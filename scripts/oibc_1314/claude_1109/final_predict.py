"""Final candidate after the one-shot HOLDOUT test (protocol docs/experiments/holdout_final_test_protocol_20261109.md, decision H1).

forecast = 0.4 * FS2+L1 3-seed site sum + 0.6 * group-direct H1 (v0.2 features + day context), fit on the full
training range 2023-06-01 01:00 ~ 2025-06-01 00:00. VPP = #990 reference. Nothing is submitted.

Usage: final_predict.py <dataset> <feat_full_dir> <fs2l1_prediction_test.csv> <v2_forecast.csv> <new_out_dir>
Sanity: H0 refit on the old window must reproduce the submitted v2 forecast (0.4*FS2+L1 + 0.6*H0) exactly.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from ctx_features import add_ctx

D, FEAT, BASE, V2, OUT = (Path(a) for a in sys.argv[1:6])
OUT.mkdir(parents=True, exist_ok=False)
KST = "Asia/Seoul"
ALPHA = 0.4
FIT_START = pd.Timestamp("2023-06-01 01:00", tz=KST)
OLD_END = pd.Timestamp("2025-02-28 00:00", tz=KST)
FULL_END = pd.Timestamp("2025-06-01 00:00", tz=KST)
CONFIGS = [dict(learning_rate=0.05, max_iter=600, max_leaf_nodes=31, min_samples_leaf=40),
           dict(learning_rate=0.03, max_iter=900, max_leaf_nodes=15, min_samples_leaf=80),
           dict(learning_rate=0.05, max_iter=500, max_leaf_nodes=63, min_samples_leaf=30)]
NON = {"ts", "group", "energy", "cap_avail", "n_sites", "group_cap_kw", "cf", "smp", "k", "cov", "split"}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


sites = pd.read_csv(D / "sites.csv")
sites["group"] = sites["current_group"].map({"A": "G1", "B": "G2"})
group_cap = sites.groupby("group")["capacity_kw"].sum()
smp = pd.read_csv(D / "train" / "smp.csv")
smp["ts"] = pd.to_datetime(smp["ts"]).dt.tz_convert(KST).dt.as_unit("us")
smp["k"] = np.where(smp.ts.dt.hour.between(11, 15), 3, 1)
smp_te = pd.read_csv(D / "smp.csv")
smp_te["ts"] = pd.to_datetime(smp_te["ts"]).dt.tz_convert(KST).dt.as_unit("us")

tr0 = pd.read_parquet(FEAT / "group_features_train.parquet").assign(split="train")
te0 = pd.read_parquet(FEAT / "group_features_test.parquet").assign(split="test")
base = [c for c in te0.columns if c not in NON]
allf = pd.concat([tr0, te0], ignore_index=True)
allf["ts"] = allf.ts.dt.as_unit("us")
assert not allf.duplicated(["group", "ts"]).any()
allf, new = add_ctx(allf)
ft = allf[allf.split == "train"].merge(smp, on="ts", how="left")
ft["cov"] = ft.cap_avail / ft.group_cap_kw
ft["cf"] = (ft.energy / ft.cap_avail).clip(0, 1.2)
keys = pd.MultiIndex.from_product([["G1", "G2"], smp_te.ts], names=["group", "ts"]).to_frame(index=False)
te = keys.merge(allf[allf.split == "test"], on=["group", "ts"], how="left")
assert len(te) == 17520 and te["cosz"].notna().all()


def direct(end, feats):
    tr = ft[(ft.ts >= FIT_START) & (ft.ts <= end) & ft.energy.notna() & (ft["cov"] >= 0.3) & (ft.cosz_max > 0)]
    sw = tr["cov"].values * tr.k.values * np.clip(tr.smp.values, 1, None) / tr.smp.clip(lower=1).mean()
    cf = np.mean([np.clip(HistGradientBoostingRegressor(loss="absolute_error", l2_regularization=1.0,
                                                        early_stopping=False, random_state=0, **c)
                          .fit(tr[feats], tr.cf, sample_weight=sw).predict(te[feats]), 0, 1) for c in CONFIGS], axis=0)
    return np.where(te.cosz_max.values > 0, cf, 0.0) * te["group"].map(group_cap).values, len(tr)


b = pd.read_csv(BASE)
b["ts"] = pd.to_datetime(b["ts"]).dt.tz_convert(KST).dt.as_unit("us")
assert len(b) == 253 * 8760 and not b.duplicated(["site_id", "ts"]).any()
assert np.isfinite(b.prediction_kwh).all() and (b.prediction_kwh >= 0).all()
assert set(b.site_id) == set(sites.site_id) and b.ts.isin(smp_te.ts).all()
bsum = b.merge(sites[["site_id", "group"]], on="site_id").groupby(["group", "ts"]).prediction_kwh.sum().rename("base").reset_index()
te = te.merge(bsum, on=["group", "ts"], how="left")
assert te.base.notna().all()

# sanity: reproduce the submitted v2 forecast from H0 on the old window
h0_old, _ = direct(OLD_END, base)
v2 = pd.read_csv(V2)
v2["ts"] = pd.to_datetime(v2["ts"]).dt.tz_convert(KST).dt.as_unit("us")
chk = te[["group", "ts"]].assign(rep=(ALPHA * te.base + (1 - ALPHA) * h0_old).clip(lower=0)).merge(v2, on=["group", "ts"])
v2_maxdiff = float((chk.rep - chk.forecast_kwh).abs().max())
assert len(chk) == 17520 and v2_maxdiff < 1e-5, v2_maxdiff

te["direct"], n_train = direct(FULL_END, base + new)
te["forecast_kwh"] = (ALPHA * te.base + (1 - ALPHA) * te.direct).clip(lower=0)
assert np.isfinite(te.forecast_kwh).all()

comp = sites[["site_id", "group"]].sort_values("site_id")
comp.to_csv(OUT / "composition.csv", index=False, lineterminator="\n")
fc = te[["group", "ts", "forecast_kwh"]].sort_values(["group", "ts"]).copy()
fc["ts"] = fc["ts"].dt.strftime("%Y-%m-%d %H:%M:%S%z").str.replace(r"(\+\d{2})(\d{2})$", r"\1:\2", regex=True)
fc.to_csv(OUT / "forecast.csv", index=False, lineterminator="\n", float_format="%.6f")

old = te.assign(m=te.ts.dt.month, v2=ALPHA * te.base + (1 - ALPHA) * h0_old)
monthly = (old.groupby("m").forecast_kwh.sum() / old.groupby("m").v2.sum() - 1).round(4)
man = {
    "candidate_id": OUT.name, "issue": "#1314 personal", "status": "SUBMISSION_CANDIDATE_UNAPPROVED", "official_S": None,
    "method": "0.4 * FS2+L1 3-seed site sum + 0.6 * group-direct H1 (v0.2 40 features + 70 day-context features, HGB L1, weight cov*k*SMP, 3-config mean); #990 reference VPP",
    "protocol": "oibc-personal docs/experiments/holdout_final_test_protocol_20261109.md (HOLDOUT one-shot test: H0 57.402M -> H1 53.779M, decision H1)",
    "fit_window": ["2023-06-01 01:00:00+09:00", "2025-06-01 00:00:00+09:00"], "holdout_used": "final test once + final training (user decision 11/09)",
    "fs2l1_part": "unchanged LightGBM FS2+L1 test prediction (fit to 2025-02), alpha 0.4 fixed",
    "external_data": False, "observed_weather_used": False,
    "inputs_sha256": {"base_prediction_test.csv": sha(BASE), "group_features_train.parquet": sha(FEAT / "group_features_train.parquet"),
                      "group_features_test.parquet": sha(FEAT / "group_features_test.parquet"), "sites.csv": sha(D / "sites.csv"),
                      "smp.csv": sha(D / "smp.csv"), "v2_forecast.csv": sha(V2)},
    "outputs_sha256": {"composition.csv": sha(OUT / "composition.csv"), "forecast.csv": sha(OUT / "forecast.csv")},
    "train_rows": n_train, "n_features": len(base) + len(new), "v2_reproduction_maxdiff_kwh": v2_maxdiff,
    "monthly_total_vs_v2": {int(k): float(v) for k, v in monthly.items()},
    "summary": {"G1_total_kwh": float(te[te.group == "G1"].forecast_kwh.sum()), "G2_total_kwh": float(te[te.group == "G2"].forecast_kwh.sum()),
                "base_total_kwh": float(te.base.sum()), "direct_total_kwh": float(te.direct.sum())},
}
json.dump(man, open(OUT / "MANIFEST.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(json.dumps(man, indent=1, ensure_ascii=False))
