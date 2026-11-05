"""Evaluation-period candidate for #1314: ALPHA * bottom-up base (3-seed site sum) + (1-ALPHA) * group-direct (3-config mean).

Usage: predict_test.py <dataset> <feat> <base_prediction_test.csv> <new_out_dir> [ALPHA=0.4] [BASE_LABEL=FS2]
The base is the team site-level prediction_test.csv (FS2 or FS2+L1 3-seed average).
Fit window = #1142 window (2023-06-01 01:00 ~ 2025-02-28 00:00); HOLDOUT 2025-03~05 not used.
Writes composition.csv / forecast.csv (README format) and a manifest; nothing is submitted.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

D, FEAT, FS2_TEST, OUT = (Path(a) for a in sys.argv[1:5])
ALPHA_FS2 = float(sys.argv[5]) if len(sys.argv) > 5 else 0.4
BASE_LABEL = sys.argv[6] if len(sys.argv) > 6 else "FS2"
assert ALPHA_FS2 in (0.4, 0.5, 0.6), "only pre-registered blend weights (#1451)"
OUT.mkdir(parents=True, exist_ok=False)
KST = "Asia/Seoul"
FIT_END = pd.Timestamp("2025-02-28 00:00", tz=KST)
CONFIGS = [dict(learning_rate=0.05, max_iter=600, max_leaf_nodes=31, min_samples_leaf=40),
           dict(learning_rate=0.03, max_iter=900, max_leaf_nodes=15, min_samples_leaf=80),
           dict(learning_rate=0.05, max_iter=500, max_leaf_nodes=63, min_samples_leaf=30)]
NON = {"ts", "group", "energy", "cap_avail", "n_sites", "group_cap_kw", "cf", "smp", "k", "cov"}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


sites = pd.read_csv(D / "sites.csv")
sites["group"] = sites["current_group"].map({"A": "G1", "B": "G2"})
smp_tr = pd.read_csv(D / "train" / "smp.csv")
smp_tr["ts"] = pd.to_datetime(smp_tr["ts"]).dt.tz_convert(KST).dt.as_unit("us")
smp_tr["k"] = np.where(smp_tr.ts.dt.hour.between(11, 15), 3, 1)
smp_te = pd.read_csv(D / "smp.csv")
smp_te["ts"] = pd.to_datetime(smp_te["ts"]).dt.tz_convert(KST).dt.as_unit("us")

ft = pd.read_parquet(FEAT / "group_features_train.parquet")
ft["ts"] = ft.ts.dt.as_unit("us")
ft = ft.merge(smp_tr, on="ts", how="left")
ft["cov"] = ft.cap_avail / ft.group_cap_kw
ft["cf"] = (ft.energy / ft.cap_avail).clip(0, 1.2)
tr = ft[(ft.ts <= FIT_END) & ft.energy.notna() & (ft["cov"] >= 0.3) & (ft.cosz_max > 0)]
assert tr.ts.max() <= FIT_END and tr.ts.min() >= pd.Timestamp("2023-06-01 01:00", tz=KST)
feats = [c for c in ft.columns if c not in NON]
sw = tr["cov"].values * tr.k.values * np.clip(tr.smp.values, 1, None) / tr.smp.clip(lower=1).mean()

te = pd.read_parquet(FEAT / "group_features_test.parquet")
te["ts"] = te.ts.dt.as_unit("us")
keys = pd.MultiIndex.from_product([["G1", "G2"], smp_te.ts], names=["group", "ts"]).to_frame(index=False)
te = keys.merge(te, on=["group", "ts"], how="left")
assert len(te) == 17520 and te["cosz"].notna().all()

cf = np.mean([np.clip(HistGradientBoostingRegressor(loss="absolute_error", l2_regularization=1.0,
                                                    early_stopping=False, random_state=0, **c)
                      .fit(tr[feats], tr.cf, sample_weight=sw).predict(te[feats]), 0, 1) for c in CONFIGS], axis=0)
cf = np.where(te.cosz_max.values > 0, cf, 0.0)
group_cap = sites.groupby("group")["capacity_kw"].sum()
te["direct"] = cf * te["group"].map(group_cap).values

fs2 = pd.read_csv(FS2_TEST)
fs2["ts"] = pd.to_datetime(fs2["ts"]).dt.tz_convert(KST).dt.as_unit("us")
# Refuse duplicates, missing or unknown keys and non-finite values before aggregation.
expected_sites = pd.Index(sites["site_id"])
expected_times = pd.DatetimeIndex(smp_te["ts"])
if expected_sites.has_duplicates or expected_times.has_duplicates:
    raise ValueError("Duplicate site or timestamp in reference inputs")
if fs2.duplicated(["site_id", "ts"]).any():
    raise ValueError("Duplicate site_id / ts in base prediction")
if not np.isfinite(fs2["prediction_kwh"]).all() or (fs2["prediction_kwh"] < 0).any():
    raise ValueError("Base predictions must be finite and non-negative")
if (len(fs2) != len(expected_sites) * len(expected_times)
        or not fs2["site_id"].isin(expected_sites).all()
        or not fs2["ts"].isin(expected_times).all()):
    raise ValueError("Base prediction does not cover the complete reference grid")
fs2 = fs2.merge(sites[["site_id", "group"]], on="site_id", validate="many_to_one")
fsum = fs2.groupby(["group", "ts"])["prediction_kwh"].sum().rename("fs2").reset_index()
te = te.merge(fsum, on=["group", "ts"], how="left")
assert te["fs2"].notna().all()
te["forecast_kwh"] = (ALPHA_FS2 * te["fs2"] + (1 - ALPHA_FS2) * te["direct"]).clip(lower=0)

comp = sites[["site_id", "group"]].sort_values("site_id")
comp.to_csv(OUT / "composition.csv", index=False, lineterminator="\n")
fc = te[["group", "ts", "forecast_kwh"]].sort_values(["group", "ts"]).copy()
fc["ts"] = fc["ts"].dt.strftime("%Y-%m-%d %H:%M:%S%z").str.replace(r"(\+\d{2})(\d{2})$", r"\1:\2", regex=True)
fc.to_csv(OUT / "forecast.csv", index=False, lineterminator="\n", float_format="%.6f")

man = {
    "candidate_id": OUT.name, "issue": "#1314", "status": "SUBMISSION_CANDIDATE_UNAPPROVED",
    "official_submission_approved": False, "official_S": None,
    "method": f"{ALPHA_FS2} * {BASE_LABEL} 3-seed bottom-up sum + {1 - ALPHA_FS2:.1f} * group-direct HGB(L1, weight cov*k*SMP, 3-config mean); #990 reference VPP",
    "base_label": BASE_LABEL, "alpha_base": ALPHA_FS2,
    "fit_window": ["2023-06-01 01:00:00+09:00", "2025-02-28 00:00:00+09:00"], "holdout_used": False,
    "external_data": False, "observed_weather_used": False,
    "inputs_sha256": {"base_prediction_test.csv": sha(FS2_TEST), "group_features_train.parquet": sha(FEAT / "group_features_train.parquet"),
                      "group_features_test.parquet": sha(FEAT / "group_features_test.parquet"), "sites.csv": sha(D / "sites.csv"), "smp.csv": sha(D / "smp.csv"),
                      "test/forecast_wa.parquet": sha(D / "test/forecast_wa.parquet"),
                      "test/forecast_wb.parquet": sha(D / "test/forecast_wb.parquet"),
                      "train/generation.parquet": sha(D / "train/generation.parquet")},
    "outputs_sha256": {"composition.csv": sha(OUT / "composition.csv"), "forecast.csv": sha(OUT / "forecast.csv")},
    "train_rows": int(len(tr)), "n_features": len(feats),
    "dev_evidence": "see docs/OIBC_1314_GROUP_DIRECT_V0_2.md (#1451: FS2+L1 base, alpha 0.4 -> group penalty 68.308M -> 64.754M won, 4/4 folds)",
    "summary": {"G1_total_kwh": float(te[te.group == "G1"].forecast_kwh.sum()), "G2_total_kwh": float(te[te.group == "G2"].forecast_kwh.sum()),
                "fs2_total_kwh": float(te.fs2.sum()), "direct_total_kwh": float(te.direct.sum())},
}
json.dump(man, open(OUT / "MANIFEST.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(json.dumps(man, indent=1, ensure_ascii=False))
