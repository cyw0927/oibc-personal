"""Fit group-direct H1 (v0.2 group features + day context, HGB L1, 3 configs) on 2023-06-01 01:00..2025-06-01 00:00 and
predict the evaluation year for the groups defined in <dataset>/sites.csv (current_group A->G1, B->G2).
Same model as final_predict.py's H1 arm. Usage: final_h1.py <dataset> <feat_dir> <out_group.parquet> [echr=<hourly fixed-box parquet>]
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from ctx_features import add_ctx

D, FEAT, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
KST = "Asia/Seoul"
FIT_START = pd.Timestamp("2023-06-01 01:00", tz=KST); FULL_END = pd.Timestamp("2025-06-01 00:00", tz=KST)
CONFIGS = [dict(learning_rate=0.05, max_iter=600, max_leaf_nodes=31, min_samples_leaf=40),
           dict(learning_rate=0.03, max_iter=900, max_leaf_nodes=15, min_samples_leaf=80),
           dict(learning_rate=0.05, max_iter=500, max_leaf_nodes=63, min_samples_leaf=30)]
NON = {"ts", "group", "energy", "cap_avail", "n_sites", "group_cap_kw", "cf", "smp", "k", "cov", "split"}
sites = pd.read_csv(D / "sites.csv"); sites["group"] = sites.current_group.map({"A": "G1", "B": "G2"})
group_cap = sites.groupby("group").capacity_kw.sum()
smp = pd.read_csv(D / "train" / "smp.csv"); smp["ts"] = pd.to_datetime(smp.ts).dt.tz_convert(KST).dt.as_unit("us")
smp["k"] = np.where(smp.ts.dt.hour.between(11, 15), 3, 1)
smp_te = pd.read_csv(D / "smp.csv"); smp_te["ts"] = pd.to_datetime(smp_te.ts).dt.tz_convert(KST).dt.as_unit("us")
tr0 = pd.read_parquet(FEAT / "group_features_train.parquet").assign(split="train")
te0 = pd.read_parquet(FEAT / "group_features_test.parquet").assign(split="test")
base = [c for c in te0.columns if c not in NON]
allf = pd.concat([tr0, te0], ignore_index=True); allf["ts"] = allf.ts.dt.as_unit("us")
assert not allf.duplicated(["group", "ts"]).any()
allf, new = add_ctx(allf)
ECHR = next((a[5:] for a in sys.argv[4:] if a.startswith("echr=")), None)
gx = []
if ECHR:
    eh = pd.read_parquet(ECHR); eh["ts"] = pd.to_datetime(eh.ts).dt.tz_convert(KST).dt.as_unit("us")
    gx = [c for c in eh.columns if c != "ts"]
    allf = allf.merge(eh, on="ts", how="left")
    print("echr test coverage", round(allf[allf.split == "test"][gx[0]].notna().mean(), 3), flush=True)
ft = allf[allf.split == "train"].merge(smp, on="ts", how="left")
ft["cov"] = ft.cap_avail / ft.group_cap_kw; ft["cf"] = (ft.energy / ft.cap_avail).clip(0, 1.2)
keys = pd.MultiIndex.from_product([["G1", "G2"], smp_te.ts], names=["group", "ts"]).to_frame(index=False)
te = keys.merge(allf[allf.split == "test"], on=["group", "ts"], how="left")
assert len(te) == 17520 and te.cosz.notna().all()
feats = base + new + gx
tr = ft[(ft.ts >= FIT_START) & (ft.ts <= FULL_END) & ft.energy.notna() & (ft["cov"] >= 0.3) & (ft.cosz_max > 0)]
sw = tr["cov"].values * tr.k.values * np.clip(tr.smp.values, 1, None) / tr.smp.clip(lower=1).mean()
cf = np.mean([np.clip(HistGradientBoostingRegressor(loss="absolute_error", l2_regularization=1.0, early_stopping=False, random_state=0, **c)
                      .fit(tr[feats], tr.cf, sample_weight=sw).predict(te[feats]), 0, 1) for c in CONFIGS], axis=0)
te["kwh"] = np.where(te.cosz_max.values > 0, cf, 0.0) * te.group.map(group_cap).values
te[["group", "ts", "kwh"]].to_parquet(OUT, index=False)
print("H1 train rows", len(tr), "features", len(feats), "total GWh", round(te.kwh.sum() / 1e6, 3), {g: round(v / 1e6, 3) for g, v in te.groupby("group").kwh.sum().items()})
