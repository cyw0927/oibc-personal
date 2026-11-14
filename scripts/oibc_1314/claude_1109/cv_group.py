"""12 monthly expanding-window folds (2024-06 .. 2025-05) for the group-direct model H1 (and H0).

Usage: cv_group.py <dataset> <feat_full_dir> <out_parquet> [spatial] [echr=<hourly fixed-box parquet>]
echr: merge hourly external fixed-box features (gx_*) on ts and fit only H1 = base + ctx + gx.
Pred = cf * cap_avail (available cohort), README provisional group penalty. Train from 2023-06-01 01:00 to fold start.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from ctx_features import add_ctx

D, FEAT, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
KST = "Asia/Seoul"
CONFIGS = [dict(learning_rate=0.05, max_iter=600, max_leaf_nodes=31, min_samples_leaf=40),
           dict(learning_rate=0.03, max_iter=900, max_leaf_nodes=15, min_samples_leaf=80),
           dict(learning_rate=0.05, max_iter=500, max_leaf_nodes=63, min_samples_leaf=30)]
NON = {"ts", "group", "energy", "cap_avail", "n_sites", "group_cap_kw", "cf", "smp", "k", "cov"}

smp = pd.read_csv(D / "train" / "smp.csv")
smp["ts"] = pd.to_datetime(smp["ts"]).dt.tz_convert(KST).dt.as_unit("us")
smp["k"] = np.where(smp.ts.dt.hour.between(11, 15), 3, 1)
ft = pd.read_parquet(FEAT / "group_features_train.parquet")
ft["ts"] = ft.ts.dt.as_unit("us")
base = [c for c in ft.columns if c not in NON]
ft, new = add_ctx(ft)
SPATIAL = len(sys.argv) > 4 and sys.argv[4] == "spatial"
ECHR = next((a[5:] for a in sys.argv[4:] if a.startswith("echr=")), None)
gx = []
if ECHR:
    eh = pd.read_parquet(ECHR); eh["ts"] = pd.to_datetime(eh.ts).dt.tz_convert(KST).dt.as_unit("us")
    gx = [c for c in eh.columns if c != "ts"]
    ft = ft.merge(eh, on="ts", how="left")
    print("echr cols", len(gx), "coverage", round(ft[gx[0]].notna().mean(), 3), flush=True)
sp = []
if SPATIAL:
    pcs = pd.read_parquet(Path(sys.argv[2]).parent / "spatial" / "pcs_train.parquet")
    pcs = pcs.drop(columns="round").groupby("ts").mean().reset_index()          # mean of rounds 10/17, like the group features
    pcs["ts"] = pcs.ts.dt.as_unit("us")
    sp = [c for c in pcs.columns if c != "ts"]
    ft = ft.merge(pcs, on="ts", how="left")
ft = ft.merge(smp, on="ts", how="left")
ft["cov"] = ft.cap_avail / ft.group_cap_kw
ft["cf"] = (ft.energy / ft.cap_avail).clip(0, 1.2)

out = []
for M in pd.period_range("2024-06", "2025-05", freq="M"):
    st = pd.Timestamp(M.start_time).tz_localize(KST) + pd.Timedelta(hours=1)
    en = pd.Timestamp(M.end_time.normalize() + pd.Timedelta(days=1)).tz_localize(KST)
    tr = ft[(ft.ts < st) & ft.energy.notna() & (ft["cov"] >= 0.3) & (ft.cosz_max > 0)]
    te = ft[(ft.ts >= st) & (ft.ts <= en) & ft.energy.notna()].copy()
    sw = tr["cov"].values * tr.k.values * np.clip(tr.smp.values, 1, None) / tr.smp.clip(lower=1).mean()
    arms = (("H1", base + new), ("H2", base + new + sp)) if SPATIAL else (("H1", base + new + gx),) if ECHR else (("H0", base), ("H1", base + new))
    for name, feats in arms:
        cf = np.mean([np.clip(HistGradientBoostingRegressor(loss="absolute_error", l2_regularization=1.0,
                                                            early_stopping=False, random_state=0, **c)
                              .fit(tr[feats], tr.cf, sample_weight=sw).predict(te[feats]), 0, 1) for c in CONFIGS], axis=0)
        te[name] = np.where(te.cosz_max.values > 0, cf, 0.0) * te.cap_avail.values
    te["fold"] = str(M)
    w = 0.1 * te.smp * te.k
    names = [a[0] for a in arms]
    print(M, {n: round((w * (te[n] - te.energy).abs()).sum() / 1e6, 3) for n in names}, flush=True)
    out.append(te[["fold", "group", "ts", "energy", "cap_avail", "smp", "k"] + names])
pd.concat(out).to_parquet(OUT, index=False)
