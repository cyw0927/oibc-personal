"""Rolling folds: site model B + stage-1 MOS features (variant BM), scored as group sums (README provisional penalty).

Usage: cv_site_mos.py <dataset> <site_train.parquet> <out_parquet> <fold> [<fold> ...]
For each fold month M the MOS model only sees observations before M (OOF inside, full fit for M).
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import mos

D, TRAIN, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
FOLDS = sys.argv[4:]
KST = "Asia/Seoul"
DROP = {"site_id", "ts", "energy", "wa_station", "wb_station", "current_group", "smp", "k", "cf", "w", "group"}
PARAMS = dict(objective="regression_l1", n_estimators=800, learning_rate=0.05, num_leaves=127, min_child_samples=200,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=-1, random_state=0)

d = pd.read_parquet(TRAIN)
d["ts"] = d.ts.dt.tz_convert(KST)
smp = pd.read_csv(D / "train" / "smp.csv")
smp["ts"] = pd.to_datetime(smp.ts).dt.tz_convert(KST)
d = d.merge(smp, on="ts", how="left")
d["k"] = np.where(d.ts.dt.hour.between(11, 15), 3, 1).astype("int8")
d = d[d.energy.notna()]
d["cf"] = (d.energy / d.capacity_kw).clip(0, 1.2).astype("float32")
d["group"] = d.current_group.map({"A": "G1", "B": "G2"})
BASEF = [c for c in d.columns if c not in DROP]
FT = {p: mos.forecast_table(D, p) for p in ("wa", "wb")}
OB = {p: mos.observed(D, p) for p in ("wa", "wb")}

out = []
for M in FOLDS:
    P = pd.Period(M, freq="M")
    st = pd.Timestamp(P.start_time).tz_localize(KST) + pd.Timedelta(hours=1)
    en = pd.Timestamp(P.end_time.normalize() + pd.Timedelta(days=1)).tz_localize(KST)
    x = d[d.ts <= en]
    mf = []
    for p in ("wa", "wb"):
        m_ = mos.build(FT[p], OB[p], p, st, en)
        x = x.merge(m_, on=[f"{p}_station", "round", "ts"], how="left")
        mf += [c for c in m_.columns if c.startswith("mos_")]
    for c in list(mf):
        raw = c[4:]
        if raw in x.columns:
            x[f"{c}_diff"] = x[c] - x[raw]; mf.append(f"{c}_diff")
    tr = x[(x.ts < st) & (x.elev > 0)]
    te = x[(x.ts >= st) & (x.ts <= en)]
    f = BASEF + mf
    m = lgb.LGBMRegressor(**PARAMS).fit(tr[f], tr.cf)
    p_ = np.where(te.elev > 0, np.clip(m.predict(te[f]), 0, None), 0.0) * te.capacity_kw.to_numpy()
    r = te[["site_id", "ts", "group", "energy", "smp", "k"]].assign(BM=p_)
    r = r.groupby(["site_id", "ts"]).agg(group=("group", "first"), energy=("energy", "first"), smp=("smp", "first"),
                                         k=("k", "first"), BM=("BM", "mean")).reset_index()
    g = r.groupby(["group", "ts"]).agg(energy=("energy", "sum"), smp=("smp", "first"), k=("k", "first"), BM=("BM", "sum")).reset_index()
    g["fold"] = M
    w = 0.1 * g.smp * g.k
    imp = pd.Series(m.feature_importances_, index=f)[mf].sort_values(ascending=False)
    print(M, len(tr), {"BM": round((w * (g.BM - g.energy).abs()).sum() / 1e6, 3)}, "mos importance", imp.head(6).to_dict(), flush=True)
    out.append(g)
pd.concat(out).to_parquet(OUT, index=False)
