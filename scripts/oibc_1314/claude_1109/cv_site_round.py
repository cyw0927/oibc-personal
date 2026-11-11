"""Candidate 1: how the two forecast issues (round 10 / 17) are used. Site model B params, rolling folds.

Usage: cv_site_round.py <dataset> <site_train.parquet> <out_parquet> <fold> [<fold> ...]
Arms (group-sum penalty, README provisional):
  B      stacked rounds, round as feature, mean of both round predictions (current)
  B10    same model, round-10 prediction only
  B17    same model, round-17 prediction only
  BR     one row per site-hour: round-17 features + round-10 values + revision (r17 - r10) for every forecast column
Both rounds are issued before D-1 17:00 KST (README), so all arms are legal.
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

D, TRAIN, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
FOLDS = sys.argv[4:]
KST = "Asia/Seoul"
DROP = {"site_id", "ts", "energy", "wa_station", "wb_station", "current_group", "smp", "k", "cf", "w", "group"}
PARAMS = dict(objective="regression_l1", n_estimators=800, learning_rate=0.05, num_leaves=127, min_child_samples=200,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=-1, random_state=0)
STATIC = {"elev", "cosz", "doy_sin", "doy_cos", "hour", "doy", "capacity_kw", "lat", "lon", "site_num", "round"}

d = pd.read_parquet(TRAIN)
d["ts"] = d.ts.dt.tz_convert(KST)
smp = pd.read_csv(D / "train" / "smp.csv")
smp["ts"] = pd.to_datetime(smp.ts).dt.tz_convert(KST)
d = d.merge(smp, on="ts", how="left")
d["k"] = np.where(d.ts.dt.hour.between(11, 15), 3, 1).astype("int8")
d = d[d.energy.notna()]
d["cf"] = (d.energy / d.capacity_kw).clip(0, 1.2).astype("float32")
d["group"] = d.current_group.map({"A": "G1", "B": "G2"})
ALL = [c for c in d.columns if c not in DROP]
FC = [c for c in ALL if c not in STATIC and d[c].dtype.kind == "f"]           # forecast-derived columns
print("features", len(ALL), "forecast-derived", len(FC), flush=True)

r17 = d[d["round"] == 17].drop(columns="round")
r10 = d[d["round"] == 10][["site_id", "ts"] + FC].rename(columns={c: f"{c}_r10" for c in FC})
w = r17.merge(r10, on=["site_id", "ts"], how="inner")
for c in FC:
    w[f"{c}_rev"] = (w[c] - w[f"{c}_r10"]).astype("float32")
WF = [c for c in ALL if c != "round"] + [f"{c}_r10" for c in FC] + [f"{c}_rev" for c in FC]
print("wide rows", len(w), "features", len(WF), flush=True)

out = []
for M in FOLDS:
    P = pd.Period(M, freq="M")
    st = pd.Timestamp(P.start_time).tz_localize(KST) + pd.Timedelta(hours=1)
    en = pd.Timestamp(P.end_time.normalize() + pd.Timedelta(days=1)).tz_localize(KST)
    tr = d[(d.ts < st) & (d.elev > 0)]
    te = d[(d.ts >= st) & (d.ts <= en)]
    m = lgb.LGBMRegressor(**PARAMS).fit(tr[ALL], tr.cf)
    p = np.where(te.elev > 0, np.clip(m.predict(te[ALL]), 0, None), 0.0) * te.capacity_kw.to_numpy()
    pr = te[["site_id", "ts", "round"]].assign(p=p).pivot_table(index=["site_id", "ts"], columns="round", values="p").reset_index()
    pr = pr.rename(columns={10: "B10", 17: "B17"})
    wtr = w[(w.ts < st) & (w.elev > 0)]
    wte = w[(w.ts >= st) & (w.ts <= en)]
    mw = lgb.LGBMRegressor(**PARAMS).fit(wtr[WF], wtr.cf)
    pw = np.where(wte.elev > 0, np.clip(mw.predict(wte[WF]), 0, None), 0.0) * wte.capacity_kw.to_numpy()
    res = wte[["site_id", "ts", "group", "energy", "smp", "k"]].assign(BR=pw).merge(pr, on=["site_id", "ts"], how="left")
    res["B"] = res[["B10", "B17"]].mean(axis=1)
    A = ["B", "B10", "B17", "BR"]
    g = res.groupby(["group", "ts"]).agg(energy=("energy", "sum"), smp=("smp", "first"), k=("k", "first"), **{a: (a, "sum") for a in A}).reset_index()
    g["fold"] = M
    ww = 0.1 * g.smp * g.k
    imp = pd.Series(mw.feature_importances_, index=WF)
    top_rev = imp[[c for c in WF if c.endswith("_rev")]].sort_values(ascending=False).head(5).to_dict()
    print(M, len(tr), {a: round((ww * (g[a] - g.energy).abs()).sum() / 1e6, 3) for a in A}, "top rev", top_rev, flush=True)
    out.append(g)
pd.concat(out).to_parquet(OUT, index=False)
