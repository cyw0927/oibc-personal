"""12 monthly expanding-window folds (2024-06 .. 2025-05) for site-level LightGBM variants, scored as group sums.

Usage: cv_site.py <dataset> <site_train.parquet> <out_parquet> <variant> [<variant> ...]
Variants:
  A  organizer-baseline features, L1                      (~ organizer baseline with L1)
  B  all features (station context, regional, cs, extras), L1
  C  B + sample weight SMP*k
  BT  B + recency sample weight (half-life 180 d; forecast-provider bias drifts over time, Kang & Tam 2015)
  L2 / L2E  B / E features and sizes with squared-error objective (conditional mean, organizer-baseline style)
Rows of both forecast rounds are stacked (round is a feature); the prediction is the mean over rounds.
Group sums over the sites present in each hour, README provisional group penalty (#990 reference groups).
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

D, TRAIN, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
VARIANTS = [v for v in sys.argv[4:] if not v.startswith(("folds=", "savesite=", "abias="))]
SAVESITE = next((v[9:] for v in sys.argv[4:] if v.startswith("savesite=")), None)   # optional site-level OOF parquet
ABIAS = next((v[6:] for v in sys.argv[4:] if v.startswith("abias=")), None)   # station|global: provider-A cloud bias drift correction
FOLDS = next((v[6:].split(",") for v in sys.argv[4:] if v.startswith("folds=")), None)
KST = "Asia/Seoul"
BASE = ["wa_cloud", "wa_temp", "wa_humidity", "wa_rain", "wa_wind_speed", "wa_ground_press",
        "wb_cloud", "wb_temp", "wb_rel_hum", "wb_uv_idx", "wb_precip_prob", "wb_total_liq", "wb_vis", "wb_wind_speed",
        "wb_dew_point", "elev", "cosz", "doy_sin", "doy_cos", "hour", "capacity_kw", "lat", "lon", "round"]
DROP = {"site_id", "ts", "energy", "wa_station", "wb_station", "current_group", "smp", "k", "cf", "w"}
PARAMS = dict(objective="regression_l1", n_estimators=800, learning_rate=0.05, num_leaves=127, min_child_samples=200,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=-1, random_state=0)

d = pd.read_parquet(TRAIN)
d["ts"] = d.ts.dt.tz_convert(KST)
smp = pd.read_csv(D / "train" / "smp.csv")
smp["ts"] = pd.to_datetime(smp.ts).dt.tz_convert(KST)
if ABIAS:
    from abias import apply as _abias
    d = _abias(d, D, ABIAS)
d = d.merge(smp, on="ts", how="left")
d["k"] = np.where(d.ts.dt.hour.between(11, 15), 3, 1).astype("int8")
d = d[d.energy.notna()]
SP = Path(str(TRAIN)).parent.parent / "spatial"
if any(v.endswith("S") for v in VARIANTS):
    pcs = pd.read_parquet(SP / "pcs_train.parquet"); pcs["ts"] = pcs.ts.dt.tz_convert(KST)
    d = d.merge(pcs, on=["round", "ts"], how="left")
    for p in ("wa", "wb"):
        nb = pd.read_parquet(SP / f"nbr_{p}_train.parquet"); nb["ts"] = nb.ts.dt.tz_convert(KST)
        d = d.merge(nb, on=[f"{p}_station", "round", "ts"], how="left")
    SPF = [c for c in d.columns if "_pch" in c or "_pcd" in c or "_nb" in c]
    print("spatial features", len(SPF), flush=True)
d["cf"] = (d.energy / d.capacity_kw).clip(0, 1.2).astype("float32")
d["group"] = d.current_group.map({"A": "G1", "B": "G2"})
ALL = [c for c in d.columns if c not in DROP and c != "group" and "_pch" not in c and "_pcd" not in c and "_nb" not in c]
FEATS = {"A": BASE, "B": ALL, "C": ALL, "E": ALL, "G": ALL, "L2": ALL, "L2E": ALL, "BT": ALL}
if any(v.endswith("S") for v in VARIANTS):
    FEATS.update({"BS": ALL + SPF, "ES": ALL + SPF})
VPARAMS_ALIAS = {"ES": "E"}
VPARAMS = {"E": dict(n_estimators=2000, learning_rate=0.03, num_leaves=255, min_child_samples=100),
           "G": dict(colsample_bytree=0.5, num_leaves=255, n_estimators=1200, learning_rate=0.04),
           "L2": dict(objective="regression"),                                    # mean target like the organizer baseline
           "L2E": dict(objective="regression", n_estimators=2000, learning_rate=0.03, num_leaves=255, min_child_samples=100)}

out = []
sites_out = []
for M in pd.period_range("2024-06", "2025-05", freq="M"):
    if FOLDS and str(M) not in FOLDS:
        continue
    st = pd.Timestamp(M.start_time).tz_localize(KST) + pd.Timedelta(hours=1)
    en = pd.Timestamp(M.end_time.normalize() + pd.Timedelta(days=1)).tz_localize(KST)
    tr = d[(d.ts < st) & (d.elev > 0)]
    te = d[(d.ts >= st) & (d.ts <= en)]
    res = te[te["round"] == 17][["site_id", "ts", "group", "energy", "smp", "k"]].reset_index(drop=True)
    for v in VARIANTS:
        f = FEATS[v]
        w = None
        if v == "C":
            w = (tr.k * tr.smp.clip(lower=1)).to_numpy(); w = w / w.mean()
        if v == "BT":                                                     # recency weight, half-life 180 days before the cut
            age = (st - tr.ts).dt.total_seconds().to_numpy() / 86400.0
            w = np.power(0.5, age / 180.0); w = w / w.mean()
        m = lgb.LGBMRegressor(**{**PARAMS, **VPARAMS.get(VPARAMS_ALIAS.get(v, v), {})}).fit(tr[f], tr.cf, sample_weight=w)
        p = np.where(te.elev > 0, np.clip(m.predict(te[f]), 0, None), 0.0) * te.capacity_kw.to_numpy()
        pr = te[["site_id", "ts"]].assign(p=p).groupby(["site_id", "ts"]).p.mean().rename(v).reset_index()
        res = res.merge(pr, on=["site_id", "ts"], how="left")
    if SAVESITE:
        sites_out.append(res.assign(fold=str(M)))
    g = res.groupby(["group", "ts"]).agg(energy=("energy", "sum"), smp=("smp", "first"), k=("k", "first"),
                                         **{v: (v, "sum") for v in VARIANTS}).reset_index()
    g["fold"] = str(M)
    ww = 0.1 * g.smp * g.k
    print(M, len(tr), {v: round((ww * (g[v] - g.energy).abs()).sum() / 1e6, 3) for v in VARIANTS}, flush=True)
    out.append(g)
pd.concat(out).to_parquet(OUT, index=False)
if SAVESITE:
    pd.concat(sites_out).to_parquet(SAVESITE, index=False)
