"""Fit a site-level LightGBM variant on all training data (to 2025-06-01 00:00) and predict the evaluation year.

Usage: final_site.py <dataset> <site_feat_dir> <variant A|B|C> <out_prefix> [seeds=0]
Writes <out_prefix>_sites.parquet (site_id, ts, kwh: mean over rounds and seeds) and <out_prefix>_group.parquet.
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

D, FD, V, OUT = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
SEEDS = [int(s) for s in (sys.argv[5] if len(sys.argv) > 5 else "0").split(",")]
ECHR = next((a[5:] for a in sys.argv[6:] if a.startswith("echr=")), None)   # ECMWF 3-hourly fixed-box ssrd by ts (no site coordinates)
KST = "Asia/Seoul"
BASE = ["wa_cloud", "wa_temp", "wa_humidity", "wa_rain", "wa_wind_speed", "wa_ground_press",
        "wb_cloud", "wb_temp", "wb_rel_hum", "wb_uv_idx", "wb_precip_prob", "wb_total_liq", "wb_vis", "wb_wind_speed",
        "wb_dew_point", "elev", "cosz", "doy_sin", "doy_cos", "hour", "capacity_kw", "lat", "lon", "round"]
DROP = {"site_id", "ts", "energy", "wa_station", "wb_station", "current_group", "smp", "k", "cf", "w", "group"}
PARAMS = dict(objective="regression_l1", n_estimators=800, learning_rate=0.05, num_leaves=127, min_child_samples=200,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=-1)
VPARAMS = {"E": dict(n_estimators=2000, learning_rate=0.03, num_leaves=255, min_child_samples=100),
           "G": dict(colsample_bytree=0.5, num_leaves=255, n_estimators=1200, learning_rate=0.04)}

tr = pd.read_parquet(FD / "site_train.parquet")
tr["ts"] = tr.ts.dt.tz_convert(KST)
smp = pd.read_csv(D / "train" / "smp.csv")
smp["ts"] = pd.to_datetime(smp.ts).dt.tz_convert(KST)
tr = tr.merge(smp, on="ts", how="left")
if ECHR:
    eh = pd.read_parquet(ECHR); eh["ts"] = pd.to_datetime(eh.ts).dt.tz_convert(KST)
    tr = tr.merge(eh, on="ts", how="left")
tr["k"] = np.where(tr.ts.dt.hour.between(11, 15), 3, 1)
tr = tr[tr.energy.notna() & (tr.elev > 0)]
tr["cf"] = (tr.energy / tr.capacity_kw).clip(0, 1.2)
feats = BASE if V == "A" else [c for c in tr.columns if c not in DROP]
assert V in ("A", "B", "C", "E", "G"), V
w = None
if V == "C":
    w = (tr.k * tr.smp.clip(lower=1)).to_numpy(); w = w / w.mean()

te = pd.read_parquet(FD / "site_test.parquet")
te["ts"] = te.ts.dt.tz_convert(KST)
if ECHR:
    te = te.merge(eh, on="ts", how="left"); print("echr test coverage", round(te[eh.columns[1]].notna().mean(), 3), flush=True)
assert set(feats) <= set(te.columns), set(feats) - set(te.columns)
pred = np.zeros(len(te))
for s in SEEDS:
    m = lgb.LGBMRegressor(random_state=s, **{**PARAMS, **VPARAMS.get(V, {})}).fit(tr[feats], tr.cf, sample_weight=w)
    pred += np.where(te.elev > 0, np.clip(m.predict(te[feats]), 0, None), 0.0) * te.capacity_kw.to_numpy() / len(SEEDS)
    print("seed", s, "done", flush=True)
te["kwh"] = pred
site = te.groupby(["site_id", "ts"]).kwh.mean().reset_index()
sites = pd.read_csv(D / "sites.csv")
idx = pd.to_datetime(pd.read_csv(D / "smp.csv").ts).dt.tz_convert(KST)
grid = pd.MultiIndex.from_product([sites.site_id, idx], names=["site_id", "ts"]).to_frame(index=False)
site = grid.merge(site, on=["site_id", "ts"], how="left").fillna({"kwh": 0.0})       # deep-night rows were not built
assert len(site) == 253 * 8760 and (site.kwh >= 0).all()
site.to_parquet(OUT + "_sites.parquet", index=False)
grp = site.merge(sites[["site_id", "current_group"]], on="site_id")
grp["group"] = grp.current_group.map({"A": "G1", "B": "G2"})
g = grp.groupby(["group", "ts"]).kwh.sum().reset_index()
g.to_parquet(OUT + "_group.parquet", index=False)
print(V, "total GWh", round(g.kwh.sum() / 1e6, 3), flush=True)
