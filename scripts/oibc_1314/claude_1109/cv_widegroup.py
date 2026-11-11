"""Group-direct model on ALL stations' raw forecasts (no compression): for every ts and round, each A/B station's
cloud/uv/humidity/precip-prob forecast becomes its own column (~134 stations x vars), plus solar/calendar.
Rows = (group, ts, round) stacked, target = group cf = energy / cap_avail; predictions averaged over rounds.
Same folds/penalty as everything else; output group-hour predictions 'W' for blending.
Usage: cv_widegroup.py <dataset> <feat_dir (group_features_train.parquet)> <out.parquet> <folds...>
All inputs are D-1 forecasts (rounds 10/17) only.
"""
import sys, warnings
from pathlib import Path
warnings.simplefilter("ignore")
import lightgbm as lgb
import numpy as np
import pandas as pd

D, FEAT, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); FOLDS = sys.argv[4:]
KST = "Asia/Seoul"
V = {"wa": ["wa_cloud", "wa_humidity", "wa_rain"], "wb": ["wb_cloud", "wb_uv_idx", "wb_precip_prob", "wb_vis"]}
wide = None
for p, cols in V.items():
    st = f"{p}_station"
    f = pd.read_parquet(D / "train" / f"forecast_{p}.parquet", columns=[st, "round", "ts"] + cols)
    f["ts"] = pd.to_datetime(f.ts).dt.tz_convert(KST)
    f = f[(f.ts.dt.minute == 0) & f.ts.dt.hour.between(6, 19) & (f.ts >= pd.Timestamp("2023-06-01", tz=KST))].drop_duplicates([st, "round", "ts"], keep="last")
    w = f.pivot_table(index=["ts", "round"], columns=st, values=cols)
    w.columns = [f"{v}@{s}" for v, s in w.columns]
    wide = w if wide is None else wide.join(w, how="outer")
wide = wide.astype("float32").reset_index()
print("wide columns", wide.shape[1] - 2, flush=True)
g = pd.read_parquet(FEAT / "group_features_train.parquet"); g["ts"] = pd.to_datetime(g.ts).dt.tz_convert(KST)
keep = ["ts", "group", "energy", "cap_avail", "group_cap_kw", "cosz", "cosz_max", "cs_ab", "doy_sin", "doy_cos", "hour"]
g = g[keep]
smp = pd.read_csv(D / "train" / "smp.csv"); smp["ts"] = pd.to_datetime(smp.ts).dt.tz_convert(KST)
g = g.merge(smp, on="ts", how="left"); g["k"] = np.where(g.ts.dt.hour.between(11, 15), 3, 1)
x = g.merge(wide, on="ts", how="inner")
x["gnum"] = (x.group == "G2").astype("int8"); x["cf"] = (x.energy / x.cap_avail).clip(0, 1.2); x["cov"] = x.cap_avail / x.group_cap_kw
F = [c for c in x.columns if "@" in c] + ["cosz", "cosz_max", "cs_ab", "doy_sin", "doy_cos", "hour", "gnum", "round"]
P = dict(objective="regression_l1", n_estimators=1500, learning_rate=0.03, num_leaves=63, min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.3, verbose=-1, n_jobs=-1, random_state=0)
out = []
for M in FOLDS:
    Pm = pd.Period(M, freq="M"); st = pd.Timestamp(Pm.start_time).tz_localize(KST) + pd.Timedelta(hours=1)
    en = pd.Timestamp(Pm.end_time.normalize() + pd.Timedelta(days=1)).tz_localize(KST)
    tr = x[(x.ts < st) & x.energy.notna() & (x["cov"] >= 0.3) & (x.cosz_max > 0)]
    te = x[(x.ts >= st) & (x.ts <= en) & x.energy.notna()].copy()
    sw = (tr["cov"] * tr.k * tr.smp.clip(lower=1)).to_numpy(); sw = sw / sw.mean()
    m = lgb.LGBMRegressor(**P).fit(tr[F], tr.cf, sample_weight=sw)
    te["W"] = np.where(te.cosz_max > 0, np.clip(m.predict(te[F]), 0, 1.2), 0) * te.cap_avail
    r = te.groupby(["group", "ts"]).agg(energy=("energy", "first"), smp=("smp", "first"), k=("k", "first"), W=("W", "mean")).reset_index()
    r["fold"] = M
    w = 0.1 * r.smp * r.k
    print(M, len(tr), "W", round((w * (r.W - r.energy).abs()).sum() / 1e6, 3), flush=True)
    out.append(r)
pd.concat(out).to_parquet(OUT, index=False)
