"""Organizer baseline (LightGBM MSE, rounds 10/17 averaged) fit to 2025-02-28 and scored on HOLDOUT 2025-03~05,
same README provisional group penalty as holdout_test.py. Also an L1 variant. Writes group preds parquet."""
import sys, json
import numpy as np, pandas as pd, lightgbm as lgb
import organizer_baseline as B

D, OUT = sys.argv[1], sys.argv[2]
KST = "Asia/Seoul"
FIT_END = pd.Timestamp("2025-02-28 00:00", tz=KST)
HO = (pd.Timestamp("2025-03-01 01:00", tz=KST), pd.Timestamp("2025-06-01 00:00", tz=KST))
sites = pd.read_csv(f"{D}/sites.csv")[["site_id", "current_group", "capacity_kw", "lat", "lon", "wa_station", "wb_station"]]
gen = pd.read_parquet(f"{D}/train/generation.parquet"); gen["ts"] = pd.to_datetime(gen.ts)
fa = pd.read_parquet(f"{D}/train/forecast_wa.parquet", columns=["wa_station", "round", "ts"] + B.WA_COLS)
fb = pd.read_parquet(f"{D}/train/forecast_wb.parquet", columns=["wb_station", "round", "ts"] + B.WB_COLS)
for d in (fa, fb): d["ts"] = pd.to_datetime(d.ts)
gtr = gen[(gen.ts >= B.TRAIN_FROM) & (gen.ts <= FIT_END)]
gho = gen[(gen.ts >= HO[0]) & (gen.ts <= HO[1])]
idx = pd.DatetimeIndex(sorted(gho.ts.unique()))
assert len(idx) == 2208
grp = sites.set_index("site_id").current_group.map({"A": "G1", "B": "G2"})
res = {}
preds = []
for obj in ("l2", "l1"):
    per_r = []
    for r in B.ROUNDS:
        d = B.features(gtr, sites, fa, fb, None, r); d = d[d.elev > 0]
        y = (d.energy / d.cap).clip(0, 1.2).astype("float32")
        m = lgb.LGBMRegressor(objective="regression" if obj == "l2" else "regression_l1", n_estimators=600, learning_rate=0.06,
                              num_leaves=127, min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                              verbose=-1, n_jobs=-1, random_state=0).fit(d[B.FEATS].astype("float32"), y)
        t = B.features(None, sites, fa, fb, idx, r)
        p = np.where(t.elev.to_numpy() > 0, m.predict(t[B.FEATS].astype("float32")), 0.0)
        t["kwh"] = np.clip(p, 0, None) * t.cap.to_numpy(); t["group"] = t.site_id.map(grp)
        per_r.append(t.groupby(["group", "ts"]).kwh.sum())
        print(obj, r, "done", flush=True)
    preds.append(pd.concat(per_r, axis=1).mean(axis=1).rename(obj))
P = pd.concat(preds, axis=1).reset_index()
act = gho.merge(sites[["site_id"]].assign(group=grp.values), on="site_id").groupby(["group", "ts"]).energy.sum().rename("actual").reset_index()
smp = pd.read_csv(f"{D}/train/smp.csv"); smp["ts"] = pd.to_datetime(smp.ts)
P = P.merge(act, on=["group", "ts"]).merge(smp, on="ts")
P["k"] = np.where(P.ts.dt.tz_convert(KST).dt.hour.between(11, 15), 3, 1)
w = 0.1 * P.smp * P.k
P["m"] = P.ts.dt.tz_convert(KST).dt.strftime("%Y-%m")
for c in ("l2", "l1"):
    pen = w * (P[c] - P.actual).abs()
    res[c] = {"total_won": float(pen.sum()), "by_month": {k: float(v) for k, v in pen.groupby(P.m).sum().items()},
              "bias_pct_smpk": float(100 * (P.smp * P.k * (P[c] - P.actual)).sum() / (P.smp * P.k * P.actual).sum())}
P.to_parquet(OUT + "_preds.parquet", index=False)
json.dump(res, open(OUT + ".json", "w"), indent=1); print(json.dumps(res, indent=1))
