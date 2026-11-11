"""Candidate 3: site-day total model + hourly shape from the hourly model. Rolling folds.

Usage: cv_day.py <dataset> <site_train.parquet> <out_parquet> <fold> [<fold> ...]
Hourly base  B: site model B (stacked rounds, mean of round predictions).
Day model  DAY: one row per site-day, irradiance(cosz)-weighted daylight means of every forecast column for round 17,
                round-17 hourly pattern 08..17 h of 5 key columns (Gala et al. 2016 daily-pattern model),
                round-10 values and revisions, day constants (context/previous-day/regional), sum cosz.
                Target kd = day energy / (capacity * sum cosz) on complete days, L1, weight capacity*sum cosz.
Arms: B, B_D = B * (DAY / sum_day B) per site-day, B_M = B * (0.5 * ratio + 0.5).
Output group-hour sums of the three arms (so the same day ratio can be applied to v4b afterwards).
"""
import sys
from pathlib import Path

import warnings
warnings.simplefilter("ignore")

import lightgbm as lgb
import numpy as np
import pandas as pd

D, TRAIN, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
FOLDS = [a for a in sys.argv[4:] if a != "perf" and not a.startswith("perflag=")]
PERF = "perf" in sys.argv[4:]
PERFLAG = next((int(a[8:]) for a in sys.argv[4:] if a.startswith("perflag=")), 0)   # test rows use perf frozen N months before the fold
KST = "Asia/Seoul"
DROP = {"site_id", "ts", "energy", "wa_station", "wb_station", "current_group", "smp", "k", "cf", "w", "group", "day"}
PARAMS = dict(objective="regression_l1", n_estimators=800, learning_rate=0.05, num_leaves=127, min_child_samples=200,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=-1, random_state=0)
DPARAMS = dict(objective="regression_l1", n_estimators=1500, learning_rate=0.03, num_leaves=63, min_child_samples=50,
               subsample=0.8, subsample_freq=1, colsample_bytree=0.7, verbose=-1, n_jobs=-1, random_state=0)
HOURLY = ["wa_cloud", "wa_temp", "wa_humidity", "wa_rain", "wa_wind_speed", "wa_snow", "wb_cloud", "wb_temp", "wb_rel_hum",
          "wb_uv_idx", "wb_precip_prob", "wb_total_liq", "wb_vis", "wb_dew_point", "wb_ceiling", "wb_snow", "cs_wa", "cs_wb"]

d = pd.read_parquet(TRAIN)
d["ts"] = d.ts.dt.tz_convert(KST)
smp = pd.read_csv(D / "train" / "smp.csv")
smp["ts"] = pd.to_datetime(smp.ts).dt.tz_convert(KST)
d = d.merge(smp, on="ts", how="left")
d["k"] = np.where(d.ts.dt.hour.between(11, 15), 3, 1).astype("int8")
d["cf"] = (d.energy / d.capacity_kw).clip(0, 1.2).astype("float32")
d["group"] = d.current_group.map({"A": "G1", "B": "G2"})
d["day"] = (d.ts - pd.Timedelta(hours=1)).dt.normalize()
ALL = [c for c in d.columns if c not in DROP]
CONST = [c for c in d.columns if c.endswith(("_dmean", "_dmin", "_dmax", "_prev", "_reg", "_regd"))] + ["doy_sin", "doy_cos", "capacity_kw", "lat", "lon", "site_num"]

# ---- site-day table
lit = d[d.elev > 0].copy()
lit["cw"] = lit.cosz.clip(lower=0)
for c in HOURLY:
    lit[f"{c}_x"] = lit[c] * lit.cw
mid = lit[lit.hour.between(11, 15)]
agg = lit.groupby(["site_id", "day", "round"]).agg(csum=("cw", "sum"), nlit=("cw", "size"), **{f"{c}_w": (f"{c}_x", "sum") for c in HOURLY},
                                                     **{c: (c, "first") for c in CONST})
for c in HOURLY:
    agg[f"{c}_w"] = (agg[f"{c}_w"] / agg.csum).astype("float32")
mm = mid.groupby(["site_id", "day", "round"]).agg(wa_cloud_mid_max=("wa_cloud", "max"), wb_cloud_mid_max=("wb_cloud", "max"),
                                                 wa_cloud_mid_min=("wa_cloud", "min"), wb_cloud_mid_min=("wb_cloud", "min"))
agg = agg.join(mm).reset_index()
VW = [f"{c}_w" for c in HOURLY] + list(mm.columns)
a17 = agg[agg["round"] == 17].drop(columns="round")
a10 = agg[agg["round"] == 10][["site_id", "day"] + VW].rename(columns={c: f"{c}_r10" for c in VW})
day = a17.merge(a10, on=["site_id", "day"], how="left")
# whole-day NWP pattern as one input vector (Gala et al. 2016): round-17 hourly values 08..17 h as separate columns
PATV = ["wa_cloud", "wb_cloud", "wb_uv_idx", "wb_precip_prob", "cs_wa"]
pt = d[(d["round"] == 17) & d.hour.between(8, 17)].pivot_table(index=["site_id", "day"], columns="hour", values=PATV)
pt.columns = [f"{v}_h{h:02d}" for v, h in pt.columns]
day = day.merge(pt.astype("float32").reset_index(), on=["site_id", "day"], how="left")
for c in VW:
    day[f"{c}_rev"] = day[c] - day[f"{c}_r10"]
obs = d[(d.elev > 0) & (d["round"] == 17)].groupby(["site_id", "day"]).agg(esum=("energy", "sum"), n_e=("energy", "count"))
day = day.merge(obs.reset_index(), on=["site_id", "day"], how="left")
day["complete"] = day.n_e == day.nlit
day["kd"] = (day.esum / (day.capacity_kw * day.csum)).clip(0, 1.5)
day["doy"] = day.day.dt.dayofyear
if PERF:
    # kim2017-style site loss factor: site energy / (capacity x fleet cf at the same hour) over windows before each month start
    o = d[(d["round"] == 17) & (d.elev > 0) & d.energy.notna()][["site_id", "ts", "energy", "capacity_kw"]]
    fl = o.groupby("ts").agg(fe=("energy", "sum"), fc=("capacity_kw", "sum"))
    o = o.join((fl.fe / fl.fc).rename("fcf"), on="ts")
    o["exp"] = o.capacity_kw * o.fcf
    rows = []
    for ms in pd.date_range(day.day.min().normalize(), day.day.max() + pd.offsets.MonthBegin(1), freq="MS", tz=KST):
        cut = ms + pd.Timedelta(hours=1)
        cols = {}
        for nd in (90, 365):
            w_ = o[(o.ts < cut) & (o.ts >= cut - pd.Timedelta(days=nd))].groupby("site_id")[["energy", "exp"]].sum()
            cols[f"perf{nd}"] = w_.energy / w_.exp.where(w_.exp > 0)
        rows.append(pd.DataFrame(cols).rename_axis("site_id").reset_index().assign(ms=ms))
    pf = pd.concat(rows, ignore_index=True)
    pf["perf_chg"] = pf.perf90 / pf.perf365
    day["ms"] = day.day.dt.to_period("M").dt.start_time.dt.tz_localize(KST)
    day = day.merge(pf, on=["site_id", "ms"], how="left").drop(columns="ms")
    print("perf features", day[["perf90", "perf365", "perf_chg"]].describe().round(3).to_dict(), flush=True)
DF = [c for c in day.columns if c not in ("site_id", "day", "esum", "n_e", "complete", "kd", "nlit")]
print("day rows", len(day), "complete", int(day.complete.sum()), "features", len(DF), flush=True)

dh = d[d.energy.notna()]
out = []
for M in FOLDS:
    P = pd.Period(M, freq="M")
    st = pd.Timestamp(P.start_time).tz_localize(KST) + pd.Timedelta(hours=1)
    en = pd.Timestamp(P.end_time.normalize() + pd.Timedelta(days=1)).tz_localize(KST)
    tr = dh[(dh.ts < st) & (dh.elev > 0)]
    te = d[(d.ts >= st) & (d.ts <= en)]                                   # all rows (the day sum of B needs every lit hour)
    m = lgb.LGBMRegressor(**PARAMS).fit(tr[ALL], tr.cf)
    p = np.where(te.elev > 0, np.clip(m.predict(te[ALL]), 0, None), 0.0) * te.capacity_kw.to_numpy()
    h = te[["site_id", "ts", "day", "group", "energy", "smp", "k"]].assign(B=p).groupby(["site_id", "ts"]).agg(
        day=("day", "first"), group=("group", "first"), energy=("energy", "first"), smp=("smp", "first"), k=("k", "first"), B=("B", "mean")).reset_index()
    dtr = day[(day.day < st.normalize()) & day.complete & (day.csum > 0)]
    dte = day[(day.day >= st.normalize()) & (day.day <= en.normalize() - pd.Timedelta(days=1))]
    if PERF and PERFLAG:
        msl = (P.start_time - pd.DateOffset(months=PERFLAG)).tz_localize(KST)
        fz = pf[pf.ms == msl][["site_id", "perf90", "perf365", "perf_chg"]]
        dte = dte.drop(columns=["perf90", "perf365", "perf_chg"]).merge(fz, on="site_id", how="left")
    md = lgb.LGBMRegressor(**DPARAMS).fit(dtr[DF], dtr.kd, sample_weight=(dtr.capacity_kw * dtr.csum).to_numpy())
    dp = dte[["site_id", "day"]].assign(DAY=np.clip(md.predict(dte[DF]), 0, None) * (dte.capacity_kw * dte.csum).to_numpy())
    h = h.merge(dp, on=["site_id", "day"], how="left")
    sb = h.groupby(["site_id", "day"]).B.transform("sum")
    ratio = (h.DAY / sb.where(sb > 0)).fillna(1.0).clip(0.2, 5)
    h["B_D"] = h.B * ratio
    h["B_M"] = h.B * (0.5 * ratio + 0.5)
    h = h[h.energy.notna()]
    A = ["B", "B_D", "B_M"]
    g = h.groupby(["group", "ts"]).agg(energy=("energy", "sum"), smp=("smp", "first"), k=("k", "first"), **{a: (a, "sum") for a in A}).reset_index()
    g["fold"] = M
    ww = 0.1 * g.smp * g.k
    imp = pd.Series(md.feature_importances_, index=DF).sort_values(ascending=False).head(6).to_dict()
    print(M, len(dtr), {a: round((ww * (g[a] - g.energy).abs()).sum() / 1e6, 3) for a in A}, "day top", imp, flush=True)
    out.append(g)
pd.concat(out).to_parquet(OUT, index=False)
