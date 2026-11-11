"""12 monthly expanding-window folds (2024-03..2025-02) for the group-direct model: base vs +day-context features.
Pre-HOLDOUT only. Metric: README provisional group penalty on available-cohort (pred = cf * cap_avail)."""
import sys, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
FEAT, D = sys.argv[1], sys.argv[2]; KST = "Asia/Seoul"
ft = pd.read_parquet(f"{FEAT}/group_features_train.parquet")
ft = ft[ft.ts <= pd.Timestamp("2025-02-28 00:00", tz=KST)].copy()
smp = pd.read_csv(f"{D}/train/smp.csv"); smp["ts"] = pd.to_datetime(smp.ts).dt.tz_convert(KST)
ft["ts"] = ft.ts.dt.tz_convert(KST); smp["ts"] = smp.ts.astype(ft.ts.dtype)
ft = ft.merge(smp, on="ts", how="left"); ft["k"] = np.where(ft.ts.dt.hour.between(11, 15), 3, 1)
ft["cov"] = ft.cap_avail / ft.group_cap_kw; ft["cf"] = (ft.energy / ft.cap_avail).clip(0, 1.2)
NON = {"ts", "group", "energy", "cap_avail", "n_sites", "group_cap_kw", "cf", "smp", "k", "cov"}
base = [c for c in ft.columns if c not in NON]

# day-context features: same D-1 issued forecast, other hours of the target day / previous day (issued D-2)
ft = ft.sort_values(["group", "ts"]).reset_index(drop=True)
day = (ft.ts - pd.Timedelta(hours=1)).dt.normalize()
ft["day"] = day
V = ["wa_cloud", "wb_cloud", "cs_ab", "wb_uv_idx", "wb_precip_prob", "wb_vis", "wb_rel_hum", "wa_humidity"]
new = []
g = ft.groupby("group")
for v in V:
    for s in (-2, -1, 1, 2):
        c = f"{v}_s{s}"; ft[c] = g[v].shift(s); new.append(c)
lit = ft.cosz > 0.05
for v in V:
    dl = ft[v].where(lit).groupby([ft.group, ft.day])
    for agg in ("mean", "min", "max", "std"):
        c = f"{v}_dl{agg}"; ft[c] = dl.transform(agg); new.append(c)
# previous-day daylight means (forecast for D-1 issued on D-2 -> available)
for v in ("wa_cloud", "wb_cloud", "wa_snow", "wb_snow", "wb_precip_prob"):
    lit = ft.cosz > 0.05
    dm = (ft[v].where(lit) if ("cloud" in v or "prob" in v) else ft[v]).groupby([ft.group, ft.day]).mean().rename("x").reset_index()
    dm["day"] = dm["day"] + pd.Timedelta(days=1)
    ft = ft.merge(dm.rename(columns={"x": f"{v}_prevday"}), on=["group", "day"], how="left"); new.append(f"{v}_prevday")
ft["cs_rel"] = ft.cs_ab / ft.cosz.where(ft.cosz > 0.02); new.append("cs_rel")
ds = ft.groupby([ft.group, ft.day]).smp
ft["smp_f"] = ft.smp; ft["smp_rel"] = ft.smp / ds.transform("mean").clip(lower=1); ft["smp_dmin"] = ds.transform("min"); ft["smp_dmean"] = ds.transform("mean")
ft["smp_mid_minus_9"] = ft.groupby([ft.group, ft.day]).smp.transform(lambda x: x.iloc[11:15].mean() - x.iloc[7:9].mean() if len(x) >= 15 else np.nan)
SMPF = ["smp_f", "smp_rel", "smp_dmin", "smp_dmean", "smp_mid_minus_9"]

CFG = dict(learning_rate=0.05, max_iter=600, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0)
def fit_pred(tr, te, feats):
    sw = tr["cov"] * tr.k * tr.smp.clip(lower=1) / tr.smp.clip(lower=1).mean()
    m = HistGradientBoostingRegressor(loss="absolute_error", early_stopping=False, random_state=0, **CFG)
    m.fit(tr[feats], tr.cf, sample_weight=sw)
    return np.where(te.cosz_max > 0, np.clip(m.predict(te[feats]), 0, 1), 0) * te.cap_avail.values
res = []
for M in pd.period_range("2024-03", "2025-02", freq="M"):
    st = pd.Timestamp(M.start_time, tz=KST) + pd.Timedelta(hours=1); en = pd.Timestamp(M.end_time.normalize() + pd.Timedelta(days=1), tz=KST)
    tr = ft[(ft.ts < st) & ft.energy.notna() & (ft["cov"] >= 0.3) & (ft.cosz_max > 0)]
    te = ft[(ft.ts >= st) & (ft.ts <= en) & ft.energy.notna() & (ft["cov"] >= 0.3)]
    w = 0.1 * te.smp * te.k
    r = {"month": str(M), "cov": round(te["cov"].mean(), 2)}
    for name, feats in (("base", base), ("ctx", base + SMPF)):
        p = fit_pred(tr, te, feats); r[name] = (w * (p - te.energy).abs()).sum() / 1e6
        r[name + "_bias%"] = 100 * ((te.smp * te.k * (p - te.energy)).sum() / (te.smp * te.k * te.energy).sum())
    res.append(r); print(r, flush=True)
R = pd.DataFrame(res); R["chg%"] = 100 * (R.ctx / R.base - 1)
print(R.round(3).to_string()); print("TOTAL base %.3f ctx %.3f (%.2f%%), improved months %d/12" % (R.base.sum(), R.ctx.sum(), 100 * (R.ctx.sum() / R.base.sum() - 1), (R.ctx < R.base).sum()))
