"""Error diagnosis of v4b (0.35E + 0.35B21 + 0.30H1) on the 5 rolling folds, group-hour level."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PS = Path(sys.argv[1])
cv = PS / "cv"
K = ["group", "ts"]
e = pd.concat([pd.read_parquet(cv / "site_cv_EG.parquet")[K + ["fold", "energy", "smp", "k", "E"]],
               pd.read_parquet(cv / "site_cv_E_feb.parquet")[K + ["fold", "energy", "smp", "k", "E"]]])
b = pd.concat([pd.read_parquet(cv / "site_cv_B2021.parquet"), pd.read_parquet(cv / "site_cv_B2021_feb.parquet")])[K + ["B"]]
h = pd.read_parquet(cv / "group_cv.parquet")[K + ["H1", "cap_avail"]]
for x in (e, b, h):
    x["ts"] = pd.to_datetime(x.ts).dt.tz_convert("Asia/Seoul")
d = e.merge(b, on=K).merge(h, on=K)
d["V"] = 0.35 * d.E + 0.35 * d.B + 0.30 * d.H1
d["w"] = 0.1 * d.smp * d.k
for c in ("V", "E", "B", "H1"):
    d[f"p_{c}"] = d.w * (d[c] - d.energy).abs()
d["err"] = d.V - d.energy
tot = d.p_V.sum()
print("rows", len(d), "v4b total M", round(tot / 1e6, 3), {c: round(d[f'p_{c}'].sum() / 1e6, 3) for c in ("E", "B", "H1")})

g = pd.read_parquet(PS / "feat_full_20261109" / "group_features_train.parquet")
g["ts"] = pd.to_datetime(g.ts).dt.tz_convert("Asia/Seoul")
g = g.rename(columns={"group_id": "gid"})
gcol = "group" if "group" in g.columns else "gid"
d = d.merge(g[["ts", gcol, "wa_cloud", "wb_cloud", "wb_cloud_daymean", "wa_cloud_daymean", "cs_ab", "wb_precip_prob",
               "wa_rain", "wa_snow", "wb_snow", "cosz", "group_cap_kw"]].rename(columns={gcol: "group"}), on=K, how="left")
d["hour"] = d.ts.dt.hour
d["day"] = (d.ts - pd.Timedelta(hours=1)).dt.normalize()
d["mon"] = d.ts.dt.month


def tab(by, name=None):
    t = d.groupby(by).agg(n=("p_V", "size"), pen=("p_V", "sum"), bias=("err", "mean"), mae=("err", lambda s: s.abs().mean()),
                          act=("energy", "mean"), E=("p_E", "sum"), B=("p_B", "sum"), H1=("p_H1", "sum"))
    t["share%"] = 100 * t.pen / tot
    t["bias%act"] = 100 * t.bias / t.act.clip(lower=1)
    for c in ("pen", "E", "B", "H1"):
        t[c] = t[c] / 1e6
    print(f"\n== by {name or by}\n", t.round(3).to_string())


tab("fold"); tab("group"); tab("hour")
d["cloud_bin"] = pd.cut((d.wa_cloud_daymean + d.wb_cloud_daymean) / 2, [-1, 20, 40, 60, 80, 101])
tab("cloud_bin", "day-mean cloud (forecast)")
d["rain"] = np.where((d.wa_rain > 0) | (d.wb_precip_prob > 50), "rainy", "dry")
tab("rain")
d["snow"] = np.where((d.wa_snow > 0) | (d.wb_snow > 0), "snow", "none"); tab("snow")

# day-level: how much is a whole-day level error?
dd = d.groupby(["group", "day"]).agg(pen=("p_V", "sum"), err_sum=("err", "sum"), act=("energy", "sum"), V=("V", "sum"))
dd["ratio"] = dd.act / dd.V.clip(lower=1)
dd = dd.sort_values("pen", ascending=False)
c = dd.pen.cumsum() / tot
print("\n== concentration: days", len(dd), "top10% days share", round(100 * dd.pen.head(len(dd) // 10).sum() / tot, 1),
      "% ; top 20 days share", round(100 * dd.pen.head(20).sum() / tot, 1), "%")
print(dd.head(15).assign(pen=lambda x: x.pen / 1e6).round(3).to_string())

# oracle: per group-day scale (best L1 scale) -> upper bound of a day-level correction
def best_scale(x):
    r = x.energy / x.V.where(x.V > 0)
    r = r.dropna(); wts = (x.w * x.V).loc[r.index]
    if len(r) == 0: return 1.0
    o = np.argsort(r.values); cw = np.cumsum(wts.values[o]); return r.values[o][np.searchsorted(cw, cw[-1] / 2)]
sc = d.groupby(["group", "day"]).apply(best_scale, include_groups=False).rename("s")
d = d.merge(sc, left_on=["group", "day"], right_index=True)
print("\n== oracle day-scale penalty M", round((d.w * (d.V * d.s - d.energy).abs()).sum() / 1e6, 3), "vs", round(tot / 1e6, 3))
# oracle hour-shape with true daily total
dt = d.groupby(["group", "day"]).energy.transform("sum") / d.groupby(["group", "day"]).V.transform("sum").clip(lower=1)
print("== oracle true daily total (shape kept) M", round((d.w * (d.V * dt - d.energy).abs()).sum() / 1e6, 3))

# zero/outage-like hours: actual << forecast at bright hours
d["ratio_h"] = d.energy / d.V.clip(lower=1)
m = (d.cosz > 0.3)
for lo, hi in [(0, 0.3), (0.3, 0.7), (0.7, 1.3), (1.3, 2), (2, 99)]:
    s = m & d.ratio_h.between(lo, hi, inclusive="left")
    print(f"act/pred in [{lo},{hi}) n={s.sum()} share={100 * d.p_V[s].sum() / tot:.1f}%")

# signed bias by regime: over- or under-forecast?
print("\n== over vs under forecast share: over", round(100 * d.p_V[d.err > 0].sum() / tot, 1), "% under", round(100 * d.p_V[d.err < 0].sum() / tot, 1), "%")
# regime-dependent best component
for cb, x in d.groupby("cloud_bin", observed=True):
    print("cloud", cb, {c: round(x[f'p_{c}'].sum() / 1e6, 2) for c in ("V", "E", "B", "H1")})
d.drop(columns=["cloud_bin"]).to_parquet(sys.argv[2], index=False)
