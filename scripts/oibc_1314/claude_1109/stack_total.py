"""Second-stage residual correction of the v11 group-hour forecast, honest rolling origin.

Usage: OIBC_HSUF=_gx stack_total.py <personal_safe> <comp.csv> <gfs_hourly.parquet> [day_suffix=_rich] [site_suffix=_gx]
v11 = (0.05 E + 0.60 B21 + 0.35 H1) * (1 + 0.3 (rd - 1)) per group-hour (12-month OOF).
Meta model: LightGBM L1 on ratio y/p (clipped) with features p, components, rd, hour, season, group, fixed-box GFS gx_*.
For each test month m (>= 4th month), fit only on OOF months < m; out = p * (1 + s * (ratio_hat - 1)), s in {0.25, 0.5, 1}.
Pre-declared rule: total over evaluated months <= -0.5 % vs v11 and better in >= 2/3 of months.
"""
import os, sys
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd

PS, COMP, GX = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
DS = sys.argv[4] if len(sys.argv) > 4 else "_rich"; SS = sys.argv[5] if len(sys.argv) > 5 else "_gx"
KST = "Asia/Seoul"; cv = PS / "cv"
grp = pd.read_csv(COMP).set_index("site_id").group


def load(fs, cols):
    out = []
    for f in fs:
        x = pd.read_parquet(cv / f); x["ts"] = pd.to_datetime(x.ts).dt.tz_convert(KST); x["fold"] = x.fold.astype(str)
        out.append(x[["site_id", "ts", "fold"] + cols])
    return pd.concat(out, ignore_index=True)


e = load([f"site_level_E_5f{SS}.parquet", f"site_level_E{SS}_7f.parquet"], ["energy", "smp", "k", "E"])
b = load([f"site_level_B21_12m{SS}.parquet"], ["B"]).rename(columns={"B": "B21"})
d23 = load([f"site_level_day23{DS}.parquet", f"site_level_day23{DS}_7f.parquet"], ["B", "D_s0", "D_s1"]).rename(columns={"B": "h23", "D_s0": "d23a", "D_s1": "d23b"})
d21 = load([f"site_level_day21{DS}.parquet", f"site_level_day21{DS}_7f.parquet"], ["B", "D_s0", "D_s1"]).rename(columns={"B": "h21", "D_s0": "d21a", "D_s1": "d21b"})
k = ["site_id", "ts", "fold"]
s = e.merge(b, on=k).merge(d23, on=k).merge(d21, on=k)
s["day"] = (s.ts - pd.Timedelta(hours=1)).dt.normalize(); s["group"] = s.site_id.map(grp)
g = s.groupby(["fold", "group", "ts"]).agg(day=("day", "first"), energy=("energy", "sum"), smp=("smp", "first"), k=("k", "first"),
                                           **{c: (c, "sum") for c in ("E", "B21", "h23", "d23a", "d23b", "h21", "d21a", "d21b")}).reset_index()
gd = g.groupby(["fold", "group", "day"])[["h23", "d23a", "d23b", "h21", "d21a", "d21b"]].sum()
r = lambda dc, hc: (gd[dc] / gd[hc].where(gd[hc] > 1)).fillna(1.0)
gd["rd"] = (r("d23a", "h23") + r("d23b", "h23") + r("d21a", "h21") + r("d21b", "h21")) / 4
g = g.merge(gd[["rd"]].reset_index(), on=["fold", "group", "day"])
h = pd.read_parquet(cv / f"group_cv_comp{os.environ.get('OIBC_HSUF', '')}.parquet"); h["ts"] = pd.to_datetime(h.ts).dt.tz_convert(KST)
g = g.merge(h[["group", "ts", "H1"]], on=["group", "ts"], how="inner")
gx = pd.read_parquet(GX); gx["ts"] = pd.to_datetime(gx.ts).dt.tz_convert(KST)
g = g.merge(gx, on="ts", how="left")
g["p"] = (0.05 * g.E + 0.60 * g.B21 + 0.35 * g.H1) * (1 + 0.3 * (g.rd - 1))
g["hour"] = g.ts.dt.hour; doy = g.ts.dt.dayofyear; g["ds"] = np.sin(2 * np.pi * doy / 365.25); g["dc"] = np.cos(2 * np.pi * doy / 365.25)
g["gi"] = (g.group == "G1").astype(int)
for c in ("E", "B21", "H1"):
    g[c + "_r"] = g[c] / g.p.where(g.p > 1)
g["w"] = 0.1 * g.smp * g.k
F = ["p", "rd", "hour", "ds", "dc", "gi", "E_r", "B21_r", "H1_r"] + [c for c in gx.columns if c != "ts"]
day = g.hour.between(6, 19) & (g.p > 50)
g["y"] = (g.energy / g.p.where(day)).clip(0, 3)
months = sorted(g.fold.unique()); SH = (0.25, 0.5, 1.0)
rows = []
for i, m in enumerate(months):
    te = g.fold == m
    base = (g.w[te] * (g.p[te] - g.energy[te]).abs()).sum() / 1e6
    rec = {"month": m, "v11": base}
    if i >= 3:
        tr = g.fold.isin(months[:i]) & day & g.y.notna()
        mdl = lgb.LGBMRegressor(objective="regression_l1", n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=80,
                                subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0)
        mdl.fit(g.loc[tr, F], g.y[tr], sample_weight=(g.w * g.p)[tr])
        rh = np.where(day[te], mdl.predict(g.loc[te, F]), 1.0)
        for sh in SH:
            out = g.p[te] * (1 + sh * (rh - 1))
            rec[f"s{sh}"] = (g.w[te] * (out - g.energy[te]).abs()).sum() / 1e6
    rows.append(rec)
R = pd.DataFrame(rows).set_index("month").round(3)
print(R.to_string())
ev = R.dropna()
for sh in SH:
    c = f"s{sh}"
    print(c, "total", round(ev.v11.sum(), 3), "->", round(ev[c].sum(), 3), f"{100 * (ev[c].sum() / ev.v11.sum() - 1):+.2f}%", "better", int((ev[c] < ev.v11).sum()), "/", len(ev))
