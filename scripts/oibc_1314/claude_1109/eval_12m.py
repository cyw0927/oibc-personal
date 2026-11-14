"""12-month re-tune of v7 blend weights (E, B21, H1) and day-correction share alpha, v7 composition.

Usage: eval_12m.py <dataset> <personal_safe> <comp.csv> <out_json> [day_suffix] [site_suffix]
(suffixes select v10-style OOF: e.g. _rich _gx; baseline is then the same files at 0.35/0.35/0.30, alpha 0.5)
Site OOF: 5 folds (site_level_E_5f, day23, day21) + 7 folds (*_7f), B21 12m; H1 group_cv_comp (12 m).
v7 = (0.35 E + 0.35 B21 + 0.30 H1) * (1 + 0.5 (rd - 1)).
Grid: wE, wB in 0..1 step 0.05 (wH = 1 - wE - wB >= 0), alpha in 0..1 step 0.1.
Honest estimate: leave-one-month-out (params picked on the other 11 months) and past-only (months before).
Pre-declared adoption rule: LOMO total <= -0.5 % vs v7 and better in >= 8 of 12 months.
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

D, PS, COMP, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
KST = "Asia/Seoul"
cv = PS / "cv"
DS = sys.argv[5] if len(sys.argv) > 5 else ""
SS = sys.argv[6] if len(sys.argv) > 6 else ""
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
print("site rows", len(s), "folds", sorted(s.fold.unique()), flush=True)
s["day"] = (s.ts - pd.Timedelta(hours=1)).dt.normalize()
s["group"] = s.site_id.map(grp)
g = s.groupby(["fold", "group", "ts"]).agg(day=("day", "first"), energy=("energy", "sum"), smp=("smp", "first"), k=("k", "first"),
                                           **{c: (c, "sum") for c in ("E", "B21", "h23", "d23a", "d23b", "h21", "d21a", "d21b")}).reset_index()
gd = g.groupby(["fold", "group", "day"])[["h23", "d23a", "d23b", "h21", "d21a", "d21b"]].sum()
r = lambda dc, hc: (gd[dc] / gd[hc].where(gd[hc] > 1)).fillna(1.0)
gd["rd"] = (r("d23a", "h23") + r("d23b", "h23") + r("d21a", "h21") + r("d21b", "h21")) / 4
g = g.merge(gd[["rd"]].reset_index(), on=["fold", "group", "day"])
h = pd.read_parquet(cv / f"group_cv_comp{__import__('os').environ.get('OIBC_HSUF', '')}.parquet"); h["ts"] = pd.to_datetime(h.ts).dt.tz_convert(KST)
g = g.merge(h[["group", "ts", "H1"]], on=["group", "ts"], how="inner")
w = (0.1 * g.smp * g.k).to_numpy()
months = sorted(g.fold.unique())
fi = g.fold.map({m: i for i, m in enumerate(months)}).to_numpy()
E, B, H, RD, Y = (g[c].to_numpy() for c in ("E", "B21", "H1", "rd", "energy"))


def pen_by_month(we, wb, a):
    p = (we * E + wb * B + (1 - we - wb) * H) * (1 + a * (RD - 1))
    return np.bincount(fi, weights=w * np.abs(p - Y), minlength=len(months)) / 1e6


grid = [(we, wb, a) for we, wb in itertools.product(np.round(np.arange(0, 1.0001, 0.05), 2), repeat=2) if we + wb <= 1.0001
        for a in np.round(np.arange(0, 1.0001, 0.1), 1)]
P = np.array([pen_by_month(*q) for q in grid])          # grid x months
base = pen_by_month(0.35, 0.35, 0.5)
best_all = int(P.sum(1).argmin())
lomo, past = np.zeros(len(months)), np.full(len(months), np.nan)
for i in range(len(months)):
    o = np.delete(np.arange(len(months)), i)
    lomo[i] = P[P[:, o].sum(1).argmin(), i]
    if i >= 3:
        past[i] = P[P[:, :i].sum(1).argmin(), i]
pm = ~np.isnan(past)
res = {"months": months, "rows": int(len(g)), "v7_by_month": base.round(3).tolist(), "v7_total": round(base.sum(), 3),
       "insample_best": dict(zip(("wE", "wB", "alpha"), map(float, grid[best_all]))), "insample_total": round(P[best_all].sum(), 3),
       "lomo_by_month": lomo.round(3).tolist(), "lomo_total": round(lomo.sum(), 3),
       "lomo_pct": round(100 * (lomo.sum() / base.sum() - 1), 2), "lomo_months_better": int((lomo < base - 1e-9).sum()),
       "past_months": [m for m, f in zip(months, pm) if f], "past_total": round(np.nansum(past), 3), "v7_on_past_months": round(base[pm].sum(), 3),
       "past_pct": round(100 * (np.nansum(past) / base[pm].sum() - 1), 2)}
res["adopt"] = bool(res["lomo_pct"] <= -0.5 and res["lomo_months_better"] >= 8)
Path(OUT).write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
