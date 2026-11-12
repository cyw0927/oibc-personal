"""Untouched-month check (7 months never used for selection): v7 vs a day-model variant, v7 composition, alpha 0.5.
Usage: eval_holdout7.py <personal_safe> <comp.csv> <variant_suffix>   (site_level_day23<suf>_7f / day21<suf>_7f vs *_7f)
"""
import sys
from pathlib import Path
import pandas as pd
PS, COMP, SUF = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
cv, K = PS / "cv", "Asia/Seoul"
grp = pd.read_csv(COMP).set_index("site_id").group


def load(f, cols):
    x = pd.read_parquet(cv / f); x["ts"] = pd.to_datetime(x.ts).dt.tz_convert(K); x["fold"] = x.fold.astype(str)
    return x[["site_id", "ts", "fold"] + cols]


def run(s23, s21):
    e = load("site_level_E_7f.parquet", ["energy", "smp", "k", "E"])
    b = load("site_level_B21_12m.parquet", ["B"]).rename(columns={"B": "B21"})
    k = ["site_id", "ts", "fold"]
    s = e.merge(b, on=k)
    for n, f in (("23", s23), ("21", s21)):
        s = s.merge(load(f, ["B", "D_s0", "D_s1"]).rename(columns={"B": f"h{n}", "D_s0": f"d{n}a", "D_s1": f"d{n}b"}), on=k)
    s["day"] = (s.ts - pd.Timedelta(hours=1)).dt.normalize(); s["group"] = s.site_id.map(grp)
    cols = ["E", "B21", "h23", "d23a", "d23b", "h21", "d21a", "d21b"]
    g = s.groupby(["fold", "group", "ts"]).agg(day=("day", "first"), energy=("energy", "sum"), smp=("smp", "first"), kk=("k", "first"), **{c: (c, "sum") for c in cols}).reset_index()
    gd = g.groupby(["fold", "group", "day"])[cols[2:]].sum()
    r = lambda d, h: (gd[d] / gd[h].where(gd[h] > 1)).fillna(1.0)
    gd["rd"] = (r("d23a", "h23") + r("d23b", "h23") + r("d21a", "h21") + r("d21b", "h21")) / 4
    g = g.merge(gd[["rd"]].reset_index(), on=["fold", "group", "day"])
    h = pd.read_parquet(cv / "group_cv_comp.parquet"); h["ts"] = pd.to_datetime(h.ts).dt.tz_convert(K)
    g = g.merge(h[["group", "ts", "H1"]], on=["group", "ts"])
    p = (0.35 * g.E + 0.35 * g.B21 + 0.3 * g.H1) * (1 + 0.5 * (g.rd - 1))
    return (0.1 * g.smp * g.kk * (p - g.energy).abs()).groupby(g.fold).sum() / 1e6, len(g)


a, n = run("site_level_day23_7f.parquet", "site_level_day21_7f.parquet")
b, m = run(f"site_level_day23{SUF}_7f.parquet", f"site_level_day21{SUF}_7f.parquet")
r = pd.DataFrame({"v7": a, "cand": b}); r["pct"] = (100 * (r.cand / r.v7 - 1)).round(2)
print(r.round(3).to_string()); print("rows", n, m, "total", round(a.sum(), 3), round(b.sum(), 3), f"{100 * (b.sum() / a.sum() - 1):+.2f}%", "months better", int((r.cand < r.v7).sum()), "/", len(r))
