"""Rolling-fold evaluation of v6 (reference composition) vs v7 (optimised composition) from site-level OOF.

Usage: eval_v7.py <dataset> <personal_safe> <comp.csv> <out_csv>
Site-level OOF: E (site_level_E_5f), B21 (site_level_B21_12m), day models (site_level_day23 / day21, seeds 0,1);
group-level H1: group_cv.parquet (reference) / group_cv_comp.parquet (new composition).
Group sums over sites present in each hour (same convention as all earlier CV).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

D, PS, COMP, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
DUMP = Path(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] != "-" else None
DSUF = sys.argv[6] if len(sys.argv) > 6 else ""                     # optional day-model file suffix, e.g. _ec          # optional: write group-hour predictions per composition
KST = "Asia/Seoul"
FOLDS = ["2024-07", "2024-10", "2025-01", "2025-02", "2025-04"]
cv = PS / "cv"
sites = pd.read_csv(D / "sites.csv")
GRP = {"ref": sites.set_index("site_id").current_group.map({"A": "G1", "B": "G2"}),
       "comp": pd.read_csv(COMP).set_index("site_id").group}


def load(f, cols):
    x = pd.read_parquet(cv / f); x["ts"] = pd.to_datetime(x.ts).dt.tz_convert(KST)
    return x[x.fold.astype(str).isin(FOLDS)][["site_id", "ts", "fold"] + cols]


e = load("site_level_E_5f.parquet", ["energy", "smp", "k", "E"])
b = load("site_level_B21_12m.parquet", ["B"]).rename(columns={"B": "B21"})
d23 = load(f"site_level_day23{DSUF}.parquet", ["B", "D_s0", "D_s1"]).rename(columns={"B": "h23", "D_s0": "d23a", "D_s1": "d23b"})
d21 = load(f"site_level_day21{DSUF}.parquet", ["B", "D_s0", "D_s1"]).rename(columns={"B": "h21", "D_s0": "d21a", "D_s1": "d21b"})
s = e.merge(b, on=["site_id", "ts", "fold"]).merge(d23, on=["site_id", "ts", "fold"]).merge(d21, on=["site_id", "ts", "fold"])
print("site rows", len(s), {n: len(x) for n, x in (("E", e), ("B21", b), ("d23", d23), ("d21", d21))})
s["day"] = (s.ts - pd.Timedelta(hours=1)).dt.normalize()
res = []
for comp, H in (("ref", "group_cv.parquet"), ("comp", "group_cv_comp.parquet")):
    x = s.assign(group=s.site_id.map(GRP[comp]))
    g = x.groupby(["fold", "group", "ts"]).agg(day=("day", "first"), energy=("energy", "sum"), smp=("smp", "first"), k=("k", "first"),
                                               E=("E", "sum"), B21=("B21", "sum"), h23=("h23", "sum"), d23a=("d23a", "sum"), d23b=("d23b", "sum"),
                                               h21=("h21", "sum"), d21a=("d21a", "sum"), d21b=("d21b", "sum")).reset_index()
    gd = g.groupby(["fold", "group", "day"])[["h23", "d23a", "d23b", "h21", "d21a", "d21b"]].sum()
    r = lambda dcol, hcol: (gd[dcol] / gd[hcol].where(gd[hcol] > 1)).fillna(1.0)
    gd["rd"] = (r("d23a", "h23") + r("d23b", "h23") + r("d21a", "h21") + r("d21b", "h21")) / 4
    g = g.merge(gd[["rd"]].reset_index(), on=["fold", "group", "day"])
    h = pd.read_parquet(cv / H); h["ts"] = pd.to_datetime(h.ts).dt.tz_convert(KST)
    g = g.merge(h[["group", "ts", "H1"]], on=["group", "ts"], how="inner")
    g["v4b"] = 0.35 * g.E + 0.35 * g.B21 + 0.30 * g.H1
    g["v6"] = g.v4b * (1 + 0.5 * (g.rd - 1))
    w = 0.1 * g.smp * g.k
    for c in ("E", "B21", "H1", "v4b", "v6"):
        g[f"p_{c}"] = w * (g[c] - g.energy).abs()
    if DUMP is not None:
        z = g.sort_values(["fold", "group", "ts"]).copy()
        z["ts"] = z.ts.dt.strftime("%Y-%m-%d %H:%M:%S%z").str.replace(r"(\+\d{2})(\d{2})$", r"\1:\2", regex=True)
        z["day"] = z.day.dt.strftime("%Y-%m-%d")
        z = z[["fold", "group", "ts", "day", "energy", "smp", "k", "E", "B21", "H1", "rd", "v4b", "v6"]].rename(columns={"v4b": "blend_before_day", "v6": "final_pred"})
        z.to_csv(DUMP / f"fold_predictions_{comp}.csv", index=False, float_format="%.6f", lineterminator="\n")
    t = g.groupby("fold")[[f"p_{c}" for c in ("E", "B21", "H1", "v4b", "v6")]].sum() / 1e6
    t.loc["TOTAL"] = t.sum(); t.insert(0, "rows", list(g.groupby("fold").size()) + [len(g)]); t.insert(0, "composition", comp)
    res.append(t)
R = pd.concat(res).round(4)
R.to_csv(OUT)
print(R.to_string())
a, b_ = R[R.composition == "ref"].loc["TOTAL", "p_v6"], R[R.composition == "comp"].loc["TOTAL", "p_v6"]
print(f"v6(ref) {a:.3f}M -> v7(comp) {b_:.3f}M ({100 * (b_ / a - 1):+.2f}%)")
