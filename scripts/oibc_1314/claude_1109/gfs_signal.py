"""Signal check: does GFS D-1 00Z f030 (09-15 KST) radiation/cloud predict v7's group-day error? LOMO linear day scale.
Usage: gfs_signal.py <sites.csv> <comp.csv> <fold_predictions_comp.csv> <gfs.parquet>
"""
import sys
import numpy as np, pandas as pd
S, C, FP, G = sys.argv[1:5]
s = pd.read_csv(S)[["site_id", "capacity_kw"]].merge(pd.read_csv(C), on="site_id")
g = pd.read_parquet(G).merge(s, on="site_id")
g["day"] = pd.to_datetime(g.day).dt.strftime("%Y-%m-%d")
gg = g.groupby(["group", "day"]).apply(lambda x: pd.Series({"dsw": np.average(x.dswrf, weights=x.capacity_kw),
                                                           "tcc": np.average(x.tcdc, weights=x.capacity_kw)}), include_groups=False).reset_index()
f = pd.read_csv(FP)
f["w"] = 0.1 * f.smp * f.k
f["pen"] = f.w * (f.final_pred - f.energy).abs()
pk = f[f.k == 3].groupby(["fold", "group", "day"]).agg(A=("energy", "sum"), F=("final_pred", "sum")).reset_index()
d = pk.merge(gg, on=["group", "day"]).query("F > 0 and A > 0")
d["y"] = np.log(d.A / d.F)
d["x1"] = np.log(d.dsw.clip(lower=5)) - np.log(d.F)
d["x2"] = d.tcc / 100
print("group-days", len(d), "corr(y,x1)", round(d.y.corr(d.x1), 3), "corr(y,tcc)", round(d.y.corr(d.x2), 3))
for fo, x in d.groupby("fold"):
    print(" ", fo, "corr x1", round(x.y.corr(x.x1), 3), "tcc", round(x.y.corr(x.x2), 3))
base = f.pen.sum() / 1e6
for cols in (["x1"], ["x1", "x2"]):
    for lam in (1.0, 0.5):
        sc = {}
        for fo in d.fold.unique():
            tr, te = d[d.fold != fo], d[d.fold == fo]
            X = np.c_[np.ones(len(tr)), tr[cols]]; b = np.linalg.lstsq(X, tr.y, rcond=None)[0]
            p = np.c_[np.ones(len(te)), te[cols]] @ b
            for (gr, dy), v in zip(te[["group", "day"]].itertuples(index=False), p):
                sc[(gr, dy)] = np.exp(lam * np.clip(v, -0.3, 0.3))
        m = f[["group", "day"]].apply(tuple, axis=1).map(sc).fillna(1.0)
        pen = (f.w * (f.final_pred * m - f.energy).abs())
        byf = pen.groupby(f.fold).sum() / 1e6; b0 = f.pen.groupby(f.fold).sum() / 1e6
        print(cols, "lam", lam, f"v7 {base:.3f} -> {pen.sum()/1e6:.3f}M ({100*(pen.sum()/1e6/base-1):+.2f}%)",
              "months better", int((byf < b0).sum()), (100 * (byf / b0 - 1)).round(1).to_dict())
# skill of each source alone for peak-hour day energy, within month (Pearson of logs)
print("within-month corr with actual: GFS dswrf vs v7 forecast vs provider-free tcc")
for fo, x in d.groupby("fold"):
    la = np.log(x.A)
    print(" ", fo, "GFS", round(la.corr(np.log(x.dsw.clip(lower=5))), 3), "v7", round(la.corr(np.log(x.F)), 3))
