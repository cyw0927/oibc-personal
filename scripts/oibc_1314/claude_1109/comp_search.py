"""Composition search on site-level OOF errors (proxy model), honest split: optimise on some months, score on others.

Usage: comp_search.py <dataset> <site_level_oof.parquet> <pred_col>
Penalty(x) = sum_t 0.1*SMP*k*(|sum_{i in G1} e_it| + |sum_{i in G2} e_it|), e = pred - actual (missing site-hours -> 0).
Constraint: each group capacity >= 20,000 kW. Greedy single-site moves from the reference composition.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

D, F, C = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
sites = pd.read_csv(D / "sites.csv")
cap = sites.set_index("site_id").capacity_kw
ids = list(sites.site_id)
x0 = (sites.current_group == "A").to_numpy()
s = pd.read_parquet(F)
s["e"] = s[C] - s.energy
E = s.pivot_table(index="ts", columns="site_id", values="e", aggfunc="first").reindex(columns=ids).fillna(0.0)
meta = s.groupby("ts").agg(smp=("smp", "first"), k=("k", "first"), fold=("fold", "first")).loc[E.index]
W = (0.1 * meta.smp * meta.k).to_numpy()
Em = E.to_numpy()
CAPV = cap.loc[ids].to_numpy(); TOT = CAPV.sum()


def pen(x, rows):
    s1 = Em[rows] @ x; s2 = Em[rows].sum(1) - s1
    return (W[rows] * (np.abs(s1) + np.abs(s2))).sum()


def search(rows, x, it=200):
    x = x.astype(float).copy()
    Er, Wr = Em[rows], W[rows]
    tot = Er.sum(1)
    for _ in range(it):
        s1 = Er @ x; s2 = tot - s1
        base = (Wr * (np.abs(s1) + np.abs(s2))).sum()
        sign = np.where(x > 0, -1.0, 1.0)                                  # moving site j changes s1 by sign_j * E_j
        n1 = s1[:, None] + Er * sign[None, :]; n2 = tot[:, None] - n1
        cost = (Wr[:, None] * (np.abs(n1) + np.abs(n2))).sum(0)
        c1 = CAPV @ x + sign * CAPV
        ok = (c1 >= 20000) & (TOT - c1 >= 20000)
        cost[~ok] = np.inf
        j = int(np.argmin(cost))
        if cost[j] >= base - 1e-6:
            break
        x[j] = 1 - x[j]
    return x


folds = sorted(meta.fold.unique())
print("folds", folds, "hours", len(meta))
ref_all = pen(x0.astype(float), np.arange(len(W)))
print("reference penalty (all folds) M", round(ref_all / 1e6, 3), "| total-sum floor M", round((W * np.abs(Em.sum(1))).sum() / 1e6, 3))
for tr_f, te_f in ((folds[:6], folds[6:]), (folds[6:], folds[:6]), (folds[::2], folds[1::2]), (folds[1::2], folds[::2])):
    tr = np.where(meta.fold.isin(tr_f))[0]; te = np.where(meta.fold.isin(te_f))[0]
    x = search(tr, x0)
    moved = int((x != x0).sum())
    a, b = pen(x0.astype(float), te), pen(x, te)
    print(f"opt on {tr_f[0]}..({len(tr_f)}) moved {moved}: train {pen(x0.astype(float), tr)/1e6:.3f}->{pen(x, tr)/1e6:.3f}M | held-out {a/1e6:.3f}->{b/1e6:.3f}M ({100*(b/a-1):+.2f}%) G1 cap {CAPV @ x:.0f}")

if len(sys.argv) > 4:                                                     # final: optimise on all months, write composition
    xa = search(np.arange(len(W)), x0)
    print("ALL months: moved", int((xa != x0).sum()), f"{ref_all/1e6:.3f}->{pen(xa, np.arange(len(W)))/1e6:.3f}M", "G1 cap", round(CAPV @ xa), "G2 cap", round(TOT - CAPV @ xa))
    pd.DataFrame({"site_id": ids, "group": np.where(xa > 0, "G1", "G2")}).to_csv(sys.argv[4], index=False, lineterminator="\n")
