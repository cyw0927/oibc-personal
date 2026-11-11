"""Held-out check of the composition gain: for each fold, optimise the composition on B21 OOF of the OTHER 11 months only,
then score the site-level part of v6 (0.5E + 0.5B21, day-adjusted alpha 0.5) under reference vs that composition on the fold.
(H1 cannot be refitted per fold cheaply, so it is left out of this check.)
Usage: eval_v7_heldout.py <dataset> <personal_safe>
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

D, PS = Path(sys.argv[1]), Path(sys.argv[2])
KST = "Asia/Seoul"; cv = PS / "cv"
FOLDS = ["2024-07", "2024-10", "2025-01", "2025-02", "2025-04"]
sites = pd.read_csv(D / "sites.csv"); ids = list(sites.site_id)
x0 = (sites.current_group == "A").to_numpy().astype(float); CAPV = sites.capacity_kw.to_numpy(); TOT = CAPV.sum()
b = pd.read_parquet(cv / "site_level_B21_12m.parquet"); b["ts"] = pd.to_datetime(b.ts).dt.tz_convert(KST); b["e"] = b.B - b.energy
Em = b.pivot_table(index="ts", columns="site_id", values="e", aggfunc="first").reindex(columns=ids).fillna(0.0)
meta = b.groupby("ts").agg(smp=("smp", "first"), k=("k", "first"), fold=("fold", "first")).loc[Em.index]
W = (0.1 * meta.smp * meta.k).to_numpy(); Em = Em.to_numpy()


def search(rows, x, it=200):
    x = x.copy(); Er, Wr = Em[rows], W[rows]; tot = Er.sum(1)
    for _ in range(it):
        s1 = Er @ x; base = (Wr * (np.abs(s1) + np.abs(tot - s1))).sum()
        sign = np.where(x > 0, -1.0, 1.0)
        n1 = s1[:, None] + Er * sign[None, :]
        cost = (Wr[:, None] * (np.abs(n1) + np.abs(tot[:, None] - n1))).sum(0)
        c1 = CAPV @ x + sign * CAPV; cost[~((c1 >= 20000) & (TOT - c1 >= 20000))] = np.inf
        j = int(np.argmin(cost))
        if cost[j] >= base - 1e-6: break
        x[j] = 1 - x[j]
    return x


def load(f, cols):
    y = pd.read_parquet(cv / f); y["ts"] = pd.to_datetime(y.ts).dt.tz_convert(KST)
    return y[y.fold.astype(str).isin(FOLDS)][["site_id", "ts", "fold"] + cols]


s = load("site_level_E_5f.parquet", ["energy", "smp", "k", "E"]).merge(load("site_level_B21_12m.parquet", ["B"]).rename(columns={"B": "B21"}), on=["site_id", "ts", "fold"])
s = s.merge(load("site_level_day23.parquet", ["B", "D_s0", "D_s1"]).rename(columns={"B": "h23", "D_s0": "a", "D_s1": "b"}), on=["site_id", "ts", "fold"])
s = s.merge(load("site_level_day21.parquet", ["B", "D_s0", "D_s1"]).rename(columns={"B": "h21", "D_s0": "c", "D_s1": "d"}), on=["site_id", "ts", "fold"])
s["day"] = (s.ts - pd.Timedelta(hours=1)).dt.normalize()


def score(f, x):
    grp = pd.Series(np.where(x > 0, "G1", "G2"), index=ids)
    y = s[s.fold == f].assign(group=lambda t: t.site_id.map(grp))
    g = y.groupby(["group", "ts"]).agg(day=("day", "first"), energy=("energy", "sum"), smp=("smp", "first"), k=("k", "first"),
                                       E=("E", "sum"), B21=("B21", "sum"), h23=("h23", "sum"), a=("a", "sum"), b=("b", "sum"), h21=("h21", "sum"), c=("c", "sum"), d=("d", "sum")).reset_index()
    gd = g.groupby(["group", "day"])[["h23", "a", "b", "h21", "c", "d"]].sum()
    r = lambda n, h: (gd[n] / gd[h].where(gd[h] > 1)).fillna(1.0)
    gd["rd"] = (r("a", "h23") + r("b", "h23") + r("c", "h21") + r("d", "h21")) / 4
    g = g.merge(gd[["rd"]].reset_index(), on=["group", "day"])
    p = (0.5 * g.E + 0.5 * g.B21) * (1 + 0.5 * (g.rd - 1))
    return (0.1 * g.smp * g.k * (p - g.energy).abs()).sum() / 1e6


tot_r = tot_c = 0
for f in FOLDS:
    rows = np.where(meta.fold.astype(str) != f)[0]
    x = search(rows, x0)
    a, c = score(f, x0), score(f, x)
    tot_r += a; tot_c += c
    print(f"{f}: composition from other 11 months (moved {int((x != x0).sum())}) | site part ref {a:.3f} -> {c:.3f}M ({100*(c/a-1):+.2f}%)")
print(f"TOTAL held-out: {tot_r:.3f} -> {tot_c:.3f}M ({100*(tot_c/tot_r-1):+.2f}%)")
x_all = (pd.read_csv(PS / "comp_opt_B21_12m.csv").set_index("site_id").group.reindex(ids) == "G1").to_numpy().astype(float)
ia = sum(score(f, x0) for f in FOLDS); ic = sum(score(f, x_all) for f in FOLDS)
print(f"same site part with the final (all-12-month) composition, in-sample: {ia:.3f} -> {ic:.3f}M ({100*(ic/ia-1):+.2f}%)")
