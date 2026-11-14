"""Blend teammate OOF (scored.parquet Candidate, #990 groups) with our v10 OOF at total level, split by v10 group shares,
scored on our v7 composition (1e42864d) group-hours. Overlap months 2024-06..2025-02. Same ts rows only.
Usage: blend_mate.py <personal_safe> <comp.csv> <scored.parquet>
"""
import sys, importlib.util
import numpy as np, pandas as pd
PS, COMP, SC = sys.argv[1:4]
spec = importlib.util.spec_from_file_location("ds", __file__.replace("blend_mate.py", "diag_spread_lib.py")); ds = importlib.util.module_from_spec(spec); spec.loader.exec_module(ds)
g = ds.build(PS, COMP, "_rich", "_gx", "")
g["p"] = (0.35 * g.E + 0.35 * g.B21 + 0.30 * g.H1) * (1 + 0.5 * (g.rd - 1))
g["w"] = 0.1 * g.smp * g.k; g["fold"] = g.fold.astype(str)
m = pd.read_parquet(SC); m["ts"] = pd.to_datetime(m.ts).dt.tz_convert("Asia/Seoul")
mt = m.groupby("ts").agg(mate=("Candidate", "sum"), mact=("actual", "sum"), mcap=("capacity", "sum"))
tot = g.groupby("ts").agg(v10=("p", "sum"), act=("energy", "sum"))
j = tot.join(mt, how="inner"); j = j[j.index.month.isin(range(1, 13))]
print("overlap hours", len(j), "months", sorted(set(j.index.strftime("%Y-%m"))))
d = j[(j.act > 0) | (j.mact > 0)]
print("actual total match: corr", round(d.act.corr(d.mact), 5), "ratio ours/mate", round(d.act.sum() / d.mact.sum(), 4), "max|diff| kWh", round((d.act - d.mact).abs().max(), 1))
# penalty at total level and at our group level for blend weights
g = g.merge(j[["mate", "v10"]].reset_index(), on="ts", how="inner")
g["share"] = np.where(g.v10 > 0, g.p / g.v10.where(g.v10 > 0), 0.0)
months = sorted(g.fold.unique()); W = np.round(np.arange(0, 1.01, 0.1), 2)
res = {}
for wm in W:
    pb = g.p * (1 - wm) + wm * g.mate * g.share
    res[wm] = (g.w * (pb - g.energy).abs()).groupby(g.fold).sum() / 1e6
R = pd.DataFrame(res); print(R.round(3).to_string())
tot_ = R.sum(); print("totals by weight", tot_.round(3).to_dict())
# LOMO choose weight on other months
lomo = []
for mth in months:
    best = R.drop(index=mth).sum().idxmin(); lomo.append((mth, best, R.loc[mth, 0.0], R.loc[mth, best]))
L = pd.DataFrame(lomo, columns=["month", "w", "v10", "lomo"]).set_index("month"); print(L.round(3).to_string())
print("LOMO total", round(L.lomo.sum(), 3), "vs v10", round(L.v10.sum(), 3), f"{100*(L.lomo.sum()/L.v10.sum()-1):+.2f}%", "better", int((L.lomo < L.v10).sum()), "/", len(L))
for wm in (0.2, 0.3, 0.5):
    print(f"fixed w={wm}", round(R[wm].sum(), 3), f"{100*(R[wm].sum()/R[0.0].sum()-1):+.2f}%", "better", int((R[wm] < R[0.0]).sum()), "/", len(R))
