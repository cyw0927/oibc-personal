"""3-way blend on 12-month OOF: v10 + AnEn + ECMWF physical; LOMO over weight grid (months with phys available)."""
import sys, importlib.util, itertools
import numpy as np, pandas as pd
PS, COMP = sys.argv[1:3]
spec = importlib.util.spec_from_file_location("ds", __file__.replace("blend3e.py", "diag_spread_lib.py")); ds = importlib.util.module_from_spec(spec); spec.loader.exec_module(ds)
g = ds.build(PS, COMP, "_rich", "_gx", ""); g["p"] = (0.35*g.E + 0.35*g.B21 + 0.30*g.H1) * (1 + 0.5*(g.rd - 1)); g["w"] = 0.1*g.smp*g.k; g["fold"] = g.fold.astype(str)
a = pd.read_parquet(f"{PS}/cv/anen_oof.parquet"); ph = pd.read_parquet(f"{PS}/cv/phys_oof_exp.parquet")
for x in (a, ph): x["ts"] = pd.to_datetime(x.ts).dt.tz_convert("Asia/Seoul"); x["fold"] = x.fold.astype(str)
g = g.merge(a[["group", "ts", "anen"]], on=["group", "ts"]).merge(ph[["group", "ts", "phys_ex_ssrd"]], on=["group", "ts"])
g = g[g.phys_ex_ssrd.notna()]
grid = [(wa, wp) for wa in (0, .05, .1, .15) for wp in (0, .05, .1, .15)]
R = pd.DataFrame({(wa, wp): (g.w * ((1-wa-wp)*g.p + wa*g.anen + wp*g.phys_ex_ssrd - g.energy).abs()).groupby(g.fold).sum()/1e6 for wa, wp in grid})
tot = R.sum().sort_values(); print("best in-sample", tot.head(5).round(3).to_dict()); base = R[(0, 0)]
for k in ((0.1, 0), (0, 0.1), (0.1, 0.05), (0.1, 0.1), (0.05, 0.05)):
    print(k, round(R[k].sum(), 3), f"{100*(R[k].sum()/base.sum()-1):+.2f}%", "better", int((R[k] < base).sum()), "/", len(base))
lomo = {m: R.drop(index=m).sum().idxmin() for m in R.index}; L = pd.Series({m: R.loc[m, k] for m, k in lomo.items()})
print("LOMO picks", lomo); print("LOMO", round(L.sum(), 3), "vs v10", round(base.sum(), 3), f"{100*(L.sum()/base.sum()-1):+.2f}%", "better", int((L < base).sum()), "/", len(base))
print("months", list(R.index))
