"""v14 vs variants with alternative AnEn component (same rows, 11 months), LOMO over small grid; criterion LOMO <= v14 - 0.3% and >= 7/11 months."""
import sys, importlib.util
import numpy as np, pandas as pd
PS, COMP = sys.argv[1:3]; KST = "Asia/Seoul"
spec = importlib.util.spec_from_file_location("ds", __file__.replace("blend_var.py", "diag_spread_lib.py")); ds = importlib.util.module_from_spec(spec); spec.loader.exec_module(ds)
g = ds.build(PS, COMP, "_rich", "_gx", ""); g["p"] = (0.35*g.E + 0.35*g.B21 + 0.30*g.H1)*(1 + 0.5*(g.rd - 1)); g["w"] = 0.1*g.smp*g.k; g["fold"] = g.fold.astype(str)
def ld(name, col, new):
    x = pd.read_parquet(f"{PS}/cv/{name}.parquet"); x["ts"] = pd.to_datetime(x.ts).dt.tz_convert(KST); return x[["group", "ts", col]].rename(columns={col: new})
for nm, c, n in (("anen_oof", "anen", "a0"), ("anen_n40_oof", "anen", "a40"), ("anen_w2_oof", "anen", "aw2"), ("anen_ex_oof", "anen", "aex"), ("phys_oof_exp", "phys_ex_ssrd", "ph")):
    g = g.merge(ld(nm, c, n), on=["group", "ts"], how="left")
g = g[g.ph.notna()]; g["aw2"] = g.aw2.fillna(g.a0); g["amix"] = g[["a0", "a40", "aw2", "aex"]].mean(1)
pen = lambda p: (g.w * (p - g.energy).abs()).groupby(g.fold).sum() / 1e6
v14 = pen(.9*g.p + .05*g.a0 + .05*g.ph)
for comp in ("a0", "a40", "aw2", "aex", "amix"):
    grid = [(wa, wp) for wa in (.05, .1, .15) for wp in (.05, .1)]
    R = pd.DataFrame({k: pen((1 - sum(k))*g.p + k[0]*g[comp] + k[1]*g.ph) for k in grid})
    lomo = {m: R.drop(index=m).sum().idxmin() for m in R.index}; L = pd.Series({m: R.loc[m, k] for m, k in lomo.items()})
    fx = R[(.05, .05)]
    print(f"{comp:5s} fixed(.05,.05) {fx.sum():.3f} ({100*(fx.sum()/v14.sum()-1):+.2f}% vs v14, {int((fx<v14).sum())}/11) | LOMO {L.sum():.3f} ({100*(L.sum()/v14.sum()-1):+.2f}% vs v14, {int((L<v14).sum())}/11)")
print("v14", round(v14.sum(), 3))
