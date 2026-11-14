"""Compare on same rows: v10 base vs v13 base (E/B21/H1 with ECMWF hourly, v10 weights), each alone and + AnEn 5% + phys(expanding PR) 5%."""
import sys, importlib.util
import numpy as np, pandas as pd
PS, COMP = sys.argv[1:3]
spec = importlib.util.spec_from_file_location("ds", __file__.replace("blend_v13v14.py", "diag_spread_lib.py")); ds = importlib.util.module_from_spec(spec); spec.loader.exec_module(ds)
def base(DS, SS, HS):
    g = ds.build(PS, COMP, DS, SS, HS); g["p"] = (0.35*g.E + 0.35*g.B21 + 0.30*g.H1) * (1 + 0.5*(g.rd - 1)); g["fold"] = g.fold.astype(str)
    return g.set_index(["group", "ts"])
b10 = base("_rich", "_gx", ""); b13 = base("_rx", "_rx", "_rx")
k = b10.index.intersection(b13.index); g = b10.loc[k, ["fold", "energy", "smp", "k", "p"]].rename(columns={"p": "v10"}); g["v13"] = b13.loc[k, "p"]
g = g.reset_index(); g["w"] = 0.1*g.smp*g.k
a = pd.read_parquet(f"{PS}/cv/anen_oof.parquet"); ph = pd.read_parquet(f"{PS}/cv/phys_oof_exp.parquet")
for x in (a, ph): x["ts"] = pd.to_datetime(x.ts).dt.tz_convert("Asia/Seoul")
g = g.merge(a[["group", "ts", "anen"]], on=["group", "ts"]).merge(ph[["group", "ts", "phys_ex_ssrd"]], on=["group", "ts"]); g = g[g.phys_ex_ssrd.notna()]
pen = lambda p: (g.w * (p - g.energy).abs()).groupby(g.fold).sum() / 1e6
R = pd.DataFrame({"v10": pen(g.v10), "v13": pen(g.v13), "v14": pen(.9*g.v10 + .05*g.anen + .05*g.phys_ex_ssrd), "v13+": pen(.9*g.v13 + .05*g.anen + .05*g.phys_ex_ssrd),
                  "v13/v10 50:50+": pen(.45*g.v10 + .45*g.v13 + .05*g.anen + .05*g.phys_ex_ssrd)})
print(R.round(3).to_string()); t = R.sum()
for c in R.columns: print(c, round(t[c], 3), f"vs v10 {100*(t[c]/t.v10-1):+.2f}%", f"vs v14 {100*(t[c]/t.v14-1):+.2f}%", "better than v14", int((R[c] < R.v14).sum()), "/", len(R))
