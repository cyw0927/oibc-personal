"""v15 emulation eval: base v10 with day ratios from 00Z-at-predict day models (_r00) and physical model PR(12Z) applied to 00Z hourly SSRD."""
import sys, importlib.util
import numpy as np, pandas as pd
PS, COMP = sys.argv[1:3]; KST = "Asia/Seoul"
spec = importlib.util.spec_from_file_location("ds", __file__.replace("v15_eval.py", "diag_spread_lib.py")); ds = importlib.util.module_from_spec(spec); spec.loader.exec_module(ds)
def base(DS):
    g = ds.build(PS, COMP, DS, "_gx", ""); g["p"] = (0.35*g.E + 0.35*g.B21 + 0.30*g.H1)*(1 + 0.5*(g.rd - 1)); g["fold"] = g.fold.astype(str); return g.set_index(["group", "ts"])
b0 = base("_rich"); b1 = base("_r00"); k = b0.index.intersection(b1.index)
g = b0.loc[k, ["fold", "energy", "smp", "k", "p"]].rename(columns={"p": "v10"}); g["v10_00"] = b1.loc[k, "p"]; g = g.reset_index(); g["w"] = 0.1*g.smp*g.k
cap = pd.read_parquet(f"{PS}/cv/group_cv_comp.parquet")[["group", "ts", "cap_avail"]]; cap["ts"] = pd.to_datetime(cap.ts).dt.tz_convert(KST)
ex = pd.read_parquet(f"{PS}/gfs/ecmwf_hourly.parquet")[["ts", "ex_ssrd"]]; e0 = pd.read_parquet(f"{PS}/gfs/ecmwf_00z.parquet")[["ts", "e0_ssrd"]]
for x in (ex, e0): x["ts"] = pd.to_datetime(x.ts).dt.tz_convert(KST)
an = pd.read_parquet(f"{PS}/cv/anen_oof.parquet")[["group", "ts", "anen"]]; an["ts"] = pd.to_datetime(an.ts).dt.tz_convert(KST)
g = g.merge(cap, on=["group", "ts"], how="left").merge(ex, on="ts", how="left").merge(e0, on="ts", how="left").merge(an, on=["group", "ts"], how="left")
g["hour"] = g.ts.dt.hour; months = sorted(g.fold.unique())
for c in ("ph12", "ph00"): g[c] = np.nan
for i, m in enumerate(months):
    tr = g[g.fold.isin(months[:i]) & (g.ex_ssrd > 20) & (g.cap_avail > 0)]
    if len(tr) < 500: continue
    pr = (tr.energy / (tr.cap_avail * tr.ex_ssrd / 1000.0)).clip(0, 2).groupby(tr.hour).median(); te = g.fold == m
    for c, src in (("ph12", "ex_ssrd"), ("ph00", "e0_ssrd")):
        g.loc[te, c] = (g.loc[te, "hour"].map(pr).fillna(0) * g.loc[te, "cap_avail"] * g.loc[te, src].fillna(0).clip(lower=0) / 1000.0).values
g = g[g.ph12.notna() & g.ph00.notna() & g.anen.notna()]
pen = lambda p: (g.w * (p - g.energy).abs()).groupby(g.fold).sum() / 1e6
R = pd.DataFrame({"v10": pen(g.v10), "v10_00day": pen(g.v10_00), "v14": pen(.9*g.v10 + .05*g.anen + .05*g.ph12),
                  "v15_day": pen(.9*g.v10_00 + .05*g.anen + .05*g.ph12), "v15_phys": pen(.9*g.v10 + .05*g.anen + .05*g.ph00),
                  "v15": pen(.9*g.v10_00 + .05*g.anen + .05*g.ph00)})
print(R.round(3).to_string()); t = R.sum()
for c in R.columns: print(c, round(t[c], 3), f"vs v14 {100*(t[c]/t.v14-1):+.2f}%", "months better than v14", int((R[c] < R.v14).sum()), "/", len(R))
