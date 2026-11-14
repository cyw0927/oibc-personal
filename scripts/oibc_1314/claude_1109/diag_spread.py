"""Diagnostic: penalty of v10 / v11 / v12(all) by forecast-disagreement tercile (same 12-month OOF rows, v7 composition).
Spread = daily std over GFS runs 00Z D-1, 18Z D-2, 12Z D-2 of the fixed-box DSWRF mean over 10-15h labels (no site coords).
Usage: diag_spread.py <personal_safe> <comp.csv>
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
PS, COMP = Path(sys.argv[1]), Path(sys.argv[2]); cv = PS / "cv"; G = PS / "gfs"; KST = "Asia/Seoul"
grp = pd.read_csv(COMP).set_index("site_id").group


def build(DS, SS, HS):
    def load(fs, cols):
        o = []
        for f in fs:
            x = pd.read_parquet(cv / f); x["ts"] = pd.to_datetime(x.ts).dt.tz_convert(KST); x["fold"] = x.fold.astype(str); o.append(x[["site_id", "ts", "fold"] + cols])
        return pd.concat(o, ignore_index=True)
    e = load([f"site_level_E_5f{SS}.parquet", f"site_level_E{SS}_7f.parquet"], ["energy", "smp", "k", "E"])
    b = load([f"site_level_B21_12m{SS}.parquet"], ["B"]).rename(columns={"B": "B21"})
    d23 = load([f"site_level_day23{DS}.parquet", f"site_level_day23{DS}_7f.parquet"], ["B", "D_s0", "D_s1"]).rename(columns={"B": "h23", "D_s0": "d23a", "D_s1": "d23b"})
    d21 = load([f"site_level_day21{DS}.parquet", f"site_level_day21{DS}_7f.parquet"], ["B", "D_s0", "D_s1"]).rename(columns={"B": "h21", "D_s0": "d21a", "D_s1": "d21b"})
    k = ["site_id", "ts", "fold"]; s = e.merge(b, on=k).merge(d23, on=k).merge(d21, on=k)
    s["day"] = (s.ts - pd.Timedelta(hours=1)).dt.normalize(); s["group"] = s.site_id.map(grp)
    g = s.groupby(["fold", "group", "ts"]).agg(day=("day", "first"), energy=("energy", "sum"), smp=("smp", "first"), k=("k", "first"),
                                               **{c: (c, "sum") for c in ("E", "B21", "h23", "d23a", "d23b", "h21", "d21a", "d21b")}).reset_index()
    gd = g.groupby(["fold", "group", "day"])[["h23", "d23a", "d23b", "h21", "d21a", "d21b"]].sum()
    r = lambda dc, hc: (gd[dc] / gd[hc].where(gd[hc] > 1)).fillna(1.0)
    gd["rd"] = (r("d23a", "h23") + r("d23b", "h23") + r("d21a", "h21") + r("d21b", "h21")) / 4
    g = g.merge(gd[["rd"]].reset_index(), on=["fold", "group", "day"])
    h = pd.read_parquet(cv / f"group_cv_comp{HS}.parquet"); h["ts"] = pd.to_datetime(h.ts).dt.tz_convert(KST)
    return g.merge(h[["group", "ts", "H1"]], on=["group", "ts"], how="inner")


cfg = {"v10": ("_rich", "_gx", "", (0.35, 0.35, 0.5)), "v11": ("_rich", "_gx", "_gx", (0.05, 0.60, 0.3)), "v12all": ("_all", "_all", "_all", (0.0, 0.65, 0.2))}
P = {}
for n, (DS, SS, HS, (we, wb, a)) in cfg.items():
    g = build(DS, SS, HS); g["p"] = (we * g.E + wb * g.B21 + (1 - we - wb) * g.H1) * (1 + a * (g.rd - 1))
    P[n] = g.set_index(["group", "ts"])[["fold", "day", "energy", "smp", "k", "p"]]
keys = P["v10"].index.intersection(P["v11"].index).intersection(P["v12all"].index)
base = P["v10"].loc[keys].copy()
for n in P: base[n] = P[n].loc[keys, "p"]
base["w"] = 0.1 * base.smp * base.k
# spread
runs = {c: pd.read_parquet(G / f).set_index("ts").gx_dsw for c, f in (("r00", "gfs_rich.parquet"), ("r18", "gfs_rich_r18.parquet"), ("r12", "gfs_rich_r12.parquet"))}
S = pd.DataFrame(runs); S.index = pd.to_datetime(S.index).tz_convert(KST)
S = S[S.index.hour.isin(range(10, 16))]; S["day"] = (S.index - pd.Timedelta(hours=1)).normalize()
sp = S.groupby("day")[["r00", "r18", "r12"]].mean().std(axis=1); rel = sp / S.groupby("day")[["r00", "r18", "r12"]].mean().mean(axis=1)
base = base.reset_index(); base["spread"] = base.day.map(sp); base["rel"] = base.day.map(rel)
days = base.drop_duplicates("day").set_index("day").spread.dropna()
q = days.quantile([1 / 3, 2 / 3]).values
base["tercile"] = np.select([base.spread <= q[0], base.spread <= q[1]], ["low", "mid"], "high")
print("spread terciles W/m2 cut", q.round(1), "rows", len(base), "days", base.day.nunique())
out = []
for t, x in base.groupby("tercile"):
    rec = {"tercile": t, "days": x.day.nunique()}
    for n in P: rec[n] = (x.w * (x[n] - x.energy).abs()).sum() / 1e6
    out.append(rec)
R = pd.DataFrame(out).set_index("tercile").loc[["low", "mid", "high"]]
R["v11_vs_v10_%"] = 100 * (R.v11 / R.v10 - 1); R["v12_vs_v11_%"] = 100 * (R.v12all / R.v11 - 1)
print(R.round(3).to_string())
# day-level: on high-spread days, share of days where v11 beat v10
dd = base.assign(e10=base.w * (base.v10 - base.energy).abs(), e11=base.w * (base.v11 - base.energy).abs(), e12=base.w * (base.v12all - base.energy).abs()).groupby(["tercile", "day"])[["e10", "e11", "e12"]].sum()
for t in ("low", "mid", "high"):
    x = dd.loc[t]; print(t, "days v11<v10", int((x.e11 < x.e10).sum()), "/", len(x), "| days v12<v11", int((x.e12 < x.e11).sum()), "/", len(x))
# gating check: low-spread days -> v10, else v11; cut from months strictly before (expanding), first 3 months use v11
base["fold"] = base.fold.astype(str); months = sorted(base.fold.unique()); res = []
dsp = base.drop_duplicates("day")[["day", "fold", "spread"]]
for i, m in enumerate(months):
    x = base[base.fold == m]
    if i < 3:
        g = x.v11
    else:
        cut = dsp[dsp.fold.isin(months[:i])].spread.quantile(1 / 3)
        g = np.where(x.spread <= cut, x.v10, x.v11)
    pen = lambda p: (x.w * (p - x.energy).abs()).sum() / 1e6
    res.append((m, pen(x.v11), pen(g)))
R2 = pd.DataFrame(res, columns=["month", "v11", "gated"]).set_index("month"); ev = R2.iloc[3:]
print(R2.round(3).to_string()); print("eval months", len(ev), "v11", round(ev.v11.sum(), 3), "gated", round(ev.gated.sum(), 3), f"{100*(ev.gated.sum()/ev.v11.sum()-1):+.2f}%", "better", int((ev.gated < ev.v11).sum()))
