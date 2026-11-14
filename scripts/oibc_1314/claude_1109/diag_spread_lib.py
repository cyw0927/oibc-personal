import numpy as np, pandas as pd
from pathlib import Path
KST='Asia/Seoul'
def build(PS, COMP, DS, SS, HS):
    PS = Path(PS); cv = PS / 'cv'; grp = pd.read_csv(COMP).set_index('site_id').group
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


