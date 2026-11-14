"""Physical NWP->PV model (Lorenz et al. 2009/2011 style) at group level and blend with v10 OOF.
pred_g,t = PR(hour, month-of-year window) * cap_avail_g,t * GHI_fcst_t / 1000, GHI = fixed-box GFS (gx_dsw) or ECMWF (ex_ssrd) hourly.
PR fitted only on OOF months strictly before the scored month (rolling, honest), as median of actual/(cap*GHI/1000) by hour, weighted.
Blend: (1-w) v10 + w phys, w in 0..0.5. Penalty on our v7 composition rows.
Usage: phys_blend.py <personal_safe> <comp.csv>
"""
import sys, importlib.util
import numpy as np, pandas as pd
PS, COMP = sys.argv[1:3]
spec = importlib.util.spec_from_file_location("ds", __file__.replace("phys_blend_exp.py", "diag_spread_lib.py")); ds = importlib.util.module_from_spec(spec); spec.loader.exec_module(ds)
g = ds.build(PS, COMP, "_rich", "_gx", "")
g["p"] = (0.35 * g.E + 0.35 * g.B21 + 0.30 * g.H1) * (1 + 0.5 * (g.rd - 1)); g["w"] = 0.1 * g.smp * g.k; g["fold"] = g.fold.astype(str)
cap = pd.read_parquet(f"{PS}/cv/group_cv_comp.parquet")[["group", "ts", "cap_avail"]]; cap["ts"] = pd.to_datetime(cap.ts).dt.tz_convert("Asia/Seoul")
g = g.merge(cap, on=["group", "ts"], how="left")
G = f"{PS}/gfs/"
gx = pd.read_parquet(G + "gfs_rich.parquet")[["ts", "gx_dsw"]]; gx["ts"] = pd.to_datetime(gx.ts).dt.tz_convert("Asia/Seoul")
ex = pd.read_parquet(G + "ecmwf_hourly.parquet")[["ts", "ex_ssrd"]]; ex["ts"] = pd.to_datetime(ex.ts).dt.tz_convert("Asia/Seoul")
g = g.merge(gx, on="ts", how="left").merge(ex, on="ts", how="left")
g["hour"] = g.ts.dt.hour
# extraterrestrial horizontal irradiance at fixed box centre (36N, 128E), mean over the 3 h window -> predicted clear-sky-like index
LAT, LON = np.radians(36.0), 128.0
def e0h(t):
    u = t.dt.tz_convert("UTC"); doy = u.dt.dayofyear.values; hr = (u.dt.hour + u.dt.minute / 60).values; ga = 2 * np.pi * (doy - 1) / 365
    dec = 0.006918 - 0.399912*np.cos(ga) + 0.070257*np.sin(ga) - 0.006758*np.cos(2*ga) + 0.000907*np.sin(2*ga) - 0.002697*np.cos(3*ga) + 0.00148*np.sin(3*ga)
    eot = 229.18 * (0.000075 + 0.001868*np.cos(ga) - 0.032077*np.sin(ga) - 0.014615*np.cos(2*ga) - 0.040849*np.sin(2*ga))
    ha = np.radians((hr * 60 + eot + 4 * LON) / 4 - 180)
    return 1361 * (1 + 0.033 * np.cos(2 * np.pi * doy / 365)) * np.clip(np.sin(LAT)*np.sin(dec) + np.cos(LAT)*np.cos(dec)*np.cos(ha), 0, None)
end = g.ts + pd.to_timedelta((3 - g.ts.dt.hour % 3) % 3, unit="h")
g["e0"] = np.mean([e0h(end - pd.Timedelta(minutes=mm)) for mm in range(5, 180, 10)], axis=0)
for s_ in ("gx_dsw", "ex_ssrd"):
    g[f"kt_{s_}"] = (g[s_] / g.e0.where(g.e0 > 20)).clip(0, 1.2)
KB = [-0.01, 0.2, 0.35, 0.5, 0.6, 0.7, 1.3]; CB = [0, 150, 350, 600, 900, 1500]
months = sorted(g.fold.unique())
for src in ("gx_dsw", "ex_ssrd"):
    g[f"phys_{src}"] = np.nan
    for i, m in enumerate(months):
        st = g.loc[g.fold == m, "ts"].min()
        tr = g[g.fold.isin(months[:i]) & (g[src] > 20) & (g.cap_avail > 0)]
        te = g.fold == m
        if len(tr) < 500:
            continue
        r = tr.energy / (tr.cap_avail * tr[src] / 1000.0)
        pr = tr.assign(r=r.clip(0, 2)).groupby("hour").r.median()
        g.loc[te, f"phys_{src}"] = (g.loc[te, "hour"].map(pr).fillna(0) * g.loc[te, "cap_avail"] * g.loc[te, src].fillna(0).clip(lower=0) / 1000.0).values
    ok = g[f"phys_{src}"].notna()
    ev = g[ok]; mo = sorted(ev.fold.unique())
    pen = lambda p: (ev.w * (p - ev.energy).abs()).groupby(ev.fold).sum() / 1e6
    R = pd.DataFrame({w: pen((1 - w) * ev.p + w * ev[f"phys_{src}"]) for w in (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0)})
    print("==", src, "months", mo); print(R.round(3).to_string()); print("totals", R.sum().round(3).to_dict())
    lomo = [(m, R.drop(index=m)[[c for c in R.columns if c < 1]].sum().idxmin()) for m in mo]
    L = pd.Series({m: R.loc[m, w] for m, w in lomo}); base = R[0.0]
    print("LOMO weights", dict(lomo)); print("LOMO", round(L.sum(), 3), "vs v10", round(base.sum(), 3), f"{100*(L.sum()/base.sum()-1):+.2f}%", "better", int((L < base).sum()), "/", len(mo))
g[["fold", "group", "ts", "phys_ex_ssrd", "phys_gx_dsw"]].to_parquet(f"{PS}/cv/phys_oof_exp.parquet", index=False)
