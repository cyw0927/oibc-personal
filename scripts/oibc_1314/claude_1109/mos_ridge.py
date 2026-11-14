"""Linear MOS (ridge) for group capacity factor, rolling honest (fit on days < scored month), then 4-way blend with v10 + AnEn + phys.
Features (D-1 17:00 info, no site coords): GFS box kt = gx_dsw/e0, gx_tcc, gx_lcc, provided wa_cloud, wb_cloud, cs_ab, cosz,
hour one-hot, and interactions kt*cosz, wa_cloud*cosz, wb_cloud*cosz. Target cf (cov>=0.3), weight smp*k*cap.
Usage: mos_ridge.py <personal_safe> <comp.csv>
"""
import sys, importlib.util, itertools
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
PS, COMP = sys.argv[1:3]; KST = "Asia/Seoul"
spec = importlib.util.spec_from_file_location("ds", __file__.replace("mos_ridge.py", "diag_spread_lib.py")); ds = importlib.util.module_from_spec(spec); spec.loader.exec_module(ds)
f = pd.read_parquet(f"{PS}/feat_full_comp/group_features_train.parquet"); f["ts"] = pd.to_datetime(f.ts).dt.tz_convert(KST)
gx = pd.read_parquet(f"{PS}/gfs/gfs_rich.parquet")[["ts", "gx_dsw", "gx_tcc", "gx_lcc"]]; gx["ts"] = pd.to_datetime(gx.ts).dt.tz_convert(KST)
f = f.merge(gx, on="ts", how="left")
LAT, LON = np.radians(36.0), 128.0
end = f.ts + pd.to_timedelta((3 - f.ts.dt.hour % 3) % 3, unit="h")
def e0h(t):
    u = t.dt.tz_convert("UTC"); doy = u.dt.dayofyear.values; hr = (u.dt.hour + u.dt.minute/60).values; ga = 2*np.pi*(doy-1)/365
    dec = 0.006918 - 0.399912*np.cos(ga) + 0.070257*np.sin(ga) - 0.006758*np.cos(2*ga) + 0.000907*np.sin(2*ga) - 0.002697*np.cos(3*ga) + 0.00148*np.sin(3*ga)
    eot = 229.18*(0.000075 + 0.001868*np.cos(ga) - 0.032077*np.sin(ga) - 0.014615*np.cos(2*ga) - 0.040849*np.sin(2*ga))
    ha = np.radians((hr*60 + eot + 4*LON)/4 - 180)
    return 1361*(1 + 0.033*np.cos(2*np.pi*doy/365))*np.clip(np.sin(LAT)*np.sin(dec) + np.cos(LAT)*np.cos(dec)*np.cos(ha), 0, None)
f["e0w"] = np.mean([e0h(end - pd.Timedelta(minutes=m)) for m in range(5, 180, 10)], axis=0)
f["kt"] = (f.gx_dsw / f.e0w.where(f.e0w > 20)).clip(0, 1.2).fillna(0)
for c in ("gx_tcc", "gx_lcc", "wa_cloud", "wb_cloud"): f[c] = f[c].fillna(f[c].median())
f["kt_c"] = f.kt * f.cosz; f["wa_c"] = f.wa_cloud * f.cosz / 100; f["wb_c"] = f.wb_cloud * f.cosz / 100; f["tcc_c"] = f.gx_tcc * f.cosz / 100
H = pd.get_dummies(f.ts.dt.hour, prefix="h").astype(float); f = pd.concat([f, H], axis=1)
X = ["kt", "gx_tcc", "gx_lcc", "wa_cloud", "wb_cloud", "cs_ab", "cosz", "kt_c", "wa_c", "wb_c", "tcc_c"] + list(H.columns)
f["cov"] = f.cap_avail / f.group_cap_kw; f["cf"] = (f.energy / f.cap_avail).clip(0, 1.2)
smp = pd.read_csv(f"C:/Users/dkrak/AppData/Local/Temp/claude/C--Users-dkrak-Downloads-----------/b8f675a9-3275-4093-bed5-50b68c2cf9a4/scratchpad/oibc_data/train/smp.csv"); smp["ts"] = pd.to_datetime(smp.ts).dt.tz_convert(KST)
f = f.merge(smp, on="ts", how="left"); f["k"] = np.where(f.ts.dt.hour.between(11, 15), 3, 1)
out = []
for M in pd.period_range("2024-06", "2025-05", freq="M"):
    st = pd.Timestamp(M.start_time).tz_localize(KST) + pd.Timedelta(hours=1); en = st + pd.offsets.MonthBegin(1)
    tr = f[(f.ts < st) & f.energy.notna() & (f["cov"] >= 0.3) & (f.cosz_max > 0)]
    te = f[(f.ts >= st) & (f.ts < en)].copy()
    mu, sd = tr[X].mean(), tr[X].std().replace(0, 1)
    sw = (tr.smp.clip(lower=1) * tr.k * tr["cov"]).values
    for grp in ("G1", "G2"):
        a = tr[tr.group == grp]; b = te[te.group == grp]
        r = Ridge(alpha=10.0).fit(((a[X] - mu) / sd).values, a.cf.values, sample_weight=sw[(tr.group == grp).values])
        b = b.assign(mos=np.where(b.cosz_max > 0, np.clip(r.predict(((b[X] - mu) / sd).values), 0, 1), 0) * b.cap_avail)
        out.append(b[["group", "ts", "mos"]])
mo = pd.concat(out)
g = ds.build(PS, COMP, "_rich", "_gx", ""); g["p"] = (0.35*g.E + 0.35*g.B21 + 0.30*g.H1)*(1 + 0.5*(g.rd - 1)); g["w"] = 0.1*g.smp*g.k; g["fold"] = g.fold.astype(str)
a = pd.read_parquet(f"{PS}/cv/anen_oof.parquet"); ph = pd.read_parquet(f"{PS}/cv/phys_oof_exp.parquet")
for x in (a, ph): x["ts"] = pd.to_datetime(x.ts).dt.tz_convert(KST)
g = g.merge(a[["group", "ts", "anen"]], on=["group", "ts"]).merge(ph[["group", "ts", "phys_ex_ssrd"]], on=["group", "ts"]).merge(mo, on=["group", "ts"], how="left")
g = g[g.phys_ex_ssrd.notna() & g.mos.notna()]
pen = lambda p: (g.w * (p - g.energy).abs()).groupby(g.fold).sum() / 1e6
print("MOS alone", round(pen(g.mos).sum(), 3), "v10", round(pen(g.p).sum(), 3))
grid = [(wa, wp, wm) for wa in (0, .05, .1) for wp in (0, .05, .1) for wm in (0, .05, .1, .15)]
R = pd.DataFrame({k: pen((1 - sum(k))*g.p + k[0]*g.anen + k[1]*g.phys_ex_ssrd + k[2]*g.mos) for k in grid})
base = R[(0, 0, 0)]; v14 = R[(0.05, 0.05, 0)]; t = R.sum().sort_values()
print("best in-sample", {str(k): round(v, 3) for k, v in t.head(6).items()})
for k in ((0.05, 0.05, 0), (0, 0, 0.1), (0.05, 0.05, 0.05), (0.05, 0.05, 0.1)):
    print(k, round(R[k].sum(), 3), f"vs v10 {100*(R[k].sum()/base.sum()-1):+.2f}%", f"vs v14 {100*(R[k].sum()/v14.sum()-1):+.2f}%", "months better than v14", int((R[k] < v14).sum()), "/", len(R))
lomo = {m: R.drop(index=m).sum().idxmin() for m in R.index}; L = pd.Series({m: R.loc[m, k] for m, k in lomo.items()})
print("LOMO picks", {m: str(k) for m, k in lomo.items()}); print("LOMO", round(L.sum(), 3), f"vs v10 {100*(L.sum()/base.sum()-1):+.2f}%", f"vs v14 {100*(L.sum()/v14.sum()-1):+.2f}%", "better than v14", int((L < v14).sum()), "/", len(R))
g[["fold", "group", "ts", "mos"]].to_parquet(f"{PS}/cv/mos_oof.parquet", index=False)
