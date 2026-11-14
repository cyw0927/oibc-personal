"""Analog ensemble (Delle Monache et al. 2013; Alessandrini et al. 2015) for group capacity factor, honest rolling.
Predictors (all D-1 17:00 available, no site coordinates): GFS fixed-box gx_dsw, gx_tcc (00Z D-1), provided forecasts wa_cloud, wb_cloud
(group aggregates), and extraterrestrial horizontal irradiance at the box centre (season/time). Window +-1 h, metric
sum_v (1/sigma_v) sqrt(sum_j (F - A)^2), N=20 analogs from days strictly before the scored month, same group and hour.
Writes OOF group-hour 'anen' and blends with v10 (and v10 + ECMWF physical 10%).
Usage: anen.py <personal_safe> <comp.csv>
"""
import sys, importlib.util
import numpy as np, pandas as pd
PS, COMP = sys.argv[1:3]; KST = "Asia/Seoul"; N = 20
spec = importlib.util.spec_from_file_location("ds", __file__.replace("anen_ex.py", "diag_spread_lib.py")); ds = importlib.util.module_from_spec(spec); spec.loader.exec_module(ds)
# history: comp group features train (2023-06..2025-05) incl. provided forecasts + actuals
f = pd.read_parquet(f"{PS}/feat_full_comp/group_features_train.parquet"); f["ts"] = pd.to_datetime(f.ts).dt.tz_convert(KST)
gx = pd.read_parquet(f"{PS}/gfs/gfs_rich.parquet")[["ts", "gx_dsw", "gx_tcc"]]; gx["ts"] = pd.to_datetime(gx.ts).dt.tz_convert(KST)
f = f.merge(gx, on="ts", how="left")
exh = pd.read_parquet(f"{PS}/gfs/ecmwf_hourly.parquet")[["ts", "ex_ssrd"]]; exh["ts"] = pd.to_datetime(exh.ts).dt.tz_convert(KST); f = f.merge(exh, on="ts", how="left")
LAT, LON = np.radians(36.0), 128.0
u = f.ts.dt.tz_convert("UTC"); doy = u.dt.dayofyear.values; hr = (u.dt.hour - 0.5).values; ga = 2 * np.pi * (doy - 1) / 365
dec = 0.006918 - 0.399912*np.cos(ga) + 0.070257*np.sin(ga) - 0.006758*np.cos(2*ga) + 0.000907*np.sin(2*ga) - 0.002697*np.cos(3*ga) + 0.00148*np.sin(3*ga)
eot = 229.18 * (0.000075 + 0.001868*np.cos(ga) - 0.032077*np.sin(ga) - 0.014615*np.cos(2*ga) - 0.040849*np.sin(2*ga))
ha = np.radians((hr * 60 + eot + 4 * LON) / 4 - 180)
f["e0"] = 1361 * (1 + 0.033*np.cos(2*np.pi*doy/365)) * np.clip(np.sin(LAT)*np.sin(dec) + np.cos(LAT)*np.cos(dec)*np.cos(ha), 0, None)
f["cov"] = f.cap_avail / f.group_cap_kw; f["cf"] = np.where((f["cov"] >= 0.3) & f.energy.notna(), (f.energy / f.cap_avail).clip(0, 1.2), np.nan)
f["day"] = (f.ts - pd.Timedelta(hours=1)).dt.normalize(); f["h"] = f.ts.dt.hour
V = ["ex_ssrd", "gx_dsw", "gx_tcc", "wa_cloud", "wb_cloud", "e0"]
sig = f[V].std()
out = []
months = pd.period_range("2024-06", "2025-05", freq="M")
for grp in ("G1", "G2"):
    x = f[f.group == grp]
    P = {v: x.pivot_table(index="day", columns="h", values=v, aggfunc="first") for v in V + ["cf"]}
    days = P["cf"].index
    for m in months:
        st = pd.Timestamp(m.start_time).tz_localize(KST); en = st + pd.offsets.MonthBegin(1)
        past = days[days < st]; tgt = days[(days >= st) & (days < en)]
        for h in range(5, 21):
            hs = list(range(h - 1, h + 1 + 1))
            def mat(dd):
                return {v: P[v].reindex(index=dd, columns=hs).to_numpy() for v in V}
            A = mat(past); F = mat(tgt)
            cfp = P["cf"].reindex(index=past, columns=[h]).to_numpy()[:, 0]
            ok = ~np.isnan(cfp)
            for v in V:
                ok &= ~np.isnan(A[v]).any(1)
            if ok.sum() < N:
                continue
            dist = np.zeros((len(tgt), ok.sum()))
            for v in V:
                diff = F[v][:, None, :] - A[v][ok][None, :, :]
                dist += np.sqrt(np.nansum(diff ** 2, axis=2)) / sig[v]
            idx = np.argsort(dist, axis=1)[:, :N]
            pred = cfp[ok][idx].mean(1)
            out.append(pd.DataFrame({"group": grp, "ts": tgt + pd.Timedelta(hours=1) + pd.to_timedelta(h - 1, unit="h"), "anen_cf": pred}))
an = pd.concat(out)
an["ts"] = an.ts.dt.tz_convert(KST) if an.ts.dt.tz is not None else an.ts.dt.tz_localize(KST)
g = ds.build(PS, COMP, "_rich", "_gx", "")
g["p"] = (0.35 * g.E + 0.35 * g.B21 + 0.30 * g.H1) * (1 + 0.5 * (g.rd - 1)); g["w"] = 0.1 * g.smp * g.k; g["fold"] = g.fold.astype(str)
cap = pd.read_parquet(f"{PS}/cv/group_cv_comp.parquet")[["group", "ts", "cap_avail"]]; cap["ts"] = pd.to_datetime(cap.ts).dt.tz_convert(KST)
g = g.merge(cap, on=["group", "ts"], how="left").merge(an, on=["group", "ts"], how="left")
g["anen"] = (g.anen_cf.fillna(0) * g.cap_avail).where(g.anen_cf.notna(), g.p)
print("anen coverage", round(g.anen_cf.notna().mean(), 3))
pen = lambda p: (g.w * (p - g.energy).abs()).groupby(g.fold).sum() / 1e6
R = pd.DataFrame({w: pen((1 - w) * g.p + w * g.anen) for w in (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0)})
print(R.round(3).to_string()); print("totals", R.sum().round(3).to_dict())
mo = list(R.index); lomo = {m: R.drop(index=m)[[c for c in R.columns if c < 1]].sum().idxmin() for m in mo}
L = pd.Series({m: R.loc[m, w] for m, w in lomo.items()}); base = R[0.0]
print("LOMO w", lomo); print("LOMO", round(L.sum(), 3), "vs v10", round(base.sum(), 3), f"{100*(L.sum()/base.sum()-1):+.2f}%", "better", int((L < base).sum()), "/", len(mo))
g[["fold", "group", "ts", "anen", "anen_cf"]].to_parquet(f"{PS}/cv/anen_ex_oof.parquet", index=False)
