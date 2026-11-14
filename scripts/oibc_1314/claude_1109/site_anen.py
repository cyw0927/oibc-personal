"""Site-level analog ensemble: per site and hour, analogs on own station forecasts (wa_cloud, wb_cloud; round mean), GFS box gx_dsw/gx_tcc,
site cosz (+-1 h window), N=20 past days strictly before the scored month; cf mean * capacity. Summed to v7 groups on observed OOF keys.
Then compare v14 vs v14 + site AnEn (grid, LOMO). Writes cv/site_anen_oof.parquet (site_id, ts, sanen_cf).
Usage: site_anen.py <personal_safe> <comp.csv>
"""
import sys, importlib.util
import numpy as np, pandas as pd
PS, COMP = sys.argv[1:3]; KST = "Asia/Seoul"; N = 20
spec = importlib.util.spec_from_file_location("ds", __file__.replace("site_anen.py", "diag_spread_lib.py")); ds = importlib.util.module_from_spec(spec); spec.loader.exec_module(ds)
x = pd.read_parquet(f"{PS}/site_feat/site_train.parquet", columns=["site_id", "ts", "energy", "capacity_kw", "wa_cloud", "wb_cloud", "cosz"])
x["ts"] = pd.to_datetime(x.ts).dt.tz_convert(KST)
x = x.groupby(["site_id", "ts"]).agg(energy=("energy", "first"), cap=("capacity_kw", "first"), wa_cloud=("wa_cloud", "mean"), wb_cloud=("wb_cloud", "mean"), cosz=("cosz", "first")).reset_index()
gx = pd.read_parquet(f"{PS}/gfs/gfs_rich.parquet")[["ts", "gx_dsw", "gx_tcc"]]; gx["ts"] = pd.to_datetime(gx.ts).dt.tz_convert(KST)
x = x.merge(gx, on="ts", how="left"); x["cf"] = (x.energy / x.cap).clip(0, 1.2)
x["day"] = (x.ts - pd.Timedelta(hours=1)).dt.normalize(); x["h"] = x.ts.dt.hour
V = ["wa_cloud", "wb_cloud", "gx_dsw", "gx_tcc", "cosz"]; sig = x[V].std()
months = pd.period_range("2024-06", "2025-05", freq="M"); out = []
for sid, y in x.groupby("site_id"):
    P = {v: y.pivot_table(index="day", columns="h", values=v, aggfunc="first") for v in V + ["cf"]}
    days = P["cf"].index
    for m in months:
        st = pd.Timestamp(m.start_time).tz_localize(KST); en = st + pd.offsets.MonthBegin(1)
        past = days[days < st]; tgt = days[(days >= st) & (days < en)]
        if len(past) < 60 or len(tgt) == 0:
            continue
        for h in range(7, 20):
            hs = [h - 1, h, h + 1]
            A = {v: P[v].reindex(index=past, columns=hs).to_numpy() for v in V}; F = {v: P[v].reindex(index=tgt, columns=hs).to_numpy() for v in V}
            cfp = P["cf"].reindex(index=past, columns=[h]).to_numpy()[:, 0]; ok = ~np.isnan(cfp)
            for v in V: ok &= ~np.isnan(A[v]).any(1)
            if ok.sum() < N:
                continue
            dist = np.zeros((len(tgt), ok.sum()))
            for v in V: dist += np.sqrt(np.nansum((F[v][:, None, :] - A[v][ok][None, :, :]) ** 2, axis=2)) / sig[v]
            out.append(pd.DataFrame({"site_id": sid, "ts": tgt + pd.to_timedelta(h, unit="h"), "sanen_cf": cfp[ok][np.argsort(dist, axis=1)[:, :N]].mean(1)}))
sa = pd.concat(out); sa["ts"] = pd.to_datetime(sa.ts); sa["ts"] = sa.ts.dt.tz_localize(KST) if sa.ts.dt.tz is None else sa.ts.dt.tz_convert(KST)
sa.to_parquet(f"{PS}/cv/site_anen_oof.parquet", index=False); print("site anen rows", len(sa))
# aggregate to groups on observed OOF keys (same rows as v10 group energy)
keys = pd.concat([pd.read_parquet(f"{PS}/cv/{f}", columns=["site_id", "ts", "fold"]) for f in ("site_level_E_5f_gx.parquet", "site_level_E_gx_7f.parquet")])
keys["ts"] = pd.to_datetime(keys.ts).dt.tz_convert(KST)
cap = x.groupby("site_id").cap.first(); grp = pd.read_csv(COMP).set_index("site_id").group
k = keys.merge(sa, on=["site_id", "ts"], how="left"); k["sanen"] = k.sanen_cf * k.site_id.map(cap); k["group"] = k.site_id.map(grp)
k["has"] = k.sanen_cf.notna()
ga = k.groupby(["group", "ts"]).agg(sanen=("sanen", "sum"), frac=("has", "mean")).reset_index()
g = ds.build(PS, COMP, "_rich", "_gx", ""); g["p"] = (0.35*g.E + 0.35*g.B21 + 0.30*g.H1)*(1 + 0.5*(g.rd - 1)); g["w"] = 0.1*g.smp*g.k; g["fold"] = g.fold.astype(str)
def ld(name, col, new):
    z = pd.read_parquet(f"{PS}/cv/{name}.parquet"); z["ts"] = pd.to_datetime(z.ts).dt.tz_convert(KST); return z[["group", "ts", col]].rename(columns={col: new})
g = g.merge(ld("anen_oof", "anen", "a0"), on=["group", "ts"]).merge(ld("phys_oof_exp", "phys_ex_ssrd", "ph"), on=["group", "ts"]).merge(ga, on=["group", "ts"], how="left")
g = g[g.ph.notna()]; g["sanen"] = np.where(g.frac.fillna(0) > 0.95, g.sanen, g.p)
pen = lambda p: (g.w * (p - g.energy).abs()).groupby(g.fold).sum() / 1e6
v14 = .9*g.p + .05*g.a0 + .05*g.ph
print("site AnEn alone", round(pen(g.sanen).sum(), 3), "| v10", round(pen(g.p).sum(), 3), "| v14", round(pen(v14).sum(), 3), "| covered rows", round((g.frac > .95).mean(), 3))
grid = [(ws,) for ws in (0, .05, .1, .15, .2)]
R = pd.DataFrame({k[0]: pen((1 - k[0])*v14 + k[0]*g.sanen) for k in grid}); base = R[0]
print(R.round(3).to_string())
lomo = {m: R.drop(index=m).sum().idxmin() for m in R.index}; L = pd.Series({m: R.loc[m, k] for m, k in lomo.items()})
print("LOMO picks", lomo); print("LOMO", round(L.sum(), 3), "vs v14", round(base.sum(), 3), f"{100*(L.sum()/base.sum()-1):+.2f}%", "better", int((L < base).sum()), "/", len(R))
