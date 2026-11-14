"""v14 = 0.90 * v10 + 0.05 * AnEn + 0.05 * ECMWF physical, evaluation year, v7 composition groups (feat_full_comp).
AnEn: analogs from all days < 2025-06-01 (same group, hour, +-1 h window, N=20) on gx_dsw, gx_tcc, wa_cloud, wb_cloud, e0; cf * group capacity.
Physical: PR(hour) = median actual/(cap_avail * ex_ssrd/1000) over all training hours with ECMWF hourly (2024-03..2025-05); pred = PR * group cap * ex_ssrd/1000.
Hours without AnEn/phys values keep v10. Inputs are D-1 17:00 forecasts only; no test actuals.
Usage: final_v14.py <personal_safe> <v10_forecast.csv> <out_dir> <dataset_comp>
"""
import sys, json, hashlib
from pathlib import Path
import numpy as np, pandas as pd
PS, V10, OUT, DC = sys.argv[1:5]; OUT = Path(OUT); OUT.mkdir(parents=True, exist_ok=True); KST = "Asia/Seoul"; N = 20
tr = pd.read_parquet(f"{PS}/feat_full_comp/group_features_train.parquet"); te = pd.read_parquet(f"{PS}/feat_full_comp/group_features_test.parquet")
for x in (tr, te): x["ts"] = pd.to_datetime(x.ts).dt.tz_convert(KST)
f = pd.concat([tr.assign(split="train"), te.assign(split="test")], ignore_index=True)
gx = pd.read_parquet(f"{PS}/gfs/gfs_rich.parquet")[["ts", "gx_dsw", "gx_tcc"]]; gx["ts"] = pd.to_datetime(gx.ts).dt.tz_convert(KST)
ex = pd.read_parquet(f"{PS}/gfs/ecmwf_hourly.parquet")[["ts", "ex_ssrd"]]; ex["ts"] = pd.to_datetime(ex.ts).dt.tz_convert(KST)
f = f.merge(gx, on="ts", how="left").merge(ex, on="ts", how="left")
LAT, LON = np.radians(36.0), 128.0
u = f.ts.dt.tz_convert("UTC"); doy = u.dt.dayofyear.values; hr = (u.dt.hour - 0.5).values; ga = 2*np.pi*(doy-1)/365
dec = 0.006918 - 0.399912*np.cos(ga) + 0.070257*np.sin(ga) - 0.006758*np.cos(2*ga) + 0.000907*np.sin(2*ga) - 0.002697*np.cos(3*ga) + 0.00148*np.sin(3*ga)
eot = 229.18*(0.000075 + 0.001868*np.cos(ga) - 0.032077*np.sin(ga) - 0.014615*np.cos(2*ga) - 0.040849*np.sin(2*ga))
ha = np.radians((hr*60 + eot + 4*LON)/4 - 180)
f["e0"] = 1361*(1 + 0.033*np.cos(2*np.pi*doy/365))*np.clip(np.sin(LAT)*np.sin(dec) + np.cos(LAT)*np.cos(dec)*np.cos(ha), 0, None)
f["cov"] = f.cap_avail / f.group_cap_kw
f["cf"] = np.where((f.split == "train") & (f["cov"] >= 0.3) & f.energy.notna(), (f.energy / f.cap_avail).clip(0, 1.2), np.nan)
f["day"] = (f.ts - pd.Timedelta(hours=1)).dt.normalize(); f["h"] = f.ts.dt.hour
V = ["gx_dsw", "gx_tcc", "wa_cloud", "wb_cloud", "e0"]; sig = f[f.split == "train"][V].std()
cut = pd.Timestamp("2025-06-01", tz=KST)
rows = []
for grp in ("G1", "G2"):
    x = f[f.group == grp]; gcap = x.group_cap_kw.iloc[0]
    P = {v: x.pivot_table(index="day", columns="h", values=v, aggfunc="first") for v in V + ["cf"]}
    days = P["cf"].index; past = days[days < cut]; tgt = P["gx_dsw"].index[P["gx_dsw"].index >= cut]
    for h in range(5, 21):
        hs = [h - 1, h, h + 1]
        A = {v: P[v].reindex(index=past, columns=hs).to_numpy() for v in V}; F = {v: P[v].reindex(index=tgt, columns=hs).to_numpy() for v in V}
        cfp = P["cf"].reindex(index=past, columns=[h]).to_numpy()[:, 0]; ok = ~np.isnan(cfp)
        for v in V: ok &= ~np.isnan(A[v]).any(1)
        dist = np.zeros((len(tgt), ok.sum()))
        for v in V: dist += np.sqrt(np.nansum((F[v][:, None, :] - A[v][ok][None, :, :]) ** 2, axis=2)) / sig[v]
        pred = cfp[ok][np.argsort(dist, axis=1)[:, :N]].mean(1)
        rows.append(pd.DataFrame({"group": grp, "ts": tgt + pd.to_timedelta(h, unit="h"), "anen": pred * gcap}))
an = pd.concat(rows)
# physical
t = f[(f.split == "train") & (f.ex_ssrd > 20) & (f.cap_avail > 0) & f.cf.notna()]
pr = (t.energy / (t.cap_avail * t.ex_ssrd / 1000.0)).clip(0, 2).groupby(t.h).median()
ft = f[f.split == "test"].copy(); ft["phys"] = np.where(ft.ex_ssrd.notna(), ft.h.map(pr).fillna(0) * ft.group_cap_kw * ft.ex_ssrd.clip(lower=0) / 1000.0, np.nan)
v = pd.read_csv(V10); v["tsd"] = pd.to_datetime(v.ts).dt.tz_convert(KST)
m = v.merge(an.rename(columns={"ts": "tsd"}), on=["group", "tsd"], how="left").merge(ft[["group", "ts", "phys"]].rename(columns={"ts": "tsd"}), on=["group", "tsd"], how="left")
both = m.anen.notna() & m.phys.notna()
m["forecast_kwh"] = np.where(both, 0.90 * m.forecast_kwh + 0.05 * m.anen + 0.05 * m.phys, m.forecast_kwh).clip(min=0)
print("rows", len(m), "blended rows", int(both.sum()), "anen cov", round(m.anen.notna().mean(), 3), "phys cov", round(m.phys.notna().mean(), 3))
print("GWh v10/anen/phys/v14", round(v.forecast_kwh.sum()/1e6, 3), round(m.anen.sum()/1e6, 3), round(m.phys.sum()/1e6, 3), round(m.forecast_kwh.sum()/1e6, 3))
print("PR by hour", pr.round(3).to_dict())
m[["group", "ts", "forecast_kwh"]].to_csv(OUT / "forecast.csv", index=False, float_format="%.6f", lineterminator="\n")
m[["group", "ts", "anen", "phys"]].to_parquet(OUT / "components_anen_phys.parquet", index=False)
