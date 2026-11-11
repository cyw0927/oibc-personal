"""Forecast-field features: whole-network weather pattern (PCA) and neighbour-station smoothing.

Outputs (train/test):
  pcs_<split>.parquet  keyed (ts, round): hourly PCs of the station cloud fields (A, B) and daily (08-17h mean) PCs.
  nbr_<split>.parquet  keyed (site_id, ts, round): inverse-distance mean of the 4 nearest other stations' cloud (A, B)
                       and its difference to the own station.
PCA is fit on training-period forecasts only (2021-01 .. 2025-05) and applied to both splits. Station positions are
approximated by the mean shifted coordinate of the sites assigned to that station (only used for station-to-station
proximity). All inputs are D-1 issued forecasts; the daily PCs use the same target day (same issue).
Usage: spatial_features.py <dataset> <out_dir>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

D, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
KST = "Asia/Seoul"
K_H, K_D, K_NB = 8, 6, 4
sites = pd.read_csv(D / "sites.csv")


def load(p: str, split: str) -> pd.DataFrame:
    f = pd.read_parquet(D / split / f"forecast_{p}.parquet", columns=[f"{p}_station", "round", "ts", f"{p}_cloud"])
    f["ts"] = pd.to_datetime(f.ts).dt.tz_convert(KST)
    return f[f.ts.dt.minute == 0].drop_duplicates([f"{p}_station", "round", "ts"], keep="last")


def field(f: pd.DataFrame, p: str) -> pd.DataFrame:
    m = f.pivot_table(index=["round", "ts"], columns=f"{p}_station", values=f"{p}_cloud", aggfunc="mean")
    return m.apply(lambda r: r.fillna(r.mean()), axis=1).fillna(50.0)


pcs, nbr = {}, {}
for p in ("wa", "wb"):
    tr, te = load(p, "train"), load(p, "test")
    Ftr, Fte = field(tr, p), field(te, p)
    Fte = Fte.reindex(columns=Ftr.columns).apply(lambda r: r.fillna(r.mean()), axis=1)
    day_tr = Ftr.index.get_level_values("ts").hour
    lit = (day_tr >= 8) & (day_tr <= 17)
    ph = PCA(n_components=K_H, random_state=0).fit(Ftr[lit].to_numpy())
    print(p, "hourly PCA explained", ph.explained_variance_ratio_.round(3), flush=True)

    def daily(Fm):
        idx = Fm.index.to_frame(index=False)
        idx["day"] = (idx.ts - pd.Timedelta(hours=1)).dt.normalize()
        h = idx.ts.dt.hour
        dm = Fm.reset_index(drop=True)[((h >= 8) & (h <= 17)).to_numpy()].groupby([idx["round"][(h >= 8) & (h <= 17)].to_numpy(), idx["day"][(h >= 8) & (h <= 17)].to_numpy()]).mean()
        return idx, dm
    itr, dtr = daily(Ftr)
    pdl = PCA(n_components=K_D, random_state=0).fit(dtr.to_numpy())
    print(p, "daily PCA explained", pdl.explained_variance_ratio_.round(3), flush=True)

    # station positions and neighbours
    pos = sites.groupby(f"{p}_station")[["lat", "lon"]].mean().reindex(Ftr.columns)
    xy = np.c_[pos.lat.to_numpy(), pos.lon.to_numpy() * np.cos(np.radians(pos.lat.mean()))]
    dist = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(dist, np.inf)
    W = np.zeros_like(dist)
    for i in range(len(xy)):
        j = np.argsort(dist[i])[:K_NB]
        W[i, j] = 1.0 / np.maximum(dist[i, j], 1e-3)
    W = W / W.sum(1, keepdims=True)

    for split, Fm, (idx, dm) in (("train", Ftr, (itr, dtr)), ("test", Fte, daily(Fte))):
        h = pd.DataFrame(ph.transform(Fm.to_numpy()), columns=[f"{p}_pch{i}" for i in range(K_H)])
        h["round"] = idx["round"].to_numpy(); h["ts"] = idx["ts"].to_numpy(); h["day"] = idx["day"].to_numpy()
        d = pd.DataFrame(pdl.transform(dm.to_numpy()), columns=[f"{p}_pcd{i}" for i in range(K_D)])
        d["round"] = dm.index.get_level_values(0); d["day"] = dm.index.get_level_values(1)
        h = h.merge(d, on=["round", "day"], how="left").drop(columns="day")
        pcs.setdefault(split, []).append(h.set_index(["round", "ts"]))
        nb = pd.DataFrame(Fm.to_numpy() @ W.T, index=Fm.index, columns=Fm.columns)          # neighbour mean per station
        long = nb.stack().rename(f"{p}_cloud_nb").reset_index().rename(columns={f"{p}_station": "st"})
        own = Fm.stack().rename("own").reset_index().rename(columns={f"{p}_station": "st"})
        long[f"{p}_cloud_nbdiff"] = long[f"{p}_cloud_nb"] - own["own"].to_numpy()
        nbr.setdefault(split, {})[p] = long

for split in ("train", "test"):
    P = pd.concat(pcs[split], axis=1).reset_index()
    P[P.select_dtypes("float64").columns] = P.select_dtypes("float64").astype("float32")
    P.to_parquet(OUT / f"pcs_{split}.parquet", index=False)
    a = nbr[split]["wa"].rename(columns={"st": "wa_station"}); b = nbr[split]["wb"].rename(columns={"st": "wb_station"})
    for x in (a, b):
        x[x.select_dtypes("float64").columns] = x.select_dtypes("float64").astype("float32")
    a.to_parquet(OUT / f"nbr_wa_{split}.parquet", index=False); b.to_parquet(OUT / f"nbr_wb_{split}.parquet", index=False)
    print(split, P.shape, a.shape, b.shape, flush=True)
