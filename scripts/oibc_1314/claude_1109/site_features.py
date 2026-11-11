"""Site-level feature table (site x ts x round) for the site model. Competition-legal inputs only.

Per round r in {10, 17}: station forecasts (A, B) joined to each site, solar geometry, calendar, plus
day-context computed on the station forecast table (same issue): +/-1, +/-2 h shifts, target-day 08-17h
mean/min/max, previous target day mean (issued one day earlier), and regional (all-station) means.
Target day = (ts - 1h).date (right-closed hours). Output: one parquet per split with rows for every
(site, ts, round); train rows carry energy (NaN where missing).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

D, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
KST = "Asia/Seoul"
START = pd.Timestamp(sys.argv[3] if len(sys.argv) > 3 else "2023-06-01 01:00", tz=KST)

WA = ["wa_cloud", "wa_temp", "wa_humidity", "wa_rain", "wa_wind_speed", "wa_ground_press", "wa_snow"]
WB = ["wb_cloud", "wb_temp", "wb_rel_hum", "wb_uv_idx", "wb_precip_prob", "wb_total_liq", "wb_vis",
      "wb_wind_speed", "wb_dew_point", "wb_ceiling", "wb_snow"]
CTX = {"wa": ["wa_cloud", "wa_humidity", "wa_rain"], "wb": ["wb_cloud", "wb_uv_idx", "wb_precip_prob", "wb_vis"]}


def solar(ts: pd.DatetimeIndex, lat, lon):
    """Mean cos(zenith) and elevation over the right-closed hour (samples at -50/-30/-10 min), NOAA approx."""
    cz = np.zeros(len(ts))
    for mnt in (50, 30, 10):
        t = ts - pd.Timedelta(minutes=mnt)
        doy = t.dayofyear.to_numpy(); hr = t.hour.to_numpy() + t.minute.to_numpy() / 60.0
        g = 2 * np.pi / 365.0 * (doy - 1 + (hr - 12) / 24.0)
        eqt = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g) - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
        dec = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
               + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
        tst = hr * 60 + eqt + 4 * (lon - 135.0)
        ha = np.radians(tst / 4.0 - 180.0); la = np.radians(lat)
        cz += np.sin(la) * np.sin(dec) + np.cos(la) * np.cos(dec) * np.cos(ha)
    cz = np.clip(cz / 3.0, -1, 1)
    return cz.astype("float32"), (90 - np.degrees(np.arccos(cz))).astype("float32")


def station_table(split: str, p: str) -> pd.DataFrame:
    st = f"{p}_station"; cols = WA if p == "wa" else WB
    frames = []
    for s in ("train", "test") if split == "test" else ("train",):
        f = pd.read_parquet(D / s / f"forecast_{p}.parquet", columns=[st, "round", "ts"] + cols)
        f["ts"] = pd.to_datetime(f.ts).dt.tz_convert(KST)
        if split == "test" and s == "train":
            f = f[f.ts >= pd.Timestamp("2025-05-25", tz=KST)]          # previous days for the first test days
        frames.append(f)
    f = pd.concat(frames, ignore_index=True)
    f = f[(f.ts.dt.minute == 0)].drop_duplicates([st, "round", "ts"], keep="last")
    f["day"] = (f.ts - pd.Timedelta(hours=1)).dt.normalize()
    f = f.sort_values([st, "round", "ts"]).reset_index(drop=True)
    key = f.set_index([st, "round", "ts"])
    for c in CTX[p]:
        for s in (-2, -1, 1, 2):
            idx = pd.MultiIndex.from_arrays([f[st], f["round"], f.ts - pd.Timedelta(hours=s)])
            f[f"{c}_s{s}"] = key[c].reindex(idx).to_numpy()
        day_mask = f.ts.dt.hour.between(8, 17)
        g = f[c].where(day_mask).groupby([f[st], f["round"], f.day])
        for a in ("mean", "min", "max"):
            f[f"{c}_d{a}"] = g.transform(a)
        dm = g.mean()
        idx = pd.MultiIndex.from_arrays([f[st], f["round"], f.day - pd.Timedelta(days=1)])
        f[f"{c}_prev"] = dm.reindex(idx).to_numpy()
        reg = f.groupby(["round", "ts"])[c].transform("mean")
        f[f"{c}_reg"] = reg
        f[f"{c}_regd"] = reg.where(day_mask).groupby([f["round"], f.day]).transform("mean")
    return f.drop(columns="day")


def build(split: str) -> pd.DataFrame:
    sites = pd.read_csv(D / "sites.csv")
    fa, fb = station_table(split, "wa"), station_table(split, "wb")
    if split == "train":
        gen = pd.read_parquet(D / "train" / "generation.parquet")
        gen["ts"] = pd.to_datetime(gen.ts).dt.tz_convert(KST)
        gen = gen[gen.ts >= START]
        base = gen[["site_id", "ts", "energy"]]
    else:
        idx = pd.to_datetime(pd.read_csv(D / "smp.csv").ts).dt.tz_convert(KST)
        base = pd.MultiIndex.from_product([sites.site_id, idx], names=["site_id", "ts"]).to_frame(index=False)
        base["energy"] = np.nan
    base = base.merge(sites[["site_id", "capacity_kw", "lat", "lon", "wa_station", "wb_station", "current_group"]], on="site_id")
    ts = pd.DatetimeIndex(base.ts)
    base["cosz"], base["elev"] = solar(ts, base.lat.to_numpy(), base.lon.to_numpy())
    base = base[base.elev > -2].copy()                                     # night rows are forecast 0 anyway
    out = []
    for r in (10, 17):
        d = base.merge(fa[fa["round"] == r].drop(columns="round"), on=["wa_station", "ts"], how="left")
        d = d.merge(fb[fb["round"] == r].drop(columns="round"), on=["wb_station", "ts"], how="left")
        d["round"] = r
        out.append(d)
    d = pd.concat(out, ignore_index=True)
    t = pd.DatetimeIndex(d.ts)
    d["doy_sin"] = np.sin(2 * np.pi * t.dayofyear / 365.25).astype("float32")
    d["doy_cos"] = np.cos(2 * np.pi * t.dayofyear / 365.25).astype("float32")
    d["hour"] = t.hour.astype("int8")
    e0 = 1 + 0.033 * np.cos(2 * np.pi * t.dayofyear / 365.0)
    cz = np.clip(d.cosz.to_numpy(), 0, None) * e0
    for p in ("wa", "wb"):
        cl = d[f"{p}_cloud"].fillna(50).to_numpy() / 100.0
        d[f"cs_{p}"] = (cz * (1 - 0.75 * cl ** 3.4)).astype("float32")
    d["site_num"] = d.site_id.str[2:].astype("int16")
    num = d.select_dtypes("float64").columns
    d[num] = d[num].astype("float32")
    return d


if __name__ == "__main__":
    for split in sys.argv[4:] if len(sys.argv) > 4 else ("train", "test"):
        d = build(split)
        d.to_parquet(OUT / f"site_{split}.parquet", index=False)
        print(split, d.shape, d.ts.min(), d.ts.max(), flush=True)
