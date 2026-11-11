"""Group-level hourly feature table for the #1314 group-direct model.

Inputs: official participant dataset only (forecasts wa/wb rounds 10 and 17, sites, generation, smp).
No observed weather, no generation lags. Output parquet files go to OUT.

v0.2 (#1468/#1469):
- train rows are restricted to [TRAIN_START, TRAIN_END] inside read_parquet (pyarrow filters), so
  HOLDOUT (2025-03-01 01:00 ~ 2025-06-01 00:00 KST) rows are never materialised or saved.
  Row groups that mix HOLDOUT and earlier rows are still opened by the reader; whether that is
  acceptable is pending PM confirmation (#1468).
- every float feature is rounded to FEATURE_DECIMALS before saving, so features no longer depend on
  the length of the input range (1e-16 float noise used to move HGB bin edges).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

D = Path(sys.argv[1])          # 01_participant_dataset
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

KST = "Asia/Seoul"
TRAIN_START = pd.Timestamp("2023-06-01 01:00", tz=KST)
TRAIN_END = pd.Timestamp("2025-06-01 00:00", tz=KST)    # full training range (HOLDOUT opened 11/09, user decision)
FEATURE_DECIMALS = 6
KEEP_EXACT = {"ts", "group", "energy", "cap_avail", "n_sites", "group_cap_kw", "group_id", "hour"}


def read_train(name: str) -> pd.DataFrame:
    """Read a train parquet keeping only TRAIN_START..TRAIN_END rows (HOLDOUT never loaded)."""
    df = pd.read_parquet(D / "train" / name, filters=[("ts", ">=", TRAIN_START), ("ts", "<=", TRAIN_END)])
    assert len(df) and df["ts"].min() >= TRAIN_START and df["ts"].max() <= TRAIN_END, name
    return df


def round_features(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if c not in KEEP_EXACT and pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].round(FEATURE_DECIMALS) + 0.0   # +0.0 folds -0.0 into 0.0
    return df

sites = pd.read_csv(D / "sites.csv")
sites["group"] = sites["current_group"].map({"A": "G1", "B": "G2"})
assert sites["group"].notna().all() and len(sites) == 253
cap = sites.set_index("site_id")["capacity_kw"]

WA = ["wa_cloud", "wa_temp", "wa_humidity", "wa_ground_press", "wa_wind_speed", "wa_rain", "wa_snow"]
WB = ["wb_temp", "wb_dew_point", "wb_wind_speed", "wb_rel_hum", "wb_vis", "wb_ceiling", "wb_uv_idx",
      "wb_precip_prob", "wb_rain_prob", "wb_snow_prob", "wb_total_liq", "wb_rain", "wb_snow", "wb_cloud"]


def solar_cosz(ts_index: pd.DatetimeIndex, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Hour-mean cos(zenith) for right-closed hours (ts-1h, ts], NOAA approximation; returns hours x plants."""
    out = np.zeros((len(ts_index), len(lat)))
    la = np.radians(lat)[None, :]
    for m in (50, 30, 10):
        t = (ts_index - pd.Timedelta(minutes=m)).tz_convert("UTC")
        doy = t.dayofyear.values[:, None]
        hr = (t.hour.values + t.minute.values / 60.0)[:, None]
        g = 2 * np.pi / 365.0 * (doy - 1 + (hr - 12) / 24.0)
        eqt = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                        - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
        dec = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
               + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
        tst = hr * 60 + eqt + 4 * lon[None, :]
        ha = np.radians(tst / 4.0 - 180.0)
        out += np.sin(la) * np.sin(dec) + np.cos(la) * np.cos(dec) * np.cos(ha)
    return out / 3.0


def round_mean(fc: pd.DataFrame, station_col: str, cols: list[str]) -> dict[str, pd.DataFrame]:
    """Per variable: ts x station matrix, mean of rounds 10 and 17 (NaN-aware), plus round-17 cloud."""
    mats = {}
    for c in cols:
        p = fc.pivot_table(index="ts", columns=[station_col, "round"], values=c, aggfunc="mean")
        r10 = p.xs(10, axis=1, level="round")
        r17 = p.xs(17, axis=1, level="round").reindex(columns=r10.columns)
        mats[c] = pd.DataFrame(np.nanmean(np.stack([r10.values, r17.values]), axis=0),
                               index=p.index, columns=r10.columns)
        if c.endswith("_cloud"):
            mats[c + "_d1710"] = r17 - r10
    return mats


def build(split: str) -> pd.DataFrame:
    if split == "train":
        fa, fb = read_train("forecast_wa.parquet"), read_train("forecast_wb.parquet")
    else:
        fa = pd.read_parquet(D / split / "forecast_wa.parquet")
        fb = pd.read_parquet(D / split / "forecast_wb.parquet")
    fa = fa[fa.ts.dt.minute == 0]
    fb = fb[fb.ts.dt.minute == 0]
    ma = round_mean(fa, "wa_station", WA)
    mb = round_mean(fb, "wb_station", WB)
    idx = ma["wa_cloud"].index.union(mb["wb_cloud"].index)
    idx = pd.DatetimeIndex(idx).sort_values()
    cosz = np.clip(solar_cosz(idx, sites["lat"].values, sites["lon"].values), 0, None)  # hours x 253
    e0 = 1 + 0.033 * np.cos(2 * np.pi * idx.dayofyear.values / 365.0)

    # per-plant cloud for clear-sky-index proxy (Kasten-Czeplak style attenuation)
    cl_a = ma["wa_cloud"].reindex(idx)[sites["wa_station"].values].values
    cl_b = mb["wb_cloud"].reindex(idx)[sites["wb_station"].values].values
    cs = cosz * e0[:, None]
    csa = cs * (1 - 0.75 * np.power(np.nan_to_num(cl_a, nan=50.0) / 100.0, 3.4))
    csb = cs * (1 - 0.75 * np.power(np.nan_to_num(cl_b, nan=50.0) / 100.0, 3.4))

    rows = []
    for g in ("G1", "G2"):
        mask = (sites["group"] == g).values
        w = sites.loc[mask, "capacity_kw"].values
        wn = w / w.sum()
        f = pd.DataFrame(index=idx)
        f["group_id"] = 0 if g == "G1" else 1
        f["cosz"] = cosz[:, mask] @ wn
        f["cosz_max"] = cosz[:, mask].max(axis=1)
        f["cs_a"] = csa[:, mask] @ wn
        f["cs_b"] = csb[:, mask] @ wn
        f["cs_ab"] = 0.5 * (f["cs_a"] + f["cs_b"])
        for st_col, mats in (("wa_station", ma), ("wb_station", mb)):
            stw = sites.loc[mask].groupby(st_col)["capacity_kw"].sum()
            for c, m in mats.items():
                m = m.reindex(index=idx, columns=stw.index)
                v = m.values
                ww = np.where(np.isnan(v), 0.0, stw.values[None, :])
                den = ww.sum(axis=1)
                f[c] = np.where(den > 0, np.nansum(np.nan_to_num(v) * ww, axis=1) / np.where(den > 0, den, 1), np.nan)
                if c in ("wa_cloud", "wb_cloud", "wb_uv_idx"):
                    mu = f[c].values[:, None]
                    f[c + "_sd"] = np.sqrt(np.where(den > 0, np.nansum(ww * (np.nan_to_num(v) - mu) ** 2, axis=1) / np.where(den > 0, den, 1), np.nan))
                    if c.endswith("cloud"):
                        f[c + "_gt80"] = np.where(den > 0, (ww * (np.nan_to_num(v) > 80)).sum(axis=1) / np.where(den > 0, den, 1), np.nan)
        day = (idx - pd.Timedelta(hours=1)).normalize()
        for c in ("wa_cloud", "wb_cloud", "cs_ab"):
            f[c + "_daymean"] = f[c].groupby(day).transform("mean")
        f["doy_sin"] = np.sin(2 * np.pi * idx.dayofyear / 365.25)
        f["doy_cos"] = np.cos(2 * np.pi * idx.dayofyear / 365.25)
        f["hour"] = idx.hour
        f["group"] = g
        f["group_cap_kw"] = w.sum()
        rows.append(f)
    out = pd.concat(rows).rename_axis("ts").reset_index()
    return out


def targets() -> pd.DataFrame:
    gen = read_train("generation.parquet")
    gen["ts"] = gen["ts"].dt.as_unit("us")
    gen = gen.merge(sites[["site_id", "group", "capacity_kw"]], on="site_id")
    t = gen.groupby(["group", "ts"]).agg(energy=("energy", "sum"), cap_avail=("capacity_kw", "sum"),
                                         n_sites=("site_id", "nunique")).reset_index()
    return t


if __name__ == "__main__":
    tr = build("train")
    te = build("test")
    tg = targets()
    tr = round_features(tr.merge(tg, on=["group", "ts"], how="left"))
    te = round_features(te)
    assert tr["ts"].max() <= TRAIN_END
    tr.to_parquet(OUT / "group_features_train.parquet", index=False)
    te.to_parquet(OUT / "group_features_test.parquet", index=False)
    print(tr.shape, te.shape, tr["energy"].notna().sum())
    print(tr.groupby("group")[["cap_avail", "n_sites"]].describe().T.to_string())
