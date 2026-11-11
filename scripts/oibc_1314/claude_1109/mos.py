"""Stage 1 (MOS): learn forecast -> observed weather per station from the training-period observations.

Observed weather is used only as a TRAINING target. At prediction time the inputs are D-1 forecasts only, so the
stage-1 output is itself a forecast-time feature. For a cut-off `cut`:
  rows with ts < cut  -> out-of-fold predictions (blocked by calendar month into K groups, all inside ts < cut)
  rows with ts >= cut -> prediction of a model fit on every observation with ts < cut
Targets: observed wa_cloud (A), observed wb_cloud and wb_uv_idx (B), clipped to their documented ranges.
"""
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

KST = "Asia/Seoul"
TARGETS = {"wa": ["wa_cloud"], "wb": ["wb_cloud", "wb_uv_idx"]}
FC = {"wa": ["wa_cloud", "wa_temp", "wa_humidity", "wa_rain", "wa_wind_speed", "wa_ground_press", "wa_snow"],
      "wb": ["wb_cloud", "wb_temp", "wb_rel_hum", "wb_uv_idx", "wb_precip_prob", "wb_total_liq", "wb_vis",
             "wb_wind_speed", "wb_dew_point", "wb_ceiling", "wb_snow"]}
CLIP = {"wa_cloud": (0, 100), "wb_cloud": (0, 100), "wb_uv_idx": (0, 13)}
PARAMS = dict(objective="regression", n_estimators=300, learning_rate=0.05, num_leaves=63, min_child_samples=300,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=-1, random_state=0)
K = 4


def forecast_table(D: Path, p: str, splits=("train",)) -> pd.DataFrame:
    st = f"{p}_station"
    fr = []
    for s in splits:
        f = pd.read_parquet(D / s / f"forecast_{p}.parquet", columns=[st, "round", "ts"] + FC[p])
        f["ts"] = pd.to_datetime(f.ts).dt.tz_convert(KST)
        fr.append(f)
    f = pd.concat(fr, ignore_index=True)
    f = f[(f.ts.dt.minute == 0) & f.ts.dt.hour.between(5, 20)].drop_duplicates([st, "round", "ts"], keep="last")
    f["day"] = (f.ts - pd.Timedelta(hours=1)).dt.normalize()
    lit = f.ts.dt.hour.between(8, 17)
    for c in TARGETS[p]:
        g = f[c].where(lit).groupby([f[st], f["round"], f.day])
        f[f"{c}_dmean"] = g.transform("mean")
        f[f"{c}_dmax"] = g.transform("max")
    f["hour"] = f.ts.dt.hour.astype("int8")
    f["doy_sin"] = np.sin(2 * np.pi * f.ts.dt.dayofyear / 365.25).astype("float32")
    f["doy_cos"] = np.cos(2 * np.pi * f.ts.dt.dayofyear / 365.25).astype("float32")
    f["st_id"] = f[st].str[2:].astype("int16")
    return f.drop(columns="day").reset_index(drop=True)


def observed(D: Path, p: str) -> pd.DataFrame:
    st = f"{p}_station"
    o = pd.read_parquet(D / "train" / f"observed_{p}.parquet", columns=[st, "ts"] + TARGETS[p])
    o["ts"] = pd.to_datetime(o.ts).dt.tz_convert(KST)
    for c in TARGETS[p]:
        lo, hi = CLIP[c]
        o.loc[(o[c] < lo) | (o[c] > hi), c] = np.nan
    return o.rename(columns={c: f"obs_{c}" for c in TARGETS[p]})


def build(f: pd.DataFrame, o: pd.DataFrame, p: str, cut: pd.Timestamp, until: pd.Timestamp) -> pd.DataFrame:
    """MOS predictions for rows with ts <= until (OOF before cut, full-fit model from cut on)."""
    st = f"{p}_station"
    rows = f[f.ts <= until].merge(o, on=[st, "ts"], how="left")
    feats = [c for c in rows.columns if c not in (st, "ts") and not c.startswith("obs_")]
    out = rows[[st, "round", "ts"]].copy()
    past = (rows.ts < cut).to_numpy()
    month = rows.ts.dt.strftime("%Y%m").astype(int)
    blocks = (month.rank(method="dense").astype(int) % K).to_numpy()
    for c in TARGETS[p]:
        y = rows[f"obs_{c}"].to_numpy()
        pred = np.full(len(rows), np.nan, dtype="float32")
        for b in range(K):
            trm = past & (blocks != b) & ~np.isnan(y)
            tem = past & (blocks == b)
            m = lgb.LGBMRegressor(**PARAMS).fit(rows.loc[trm, feats], y[trm])
            pred[tem] = m.predict(rows.loc[tem, feats])
        trm = past & ~np.isnan(y)
        if (~past).any():
            m = lgb.LGBMRegressor(**PARAMS).fit(rows.loc[trm, feats], y[trm])
            pred[~past] = m.predict(rows.loc[~past, feats])
        lo, hi = CLIP[c]
        out[f"mos_{c}"] = np.clip(pred, lo, hi)
    return out
