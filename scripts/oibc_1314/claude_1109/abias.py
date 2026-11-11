"""Provider-A cloud bias drift correction (Kang & Tam 2015 hypothesis), past-only.

For target day D, bias = mean(forecast wa_cloud - observed wa_cloud) over 11-15h of target days in
[D-91, D-2] (observations up to D-2 only, so always known before D-1 17:00 KST), per (station, round)
and pooled over all stations (used for the regional-mean columns). Requires >= 30 days in the window.
apply(d, D, mode): subtract bias from every wa_cloud* column of a site feature table, clip to [0, 100],
recompute cs_wa. mode = "station" (per-station bias) or "global" (pooled bias everywhere).
"""
from pathlib import Path

import numpy as np
import pandas as pd

KST = "Asia/Seoul"
WIN, LAG, MINDAYS = 90, 2, 30


def bias_table(D: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    f = pd.read_parquet(D / "train" / "forecast_wa.parquet", columns=["wa_station", "round", "ts", "wa_cloud"])
    o = pd.read_parquet(D / "train" / "observed_wa.parquet", columns=["wa_station", "ts", "wa_cloud"])
    for x in (f, o):
        x["ts"] = pd.to_datetime(x.ts).dt.tz_convert(KST)
    f = f[(f.ts.dt.minute == 0) & f["round"].isin([10, 17])].drop_duplicates(["wa_station", "round", "ts"], keep="last")
    m = f.merge(o.rename(columns={"wa_cloud": "obs"}), on=["wa_station", "ts"], how="inner")
    m = m[m.ts.dt.hour.between(11, 15) & m.obs.notna() & m.wa_cloud.notna()]
    m["day"] = (m.ts - pd.Timedelta(hours=1)).dt.normalize().dt.tz_localize(None)
    m["e"] = m.wa_cloud - m.obs
    days = pd.date_range(m.day.min(), pd.Timestamp("2026-01-01"), freq="D", name="day")

    def roll(daily: pd.DataFrame) -> pd.DataFrame:      # daily: index day, cols key -> (sum, n)
        s = daily["s"].reindex(days).fillna(0).rolling(WIN, min_periods=1).sum().shift(LAG)
        n = daily["n"].reindex(days).fillna(0).rolling(WIN, min_periods=1).sum().shift(LAG)
        dn = daily["d"].reindex(days).fillna(0).rolling(WIN, min_periods=1).sum().shift(LAG)
        return (s / n).where(dn >= MINDAYS)

    g = m.groupby(["day", "wa_station", "round"]).e.agg(["sum", "count"]).reset_index()
    st = g.pivot_table(index="day", columns=["wa_station", "round"], values=["sum", "count"])
    daily = {"s": st["sum"], "n": st["count"], "d": (st["count"] > 0).astype(float)}
    bs = roll(daily).stack(["wa_station", "round"], future_stack=True).rename("bias_s").reset_index().rename(columns={"index": "day"})
    gg = m.groupby(["day", "round"]).e.agg(["sum", "count"]).reset_index()
    gt = gg.pivot_table(index="day", columns="round", values=["sum", "count"])
    bg = roll({"s": gt["sum"], "n": gt["count"], "d": (gt["count"] > 0).astype(float)})
    bg = bg.stack(future_stack=True).rename("bias_g").reset_index().rename(columns={"index": "day"})
    return bs.dropna(), bg.dropna()


def apply(d: pd.DataFrame, D: Path, mode: str) -> pd.DataFrame:
    bs, bg = bias_table(D)
    d = d.copy()
    d["day"] = (d.ts - pd.Timedelta(hours=1)).dt.normalize().dt.tz_localize(None)
    d = d.merge(bg, on=["day", "round"], how="left")
    if mode == "station":
        d = d.merge(bs, on=["day", "wa_station", "round"], how="left")
        d["bias_s"] = d.bias_s.fillna(d.bias_g)
    else:
        d["bias_s"] = d.bias_g
    d[["bias_s", "bias_g"]] = d[["bias_s", "bias_g"]].fillna(0.0)
    cols = [c for c in d.columns if c.startswith("wa_cloud")]
    for c in cols:
        b = d.bias_g if c.endswith(("_reg", "_regd")) else d.bias_s
        d[c] = (d[c] - b).clip(0, 100).astype("float32")
    t = pd.DatetimeIndex(d.ts)
    e0 = 1 + 0.033 * np.cos(2 * np.pi * t.dayofyear / 365.0)
    cz = np.clip(d.cosz.to_numpy(), 0, None) * e0
    cl = d.wa_cloud.fillna(50).to_numpy() / 100.0
    d["cs_wa"] = (cz * (1 - 0.75 * cl ** 3.4)).astype("float32")
    print("abias", mode, "cols", len(cols), "mean bias_s by year",
          d.groupby(d.ts.dt.year).bias_s.mean().round(2).to_dict(), flush=True)
    return d.drop(columns=["day", "bias_s", "bias_g"])
