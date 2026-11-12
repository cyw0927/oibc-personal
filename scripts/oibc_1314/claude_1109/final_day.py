"""Final day-total adjustment (candidate 3) for the evaluation year.

Usage: final_day.py <dataset> <site_feat_dir> <hourly_sites.parquet> <base_forecast.csv> <out_dir> <alpha> [seeds=0]
1. Site-day model (same features/params as cv_day.py) fit on every training day (< 2025-06-01), predicts each test site-day.
2. Site ratio = DAY / sum_day(hourly B), clipped [0.2, 5]; B_D = B * ratio; group-day ratio rd = sum B_D / sum B.
3. out = base * (1 + alpha * (rd - 1)) per group-hour (#990 reference groups). Writes composition.csv, forecast.csv,
   day_ratio.parquet and MANIFEST.json.
Inputs are D-1 forecasts (rounds 10/17) only; no test actuals exist or are used.
"""
import hashlib, json, sys, warnings
from pathlib import Path
warnings.simplefilter("ignore")

import lightgbm as lgb
import numpy as np
import pandas as pd

D, FD, HB, BASE, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5])
ALPHA = float(sys.argv[6])
SEEDS = [int(s) for s in (sys.argv[7] if len(sys.argv) > 7 else "0").split(",")]
ECBOX = next((a[6:] for a in sys.argv[8:] if a.startswith("ecbox=")), None)   # optional ECMWF fixed-box daily means (no site coordinates)
ECMWF = next((a[6:] for a in sys.argv[8:] if a.startswith("ecmwf=")), None)   # optional ECMWF open-data site-day ssrd parquet (same features as cv_day3)
KST = "Asia/Seoul"
CUT = pd.Timestamp("2025-06-01", tz=KST)
DPARAMS = dict(objective="regression_l1", n_estimators=1500, learning_rate=0.03, num_leaves=63, min_child_samples=50,
               subsample=0.8, subsample_freq=1, colsample_bytree=0.7, verbose=-1, n_jobs=-1)
HOURLY = ["wa_cloud", "wa_temp", "wa_humidity", "wa_rain", "wa_wind_speed", "wa_snow", "wb_cloud", "wb_temp", "wb_rel_hum",
          "wb_uv_idx", "wb_precip_prob", "wb_total_liq", "wb_vis", "wb_dew_point", "wb_ceiling", "wb_snow", "cs_wa", "cs_wb"]
PATV = ["wa_cloud", "wb_cloud", "wb_uv_idx", "wb_precip_prob", "cs_wa"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def day_table(d):
    d["day"] = (d.ts - pd.Timedelta(hours=1)).dt.normalize()
    CONST = [c for c in d.columns if c.endswith(("_dmean", "_dmin", "_dmax", "_prev", "_reg", "_regd"))] + ["doy_sin", "doy_cos", "capacity_kw", "lat", "lon", "site_num"]
    lit = d[d.elev > 0].copy()
    lit["cw"] = lit.cosz.clip(lower=0)
    for c in HOURLY:
        lit[f"{c}_x"] = lit[c] * lit.cw
    mid = lit[lit.hour.between(11, 15)]
    agg = lit.groupby(["site_id", "day", "round"]).agg(csum=("cw", "sum"), nlit=("cw", "size"), **{f"{c}_w": (f"{c}_x", "sum") for c in HOURLY},
                                                         **{c: (c, "first") for c in CONST})
    for c in HOURLY:
        agg[f"{c}_w"] = (agg[f"{c}_w"] / agg.csum).astype("float32")
    mm = mid.groupby(["site_id", "day", "round"]).agg(wa_cloud_mid_max=("wa_cloud", "max"), wb_cloud_mid_max=("wb_cloud", "max"),
                                                     wa_cloud_mid_min=("wa_cloud", "min"), wb_cloud_mid_min=("wb_cloud", "min"))
    agg = agg.join(mm).reset_index()
    VW = [f"{c}_w" for c in HOURLY] + list(mm.columns)
    a17 = agg[agg["round"] == 17].drop(columns="round")
    a10 = agg[agg["round"] == 10][["site_id", "day"] + VW].rename(columns={c: f"{c}_r10" for c in VW})
    day = a17.merge(a10, on=["site_id", "day"], how="left")
    pt = d[(d["round"] == 17) & d.hour.between(8, 17)].pivot_table(index=["site_id", "day"], columns="hour", values=PATV)
    pt.columns = [f"{v}_h{h:02d}" for v, h in pt.columns]
    day = day.merge(pt.astype("float32").reset_index(), on=["site_id", "day"], how="left")
    rev = pd.DataFrame({f"{c}_rev": day[c] - day[f"{c}_r10"] for c in VW})
    day = pd.concat([day, rev], axis=1)
    if "energy" in d:
        obs = d[(d.elev > 0) & (d["round"] == 17)].groupby(["site_id", "day"]).agg(esum=("energy", "sum"), n_e=("energy", "count"))
        day = day.merge(obs.reset_index(), on=["site_id", "day"], how="left")
    day["doy"] = day.day.dt.dayofyear
    if ECMWF:
        ec = pd.read_parquet(ECMWF)[["site_id", "day", "ssrd"]]
        ec["day"] = pd.to_datetime(ec.day).dt.tz_localize(KST)
        ec["ec_ssrd_reg"] = ec.groupby("day").ssrd.transform("mean")
        day = day.merge(ec.rename(columns={"ssrd": "ec_ssrd"}), on=["site_id", "day"], how="left")
        day["ec_kt"] = (day.ec_ssrd / (day.csum.clip(lower=0.1) / day.nlit.clip(lower=1) * 1000)).astype("float32")
        print("ecmwf coverage", round(day.ec_ssrd.notna().mean(), 3), flush=True)
    if ECBOX:
        eb = pd.read_parquet(ECBOX); eb["day"] = pd.to_datetime(eb.day).dt.tz_localize(KST)
        day = day.merge(eb, on="day", how="left")
        print("ecmwf box coverage", round(day.ec_box.notna().mean(), 3), flush=True)
    return day


tr = pd.read_parquet(FD / "site_train.parquet"); tr["ts"] = tr.ts.dt.tz_convert(KST)
te = pd.read_parquet(FD / "site_test.parquet"); te["ts"] = te.ts.dt.tz_convert(KST)
dtr = day_table(tr)
dtr = dtr[(dtr.day < CUT) & (dtr.n_e == dtr.nlit) & (dtr.csum > 0) & dtr.esum.notna()].copy()
dtr["kd"] = (dtr.esum / (dtr.capacity_kw * dtr.csum)).clip(0, 1.5)
dte = day_table(te.drop(columns=["energy"], errors="ignore"))
dte = dte[dte.day >= CUT]
DF = [c for c in dtr.columns if c not in ("site_id", "day", "esum", "n_e", "kd", "nlit")]
assert set(DF) <= set(dte.columns), set(DF) - set(dte.columns)
print("train days", len(dtr), "test site-days", len(dte), "features", len(DF), flush=True)
pred = np.zeros(len(dte))
for s in SEEDS:
    md = lgb.LGBMRegressor(random_state=s, **DPARAMS).fit(dtr[DF], dtr.kd, sample_weight=(dtr.capacity_kw * dtr.csum).to_numpy())
    pred += np.clip(md.predict(dte[DF]), 0, None) / len(SEEDS)
    print("seed", s, "done", flush=True)
dp = dte[["site_id", "day"]].assign(DAY=pred * (dte.capacity_kw * dte.csum).to_numpy())

hb = pd.read_parquet(HB); hb["ts"] = pd.to_datetime(hb.ts).dt.tz_convert(KST)
hb["day"] = (hb.ts - pd.Timedelta(hours=1)).dt.normalize()
hb = hb.merge(dp, on=["site_id", "day"], how="left")
sb = hb.groupby(["site_id", "day"]).kwh.transform("sum")
hb["ratio"] = (hb.DAY / sb.where(sb > 0)).fillna(1.0).clip(0.2, 5)
hb["BD"] = hb.kwh * hb.ratio
sites = pd.read_csv(D / "sites.csv")
hb = hb.merge(sites[["site_id", "current_group"]], on="site_id")
hb["group"] = hb.current_group.map({"A": "G1", "B": "G2"})
gd = hb.groupby(["group", "day"]).agg(B=("kwh", "sum"), BD=("BD", "sum")).reset_index()
gd["rd"] = (gd.BD / gd.B.where(gd.B > 1)).fillna(1.0)

bf = pd.read_csv(BASE); bf["ts"] = pd.to_datetime(bf.ts).dt.tz_convert(KST)
bf["day"] = (bf.ts - pd.Timedelta(hours=1)).dt.normalize()
bf = bf.merge(gd[["group", "day", "rd"]], on=["group", "day"], how="left")
assert len(bf) == 17520 and bf.rd.notna().all()
bf["out"] = (bf.forecast_kwh * (1 + ALPHA * (bf.rd - 1))).clip(lower=0)
OUT.mkdir(parents=True, exist_ok=False)
sites.assign(group=sites.current_group.map({"A": "G1", "B": "G2"}))[["site_id", "group"]].sort_values("site_id").to_csv(OUT / "composition.csv", index=False, lineterminator="\n")
fc = bf[["group", "ts", "out"]].rename(columns={"out": "forecast_kwh"}).sort_values(["group", "ts"])
fc["ts"] = fc["ts"].dt.strftime("%Y-%m-%d %H:%M:%S%z").str.replace(r"(\+\d{2})(\d{2})$", r"\1:\2", regex=True)
fc.to_csv(OUT / "forecast.csv", index=False, lineterminator="\n", float_format="%.6f")
gd.to_parquet(OUT / "day_ratio.parquet", index=False)
man = {"candidate_id": OUT.name, "method": "base forecast x (1 + alpha*(group-day ratio - 1)); ratio from site-day model / hourly B",
       "alpha": ALPHA, "seeds": SEEDS, "external_data": ({"source": "ECMWF IFS open data (CC BY 4.0), 12Z D-2 run, ssrd 36-42h", "file": ECMWF, "sha256": sha(ECMWF)} if ECMWF else ({"source": "ECMWF IFS open data (CC BY 4.0), 12Z D-2 run, ssrd 36-42h, fixed-box means (no site coordinates)", "file": ECBOX, "sha256": sha(ECBOX)} if ECBOX else False)),
       "inputs_sha256": {"base_forecast.csv": sha(BASE), "hourly_sites": sha(HB), "site_train": sha(FD / "site_train.parquet"), "site_test": sha(FD / "site_test.parquet")},
       "outputs_sha256": {"composition.csv": sha(OUT / "composition.csv"), "forecast.csv": sha(OUT / "forecast.csv")},
       "totals_gwh": {"base": round(bf.forecast_kwh.sum() / 1e6, 3), "out": round(bf.out.sum() / 1e6, 3)},
       "ratio_stats": gd.rd.describe().round(4).to_dict()}
json.dump(man, open(OUT / "MANIFEST.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(json.dumps(man, indent=1, ensure_ascii=False))
