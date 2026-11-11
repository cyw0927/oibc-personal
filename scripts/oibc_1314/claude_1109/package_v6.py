"""Package the v6 review bundle (no training): submission, fold predictions, day-model OOF and ratios, features, code, logs."""
import hashlib, json, shutil, sys
from pathlib import Path
import numpy as np
import pandas as pd

PS, CAND, CODE, REPLAY, TASKS, OUT = (Path(a) for a in sys.argv[1:7])
KST = "Asia/Seoul"; K = ["group", "ts"]
FOLDS = ["2024-07", "2024-10", "2025-01", "2025-02", "2025-04"]
OUT.mkdir(parents=True, exist_ok=False)
cv = PS / "cv"
ts_str = lambda s: s.dt.strftime("%Y-%m-%d %H:%M:%S%z").str.replace(r"(\+\d{2})(\d{2})$", r"\1:\2", regex=True)


def L(f, cols, ren={}):
    x = pd.read_parquet(cv / f).rename(columns=ren); x["ts"] = pd.to_datetime(x.ts).dt.tz_convert(KST); return x[K + cols]


# 1 submission
o = OUT / "1_submission"; o.mkdir()
for n in ("composition.csv", "forecast.csv", "MANIFEST.json", "validator_report.json", "official_validator.log"):
    shutil.copy2(CAND / n, o / n)

# 2 fold predictions + evaluation
d = pd.concat([L("site_cv_EG.parquet", ["fold", "energy", "smp", "k", "E"]), L("site_cv_E_feb.parquet", ["fold", "energy", "smp", "k", "E"])])
d = d.merge(pd.concat([L("site_cv_B2021.parquet", ["B21"], {"B": "B21"}), L("site_cv_B2021_feb.parquet", ["B21"], {"B": "B21"})]), on=K)
d = d.merge(L("group_cv.parquet", ["H1"]), on=K)
d = d.merge(L("site_cv_day.parquet", ["B23_hourly", "B23_day"], {"B": "B23_hourly", "B_D": "B23_day"}), on=K)
d = d.merge(L("site_cv_day2021.parquet", ["B21_hourly", "B21_day"], {"B": "B21_hourly", "B_D": "B21_day"}), on=K)
d["day"] = (d.ts - pd.Timedelta(hours=1)).dt.normalize()
gd = d.groupby(["fold", "group", "day"])[["B23_hourly", "B23_day", "B21_hourly", "B21_day"]].sum().reset_index()
gd["rd_2023"] = (gd.B23_day / gd.B23_hourly.where(gd.B23_hourly > 1)).fillna(1.0)
gd["rd_2021"] = (gd.B21_day / gd.B21_hourly.where(gd.B21_hourly > 1)).fillna(1.0)
gd["rd_avg"] = (gd.rd_2023 + gd.rd_2021) / 2
d = d.merge(gd[["fold", "group", "day", "rd_2023", "rd_2021", "rd_avg"]], on=["fold", "group", "day"])
d["v4b"] = 0.35 * d.E + 0.35 * d.B21 + 0.30 * d.H1
d["v6"] = d.v4b * (1 + 0.5 * (d.rd_avg - 1))
w = 0.1 * d.smp * d.k
o = OUT / "2_fold_predictions"; o.mkdir()
x = d.sort_values(["fold", "group", "ts"]).copy(); x["ts"] = ts_str(x.ts); x["day"] = x.day.dt.strftime("%Y-%m-%d")
x[["fold", "group", "ts", "day", "energy", "smp", "k", "E", "B21", "H1", "v4b", "rd_avg", "v6"]].to_csv(o / "v4b_v6_fold_predictions.csv", index=False, float_format="%.6f", lineterminator="\n")
ev = pd.DataFrame({c: (w * (d[c] - d.energy).abs()).groupby(d.fold).sum() / 1e6 for c in ("E", "B21", "H1", "v4b", "v6")})
ev.insert(0, "rows", d.groupby("fold").size())
ev.loc["TOTAL_5fold"] = ev.sum(); ev.loc["TOTAL_4fold_excl_2025-02"] = ev.loc[[f for f in FOLDS if f != "2025-02"]].sum()
ev["v6_vs_v4b_%"] = 100 * (ev.v6 / ev.v4b - 1)
ev.round(6).to_csv(o / "v4b_v6_eval_by_fold.csv", lineterminator="\n")
lvl = pd.DataFrame({c: [d[c].sum() / d.energy.sum(), d[c][d.k == 3].sum() / d.energy[d.k == 3].sum()] for c in ("E", "B21", "H1", "v4b", "v6")}, index=["level_all", "level_11_15h"])
lvl.round(5).to_csv(o / "level_pred_over_actual.csv", lineterminator="\n")
print(ev.round(3).to_string())

# 3 day-model OOF (group level) and daily ratios; final test ratios
o = OUT / "3_day_model"; o.mkdir()
for f, n in (("site_cv_day.parquet", "day2023"), ("site_cv_day2021.parquet", "day2021")):
    y = pd.read_parquet(cv / f); y["ts"] = ts_str(pd.to_datetime(y.ts).dt.tz_convert(KST))
    y.rename(columns={"B": "hourly_B", "B_D": "B_x_dayratio", "B_M": "B_x_half_dayratio"}).to_csv(o / f"oof_{n}_group_hour.csv", index=False, float_format="%.6f", lineterminator="\n")
g2 = gd.copy(); g2["day"] = g2.day.dt.strftime("%Y-%m-%d")
g2.to_csv(o / "oof_group_day_ratios.csv", index=False, float_format="%.6f", lineterminator="\n")
for n in ("day23_a1", "day21_a1"):
    r = pd.read_parquet(PS / "final_day" / n / "day_ratio.parquet"); r["day"] = pd.to_datetime(r.day).dt.tz_convert(KST).dt.strftime("%Y-%m-%d")
    r.to_csv(o / f"test_group_day_ratio_{n}.csv", index=False, float_format="%.6f", lineterminator="\n")
    shutil.copy2(PS / "final_day" / n / "MANIFEST.json", o / f"final_{n}_MANIFEST.json")

# 5 code
o = OUT / "5_code"; o.mkdir()
shutil.copy2(REPLAY / "cv_day_v6_cv_version.py", o / "cv_day_v6_cv_version.py")
for n in ("final_day.py", "combine_day.py", "site_features.py", "ctx_features.py", "make_blend.py", "final_site.py", "package_v6.py"):
    shutil.copy2(CODE / n, o / n)
shutil.copy2(CODE / "cv_day.py", o / "cv_day_current_with_later_perf_experiment.py")

# 8 logs
o = OUT / "8_logs"; o.mkdir()
for t, n in (("bqw95gwso", "cv_day_2023_5fold.log"), ("bte6jrfz1", "cv_day_2021_5fold.log"), ("bd93fub3y", "final_day_2023_2021.log")):
    p = TASKS / f"{t}.output"
    if p.exists():
        shutil.copy2(p, o / n)
print("done")
