"""Average group-day ratios of several final_day runs and apply them to a base forecast.

Usage: combine_day.py <dataset> <base_forecast.csv> <out_dir> <alpha> <day_ratio_dir> [<day_ratio_dir> ...]
out = base * (1 + alpha * (mean_rd - 1)), #990 reference composition.
"""
import hashlib, json, sys
from pathlib import Path

import pandas as pd

D, BASE, OUT, ALPHA = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), float(sys.argv[4])
DIRS = [Path(a) for a in sys.argv[5:]]
KST = "Asia/Seoul"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

rs = []
for i, p in enumerate(DIRS):
    r = pd.read_parquet(p / "day_ratio.parquet")[["group", "day", "rd"]].rename(columns={"rd": f"rd{i}"})
    r["day"] = pd.to_datetime(r.day).dt.tz_convert(KST)
    rs.append(r.set_index(["group", "day"]))
r = pd.concat(rs, axis=1)
r["rd"] = r.mean(axis=1)
bf = pd.read_csv(BASE); bf["ts"] = pd.to_datetime(bf.ts).dt.tz_convert(KST)
bf["day"] = (bf.ts - pd.Timedelta(hours=1)).dt.normalize()
bf = bf.merge(r[["rd"]].reset_index(), on=["group", "day"], how="left")
assert len(bf) == 17520 and bf.rd.notna().all()
bf["out"] = (bf.forecast_kwh * (1 + ALPHA * (bf.rd - 1))).clip(lower=0)
OUT.mkdir(parents=True, exist_ok=False)
sites = pd.read_csv(D / "sites.csv")
sites.assign(group=sites.current_group.map({"A": "G1", "B": "G2"}))[["site_id", "group"]].sort_values("site_id").to_csv(OUT / "composition.csv", index=False, lineterminator="\n")
fc = bf[["group", "ts", "out"]].rename(columns={"out": "forecast_kwh"}).sort_values(["group", "ts"])
fc["ts"] = fc["ts"].dt.strftime("%Y-%m-%d %H:%M:%S%z").str.replace(r"(\+\d{2})(\d{2})$", r"\1:\2", regex=True)
fc.to_csv(OUT / "forecast.csv", index=False, lineterminator="\n", float_format="%.6f")
man = {"candidate_id": OUT.name, "method": "base x (1 + alpha*(mean group-day ratio - 1)); ratios from site-day models (Gala 2016 day-total structure)",
       "alpha": ALPHA, "ratio_runs": [str(p) for p in DIRS], "external_data": False,
       "inputs_sha256": {"base_forecast.csv": sha(BASE), **{f"day_ratio_{i}": sha(p / "day_ratio.parquet") for i, p in enumerate(DIRS)}},
       "outputs_sha256": {"composition.csv": sha(OUT / "composition.csv"), "forecast.csv": sha(OUT / "forecast.csv")},
       "totals_gwh": {"base": round(bf.forecast_kwh.sum() / 1e6, 3), "out": round(bf.out.sum() / 1e6, 3)},
       "rd_stats": r.rd.describe().round(4).to_dict(), "rd_run_corr": round(float(r.iloc[:, 0].corr(r.iloc[:, 1])), 3) if len(DIRS) > 1 else None}
json.dump(man, open(OUT / "MANIFEST.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(json.dumps(man, indent=1, ensure_ascii=False))
