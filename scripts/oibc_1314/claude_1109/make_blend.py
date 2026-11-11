"""Blend evaluation-year group forecasts into a submission candidate.

Usage: make_blend.py <dataset> <v3_dir> <fs2l1_prediction_test.csv> <out_dir> <name=weight> [<name=weight> ...] [site=<prefix>:<name>]
Components:
  H1      group-direct H1 (fit to 2025-05), recovered exactly from v3: (v3 - 0.4*FS2L1) / 0.6
  FS2L1   team LightGBM FS2+L1 3-seed site sum (fit to 2025-02)
  others  given as site=<parquet prefix>:<name> -> <prefix>_group.parquet (group, ts, kwh)
Weights must sum to 1. VPP = #990 reference.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

D, V3, BASE, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
args = sys.argv[5:]
W = {a.split("=")[0]: float(a.split("=")[1]) for a in args if not a.startswith("site=")}
EXTRA = {a[5:].rsplit(":", 1)[1]: a[5:].rsplit(":", 1)[0] for a in args if a.startswith("site=")}
assert abs(sum(W.values()) - 1) < 1e-9, W
KST = "Asia/Seoul"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


sites = pd.read_csv(D / "sites.csv")
sites["group"] = sites["current_group"].map({"A": "G1", "B": "G2"})
smp = pd.read_csv(D / "smp.csv")
keys = pd.MultiIndex.from_product([["G1", "G2"], pd.to_datetime(smp.ts).dt.tz_convert(KST)], names=["group", "ts"]).to_frame(index=False)

b = pd.read_csv(BASE)
b["ts"] = pd.to_datetime(b.ts).dt.tz_convert(KST)
b = b.merge(sites[["site_id", "group"]], on="site_id").groupby(["group", "ts"]).prediction_kwh.sum().rename("FS2L1").reset_index()
v3 = pd.read_csv(V3 / "forecast.csv")
v3["ts"] = pd.to_datetime(v3.ts).dt.tz_convert(KST)
t = keys.merge(b, on=["group", "ts"]).merge(v3, on=["group", "ts"])
t["H1"] = ((t.forecast_kwh - 0.4 * t.FS2L1) / 0.6).clip(lower=0)
for name, prefix in EXTRA.items():
    g = pd.read_parquet(prefix + "_group.parquet")
    g["ts"] = pd.to_datetime(g.ts).dt.tz_convert(KST)
    t = t.merge(g.rename(columns={"kwh": name}), on=["group", "ts"], how="left")
assert len(t) == 17520 and t[list(W)].notna().all().all()
OUT.mkdir(parents=True, exist_ok=False)
t["out"] = sum(w * t[n] for n, w in W.items()).clip(lower=0)

sites[["site_id", "group"]].sort_values("site_id").to_csv(OUT / "composition.csv", index=False, lineterminator="\n")
fc = t[["group", "ts", "out"]].rename(columns={"out": "forecast_kwh"}).sort_values(["group", "ts"])
fc["ts"] = fc["ts"].dt.strftime("%Y-%m-%d %H:%M:%S%z").str.replace(r"(\+\d{2})(\d{2})$", r"\1:\2", regex=True)
fc.to_csv(OUT / "forecast.csv", index=False, lineterminator="\n", float_format="%.6f")
man = {"candidate_id": OUT.name, "weights": W, "components": {"H1": "group-direct H1 fit to 2025-05 (from v3)", "FS2L1": "team LightGBM FS2+L1 (fit to 2025-02)",
       **{n: p for n, p in EXTRA.items()}}, "official_S": None, "external_data": False,
       "inputs_sha256": {"v3_forecast.csv": sha(V3 / "forecast.csv"), "fs2l1": sha(BASE), **{n: sha(Path(p + "_group.parquet")) for n, p in EXTRA.items()}},
       "outputs_sha256": {"composition.csv": sha(OUT / "composition.csv"), "forecast.csv": sha(OUT / "forecast.csv")},
       "totals_gwh": {c: round(float(t[c].sum()) / 1e6, 3) for c in list(W) + ["out"]}}
json.dump(man, open(OUT / "MANIFEST.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(json.dumps(man, indent=1, ensure_ascii=False))
