"""One-shot HOLDOUT test (pre-registered: docs/holdout_final_test_protocol_20261109.md). H0 = v0.2 features, H1 = + day context.

Usage: holdout_test.py <dataset> <feat_full_dir> <out_json>
Fit 2023-06-01 01:00 ~ 2025-02-28 00:00, evaluate 2025-03-01 01:00 ~ 2025-06-01 00:00 (README provisional group penalty).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from ctx_features import add_ctx

D, FEAT, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
KST = "Asia/Seoul"
FIT_START = pd.Timestamp("2023-06-01 01:00", tz=KST)
FIT_END = pd.Timestamp("2025-02-28 00:00", tz=KST)
HO_START = pd.Timestamp("2025-03-01 01:00", tz=KST)
HO_END = pd.Timestamp("2025-06-01 00:00", tz=KST)
CONFIGS = [dict(learning_rate=0.05, max_iter=600, max_leaf_nodes=31, min_samples_leaf=40),
           dict(learning_rate=0.03, max_iter=900, max_leaf_nodes=15, min_samples_leaf=80),
           dict(learning_rate=0.05, max_iter=500, max_leaf_nodes=63, min_samples_leaf=30)]
NON = {"ts", "group", "energy", "cap_avail", "n_sites", "group_cap_kw", "cf", "smp", "k", "cov"}

smp = pd.read_csv(D / "train" / "smp.csv")
smp["ts"] = pd.to_datetime(smp["ts"]).dt.tz_convert(KST).dt.as_unit("us")
smp["k"] = np.where(smp.ts.dt.hour.between(11, 15), 3, 1)
ft = pd.read_parquet(FEAT / "group_features_train.parquet")
ft["ts"] = ft.ts.dt.as_unit("us")
base = [c for c in ft.columns if c not in NON]
ft, new = add_ctx(ft)
ft = ft.merge(smp, on="ts", how="left")
ft["cov"] = ft.cap_avail / ft.group_cap_kw
ft["cf"] = (ft.energy / ft.cap_avail).clip(0, 1.2)
tr = ft[(ft.ts >= FIT_START) & (ft.ts <= FIT_END) & ft.energy.notna() & (ft["cov"] >= 0.3) & (ft.cosz_max > 0)]
ho = ft[(ft.ts >= HO_START) & (ft.ts <= HO_END)].copy()
assert len(ho) == 2 * 2208 and ho.energy.notna().all() and ho.smp.notna().all(), (len(ho), ho.energy.isna().sum())
sw = tr["cov"].values * tr.k.values * np.clip(tr.smp.values, 1, None) / tr.smp.clip(lower=1).mean()


def predict(feats):
    cf = np.mean([np.clip(HistGradientBoostingRegressor(loss="absolute_error", l2_regularization=1.0,
                                                        early_stopping=False, random_state=0, **c)
                          .fit(tr[feats], tr.cf, sample_weight=sw).predict(ho[feats]), 0, 1) for c in CONFIGS], axis=0)
    return np.where(ho.cosz_max.values > 0, cf, 0.0) * ho.cap_avail.values


ho["H0"] = predict(base)
ho["H1"] = predict(base + new)
ho["m"] = ho.ts.dt.strftime("%Y-%m")
w = 0.1 * ho.smp * ho.k
res = {"protocol": "docs/holdout_final_test_protocol_20261109.md", "n_rows": int(len(ho)), "cov_mean": float(ho["cov"].mean()),
       "n_base": len(base), "n_new": len(new)}
for h in ("H0", "H1"):
    pen = (w * (ho[h] - ho.energy).abs())
    res[h] = {"total_won": float(pen.sum()), "by_month": {m: float(v) for m, v in pen.groupby(ho.m).sum().items()},
              "bias_pct_smpk": float(100 * (ho.smp * ho.k * (ho[h] - ho.energy)).sum() / (ho.smp * ho.k * ho.energy).sum()),
              "group_wmae": float((ho.smp * ho.k * (ho[h] - ho.energy).abs()).sum() / (ho.smp * ho.k * ho.energy).sum())}
months_better = sum(res["H1"]["by_month"][m] < res["H0"]["by_month"][m] for m in res["H0"]["by_month"])
res["H1_months_better"] = months_better
res["decision"] = "H1" if (res["H1"]["total_won"] < res["H0"]["total_won"] and months_better >= 2) else "H0"
json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)
ho[["group", "ts", "energy", "smp", "k", "H0", "H1"]].to_parquet(str(OUT).replace(".json", "_preds.parquet"), index=False)
print(json.dumps(res, indent=1))
