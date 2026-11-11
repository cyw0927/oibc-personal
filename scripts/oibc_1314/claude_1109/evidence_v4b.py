"""Evidence for the v4b review (no new model training): fold windows, same-condition comparison, feature timing, SHAs.

Usage: evidence_v4b.py <dataset> <personal_safe_dir> <candidates_dir> <out_dir>
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

D, PS, CAND, OUT = (Path(a) for a in sys.argv[1:5])
OUT.mkdir(parents=True, exist_ok=True)
KST = "Asia/Seoul"
FOLDS4 = ["2024-07", "2024-10", "2025-01", "2025-04"]
ev = {}


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def window(M: str):
    P = pd.Period(M, freq="M")
    st = pd.Timestamp(P.start_time).tz_localize(KST) + pd.Timedelta(hours=1)
    en = pd.Timestamp(P.end_time.normalize() + pd.Timedelta(days=1)).tz_localize(KST)
    return st, en


# 1) fold windows recomputed from the same filters as cv_site.py / cv_group.py
folds = []
for name, path in (("site_2023start(B,E)", PS / "site_feat/site_train.parquet"), ("site_2021start(B21)", PS / "site_feat2021/site_train.parquet")):
    d = pd.read_parquet(path, columns=["ts", "elev", "energy", "round"])
    d["ts"] = d.ts.dt.tz_convert(KST)
    d = d[d.energy.notna()]
    for M in FOLDS4:
        st, en = window(M)
        tr = d[(d.ts < st) & (d.elev > 0)]
        te = d[(d.ts >= st) & (d.ts <= en)]
        folds.append({"model": name, "fold": M, "train_first_ts": str(tr.ts.min()), "train_last_ts": str(tr.ts.max()),
                      "train_rows": int(len(tr)), "eval_first_ts": str(te.ts.min()), "eval_last_ts": str(te.ts.max()),
                      "train_after_eval_start": int((tr.ts >= st).sum())})
g = pd.read_parquet(PS / "feat_full_20261109/group_features_train.parquet", columns=["ts", "energy", "cap_avail", "group_cap_kw", "cosz_max"])
g["ts"] = g.ts.dt.tz_convert(KST)
for M in FOLDS4:
    st, en = window(M)
    tr = g[(g.ts < st) & g.energy.notna() & (g.cap_avail / g.group_cap_kw >= 0.3) & (g.cosz_max > 0)]
    folds.append({"model": "group_H1", "fold": M, "train_first_ts": str(tr.ts.min()), "train_last_ts": str(tr.ts.max()),
                  "train_rows": int(len(tr)), "eval_first_ts": str(st), "eval_last_ts": str(en), "train_after_eval_start": int((tr.ts >= st).sum())})
F = pd.DataFrame(folds)
F.to_csv(OUT / "fold_windows.csv", index=False)
ev["fold_windows"] = folds

# 2) same-condition comparison on rolling folds
s = pd.read_parquet(PS / "cv/site_cv_ABC.parquet")
e = pd.read_parquet(PS / "cv/site_cv_EG.parquet")
b21 = pd.read_parquet(PS / "cv/site_cv_B2021.parquet").rename(columns={"B": "B21"})
gc = pd.read_parquet(PS / "cv/group_cv.parquet")
for x in (s, e, b21, gc):
    x["ts"] = x.ts.dt.tz_convert("UTC")
m = s.merge(e[["group", "ts", "E"]], on=["group", "ts"]).merge(b21[["group", "ts", "B21"]], on=["group", "ts"]) \
     .merge(gc[["group", "ts", "H0", "H1"]], on=["group", "ts"])
m["v4b"] = 0.35 * m.E + 0.35 * m.B21 + 0.30 * m.H1
oof = pd.read_parquet(PS / "fs2l1_oof/oof_dev.parquet")
sites = pd.read_csv(D / "sites.csv")
oof = oof.merge(sites[["site_id", "current_group"]], on="site_id")
oof["group"] = oof.current_group.map({"A": "G1", "B": "G2"})
oof["ts"] = pd.to_datetime(oof.ts, utc=True)
fo = oof.groupby(["group", "ts"]).agg(FS2L1=("prediction_kwh", "sum"), act_o=("actual_kwh", "sum")).reset_index()
mm = m.merge(fo, on=["group", "ts"], how="inner")
mm = mm[(mm.act_o - mm.energy).abs() < 1.0].copy()          # identical plant cohort and actuals only
mm["v2_5th_submission"] = 0.4 * mm.FS2L1 + 0.6 * mm.H0
mm["v3"] = 0.4 * mm.FS2L1 + 0.6 * mm.H1


def table(x: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    gross = float((x.smp * x.energy).sum())
    for c in cols:
        ae = (x[c] - x.energy).abs()
        pen = 0.1 * x.smp * x.k * ae
        rows.append({"candidate": c, "penalty_total": float(pen.sum()), "penalty_G1": float(pen[x.group == "G1"].sum()),
                     "penalty_G2": float(pen[x.group == "G2"].sum()), "S_README": gross - float(pen.sum()),
                     "MAE_kwh": float(ae.mean()), "wMAE_smpk": float((x.smp * x.k * ae).sum() / (x.smp * x.k * x.energy).sum())})
    return pd.DataFrame(rows)


A = table(mm, ["v2_5th_submission", "v3", "v4b", "E", "B21", "H1", "FS2L1"])
A["hours"] = len(mm) // 2
A["months"] = ",".join(sorted(mm.fold.unique()))
A.to_csv(OUT / "compare_matched_v2_v3_v4b.csv", index=False)
per = mm.groupby("fold").apply(lambda x: table(x, ["v2_5th_submission", "v3", "v4b"]).set_index("candidate").penalty_total).reset_index()
per.to_csv(OUT / "compare_matched_by_fold.csv", index=False)
B = table(m[m.fold.isin(FOLDS4)], ["v4b", "E", "B21", "H1", "H0"])
B.to_csv(OUT / "compare_4fold_v4b_components.csv", index=False)
ev["matched_hours"] = {k: int(v) for k, v in mm.fold.value_counts().sort_index().items()}

# 3) feature timing: shifted / day-aggregate features never leave the target day on daylight rows
chk = {}
for name, path in (("site_test", PS / "site_feat/site_test.parquet"), ("site_train", PS / "site_feat/site_train.parquet")):
    t = pd.read_parquet(path, columns=["ts", "elev"])
    t = t[t.elev > 0].ts.dt.tz_convert(KST).drop_duplicates()
    day = (t - pd.Timedelta(hours=1)).dt.normalize()
    chk[name] = {"daylight_hours_checked": int(len(t)), "hour_min": int(t.dt.hour.min()), "hour_max": int(t.dt.hour.max()),
                 "plus2h_crosses_target_day": int(((t + pd.Timedelta(hours=1)).dt.normalize() != day).sum())}
gf = pd.concat([pd.read_parquet(PS / "feat_full_20261109/group_features_test.parquet", columns=["ts", "cosz_max"]),
                pd.read_parquet(PS / "feat_full_20261109/group_features_train.parquet", columns=["ts", "cosz_max"])])
t = gf[gf.cosz_max > 0].ts.dt.tz_convert(KST).drop_duplicates()
chk["group_all"] = {"daylight_hours_checked": int(len(t)), "plus2h_crosses_target_day":
                    int(((t + pd.Timedelta(hours=1)).dt.normalize() != (t - pd.Timedelta(hours=1)).dt.normalize()).sum())}
ev["feature_timing"] = chk

# 4) SHAs
code = PS.parent.parent / "oibc-personal/scripts/oibc_1314/claude_1109"
ev["code_sha256"] = {p.name: sha(p) for p in sorted(code.glob("*.py")) if p.name in {
    "site_features.py", "final_site.py", "cv_site.py", "cv_group.py", "ctx_features.py", "prep_group_features_full.py",
    "final_predict.py", "make_blend.py", "holdout_test.py", "evidence_v4b.py"}}
ev["inputs_sha256"] = {"site_train(2023)": sha(PS / "site_feat/site_train.parquet"), "site_train(2021)": sha(PS / "site_feat2021/site_train.parquet"),
                       "site_test": sha(PS / "site_feat/site_test.parquet"), "E_group": sha(PS / "final_site/E_group.parquet"),
                       "B21_group": sha(PS / "final_site/B21_group.parquet"), "v3_forecast(H1 source)": sha(CAND / "Y1314_h1_ctx_full_v3/forecast.csv"),
                       "fs2l1_prediction_test": sha(PS / "fs2l1_test_1452/prediction_test.csv"), "sites.csv": sha(D / "sites.csv"), "smp.csv": sha(D / "smp.csv")}
ev["outputs_sha256"] = {"composition.csv": sha(CAND / "Y1314_v4b_siteEB21_H1/composition.csv"), "forecast.csv": sha(CAND / "Y1314_v4b_siteEB21_H1/forecast.csv")}
comp = pd.read_csv(CAND / "Y1314_v4b_siteEB21_H1/composition.csv").merge(sites[["site_id", "current_group", "capacity_kw"]], on="site_id")
ev["composition_check"] = {"rows": int(len(comp)), "same_as_current_group": bool((comp.group == comp.current_group.map({"A": "G1", "B": "G2"})).all()),
                           "capacity_kw": {k: float(v) for k, v in comp.groupby("group").capacity_kw.sum().round(1).items()}}
json.dump(ev, open(OUT / "evidence_v4b.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(F.to_string()); print(A.round(4).to_string()); print(per.round(0).to_string()); print(B.round(4).to_string())
print(json.dumps({k: ev[k] for k in ("matched_hours", "feature_timing", "composition_check")}, indent=1))
