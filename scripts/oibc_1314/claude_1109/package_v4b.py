"""Package v4b reproduction bundle: fold OOF (E/B21/H1), v4b rolling eval, final component predictions, code, manifest/logs."""
import hashlib, json, shutil, sys
from pathlib import Path
import pandas as pd

PS, CAND, PARAMS, CODE, OUT = (Path(a) for a in sys.argv[1:6])
FOLDS4 = ["2024-07", "2024-10", "2025-01", "2025-04"]
K = ["group", "ts"]
KST = "Asia/Seoul"
OUT.mkdir(parents=True, exist_ok=False)
cv = PS / "cv"


def L(f, src, name):
    x = pd.read_parquet(cv / f)
    x["ts"] = pd.to_datetime(x.ts).dt.tz_convert(KST)
    return x[["fold", "group", "ts", "energy", "smp", "k", src]].rename(columns={src: name})


def ts_str(s):
    return s.dt.strftime("%Y-%m-%d %H:%M:%S%z").str.replace(r"(\+\d{2})(\d{2})$", r"\1:\2", regex=True)


o = OUT / "1_fold_oof"; o.mkdir()
comp = {"E": pd.concat([L("site_cv_EG.parquet", "E", "pred"), L("site_cv_E_feb.parquet", "E", "pred")]),
        "B21": pd.concat([L("site_cv_B2021.parquet", "B", "pred"), L("site_cv_B2021_feb.parquet", "B", "pred")]),
        "H1": L("group_cv.parquet", "H1", "pred")}
for n, x in comp.items():
    x = x.sort_values(["fold", "group", "ts"])
    for tag, folds in (("4fold", FOLDS4), ("2025-02_extra", ["2025-02"])):
        y = x[x.fold.isin(folds)].copy(); y["ts"] = ts_str(y.ts)
        y.to_csv(o / f"{n}_oof_{tag}.csv", index=False, float_format="%.6f", lineterminator="\n")

r = OUT / "2_v4b_rolling"; r.mkdir()
m = comp["E"].rename(columns={"pred": "E"}).merge(comp["B21"][["fold"] + K + ["pred"]].rename(columns={"pred": "B21"}), on=["fold"] + K) \
    .merge(comp["H1"][["fold"] + K + ["pred"]].rename(columns={"pred": "H1"}), on=["fold"] + K)
m["v4b"] = 0.35 * m.E + 0.35 * m.B21 + 0.30 * m.H1
w = 0.1 * m.smp * m.k
for c in ("E", "B21", "H1", "v4b"):
    m[f"pen_{c}"] = w * (m[c] - m.energy).abs()
ev = m.groupby("fold").agg(rows=("v4b", "size"), **{f"{c}_M": (f"pen_{c}", "sum") for c in ("E", "B21", "H1", "v4b")})
for c in ("E", "B21", "H1", "v4b"):
    ev[f"{c}_M"] /= 1e6
ev.loc["TOTAL_4fold"] = ev.loc[FOLDS4].sum()
ev.loc["TOTAL_5fold_incl_2025-02"] = ev.loc[FOLDS4 + ["2025-02"]].sum()
ev.round(6).to_csv(r / "v4b_eval_by_fold.csv", lineterminator="\n")
mm = m.sort_values(["fold", "group", "ts"]).copy(); mm["ts"] = ts_str(mm.ts)
mm[mm.fold.isin(FOLDS4)].drop(columns=[c for c in mm if c.startswith("pen_")]).to_csv(r / "v4b_oof_4fold.csv", index=False, float_format="%.6f", lineterminator="\n")
mm[mm.fold == "2025-02"].drop(columns=[c for c in mm if c.startswith("pen_")]).to_csv(r / "v4b_oof_2025-02_extra.csv", index=False, float_format="%.6f", lineterminator="\n")
print(ev.round(3).to_string())

f = OUT / "3_final_predictions"; f.mkdir()
fc = pd.read_csv(CAND / "forecast.csv"); fc["ts"] = pd.to_datetime(fc.ts).dt.tz_convert(KST)
v3 = pd.read_csv(Path(sys.argv[6]) / "forecast.csv"); v3["ts"] = pd.to_datetime(v3.ts).dt.tz_convert(KST)
for n in ("E", "B21"):
    g = pd.read_parquet(PS / "final_site" / f"{n}_group.parquet"); g["ts"] = pd.to_datetime(g.ts).dt.tz_convert(KST)
    fc = fc.merge(g.rename(columns={"kwh": n}), on=K)
    shutil.copy2(PS / "final_site" / f"{n}_sites.parquet", f / f"{n}_sites_final.parquet")
fc["H1"] = (fc.forecast_kwh - 0.35 * fc.E - 0.35 * fc.B21) / 0.30
chk = (0.35 * fc.E + 0.35 * fc.B21 + 0.30 * fc.H1 - fc.forecast_kwh).abs().max()
fc2 = fc.copy(); fc2["ts"] = ts_str(fc2.ts)
fc2[K + ["E", "B21", "H1", "forecast_kwh"]].rename(columns={"forecast_kwh": "v4b"}).to_csv(f / "components_final_group.csv", index=False, float_format="%.6f", lineterminator="\n")
for n in ("composition.csv", "forecast.csv"):
    shutil.copy2(CAND / n, f / f"v4b_{n}")
shutil.copy2(Path(sys.argv[6]) / "forecast.csv", f / "v3_forecast_for_H1_recovery.csv")
print("H1 recon max abs err", chk, "H1 min", round(fc.H1.min(), 6))

c = OUT / "4_code"; c.mkdir()
for n in ["ctx_features.py", "prep_group_features_full.py", "site_features.py", "cv_site.py", "cv_group.py", "final_site.py",
          "final_predict.py", "make_blend.py", "validate.sh", "evidence_v4b.py", "package_v4b.py"]:
    shutil.copy2(CODE / n, c / n)

e = OUT / "5_manifest_logs"; e.mkdir()
for n in ("MANIFEST.json", "validator_report.json", "official_validator.log"):
    shutil.copy2(CAND / n, e / n)
shutil.copytree(PARAMS / "evidence", e / "evidence")
for n in ("v4b_parameters.json", "v4b_파라미터_요약.md"):
    shutil.copy2(PARAMS / n, e / n)

sha = {str(p.relative_to(OUT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.rglob("*")) if p.is_file()}
json.dump(sha, open(OUT / "SHA256SUMS.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(len(sha), "files")
