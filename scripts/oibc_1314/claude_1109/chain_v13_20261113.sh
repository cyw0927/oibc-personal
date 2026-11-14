# v13 chain: GFS hourly (v10) + ECMWF hourly SSRD (GCP mirror) in E/B21/H1; day models unchanged (_rich); v10 weights fixed (0.35/0.35/0.30, a=0.5)
cd "$(dirname "$0")"
D='C:/Users/dkrak/AppData/Local/Temp/claude/C--Users-dkrak-Downloads-----------/b8f675a9-3275-4093-bed5-50b68c2cf9a4/scratchpad/oibc_data'
P=C:/Users/dkrak/Documents/Codex/OIBC_1314_review/personal_safe; G=$P/gfs; O=$P/v13_20261113; mkdir -p $O
COMP="C:/Users/dkrak/Downloads/인천공항 작업 파일/_ai_analysis/OIBC/submit_v7_20261111/composition.csv"
F12=folds=2024-07,2024-10,2025-01,2025-02,2025-04,2024-06,2024-08,2024-09,2024-11,2024-12,2025-03,2025-05
log(){ echo "[$(date +%H:%M)] $*" | tee -a $O/summary.txt; }
log "wait ecmwf_hourly"; until grep -q " days" $G/ecmwf_hourly.log 2>/dev/null; do sleep 30; done; log "ecmwf: $(grep -c FAIL $G/ecmwf_hourly.log) fails"
python - <<PY
import pandas as pd
g = pd.read_parquet("$G/gfs_rich.parquet"); g["ts"] = pd.to_datetime(g.ts).dt.tz_convert("Asia/Seoul")
e = pd.read_parquet("$G/ecmwf_hourly.parquet"); e["ts"] = pd.to_datetime(e.ts).dt.tz_convert("Asia/Seoul")
m = g.merge(e, on="ts", how="left"); m.to_parquet("$G/gfs_rx.parquet", index=False)
t = m[m.ts >= "2025-06-01"]; print("gfs_rx", m.shape, "ex coverage all", round(m.ex_ssrd.notna().mean(), 3), "test", round(t.ex_ssrd.notna().mean(), 3))
PY
(python -u cv_site.py $D $P/site_feat2021/site_train.parquet $O/cvB21_rx.parquet B $F12 echr=$G/gfs_rx.parquet savesite=$O/B21_rx.parquet > $O/B21_rx.log 2>&1) &
(python -u cv_site.py $D $P/site_feat/site_train.parquet $O/cvE_rx.parquet E $F12 echr=$G/gfs_rx.parquet savesite=$O/E_rx.parquet > $O/E_rx.log 2>&1) &
(python -u cv_group.py $D $P/feat_full_comp $P/cv/group_cv_comp_rx.parquet echr=$G/gfs_rx.parquet > $O/H1_rx.log 2>&1) &
wait; log "CV done"
python - <<'PY'
import sys, pandas as pd
P = "C:/Users/dkrak/Documents/Codex/OIBC_1314_review/personal_safe/"; O = P + "v13_20261113/"
m7 = ["2024-06", "2024-08", "2024-09", "2024-11", "2024-12", "2025-03", "2025-05"]
x = pd.read_parquet(O + "E_rx.parquet"); x["fold"] = x.fold.astype(str)
x[~x.fold.isin(m7)].to_parquet(P + "cv/site_level_E_5f_rx.parquet", index=False); x[x.fold.isin(m7)].to_parquet(P + "cv/site_level_E_rx_7f.parquet", index=False)
pd.read_parquet(O + "B21_rx.parquet").to_parquet(P + "cv/site_level_B21_12m_rx.parquet", index=False)
import shutil
for n in ("day23", "day21"):
    shutil.copy(P + f"cv/site_level_{n}_rich.parquet", P + f"cv/site_level_{n}_rx.parquet"); shutil.copy(P + f"cv/site_level_{n}_rich_7f.parquet", P + f"cv/site_level_{n}_rx_7f.parquet")
PY
OIBC_HSUF=_rx python eval_12m.py $D $P "$COMP" $O/eval12m_rx.json _rx _rx > /dev/null 2>&1
python - <<PY | tee -a $O/summary.txt
import json
a = json.load(open("$P/eval12m_v10.json")); b = json.load(open("$O/eval12m_rx.json"))
v10 = a["v7_by_month"]; c = b["v7_by_month"]   # both at fixed v10 weights 0.35/0.35/0.30, alpha 0.5
sel = {"2024-07", "2024-10", "2025-01", "2025-02", "2025-04"}
print("v13 (v10 weights) 12m", round(sum(c), 3), "vs v10", round(sum(v10), 3), f"{100*(sum(c)/sum(v10)-1):+.2f}%", "better", sum(x < y for x, y in zip(c, v10)), "/ 12")
for g in ("5", "7"):
    ms = [i for i, m in enumerate(a["months"]) if (m in sel) == (g == "5")]
    s0 = sum(v10[i] for i in ms); s1 = sum(c[i] for i in ms); print(g, "months", round(s0, 3), "->", round(s1, 3), f"{100*(s1/s0-1):+.2f}%", "better", sum(c[i] < v10[i] for i in ms), "/", len(ms))
for m, x, y in zip(a["months"], v10, c): print(m, round(x, 3), round(y, 3), f"{100*(y/x-1):+.1f}%")
PY
log "done"
