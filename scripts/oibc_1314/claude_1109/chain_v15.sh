# v15 emulation: train with ECMWF 12Z D-2, predict with ECMWF 00Z D-1 (day model ec_box*, physical model) on 12-month OOF; compare with v14
cd "$(dirname "$0")"
D='C:/Users/dkrak/AppData/Local/Temp/claude/C--Users-dkrak-Downloads-----------/b8f675a9-3275-4093-bed5-50b68c2cf9a4/scratchpad/oibc_data'
P=C:/Users/dkrak/Documents/Codex/OIBC_1314_review/personal_safe; G=$P/gfs; O=$P/v15_20261113; mkdir -p $O
COMP="C:/Users/dkrak/Downloads/인천공항 작업 파일/_ai_analysis/OIBC/submit_v7_20261111/composition.csv"
M12="2024-07 2024-10 2025-01 2025-02 2025-04 2024-06 2024-08 2024-09 2024-11 2024-12 2025-03 2025-05"
log(){ echo "[$(date +%H:%M)] $*" | tee -a $O/summary.txt; }
log "wait ecmwf_00z"; until grep -q " days" $G/ecmwf_00z.log 2>/dev/null; do sleep 30; done; log "00z: $(grep -c FAIL $G/ecmwf_00z.log) fails"
python - <<PY
import pandas as pd
e = pd.read_parquet("$G/ecmwf_00z.parquet"); e["ts"] = pd.to_datetime(e.ts).dt.tz_convert("Asia/Seoul")
e["day"] = (e.ts - pd.Timedelta(hours=1)).dt.tz_localize(None).dt.normalize()
d = e.groupby("day")[["e0_box", "e0_boxN", "e0_boxS", "lm_utc"]].first().reset_index()
d[["day", "e0_box", "e0_boxN", "e0_boxS"]].rename(columns={"e0_box": "ec_box", "e0_boxN": "ec_boxN", "e0_boxS": "ec_boxS"}).to_parquet("$G/e0box_pred.parquet", index=False)
d["lm"] = pd.to_datetime(d.lm_utc, format="%a, %d %b %Y %H:%M:%S GMT", utc=True).dt.tz_convert("Asia/Seoul")
d["ok17"] = d.lm < (d.day.dt.tz_localize("Asia/Seoul") - pd.Timedelta(days=1) + pd.Timedelta(hours=17))
d.to_parquet("$G/ecmwf_00z_daily.parquet", index=False)
t = d[d.day >= "2025-06-01"]; print("test days", len(t), "published before D-1 17:00", int(t.ok17.sum()), "first ok", t[t.ok17].day.min(), "last not ok", t[~t.ok17].day.max())
b = pd.read_parquet("$G/box_rich.parquet"); b["day"] = pd.to_datetime(b.day); m = b.merge(d[["day", "e0_box"]], on="day")
print("corr 12Z vs 00Z box", round(m.ec_box.corr(m.e0_box), 4))
PY
(python -u cv_day3.py $D $P/site_feat/site_train.parquet $O/cv23.parquet $M12 ecbox=$G/box_rich.parquet ecboxpred=$G/e0box_pred.parquet savesite=$O/d23.parquet > $O/d23.log 2>&1) &
(python -u cv_day3.py $D $P/site_feat2021/site_train.parquet $O/cv21.parquet $M12 ecbox=$G/box_rich.parquet ecboxpred=$G/e0box_pred.parquet savesite=$O/d21.parquet > $O/d21.log 2>&1) &
wait; log "day CV done"
python - <<'PY'
import pandas as pd
P = "C:/Users/dkrak/Documents/Codex/OIBC_1314_review/personal_safe/"; O = P + "v15_20261113/"
m7 = ["2024-06", "2024-08", "2024-09", "2024-11", "2024-12", "2025-03", "2025-05"]
for n in ("23", "21"):
    x = pd.read_parquet(O + f"d{n}.parquet"); x["fold"] = x.fold.astype(str)
    x[~x.fold.isin(m7)].to_parquet(P + f"cv/site_level_day{n}_r00.parquet", index=False); x[x.fold.isin(m7)].to_parquet(P + f"cv/site_level_day{n}_r00_7f.parquet", index=False)
PY
python v15_eval.py $P "$COMP" | tee -a $O/summary.txt
log done
