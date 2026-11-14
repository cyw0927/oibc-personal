#!/bin/sh
# Overnight chain (11/12): GFS multi-variable (day + hourly site models) and ECMWF ENS (day model), 5 selection months + 7 untouched months.
# All external inputs are fixed-box summaries (no site coordinates). Results -> $O/summary.txt. Nothing is submitted.
D='C:/Users/dkrak/AppData/Local/Temp/claude/C--Users-dkrak-Downloads-----------/b8f675a9-3275-4093-bed5-50b68c2cf9a4/scratchpad/oibc_data'
P=C:/Users/dkrak/Documents/Codex/OIBC_1314_review/personal_safe; G=$P/gfs; O=$P/overnight_20261112; mkdir -p $O
COMP="C:/Users/dkrak/Downloads/인천공항 작업 파일/_ai_analysis/OIBC/submit_v7_20261111/composition.csv"
M12="2024-07 2024-10 2025-01 2025-02 2025-04 2024-06 2024-08 2024-09 2024-11 2024-12 2025-03 2025-05"; F12=folds=$(echo $M12 | tr ' ' ',')
log(){ echo "[$(date +%H:%M)] $*" | tee -a $O/summary.txt; }
split7(){ python - "$1" "$2" <<'PY'
import sys, pandas as pd
src, stem = sys.argv[1], sys.argv[2]
x = pd.read_parquet(src); x["fold"] = x.fold.astype(str)
m7 = ["2024-06", "2024-08", "2024-09", "2024-11", "2024-12", "2025-03", "2025-05"]
x[~x.fold.isin(m7)].to_parquet(stem + ".parquet", index=False); x[x.fold.isin(m7)].to_parquet(stem + "_7f.parquet", index=False)
PY
}
evalboth(){ log "== $1"; python eval_v7.py $D $P "$COMP" $O/eval5_$1.csv - $2 $3 2>&1 | tail -1 | tee -a $O/summary.txt
  python eval_v7.py $D $P "$COMP" $O/eval5_$1.csv - $2 $3 >/dev/null 2>&1; python -c "import pandas as pd;r=pd.read_csv('$O/eval5_$1.csv');print(r[r.composition=='comp'][['fold','p_v6']].round(3).to_string(index=False))" | tee -a $O/summary.txt
  python eval_holdout7.py $P "$COMP" $2 $3 | tee -a $O/summary.txt; }

log "wait gfs_rich"; until grep -q " days" $G/gfs_rich.log 2>/dev/null; do sleep 60; done; log "gfs_rich: $(grep -c FAIL $G/gfs_rich.log) fails"
python - <<PY
import pandas as pd
g = pd.read_parquet("$G/gfs_rich.parquet"); g["day"] = (g.ts - pd.Timedelta(hours=1)).dt.tz_localize(None).dt.normalize()
h = g.ts.dt.hour; cols = [c for c in g.columns if c.startswith("gx_")]
pk = g[h.between(10, 15)].groupby("day")[cols].mean().add_suffix("_pk"); dy = g[h.between(7, 18)].groupby("day")[cols].mean().add_suffix("_dy")
b = pd.read_parquet("$G/box_ecgf.parquet"); b["day"] = pd.to_datetime(b.day)
b.merge(pk.join(dy).reset_index(), on="day", how="left").to_parquet("$G/box_rich.parquet", index=False)
print("box_rich", b.shape)
PY
cp $G/box_rich.parquet $O/
(python -u cv_day3.py $D $P/site_feat/site_train.parquet $O/cv23_rich.parquet $M12 ecbox=$G/box_rich.parquet savesite=$O/d23_rich.parquet > $O/d23_rich.log 2>&1) &
(python -u cv_day3.py $D $P/site_feat2021/site_train.parquet $O/cv21_rich.parquet $M12 ecbox=$G/box_rich.parquet savesite=$O/d21_rich.parquet > $O/d21_rich.log 2>&1) &
(python -u cv_site.py $D $P/site_feat2021/site_train.parquet $O/cvB21_gx.parquet B $F12 echr=$G/gfs_rich.parquet savesite=$O/B21_gx.parquet > $O/B21_gx.log 2>&1) &
(python -u cv_site.py $D $P/site_feat/site_train.parquet $O/cvE_gx.parquet E $F12 echr=$G/gfs_rich.parquet savesite=$O/E_gx.parquet > $O/E_gx.log 2>&1) &
wait; log "rich day + gx site models done"
split7 $O/d23_rich.parquet $P/cv/site_level_day23_rich; split7 $O/d21_rich.parquet $P/cv/site_level_day21_rich
split7 $O/E_gx.parquet $P/cv/site_level_E_5f_gx; mv $P/cv/site_level_E_5f_gx_7f.parquet $P/cv/site_level_E_gx_7f.parquet
cp $O/B21_gx.parquet $P/cv/site_level_B21_12m_gx.parquet
evalboth day_rich _rich ""
evalboth day_rich_site_gx _rich _gx

log "wait ens"; until grep -q " days" $G/ecmwf_ens_box.log 2>/dev/null; do sleep 60; done; log "ens: $(grep -c FAIL $G/ecmwf_ens_box.log) fails"
python - <<PY
import pandas as pd
b = pd.read_parquet("$G/box_rich.parquet"); e = pd.read_parquet("$G/ecmwf_ens_box.parquet"); e["day"] = pd.to_datetime(e.day)
b.merge(e, on="day", how="left").to_parquet("$G/box_ens.parquet", index=False); print("box_ens ok")
PY
(python -u cv_day3.py $D $P/site_feat/site_train.parquet $O/cv23_ens.parquet $M12 ecbox=$G/box_ens.parquet savesite=$O/d23_ens.parquet > $O/d23_ens.log 2>&1) &
(python -u cv_day3.py $D $P/site_feat2021/site_train.parquet $O/cv21_ens.parquet $M12 ecbox=$G/box_ens.parquet savesite=$O/d21_ens.parquet > $O/d21_ens.log 2>&1) &
wait
split7 $O/d23_ens.parquet $P/cv/site_level_day23_ens; split7 $O/d21_ens.parquet $P/cv/site_level_day21_ens
evalboth day_ens _ens ""
evalboth day_ens_site_gx _ens _gx
log "ALL DONE (reference: v9 = day _egf: 5f 73.311M, 7f 102.948M)"
