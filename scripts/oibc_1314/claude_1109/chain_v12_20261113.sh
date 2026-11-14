# v12 chain: wait for GFS lagged runs (18Z/12Z D-2), extra vars, quadrant boxes -> gfs_all hourly + box_all daily -> 12-month CV -> LOMO retune
cd "$(dirname "$0")"
D='C:/Users/dkrak/AppData/Local/Temp/claude/C--Users-dkrak-Downloads-----------/b8f675a9-3275-4093-bed5-50b68c2cf9a4/scratchpad/oibc_data'
P=C:/Users/dkrak/Documents/Codex/OIBC_1314_review/personal_safe; G=$P/gfs; O=$P/v12_20261113; mkdir -p $O
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
log "wait downloads"
for f in gfs_rich_r18 gfs_rich_r12 gfs_extra gfs_quad; do until grep -q " days" $G/$f.log 2>/dev/null; do sleep 60; done; log "$f: $(grep -c FAIL $G/$f.log) fails"; done
python - <<PY
import pandas as pd, numpy as np
G = "$G"
a = pd.read_parquet(f"{G}/gfs_rich.parquet")
for f in ("gfs_quad", "gfs_extra"):
    x = pd.read_parquet(f"{G}/{f}.parquet"); a = a.merge(x, on="ts", how="left")
r18 = pd.read_parquet(f"{G}/gfs_rich_r18.parquet").set_index("ts"); r12 = pd.read_parquet(f"{G}/gfs_rich_r12.parquet").set_index("ts")
a = a.set_index("ts")
for v in ("dsw", "tcc", "lcc", "mcc", "hcc", "prate", "t2", "rh2"):
    c = f"gx_{v}"; st = pd.concat([a[c], r18[c].reindex(a.index), r12[c].reindex(a.index)], axis=1)
    a[c + "_lm"] = st.mean(1); a[c + "_ls"] = st.std(1)
a = a.reset_index(); a.to_parquet(f"{G}/gfs_all.parquet", index=False); print("gfs_all", a.shape, "cols", a.shape[1] - 1)
a["day"] = (a.ts - pd.Timedelta(hours=1)).dt.tz_localize(None).dt.normalize(); h = a.ts.dt.hour; cols = [c for c in a.columns if c.startswith("gx_")]
pk = a[h.between(10, 15)].groupby("day")[cols].mean().add_suffix("_pk"); dy = a[h.between(7, 18)].groupby("day")[cols].mean().add_suffix("_dy")
b = pd.read_parquet(f"{G}/box_ecgf.parquet"); b["day"] = pd.to_datetime(b.day)
b.merge(pk.join(dy).reset_index(), on="day", how="left").to_parquet(f"{G}/box_all.parquet", index=False); print("box_all", b.shape)
PY
log "features built"
(python -u cv_day3.py $D $P/site_feat/site_train.parquet $O/cv23_all.parquet $M12 ecbox=$G/box_all.parquet savesite=$O/d23_all.parquet > $O/d23_all.log 2>&1) &
(python -u cv_day3.py $D $P/site_feat2021/site_train.parquet $O/cv21_all.parquet $M12 ecbox=$G/box_all.parquet savesite=$O/d21_all.parquet > $O/d21_all.log 2>&1) &
(python -u cv_site.py $D $P/site_feat2021/site_train.parquet $O/cvB21_all.parquet B $F12 echr=$G/gfs_all.parquet savesite=$O/B21_all.parquet > $O/B21_all.log 2>&1) &
(python -u cv_site.py $D $P/site_feat/site_train.parquet $O/cvE_all.parquet E $F12 echr=$G/gfs_all.parquet savesite=$O/E_all.parquet > $O/E_all.log 2>&1) &
(python -u cv_group.py $D $P/feat_full_comp $P/cv/group_cv_comp_all.parquet echr=$G/gfs_all.parquet > $O/H1_all.log 2>&1) &
wait; log "CV done"
split7 $O/d23_all.parquet $P/cv/site_level_day23_all; split7 $O/d21_all.parquet $P/cv/site_level_day21_all
split7 $O/E_all.parquet $P/cv/site_level_E_5f_all; mv $P/cv/site_level_E_5f_all_7f.parquet $P/cv/site_level_E_all_7f.parquet
cp $O/B21_all.parquet $P/cv/site_level_B21_12m_all.parquet
tail -13 $O/H1_all.log | tee -a $O/summary.txt
OIBC_HSUF=_all python eval_12m.py $D $P "$COMP" $O/eval12m_all.json _all _all > /dev/null 2>&1
python - <<PY | tee -a $O/summary.txt
import json
a = json.load(open("$P/eval12m_v11h.json")); b = json.load(open("$O/eval12m_all.json"))
v11 = a["lomo_by_month"]; c = b["lomo_by_month"]
print("v12(all) LOMO", round(sum(c), 3), "vs v11 LOMO", round(sum(v11), 3), f"{100*(sum(c)/sum(v11)-1):+.2f}%", "better", sum(x < y for x, y in zip(c, v11)), "/ 12", "insample best", b["insample_best"])
for m, x, y in zip(a["months"], v11, c): print(m, round(x, 3), round(y, 3))
PY
log "done"
