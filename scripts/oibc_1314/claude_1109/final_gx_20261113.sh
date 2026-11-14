# v10 candidate: site models E/B21 + hourly GFS box (echr), day models + GFS rich box (ecbox=box_rich)
S=/c/Users/dkrak/AppData/Local/Temp/claude/C--Users-dkrak-Downloads-----------/b8f675a9-3275-4093-bed5-50b68c2cf9a4/scratchpad
P=C:/Users/dkrak/Documents/Codex/OIBC_1314_review/personal_safe; G=$P/gfs; F=$P/final_site; FD=$P/final_day
(python -u final_site.py $S/oibc_data $P/site_feat E $F/E_gx 0,1 echr=$G/gfs_rich.parquet > $F/E_gx.log 2>&1) &
(python -u final_site.py $S/oibc_data $P/site_feat2021 B $F/B21_gx 0,1 echr=$G/gfs_rich.parquet > $F/B21_gx.log 2>&1) &
(python -u final_day.py $S/oibc_data_comp $P/site_feat $F/B_sites.parquet $F/v4b_comp_forecast.csv $FD/comp_day23_rich 1.0 0,1 ecbox=$G/box_rich.parquet > $FD/comp_day23_rich.log 2>&1) &
(python -u final_day.py $S/oibc_data_comp $P/site_feat2021 $F/B21_sites.parquet $F/v4b_comp_forecast.csv $FD/comp_day21_rich 1.0 0,1 ecbox=$G/box_rich.parquet > $FD/comp_day21_rich.log 2>&1) &
wait; echo ALL_DONE
