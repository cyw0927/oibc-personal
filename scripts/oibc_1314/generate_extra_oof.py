"""Generate a fixed historical FS2+L1 OOF using safe prelock data only."""
import argparse,gc,json,platform,time
from pathlib import Path
import numpy as np
import pandas as pd
from fs2l1_safe_control import ROOT,DATA,CONTRACTS,LOCK,lightgbm,safe_metadata,sha,load_contracts,OfficialData,training_rows,validation_grid,build_nonlinear

MONTHS={'MAY_2024':('2024-05-01 01:00','2024-06-01 00:00','2024-04-30 00:00'),
        'AUG_2024':('2024-08-01 01:00','2024-09-01 00:00','2024-07-31 00:00')}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('month',choices=MONTHS);a=ap.parse_args()
 control=ROOT/'personal_safe'/'fs2l1_control_20261106'/'results.json'
 if json.loads(control.read_text(encoding='utf8'))['max_abs_diff_kwh']!=0:raise ValueError('original JUL reproduction mismatch')
 inputs=[DATA/'train'/'generation.parquet',DATA/'train'/'forecast_wa.parquet',DATA/'train'/'forecast_wb.parquet']
 for p in inputs:safe_metadata(p)
 start,end,cut=[pd.Timestamp(x,tz='Asia/Seoul') for x in MONTHS[a.month]]
 if not cut<start<=end<LOCK:raise ValueError('bad cutoff')
 c=load_contracts(CONTRACTS);data=OfficialData(DATA,c);ix=pd.date_range(start,end,freq='h')
 gen=data.generation();v=gen[gen.ts.between(start,end)]
 if v.duplicated(['site_id','ts']).any():raise ValueError('duplicate generation')
 counts=v.groupby('site_id').ts.nunique();ids=tuple(sorted(counts[counts.eq(len(ix))].index))
 train=training_rows(data,pd.Timestamp('2023-06-01 01:00',tz='Asia/Seoul'),cut);grid=validation_grid(data,ids,ix)
 if not ids or len(grid)!=len(ids)*len(ix) or train.ts.max()>cut:raise ValueError('bad validation grid')
 print(a.month,'sites',len(ids),'training',len(train),'validation',len(grid),flush=True)
 pred=[]
 for seed in (11,22,33):
  t=time.time();m=build_nonlinear('lgbm_cf',{'seed':seed,'n_jobs':2,'feature_set':'FS2','loss':'L1'},data);m.fit(train);y=m.predict(grid)['prediction_kwh'].to_numpy(float)
  if len(y)!=len(grid) or not np.isfinite(y).all() or (y<0).any():raise ValueError('invalid prediction')
  pred.append(y);del m;gc.collect();print(a.month,'seed',seed,'seconds',round(time.time()-t,1),flush=True)
 grid=grid.copy();grid['base']=np.mean(pred,axis=0);grid['group']=grid.current_group.map({'A':'G1','B':'G2'})
 if grid.group.isna().any():raise ValueError('unknown group')
 out=ROOT/'personal_safe'/'expanded_oof_20261106'/a.month.lower();out.mkdir(parents=True,exist_ok=True)
 grid[['site_id','ts','group','energy','capacity_kw','base']].to_parquet(out/'site_oof.parquet',index=False)
 g=grid.groupby(['group','ts'],as_index=False).agg(actual=('energy','sum'),base=('base','sum'),cap=('capacity_kw','sum'),site_count=('site_id','nunique'))
 smp=pd.read_csv(ROOT/'personal_safe'/'winter_20261105'/'smp_to_20250228.csv',parse_dates=['ts']);smp.ts=pd.to_datetime(smp.ts,utc=True).dt.tz_convert('Asia/Seoul').dt.as_unit('us')
 if smp.ts.max()>=LOCK or smp.ts.duplicated().any():raise ValueError('unsafe SMP')
 smp['k']=np.where(smp.ts.dt.hour.between(11,15),3,1);g=g.merge(smp,on='ts',validate='many_to_one')
 if g.smp.isna().any() or g.duplicated(['group','ts']).any():raise ValueError('invalid group grid')
 g.to_parquet(out/'group_oof.parquet',index=False)
 info={'period':[str(start),str(end)],'cutoff':str(cut),'train_rows':len(train),'cohort_sites':len(ids),'site_hours':len(grid),'group_hours':len(g),'group_capacity_kw':{str(k):float(v) for k,v in g.groupby('group').cap.first().items()},'site_sha256':sha(out/'site_oof.parquet'),'group_sha256':sha(out/'group_oof.parquet'),'input_sha256':{str(x.relative_to(DATA)):sha(x) for x in inputs},'lightgbm':lightgbm.__version__,'python':platform.python_version(),'seeds':[11,22,33],'historical_data_only':True,'excluded_holdout':True,'cohort':'retrospective complete records; not all sites'}
 (out/'manifest.json').write_text(json.dumps(info,indent=2),encoding='utf8');print(json.dumps(info),flush=True)
if __name__=='__main__':main()
