"""Standalone Extra Trees group forecasting experiment; no locked inputs.

See docs/extra_trees_group_protocol_20261105.md. Outputs stay outside Git.
"""
import argparse,hashlib,json,platform
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesRegressor
from safe_reproduce import safe_parquet
from residual_stacking_dev import read_real

KST='Asia/Seoul'
START=pd.Timestamp('2023-06-01 01:00',tz=KST)
END=pd.Timestamp('2024-11-01 00:00',tz=KST)
FOLDS={'DEV_JUL_2024':('2024-06-30 00:00',224),'DEV_OCT_2024':('2024-09-30 00:00',252)}
NON={'ts','group','energy','cap_avail','n_sites','group_cap_kw','cf','smp','k','cov'}

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''): h.update(b)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('safe_root',type=Path);p.add_argument('reference_dir',type=Path);p.add_argument('features',type=Path);p.add_argument('smp',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
 feat=safe_parquet(a.features)
 if feat.ts.max()>END or feat.duplicated(['group','ts']).any():raise ValueError('unsafe features')
 feat=feat[feat.ts>=START].copy()
 smp=pd.read_csv(a.smp,parse_dates=['ts']);smp.ts=pd.to_datetime(smp.ts,utc=True).dt.tz_convert(KST)
 if smp.ts.max()>END or smp.ts.duplicated().any() or smp.smp.isna().any() or (smp.smp<0).any():raise ValueError('unsafe/invalid SMP')
 smp['k']=np.where(smp.ts.dt.hour.between(11,15),3,1)
 feat=feat.merge(smp,on='ts',how='left',validate='many_to_one');feat['cov']=feat.cap_avail/feat.group_cap_kw;feat['cf']=(feat.energy/feat.cap_avail).clip(0,1.2)
 columns=[x for x in feat.columns if x not in NON]
 result={'protocol':'docs/extra_trees_group_protocol_20261105.md','method':'ExtraTrees standalone group; weighted squared loss','settlement':'README provisional D05 HOLD, official_S=null','python':platform.python_version(),'sklearn':sklearn.__version__,'feature_file_sha256':sha(a.features),'smp_file_sha256':sha(a.smp),'folds':{}}
 a.out.mkdir(parents=True,exist_ok=True)
 for fold,(cut,nsite) in FOLDS.items():
  g,paths=read_real(a.safe_root,a.reference_dir,fold,744)
  if g.duplicated(['group','ts']).any() or g.ts.nunique()!=744 or len(g)!=1488:raise ValueError('reference keys invalid')
  site=safe_parquet(paths[0]);actual_sites=site.site_id.nunique()
  if actual_sites!=nsite:raise ValueError('cohort site count changed')
  cutoff=pd.Timestamp(cut,tz=KST)
  train=feat[(feat.ts<=cutoff)&feat.energy.notna()&(feat['cov']>=.3)&(feat.cosz_max>0)].copy()
  valid=g[['group','ts']].merge(feat.drop(columns=['smp','k']),on=['group','ts'],how='left',validate='one_to_one')
  if train.empty or train.smp.isna().any() or valid.cosz.isna().any() or not train.ts.max()<g.ts.min():raise ValueError('missing inputs or temporal leak')
  w=train.cap_avail*train.smp.clip(lower=1)*train.k;w=w/w.mean()
  model=ExtraTreesRegressor(n_estimators=250,max_features=.8,min_samples_leaf=12,n_jobs=4,random_state=42)
  model.fit(train[columns],train.cf,sample_weight=w)
  g['extra']=np.where(valid.cosz_max.to_numpy()>0,np.clip(model.predict(valid[columns]),0,1)*g.cap,0)
  if g.extra.isna().any() or (g.extra<0).any():raise ValueError('invalid predictions')
  den=(g.smp*g.k*g.actual).sum()
  def metric(c):
   err=(g[c]-g.actual).abs();return {'penalty':float((.1*g.smp*g.k*err).sum()),'group_wmae':float((g.smp*g.k*err).sum()/den)}
  result['folds'][fold]={'train_rows':len(train),'train_max_ts':str(train.ts.max()),'validation_start':str(g.ts.min()),'validation_end':str(g.ts.max()),'sites':nsite,'hours':744,'capacity_kw':g.groupby('group').cap.first().to_dict(),'extra_trees':metric('extra'),'reference_1314':metric('reference_1314'),'input_sha256':{str(x):sha(x) for x in paths},'keys_actual_smp':'PASS'}
  print(fold,json.dumps(result['folds'][fold]),flush=True)
  g[['group','ts','actual','smp','k','cap','extra','reference_1314']].to_parquet(a.out/f'{fold.lower()}_predictions.parquet',index=False)
 (a.out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')

if __name__=='__main__':main()
