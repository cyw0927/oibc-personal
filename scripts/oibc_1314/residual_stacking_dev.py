"""Pre-registered cross-fitted group residual model; no locked inputs.

APR->AUG uses proxy comparison only. JUL->OCT additionally compares true #1314
on its documented v2 JUL/OCT DEV reference. See residual_stacking_protocol.
Predictions, actuals and results are confidential derived data: keep outside Git.
"""
import argparse,hashlib,json,platform
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from safe_reproduce import safe_parquet

KST='Asia/Seoul'
FEATURES=['group_id','cosz','cs_ab','wa_cloud','wb_cloud','wa_temp','wb_temp','wb_uv_idx','doy_sin','doy_cos','hour','site_cf','direct_cf','gap_cf']

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def check_group(d,hours):
 if d.duplicated(['group','ts']).any() or len(d)!=2*hours or d.ts.nunique()!=hours or set(d.group)!={'G1','G2'}:raise ValueError('group keys incomplete')
 if d[['actual','site','direct','smp','k','cap']].isna().any().any() or (d.cap<=0).any() or (d.smp<0).any():raise ValueError('invalid group data')
 return d.sort_values(['group','ts']).reset_index(drop=True)

def read_proxy(root,fold,hours):
 path=root/'group_penalty_20261105'/f'{fold.lower()}_group_predictions.parquet'
 d=safe_parquet(path)[['group','ts','actual','site','cap','smp','k','coverage']].rename(columns={'coverage':'direct'})
 return check_group(d,hours),[path]

def read_real(root,refroot,fold,hours):
 path=root/'weather_l1_smpk_20261105'/f'{fold.lower()}_site_predictions.parquet'
 refpath=refroot/f'{fold}_compare.parquet'
 d=safe_parquet(path)
 if d.duplicated(['site_id','ts']).any() or not np.allclose(d.energy,d.actual_kwh,rtol=0,atol=1e-9):raise ValueError('site OOF actual mismatch')
 d=d.groupby(['group','ts'],as_index=False).agg(actual=('energy','sum'),site=('site_pred','sum'),cap=('capacity_kw','sum'),team=('prediction_kwh','sum'))
 r=safe_parquet(refpath)[['group','ts','actual','smp','k','2023']].rename(columns={'actual':'ref_actual','2023':'direct'})
 d=d.merge(r,on=['group','ts'],how='outer',validate='one_to_one',indicator=True)
 if not d['_merge'].eq('both').all() or not np.allclose(d.actual,d.ref_actual,rtol=0,atol=1e-6):raise ValueError('reference keys or actual mismatch')
 d=d.drop(columns=['_merge'])
 d['reference_1314']=.4*d.team+.6*d.direct
 return check_group(d,hours),[path,refpath]

def attach(d,features):
 x=d.merge(features,on=['group','ts'],how='left',validate='one_to_one',indicator=True)
 if not x['_merge'].eq('both').all() or x['cosz'].isna().any():raise ValueError('missing group forecast features')
 x['site_cf']=x.site/x.cap; x['direct_cf']=x.direct/x.cap; x['gap_cf']=x.site_cf-x.direct_cf
 x['raw']=.4*x.site+.6*x.direct
 return x

def main():
 p=argparse.ArgumentParser();p.add_argument('safe_root',type=Path);p.add_argument('reference_dir',type=Path);p.add_argument('features',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
 feat=safe_parquet(a.features)
 if feat.ts.max()>pd.Timestamp('2024-11-01',tz=KST) or feat.duplicated(['group','ts']).any():raise ValueError('unsafe features')
 feat=feat[['group','ts']+[x for x in FEATURES if x not in ('site_cf','direct_cf','gap_cf')]]
 meta={
  'proxy':json.loads((a.safe_root/'group_penalty_20261105'/'results.json').read_text()),
  'real':json.loads((a.safe_root/'weather_l1_smpk_20261105'/'results.json').read_text())}
 cases=[('APR_to_AUG','proxy','APR_2024','AUG_2024',720,744),('JUL_to_OCT','real','DEV_JUL_2024','DEV_OCT_2024',744,744)]
 result={'protocol':'docs/residual_stacking_protocol_20261105.md','metric':'README provisional D05 HOLD; official_S=null','method':'fixed 40/60 plus OOF residual HGB; historical SMP/k used as training weight only','seed':7,'python':platform.python_version(),'sklearn':sklearn.__version__,'features_sha256':sha(a.features),'cases':{}}
 a.output.mkdir(parents=True,exist_ok=True)
 for case,kind,train_fold,test_fold,nh_train,nh_test in cases:
  reader=read_proxy if kind=='proxy' else read_real
  if kind=='proxy': train,train_paths=reader(a.safe_root,train_fold,nh_train);test,test_paths=reader(a.safe_root,test_fold,nh_test)
  else: train,train_paths=reader(a.safe_root,a.reference_dir,train_fold,nh_train);test,test_paths=reader(a.safe_root,a.reference_dir,test_fold,nh_test)
  train=attach(train,feat);test=attach(test,feat)
  # Check recorded base model cutoffs precede each OOF period, and meta period precedes evaluation.
  record=meta[kind]['folds']; tc=pd.Timestamp(record[train_fold]['cutoff'] if kind=='proxy' else record[train_fold]['train_max_ts']); vc=pd.Timestamp(record[test_fold]['cutoff'] if kind=='proxy' else record[test_fold]['train_max_ts'])
  if not (tc<train.ts.min() and train.ts.max()<test.ts.min() and vc<test.ts.min()):raise ValueError('temporal leakage')
  train=train[train.cosz>0].copy(); w=train.smp*train.k*train.cap
  if (w<0).any() or w.sum()<=0:raise ValueError('invalid weights')
  model=HistGradientBoostingRegressor(loss='absolute_error',learning_rate=.05,max_iter=120,max_leaf_nodes=7,min_samples_leaf=40,l2_regularization=5,early_stopping=False,random_state=7)
  model.fit(train[FEATURES].astype('float32'),(train.actual-train.raw)/train.cap,sample_weight=w/w.mean())
  correction=model.predict(test[FEATURES].astype('float32'))
  test['stacked']=np.where(test.cosz>0,np.clip(test.raw/test.cap+correction,0,1.2)*test.cap,0)
  den=(test.smp*test.k*test.actual).sum()
  def metrics(column):
   err=(test[column]-test.actual).abs();return {'penalty':float((.1*test.smp*test.k*err).sum()),'wmae':float((test.smp*test.k*err).sum()/den)}
  case_result={'meta_train_start':str(train.ts.min()),'meta_train_end':str(train.ts.max()),'base_fit_cutoff_train':str(tc),'base_fit_cutoff_validation':str(vc),'validation_start':str(test.ts.min()),'validation_end':str(test.ts.max()),'meta_training_rows':len(train),'group_hours':len(test),'cohort_capacity_kw':test.groupby('group').cap.first().to_dict(),'proxy_raw':metrics('raw'),'stacked':metrics('stacked'),'input_sha256':{str(q):sha(q) for q in train_paths+test_paths}}
  if kind=='real':case_result['actual_1314']=metrics('reference_1314')
  result['cases'][case]=case_result
  print(case,json.dumps(case_result),flush=True)
  test[['group','ts','actual','site','direct','raw','stacked','smp','k','cap']].to_parquet(a.output/f'{case.lower()}_predictions.parquet',index=False)
 (a.output/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')

if __name__=='__main__':main()
