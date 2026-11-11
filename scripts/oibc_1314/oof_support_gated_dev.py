"""Chronological, past-OOF-only group correction. No locked input permitted."""
import argparse,hashlib,json,platform
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from analog_group_dev import weighted_median
from sklearn.preprocessing import PolynomialFeatures,RobustScaler,StandardScaler
from safe_reproduce import safe_parquet,LOCK

TRAIN={'DEV_JAN':(['DEV_JUL_2024','DEV_OCT_2024'],'2024-12-31 00:00'),
       'DEV_FEB':(['DEV_JUL_2024','DEV_OCT_2024','DEV_JAN'],'2025-01-31 00:00')}
FEATS=['ref_cf','cosz','cs_ab','wa_cloud','wb_cloud','wa_temp','wb_temp','wa_humidity','wb_rel_hum','wb_precip_prob','wb_total_liq','doy_sin','doy_cos','hour','group_id']

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def main():
 ap=argparse.ArgumentParser()
 for n in ('features','input_dir','out'):ap.add_argument(n,type=Path)
 a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 ft=safe_parquet(a.features)
 if ft.ts.max()>=LOCK or ft.duplicated(['group','ts']).any():raise ValueError('unsafe features')
 folds=['DEV_JUL_2024','DEV_OCT_2024','DEV_JAN','DEV_FEB']; paths={f:a.input_dir/f'{f.lower()}_analog.parquet' for f in folds}
 outputs=[]
 for f,p in paths.items():
  d=safe_parquet(p)
  if not d.fold.eq(f).all() or d.duplicated(['group','ts']).any():raise ValueError('bad OOF')
  outputs.append(d[['fold','group','ts','actual','base','cap','smp','k','reference']])
 d=pd.concat(outputs,ignore_index=True).merge(ft[['group','ts','group_id','cosz','cosz_max','cs_ab','wa_cloud','wb_cloud','wa_temp','wb_temp','wa_humidity','wb_rel_hum','wb_precip_prob','wb_total_liq','doy_sin','doy_cos','hour']],on=['group','ts'],validate='one_to_one',how='left')
 if d.cosz.isna().any() or (d.cap<=0).any() or d.reference.isna().any():raise ValueError('missing model inputs')
 d['ref_cf']=d.reference/d.cap
 result={'protocol':'docs/oof_support_gated_20261105.md','metric':'README provisional D05 HOLD official_S=null','method':'OOF residual nearest-neighbor with temperature-support abstention','python':platform.python_version(),'sklearn':sklearn.__version__,'feature_sha256':sha(a.features),'oof_sha256':{f:sha(p) for f,p in paths.items()},'provenance':'features created in the past from potentially mixed locked rowgroups; actual arrival unverified; JAN/FEB v2 independently re-trained','folds':{}}
 for fold,(train_folds,cut) in TRAIN.items():
  tr=d[d.fold.isin(train_folds)&(d.cosz_max>0)].copy();v=d[d.fold.eq(fold)].sort_values(['group','ts']).copy();cutoff=pd.Timestamp(cut,tz='Asia/Seoul')
  if len(v) not in (1440,1296) or tr.empty or tr.ts.max()>cutoff or tr.ts.max()>=v.ts.min():raise ValueError('leakage or cohort mismatch')
  v['corrected']=v.reference.to_numpy(); supported=0
  for group in ['G1','G2']:
   hist=tr[tr.group.eq(group)].copy()
   test=v[v.group.eq(group)&(v.cosz_max>0)].copy()
   if len(hist)<40 or hist[['wa_temp','wb_temp']].isna().all().any():raise ValueError('insufficient data')
   eligible=np.ones(len(test),dtype=bool)
   for col in ['wa_temp','wb_temp']:
    eligible &= test[col].notna().to_numpy() & (test[col].to_numpy()>=hist[col].min()) & (test[col].to_numpy()<=hist[col].max())
   selected=test.iloc[np.flatnonzero(eligible)]
   if selected.empty:continue
   imp=SimpleImputer(strategy='median');tx=imp.fit_transform(hist[FEATS]);vx=imp.transform(selected[FEATS])
   if tx.shape[1]!=len(FEATS):raise ValueError('missing feature')
   scale=RobustScaler();tx=scale.fit_transform(tx);vx=scale.transform(vx)
   nn=NearestNeighbors(n_neighbors=40,algorithm='brute',n_jobs=1).fit(tx);idx=nn.kneighbors(vx,return_distance=False)
   weight=hist.smp.clip(lower=1).to_numpy()*hist.k.to_numpy()*hist.cap.to_numpy()
   residual=((hist.actual-hist.reference)/hist.cap).to_numpy()
   delta=weighted_median(residual[idx],weight[idx]);val=np.maximum(selected.reference.to_numpy()+delta*selected.cap.to_numpy(),0)
   v.loc[selected.index,'corrected']=val;supported+=len(selected)
  if not np.isfinite(v.corrected).all() or (v.corrected<0).any():raise ValueError('invalid forecast')
  def metric(c):
   e=v.smp*v.k*abs(v[c]-v.actual);return {'penalty':float(.1*e.sum()),'wmae':float(e.sum()/(v.smp*v.k*v.actual).sum())}
  result['folds'][fold]={'training_folds':train_folds,'train_rows':len(tr),'train_last':str(tr.ts.max()),'cutoff':str(cutoff),'validation_start':str(v.ts.min()),'validation_end':str(v.ts.max()),'group_hours':len(v),'supported_daytime_rows':supported,'daytime_rows':int((v.cosz_max>0).sum()),'baseline':metric('reference'),'corrected':metric('corrected')}
  v[['fold','group','ts','actual','reference','cap','smp','k','corrected']].to_parquet(a.out/f'{fold.lower()}_corrected.parquet',index=False)
  print(fold,json.dumps(result['folds'][fold]),flush=True)
 (a.out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf8')
if __name__=='__main__':main()
