"""Chronological OOF-to-group model, no target-period labels as inputs."""
import argparse, hashlib, json, platform
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import QuantileRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from safe_reproduce import safe_parquet, LOCK

FEATURES=['base_cf','cosz','cs_ab','wa_cloud','wb_cloud','wa_temp','wb_temp','wa_humidity','wb_rel_hum','wb_precip_prob','wb_total_liq','doy_sin','doy_cos','hour','group_id']
PLAN={'DEV_JAN':(['DEV_MAY_2024','DEV_JUL_2024','DEV_AUG_2024','DEV_SEP_2024','DEV_OCT_2024'],'2024-12-31 00:00'), 'DEV_FEB':(['DEV_MAY_2024','DEV_JUL_2024','DEV_AUG_2024','DEV_SEP_2024','DEV_OCT_2024','DEV_JAN'],'2025-01-31 00:00')}

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser()
 for x in ('features','analog_dir','sep','sep_result','extra_dir','out'):p.add_argument(x,type=Path)
 a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 ft=safe_parquet(a.features)
 if ft.ts.max()>=LOCK or ft.duplicated(['group','ts']).any():raise ValueError('bad feature metadata or keys')
 parts=[]; paths={}
 for fold in ('DEV_JUL_2024','DEV_OCT_2024','DEV_JAN','DEV_FEB'):
  q=a.analog_dir/f'{fold.lower()}_analog.parquet';paths[fold]=q;d=safe_parquet(q)
  if not d.fold.eq(fold).all():raise ValueError('fold mismatch')
  parts.append(d[['fold','group','ts','actual','base','cap','smp','k','reference']])
 sep_info=json.loads(a.sep_result.read_text(encoding='utf8'))
 if sep_info['control_max_diff']!=0 or pd.Timestamp(sep_info['cutoff'])>=pd.Timestamp(sep_info['period'][0]):raise ValueError('invalid SEP provenance')
 d=safe_parquet(a.sep);d['fold']='DEV_SEP_2024';paths['DEV_SEP_2024']=a.sep
 if d.ts.min()!=pd.Timestamp('2024-09-01 01:00',tz='Asia/Seoul') or d.ts.max()!=pd.Timestamp('2024-10-01 00:00',tz='Asia/Seoul'):raise ValueError('SEP interval')
 parts.append(d[['fold','group','ts','actual','base','cap','smp','k','reference']])
 for month in ('may_2024','aug_2024'):
  q=a.extra_dir/month/'group_oof.parquet';manifest=a.extra_dir/month/'manifest.json'; info=json.loads(manifest.read_text(encoding='utf8'))
  if sha(q)!=info['group_sha256'] or pd.Timestamp(info['cutoff'])>=pd.Timestamp(info['period'][0]) or info['seeds']!=[11,22,33]:raise ValueError('bad historical OOF provenance')
  d=safe_parquet(q);d['fold']='DEV_'+month.upper();d['reference']=np.nan;paths[d.fold.iloc[0]]=q
  if d.ts.min()!=pd.Timestamp(info['period'][0]) or d.ts.max()!=pd.Timestamp(info['period'][1]):raise ValueError('OOF interval mismatch')
  parts.append(d[['fold','group','ts','actual','base','cap','smp','k','reference']])
 d=pd.concat(parts,ignore_index=True)
 if d.duplicated(['group','ts']).any() or (d.cap<=0).any() or d[['actual','base','smp','k']].isna().any().any():raise ValueError('invalid OOF')
 used=['group','ts','group_id','cosz','cosz_max','cs_ab','wa_cloud','wb_cloud','wa_temp','wb_temp','wa_humidity','wb_rel_hum','wb_precip_prob','wb_total_liq','doy_sin','doy_cos','hour']
 d=d.merge(ft[used],on=['group','ts'],how='left',validate='one_to_one')
 if d.cosz.isna().any() or (d.smp<0).any():raise ValueError('invalid inputs')
 d['base_cf']=d.base/d.cap
 result={'protocol':'docs/oof_ablation_quantile_plan_20261106.md','metric':'README provisional D05 HOLD official_S=null','method':'weighted quantile linear regression from previous OOF','features':FEATURES,'python':platform.python_version(),'sklearn':sklearn.__version__,'features_sha256':sha(a.features),'input_sha256':{n:sha(q) for n,q in paths.items()},'sep_provenance_sha256':sha(a.sep_result),'caveat':'reused DEV, JUL/OCT partial cohorts, JAN/FEB #1314 reconstructed, weather arrival unverified, prelock features historically may have opened mixed locked rowgroups','folds':{}}
 for fold,(hist,cut) in PLAN.items():
  tr=d[d.fold.isin(hist)&(d.cosz_max>0)].copy(); v=d[d.fold.eq(fold)].sort_values(['group','ts']).copy();cut=pd.Timestamp(cut,tz='Asia/Seoul')
  if tr.empty or v.reference.isna().any() or tr.ts.max()>cut or tr.ts.max()>=v.ts.min() or len(v) not in (1440,1296):raise ValueError('training and evaluation not separated')
  w=tr.smp.clip(lower=1).to_numpy()*tr.k.to_numpy()*tr.cap.to_numpy();target=(tr.actual/tr.cap).to_numpy()
  imp=SimpleImputer(strategy='median',keep_empty_features=True); tx=imp.fit_transform(tr[FEATURES]);vx=imp.transform(v[FEATURES])
  scale=StandardScaler();tx=scale.fit_transform(tx);vx=scale.transform(vx)
  m=QuantileRegressor(quantile=.5,alpha=.01,solver='highs')
  m.fit(tx,target,sample_weight=w/w.mean());pred=np.clip(m.predict(vx),0,1);v['candidate']=np.where(v.cosz_max>0,pred,0)*v.cap
  if not np.isfinite(v.candidate).all() or (v.candidate<0).any():raise ValueError('bad forecast')
  def metric(col):
   e=v.smp*v.k*abs(v[col]-v.actual);return {'penalty':float(.1*e.sum()),'wmae':float(e.sum()/(v.smp*v.k*v.actual).sum())}
  result['folds'][fold]={'training_folds':hist,'training_rows':len(tr),'training_max':str(tr.ts.max()),'validation_min':str(v.ts.min()),'validation_rows':len(v),'baseline':metric('reference'),'candidate':metric('candidate')}
  v[['fold','group','ts','actual','base','cap','smp','k','reference','candidate']].to_parquet(a.out/f'{fold.lower()}_predictions.parquet',index=False)
  print(fold,json.dumps(result['folds'][fold]),flush=True)
 (a.out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf8')
if __name__=='__main__':main()
