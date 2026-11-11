"""Cross-fitted plant and group forecasts for every divisor of 24 hours."""
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from wls_spatial_reconcile import safe_parquet,LOCK,ROOT,OOF,SITES,OCT,WINTER,sha
from cross_temporal_reconcile import FEATURES,DAILY_COLS

OUT=ROOT/'multiscale_20261109'; DURATIONS=(1,2,3,4,6,8,12,24)
PREVIOUS={'DEV_OCT_2024':['DEV_MAY_2024','DEV_JUL_2024','DEV_AUG_2024'],
          'DEV_JAN':['DEV_MAY_2024','DEV_JUL_2024','DEV_AUG_2024','DEV_OCT_2024'],
          'DEV_FEB':['DEV_MAY_2024','DEV_JUL_2024','DEV_AUG_2024','DEV_OCT_2024','DEV_JAN']}
CUT={'DEV_OCT_2024':'2024-09-30 00:00','DEV_JAN':'2024-12-31 00:00','DEV_FEB':'2025-01-31 00:00'}

def ridge_fit(x,y,alpha=1.):
 x=np.asarray(x,float);y=np.asarray(y,float); med=np.nanmedian(x,axis=0)
 if not np.isfinite(med).all() or not np.isfinite(y).all():raise ValueError('invalid training features')
 x=np.where(np.isfinite(x),x,med);mu=x.mean(0);sd=x.std(0);sd=np.where(sd>1e-12,sd,1.);z=np.column_stack([np.ones(len(x)),(x-mu)/sd]);pen=np.eye(z.shape[1])*alpha;pen[0,0]=0
 return med,mu,sd,np.linalg.solve(z.T@z+pen,z.T@y)
def ridge_predict(model,x):
 med,mu,sd,b=model;x=np.asarray(x,float);x=np.where(np.isfinite(x),x,med);return np.column_stack([np.ones(len(x)),(x-mu)/sd])@b

def with_blocks(d,h):
 d=d.copy();t=d.ts-pd.Timedelta(hours=1);d['day']=t.dt.floor('D');d['slot']=t.dt.hour//h;return d

def site_period(d,h):
 z=with_blocks(d,h);s=z.groupby(['site_id','group','capacity_kw','day','slot'],as_index=False).agg(actual=('actual_kwh','mean'),base=('prediction_kwh','mean'),hours=('ts','nunique'),last=('ts','max'))
 s=s[s.hours.eq(h)].copy();s['base_cf']=s.base/s.capacity_kw;s['actual_cf']=s.actual/s.capacity_kw;s['base2']=s.base_cf**2
 s['group_id']=(s.group=='G2').astype(float);s['hour_sin']=np.sin(2*np.pi*(s.slot*h+h/2)/24);s['hour_cos']=np.cos(2*np.pi*(s.slot*h+h/2)/24)
 s['doy_sin']=np.sin(2*np.pi*s.day.dt.dayofyear/365.25);s['doy_cos']=np.cos(2*np.pi*s.day.dt.dayofyear/365.25)
 s['log_cap']=np.log1p(s.capacity_kw);return s

SITE_COLS=['base_cf','base2','group_id','hour_sin','hour_cos','doy_sin','doy_cos','log_cap']

def group_period(ft,h):
 z=with_blocks(ft,h);z['target_cf']=z.energy/z.cap_avail;z['good']=z.energy.notna()&(z.cap_avail/z.group_cap_kw>=.3)&np.isfinite(z.target_cf)
 v=z.groupby(['group','day','slot'],as_index=False).agg(**{c:(c,'mean') for c in DAILY_COLS},target_cf=('target_cf','mean'),good=('good','sum'),hours=('ts','nunique'),last=('ts','max'))
 return v[v.hours.eq(h)].copy()

def main():
 OUT.mkdir(parents=True,exist_ok=True);ft=safe_parquet(FEATURES);all_oof=safe_parquet(OOF)
 sites=pd.read_csv(SITES,usecols=['site_id','current_group','capacity_kw']);sites['group']=sites.current_group.map({'A':'G1','B':'G2'})
 if len(sites)!=253 or sites.site_id.duplicated().any() or sites.group.isna().any():raise ValueError('bad site contract')
 if ft.ts.max()>=LOCK or ft.duplicated(['group','ts']).any():raise ValueError('unsafe features')
 sources={}; source_paths={}
 for fold in ['DEV_JUL_2024','DEV_OCT_2024','DEV_JAN','DEV_FEB']:
  d=all_oof[all_oof.fold.eq(fold)][['site_id','ts','actual_kwh','prediction_kwh']].merge(sites[['site_id','group','capacity_kw']],on='site_id',validate='many_to_one');sources[fold]=d
 for month in ['may_2024','aug_2024']:
  p=ROOT/'expanded_oof_20261106'/month/'site_oof.parquet';meta=ROOT/'expanded_oof_20261106'/month/'manifest.json';m=json.loads(meta.read_text(encoding='utf8'))
  if sha(p)!=m['site_sha256'] or pd.Timestamp(m['cutoff'])>=pd.Timestamp(m['period'][0]):raise ValueError('extra OOF provenance')
  d=safe_parquet(p).rename(columns={'energy':'actual_kwh','base':'prediction_kwh'});sources['DEV_'+month.upper()]=d;source_paths[month]=sha(p)
 group_data={'DEV_OCT_2024':safe_parquet(OCT).query("fold == 'DEV_OCT_2024'").rename(columns={'actual_kwh':'actual','l1_kwh':'base','direct_kwh':'direct','prediction_kwh':'reference'})}
 for f in ['DEV_JAN','DEV_FEB']:group_data[f]=safe_parquet(WINTER/f'{f.lower()}_predictions.parquet').rename(columns={'baseline':'reference'})
 info={'durations':list(DURATIONS),'input_sha256':{'base_oof':sha(OOF),'features':sha(FEATURES),'sites':sha(SITES),**source_paths},'folds':{},'protocol':'docs/cross_temporal_full_scales_plan_20261109.md'}
 for fold in ['DEV_OCT_2024','DEV_JAN','DEV_FEB']:
  cutoff=pd.Timestamp(CUT[fold],tz='Asia/Seoul');test=sources[fold].copy();tr=pd.concat([sources[x] for x in PREVIOUS[fold]],ignore_index=True)
  if tr.ts.max()>cutoff or tr.ts.max()>=test.ts.min() or tr.duplicated(['site_id','ts']).any() or test.duplicated(['site_id','ts']).any():raise ValueError('site OOF chronology/keys')
  g=group_data[fold].copy();g=with_blocks(g,1); site_result=[];group_result=[];counts={}
  for h in DURATIONS[1:]:
   st=site_period(tr,h);sv=site_period(test,h)
   if st.empty or sv.empty or st['last'].max()>cutoff:raise ValueError('site period training invalid')
   model=ridge_fit(st[SITE_COLS],st.actual_cf);sv['prediction_cf']=np.clip(ridge_predict(model,sv[SITE_COLS]),0,1);sv.loc[sv.base_cf<=0,'prediction_cf']=0.;sv['duration']=h
   site_result.append(sv[['site_id','group','day','slot','duration','prediction_cf']]);
   gp=group_period(ft,h);gt=gp[(gp['last']<=cutoff)&(gp.good==h)&(gp.day>=pd.Timestamp('2023-06-01',tz='Asia/Seoul'))].copy()
   wanted=with_blocks(group_data[fold],h).groupby(['group','day','slot'],as_index=False).agg(hours=('ts','nunique'))
   gv=wanted.merge(gp[['group','day','slot']+DAILY_COLS],on=['group','day','slot'],validate='one_to_one')
   if gt.empty or not wanted.hours.eq(h).all() or len(gv)!=len(wanted) or gt['last'].max()>cutoff:raise ValueError('group period invalid')
   model=ridge_fit(gt[DAILY_COLS],gt.target_cf);gv['prediction_cf']=np.clip(ridge_predict(model,gv[DAILY_COLS]),0,1);gv['duration']=h
   group_result.append(gv[['group','day','slot','duration','prediction_cf']]);counts[str(h)]={'site_training_rows':len(st),'site_validation_rows':len(sv),'group_training_rows':len(gt),'group_validation_rows':len(gv)}
  sp=pd.concat(site_result,ignore_index=True);gp=pd.concat(group_result,ignore_index=True)
  sp.to_parquet(OUT/f'{fold.lower()}_site_levels.parquet',index=False);gp.to_parquet(OUT/f'{fold.lower()}_group_levels.parquet',index=False)
  info['folds'][fold]={'cutoff':str(cutoff),'training_folds':PREVIOUS[fold],'site_training_max':str(tr.ts.max()),'validation_start':str(test.ts.min()),'validation_end':str(test.ts.max()),'sites':int(test.site_id.nunique()),'levels':counts,'site_levels_sha256':sha(OUT/f'{fold.lower()}_site_levels.parquet'),'group_levels_sha256':sha(OUT/f'{fold.lower()}_group_levels.parquet')}
  print(fold,'sites',test.site_id.nunique(),'site levels',len(sp),'group levels',len(gp),flush=True)
 (OUT/'base_manifest.json').write_text(json.dumps(info,indent=2),encoding='utf8')
if __name__=='__main__':main()
