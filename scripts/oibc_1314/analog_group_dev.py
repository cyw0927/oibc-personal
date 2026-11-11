"""Historical weather-analog group model, no locked data. See preregistered protocol."""
import argparse,hashlib,json,platform
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler
from safe_reproduce import safe_parquet, LOCK

FOLDS={'DEV_JUL_2024':('2024-06-30 00:00',224,744),
       'DEV_OCT_2024':('2024-09-30 00:00',252,744),
       'DEV_JAN':('2024-12-31 00:00',253,720),
       'DEV_FEB':('2025-01-31 00:00',253,648)}
FEATURES=['cosz','cs_ab','wa_cloud','wb_cloud','wa_temp','wb_temp','wa_humidity','wb_rel_hum','wb_precip_prob','wb_total_liq','wa_cloud_sd','wb_cloud_sd','doy_sin','doy_cos']
K=40

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def weighted_median(values,weights):
 order=np.argsort(values,axis=1,kind='stable');v=np.take_along_axis(values,order,axis=1);w=np.take_along_axis(weights,order,axis=1)
 if not np.isfinite(v).all() or not np.isfinite(w).all() or (w<0).any() or (w.sum(axis=1)<=0).any():raise ValueError('invalid neighbor weights')
 return v[np.arange(len(v)),(np.cumsum(w,axis=1)>=w.sum(axis=1)[:,None]/2).argmax(axis=1)]

def main():
 ap=argparse.ArgumentParser()
 for n in ('features','oof','sites','smp','ref','winter_dir','out'):ap.add_argument(n,type=Path)
 a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 ft=safe_parquet(a.features);oof=safe_parquet(a.oof);ref=safe_parquet(a.ref)
 winters={f:safe_parquet(a.winter_dir/f'{f.lower()}_predictions.parquet') for f in ('DEV_JAN','DEV_FEB')}
 if ft.ts.max()>=LOCK or ft.duplicated(['group','ts']).any():raise ValueError('unsafe/duplicate features')
 sites=pd.read_csv(a.sites,usecols=['site_id','current_group','capacity_kw']);sites['group']=sites.current_group.map({'A':'G1','B':'G2'})
 if len(sites)!=253 or sites.site_id.duplicated().any() or sites.group.isna().any():raise ValueError('sites invalid')
 smp=pd.read_csv(a.smp,parse_dates=['ts']);smp.ts=pd.to_datetime(smp.ts,utc=True).dt.tz_convert('Asia/Seoul').dt.as_unit('us')
 if smp.ts.max()>=LOCK or smp.ts.duplicated().any() or smp.smp.isna().any() or (smp.smp<0).any():raise ValueError('market inputs invalid')
 smp['k']=np.where(smp.ts.dt.hour.between(11,15),3,1)
 if oof.duplicated(['fold','site_id','ts']).any():raise ValueError('duplicate OOF')
 oof=oof[oof.fold.isin(FOLDS)].merge(sites[['site_id','group','capacity_kw']],on='site_id',validate='many_to_one')
 agg=oof.groupby(['fold','group','ts'],as_index=False).agg(actual=('actual_kwh','sum'),base=('prediction_kwh','sum'),cap=('capacity_kw','sum'),n=('site_id','nunique'))
 agg=agg.merge(smp,on='ts',validate='many_to_one',how='left')
 if agg.smp.isna().any():raise ValueError('missing price')
 ft.ts=ft.ts.dt.as_unit('us');ft=ft.merge(smp,on='ts',how='left',validate='many_to_one');ft['cov']=ft.cap_avail/ft.group_cap_kw;ft['cf']=(ft.energy/ft.cap_avail).clip(0,1.2)
 result={'protocol':'docs/analog_group_architecture_20261105.md','metric':'README provisional D05 HOLD official_S=null','method':'same-group same-hour 40 nearest weather-forecast analogues, historic weighted median','python':platform.python_version(),'sklearn':sklearn.__version__,'provenance':'feature file past creation may have opened mixed locked row groups; original winter v2 not available; weather issue/arrival unverified','inputs':{n:sha(getattr(a,n)) for n in ('features','oof','sites','smp','ref')},'winter_inputs':{f:sha(a.winter_dir/f'{f.lower()}_predictions.parquet') for f in winters},'folds':{}}
 for fold,(cut,nsites,hours) in FOLDS.items():
  val=agg[agg.fold.eq(fold)].sort_values(['group','ts']).reset_index(drop=True)
  if len(val)!=hours*2 or val.ts.nunique()!=hours or oof[oof.fold.eq(fold)].site_id.nunique()!=nsites:raise ValueError('cohort mismatch')
  if nsites==253: existing=winters[fold][['group','ts','actual','base','cap','smp','k','baseline']].rename(columns={'baseline':'reference'})
  else: existing=ref[ref.fold.eq(fold)][['group','ts','actual_kwh','l1_kwh','smp','k','prediction_kwh']].rename(columns={'actual_kwh':'actual','l1_kwh':'base','prediction_kwh':'reference'})
  check=val.merge(existing,on=['group','ts'],how='outer',validate='one_to_one',indicator=True,suffixes=('','_ref'))
  if not check._merge.eq('both').all():raise ValueError('reference key mismatch')
  for c in ['actual','base','smp','k']:
   if not np.allclose(check[c],check[c+'_ref'],rtol=0,atol=1e-6):raise ValueError('reference mismatch '+c)
  val['reference']=check.reference.to_numpy()
  tr=ft[(ft.ts>=pd.Timestamp('2023-06-01 01:00',tz='Asia/Seoul'))&(ft.ts<=pd.Timestamp(cut,tz='Asia/Seoul'))&ft.energy.notna()&(ft['cov']>=.3)&(ft.cosz_max>0)].copy()
  va=val[['group','ts']].merge(ft[['group','ts','cosz_max']+FEATURES],on=['group','ts'],validate='one_to_one',how='left')
  if tr.smp.isna().any() or va.cosz_max.isna().any() or not tr.ts.max()<val.ts.min():raise ValueError('missing inputs or temporal leakage')
  val['analog_cf']=0.0; neighbor_min=10**9
  for group in ['G1','G2']:
   for hour in range(24):
    idx=va.index[(va.group==group)&(va.ts.dt.hour==hour)&(va.cosz_max>0)]
    if len(idx)==0:continue
    hist=tr[(tr.group==group)&(tr.ts.dt.hour==hour)]
    if len(hist)<K:raise ValueError('insufficient training analogues')
    inp=SimpleImputer(strategy='median');train_x=inp.fit_transform(hist[FEATURES]);valid_x=inp.transform(va.loc[idx,FEATURES])
    if train_x.shape[1]!=len(FEATURES):raise ValueError('all-missing feature')
    scale=RobustScaler();train_x=scale.fit_transform(train_x);valid_x=scale.transform(valid_x)
    nn=NearestNeighbors(n_neighbors=K,algorithm='brute',metric='euclidean',n_jobs=1).fit(train_x);neighbors=nn.kneighbors(valid_x,return_distance=False)
    weights=hist['cov'].to_numpy()*hist.smp.clip(lower=1).to_numpy()*hist.k.to_numpy()
    val.loc[idx,'analog_cf']=np.clip(weighted_median(hist.cf.to_numpy()[neighbors],weights[neighbors]),0,1)
    neighbor_min=min(neighbor_min,len(hist))
  val['analog_direct']=val.analog_cf*val.cap;val['candidate']=.4*val.base+.6*val.analog_direct
  if not np.isfinite(val.candidate).all() or (val.candidate<0).any():raise ValueError('invalid output')
  def metric(c):
   e=val.smp*val.k*abs(val[c]-val.actual);return {'penalty':float(.1*e.sum()),'wmae':float(e.sum()/(val.smp*val.k*val.actual).sum())}
  result['folds'][fold]={'sites':nsites,'group_hours':len(val),'training_rows':len(tr),'min_group_hour_history':int(neighbor_min),'train_max':str(tr.ts.max()),'start':str(val.ts.min()),'end':str(val.ts.max()),'reference_type':'stored v2' if nsites!=253 else 'retrained v2', 'reference':metric('reference'),'analog_direct':metric('analog_direct'),'analog_blend':metric('candidate'),'key_actual_smp_k':'PASS'}
  val[['fold','group','ts','actual','base','cap','smp','k','reference','analog_direct','candidate']].to_parquet(a.out/f'{fold.lower()}_analog.parquet',index=False)
  print(fold,json.dumps(result['folds'][fold]),flush=True)
 (a.out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf8')
if __name__=='__main__':main()
