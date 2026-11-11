"""Independent weather-based site model: safe JUL/OCT DEV benchmark only.

Inputs must already be physically separated from locked periods. No team predictions
are used for model fitting. Historical team OOF is used solely for comparison keys
and the #1314 reference blend. Never commit inputs, predictions or output artifacts.
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from safe_reproduce import safe_parquet

KST='Asia/Seoul'
END=pd.Timestamp('2024-11-01 00:00',tz=KST)
START=pd.Timestamp('2023-06-01 01:00',tz=KST)
FOLDS={'DEV_JUL_2024':('2024-06-30 00:00','2024-07-01 01:00','2024-08-01 00:00',224),
       'DEV_OCT_2024':('2024-09-30 00:00','2024-10-01 01:00','2024-11-01 00:00',252)}
WA=['wa_cloud','wa_temp','wa_humidity','wa_rain','wa_wind_speed','wa_ground_press']
WB=['wb_cloud','wb_temp','wb_rel_hum','wb_uv_idx','wb_precip_prob','wb_total_liq','wb_vis','wb_wind_speed','wb_dew_point']
FEATURES=WA+WB+['cosz','doy_sin','doy_cos','hour','capacity_kw','lat','lon']

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''): h.update(chunk)
    return h.hexdigest()

def read_weather(path,station,cols):
    d=safe_parquet(path)
    if d.ts.max()>END: raise ValueError('Weather extends past the safe split')
    d=d[d['round'].isin([10,17])][[station,'round','ts']+cols].copy()
    if d.duplicated([station,'round','ts']).any(): raise ValueError(f'duplicate {station} forecast key')
    # Pre-registered baseline: mean of available issued rounds, no observed weather.
    return d.groupby([station,'ts'],as_index=False,sort=False)[cols].mean()

def solar_cosz(ts,lat,lon):
    utc=(pd.DatetimeIndex(ts)-pd.Timedelta(minutes=30)).tz_convert('UTC')
    doy=utc.dayofyear.to_numpy(); hr=utc.hour.to_numpy()+utc.minute.to_numpy()/60
    g=2*np.pi/365*(doy-1+(hr-12)/24)
    eq=229.18*(.000075+.001868*np.cos(g)-.032077*np.sin(g)-.014615*np.cos(2*g)-.040849*np.sin(2*g))
    dec=.006918-.399912*np.cos(g)+.070257*np.sin(g)-.006758*np.cos(2*g)+.000907*np.sin(2*g)-.002697*np.cos(3*g)+.00148*np.sin(3*g)
    ha=np.radians((hr*60+eq+4*lon)/4-180); la=np.radians(lat)
    return np.sin(la)*np.sin(dec)+np.cos(la)*np.cos(dec)*np.cos(ha)

def penalty(df,pred): return float((.1*df.smp*df.k*(df[pred]-df.actual).abs()).sum())

def main():
    ap=argparse.ArgumentParser()
    for name in ('generation','wa','wb','sites','smp','oof','reference_dir','out'):
        ap.add_argument(name,type=Path)
    a=ap.parse_args()
    sites=pd.read_csv(a.sites)[['site_id','current_group','capacity_kw','lat','lon','wa_station','wb_station']]
    if len(sites)!=253 or sites.site_id.duplicated().any(): raise ValueError('invalid sites')
    sites['group']=sites.current_group.map({'A':'G1','B':'G2'})
    if sites.group.isna().any(): raise ValueError('invalid mapping')
    gen=safe_parquet(a.generation)
    if gen.ts.max()>END or gen.duplicated(['site_id','ts']).any(): raise ValueError('unsafe or duplicate generation')
    gen=gen[(gen.ts>=START)&(gen.ts<=END)].merge(sites,on='site_id',validate='many_to_one')
    wa=read_weather(a.wa,'wa_station',WA); wb=read_weather(a.wb,'wb_station',WB)
    gen=gen.merge(wa,on=['wa_station','ts'],how='left',validate='many_to_one').merge(wb,on=['wb_station','ts'],how='left',validate='many_to_one')
    gen['cosz']=solar_cosz(gen.ts,gen.lat.to_numpy(),gen.lon.to_numpy())
    gen['hour']=gen.ts.dt.hour; doy=gen.ts.dt.dayofyear
    gen['doy_sin']=np.sin(2*np.pi*doy/365.25);gen['doy_cos']=np.cos(2*np.pi*doy/365.25)
    gen['cf']=(gen.energy/gen.capacity_kw).clip(0,1.2)
    oof=safe_parquet(a.oof); oof=oof[oof.fold.isin(FOLDS)][['fold','site_id','ts','actual_kwh','prediction_kwh']]
    if oof.duplicated(['fold','site_id','ts']).any(): raise ValueError('duplicate OOF keys')
    smp=pd.read_csv(a.smp,parse_dates=['ts']); smp.ts=pd.to_datetime(smp.ts,utc=True).dt.tz_convert(KST)
    if smp.ts.max()>END or smp.ts.duplicated().any(): raise ValueError('unsafe or duplicate SMP')
    smp['k']=np.where(smp.ts.dt.hour.between(11,15),3,1)
    a.out.mkdir(parents=True,exist_ok=True)
    result={'metric':'README provisional; D05 HOLD; official_S=null','method':'HGB absolute-error, 22-feature weather+solar+site; rounds 10/17 feature mean; 250k deterministic training sample','seed':42,'python':platform.python_version(),'sklearn':sklearn.__version__,'inputs':{k:sha(getattr(a,k)) for k in ('generation','wa','wb','sites','smp','oof')},'reference_sha256':{fold:sha(a.reference_dir/f'{fold}_compare.parquet') for fold in FOLDS},'folds':{}}
    for fold,(end,vstart,vend,nsite) in FOLDS.items():
        train=gen[(gen.ts<=pd.Timestamp(end,tz=KST)) & (gen.cosz>0) & gen.energy.notna()].copy()
        valid=gen[(gen.ts>=pd.Timestamp(vstart,tz=KST))&(gen.ts<=pd.Timestamp(vend,tz=KST))].copy()
        keys=oof[oof.fold.eq(fold)]
        valid=valid.merge(keys,on=['site_id','ts'],how='inner',validate='one_to_one')
        if len(valid)!=len(keys) or valid.site_id.nunique()!=nsite or valid.ts.nunique()!=744 or len(valid)!=nsite*744: raise ValueError('Validation cohort mismatch')
        if not np.allclose(valid.energy,valid.actual_kwh,rtol=0,atol=1e-9): raise ValueError('actual mismatch')
        if train[WA+WB].isna().all(axis=1).any() or valid[WA+WB].isna().all(axis=1).any(): raise ValueError('entire weather missing')
        if len(train)>250000: train=train.sample(n=250000,random_state=42)
        model=HistGradientBoostingRegressor(loss='absolute_error',learning_rate=.05,max_iter=200,max_leaf_nodes=31,min_samples_leaf=100,l2_regularization=1,early_stopping=False,random_state=42)
        model.fit(train[FEATURES].astype('float32'),train.cf)
        valid['site_pred']=np.where(valid.cosz>0,np.clip(model.predict(valid[FEATURES].astype('float32')),0,1.2)*valid.capacity_kw,0)
        valid.to_parquet(a.out/f'{fold.lower()}_site_predictions.parquet',index=False)
        valid=valid.merge(smp,on='ts',how='left',validate='many_to_one')
        if valid.smp.isna().any(): raise ValueError('missing SMP')
        g=valid.groupby(['group','ts'],as_index=False).agg(actual=('energy','sum'),weather=('site_pred','sum'),base=('prediction_kwh','sum'),smp=('smp','first'),k=('k','first'))
        ref=safe_parquet(a.reference_dir/f'{fold}_compare.parquet')
        ref=ref[['group','ts','actual','smp','k','2023']].rename(columns={'actual':'ref_actual','smp':'ref_smp','k':'ref_k','2023':'direct'})
        g=g.merge(ref,on=['group','ts'],how='outer',indicator=True,validate='one_to_one')
        if not g['_merge'].eq('both').all() or not np.allclose(g.actual,g.ref_actual,rtol=0,atol=1e-6) or not np.allclose(g.smp,g.ref_smp,rtol=0,atol=1e-9) or not (g.k==g.ref_k).all(): raise ValueError('comparison keys/actual/price mismatch')
        g['1314']=.4*g.base+.6*g.direct
        den=(g.smp*g.k*g.actual).sum()
        def wmae(col): return float((g.smp*g.k*(g[col]-g.actual).abs()).sum()/den)
        result['folds'][fold]={'train_rows':len(train),'train_max_ts':str(train.ts.max()),'sites':nsite,'site_hours':len(valid),'weather_missing_any':int(valid[WA+WB].isna().any(axis=1).sum()),'site_wmae':float((valid.smp*valid.k*(valid.site_pred-valid.energy).abs()).sum()/(valid.smp*valid.k*valid.energy).sum()),'group_wmae_weather':wmae('weather'),'group_wmae_1314':wmae('1314'),'penalty_weather':penalty(g,'weather'),'penalty_1314':penalty(g,'1314'),'comparison_keys':'PASS','actual_and_smp':'PASS'}
        print(fold,json.dumps(result['folds'][fold]),flush=True)
    (a.out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')

if __name__=='__main__': main()

