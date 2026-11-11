"""Exploratory daily/hourly reconciliation. Inputs must be pre-lock safe files."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from safe_reproduce import safe_parquet

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser()
    for name in ('features','oof','sites','reference','out'): p.add_argument(name,type=Path)
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    ft=safe_parquet(a.features); oo=safe_parquet(a.oof); ref=safe_parquet(a.reference)
    sites=pd.read_csv(a.sites,usecols=['site_id','current_group','capacity_kw'])
    sites['group']=sites.current_group.map({'A':'G1','B':'G2'})
    if sites.site_id.duplicated().any() or sites.group.isna().any(): raise ValueError('invalid sites')
    if ft.ts.max()>pd.Timestamp('2024-11-01',tz='Asia/Seoul') or ft.duplicated(['group','ts']).any(): raise ValueError('unsafe features')
    ft['day']=(ft.ts-pd.Timedelta(hours=1)).dt.floor('D')
    # 24 forecast hours are summarized separately; no observed target in features.
    cols=['cosz','cosz_max','cs_a','cs_b','cs_ab','wa_cloud','wa_temp','wa_humidity','wa_rain','wb_temp','wb_rel_hum','wb_uv_idx','wb_precip_prob','wb_total_liq','wb_cloud','doy_sin','doy_cos','group_id']
    ft['coverage']=ft.cap_avail/ft.group_cap_kw
    ft['target']=ft.energy/ft.cap_avail
    good=ft.energy.notna() & (ft.coverage>=.3) & np.isfinite(ft.target)
    ft['good']=good
    df=ft.groupby(['group','day'],as_index=False).agg(**{c:(c,'mean') for c in cols}, target=('target','sum'), good=('good','sum'),hours=('ts','nunique'),last_ts=('ts','max'))
    df=df[df.hours.eq(24)].copy()
    oo=oo[oo.fold.isin(['DEV_JUL_2024','DEV_OCT_2024'])].merge(sites[['site_id','group','capacity_kw']],on='site_id',validate='many_to_one')
    if oo.duplicated(['fold','site_id','ts']).any(): raise ValueError('duplicate oof')
    caps=oo.groupby(['fold','group','ts'],as_index=False).capacity_kw.sum().rename(columns={'capacity_kw':'cap'})
    r=ref.merge(caps,on=['fold','group','ts'],validate='one_to_one')
    if len(r)!=len(ref) or r.duplicated(['fold','group','ts']).any():raise ValueError('reference keys')
    r['day']=(r.ts-pd.Timedelta(hours=1)).dt.floor('D')
    result={'description':'Daily model + proportional temporal reconciliation, exploratory, no alpha search','source':'Yagli et al 2019 sequential reconciliation; daily estimator is our adaptation','metric':'README provisional D05 HOLD official_S=null','inputs':{n:sha(getattr(a,n)) for n in ('features','oof','sites','reference')},'folds':{}}
    for fold in ['DEV_JUL_2024','DEV_OCT_2024']:
        test=r[r.fold.eq(fold)].copy(); start=test.ts.min(); end=test.ts.max()
        if len(test)!=1488: raise ValueError('incomplete test')
        train=df[(df.last_ts<start)&(df.good==24)&(df.day>=pd.Timestamp('2023-06-01',tz='Asia/Seoul'))].copy()
        val=test.groupby(['group','day'],as_index=False).agg(cap=('cap','first'),count=('ts','size'),first=('ts','min'))
        val=val.merge(df[['group','day']+cols],on=['group','day'],validate='one_to_one')
        if train.empty or not val['count'].eq(24).all() or len(val)!=62 or train.last_ts.max()>=start:raise ValueError('training boundary/day coverage')
        model=HistGradientBoostingRegressor(loss='absolute_error',learning_rate=.05,max_iter=200,max_leaf_nodes=15,min_samples_leaf=20,l2_regularization=1,early_stopping=False,random_state=42)
        model.fit(train[cols],train.target)
        val['daily_prediction']=np.maximum(model.predict(val[cols]),0)*val.cap
        test=test.merge(val[['group','day','daily_prediction']],on=['group','day'],validate='many_to_one')
        sums=test.groupby(['group','day']).prediction_kwh.transform('sum')
        # Preserve the original intraday shape and impose independent daily total.
        # Zero-total days stay zero; no future observed data used for redistribution.
        test['reconciled']=np.where(sums>1e-9,test.prediction_kwh*test.daily_prediction/sums,0)
        if not np.isfinite(test.reconciled).all() or (test.reconciled<0).any():raise ValueError('invalid prediction')
        def metric(x):return float((.1*test.smp*test.k*(test[x]-test.actual_kwh).abs()).sum())
        result['folds'][fold]={'train_days':len(train),'train_end':str(train.last_ts.max()),'test_start':str(start),'test_end':str(end),'baseline_penalty':metric('prediction_kwh'),'reconciled_penalty':metric('reconciled'),'reference_count':len(test),'zero_total_days':int(((sums<=1e-9).sum())/24),'validation':'keys and time boundaries PASS'}
        test[['fold','group','ts','actual_kwh','prediction_kwh','reconciled','smp','k']].to_parquet(a.out/f'{fold.lower()}_daily_reconciled.parquet',index=False)
        print(fold,json.dumps(result['folds'][fold]),flush=True)
    (a.out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf8')
if __name__=='__main__':main()
