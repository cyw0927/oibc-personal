"""Small independent historical site/hour climatology baseline for JUL DEV.

It uses safe pre-separated official data, not forecasts or team OOF. Never commit outputs.
It is a feasibility benchmark, not a submission model or a model selection experiment.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from safe_reproduce import safe_parquet

def main():
    p=argparse.ArgumentParser()
    for arg in ('generation','sites','smp','output'): p.add_argument(arg,type=Path)
    a=p.parse_args()
    gen=safe_parquet(a.generation)
    sites=pd.read_csv(a.sites)[['site_id','current_group','capacity_kw']]
    if sites.site_id.duplicated().any() or len(sites)!=253: raise ValueError('invalid sites')
    smp=pd.read_csv(a.smp,parse_dates=['ts'])
    smp.ts=pd.to_datetime(smp.ts,utc=True).dt.tz_convert('Asia/Seoul')
    if smp.ts.max()>pd.Timestamp('2024-11-01',tz='Asia/Seoul'): raise ValueError('SMP not pre-cut')
    gen=gen.merge(sites,on='site_id',validate='many_to_one')
    gen['hour']=gen.ts.dt.hour
    start=pd.Timestamp('2023-06-01 01:00',tz='Asia/Seoul')
    cutoff=pd.Timestamp('2024-06-30 00:00',tz='Asia/Seoul')
    vstart=pd.Timestamp('2024-07-01 01:00',tz='Asia/Seoul')
    vend=pd.Timestamp('2024-08-01 00:00',tz='Asia/Seoul')
    train=gen[(gen.ts>=start)&(gen.ts<=cutoff)].copy()
    val=gen[(gen.ts>=vstart)&(gen.ts<=vend)].copy()
    cohort=val.groupby('site_id').ts.nunique(); cohort=cohort[cohort==744].index
    val=val[val.site_id.isin(cohort)].copy()
    if val.duplicated(['site_id','ts']).any(): raise ValueError('duplicate actual keys')
    train['cf']=(train.energy/train.capacity_kw).clip(0,1.2)
    bysite=train.groupby(['site_id','hour'],as_index=False).cf.median().rename(columns={'cf':'cf_site'})
    byhour=train.groupby('hour',as_index=False).cf.median().rename(columns={'cf':'cf_fallback'})
    val=val.merge(bysite,on=['site_id','hour'],how='left',validate='many_to_one').merge(byhour,on='hour',validate='many_to_one')
    val['prediction']=val.cf_site.fillna(val.cf_fallback)*val.capacity_kw
    val=val.merge(smp,on='ts',how='left',validate='many_to_one')
    if val[['prediction','smp','energy']].isna().any().any(): raise ValueError('missing inputs')
    val['k']=np.where(val.hour.between(11,15),3,1)
    grouped=val.groupby(['current_group','ts'],as_index=False).agg(actual=('energy','sum'),prediction=('prediction','sum'),smp=('smp','first'),k=('k','first'))
    penalty=float((.1*grouped.smp*grouped.k*(grouped.actual-grouped.prediction).abs()).sum())
    weighted_error=float((grouped.smp*grouped.k*(grouped.actual-grouped.prediction).abs()).sum()/(grouped.smp*grouped.k*grouped.actual).sum())
    result={'model':'independent site-hour median climatology','train_start':str(train.ts.min()),'train_end':str(train.ts.max()),'validation':'DEV_JUL_2024','site_count':int(len(cohort)),'rows':len(val),'fallback_rows':int(val.cf_site.isna().sum()),'group_wmae':weighted_error,'penalty_won_provisional':penalty,'caveat':'README provisional, D05 HOLD; cohort derived independently; no team OOF, no weather forecast'}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps(result,indent=2))

if __name__=='__main__': main()
