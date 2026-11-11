"""Pre-registered APR/AUG group-penalty experiment; never reads locked inputs.

See docs/group_penalty_experiment_20261105.md. Outputs are data derivatives: outside Git.
The independently trained site model makes the #1314 comparison a PROXY, not #1314.
"""
import argparse, hashlib, json, platform
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from safe_reproduce import safe_parquet
from weather_site_dev import read_weather, solar_cosz, FEATURES, WA, WB, START, END, KST

FOLDS={'APR_2024':('2024-03-31 00:00','2024-04-01 01:00','2024-05-01 00:00',720),
       'AUG_2024':('2024-07-31 00:00','2024-08-01 01:00','2024-09-01 00:00',744)}
CONFIGS=[dict(learning_rate=.05,max_iter=600,max_leaf_nodes=31,min_samples_leaf=40),
         dict(learning_rate=.03,max_iter=900,max_leaf_nodes=15,min_samples_leaf=80),
         dict(learning_rate=.05,max_iter=500,max_leaf_nodes=63,min_samples_leaf=30)]
NON={'ts','group','energy','cap_avail','n_sites','group_cap_kw','cf','smp','k','cov'}

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''): h.update(chunk)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    for name in ('generation','wa','wb','sites','smp','group_features','out'): ap.add_argument(name,type=Path)
    a=ap.parse_args()
    sites=pd.read_csv(a.sites)[['site_id','current_group','capacity_kw','lat','lon','wa_station','wb_station']]
    if len(sites)!=253 or sites.site_id.duplicated().any(): raise ValueError('site master invalid')
    sites['group']=sites.current_group.map({'A':'G1','B':'G2'})
    if sites.group.isna().any(): raise ValueError('group mapping invalid')
    gen=safe_parquet(a.generation)
    if gen.ts.max()>END or gen.duplicated(['site_id','ts']).any(): raise ValueError('unsafe or duplicate generation')
    gen=gen[(gen.ts>=START)&(gen.ts<=END)].merge(sites,on='site_id',validate='many_to_one')
    wa=read_weather(a.wa,'wa_station',WA);wb=read_weather(a.wb,'wb_station',WB)
    gen=gen.merge(wa,on=['wa_station','ts'],how='left',validate='many_to_one').merge(wb,on=['wb_station','ts'],how='left',validate='many_to_one')
    gen['cosz']=solar_cosz(gen.ts,gen.lat.to_numpy(),gen.lon.to_numpy()); gen['hour']=gen.ts.dt.hour
    doy=gen.ts.dt.dayofyear;gen['doy_sin']=np.sin(2*np.pi*doy/365.25);gen['doy_cos']=np.cos(2*np.pi*doy/365.25)
    gen['cf']=(gen.energy/gen.capacity_kw).clip(0,1.2)
    smp=pd.read_csv(a.smp,parse_dates=['ts']);smp.ts=pd.to_datetime(smp.ts,utc=True).dt.tz_convert(KST)
    if smp.ts.max()>END or smp.ts.duplicated().any() or (smp.smp<0).any(): raise ValueError('SMP invalid')
    smp['k']=np.where(smp.ts.dt.hour.between(11,15),3,1)
    ft=safe_parquet(a.group_features)
    if ft.ts.max()>END: raise ValueError('unsafe features')
    ft=ft[ft.ts>=START].merge(smp,on='ts',how='left',validate='many_to_one')
    ft['cov']=ft.cap_avail/ft.group_cap_kw;ft['cf']=(ft.energy/ft.cap_avail).clip(0,1.2)
    feats=[c for c in ft.columns if c not in NON]
    result={'protocol':'docs/group_penalty_experiment_20261105.md','settlement':'README provisional; D05 HOLD; official_S=null','comparison':'proxy using independently trained site baseline; NOT exact #1314','seed':42,'python':platform.python_version(),'sklearn':sklearn.__version__,'input_sha256':{k:sha(getattr(a,k)) for k in ('generation','wa','wb','sites','smp','group_features')},'folds':{}}
    a.out.mkdir(parents=True,exist_ok=True)
    for fold,(cut,vstart,vend,hours) in FOLDS.items():
        cut=pd.Timestamp(cut,tz=KST); vstart=pd.Timestamp(vstart,tz=KST); vend=pd.Timestamp(vend,tz=KST)
        tr=gen[(gen.ts<=cut)&(gen.cosz>0)&gen.energy.notna()].copy()
        val=gen[(gen.ts>=vstart)&(gen.ts<=vend)].copy()
        coverage=val.groupby('site_id').ts.nunique();cohort=coverage[coverage==hours].index
        val=val[val.site_id.isin(cohort)].copy()
        if val.duplicated(['site_id','ts']).any() or len(val)!=len(cohort)*hours or val.ts.nunique()!=hours: raise ValueError('invalid cohort')
        if val[WA+WB].isna().all(axis=1).any(): raise ValueError('forecast missing')
        if len(tr)>250000:tr=tr.sample(n=250000,random_state=42)
        tr=tr.merge(smp,on='ts',how='left',validate='many_to_one')
        if tr.smp.isna().any():raise ValueError('missing train SMP')
        site=HistGradientBoostingRegressor(loss='absolute_error',learning_rate=.05,max_iter=200,max_leaf_nodes=31,min_samples_leaf=100,l2_regularization=1,early_stopping=False,random_state=42)
        site.fit(tr[FEATURES].astype('float32'),tr.cf,sample_weight=tr.smp*tr.k*tr.capacity_kw)
        val['site_pred']=np.where(val.cosz>0,np.clip(site.predict(val[FEATURES].astype('float32')),0,1.2)*val.capacity_kw,0)
        g=val.groupby(['group','ts'],as_index=False).agg(actual=('energy','sum'),site=('site_pred','sum'),cap=('capacity_kw','sum'))
        g=g.merge(smp,on='ts',how='left',validate='many_to_one')
        gt=ft[(ft.ts<=cut)&ft.energy.notna()&(ft['cov']>=.3)&(ft.cosz_max>0)].copy()
        gv=g[['group','ts']].merge(ft.drop(columns=['smp','k']),on=['group','ts'],how='left',validate='one_to_one')
        if len(g)!=2*hours or g.smp.isna().any() or gt.smp.isna().any() or gv.cosz.isna().any():raise ValueError('group inputs invalid')
        for variant in ('coverage','energy'):
            size=gt['cov'] if variant=='coverage' else gt['cap_avail']/gt['group_cap_kw'].mean()
            weight=size*gt.smp.clip(lower=1)*gt.k/gt.smp.clip(lower=1).mean()
            predictions=[]
            for cfg in CONFIGS:
                m=HistGradientBoostingRegressor(loss='absolute_error',l2_regularization=1,early_stopping=False,random_state=0,**cfg)
                m.fit(gt[feats],gt.cf,sample_weight=weight)
                predictions.append(np.clip(m.predict(gv[feats]),0,1))
            g[variant]=np.where(gv.cosz_max.to_numpy()>0,np.mean(predictions,axis=0),0)*g.cap
        g['proxy_existing']=.4*g.site+.6*g.coverage
        g['proxy_new']=.4*g.site+.6*g.energy
        den=(g.smp*g.k*g.actual).sum()
        def metric(col):
            e=(g[col]-g.actual).abs();return {'penalty':float((.1*g.smp*g.k*e).sum()),'wmae':float((g.smp*g.k*e).sum()/den)}
        result['folds'][fold]={'cutoff':str(cut),'site_train_max':str(tr.ts.max()),'group_train_max':str(gt.ts.max()),'validation_start':str(g.ts.min()),'validation_end':str(g.ts.max()),'cohort_sites':len(cohort),'group_site_counts':val.groupby('group').site_id.nunique().to_dict(),'site_rows':len(val),'group_hours':len(g),'site_train_rows':len(tr),'group_train_rows':len(gt),'standalone_new':metric('energy'),'group_old':metric('coverage'),'proxy_existing':metric('proxy_existing'),'proxy_new':metric('proxy_new')}
        print(fold,json.dumps(result['folds'][fold]),flush=True)
        g.to_parquet(a.out/f'{fold.lower()}_group_predictions.parquet',index=False)
    (a.out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')

if __name__=='__main__':main()
