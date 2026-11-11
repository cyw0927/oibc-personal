"""Cloud/solar normalized group model, pre-HOLDOUT data only."""
import argparse,hashlib,json,platform
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from safe_reproduce import safe_parquet,CONFIGS,END

FOLDS={'APR_2024':('2024-03-31 00:00',720,174,'proxy'),
       'DEV_JUL_2024':('2024-06-30 00:00',744,224,'1314'),
       'AUG_2024':('2024-07-31 00:00',744,250,'proxy'),
       'DEV_OCT_2024':('2024-09-30 00:00',744,252,'1314')}
NON={'ts','group','energy','cap_avail','n_sites','group_cap_kw','cf','smp','k','cov'}

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    for key in ('features','oof','sites','smp','ref','proxy_dir','out'):ap.add_argument(key,type=Path)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    ft=safe_parquet(a.features); oo=safe_parquet(a.oof); ref=safe_parquet(a.ref)
    proxy_paths={f:a.proxy_dir/f'{f.lower()}_group_predictions.parquet' for f in ('APR_2024','AUG_2024')}
    proxies={f:safe_parquet(p) for f,p in proxy_paths.items()}
    if ft.ts.max()>END or ft.duplicated(['group','ts']).any():raise ValueError('unsafe features')
    sites=pd.read_csv(a.sites,usecols=['site_id','current_group','capacity_kw']);sites['group']=sites.current_group.map({'A':'G1','B':'G2'})
    if len(sites)!=253 or sites.site_id.duplicated().any() or sites.group.isna().any():raise ValueError('invalid sites')
    smp=pd.read_csv(a.smp,parse_dates=['ts']);smp.ts=pd.to_datetime(smp.ts,utc=True).dt.tz_convert('Asia/Seoul')
    if smp.ts.max()>END or smp.ts.duplicated().any() or smp.smp.isna().any() or (smp.smp<0).any():raise ValueError('invalid historical SMP')
    smp['k']=np.where(smp.ts.dt.hour.between(11,15),3,1)
    oo=oo[oo.fold.isin(['DEV_JUL_2024','DEV_OCT_2024'])].merge(sites[['site_id','group','capacity_kw']],on='site_id',validate='many_to_one')
    if oo.duplicated(['fold','site_id','ts']).any():raise ValueError('duplicate OOF')
    agg=oo.groupby(['fold','group','ts'],as_index=False).agg(actual=('actual_kwh','sum'),site=('prediction_kwh','sum'),cap=('capacity_kw','sum'),n=('site_id','nunique'))
    ag=agg.merge(ref,on=['fold','group','ts'],how='outer',validate='one_to_one',indicator=True)
    if not ag._merge.eq('both').all() or not np.allclose(ag.actual,ag.actual_kwh,rtol=0,atol=1e-6) or not np.allclose(ag.site,ag.l1_kwh,rtol=0,atol=1e-6) or not (ag.n==ag.n_sites).all():raise ValueError('OOF reference mismatch')
    ag=ag.rename(columns={'prediction_kwh':'baseline'})
    ft=ft[ft.ts>=pd.Timestamp('2023-06-01 01:00',tz='Asia/Seoul')].merge(smp,on='ts',how='left',validate='many_to_one')
    ft['cov']=ft.cap_avail/ft.group_cap_kw;ft['cf']=(ft.energy/ft.cap_avail).clip(0,1.2)
    features=[c for c in ft.columns if c not in NON]
    result={'protocol':'docs/group_cloud_normalized_20261105.md','metric':'README provisional D05 HOLD official_S=null','method':'cloud-solar normalized HGB L1, 3 configurations','python':platform.python_version(),'sklearn':sklearn.__version__,'inputs':{x:sha(getattr(a,x)) for x in ('features','oof','sites','smp','ref')},'proxy_inputs':{f:sha(p) for f,p in proxy_paths.items()},'folds':{}}
    for fold,(cut,hours,nsites,mode) in FOLDS.items():
        if mode=='1314':
            v=ag[ag.fold.eq(fold)].copy()
            if oo[oo.fold.eq(fold)].site_id.nunique()!=nsites:raise ValueError('site mismatch')
        else:
            v=proxies[fold].copy().rename(columns={'proxy_existing':'baseline'})
            if not np.allclose(v.baseline,.4*v.site+.6*v.coverage,rtol=0,atol=1e-6):raise ValueError('proxy baseline mismatch')
        v=v.sort_values(['group','ts']).reset_index(drop=True)
        if v.duplicated(['group','ts']).any() or len(v)!=2*hours or v.ts.nunique()!=hours:raise ValueError('bad validation keys')
        t=v[['ts','smp','k']].drop_duplicates('ts').merge(smp,on='ts',validate='one_to_one',suffixes=('','_check'))
        if len(t)!=hours or not np.allclose(t.smp,t.smp_check,rtol=0,atol=1e-9) or not (t.k==t.k_check).all():raise ValueError('evaluation SMP mismatch')
        cutoff=pd.Timestamp(cut,tz='Asia/Seoul');tr=ft[(ft.ts<=cutoff)&ft.energy.notna()&(ft['cov']>=.3)&(ft.cosz_max>0)].copy()
        va=v[['group','ts']].merge(ft.drop(columns=['smp','k']),on=['group','ts'],how='left',validate='one_to_one')
        if tr.empty or tr.smp.isna().any() or tr.cs_ab.isna().any() or va.cs_ab.isna().any() or not tr.ts.max()<v.ts.min():raise ValueError('training or input invalid')
        scale=np.maximum(tr.cs_ab.to_numpy(),.1); y=tr.cf.to_numpy()/scale
        weights=tr['cov'].to_numpy()*tr.k.to_numpy()*tr.smp.clip(lower=1).to_numpy()*scale
        predictions=[]
        for cfg in CONFIGS:
            model=HistGradientBoostingRegressor(loss='absolute_error',l2_regularization=1,early_stopping=False,random_state=0,**cfg)
            model.fit(tr[features],y,sample_weight=weights)
            predictions.append(model.predict(va[features]))
        cf=np.clip(np.mean(predictions,axis=0)*np.maximum(va.cs_ab.to_numpy(),.1),0,1)
        v['direct_cloud']=np.where(va.cosz_max.to_numpy()>0,cf,0)*v.cap.to_numpy()
        v['candidate']=.4*v.site+.6*v.direct_cloud
        if not np.isfinite(v.candidate).all() or (v.candidate<0).any():raise ValueError('invalid prediction')
        def metric(c):
            numerator=(v.smp*v.k*abs(v[c]-v.actual)).sum();return {'penalty':float(.1*numerator),'wmae':float(numerator/(v.smp*v.k*v.actual).sum())}
        result['folds'][fold]={'type':mode,'sites':nsites,'group_hours':len(v),'training_rows':len(tr),'train_max':str(tr.ts.max()),'start':str(v.ts.min()),'end':str(v.ts.max()),'group_cap_kw':v.groupby('group').cap.first().to_dict(),'baseline':metric('baseline'),'candidate':metric('candidate'),'same_keys_actual_smp_k':'PASS'}
        v[['group','ts','actual','site','baseline','direct_cloud','candidate','cap','smp','k']].to_parquet(a.out/f'{fold.lower()}_predictions.parquet',index=False)
        print(fold,json.dumps(result['folds'][fold]),flush=True)
    (a.out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf8')

if __name__=='__main__':main()
