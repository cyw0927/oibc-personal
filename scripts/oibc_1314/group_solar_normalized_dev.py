"""Solar-normalized direct model on safe DEV only; see protocol before execution."""
import argparse, hashlib, json, platform
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from safe_reproduce import safe_parquet, CONFIGS, END

FOLDS={'DEV_JUL_2024':('2024-06-30 00:00',224),'DEV_OCT_2024':('2024-09-30 00:00',252)}
NON={'ts','group','energy','cap_avail','n_sites','group_cap_kw','cf','smp','k','cov'}

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    for name in ('features','oof','sites','smp','ref','out'): ap.add_argument(name,type=Path)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    ft=safe_parquet(a.features); oo=safe_parquet(a.oof); ref=safe_parquet(a.ref)
    if ft.ts.max()>END or ft.duplicated(['group','ts']).any(): raise ValueError('unsafe features or duplicate keys')
    sites=pd.read_csv(a.sites,usecols=['site_id','current_group','capacity_kw'])
    sites['group']=sites.current_group.map({'A':'G1','B':'G2'})
    if len(sites)!=253 or sites.site_id.duplicated().any() or sites.group.isna().any():raise ValueError('invalid sites')
    smp=pd.read_csv(a.smp,parse_dates=['ts']);smp.ts=pd.to_datetime(smp.ts,utc=True).dt.tz_convert('Asia/Seoul')
    if smp.ts.max()>END or smp.ts.duplicated().any() or smp.smp.isna().any() or (smp.smp<0).any():raise ValueError('unsafe market inputs')
    smp['k']=np.where(smp.ts.dt.hour.between(11,15),3,1)
    oo=oo[oo.fold.isin(FOLDS)].merge(sites[['site_id','group','capacity_kw']],on='site_id',validate='many_to_one')
    if oo.duplicated(['fold','site_id','ts']).any():raise ValueError('duplicate OOF keys')
    ag=oo.groupby(['fold','group','ts'],as_index=False).agg(actual=('actual_kwh','sum'),base=('prediction_kwh','sum'),cap=('capacity_kw','sum'),n=('site_id','nunique'))
    v=ag.merge(ref,on=['fold','group','ts'],how='outer',validate='one_to_one',indicator=True)
    if not v._merge.eq('both').all() or not np.allclose(v.actual,v.actual_kwh,rtol=0,atol=1e-6) or not np.allclose(v.base,v.l1_kwh,rtol=0,atol=1e-6) or not (v.n==v.n_sites).all():raise ValueError('OOF/reference mismatch')
    v=v.merge(smp.rename(columns={'smp':'check_smp','k':'check_k'}),on='ts',how='left',validate='many_to_one')
    if not np.allclose(v.smp,v.check_smp,rtol=0,atol=1e-9) or not (v.k==v.check_k).all():raise ValueError('market mismatch')
    ft=ft[ft.ts>=pd.Timestamp('2023-06-01 01:00',tz='Asia/Seoul')].merge(smp,on='ts',how='left',validate='many_to_one')
    ft['cov']=ft.cap_avail/ft.group_cap_kw; ft['cf']=(ft.energy/ft.cap_avail).clip(0,1.2)
    features=[c for c in ft.columns if c not in NON]
    result={'protocol':'docs/group_solar_normalized_20261105.md','metric':'README provisional D05 HOLD official_S=null','model':'pooled HGB L1 with solar-angle-normalized target, 3 configs','python':platform.python_version(),'sklearn':sklearn.__version__,'input_sha256':{x:sha(getattr(a,x)) for x in ('features','oof','sites','smp','ref')},'folds':{}}
    for fold,(cut,nsites) in FOLDS.items():
        vf=v[v.fold.eq(fold)].copy().reset_index(drop=True)
        if len(vf)!=1488 or oo[oo.fold.eq(fold)].site_id.nunique()!=nsites:raise ValueError('cohort changed')
        cutoff=pd.Timestamp(cut,tz='Asia/Seoul'); tr=ft[(ft.ts<=cutoff)&ft.energy.notna()&(ft['cov']>=.3)&(ft.cosz_max>0)].copy()
        if tr.smp.isna().any() or not tr.ts.max()<vf.ts.min():raise ValueError('unsafe training')
        va=vf[['group','ts']].merge(ft.drop(columns=['smp','k']),on=['group','ts'],how='left',validate='one_to_one')
        if va.cosz.isna().any():raise ValueError('missing validation inputs')
        scale=np.maximum(tr.cosz.to_numpy(),0.1)
        target=tr.cf.to_numpy()/scale
        weights=tr['cov'].to_numpy()*tr.k.to_numpy()*tr.smp.clip(lower=1).to_numpy()*scale
        preds=[]
        for cfg in CONFIGS:
            model=HistGradientBoostingRegressor(loss='absolute_error',l2_regularization=1,early_stopping=False,random_state=0,**cfg)
            model.fit(tr[features],target,sample_weight=weights)
            preds.append(model.predict(va[features]))
        val_scale=np.maximum(va.cosz.to_numpy(),0.1)
        cf=np.clip(np.mean(preds,axis=0)*val_scale,0,1)
        direct=np.where(va.cosz_max.to_numpy()>0,cf,0)*vf.cap.to_numpy()
        trained={'pooled':{'rows':len(tr),'train_max':str(tr.ts.max()),'solar_scale_floor':0.1}}
        vf['specialist_direct']=direct;vf['specialist_1314']=.4*vf.base+.6*vf.specialist_direct
        if not np.isfinite(vf.specialist_1314).all() or (vf.specialist_1314<0).any():raise ValueError('invalid prediction')
        def metric(col):
            numerator=(vf.smp*vf.k*abs(vf[col]-vf.actual)).sum()
            return {'penalty':float(.1*numerator),'wmae':float(numerator/(vf.smp*vf.k*vf.actual).sum())}
        result['folds'][fold]={'sites':nsites,'group_hours':len(vf),'start':str(vf.ts.min()),'end':str(vf.ts.max()),'train':trained,'reference':metric('prediction_kwh'),'new':metric('specialist_1314'),'reference_direct':metric('direct_kwh'),'specialist_direct':metric('specialist_direct'),'validation':'same keys, actual, SMP, site counts PASS'}
        vf[['fold','group','ts','actual','base','prediction_kwh','specialist_direct','specialist_1314','smp','k']].to_parquet(a.out/f'{fold.lower()}_group_specialist.parquet',index=False)
        print(fold,json.dumps(result['folds'][fold]),flush=True)
    (a.out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf8')

if __name__=='__main__':main()
