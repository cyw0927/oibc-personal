"""Historical pre-lock derived features only; see winter reproduction protocol."""
import argparse,hashlib,json,platform
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from safe_reproduce import safe_parquet,CONFIGS,NON,LOCK

FOLDS={'DEV_JUL_2024':('2024-06-30 00:00',224,744),
       'DEV_OCT_2024':('2024-09-30 00:00',252,744),
       'DEV_JAN':('2024-12-31 00:00',253,720),
       'DEV_FEB':('2025-01-31 00:00',253,648)}

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    for x in ('features','oof','sites','smp','control','out'):ap.add_argument(x,type=Path)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    ft=safe_parquet(a.features);oof=safe_parquet(a.oof);control=safe_parquet(a.control)
    if ft.ts.max()>=LOCK or ft.duplicated(['group','ts']).any():raise ValueError('unsafe features')
    sites=pd.read_csv(a.sites,usecols=['site_id','current_group','capacity_kw']);sites['group']=sites.current_group.map({'A':'G1','B':'G2'})
    if len(sites)!=253 or sites.site_id.duplicated().any() or sites.group.isna().any():raise ValueError('sites invalid')
    smp=pd.read_csv(a.smp,parse_dates=['ts']);smp.ts=pd.to_datetime(smp.ts,utc=True).dt.tz_convert('Asia/Seoul').dt.as_unit('us')
    if smp.ts.max()>=LOCK or smp.ts.duplicated().any() or smp.smp.isna().any() or (smp.smp<0).any():raise ValueError('smp invalid')
    smp['k']=np.where(smp.ts.dt.hour.between(11,15),3,1)
    if oof.duplicated(['fold','site_id','ts']).any():raise ValueError('OOF duplicate')
    oof=oof[oof.fold.isin(FOLDS)].merge(sites[['site_id','group','capacity_kw']],on='site_id',validate='many_to_one')
    g=oof.groupby(['fold','group','ts'],as_index=False).agg(actual=('actual_kwh','sum'),base=('prediction_kwh','sum'),cap=('capacity_kw','sum'),count=('site_id','nunique'))
    g=g.merge(smp,on='ts',how='left',validate='many_to_one')
    if g.smp.isna().any():raise ValueError('missing group SMP')
    ft.ts=ft.ts.dt.as_unit('us');ft=ft.merge(smp,on='ts',how='left',validate='many_to_one')
    ft['cov']=ft.cap_avail/ft.group_cap_kw;ft['cf']=(ft.energy/ft.cap_avail).clip(0,1.2)
    feats=[c for c in ft.columns if c not in NON]
    result={'protocol':'docs/winter_1314_solar_reproduction_20261105.md','metric':'README provisional D05 HOLD official_S=null','caveat':'prior creation opened mixed original rowgroups; current inputs contain no locked timestamps. JAN/FEB original v2 predictions unavailable. Weather issuance not independently verified.','python':platform.python_version(),'sklearn':sklearn.__version__,'input_sha256':{k:sha(getattr(a,k)) for k in ('features','oof','sites','smp','control')},'folds':{}}
    for fold,(cut,nsites,hours) in FOLDS.items():
        val=g[g.fold.eq(fold)].sort_values(['group','ts']).reset_index(drop=True)
        if len(val)!=hours*2 or val.ts.nunique()!=hours or oof[oof.fold.eq(fold)].site_id.nunique()!=nsites:raise ValueError('cohort mismatch')
        tr=ft[(ft.ts<=pd.Timestamp(cut,tz='Asia/Seoul'))&ft.energy.notna()&(ft['cov']>=.3)&(ft.cosz_max>0)].copy()
        va=val[['group','ts']].merge(ft.drop(columns=['smp','k']),on=['group','ts'],how='left',validate='one_to_one')
        if tr.smp.isna().any() or va.cosz.isna().any() or not tr.ts.max()<val.ts.min():raise ValueError('unsafe train/validation')
        if nsites==253 and (not np.allclose(va.energy,val.actual,atol=1e-6,rtol=0) or not np.allclose(va.cap_avail,val.cap,atol=1e-6,rtol=0)):raise ValueError('winter actual/cap mismatch')
        baseweight=tr['cov'].to_numpy()*tr.k.to_numpy()*tr.smp.clip(lower=1).to_numpy()/tr.smp.clip(lower=1).mean()
        predictions=[]; solar=[];scale=np.maximum(tr.cosz.to_numpy(),.1);ys=tr.cf.to_numpy()/scale
        for cfg in CONFIGS:
            model=HistGradientBoostingRegressor(loss='absolute_error',l2_regularization=1,early_stopping=False,random_state=0,**cfg)
            model.fit(tr[feats],tr.cf,sample_weight=baseweight)
            predictions.append(np.clip(model.predict(va[feats]),0,1))
            if nsites==253:
                model2=HistGradientBoostingRegressor(loss='absolute_error',l2_regularization=1,early_stopping=False,random_state=0,**cfg)
                model2.fit(tr[feats],ys,sample_weight=baseweight*scale)
                solar.append(model2.predict(va[feats]))
        daytime=va.cosz_max.to_numpy()>0
        val['direct']=np.where(daytime,np.mean(predictions,axis=0),0)*val.cap.to_numpy()
        val['baseline']=.4*val.base+.6*val.direct
        info={'sites':nsites,'group_hours':len(val),'train_rows':len(tr),'train_max':str(tr.ts.max()),'start':str(val.ts.min()),'end':str(val.ts.max())}
        if nsites!=253:
            orig=control[control.fold.eq(fold)][['group','ts','direct_ens']]
            cmp=val.merge(orig,on=['group','ts'],how='outer',validate='one_to_one',indicator=True)
            if not cmp._merge.eq('both').all():raise ValueError('control key mismatch')
            info['control_direct_max_abs_diff_kwh']=float(abs(cmp.direct-cmp.direct_ens).max())
        else:
            val['solar_direct']=np.where(daytime,np.clip(np.mean(solar,axis=0)*np.maximum(va.cosz.to_numpy(),.1),0,1),0)*val.cap.to_numpy()
            val['solar_candidate']=.4*val.base+.6*val.solar_direct
            if not np.isfinite(val.solar_candidate).all() or (val.solar_candidate<0).any():raise ValueError('invalid forecast')
            def metric(c):
                err=val.smp*val.k*abs(val[c]-val.actual);return {'penalty':float(.1*err.sum()),'wmae':float(err.sum()/(val.smp*val.k*val.actual).sum())}
            info['baseline']=metric('baseline');info['solar']=metric('solar_candidate');info['actual_capacity']='PASS'
            val[['fold','group','ts','actual','base','cap','smp','k','direct','baseline','solar_direct','solar_candidate']].to_parquet(a.out/f'{fold.lower()}_predictions.parquet',index=False)
        result['folds'][fold]=info;print(fold,json.dumps(info),flush=True)
    (a.out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf8')

if __name__=='__main__':main()
