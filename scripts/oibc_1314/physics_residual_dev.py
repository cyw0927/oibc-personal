"""Two-stage physics prior plus learned residual; chronological DEV-only exploration."""
import argparse, hashlib, json, platform
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from safe_reproduce import safe_parquet, NON, LOCK, CONFIGS
from winter_solar_reproduce import FOLDS

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    for arg in ('features','oof','sites','smp','reference_dir','out'): ap.add_argument(arg,type=Path)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    ft=safe_parquet(a.features); oof=safe_parquet(a.oof)
    if ft.ts.max()>=LOCK or ft.duplicated(['group','ts']).any(): raise ValueError('unsafe features')
    sites=pd.read_csv(a.sites,usecols=['site_id','current_group','capacity_kw'])
    sites['group']=sites.current_group.map({'A':'G1','B':'G2'})
    if len(sites)!=253 or sites.site_id.duplicated().any() or sites.group.isna().any(): raise ValueError('sites invalid')
    smp=pd.read_csv(a.smp,parse_dates=['ts']); smp.ts=pd.to_datetime(smp.ts,utc=True).dt.tz_convert('Asia/Seoul').dt.as_unit('us')
    if smp.ts.max()>=LOCK or smp.ts.duplicated().any() or smp.smp.isna().any() or (smp.smp<0).any(): raise ValueError('bad smp')
    smp['k']=np.where(smp.ts.dt.hour.between(11,15),3,1)
    if oof.duplicated(['fold','site_id','ts']).any(): raise ValueError('duplicate OOF')
    oof=oof[oof.fold.isin(FOLDS)].merge(sites[['site_id','group','capacity_kw']],on='site_id',validate='many_to_one')
    g=oof.groupby(['fold','group','ts'],as_index=False).agg(actual=('actual_kwh','sum'),base=('prediction_kwh','sum'),cap=('capacity_kw','sum'),count=('site_id','nunique'))
    g=g.merge(smp,on='ts',how='left',validate='many_to_one')
    if g.smp.isna().any(): raise ValueError('missing SMP')
    ft.ts=ft.ts.dt.as_unit('us'); ft=ft.merge(smp,on='ts',how='left',validate='many_to_one')
    ft['cov']=ft.cap_avail/ft.group_cap_kw; ft['cf']=(ft.energy/ft.cap_avail).clip(0,1.2)
    feats=[c for c in ft.columns if c not in NON]
    result={'metric':'README provisional D05 HOLD official_S=null','model':'weighted Ridge clear-sky/solar prior + HGB L1 residual, 3 configs, equal mean', 'caveat':'reused DEV; historical group features may have been made by opening mixed locked row groups; current input metadata prelock; JAN/FEB reference reconstructed not original; weather arrival unknown', 'python':platform.python_version(),'sklearn':sklearn.__version__,'inputs':{k:sha(getattr(a,k)) for k in ('features','oof','sites','smp')},'folds':{}}
    for fold,(cut,nsites,hours) in FOLDS.items():
        v=g[g.fold.eq(fold)].sort_values(['group','ts']).reset_index(drop=True)
        if len(v)!=hours*2 or v.ts.nunique()!=hours or oof[oof.fold.eq(fold)].site_id.nunique()!=nsites: raise ValueError('cohort')
        tr=ft[(ft.ts<=pd.Timestamp(cut,tz='Asia/Seoul')) & ft.energy.notna() & (ft['cov']>=.3) & (ft.cosz_max>0)].copy()
        va=v[['group','ts']].merge(ft,on=['group','ts'],how='left',validate='one_to_one')
        if tr.smp.isna().any() or va.cosz.isna().any() or not tr.ts.max()<v.ts.min(): raise ValueError('training validation or missing values')
        if nsites==253 and (not np.allclose(va.energy,v.actual,atol=1e-6,rtol=0) or not np.allclose(va.cap_avail,v.cap,atol=1e-6,rtol=0)): raise ValueError('actual capacity mismatch')
        w=tr['cov'].to_numpy()*tr.k.to_numpy()*tr.smp.clip(lower=1).to_numpy()/tr.smp.clip(lower=1).mean()
        prior_feats=['cs_ab','cosz']
        imp=SimpleImputer(strategy='median',keep_empty_features=True)
        px=imp.fit_transform(tr[prior_feats]); pv=imp.transform(va[prior_feats])
        prior=Ridge(alpha=1.0).fit(px,tr.cf,sample_weight=w)
        residual=tr.cf.to_numpy()-prior.predict(px); preds=[]
        for cfg in CONFIGS:
            m=HistGradientBoostingRegressor(loss='absolute_error',l2_regularization=1.0,early_stopping=False,random_state=0,**cfg)
            m.fit(tr[feats],residual,sample_weight=w)
            preds.append(m.predict(va[feats]))
        cf=np.clip(prior.predict(pv)+np.mean(preds,axis=0),0,1)
        v['candidate']=.4*v.base+.6*np.where(va.cosz_max.to_numpy()>0,cf,0)*v.cap.to_numpy()
        rp=a.reference_dir/f'{fold.lower()}_analog.parquet'; ref=safe_parquet(rp)
        cols=['group','ts','actual','base','cap','smp','k','reference']
        v=v.merge(ref[cols],on=['group','ts'],validate='one_to_one',suffixes=('','_r'),how='outer',indicator=True)
        if not v._merge.eq('both').all(): raise ValueError('reference keys')
        for c in ['actual','base','cap','smp','k']:
            if not np.allclose(v[c],v[c+'_r'],rtol=0,atol=1e-6): raise ValueError('reference mismatch '+c)
        if not np.isfinite(v.candidate).all() or (v.candidate<0).any(): raise ValueError('invalid candidate')
        def metric(c):
            err=v.smp*v.k*abs(v[c]-v.actual); return {'penalty':float(.1*err.sum()),'wmae':float(err.sum()/(v.smp*v.k*v.actual).sum())}
        info={'sites':nsites,'group_hours':len(v),'train_rows':len(tr),'train_max':str(tr.ts.max()),'start':str(v.ts.min()),'reference_sha256':sha(rp),'reference':metric('reference'),'candidate':metric('candidate')}
        result['folds'][fold]=info; print(fold,json.dumps(info),flush=True)
        v[['fold','group','ts','actual','base','cap','smp','k','reference','candidate']].to_parquet(a.out/f'{fold.lower()}_predictions.parquet',index=False)
    (a.out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf8')
if __name__=='__main__': main()
