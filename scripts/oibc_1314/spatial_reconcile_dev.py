"""Nonnegative weighted spatial reconciliation to the existing #1314 group OOF."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from safe_reproduce import safe_parquet

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def reconcile(pred, capacity, target):
    """Weighted L2 projection: minimize sum((x-pred)^2/capacity^2)."""
    pred=np.asarray(pred,dtype=float); capacity=np.asarray(capacity,dtype=float)
    if not np.isfinite(pred).all() or not np.isfinite(capacity).all() or np.any(capacity<=0) or not np.isfinite(target) or target<0:
        raise ValueError('invalid projection inputs')
    if target==0: return np.zeros_like(pred)
    weights=capacity**2
    # The active set only shrinks: x_i = max(0, pred_i + lambda*capacity_i^2).
    active=np.ones(len(pred),dtype=bool)
    while True:
        if not active.any(): raise ValueError('infeasible active set')
        lam=(target-pred[active].sum())/weights[active].sum()
        bad=active & ((pred+lam*weights)<0)
        if not bad.any(): break
        active[bad]=False
    x=np.where(active,pred+lam*weights,0.0)
    if np.min(x)<-1e-8 or not np.isclose(x.sum(),target,rtol=1e-11,atol=1e-7):raise ValueError('projection error')
    return x

def main():
    ap=argparse.ArgumentParser()
    for arg in ['oof','sites','group_oof','out']:ap.add_argument(arg,type=Path)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    oof=safe_parquet(a.oof); ref=safe_parquet(a.group_oof)
    sites=pd.read_csv(a.sites,usecols=['site_id','current_group','capacity_kw'])
    sites['group']=sites.current_group.map({'A':'G1','B':'G2'})
    if len(sites)!=253 or sites.site_id.duplicated().any() or sites.group.isna().any() or (sites.capacity_kw<=0).any():raise ValueError('sites invalid')
    oof=oof[oof.fold.isin(['DEV_JUL_2024','DEV_OCT_2024'])].copy()
    if oof.duplicated(['fold','site_id','ts']).any() or ref.duplicated(['fold','group','ts']).any():raise ValueError('duplicate keys')
    oof=oof.merge(sites[['site_id','group','capacity_kw']],on='site_id',validate='many_to_one')
    agg=oof.groupby(['fold','group','ts'],as_index=False).agg(check_actual=('actual_kwh','sum'),check_l1=('prediction_kwh','sum'),n=('site_id','nunique'))
    check=ref.merge(agg,on=['fold','group','ts'],how='outer',validate='one_to_one',indicator=True)
    if not check._merge.eq('both').all() or not np.allclose(check.actual_kwh,check.check_actual,rtol=0,atol=1e-6) or not np.allclose(check.l1_kwh,check.check_l1,rtol=0,atol=1e-6) or not (check.n==check.n_sites).all():raise ValueError('reference mismatch')
    oof=oof.merge(ref[['fold','group','ts','prediction_kwh','smp','k']].rename(columns={'prediction_kwh':'target'}),on=['fold','group','ts'],validate='many_to_one')
    oof['reconciled']=np.nan
    for _,g in oof.groupby(['fold','group','ts'],sort=False):
        oof.loc[g.index,'reconciled']=reconcile(g.prediction_kwh.to_numpy(),g.capacity_kw.to_numpy(),float(g.target.iloc[0]))
    test=oof.groupby(['fold','group','ts'],as_index=False).agg(actual=('actual_kwh','sum'),prior=('prediction_kwh','sum'),adjusted=('reconciled','sum'),target=('target','first'),smp=('smp','first'),k=('k','first'))
    if not np.allclose(test.adjusted,test.target,rtol=0,atol=1e-6):raise ValueError('coherence failed')
    result={'method':'fixed #1314 group target, nonnegative capacity-squared weighted L2 projection','metric':'README provisional D05 HOLD official_S=null','protocol':'docs/spatial_reconciliation_20261105.md','input_sha256':{k:sha(getattr(a,k)) for k in ('oof','sites','group_oof')},'folds':{}}
    for fold in ['DEV_JUL_2024','DEV_OCT_2024']:
        s=oof[oof.fold.eq(fold)].copy();g=test[test.fold.eq(fold)].copy()
        weighted=s.smp*s.k; den=(weighted*s.actual_kwh).sum()
        if den<=0 or s.smp.isna().any() or s.reconciled.isna().any():raise ValueError('invalid data')
        result['folds'][fold]={'sites':int(s.site_id.nunique()),'site_hours':len(s),'group_hours':len(g),'group_penalty_1314':float((.1*g.smp*g.k*abs(g.target-g.actual)).sum()),'group_penalty_reconciled':float((.1*g.smp*g.k*abs(g.adjusted-g.actual)).sum()),'site_wmae_original':float((weighted*abs(s.prediction_kwh-s.actual_kwh)).sum()/den),'site_wmae_reconciled':float((weighted*abs(s.reconciled-s.actual_kwh)).sum()/den),'max_group_error_kwh':float(abs(g.adjusted-g.target).max()),'negative_site_values':int((s.reconciled<0).sum())}
        s[['fold','group','site_id','ts','actual_kwh','prediction_kwh','reconciled','capacity_kw']].to_parquet(a.out/f'{fold.lower()}_site_reconciled.parquet',index=False)
        print(fold,json.dumps(result['folds'][fold]),flush=True)
    (a.out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf8')

if __name__=='__main__':main()
