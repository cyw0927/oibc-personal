"""Reproduce JUL/OCT group-direct runs using pre-separated pre-HOLDOUT inputs.

Usage: python safe_reproduce.py FEATURES OOF SITES SMP OUTPUT
Only files whose parquet metadata proves every row group ends before the lock are read.
SMP must itself be a pre-cut file. Outputs contain competition derivatives: keep outside Git.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingRegressor

LOCK = pd.Timestamp('2025-03-01 01:00', tz='Asia/Seoul')
END = pd.Timestamp('2024-11-01 00:00', tz='Asia/Seoul')
FOLDS = {'DEV_JUL_2024': '2024-06-30 00:00', 'DEV_OCT_2024': '2024-09-30 00:00'}
CONFIGS = [dict(learning_rate=.05, max_iter=600, max_leaf_nodes=31, min_samples_leaf=40),
           dict(learning_rate=.03, max_iter=900, max_leaf_nodes=15, min_samples_leaf=80),
           dict(learning_rate=.05, max_iter=500, max_leaf_nodes=63, min_samples_leaf=30)]
NON = {'ts', 'group', 'energy', 'cap_avail', 'n_sites', 'group_cap_kw', 'cf', 'smp', 'k', 'cov'}

def safe_parquet(path):
    pf = pq.ParquetFile(path)
    for i in range(pf.num_row_groups):
        row = pf.metadata.row_group(i)
        ts = next((row.column(j) for j in range(row.num_columns) if row.column(j).path_in_schema == 'ts'), None)
        if ts is None or not ts.statistics or not ts.statistics.has_min_max:
            raise ValueError(f'{path}: missing timestamp statistics')
        if pd.Timestamp(ts.statistics.max) >= LOCK:
            raise ValueError(f'{path}: row group intersects locked period')
    return pd.read_parquet(path)

def main():
    p = argparse.ArgumentParser()
    for x in ('features', 'oof', 'sites', 'smp', 'output'):
        p.add_argument(x, type=Path)
    a = p.parse_args()
    ft, oof = safe_parquet(a.features), safe_parquet(a.oof)
    if ft.ts.max() > END:
        raise ValueError('Features extend beyond the safe 2024-11-01 boundary')
    smp = pd.read_csv(a.smp, parse_dates=['ts'])
    if pd.to_datetime(smp.ts, utc=True).max() > END.tz_convert('UTC'):
        raise ValueError('SMP input must be pre-cut before use')
    smp.ts = pd.to_datetime(smp.ts, utc=True).dt.tz_convert('Asia/Seoul').dt.as_unit('us')
    smp['k'] = np.where(smp.ts.dt.hour.between(11,15),3,1)
    sites = pd.read_csv(a.sites)
    sites['group'] = sites.current_group.map({'A':'G1','B':'G2'})
    if len(sites)!=253 or sites.site_id.duplicated().any() or sites.group.isna().any():
        raise ValueError('Invalid site mapping')
    oof = oof[oof.fold.isin(FOLDS)].copy()
    if oof.duplicated(['fold','site_id','ts']).any():
        raise ValueError('Duplicate OOF keys')
    oof = oof.merge(sites[['site_id','group','capacity_kw']], on='site_id', validate='many_to_one')
    oof = oof.merge(smp, on='ts', how='left', validate='many_to_one')
    if oof.smp.isna().any(): raise ValueError('Missing market price')
    gh = oof.groupby(['fold','group','ts'], as_index=False).agg(actual=('actual_kwh','sum'), base=('prediction_kwh','sum'), cap=('capacity_kw','sum'), smp=('smp','first'), k=('k','first'))
    ft.ts = ft.ts.dt.as_unit('us')
    ft = ft[ft.ts >= pd.Timestamp('2023-06-01 01:00',tz='Asia/Seoul')].merge(smp,on='ts',how='left',validate='many_to_one')
    ft['cov'] = ft.cap_avail / ft.group_cap_kw
    ft['cf'] = (ft.energy / ft.cap_avail).clip(0,1.2)
    features = [c for c in ft.columns if c not in NON]
    results = {}
    for fold, cutoff in FOLDS.items():
        v = gh[gh.fold.eq(fold)].copy()
        tr = ft[(ft.ts <= pd.Timestamp(cutoff,tz='Asia/Seoul')) & ft.energy.notna() & (ft['cov']>=.3) & (ft.cosz_max>0)]
        va = v[['group','ts']].merge(ft,on=['group','ts'],how='left',validate='one_to_one')
        if va.cosz.isna().any() or tr.smp.isna().any(): raise ValueError('Missing training/evaluation input')
        weights = tr['cov'].values * tr.k.values * np.clip(tr.smp.values,1,None) / tr.smp.clip(lower=1).mean()
        predictions=[]
        for config in CONFIGS:
            m=HistGradientBoostingRegressor(loss='absolute_error',l2_regularization=1,early_stopping=False,random_state=0,**config)
            m.fit(tr[features],tr.cf,sample_weight=weights)
            predictions.append(np.clip(m.predict(va[features]),0,1))
        v['direct']=np.where(va.cosz_max.values>0,np.mean(predictions,axis=0),0)*v.cap.values
        v['blend']=.4*v.base+.6*v.direct
        penalty=lambda col: float((.1*v.smp*v.k*(v[col]-v.actual).abs()).sum())
        results[fold]={'training_rows':len(tr),'group_hours':len(v),'sites':int(oof[oof.fold.eq(fold)].site_id.nunique()),'base_penalty':penalty('base'),'direct_penalty':penalty('direct'),'blend_penalty':penalty('blend')}
        print(fold,results[fold],flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps({'settlement':'README provisional; D05 HOLD','folds':results},indent=2),encoding='utf-8')

if __name__ == '__main__': main()
