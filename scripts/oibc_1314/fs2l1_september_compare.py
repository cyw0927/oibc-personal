"""Reconstructed prelock SEPTEMBER comparison; see preregistered protocol."""
import gc,hashlib,json,platform,time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from safe_reproduce import safe_parquet,CONFIGS,NON
from fs2l1_safe_control import ROOT,DATA,CONTRACTS,LOCK,lightgbm,safe_metadata,sha,load_contracts,OfficialData,training_rows,validation_grid,build_nonlinear

def main():
    control=ROOT/'personal_safe'/'fs2l1_control_20261106'/'results.json'
    if not control.is_file():raise ValueError('JUL control missing')
    check=json.loads(control.read_text(encoding='utf8'))
    if check['max_abs_diff_kwh']>1e-6:raise ValueError('JUL reproduction not verified; stop')
    inputs=[DATA/'train'/'generation.parquet',DATA/'train'/'forecast_wa.parquet',DATA/'train'/'forecast_wb.parquet']
    features=ROOT/'safe_features'/'group_features_train.parquet';smp_path=ROOT/'personal_safe'/'winter_20261105'/'smp_to_20250228.csv'
    for p in inputs+[features]:safe_metadata(p)
    c=load_contracts(CONTRACTS);data=OfficialData(DATA,c)
    start=pd.Timestamp('2024-09-01 01:00',tz='Asia/Seoul');end=pd.Timestamp('2024-10-01 00:00',tz='Asia/Seoul');cut=pd.Timestamp('2024-08-31 00:00',tz='Asia/Seoul')
    if end>=LOCK:raise ValueError('locked date')
    ix=pd.date_range(start,end,freq='h');gen=data.generation();v=gen[gen.ts.between(start,end)]
    if v.duplicated(['site_id','ts']).any():raise ValueError('duplicate')
    counts=v.groupby('site_id').ts.nunique();ids=tuple(sorted(counts[counts.eq(len(ix))].index));train=training_rows(data,pd.Timestamp('2023-06-01 01:00',tz='Asia/Seoul'),cut);grid=validation_grid(data,ids,ix)
    if not ids or train.ts.max()>cut or len(grid)!=len(ids)*len(ix):raise ValueError('grid')
    print('sites',len(ids),'training',len(train),'validation',len(grid),flush=True)
    predictions=[]
    for seed in (11,22,33):
        t=time.time(); model=build_nonlinear('lgbm_cf',{'seed':seed,'n_jobs':2,'feature_set':'FS2','loss':'L1'},data);model.fit(train)
        pred=model.predict(grid)['prediction_kwh'].to_numpy(float)
        if not np.isfinite(pred).all() or (pred<0).any():raise ValueError('bad forecast')
        predictions.append(pred);del model;gc.collect();print('seed',seed,'seconds',round(time.time()-t,1),flush=True)
    grid=grid.copy();grid['base']=np.mean(predictions,axis=0);grid['group']=grid.current_group.map({'A':'G1','B':'G2'})
    if grid.group.isna().any():raise ValueError('unknown group')
    val=grid.groupby(['group','ts'],as_index=False).agg(actual=('energy','sum'),base=('base','sum'),cap=('capacity_kw','sum'),site_count=('site_id','nunique')).sort_values(['group','ts']).reset_index(drop=True)
    smp=pd.read_csv(smp_path,parse_dates=['ts']);smp.ts=pd.to_datetime(smp.ts,utc=True).dt.tz_convert('Asia/Seoul').dt.as_unit('us')
    if smp.ts.max()>=LOCK or smp.ts.duplicated().any():raise ValueError('bad smp')
    smp['k']=np.where(smp.ts.dt.hour.between(11,15),3,1);val=val.merge(smp,on='ts',validate='many_to_one')
    ft=safe_parquet(features);ft.ts=ft.ts.dt.as_unit('us');ft=ft.merge(smp,on='ts',how='left',validate='many_to_one');ft['cov']=ft.cap_avail/ft.group_cap_kw;ft['cf']=(ft.energy/ft.cap_avail).clip(0,1.2)
    feats=[x for x in ft.columns if x not in NON];tr=ft[(ft.ts<=cut)&ft.energy.notna()&(ft['cov']>=.3)&(ft.cosz_max>0)].copy();va=val[['group','ts']].merge(ft.drop(columns=['smp','k']),on=['group','ts'],how='left',validate='one_to_one')
    if tr.ts.max()>cut or tr.smp.isna().any() or va.cosz.isna().any():raise ValueError('group data cutoff/missing')
    w=tr['cov'].to_numpy()*tr.k.to_numpy()*tr.smp.clip(lower=1).to_numpy()/tr.smp.clip(lower=1).mean()
    imp=SimpleImputer(strategy='median',keep_empty_features=True);px=imp.fit_transform(tr[['cs_ab','cosz']]);pv=imp.transform(va[['cs_ab','cosz']]);prior=Ridge(alpha=1).fit(px,tr.cf,sample_weight=w);residual=tr.cf.to_numpy()-prior.predict(px)
    baseline=[];candidate=[]
    for config in CONFIGS:
        a=HistGradientBoostingRegressor(loss='absolute_error',l2_regularization=1,early_stopping=False,random_state=0,**config);a.fit(tr[feats],tr.cf,sample_weight=w);baseline.append(np.clip(a.predict(va[feats]),0,1));del a
        b=HistGradientBoostingRegressor(loss='absolute_error',l2_regularization=1,early_stopping=False,random_state=0,**config);b.fit(tr[feats],residual,sample_weight=w);candidate.append(b.predict(va[feats]));del b
    day=va.cosz_max.to_numpy()>0;old=np.where(day,np.mean(baseline,axis=0),0)*val.cap.to_numpy();new=np.where(day,np.clip(prior.predict(pv)+np.mean(candidate,axis=0),0,1),0)*val.cap.to_numpy()
    val['reference']=.4*val.base+.6*old;val['candidate']=.4*val.base+.6*new
    if not np.isfinite(val[['reference','candidate','actual','smp']].to_numpy()).all() or (val[['reference','candidate']]<0).any().any():raise ValueError('invalid')
    def metric(col):
        e=val.smp*val.k*abs(val[col]-val.actual);return {'penalty':float(.1*e.sum()),'wmae_group':float(e.sum()/(val.smp*val.k*val.actual).sum())}
    out=ROOT/'personal_safe'/'fs2l1_sep_20261106';out.mkdir(parents=True,exist_ok=True);val.to_parquet(out/'group_comparison.parquet',index=False)
    info={'metric':'README provisional D05 HOLD official_S=null','period':[str(start),str(end)],'cutoff':str(cut),'sites':len(ids),'group_capacity_kw':{str(k):float(v) for k,v in val.groupby('group').cap.first().items()},'group_counts':{str(k):int(v) for k,v in val.groupby('group').site_count.first().items()},'group_hours':len(val),'training_rows':len(train),'lightgbm':lightgbm.__version__,'python':platform.python_version(),'pandas':pd.__version__,'control_max_diff':check['max_abs_diff_kwh'],'reference':metric('reference'),'candidate':metric('candidate'),'input_sha256':{str(p):sha(p) for p in inputs+[features,smp_path]}}
    (out/'results.json').write_text(json.dumps(info,indent=2),encoding='utf8');print(json.dumps(info),flush=True)
if __name__=='__main__':main()
