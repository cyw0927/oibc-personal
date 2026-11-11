"""Nonnegative joint spatial/hourly-daily GLS approximation; prelock OOF only."""
import hashlib,json,platform,time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize, check_grad
from wls_spatial_reconcile import safe_parquet,LOCK,ROOT,OOF,SITES,OCT,WINTER,sha

OUT=ROOT/'cross_temporal_20261109'
FEATURES=ROOT/'winter_20261105'/'group_features_prelock_v2.parquet'
DAILY_COLS=['cosz','cosz_max','cs_a','cs_b','cs_ab','wa_cloud','wa_temp','wa_humidity','wa_rain','wb_temp','wb_rel_hum','wb_uv_idx','wb_precip_prob','wb_total_liq','wb_cloud','doy_sin','doy_cos','group_id']

def daily_table(ft):
 ft=ft.copy();ft['day']=(ft.ts-pd.Timedelta(hours=1)).dt.floor('D');ft['coverage']=ft.cap_avail/ft.group_cap_kw
 ft['target']=ft.energy/ft.cap_avail;ft['good']=ft.energy.notna()&(ft.coverage>=.3)&np.isfinite(ft.target)
 df=ft.groupby(['group','day'],as_index=False).agg(**{c:(c,'mean') for c in DAILY_COLS},target=('target','sum'),good=('good','sum'),hours=('ts','nunique'),last_ts=('ts','max'))
 return df[df.hours.eq(24)].copy()

def daily_predictions(df,gg,cut):
 train=df[(df.last_ts<=cut)&(df.good==24)&(df.day>=pd.Timestamp('2023-06-01',tz='Asia/Seoul'))].copy()
 g=gg.copy();g['day']=(g.ts-pd.Timedelta(hours=1)).dt.floor('D')
 val=g.groupby(['group','day'],as_index=False).agg(cap=('cap','first'),hours=('ts','size'))
 val=val.merge(df[['group','day']+DAILY_COLS],on=['group','day'],validate='one_to_one')
 if train.empty or not val.hours.eq(24).all() or train.last_ts.max()>cut:raise ValueError('daily cutoff/grid invalid')
 x=train[DAILY_COLS].to_numpy(float);z=val[DAILY_COLS].to_numpy(float);median=np.nanmedian(x,axis=0)
 if not np.isfinite(median).all():raise ValueError('unavailable daily feature')
 x=np.where(np.isfinite(x),x,median);z=np.where(np.isfinite(z),z,median)
 mean=x.mean(axis=0);sd=x.std(axis=0);sd=np.where(sd>1e-12,sd,1.)
 x=(x-mean)/sd;z=(z-mean)/sd;x=np.column_stack([np.ones(len(x)),x]);z=np.column_stack([np.ones(len(z)),z])
 reg=np.eye(x.shape[1]);reg[0,0]=0.;coef=np.linalg.solve(x.T@x+reg,x.T@train.target.to_numpy(float))
 val['daily_pred']=np.maximum(z@coef,0)*val.cap
 return val[['group','day','daily_pred']],{'train_rows':len(train),'train_max':str(train.last_ts.max())}

def shrink_cov(x):
 x=np.asarray(x,float)
 if x.ndim!=2 or len(x)<3 or not np.isfinite(x).all():raise ValueError('bad covariance sample')
 sd=x.std(axis=0);good=sd[sd>1e-10]
 if len(good)==0:raise ValueError('degenerate covariance')
 floor=float(np.median(good))*1e-3;sd=np.maximum(sd,floor)
 X=x/sd;X=X-X.mean(axis=0);n,p=X.shape;emp=X.T@X/n;mu=np.trace(emp)/p
 beta=(np.sum((X*X).T@(X*X))/n-np.sum(emp*emp))/(p*n)
 delta=np.sum((emp-mu*np.eye(p))**2)/p
 shrink=0. if delta<=0 else float(np.clip(beta/delta,0.,1.))
 corr=(1-shrink)*emp+shrink*mu*np.eye(p);c=corr*sd[:,None]*sd[None,:]
 c.flat[::len(sd)+1]+=max(float(np.median(np.diag(c)))*1e-6,1e-12)
 return c,shrink

def objective(x,shape,A,T,Y,si,ti):
 z=x.reshape(shape); e=A@z@T.T-Y; q=si@e@ti
 return float((e*q).sum()),(2*A.T@q@T).ravel()

def synthetic_check():
 rng=np.random.default_rng(11);A=np.array([[1.,0.],[0.,1.],[.4,.6]]);T=np.array([[1.,0.],[0.,1.],[1.,1.]])
 Y=rng.random((3,3));si=np.array([[2.,.1,0.],[.1,1.,.1],[0.,.1,1.]]);ti=si.copy();x=rng.random(4)
 err=check_grad(lambda y:objective(y,(2,2),A,T,Y,si,ti)[0],lambda y:objective(y,(2,2),A,T,Y,si,ti)[1],x)
 if err>1e-5:raise ValueError('gradient failure')
 return float(err)

def main():
 grad=synthetic_check();ft=safe_parquet(FEATURES);site=safe_parquet(OOF);oct=safe_parquet(OCT);oct=oct[oct.fold.eq('DEV_OCT_2024')].rename(columns={'actual_kwh':'actual','l1_kwh':'base','direct_kwh':'direct','prediction_kwh':'reference'})
 sites=pd.read_csv(SITES,usecols=['site_id','current_group','capacity_kw']).sort_values('site_id').reset_index(drop=True);sites['group']=sites.current_group.map({'A':'G1','B':'G2'})
 if len(sites)!=253 or sites.site_id.duplicated().any() or sites.group.isna().any() or ft.ts.max()>=LOCK:raise ValueError('bad inputs')
 ids=sites.site_id.tolist();caps=sites.capacity_kw.to_numpy();groups=sites.group.to_numpy();gcaps=np.array([caps[groups==g].sum() for g in ('G1','G2')]);
 A=np.zeros((255,253));A[:253]=np.eye(253)
 for j,g in enumerate(('G1','G2')):A[253+j,groups==g]=caps[groups==g]/gcaps[j]
 T=np.vstack([np.eye(24),np.ones((1,24))]); groups_data={'DEV_OCT_2024':oct}
 for fold in ('DEV_JAN','DEV_FEB'):groups_data[fold]=safe_parquet(WINTER/f'{fold.lower()}_predictions.parquet').rename(columns={'baseline':'reference'})
 site=site[site.fold.isin(groups_data)].merge(sites[['site_id','group','capacity_kw']],on='site_id',validate='many_to_one')
 if site.duplicated(['fold','site_id','ts']).any():raise ValueError('duplicate site')
 oct_caps=site[site.fold.eq('DEV_OCT_2024')].groupby(['group','ts'],as_index=False).capacity_kw.sum().rename(columns={'capacity_kw':'cap'})
 groups_data['DEV_OCT_2024']=groups_data['DEV_OCT_2024'].merge(oct_caps,on=['group','ts'],validate='one_to_one')
 df=daily_table(ft);daily={};daily_info={}
 for fold,cut in [('DEV_OCT_2024','2024-09-30 00:00'),('DEV_JAN','2024-12-31 00:00'),('DEV_FEB','2025-01-31 00:00')]:
  daily[fold],daily_info[fold]=daily_predictions(df,groups_data[fold],pd.Timestamp(cut,tz='Asia/Seoul'))
  print('daily',fold,daily_info[fold],flush=True)
 # Cohort-aware arrays: absent historical sites are marked rather than treated as measurements.
 def arrays(fold):
  ss=site[site.fold.eq(fold)];gg=groups_data[fold];times=pd.DatetimeIndex(sorted(ss.ts.unique()))
  if len(times)%24 or len(times)!=gg.ts.nunique():raise ValueError('incomplete time grid')
  present=set(ss.site_id);site_cap=sites.set_index('site_id').capacity_kw
  actual=ss.pivot(index='ts',columns='site_id',values='actual_kwh').reindex(index=times,columns=ids)
  base=ss.pivot(index='ts',columns='site_id',values='prediction_kwh').reindex(index=times,columns=ids)
  for col in ids:
   if col not in present:actual[col]=0.;base[col]=0.
  if actual.isna().any().any() or base.isna().any().any():raise ValueError('site grid incomplete')
  cohort_caps=np.array([float(site_cap[x]) if x in present else 0. for x in ids]);group_cap=np.array([cohort_caps[groups==g].sum() for g in ('G1','G2')])
  gv=gg.pivot(index='ts',columns='group',values='direct').reindex(index=times,columns=['G1','G2']);truth=gg.pivot(index='ts',columns='group',values='actual').reindex(index=times,columns=['G1','G2']);baseg=gg.pivot(index='ts',columns='group',values='base').reindex(index=times,columns=['G1','G2'])
  if gv.isna().any().any() or not np.allclose(np.column_stack([actual.to_numpy()[:,groups==g].sum(axis=1) for g in ('G1','G2')]),truth,atol=1e-6,rtol=0) or not np.allclose(np.column_stack([base.to_numpy()[:,groups==g].sum(axis=1) for g in ('G1','G2')]),baseg,atol=1e-6,rtol=0):raise ValueError('aggregate mismatch')
  if not np.allclose(gg.reference,.4*gg.base+.6*gg.direct,atol=1e-6,rtol=0):raise ValueError('reference mismatch')
  da=(times-pd.Timedelta(hours=1)).floor('D');daykeys=pd.DatetimeIndex(da.unique());daymat=daily[fold].pivot(index='day',columns='group',values='daily_pred').reindex(index=daykeys,columns=['G1','G2'])
  if daymat.isna().any().any():raise ValueError('missing daily predictions')
  sy=base.to_numpy()/caps;sa=actual.to_numpy()/caps;gy=gv.to_numpy()/group_cap;ga=truth.to_numpy()/group_cap
  return {'times':times,'site_pred':sy.reshape(-1,24,253),'site_actual':sa.reshape(-1,24,253),'group_pred':gy.reshape(-1,24,2),'group_actual':ga.reshape(-1,24,2),'group_daily':daymat.to_numpy()/group_cap,'group_cap':group_cap,'present':present,'gg':gg}
 data={f:arrays(f) for f in groups_data}
 result={'method':'nonnegative joint spatial/hourly-daily separable shrinkage GLS approximation','protocol':'docs/cross_temporal_reconciliation_plan_20261109.md','metric':'README provisional D05 HOLD official_S=null','grad_check_abs':grad,'daily_models':daily_info,'caveat':'hour/day only; site daily base is sum hourly; separable covariance and 252-site OCT; JAN/FEB reference reconstructed; DEV reused; actual forecast arrival unknown; historical winter feature may have opened mixed row groups','input_sha256':{str(p):sha(p) for p in [OOF,SITES,OCT,FEATURES,WINTER/'dev_jan_predictions.parquet',WINTER/'dev_feb_predictions.parquet']},'python':platform.python_version(),'folds':{}}
 OUT.mkdir(parents=True,exist_ok=True)
 for fold,hist in [('DEV_JAN',['DEV_OCT_2024']),('DEV_FEB',['DEV_OCT_2024','DEV_JAN'])]:
  v=data[fold];maxtrain=max(data[f]['times'].max() for f in hist)
  if maxtrain>=v['times'].min():raise ValueError('lookahead')
  spatial_samples=[];temporal_samples=[]
  for f in hist:
   u=data[f];se=u['site_actual']-u['site_pred'];ge=u['group_actual']-u['group_pred'];space=np.concatenate([se,ge],axis=2);spatial_samples.append(space.reshape(-1,255))
   # Each series/day contributes a 24-hour vector and an independently predicted group-day value.
   site_day=se.sum(axis=1);group_day=u['group_actual'].sum(axis=1)-u['group_daily'];day=np.concatenate([site_day,group_day],axis=1)
   temporal_samples.append(np.concatenate([space.transpose(0,2,1),day[:,:,None]],axis=2).reshape(-1,25))
  sc,ssh=shrink_cov(np.concatenate(spatial_samples));tc,tsh=shrink_cov(np.concatenate(temporal_samples));si=np.linalg.inv(sc);ti=np.linalg.inv(tc)
  new_sites=[];optim=[];hours=v['times'];prediction_ref=v['gg'].set_index(['group','ts']);t0=time.time()
  for d in range(len(v['site_pred'])):
   x0=v['site_pred'][d].T;gh=v['group_pred'][d].T;gd=v['group_daily'][d];Y=np.empty((255,25));Y[:253,:24]=x0;Y[:253,24]=x0.sum(axis=1);Y[253:,:24]=gh;Y[253:,24]=gd
   # Freeze physically dark hours only when every base site and both group predictions are zero.
   dark=(np.max(x0,axis=0)==0)&(np.max(gh,axis=0)==0);bounds=[(0.,0.) if dark[t] else (0.,None) for i in range(253) for t in range(24)]
   fun=lambda x:objective(x,(253,24),A,T,Y,si,ti)
   res=minimize(fun,x0.ravel(),jac=True,bounds=bounds,method='L-BFGS-B',options={'maxiter':10000,'ftol':1e-8,'gtol':1e-6,'maxls':30})
   initial=fun(x0.ravel())[0]
   if not np.isfinite(res.fun) or res.fun>initial+1e-6 or (res.x< -1e-10).any():raise ValueError('failed constrained optimizer')
   new_sites.append(res.x.reshape(253,24).T*caps);optim.append({'success':bool(res.success),'iterations':int(res.nit),'message':str(res.message),'initial':float(initial),'final':float(res.fun)})
   if (d+1)%10==0:print(fold,'days',d+1,'/',len(v['site_pred']),'seconds',round(time.time()-t0,1),flush=True)
  out=np.stack(new_sites).reshape(-1,253);row=[]
  for k,g in enumerate(('G1','G2')):
   total=out[:,groups==g].sum(axis=1);refs=prediction_ref.loc[g].reindex(hours)
   r=refs[['actual','base','direct','reference','smp','k']].copy();r.index.name='ts';r['candidate']=total;r['group']=g;r['site_sum']=total;row.append(r.reset_index())
  val=pd.concat(row,ignore_index=True)
  def metric(col):
   e=val.smp*val.k*abs(val[col]-val.actual);return {'penalty':float(.1*e.sum()),'wmae':float(e.sum()/(val.smp*val.k*val.actual).sum())}
  val.to_parquet(OUT/f'{fold.lower()}_group.parquet',index=False)
  siteout=pd.DataFrame(out,index=hours,columns=ids).rename_axis(index='ts',columns='site_id').stack().rename('prediction_kwh').reset_index();siteout.to_parquet(OUT/f'{fold.lower()}_site.parquet',index=False)
  info={'training_folds':hist,'training_max':str(maxtrain),'validation_start':str(hours.min()),'spatial_shrinkage':ssh,'temporal_shrinkage':tsh,'baseline':metric('reference'),'candidate':metric('candidate'),'days':len(optim),'optimizer_converged':sum(o['success'] for o in optim),'optimizer_status_counts':pd.Series([o['message'] for o in optim]).value_counts().to_dict(),'max_iterations':max(o['iterations'] for o in optim),'nonnegative_min':float(out.min()),'rows':len(val),'optimizer':optim}
  result['folds'][fold]=info;print(fold,'baseline',info['baseline'],'candidate',info['candidate'],'converged',info['optimizer_converged'],'/',info['days'],flush=True)
 (OUT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf8')
if __name__=='__main__':main()
