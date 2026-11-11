"""Nonnegative joint spatial/multiscale GLS using cross-fitted base forecasts."""
import json,platform,time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.optimize import minimize
from wls_spatial_reconcile import safe_parquet,LOCK,ROOT,OOF,SITES,OCT,WINTER,sha
from cross_temporal_reconcile import FEATURES,shrink_cov,objective,synthetic_check
from multiscale_base_forecasts import OUT,DURATIONS

def main():
 grad=synthetic_check();manifest=json.loads((OUT/'base_manifest.json').read_text());site=safe_parquet(OOF)
 sites=pd.read_csv(SITES,usecols=['site_id','current_group','capacity_kw']).sort_values('site_id').reset_index(drop=True);sites['group']=sites.current_group.map({'A':'G1','B':'G2'})
 if len(sites)!=253 or sites.site_id.duplicated().any() or sites.group.isna().any():raise ValueError('bad sites')
 ids=sites.site_id.tolist();caps=sites.capacity_kw.to_numpy();groups=sites.group.to_numpy();gcaps=np.array([caps[groups==g].sum() for g in ('G1','G2')]);A=np.zeros((255,253));A[:253]=np.eye(253)
 for j,g in enumerate(('G1','G2')):A[253+j,groups==g]=caps[groups==g]/gcaps[j]
 # Means instead of sums improve conditioning; the equivalent total is mean*duration.
 rows=[];level_order=[]
 for h in DURATIONS:
  for slot in range(24//h):
   row=np.zeros(24);row[slot*h:(slot+1)*h]=1/h;rows.append(row);level_order.append((h,slot))
 T=np.array(rows)
 if T.shape!=(60,24):raise ValueError('temporal hierarchy dimension')
 oct=safe_parquet(OCT);oct=oct[oct.fold.eq('DEV_OCT_2024')].rename(columns={'actual_kwh':'actual','l1_kwh':'base','direct_kwh':'direct','prediction_kwh':'reference'})
 group_data={'DEV_OCT_2024':oct}
 for f in ('DEV_JAN','DEV_FEB'):group_data[f]=safe_parquet(WINTER/f'{f.lower()}_predictions.parquet').rename(columns={'baseline':'reference'})
 site=site[site.fold.isin(group_data)].merge(sites[['site_id','group','capacity_kw']],on='site_id',validate='many_to_one')
 if site.duplicated(['fold','site_id','ts']).any():raise ValueError('duplicate OOF')
 def arrays(fold):
  meta=manifest['folds'][fold];cut=pd.Timestamp(meta['cutoff']);ss=site[site.fold.eq(fold)];gg=group_data[fold].copy();times=pd.DatetimeIndex(sorted(ss.ts.unique()));days=pd.DatetimeIndex((times-pd.Timedelta(hours=1)).floor('D').unique())
  if len(times)!=len(days)*24 or times.min()<=cut or pd.Timestamp(meta['site_training_max'])>cut:raise ValueError('nonchronological/incomplete grid')
  present=set(ss.site_id);act=ss.pivot(index='ts',columns='site_id',values='actual_kwh').reindex(index=times,columns=ids);base=ss.pivot(index='ts',columns='site_id',values='prediction_kwh').reindex(index=times,columns=ids)
  for col in ids:
   if col not in present:act[col]=0.;base[col]=0.
  if act.isna().any().any() or base.isna().any().any():raise ValueError('incomplete plant grid')
  cohort_caps=np.array([c if i in present else 0. for i,c in zip(ids,caps)]);groupcap=np.array([cohort_caps[groups==g].sum() for g in ('G1','G2')]);At=A.copy()
  for j,g in enumerate(('G1','G2')):At[253+j,:]=0.;At[253+j,groups==g]=cohort_caps[groups==g]/groupcap[j]
  gh=gg.pivot(index='ts',columns='group',values='direct').reindex(index=times,columns=['G1','G2']);ga=gg.pivot(index='ts',columns='group',values='actual').reindex(index=times,columns=['G1','G2']);gb=gg.pivot(index='ts',columns='group',values='base').reindex(index=times,columns=['G1','G2'])
  if gh.isna().any().any() or not np.allclose(np.column_stack([act.to_numpy()[:,groups==g].sum(1) for g in ('G1','G2')]),ga,atol=1e-6,rtol=0) or not np.allclose(np.column_stack([base.to_numpy()[:,groups==g].sum(1) for g in ('G1','G2')]),gb,atol=1e-6,rtol=0) or not np.allclose(gg.reference,.4*gg.base+.6*gg.direct,atol=1e-6,rtol=0):raise ValueError('site group baseline mismatch')
  sp=OUT/f'{fold.lower()}_site_levels.parquet';gp=OUT/f'{fold.lower()}_group_levels.parquet'
  if sha(sp)!=meta['site_levels_sha256'] or sha(gp)!=meta['group_levels_sha256']:raise ValueError('level hash mismatch')
  for path in (sp,gp):
   pf=pq.ParquetFile(path);idx=pf.schema_arrow.get_field_index('day')
   if idx<0:raise ValueError('level timestamp missing')
   for rg in range(pf.num_row_groups):
    st=pf.metadata.row_group(rg).column(idx).statistics
    if st is None or not st.has_min_max or pd.Timestamp(st.max)>=LOCK:raise ValueError('unsafe level row group')
  x=pd.read_parquet(sp);y=pd.read_parquet(gp)
  if x.duplicated(['site_id','day','duration','slot']).any() or y.duplicated(['group','day','duration','slot']).any():raise ValueError('duplicate hierarchy key')
  if x.day.max()>=LOCK or y.day.max()>=LOCK:raise ValueError('locked level')
  nd=len(days);Y=np.zeros((nd,255,60));Y[:,:253,:24]=(base.to_numpy()/caps).reshape(nd,24,253).transpose(0,2,1);Y[:,253:,:24]=(gh.to_numpy()/groupcap).reshape(nd,24,2).transpose(0,2,1)
  for j,(h,slot) in enumerate(level_order[24:],start=24):
   sx=x[(x.duration==h)&(x.slot==slot)].pivot(index='day',columns='site_id',values='prediction_cf').reindex(index=days,columns=ids)
   for col in ids:
    if col not in present:sx[col]=0.
   gy=y[(y.duration==h)&(y.slot==slot)].pivot(index='day',columns='group',values='prediction_cf').reindex(index=days,columns=['G1','G2'])
   if sx.isna().any().any() or gy.isna().any().any():raise ValueError('missing forecast level')
   Y[:,:253,j]=sx.to_numpy();Y[:,253:,j]=gy.to_numpy()
  truth=(act.to_numpy()/caps).reshape(nd,24,253).transpose(0,2,1);return {'times':times,'days':days,'Y':Y,'truth':truth,'A':At,'group_cap':groupcap,'gg':gg,'present':present}
 data={f:arrays(f) for f in group_data};outinfo={'protocol':'docs/cross_temporal_full_scales_plan_20261109.md','metric':'README provisional D05 HOLD official_S=null','method':'joint nonnegative spatial x 60 temporal levels; separable shrinkage covariance','grad_check_abs':grad,'durations':list(DURATIONS),'approximation':'separable covariance, ridge aggregate base forecasts based on earlier OOF, not exact full unconstrained covariance/paper original base models','base_manifest_sha256':sha(OUT/'base_manifest.json'),'oof_sha256':sha(OOF),'python':platform.python_version(),'folds':{}}
 for fold,hist in [('DEV_JAN',['DEV_OCT_2024']),('DEV_FEB',['DEV_OCT_2024','DEV_JAN'])]:
  v=data[fold];maxtr=max(data[f]['times'].max() for f in hist)
  if maxtr>=v['times'].min():raise ValueError('lookahead covariance')
  sp=[];te=[]
  for f in hist:
   u=data[f];err=np.stack([u['A']@x@T.T-y for x,y in zip(u['truth'],u['Y'])]);sp.append(err[:,:,:24].transpose(0,2,1).reshape(-1,255));te.append(err.reshape(-1,60))
  sc,ssh=shrink_cov(np.concatenate(sp));tc,tsh=shrink_cov(np.concatenate(te))
  historical_present=set().union(*(data[f]['present'] for f in hist));missing=[i for i,sid in enumerate(ids) if sid not in historical_present]
  for i in missing:
   peers=[j for j in range(253) if j not in missing and groups[j]==groups[i]];fallback=float(np.median(np.diag(sc)[peers]))
   if not np.isfinite(fallback) or fallback<=0:raise ValueError('invalid missing plant covariance fallback')
   sc[i,:]=0.;sc[:,i]=0.;sc[i,i]=fallback
  si=np.linalg.inv(sc);ti=np.linalg.inv(tc);news=[];optim=[];start=time.time()
  for d,Y in enumerate(v['Y']):
   x0=Y[:253,:24];dark=(np.max(x0,axis=0)==0)&(np.max(Y[253:,:24],axis=0)==0);bounds=[(0.,0.) if dark[t] else (0.,None) for i in range(253) for t in range(24)]
   fun=lambda x:objective(x,(253,24),A,T,Y,si,ti)
   res=minimize(fun,x0.ravel(),jac=True,bounds=bounds,method='L-BFGS-B',options={'maxiter':10000,'ftol':1e-8,'gtol':1e-6,'maxls':30})
   initial=fun(x0.ravel())[0]
   if not np.isfinite(res.fun) or res.fun>initial+1e-6 or (res.x< -1e-10).any():raise ValueError('invalid optimizer')
   x=res.x;grad=res.jac.copy();grad[(x<=1e-10)&(grad>0)]=0.;grad[np.array([lo==hi for lo,hi in bounds])]=0.
   news.append(x.reshape(253,24).T*caps);optim.append({'success':bool(res.success),'iterations':int(res.nit),'message':str(res.message),'initial':float(initial),'final':float(res.fun),'projected_gradient_max':float(np.abs(grad).max())})
   if (d+1)%10==0:print(fold,'days',d+1,'/',len(v['days']),'seconds',round(time.time()-start,1),flush=True)
  out=np.stack(news).reshape(-1,253);ref=v['gg'].set_index(['group','ts']);result_rows=[]
  for g in ('G1','G2'):
   total=out[:,groups==g].sum(1);z=ref.loc[g].reindex(v['times'])[['actual','base','direct','reference','smp','k']].copy();z.index.name='ts';z['candidate']=total;z['group']=g;result_rows.append(z.reset_index())
  val=pd.concat(result_rows,ignore_index=True)
  def metric(col):
   e=val.smp*val.k*abs(val[col]-val.actual);return {'penalty':float(.1*e.sum()),'wmae':float(e.sum()/(val.smp*val.k*val.actual).sum())}
  val.to_parquet(OUT/f'{fold.lower()}_group_result.parquet',index=False)
  plant=pd.DataFrame(out,index=v['times'],columns=ids).rename_axis(index='ts',columns='site_id').stack().rename('prediction_kwh').reset_index();plant.to_parquet(OUT/f'{fold.lower()}_site_result.parquet',index=False)
  info={'training_folds':hist,'training_max':str(maxtr),'validation_start':str(v['times'].min()),'spatial_shrinkage':ssh,'temporal_shrinkage':tsh,'missing_training_site_fallback_count':len(missing),'baseline':metric('reference'),'candidate':metric('candidate'),'days':len(optim),'optimizer_converged':sum(x['success'] for x in optim),'optimizer_status_counts':pd.Series([x['message'] for x in optim]).value_counts().to_dict(),'max_iterations':max(x['iterations'] for x in optim),'nonnegative_min':float(out.min()),'optimizer':optim}
  outinfo['folds'][fold]=info;print(fold,'baseline',info['baseline'],'candidate',info['candidate'],'converged',info['optimizer_converged'],'/',info['days'],flush=True)
 (OUT/'results.json').write_text(json.dumps(outinfo,indent=2),encoding='utf8')
if __name__=='__main__':main()
