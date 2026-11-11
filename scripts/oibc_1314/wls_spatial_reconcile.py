"""Diagonal WLS, nonnegative spatial reconciliation with past-only OOF variances."""
import hashlib,json,platform
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.optimize import minimize
LOCK=pd.Timestamp('2025-03-01 01:00',tz='Asia/Seoul')

def safe_parquet(path):
 pf=pq.ParquetFile(path)
 for i in range(pf.num_row_groups):
  rg=pf.metadata.row_group(i);col=next((rg.column(j) for j in range(rg.num_columns) if rg.column(j).path_in_schema=='ts'),None)
  if col is None or not col.statistics or not col.statistics.has_min_max or pd.Timestamp(col.statistics.max)>=LOCK:raise ValueError('locked or unverified row group')
 return pd.read_parquet(path)

ROOT=Path(r'C:\Users\dkrak\Documents\Codex\OIBC_1314_review\personal_safe')
OOF=Path(r'C:\Users\dkrak\Documents\Codex\OIBC_shared\1314_dev_20261104\oof_dev.parquet')
SITES=ROOT.parent/'unlocked_data'/'sites.csv'
OCT=ROOT/'handoff_5_20261105'/'v2_jul_oct_group_oof.parquet'
WINTER=ROOT/'winter_20261105'
OUT=ROOT/'wls_spatial_20261109'

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def reconcile(p,g,site_var,group_var):
 """Exact diagonal WLS solution via monotone active set; rows are hours."""
 p=np.asarray(p,float); g=np.asarray(g,float); site_var=np.asarray(site_var,float)
 if p.ndim!=2 or (p<0).any() or (site_var<=0).any() or group_var<=0:raise ValueError('invalid WLS input')
 free=np.ones_like(p,dtype=bool)
 for _ in range(p.shape[1]+1):
  t=(g-(p*free).sum(axis=1))/(group_var+(free*site_var).sum(axis=1))
  raw=p+t[:,None]*site_var; new=free & (raw>0)
  if np.array_equal(new,free):break
  free=new
 else:raise RuntimeError('active set did not converge')
 x=np.where(free,np.maximum(raw,0),0)
 if not np.isfinite(x).all():raise ValueError('nonfinite result')
 return x

def synthetic_check():
 p=np.array([[2.,1.,.5],[1.,2.,1.],[0.,0.,0.]]);g=np.array([.1,5.,0.]);v=np.array([2.,.5,1.]);vg=.7
 got=reconcile(p,g,v,vg)
 for i in range(len(p)):
  fun=lambda x: np.sum((x-p[i])**2/v)+(x.sum()-g[i])**2/vg
  res=minimize(fun,np.maximum(p[i],0),bounds=[(0,None)]*len(v),method='L-BFGS-B',options={'ftol':1e-14,'gtol':1e-10})
  if not res.success or abs(fun(got[i])-res.fun)>1e-7:raise AssertionError('WLS synthetic optimization mismatch')

def main():
 synthetic_check()
 sites=pd.read_csv(SITES,usecols=['site_id','current_group','capacity_kw']);sites['group']=sites.current_group.map({'A':'G1','B':'G2'})
 if len(sites)!=253 or sites.site_id.duplicated().any() or sites.group.isna().any():raise ValueError('site table invalid')
 o=safe_parquet(OOF);g0=safe_parquet(OCT);g0=g0[g0.fold.eq('DEV_OCT_2024')].rename(columns={'actual_kwh':'actual','l1_kwh':'base','direct_kwh':'direct','prediction_kwh':'reference'})
 group={'DEV_OCT_2024':g0}
 for fold in ('DEV_JAN','DEV_FEB'):
  g=safe_parquet(WINTER/f'{fold.lower()}_predictions.parquet').rename(columns={'baseline':'reference'});group[fold]=g
 o=o[o.fold.isin(group)].merge(sites[['site_id','group','capacity_kw']],on='site_id',validate='many_to_one')
 if o.duplicated(['fold','site_id','ts']).any() or o.ts.max()>=LOCK:raise ValueError('bad OOF keys')
 for fold in group:
  ss=o[o.fold.eq(fold)];gg=group[fold]
  sums=ss.groupby(['group','ts'],as_index=False).agg(actual_check=('actual_kwh','sum'),base_check=('prediction_kwh','sum'))
  check=gg.merge(sums,on=['group','ts'],how='outer',validate='one_to_one',indicator=True)
  if not check._merge.eq('both').all() or not np.allclose(check.actual,check.actual_check,rtol=0,atol=1e-6) or not np.allclose(check.base,check.base_check,rtol=0,atol=1e-6) or not np.allclose(check.reference,.4*check.base+.6*check.direct,rtol=0,atol=1e-6):raise ValueError('site/group/1314 mismatch')
 result={'method':'nonnegative diagonal WLS spatial reconciliation, sample variance, no temporal reconciliation','protocol':'docs/wls_spatial_reconciliation_plan_20261109.md','metric':'README provisional D05 HOLD official_S=null','caveat':'site and group errors correlated; diagonal ignores correlations; OCT partial cohort; JAN/FEB group-direct reconstructed; historical group feature origin and actual forecast arrival not verified; reused DEV','input_sha256':{str(p):sha(p) for p in [OOF,SITES,OCT,WINTER/'dev_jan_predictions.parquet',WINTER/'dev_feb_predictions.parquet']},'synthetic_numeric_optimization':'PASS','python':platform.python_version(),'folds':{}}
 OUT.mkdir(parents=True,exist_ok=True)
 for fold,hist,cut in [('DEV_JAN',['DEV_OCT_2024'],'2024-12-31 00:00'),('DEV_FEB',['DEV_OCT_2024','DEV_JAN'],'2025-01-31 00:00')]:
  tr=o[o.fold.isin(hist)].copy();ve=o[o.fold.eq(fold)].copy();gg=group[fold]
  if tr.ts.max()>pd.Timestamp(cut,tz='Asia/Seoul') or tr.ts.max()>=ve.ts.min():raise ValueError('lookahead')
  tr['error']=tr.actual_kwh-tr.prediction_kwh
  var=tr.groupby('site_id').error.var(ddof=1);ve=ve.merge(var.rename('variance'),left_on='site_id',right_index=True,how='left',validate='many_to_one')
  group_hist=pd.concat([group[f][['group','ts','actual','direct']] for f in hist]);group_hist['error']=group_hist.actual-group_hist.direct
  group_vars=group_hist.groupby('group').error.var(ddof=1)
  rows=[];site_rows=[];details={}
  for name in ('G1','G2'):
   z=ve[ve.group.eq(name)].copy();sg=sites[sites.group.eq(name)].sort_values('site_id');ids=sg.site_id.tolist()
   var_site=z[['site_id','capacity_kw','variance']].drop_duplicates().set_index('site_id').reindex(ids)
   if len(var_site)!=len(ids) or var_site.capacity_kw.isna().any():raise ValueError('missing evaluation site')
   ratio=(var_site.variance/var_site.capacity_kw.pow(2)).dropna();fallback=float(ratio.median());missing=int(var_site.variance.isna().sum())
   if not np.isfinite(fallback) or fallback<=0:raise ValueError('bad variance fallback')
   var_site['variance']=var_site.variance.fillna(fallback*var_site.capacity_kw.pow(2));floor=max(1e-8,float(var_site.variance.median())*1e-6);sv=var_site.variance.clip(lower=floor).to_numpy();gv=max(float(group_vars[name]),floor)
   mat=z.pivot(index='ts',columns='site_id',values='prediction_kwh').reindex(columns=ids).sort_index()
   target=gg[gg.group.eq(name)].set_index('ts').reindex(mat.index)
   if mat.isna().any().any() or target[['direct','reference','actual','smp','k']].isna().any().any():raise ValueError('grid gap')
   x=reconcile(mat.to_numpy(),target.direct.to_numpy(),sv,gv); total=x.sum(axis=1)
   site_part=pd.DataFrame(x,index=mat.index,columns=ids).rename_axis(index='ts',columns='site_id').stack().rename('prediction_kwh').reset_index();site_part['group']=name;site_rows.append(site_part)
   row=target[['actual','base','direct','reference','smp','k']].copy();row['candidate']=total;row['group']=name;row['min_site']=x.min(axis=1);row['site_sum']=x.sum(axis=1);rows.append(row.reset_index())
   details[name]={'sites':len(ids),'fallback_sites':missing,'group_variance_kwh2':gv,'median_site_variance_kwh2':float(np.median(sv)),'hours':len(mat),'site_group_max_diff':float(np.max(np.abs(total-x.sum(axis=1))))}
  val=pd.concat(rows,ignore_index=True)
  if val.duplicated(['group','ts']).any() or (val.min_site<0).any() or not np.isfinite(val.candidate).all():raise ValueError('bad output')
  site_output=pd.concat(site_rows,ignore_index=True)
  independently=site_output.groupby(['group','ts'],as_index=False).prediction_kwh.sum().merge(val[['group','ts','candidate']],on=['group','ts'],validate='one_to_one')
  if site_output.duplicated(['site_id','ts']).any() or not np.allclose(independently.prediction_kwh,independently.candidate,rtol=0,atol=1e-8):raise ValueError('site/group coherence')
  def metric(col):
   err=val.smp*val.k*abs(val[col]-val.actual);return {'penalty':float(.1*err.sum()),'wmae':float(err.sum()/(val.smp*val.k*val.actual).sum())}
  val.to_parquet(OUT/f'{fold.lower()}_group.parquet',index=False)
  site_output.to_parquet(OUT/f'{fold.lower()}_site.parquet',index=False)
  info={'training_folds':hist,'training_max':str(tr.ts.max()),'validation_start':str(ve.ts.min()),'rows':len(val),'baseline':metric('reference'),'candidate':metric('candidate'),'groups':details}
  result['folds'][fold]=info;print(fold,json.dumps(info),flush=True)
 (OUT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf8')
if __name__=='__main__':main()
