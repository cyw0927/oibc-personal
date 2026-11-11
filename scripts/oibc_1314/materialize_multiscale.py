"""Materialize and independently verify all coherent temporal output levels."""
import json
import numpy as np
import pandas as pd
from wls_spatial_reconcile import safe_parquet,SITES,sha
from multiscale_base_forecasts import OUT,DURATIONS

def main():
 sites=pd.read_csv(SITES,usecols=['site_id','current_group']);sites['group']=sites.current_group.map({'A':'G1','B':'G2'})
 if sites.site_id.duplicated().any() or sites.group.isna().any():raise ValueError('site mapping')
 result={}
 for fold in ['dev_jan','dev_feb']:
  site=safe_parquet(OUT/f'{fold}_site_result.parquet').merge(sites[['site_id','group']],on='site_id',validate='many_to_one');group=safe_parquet(OUT/f'{fold}_group_result.parquet')
  for d in (site,group):
   t=d.ts-pd.Timedelta(hours=1);d['day']=t.dt.floor('D');d['hour']=t.dt.hour
  sp=[];gp=[];maxdiff=0.
  for h in DURATIONS:
   site['slot']=site.hour//h;group['slot']=group.hour//h
   s=site.groupby(['site_id','group','day','slot'],as_index=False).agg(prediction_kwh=('prediction_kwh','sum'),hours=('ts','nunique'));s['duration']=h
   g=group.groupby(['group','day','slot'],as_index=False).agg(prediction_kwh=('candidate','sum'),hours=('ts','nunique'));g['duration']=h
   if not s.hours.eq(h).all() or not g.hours.eq(h).all():raise ValueError('incomplete aggregation')
   chk=s.groupby(['group','day','slot'],as_index=False).prediction_kwh.sum().merge(g,on=['group','day','slot'],validate='one_to_one',suffixes=('_site','_group'))
   diff=float(np.max(abs(chk.prediction_kwh_site-chk.prediction_kwh_group)));maxdiff=max(maxdiff,diff)
   if diff>1e-7:raise ValueError('spatial and temporal coherence failure')
   sp.append(s.drop(columns='hours'));gp.append(g.drop(columns='hours'))
  so=pd.concat(sp,ignore_index=True);go=pd.concat(gp,ignore_index=True)
  spath=OUT/f'{fold}_site_adjusted_levels.parquet';gpath=OUT/f'{fold}_group_adjusted_levels.parquet';so.to_parquet(spath,index=False);go.to_parquet(gpath,index=False)
  result[fold]={'levels':list(DURATIONS),'site_rows':len(so),'group_rows':len(go),'spatial_max_diff_kwh':maxdiff,'site_levels_sha256':sha(spath),'group_levels_sha256':sha(gpath)};print(fold,result[fold],flush=True)
 (OUT/'coherence_checks.json').write_text(json.dumps(result,indent=2),encoding='utf8')
if __name__=='__main__':main()
