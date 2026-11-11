"""Hindsight diagnostic for grouping potential on pre-lock DEV site forecasts.

For fixed per-site forecasts, triangle inequality gives |sum(errors)| as a
lower bound for sum(|group errors|), if weights are nonnegative and identical
across groups. It ignores feasibility, making this only an optimistic bound.
Never use these diagnostic actuals to generate or select a submission.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from safe_reproduce import safe_parquet

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''): h.update(b)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('predictions',nargs='+',type=Path);p.add_argument('--smp',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args()
 smp=pd.read_csv(a.smp,parse_dates=['ts']);smp.ts=pd.to_datetime(smp.ts,utc=True).dt.tz_convert('Asia/Seoul')
 if smp.ts.max()>pd.Timestamp('2024-11-01',tz='Asia/Seoul') or smp.ts.duplicated().any() or smp.smp.isna().any() or (smp.smp<0).any():raise ValueError('SMP must be verified, pre-cut, unique and nonnegative')
 smp['k']=np.where(smp.ts.dt.hour.between(11,15),3,1);out={'basis':'site forecast fixed; optimistic unconstrained aggregation lower bound; README provisional D05 HOLD; not official','smp_sha256':sha(a.smp),'folds':{}}
 for path in a.predictions:
  d=safe_parquet(path)
  if d.duplicated(['site_id','ts']).any() or not d.group.isin(['G1','G2']).all() or d[['energy','site_pred']].isna().any().any():raise ValueError('invalid site predictions')
  d['err']=d.site_pred-d.energy
  groups=d.groupby(['group','ts'],as_index=False).agg(err=('err','sum'))
  pivot=groups.pivot(index='ts',columns='group',values='err').reset_index()
  if pivot[['G1','G2']].isna().any().any():raise ValueError('missing group-hour')
  pivot=pivot.merge(smp,on='ts',how='left',validate='one_to_one')
  if pivot.smp.isna().any():raise ValueError('missing SMP')
  w=.1*pivot.smp*pivot.k
  current=float((w*(pivot.G1.abs()+pivot.G2.abs())).sum())
  lower=float((w*(pivot.G1+pivot.G2).abs()).sum())
  if lower>current+1e-6:raise AssertionError('triangle inequality violated')
  out['folds'][path.stem]={'sha256':sha(path),'sites':int(d.site_id.nunique()),'hours':int(d.ts.nunique()),'rows':len(d),'current_penalty':current,'theoretical_lower_bound':lower,'optimistic_max_grouping_saving':current-lower,'optimistic_share':(current-lower)/current if current else None}
  print(path.stem,json.dumps(out['folds'][path.stem]),flush=True)
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2),encoding='utf-8')

if __name__=='__main__':main()
