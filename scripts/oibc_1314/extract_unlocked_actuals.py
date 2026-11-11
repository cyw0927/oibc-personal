"""Extract JAN/FEB DEV actuals only from a proven pre-lock OOF artifact.

The source is a collaborator-derived artifact, not the original measurement file.
Never commit outputs. It does not recover December actuals or weather forecasts.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

LOCK = pd.Timestamp('2025-03-01 01:00', tz='Asia/Seoul')
FOLDS = {'DEV_JAN': (182160, 720), 'DEV_FEB': (163944, 648)}

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('source', type=Path)
    ap.add_argument('destination', type=Path)
    a=ap.parse_args()
    pf=pq.ParquetFile(a.source)
    for i in range(pf.num_row_groups):
        rg=pf.metadata.row_group(i)
        c=next((rg.column(j) for j in range(rg.num_columns) if rg.column(j).path_in_schema=='ts'),None)
        if c is None or not c.statistics or not c.statistics.has_min_max or pd.Timestamp(c.statistics.max)>=LOCK:
            raise ValueError('Refusing to read: locked or unverified row group')
    df=pd.read_parquet(a.source,columns=['fold','site_id','ts','actual_kwh'])
    a.destination.mkdir(parents=True,exist_ok=True)
    manifest={'source_sha256':sha(a.source),'origin':'existing team DEV OOF; actual columns only','folds':{}}
    for fold,(rows,hours) in FOLDS.items():
        d=df[df.fold.eq(fold)].copy().sort_values(['site_id','ts'])
        if len(d)!=rows or d.site_id.nunique()!=253 or d.ts.nunique()!=hours or d.duplicated(['site_id','ts']).any() or d.actual_kwh.isna().any():
            raise ValueError(f'{fold}: incomplete or duplicate keys')
        if d.groupby('site_id').ts.nunique().min()!=hours: raise ValueError('Incomplete per-site time axis')
        out=a.destination / f'{fold.lower()}_actuals.parquet'
        d.to_parquet(out,index=False)
        manifest['folds'][fold]={'rows':rows,'sites':253,'hours':hours,'start':str(d.ts.min()),'end':str(d.ts.max()),'sha256':sha(out)}
    (a.destination/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__': main()
