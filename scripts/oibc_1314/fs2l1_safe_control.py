"""Rebuild the saved JUL FS2+L1 ensemble using metadata-checked prelock files only."""
import gc, hashlib, json, platform, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT=Path(r'C:\Users\dkrak\Documents\Codex\OIBC_1314_review')
sys.path.insert(0,str(ROOT/'python_deps'))
sys.path.insert(0,r'C:\dev\icn-solar-oibc\oibc\src')
import lightgbm
from hf_forecast.adapters.oibc2026.contracts import load_contracts
from hf_forecast.adapters.oibc2026.data import OfficialData,training_rows,validation_grid
from hf_forecast.adapters.oibc2026.nonlinear import build_nonlinear

LOCK=pd.Timestamp('2025-03-01 01:00',tz='Asia/Seoul')
DATA=ROOT/'unlocked_data'
CONTRACTS=Path(r'C:\dev\icn-solar-oibc\workspaces\w1\oibc-data-validation')
REF=Path(r'C:\Users\dkrak\Documents\Codex\OIBC_shared\1314_dev_20261104\oof_dev.parquet')
OUT=ROOT/'personal_safe'/'fs2l1_control_20261106'

def safe_metadata(p):
    pf=pq.ParquetFile(p)
    for i in range(pf.num_row_groups):
        rg=pf.metadata.row_group(i); col=next((rg.column(j) for j in range(rg.num_columns) if rg.column(j).path_in_schema=='ts'),None)
        if col is None or not col.statistics or not col.statistics.has_min_max or pd.Timestamp(col.statistics.max)>=LOCK:
            raise ValueError(f'locked or unverified rowgroup: {p}')

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    for p in [DATA/'train'/'generation.parquet',DATA/'train'/'forecast_wa.parquet',DATA/'train'/'forecast_wb.parquet',REF]: safe_metadata(p)
    if lightgbm.__version__!='4.7.0':raise ValueError('unexpected LightGBM version')
    c=load_contracts(CONTRACTS); fold=c.folds['DEV_JUL_2024']
    if fold.val_end>=LOCK or fold.train_until>=fold.val_start:raise ValueError('unsafe fold')
    data=OfficialData(DATA,c); train=training_rows(data,fold.train_from,fold.train_until); grid=validation_grid(data,fold.site_ids,fold.val_index())
    if train.ts.max()>fold.train_until or grid.ts.max()>=LOCK:raise ValueError('bad cutoff')
    print(f'train={len(train)} grid={len(grid)}',flush=True)
    predictions=[]
    for seed in (11,22,33):
        t=time.time(); model=build_nonlinear('lgbm_cf',{'seed':seed,'n_jobs':2,'feature_set':'FS2','loss':'L1'},data)
        model.fit(train); pred=model.predict(grid)['prediction_kwh'].to_numpy(dtype=float)
        if len(pred)!=len(grid) or not np.isfinite(pred).all() or (pred<0).any():raise ValueError('invalid')
        predictions.append(pred); print('seed',seed,'seconds',round(time.time()-t,1),flush=True); del model;gc.collect()
    out=grid[['site_id','ts','energy']].copy();out['prediction']=np.mean(predictions,axis=0)
    r=pd.read_parquet(REF,columns=['fold','site_id','ts','actual_kwh','prediction_kwh']);r=r[r.fold.eq('DEV_JUL_2024')]
    out=out.merge(r,on=['site_id','ts'],how='outer',validate='one_to_one',indicator=True)
    if not out._merge.eq('both').all() or not np.allclose(out.energy,out.actual_kwh,atol=1e-8,rtol=0):raise ValueError('reference mismatch')
    delta=out.prediction-out.prediction_kwh
    info={'rows':len(out),'max_abs_diff_kwh':float(delta.abs().max()),'mean_abs_diff_kwh':float(delta.abs().mean()),'within_1e-6_fraction':float((delta.abs()<=1e-6).mean()),'cutoff':str(fold.train_until),'train_rows':len(train),'lightgbm':lightgbm.__version__,'pandas':pd.__version__,'python':platform.python_version(),'data_sha256':{str(p.relative_to(DATA)):sha(p) for p in [DATA/'train'/'generation.parquet',DATA/'train'/'forecast_wa.parquet',DATA/'train'/'forecast_wb.parquet']},'reference_sha256':sha(REF),'seeds':[11,22,33],'n_jobs':2}
    OUT.mkdir(parents=True,exist_ok=True);out[['site_id','ts','prediction']].to_parquet(OUT/'jul_control_prediction.parquet',index=False);(OUT/'results.json').write_text(json.dumps(info,indent=2),encoding='utf8');print(json.dumps(info),flush=True)
if __name__=='__main__':main()
