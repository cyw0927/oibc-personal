"""Variant check: direct model = mean of 3 pre-registered HGB configs (smpk weight). Reuses dev_eval helpers' logic."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
D=Path(sys.argv[1]); FEAT=Path(sys.argv[2]); OUT=Path(sys.argv[3])
KST="Asia/Seoul"
FOLDS={"DEV_JUL_2024":"2024-06-30 00:00","DEV_OCT_2024":"2024-09-30 00:00","DEV_JAN":"2024-12-31 00:00","DEV_FEB":"2025-01-31 00:00"}
CONFIGS=[dict(learning_rate=0.05,max_iter=600,max_leaf_nodes=31,min_samples_leaf=40),
         dict(learning_rate=0.03,max_iter=900,max_leaf_nodes=15,min_samples_leaf=80),
         dict(learning_rate=0.05,max_iter=500,max_leaf_nodes=63,min_samples_leaf=30)]
NON={"ts","group","energy","cap_avail","n_sites","group_cap_kw","cf","smp","k","cov"}
smp=pd.read_csv(D/"train"/"smp.csv"); smp["ts"]=pd.to_datetime(smp["ts"]).dt.tz_convert(KST).dt.as_unit("us"); smp["k"]=np.where(smp.ts.dt.hour.between(11,15),3,1)
ft=pd.read_parquet(FEAT/"group_features_train.parquet"); ft["ts"]=ft.ts.dt.as_unit("us"); ft=ft.merge(smp,on="ts",how="left")
ft["cov"]=ft.cap_avail/ft.group_cap_kw; ft["cf"]=(ft.energy/ft.cap_avail).clip(0,1.2)
gp=pd.read_parquet(OUT/"dev_group_preds.parquet")
feats=[c for c in ft.columns if c not in NON]
res=[]
for fold,cut in FOLDS.items():
    cut=pd.Timestamp(cut,tz=KST); vk=gp[gp.fold==fold]
    tr=ft[(ft.ts<=cut)&ft.energy.notna()&(ft["cov"]>=0.3)&(ft.cosz_max>0)]
    va=vk[["group","ts"]].merge(ft.drop(columns=["smp","k"]),on=["group","ts"],how="left")
    sw=tr["cov"].values*tr.k.values*np.clip(tr.smp.values,1,None)/tr.smp.clip(lower=1).mean()
    ps=[]
    for c in CONFIGS:
        m=HistGradientBoostingRegressor(loss="absolute_error",l2_regularization=1.0,early_stopping=False,random_state=0,**c).fit(tr[feats],tr.cf,sample_weight=sw)
        ps.append(np.clip(m.predict(va[feats]),0,1))
    o=vk.copy()
    for i,p in enumerate(ps): o[f"cfg{i}"]=np.where(va.cosz_max.values>0,p,0)*vk.cap_sub.values
    o["direct_ens"]=o[["cfg0","cfg1","cfg2"]].mean(axis=1); res.append(o)
gp2=pd.concat(res,ignore_index=True); gp2.to_parquet(OUT/"dev_group_preds_ens.parquet",index=False)
pen=lambda d,p: float((0.10*d.smp*d.k*(p-d.actual).abs()).sum())
for c in ["fs2","direct_smpk","cfg0","cfg1","cfg2","direct_ens"]:
    print(c,{f:round(pen(g,g[c])/1e6,3) for f,g in gp2.groupby("fold")},"ALL",round(pen(gp2,gp2[c])/1e6,3))
A=[0,0.25,0.3,0.4,0.5,0.6,1]
grid={a:{f:pen(g,a*g.fs2+(1-a)*g.direct_ens) for f,g in gp2.groupby("fold")} for a in A}
for a in A: print("blend",a,{f:round(v/1e6,3) for f,v in grid[a].items()},"ALL",round(sum(grid[a].values())/1e6,3))
lofo=0;pick={}
for f in grid[0]:
    b=min(A,key=lambda a:sum(v for ff,v in grid[a].items() if ff!=f)); pick[f]=b; lofo+=grid[b][f]
print("LOFO",pick,round(lofo/1e6,3))
