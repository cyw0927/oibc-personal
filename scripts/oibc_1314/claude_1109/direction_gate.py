"""DIRECTION-GATE v1 (proposal doc, 11/11): predict the JOINT error direction of both groups on a day from D-1 forecasts
only, and nudge both groups' 11-15h forecasts by +/-5% when the classifier is confident (p >= 0.70); else keep v7.

Time order: for evaluation month M the classifier is trained only on days before M. Training labels come from a proxy
residual available for all 12 rolling months (B21 site-level OOF summed into the evaluated composition), because the
v7 OOF itself only exists for the 5 evaluation months. Threshold = per-group median |E| of the training days.
Features (regional = all-station means, 11-15h, per issue round): A/B cloud, B uv, B precip prob, A humidity, B vis;
17h-10h change; A-B provider gap; within-day cloud spread; proxy midday capacity factor.
Usage: direction_gate.py <dataset> <personal_safe> <fold_predictions.csv> <composition csv or 'ref'> <out_json>
"""
import json, sys, warnings
from pathlib import Path
warnings.simplefilter("ignore")
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

D, PS, FP, COMP, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], Path(sys.argv[5])
KST = "Asia/Seoul"; P_THR, STEP = 0.70, 0.05
FOLDS = ["2024-07", "2024-10", "2025-01", "2025-02", "2025-04"]

# ---- day features from raw D-1 forecasts (rounds 10 and 17)
feats = []
for p, cols in (("wa", ["wa_cloud", "wa_humidity"]), ("wb", ["wb_cloud", "wb_uv_idx", "wb_precip_prob", "wb_vis"])):
    f = pd.read_parquet(D / "train" / f"forecast_{p}.parquet", columns=[f"{p}_station", "round", "ts"] + cols)
    f["ts"] = pd.to_datetime(f.ts).dt.tz_convert(KST)
    f = f[(f.ts.dt.minute == 0) & f.ts.dt.hour.between(11, 15) & (f.ts >= pd.Timestamp("2024-05-01", tz=KST))]
    f["day"] = (f.ts - pd.Timedelta(hours=1)).dt.normalize()
    m = f.groupby(["day", "round"])[cols].mean().unstack("round")
    m.columns = [f"{c}_r{r}" for c, r in m.columns]
    for c in cols:
        m[f"{c}_d1710"] = m[f"{c}_r17"] - m[f"{c}_r10"]
    if p == "wa":
        sd = f[f["round"] == 17].groupby(["day", f.ts.dt.hour]).wa_cloud.mean().groupby("day").std().rename("wa_cloud_hourly_sd")
        m = m.join(sd)
    feats.append(m)
X = feats[0].join(feats[1], how="inner")
X["gap_cloud_r17"] = X.wa_cloud_r17 - X.wb_cloud_r17
X["doy_sin"] = np.sin(2 * np.pi * X.index.dayofyear / 365.25); X["doy_cos"] = np.cos(2 * np.pi * X.index.dayofyear / 365.25)

# ---- proxy residuals (B21 OOF, 12 months) summed into the evaluated composition
sites = pd.read_csv(D / "sites.csv")
grp = sites.set_index("site_id").current_group.map({"A": "G1", "B": "G2"}) if COMP == "ref" else pd.read_csv(COMP).set_index("site_id").group
b = pd.read_parquet(PS / "cv" / "site_level_B21_12m.parquet"); b["ts"] = pd.to_datetime(b.ts).dt.tz_convert(KST)
b["group"] = b.site_id.map(grp)
gb = b.groupby(["group", "ts"]).agg(energy=("energy", "sum"), pred=("B", "sum"), smp=("smp", "first"), k=("k", "first"), fold=("fold", "first")).reset_index()
gb = gb[gb.ts.dt.hour.between(11, 15)]
gb["day"] = (gb.ts - pd.Timedelta(hours=1)).dt.normalize(); gb["w"] = 0.1 * gb.smp * gb.k
gb["E"] = gb.w * (gb.energy - gb.pred)
caps = sites.assign(group=sites.site_id.map(grp)).groupby("group").capacity_kw.sum()
pe = gb.groupby(["day", "group"]).E.sum().unstack("group")
pcf = (gb.groupby(["day", "group"]).pred.sum().unstack("group") / caps / 5).mean(axis=1).rename("proxy_midday_cf")
X = X.join(pcf, how="left")
F = [c for c in X.columns]

# ---- evaluation on v7 (or v6) fold predictions
d = pd.read_csv(FP); d["ts"] = pd.to_datetime(d.ts).dt.tz_convert(KST); d["day"] = (d.ts - pd.Timedelta(hours=1)).dt.normalize()
d["w"] = 0.1 * d.smp * d.k; mid = d.ts.dt.hour.between(11, 15)
res = {"rule": f"p>={P_THR}, +/-{STEP:.0%} on 11-15h both groups", "months": []}
newp = d.final_pred.to_numpy().copy()
for M in FOLDS:
    st = pd.Timestamp(M + "-01", tz=KST)
    tr_days = pe.index[(pe.index < st)].intersection(X.dropna().index)
    if len(tr_days) < 20:
        res["months"].append({"month": M, "skipped": "too few training days", "train_days": int(len(tr_days))}); continue
    Etr = pe.loc[tr_days]
    thr = Etr.abs().median()
    y = np.where((Etr.G1 > thr.G1) & (Etr.G2 > thr.G2), 1, np.where((Etr.G1 < -thr.G1) & (Etr.G2 < -thr.G2), -1, 0))
    sc = StandardScaler().fit(X.loc[tr_days, F]); clf = LogisticRegression(C=0.5, max_iter=2000)
    clf.fit(sc.transform(X.loc[tr_days, F]), y)
    te_days = sorted(set(d.day[d.fold == M]))
    te_days = [t for t in te_days if t in X.index and X.loc[t, F].notna().all()]
    pr = pd.DataFrame(clf.predict_proba(sc.transform(X.loc[te_days, F])), index=te_days, columns=clf.classes_)
    s = pd.Series(0, index=te_days)
    if 1 in pr: s[pr[1] >= P_THR] = 1
    if -1 in pr: s[pr[-1] >= P_THR] = -1
    sel = (d.fold == M) & mid & d.day.isin(s.index)
    sgn = d.loc[sel, "day"].map(s).to_numpy()
    newp[sel.to_numpy()] = d.loc[sel, "final_pred"].to_numpy() * (1 + STEP * sgn)
    # truth for the applied days (v7 residual direction, for hit rate only)
    dm = d[(d.fold == M) & mid]
    ev = (dm.w * (dm.energy - dm.final_pred)).groupby([dm.day, dm.group]).sum().unstack("group")
    hits = int(sum((np.sign(ev.loc[t]).eq(s[t]).all()) for t in s.index[s != 0] if t in ev.index))
    fm = (d.fold == M).to_numpy()
    base = (d.w * (d.final_pred - d.energy).abs())[fm].sum(); new = (d.w[fm] * np.abs(newp[fm] - d.energy[fm])).sum()
    res["months"].append({"month": M, "train_days": int(len(tr_days)), "class_counts_train": {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
                          "applied_up": int((s == 1).sum()), "applied_down": int((s == -1).sum()), "joint_direction_hits": hits,
                          "max_p_up": round(float(pr[1].max()) if 1 in pr else 0, 3), "max_p_down": round(float(pr[-1].max()) if -1 in pr else 0, 3),
                          "base_M": round(base / 1e6, 4), "gate_M": round(new / 1e6, 4)})
w = d.w.to_numpy(); y_ = d.energy.to_numpy()
res["total_base_M"] = round((w * np.abs(d.final_pred - y_)).sum() / 1e6, 4); res["total_gate_M"] = round((w * np.abs(newp - y_)).sum() / 1e6, 4)
res["change_pct"] = round(100 * (res["total_gate_M"] / res["total_base_M"] - 1), 3)
# sensitivity (reported, not used for selection)
json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(json.dumps(res, indent=1, ensure_ascii=False))
