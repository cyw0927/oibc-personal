"""Group-day total model (direct on the scored aggregate): capacity-weighted daily features of the group's sites,
target = group day energy / sum(capacity*sum cosz) over the sites with a complete day; rolling folds.
Output: group-day ratio rd_g = predicted group day energy / sum of hourly-B day predictions (same sites).
Usage: cv_groupday.py <site_feat_dir> <site_level_day.parquet (has hourly B per site-hour)> <sites.csv> <out.parquet> <folds...>
"""
import sys, warnings
from pathlib import Path
warnings.simplefilter("ignore")
import lightgbm as lgb
import numpy as np
import pandas as pd

FD, SL, SITES, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
FOLDS = sys.argv[5:]
KST = "Asia/Seoul"
src = open(Path(__file__).with_name("final_day.py"), encoding="utf-8").read()
src = src[:src.index("tr = pd.read_parquet")]
src = src[:src.index("D, FD, HB, BASE, OUT")] + src[src.index("KST = "):]
g = {"__file__": __file__}; exec(src, g)
tr = pd.read_parquet(FD / "site_train.parquet"); tr["ts"] = tr.ts.dt.tz_convert(KST)
day = g["day_table"](tr)
sites = pd.read_csv(SITES); grp = sites.set_index("site_id").current_group.map({"A": "G1", "B": "G2"})
day["group"] = day.site_id.map(grp)
day["complete"] = (day.n_e == day.nlit) & day.esum.notna()
num = [c for c in day.columns if c not in ("site_id", "day", "group", "esum", "n_e", "nlit", "complete", "site_num", "lat", "lon") and day[c].dtype.kind in "fi"]
day["wcap"] = day.capacity_kw * day.csum
agg = {}
for c in num:
    if c in ("capacity_kw", "csum", "wcap", "doy", "doy_sin", "doy_cos"):
        continue
    day[f"{c}__x"] = day[c] * day.capacity_kw
gd = day.groupby(["group", "day"]).agg(cap=("capacity_kw", "sum"), wcap=("wcap", "sum"), doy=("doy", "first"), doy_sin=("doy_sin", "first"), doy_cos=("doy_cos", "first"),
                                       **{c: (f"{c}__x", "sum") for c in num if f"{c}__x" in day})
for c in num:
    if c in gd:
        gd[c] = gd[c] / gd.cap
cm = day[day.complete].groupby(["group", "day"]).agg(e=("esum", "sum"), wc=("wcap", "sum"), capc=("capacity_kw", "sum"))
gd = gd.join(cm); gd["cov"] = gd.capc / gd.cap
gd["kd"] = (gd.e / gd.wc).clip(0, 1.5)
gd = gd.reset_index(); gd["gnum"] = (gd.group == "G2").astype(int)
F = [c for c in gd.columns if c not in ("group", "day", "e", "wc", "capc", "kd", "cov", "cap")]
print("group-days", len(gd), "features", len(F), flush=True)
sl = pd.read_parquet(SL); sl["ts"] = pd.to_datetime(sl.ts).dt.tz_convert(KST); sl["day"] = (sl.ts - pd.Timedelta(hours=1)).dt.normalize()
sl["group"] = sl.site_id.map(grp)
P = dict(objective="regression_l1", n_estimators=1500, learning_rate=0.02, num_leaves=15, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.6, verbose=-1, n_jobs=-1)
out = []
for M in FOLDS:
    Pm = pd.Period(M, freq="M"); st = pd.Timestamp(Pm.start_time).tz_localize(KST); en = st + pd.offsets.MonthBegin(1)
    trn = gd[(gd.day < st) & gd.kd.notna() & (gd["cov"] > 0.5)]
    tst = gd[(gd.day >= st) & (gd.day < en)].copy()
    pr = np.zeros(len(tst))
    for s in range(3):
        pr += lgb.LGBMRegressor(random_state=s, **P).fit(trn[F], trn.kd, sample_weight=trn.wc.to_numpy()).predict(tst[F]) / 3
    tst["kd_pred"] = np.clip(pr, 0, None)
    # sites present in OOF hourly B for the fold: predicted day energy for those sites = kd_pred * sum(cap*csum) of those sites
    pres = sl[sl.fold == M][["site_id", "day", "group"]].drop_duplicates()
    wsite = day[["site_id", "day", "wcap"]].merge(pres, on=["site_id", "day"])
    wg = wsite.groupby(["group", "day"]).wcap.sum().rename("wcap_present")
    bsum = sl[sl.fold == M].groupby(["group", "day"]).B.sum().rename("Bday")
    t = tst.set_index(["group", "day"]).join(wg).join(bsum)
    t["rd_gd"] = (t.kd_pred * t.wcap_present / t.Bday.where(t.Bday > 1)).fillna(1.0).clip(0.3, 3)
    t["fold"] = M
    out.append(t.reset_index()[["fold", "group", "day", "rd_gd"]])
    print(M, len(trn), "rd mean", round(t.rd_gd.mean(), 4), flush=True)
pd.concat(out).to_parquet(OUT, index=False)
