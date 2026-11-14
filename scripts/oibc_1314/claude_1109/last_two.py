"""(1) Daily-total AnEn correction on top of v14: daily analogs (same group, N=20, days < scored month) on daily box predictors
    (ECMWF ec_box, GFS gf_dsw/gf_tcc, provided wa/wb daily mean cloud, daily clear-sky sum) -> daily cf; ratio = AnEn daily / v14 daily;
    out = v14 * (1 + b (ratio - 1)), b in {0, .1, .2, .3}.
(2) Hour-split v14 weights: (AnEn, phys) weights separately for k=3 hours (11-15) and other hours.
Both LOMO on the same 11-month rows vs v14."""
import sys, importlib.util, itertools
import numpy as np, pandas as pd
PS, COMP = sys.argv[1:3]; KST = "Asia/Seoul"; N = 20
spec = importlib.util.spec_from_file_location("ds", __file__.replace("last_two.py", "diag_spread_lib.py")); ds = importlib.util.module_from_spec(spec); spec.loader.exec_module(ds)
g = ds.build(PS, COMP, "_rich", "_gx", ""); g["p"] = (0.35*g.E + 0.35*g.B21 + 0.30*g.H1)*(1 + 0.5*(g.rd - 1)); g["w"] = 0.1*g.smp*g.k; g["fold"] = g.fold.astype(str)
def ld(name, col, new):
    z = pd.read_parquet(f"{PS}/cv/{name}.parquet"); z["ts"] = pd.to_datetime(z.ts).dt.tz_convert(KST); return z[["group", "ts", col]].rename(columns={col: new})
g = g.merge(ld("anen_oof", "anen", "a0"), on=["group", "ts"]).merge(ld("phys_oof_exp", "phys_ex_ssrd", "ph"), on=["group", "ts"]); g = g[g.ph.notna()].copy()
g["v14"] = .9*g.p + .05*g.a0 + .05*g.ph
pen = lambda p: (g.w * (p - g.energy).abs()).groupby(g.fold).sum() / 1e6
base = pen(g.v14)
def report(name, R):
    lomo = {m: R.drop(index=m).sum().idxmin() for m in R.index}; L = pd.Series({m: R.loc[m, k] for m, k in lomo.items()})
    print(name, "in-sample best", R.sum().idxmin(), round(R.sum().min(), 3), "| LOMO", round(L.sum(), 3), f"{100*(L.sum()/base.sum()-1):+.2f}% vs v14", "better", int((L < base).sum()), "/", len(R), "| picks", list(lomo.values()))
# (1) daily AnEn
f = pd.read_parquet(f"{PS}/feat_full_comp/group_features_train.parquet"); f["ts"] = pd.to_datetime(f.ts).dt.tz_convert(KST)
f["day"] = (f.ts - pd.Timedelta(hours=1)).dt.tz_localize(None).dt.normalize()
dd = f.groupby(["group", "day"]).agg(e=("energy", "sum"), n=("energy", "count"), cap=("cap_avail", "mean"), gcap=("group_cap_kw", "first"),
                                      wa=("wa_cloud", "mean"), wb=("wb_cloud", "mean"), cs=("cs_ab", "sum")).reset_index()
bx = pd.read_parquet(f"{PS}/gfs/box_ecgf.parquet"); bx["day"] = pd.to_datetime(bx.day)
dd = dd.merge(bx[["day", "ec_box", "gf_dsw", "gf_tcc"]], on="day", how="left")
dd["cf"] = np.where((dd.n >= 20) & (dd.cap / dd.gcap >= 0.3), dd.e / (dd.cap * 24), np.nan)
V = ["ec_box", "gf_dsw", "gf_tcc", "wa", "wb", "cs"]; sig = dd[V].std()
rows = []
for grp, y in dd.groupby("group"):
    y = y.set_index("day")
    for m in sorted(g.fold.unique()):
        st = pd.Timestamp(m + "-01"); en = st + pd.offsets.MonthBegin(1)
        past = y[(y.index < st) & y.cf.notna()].dropna(subset=["gf_dsw", "gf_tcc", "wa", "wb", "cs"]); tgt = y[(y.index >= st) & (y.index < en)]
        A = past[V].fillna(past[V].mean()).to_numpy(); F = tgt[V].fillna(past[V].mean()).to_numpy()
        d = (np.abs(F[:, None, :] - A[None, :, :]) / sig.values).sum(2)
        rows.append(pd.DataFrame({"group": grp, "day": tgt.index, "dcf": past.cf.to_numpy()[np.argsort(d, 1)[:, :N]].mean(1)}))
dc = pd.concat(rows)
g["day"] = (g.ts - pd.Timedelta(hours=1)).dt.tz_localize(None).dt.normalize()
cap = pd.read_parquet(f"{PS}/cv/group_cv_comp.parquet")[["group", "ts", "cap_avail"]]; cap["ts"] = pd.to_datetime(cap.ts).dt.tz_convert(KST)
g = g.merge(cap, on=["group", "ts"], how="left").merge(dc, on=["group", "day"], how="left")
gd = g.groupby(["group", "day"]).agg(v=("v14", "sum"), c=("cap_avail", "mean")).reset_index(); gd = gd.merge(dc, on=["group", "day"], how="left")
gd["ratio"] = (gd.dcf * gd.c * 24 / gd.v.where(gd.v > 1)).clip(0.5, 2).fillna(1.0)
g = g.merge(gd[["group", "day", "ratio"]], on=["group", "day"], how="left")
#R1 = pd.DataFrame({b: pen(g.v14 * (1 + b * (g.ratio - 1))) for b in (0, .05, .1, .2, .3)})
#report1
# (2) hour-split weights
pk = g.ts.dt.hour.between(11, 15)
W = [(.05, .05), (.1, .05), (.05, .1), (.1, .1), (0, .05), (.05, 0), (0, 0)]
R2 = pd.DataFrame({f"pk{a}_ot{b}": pen(np.where(pk, (1-sum(a))*g.p + a[0]*g.a0 + a[1]*g.ph, (1-sum(b))*g.p + b[0]*g.a0 + b[1]*g.ph)) for a in W for b in W})
report("(2) hour-split", R2)
