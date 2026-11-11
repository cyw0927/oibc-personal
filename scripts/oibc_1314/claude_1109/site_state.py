import sys, numpy as np, pandas as pd
D = sys.argv[1]
END = pd.Timestamp("2025-02-28 00:00", tz="Asia/Seoul")   # pre-HOLDOUT only
s = pd.read_csv(f"{D}/sites.csv")
g = pd.read_parquet(f"{D}/train/generation.parquet")
g["ts"] = pd.to_datetime(g.ts).dt.tz_convert("Asia/Seoul")
g = g[g.ts <= END]
g = g[g.ts.dt.hour.between(11, 15)]
g["ym"] = g.ts.dt.strftime("%Y-%m")
first = g.groupby("site_id").ts.min()
cap = s.set_index("site_id").capacity_kw
g["cf"] = g.energy / g.site_id.map(cap)
months = [m for m in sorted(g.ym.unique()) if m >= "2024-06"]
hours_per_month = {m: g[g.ym == m].ts.nunique() for m in months}
rows = g[g.ym.isin(months)].groupby(["site_id", "ym"]).agg(n=("cf", "size"), cf=("cf", "median"), zero=("cf", lambda x: (x < 0.01).mean()))
cov = rows.n.unstack() / pd.Series(hours_per_month)
cfm = rows.cf.unstack()
# capacity-weighted share of midday site-hours that are missing, among sites already commissioned before 2024-06
old = first[first < pd.Timestamp("2024-06-01", tz="Asia/Seoul")].index
print("sites commissioned before 2024-06:", len(old), "cap", round(cap[old].sum()))
w = cap.reindex(cov.index)
for m in months:
    c = cov[m].reindex(old).fillna(0)
    print(m, "cap-weighted midday coverage(old sites)", round((c * cap[old]).sum() / cap[old].sum(), 4),
          "| sites cov<0.5:", int((c < 0.5).sum()), "cap", round(cap[old][c < 0.5].sum()))
# sites whose last 2 months (2025-01,02) look dead or missing
last = ["2025-01", "2025-02"]
z = rows.zero.unstack()
bad = pd.DataFrame({"cap": cap, "cov_last": cov[last].mean(axis=1), "zero_last": z[last].mean(axis=1),
                    "cf_last": cfm[last].mean(axis=1), "cf_prev": cfm[["2024-01", "2024-02"]].mean(axis=1) if "2024-01" in cfm else np.nan})
bad = bad[(bad.cov_last < 0.7) | (bad.zero_last > 0.3)].sort_values("cap", ascending=False)
print("\nsuspect sites (Jan-Feb 2025 midday):", len(bad), "cap", round(bad.cap.sum()))
print(bad.round(3).head(25).to_string())
