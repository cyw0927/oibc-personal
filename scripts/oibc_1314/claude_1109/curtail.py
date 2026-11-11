import sys, numpy as np, pandas as pd
D = sys.argv[1]; KST = "Asia/Seoul"
END = pd.Timestamp("2025-02-28 00:00", tz=KST)
s = pd.read_csv(f"{D}/sites.csv").set_index("site_id")
g = pd.read_parquet(f"{D}/train/generation.parquet"); g["ts"] = pd.to_datetime(g.ts).dt.tz_convert(KST)
g = g[(g.ts <= END) & (g.ts >= pd.Timestamp("2023-03-01", tz=KST)) & g.ts.dt.hour.between(11, 14)]
fa = pd.read_parquet(f"{D}/train/forecast_wa.parquet", columns=["wa_station", "round", "ts", "wa_cloud"])
fa = fa[fa["round"] == 17]; fa["ts"] = pd.to_datetime(fa.ts).dt.tz_convert(KST)
g = g.merge(s[["wa_station", "capacity_kw"]], left_on="site_id", right_index=True)
g = g.merge(fa[["wa_station", "ts", "wa_cloud"]], on=["wa_station", "ts"], how="left")
g["cf"] = g.energy / g.capacity_kw
clear = g[g.wa_cloud <= 20]
clear = clear.assign(ym=clear.ts.dt.strftime("%Y-%m"), dow=clear.ts.dt.dayofweek)
# fraction of clear-forecast midday site-hours with near-zero output, by month
t = clear.groupby("ym").agg(n=("cf", "size"), zero=("cf", lambda x: (x < 0.05).mean()), med=("cf", "median"))
print(t.round(3).to_string())
# simultaneous zeros: per hour, capacity share of sites with cf<0.05 under clear forecast
h = clear.groupby("ts").apply(lambda d: pd.Series({"share0": d.capacity_kw[d.cf < 0.05].sum() / d.capacity_kw.sum(), "n": len(d)}))
h = h[h.n >= 50]
h["ym"] = h.index.strftime("%Y-%m"); h["wkend"] = h.index.dayofweek >= 5
print("\nhours with >=30% capacity at ~0 under clear forecast, by month/weekend")
print(h.assign(big=h.share0 >= 0.3).groupby(["ym", "wkend"]).big.sum().unstack().fillna(0).astype(int).to_string())
print("\nworst hours:"); print(h.sort_values("share0", ascending=False).head(15).round(3).to_string())
