"""Day-context features for the #1314 group-direct model (pre-registered 2026-11-09, H1).

All inputs are D-1 issued forecasts already in the group feature table (rounds 10/17 mean), so every
feature is available at the D-1 17:00 KST cutoff: other hours of the target day come from the same
forecast issue, previous-day values from the D-2 issue. Shifts are by clock time (not row order).
"""
import numpy as np
import pandas as pd

V = ["wa_cloud", "wb_cloud", "cs_ab", "wb_uv_idx", "wb_precip_prob", "wb_vis", "wb_rel_hum", "wa_humidity"]
PREV = ["wa_cloud", "wb_cloud", "wa_snow", "wb_snow", "wb_precip_prob"]
SHIFTS = (-2, -1, 1, 2)
AGGS = ("mean", "min", "max", "std")


def add_ctx(ft: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return (frame with context columns, list of new column names). Needs columns group, ts, cosz + V + PREV."""
    ft = ft.sort_values(["group", "ts"]).reset_index(drop=True)
    day = (ft.ts - pd.Timedelta(hours=1)).dt.normalize()
    ft["_day"] = day
    new = []
    key = ft.set_index(["group", "ts"])
    for v in V:
        for s in SHIFTS:
            c = f"{v}_s{s}"
            idx = pd.MultiIndex.from_arrays([ft.group, ft.ts - pd.Timedelta(hours=s)])
            ft[c] = key[v].reindex(idx).values
            new.append(c)
    lit = ft.cosz > 0.05
    for v in V:
        dl = ft[v].where(lit).groupby([ft.group, ft._day])
        for agg in AGGS:
            c = f"{v}_dl{agg}"
            ft[c] = dl.transform(agg)
            new.append(c)
    for v in PREV:
        src = ft[v].where(lit) if ("cloud" in v or "prob" in v) else ft[v]
        dm = src.groupby([ft.group, ft._day]).mean()
        idx = pd.MultiIndex.from_arrays([ft.group, ft._day - pd.Timedelta(days=1)])
        c = f"{v}_prevday"
        ft[c] = dm.reindex(idx).values
        new.append(c)
    ft["cs_rel"] = ft.cs_ab / ft.cosz.where(ft.cosz > 0.02)
    new.append("cs_rel")
    return ft.drop(columns="_day"), new
