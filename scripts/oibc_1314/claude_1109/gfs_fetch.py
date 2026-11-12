"""Fetch NOAA GFS (public domain, AWS noaa-gfs-bdp-pds) D-1 00Z run fields for target days, site-nearest grid.

Run 00Z of D-1 (09 KST) is published ~13-14 KST, before the D-1 17:00 KST cutoff.
Lead f030 = 24-30 h average = target day 09-15 KST (DSWRF surface, TCDC entire atmosphere).
Usage: gfs_fetch.py <sites.csv> <out.parquet> <YYYY-MM> [<YYYY-MM> ...]   (months of target days)
Output: site_id, day (target date), dswrf, tcdc.
"""
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import eccodes
import numpy as np
import pandas as pd

SITES, OUT, MONTHS = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
B = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
FIELDS = {"DSWRF:surface": "dswrf", "TCDC:entire atmosphere": "tcdc"}
LEAD = 30
s = pd.read_csv(SITES)


def url(run: pd.Timestamp) -> str:
    sub = "atmos/" if run >= pd.Timestamp("2021-03-22") else ""
    return f"{B}/gfs.{run:%Y%m%d}/00/{sub}gfs.t00z.pgrb2.0p25.f{LEAD:03d}"


def get(day: pd.Timestamp):
    u = url(day - pd.Timedelta(days=1))
    for attempt in range(4):
        try:
            idx = urllib.request.urlopen(u + ".idx", timeout=60).read().decode().splitlines()
            off = [int(l.split(":")[1]) for l in idx]
            rec = {}
            for i, l in enumerate(idx):
                for key, nm in FIELDS.items():
                    if f":{key}:" in l and "ave" in l:
                        r = urllib.request.Request(u, headers={"Range": f"bytes={off[i]}-{off[i + 1] - 1}"})
                        h = eccodes.codes_new_from_message(urllib.request.urlopen(r, timeout=60).read())
                        vals = eccodes.codes_get_values(h).reshape(eccodes.codes_get(h, "Nj"), eccodes.codes_get(h, "Ni"))
                        eccodes.codes_release(h)
                        # 0.25 deg global grid, lat 90 -> -90, lon 0 -> 359.75
                        iy = np.rint((90 - s.lat.to_numpy()) / 0.25).astype(int)
                        ix = np.rint(s.lon.to_numpy() / 0.25).astype(int)
                        rec[nm] = vals[iy, ix]
            return pd.DataFrame({"site_id": s.site_id, "day": day, **rec})
        except Exception as e:  # noqa: BLE001
            err = e
    print("FAIL", day.date(), err, flush=True)
    return None


days = [d for m in MONTHS for d in pd.date_range(f"{m}-01", pd.Timestamp(f"{m}-01") + pd.offsets.MonthEnd(0))]
with ThreadPoolExecutor(16) as ex:
    res = [r for r in ex.map(get, days) if r is not None]
out = pd.concat(res, ignore_index=True)
out.to_parquet(OUT, index=False)
print(len(days), "days", out.shape, out[["dswrf", "tcdc"]].describe().round(1).to_dict(), flush=True)
