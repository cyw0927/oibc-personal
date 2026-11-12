"""Fetch ECMWF IFS open data (CC BY 4.0, AWS ecmwf-forecasts) for target days, site-nearest grid.

Run 12Z of D-2 (21 KST D-2, published ~D-1 early morning KST) -> safely before the D-1 17:00 KST cutoff.
Leads 36 h / 42 h = 00Z / 06Z of D = 09 / 15 KST. ssrd accumulated -> mean W/m2 over 09-15 KST.
Usage: ecmwf_fetch.py <sites.csv> <out.parquet> <YYYY-MM> [...]
"""
import json, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import eccodes, numpy as np, pandas as pd

SITES, OUT, MONTHS = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
B = "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com"
s = pd.read_csv(SITES)


def base(run):
    sub = "ifs/0p25" if run >= pd.Timestamp("2024-02-01") else "0p4-beta"
    return f"{B}/{run:%Y%m%d}/12z/{sub}/oper/{run:%Y%m%d}120000"


def field(run, step, param="ssrd"):
    u = f"{base(run)}-{step}h-oper-fc"
    for l in urllib.request.urlopen(u + ".index", timeout=60):
        j = json.loads(l)
        if j["param"] == param:
            r = urllib.request.Request(u + ".grib2", headers={"Range": f"bytes={j['_offset']}-{j['_offset'] + j['_length'] - 1}"})
            h = eccodes.codes_new_from_message(urllib.request.urlopen(r, timeout=60).read())
            ni, nj = eccodes.codes_get(h, "Ni"), eccodes.codes_get(h, "Nj")
            la0, lo0 = eccodes.codes_get(h, "latitudeOfFirstGridPointInDegrees"), eccodes.codes_get(h, "longitudeOfFirstGridPointInDegrees")
            dx = eccodes.codes_get(h, "iDirectionIncrementInDegrees")
            v = eccodes.codes_get_values(h).reshape(nj, ni); eccodes.codes_release(h)
            iy = np.rint((la0 - s.lat.to_numpy()) / dx).astype(int)
            ix = np.rint(((s.lon.to_numpy() - lo0) % 360) / dx).astype(int)
            return v[iy, ix]
    raise KeyError(param)


def get(day):
    run = day - pd.Timedelta(days=2)
    for t in range(6):
        try:
            a, b = field(run, 36), field(run, 42)
            return pd.DataFrame({"site_id": s.site_id, "day": day, "ssrd": (b - a) / 21600.0})
        except Exception as e:  # noqa: BLE001
            err = e; time.sleep(2 + 3 * t)
    print("FAIL", day.date(), err, flush=True)


days = [d for m in MONTHS for d in pd.date_range(f"{m}-01", pd.Timestamp(f"{m}-01") + pd.offsets.MonthEnd(0))]
with ThreadPoolExecutor(12) as ex:
    res = [r for r in ex.map(get, days) if r is not None]
out = pd.concat(res, ignore_index=True); out.to_parquet(OUT, index=False)
print(len(days), "days", out.shape, out.ssrd.describe().round(1).to_dict(), flush=True)
