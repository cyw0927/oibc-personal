"""ECMWF IFS open data (CC BY 4.0) fixed-box daily summaries, independent of site coordinates.

Run 12Z D-2, ssrd (+42h - +36h)/21600 = mean W/m2 over target day 09-15 KST, averaged over fixed lat/lon boxes:
box 34-38N 126-130E (whole region), boxN 36-38N, boxS 34-36N (same longitudes). No site lat/lon is used.
Usage: ecmwf_box_fetch.py <out.parquet> <first_day> <last_day>
"""
import json, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
import eccodes, numpy as np, pandas as pd

OUT, D0, D1 = sys.argv[1], sys.argv[2], sys.argv[3]
B = "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com"
BOX = {"ec_box": (34, 38), "ec_boxN": (36, 38), "ec_boxS": (34, 36)}


def field(run, step):
    u = f"{B}/{run:%Y%m%d}/12z/ifs/0p25/oper/{run:%Y%m%d}120000-{step}h-oper-fc"
    for l in urllib.request.urlopen(u + ".index", timeout=60):
        j = json.loads(l)
        if j["param"] == "ssrd":
            r = urllib.request.Request(u + ".grib2", headers={"Range": f"bytes={j['_offset']}-{j['_offset'] + j['_length'] - 1}"})
            h = eccodes.codes_new_from_message(urllib.request.urlopen(r, timeout=60).read())
            v, la, lo = (eccodes.codes_get_values(h), eccodes.codes_get_array(h, "latitudes"), eccodes.codes_get_array(h, "longitudes"))
            eccodes.codes_release(h)
            return v, la, np.where(lo > 180, lo - 360, lo)
    raise KeyError("ssrd")


def get(day):
    run = day - pd.Timedelta(days=2)
    for t in range(6):
        try:
            (a, la, lo), (b, _, _) = field(run, 36), field(run, 42)
            x = (b - a) / 21600.0
            row = {"day": day}
            for k, (s, n) in BOX.items():
                m = (la >= s) & (la <= n) & (lo >= 126) & (lo <= 130)
                row[k] = float(x[m].mean())
            return row
        except Exception as e:  # noqa: BLE001
            err = e; time.sleep(2 + 3 * t)
    print("FAIL", day.date(), err, flush=True)


days = pd.date_range(D0, D1)
with ThreadPoolExecutor(6) as ex:
    rows = [r for r in ex.map(get, days) if r]
out = pd.DataFrame(rows); out.to_parquet(OUT, index=False)
print(len(days), "days ->", len(out), out.describe().round(1).to_dict(), flush=True)
