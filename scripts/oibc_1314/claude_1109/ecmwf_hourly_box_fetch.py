"""ECMWF IFS open data (CC BY 4.0) 3-hourly fixed-box ssrd for target days, no site coordinates.
Run 12Z D-2 (published D-1 early morning KST). Steps 30..48 (every 3 h) -> 3 h mean W/m2 for KST windows
03-06, 06-09, 09-12, 12-15, 15-18, 18-21 of target day D. Box 34-38N/126-130E.
Output: ts (hour-ending KST, right-closed hourly labels 04..21 of D), ech_box (mean of the 3 h window containing the hour).
Usage: ecmwf_hourly_box_fetch.py <out.parquet> <first_day> <last_day>
"""
import json, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
import threading
import eccodes, numpy as np, pandas as pd
LOCK = threading.Lock()   # eccodes decoding is not thread-safe
OUT, D0, D1 = sys.argv[1:4]
B = "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com"
STEPS = list(range(30, 49, 3))


def box(run, step):
    u = f"{B}/{run:%Y%m%d}/12z/ifs/0p25/oper/{run:%Y%m%d}120000-{step}h-oper-fc"
    for l in urllib.request.urlopen(u + ".index", timeout=60):
        j = json.loads(l)
        if j["param"] == "ssrd":
            r = urllib.request.Request(u + ".grib2", headers={"Range": f"bytes={j['_offset']}-{j['_offset'] + j['_length'] - 1}"})
            msg = urllib.request.urlopen(r, timeout=60).read()
            with LOCK:
                h = eccodes.codes_new_from_message(msg)
                v, la, lo = eccodes.codes_get_values(h), eccodes.codes_get_array(h, "latitudes"), eccodes.codes_get_array(h, "longitudes")
                eccodes.codes_release(h); lo = np.where(lo > 180, lo - 360, lo)
            return float(v[(la >= 34) & (la <= 38) & (lo >= 126) & (lo <= 130)].mean())
    raise KeyError("ssrd")


def get(day):
    run = day - pd.Timedelta(days=2)
    for t in range(10):
        try:
            acc = [box(run, s) for s in STEPS]
            rows = []
            for i in range(len(STEPS) - 1):
                w = (acc[i + 1] - acc[i]) / 10800.0
                start = pd.Timestamp(day).tz_localize("Asia/Seoul") + pd.Timedelta(hours=3 + 3 * i)   # 03, 06, ... KST
                for k in (1, 2, 3):
                    rows.append({"ts": start + pd.Timedelta(hours=k), "ech_box": w})
            return rows
        except Exception as e:  # noqa: BLE001
            err = e; time.sleep(2 + 3 * t)
    print("FAIL", day.date(), err, flush=True)


with ThreadPoolExecutor(4) as ex:
    rows = [x for r in ex.map(get, pd.date_range(D0, D1)) if r for x in r]
out = pd.DataFrame(rows); out.to_parquet(OUT, index=False)
print(out.ts.dt.date.nunique(), "days", len(out), out.ech_box.describe().round(1).to_dict(), flush=True)
