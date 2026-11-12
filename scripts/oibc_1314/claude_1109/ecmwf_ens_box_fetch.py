"""ECMWF ENS open data (CC BY 4.0) fixed-box ssrd for target days, members cf + pf1..pf9 (10), no site coordinates.
Run 12Z D-2, (+42h - +36h)/21600 = target day 09-15 KST mean W/m2, box 34-38N/126-130E.
Output per day: ens_med, ens_mean, ens_sd, ens_p10, ens_p90 over the 10 members.
Usage: ecmwf_ens_box_fetch.py <out.parquet> <first_day> <last_day>
"""
import json, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
import eccodes, numpy as np, pandas as pd
LOCK = threading.Lock()   # eccodes decoding is not thread-safe
OUT, D0, D1 = sys.argv[1:4]
B = "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com"
MEM = {("cf", 0)} | {("pf", i) for i in range(1, 10)}


def boxes(run, step):
    u = f"{B}/{run:%Y%m%d}/12z/ifs/0p25/enfo/{run:%Y%m%d}120000-{step}h-enfo-ef"
    out = {}
    for l in urllib.request.urlopen(u + ".index", timeout=60):
        j = json.loads(l)
        if j["param"] == "ssrd" and (j.get("type"), int(j.get("number", 0))) in MEM:
            r = urllib.request.Request(u + ".grib2", headers={"Range": f"bytes={j['_offset']}-{j['_offset'] + j['_length'] - 1}"})
            msg = urllib.request.urlopen(r, timeout=90).read()
            with LOCK:
                h = eccodes.codes_new_from_message(msg)
                v, la, lo = eccodes.codes_get_values(h), eccodes.codes_get_array(h, "latitudes"), eccodes.codes_get_array(h, "longitudes")
                eccodes.codes_release(h)
            lo = np.where(lo > 180, lo - 360, lo)
            out[(j.get("type"), int(j.get("number", 0)))] = float(v[(la >= 34) & (la <= 38) & (lo >= 126) & (lo <= 130)].mean())
    return out


def get(day):
    run = day - pd.Timedelta(days=2)
    for t in range(10):
        try:
            a, b = boxes(run, 36), boxes(run, 42)
            x = np.array([(b[k] - a[k]) / 21600.0 for k in a if k in b])
            if len(x) < 8:
                raise ValueError(f"members {len(x)}")
            return {"day": day, "ens_n": len(x), "ens_med": np.median(x), "ens_mean": x.mean(), "ens_sd": x.std(),
                    "ens_p10": np.quantile(x, 0.1), "ens_p90": np.quantile(x, 0.9)}
        except Exception as e:  # noqa: BLE001
            err = e; time.sleep(3 + 4 * t)
    print("FAIL", day.date(), err, flush=True)


with ThreadPoolExecutor(4) as ex:
    rows = [r for r in ex.map(get, pd.date_range(D0, D1)) if r]
out = pd.DataFrame(rows); out.to_parquet(OUT, index=False)
print(len(out), "days", out.drop(columns="day").describe().round(1).to_dict(), flush=True)
