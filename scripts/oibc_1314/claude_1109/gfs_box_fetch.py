"""NOAA GFS (public domain, AWS noaa-gfs-bdp-pds) fixed-box daily summaries, no site coordinates.
Run 00Z of D-1 (09 KST, published ~13-14 KST D-1, before the 17:00 cutoff); f030 = 24-30 h average = target day 09-15 KST.
DSWRF surface and TCDC entire atmosphere averaged over boxes 34-38N/126-130E (gf_*), N 36-38N, S 34-36N.
Usage: gfs_box_fetch.py <out.parquet> <first_day> <last_day>
"""
import sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
import threading
import eccodes, numpy as np, pandas as pd
LOCK = threading.Lock()   # eccodes decoding is not thread-safe
OUT, D0, D1 = sys.argv[1:4]
B = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
BOX = {"": (34, 38), "N": (36, 38), "S": (34, 36)}
FIELDS = {"DSWRF:surface": "dsw", "TCDC:entire atmosphere": "tcc"}


def get(day):
    run = day - pd.Timedelta(days=1)
    u = f"{B}/gfs.{run:%Y%m%d}/00/{'atmos/' if run >= pd.Timestamp('2021-03-22') else ''}gfs.t00z.pgrb2.0p25.f030"
    for t in range(10):
        try:
            idx = urllib.request.urlopen(u + ".idx", timeout=60).read().decode().splitlines()
            off = [int(l.split(":")[1]) for l in idx]
            row = {"day": day}
            for i, l in enumerate(idx):
                for key, nm in FIELDS.items():
                    if f":{key}:" in l and "ave" in l:
                        r = urllib.request.Request(u, headers={"Range": f"bytes={off[i]}-{off[i + 1] - 1}"})
                        msg = urllib.request.urlopen(r, timeout=60).read()
                        with LOCK:
                            h = eccodes.codes_new_from_message(msg)
                            v, la, lo = eccodes.codes_get_values(h), eccodes.codes_get_array(h, "latitudes"), eccodes.codes_get_array(h, "longitudes")
                            eccodes.codes_release(h)
                        for bk, (s, n) in BOX.items():
                            m = (la >= s) & (la <= n) & (lo >= 126) & (lo <= 130)
                            row[f"gf_{nm}{bk}"] = float(v[m].mean())
            return row
        except Exception as e:  # noqa: BLE001
            err = e; time.sleep(2 + 3 * t)
    print("FAIL", day.date(), err, flush=True)


with ThreadPoolExecutor(8) as ex:
    rows = [r for r in ex.map(get, pd.date_range(D0, D1)) if r]
out = pd.DataFrame(rows); out.to_parquet(OUT, index=False)
print(len(out), "days", out.drop(columns="day").describe().round(1).to_dict(), flush=True)
