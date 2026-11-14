"""ECMWF IFS open data (CC BY 4.0) 3-hourly SSRD fixed-box series from the official Google Cloud mirror
(storage.googleapis.com/ecmwf-open-data, same files as data.ecmwf.int / AWS), no site coordinates.
Run 12Z D-2 (published early D-1 KST). Valid D 00 KST = +27 h. Window ending at label hour E (06..21 KST) = steps E+24 -> E+27,
mean W/m2 = (ssrd[E+27] - ssrd[E+24]) / 10800. Boxes 34-38N/126-130E (+N 36-38, S 34-36).
Output hourly rows ts (KST labels 04..21 of D), ex_ssrd/ex_ssrdN/ex_ssrdS. Resumable cache <out>.jsonl.
Usage: ecmwf_00z_fetch.py <out.parquet> <first_day> <last_day>
00Z D-1 variant: valid D 00 KST = +15 h; window ending label E -> steps E+12 -> E+15; also daily 09-15 KST mean (ssrd[30]-ssrd[24])/21600
and AWS/GCP Last-Modified of the +36 h index (publication evidence) per target day in column lm_utc.
"""
import json, os, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
import eccodes, numpy as np, pandas as pd
LOCK = threading.Lock(); WL = threading.Lock()
OUT, D0, D1 = sys.argv[1:4]; CACHE = OUT + ".jsonl"
B = "https://storage.googleapis.com/ecmwf-open-data"
STEPS = [18, 21, 24, 27, 30, 33, 36]
BOX = {"": (34, 38), "N": (36, 38), "S": (34, 36)}


def boxes(run, step):
    u = f"{B}/{run:%Y%m%d}/00z/ifs/0p25/oper/{run:%Y%m%d}000000-{step}h-oper-fc"
    for l in urllib.request.urlopen(u + ".index", timeout=60):
        j = json.loads(l)
        if j["param"] == "ssrd":
            r = urllib.request.Request(u + ".grib2", headers={"Range": f"bytes={j['_offset']}-{j['_offset'] + j['_length'] - 1}"})
            msg = urllib.request.urlopen(r, timeout=120).read()
            with LOCK:
                h = eccodes.codes_new_from_message(msg)
                v, la, lo = eccodes.codes_get_values(h), eccodes.codes_get_array(h, "latitudes"), eccodes.codes_get_array(h, "longitudes")
                eccodes.codes_release(h)
            lo = np.where(lo > 180, lo - 360, lo)
            return {k: float(v[(la >= s) & (la <= n) & (lo >= 126) & (lo <= 130)].mean()) for k, (s, n) in BOX.items()}
    raise ValueError("no ssrd")


def get(day):
    run = day - pd.Timedelta(days=1)
    for t in range(6):
        try:
            acc = {s: boxes(run, s) for s in STEPS}
            lm = urllib.request.urlopen(urllib.request.Request(f"{B}/{run:%Y%m%d}/00z/ifs/0p25/oper/{run:%Y%m%d}000000-36h-oper-fc.index", method="HEAD"), timeout=60).headers.get("Last-Modified")
            rows = []
            for e in range(6, 22, 3):
                L1, L0 = e + 15, e + 12
                vals = {f"e0_ssrd{k}": (acc[L1][k] - acc[L0][k]) / 10800.0 for k in BOX}
                vals.update({f"e0_box{k}": (acc[30][k] - acc[24][k]) / 21600.0 for k in BOX}); vals["lm_utc"] = lm
                end = pd.Timestamp(day).tz_localize("Asia/Seoul") + pd.Timedelta(hours=e)
                for kk in (2, 1, 0):
                    rows.append({"ts": str(end - pd.Timedelta(hours=kk)), **vals})
            with WL, open(CACHE, "a", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r) + chr(10))
            return
        except Exception as ex:  # noqa: BLE001
            err = ex; time.sleep(2 + 3 * t)
    print("FAIL", day.date(), err, flush=True)


done = set()
if os.path.exists(CACHE):
    done = {json.loads(l)["ts"][:10] for l in open(CACHE, encoding="utf-8") if l.strip()}
todo = [d for d in pd.date_range(D0, D1) if str(d.date()) not in done]
print("todo", len(todo), flush=True)
with ThreadPoolExecutor(12) as ex:
    list(ex.map(get, todo))
out = pd.read_json(CACHE, lines=True, convert_dates=False); out["ts"] = pd.to_datetime(out.ts)
out = out.drop_duplicates("ts").sort_values("ts"); out.to_parquet(OUT, index=False)
print(out.ts.dt.date.nunique(), "days", out.shape, out.drop(columns="ts").describe().loc[["count", "mean", "min", "max"]].round(1).to_dict(), flush=True)
