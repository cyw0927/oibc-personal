"""NOAA GFS 1.0 deg (public domain, AWS noaa-gfs-bdp-pds) multi-variable 3-hourly fixed-box series, no site coordinates.
Run 00Z of D-1 (published ~12:40-13:00 KST D-1, before the 17:00 cutoff). Leads 21..36 -> target day KST windows
03-06, 06-09, 09-12, 12-15, 15-18, 18-21. Average fields (DSWRF, TCDC, LCDC, MCDC, HCDC, PRATE): 3 h means (6 h aves
de-accumulated); instant fields (TMP 2m, RH 2m, PWAT) at the window end. Box 34-38N/126-130E (+N/S halves for dsw/tcc).
Output hourly rows: ts (KST, hour-ending labels 04..21 of D), gx_* columns = value of the 3 h window containing the hour.
Usage: gfs_rich_fetch.py <out.parquet> <first_day> <last_day> [cycle=0]
cycle 0 = 00Z D-1 (default). cycle 18/12 = 18Z/12Z D-2 (published D-1 ~08:40 / D-2 ~02:40 KST), leads shifted by 24-cycle so the
same target-day windows are covered (time-lagged ensemble members). Output columns keep the gx_ names. extra=1 fetches a second variable set (tccbl, usw, sunsd, vis, cape).
"""
import sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
import eccodes, numpy as np, pandas as pd
LOCK = threading.Lock()   # eccodes decoding is not thread-safe
OUT, D0, D1 = sys.argv[1:4]
CYC = int(next((a[6:] for a in sys.argv[4:] if a.startswith("cycle=")), "0"))
SHIFT = 0 if CYC == 0 else 24 - CYC
B = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
AVE = {"DSWRF:surface": "dsw", "TCDC:entire atmosphere": "tcc", "LCDC:low cloud layer": "lcc", "MCDC:middle cloud layer": "mcc",
       "HCDC:high cloud layer": "hcc", "PRATE:surface": "prate"}
INST = {"TMP:2 m above ground": "t2", "RH:2 m above ground": "rh2", "PWAT:entire atmosphere (considered as a single layer)": "pwat"}
SPLIT = {"dsw", "tcc"}
QUAD = any(a == "quad=1" for a in sys.argv[4:])   # 2x2 deg quadrant boxes (NW/NE/SW/SE of 34-38N/126-130E) for dsw, tcc, lcc
if QUAD:
    AVE = {"DSWRF:surface": "dsw", "TCDC:entire atmosphere": "tcc", "LCDC:low cloud layer": "lcc"}
    INST = {}
if any(a == "extra=1" for a in sys.argv[4:]):   # second variable set: sunshine, visibility, CAPE, boundary-layer cloud, upward SW
    AVE = {"TCDC:boundary layer cloud layer": "tccbl", "USWRF:surface": "usw"}
    INST = {"SUNSD:surface": "sunsd", "VIS:surface": "vis", "CAPE:surface": "cape"}
    SPLIT = set()
LEADS = [21 + SHIFT, 24 + SHIFT, 27 + SHIFT, 30 + SHIFT, 33 + SHIFT, 36 + SHIFT]


def lead_fields(run, L):
    u = f"{B}/gfs.{run:%Y%m%d}/{CYC:02d}/{'atmos/' if run >= pd.Timestamp('2021-03-22') else ''}gfs.t{CYC:02d}z.pgrb2.1p00.f{L:03d}"
    idx = urllib.request.urlopen(u + ".idx", timeout=60).read().decode().splitlines()
    off = [int(l.split(":")[1]) for l in idx]
    out = {}
    for i, l in enumerate(idx):
        f = l.split(":")
        key, kind = f"{f[3]}:{f[4]}", f[5]
        nm = AVE.get(key) if "ave" in kind else INST.get(key) if kind.endswith("hour fcst") and "ave" not in kind else None
        if nm is None or i + 1 >= len(off):
            continue
        msg = urllib.request.urlopen(urllib.request.Request(u, headers={"Range": f"bytes={off[i]}-{off[i + 1] - 1}"}), timeout=60).read()
        with LOCK:
            h = eccodes.codes_new_from_message(msg)
            v, la, lo = eccodes.codes_get_values(h), eccodes.codes_get_array(h, "latitudes"), eccodes.codes_get_array(h, "longitudes")
            eccodes.codes_release(h)
        if QUAD:
            for bk, (s, n, w, e) in {"NW": (36, 38, 126, 128), "NE": (36, 38, 128, 130), "SW": (34, 36, 126, 128), "SE": (34, 36, 128, 130)}.items():
                out[nm + bk] = float(v[(la >= s) & (la <= n) & (lo >= w) & (lo <= e)].mean())
            continue
        for bk, (s, n) in ({"": (34, 38), "N": (36, 38), "S": (34, 36)} if nm in SPLIT else {"": (34, 38)}).items():
            out[nm + bk] = float(v[(la >= s) & (la <= n) & (lo >= 126) & (lo <= 130)].mean())
    return out


def get(day):
    run = day - pd.Timedelta(days=1 if CYC == 0 else 2)
    for t in range(8):
        try:
            raw = {L: lead_fields(run, L) for L in LEADS}
            rows = []
            for L in LEADS:
                vals = {}
                for k, x in raw[L].items():
                    base = k[:-2] if QUAD else k.rstrip("NS") if k[-1] in "NS" and k[:-1] in SPLIT else k
                    if base in AVE.values() and L % 6 == 0:          # GFS aves reset every 6 h of lead for every cycle          # 6 h ave -> 3 h mean of the second half
                        prev = raw[L - 3].get(k)
                        x = 2 * x - prev if prev is not None else np.nan
                    vals[f"gx_{k}"] = x
                end = pd.Timestamp(day).tz_localize("Asia/Seoul") + pd.Timedelta(hours=L - SHIFT - 15)
                for kk in (2, 1, 0):
                    rows.append({"ts": end - pd.Timedelta(hours=kk), **vals})
            return rows
        except Exception as e:  # noqa: BLE001
            err = e; time.sleep(2 + 3 * t)
    print("FAIL", day.date(), err, flush=True)


with ThreadPoolExecutor(10) as ex:
    rows = [x for r in ex.map(get, pd.date_range(D0, D1)) if r for x in r]
out = pd.DataFrame(rows).sort_values("ts"); out.to_parquet(OUT, index=False)
print(out.ts.dt.date.nunique(), "days", out.shape, out.drop(columns="ts").describe().loc[["count", "mean", "min", "max"]].round(1).to_dict(), flush=True)
