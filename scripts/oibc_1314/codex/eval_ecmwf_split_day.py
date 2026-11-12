'''Test if fixed-box ECMWF radiation timing improves v8-box on forward-only folds.'''
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from email.utils import parsedate_to_datetime
from hashlib import sha256
import json
from pathlib import Path
from threading import Lock
from uuid import uuid4
import time

import eccodes as ec
import numpy as np
import pandas as pd
import requests

BUCKETS = ("https://s3.eu-central-1.amazonaws.com/ecmwf-forecasts",
           "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com")
KST = "Asia/Seoul"
KEY = ["fold", "day", "group"]
STEPS = (36, 39, 42)
SHRINKAGE = .5
MAX_SHIFT = .05
GRIB_LOCK = Lock()


def read_field(run, step, day):
    stamp = run.strftime("%Y%m%d")
    key = f"{stamp}/12z/ifs/0p25/oper/{stamp}120000-{step}h-oper-fc"
    cutoff = day.tz_localize(KST) - pd.Timedelta(hours=7)
    for attempt in range(6):
        root = f"{BUCKETS[attempt % len(BUCKETS)]}/{key}"
        query = "" if attempt < 2 else f"?oibc_audit={uuid4().hex[:10]}"
        try:
            ix = requests.get(root+".index"+query, timeout=12)
            ix.raise_for_status()
            entry, = [v for line in ix.text.splitlines() if (v := json.loads(line))["param"] == "ssrd"]
            offset, length = entry["_offset"], entry["_length"]
            raw = requests.get(root+".grib2"+query, timeout=15,
                               headers={"Range": f"bytes={offset}-{offset+length-1}"})
            raw.raise_for_status()
            if raw.status_code != 206 or len(raw.content) != length:
                raise ValueError("Bad GRIB range response")
            latest = max(pd.Timestamp(parsedate_to_datetime(x.headers["Last-Modified"])) for x in (ix, raw))
            if latest >= cutoff:
                raise ValueError("AWS object timestamp after cutoff")
            with GRIB_LOCK:
                handle = ec.codes_new_from_message(raw.content)
                try:
                    def get(k):
                        return ec.codes_get(handle, k)
                    expected = run + pd.Timedelta(hours=12+step)
                    actual = (get("dataDate"), get("dataTime"), get("endStep"), get("validityDate"), get("validityTime"))
                    if actual != (int(stamp), 1200, step, int(expected.strftime("%Y%m%d")), expected.hour*100):
                        raise ValueError(f"Unexpected forecast clock: {actual}")
                    if get("shortName") != "ssrd" or get("stepType") != "accum":
                        raise ValueError("Unexpected forecast field")
                    lat = ec.codes_get_array(handle,"latitudes")
                    lon = ec.codes_get_array(handle,"longitudes")
                    mask = (lat>=34)&(lat<=38)&(lon>=126)&(lon<=130)
                    if mask.sum() != 289:
                        raise ValueError("Fixed box grid mismatch")
                    return (ec.codes_get_values(handle)[mask].mean(), latest.isoformat(), get("startStep"))
                finally:
                    ec.codes_release(handle)
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            time.sleep(.5 * (attempt+1))
    raise RuntimeError(f"{day.date()} step {step}: {last_error}")


def fetch(days, reference, folder):
    folder.mkdir(exist_ok=True, parents=True)
    def one(day):
        cache = folder / f"{day:%Y-%m-%d}.json"
        if cache.exists():
            return json.loads(cache.read_text(encoding="utf-8"))
        run = day-pd.Timedelta(days=2)
        a,b,c = [read_field(run,s,day) for s in STEPS]
        starts = [field[2] for field in (a,b,c)]
        if starts != [0,0,0] and not (starts == list(STEPS) and run < pd.Timestamp("2024-11-12")):
            raise ValueError(f"Unexpected accumulation origin on {day}: {starts}")
        am, pm = (b[0]-a[0])/10800, (c[0]-b[0])/10800
        if min(am,pm) < -1 or abs((am+pm)/2-reference.loc[day,"ec_box"]) > 1e-7:
            raise ValueError(f"Original 6h mean mismatch on {day.date()}")
        row = dict(day=day.strftime("%Y-%m-%d"), am_wm2=am, pm_wm2=pm,
                   aws_modified=max(a[1],b[1],c[1]), starts=starts)
        cache.write_text(json.dumps(row)+"\n",encoding="utf-8")
        return row

    rows, errors = [], {}
    with ThreadPoolExecutor(12) as pool:
        futures = {pool.submit(one,d):d for d in days}
        for i,f in enumerate(as_completed(futures),1):
            try:
                rows.append(f.result())
            except (RuntimeError, ValueError) as e:
                errors[str(futures[f].date())] = str(e)
            if i%20==0 or i==len(days):
                print(f"ECMWF {i}/{len(days)}, obtained {len(rows)}, errors {len(errors)}",flush=True)
    return pd.DataFrame(rows), errors


def fit_and_predict(train, test):
    controls = ["base_pm_share","day_sin","day_cos"]
    features = controls+["radiation_pm_share"]
    def predict(cols):
        x = train[cols].to_numpy()
        center = x.mean(axis=0)
        scale = x.std(axis=0)
        scale[scale == 0] = 1
        x = (x-center)/scale
        y = train.target_shift.to_numpy()
        coefficients = np.linalg.solve(x.T@x + 10*np.eye(x.shape[1]), x.T@(y-y.mean()))
        return y.mean() + ((test[cols].to_numpy()-center)/scale)@coefficients
    return predict(controls),predict(features)


def score(rows, shifts, baseline):
    merged = rows.merge(shifts.rename("shift").reset_index(),on=KEY,how="left",validate="many_to_one")
    delta = merged["shift"].fillna(0).clip(-MAX_SHIFT,MAX_SHIFT)*SHRINKAGE
    pm = merged.base_pm_share
    factors = np.where(merged.ts.dt.hour.ge(13),1+delta/pm,1-delta/(1-pm))
    predicted = merged.final_pred*factors
    assert np.isfinite(predicted).all() and (predicted>=0).all()
    if len(shifts):
        check = merged.assign(predicted=predicted).groupby(KEY)[["predicted","final_pred"]].sum()
        assert np.allclose(check.predicted,check.final_pred,rtol=1e-12,atol=1e-8)
    new = (.1*merged.smp*merged.k*(predicted-merged.energy).abs()).sum()
    return float(new+baseline-rows.penalty.sum())


def evaluate(frame, weather):
    frame["penalty"] = .1*frame.smp*frame.k*(frame.final_pred-frame.energy).abs()
    # Timestamp labels mark the end of each hour: 10–12 and 13–15.
    rows = frame[frame.ts.dt.hour.between(10,15)].copy()
    rows["pm"] = rows.ts.dt.hour.ge(13)
    sums = rows.groupby(KEY+["pm"]).agg(p=("final_pred","sum"),a=("energy","sum"),n=("energy","size")).unstack("pm")
    sums.columns = [f"{col}_{'pm' if pm else 'am'}" for col,pm in sums]
    groups = sums.reset_index()
    valid = groups.n_am.eq(3)&groups.n_pm.eq(3)&groups.p_am.gt(0)&groups.p_pm.gt(0)
    groups = groups[valid].merge(weather,on="day",how="inner",validate="many_to_one")
    groups["base_pm_share"] = groups.p_pm/(groups.p_pm+groups.p_am)
    groups["radiation_pm_share"] = groups.pm_wm2/(groups.am_wm2+groups.pm_wm2)
    angle = 2*np.pi*groups.day.dt.dayofyear/365.25
    groups["day_sin"], groups["day_cos"] = np.sin(angle),np.cos(angle)
    groups["target_shift"] = groups.a_pm/(groups.a_pm+groups.a_am).replace(0,np.nan)-groups.base_pm_share
    groups = groups.set_index(KEY).sort_index()
    rows = rows.merge(groups[["base_pm_share"]].reset_index(),on=KEY,how="inner",validate="many_to_one")
    out=[]
    for month, ref in frame.groupby("fold",sort=True):
        train = groups[groups.index.get_level_values("fold")<month].dropna(subset=["target_shift"])
        test = groups.xs(month,level="fold",drop_level=False)
        base = float(ref.penalty.sum())
        result = dict(month=month,baseline=base,adjusted=base,gate=False,
                      train_days=train.index.get_level_values("day").nunique(),
                      evaluated_group_days=len(test))
        if result["train_days"]>=28:
            days = train.index.get_level_values("day").unique().sort_values()
            pivot = days[max(14,int(len(days)*.65))]
            history = train[train.index.get_level_values("day")<pivot]
            gate = train[train.index.get_level_values("day")>=pivot]
            ctl, wx = fit_and_predict(history,gate)
            mae = lambda p,t: float(np.mean(abs(p-t.target_shift.to_numpy())))
            prior_rows = rows.merge(gate.index.to_frame(index=False),on=KEY,how="inner",validate="many_to_one")
            old = float(prior_rows.penalty.sum())
            changed = score(prior_rows,pd.Series(wx,index=gate.index),old)
            result.update(gate_days=gate.index.get_level_values("day").nunique(),
                          prior_control_mae=mae(ctl,gate),prior_weather_mae=mae(wx,gate),
                          prior_base_penalty=old,prior_adjusted_penalty=changed)
            result["gate"] = result["prior_weather_mae"]<result["prior_control_mae"] and changed<old
            ctl, wx = fit_and_predict(train,test)
            actual_test = test.target_shift.notna().to_numpy()
            result.update(holdout_control_mae=mae(ctl[actual_test],test[actual_test]),
                          holdout_weather_mae=mae(wx[actual_test],test[actual_test]))
            if result["gate"]:
                result["adjusted"] = score(rows[rows.fold.eq(month)],pd.Series(wx,index=test.index),base)
        out.append(result)
        print(result,flush=True)
    return out


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ("dump","box","output"):
        parser.add_argument(arg,type=Path)
    parser.add_argument("--cached-only",action="store_true",help="Use fetched days and leave unavailable days at baseline")
    args=parser.parse_args()
    f=pd.read_csv(args.dump)
    f["day"]=pd.to_datetime(f.day)
    f["ts"]=pd.to_datetime(f.ts,utc=True).dt.tz_convert(KST)
    assert len(f)==3749 and not f.duplicated(["fold","group","ts"]).any()
    e=pd.read_parquet(args.box).set_index("day")
    assert sha256(args.box.read_bytes()).hexdigest().startswith("7afc64ee")
    days = sorted(f.day.unique())
    raw_folder = args.output/"raw_means"
    if args.cached_only:
        wx = pd.DataFrame([json.loads(p.read_text(encoding="utf-8")) for p in raw_folder.glob("*.json")])
        missing = sorted({d.strftime("%Y-%m-%d") for d in days}-set(wx.day))
    else:
        wx,errors=fetch(days,e,raw_folder)
        if errors:
            print(json.dumps(errors,indent=2),flush=True)
            raise SystemExit("Coverage incomplete; rerun to retry missing dates")
        missing = []
    wx["day"]=pd.to_datetime(wx.day)
    folds=evaluate(f,wx)
    summary=dict(days=len(wx),expected_days=len(days),missing_days=missing,steps=STEPS,dump_sha256=sha256(args.dump.read_bytes()).hexdigest(),folds=folds,
                 total_base=sum(x["baseline"] for x in folds),total_adjusted=sum(x["adjusted"] for x in folds))
    assert abs(summary["total_base"]-73639060.48)<.01
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
