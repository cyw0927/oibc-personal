"""Evaluate a solar envelope times learned hourly and daily weather attenuation."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from eval_v7_full_heldout import GROUP_KEYS, KST, MONTHS, require, time_bounds
from eval_round_disagreement import baseline

SOLAR = ["group_id", "hour", "doy_sin", "doy_cos", "cosz", "cosz_max"]
DAY_KEYS = ["group", "day"]
WEATHER = ["wa_cloud", "wb_cloud", "wa_rain", "wb_precip_prob", "wb_uv_idx",
           "wa_humidity", "wb_rel_hum", "cs_a", "cs_b"]


def regressor(objective, **extra):
    return LGBMRegressor(objective=objective, n_estimators=350, learning_rate=0.045,
                        num_leaves=15, min_child_samples=30, n_jobs=4, random_state=0, verbosity=-1, **extra)


def daily_features(frame):
    noon = frame[frame.ts.dt.hour.between(11, 15)]
    full = frame.groupby(DAY_KEYS)
    return full[SOLAR[2:4] + ["group_id"]].first().join(
        full[["envelope", "cosz"]].sum()).join(
        full[WEATHER].mean().add_suffix("_day")).join(
        noon.groupby(DAY_KEYS)[WEATHER].mean().add_suffix("_noon"))


def predict_month(frame, month, weather_features):
    start, end = time_bounds(month)
    history = frame[(frame.ts < start) & (frame.cosz_max > 0.03)].copy()
    future = frame[frame.ts.between(start, end)].copy()
    require(history.ts.max() < future.ts.min(), "Training cutoff overlaps evaluation")
    eligible = history.cf.notna() & history.smp.notna() & history.coverage.ge(0.30)
    clear = history[eligible & history.wa_cloud.le(35) & history.wb_cloud.le(35)]
    require(clear.groupby("group").size().ge(50).all(), "Insufficient clear conditions")
    envelope = regressor("quantile", alpha=0.85).fit(clear[SOLAR], clear.cf)
    for data in (history, future):
        data["envelope"] = np.where(data.cosz_max > 0.03,
                                    envelope.predict(data[SOLAR]).clip(0.01, 1), 0)
    train = history[eligible]
    features = [*weather_features, "envelope"]
    weight = train.coverage * train.k * train.smp.clip(lower=1)
    ratio = (train.cf / train.envelope).clip(0, 1.35)
    attenuation = regressor("regression_l1").fit(train[features], ratio, sample_weight=weight * train.envelope)
    direct = regressor("regression_l1").fit(train[features], train.cf, sample_weight=weight)
    future["hourly_cf"] = future.envelope * attenuation.predict(future[features]).clip(0, 1.35)
    future["direct_cf"] = np.where(future.cosz_max > 0.03, direct.predict(future[features]).clip(0, 1), 0)

    # A separate day model predicts the lost fraction for the whole lit period.
    # Test-day features come entirely from forecasts and solar geometry.
    hdays = daily_features(history)
    fdays = daily_features(future[future.cosz_max > 0.03])
    target = train.groupby(DAY_KEYS).agg(cf=("cf", "sum"), labelled=("cf", "size"))
    target = target.join(history.groupby(DAY_KEYS).size().rename("expected"))
    target = target[target.labelled.eq(target.expected)].join(hdays)
    target["ratio"] = (target.cf / target.envelope).clip(0, 1.35)
    daily = regressor("regression_l1").fit(target[hdays.columns], target.ratio, sample_weight=target.envelope)
    fdays["desired_cf"] = daily.predict(fdays[hdays.columns]).clip(0, 1.35) * fdays.envelope
    current = future.groupby(DAY_KEYS).hourly_cf.sum()
    factor = (fdays.desired_cf / current.where(current > 1e-8)).clip(0.65, 1.35).fillna(1)
    future = future.merge(factor.rename("factor"), on=DAY_KEYS, how="left", validate="many_to_one")
    future["daily_cf"] = future.hourly_cf * future.factor.fillna(1)
    for arm in ("hourly", "daily", "direct"):
        future[arm] = future[f"{arm}_cf"].clip(0, 1) * future.cap_avail
    return future[GROUP_KEYS + ["hourly", "daily", "direct"]], {"clear_rows": len(clear), "weather_rows": len(train),
                                                             "daily_training_rows": len(target)}


def run(dataset, personal_safe, output):
    original = baseline(dataset, personal_safe)
    file = personal_safe / "feat_full_20261109" / "group_features_train.parquet"
    frame = pd.read_parquet(file)
    frame["ts"] = frame.ts.dt.as_unit("us")
    frame["day"] = (frame.ts - pd.Timedelta(hours=1)).dt.normalize()
    require(not frame.duplicated(GROUP_KEYS).any(), "Duplicate group-weather keys")
    excluded = {"ts", "day", "group", "energy", "cap_avail", "n_sites", "group_cap_kw"}
    weather_features = [c for c in frame if c not in excluded]
    market = pd.read_csv(dataset / "train" / "smp.csv")
    market["ts"] = pd.to_datetime(market.ts).dt.tz_convert(KST).dt.as_unit("us")
    frame = frame.merge(market, on="ts", validate="many_to_one")
    frame["k"] = np.where(frame.ts.dt.hour.between(11, 15), 3, 1)
    frame["coverage"] = frame.cap_avail / frame.group_cap_kw
    frame["cf"] = (frame.energy / frame.cap_avail).clip(0, 1.2)
    results = []
    for month in MONTHS:
        predictions, counts = predict_month(frame, month, weather_features)
        scored = original[original.fold.eq(month)].merge(predictions, on=GROUP_KEYS, validate="one_to_one", how="left")
        scored["v6"] = scored.pred
        scored["daily_blend30"] = 0.7 * scored.v6 + 0.3 * scored.daily
        weight = 0.1 * scored.smp * scored.k
        penalty = {}
        for arm in ("v6", "direct", "hourly", "daily", "daily_blend30"):
            require(np.isfinite(scored[arm]).all() and scored[arm].ge(0).all(), f"Invalid {arm} in {month}")
            penalty[arm] = float((weight * (scored[arm] - scored.actual).abs()).sum())
        require(abs(penalty["v6"] - scored.penalty.sum()) < 1e-6, "v6 changed")
        record = dict(month=month, rows=len(scored), penalty=penalty, **counts)
        print(json.dumps(record), flush=True)
        results.append(record)
    totals = {arm: sum(row["penalty"][arm] for row in results) for arm in results[0]["penalty"]}
    report = dict(metric="README provisional D05 HOLD", group_features_sha256=sha256(file.read_bytes()).hexdigest(),
                  months=results, totals=totals)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("TOTAL", totals, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("personal_safe", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=4):
        run(args.dataset, args.personal_safe, args.output)
