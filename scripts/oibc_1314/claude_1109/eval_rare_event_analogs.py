"""Walk-forward rare-event analog search for same-direction VPP forecast errors."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

from eval_portfolio_day_calibration import full_v7, label_days, proxy_history
from eval_v7_full_heldout import KST, MONTHS, load, require

WEATHER = ["clear_proxy", "wa_cloud", "wb_cloud", "wb_temp", "wb_rel_hum"]
WEATHER_WEIGHTS = np.array([2., 1., 1., .5, .5])
HOURS = np.arange(11, 16)
GROUPS = ["G1", "G2"]
N_ANALOGS = 12
ALARM_PROBABILITY = 0.5
TAIL = 0.10
ARMS = {"same_hour": (False, False), "adjacent_hour": (True, False),
        "adjacent_hour_bias": (True, True)}


def forecast_features(path):
    columns = ["ts", "group", "cosz", "cs_ab", *WEATHER[1:]]
    weather = pd.read_parquet(path, columns=columns)
    weather["ts"] = weather.ts.dt.tz_convert(KST)
    require(not weather.duplicated(["group", "ts"]).any(), "Duplicate weather keys")
    weather["day"] = (weather.ts - pd.Timedelta(hours=1)).dt.normalize()
    weather["hour"] = weather.ts.dt.hour
    sun = 1 + .033 * np.cos(2 * np.pi * weather.ts.dt.dayofyear / 365)
    weather["clear_proxy"] = weather.cs_ab / (sun * weather.cosz).where(weather.cosz > .02)
    grid = weather.pivot(index="day", columns=["group", "hour"], values=[*WEATHER, "cosz"])
    windows = {}
    for shift in (-1, 0, 1):
        keys = pd.MultiIndex.from_product([WEATHER, GROUPS, HOURS + shift])
        windows[shift] = grid.reindex(columns=keys).to_numpy().reshape(len(grid), len(WEATHER), 2, 5)
    sun_keys = pd.MultiIndex.from_product([["cosz"], GROUPS, HOURS])
    geometry = grid.reindex(columns=sun_keys).to_numpy()
    require(np.isfinite(geometry).all(), "Missing solar geometry")
    return grid.index, windows, geometry


def labels(hourly):
    by_group = pd.concat([label_days(hourly[hourly.group == group]).scale.rename(group) for group in GROUPS],
                         axis=1, join="inner")
    result = by_group.join(label_days(hourly).scale.rename("portfolio")).reset_index().set_index("day")
    require(result.index.is_unique, "Multiple labels on one day")
    require(result[GROUPS + ["portfolio"]].notna().all().all(), "Incomplete analog labels")
    return result


def predict(days, windows, geometry, library, test_days, adjacent, bias):
    train_pos = days.get_indexer(library.index)
    test_pos = days.get_indexer(test_days)
    require((train_pos >= 0).all() and (test_pos >= 0).all(), "Missing weather features")
    require(library.index.max() < test_days.min(), "Analog archive overlaps evaluation")
    center = windows[0][train_pos]
    pool = center.transpose(1, 0, 2, 3).reshape(len(WEATHER), -1)
    median = np.nanmedian(pool, axis=1)
    stdev = np.maximum(np.nanstd(pool, axis=1), 1e-6)
    require(np.isfinite(median).all() and np.isfinite(stdev).all(), "Missing archive predictors")

    def impute(window):
        return np.where(np.isfinite(window), window, median[:, None, None])

    center = impute(center)
    response = library[GROUPS + ["portfolio"]].to_numpy()
    sunshine = center[:, 0].mean(axis=(1, 2))
    centered = sunshine - sunshine.mean()
    variance = centered @ centered
    slope = ((centered @ response) / variance).clip(-1.5, 1.5) if bias and variance > 1e-8 else np.zeros(3)
    offsets = (0, -1, 1) if adjacent else (0,)
    candidates = np.stack([impute(windows[shift][train_pos]) for shift in offsets])
    weight = WEATHER_WEIGHTS / stdev**2
    predictions = []
    for day, row in zip(test_days, test_pos, strict=True):
        target = impute(windows[0][row])
        weather_distance = np.einsum("snfgh,f->sn", (candidates - target)**2, weight) / 10
        geometry_distance = ((geometry[train_pos] - geometry[row])**2).mean(axis=1) / .15**2
        distance = weather_distance + geometry_distance[None, :] + .10 * np.abs(offsets)[:, None]
        # Each historic date contributes at most one member, using its nearest window.
        best_window = distance.argmin(axis=0)
        ordering = np.argsort(distance.min(axis=0), kind="stable")[:N_ANALOGS]
        require(len(ordering) == N_ANALOGS, "Too few past analogs")
        chosen_weather = candidates[best_window[ordering], ordering, 0].mean(axis=(1, 2))
        adjustment = (target[0].mean() - chosen_weather)[:, None] * slope
        analog = response[ordering] + adjustment
        lower = (analog[:, :2] <= 1 - TAIL).all(axis=1)
        upper = (analog[:, :2] >= 1 + TAIL).all(axis=1)
        p_low, p_high = float(lower.mean()), float(upper.mean())
        direction = -1 if p_low > p_high else 1
        matching = lower if direction < 0 else upper
        active = max(p_low, p_high) >= ALARM_PROBABILITY
        factor = float(np.clip(np.median(analog[matching, 2]), .8, 1.2)) if active else 1.
        # The same partial adjustment is applied to both VPPs.
        factor = 1 + .5 * (factor - 1)
        if (factor - 1) * direction <= 0:
            factor = 1.
        predictions.append(dict(day=day, risk=p_low+p_high, action=direction if factor != 1 else 0,
                                factor=factor, displaced_analogs=int(np.count_nonzero(best_window[ordering]))))
    return pd.DataFrame(predictions).set_index("day")


def evaluate(v7, truth, predictions):
    daily = predictions.join(truth[GROUPS], how="left")
    require(daily[GROUPS].notna().all().all(), "Missing risk labels")
    lower = daily[GROUPS].le(1 - TAIL).all(axis=1)
    upper = daily[GROUPS].ge(1 + TAIL).all(axis=1)
    daily["event"] = lower | upper
    daily["hit"] = (lower & daily.action.eq(-1)) | (upper & daily.action.eq(1))
    hourly = v7.join(predictions.factor.rename("analog_factor"), on="day", validate="many_to_one")
    require(hourly.analog_factor.notna().all(), "Unmatched evaluation hour")
    hourly["adjusted"] = hourly.pred * hourly.analog_factor
    require(np.isfinite(hourly.adjusted).all() and hourly.adjusted.ge(0).all(), "Invalid adjusted forecast")
    penalty = float((hourly.weight * (hourly.actual - hourly.adjusted).abs()).sum())
    return penalty, daily


def run(personal, composition, output):
    feature_path = personal / "feat_full_comp/group_features_train.parquet"
    days, windows, geometry = forecast_features(feature_path)
    mapping = pd.read_csv(composition).set_index("site_id").group
    cv = personal / "cv"
    raw = load(cv / "site_level_B21_12m.parquet")
    library = labels(proxy_history(raw, mapping))
    v7 = full_v7(cv, raw, mapping)
    truth = labels(v7)
    results = []
    traces = {name: [] for name in ARMS}
    for month in MONTHS:
        prior = library.loc[library.fold < month]
        observed = v7[v7.fold == month]
        target_days = pd.DatetimeIndex(observed.day.unique()).sort_values()
        record = dict(month=month, library_days=len(prior), rows=len(observed),
                      library_end=prior.index.max().date().isoformat(), v7=float(observed.penalty.sum()), arms={})
        for name, (adjacent, bias) in ARMS.items():
            forecasts = predict(days, windows, geometry, prior, target_days, adjacent, bias)
            penalty, daily = evaluate(observed, truth, forecasts)
            traces[name].append(daily)
            record["arms"][name] = dict(penalty=penalty, true_event_days=int(daily.event.sum()),
                                        flagged_days=int(daily.action.ne(0).sum()), correct_direction_days=int(daily.hit.sum()),
                                        displaced_analogs=int(daily.displaced_analogs.sum()))
        print(json.dumps(record), flush=True)
        results.append(record)
    totals = {"v7": sum(row["v7"] for row in results), "rows": sum(row["rows"] for row in results)}
    for name in ARMS:
        daily = pd.concat(traces[name])
        totals[name] = dict(penalty=sum(row["arms"][name]["penalty"] for row in results),
                            evaluated_days=len(daily), event_days=int(daily.event.sum()),
                            event_prevalence=float(daily.event.mean()),
                            average_precision=float(average_precision_score(daily.event, daily.risk)),
                            flagged_days=int(daily.action.ne(0).sum()), correct_direction_days=int(daily.hit.sum()))
    report = dict(metric="README provisional D05 HOLD", weather_features=WEATHER, neighbors=N_ANALOGS,
                  same_direction_ratio_threshold=TAIL, alert_probability=ALARM_PROBABILITY, months=results, totals=totals,
                  weather_sha256=sha256(feature_path.read_bytes()).hexdigest(),
                  mapping_sha256=sha256(composition.read_bytes()).hexdigest())
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(totals), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("personal", "composition", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=4):
        run(args.personal, args.composition, args.output)
