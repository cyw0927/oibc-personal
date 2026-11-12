'''Compare pooled and weather-regime VPP forecasters on the frozen v7 grid.'''
import argparse
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

KST = "Asia/Seoul"
KEYS = ["group", "ts"]
DAILY_KEYS = ["group", "day"]
WEATHER = ["wa_cloud", "wb_cloud", "wa_cloud_prev", "wb_cloud_prev",
           "wa_temp", "wb_temp", "wa_humidity", "wb_rel_hum", "wb_uv_idx",
           "wb_precip_prob", "wb_vis", "cs_wa", "cs_wb", "cosz", "elev", "doy_sin", "doy_cos"]
REGIMES = ["steady", "clearing", "clouding", "mixed"]
MIN_DAYS = 45
PARAMS = dict(objective="regression_l1", n_estimators=350, num_leaves=31,
              learning_rate=.035, min_child_samples=40, random_state=17,
              deterministic=True, force_col_wise=True, n_jobs=3, verbosity=-1)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def kst(values):
    return pd.to_datetime(values, utc=True).dt.tz_convert(KST).dt.as_unit("us")


def weather_features(sites):
    # Aggregate the weather on all mapped sites, regardless of observed generation.
    weather = sites[WEATHER].mul(sites.capacity_kw, axis=0)
    weights = sites[WEATHER].notna().mul(sites.capacity_kw, axis=0)
    keys = [sites.group, sites.ts]
    hourly = (weather.groupby(keys).sum() / weights.groupby(keys).sum()).reset_index()
    hourly["day"] = (hourly.ts - pd.Timedelta(hours=1)).dt.normalize()
    hourly["hour"] = hourly.ts.dt.hour
    hourly["cloud"] = hourly[["wa_cloud", "wb_cloud"]].mean(axis=1)
    hourly["previous_cloud"] = hourly[["wa_cloud_prev", "wb_cloud_prev"]].mean(axis=1)
    daytime = hourly[hourly.hour.between(8, 17)]
    daily = daytime.groupby(DAILY_KEYS).agg(
        mean_cloud=("cloud", "mean"), previous_cloud_mean=("previous_cloud", "mean"),
        low_cloud=("cloud", "min"), high_cloud=("cloud", "max"),
    )
    for label, low, high in [("morning_cloud", 8, 10), ("afternoon_cloud", 14, 17)]:
        daily[label] = daytime[daytime.hour.between(low, high)].groupby(DAILY_KEYS).cloud.mean()
    daily["within_day"] = daily.afternoon_cloud - daily.morning_cloud
    daily["between_days"] = daily.mean_cloud - daily.previous_cloud_mean
    daily["cloud_range"] = daily.high_cloud - daily.low_cloud
    trend = daily.within_day.where(daily.within_day.abs() >= 15, daily.between_days)
    daily["regime"] = np.select(
        [trend.le(-15), trend.ge(15), daily.cloud_range.ge(25)],
        ["clearing", "clouding", "mixed"], default="steady")
    daily.loc[daily.within_day.isna() | daily.between_days.isna(), "regime"] = "unknown"
    return hourly.merge(daily.reset_index(), on=DAILY_KEYS, how="left", validate="many_to_one")


def targets(sites):
    observed = sites[sites.energy.notna()]
    return observed.groupby(KEYS).agg(energy=("energy", "sum"),
                                      scored_kw=("capacity_kw", "sum")).reset_index()


def predict(model, rows, features):
    cf = model.predict(rows[features])
    return np.where(rows.elev > 0, np.clip(cf, 0, None), 0) * rows.scored_kw


def run(features_file, package, output):
    mapping_file = package / "1_submit" / "composition.csv"
    ref_file = package / "2_evidence" / "fold_predictions_comp.csv"
    assert digest(mapping_file).startswith("1e42864d")
    assert digest(ref_file).startswith("a1a2dc22")
    mapping = pd.read_csv(mapping_file).set_index("site_id").group
    data = pd.read_parquet(features_file, columns=["site_id", "ts", "energy", "capacity_kw", "round", *WEATHER])
    data = data.loc[data["round"].eq(17)].drop(columns="round")
    data["ts"] = kst(data.ts)
    data["group"] = data.site_id.map(mapping)
    assert data.group.notna().all() and not data.duplicated(["site_id", "ts"]).any()
    capacity = data.drop_duplicates("site_id").groupby("group").capacity_kw.sum()
    assert len(mapping) == 253 and capacity.ge(20000).all()
    forecast = weather_features(data)
    observed = targets(data)
    data = forecast.merge(observed, on=KEYS, how="inner", validate="one_to_one")
    data["available_fraction"] = data.scored_kw / data.group.map(capacity)
    data["cf"] = (data.energy / data.scored_kw).clip(0, 1.2)
    features = forecast.select_dtypes("number").columns.tolist()
    features.append("is_g2")
    data["is_g2"] = data.group.eq("G2").astype(int)
    ref = pd.read_csv(ref_file)
    ref["ts"] = kst(ref.ts)
    report = dict(reference_sha256=digest(ref_file), mapping_sha256=digest(mapping_file),
                  forecast_round=17, cloud_threshold_points=15, mixed_range_points=25,
                  min_regime_days=MIN_DAYS, min_train_coverage=.60,
                  model=PARAMS, features=features, folds=[])
    for month, frame in ref.groupby("fold", sort=True):
        cutoff = pd.Timestamp(month+"-01", tz=KST) - pd.Timedelta(hours=7)
        train = data[data.ts.le(cutoff) & data.elev.gt(0) & data.available_fraction.ge(.60)]
        eval_rows = frame.merge(data.drop(columns="energy"), on=KEYS, validate="one_to_one")
        check = frame.merge(observed, on=KEYS, validate="one_to_one")
        assert len(eval_rows) == len(frame) and len(check) == len(frame)
        assert np.allclose(check.energy_x, check.energy_y, rtol=1e-7, atol=.003)
        pooled = lgb.LGBMRegressor(**PARAMS).fit(train[features], train.cf, sample_weight=train.scored_kw)
        pooled_pred = predict(pooled, eval_rows, features)
        expert_pred = pooled_pred.copy()
        days = {}
        routed = {}
        for regime in REGIMES:
            subset = train[train.regime.eq(regime)]
            days[regime] = int(subset.day.nunique())
            test_mask = eval_rows.regime.eq(regime)
            routed[regime] = int(test_mask.sum()) if days[regime] >= MIN_DAYS else 0
            if routed[regime]:
                expert = lgb.LGBMRegressor(**PARAMS).fit(subset[features], subset.cf, sample_weight=subset.scored_kw)
                expert_pred.loc[test_mask] = predict(expert, eval_rows.loc[test_mask], features)
        candidates = {"v7":eval_rows.final_pred, "pooled":pooled_pred, "experts":expert_pred,
                      "v7_experts_50_50":.5*(eval_rows.final_pred + expert_pred)}
        weight = .1 * eval_rows.smp * eval_rows.k
        assert all(np.isfinite(values).all() and (values >= 0).all() for values in candidates.values())
        scores = {name:float((weight * (values - eval_rows.energy).abs()).sum())
                  for name, values in candidates.items()}
        report["folds"].append(dict(month=month, rows=len(eval_rows), training_rows=len(train),
                                    last_training=str(train.ts.max()), regime_training_days=days,
                                    expert_rows=routed, penalty_krw=scores))
        print(month, scores, "routed", routed, flush=True)
    assert len(report["folds"]) == 5 and sum(x["rows"] for x in report["folds"]) == 3749
    report["total_penalty_krw"] = {name:sum(f["penalty_krw"][name] for f in report["folds"])
                                   for name in candidates}
    assert abs(report["total_penalty_krw"]["v7"] - 74389091.15) < 1
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(report["total_penalty_krw"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ["features", "package", "output"]:
        parser.add_argument(arg, type=Path)
    args = parser.parse_args()
    run(args.features, args.package, args.output)
