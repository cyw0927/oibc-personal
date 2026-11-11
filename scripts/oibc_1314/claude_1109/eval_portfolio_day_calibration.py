"""Fit one shared, settlement-weighted daily adjustment on earlier forecast errors."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from eval_v7_full_heldout import GROUP_KEYS, MONTHS, join_site_oof, load, require, score

WEATHER = [
    "cosz", "cosz_max", "cs_ab", "wa_cloud", "wb_cloud", "wa_cloud_sd", "wb_cloud_sd",
    "wa_cloud_d1710", "wb_cloud_d1710", "wb_precip_prob", "wb_uv_idx",
    "wa_rain", "wb_snow", "wa_temp", "wa_humidity", "wb_vis",
]
KEYS = ["fold", "day"]


def weighted_median(ratios, weights):
    order = np.argsort(ratios)
    ordered = ratios[order]
    cumulative = np.cumsum(weights[order])
    return float(ordered[np.searchsorted(cumulative, cumulative[-1] / 2)])


def label_days(hourly):
    rows = hourly[(hourly.pred > 0) & (hourly.weight > 0)].copy()
    rows["ratio"] = rows.actual / rows.pred
    rows["target_weight"] = rows.weight * rows.pred
    rows = rows.groupby(KEYS).apply(
        lambda x: pd.Series({
            "scale": weighted_median(x.ratio.to_numpy(), x.target_weight.to_numpy()),
            "weight": x.target_weight.sum(),
        }), include_groups=False)
    return rows


def features(path):
    weather = pd.read_parquet(path, columns=["ts", "group", "doy_sin", "doy_cos", *WEATHER])
    weather["day"] = (weather.ts - pd.Timedelta(hours=1)).dt.normalize()
    weather["fold"] = weather.day.dt.strftime("%Y-%m")
    midday = weather.ts.dt.hour.between(11, 15)
    mean = weather.loc[midday].groupby(KEYS)[WEATHER].mean().add_suffix("_mid")
    lit = weather.cosz_max > 0.05
    shape = weather.loc[lit].groupby(KEYS)[WEATHER].std().add_suffix("_spread")
    date = weather.groupby(KEYS)[["doy_sin", "doy_cos"]].first()
    return date.join(mean).join(shape)


def with_weights(x):
    x["day"] = (x.ts - pd.Timedelta(hours=1)).dt.normalize()
    x["weight"] = 0.1 * x.smp * x.k
    require(np.isfinite(x.weight).all() and x.weight.ge(0).all(), "Invalid market weights")
    return x


def proxy_history(oof, mapping):
    oof = oof.assign(group=oof.site_id.map(mapping))
    require(oof.group.notna().all(), "Missing mapping")
    h = oof.groupby(["fold", *GROUP_KEYS]).agg(
        actual=("energy", "sum"), pred=("B", "sum"), smp=("smp", "first"), k=("k", "first")).reset_index()
    return with_weights(h)


def full_v7(cv, oof, mapping):
    sites = join_site_oof(cv, oof)
    h1 = pd.read_parquet(cv / "group_cv_comp.parquet")
    h1["ts"] = h1.ts.dt.as_unit("us")
    out = []
    for month in MONTHS:
        h = score(sites.loc[sites.fold == month], mapping,
                  h1.loc[h1.fold == month, GROUP_KEYS + ["H1", "energy"]])
        out.append(h.reset_index().assign(fold=month))
    h = with_weights(pd.concat(out, ignore_index=True))
    require(len(h) == 3749 and abs(h.penalty.sum() / 1e6 - 74.3891) < 5.1e-5, "v7 replay changed")
    return h


def total(hourly, factor):
    return float((hourly.weight * (hourly.actual - hourly.pred * factor).abs()).sum())


def oracle(hourly, clip):
    labels = label_days(hourly)
    scale = labels.scale.clip(*clip).rename("oracle_scale")
    matched = hourly.join(scale, on=KEYS)
    return total(matched, matched.oracle_scale.fillna(1))


def run(personal, composition, output):
    cv = personal / "cv"
    feature_path = personal / "feat_full_comp/group_features_train.parquet"
    f = features(feature_path)
    mapping = pd.read_csv(composition).set_index("site_id").group
    oof = load(cv / "site_level_B21_12m.parquet")
    proxy = proxy_history(oof, mapping)
    v7 = full_v7(cv, oof, mapping)
    labels = label_days(proxy)
    labels_v7 = label_days(v7)
    rows = []
    for month in sorted(proxy.fold.unique())[1:]:
        train = labels.loc[labels.index.get_level_values("fold") < month].join(f)
        today = f.loc[f.index.get_level_values("fold") == month]
        require(len(train) >= 20 and len(today) >= 28, "Incomplete prior history")
        # Fit the conditional weighted median of the whole-portfolio daily correction.
        model = HistGradientBoostingRegressor(loss="absolute_error", max_iter=120, max_leaf_nodes=5,
                                              min_samples_leaf=16, l2_regularization=2.0,
                                              early_stopping=False, random_state=0)
        model.fit(train[f.columns], train.scale.clip(0.6, 1.4), sample_weight=train.weight / train.weight.mean())
        scale = pd.Series(model.predict(today).clip(0.8, 1.2), index=today.index, name="scale")
        earlier_v7 = labels_v7.loc[labels_v7.index.get_level_values("fold") < month]
        if len(earlier_v7) >= 20:
            # Same-model sensitivity, trained on whichever earlier v7 folds exist.
            same = earlier_v7.join(f)
            model.fit(same[f.columns], same.scale.clip(0.6, 1.4), sample_weight=same.weight / same.weight.mean())
            scale_v7 = pd.Series(model.predict(today).clip(0.8, 1.2), index=today.index, name="same_model_scale")
        else:
            scale_v7 = pd.Series(1.0, index=today.index, name="same_model_scale")
        record = dict(month=month, training_days=len(train), earlier_v7_days=len(earlier_v7))
        for name, hourly in (("proxy", proxy), ("v7", v7)):
            test = hourly.loc[hourly.fold == month].join(scale, on=KEYS).join(scale_v7, on=KEYS)
            if test.empty:
                continue
            require(test.scale.notna().all() and test.same_model_scale.notna().all(), "Missing adjustment")
            record[name] = dict(rows=len(test), baseline=total(test, 1), full=total(test, test.scale),
                                half=total(test, 0.5 + 0.5 * test.scale),
                                shared_oracle=oracle(test, (0, np.inf)),
                                shared_oracle_20pct=oracle(test, (0.8, 1.2)))
            if name == "v7":
                record[name]["same_model_half"] = total(test, 0.5 + 0.5 * test.same_model_scale)
        rows.append(record)
        print(json.dumps(record), flush=True)
    totals = {}
    for name in ("proxy", "v7"):
        completed = [row[name] for row in rows if name in row]
        totals[name] = {k: sum(row[k] for row in completed) for k in completed[0]}
    report = dict(metric="README provisional D05 HOLD", source_sha256=sha256(feature_path.read_bytes()).hexdigest(),
                  months=rows, totals=totals)
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
