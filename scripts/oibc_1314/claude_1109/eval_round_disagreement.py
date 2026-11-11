"""Walk-forward reliability of the 10:00 and 17:00 issues on large-revision days."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from eval_v7_full_heldout import (
    GROUP_KEYS, KST, MONTHS, group_preparation, join_site_oof, load, require, score, time_bounds,
)

ACTIONS = (0, -1, 1)  # equal, favour 10, favour 17. In a tie retain equal weight.
ALPHA = 0.5
QUANTILE = 0.8
MIN_DAYS = 8


def issue_frame(prep, namespace, issue):
    def issue_matrices(forecast, station, columns):
        rows = forecast[forecast["round"] == issue]
        require(not rows.duplicated(["ts", station]).any(), f"Duplicate {station} issue {issue}")
        return {col: rows.pivot(index="ts", columns=station, values=col) for col in columns}

    namespace["round_mean"] = issue_matrices
    return prep["round_features"](prep["build"]("train")).assign(issue=issue)


def round_oof(dataset, scratch):
    prep, namespace = group_preparation(dataset, scratch)
    frames = [issue_frame(prep, namespace, issue) for issue in (10, 17)]
    data = pd.concat(frames, ignore_index=True)
    require(not data.duplicated(GROUP_KEYS + ["issue"]).any(), "Duplicate forecast keys")
    data = data.merge(prep["targets"](), on=GROUP_KEYS, how="left", validate="many_to_one")
    data["ts"] = data.ts.dt.as_unit("us")
    data["day"] = (data.ts - pd.Timedelta(hours=1)).dt.normalize()
    market = pd.read_csv(dataset / "train" / "smp.csv")
    market["ts"] = pd.to_datetime(market.ts).dt.tz_convert(KST).dt.as_unit("us")
    require(market.ts.is_unique, "Duplicate SMP keys")
    data = data.merge(market, on="ts", how="left", validate="many_to_one")
    data["k"] = np.where(data.ts.dt.hour.between(11, 15), 3, 1)
    data["coverage"] = data.cap_avail / data.group_cap_kw
    data["cf"] = (data.energy / data.cap_avail).clip(0, 1.2)
    excluded = {"ts", "day", "group", "energy", "cap_avail", "n_sites", "smp", "k", "coverage", "cf"}
    columns = [c for c in data if c not in excluded]
    out = []
    for month in map(str, pd.period_range("2024-03", "2025-05", freq="M")):
        start, end = time_bounds(month)
        train = data[(data.ts < start) & data.energy.notna() & data.smp.notna()
                     & (data.coverage >= 0.3) & (data.cosz_max > 0)]
        test = data[data.ts.between(start, end)].copy()
        require(train.ts.max() < test.ts.min(), "Training overlaps evaluation")
        weight = train.coverage * train.k * train.smp.clip(lower=1)
        model = LGBMRegressor(objective="regression_l1", n_estimators=400, learning_rate=0.05,
                              num_leaves=31, min_child_samples=40, n_jobs=4, verbosity=-1, random_state=0)
        model.fit(train[columns], train.cf, sample_weight=weight / weight.mean())
        test["pred_cf"] = np.where(test.cosz_max > 0, model.predict(test[columns]).clip(0, 1), 0)
        test["weather_total"] = test.pred_cf * test.group_cap_kw
        test["scored"] = test.pred_cf * test.cap_avail
        # Both branches share the same forecast keys and evaluation targets.
        keys = ["day", *GROUP_KEYS]
        fields = ["weather_total", "scored"]
        ref = test[test.issue == 10].set_index(keys).sort_index()
        late = test[test.issue == 17].set_index(keys).sort_index()
        require(ref.index.equals(late.index), f"Round coverage differs in {month}")
        require(ref[["energy", "smp", "k"]].equals(late[["energy", "smp", "k"]]), "Target mismatch")
        for field in fields:
            ref[f"{field}10"] = ref[field]
            ref[f"{field}17"] = late[field]
        ref["fold"] = month
        out.append(ref.reset_index()[keys + ["fold", "energy", "smp", "k", *[f"{c}{r}" for c in fields for r in (10, 17)]]])
        print(f"Round OOF {month} | training {len(train):,} | forecast {len(ref):,}", flush=True)
    return pd.concat(out, ignore_index=True)


def day_evidence(oof):
    # The day-selection signal uses only the forecast and fixed full group capacities.
    noon = oof[oof.ts.dt.hour.between(11, 15)]
    keys = ["fold", "day"]
    require(noon.groupby(keys).size().eq(10).all(), "Incomplete midday forecast")
    days = noon.groupby(keys)[["weather_total10", "weather_total17"]].sum()
    total = days.weather_total10 + days.weather_total17
    days["revision"] = ((days.weather_total17 - days.weather_total10) / total.where(total > 0)).fillna(0)
    days["magnitude"] = days.revision.abs()
    labeled = oof.dropna(subset=["energy", "smp", "scored10", "scored17"]).copy()
    equal = (labeled.scored10 + labeled.scored17) / 2
    weight = 0.1 * labeled.smp * labeled.k
    for action in ACTIONS:
        favored = equal + action * ALPHA * (labeled.scored17 - labeled.scored10) / 2
        labeled[f"loss_{action}"] = weight * (favored - labeled.energy).abs()
    for issue in (10, 17):
        labeled[f"raw_{issue}"] = weight * (labeled[f"scored{issue}"] - labeled.energy).abs()
    return days.join(labeled.groupby(keys)[[f"loss_{a}" for a in ACTIONS] + ["raw_10", "raw_17"]].sum()).reset_index()


def learn_policy(days, month):
    history = days[(days.fold < month) & days.loss_0.notna()]
    require(not history.empty, f"No earlier months for {month}")
    threshold = float(history.magnitude.quantile(QUANTILE))
    large = history[history.magnitude >= threshold]
    chosen, counts = {}, {}
    for sign in (-1, 1):
        bucket = large[np.sign(large.revision) == sign]
        counts[sign] = len(bucket)
        chosen[sign] = min(ACTIONS, key=lambda action: bucket[f"loss_{action}"].sum()) if len(bucket) >= MIN_DAYS else 0
    return threshold, chosen, counts


def learn_previous_month(days, month):
    previous = str(pd.Period(month, freq="M") - 1)
    history = days[(days.fold == previous) & days.loss_0.notna()]
    require(not history.empty, f"No prior month for {month}")
    threshold = float(history.magnitude.quantile(QUANTILE))
    large = history[history.magnitude >= threshold]
    action = min(ACTIONS, key=lambda choice: large[f"loss_{choice}"].sum()) if len(large) >= 5 else 0
    return previous, threshold, action, len(large)


def baseline(dataset, personal_safe):
    cv = personal_safe / "cv"
    sites = pd.read_csv(dataset / "sites.csv")
    mapping = sites.set_index("site_id").current_group.map({"A": "G1", "B": "G2"})
    frame = join_site_oof(cv, load(cv / "site_level_B21_12m.parquet"))
    h1 = pd.read_parquet(cv / "group_cv.parquet")
    h1["ts"] = h1.ts.dt.as_unit("us")
    groups = [
        score(frame[frame.fold == month], mapping,
              h1.loc[h1.fold == month, GROUP_KEYS + ["H1", "energy"]]).assign(fold=month).reset_index()
        for month in MONTHS
    ]
    v6 = pd.concat(groups, ignore_index=True)
    require(len(v6) == 3749 and abs(v6.penalty.sum() - 76_126_029.82) < 1, "v6 replay mismatch")
    return v6


def evaluate(v6, days, output):
    summary = []
    for month in sorted(days.loc[days.fold >= "2024-06", "fold"].unique()):
        threshold, actions, counts = learn_policy(days, month)
        previous, prev_threshold, prev_action, prev_days = learn_previous_month(days, month)
        month_days = days[days.fold == month].copy()
        month_days["prev_large"] = month_days.magnitude.ge(prev_threshold)
        month_days["large"] = month_days.magnitude.ge(threshold)
        month_days["action"] = np.sign(month_days.revision).map(actions).fillna(0).where(month_days.large, 0)
        hard_days = month_days[month_days.large & month_days.raw_10.notna()]
        first = int((hard_days.raw_10 < hard_days.raw_17).sum())
        late = int((hard_days.raw_17 < hard_days.raw_10).sum())
        metrics = dict(month=month, threshold_fraction_of_mean=2 * threshold, actions=actions,
                       calibration_days_by_direction=counts, big_days=len(hard_days), round_10_wins=first,
                       round_17_wins=late, round_10_penalty=float(hard_days.raw_10.sum()),
                       round_17_penalty=float(hard_days.raw_17.sum()),
                       previous_month=dict(month=previous, threshold_fraction_of_mean=2 * prev_threshold,
                                           action=prev_action, calibration_days=prev_days))
        examples = hard_days.nlargest(3, "magnitude")
        metrics["largest_revision_days"] = [
            dict(day=row.day.date().isoformat(), relative_change=2 * float(row.revision),
                 noon_forecast_10_kwh=float(row.weather_total10), noon_forecast_17_kwh=float(row.weather_total17),
                 round_10_penalty=float(row.raw_10), round_17_penalty=float(row.raw_17))
            for row in examples.itertuples()
        ]
        if month in MONTHS:
            rows = v6[v6.fold == month].merge(month_days[["day", "large", "prev_large", "revision", "action"]],
                                             on="day", how="left", validate="many_to_one")
            require(rows.large.notna().all(), "Missing day signal")
            weight = 0.1 * rows.smp * rows.k
            arms = {"baseline": 0, "rolling": rows.action, "always_10_on_large": -rows.large.astype(int),
                    "always_17_on_large": rows.large.astype(int)}
            prev_prediction = rows.pred * (1 + ALPHA * rows.revision * prev_action * rows.prev_large)
            require(np.isfinite(prev_prediction).all() and prev_prediction.ge(0).all(), "Invalid prior-month prediction")
            rows["previous_month_only"] = weight * (prev_prediction - rows.actual).abs()
            for name, action in arms.items():
                prediction = rows.pred * (1 + ALPHA * rows.revision * action)
                require(np.isfinite(prediction).all() and prediction.ge(0).all(), "Invalid adjusted prediction")
                rows[name] = weight * (prediction - rows.actual).abs()
            require(np.allclose(rows.baseline, rows.penalty), "Baseline scores differ")
            metrics["v6_comparison"] = {
                "rows": len(rows), "affected_days": int(rows.loc[rows.action.ne(0) & rows.revision.ne(0), "day"].nunique()),
                "large_days": int(rows.loc[rows.large, "day"].nunique()),
                "previous_month_penalty": float(rows.previous_month_only.sum()),
                "previous_month_affected_days": int(rows.loc[rows.prev_large & (prev_action != 0), "day"].nunique()),
                "penalty": {name: float(rows[name].sum()) for name in arms},
                "large_day_penalty": {name: float(rows.loc[rows.large, name].sum()) for name in arms},
                "oracle_on_large_days": float(rows.groupby("day")[list(arms)].sum().min(axis=1).sum()),
            }
        summary.append(metrics)
        print(json.dumps(metrics), flush=True)
    scored = [row["v6_comparison"] for row in summary if "v6_comparison" in row]
    totals = {a: sum(row["penalty"][a] for row in scored) for a in scored[0]["penalty"]}
    report = dict(alpha=ALPHA, disagreement_quantile=QUANTILE, min_history_days_per_sign=MIN_DAYS, months=summary,
                  totals=totals, large_day_totals={a: sum(row["large_day_penalty"][a] for row in scored) for a in totals},
                  oracle_total=sum(row["oracle_on_large_days"] for row in scored),
                  previous_month_only_total=sum(row["previous_month_penalty"] for row in scored))
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("TOTAL", json.dumps(totals), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("personal_safe", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=4):
        v6 = baseline(args.dataset, args.personal_safe)
        rounds = round_oof(args.dataset, args.output)
        evidence = day_evidence(rounds)
        evaluate(v6, evidence, args.output)


if __name__ == "__main__":
    main()
