"""Compare full v7 against v6 with a separate composition for each evaluation month."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from ctx_features import add_ctx

KST = "Asia/Seoul"
MONTHS = ("2024-07", "2024-10", "2025-01", "2025-02", "2025-04")
KEYS = ["site_id", "ts", "fold"]
GROUP_KEYS = ["group", "ts"]
CONFIGS = (
    dict(learning_rate=0.05, max_iter=600, max_leaf_nodes=31, min_samples_leaf=40),
    dict(learning_rate=0.03, max_iter=900, max_leaf_nodes=15, min_samples_leaf=80),
    dict(learning_rate=0.05, max_iter=500, max_leaf_nodes=63, min_samples_leaf=30),
)
NON_FEATURES = {"ts", "group", "energy", "cap_avail", "n_sites", "group_cap_kw", "cf", "smp", "k", "cov"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def time_bounds(month):
    start = pd.Period(month).start_time.tz_localize(KST)
    return start + pd.Timedelta(hours=1), start + pd.offsets.MonthBegin()


def load(path):
    frame = pd.read_parquet(path)
    frame["ts"] = frame.ts.dt.tz_convert(KST).dt.as_unit("us")
    require(not frame.duplicated(KEYS).any(), f"Duplicate keys: {path.name}")
    day = frame.ts - pd.Timedelta(hours=1)
    require((frame.fold == day.dt.strftime("%Y-%m")).all(), f"Invalid fold labels: {path.name}")
    require(np.isfinite(frame.select_dtypes("number")).all().all(), f"Non-finite values: {path.name}")
    return frame


def join_site_oof(cv, b21):
    frame = load(cv / "site_level_E_5f.parquet").set_index(KEYS).sort_index()
    inputs = (
        (b21, {"B": "B21"}),
        (load(cv / "site_level_day23.parquet"), {"B": "h23", "D_s0": "d23a", "D_s1": "d23b"}),
        (load(cv / "site_level_day21.parquet"), {"B": "h21", "D_s0": "d21a", "D_s1": "d21b"}),
    )
    for source, columns in inputs:
        source = source[source.fold.isin(MONTHS)].set_index(KEYS).sort_index()
        require(frame.index.equals(source.index), "Site OOF keys differ")
        for col in ("energy", "smp", "k"):
            require(np.allclose(frame[col], source[col], atol=1e-7, rtol=1e-7), f"Site OOF {col} differs")
        frame = frame.join(source[list(columns)].rename(columns=columns), validate="one_to_one")
    return frame.reset_index()


def optimise(errors, weights, capacities, reference):
    assignment = reference.astype(float).copy()
    combined = errors.sum(axis=1)
    total_capacity = capacities.sum()
    for _ in range(200):
        first = errors @ assignment
        baseline = weights @ (abs(first) + abs(combined - first))
        direction = 1 - 2 * assignment
        candidate_first = first[:, None] + errors * direction
        costs = (weights[:, None] * (abs(candidate_first) + abs(combined[:, None] - candidate_first))).sum(axis=0)
        capacity = capacities @ assignment + direction * capacities
        costs[(capacity < 20000) | (total_capacity - capacity < 20000)] = np.inf
        best = int(costs.argmin())
        if costs[best] >= baseline - 1e-6:
            break
        assignment[best] = 1 - assignment[best]
    return assignment.astype(bool)


def group_preparation(dataset, scratch):
    # Load the original feature builder in isolation, then reuse read-only train tables.
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], str(dataset), str(scratch)]
        module = runpy.run_path(str(Path(__file__).with_name("prep_group_features_full.py")))
    finally:
        sys.argv = original_argv
    namespace = module["build"].__globals__
    tables = {name: module["read_train"](name) for name in (
        "forecast_wa.parquet", "forecast_wb.parquet", "generation.parquet"
    )}
    namespace["read_train"] = lambda name: tables[name].copy()
    return module, namespace


def fit_h1(prep, namespace, sites, market, month):
    namespace["sites"] = sites
    features = prep["build"]("train").merge(prep["targets"](), on=GROUP_KEYS, how="left")
    features = prep["round_features"](features)
    features["ts"] = features.ts.dt.as_unit("us")
    base = [column for column in features.columns if column not in NON_FEATURES]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
        features, context = add_ctx(features)
    features = features.copy()
    features = features.merge(market, on="ts", how="left", validate="many_to_one")
    features["coverage"] = features.cap_avail / features.group_cap_kw
    start, end = time_bounds(month)
    lit = features.cosz_max.gt(0)
    train = features[(features.ts < start) & features.energy.notna() & (features.coverage >= 0.3) & lit]
    test = features[features.ts.between(start, end) & features.energy.notna()].copy()
    require(not train.empty and not test.empty and train.ts.max() < test.ts.min(), "Invalid H1 cutoff")
    require(np.isfinite(train.smp).all(), "Missing SMP in H1 training")
    # The existing context includes t+1/t+2. At times using nonzero H1 these
    # hours must be within the same issue's target day.
    used = pd.concat([train.ts, test.loc[test.cosz_max > 0, "ts"]])
    require(((used - pd.Timedelta(hours=1)).dt.normalize() ==
             (used + pd.Timedelta(hours=1)).dt.normalize()).all(), "Future context crosses issue day")
    target = (train.energy / train.cap_avail).clip(0, 1.2)
    smp = train.smp.clip(lower=1)
    weight = train.coverage * train.k * smp / smp.mean()
    columns = base + context
    prediction = [
        HistGradientBoostingRegressor(loss="absolute_error", l2_regularization=1.0,
                                      early_stopping=False, random_state=0, **config)
        .fit(train[columns], target, sample_weight=weight).predict(test[columns]).clip(0, 1)
        for config in CONFIGS
    ]
    test["H1"] = np.where(test.cosz_max > 0, np.mean(prediction, axis=0), 0) * test.cap_avail
    return test[GROUP_KEYS + ["H1", "energy", "cap_avail"]]


def score(site_oof, mapping, h1):
    site = site_oof.assign(group=site_oof.site_id.map(mapping))
    require(site.group.notna().all(), "Site absent from composition")
    site["day"] = (site.ts - pd.Timedelta(hours=1)).dt.normalize()
    daily_cols = ["h23", "d23a", "d23b", "h21", "d21a", "d21b"]
    grouped = site.groupby(GROUP_KEYS).agg(
        day=("day", "first"), actual=("energy", "sum"), smp=("smp", "first"), k=("k", "first"),
        **{name: (name, "sum") for name in ["E", "B21", *daily_cols]},
    ).reset_index()
    day = grouped.groupby(["group", "day"])[daily_cols].sum()
    ratios = [(day[col] / day[base].where(day[base] > 1)).fillna(1)
              for base, col in (("h23", "d23a"), ("h23", "d23b"), ("h21", "d21a"), ("h21", "d21b"))]
    day["factor"] = 1 + 0.5 * (sum(ratios) / 4 - 1)
    joined = grouped.merge(day[["factor"]], on=["group", "day"], validate="many_to_one")
    joined = joined.merge(h1, on=GROUP_KEYS, how="left", validate="one_to_one")
    require(joined.H1.notna().all(), "H1 is missing for a scored group-hour")
    joined["pred"] = (0.35 * joined.E + 0.35 * joined.B21 + 0.30 * joined.H1) * joined.factor
    require(np.isfinite(joined.pred).all() and joined.pred.ge(0).all(), "Invalid prediction")
    joined["penalty"] = 0.1 * joined.smp * joined.k * (joined.pred - joined.actual).abs()
    return joined.set_index(GROUP_KEYS).sort_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("personal_safe", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--protocol", choices=["leave_month_out", "past_only"])
    parser.add_argument("--replay-reference", action="store_true")
    parser.add_argument("--month", choices=MONTHS, action="append")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cv = args.personal_safe / "cv"
    sites = pd.read_csv(args.dataset / "sites.csv")
    ids = sites.site_id
    reference = sites.current_group.eq("A").to_numpy()
    capacities = sites.capacity_kw.to_numpy()
    require(len(sites) == 253 and ids.is_unique and sites.current_group.isin(["A", "B"]).all(), "Invalid sites")
    b21 = load(cv / "site_level_B21_12m.parquet")
    require(b21.site_id.isin(ids).all(), "Unknown site")
    source = b21.assign(error=b21.B - b21.energy)
    market_cols = ["smp", "k", "fold"]
    market_check = source.groupby("ts")[market_cols].nunique()
    require((market_check == 1).all().all(), "Market or fold values disagree by plant")
    errors = source.pivot(index="ts", columns="site_id", values="error").reindex(columns=ids).fillna(0)
    market_oof = source.groupby("ts")[market_cols].first().loc[errors.index]
    weights = (0.1 * market_oof.smp * market_oof.k).to_numpy()
    market = pd.read_csv(args.dataset / "train" / "smp.csv")
    market["ts"] = pd.to_datetime(market.ts).dt.tz_convert(KST).dt.as_unit("us")
    market["k"] = np.where(market.ts.dt.hour.between(11, 15), 3, 1)
    site_oof = join_site_oof(cv, b21)
    stored_h1 = pd.read_parquet(cv / "group_cv.parquet")
    stored_h1["ts"] = stored_h1.ts.dt.tz_convert(KST).dt.as_unit("us")
    output = args.output / "results.json"
    selected_months = args.month or MONTHS
    results = []
    if args.month and output.exists():
        results = [row for row in json.loads(output.read_text(encoding="utf-8"))
                   if row["month"] not in selected_months]
        require(all(args.protocol is None or row["protocol"] == args.protocol for row in results),
                "Existing output uses a different protocol")
    with threadpool_limits(limits=4):
        prep, namespace = group_preparation(args.dataset, args.output)
        for month in selected_months:
            fold = site_oof[site_oof.fold == month]
            old_sites = sites.assign(group=np.where(reference, "G1", "G2"))
            old_mapping = old_sites.set_index("site_id").group
            # Keep scoring on the site OOF cohort, as in eval_v7.py.
            old_score = score(fold, old_mapping, stored_h1[stored_h1.fold == month][GROUP_KEYS + ["H1", "energy"]])
            # Check parity once, then use the existing rolling v6 H1 forecasts.
            max_delta = None
            if args.replay_reference and month == MONTHS[0]:
                fresh = score(fold, old_mapping, fit_h1(prep, namespace, old_sites, market, month))
                max_delta = float((old_score.H1 - fresh.H1).abs().max())
                require(max_delta < 1e-5, "Reference H1 replay failed")
            actual_gap = float((old_score.actual - old_score.energy).abs().max())
            print(f"{month} v6 {old_score.penalty.sum():.2f}, H1 replay max delta {max_delta}", flush=True)
            mappings = []
            excluded_times = set()
            for protocol, mask in (("leave_month_out", market_oof.fold.ne(month)),
                                   ("past_only", market_oof.fold.lt(month))):
                assignment = optimise(errors.to_numpy()[mask], weights[mask], capacities, reference)
                capacity = [float(capacities[assignment].sum()), float(capacities[~assignment].sum())]
                require(min(capacity) >= 20000, "Composition violates capacity floor")
                new_sites = sites.assign(group=np.where(assignment, "G1", "G2"))
                mapping = new_sites.set_index("site_id").group
                keys = pd.MultiIndex.from_frame(fold[["ts"]].assign(group=fold.site_id.map(mapping))
                                                [GROUP_KEYS].drop_duplicates())
                # A sparse hour can have observations for only one of the groups.
                # Exclude that entire hour from all arms when their scored keys differ.
                excluded_times.update(old_score.index.symmetric_difference(keys).get_level_values("ts"))
                mappings.append((protocol, mask, assignment, capacity, new_sites, mapping))

            original_v6 = float(old_score.penalty.sum())
            old_score = old_score[~old_score.index.get_level_values("ts").isin(excluded_times)]
            for protocol, mask, assignment, capacity, new_sites, mapping in mappings:
                if args.protocol is not None and protocol != args.protocol:
                    continue
                candidate = score(fold, mapping, fit_h1(prep, namespace, new_sites, market, month))
                candidate = candidate[~candidate.index.get_level_values("ts").isin(excluded_times)]
                require(old_score.index.equals(candidate.index), "Unequal scored group-time keys")
                by_time = ["actual", "smp", "k"]
                old_total = old_score[by_time].astype(float).groupby("ts").sum()
                new_total = candidate[by_time].astype(float).groupby("ts").sum()
                actual_delta = float((old_total.actual - new_total.actual).abs().max())
                require(np.allclose(old_total.actual, new_total.actual, rtol=3e-7, atol=1e-5),
                        f"Portfolio actual changed by {actual_delta:.9g} kWh")
                require(old_total[["smp", "k"]].equals(new_total[["smp", "k"]]), "Market values changed")
                a, b = float(old_score.penalty.sum()), float(candidate.penalty.sum())
                record = dict(month=month, protocol=protocol, selection_months=sorted(market_oof.fold[mask].unique()),
                              rows=len(old_score), excluded_times=sorted(t.isoformat() for t in excluded_times),
                              max_portfolio_actual_delta_kwh=actual_delta, max_h1_vs_site_actual_kwh=actual_gap,
                              sites=int(fold.site_id.nunique()), moved=int((assignment != reference).sum()),
                              capacity_kw=capacity, v6=a, v7=b, reduction_pct=100 * (1 - b / a),
                              stored_v6=original_v6, h1_replay_max_kwh=max_delta,
                              mapping_sha256=hashlib.sha256(mapping.to_csv(lineterminator="\n").encode()).hexdigest())
                results.append(record)
                print(json.dumps(record), flush=True)
                output.write_text(json.dumps(sorted(results, key=lambda row: (row["month"], row["protocol"])),
                                             indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
