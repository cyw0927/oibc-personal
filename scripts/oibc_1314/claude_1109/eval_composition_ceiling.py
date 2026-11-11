"""Bound savings from VPP reassignment with site forecasts or group forecasts held fixed."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from eval_v7_full_heldout import GROUP_KEYS, MONTHS, join_site_oof, load, optimise, require, score


def cost(data, weight, mask, offset, portfolio, assignment):
    first = data[mask] @ assignment - offset[mask]
    return float(weight[mask] @ (np.abs(first) + np.abs(portfolio[mask] - first)))


def anchored_search(data, weight, capacity, initial, offset, portfolio):
    """Greedy capacity-feasible single-site moves, keeping group forecast totals fixed."""
    assignment = initial.astype(float).copy()
    total_capacity = capacity.sum()
    for _ in range(200):
        first = data @ assignment - offset
        baseline = weight @ (np.abs(first) + np.abs(portfolio - first))
        direction = 1 - 2 * assignment
        candidates = first[:, None] + data * direction
        penalties = (weight[:, None] * (np.abs(candidates) + np.abs(portfolio[:, None] - candidates))).sum(axis=0)
        size = capacity @ assignment + direction * capacity
        penalties[(size < 20000) | (total_capacity - size < 20000)] = np.inf
        best = int(penalties.argmin())
        if penalties[best] >= baseline - 1e-6:
            break
        assignment[best] = 1 - assignment[best]
    return assignment.astype(bool)


def refine_swaps(data, weight, capacity, initial, offset, portfolio):
    assignment = initial.copy()
    total_capacity = capacity.sum()
    # Zero-weight hours make no contribution to any candidate's cost.
    nonzero = weight > 0
    data, weight = data[nonzero], weight[nonzero]
    offset, portfolio = offset[nonzero], portfolio[nonzero]
    for iteration in range(30):
        first = data @ assignment.astype(float) - offset
        base = float(weight @ (abs(first) + abs(portfolio - first)))
        left, right = np.where(assignment)[0], np.where(~assignment)[0]
        removed, added = np.repeat(left, len(right)), np.tile(right, len(left))
        size = capacity[assignment].sum() - capacity[removed] + capacity[added]
        valid = (size >= 20000) & (total_capacity - size >= 20000)
        removed, added = removed[valid], added[valid]
        best_cost, best_pair = base - 1e-6, None
        for batch in range(0, len(removed), 128):
            rem, add = removed[batch:batch+128], added[batch:batch+128]
            revised = first[:, None] + data[:, add] - data[:, rem]
            loss = (weight[:, None] * (abs(revised) + abs(portfolio[:, None] - revised))).sum(axis=0)
            index = int(loss.argmin())
            if loss[index] < best_cost:
                best_cost, best_pair = float(loss[index]), (rem[index], add[index])
        if best_pair is None:
            return assignment, iteration, True
        removed_site, added_site = best_pair
        assignment[removed_site] = False
        assignment[added_site] = True
    return assignment, 30, False


def assess(data, weight, months, capacity, reference, v7, offset=None, portfolio=None):
    fixed_groups = offset is not None
    if not fixed_groups:
        offset = np.zeros(len(data))
        portfolio = data.sum(axis=1)

    def find(mask, start):
        if fixed_groups:
            return anchored_search(data[mask], weight[mask], capacity, start, offset[mask], portfolio[mask])
        return optimise(data[mask], weight[mask], capacity, start)

    def evaluate(mask, assignment):
        return cost(data, weight, mask, offset, portfolio, assignment)

    all_rows = np.ones(len(data), dtype=bool)
    base = evaluate(all_rows, v7)
    original = evaluate(all_rows, reference)
    lower = float(weight @ np.abs(portfolio))
    require(lower <= base + 1e-7, "Invalid lower bound")
    best = min((v7, find(all_rows, v7), find(all_rows, reference)),
               key=lambda assignment: evaluate(all_rows, assignment))
    best, swaps, stopped = refine_swaps(data, weight, capacity, best, offset, portfolio)
    best_after_moves = find(all_rows, best)
    best = min((best, best_after_moves), key=lambda assignment: evaluate(all_rows, assignment))
    achieved = evaluate(all_rows, best)
    require(lower <= achieved + 1e-7, "Candidate below the lower bound")
    require(min(capacity[best].sum(), capacity[~best].sum()) >= 20000, "Capacity violation")
    validation = []
    all_months = np.unique(months)
    for month in all_months[1:]:
        train = months < month
        test = months == month
        assignment = find(train, reference)
        require(min(capacity[assignment].sum(), capacity[~assignment].sum()) >= 20000, "Capacity violation")
        validation.append(dict(month=str(month), current_v7_mapping=evaluate(test, v7),
                               past_selected_mapping=evaluate(test, assignment),
                               moved_from_v7=int(np.count_nonzero(assignment != v7))))
    return dict(months=list(all_months), hours=len(data), reference_penalty=original, v7_mapping_penalty=base,
                theoretical_floor=lower, theoretical_max_savings=base-lower, feasible_in_sample_penalty=achieved,
                feasible_in_sample_savings=base-achieved, accepted_swaps=swaps, swap_search_stopped=stopped, moves_from_v7=int(np.count_nonzero(best != v7)),
                feasible_capacity_kw=[float(capacity[best].sum()), float(capacity[~best].sum())],
                rolling=validation,
                rolling_savings=sum(row["current_v7_mapping"]-row["past_selected_mapping"] for row in validation))


def run(dataset, personal_safe, composition, forecast, output):
    cv = personal_safe / "cv"
    sites = pd.read_csv(dataset / "sites.csv")
    ids = sites.site_id
    require(len(ids) == 253 and ids.is_unique, "Invalid site list")
    mapping = pd.read_csv(composition).set_index("site_id").group
    require(mapping.index.is_unique and set(mapping.index) == set(ids)
            and set(mapping) == {"G1", "G2"}, "Invalid v7 composition")
    v7 = mapping.reindex(ids).eq("G1").to_numpy()
    reference = sites.current_group.eq("A").to_numpy()
    capacity = sites.capacity_kw.to_numpy()

    oof = load(cv / "site_level_B21_12m.parquet")
    oof["error"] = oof.B - oof.energy
    require(oof.site_id.isin(ids).all(), "Unknown site")
    require(oof.groupby("ts")[["smp", "k", "fold"]].nunique().eq(1).all().all(), "Inconsistent market keys")
    error = oof.pivot(index="ts", columns="site_id", values="error").reindex(columns=ids).fillna(0)
    market = oof.groupby("ts")[["smp", "k", "fold"]].first().loc[error.index]
    weight = (0.1 * market.smp * market.k).to_numpy()
    require(np.isfinite(weight).all() and (weight >= 0).all(), "Negative or missing weight invalidates bound")
    twelve = assess(error.to_numpy(), weight, market.fold.to_numpy(), capacity, reference, v7)
    require(len(twelve["months"]) == 12, "Missing historical months")
    twelve["site_rows"] = len(oof)

    # Full historical v7 forecasts are available for five folds, with original mapping, H1 and daily correction.
    v7_sites = join_site_oof(cv, oof)
    group_model = pd.read_parquet(cv / "group_cv_comp.parquet")
    group_model["ts"] = group_model.ts.dt.as_unit("us")
    history = pd.concat([
        score(v7_sites[v7_sites.fold.eq(month)], mapping,
              group_model.loc[group_model.fold.eq(month), GROUP_KEYS + ["H1", "energy"]])
        .reset_index().assign(fold=month) for month in MONTHS
    ], ignore_index=True)
    require(len(history) == 3749 and abs(history.penalty.sum()/1e6 - 74.3891) < 5.1e-5, "v7 replay changed")
    fixed = history.pivot(index="ts", columns="group", values="pred").dropna()
    actual = v7_sites.pivot(index="ts", columns="site_id", values="energy").reindex(index=fixed.index, columns=ids).fillna(0)
    market = history.groupby("ts")[["smp", "k", "fold"]].first().loc[actual.index]
    weight = (0.1 * market.smp * market.k).to_numpy()
    require(np.isfinite(weight).all() and (weight >= 0).all(), "Negative or missing weight invalidates bound")
    actual = actual.astype(float)
    group_delta = actual.sum(axis=1) - fixed.sum(axis=1)
    five = assess(actual.to_numpy(), weight, market.fold.to_numpy(), capacity, reference, v7,
                  offset=fixed.G1.to_numpy(), portfolio=group_delta.to_numpy())
    matched_penalty = history.loc[history.ts.isin(fixed.index), "penalty"].sum()
    require(abs(five["v7_mapping_penalty"] - matched_penalty) < 50, "Anchored score mismatch")
    five["omitted_incomplete_group_hours"] = history.ts.nunique() - len(fixed)

    report = dict(metric="README provisional D05 HOLD", b21_12_months_site_forecast_fixed=twelve,
                  v7_5_months_group_forecast_fixed=five,
                  public_seventh_gap_krw=24_851_594,
                  package_sha256={path.name: sha256(path.read_bytes()).hexdigest() for path in (composition, forecast)})
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "personal_safe", "composition", "forecast", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=4):
        run(args.dataset, args.personal_safe, args.composition, args.forecast, args.output)
