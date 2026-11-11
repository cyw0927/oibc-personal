"""Compare site-only and joint site/VPP LightGBM losses on the frozen v7 folds."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

KST = "Asia/Seoul"
MONTHS = ("2024-07", "2024-10", "2025-01", "2025-02", "2025-04")
SITE_KEYS = ["site_id", "ts"]
GROUP_KEYS = ["group", "ts"]
EXCLUDED = {"site_id", "ts", "energy", "wa_station", "wb_station", "current_group", "smp", "k", "cf", "w", "group"}
BASE_SHA = "a1a2dc229f8bfebe1394bb7b8f4f896e3ec935c733a7ed3e6be5d308f13ffd18"
MAPPING_SHA = "1e42864de9b9545f96b5700efd6682f90def2b8fb1483131ac052e5c331b38e2"
TREES = 240
PARAMS = dict(learning_rate=.05, num_leaves=47, min_data_in_leaf=200, max_bin=63,
              feature_fraction=.8, max_delta_step=1., verbosity=-1,
              num_threads=8, force_col_wise=True, deterministic=True, seed=17)


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def digest(path):
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def kst(series):
    return pd.to_datetime(series, utc=True).dt.tz_convert(KST).dt.as_unit("us")


def huber_gradient(error, width):
    magnitude = np.hypot(error, width)
    return error / magnitude, width**2 / magnitude**3


class Objective:
    def __init__(self, target, shares, group, group_factor, market_weight, actual):
        self.target = target
        self.shares = shares
        self.group = group
        self.group_factor = group_factor
        size = int(group.max() + 1)
        total_coeff = np.bincount(group, weights=group_factor, minlength=size)
        self.actual = np.bincount(group, weights=group_factor * actual, minlength=size)
        self.price = np.bincount(group, weights=group_factor * market_weight, minlength=size) / total_coeff
        self.price /= self.price.mean()
        self.mass = shares.sum() / size
        self.group_majorizer = group_factor * total_coeff[group]

    def loss(self, prediction, group_weight):
        site_grad, site_hess = huber_gradient(prediction - self.target, .07)
        gradient, hessian = site_grad * self.shares, site_hess * self.shares
        if group_weight:
            summed = np.bincount(self.group, weights=self.group_factor * prediction, minlength=len(self.actual))
            group_grad, group_hess = huber_gradient(summed - self.actual, .035)
            scale = group_weight * self.mass * self.price
            gradient += (scale * group_grad)[self.group] * self.group_factor
            # Diagonal majorizer for the coupled group-hour Hessian.
            hessian += (scale * group_hess)[self.group] * self.group_majorizer
        return gradient, np.maximum(hessian, .05 * self.shares)


def build_objective(train, capacity_scale):
    market_weight = train.smp.to_numpy() * train.k.to_numpy()
    require(np.isfinite(market_weight).all() and (market_weight >= 0).all(), "Invalid training prices")
    group, _ = pd.factorize(pd.MultiIndex.from_frame(train[GROUP_KEYS]), sort=False)
    rounds = train.groupby(SITE_KEYS, sort=False).round.transform("size").to_numpy()
    require(np.isin(rounds, [1, 2]).all(), "Unexpected forecast issue count")
    shares = 1 / rounds
    cf = train.energy.to_numpy(dtype=float) / train.capacity_kw.to_numpy(dtype=float)
    return Objective(cf.clip(0, 1.2), shares, group,
                     shares * train.capacity_kw.to_numpy(dtype=float) / capacity_scale, market_weight, cf)


def read_reference(dataset, package, personal):
    composition = package / "1_submit/composition.csv"
    folds = package / "2_evidence/fold_predictions_comp.csv"
    require(digest(composition) == MAPPING_SHA and digest(folds) == BASE_SHA, "Unexpected v7 inputs")
    mapping = pd.read_csv(composition).set_index("site_id").group
    sites = pd.read_csv(dataset / "sites.csv").set_index("site_id").capacity_kw
    require(mapping.index.is_unique and len(mapping) == 253 and set(mapping.index) == set(sites.index), "Bad composition")
    groups = sites.groupby(mapping).sum()
    require(set(groups.index) == {"G1", "G2"} and groups.ge(20000).all(), "v7 capacity constraint")
    ref = pd.read_csv(folds)
    ref["ts"] = kst(ref.ts)
    require(len(ref) == 3749 and not ref.duplicated(["fold", *GROUP_KEYS]).any(), "Unexpected reference keys")
    oof = pd.read_parquet(personal / "cv/site_level_E_5f.parquet", columns=[*SITE_KEYS, "energy", "fold"])
    oof["ts"] = kst(oof.ts)
    oof["group"] = oof.site_id.map(mapping)
    require(len(oof) == 460217 and not oof.duplicated(SITE_KEYS).any() and oof.group.notna().all(), "Unexpected site evaluation keys")
    return mapping, sites, ref, oof


def read_train(path, mapping, market):
    table = pd.read_parquet(path)
    table["ts"] = kst(table.ts)
    require(table["round"].isin([10, 17]).all(), "Forecast issue outside allowed rounds")
    require(not table.duplicated([*SITE_KEYS, "round"]).any(), "Duplicate site/hour/issue")
    table["group"] = table.site_id.map(mapping)
    require(table.group.notna().all(), "Unknown plant")
    table = table.merge(market, on="ts", how="left", validate="many_to_one", copy=False)
    table["k"] = np.where(table.ts.dt.hour.between(11, 15), 3, 1).astype("int8")
    features = [c for c in table if c not in EXCLUDED and all(tag not in c for tag in ("_pch", "_pcd", "_nb"))]
    require(all(pd.api.types.is_numeric_dtype(table[c]) for c in features), "Unexpected model input type")
    return table, features


def evaluate(model, test, features, expected, reference):
    pred = model.predict(test[features], num_threads=8)
    values = np.where(test.elev.to_numpy() > 0, pred.clip(0, None), 0) * test.capacity_kw.to_numpy()
    site = test[SITE_KEYS].assign(prediction=values).groupby(SITE_KEYS, sort=False).prediction.mean().reset_index()
    joined = expected.merge(site, on=SITE_KEYS, how="left", validate="one_to_one")
    require(joined.prediction.notna().all() and len(joined) == len(site), "Evaluation site grid changed")
    group = joined.groupby(GROUP_KEYS).agg(actual=("energy", "sum"), prediction=("prediction", "sum")).reset_index()
    group = reference.merge(group, on=GROUP_KEYS, how="outer", validate="one_to_one", indicator=True)
    require(group["_merge"].eq("both").all(), "Evaluation group/time grid changed")
    require(np.allclose(group.actual, group.energy, atol=.003, rtol=1e-7), "Evaluation actuals differ")
    weight = .1 * group.smp * group.k
    return dict(penalty_krw=float((weight * (group.prediction - group.energy).abs()).sum()),
                site_mae_kwh=float((joined.prediction - joined.energy).abs().mean()),
                midpoint_penalty_krw=float((weight * (group.prediction - group.energy).abs())
                                           .loc[group.ts.dt.hour.between(11, 15)].sum()))


def run(dataset, package, personal, output):
    mapping, capacity, ref, oof = read_reference(dataset, package, personal)
    market = pd.read_csv(dataset / "train/smp.csv")
    market["ts"] = kst(market.ts)
    feature_file = personal / "site_feat2021/site_train.parquet"
    all_data, features = read_train(feature_file, mapping, market)
    print(f"Loaded {len(all_data):,} rows, {len(features)} unchanged B21 predictors", flush=True)
    report = dict(metric="README provisional D05 HOLD", features=features, input_sha256=digest(feature_file),
                  frozen_composition_sha256=MAPPING_SHA, v7_evaluation_sha256=BASE_SHA,
                  rounds=TREES, parameters=PARAMS, site_width=.07, group_width=.035,
                  group_loss_weight=1., folds=[])
    for month in MONTHS:
        start = pd.Timestamp(f"{month}-01", tz=KST)
        end = start + pd.offsets.MonthBegin()
        fit_until = start - pd.Timedelta(hours=7)  # 17:00 KST before the first target day.
        training_mask = (all_data.ts <= fit_until) & all_data.energy.notna() & all_data.elev.gt(0)
        evaluation_mask = all_data.ts.between(start + pd.Timedelta(hours=1), end)
        ref_month = ref.loc[ref.fold == month]
        expected = oof.loc[oof.fold == month, [*SITE_KEYS, "group", "energy"]]
        test = all_data.loc[evaluation_mask].merge(expected[SITE_KEYS], on=SITE_KEYS, validate="many_to_one")
        require(not test.empty and len(test[SITE_KEYS].drop_duplicates()) == len(expected), f"Incomplete target month {month}")
        actuals = test[[*SITE_KEYS, "energy"]].merge(expected, on=SITE_KEYS, validate="many_to_one")
        require(np.allclose(actuals.energy_x, actuals.energy_y, atol=1e-5, rtol=1e-7), "Source actuals differ")
        train = all_data.loc[training_mask]
        require(not train.empty and train.ts.max() <= fit_until and train.ts.max() < test.ts.min(), "Overlapping folds")
        obj = build_objective(train, capacity.sum() / 2)
        data = lgb.Dataset(train[features], label=obj.target, params={"max_bin": PARAMS["max_bin"]})
        row = dict(month=month, train_rows=len(train), train_max=train.ts.max().isoformat(), site_rows=len(expected),
                   group_rows=len(ref_month),
                   v7=float((.1 * ref_month.smp * ref_month.k * (ref_month.final_pred - ref_month.energy).abs()).sum()))
        print(f"{month} training {len(train):,} issue rows, last actual {row['train_max']}", flush=True)
        for name, group_weight in (("site_only", 0.), ("site_and_group", 1.)):
            def objective(prediction, _dataset):
                return obj.loss(prediction, group_weight)
            model = lgb.train({**PARAMS, "objective": objective}, data, num_boost_round=TREES)
            row[name] = evaluate(model, test, features, expected, ref_month)
            print(name, row[name], flush=True)
        report["folds"].append(row)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["totals"] = dict(v7=sum(row["v7"] for row in report["folds"]),
                            **{name: sum(row[name]["penalty_krw"] for row in report["folds"])
                               for name in ("site_only", "site_and_group")})
    require(sum(row["group_rows"] for row in report["folds"]) == 3749
            and abs(report["totals"]["v7"] - 74_389_091.15) < 1, "v7 replay changed")
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["totals"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "package", "personal", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run(args.dataset, args.package, args.personal, args.output)
