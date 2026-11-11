"""Independently verify the v7 review ZIP against its submission and reported fold scores."""

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

EXPECTED_HASHES = {
    "composition.csv": "1e42864de9b9545f96b5700efd6682f90def2b8fb1483131ac052e5c331b38e2",
    "forecast.csv": "7e1bbe90301a7dfe1111fef501833cb5c4ea0fd47309fe300621880e99e950c4",
}
KEY = ["group", "ts"]


def check(condition, detail):
    if not condition:
        raise ValueError(detail)


def csv(archive, path):
    return pd.read_csv(BytesIO(archive[path]))


def verify_archive(path):
    with ZipFile(path) as archive:
        files = [entry.filename for entry in archive.infolist() if not entry.is_dir()]
        check(len(set(files)) == len(files), "Duplicate paths in ZIP")
        roots = {name.split("/", 1)[0] for name in files}
        check(len(roots) == 1, "Unexpected package roots")
        contents = {name.split("/", 1)[1]: archive.read(name) for name in files}
    manifest = json.loads(contents["SHA256SUMS.json"])
    check(set(manifest) == set(contents) - {"SHA256SUMS.json"}, "SHA manifest inventory mismatch")
    for name, digest in manifest.items():
        check(sha256(contents[name]).hexdigest() == digest, f"ZIP checksum mismatch: {name}")
    return contents, len(manifest)


def verify_submission(archive, submission, sites_path):
    manifest = json.loads(archive["1_submission/MANIFEST.json"])
    check(manifest["outputs_sha256"] == EXPECTED_HASHES, "Unexpected submission SHA")
    for filename, digest in EXPECTED_HASHES.items():
        disk = (submission / filename).read_bytes()
        check(archive[f"1_submission/{filename}"] == disk, f"ZIP and submission disagree: {filename}")
        check(sha256(disk).hexdigest() == digest, f"Submission checksum mismatch: {filename}")
    composition = csv(archive, "1_submission/composition.csv")
    sites = pd.read_csv(sites_path)
    check(len(sites) == 253 and sites.site_id.is_unique, "Unexpected source site inventory")
    group = composition.merge(sites, on="site_id", how="outer", validate="one_to_one", indicator=True)
    check(group["_merge"].eq("both").all() and set(group.group) == {"G1", "G2"}, "Bad site or group mapping")
    check(group.current_group.isin(["A", "B"]).all(), "Invalid reference labels")
    sizes = group.groupby("group").capacity_kw.agg(["size", "sum"])
    check(sizes["sum"].ge(20_000).all(), "A group is below 20 MW")
    moved = int(group.group.ne(group.current_group.map({"A": "G1", "B": "G2"})).sum())
    check(moved == 81, "Changed-site count differs")
    forecast = csv(archive, "1_submission/forecast.csv").set_index(KEY).sort_index()
    components = csv(archive, "3_final_components/v7_components_group_hour.csv").set_index(KEY).sort_index()
    hours = pd.date_range("2025-06-01 01:00", periods=8760, freq="h", tz="Asia/Seoul").strftime("%Y-%m-%d %H:%M:%S%z")
    hours = hours.str.replace(r"(\+\d{2})(\d{2})$", r"\1:\2", regex=True)
    expected = pd.MultiIndex.from_product([["G1", "G2"], hours], names=KEY)
    check(forecast.index.equals(expected) and components.index.equals(expected), "Unexpected submission time grid")
    check(np.isfinite(forecast.forecast_kwh).all() and forecast.forecast_kwh.ge(0).all(), "Invalid forecast values")
    blend = .35 * components.E + .35 * components.B21 + .30 * components.H1
    ratio = (components.comp_day23_a1 + components.comp_day21_a1) / 2
    prediction = blend * (1 + .5 * (ratio - 1))
    # Component inputs have been stored to six decimal places, including each daily ratio.
    rounding_tolerance = 2e-6 + .25e-6 * blend.abs()
    errors = (prediction - forecast.forecast_kwh).abs()
    check(errors.le(rounding_tolerance).all(), "Forecast fails independent component replay")
    exact_col_error = float((components.v7 - forecast.forecast_kwh).abs().max())
    check(exact_col_error <= 2e-6, "Recorded final components do not match submission")
    return dict(groups=sizes.to_dict(orient="index"), moved_sites=moved, forecast_rows=len(forecast),
                component_replay_max_delta_kwh=float(errors.max()),
                recorded_forecast_max_delta_kwh=exact_col_error, submission_sha256=EXPECTED_HASHES)


def verify_full_precision(archive, personal):
    manifest = json.loads(archive["1_submission/MANIFEST.json"])
    source = {
        "base_forecast.csv": personal / "final_site/v4b_comp_forecast.csv",
        "day_ratio_0": personal / "final_day/comp_day23_a1/day_ratio.parquet",
        "day_ratio_1": personal / "final_day/comp_day21_a1/day_ratio.parquet",
    }
    hashes = {name: sha256(path.read_bytes()).hexdigest() for name, path in source.items()}
    check(hashes == manifest["inputs_sha256"], "Full-precision source hash mismatch")
    ratios = []
    for name in ("day_ratio_0", "day_ratio_1"):
        ratio = pd.read_parquet(source[name], columns=["group", "day", "rd"]).set_index(["group", "day"])
        check(ratio.index.is_unique and len(ratio) == 730 and np.isfinite(ratio.rd).all(),
              f"Invalid daily ratio file: {name}")
        ratios.append(ratio.rd)
    check(ratios[0].index.sort_values().equals(ratios[1].index.sort_values()), "Ratio grids disagree")
    ratio = ((ratios[0] + ratios[1]) / 2).rename("ratio")
    base = pd.read_csv(source["base_forecast.csv"])
    base["day"] = (pd.to_datetime(base.ts, utc=True).dt.tz_convert("Asia/Seoul")
                   - pd.Timedelta(hours=1)).dt.normalize()
    values = base.join(ratio, on=["group", "day"], validate="many_to_one")
    check(values.ratio.notna().all() and len(values) == 17520, "Source replay is missing rows")
    values["forecast_kwh"] = (values.forecast_kwh * (1 + .5 * (values.ratio - 1))).clip(lower=0)
    replay = values[[*KEY, "forecast_kwh"]].sort_values(KEY)
    encoded = replay.to_csv(index=False, lineterminator="\n", float_format="%.6f").encode()
    check(encoded == archive["1_submission/forecast.csv"], "Full-precision submission SHA could not be reproduced")
    return dict(forecast_sha256=sha256(encoded).hexdigest(), source_sha256=hashes)


def verify_scores(archive):
    expected = csv(archive, "2_fold_predictions/eval_by_fold_ref_vs_comp.csv")
    output = {}
    for variant in ("ref", "comp"):
        frame = csv(archive, f"2_fold_predictions/fold_predictions_{variant}.csv")
        check(not frame.duplicated(["fold", *KEY]).any() and len(frame) == 3749, f"Bad fold keys: {variant}")
        fold_start = pd.to_datetime(frame.ts, utc=True).dt.tz_convert("Asia/Seoul") - pd.Timedelta(hours=1)
        check(fold_start.dt.strftime("%Y-%m").eq(frame.fold).all(), f"Bad fold timestamps: {variant}")
        check(np.array_equal(np.where(pd.to_datetime(frame.ts, utc=True).dt.tz_convert("Asia/Seoul")
                                     .dt.hour.between(11, 15), 3, 1), frame.k), "Unexpected k rule")
        weight = .1 * frame.smp * frame.k
        frame["penalty"] = weight * (frame.final_pred - frame.energy).abs()
        rollup = frame.groupby("fold").penalty.agg(["size", "sum"])
        rollup.loc["TOTAL"] = rollup.sum()
        table = expected.loc[expected.composition == variant].set_index("fold").loc[rollup.index]
        check(np.array_equal(rollup["size"], table.rows), f"Fold row-count mismatch: {variant}")
        check(np.allclose(rollup["sum"] / 1e6, table.p_v6, rtol=0, atol=.0000501),
              f"Fold penalty differs from reported rounding: {variant}")
        output[variant] = {name: float(value) for name, value in rollup["sum"].items()}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("zip", "submission", "sites", "output"):
        parser.add_argument(arg, type=Path)
    parser.add_argument("--personal-safe", type=Path)
    args = parser.parse_args()
    contents, checksums = verify_archive(args.zip)
    result = dict(package_sha256=sha256(args.zip.read_bytes()).hexdigest(), manifest_entries_verified=checksums,
                  submission=verify_submission(contents, args.submission, args.sites),
                  fold_penalty_krw=verify_scores(contents))
    if args.personal_safe:
        result["full_precision_replay"] = verify_full_precision(contents, args.personal_safe)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
