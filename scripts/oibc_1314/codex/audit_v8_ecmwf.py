'''Independently replay the OIBC v8 package and sample original ECMWF forecast fields.'''
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import eccodes
import numpy as np
import pandas as pd
import requests


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def field(day, step, sites):
    run = day - timedelta(days=2)
    stamp = run.strftime("%Y%m%d")
    root = f"https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com/{stamp}/12z/ifs/0p25/oper/{stamp}120000-{step}h-oper-fc"
    idx = requests.get(root+".index", timeout=25)
    idx.raise_for_status()
    record, = [j for line in idx.text.splitlines() if (j := json.loads(line))["param"] == "ssrd"]
    offset, length = record["_offset"], record["_length"]
    raw = requests.get(root+".grib2", headers={"Range": f"bytes={offset}-{offset+length-1}"}, timeout=25)
    raw.raise_for_status()
    assert raw.status_code == 206 and len(raw.content) == length
    h = eccodes.codes_new_from_message(raw.content)
    try:
        def get(key):
            return eccodes.codes_get(h, key)
        assert get("shortName") == "ssrd" and get("stepType") == "accum"
        assert (get("dataDate"), get("dataTime"), get("endStep")) == (int(stamp), 1200, step)
        start_step = get("startStep")
        valid = datetime.strptime(f"{get('validityDate')} {get('validityTime'):04d}", "%Y%m%d %H%M")
        assert valid == datetime.combine(day, datetime.min.time()) + timedelta(hours=step-36)
        values = np.array([eccodes.codes_grib_find_nearest(h, lat, lon)[0]["value"]
                           for lat, lon in zip(sites.lat, sites.lon)])
    finally:
        eccodes.codes_release(h)
    modified = max(parsedate_to_datetime(x.headers["Last-Modified"]) for x in (idx, raw))
    cutoff = datetime.combine(day-timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)+timedelta(hours=8)
    return values, dict(step=step, start_step=start_step, url=root+".grib2", last_modified=modified.isoformat(),
                        before_cutoff=modified<=cutoff, valid_utc=valid.isoformat())


def sample(day, sites, archive):
    a, ainfo = field(day, 36, sites)
    b, binfo = field(day, 42, sites)
    saved = archive.loc[archive.day.eq(pd.Timestamp(day))].set_index("site_id").ssrd.reindex(sites.site_id)
    diff = np.abs((b-a)/21600 - saved.to_numpy())
    assert len(diff) == 253 and np.isfinite(diff).all() and diff.max() < 1e-7
    return dict(day=str(day), rows=len(diff), max_abs_wm2=float(diff.max()), same_accumulation_origin=ainfo["start_step"] == binfo["start_step"], objects=[ainfo,binfo])


def run(dataset, personal, project, candidate_root, output, repo):
    output.mkdir(exist_ok=True, parents=True)
    v7 = candidate_root / "Y1314_v7_comp_dayavg_a05"
    v8 = candidate_root / "Y1314_v8_ecmwf_day_a05"
    sites = pd.read_csv(dataset/"sites.csv")
    mapping = pd.read_csv(v8/"composition.csv")
    cap = mapping.merge(sites, on="site_id", validate="one_to_one").groupby("group").capacity_kw.sum()
    assert len(mapping) == len(sites) == 253 and set(cap.index)=={"G1","G2"} and cap.ge(20000).all()
    composition = sha(v8/"composition.csv")
    assert composition == sha(v7/"composition.csv") and composition.startswith("1e42864d")
    site_file = str(dataset/"sites.csv")
    composition_file = str(v8/"composition.csv")
    forecast_file = str(v8/"forecast.csv")
    validators = [
        [sys.executable, str(project/"scripts/oibc_submission_validator.py"), "--sites", site_file,
         "--composition", composition_file, "--forecast", forecast_file, "--evaluation-smp", str(dataset/"smp.csv"),
         "--rule", str(project/"scripts/oibc2026_capacity_rule_v0_1.json"), "--expected-time-count", "8760", "--input-capacity-unit", "kW"],
        [sys.executable, "-I", "-X", "utf8", str(repo.parent/"OIBC/official_validator_20261102.py"),
         "--composition", composition_file, "--forecast", forecast_file, "--data", str(dataset)],
    ]
    for num, args in enumerate(validators, start=1):
        result = subprocess.run(args, capture_output=True)
        (output/f"validator_{num}.log").write_bytes(result.stdout+result.stderr)
        result.check_returncode()
        if num == 1:
            assert json.loads(result.stdout)["passed"]
    replay = {}
    for source, suffix, prefix in [(v7,"a1","7e1bbe90"),(v8,"ec","68b25cd6")]:
        dest = output / ("replay_" + suffix)
        run_path = ["scripts/oibc_1314/claude_1109/combine_day.py", str(dataset),
                    str(personal/"final_site/v4b_comp_forecast.csv"), str(dest), "0.5"]
        subprocess.run([sys.executable, *run_path, *[str(personal/f"final_day/comp_day{year}_{suffix}")
                        for year in (23,21)]], cwd=repo, check=True, capture_output=True)
        expected = sha(source/"forecast.csv")
        assert expected.startswith(prefix) and sha(dest/"forecast.csv") == expected
        replay[suffix] = expected
    keys = ["group","ts"]
    prev = pd.read_csv(v7/"forecast.csv").set_index(keys).sort_index()
    new = pd.read_csv(v8/"forecast.csv").set_index(keys).sort_index()
    assert new.index.is_unique and prev.index.equals(new.index) and len(new) == 17520
    assert set(new.index.get_level_values("group")) == set(cap.index)
    changed = int(new.forecast_kwh.ne(prev.forecast_kwh).sum())
    total = float(new.forecast_kwh.sum()/1e6)
    key = ["fold",*keys]
    a = pd.read_csv(candidate_root.parent/"final_v7_20261111/2_evidence/fold_predictions_comp.csv").set_index(key).sort_index()
    b = pd.read_csv(personal/"cv/dump_ec/fold_predictions_comp.csv").set_index(key).sort_index()
    assert len(a) == 3749 and a.index.is_unique and a.index.equals(b.index)
    fixed = ["energy","smp","k","E","B21","H1","blend_before_day"]
    pd.testing.assert_frame_equal(a[fixed],b[fixed])
    scores = pd.DataFrame({n:(.1*x.smp*x.k*(x.energy-x.final_pred).abs()).groupby("fold").sum()
                           for n,x in [("v7",a),("v8",b)]})
    path = personal/"gfs/ecmwf_all.parquet"
    external = pd.read_parquet(path)
    assert sha(path).startswith("98271985") and external.day.nunique() == 816
    assert not external.duplicated(["site_id","day"]).any() and np.isfinite(external.ssrd).all()
    eval_dates = pd.date_range("2025-06-01","2026-05-31")
    eval_ec = external[external.day.isin(eval_dates)]
    assert eval_ec.day.nunique()==365 and len(eval_ec)==365*253 and set(eval_ec.site_id)==set(sites.site_id)
    dates = [date.fromisoformat(x) for x in ("2024-07-15","2024-10-15","2025-01-15","2025-02-15","2025-04-15","2025-06-01","2025-12-15","2026-05-31")]
    def audit(day):
        try:
            return sample(day,sites,external)
        except Exception as exc:
            return dict(day=str(day), error=repr(exc))
    with ThreadPoolExecutor(4) as ex:
        samples = list(ex.map(audit, dates))
    report = dict(composition_sha256=composition, forecast_replay=replay, changed_rows=changed,
                  forecast_gwh=total, capacity_kw=cap.to_dict(), scores=scores.to_dict("index"),
                  totals=scores.sum().to_dict(), ecmwf_sha256=sha(path), archive_days=816,
                  archive_rows=len(external), evaluation_archive_rows=len(eval_ec), samples=samples)
    (output/"summary.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    for name in ("dataset","personal","project","candidates","output","repo"):
        cli.add_argument(name,type=Path)
    run(**dict(zip(("dataset","personal","project","candidate_root","output","repo"),vars(cli.parse_args()).values())))
