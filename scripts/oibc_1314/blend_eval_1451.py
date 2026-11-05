"""#1451 DEV re-blend of the #1314 group-direct model on a new bottom-up reference (#1456 criteria).

Compares only the pre-registered candidates (no weight search):
  A_REF      = bottom-up reference (e.g. FS2+L1 3-seed average) group sum
  GD_ONLY    = group-direct (3-config mean)
  B_04_06    = 0.4 * A_REF + 0.6 * GD_ONLY
  C_05_05    = 0.5 * A_REF + 0.5 * GD_ONLY
  C_06_04    = 0.6 * A_REF + 0.4 * GD_ONLY

Inputs are DEV OOF only (ts < 2025-03-01 KST); HOLDOUT is never read.
Metric (README settlement, provisional, D05 HOLD): group penalty = sum 0.10*SMP*k*|sum(pred)-sum(actual)|.

Usage:
  python -I blend_eval_1451.py --fs2-inbox <dir with 3 FS2 run folders> --ref-oof <reference oof_dev.parquet>
         --group-preds <dev_group_preds_ens.parquet> --sites <sites.csv> --out <result.json>
The FS2 runs are used only to check that the group-direct file was built on the same DEV keys.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["site_id", "ts", "fold"]
GK = ["fold", "group", "ts"]
LIMIT = pd.Timestamp("2025-03-01", tz="Asia/Seoul")
FOLDS = ["DEV_JUL_2024", "DEV_OCT_2024", "DEV_JAN", "DEV_FEB"]
WEIGHTS = {"B_04_06": 0.4, "C_05_05": 0.5, "C_06_04": 0.6}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_oof(p: Path) -> pd.DataFrame:
    d = pd.read_parquet(p, columns=KEY + ["actual_kwh", "prediction_kwh"]).sort_values(KEY).reset_index(drop=True)
    assert d.ts.max() < LIMIT and not d.duplicated(KEY).any(), p
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    for a in ("--fs2-inbox", "--ref-oof", "--group-preds", "--sites", "--out"):
        ap.add_argument(a, required=True, type=Path)
    args = ap.parse_args()

    fs2_paths = sorted(args.fs2_inbox.glob("*/oof_dev.parquet"))
    assert len(fs2_paths) == 3, fs2_paths
    fs2 = [load_oof(p) for p in fs2_paths]
    a = fs2[0][KEY + ["actual_kwh"]].copy()
    for f in fs2[1:]:
        assert a[KEY].equals(f[KEY]) and np.array_equal(a.actual_kwh, f.actual_kwh)
    a["fs2"] = sum(f.prediction_kwh for f in fs2) / 3

    ref = load_oof(args.ref_oof)
    assert a[KEY].equals(ref[KEY]), "reference keys differ from FS2 keys"
    assert np.allclose(a.actual_kwh, ref.actual_kwh, rtol=0, atol=1e-9), "actual differs"
    a["ref"] = ref.prediction_kwh.values
    assert np.isfinite(a.ref).all() and (a.ref >= 0).all()

    sites = pd.read_csv(args.sites, usecols=["site_id", "current_group"])
    sites["group"] = sites.current_group.map({"A": "G1", "B": "G2"})   # #990 reference mapping
    assert len(sites) == 253 and sites.site_id.is_unique and sites.group.notna().all()
    a = a.merge(sites[["site_id", "group"]], on="site_id", validate="many_to_one")

    gp = pd.read_parquet(args.group_preds)
    assert gp.ts.max() < LIMIT and not gp.duplicated(GK).any()
    g = a.groupby(GK, as_index=False).agg(actual=("actual_kwh", "sum"), fs2=("fs2", "sum"), ref=("ref", "sum"))
    g = g.merge(gp[GK + ["smp", "k", "fs2", "actual", "direct_ens"]], on=GK, suffixes=("", "_gp"), validate="one_to_one")
    assert np.allclose(g.fs2, g.fs2_gp, atol=1e-6) and np.allclose(g.actual, g.actual_gp, atol=1e-6), \
        "group-direct file was not built on the same DEV keys"

    cand = {"A_REF": g.ref, "GD_ONLY": g.direct_ens}
    cand.update({k: w * g.ref + (1 - w) * g.direct_ens for k, w in WEIGHTS.items()})
    for k, v in cand.items():
        g[k] = v

    def wmae(d, p):
        w = d.smp * d.k
        return float((w * (d[p] - d.actual).abs()).sum() / (w * d.actual).sum())

    def pen(d, p):
        return float((0.10 * d.smp * d.k * (d[p] - d.actual).abs()).sum())

    def bias(d, p):
        return float((d[p] - d.actual).sum() / d.actual.sum())

    res = {"candidates": {}}
    for k in cand:
        r = {"group_wmae": {"ALL": wmae(g, k), **{f: wmae(g[g.fold == f], k) for f in FOLDS}},
             "penalty_won": {"ALL": pen(g, k), **{f: pen(g[g.fold == f], k) for f in FOLDS}},
             "bias_rel": bias(g, k),
             "midday_bias_rel": {f: bias(g[(g.fold == f) & g.ts.dt.hour.between(11, 15)], k) for f in FOLDS},
             "nan": int(g[k].isna().sum()), "inf": int(np.isinf(g[k]).sum()), "neg": int((g[k] < 0).sum())}
        r["fold_improve_vs_A"] = sum(r["penalty_won"][f] < pen(g[g.fold == f], "A_REF") for f in FOLDS)
        res["candidates"][k] = r
    res["rows_group"] = len(g)
    res["sha256"] = {"ref_oof": sha(args.ref_oof), "group_preds": sha(args.group_preds), "sites": sha(args.sites),
                     **{p.parent.name: sha(p) for p in fs2_paths}}
    res["holdout_used_for_selection"] = False
    args.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    for k, r in res["candidates"].items():
        print(f"{k:8s} wmae {r['group_wmae']['ALL']:.6f} pen {r['penalty_won']['ALL'] / 1e6:.3f}M "
              f"folds {[round(r['penalty_won'][f] / 1e6, 3) for f in FOLDS]} improve_vs_A {r['fold_improve_vs_A']}/4")


if __name__ == "__main__":
    main()
