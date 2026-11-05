"""DEV evaluation for the #1314 group-direct model vs FS2 3-seed bottom-up (reference VPP).

Metric (README settlement, provisional): group penalty = sum 0.10*SMP*k*|sum(pred)-sum(actual)| per group-hour.
Same DEV folds, keys and fit windows as the team FS2 OOF (#1144). HOLDOUT (2025-03~05) never read.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

D = Path(sys.argv[1])
FEAT = Path(sys.argv[2])
INBOX = Path(sys.argv[3])
OUT = Path(sys.argv[4])
OUT.mkdir(parents=True, exist_ok=True)

FOLDS = {  # from FS2 manifest (fit_start 2023-06-01 01:00 for all)
    "DEV_JUL_2024": "2024-06-30 00:00",
    "DEV_OCT_2024": "2024-09-30 00:00",
    "DEV_JAN": "2024-12-31 00:00",
    "DEV_FEB": "2025-01-31 00:00",
}
KST = "Asia/Seoul"
NON_FEAT = {"ts", "group", "energy", "cap_avail", "n_sites", "group_cap_kw"}

sites = pd.read_csv(D / "sites.csv")
sites["group"] = sites["current_group"].map({"A": "G1", "B": "G2"})
smp = pd.read_csv(D / "train" / "smp.csv")
smp["ts"] = pd.to_datetime(smp["ts"]).dt.tz_convert(KST).dt.as_unit("us")
smp["k"] = np.where(smp["ts"].dt.hour.between(11, 15), 3, 1)


def load_fs2() -> pd.DataFrame:
    runs = sorted(p for p in INBOX.iterdir() if p.is_dir() and "fs2-s" in p.name)
    assert len(runs) == 3, runs
    base = None
    for i, r in enumerate(runs):
        d = pd.read_parquet(r / "oof_dev.parquet", columns=["site_id", "ts", "fold", "actual_kwh", "prediction_kwh"])
        d = d.sort_values(["fold", "site_id", "ts"]).reset_index(drop=True)
        if base is None:
            base = d.rename(columns={"prediction_kwh": "p0"})
        else:
            assert (base[["site_id", "ts", "fold"]].values == d[["site_id", "ts", "fold"]].values).all()
            assert np.allclose(base["actual_kwh"], d["actual_kwh"])
            base[f"p{i}"] = d["prediction_kwh"].values
    base["fs2"] = base[["p0", "p1", "p2"]].mean(axis=1)
    base["ts"] = base["ts"].dt.as_unit("us")
    return base


def site_wmae(df: pd.DataFrame, col: str) -> float:
    w = df["smp"] * df["k"]
    return float((w * (df["actual_kwh"] - df[col]).abs()).sum() / (w * df["actual_kwh"]).sum())


def fit_predict(tr: pd.DataFrame, va: pd.DataFrame, weight: str, seed: int = 0) -> np.ndarray:
    feats = [c for c in tr.columns if c not in NON_FEAT and c not in ("cf", "w", "smp", "k", "cov")]
    sw = tr["cov"].values.copy()
    if weight in ("k", "smpk"):
        sw = sw * tr["k"].values
    if weight == "smpk":
        sw = sw * np.clip(tr["smp"].values, 1.0, None) / tr["smp"].clip(lower=1.0).mean()
    m = HistGradientBoostingRegressor(loss="absolute_error", learning_rate=0.05, max_iter=600,
                                      max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0,
                                      early_stopping=False, random_state=seed)
    m.fit(tr[feats], tr["cf"], sample_weight=sw)
    return np.clip(m.predict(va[feats]), 0, 1.0)


def main():
    fs2 = load_fs2()
    fs2 = fs2.merge(smp, on="ts", how="left")
    assert fs2["smp"].notna().all()
    fs2 = fs2.merge(sites[["site_id", "group", "capacity_kw"]], on="site_id")
    res = {"fs2_site_wmae_all": site_wmae(fs2, "fs2"),
           "fs2_site_wmae_by_fold": {f: site_wmae(g, "fs2") for f, g in fs2.groupby("fold")}}

    # group-hour actual / FS2 sums over the OOF key set (subset plants for JUL/OCT)
    gh = fs2.groupby(["fold", "group", "ts"]).agg(actual=("actual_kwh", "sum"), fs2=("fs2", "sum"),
                                                  cap_sub=("capacity_kw", "sum"), smp=("smp", "first"),
                                                  k=("k", "first")).reset_index()

    ft = pd.read_parquet(FEAT / "group_features_train.parquet")
    ft["ts"] = ft["ts"].dt.as_unit("us")
    ft = ft.merge(smp, on="ts", how="left")
    ft["cov"] = ft["cap_avail"] / ft["group_cap_kw"]
    ft["cf"] = (ft["energy"] / ft["cap_avail"]).clip(0, 1.2)
    preds = []
    for fold, cut in FOLDS.items():
        cut = pd.Timestamp(cut, tz=KST)
        vk = gh[gh.fold == fold]
        tr = ft[(ft.ts <= cut) & ft["energy"].notna() & (ft["cov"] >= 0.3) & (ft["cosz_max"] > 0)]
        va = vk[["group", "ts"]].merge(ft.drop(columns=["smp", "k"]), on=["group", "ts"], how="left")
        assert len(va) == len(vk) and va["cosz"].notna().all()
        out = vk.copy()
        for wname in ("one", "k", "smpk"):
            cf = fit_predict(tr, va, wname)
            cf = np.where(va["cosz_max"].values > 0, cf, 0.0)
            out[f"direct_{wname}"] = cf * vk["cap_sub"].values
        preds.append(out)
        print(fold, "train rows", len(tr), flush=True)
    gp = pd.concat(preds, ignore_index=True)
    gp.to_parquet(OUT / "dev_group_preds.parquet", index=False)

    cols = ["fs2", "direct_one", "direct_k", "direct_smpk"]
    def pen(df, c):
        return float((0.10 * df["smp"] * df["k"] * (df[c] - df["actual"]).abs()).sum())
    table = {}
    for c in cols:
        table[c] = {f: pen(g, c) for f, g in gp.groupby("fold")}
        table[c]["ALL"] = pen(gp, c)
    res["group_penalty_won"] = table
    res["group_wmae"] = {c: float((gp.smp * gp.k * (gp[c] - gp.actual).abs()).sum() / (gp.smp * gp.k * gp.actual).sum()) for c in cols}
    json.dump(res, open(OUT / "dev_eval.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


main()
