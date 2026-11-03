"""XGBoost stacks on the full train split: chosen on validation, frozen, then scored on test.

A POST-HOC look, asked for on 2 Nov 2026, after the final pipeline (MLP + TCN, logistic) was
frozen and scored once on test (`flex.score_test`). It does not change that pipeline or its
headline. Anything it shows on test must be reported as a second look at test.

Same protocol as `flex.run_gbt`, for three sets of inputs:

    XGBoost              the trees' 68 inputs
    XGBoost + TCN        + the TCN's score
    XGBoost + MLP + TCN  + the MLP's and TCN's scores and the 7 cosines (as GBT + MLP + TCN)

1. Choose on validation. Fit on all 274 train classes; validation AP every 20 trees picks
   (tree size, up-weighting of raises, number of trees). The choices are written to disk
   before any test row is scored, and nothing after that changes them.
2. Score. Each model is fitted two ways with the frozen settings:
   * fold models, as the main pipeline: 4 fits on 3/4 of the train classes, out-of-fold train
     scores, each validation or test session scored by its assigned fold model. Directly
     comparable with MLP + TCN, whose scores are made this way.
   * one fit on all 274 train classes.
   Specialist scores come from the saved runs: out-of-fold on train, fold-assigned on
   validation and test. The MLP and TCN are not retrained.
3. Report, with the same session bootstrap (1,000 paired resamples, same seed) as the notebook.

Reads data/results/flex/ and never writes there; writes data/results/xgb/ (local, gitignored).
`python xgb_full.py` from pipeline1/. Delete data/results/xgb/ to choose again.
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from oncue import baselines, config, flex, metrics  # noqa: E402

OUT = config.DATA_DIR / "results" / "xgb"
STEP, MAX_TREES = flex.STAGE_EVERY, flex.GBT_MAX_TREES   # 20, 600 as HGB

# The grid of xgb_vs_hgb.ipynb, so this run and that one test the same model.
XGB_BASE = dict(learning_rate=0.05, reg_lambda=1.0, colsample_bynode=0.8, tree_method="hist",
                grow_policy="lossguide", max_depth=0, enable_categorical=True,
                random_state=config.SEED, n_jobs=-1)
XGB_GRID = ({"max_leaves": 15, "min_child_weight": 5}, {"max_leaves": 31, "min_child_weight": 5},
            {"max_leaves": 63, "min_child_weight": 10})
XGB_NAMES = {"xgb": "XGBoost", "xgb_tcn": "XGBoost + TCN", "xgb_mlp_tcn": "XGBoost + MLP + TCN"}
REF_NAMES = {"mlp_tcn_lr": "MLP + TCN, logistic (final)", "tcn": "TCN", "mlp": "MLP",
             "gbt": "HGB", "gbt_mlp_tcn": "HGB + MLP + TCN"}


def log(*a):
    print(*a, flush=True)


def _xgb(setting: dict, spw: float, n_trees: int) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(**XGB_BASE, **setting, scale_pos_weight=spw, n_estimators=int(n_trees))


def _load(name: str):
    path = OUT / f"{name}.pkl"
    if path.exists():
        with open(path, "rb") as fh:
            return pickle.load(fh)
    return None


def _save(obj, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{name}.pkl", "wb") as fh:
        pickle.dump(obj, fh)


def choose(X: pd.DataFrame, y: np.ndarray, tr: np.ndarray, va: np.ndarray) -> dict:
    """Fit each setting once with 600 trees on all train; validation AP every 20 trees picks.
    Touches train and validation rows only."""
    Xtr, Xva = X.iloc[tr], X.iloc[va]
    ytr, yva = y[tr], y[va]
    spw_up = float(np.sqrt((len(ytr) - ytr.sum()) / ytr.sum()))
    best, curve = None, []
    for st in XGB_GRID:
        for spw in (1.0, spw_up):
            t0 = time.time()
            m = _xgb(st, spw, MAX_TREES).fit(Xtr, ytr)
            for n in range(STEP, MAX_TREES + 1, STEP):
                ap = average_precision_score(yva, m.predict_proba(Xva, iteration_range=(0, n))[:, 1])
                curve.append({**st, "spw": spw, "trees": n, "val_AP": ap})
                if best is None or ap > best["val_AP"]:
                    best = {"setting": st, "spw": spw, "trees": n, "val_AP": ap}
            log(f"    {st['max_leaves']} leaves, pos. weight {spw:.1f}: best val AP so far "
                f"{best['val_AP']:.4f} ({time.time() - t0:.0f} s)")
    return {**best, "curve": pd.DataFrame(curve)}


def score(X: pd.DataFrame, y: np.ndarray, rows: dict, cid: np.ndarray, choice: dict) -> dict:
    """Fold models (out-of-fold train, fold-assigned validation and test) and one all-train fit."""
    tr, va, te = rows["train"], rows["validation"], rows["test"]
    st, spw, n = choice["setting"], choice["spw"], choice["trees"]
    folds, vf, tf = flex.fold_labels(cid[tr]), flex.val_folds(cid[va]), flex.val_folds(cid[te])
    Xtr, Xva, Xte = X.iloc[tr], X.iloc[va], X.iloc[te]
    ytr = y[tr]
    oof, val, test = np.zeros(len(tr)), np.zeros(len(va)), np.zeros(len(te))
    for k in range(flex.N_FOLDS):
        m = _xgb(st, spw, n).fit(Xtr[folds != k], ytr[folds != k])
        oof[folds == k] = m.predict_proba(Xtr[folds == k])[:, 1]
        val[vf == k] = m.predict_proba(Xva[vf == k])[:, 1]
        test[tf == k] = m.predict_proba(Xte[tf == k])[:, 1]
    m = _xgb(st, spw, n).fit(Xtr, ytr)
    return {"oof": oof, "val": val, "test": test,
            "val_full": m.predict_proba(Xva)[:, 1], "test_full": m.predict_proba(Xte)[:, 1]}


def table(frame_y: pd.DataFrame, frame_guess: pd.DataFrame, preds: dict, names: dict, taus: dict,
          raises: pd.DataFrame, ci: pd.DataFrame) -> pd.DataFrame:
    y, c = frame_y["y"].to_numpy(), frame_y["class_id"].to_numpy()
    rand_w, rand = metrics.ap_within(y, np.full(len(y), 0.5), c), y.mean()
    rows = {}
    for i, p in preds.items():
        pm = baselines.point_metrics(frame_y, p)
        row = {"AP_w": round(pm["AP_w"], 4), "x random": round(pm["AP_w"] / rand_w, 1),
               "AP_w 95%": f"{ci.loc[i, 'AP_w low']:.3f}-{ci.loc[i, 'AP_w high']:.3f}",
               "AP": round(pm["AP"], 4), "AP x random": round(pm["AP"] / rand, 1),
               "AP 95%": f"{ci.loc[i, 'AP low']:.3f}-{ci.loc[i, 'AP high']:.3f}",
               "ROC AUC": round(pm["ROC_AUC"], 3)}
        if i in taus:
            s = baselines.guess_scores(frame_guess, p, taus[i], raises)
            row |= {"warnings": s["warnings"], "right": s["warnings_right"],
                    "% right": f"{100 * s['pct_warnings_right']:.1f}%" if s["warnings"] else "-",
                    "raises warned": f"{s['raises_warned']} of {s['raises']}"}
        rows[names[i]] = row
    return pd.DataFrame(rows).T.fillna("")


def gains(diff: pd.DataFrame, names: dict) -> pd.DataFrame:
    return pd.DataFrame(
        {"AP_w gain of final": [f"{g:+.4f} ({lo:+.4f} to {hi:+.4f})" for g, lo, hi in
                                zip(diff["AP_w gain"], diff["AP_w low"], diff["AP_w high"])],
         "final wins (AP_w)": [f"{100 * w:.0f}%" for w in diff["AP_w: ref better in"]],
         "AP gain of final": [f"{g:+.4f} ({lo:+.4f} to {hi:+.4f})" for g, lo, hi in
                              zip(diff["AP gain"], diff["AP low"], diff["AP high"])],
         "final wins (AP)": [f"{100 * w:.0f}%" for w in diff["AP: ref better in"]]},
        index=[names[i] for i in diff.index])


def main() -> None:
    pd.set_option("display.width", 240, "display.max_columns", 30, "display.max_colwidth", 60)
    t0 = time.time()
    df = flex.load()
    cols = flex.gbt_inputs(df.columns)
    split, y, cid = df["split"].to_numpy(), df["y"].to_numpy(), df["class_id"].to_numpy()
    rows = {s: np.flatnonzero(split == s) for s in ("train", "validation", "test")}
    tr, va, te = rows["train"], rows["validation"], rows["test"]
    mlp, tcn = flex.load_result("mlp"), flex.load_result("tcn")
    comb, T = flex.load_result(flex.FINAL), flex.load_result("test_final")
    assert (flex.fold_labels(cid[tr]) == mlp["folds"]).all()    # the specialists' own folds

    def scores(oof, val, test):
        v = np.full(len(df), np.nan)
        v[tr], v[va], v[te] = oof, val, test
        assert np.isfinite(v).all()
        return v

    mlp_s = scores(mlp["oof"], mlp["val"], T["mlp_test"])
    tcn_s = scores(tcn["oof"], tcn["val"], T["tcn_test"])
    cos = pd.DataFrame(flex.vector_inputs(df)["cosines"], columns=list(flex.COSINES), index=df.index)
    inputs = {"xgb": df[cols], "xgb_tcn": df[cols].assign(tcn_score=tcn_s),
              "xgb_mlp_tcn": pd.concat([df[cols].assign(mlp_score=mlp_s, tcn_score=tcn_s), cos], axis=1)}
    log(f"{len(df):,} moments; train {len(tr):,}, validation {len(va):,}, test {len(te):,}; "
        f"inputs {({k: v.shape[1] for k, v in inputs.items()})}")

    # 1. Choose on validation; frozen on disk before any test row is scored.
    choices = _load("choices")
    if choices is None:
        choices = {}
        for name, X in inputs.items():
            log(f"choosing {XGB_NAMES[name]} on validation")
            choices[name] = choose(X, y, tr, va)
        _save(choices, "choices")
    for name, c in choices.items():
        log(f"FROZEN {XGB_NAMES[name]}: {c['setting']['max_leaves']} leaves, {c['trees']} trees, "
            f"pos. weight {c['spw']:.1f}; validation AP {c['val_AP']:.4f}")

    # 2. Score.
    res = _load("scores")
    if res is None:
        res = {}
        for name, X in inputs.items():
            t1 = time.time()
            res[name] = score(X, y, rows, cid, choices[name])
            log(f"scored {XGB_NAMES[name]} ({time.time() - t1:.0f} s)")
        _save(res, "scores")

    # 3. Report, with the saved models as references.
    gbt, gbt3 = flex.load_result("gbt"), flex.load_result("gbt_mlp_tcn")
    names = {**REF_NAMES, **XGB_NAMES, **{f"{k}_all": f"{v}, all train" for k, v in XGB_NAMES.items()}}
    p_val = {"mlp_tcn_lr": comb["val"], "tcn": tcn["val"], "mlp": mlp["val"], "gbt": gbt["val"],
             "gbt_mlp_tcn": gbt3["val"]}
    p_test = {"mlp_tcn_lr": T["test"], "tcn": T["tcn_test"], "mlp": T["mlp_test"]}
    for k in XGB_NAMES:
        p_val |= {k: res[k]["val"], f"{k}_all": res[k]["val_full"]}
        p_test |= {k: res[k]["test"], f"{k}_all": res[k]["test_full"]}
    ytr = y[tr]
    taus = {"mlp_tcn_lr": baselines.tau_like_truth(ytr, comb["oof"]),
            "tcn": baselines.tau_like_truth(ytr, tcn["oof"]),
            "mlp": baselines.tau_like_truth(ytr, mlp["oof"]),
            "gbt": baselines.tau_like_truth(ytr, gbt["oof"]),
            "gbt_mlp_tcn": baselines.tau_like_truth(ytr, gbt3["oof"]),
            **{k: baselines.tau_like_truth(ytr, res[k]["oof"]) for k in XGB_NAMES}}

    data = baselines.load(config.TABLES_DIR, config.DATA_DIR / "03_split" / "split.csv")
    raises = baselines.onsets(config.TABLES_DIR, data["class_id"].unique())
    report = {}
    for s, preds, title in (("validation", p_val, "VALIDATION, 86 sessions"), ("test", p_test, "TEST, 86 sessions (second look)")):
        frame = df[df["split"] == s].reset_index(drop=True)
        guess = data[data["split"] == s]
        assert (guess[["class_id", "t_s"]].to_numpy() == frame[["class_id", "t_s"]].to_numpy()).all()
        t1 = time.time()
        ci, diff = flex.session_bootstrap(frame, preds, ref="mlp_tcn_lr")
        log(f"\n{title} (bootstrap {time.time() - t1:.0f} s)")
        tab = table(frame, guess, preds, names, taus, raises, ci)
        log(tab.to_string())
        gain = gains(diff, names)
        log("\nDoes the final pipeline really beat them? Paired session bootstrap, 1,000 resamples")
        log(gain.to_string())
        report[s] = {"table": tab, "gains": gain, "ci": ci, "diff": diff}
    _save(report, "report")

    # Reproduces the saved references?
    TB = flex.load_result("test_bootstrap")
    frame = df[df["split"] == "test"].reset_index(drop=True)
    now = baselines.point_metrics(frame, T["test"])
    log(f"\ncheck, final pipeline on test: AP_w {now['AP_w']:.4f} (saved {TB['point'][flex.FINAL]['AP_w']:.4f}); "
        f"interval {report['test']['ci'].loc[flex.FINAL, 'AP_w low']:.4f}-"
        f"{report['test']['ci'].loc[flex.FINAL, 'AP_w high']:.4f} "
        f"(saved {TB['ci'].loc[flex.FINAL, 'AP_w low']:.4f}-{TB['ci'].loc[flex.FINAL, 'AP_w high']:.4f})")
    log(f"RUN OK in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
