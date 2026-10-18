"""The naive models (notebook Step 2): what a simple approach gets, before the main pipeline.

Task: at every counting moment t (2 s grid, `points.at_risk`), predict y = 1 when Brady's hand
goes up in (t + 2 s, t + 15 s] (`y_H15_g2`). Train on the train split, choose each model's
one setting on validation, and keep test untouched until the very end.

Models, simplest first:

| id | model | inputs |
|---|---|---|
| B0 | the base rate (a constant) | none |
| R | the professor-question rule (fixed formula, spec §5.5) | q_open, q_age_s, q_inv_age_s |
| QT | question-timing logistic regression (learned) | the question inputs only |
| NB | multinomial Naive Bayes | the words of the last 60 s |
| TFIDF | TF-IDF + logistic regression | the words of the last 60 s |

Evaluation: average precision (AP), AP within class (AP_w), ROC AUC; and the **timeline**:
the model fires when its probability reaches a threshold (at most once per 15 s); a raise is
*hit* when the model fired in [onset - 15 s, onset - 2 s] and the gap is the *lead time*; a
firing with no raise in (t + 2 s, t + 15 s] is a *false alarm*. The threshold is fixed on the
TRAIN predictions at a false-alarm budget (default 6 per hour of counting time), then the
validation predictions are scored with it.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer

from . import config, metrics, models

LABEL = "y_H15_g2"
H_S, G_S = config.H_S, config.G_S          # 15 s horizon, 2 s guard
REFRACTORY_S = 15.0                        # at most one firing per 15 s
FA_BUDGETS = (6.0, 30.0)                   # thresholds fixed on train: 6 / 30 false alarms per hour
FA_PER_HOUR = FA_BUDGETS[0]
Q_COLS = ("q_open", "q_age_s", "q_inv_age_s", "q_kind")
TEXT_COLS = ("class_id", "txt_all60")


# --------------------------------------------------------------------------- data


def load(tables, split_path) -> pd.DataFrame:
    """Counting moments with the label, the split, the question inputs and the 60 s words."""
    pts = pd.read_parquet(tables / "points.parquet", columns=["class_id", "t_s", "at_risk", LABEL])
    pts = pts[pts["at_risk"].astype(bool) & pts[LABEL].notna()]
    feat = pd.read_parquet(tables / "features.parquet", columns=["class_id", "t_s", *Q_COLS])
    win = pd.read_parquet(tables / "windows.parquet", columns=["class_id", "t_s", "txt_all60"])
    split = pd.read_csv(split_path, usecols=["class_id", "split"])
    df = (pts.merge(feat, on=["class_id", "t_s"], how="left")
             .merge(win, on=["class_id", "t_s"], how="left")
             .merge(split, on="class_id", how="left"))
    df["y"] = df[LABEL].astype(int)
    df["txt_all60"] = df["txt_all60"].fillna("")
    return df.sort_values(["class_id", "t_s"]).reset_index(drop=True)


def onsets(tables, class_ids) -> pd.DataFrame:
    """Brady's raises (bids with a visible start), re-timed onsets."""
    e = pd.read_parquet(tables / "hand_episodes.parquet")
    b = e[(e["ep_type"] == config.EP_BID) & ~e["left_censored"].astype(bool)
          & e["class_id"].isin(class_ids)]
    t = b["onset_retimed_s"].where(b["onset_retimed_s"].notna(), b["onset_s"])
    return pd.DataFrame({"class_id": b["class_id"].to_numpy(), "onset_s": t.to_numpy(float)})


# --------------------------------------------------------------------------- models


def _question_inputs(X: pd.DataFrame) -> pd.DataFrame:
    cap = config.Q_AGE_CAP_S
    out = pd.DataFrame({
        "q_open": pd.to_numeric(X["q_open"], errors="coerce").fillna(0.0),
        "q_age": np.clip(pd.to_numeric(X["q_age_s"], errors="coerce").fillna(cap), 0, cap),
        "q_inv_age": np.clip(pd.to_numeric(X["q_inv_age_s"], errors="coerce").fillna(cap), 0, cap),
        "q_kind": X["q_kind"].astype("object").fillna("none").astype(str),
    }, index=X.index)
    return out


class QuestionTiming(Pipeline):
    """P(raise in the next 2-15 s | seconds since the professor's question, is it still open,
    seconds since an open invitation, kind of question): a logistic regression with smooth
    (spline) curves of the two times, so the shape of "how soon after a question" is learned."""

    def __init__(self, C: float = 1.0):
        self.C = C
        ct = ColumnTransformer([
            ("age", SplineTransformer(n_knots=8, degree=3), ["q_age"]),
            ("inv", SplineTransformer(n_knots=6, degree=3), ["q_inv_age"]),
            ("open", "passthrough", ["q_open"]),
            ("kind", OneHotEncoder(handle_unknown="ignore"), ["q_kind"])])
        super().__init__([("ct", ct), ("lr", LogisticRegression(C=C, max_iter=2000))])

    def fit(self, X, y=None, **kw):
        return super().fit(_question_inputs(X), y, **kw)

    def predict_proba(self, X, **kw):
        return super().predict_proba(_question_inputs(X), **kw)


def make(model_id: str, setting: float | None = None):
    """Unfitted model; `setting` = alpha (NB) or C (QT, TFIDF)."""
    if model_id == "B0":
        return models.PriorProba()
    if model_id == "R":
        return models.RuleR()
    if model_id == "QT":
        return QuestionTiming(C=setting or 1.0)
    if model_id == "NB":
        ct = ColumnTransformer([("txt", models.MultiTextVectorizer("count"), list(TEXT_COLS))])
        return Pipeline([("ct", ct), ("nb", MultinomialNB(alpha=setting or 1.0))])
    if model_id == "TFIDF":
        ct = ColumnTransformer([("txt", models.MultiTextVectorizer("tfidf"), list(TEXT_COLS))])
        return Pipeline([("ct", ct), ("lr", LogisticRegression(C=setting or 1.0, max_iter=2000))])
    raise KeyError(model_id)


SETTINGS = {"QT": (0.1, 1.0, 10.0), "NB": (0.1, 0.3, 1.0), "TFIDF": (0.03, 0.1, 0.3, 1.0)}
NAMES = {"B0": "Base rate", "R": "Professor-question rule", "QT": "Question timing (LR)",
         "NB": "Naive Bayes (words)", "TFIDF": "TF-IDF + LR (words)"}


def proba(model, X) -> np.ndarray:
    return np.asarray(model.predict_proba(X))[:, 1]


# --------------------------------------------------------------------------- metrics


def point_metrics(df: pd.DataFrame, p: np.ndarray) -> dict:
    y = df["y"].to_numpy()
    return {"AP": float(average_precision_score(y, p)),
            "AP_w": float(metrics.ap_within(y, p, df["class_id"].to_numpy())),
            "ROC_AUC": float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan"),
            "base_rate": float(y.mean())}


def firings(df: pd.DataFrame, p: np.ndarray, tau: float) -> pd.DataFrame:
    """Firing moments: p >= tau, at most one per REFRACTORY_S within a class."""
    out = []
    for cid, idx in df.groupby("class_id", sort=False).indices.items():
        t = df["t_s"].to_numpy(float)[idx]
        hit = np.flatnonzero(p[idx] >= tau)
        last = -np.inf
        for k in hit:
            if t[k] - last >= REFRACTORY_S:
                out.append((cid, t[k]))
                last = t[k]
    return pd.DataFrame(out, columns=["class_id", "t_s"])


def timeline(df: pd.DataFrame, p: np.ndarray, tau: float, raises: pd.DataFrame) -> dict:
    f = firings(df, p, tau)
    hours = len(df) * 2.0 / 3600.0
    n_fa, hits, leads = 0, 0, []
    raises = raises[raises["class_id"].isin(df["class_id"].unique())]
    for cid, g in f.groupby("class_id"):
        on = np.sort(raises.loc[raises["class_id"] == cid, "onset_s"].to_numpy())
        for t in g["t_s"].to_numpy():
            n_fa += not np.any((on > t + G_S) & (on <= t + H_S))
    for cid, g in raises.groupby("class_id"):
        ft = np.sort(f.loc[f["class_id"] == cid, "t_s"].to_numpy())
        for o in g["onset_s"].to_numpy():
            w = ft[(ft >= o - H_S) & (ft < o - G_S)]
            if len(w):
                hits += 1
                leads.append(o - w.min())
    n = len(raises)
    leads_a = np.asarray(leads)
    return {"firings": int(len(f)), "false_alarms_per_hour": n_fa / hours if hours else np.nan,
            "raises": int(n), "hit_rate": hits / n if n else np.nan,
            "warned_ge10s": float((leads_a >= 10).sum() / n) if n else np.nan,
            "warned_ge5s": float((leads_a >= 5).sum() / n) if n else np.nan,
            "precision": (len(f) - n_fa) / len(f) if len(f) else np.nan,
            "median_lead_s": float(np.median(leads)) if leads else np.nan}


def threshold_for_budget(df: pd.DataFrame, p: np.ndarray, raises: pd.DataFrame,
                         fa_per_hour: float = FA_PER_HOUR) -> float:
    """The lowest threshold whose false-alarm rate on `df` stays within the budget."""
    cands = np.unique(np.quantile(p, np.linspace(0.5, 0.9999, 60)))
    best = float(cands[-1])
    for tau in cands[::-1]:
        if timeline(df, p, tau, raises)["false_alarms_per_hour"] <= fa_per_hour:
            best = float(tau)
        else:
            break
    return best


# --------------------------------------------------------------------------- run


def run(df: pd.DataFrame, raises: pd.DataFrame, model_ids=("B0", "R", "QT", "NB", "TFIDF"),
        log=print) -> tuple[pd.DataFrame, dict]:
    """Fit each model on train (choosing its setting on validation), score validation.
    Returns the results table and {model_id: (model, p_train, p_val, tau)}."""
    tr, va = df[df["split"] == "train"], df[df["split"] == "validation"]
    rows, fitted = [], {}
    for mid in model_ids:
        t0 = time.time()
        best = None
        for s in SETTINGS.get(mid, (None,)):
            m = make(mid, s)
            m.fit(tr, tr["y"])
            pv = proba(m, va)
            ap = average_precision_score(va["y"], pv)
            if best is None or ap > best[0]:
                best = (ap, s, m, pv)
        _, s, m, pv = best
        pt = proba(m, tr)
        row = {"model": NAMES[mid], "id": mid, "setting": s, **point_metrics(va, pv)}
        taus = {}
        for b in FA_BUDGETS:
            k = f"{b:g}/h"
            keys = ("hit_rate", "warned_ge5s", "warned_ge10s", "precision", "lead_s", "FA_per_h")
            if mid == "B0":                  # a constant fires everywhere or nowhere
                row.update({f"{x}@{k}": np.nan for x in keys})
                continue
            tau = threshold_for_budget(tr, pt, raises, b)
            tl = timeline(va, pv, tau, raises)
            taus[b] = tau
            row.update({f"hit_rate@{k}": tl["hit_rate"], f"warned_ge5s@{k}": tl["warned_ge5s"],
                        f"warned_ge10s@{k}": tl["warned_ge10s"], f"precision@{k}": tl["precision"],
                        f"lead_s@{k}": tl["median_lead_s"], f"FA_per_h@{k}": tl["false_alarms_per_hour"]})
        row["raises"] = int(raises["class_id"].isin(va["class_id"].unique()).sum())
        row["fit_s"] = time.time() - t0
        rows.append(row)
        fitted[mid] = (m, pt, pv, taus)
        log(f"{mid}: AP {row['AP']:.4f} ({row['fit_s']:.0f} s)")
    return pd.DataFrame(rows), fitted


def alarm_curve(df: pd.DataFrame, p: np.ndarray, raises: pd.DataFrame, n: int = 25) -> pd.DataFrame:
    """Hit rate against false alarms per hour as the threshold sweeps (one model, one split)."""
    rows = []
    for tau in np.unique(np.quantile(p, np.linspace(0.80, 0.9995, n))):
        tl = timeline(df, p, tau, raises)
        rows.append({"threshold": tau, "FA_per_h": tl["false_alarms_per_hour"],
                     "hit_rate": tl["hit_rate"], "lead_s": tl["median_lead_s"]})
    return pd.DataFrame(rows)


def firing_log(df: pd.DataFrame, p: np.ndarray, tau: float, raises: pd.DataFrame,
               class_id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One class: every firing with what followed (a raise 2-15 s later = hit, with the lead
    time; otherwise a false alarm), and every raise with whether it was warned."""
    m = (df["class_id"] == class_id).to_numpy()
    f = firings(df[m], p[m], tau)
    on = np.sort(raises.loc[raises["class_id"] == class_id, "onset_s"].to_numpy())
    rows = []
    for t in f["t_s"].to_numpy():
        nxt = on[(on > t + G_S) & (on <= t + H_S)]
        rows.append({"fired_at_min": round(t / 60, 2),
                     "outcome": "hit" if len(nxt) else "false alarm",
                     "raise_at_min": round(nxt.min() / 60, 2) if len(nxt) else np.nan,
                     "lead_s": round(nxt.min() - t, 1) if len(nxt) else np.nan})
    ft = f["t_s"].to_numpy()
    warned = [{"raise_at_min": round(o / 60, 2),
               "warned": bool(np.any((ft >= o - H_S) & (ft < o - G_S))),
               "lead_s": round(o - ft[(ft >= o - H_S) & (ft < o - G_S)].min(), 1)
               if np.any((ft >= o - H_S) & (ft < o - G_S)) else np.nan} for o in on]
    return pd.DataFrame(rows), pd.DataFrame(warned)


def lead_breakdown(df: pd.DataFrame, p: np.ndarray, tau: float, raises: pd.DataFrame) -> dict:
    """Over every class in `df`: firings and how many were right, and every raise by lead bucket
    (warned 10-15 s, 5-10 s or 2-5 s ahead, or missed), plus the per-class warned share."""
    f = firings(df, p, tau)
    raises = raises[raises["class_id"].isin(df["class_id"].unique())]
    right = 0
    for cid, g in f.groupby("class_id"):
        on = raises.loc[raises["class_id"] == cid, "onset_s"].to_numpy()
        right += sum(bool(np.any((on > t + G_S) & (on <= t + H_S))) for t in g["t_s"].to_numpy())
    buckets = {"10-15 s": 0, "5-10 s": 0, "2-5 s": 0, "missed": 0}
    per_class = []
    for cid, g in raises.groupby("class_id"):
        ft = f.loc[f["class_id"] == cid, "t_s"].to_numpy()
        warned = 0
        for o in g["onset_s"].to_numpy():
            w = ft[(ft >= o - H_S) & (ft < o - G_S)]
            if not len(w):
                buckets["missed"] += 1
                continue
            lead = o - w.min()
            warned += 1
            buckets["10-15 s" if lead >= 10 else "5-10 s" if lead >= 5 else "2-5 s"] += 1
        per_class.append(warned / len(g))
    return {"classes": int(df["class_id"].nunique()), "raises": int(len(raises)),
            "firings": int(len(f)), "firings_right": int(right),
            **{f"warned {k} ahead" if k != "missed" else "missed": v for k, v in buckets.items()},
            "median warned share per class": float(np.median(per_class)) if per_class else np.nan}


def score_split(fitted: dict, df: pd.DataFrame, raises: pd.DataFrame) -> pd.DataFrame:
    """The frozen models (settings and thresholds from train/validation) on another split."""
    rows = []
    for mid, (m, pt, pv, taus) in fitted.items():
        p = proba(m, df)
        row = {"model": NAMES[mid], "id": mid, **point_metrics(df, p)}
        for b in FA_BUDGETS:
            k = f"{b:g}/h"
            if mid == "B0":
                continue
            tl = timeline(df, p, taus[b], raises)
            row.update({f"hit_rate@{k}": tl["hit_rate"], f"warned_ge5s@{k}": tl["warned_ge5s"],
                        f"warned_ge10s@{k}": tl["warned_ge10s"], f"precision@{k}": tl["precision"],
                        f"lead_s@{k}": tl["median_lead_s"], f"FA_per_h@{k}": tl["false_alarms_per_hour"]})
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- guesses (18 Oct)
# Brady: no cooldowns or firing rules ("the model is supposed to work that out itself"), and
# score only the guesses. Every counting moment whose probability is at or above the
# threshold is a guess "my hand goes up in (t + 2 s, t + 15 s]": right if it did, wrong if
# not. The threshold is fixed on train so the model says yes as often as the truth is yes
# there (as many guesses as positive moments). For a flexible model the train scores must be
# out-of-fold, or the threshold is set on overfit scores.


def tau_like_truth(y_train, p_train) -> float:
    """The threshold at which the train scores give as many guesses as there are positives."""
    y, p = np.asarray(y_train), np.asarray(p_train, float)
    n_pos = int(y.sum())
    return float(np.sort(p)[::-1][min(max(n_pos, 1), len(p)) - 1])


WARN_WINDOWS = ((2, 5), (5, 10), (10, 15))        # seconds before the raise; 2-5 s is the ideal


def raises_warned(df: pd.DataFrame, said: np.ndarray, raises: pd.DataFrame) -> dict:
    """Per raise in df's classes: warned (a guess in [onset - 15, onset - 2) s), a guess in
    each WARN_WINDOWS window, and whether a perfect model could warn it at all (a counting
    moment in that 2-15 s window; a perfect model also has one in each sub-window it covers)."""
    t_all = df["t_s"].to_numpy(float)
    out = {"warned": [], "warnable": [], **{f"in_{a}_{b}": [] for a, b in WARN_WINDOWS},
           **{f"can_{a}_{b}": [] for a, b in WARN_WINDOWS}}
    for cid, idx in df.groupby("class_id", sort=False).indices.items():
        tc = t_all[idx]
        ft = tc[said[idx]]
        for o in raises.loc[raises["class_id"] == cid, "onset_s"].to_numpy():
            out["warned"].append(bool(np.any((ft >= o - H_S) & (ft < o - G_S))))
            out["warnable"].append(bool(np.any((tc >= o - H_S) & (tc < o - G_S))))
            for a, b in WARN_WINDOWS:
                out[f"in_{a}_{b}"].append(bool(np.any((ft >= o - b) & (ft < o - a))))     # lead in (a, b]
                out[f"can_{a}_{b}"].append(bool(np.any((tc >= o - b) & (tc < o - a))))
    return {k: np.asarray(v, bool) for k, v in out.items()}


def warning_runs(df: pd.DataFrame, said: np.ndarray) -> tuple[int, int]:
    """(warnings, right). A warning is an unbroken run of yes moments (consecutive counting
    moments, 2 s apart): back-to-back yeses are one warning, a gap starts a new one. It is
    right if any of its moments is positive (my hand went up 2-15 s after it). Counting only:
    the model still says yes at every moment it likes."""
    t_all, y_all = df["t_s"].to_numpy(float), df["y"].to_numpy().astype(bool)
    n = right = 0
    for _, idx in df.groupby("class_id", sort=False).indices.items():
        s_, t, y = said[idx], t_all[idx], y_all[idx]
        if not s_.any():
            continue
        cont = np.r_[False, s_[:-1] & (np.diff(t) <= 2.0 + 1e-6)]
        run = np.cumsum(s_ & ~cont)
        ids = run[s_]
        n += int(ids.max())
        right += int(len(np.unique(ids[y[s_]])))
    return n, right


def guess_scores(df: pd.DataFrame, p: np.ndarray, tau: float, raises: pd.DataFrame) -> dict:
    """Guesses (p >= tau), right (a raise 2-15 s later), wrong; raises warned 2-15 s ahead and
    in each WARN_WINDOWS window; and the same counts for a perfect model (guesses exactly at
    the positive moments)."""
    y, said = df["y"].to_numpy().astype(bool), np.asarray(p) >= tau
    right, guesses = int((said & y).sum()), int(said.sum())
    r = raises_warned(df, said, raises)
    out = {"guesses": guesses, "right": right, "wrong": guesses - right,
           "pct_right": right / guesses if guesses else np.nan,
           "guesses_per_class": guesses / max(1, df["class_id"].nunique()),
           "raises": int(len(r["warned"])), "raises_warned": int(r["warned"].sum()),
           "perfect_warned": int(r["warnable"].sum()), "positives": int(y.sum())}
    n_w, right_w = warning_runs(df, said)
    out.update({"warnings": n_w, "warnings_right": right_w, "warnings_wrong": n_w - right_w,
                "pct_warnings_right": right_w / n_w if n_w else np.nan,
                "warnings_per_class": n_w / max(1, df["class_id"].nunique())})
    for a, b in WARN_WINDOWS:
        out[f"warned_{a}_{b}"] = int(r[f"in_{a}_{b}"].sum())
        out[f"perfect_{a}_{b}"] = int(r[f"can_{a}_{b}"].sum())
    return out


def raise_rate_per_hour(df: pd.DataFrame, raises: pd.DataFrame) -> float:
    r = raises[raises["class_id"].isin(df["class_id"].unique())]
    return len(r) / (len(df) * 2.0 / 3600.0)


def peri_raise(df: pd.DataFrame, p: np.ndarray, raises: pd.DataFrame,
               offsets=np.arange(-60, 1, 2)) -> pd.DataFrame:
    """Mean model probability at fixed offsets before each raise (counting moments only)."""
    out = {o: [] for o in offsets}
    t_all = df["t_s"].to_numpy(float)
    for cid, idx in df.groupby("class_id", sort=False).indices.items():
        t, q = t_all[idx], p[idx]
        for o_s in raises.loc[raises["class_id"] == cid, "onset_s"].to_numpy():
            for off in offsets:
                k = np.searchsorted(t, o_s + off - 1.0)
                if k < len(t) and abs(t[k] - (o_s + off)) <= 1.0:
                    out[off].append(q[k])
    return pd.DataFrame({"offset_s": offsets, "mean_p": [np.mean(out[o]) if out[o] else np.nan for o in offsets],
                         "n": [len(out[o]) for o in offsets]})
