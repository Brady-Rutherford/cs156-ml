"""Point- and event-level metrics, timing decomposition and inference (spec §8.2-§8.5, A9).

Point level (§8.2): `ap` (step-wise, identical to sklearn's average_precision_score), lift,
`ap_within` (AP_w, the primary statistic: per-class AP weighted by the class's bid count K_c),
`macro_ap`, `ece` (equal-mass bins, ties never split), `brier`, `brier_skill` (A9), `roc_auc`.

Event level (§8.3, seizure-prediction convention). `EventData` holds the per-class decision
grids once; everything else is computed from it:
* the causal grid G (`in_G`) is where firings may happen; label-side exclusions never remove
  points from G, so a guard point can fire and counts as neutral;
* firings: `p(s) >= tau` and no earlier firing in (s - (H - g), s), built in time order, per
  class (`firings` never reads labels). A firing claims the occurrence period (s + g, s + H];
* FAR = false firings per minute of G_def (defined labels); the firing rate counts all firings
  per minute of G; recall counts *evaluable* bids (>= 50% of the grid in [o - H, o - g) in G)
  with a firing in [o - H, o - g); lead = o - earliest such firing;
* time-in-warning is time-weighted by default: each grid point owns [t, t + delta), and TiW is
  the share of G-time inside the warning intervals. That makes TiW equal R0 = rate (H - g) / 60
  for any process independent of the onsets (always-on: 13/14), which the spec's own checks
  need; `mode="points"` gives the literal point count (always-on: 6/7);
* `froc` (lower envelope over tau), `recall_at_far`, `pafroc`, `r0`, `tau_rate_matched`
  (inner-OOF rate matching), `random_alarm` (matched-rate random reference, A9) and
  `eventlift_shift` (EventLift against shifted predictions).

Timing decomposition (§8.4): `shift_global`, `shift_local`, `TimingGain` / `timing_gain`.

Inference (§8.5): every statistic that has to be bootstrapped is a callable of a class-weight
vector (bootstrap multiplicities), so `cluster_bootstrap` and `paired_delta` resample whole
classes without re-sorting points (`PooledAP`, `WithinAP`, `TimingGain`, `EventStats.summary`,
`ClassResampled` for anything else). Plus `mde`, `holm`, `fixed_sequence_holm`, `iut_p`.
"""

from __future__ import annotations

import bisect
import itertools
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np
import pandas as pd

from . import config

_TOL = 1e-9

# --------------------------------------------------------------------------- helpers


def _f(a) -> np.ndarray:
    return np.asarray(a, dtype=float)


def _defined(y, p) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(y, p, mask) restricted to defined labels; NaN scores on defined points are an error."""
    y, p = _f(y), _f(p)
    if y.shape != p.shape:
        raise ValueError(f"y and p have different shapes {y.shape} vs {p.shape}")
    m = ~np.isnan(y)
    if np.isnan(p[m]).any():
        raise ValueError("NaN scores on points with a defined label")
    return y[m], p[m], m


def _class_index(classes: np.ndarray, groups) -> np.ndarray:
    g = np.asarray(groups)
    if classes.size == 0:
        if g.size:
            raise ValueError("groups not found in classes")
        return np.zeros(0, dtype=np.int64)
    idx = np.searchsorted(classes, g)
    idx = np.clip(idx, 0, len(classes) - 1)
    if g.size and not np.array_equal(classes[idx], g):
        raise ValueError("groups contain classes missing from `classes`")
    return idx.astype(np.int64)


def _classes(groups, classes=None) -> np.ndarray:
    return np.unique(np.asarray(groups)) if classes is None else np.sort(np.asarray(classes))


def _weights(w, n: int) -> np.ndarray:
    if w is None:
        return np.ones(n)
    w = _f(w)
    if w.shape != (n,):
        raise ValueError(f"class weights must have shape ({n},), got {w.shape}")
    return w


def _wmedian(x: np.ndarray, w: np.ndarray) -> float:
    """Median of x with integer weights (== np.median of the materialised resample)."""
    keep = w > 0
    if not keep.any():
        return float("nan")
    x, w = x[keep], w[keep]
    if np.allclose(w, np.round(w)):
        return float(np.median(np.repeat(x, np.round(w).astype(int))))
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    return float(x[o][np.searchsorted(cw, cw[-1] / 2.0)])


# --------------------------------------------------------------------------- point metrics


def _presort(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Descending order and the last index of each block of tied scores."""
    order = np.argsort(-p, kind="mergesort")
    ps = p[order]
    last = np.flatnonzero(np.r_[ps[1:] != ps[:-1], True]) if ps.size else np.zeros(0, int)
    return order, last


def _ap_sorted(ys: np.ndarray, w: np.ndarray, last: np.ndarray) -> float:
    """AP = sum_n (R_n - R_{n-1}) P_n over distinct thresholds (sklearn's definition)."""
    if ys.size == 0:
        return float("nan")
    tp = np.cumsum(w * ys)[last]
    fp = np.cumsum(w * (1.0 - ys))[last]
    pos = tp[-1]
    if not pos > 0:
        return float("nan")
    denom = tp + fp
    prec = np.divide(tp, denom, out=np.zeros_like(tp), where=denom > 0)
    rec = tp / pos
    return float(np.sum(np.diff(rec, prepend=0.0) * prec))


def _ap_from_counts(tp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """AP from (weighted) cumulative TP / FP at the score blocks that hold a positive (last
    axis); blocks without a positive add nothing to sum (R_n - R_n-1) P_n, so this is exact."""
    tp, fp = np.atleast_2d(tp), np.atleast_2d(fp)
    if tp.shape[1] == 0:
        return np.full(len(tp), np.nan)
    pos = tp[:, -1:]
    denom = tp + fp
    prec = np.divide(tp, denom, out=np.zeros_like(tp), where=denom > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.sum(np.diff(tp, prepend=0.0, axis=1) / pos * prec, axis=1)
    return np.where(pos[:, 0] > 0, out, np.nan)


def ap(y, p, sample_weight=None) -> float:
    """Average precision over defined labels (NaN y dropped); NaN if there is no positive."""
    y, p, m = _defined(y, p)
    w = np.ones(y.size) if sample_weight is None else _f(sample_weight)[m]
    order, last = _presort(p)
    return _ap_sorted(y[order], w[order], last)


def base_rate(y) -> float:
    y = _f(y)
    y = y[~np.isnan(y)]
    return float(y.mean()) if y.size else float("nan")


def lift(y, p) -> float:
    """AP / pi."""
    return ap(y, p) / base_rate(y)


class PooledAP:
    """Pooled AP as a function of class weights (bootstrap multiplicities).

    Scores are sorted once. A resample that draws class c twice is the same as weight 2 on
    its points, and AP only moves at score blocks that hold a positive, so per-class
    cumulative TP / FP counts at those blocks (C x J) turn every draw into two dot products
    (`batch` does all draws at once)."""

    def __init__(self, y, p, groups, *, classes=None):
        y, p, m = _defined(y, p)
        g = np.asarray(groups)[m]
        self.classes = _classes(np.asarray(groups), classes)
        cls = _class_index(self.classes, g)
        order, last = _presort(p)
        ys, cs = y[order], cls[order]
        block = np.searchsorted(last, np.arange(ys.size))       # score block of each point
        ends = np.unique(last[block[ys == 1]]) if ys.size else np.zeros(0, int)
        C = len(self.classes)
        self._tp, self._fp = np.zeros((C, ends.size)), np.zeros((C, ends.size))
        if ys.size:
            by = np.lexsort((np.arange(ys.size), cs))           # positions grouped by class
            bounds = np.searchsorted(cs[by], np.arange(C + 1))
            for c in range(C):
                pos = by[bounds[c]:bounds[c + 1]]
                if not pos.size:
                    continue
                yc = ys[pos] == 1
                self._tp[c] = np.searchsorted(pos[yc], ends, side="right")
                self._fp[c] = np.searchsorted(pos[~yc], ends, side="right")

    def __call__(self, w=None) -> float:
        wc = _weights(w, len(self.classes))
        return float(_ap_from_counts(wc @ self._tp, wc @ self._fp)[0])

    def batch(self, W: np.ndarray) -> np.ndarray:
        """The statistic for every row of a (B x C) class-weight matrix. Integer weights and
        counts make float32 products exact (sums < 2**24), and twice as fast."""
        W = np.asarray(W)
        if np.issubdtype(W.dtype, np.integer) and (W.sum(axis=1).max(initial=0)
                                                   * max(self._tp.max(initial=0),
                                                         self._fp.max(initial=0)) < 2 ** 24):
            Wf = W.astype(np.float32)
            return _ap_from_counts((Wf @ self._tp.astype(np.float32)).astype(float),
                                   (Wf @ self._fp.astype(np.float32)).astype(float))
        W = W.astype(float)
        return _ap_from_counts(W @ self._tp, W @ self._fp)


def per_class_ap(y, p, groups) -> pd.Series:
    """AP of each class (NaN for classes without a positive), indexed by class id."""
    y, p, m = _defined(y, p)
    g = np.asarray(groups)[m]
    classes = np.unique(np.asarray(groups))
    cls = _class_index(classes, g)
    out = np.full(len(classes), np.nan)
    order = np.argsort(cls, kind="stable")
    bounds = np.searchsorted(cls[order], np.arange(len(classes) + 1))
    for c in range(len(classes)):
        idx = order[bounds[c]:bounds[c + 1]]
        if idx.size and y[idx].sum() > 0:
            out[c] = ap(y[idx], p[idx])
    return pd.Series(out, index=pd.Index(classes, name="class_id"), name="ap")


def bid_counts(y, groups, onset) -> pd.Series:
    """K_c: distinct bid onsets (`next_bid_s`) among the positive points of each class."""
    df = pd.DataFrame({"y": _f(y), "g": np.asarray(groups), "o": _f(onset)})
    pos = df[df["y"] == 1]
    k = pos.groupby("g")["o"].nunique()
    return k.reindex(np.unique(df["g"]), fill_value=0).rename("K")


class WithinAP:
    """AP_w = sum_c K_c AP_c / sum_c K_c over classes with >= 1 positive (§8.2).

    Each class is ranked only against itself, so fold score scales never mix. `weights` is
    K_c: a mapping/Series class -> bid count (see `bid_counts`), "equal" (macro-AP), or None
    (number of positive points, a proxy for K_c). Callable with bootstrap class weights.
    """

    def __init__(self, y, p, groups, *, weights=None, classes=None):
        self.classes = _classes(np.asarray(groups), classes)
        ap_c = per_class_ap(y, p, groups).reindex(self.classes)
        if weights is None:
            yy = _f(y)
            pos = pd.Series(yy == 1).groupby(np.asarray(groups)).sum()
            k = pos.reindex(self.classes, fill_value=0).to_numpy(float)
        elif isinstance(weights, str):
            if weights != "equal":
                raise ValueError(f"unknown weights {weights!r}")
            k = np.ones(len(self.classes))
        else:
            k = pd.Series(weights).reindex(self.classes).to_numpy(float)
            if np.isnan(k[ap_c.notna().to_numpy()]).any():
                raise ValueError("weights missing for classes with positives")
        self.ap_c = ap_c.to_numpy(float)
        self.k = np.nan_to_num(k)
        self._valid = ~np.isnan(self.ap_c) & (self.k > 0)

    def __call__(self, w=None) -> float:
        wc = _weights(w, len(self.classes))
        kw = (wc * self.k)[self._valid]
        den = kw.sum()
        return float(np.dot(kw, self.ap_c[self._valid]) / den) if den > 0 else float("nan")

    def batch(self, W: np.ndarray) -> np.ndarray:
        KW = (np.asarray(W, float) * self.k)[:, self._valid]
        den = KW.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, KW @ self.ap_c[self._valid] / den, np.nan)


def ap_within(y, p, groups, weights=None) -> float:
    """AP_w, the primary statistic (see `WithinAP`)."""
    return WithinAP(y, p, groups, weights=weights)()


def macro_ap(y, p, groups) -> float:
    """Unweighted mean of per-class AP over classes with >= 1 positive."""
    return WithinAP(y, p, groups, weights="equal")()


def fold_ap(y, p, folds) -> tuple[float, float, pd.Series]:
    """Mean and sd (ddof=1) of per-fold AP, plus the per-fold Series."""
    yy, pp, m = _defined(y, p)
    f = np.asarray(folds)[m]
    per = pd.Series({k: ap(yy[f == k], pp[f == k]) for k in np.unique(f)}, name="ap")
    sd = float(per.std(ddof=1)) if per.notna().sum() > 1 else float("nan")
    return float(per.mean()), sd, per


def reliability(y, p, n_bins: int = config.ECE_BINS) -> pd.DataFrame:
    """Equal-mass bins on p (ties stay together): bin, n, p_mean, y_mean."""
    y, p, _ = _defined(y, p)
    if y.size == 0:
        return pd.DataFrame(columns=["bin", "n", "p_mean", "y_mean"])
    order = np.argsort(p, kind="mergesort")
    ps, ys = p[order], y[order]
    cuts = ps[(np.arange(1, n_bins) * ps.size) // n_bins]
    b = np.searchsorted(cuts, ps, side="right")
    n = np.bincount(b, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        pm = np.bincount(b, weights=ps, minlength=n_bins) / n
        ym = np.bincount(b, weights=ys, minlength=n_bins) / n
    df = pd.DataFrame({"bin": np.arange(n_bins), "n": n, "p_mean": pm, "y_mean": ym})
    return df[df["n"] > 0].reset_index(drop=True)


def ece(y, p, n_bins: int = config.ECE_BINS) -> float:
    """ECE = sum_b (n_b / N) |ybar_b - pbar_b| over equal-mass bins (use p_cal)."""
    r = reliability(y, p, n_bins)
    if r.empty:
        return float("nan")
    return float(np.sum(r["n"] / r["n"].sum() * np.abs(r["y_mean"] - r["p_mean"])))


def brier(y, p) -> float:
    y, p, _ = _defined(y, p)
    return float(np.mean((p - y) ** 2)) if y.size else float("nan")


def brier_skill(y, p, ref=None) -> float:
    """Brier skill score 1 - Brier / Brier_ref (A9).

    `ref`: None = the evaluated sample's own base rate (climatology, pi (1 - pi)); a float =
    a constant reference such as the training-fold prevalence B0; an array = reference scores.
    """
    y, p, m = _defined(y, p)
    if y.size == 0:
        return float("nan")
    if ref is None:
        b_ref = float(np.mean((y.mean() - y) ** 2))
    elif np.ndim(ref) == 0:
        b_ref = float(np.mean((float(ref) - y) ** 2))
    else:
        b_ref = float(np.mean((_f(ref)[m] - y) ** 2))
    return 1.0 - float(np.mean((p - y) ** 2)) / b_ref if b_ref > 0 else float("nan")


def roc_auc(y, p) -> float:
    """ROC-AUC (secondary; it flatters at pi ~ 2%). NaN when only one class is present."""
    from sklearn.metrics import roc_auc_score

    y, p, _ = _defined(y, p)
    if np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def point_metrics(y, p, groups, *, weights=None, ref=None,
                  n_bins: int = config.ECE_BINS) -> dict:
    """Point-level columns of the `results` table for one set of OOF predictions."""
    _, _, m = _defined(y, p)
    a = ap(y, p)
    pi = base_rate(y)
    return {
        "n_points": int(m.sum()), "n_classes": int(np.unique(np.asarray(groups)[m]).size),
        "base_rate": pi, "ap": a, "ap_lift": a / pi if pi > 0 else float("nan"),
        "ap_within": ap_within(y, p, groups, weights), "macro_ap": macro_ap(y, p, groups),
        "roc_auc": roc_auc(y, p), "brier": brier(y, p), "brier_skill": brier_skill(y, p, ref),
        "ece": ece(y, p, n_bins),
    }


# --------------------------------------------------------------------------- event metrics


def r0(rate, H: float = config.H_S, g: float = config.G_S):
    """Analytic chance recall R0 = min(1, rate (H - g) / 60) for firings independent of onsets
    (refractory firings have disjoint warnings, so recall = time in warning)."""
    out = np.minimum(1.0, _f(rate) * (float(H) - float(g)) / 60.0)
    return float(out) if np.ndim(out) == 0 else out


def r0_poisson(rate, H: float = config.H_S, g: float = config.G_S):
    """No-refractory (Poisson) approximation 1 - exp(-rate (H - g) / 60)."""
    out = 1.0 - np.exp(-_f(rate) * (float(H) - float(g)) / 60.0)
    return float(out) if np.ndim(out) == 0 else out


def _greedy_refractory(times: np.ndarray, refractory: float) -> np.ndarray:
    """Positions that fire: the earliest candidate, then the next one >= last + refractory."""
    n = times.size
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    if refractory <= 0:
        return np.arange(n, dtype=np.int64)
    lst = times.tolist()
    thr = refractory - _TOL
    out, j = [], 0
    while j < n:
        out.append(j)
        j = bisect.bisect_left(lst, lst[j] + thr, j + 1)
    return np.asarray(out, dtype=np.int64)


def _merge(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Union of intervals (a_i, b_i] as sorted disjoint intervals."""
    if a.size == 0:
        return a, b
    o = np.argsort(a, kind="mergesort")
    a, b = a[o], b[o]
    cmax = np.maximum.accumulate(b)
    start = np.r_[True, a[1:] > cmax[:-1]]
    idx = np.flatnonzero(start)
    return a[idx], np.maximum.reduceat(b, idx)


def _measure_le(a: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Length of (union of disjoint sorted intervals (a, b]) intersected with (-inf, x]."""
    if a.size == 0:
        return np.zeros_like(x)
    cum = np.r_[0.0, np.cumsum(b - a)]
    i = np.searchsorted(a, x, side="left")      # intervals with a < x
    j = np.maximum(i - 1, 0)
    part = np.minimum(x, b[j]) - a[j]
    return np.where(i > 0, cum[j] + part, 0.0)


def _n_grid_in(a: np.ndarray, b: np.ndarray, t0: np.ndarray, delta: float) -> np.ndarray:
    """Number of regular grid points t0 + k delta in [a, b)."""
    k_hi = np.ceil((b - t0) / delta - _TOL)
    k_lo = np.ceil((a - t0) / delta - _TOL)
    return np.maximum(k_hi - k_lo, 0)


@dataclass
class EventStats:
    """Per-class sufficient statistics at one threshold; `summary(w)` aggregates them with
    class weights (bootstrap multiplicities), so event metrics bootstrap by class for free."""

    classes: np.ndarray
    H: float
    g: float
    delta: float
    n_grid: np.ndarray       # |G| per class
    n_def: np.ndarray        # |G_def|
    n_fire: np.ndarray
    n_true: np.ndarray
    n_false: np.ndarray
    n_neutral: np.ndarray
    warn_s: np.ndarray       # seconds of G-time inside warning intervals
    warn_pts: np.ndarray     # G points inside warning intervals
    n_eval: np.ndarray       # evaluable bids
    n_hit: np.ndarray        # evaluable bids with a firing in [o - H, o - g)
    lead_s: np.ndarray = field(default_factory=lambda: np.zeros(0))
    lead_cls: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    has_tiw: bool = True

    def summary(self, w=None) -> dict:
        wc = _weights(w, len(self.classes))

        def tot(a):
            return float(np.dot(wc, a))

        grid_min = tot(self.n_grid) * self.delta / 60.0
        def_min = tot(self.n_def) * self.delta / 60.0
        n_fire, n_eval = tot(self.n_fire), tot(self.n_eval)
        rate = n_fire / grid_min if grid_min > 0 else float("nan")
        far = tot(self.n_false) / def_min if def_min > 0 else float("nan")
        rec = tot(self.n_hit) / n_eval if n_eval > 0 else float("nan")
        chance = r0(rate, self.H, self.g) if not math.isnan(rate) else float("nan")
        nan = float("nan")
        tiw = tot(self.warn_s) / (tot(self.n_grid) * self.delta) if self.has_tiw and \
            grid_min > 0 else nan
        tiw_pts = tot(self.warn_pts) / tot(self.n_grid) if self.has_tiw and grid_min > 0 \
            else nan
        lead_med = _wmedian(self.lead_s, wc[self.lead_cls]) if self.lead_s.size else nan
        return {
            "far": far, "far_per_hour": far * 60.0, "rate": rate, "recall": rec, "tiw": tiw,
            "tiw_points": tiw_pts, "r0": chance,
            "eventlift0": rec / chance if chance and chance > 0 else nan,
            "lead_med_s": lead_med, "n_fire": n_fire, "n_true": tot(self.n_true),
            "n_false": tot(self.n_false), "n_neutral": tot(self.n_neutral),
            "n_eval": n_eval, "n_hit": tot(self.n_hit), "grid_min": grid_min,
            "def_min": def_min,
        }

    def per_class(self) -> pd.DataFrame:
        return pd.DataFrame({
            "class_id": self.classes, "n_grid": self.n_grid, "n_def": self.n_def,
            "n_fire": self.n_fire, "n_true": self.n_true, "n_false": self.n_false,
            "n_neutral": self.n_neutral, "warn_s": self.warn_s, "n_eval": self.n_eval,
            "n_hit": self.n_hit})


class EventData:
    """Decision grids and bid onsets of a set of classes, prepared once for event scoring.

    Points (any order) carry t, class, `in_grid` (the causal grid G, `points.in_G`) and the
    label y for the same (H, g) (1 / 0 / NaN). `onsets` are bid onsets o_k (ep_type == bid,
    not left-censored; see `bid_onsets`) with their classes. Pass every grid point of a class
    (in G or not) or only its G points: evaluability counts grid slots arithmetically.

    Internally points are sorted by (class, t) and laid on one time axis with a gap between
    classes, so the refractory and the windows never cross a class boundary.
    """

    def __init__(self, t, groups=None, *, in_grid=None, y=None, onsets=None,
                 onset_groups=None, H: float = config.H_S, g: float = config.G_S,
                 delta: float = config.DELTA_S,
                 evaluable_min: float = config.EVALUABLE_MIN_FRAC):
        t = _f(t)
        n = t.size
        if np.isnan(t).any():
            raise ValueError("NaN decision times")
        groups = np.zeros(n, dtype=int) if groups is None else np.asarray(groups)
        grid = (np.ones(n, bool) if in_grid is None
                else pd.Series(in_grid, dtype="boolean").fillna(False).to_numpy(dtype=bool))
        yy = np.full(n, np.nan) if y is None else _f(y)
        if not (len(groups) == len(grid) == len(yy) == n):
            raise ValueError("t, groups, in_grid and y must have the same length")
        self.H, self.g, self.delta = float(H), float(g), float(delta)
        self.refractory = config.alarm_refractory(H, g)
        self.n_points = n
        self.classes = np.unique(groups)
        cls = _class_index(self.classes, groups)
        self.order = np.lexsort((t, cls))
        self.cls, self.t = cls[self.order], t[self.order]
        self.in_grid, self.y = grid[self.order], yy[self.order]
        C = len(self.classes)

        # onsets of classes that have points
        o = _f(onsets) if onsets is not None else np.zeros(0)
        if onsets is not None and onset_groups is None:
            if np.unique(groups).size > 1:
                raise ValueError("onset_groups is required with several classes")
            og = np.full(o.size, self.classes[0] if C else 0)
        else:
            og = np.asarray(onset_groups) if onset_groups is not None else np.zeros(0)
        keep = np.isin(og, self.classes) & ~np.isnan(o)
        self.n_onsets_dropped = int((~keep).sum())
        o, oc = o[keep], _class_index(self.classes, og[keep])

        # one time axis: integer offsets, gaps wider than any window
        lo = np.full(C, np.inf)
        hi = np.full(C, -np.inf)
        np.minimum.at(lo, self.cls, self.t)
        np.maximum.at(hi, self.cls, self.t + self.delta)
        np.minimum.at(lo, oc, o - self.H)
        np.maximum.at(hi, oc, o + self.H)
        gap = 4.0 * (self.H + self.delta) + 10.0
        off = np.zeros(C)
        start = 0.0
        for c in range(C):
            off[c] = math.ceil(start - lo[c])
            start = off[c] + hi[c] + gap
        self._off = off
        self.tg = self.t + off[self.cls]
        self.onset_t, self.onset_cls = o, oc
        self.onset_tg = o + off[oc]

        # G points; evaluable onsets (>= evaluable_min of the grid slots in [o - H, o - g))
        self._gi = np.flatnonzero(self.in_grid)
        self._tgG = self.tg[self._gi]
        self._clsG = self.cls[self._gi]
        first_t = np.full(C, np.inf)
        np.minimum.at(first_t, self.cls, self.tg)
        a, b = self.onset_tg - self.H, self.onset_tg - self.g
        n_in = np.searchsorted(self._tgG, b, "left") - np.searchsorted(self._tgG, a, "left")
        slots = _n_grid_in(a, b, first_t[oc], self.delta) if oc.size else np.zeros(0)
        self.onset_eval = (slots > 0) & (n_in >= evaluable_min * slots - _TOL)
        defined = self.in_grid & ~np.isnan(self.y)
        self.n_grid = np.bincount(self._clsG, minlength=C)
        self.n_def = np.bincount(self.cls[defined], minlength=C)

    @classmethod
    def from_frames(cls, points: pd.DataFrame, onsets: pd.DataFrame | None = None, *,
                    y_col: str = config.HEADLINE_LABEL, t_col: str = "t_s",
                    group_col: str = "class_id", grid_col: str = "in_G",
                    onset_col: str = "onset_s", **kw) -> EventData:
        """From a points-like frame and a bid-onset frame (see `bid_onsets`)."""
        return cls(points[t_col].to_numpy(float), points[group_col].to_numpy(),
                   in_grid=points[grid_col].to_numpy() if grid_col in points else None,
                   y=points[y_col].to_numpy(float) if y_col in points else None,
                   onsets=None if onsets is None else onsets[onset_col].to_numpy(float),
                   onset_groups=None if onsets is None else onsets[group_col].to_numpy(),
                   **kw)

    # ----------------------------------------------------------------- firing

    def _sorted(self, a, name: str) -> np.ndarray:
        a = _f(a)
        if a.ndim == 0:
            return np.full(self.n_points, float(a))
        if a.shape != (self.n_points,):
            raise ValueError(f"{name} must have {self.n_points} values, got {a.shape}")
        return a[self.order]

    def _fire_sorted(self, ps: np.ndarray, taus: np.ndarray) -> np.ndarray:
        cand = np.flatnonzero(self.in_grid & (ps >= taus))
        fired = np.zeros(self.n_points, bool)
        fired[cand[_greedy_refractory(self.tg[cand], self.refractory)]] = True
        return fired

    def firings(self, p, tau) -> np.ndarray:
        """Boolean firing mask in the caller's point order. `tau` is a scalar or one value per
        point (e.g. each fold's rate-matched threshold). Labels are never read."""
        fired = self._fire_sorted(self._sorted(p, "p"), self._sorted(tau, "tau"))
        out = np.zeros(self.n_points, bool)
        out[self.order] = fired
        return out

    def fired_sorted(self, fired) -> np.ndarray:
        """A caller-order mask in internal (sorted) order."""
        f = np.asarray(fired, dtype=bool)
        if f.shape != (self.n_points,):
            raise ValueError("fired mask has the wrong length")
        return f[self.order]

    def rate(self, p, tau, w=None) -> float:
        """All firings per minute of G (cheap: no TiW, no recall)."""
        fired = self._fire_sorted(self._sorted(p, "p"), self._sorted(tau, "tau"))
        wc = _weights(w, len(self.classes))
        n_fire = np.bincount(self.cls[fired], minlength=len(self.classes))
        grid_min = float(np.dot(wc, self.n_grid)) * self.delta / 60.0
        return float(np.dot(wc, n_fire)) / grid_min if grid_min > 0 else float("nan")

    # ----------------------------------------------------------------- statistics

    def stats_from_sorted(self, fired: np.ndarray, *, tiw: bool = True) -> EventStats:
        C = len(self.classes)
        fi = np.flatnonzero(fired & self.in_grid)
        ft, fc, fy = self.tg[fi], self.cls[fi], self.y[fi]
        n_fire = np.bincount(fc, minlength=C)
        n_true = np.bincount(fc[fy == 1], minlength=C)
        n_false = np.bincount(fc[fy == 0], minlength=C)
        if tiw:
            a, b = _merge(ft + self.g, ft + self.H)
            cover = (_measure_le(a, b, self._tgG + self.delta) - _measure_le(a, b, self._tgG))
            warn_s = np.bincount(self._clsG, weights=cover, minlength=C)
            lo = np.searchsorted(ft, self._tgG - self.H, "left")
            hi = np.searchsorted(ft, self._tgG - self.g, "left")
            warn_pts = np.bincount(self._clsG, weights=(hi > lo).astype(float), minlength=C)
        else:
            warn_s = warn_pts = np.zeros(C)
        e = self.onset_eval
        ot, oc = self.onset_tg[e], self.onset_cls[e]
        lo = np.searchsorted(ft, ot - self.H, "left")
        hi = np.searchsorted(ft, ot - self.g, "left")
        hit = hi > lo
        return EventStats(
            classes=self.classes, H=self.H, g=self.g, delta=self.delta, n_grid=self.n_grid,
            n_def=self.n_def, n_fire=n_fire, n_true=n_true, n_false=n_false,
            n_neutral=n_fire - n_true - n_false, warn_s=warn_s, warn_pts=warn_pts,
            n_eval=np.bincount(oc, minlength=C), n_hit=np.bincount(oc[hit], minlength=C),
            lead_s=ot[hit] - ft[lo[hit]], lead_cls=oc[hit], has_tiw=tiw)

    def stats(self, p, tau, *, tiw: bool = True) -> EventStats:
        fired = self._fire_sorted(self._sorted(p, "p"), self._sorted(tau, "tau"))
        return self.stats_from_sorted(fired, tiw=tiw)

    def stats_from_mask(self, fired, *, tiw: bool = True) -> EventStats:
        return self.stats_from_sorted(self.fired_sorted(fired), tiw=tiw)

    def class_weights(self, keep_classes) -> np.ndarray:
        """1 for classes in `keep_classes`, 0 otherwise (aligned with `self.classes`)."""
        return np.isin(self.classes, np.asarray(keep_classes)).astype(float)


def bid_onsets(hand_episodes: pd.DataFrame, *, retimed: bool = False) -> pd.DataFrame:
    """Bid onsets for recall: ep_type == bid, not left-censored, with an onset.

    With `retimed`, `onset_retimed_s` replaces `onset_s` where re-timing did not fail.
    """
    e = hand_episodes
    keep = (e["ep_type"] == config.EP_BID) & ~e["left_censored"].fillna(False).astype(bool)
    e = e[keep]
    onset = e["onset_s"].to_numpy(float)
    if retimed and "onset_retimed_s" in e:
        ok = ~e["retime_failed"].fillna(True).astype(bool).to_numpy()
        rt = e["onset_retimed_s"].to_numpy(float)
        onset = np.where(ok & ~np.isnan(rt), rt, onset)
    out = pd.DataFrame({"class_id": e["class_id"].to_numpy(), "onset_s": onset})
    return out[out["onset_s"].notna()].reset_index(drop=True)


# Array-level wrappers with the names of spec §13 (each builds an EventData).


def firings(p, t, tau, H: float = config.H_S, g: float = config.G_S, *, in_grid=None,
            groups=None) -> np.ndarray:
    """Firing mask F_tau (refractory H - g on the causal grid, per class). Never reads labels."""
    return EventData(t, groups, in_grid=in_grid, H=H, g=g).firings(p, tau)


def far(fired, y, in_grid=None, *, delta: float = config.DELTA_S) -> float:
    """False firings per at-risk minute of G_def (x 60 for per hour)."""
    fired = np.asarray(fired, bool)
    y = _f(y)
    grid = np.ones(fired.size, bool) if in_grid is None else np.asarray(in_grid, bool)
    n_def = int((grid & ~np.isnan(y)).sum())
    if n_def == 0:
        return float("nan")
    return float((fired & grid & (y == 0)).sum()) / (n_def * delta / 60.0)


def firing_rate(fired, in_grid=None, *, delta: float = config.DELTA_S) -> float:
    """All firings per minute of G."""
    fired = np.asarray(fired, bool)
    grid = np.ones(fired.size, bool) if in_grid is None else np.asarray(in_grid, bool)
    n = int(grid.sum())
    return float((fired & grid).sum()) / (n * delta / 60.0) if n else float("nan")


def _mask_stats(fired, t, *, groups, in_grid, onsets, onset_groups, H, g, delta,
                tiw=True) -> EventStats:
    d = EventData(t, groups, in_grid=in_grid, onsets=onsets, onset_groups=onset_groups,
                  H=H, g=g, delta=delta)
    return d.stats_from_mask(fired, tiw=tiw)


def recall(fired, t, onsets, H: float = config.H_S, g: float = config.G_S, *, groups=None,
           onset_groups=None, in_grid=None, delta: float = config.DELTA_S) -> float:
    """Share of evaluable bids with a firing in [o - H, o - g)."""
    s = _mask_stats(fired, t, groups=groups, in_grid=in_grid, onsets=onsets,
                    onset_groups=onset_groups, H=H, g=g, delta=delta, tiw=False)
    return s.summary()["recall"]


def lead(fired, t, onsets, H: float = config.H_S, g: float = config.G_S, *, groups=None,
         onset_groups=None, in_grid=None, delta: float = config.DELTA_S) -> np.ndarray:
    """Lead o - (earliest firing in [o - H, o - g)) of every detected evaluable bid."""
    s = _mask_stats(fired, t, groups=groups, in_grid=in_grid, onsets=onsets,
                    onset_groups=onset_groups, H=H, g=g, delta=delta, tiw=False)
    return s.lead_s


def time_in_warning(fired, t, H: float = config.H_S, g: float = config.G_S, *, groups=None,
                    in_grid=None, delta: float = config.DELTA_S, mode: str = "time") -> float:
    """Share of G inside the warning intervals (s + g, s + H]: G-time ("time", default) or
    G points ("points", the literal §8.3 count)."""
    s = _mask_stats(fired, t, groups=groups, in_grid=in_grid, onsets=None,
                    onset_groups=None, H=H, g=g, delta=delta)
    key = {"time": "tiw", "points": "tiw_points"}[mode]
    return s.summary()[key]


# --------------------------------------------------------------------------- FROC, thresholds


def froc_thresholds(data: EventData, p, n: int = 200) -> np.ndarray:
    """Descending thresholds, +inf first, dense at the top: exceedance counts are spaced
    geometrically over the distinct scores on G (low FAR needs fine resolution)."""
    ps = data._sorted(p, "p")[data.in_grid]
    u = np.unique(ps[np.isfinite(ps)])[::-1]
    if u.size == 0:
        return np.array([np.inf])
    k = np.unique(np.round(np.geomspace(1, u.size, num=min(n, u.size))).astype(int))
    return np.r_[np.inf, u[k - 1]]


def froc(data: EventData, p, *, thresholds=None, n_thresholds: int = 200, w=None,
         tiw: bool = True, far_stop: float | None = None) -> pd.DataFrame:
    """One row per threshold (descending): tau, far, rate, recall, tiw, r0, eventlift0, lead
    and counts. `w` restricts or reweights classes (e.g. kept classes of a shift).

    `far_stop` ends the scan after the first threshold whose FAR exceeds it: low thresholds
    fire constantly, cost the most and are never read when only FAR <= far_stop / 2 matters
    (callers pass a 2x margin because FAR is not monotone in tau)."""
    taus = froc_thresholds(data, p, n_thresholds) if thresholds is None else _f(thresholds)
    ps = data._sorted(p, "p")
    rows = []
    for tau in np.sort(taus)[::-1]:
        fired = data._fire_sorted(ps, np.full(data.n_points, tau))
        rows.append({"tau": float(tau), **data.stats_from_sorted(fired, tiw=tiw).summary(w)})
        if far_stop is not None and rows[-1]["far"] > far_stop:
            break
    return pd.DataFrame(rows)


def froc_envelope(curve: pd.DataFrame) -> pd.DataFrame:
    """The FROC frontier (lower envelope in FAR): for each recall the smallest FAR reaching
    it. FAR is not monotone in tau under the refractory rule, so this is what gets drawn."""
    c = curve[["far", "recall", "tau"]].dropna(subset=["far", "recall"])
    c = c.sort_values(["far", "recall"], ascending=[True, False], kind="mergesort")
    best = np.maximum.accumulate(c["recall"].to_numpy())
    keep = np.r_[True, c["recall"].to_numpy()[1:] > best[:-1]] if len(c) else np.zeros(0, bool)
    return c[keep].reset_index(drop=True)


def recall_at_far(curve: pd.DataFrame, far_budget: float, *, interpolate: bool = False
                  ) -> float:
    """Best recall with FAR <= budget on the envelope (step), or linearly interpolated
    between envelope vertices (achievable by randomising between two thresholds)."""
    env = froc_envelope(curve)
    if env.empty:
        return float("nan")
    f, r = env["far"].to_numpy(), env["recall"].to_numpy()
    if interpolate:
        return float(np.interp(far_budget, f, r, left=0.0))
    ok = f <= far_budget + _TOL
    return float(r[ok].max()) if ok.any() else 0.0


def pafroc(curve: pd.DataFrame, far_max: float = config.FROC_FAR_MAX, *,
           interpolate: bool = False, normalise: bool = True) -> float:
    """Partial area under the FROC envelope on [0, far_max] (normalised: mean recall)."""
    env = froc_envelope(curve)
    if env.empty:
        return float("nan")
    f, r = env["far"].to_numpy(), env["recall"].to_numpy()
    if interpolate:
        xs = np.r_[0.0, f[f < far_max], far_max]
        area = float(np.trapezoid(np.interp(xs, f, r, left=0.0), xs))
    else:
        nxt = np.r_[f[1:], np.inf]
        area = float(np.sum(r * np.clip(np.minimum(nxt, far_max) - f, 0.0, None)))
    return area / far_max if normalise else area


def rate_at_far(curve: pd.DataFrame, far_budget: float) -> float:
    """Firing rate at which FAR first reaches the budget as tau decreases (linear
    interpolation between the two thresholds around the crossing)."""
    c = curve.sort_values("tau", ascending=False, kind="mergesort")
    f, r = c["far"].to_numpy(float), c["rate"].to_numpy(float)
    over = np.flatnonzero(f > far_budget + _TOL)
    if over.size == 0:
        return float(np.nanmax(r)) if r.size else 0.0
    i = int(over[0])
    if i == 0:
        return 0.0
    f0, f1, r0_, r1 = f[i - 1], f[i], r[i - 1], r[i]
    return float(r0_ + (far_budget - f0) * (r1 - r0_) / (f1 - f0))


def tau_for_rate(data: EventData, p, rate: float, *, w=None) -> float:
    """Threshold whose firing rate on G is closest to `rate` (ties -> the higher tau).

    The greedy refractory count is monotone in the candidate set (greedy earliest-first is
    the largest spaced subset), so the rate is monotone in tau. The search walks down from
    the top scores (exponential, then bisection), so it never fires at every point.
    """
    ps = data._sorted(p, "p")
    u = np.unique(ps[data.in_grid & np.isfinite(ps)])
    n = u.size
    if not rate > 0 or n == 0:
        return float("inf")
    wc = _weights(w, len(data.classes))
    grid_min = float(np.dot(wc, data.n_grid)) * data.delta / 60.0
    cache: dict[int, float] = {0: 0.0}

    def tau_of(j: int) -> float:          # j = number of top distinct scores that may fire
        return float(u[n - j]) if j > 0 else float("inf")

    def r(j: int) -> float:
        if j not in cache:
            fired = data._fire_sorted(ps, np.full(data.n_points, u[n - j]))
            n_fire = np.bincount(data.cls[fired], minlength=len(data.classes))
            cache[j] = float(np.dot(wc, n_fire)) / grid_min
        return cache[j]

    lo, hi = 0, 1                         # r(lo) < rate
    while r(hi) < rate:
        if hi >= n:
            return float(u[0])            # even firing at every candidate is below `rate`
        lo, hi = hi, min(2 * hi, n)
    while hi - lo > 1:                    # r(lo) < rate <= r(hi)
        mid = (lo + hi) // 2
        if r(mid) >= rate:
            hi = mid
        else:
            lo = mid
    return tau_of(hi) if (r(hi) - rate) < (rate - r(lo)) else tau_of(lo)


def tau_rate_matched(inner: EventData, p_inner, train: EventData, p_train, far_budget: float,
                     *, n_thresholds: int = 200) -> dict:
    """Rate-matched threshold (§8.3): the inner-OOF firing rate at FAR = budget, matched by
    the outer model's firing rate on the training fold's G. Apply `tau` to the test fold."""
    rate_star = rate_at_far(froc(inner, p_inner, n_thresholds=n_thresholds, tiw=False,
                                 far_stop=2 * far_budget), far_budget)
    tau = tau_for_rate(train, p_train, rate_star)
    return {"tau": tau, "rate_star": rate_star, "rate_train": train.rate(p_train, tau),
            "far_budget": float(far_budget)}


def random_alarm(data: EventData, rate: float, *, n_rep: int = config.SHIFT_REPEATS,
                 seed: int = config.SEED, w=None) -> dict:
    """Matched-rate random alarm reference (A9): iid uniform scores thresholded to the given
    firing rate, same firing rule. Returns means (and per-rep recalls)."""
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(n_rep):
        u = rng.random(data.n_points)
        tau = tau_for_rate(data, u, rate, w=w)
        reps.append(data.stats(u, tau, tiw=True).summary(w))
    df = pd.DataFrame(reps)
    return {"recall": float(df["recall"].mean()), "recall_sd": float(df["recall"].std()),
            "far": float(df["far"].mean()), "rate": float(df["rate"].mean()),
            "tiw": float(df["tiw"].mean()), "recalls": df["recall"].to_numpy()}


# --------------------------------------------------------------------------- shifts (§8.4)


def _sort_order(groups, t) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(groups)
    classes = np.unique(g)
    cls = _class_index(classes, g)
    order = np.lexsort((np.arange(g.size) if t is None else _f(t), cls))
    return order, cls[order]


def _runs(keys: list[np.ndarray]) -> np.ndarray:
    """Start positions of maximal runs of equal consecutive keys, plus the end."""
    n = keys[0].size
    change = np.zeros(n, bool)
    if n:
        change[0] = True
    for k in keys:
        change[1:] |= k[1:] != k[:-1]
    return np.r_[np.flatnonzero(change), n]


def _shift_plan(groups, t, mask, segments=None) -> tuple[np.ndarray, np.ndarray]:
    """(sel, bounds): masked point indices in (class, t) order and the boundaries of the runs
    to shift within (classes; with `segments`, maximal same-segment runs of a class)."""
    order, cls = _sort_order(groups, t)
    m_ord = mask[order]
    sel = order[m_ord]
    keys = [cls[m_ord]]
    if segments is not None:
        seg = pd.factorize(pd.Series(np.asarray(segments, dtype=object)),
                           use_na_sentinel=True)[0]
        keys.append(seg[sel])
    return sel, _runs(keys)


def _roll_global(p, sel, bounds, rng, min_shift, min_n) -> tuple[np.ndarray, np.ndarray]:
    out = np.full(p.size, np.nan)
    kept = np.zeros(p.size, bool)
    for a, b in itertools.pairwise(bounds):
        n = b - a
        if n < max(min_n, 2 * min_shift + 1):
            continue
        idx = sel[a:b]
        out[idx] = np.roll(p[idx], int(rng.integers(min_shift, n - min_shift + 1)))
        kept[idx] = True
    return out, kept


def _roll_local(p, mask, sel, bounds, rng, block, min_block, min_shift
                ) -> tuple[np.ndarray, np.ndarray]:
    out = np.where(mask, p, np.nan)
    shifted = np.zeros(p.size, bool)
    for a, b in itertools.pairwise(bounds):
        n = b - a
        if n < min_block:
            continue
        starts = list(range(0, n, block))
        if len(starts) > 1 and n - starts[-1] < min_block:
            starts.pop()
        for s0, e0 in zip(starts, starts[1:] + [n], strict=True):
            idx = sel[a + s0:a + e0]
            out[idx] = np.roll(p[idx], int(rng.integers(min_shift, (e0 - s0) - min_shift + 1)))
            shifted[idx] = True
    return out, shifted


def shift_global(p, groups, *, t=None, mask=None, seed: int = config.SEED, rng=None,
                 min_shift: int = config.SHIFT_GLOBAL_MIN,
                 min_n: int = config.SHIFT_GLOBAL_MIN_N) -> tuple[np.ndarray, np.ndarray]:
    """Circular shift of each class's scores by u_c ~ U{min_shift, ..., n_c - min_shift}
    indices over its points in time order (labels stay put). Classes with n_c < min_n are
    skipped: their scores become NaN and `kept` is False there. Points outside `mask` -> NaN.
    Returns (p_shifted, kept) in the caller's order."""
    rng = np.random.default_rng(seed) if rng is None else rng
    p = _f(p)
    m = np.ones(p.size, bool) if mask is None else np.asarray(mask, bool)
    sel, bounds = _shift_plan(groups, t, m)
    return _roll_global(p, sel, bounds, rng, min_shift, min_n)


def shift_local(p, groups, segments, *, t=None, mask=None, seed: int = config.SEED, rng=None,
                block: int = config.SHIFT_LOCAL_BLOCK,
                min_block: int = config.SHIFT_LOCAL_MIN_BLOCK,
                min_shift: int = config.SHIFT_LOCAL_MIN_U) -> tuple[np.ndarray, np.ndarray]:
    """Within-moment shift: maximal same-segment runs of each class are cut into consecutive
    `block`-point blocks; a short tail block merges into the previous block of the run, a
    run shorter than `min_block` stays unshifted. Each block of n_b points is rolled by
    u_b ~ U{min_shift, ..., n_b - min_shift}. Returns (p_shifted, shifted) in caller order;
    unshifted points keep their score, points outside `mask` -> NaN."""
    rng = np.random.default_rng(seed) if rng is None else rng
    p = _f(p)
    m = np.ones(p.size, bool) if mask is None else np.asarray(mask, bool)
    sel, bounds = _shift_plan(groups, t, m, segments)
    return _roll_local(p, m, sel, bounds, rng, block, min_block, min_shift)


class TimingGain:
    """TG = AP - mean_r AP(shift_r) on the same points (global: "which class" floor; local:
    "which moment"). Callable with bootstrap class weights; `summary()` gives the parts.

    Uses defined labels only (pass the at-risk defined points of each class in time order,
    or give `t`). Global: classes with n_c < 400 are dropped from both terms and counted.
    `within=True` gives TG_w on AP_w (with `weights` = K_c as in `WithinAP`).
    """

    def __init__(self, y, p, groups, *, t=None, segments=None, kind: str = "global",
                 n_rep: int = config.SHIFT_REPEATS, seed: int = config.SEED,
                 within: bool = False, weights=None, classes=None):
        y, p, m = _defined(y, p)
        g = np.asarray(groups)[m]
        tt = None if t is None else _f(t)[m]
        self.kind, self.n_rep, self.within = kind, n_rep, within
        self.classes = _classes(np.asarray(groups), classes)
        rng = np.random.default_rng(seed)
        if kind not in ("global", "local"):
            raise ValueError(f"unknown shift kind {kind!r}")
        if kind == "local" and segments is None:
            raise ValueError("local shift needs segments")
        everywhere = np.ones(y.size, bool)
        sel, bounds = _shift_plan(g, tt, everywhere, None if kind == "global" else
                                  np.asarray(segments, dtype=object)[m])
        reps, masks = [], []
        for _ in range(n_rep):
            if kind == "global":
                ps, mk = _roll_global(p, sel, bounds, rng, config.SHIFT_GLOBAL_MIN,
                                      config.SHIFT_GLOBAL_MIN_N)
            else:
                ps, mk = _roll_local(p, everywhere, sel, bounds, rng, config.SHIFT_LOCAL_BLOCK,
                                     config.SHIFT_LOCAL_MIN_BLOCK, config.SHIFT_LOCAL_MIN_U)
            reps.append(ps)
            masks.append(mk)
        keep = masks[0] if kind == "global" else np.ones(y.size, bool)
        self.n_classes_skipped = int(np.unique(g).size - np.unique(g[keep]).size)
        self.share_unshifted = float(np.mean([~mk for mk in masks])) if kind == "local" else 0.0

        def make(scores):
            if within:
                return WithinAP(y[keep], scores[keep], g[keep], weights=weights,
                                classes=self.classes)
            return PooledAP(y[keep], scores[keep], g[keep], classes=self.classes)

        self._base = make(p)
        self._shifted = [make(ps) for ps in reps]

    def __call__(self, w=None) -> float:
        return self._base(w) - float(np.mean([s(w) for s in self._shifted]))

    def batch(self, W: np.ndarray) -> np.ndarray:
        return self._base.batch(W) - np.mean([s.batch(W) for s in self._shifted], axis=0)

    def summary(self) -> dict:
        a = self._base()
        a_shift = float(np.mean([s() for s in self._shifted]))
        return {"ap": a, "ap_shift": a_shift, "tg": a - a_shift, "kind": self.kind,
                "n_rep": self.n_rep, "n_classes_skipped": self.n_classes_skipped,
                "share_unshifted": self.share_unshifted}


def timing_gain(y, p, groups, **kw) -> dict:
    """TG / TG_local point estimates (see `TimingGain`)."""
    return TimingGain(y, p, groups, **kw).summary()


@dataclass
class ShiftReference:
    """FROC curves of shifted predictions (same firing rule, labels fixed) and the class
    weights they are valid on; `recall_at(far)` is the chance recall at a matched FAR."""

    curves: list
    w: np.ndarray
    n_classes_skipped: int
    kind: str

    def recall_at(self, far: float) -> float:
        if not self.curves or not np.isfinite(far):
            return float("nan")
        return float(np.mean([recall_at_far(c, far, interpolate=True) for c in self.curves]))


def shift_reference(data: EventData, p, *, kind: str = "global", segments=None,
                    n_rep: int = config.SHIFT_REPEATS, seed: int = config.SEED,
                    n_thresholds: int = 200, far_stop: float | None = None
                    ) -> ShiftReference:
    """Shift scores over each class's G points (global: U{150, n - 150}, classes with < 400 G
    points left out of both sides; local: same-segment blocks) and build their FROC curves
    (up to `far_stop`, see `froc`)."""
    if kind not in ("global", "local"):
        raise ValueError(f"unknown shift kind {kind!r}")
    if kind == "local" and segments is None:
        raise ValueError("local shift needs segments")
    rng = np.random.default_rng(seed)
    ps0 = data._sorted(p, "p")
    seg = None if kind == "global" else np.asarray(segments, dtype=object)[data.order]
    sel, bounds = _shift_plan(data.cls, None, data.in_grid, seg)
    keep = np.ones(len(data.classes), bool)
    shifted = []
    for _ in range(n_rep):
        if kind == "global":
            sh, kept = _roll_global(ps0, sel, bounds, rng, config.SHIFT_GLOBAL_MIN,
                                    config.SHIFT_GLOBAL_MIN_N)
            keep &= np.isin(np.arange(len(data.classes)), np.unique(data.cls[kept]))
        else:
            sh, _ = _roll_local(ps0, data.in_grid, sel, bounds, rng, config.SHIFT_LOCAL_BLOCK,
                                config.SHIFT_LOCAL_MIN_BLOCK, config.SHIFT_LOCAL_MIN_U)
        unsorted = np.empty(data.n_points)
        unsorted[data.order] = sh
        shifted.append(unsorted)
    w = keep.astype(float)
    curves = [froc(data, x, n_thresholds=n_thresholds, w=w, tiw=False, far_stop=far_stop)
              for x in shifted]
    return ShiftReference(curves=curves, w=w, n_classes_skipped=int((~keep).sum()), kind=kind)


def eventlift_shift(data: EventData, p, tau, *, kind: str = "global", segments=None,
                    n_rep: int = config.SHIFT_REPEATS, seed: int = config.SEED,
                    n_thresholds: int = 200, ref: ShiftReference | None = None) -> dict:
    """EventLift = Recall(tau) / Recall_shift(FAR_realised) (§8.3 primary chance level).

    The shifted predictions' recall is read at the model's realised FAR, linearly
    interpolated on their FROC envelope (conservative). Both sides use only the classes the
    shift kept. Pass `ref` to reuse one `shift_reference` across thresholds."""
    if ref is None:
        far = data.stats(p, tau, tiw=False).summary()["far"]
        ref = shift_reference(data, p, kind=kind, segments=segments, n_rep=n_rep, seed=seed,
                              n_thresholds=n_thresholds,
                              far_stop=2 * far if np.isfinite(far) and far > 0 else None)
    model = data.stats(p, tau).summary(ref.w)
    r_shift = ref.recall_at(model["far"])
    return {"recall": model["recall"], "far": model["far"], "tiw": model["tiw"],
            "recall_shift": r_shift,
            "eventlift": model["recall"] / r_shift if r_shift > 0 else float("nan"),
            "n_classes_skipped": ref.n_classes_skipped}


def budget_key(far_budget: float) -> str:
    """0.1 -> "far01", 0.5 -> "far05" (results column suffixes)."""
    return f"far{round(far_budget * 10):02d}"


def event_metrics(data: EventData, p, taus: Mapping[float, object], *,
                  n_rep: int = config.SHIFT_REPEATS, seed: int = config.SEED,
                  far_max: float = config.FROC_FAR_MAX, n_thresholds: int = 200) -> dict:
    """Event columns of a `results` row. `taus` maps each FAR budget to its rate-matched
    threshold (scalar or per point, e.g. per fold); lead is reported at the first budget.
    `pafroc` is threshold-free and always computed."""
    row: dict = {}
    summ = {b: data.stats(p, tau).summary() for b, tau in taus.items()}
    ref = None
    if taus:
        top = max([v["far"] for v in summ.values() if np.isfinite(v["far"])] + list(taus))
        ref = shift_reference(data, p, n_rep=n_rep, seed=seed, n_thresholds=n_thresholds,
                              far_stop=2 * top)
    for i, (budget, tau) in enumerate(taus.items()):
        k = budget_key(budget)
        s = summ[budget]
        row[f"recall_{k}"] = s["recall"]
        row[f"realised_{k}"] = s["far"]
        row[f"tiw_{k}"] = s["tiw"]
        row[f"eventlift0_{k}"] = s["eventlift0"]
        row[f"eventlift_shift_{k}"] = eventlift_shift(data, p, tau, ref=ref)["eventlift"]
        row[f"recall_random_{k}"] = random_alarm(data, s["rate"], n_rep=n_rep,
                                                 seed=seed)["recall"]
        if i == 0:
            row["lead_med_s"] = s["lead_med_s"]
            row["n_onsets"] = int(s["n_eval"])
    row["pafroc"] = pafroc(froc(data, p, n_thresholds=n_thresholds, tiw=False,
                                far_stop=2 * far_max), far_max)
    return row


def segment_index(t, groups, segments: pd.DataFrame) -> np.ndarray:
    """Index of the timeline segment current at t (by class; -1 before the first segment).
    `segments` has class_id, seg_idx, start_s (the segments table). Label-side use only
    (the local shift); never a feature."""
    t, g = _f(t), np.asarray(groups)
    out = np.full(t.size, -1, dtype=np.int64)
    for c, seg in segments.sort_values(["class_id", "start_s"]).groupby("class_id"):
        m = g == c
        if m.any():
            k = np.searchsorted(seg["start_s"].to_numpy(float), t[m], side="right") - 1
            out[m] = np.where(k >= 0, seg["seg_idx"].to_numpy()[np.maximum(k, 0)], -1)
    return out


def _bool_col(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df:
        return np.ones(len(df), bool)
    return pd.Series(df[col], dtype="boolean").fillna(False).to_numpy(dtype=bool)


def fold_thresholds(test: pd.DataFrame, inner_rows: pd.DataFrame,
                    train_scores: pd.DataFrame | None, *, H: float = config.H_S,
                    g: float = config.G_S, score_col: str = "p_cal",
                    far_budgets=config.FAR_BUDGETS, n_thresholds: int = 200
                    ) -> tuple[dict, str]:
    """Per-point rate-matched thresholds of the test rows (§8.3), fold by fold.

    Step 1: on fold k's inner OOF rows (the training fold's causal grid G, undefined labels
    included), the firing rate rate* at which FAR = budget. Step 2: tau such that the outer
    model's firing rate on the training fold's G (`train_scores`) equals rate*. The test
    fold's own scores are never used: without training-fold scores, or without inner rows
    for every test fold, the thresholds are NaN (`method` says why). Returns
    ({budget: tau per test row, in `test`'s row order}, method)."""
    fold_te = test["fold"].to_numpy() if "fold" in test else np.zeros(len(test), int)
    have = len(inner_rows) and set(np.unique(fold_te)) <= set(np.unique(inner_rows["fold"]))
    if not have:
        return {}, "none_no_inner"
    if train_scores is None or not len(train_scores):
        return {}, "none_no_train_scores"
    taus = {b: np.full(len(test), np.nan) for b in far_budgets}
    for k in np.unique(fold_te):
        ik, mk = inner_rows[inner_rows["fold"] == k], fold_te == k
        inner_k = EventData(ik["t_s"].to_numpy(float), ik["class_id"].to_numpy(),
                            in_grid=_bool_col(ik, "in_G"), y=ik["y"].to_numpy(float), H=H, g=g)
        curve_k = froc(inner_k, ik[score_col].to_numpy(float), n_thresholds=n_thresholds,
                       tiw=False, far_stop=2 * max(far_budgets))
        tr = train_scores[train_scores["fold"] == k]
        if not len(tr):
            continue
        ref_k = EventData(tr["t_s"].to_numpy(float), tr["class_id"].to_numpy(),
                          in_grid=_bool_col(tr, "in_G"), H=H, g=g)
        for budget in far_budgets:
            taus[budget][mk] = tau_for_rate(ref_k, tr[score_col].to_numpy(float),
                                            rate_at_far(curve_k, budget))
    return taus, "rate_matched"


def _boot_cols(prefix: str, samples: np.ndarray, ci: float = 0.95) -> dict:
    ok = np.asarray(samples, float)
    ok = ok[np.isfinite(ok)]
    if not ok.size:
        return {f"{prefix}_lo": float("nan"), f"{prefix}_hi": float("nan")}
    lo, hi = np.percentile(ok, [50.0 * (1 - ci), 50.0 * (1 + ci)])
    return {f"{prefix}_lo": float(lo), f"{prefix}_hi": float(hi)}


def event_boot(stats: EventStats, W: np.ndarray) -> dict[str, np.ndarray]:
    """Recall, FAR, firing rate and median lead of one threshold for every bootstrap weight
    row (class resampling, vectorised)."""
    W = np.asarray(W, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        n_eval = W @ stats.n_eval
        recall = np.where(n_eval > 0, (W @ stats.n_hit) / n_eval, np.nan)
        def_min = (W @ stats.n_def) * stats.delta / 60.0
        far = np.where(def_min > 0, (W @ stats.n_false) / def_min, np.nan)
        grid_min = (W @ stats.n_grid) * stats.delta / 60.0
        rate = np.where(grid_min > 0, (W @ stats.n_fire) / grid_min, np.nan)
    lead = np.array([_wmedian(stats.lead_s, w[stats.lead_cls]) if stats.lead_s.size
                     else np.nan for w in W])
    return {"recall": recall, "far": far, "rate": rate, "lead": lead}


def condition_metrics(oof: pd.DataFrame, points: pd.DataFrame, *, H: float = config.H_S,
                      g: float = config.G_S, inner: pd.DataFrame | None = None,
                      onsets: pd.DataFrame | None = None, segments: pd.DataFrame | None = None,
                      train_scores: pd.DataFrame | None = None,
                      score_col: str = "p_cal", far_budgets=config.FAR_BUDGETS,
                      n_boot: int = config.N_BOOT, n_rep: int = config.SHIFT_REPEATS,
                      seed: int = config.SEED, n_thresholds: int = 200) -> dict:
    """Within-class, timing-gain and event columns of one condition's pooled `results` row
    (the `experiments.evaluate` scorer hook), each with its class-bootstrap 95% CI (§8.5:
    AP_w, pooled AP, TG, TG_local, Recall@lambda and lead share one set of resamples).

    * `oof`: OOF rows (class_id, t_s, fold, y, p_raw, p_cal, inner, and the event grid
      `in_G`). Test rows must cover the whole causal grid G of their classes (undefined
      labels included, §8.3). Inner rows (inner == True, here or in `inner`) must cover
      each training fold's whole G too: they set the rate-matched thresholds.
    * `points`: in_G (when the OOF rows carry none) and next_bid_s (bid onsets default to
      the distinct next_bid_s of each class; pass `onsets` (class_id, onset_s) for other
      targets, e.g. T-speech).
    * `segments`: the segments table, for the local shift (TG_local); NaN without it.
    * `train_scores`: outer-model scores on each training fold's G (class_id, t_s, fold,
      score_col), step 2 of the thresholds (`fold_thresholds`). Without them, or without
      inner rows, the threshold columns are NaN: test scores never set a threshold and the
      oracle value is never substituted. `tau_method` (not a results column) says why.
    """
    is_inner = _bool_col(oof, "inner") if "inner" in oof else np.zeros(len(oof), bool)
    test = oof[~is_inner]
    inner_rows = oof[is_inner] if inner is None else inner
    keys = ["class_id", "t_s"]
    pcols = [c for c in ("in_G", "next_bid_s") if c in points]
    P = points[keys + pcols].drop_duplicates(keys)

    def with_points(df: pd.DataFrame) -> pd.DataFrame:
        """Points columns the rows do not carry (an OOF row's own event grid wins)."""
        add = [c for c in pcols if c not in df]
        return df.merge(P[keys + add], on=keys, how="left") if add else df.copy()

    te = with_points(test).sort_values(keys, kind="mergesort").reset_index(drop=True)
    if "fold" not in te:
        te["fold"] = 0
    cls, t = te["class_id"].to_numpy(), te["t_s"].to_numpy(float)
    y, p = te["y"].to_numpy(float), te[score_col].to_numpy(float)
    if onsets is None and "next_bid_s" in points:
        nb = points[["class_id", "next_bid_s"]].dropna().drop_duplicates()
        onsets = nb.rename(columns={"next_bid_s": "onset_s"})
    out: dict = {}
    classes = np.unique(cls)
    W = bootstrap_weights(len(classes), n_boot, seed)

    # within-class and pooled AP (K_c = distinct bids among positives)
    d = ~np.isnan(y)
    nxt = te["next_bid_s"].to_numpy(float) if "next_bid_s" in te else np.full(len(te), np.nan)
    k_c = bid_counts(y[d], cls[d], nxt[d]) if np.isfinite(nxt[d & (y == 1)]).all() else None
    apw = WithinAP(y[d], p[d], cls[d], weights=k_c, classes=classes)
    boot = _result(apw(), _evaluate(apw, W), 0.95, len(W), seed)
    out.update({"ap_within": boot.point, "ap_within_lo": boot.lo, "ap_within_hi": boot.hi})
    out.update(_boot_cols("ap", _evaluate(PooledAP(y[d], p[d], cls[d], classes=classes), W)))

    # timing decomposition on the at-risk defined points
    tg = TimingGain(y[d], p[d], cls[d], t=t[d], kind="global", n_rep=n_rep, seed=seed,
                    classes=classes)
    sg = tg.summary()
    out.update({"ap_shift": sg["ap_shift"], "timing_gain": sg["tg"], "ap_shift_base": sg["ap"],
                "n_classes_shift_skipped": sg["n_classes_skipped"]})
    out.update(_boot_cols("timing_gain", _evaluate(tg, W)))
    if segments is not None:
        seg = segment_index(t, cls, segments)
        tlo = TimingGain(y[d], p[d], cls[d], t=t[d], segments=seg[d], kind="local",
                         n_rep=n_rep, seed=seed, classes=classes)
        tl = tlo.summary()
        out.update({"ap_local": tl["ap_shift"], "timing_gain_local": tl["tg"],
                    "ap_local_base": tl["ap"], "share_unshifted_local": tl["share_unshifted"]})
        out.update(_boot_cols("timing_gain_local", _evaluate(tlo, W)))
    else:
        out.update({"ap_local": float("nan"), "timing_gain_local": float("nan"),
                    "ap_local_base": float("nan")})

    # event metrics on the grid, thresholds per fold (never from test scores)
    o = onsets if onsets is not None else pd.DataFrame({"class_id": [], "onset_s": []})
    grid = _bool_col(te, "in_G")
    data = EventData(t, cls, in_grid=grid, y=y, onsets=o["onset_s"].to_numpy(float),
                     onset_groups=o["class_id"].to_numpy(), H=H, g=g)
    ir = with_points(inner_rows) if len(inner_rows) else inner_rows
    taus, method = fold_thresholds(te, ir, train_scores, H=H, g=g, score_col=score_col,
                                   far_budgets=far_budgets, n_thresholds=n_thresholds)
    out["tau_method"] = method
    out.update(event_metrics(data, p, taus, n_rep=n_rep, seed=seed,
                             n_thresholds=n_thresholds))
    for i, (budget, tau) in enumerate(taus.items()):
        k = budget_key(budget)
        if not np.isfinite(tau).all():
            continue
        eb = event_boot(data.stats(p, tau, tiw=False), W)
        out.update(_boot_cols(f"recall_{k}", eb["recall"]))
        if i == 0:
            out.update(_boot_cols("lead_med_s", eb["lead"]))
    for budget in far_budgets:
        k = budget_key(budget)
        for col in ("recall", "realised", "tiw", "eventlift0", "eventlift_shift",
                    "recall_random", "recall_{}_lo", "recall_{}_hi"):
            name = col.format(k) if "{}" in col else f"{col}_{k}"
            out.setdefault(name, float("nan"))
    for col in ("lead_med_s", "lead_med_s_lo", "lead_med_s_hi"):
        out.setdefault(col, float("nan"))
    return out


# --------------------------------------------------------------------------- EV (S3)


class FrocCounts:
    """Per-class firing counts of one score vector at a set of thresholds, so FROC curves
    (and recall at any FAR on their envelope) can be re-aggregated for every bootstrap
    resample of classes without re-running the firing rule."""

    def __init__(self, data: EventData, p, *, thresholds=None, n_thresholds: int = 200,
                 far_stop: float | None = None, classes=None):
        taus = froc_thresholds(data, p, n_thresholds) if thresholds is None else _f(thresholds)
        ps = data._sorted(p, "p")
        idx = (np.arange(len(data.classes)) if classes is None
               else _class_index(data.classes, np.asarray(classes)))
        C = len(data.classes)
        self.n_def = data.n_def[idx].astype(float)
        nf, nh, ne = [], [], None
        for tau in np.sort(taus)[::-1]:
            st = data.stats_from_sorted(data._fire_sorted(ps, np.full(data.n_points, tau)),
                                        tiw=False)
            nf.append(st.n_false.astype(float))
            nh.append(st.n_hit.astype(float))
            ne = st.n_eval.astype(float)
            if far_stop is not None and st.summary()["far"] > far_stop:
                break
        self.n_false = np.asarray(nf).reshape(len(nf), C)[:, idx]
        self.n_hit = np.asarray(nh).reshape(len(nh), C)[:, idx]
        self.n_eval = (ne if ne is not None else np.zeros(C))[idx]
        self.delta = data.delta

    def recall_at(self, far, W) -> np.ndarray:
        """Recall at FAR `far` (one value per weight row) on each row's FROC envelope,
        linearly interpolated between vertices (as for the shifted references)."""
        W = np.atleast_2d(np.asarray(W, float))
        far = np.broadcast_to(np.asarray(far, float), (len(W),))
        with np.errstate(invalid="ignore", divide="ignore"):
            F = (W @ self.n_false.T) / ((W @ self.n_def) * self.delta / 60.0)[:, None]
            R = (W @ self.n_hit.T) / (W @ self.n_eval)[:, None]
        out = np.full(len(W), np.nan)
        for b in range(len(W)):
            if np.isfinite(far[b]):
                out[b] = _envelope_interp(F[b], R[b], far[b])
        return out


def _envelope_interp(f: np.ndarray, r: np.ndarray, x: float) -> float:
    """`recall_at_far(..., interpolate=True)` on raw arrays (FROC lower envelope)."""
    ok = np.isfinite(f) & np.isfinite(r)
    f, r = f[ok], r[ok]
    if not f.size:
        return float("nan")
    o = np.lexsort((-r, f))
    f, r = f[o], r[o]
    best = np.maximum.accumulate(r)
    keep = np.r_[True, r[1:] > best[:-1]]
    return float(np.interp(x, f[keep], r[keep], left=0.0))


def event_comparison(model: pd.DataFrame, refs: Mapping[str, pd.DataFrame],
                     points: pd.DataFrame, *, condition: str = "K0-M3",
                     H: float = config.H_S, g: float = config.G_S,
                     onsets: pd.DataFrame | None = None,
                     train_scores: pd.DataFrame | None = None, score_col: str = "p_cal",
                     far_budgets=config.FAR_BUDGETS, n_boot: int = config.N_BOOT,
                     n_rep: int = config.SHIFT_REPEATS, seed: int = config.SEED,
                     n_thresholds: int = 200) -> pd.DataFrame:
    """EV (S3, §8.3, A9): the model's event recall at each FAR budget (rate-matched tau,
    realised FAR, TiW, lead, class-bootstrap CI) against references read at the model's
    realised FAR on their FROC envelopes: R and B-min (`refs`, their OOF rows), the
    globally shifted model scores ("shift") and the matched-rate random alarm ("random");
    plus the always-on point ("AO", at its own FAR). Every reference recall and every
    model-minus-reference delta uses the same class resamples (paired). `s3_pass` (model
    rows): the Recall CI lower bound exceeds both the shift and R recall at matched FAR.
    Rows follow the `ev` schema (aggregate only)."""
    keys = ["class_id", "t_s"]
    pcols = [c for c in ("in_G", "next_bid_s") if c in points]
    P = points[keys + pcols].drop_duplicates(keys)

    def prep(df):
        df = df[~_bool_col(df, "inner")] if "inner" in df else df
        add = [c for c in pcols if c not in df]
        df = df.merge(P[keys + add], on=keys, how="left") if add else df
        return df.sort_values(keys, kind="mergesort").reset_index(drop=True)

    te = prep(model)
    inner_rows = model[_bool_col(model, "inner")] if "inner" in model else model.iloc[:0]
    if len(inner_rows):
        add = [c for c in pcols if c not in inner_rows]
        inner_rows = inner_rows.merge(P[keys + add], on=keys, how="left") if add else inner_rows
    if onsets is None and "next_bid_s" in points:
        onsets = points[["class_id", "next_bid_s"]].dropna().drop_duplicates().rename(
            columns={"next_bid_s": "onset_s"})
    o = onsets if onsets is not None else pd.DataFrame({"class_id": [], "onset_s": []})
    cls, t = te["class_id"].to_numpy(), te["t_s"].to_numpy(float)
    data = EventData(t, cls, in_grid=_bool_col(te, "in_G"), y=te["y"].to_numpy(float),
                     onsets=o["onset_s"].to_numpy(float), onset_groups=o["class_id"].to_numpy(),
                     H=H, g=g)
    p = te[score_col].to_numpy(float)
    classes = data.classes
    W = bootstrap_weights(len(classes), n_boot, seed)
    taus, method = fold_thresholds(te, inner_rows, train_scores, H=H, g=g,
                                   score_col=score_col, far_budgets=far_budgets,
                                   n_thresholds=n_thresholds)
    n_onsets = int(data.onset_eval.sum())
    base = {"condition": condition, "n_classes": len(classes), "n_onsets": n_onsets}
    rows = []
    ao = data.stats(np.ones(data.n_points), 0.5).summary()
    if not taus:
        return pd.DataFrame([{**base, "reference": "AO", "budget": np.nan, "far": ao["far"],
                              "recall": ao["recall"], "tiw": ao["tiw"], "tau_method": method}])
    ref_counts = {}
    for name, df in refs.items():
        r = prep(df)
        r = r[np.isin(r["class_id"].to_numpy(), classes)]
        if not len(r):
            continue
        rd = EventData(r["t_s"].to_numpy(float), r["class_id"].to_numpy(),
                       in_grid=_bool_col(r, "in_G"), y=r["y"].to_numpy(float),
                       onsets=o["onset_s"].to_numpy(float),
                       onset_groups=o["class_id"].to_numpy(), H=H, g=g)
        missing = np.setdiff1d(classes, rd.classes)
        if len(missing):                   # a reference must cover every model class
            continue
        ref_counts[name] = FrocCounts(rd, r[score_col].to_numpy(float),
                                      n_thresholds=n_thresholds, classes=classes,
                                      far_stop=2 * max(far_budgets) + 1.0)
    # globally shifted model scores (same shifts as `shift_reference`), classes the shift
    # skips (< 400 G points) left out of both sides of the shift comparison
    rng = np.random.default_rng(seed)
    shifted = []
    sel, bounds = _shift_plan(data.cls, None, data.in_grid)
    ps0 = data._sorted(p, "p")
    keep = np.ones(len(classes), bool)
    for _ in range(n_rep):
        sh, kept = _roll_global(ps0, sel, bounds, rng, config.SHIFT_GLOBAL_MIN,
                                config.SHIFT_GLOBAL_MIN_N)
        keep &= np.isin(np.arange(len(classes)), np.unique(data.cls[kept]))
        unsorted = np.empty(data.n_points)
        unsorted[data.order] = sh
        shifted.append(FrocCounts(data, np.nan_to_num(unsorted, nan=-np.inf),
                                  n_thresholds=n_thresholds,
                                  far_stop=2 * max(far_budgets) + 1.0))
    for budget, tau in taus.items():
        if not np.isfinite(tau).all():
            continue
        st = data.stats(p, tau)
        summ = st.summary()
        eb = event_boot(st, W)
        far_b = eb["far"]
        m_row = {**base, "reference": "model", "budget": float(budget), "far": summ["far"],
                 "recall": summ["recall"], "tiw": summ["tiw"], "lead_med_s": summ["lead_med_s"],
                 **_ci(eb["recall"], "recall")}
        ref_rec = {}
        for name, fc in ref_counts.items():
            point = float(fc.recall_at(summ["far"], np.ones((1, len(classes))))[0])
            samples = fc.recall_at(far_b, W)
            ref_rec[name] = point
            rows.append({**base, "reference": name, "budget": float(budget), "far": summ["far"],
                         "recall": point, **_ci(samples, "recall"),
                         **_delta(summ["recall"] - point, eb["recall"] - samples)})
        # shift reference: model and shifted curves on the classes the shift kept
        wk = W * keep[None, :]
        m_keep = event_boot(st, wk)
        sh_point = float(np.mean([fc.recall_at(st.summary(keep.astype(float))["far"],
                                               keep[None, :].astype(float))[0]
                                  for fc in shifted]))
        sh_samples = np.mean([fc.recall_at(m_keep["far"], wk) for fc in shifted], axis=0)
        model_keep = st.summary(keep.astype(float))["recall"]
        ref_rec["shift"] = sh_point
        rows.append({**base, "reference": "shift", "budget": float(budget),
                     "far": st.summary(keep.astype(float))["far"], "recall": sh_point,
                     **_ci(sh_samples, "recall"),
                     **_delta(model_keep - sh_point, m_keep["recall"] - sh_samples)})
        rnd = random_alarm(data, summ["rate"], n_rep=n_rep, seed=seed)
        rows.append({**base, "reference": "random", "budget": float(budget), "far": rnd["far"],
                     "recall": rnd["recall"],
                     **_delta(summ["recall"] - rnd["recall"], np.full(len(W), np.nan))})
        s3 = all(np.isfinite(ref_rec.get(r, np.nan)) and m_row["recall_lo"] > ref_rec[r]
                 for r in ("shift", "R"))
        rows.append({**m_row, "s3_pass": bool(s3)})
    rows.append({**base, "reference": "AO", "budget": np.nan, "far": ao["far"],
                 "recall": ao["recall"], "tiw": ao["tiw"]})
    out = pd.DataFrame(rows)
    for c in ("far", "recall", "recall_lo", "recall_hi", "tiw", "lead_med_s", "delta",
              "delta_lo", "delta_hi", "delta_p_le0"):
        out[c] = out[c].astype(float) if c in out else np.nan
    out["s3_pass"] = pd.array(out["s3_pass"] if "s3_pass" in out else [pd.NA] * len(out),
                              dtype="boolean")
    out["tau_method"] = method
    return out


def _ci(samples, name: str) -> dict:
    c = _boot_cols(name, samples)
    return {"recall_lo": c[f"{name}_lo"], "recall_hi": c[f"{name}_hi"]}


def _delta(point: float, samples: np.ndarray) -> dict:
    s = np.asarray(samples, float)
    ok = s[np.isfinite(s)]
    c = _boot_cols("d", s)
    return {"delta": float(point), "delta_lo": c["d_lo"], "delta_hi": c["d_hi"],
            "delta_p_le0": float(np.mean(ok <= 0)) if ok.size else float("nan")}


# --------------------------------------------------------------------------- inference (§8.5)


def bootstrap_weights(n_clusters: int, n_boot: int = config.N_BOOT,
                      seed: int = config.SEED) -> np.ndarray:
    """(n_boot, n_clusters) multiplicities of classes drawn with replacement."""
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_clusters, size=(n_boot, n_clusters))
    W = np.zeros((n_boot, n_clusters), dtype=np.int32)
    np.add.at(W, (np.repeat(np.arange(n_boot), n_clusters), draws.ravel()), 1)
    return W


@dataclass
class BootResult:
    """A percentile cluster-bootstrap result (conditional on the fitted fold models)."""

    point: float
    samples: np.ndarray
    lo: float
    hi: float
    se: float
    ci: float
    n_boot: int
    seed: int
    n_nan: int

    @property
    def p_le0(self) -> float:
        """P(stat <= 0) over resamples (one-sided bootstrap p for a positive delta)."""
        s = self.samples[~np.isnan(self.samples)]
        return float(np.mean(s <= 0)) if s.size else float("nan")

    @property
    def p_two_sided(self) -> float:
        s = self.samples[~np.isnan(self.samples)]
        if not s.size:
            return float("nan")
        return float(min(1.0, 2 * min(np.mean(s <= 0), np.mean(s >= 0))))

    @property
    def mde(self) -> float:
        return mde(self.se)

    def as_dict(self) -> dict:
        return {"point": self.point, "lo": self.lo, "hi": self.hi, "se": self.se,
                "p_le0": self.p_le0, "p_two_sided": self.p_two_sided, "mde": self.mde,
                "n_boot": self.n_boot, "n_nan": self.n_nan}


def _n_clusters(stat, n_clusters):
    if n_clusters is not None:
        return int(n_clusters)
    if hasattr(stat, "classes"):
        return len(stat.classes)
    raise ValueError("n_clusters is required for a stat without `.classes`")


def _result(point, samples, ci, n_boot, seed) -> BootResult:
    samples = np.asarray(samples, float)
    ok = samples[~np.isnan(samples)]
    q = 50.0 * (1 - ci), 50.0 * (1 + ci)
    lo, hi = (np.percentile(ok, q) if ok.size else (np.nan, np.nan))
    se = float(np.std(ok, ddof=1)) if ok.size > 1 else float("nan")
    return BootResult(point=float(point), samples=samples, lo=float(lo), hi=float(hi), se=se,
                      ci=ci, n_boot=n_boot, seed=seed, n_nan=int(np.isnan(samples).sum()))


def cluster_bootstrap(stat: Callable, n_clusters: int | None = None, *,
                      n_boot: int = config.N_BOOT, seed: int = config.SEED, ci: float = 0.95,
                      weights: np.ndarray | None = None) -> BootResult:
    """Cluster bootstrap over classes (B = 2000, percentile CI).

    `stat(w)` takes a class-weight vector (multiplicities aligned with `stat.classes`) and
    returns a float; `stat(None)` / unit weights give the point estimate. Pass precomputed
    `weights` (from `bootstrap_weights`) to share draws across statistics.
    """
    n = _n_clusters(stat, n_clusters)
    W = bootstrap_weights(n, n_boot, seed) if weights is None else np.asarray(weights)
    samples = _evaluate(stat, W)
    return _result(stat(np.ones(n)), samples, ci, len(W), seed)


def _evaluate(stat: Callable, W: np.ndarray) -> np.ndarray:
    """stat(w) for every weight row (vectorised when the statistic has `batch`)."""
    if hasattr(stat, "batch"):
        return np.asarray(stat.batch(W), float)
    return np.array([stat(w) for w in W], float)


def paired_delta(stat_a: Callable, stat_b: Callable, n_clusters: int | None = None, *,
                 n_boot: int = config.N_BOOT, seed: int = config.SEED, ci: float = 0.95,
                 weights: np.ndarray | None = None) -> BootResult:
    """Paired bootstrap of stat_a - stat_b on the *same* resampled classes. Report the CI,
    `p_le0` = P(delta <= 0) and the MDE."""
    ca, cb = getattr(stat_a, "classes", None), getattr(stat_b, "classes", None)
    if ca is not None and cb is not None and not np.array_equal(ca, cb):
        raise ValueError("paired_delta: the two statistics cover different classes")
    n = _n_clusters(stat_a if ca is not None else stat_b, n_clusters)
    W = bootstrap_weights(n, n_boot, seed) if weights is None else np.asarray(weights)
    d = _evaluate(stat_a, W) - _evaluate(stat_b, W)
    ones = np.ones(n)
    return _result(stat_a(ones) - stat_b(ones), d, ci, len(W), seed)


class ClassResampled:
    """Wrap any fn(*arrays) -> float so it can be bootstrapped by class: the arrays are
    physically resampled (class c repeated w_c times). Slower than the weighted stats."""

    def __init__(self, fn: Callable, groups, *arrays, classes=None):
        self.fn = fn
        g = np.asarray(groups)
        self.classes = _classes(g, classes)
        cls = _class_index(self.classes, g)
        self._order = np.argsort(cls, kind="stable")
        self._bounds = np.searchsorted(cls[self._order], np.arange(len(self.classes) + 1))
        self.arrays = [np.asarray(a) for a in arrays]

    def __call__(self, w=None) -> float:
        if w is None:
            return float(self.fn(*self.arrays))
        wc = np.round(_weights(w, len(self.classes))).astype(int)
        parts = [self._order[self._bounds[c]:self._bounds[c + 1]]
                 for c in np.flatnonzero(wc) for _ in range(wc[c])]
        idx = np.concatenate(parts) if parts else np.zeros(0, int)
        return float(self.fn(*[a[idx] for a in self.arrays]))


def mde(se: float, *, alpha: float = config.ALPHA, power: float = 0.80) -> float:
    """Minimum detectable effect (z_{1-alpha} + z_power) * SE; = 2.49 SE at 0.05 / 80%."""
    z = NormalDist().inv_cdf
    return (z(1 - alpha) + z(power)) * float(se)


def iut_p(*p_values: float) -> float:
    """Intersection-union test p-value (every component must reject): the largest p."""
    return float(max(p_values))


def holm(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values."""
    names = list(p_values)
    ps = np.array([float(p_values[k]) for k in names])
    o = np.argsort(ps, kind="mergesort")
    m = len(ps)
    adj = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * ps[o]))
    out = np.empty(m)
    out[o] = adj
    return {k: float(v) for k, v in zip(names, out, strict=True)}


def fixed_sequence_holm(p_values: Mapping[str, float | None], *, primary: str = "H1",
                        alpha: float = config.ALPHA, strict: bool = False) -> pd.DataFrame:
    """Fixed-sequence gatekeeping (§7): the primary at alpha; only if it rejects are the
    others tested, Holm-corrected at alpha. A None/NaN p (e.g. H3 underpowered) leaves the
    family. `p_adj` = max(p_primary, Holm p): reject iff p_adj <= alpha (`strict`: < alpha;
    with one-sided bootstrap p = P(delta* <= 0) and alpha = ALPHA_ONE_SIDED this rejects
    exactly when the 95% percentile CI lies above 0)."""
    if primary not in p_values or p_values[primary] is None:
        raise ValueError(f"the primary hypothesis {primary!r} needs a p-value")

    def ok(x: float) -> bool:
        return x < alpha if strict else x <= alpha
    p1 = float(p_values[primary])
    gate = ok(p1)
    family = {k: float(v) for k, v in p_values.items()
              if k != primary and v is not None and not math.isnan(float(v))}
    adj = holm(family) if family else {}
    rows = [{"hypothesis": primary, "p": p1, "in_family": True, "tested": True,
             "p_holm": p1, "p_adj": p1, "reject": gate}]
    for k, v in p_values.items():
        if k == primary:
            continue
        if k in family:
            p_adj = max(p1, adj[k])
            rows.append({"hypothesis": k, "p": family[k], "in_family": True, "tested": gate,
                         "p_holm": adj[k], "p_adj": p_adj, "reject": bool(gate and ok(p_adj))})
        else:
            rows.append({"hypothesis": k, "p": float("nan") if v is None else float(v),
                         "in_family": False, "tested": False, "p_holm": float("nan"),
                         "p_adj": float("nan"), "reject": False})
    return pd.DataFrame(rows).set_index("hypothesis")
