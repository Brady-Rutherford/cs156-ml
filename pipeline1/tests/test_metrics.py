"""Tests for oncue.metrics (spec §8.2-§8.5, A9). Synthetic grids and scores only.

Includes the spec's five `test_metrics.py` checks (§8.3) and the critique blocker: an
always-on or constant-score predictor must not reach high recall at a low false-firing rate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from oncue import config, schemas
from oncue import metrics as M
from sklearn.metrics import average_precision_score

H, G, D = 15.0, 2.0, 2.0
T0 = 120.0


# --------------------------------------------------------------------------- helpers


def grid(n: int, t0: float = T0) -> np.ndarray:
    return t0 + D * np.arange(n)


def labels(t: np.ndarray, onsets, *, guard: bool = True, H: float = H, g: float = G
           ) -> np.ndarray:
    """§3 label rule (all observed): 1 if the first onset in (t, t + H] lies in (t + g, t + H],
    0 if there is none, undefined (NaN) if an onset falls in the guard (t, t + g].
    `guard=False` gives fully defined labels (1 iff an onset in (t + g, t + H])."""
    o = np.sort(np.asarray(onsets, float))
    in_guard = np.searchsorted(o, t + g, "right") - np.searchsorted(o, t, "right")
    in_occ = np.searchsorted(o, t + H, "right") - np.searchsorted(o, t + g, "right")
    if not guard:
        return (in_occ > 0).astype(float)
    return np.where(in_guard > 0, np.nan, (in_occ > 0).astype(float))


def world(n_classes: int, n_points: int, n_onsets: int, rng, *, guard: bool = True):
    """Concatenated contiguous grids with random onsets: (t, cls, y, onsets, onset_cls)."""
    ts, cs, ys, os_, ocs = [], [], [], [], []
    for c in range(n_classes):
        t = grid(n_points)
        o = np.sort(rng.uniform(t[0] + H, t[-1] + D, n_onsets))
        ts.append(t)
        cs.append(np.full(n_points, c))
        ys.append(labels(t, o, guard=guard))
        os_.append(o)
        ocs.append(np.full(n_onsets, c))
    return (np.concatenate(ts), np.concatenate(cs), np.concatenate(ys), np.concatenate(os_),
            np.concatenate(ocs))


def oracle_scores(t, cls, onsets, onset_cls) -> np.ndarray:
    """p = 1 exactly on [o - H, o - g) of every onset of the point's class, else 0."""
    p = np.zeros(t.size)
    for o, c in zip(onsets, onset_cls, strict=True):
        p[(cls == c) & (t >= o - H) & (t < o - G)] = 1.0
    return p


# --------------------------------------------------------------------------- point metrics


def test_ap_matches_sklearn_with_ties_and_weights():
    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.1).astype(float)
    p = np.round(rng.random(500), 1)                     # heavy ties
    assert M.ap(y, p) == pytest.approx(average_precision_score(y, p), abs=1e-12)
    w = rng.integers(1, 4, 500).astype(float)
    assert M.ap(y, p, sample_weight=w) == pytest.approx(
        average_precision_score(y, p, sample_weight=w), abs=1e-12)


def test_ap_drops_undefined_and_is_nan_without_positives():
    y = np.array([1.0, np.nan, 0.0, 0.0])
    p = np.array([0.9, 0.99, 0.1, 0.2])
    assert M.ap(y, p) == pytest.approx(1.0)
    assert np.isnan(M.ap([0.0, 0.0], [0.3, 0.4]))
    with pytest.raises(ValueError):
        M.ap([1.0, 0.0], [np.nan, 0.2])
    assert M.lift([1, 0, 0, 0], [0.9, 0.1, 0.2, 0.3]) == pytest.approx(4.0)


def _three_classes():
    # A: AP = 0.5 * 1 + 0.5 * 0.5 = 0.75; B: AP = 0.5; C: no positive (skipped)
    y = np.array([1, 0, 0, 1, 0, 1, 0, 0], float)
    p = np.array([0.9, 0.8, 0.7, 0.1, 0.9, 0.1, 0.5, 0.4])
    g = np.array(["A", "A", "A", "A", "B", "B", "C", "C"])
    return y, p, g


def test_ap_within_known_answer():
    y, p, g = _three_classes()
    per = M.per_class_ap(y, p, g)
    assert per["A"] == pytest.approx(0.75) and per["B"] == pytest.approx(0.5)
    assert np.isnan(per["C"])
    k = pd.Series({"A": 2, "B": 1, "C": 0})
    assert M.ap_within(y, p, g, weights=k) == pytest.approx(2 / 3)
    # default K_c = positive points per class (A has 2, B has 1): same answer here
    assert M.ap_within(y, p, g) == pytest.approx(2 / 3)
    assert M.ap_within(y, p, g, weights={"A": 1, "B": 3}) == pytest.approx(
        (0.75 + 3 * 0.5) / 4)
    assert M.macro_ap(y, p, g) == pytest.approx(0.625)


def test_ap_within_ignores_cross_class_score_scale():
    y, p, g = _three_classes()
    p2 = p.copy()
    p2[g == "B"] = p2[g == "B"] * 1e-3            # class B scored by a different fold model
    p2[g == "A"] = p2[g == "A"] + 5.0
    assert M.ap_within(y, p2, g) == pytest.approx(M.ap_within(y, p, g))
    assert M.ap(y, p2) != pytest.approx(M.ap(y, p))   # pooled AP does change


def test_bid_counts_distinct_onsets_of_positives():
    y = np.array([1, 1, 0, 1, np.nan, 1])
    g = np.array([1, 1, 1, 1, 2, 2])
    o = np.array([200.0, 200.0, 260.0, 300.0, 400.0, 500.0])
    k = M.bid_counts(y, g, o)
    assert k.to_dict() == {1: 2, 2: 1}


def test_ece_equal_mass_bins_known_values():
    assert M.ece([0, 0, 1, 1], [0.1, 0.1, 0.9, 0.9], n_bins=2) == pytest.approx(0.1)
    # a constant forecast equal to the base rate is perfectly calibrated (ties never split)
    y = np.r_[np.ones(20), np.zeros(80)]
    np.random.default_rng(1).shuffle(y)
    assert M.ece(y, np.full(100, 0.2)) == pytest.approx(0.0, abs=1e-12)
    rel = M.reliability(y, np.full(100, 0.2))
    assert len(rel) == 1 and rel["n"].iloc[0] == 100
    rng = np.random.default_rng(2)
    p = rng.random(200_000) * 0.2
    yy = (rng.random(p.size) < p).astype(float)
    assert M.ece(yy, p) < 0.005
    r = M.reliability(yy, p, n_bins=15)
    assert len(r) == 15 and abs(r["n"].max() - r["n"].min()) <= 1


def test_brier_and_brier_skill():
    y = np.array([1, 0, 0, 0], float)
    assert M.brier(y, [1, 0, 0, 0]) == 0.0
    assert M.brier(y, [0.5] * 4) == pytest.approx(0.25)
    assert M.brier_skill(y, [1, 0, 0, 0]) == pytest.approx(1.0)
    assert M.brier_skill(y, [0.25] * 4) == pytest.approx(0.0)       # climatology
    assert M.brier_skill(y, [0.25] * 4, ref=0.5) == pytest.approx(1 - 0.1875 / 0.25)
    assert M.brier_skill(y, [0.25] * 4, ref=np.array([0.5] * 4)) == pytest.approx(0.25)
    assert M.brier_skill(y, [0.9, 0.9, 0.9, 0.9]) < 0


def test_fold_ap_and_point_metrics():
    rng = np.random.default_rng(3)
    y = (rng.random(1000) < 0.1).astype(float)
    p = y * 0.5 + rng.random(1000)
    folds = np.repeat(np.arange(5), 200)
    mean, sd, per = M.fold_ap(y, p, folds)
    assert len(per) == 5 and mean == pytest.approx(per.mean()) and sd > 0
    row = M.point_metrics(y, p, np.repeat(np.arange(10), 100))
    for key in ("n_points", "n_classes", "base_rate", "ap", "ap_lift", "ap_within",
                "macro_ap", "roc_auc", "brier", "brier_skill", "ece"):
        assert key in row
    assert row["n_classes"] == 10 and row["ap_lift"] > 2


# --------------------------------------------------------------------------- §8.3 checks


def test_always_on_fires_every_14s_and_far_formula():
    """§8.3 test 1: p = 1 fires every Δ⌈(H - g)/Δ⌉ = 14 s; FAR = (60/14) x false share."""
    rng = np.random.default_rng(4)
    t = grid(1792)                                      # |G| = 7 x 256: exactly 256 firings
    o = np.sort(rng.uniform(t[0] + H, t[-1], 12))
    y = labels(t, o, guard=False)                       # G_def = G
    d = M.EventData(t, y=y, onsets=o)
    fired = d.firings(np.ones(t.size), 1.0)
    assert np.all(np.diff(t[fired]) == 14.0)
    s = d.stats(np.ones(t.size), 1.0).summary()
    assert s["n_fire"] == 256 and s["n_neutral"] == 0
    assert s["far"] == pytest.approx(60 / 14 * s["n_false"] / s["n_fire"], abs=1e-9)
    assert s["rate"] == pytest.approx(60 / 14, abs=1e-9)
    assert s["tiw"] == pytest.approx(13 / 14, abs=0.002)        # time-weighted
    assert s["tiw_points"] == pytest.approx(6 / 7, abs=0.002)   # literal point count


def test_always_on_recall_bounded_by_tiw_over_500_placements():
    """§8.3 test 1: recall <= TiW + 0.02 over 500 random onset placements."""
    rng = np.random.default_rng(5)
    t = grid(1800)
    p = np.ones(t.size)
    recalls, tiws = [], []
    for _ in range(500):
        o = rng.uniform(t[0] + H, t[-1] + D, 8)
        d = M.EventData(t, y=labels(t, o), onsets=o)
        s = d.stats(p, 1.0).summary()
        recalls.append(s["recall"])
        tiws.append(s["tiw"])
    assert np.ptp(tiws) == 0                              # TiW never depends on onsets
    assert np.mean(recalls) <= tiws[0] + 0.02
    assert np.mean(recalls) == pytest.approx(13 / 14, abs=0.02)


def test_always_on_eventlift_is_one():
    """§8.3 test 1: EventLift_0 in [0.9, 1.1]; EventLift vs shifted predictions = 1."""
    rng = np.random.default_rng(6)
    t, cls, y, o, oc = world(3, 1800, 10, rng)
    d = M.EventData(t, cls, y=y, onsets=o, onset_groups=oc)
    p = np.ones(t.size)
    s = d.stats(p, 1.0).summary()
    assert 0.9 <= s["eventlift0"] <= 1.1
    ev = M.eventlift_shift(d, p, 1.0, n_rep=3)
    assert ev["n_classes_skipped"] == 0
    assert ev["eventlift"] == pytest.approx(1.0, abs=1e-12)


def test_firings_never_read_labels():
    """§8.3 test 2: F_tau is identical when labels (and onsets) are recomputed."""
    rng = np.random.default_rng(7)
    t, cls, y, o, oc = world(4, 600, 6, rng)
    p = rng.random(t.size)
    in_g = rng.random(t.size) > 0.2
    d1 = M.EventData(t, cls, in_grid=in_g, y=y, onsets=o, onset_groups=oc)
    o2 = rng.uniform(T0, t.max(), o.size)
    y2 = np.concatenate([labels(t[cls == c], o2[oc == c]) for c in range(4)])
    d2 = M.EventData(t, cls, in_grid=in_g, y=y2, onsets=o2, onset_groups=oc)
    d3 = M.EventData(t, cls, in_grid=in_g)                            # no labels at all
    f1 = d1.firings(p, 0.8)
    assert np.array_equal(f1, d2.firings(p, 0.8)) and np.array_equal(f1, d3.firings(p, 0.8))
    assert np.array_equal(f1, M.firings(p, t, 0.8, H, G, in_grid=in_g, groups=cls))
    assert not (f1 & ~in_g).any()                       # only causal-grid points fire


def test_random_scores_eventlift0_near_one():
    """§8.3 test 3: random p independent of onsets, 200 classes: mean EventLift_0 ~ 1."""
    rng = np.random.default_rng(8)
    t, cls, y, o, oc = world(200, 1800, 10, rng)
    d = M.EventData(t, cls, y=y, onsets=o, onset_groups=oc)
    st = d.stats(rng.random(t.size), 0.9)
    pooled = st.summary()
    assert 0.9 <= pooled["eventlift0"] <= 1.1
    per = st.per_class()
    rate_c = per["n_fire"] / (per["n_grid"] * D / 60)
    lift_c = (per["n_hit"] / per["n_eval"]) / M.r0(rate_c.to_numpy(), H, G)
    assert 0.9 <= float(np.nanmean(lift_c)) <= 1.1


def test_oracle_recall_one_far_zero():
    """§8.3 test 4: p = 1 exactly on [o - H, o - g): recall 1, FAR 0."""
    rng = np.random.default_rng(9)
    t = grid(1800)
    o = T0 + 60 + np.cumsum(rng.uniform(40, 120, 20))     # well separated onsets
    o = o[o < t[-1]]
    y = labels(t, o)
    d = M.EventData(t, y=y, onsets=o)
    p = oracle_scores(t, np.zeros(t.size), o, np.zeros(o.size))
    s = d.stats(p, 1.0).summary()
    assert s["recall"] == 1.0 and s["far"] == 0.0
    assert s["n_true"] == len(o) and s["n_false"] == 0 and s["n_eval"] == len(o)
    assert np.all((d.stats(p, 1.0).lead_s > G) & (d.stats(p, 1.0).lead_s <= H))


def test_guard_point_fires_and_counts_neutral():
    """§8.3 test 5: label-side rules never remove points from G; a guard point fires."""
    t = grid(100)
    o = np.array([161.0])                                  # guard point: t = 160
    y = labels(t, o)
    assert np.isnan(y[t == 160.0][0])
    p = np.where(t == 160.0, 1.0, 0.0)
    d = M.EventData(t, y=y, onsets=o)
    s = d.stats(p, 0.5).summary()
    assert s["n_fire"] == 1 and s["n_neutral"] == 1
    assert s["n_true"] == 0 and s["n_false"] == 0 and s["far"] == 0.0
    assert s["recall"] == 0.0                              # firing is after o - g


# --------------------------------------------------------------------------- critique blocker


@pytest.mark.parametrize("const", [1.0, 0.37, 0.0])
def test_constant_predictor_gets_no_recall_at_low_far(const):
    """Blocker (§16 #1, #38): an always-on / constant-score predictor must not achieve high
    recall at a low false-firing budget. Its only non-trivial operating point fires every
    14 s (~4.3/min), far above both budgets; below that it fires nothing."""
    rng = np.random.default_rng(10)
    t, cls, y, o, oc = world(20, 1800, 6, rng)
    d = M.EventData(t, cls, y=y, onsets=o, onset_groups=oc)
    p = np.full(t.size, const)
    curve = M.froc(d, p)
    assert len(curve) == 2                                       # tau = inf, tau = const
    on = curve.iloc[-1]
    assert on["far"] > 3.0 and on["rate"] == pytest.approx(60 / 14, rel=0.01)
    assert 0.9 <= on["eventlift0"] <= 1.1                         # recall is just TiW
    for budget in config.FAR_BUDGETS:
        assert M.recall_at_far(curve, budget) == 0.0
        # even randomising between "never" and "always" only buys chance-level recall
        mixed = M.recall_at_far(curve, budget, interpolate=True)
        assert mixed <= 1.1 * M.r0(budget / on["far"] * on["rate"], H, G) + 1e-9
    assert M.pafroc(curve) == 0.0
    # rate-matched thresholds from inner OOF (other classes) give it nothing either
    t2, c2, y2, o2, oc2 = world(10, 1800, 6, np.random.default_rng(11))
    inner = M.EventData(t2, c2, y=y2, onsets=o2, onset_groups=oc2)
    for budget in config.FAR_BUDGETS:
        tm = M.tau_rate_matched(inner, np.full(t2.size, const), d, p, budget)
        s = d.stats(p, tm["tau"]).summary()
        assert s["recall"] == 0.0 and s["far"] == 0.0 and np.isinf(tm["tau"])


def test_oracle_contrast_reaches_full_recall_at_zero_far():
    rng = np.random.default_rng(12)
    t, cls, y, o, oc = world(5, 1800, 1, rng)
    d = M.EventData(t, cls, y=y, onsets=o, onset_groups=oc)
    curve = M.froc(d, oracle_scores(t, cls, o, oc))
    assert M.recall_at_far(curve, 0.1) == 1.0
    assert M.pafroc(curve) == pytest.approx(1.0)


# --------------------------------------------------------------------------- event details


def test_refractory_resets_per_class_and_input_order_is_irrelevant():
    t = np.r_[grid(50), grid(50)]
    cls = np.r_[np.zeros(50), np.ones(50)]
    p = np.where(t == T0, 1.0, 0.0)
    assert M.firings(p, t, 0.5, groups=cls).sum() == 2           # same t, two classes
    rng = np.random.default_rng(13)
    p = rng.random(t.size)
    f = M.firings(p, t, 0.6, groups=cls)
    perm = rng.permutation(t.size)
    fp = M.firings(p[perm], t[perm], 0.6, groups=cls[perm])
    assert np.array_equal(fp, f[perm])


def test_refractory_open_interval():
    """No earlier firing in (s - (H - g), s): exactly H - g apart may both fire."""
    t = np.array([0.0, 12.0, 13.0, 26.0, 38.0])
    f = M.firings(np.ones(5), t, 0.5)
    assert t[f].tolist() == [0.0, 13.0, 26.0]


def test_per_point_tau_for_fold_thresholds():
    t = np.r_[grid(70), grid(70)]
    cls = np.r_[np.zeros(70), np.ones(70)]
    p = np.where(cls == 0, 0.6, 0.3)
    tau = np.where(cls == 0, 0.5, 0.2)
    f = M.firings(p, t, tau, groups=cls)
    assert f[cls == 0].sum() == 10 and f[cls == 1].sum() == 10
    assert M.firings(p, t, 0.5, groups=cls)[cls == 1].sum() == 0


def test_evaluable_bids_need_half_the_window_in_G():
    t = grid(200)
    in_g = ~((t >= 300) & (t < 400))
    o = np.array([380.0, 306.0, 309.0, 250.0])
    d = M.EventData(t, in_grid=in_g, onsets=o)
    # 380: window [365, 378) has no G point; 306: 4 of 6 slots in G; 309: 3 of 7
    assert d.onset_eval.tolist() == [False, True, False, True]
    # only G points supplied: same answer (slots are counted arithmetically)
    d2 = M.EventData(t[in_g], onsets=o)
    assert d2.onset_eval.tolist() == [False, True, False, True]


def test_lead_and_window_boundaries():
    t = grid(40, t0=100.0)
    p = (t >= 122) & (t < 135)
    f = M.firings(p.astype(float), t, 0.5)
    assert t[f].tolist() == [122.0]
    assert M.lead(f, t, [137.0]).tolist() == [15.0]         # firing at o - H counts
    assert M.recall(f, t, [137.0]) == 1.0
    assert M.recall(f, t, [124.0]) == 0.0                   # firing at o - g does not
    f2 = M.firings(((t >= 124) & (t < 137)).astype(float), t, 0.5)
    assert M.lead(f2, t, [138.5]).tolist() == [14.5]


def test_standalone_wrappers_match_eventdata():
    rng = np.random.default_rng(14)
    t, cls, y, o, oc = world(3, 900, 5, rng)
    in_g = rng.random(t.size) > 0.1
    p = rng.random(t.size)
    d = M.EventData(t, cls, in_grid=in_g, y=y, onsets=o, onset_groups=oc)
    f = d.firings(p, 0.85)
    s = d.stats(p, 0.85).summary()
    assert M.far(f, y, in_g) == pytest.approx(s["far"])
    assert M.firing_rate(f, in_g) == pytest.approx(s["rate"])
    kw = {"groups": cls, "onset_groups": oc, "in_grid": in_g}
    assert M.recall(f, t, o, **kw) == pytest.approx(s["recall"])
    assert np.median(M.lead(f, t, o, **kw)) == pytest.approx(s["lead_med_s"])
    assert M.time_in_warning(f, t, groups=cls, in_grid=in_g) == pytest.approx(s["tiw"])
    assert M.time_in_warning(f, t, groups=cls, in_grid=in_g, mode="points") == \
        pytest.approx(s["tiw_points"])


def test_r0_values_from_spec():
    assert M.r0(0.1) == pytest.approx(0.0217, abs=5e-4)
    assert M.r0(0.5) == pytest.approx(0.108, abs=5e-4)
    assert M.r0(10.0) == 1.0
    assert abs(M.r0_poisson(0.5) - M.r0(0.5)) / M.r0(0.5) < 0.06


def test_froc_envelope_recall_at_far_and_pafroc():
    curve = pd.DataFrame({"tau": [np.inf, 0.9, 0.8, 0.7, 0.6],
                          "far": [0.0, 0.1, 0.05, 0.3, 0.6],
                          "recall": [0.0, 0.2, 0.3, 0.5, 0.9]})
    env = M.froc_envelope(curve)
    assert env["far"].tolist() == [0.0, 0.05, 0.3, 0.6]     # (0.1, 0.2) is dominated
    assert M.recall_at_far(curve, 0.1) == 0.3
    assert M.recall_at_far(curve, 0.3) == 0.5
    assert M.recall_at_far(curve, 0.2, interpolate=True) == pytest.approx(0.3 + 0.2 * 0.6)
    # step area on [0, 0.5]: 0 * 0.05 + 0.3 * 0.25 + 0.5 * 0.2 = 0.175
    assert M.pafroc(curve, 0.5, normalise=False) == pytest.approx(0.175)
    assert M.pafroc(curve, 0.5) == pytest.approx(0.35)


def test_rate_at_far_first_crossing_interpolates():
    curve = pd.DataFrame({"tau": [np.inf, 0.9, 0.8, 0.7], "far": [0.0, 0.05, 0.15, 0.08],
                          "rate": [0.0, 0.5, 1.0, 1.5]})
    assert M.rate_at_far(curve, 0.1) == pytest.approx(0.75)   # stops at the first crossing
    assert M.rate_at_far(curve, 0.2) == pytest.approx(1.5)


def test_tau_for_rate_and_rate_matching_are_scale_free():
    rng = np.random.default_rng(15)
    t, cls, y, o, oc = world(40, 1500, 6, rng)
    d = M.EventData(t, cls, y=y, onsets=o, onset_groups=oc)
    signal = oracle_scores(t, cls, o, oc)
    p_inner = signal * 0.3 + rng.random(t.size)
    rates = [d.rate(p_inner, tau) for tau in np.linspace(0.2, 1.3, 12)]
    assert np.all(np.diff(rates) <= 1e-12)                    # monotone in tau
    target = 0.4
    tau = M.tau_for_rate(d, p_inner, target)
    step = 1.0 / (d.n_grid.sum() * D / 60)                    # one firing
    assert abs(d.rate(p_inner, tau) - target) <= step + 1e-12
    # the outer model scores on another scale (monotone transform): same firings
    p_train = 10 * p_inner ** 3 - 4
    tm = M.tau_rate_matched(d, p_inner, d, p_train, 0.1)
    assert abs(tm["rate_train"] - tm["rate_star"]) <= step + 1e-12
    assert d.stats(p_train, tm["tau"]).summary()["far"] == pytest.approx(0.1, abs=0.02)
    assert np.isinf(M.tau_for_rate(d, p_inner, 0.0))


def test_random_alarm_matches_r0():
    rng = np.random.default_rng(16)
    t, cls, y, o, oc = world(100, 1800, 10, rng)
    d = M.EventData(t, cls, y=y, onsets=o, onset_groups=oc)
    ra = M.random_alarm(d, 0.5, n_rep=5, seed=1)
    assert ra["rate"] == pytest.approx(0.5, abs=0.01)
    assert ra["recall"] == pytest.approx(M.r0(0.5), abs=0.03)
    assert ra["tiw"] == pytest.approx(M.r0(ra["rate"]), abs=0.005)


def test_bid_onsets_filters_votes_glitches_and_censored():
    eps = pd.DataFrame({
        "class_id": [1, 1, 1, 1, 2], "ep_type": ["bid", "vote", "glitch", "bid", "bid"],
        "onset_s": [200.0, 300.0, 400.0, 500.0, np.nan],
        "onset_retimed_s": [199.6, 300.0, 400.0, 500.2, np.nan],
        "retime_failed": [False, False, False, True, True],
        "left_censored": [False, False, False, False, True]})
    assert M.bid_onsets(eps)["onset_s"].tolist() == [200.0, 500.0]
    assert M.bid_onsets(eps, retimed=True)["onset_s"].tolist() == [199.6, 500.0]


def test_event_metrics_row_keys():
    rng = np.random.default_rng(17)
    t, cls, y, o, oc = world(4, 1800, 5, rng)
    d = M.EventData(t, cls, y=y, onsets=o, onset_groups=oc)
    p = oracle_scores(t, cls, o, oc) * 0.5 + rng.random(t.size) * 0.6
    row = M.event_metrics(d, p, {0.1: 1.0, 0.5: 0.9}, n_rep=2)
    for key in ("recall_far01", "recall_far05", "realised_far01", "realised_far05",
                "tiw_far01", "tiw_far05", "eventlift_shift_far01", "eventlift0_far01",
                "lead_med_s", "pafroc", "recall_random_far01", "recall_random_far05"):
        assert key in row
    assert M.budget_key(0.1) == "far01" and M.budget_key(0.5) == "far05"


# --------------------------------------------------------------------------- shifts (§8.4)


def test_shift_global_is_a_within_class_rotation():
    rng = np.random.default_rng(18)
    g = np.r_[np.zeros(500, int), np.ones(300, int), np.full(450, 2)]
    p = rng.random(g.size)
    ps, kept = M.shift_global(p, g, seed=3)
    assert not kept[g == 1].any() and np.isnan(ps[g == 1]).all()      # n_c < 400: skipped
    for c, n in ((0, 500), (2, 450)):
        orig, sh = p[g == c], ps[g == c]
        us = [u for u in range(n) if np.array_equal(np.roll(orig, u), sh)]
        assert len(us) == 1 and 150 <= us[0] <= n - 150
    ps2, _ = M.shift_global(p, g, seed=3)
    assert np.array_equal(ps, ps2, equal_nan=True)
    ps3, _ = M.shift_global(p, g, seed=4)
    assert not np.array_equal(ps, ps3, equal_nan=True)


def test_shift_local_blocks_within_segments():
    rng = np.random.default_rng(19)
    seg = np.r_[np.full(200, "a"), np.full(20, "b"), np.full(95, "c")].astype(object)
    g = np.zeros(seg.size, int)
    p = rng.random(seg.size)
    ps, shifted = M.shift_local(p, g, seg, seed=5)
    assert not shifted[200:220].any() and np.array_equal(ps[200:220], p[200:220])
    # run "a" (200): blocks [0, 90) and [90, 200) (the 20-point tail merges); run "c": one
    for a, b in ((0, 90), (90, 200), (220, 315)):
        n = b - a
        us = [u for u in range(n) if np.array_equal(np.roll(p[a:b], u), ps[a:b])]
        assert len(us) == 1 and 15 <= us[0] <= n - 15
    assert shifted.mean() == pytest.approx(295 / 315)


def test_timing_gain_constant_per_class_is_zero_and_oracle_positive():
    rng = np.random.default_rng(20)
    t, cls, y, _, _ = world(6, 900, 8, rng, guard=False)
    seg = (t // 300).astype(int)
    const = (cls + 1) / 10.0
    for kind in ("global", "local"):
        tg = M.timing_gain(y, const, cls, t=t, segments=seg, kind=kind, n_rep=3)
        assert tg["tg"] == pytest.approx(0.0, abs=1e-12)
    p = y + 0.5 * rng.random(t.size)
    tg = M.timing_gain(y, p, cls, t=t, segments=seg, kind="global", n_rep=3)
    assert tg["tg"] > 0.3 and tg["n_classes_skipped"] == 0
    tgl = M.TimingGain(y, p, cls, t=t, segments=seg, kind="local", n_rep=3, within=True)
    assert tgl() == pytest.approx(tgl.summary()["tg"]) and tgl() > 0.3
    assert 0 <= tgl.summary()["share_unshifted"] < 0.1
    short = M.timing_gain(y[:300], p[:300], cls[:300], kind="global", n_rep=2)
    assert short["n_classes_skipped"] == 1


# --------------------------------------------------------------------------- inference (§8.5)


def _boot_data(n_classes=30, seed=21):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(n_classes), 120)
    y = (rng.random(g.size) < 0.08).astype(float)
    p = y * rng.random(g.size) + rng.random(g.size)
    return y, p, g


def test_bootstrap_is_reproducible():
    y, p, g = _boot_data()
    stat = M.WithinAP(y, p, g)
    a = M.cluster_bootstrap(stat, n_boot=300, seed=7)
    b = M.cluster_bootstrap(stat, n_boot=300, seed=7)
    c = M.cluster_bootstrap(stat, n_boot=300, seed=8)
    assert np.array_equal(a.samples, b.samples) and (a.lo, a.hi) == (b.lo, b.hi)
    assert not np.array_equal(a.samples, c.samples)
    assert a.lo <= a.point <= a.hi and a.n_boot == 300 and a.n_nan == 0
    W = M.bootstrap_weights(30, 50, seed=1)
    assert W.shape == (50, 30) and np.all(W.sum(axis=1) == 30)
    assert np.array_equal(W, M.bootstrap_weights(30, 50, seed=1))


def test_weighted_stats_equal_materialised_resample():
    y, p, g = _boot_data(12)
    w = M.bootstrap_weights(12, 1, seed=3)[0]
    idx = np.concatenate([np.flatnonzero(g == c) for c in range(12) for _ in range(w[c])])
    new_g = np.concatenate([np.full((g == c).sum(), 100 * c + k) for c in range(12)
                            for k in range(w[c])])
    assert M.PooledAP(y, p, g)(w) == pytest.approx(
        average_precision_score(y[idx], p[idx]), abs=1e-12)
    assert M.WithinAP(y, p, g)(w) == pytest.approx(M.ap_within(y[idx], p[idx], new_g))
    cr = M.ClassResampled(M.brier, g, y, p)
    assert cr(w) == pytest.approx(M.brier(y[idx], p[idx]))
    assert cr() == pytest.approx(M.brier(y, p))


def test_event_stats_weights_equal_materialised_classes():
    rng = np.random.default_rng(22)
    t, cls, y, o, oc = world(4, 900, 6, rng)
    p = rng.random(t.size)
    d = M.EventData(t, cls, y=y, onsets=o, onset_groups=oc)
    w = np.array([2, 0, 1, 3])
    got = d.stats(p, 0.9).summary(w)
    parts = [(c, k) for c in range(4) for k in range(w[c])]
    sel = [np.flatnonzero(cls == c) for c, _ in parts]
    osel = [np.flatnonzero(oc == c) for c, _ in parts]
    d2 = M.EventData(np.concatenate([t[s] for s in sel]),
                     np.concatenate([np.full(s.size, 10 * c + k) for (c, k), s
                                     in zip(parts, sel, strict=True)]),
                     y=np.concatenate([y[s] for s in sel]),
                     onsets=np.concatenate([o[s] for s in osel]),
                     onset_groups=np.concatenate([np.full(s.size, 10 * c + k) for (c, k), s
                                                  in zip(parts, osel, strict=True)]))
    want = d2.stats(np.concatenate([p[s] for s in sel]), 0.9).summary()
    for key in ("far", "rate", "recall", "tiw", "tiw_points", "lead_med_s", "n_eval"):
        assert got[key] == pytest.approx(want[key], abs=1e-12)
    boot = M.cluster_bootstrap(lambda w: d.stats(p, 0.9).summary(w)["recall"], 4,
                               n_boot=50, seed=1)
    assert 0 <= boot.lo <= boot.hi <= 1


def test_paired_delta():
    y, p, g = _boot_data()
    a = M.WithinAP(y, p, g)
    same = M.paired_delta(a, a, n_boot=200, seed=1)
    assert same.point == 0 and same.lo == 0 and same.hi == 0
    rng = np.random.default_rng(4)
    b = M.WithinAP(y, p + 3 * rng.random(p.size), g)          # a noisier model
    d = M.paired_delta(a, b, n_boot=400, seed=1)
    assert d.point > 0 and d.lo > 0 and d.p_le0 < 0.025
    assert d.mde == pytest.approx(2.486 * d.se, rel=1e-3)
    assert set(d.as_dict()) >= {"point", "lo", "hi", "se", "p_le0", "mde"}
    with pytest.raises(ValueError):
        M.paired_delta(a, M.WithinAP(y[:600], p[:600], g[:600]), n_boot=10)


def test_mde():
    assert M.mde(1.0) == pytest.approx(2.486, abs=1e-3)
    assert M.mde(0.01) == pytest.approx(config.MDE_FACTOR * 0.01, abs=1e-4)


def test_holm_and_fixed_sequence():
    adj = M.holm({"H2": 0.01, "H3": 0.04, "H4": 0.03, "H5": 0.2})
    assert adj == pytest.approx({"H2": 0.04, "H3": 0.09, "H4": 0.09, "H5": 0.2})
    out = M.fixed_sequence_holm({"H1": 0.01, "H2": 0.01, "H3": 0.04, "H4": 0.03, "H5": 0.2})
    assert out.loc["H1", "reject"] and out.loc["H2", "reject"]
    assert not out.loc[["H3", "H4", "H5"], "reject"].any()
    # H1 fails: nothing else is tested, however small its p
    out = M.fixed_sequence_holm({"H1": 0.2, "H2": 1e-6, "H4": 1e-6})
    assert not out["reject"].any() and not out.loc["H2", "tested"]
    assert out.loc["H2", "p_adj"] == pytest.approx(0.2)
    # H3 underpowered (None): leaves the family, so Holm uses m = 2
    out = M.fixed_sequence_holm({"H1": 0.001, "H2": 0.02, "H3": None, "H4": 0.03})
    assert not out.loc["H3", "in_family"] and out.loc["H2", "p_holm"] == pytest.approx(0.04)
    assert out.loc["H4", "reject"] and out.loc["H2", "reject"]
    with pytest.raises(ValueError):
        M.fixed_sequence_holm({"H2": 0.01})
    assert M.iut_p(0.01, 0.04) == 0.04


# --------------------------------------------------------------------------- scorer hook


def _condition_frames(n_classes=10, n_points=900, seed=30):
    """points, test OOF over all of G, inner OOF (defined rows of the other folds' classes),
    and a segments table, for `condition_metrics`."""
    rng = np.random.default_rng(seed)
    t, cls, y, o, oc = world(n_classes, n_points, 6, rng)
    in_g = np.ones(t.size, bool)
    in_g[(t - T0) % 600 < 20] = False                         # some causal exclusions
    y = np.where(in_g, y, np.nan)
    nxt = np.full(t.size, np.nan)
    for c in range(n_classes):
        m, oo = cls == c, np.sort(o[oc == c])
        k = np.searchsorted(oo, t[m], "right")
        nxt[m] = np.where(k < oo.size, oo[np.minimum(k, oo.size - 1)], np.nan)
    points = pd.DataFrame({"class_id": cls, "t_s": t, "in_G": in_g, "next_bid_s": nxt,
                           config.HEADLINE_LABEL: y})
    fold = cls % 5
    signal = oracle_scores(t, cls, o, oc)
    p = 0.6 * signal + 0.5 * rng.random(t.size)
    test = pd.DataFrame({"class_id": cls, "t_s": t, "fold": fold, "y": y, "p_raw": p,
                         "p_cal": p, "inner": False})[in_g]
    parts = []
    for k in range(5):
        m = (fold != k) & ~np.isnan(y)
        q = 0.6 * signal[m] + 0.5 * rng.random(m.sum())
        parts.append(pd.DataFrame({"class_id": cls[m], "t_s": t[m], "fold": k, "y": y[m],
                                   "p_raw": q, "p_cal": q, "inner": True}))
    inner = pd.concat(parts, ignore_index=True)
    segs = pd.DataFrame([{"class_id": c, "seg_idx": j, "start_s": T0 + 300.0 * j}
                         for c in range(n_classes) for j in range(7)])
    return points, test, inner, segs


def test_condition_metrics_with_inner_rows():
    points, test, inner, segs = _condition_frames()
    oof = pd.concat([test, inner], ignore_index=True)
    train = inner.drop(columns=["y", "inner"])          # outer model on the training G
    row = M.condition_metrics(oof, points, segments=segs, train_scores=train, n_boot=200,
                              n_rep=3)
    assert row["tau_method"] == "rate_matched"
    assert row["ap_within_lo"] <= row["ap_within"] <= row["ap_within_hi"]
    assert row["timing_gain"] > 0.1 and np.isfinite(row["timing_gain_local"])
    for k in ("far01", "far05"):
        assert 0 <= row[f"recall_{k}"] <= 1 and np.isfinite(row[f"realised_{k}"])
        assert np.isfinite(row[f"recall_random_{k}"]) and np.isfinite(row[f"tiw_{k}"])
        assert row[f"recall_{k}_lo"] <= row[f"recall_{k}"] <= row[f"recall_{k}_hi"]
    assert row["eventlift0_far01"] > 2 and row["eventlift_shift_far01"] > 1
    assert row["realised_far05"] >= row["realised_far01"] - 0.05
    assert 0 < row["pafroc"] <= 1 and np.isfinite(row["lead_med_s"])
    assert row["lead_med_s_lo"] <= row["lead_med_s"] <= row["lead_med_s_hi"]
    # §8.5: CIs for pooled AP, TG and TG_local from the same class resamples
    te = test[test["y"].notna()]
    assert row["ap_lo"] <= M.ap(te["y"], te["p_cal"]) <= row["ap_hi"]
    assert row["timing_gain_lo"] <= row["timing_gain"] <= row["timing_gain_hi"]
    assert row["timing_gain_local_lo"] <= row["timing_gain_local"] <= row["timing_gain_local_hi"]
    floats = {c.name for c in schemas.schema("results") if c.dtype == "float64"}
    assert {"ap_within", "ap_within_lo", "ap_within_hi", "ap_shift", "timing_gain",
            "ap_local", "timing_gain_local", "recall_far01", "eventlift_shift_far01",
            "eventlift0_far01", "lead_med_s", "pafroc", "recall_random_far01", "ap_lo",
            "ap_hi", "timing_gain_lo", "timing_gain_hi", "recall_far01_lo", "lead_med_s_hi",
            "ap_shift_base"} <= floats


def test_thresholds_are_never_fit_on_test_scores():
    """Without training-fold scores the event thresholds are NaN (formerly they were matched
    on the test fold's own scores): rescaling the test scores can never move a threshold."""
    points, test, inner, _ = _condition_frames(n_classes=6)
    oof = pd.concat([test, inner], ignore_index=True)
    row = M.condition_metrics(oof, points, n_boot=50, n_rep=2)
    assert row["tau_method"] == "none_no_train_scores"
    assert np.isnan(row["recall_far01"]) and np.isnan(row["realised_far01"])
    train = inner.assign(p_cal=inner["p_cal"])
    a = M.condition_metrics(oof, points, train_scores=train, n_boot=50, n_rep=2)
    te = ~oof["inner"]
    warped = oof.assign(p_cal=np.where(te, oof["p_cal"] ** 0.3, oof["p_cal"]))
    b = M.condition_metrics(warped, points, train_scores=train, n_boot=50, n_rep=2)
    # same thresholds, monotone-warped test scores: the realised firing moves
    assert a["realised_far01"] != b["realised_far01"]


def test_condition_metrics_without_inner_rows_never_uses_oracle():
    points, test, _, _ = _condition_frames(n_classes=6)
    row = M.condition_metrics(test, points, n_boot=100, n_rep=2)
    assert row["tau_method"] == "none_no_inner"
    assert np.isnan(row["recall_far01"]) and np.isnan(row["realised_far05"])
    assert np.isnan(row["ap_local"]) and np.isfinite(row["pafroc"])
    assert np.isfinite(row["ap_within"]) and np.isfinite(row["timing_gain"])
    # the OOF rows' own event grid (experiments writes the target's grid there; T-speech's
    # differs from points.in_G) is used; without one, points.in_G is merged in
    same = test.merge(points[["class_id", "t_s", "in_G"]], on=["class_id", "t_s"])
    row2 = M.condition_metrics(same, points, n_boot=100, n_rep=2)
    assert row2["pafroc"] == row["pafroc"] and row2["ap_within"] == row["ap_within"]
    row3 = M.condition_metrics(test.assign(in_G=False), points, n_boot=100, n_rep=2)
    assert np.isnan(row3["pafroc"]) or row3["pafroc"] == 0     # nothing on the grid fires


def test_segment_index_and_shift_reference():
    segs = pd.DataFrame({"class_id": [1, 1, 2], "seg_idx": [0, 1, 0],
                         "start_s": [100.0, 400.0, 50.0]})
    idx = M.segment_index([90, 100, 399, 400, 60], [1, 1, 1, 1, 2], segs)
    assert idx.tolist() == [-1, 0, 0, 1, 0]
    rng = np.random.default_rng(31)
    t, cls, y, o, oc = world(3, 600, 4, rng)
    d = M.EventData(t, cls, y=y, onsets=o, onset_groups=oc)
    ref = M.shift_reference(d, rng.random(t.size), n_rep=2)
    assert ref.n_classes_skipped == 0 and len(ref.curves) == 2
    ref_local = M.shift_reference(d, rng.random(t.size), kind="local",
                                  segments=(t // 300).astype(int), n_rep=2)
    assert 0 <= ref_local.recall_at(0.5) <= 1


# --------------------------------------------------------------------------- review fixes


def test_vectorised_bootstrap_statistics_equal_the_definitions():
    """PooledAP / WithinAP / TimingGain evaluated for a weight matrix equal the per-draw
    statistics, and PooledAP equals sklearn's AP on the materialised resample."""
    rng = np.random.default_rng(5)
    t, cls, y, _, _ = world(8, 500, 5, rng)
    p = np.round(rng.random(t.size) + 0.4 * np.nan_to_num(y), 2)       # ties included
    d = ~np.isnan(y)
    pooled = M.PooledAP(y[d], p[d], cls[d])
    within = M.WithinAP(y[d], p[d], cls[d])
    tg = M.TimingGain(y[d], p[d], cls[d], t=t[d], n_rep=3, seed=1)
    W = M.bootstrap_weights(8, 40, seed=3)
    for stat in (pooled, within, tg):
        np.testing.assert_allclose(stat.batch(W), [stat(w) for w in W], rtol=1e-10)
    w = W[0]
    idx = np.concatenate([np.flatnonzero(cls[d] == c) for c in range(8) for _ in range(w[c])])
    assert pooled(w) == pytest.approx(average_precision_score(y[d][idx], p[d][idx]))
    assert pooled() == pytest.approx(average_precision_score(y[d], p[d]))


def test_timing_gain_reports_its_own_base_and_skipped_classes():
    """ap_shift is over the classes the global shift keeps (n_c >= 400): the base AP of
    those same classes (ap_shift_base) is reported, so ap_shift_base - ap_shift == TG."""
    rng = np.random.default_rng(2)
    t1, c1, y1, o1, oc1 = world(6, 300, 4, rng)                       # skipped (< 400)
    t2, c2, y2, o2, oc2 = world(6, 600, 4, rng)
    c2 = c2 + 6
    t, cls, y = np.r_[t1, t2], np.r_[c1, c2], np.r_[y1, y2]
    p = 0.6 * oracle_scores(t, cls, np.r_[o1, o2], np.r_[oc1, oc2 + 6]) + rng.random(t.size)
    p[cls < 6] += 1.0                                                # skipped classes score high
    pts = pd.DataFrame({"class_id": cls, "t_s": t, "in_G": True, config.HEADLINE_LABEL: y})
    oof = pd.DataFrame({"class_id": cls, "t_s": t, "fold": 0, "y": y, "p_raw": p,
                        "p_cal": p, "inner": False})
    row = M.condition_metrics(oof, pts, n_boot=30, n_rep=3)
    assert row["n_classes_shift_skipped"] == 6
    assert row["timing_gain"] == pytest.approx(row["ap_shift_base"] - row["ap_shift"])
    d = ~np.isnan(y) & (cls >= 6)
    assert row["ap_shift_base"] == pytest.approx(M.ap(y[d], p[d]))
    assert M.ap(y[~np.isnan(y)], p[~np.isnan(y)]) != pytest.approx(row["ap_shift_base"])


def test_froc_counts_reaggregate_curves_for_any_class_weights():
    rng = np.random.default_rng(9)
    t, cls, y, o, oc = world(5, 600, 5, rng)
    p = 0.5 * oracle_scores(t, cls, o, oc) + rng.random(t.size)
    data = M.EventData(t, cls, y=y, onsets=o, onset_groups=oc)
    fc = M.FrocCounts(data, p, n_thresholds=60)
    taus = M.froc_thresholds(data, p, 60)
    for w in (np.ones(5), np.array([2.0, 0, 1, 1, 3])):
        curve = M.froc(data, p, thresholds=taus, w=w, tiw=False)
        for far in (0.1, 0.5, 1.0):
            assert fc.recall_at(far, w[None, :])[0] == pytest.approx(
                M.recall_at_far(curve, far, interpolate=True))


def test_event_comparison_pairs_references_at_matched_far():
    """EV rows: references read at the model's realised FAR; deltas paired on the same
    class resamples; S3 needs the recall CI lower bound above shift and R."""
    rng = np.random.default_rng(4)
    t, cls, y, o, oc = world(8, 700, 5, rng)
    sig = oracle_scores(t, cls, o, oc)
    nxt = np.full(t.size, np.nan)
    for c in range(8):
        m, oo = cls == c, np.sort(o[oc == c])
        k = np.searchsorted(oo, t[m], "right")
        nxt[m] = np.where(k < oo.size, oo[np.minimum(k, oo.size - 1)], np.nan)
    pts = pd.DataFrame({"class_id": cls, "t_s": t, "in_G": True, "next_bid_s": nxt,
                        config.HEADLINE_LABEL: y})

    def oof(strength, seed):
        r = np.random.default_rng(seed)
        q = strength * sig + r.random(t.size)
        test = pd.DataFrame({"class_id": cls, "t_s": t, "fold": cls % 4, "y": y, "p_raw": q,
                             "p_cal": q, "inner": False})
        inner = test.assign(inner=True, p_cal=strength * sig + r.random(t.size))
        return pd.concat([test, inner], ignore_index=True), inner
    model, inner = oof(1.0, 1)
    train = inner.drop(columns=["y", "inner"])
    weak, _ = oof(0.05, 2)
    ev = M.event_comparison(model, {"R": weak}, pts, train_scores=train, n_boot=200,
                            n_rep=3, n_thresholds=80)
    mrow = ev[(ev["reference"] == "model") & (ev["budget"] == 0.1)].iloc[0]
    rrow = ev[(ev["reference"] == "R") & (ev["budget"] == 0.1)].iloc[0]
    assert rrow["far"] == mrow["far"] and rrow["delta"] > 0
    assert rrow["delta_lo"] > 0 and rrow["delta_p_le0"] < 0.05
    assert mrow["recall_lo"] <= mrow["recall"] <= mrow["recall_hi"]
    assert bool(mrow["s3_pass"]) == bool(
        mrow["recall_lo"] > ev.loc[(ev["reference"] == "shift") & (ev["budget"] == 0.1),
                                   "recall"].item() and mrow["recall_lo"] > rrow["recall"])
    no_train = M.event_comparison(model, {"R": weak}, pts, n_boot=20, n_rep=2)
    assert set(no_train["reference"]) == {"AO"}                       # never fit on test
