"""report.hypothesis_table (§7, §8.5): the decision rule matches the CI shown, the §5.6 PCW
coverage rule, the pre-registered family never shrinks, K_c as in the results table, and
every row carries n_classes / n_onsets / base_rate. Invented OOF scores only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from oncue import config, metrics, report


def _base(classes, rng, n=600, bids=4):
    """Held-out rows with planted bids (6 positive points before each onset) and the bid
    onsets themselves (class ids as 12-hex-like strings, as in the bundle)."""
    rows, ons = [], []
    for c in classes:
        t = 120 + 2.0 * np.arange(n)
        y = np.zeros(n)
        for o in rng.choice(np.arange(20, n - 20), bids, replace=False):
            y[o - 7:o - 1] = 1
            ons.append({"class_hash": f"c{c:03d}", "onset_s": float(t[o]), "ep_type": "bid",
                        "left_censored": False})
        rows.append(pd.DataFrame({"class_id": f"c{c:03d}", "t_s": t, "y": y}))
    return pd.concat(rows, ignore_index=True), pd.DataFrame(ons)


def _oof(base, strength, rng):
    p = 1 / (1 + np.exp(-(strength * base["y"] + rng.normal(0, 1, len(base)))))
    return base.assign(fold=0, p_raw=p, p_cal=p, inner=False)


@pytest.fixture(scope="module")
def scenario():
    rng = np.random.default_rng(7)
    base, bids = _base(range(60), rng)
    small, small_bids = _base(range(100, 103), rng)                # a 3-class PCW subset
    oofs = {"K0-M3": _oof(base, 1.2, rng), "F0-Fstruct": _oof(base, 0.6, rng),
            "F0-Fall-me_hand": _oof(base, 1.2, rng), "F0-Fstruct-me_hand": _oof(base, 0.6, rng),
            "K0-R": _oof(base, 1.1, rng), "K1-M4": _oof(base, 1.0, rng),
            "K0-M2": _oof(base, 0.95, rng),
            "F1-Fall+pcwA": _oof(small, 0.5, rng), "F1-Fall-pcw": _oof(small, 0.5, rng)}
    segs = pd.DataFrame([{"class_id": f"c{c:03d}", "seg_idx": i, "start_s": 120.0 + 400 * i}
                         for c in range(60) for i in range(4)])
    return oofs, pd.concat([bids, small_bids], ignore_index=True), segs


def test_reject_implies_the_95_ci_lies_above_zero():
    """One-sided bootstrap p in (0.025, 0.05] used to reject while the 95% CI shown
    included 0; the rule is now p < alpha / 2, the same as "CI lower bound > 0"."""
    for seed in range(200):
        d = np.random.default_rng(seed).normal(0.012, 0.05, 40)
        st = metrics.ClassResampled(lambda x: float(np.mean(x)), np.arange(40), d)
        b = metrics.cluster_bootstrap(st, n_boot=2000, seed=156)
        if 0.025 < b.p_le0 <= 0.05:
            break
    assert b.lo <= 0 < b.point
    old = metrics.fixed_sequence_holm({"H1": b.p_le0, "H4": 0.001, "H5": 0.01})
    assert old.loc["H1", "reject"]                                 # the inconsistency
    new = metrics.fixed_sequence_holm({"H1": b.p_le0, "H4": 0.001, "H5": 0.01},
                                      alpha=config.ALPHA_ONE_SIDED, strict=True)
    assert not new["reject"].any()
    edge = metrics.fixed_sequence_holm({"H1": config.ALPHA_ONE_SIDED},
                                       alpha=config.ALPHA_ONE_SIDED, strict=True)
    assert not edge.loc["H1", "reject"]


def test_hypothesis_rows_reject_only_with_ci_above_zero(scenario):
    oofs, bids, segs = scenario
    ht = report.hypothesis_table(oofs, bids, segments=segs, n_boot=500, n_rep=3)
    assert (ht["alpha"] == config.ALPHA_ONE_SIDED).all()
    rej = ht["reject"].fillna(False).astype(bool)
    assert rej.any()
    assert (ht.loc[rej, "lo"] > 0).all()
    for h in ("H1", "H2"):               # an IUT hypothesis needs both components above 0
        parts = ht[ht["hypothesis"].str.startswith(h)]
        if parts["reject"].fillna(False).astype(bool).any():
            assert (parts["lo"] > 0).all()


def test_pcw_coverage_rule_takes_h3_out_of_the_family(scenario):
    """§5.6: fewer than 40 PCW classes or 150 bids -> H3 is exploratory and Holm's m does
    not count it (H4's adjusted p equals the one computed without H3)."""
    oofs, bids, segs = scenario
    with_h3 = report.hypothesis_table(oofs, bids, segments=segs, n_boot=400, n_rep=3)
    without = report.hypothesis_table({k: v for k, v in oofs.items() if not k.startswith("F1")},
                                      bids, segments=segs, n_boot=400, n_rep=3)
    a, b = with_h3.set_index("hypothesis"), without.set_index("hypothesis")
    assert a.loc["H3", "n_classes"] == 3 and "coverage rule" in a.loc["H3", "note"]
    assert not a.loc["H3", "in_family"] and np.isfinite(a.loc["H3", "estimate"])
    assert a.loc["H4", "p_adj"] == pytest.approx(b.loc["H4", "p_adj"])


def test_h2_never_silently_leaves_the_family(scenario, tmp_path):
    """Without the segments table (bundle re-run) H2 is "not computable here" and p_adj /
    reject of H2-H5 stay empty instead of shrinking Holm's family; the committed
    hypotheses.parquet restores the local H2 rows and the full family."""
    oofs, bids, segs = scenario
    local = report.hypothesis_table(oofs, bids, segments=segs, n_boot=400, n_rep=3)
    bundle = report.hypothesis_table(oofs, bids, n_boot=400, n_rep=3).set_index("hypothesis")
    assert bundle.loc[["H2a", "H2b"], "note"].eq(report.H2_NOT_COMPUTABLE).all()
    assert bundle.loc[["H2a", "H3", "H4", "H5"], "p_adj"].isna().all()
    assert bundle.loc[["H2a", "H3", "H4", "H5"], "reject"].isna().all()
    assert bundle.loc[["H1a", "H1b"], "p_adj"].notna().all()       # H1 is tested alone
    report.write_hypotheses(local, tmp_path)
    got = report.load_bundle(tmp_path).hypotheses
    re_run = report.hypothesis_table(oofs, bids, h2=got, n_boot=400, n_rep=3)
    a, b = local.set_index("hypothesis"), re_run.set_index("hypothesis")
    assert b.loc["H2a", "note"] == "precomputed locally"
    np.testing.assert_allclose(a.loc[["H4", "H5"], "p_adj"], b.loc[["H4", "H5"], "p_adj"])


def test_bid_weights_match_the_results_table(scenario):
    """K_c in the hypothesis tests is the results-table K_c (bids with a positive scored
    point), so each estimate is the difference of the AP_w values as reported."""
    oofs, bids, _ = scenario
    m = report._merged(oofs["K0-M3"], oofs["K0-R"])
    on = report.onsets_from_bids(bids)
    w = report.bid_weights(m, on, "y_a")
    nxt = np.full(len(m), np.nan)
    for c, g in m.groupby("class_id"):
        o = np.sort(on.loc[on["class_id"] == c, "onset_s"].to_numpy())
        k = np.searchsorted(o, g["t_s"].to_numpy(), side="right")
        nxt[g.index] = np.where(k < len(o), o[np.minimum(k, len(o) - 1)], np.nan)
    k_res = metrics.bid_counts(m["y_a"].to_numpy(), m["class_id"].to_numpy(), nxt)
    pd.testing.assert_series_equal(w.sort_index(), k_res.astype(float).sort_index(),
                                   check_names=False)
    ht = report.hypothesis_table({"K0-M3": oofs["K0-M3"], "K0-R": oofs["K0-R"],
                                  "F0-Fstruct": oofs["F0-Fstruct"]}, bids, n_boot=50)
    h4 = ht.set_index("hypothesis").loc["H4"]
    a = metrics.ap_within(m["y_a"], m["p_cal_a"], m["class_id"], k_res)
    b = metrics.ap_within(m["y_a"], m["p_cal_b"], m["class_id"], k_res)
    assert h4["estimate"] == pytest.approx(a - b)


def test_hypothesis_rows_carry_sizes(scenario, tmp_path):
    oofs, bids, segs = scenario
    ht = report.hypothesis_table(oofs, bids, segments=segs, n_boot=100, n_rep=2)
    assert {"n_classes", "n_onsets", "base_rate"} <= set(ht.columns)
    h = ht.set_index("hypothesis")
    assert h.loc["H1a", "n_classes"] == 60 and h.loc["H1a", "n_onsets"] == 240
    assert 0 < h.loc["H4", "base_rate"] < 0.1
    s = report.summary(None, ht)
    assert s["hypotheses"]["H4"]["n_onsets"] == 240.0
    assert s["hypotheses"]["H4"]["alpha"] == config.ALPHA_ONE_SIDED
