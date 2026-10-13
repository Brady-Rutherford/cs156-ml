"""Positive and negative controls (spec §9.2): P-clock, P-plant, CALLON, A-mismatch, N-name,
IH cut-off shift, label shift, A-jitter and A-stream. Synthetic data only: the invented world
of `oncue.synthetic` and small hand-made arrays; no network (A-stream's transcriber is a
fake injected here)."""

from __future__ import annotations

import json
from argparse import Namespace

import numpy as np
import pandas as pd
import pytest
from oncue import config, controls, synthetic

NB = 200          # bootstrap resamples in tests


def _points(sc, episodes: pd.DataFrame) -> pd.DataFrame:
    """A minimal points table for one synthetic class (headline label, at-risk mask)."""
    t = np.arange(config.T_OPEN_S, sc.T_end_s - config.H_S + 1e-9, config.DELTA_S)
    bids = np.sort(episodes.loc[episodes["ep_type"] == config.EP_BID, "onset_s"].to_numpy())
    y = np.full(len(t), np.nan)
    for i, ti in enumerate(t):
        if not ((bids > ti) & (bids <= ti + 2)).any():
            y[i] = float(((bids > ti + 2) & (bids <= ti + 15)).any())
    up = np.zeros(len(t), bool)
    for r in episodes.itertuples():
        up |= (t >= r.onset_s) & (t < r.offset_s + config.R_LOW_S)
    nxt = np.array([bids[bids > ti][0] if (bids > ti).any() else np.nan for ti in t])
    return pd.DataFrame({"class_id": sc.class_id, "t_s": t, config.HEADLINE_LABEL: y,
                         "in_G": ~up, "at_risk": ~up & np.isfinite(y), "next_bid_s": nxt,
                         "in_vault": False})


@pytest.fixture(scope="module")
def tables(world, truth):
    """Concatenated truth tables of the world plus a points table."""
    out = {name: pd.concat([truth[sc.class_id][name] for sc in world], ignore_index=True)
           for name in ("words", "hand_episodes", "segments", "hand_frames")}
    out["points"] = pd.concat([_points(sc, truth[sc.class_id]["hand_episodes"])
                               for sc in world], ignore_index=True)
    return out


# --------------------------------------------------------------------------- P-clock


def _speech(rng, T=900.0):
    """Alternating speech/silence intervals and 0.3 s words inside the speech."""
    t, iv = 0.0, []
    while t < T:
        on = rng.uniform(1.0, 6.0)
        iv.append((t, t + on))
        t += on + rng.uniform(1.0, 5.0)
    words = [(a, min(a + 0.3, b)) for s, e in iv for a, b in
             [(x, x + 0.3) for x in np.arange(s, e, 0.32)]]
    return np.array(iv), pd.DataFrame(words, columns=["start_s", "end_s"])


def _border_samples(cid, iv, lag, T=900.0):
    t = np.arange(0.0, T, 0.5)
    z = ((iv[:, None, 0] + lag <= t) & (t < iv[:, None, 1] + lag)).any(axis=0)
    return pd.DataFrame({"class_id": cid, "t_s": t, "z": z.astype(float)})


def test_p_clock_passes_when_aligned_and_fails_when_shifted():
    rng = np.random.default_rng(0)
    samples, words, shifted = [], [], []
    for cid, lag in ((1, 0.2), (2, 0.35), (3, 0.5)):
        iv, w = _speech(rng)
        samples.append(_border_samples(cid, iv, lag))
        shifted.append(_border_samples(cid, iv, lag + 3.0))     # video 3 s behind audio
        words.append(w.assign(class_id=cid))
    words = pd.concat(words, ignore_index=True)
    ok = controls.p_clock(pd.concat(samples), words)
    assert ok["pass"], ok["checks"]
    assert 0.0 <= ok["delta_star"] <= 0.5 and 0.1 <= ok["lag_s"] <= 0.6
    assert ok["sharpness"] >= config.P_CLOCK_SHARPNESS
    assert list(ok["curve"]["delta_s"]) == list(config.P_CLOCK_DELTAS_S)
    bad = controls.p_clock(pd.concat(shifted), words)
    assert not bad["pass"] and not bad["checks"]["peak_in_range"]
    flat = pd.concat(samples).assign(z=lambda d: (np.random.default_rng(1).random(len(d))
                                                  < 0.4).astype(float))
    assert not controls.p_clock(flat, words)["pass"]


def test_p_clock_recovers_the_synthetic_world_lag(world, truth):
    sc = world.classes[0]
    samples, w = controls.p_clock_inputs(truth[sc.class_id]["hand_frames"],
                                         truth[sc.class_id]["words"], any_speaker=True)
    res = controls.p_clock(samples, w.assign(class_id=sc.class_id))
    lag = synthetic.L_UI_S - sc.meta["cal_offset_s"]           # border lag behind audio
    assert abs(res["lag_s"] - lag) <= 0.3


# --------------------------------------------------------------------------- CALLON


def test_callon_check_passes_on_truth_and_flags_late_labels(tables):
    words, pts, segs = tables["words"], tables["points"], tables["segments"]
    onsets = controls.bid_onsets(episodes=tables["hand_episodes"])
    ok = controls.callon_check(words, onsets, pts, segs, n_boot=NB)
    assert ok["pass"] and ok["n_bids"] == len(onsets)
    assert ok["n_matched"] > len(onsets)
    late = onsets.assign(onset_s=onsets["onset_s"] + 8.0)       # labels 8 s late
    bad = controls.callon_check(words, late, pts, segs, n_boot=NB)
    assert not bad["pass"] and bad["rate_bid"] > ok["rate_bid"]


def test_matched_points_rules(tables):
    onsets = controls.bid_onsets(episodes=tables["hand_episodes"])
    m = controls.matched_points(tables["points"], onsets, tables["segments"])
    assert len(m) <= config.CALLON_MATCHED * len(onsets)
    d = (m["t_s"] - m["onset_s"]).abs()
    assert d.between(*config.CALLON_MATCH_DIST_S).all()
    p = tables["points"].set_index(["class_id", "t_s"])
    sel = p.loc[list(zip(m["class_id"], m["t_s"]))]
    assert (sel[config.HEADLINE_LABEL] == 0).all() and sel["at_risk"].all()
    seg_o = [controls.seg_type_at(tables["segments"], c, [o])[0]
             for c, o in zip(m["class_id"], m["onset_s"])]
    seg_b = [controls.seg_type_at(tables["segments"], c, [t])[0]
             for c, t in zip(m["class_id"], m["t_s"])]
    assert seg_o == seg_b


def test_controls_refuse_vault_rows(tables):
    with pytest.raises(AssertionError):
        controls.assert_dev(tables["points"].assign(in_vault=True))
    assert len(controls.dev_only(tables["points"].assign(in_vault=True))) == 0


# --------------------------------------------------------------------------- P-plant


def test_plant_design_marks_exactly_the_visible_windows():
    X = pd.DataFrame({"class_id": 1, "t_s": np.arange(100.0, 200.0, 2.0),
                      "txt_all60": "p_a", "txt_all15": "p_b", "txt_all60_name": "p_a",
                      "txt_chat60": "c_x"})
    planted = pd.DataFrame({"class_id": [1], "onset_s": [150.0]})
    out, vis = controls.plant_design(X, planted)
    end = 150.0 - config.PLANT_LEAD_S + controls.PLANT_WORD_DUR_S
    t = X["t_s"].to_numpy()
    exp60 = (end > t - 60) & (end <= t - config.EPS_W_S)
    exp15 = (end > t - 15) & (end <= t - config.EPS_W_S)
    tok = "p_" + config.PLANT_TOKEN
    assert np.array_equal(vis, exp60)
    assert np.array_equal(out["txt_all60"].str.contains(tok).to_numpy(), exp60)
    assert np.array_equal(out["txt_all60_name"].str.contains(tok).to_numpy(), exp60)
    assert np.array_equal(out["txt_all15"].str.contains(tok).to_numpy(), exp15)
    assert not out["txt_chat60"].str.contains(tok).any()
    assert not (t[exp15] < 150.0 - config.PLANT_LEAD_S).any()   # never before it is said


def test_plant_words_and_choice(tables):
    onsets = controls.bid_onsets(episodes=tables["hand_episodes"])
    planted = controls.choose_planted(onsets, seed=1)
    assert len(planted) == round(config.PLANT_FRAC * len(onsets))
    pd.testing.assert_frame_equal(planted, controls.choose_planted(onsets, seed=1))
    w = controls.plant_words(tables["words"], planted)
    new = w[w["text_norm"] == config.PLANT_TOKEN]
    assert len(new) == len(planted) and (new["role"] == config.ROLE_PROF).all()
    assert np.allclose(np.sort(new["start_s"]), np.sort(planted["onset_s"] - 8.0))
    assert len(w) == len(tables["words"]) + len(planted)


def test_planted_ap():
    y = np.array([1, 1, 0, 0, 0, 1])
    p = np.array([0.9, 0.2, 0.1, 0.3, 0.05, 0.95])
    vis = np.array([True, False, False, False, False, True])
    assert controls.planted_ap(y, p, vis) == 1.0               # invisible positive dropped


# --------------------------------------------------------------------------- A-mismatch


def test_derangements_within_section_without_fixed_points():
    classes = pd.DataFrame({"class_id": range(10), "section_id": [0] * 4 + [1] * 5 + [2]})
    maps = controls.derangements(classes, n=5, seed=3)
    assert len(maps) == 5 and maps == controls.derangements(classes, n=5, seed=3)
    sec = classes.set_index("class_id")["section_id"]
    for m in maps:
        assert set(m) == set(range(9))                         # the singleton section drops
        assert all(k != v and sec[k] == sec[v] for k, v in m.items())
        assert sorted(m.values()) == sorted(m)                 # a permutation
    assert len({tuple(sorted(m.items())) for m in maps}) > 1


def test_mismatch_pcw_recomputes_similarity_columns_only():
    feats = pd.DataFrame({"class_id": [1, 1, 2, 3], "t_s": [0.0, 2.0, 0.0, 0.0],
                          "pcw_has": [1.0, 1, 1, 0], "pcw_my_words": [10.0, 10, 20, 0],
                          "pcw_res_A": [0.1, 0.2, 0.3, 0.4], "pcw_unspent": [1.0, 2, 3, 4],
                          "lo_sim": [0.0, 0, 0, 0], "ctx_min": [1.0, 2, 3, 4]})
    seen = []

    def recompute(cid, donor, rows):
        seen.append((cid, donor))
        return pd.DataFrame({"pcw_res_A": -donor, "pcw_unspent": -donor, "lo_sim": 9.0},
                            index=rows.index)
    out = controls.mismatch_pcw(feats, {1: 2, 2: 1}, recompute)
    assert sorted(seen) == [(1, 2), (2, 1)] and set(out["class_id"]) == {1, 2}
    assert (out.loc[out["class_id"] == 1, "pcw_res_A"] == -2).all()
    for c in ("pcw_has", "pcw_my_words", "ctx_min"):
        pd.testing.assert_series_equal(out[c], feats.loc[out.index, c])
    with pytest.raises(ValueError):
        controls.mismatch_pcw(feats, {1: 2}, lambda c, d, r: pd.DataFrame(
            {"pcw_has": 0.0}, index=r.index))
    assert set(controls.mismatch_columns(feats.columns)) == {"pcw_res_A", "pcw_unspent",
                                                             "lo_sim"}
    runs = pd.DataFrame({"ap_within": [0.2, 0.3], "ap": [0.1, 0.1]})
    s = controls.mismatch_summary(runs)
    assert s["ap_within"]["mean"] == pytest.approx(0.25) and s["ap"]["n"] == 2


# --------------------------------------------------------------------------- N-name


def test_name_rates_and_counts(tables):
    onsets = controls.bid_onsets(episodes=tables["hand_episodes"])
    m = controls.matched_points(tables["points"], onsets, tables["segments"])
    rates = controls.name_rates(tables["words"], onsets, m, n_boot=NB)
    assert set(rates["window"]) == {"pre_bid", "post_bid", "matched"}
    post = rates.set_index("window").loc["post_bid"]
    assert post["rate_per_min"] > 0 and post["ci_lo"] <= post["rate_per_min"] <= post["ci_hi"]
    assert not any(c.startswith(("text", "word")) for c in rates.columns)   # aggregates only


def test_name_counts_are_causal():
    words = pd.DataFrame({"class_id": 1, "start_s": [1.0, 5.0, 9.6],
                          "end_s": [1.5, 5.5, 9.9], "role": "prof",
                          "text_masked": ["[NAME],", "hi", "[NAME]"]})
    pts = pd.DataFrame({"class_id": [1, 1], "t_s": [10.0, 20.0]})
    c = controls.name_counts_at(pts, words, windows=(15.0, 60.0))
    assert c["name_n_60"].tolist() == [1, 2]                  # the 9.9 word ends > t - eps_w
    assert c["name_n_15"].tolist() == [1, 1]                  # (5, 19.75] holds one name
    lift = controls.name_lr_lift(pd.DataFrame({"n": np.r_[np.ones(20), np.zeros(80)]}),
                                 np.r_[np.ones(20), np.zeros(80)], np.arange(100) % 10,
                                 n_boot=50)
    assert lift["ap"] == pytest.approx(1.0) and lift["lift"] > 1
    assert controls.name_flag(0.03) and not controls.name_flag(0.01)
    assert not controls.name_flag(float("nan"))


# --------------------------------------------------------------------------- IH, LS, jitter


def test_cutoff_shift_label_shift_and_jitter(tables):
    pts = tables["points"].head(5)
    ih = controls.cutoff_shift(pts, 30, include_me=True)
    assert (ih["t_feat_s"] - ih["t_s"] == 30).all() and ih["include_me"].all()
    assert ih["illegal"].all() and not controls.cutoff_shift(pts, -10)["illegal"].any()
    ep = tables["hand_episodes"]
    ls = controls.label_shift(ep, 2)
    assert np.allclose(ls["onset_s"], ep["onset_s"] - 2)
    assert np.allclose(ls["offset_s"], ep["offset_s"])
    j = controls.jitter_times({"words": tables["words"], "points": pts}, seed=4)
    w0, w1 = tables["words"], j["words"]
    assert np.allclose(w1["end_s"] - w1["start_s"], w0["end_s"] - w0["start_s"])
    d = w1["start_s"] - w0["start_s"]
    assert 0.3 < d.std() < 0.7 and abs(d.mean()) < 0.05
    pd.testing.assert_frame_equal(j["points"], pts)            # labels untouched
    assert j["words"].equals(controls.jitter_times({"words": w0}, seed=4)["words"])


# --------------------------------------------------------------------------- A-stream


def test_stream_audit_needs_injected_callables_and_splices(tables):
    s = controls.stream_sample(tables["points"], n=20, seed=2)
    assert (s[config.HEADLINE_LABEL] == 1).sum() == 10 and len(s) == 20
    with pytest.raises(RuntimeError):
        controls.stream_audit(s)                                # no default ASR path
    w = tables["words"]
    by_class = {c: g for c, g in w.groupby("class_id")}

    def fake_transcribe(cid, t0, t1):                           # "re-transcribed" words
        seg = by_class[cid]
        seg = seg[(seg["start_s"] >= t0) & (seg["end_s"] <= t1)]
        return seg.assign(text_masked=seg["text_masked"].str.lower())

    seen = []

    def rebuild(cid, t, words):
        seen.append(len(words))
        return {"n": len(words)}
    out = controls.stream_audit(s.head(4), transcribe=fake_transcribe, rebuild=rebuild,
                                score=lambda row: 0.5, words_by_class=by_class)
    assert out["p_stream"].eq(0.5).all() and len(seen) == 4
    one = by_class[s["class_id"].iloc[0]]
    t1 = float(s["t_s"].iloc[0])
    spliced = controls.splice_stream_words(one, fake_transcribe(one["class_id"].iloc[0],
                                                                t1 - 180, t1), t1 - 180, t1)
    inside = spliced[(spliced["start_s"] >= t1 - 180) & (spliced["end_s"] <= t1)]
    assert (inside["text_masked"] == inside["text_masked"].str.lower()).all()
    dis = controls.punct_case_disagreement(one, fake_transcribe(one["class_id"].iloc[0],
                                                                 0, 1e9))
    assert dis["n_matched"] > 0 and dis["case"] > 0 and dis["punct"] == 0
    summ = controls.stream_summary(s.assign(p_full=np.linspace(0, 1, 20),
                                            p_stream=np.linspace(0, 1, 20)), n_boot=50)
    assert summ["spearman"] == pytest.approx(1.0) and summ["delta_ap"] == pytest.approx(0.0)


def test_cluster_ci_brackets_the_estimate():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"class_id": np.repeat(np.arange(30), 10), "v": rng.normal(size=300)})
    est, lo, hi = controls.cluster_ci(df, lambda d: d["v"].mean(), n_boot=300)
    assert lo < est < hi and hi - lo < 1.0


# --------------------------------------------------------------------------- CLI


def test_cli_writes_aggregate_json(tables, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TABLES_DIR", tmp_path / "tables")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    (tmp_path / "tables").mkdir()
    for name in ("points", "words", "hand_episodes", "segments", "hand_frames"):
        tables[name].to_parquet(config.table_path(name), index=False)
    assert controls.cli(Namespace(extra=["--n-boot", "50"])) == 0
    for name in controls.CONTROL_NAMES:
        blob = (tmp_path / "controls" / f"{name}.json").read_text()
        json.loads(blob)
        assert "[NAME]" not in blob and "go ahead" not in blob     # no words leave
