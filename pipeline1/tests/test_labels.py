"""Labels (§2.4-§3, A1): live state, speech onsets, every exclusion code, asymmetric
observability, the g sweep, glitch windows, onset_pending and T-speech's 30 s rule.

Hand-built scenarios use invented timings only; the synthetic world checks the whole table.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from oncue import config, labels, schemas
from oncue.causal import live_states

CID = 1
UP, DOWN, NA = config.STATE_UP, config.STATE_DOWN, config.STATE_NA


# --------------------------------------------------------------------------- scenario builder


def scenario(T_end: float = 600.0, *, episodes=(), na=(), segments=(("intro", 3.0),),
             restarts=(), voice=(), words=(), qa_pass=True, eps_f=0.6) -> dict:
    """Minimal schema-shaped tables for class CID.

    episodes: dicts with onset, offset, ep_type (default bid) and optional left_censored /
    eligible / outcome; the hand is UP on samples in [onset, offset).
    na: (a, b) spans of NA samples. voice: dicts (role, start, end, main). words: dicts
    (role, start, end, text).
    """
    segs = [[1_000, 2_000, 0.0]] + [[3_000, 4_000, float(r)] for r in restarts]
    classes = pd.DataFrame([{
        "class_id": CID, "T_end_s": T_end, "eps_F_s": eps_f, "rec_offset_s": 0.0,
        "rec_segments_json": json.dumps(segs), "qa_pass": qa_pass, "timing_precision": "ms",
        "pcw_status": "none", "course_code": "CS901"}])
    t = np.arange(0.0, T_end + 1e-9, 1.0 / config.F_SCAN)
    raw = np.full(len(t), DOWN, dtype=object)
    for e in episodes:
        raw[(t >= e["onset"]) & (t < e["offset"])] = UP
    for a, b in na:
        raw[(t >= a) & (t <= b)] = NA
    hf = pd.DataFrame({"class_id": CID, "t_s": t, "n_tiles": 9, "tile_found": raw != NA,
                       "raw_state": raw, "live_state": live_states(raw)})
    eps = pd.DataFrame([{
        "class_id": CID, "ep_id": i, "onset_s": e["onset"], "onset_retimed_s": e["onset"],
        "retime_failed": False, "offset_s": e["offset"], "offset_live_s": e["offset"] + 0.5,
        "dur_s": e["offset"] - e["onset"], "n_toggles": 0,
        "ep_type": e.get("ep_type", config.EP_BID), "delta_n_others": 0, "vote_lex": False,
        "contested": e.get("contested", False), "left_censored": e.get("left_censored", False),
        "me_hand_eligible": e.get("eligible", True), "outcome": e.get("outcome", "withdrawn"),
        "call_latency_s": np.nan, "seg_type": "intro"} for i, e in enumerate(episodes)],
        columns=schemas.columns("hand_episodes"))
    seg = pd.DataFrame([{"class_id": CID, "seg_idx": i, "start_s": s, "title_masked": typ,
                         "section_masked": "", "seg_type": typ}
                        for i, (typ, s) in enumerate(segments)])
    v = pd.DataFrame([{"class_id": CID, "spk_hash": "u_me" if x["role"] == "me" else "u_x",
                       "role": x["role"], "start_s": x["start"], "end_s": x["end"],
                       "room_main": x.get("main", True), "precise": True,
                       "word_bearing": True, "first_word_avail_s": np.nan} for x in voice],
                     columns=schemas.columns("voice"))
    w = pd.DataFrame([{"class_id": CID, "w_idx": i, "start_s": x["start"], "end_s": x["end"],
                       "text_masked": x.get("text", "word"), "text_norm": x.get("text", "word"),
                       "has_qmark": False, "conf": 0.9, "filler": False,
                       "spk_hash": "u_me" if x["role"] == "me" else "u_x", "role": x["role"],
                       "role_event_start_s": x["start"] - 0.1,
                       "role_avail_s": x["end"] + config.EPS_W_S, "role_fused": x["role"],
                       "how": "forum"} for i, x in enumerate(words)],
                     columns=schemas.columns("words"))
    return {"classes": classes, "hand_frames": hf, "hand_episodes": eps, "segments": seg,
            "voice": schemas.coerce(v, "voice"), "words": schemas.coerce(w, "words")}


def at(p: pd.DataFrame, t: float) -> pd.Series:
    row = p[np.isclose(p["t_s"], t)]
    assert len(row) == 1, t
    return row.iloc[0]


def reason(p, t):
    r = at(p, t)["excl_reason"]
    return None if pd.isna(r) else str(r)


# --------------------------------------------------------------------------- live state


def test_live_state_uses_samples_up_to_t_only():
    rng = np.random.default_rng(3)
    raw = rng.choice([UP, DOWN, NA], size=400, p=[0.3, 0.6, 0.1])
    full = live_states(raw)
    for k in range(1, len(raw) + 1, 7):
        assert list(live_states(raw[:k])) == list(full[:k])


def test_live_state_needs_k_agreeing_samples():
    raw = [DOWN, DOWN, UP, DOWN, UP, UP, UP, DOWN, DOWN]
    live = list(live_states(raw))
    assert live[0] == NA                       # nothing confirmed yet
    assert live[1:4] == [DOWN, DOWN, DOWN]     # a single UP sample is not confirmed
    assert live[4] == DOWN and live[5] == UP   # live at the second agreeing sample
    assert live[7] == UP and live[8] == DOWN


def test_live_state_matches_synthetic_detector(truth):
    for tables in truth.values():
        hf = tables["hand_frames"]
        mine = live_states(hf["raw_state"])
        assert (mine[1:] == hf["live_state"].astype(str).to_numpy()[1:]).all()


# --------------------------------------------------------------------------- speech onsets


def test_speech_onsets_match_synthetic_truth(world, truth):
    for sc in world:
        tb = truth[sc.class_id]
        mine = labels.speech_onsets(tb["voice"], tb["words"], tb["hand_episodes"])
        ref = tb["speech_onsets"]
        schemas.validate(mine, "speech_onsets", strict=True)
        np.testing.assert_allclose(mine["onset_s"], ref["onset_s"])
        np.testing.assert_allclose(mine["turn_end_s"], ref["turn_end_s"])
        assert list(mine["fresh"]) == list(ref["fresh"])
        assert list(mine["hand_preceded"]) == list(ref["hand_preceded"])


def test_speech_onsets_merge_fresh_and_drop_blips():
    voice = [{"role": "me", "start": 100, "end": 104}, {"role": "me", "start": 105.5, "end": 108},
             {"role": "me", "start": 111, "end": 113}, {"role": "me", "start": 150, "end": 152},
             {"role": "me", "start": 200, "end": 200.3}]                  # blip: no words
    words = [{"role": "me", "start": 101, "end": 101.5}, {"role": "me", "start": 106, "end": 106.4},
             {"role": "me", "start": 111.5, "end": 112}, {"role": "me", "start": 150.2, "end": 150.6}]
    tb = scenario(voice=voice, words=words)
    on = labels.speech_onsets(tb["voice"], tb["words"])
    assert list(on["onset_s"]) == [100, 111, 150]       # 1.5 s gap merged; 3 s gap not
    assert list(on["turn_end_s"]) == [108, 113, 152]
    assert list(on["fresh"]) == [True, False, True]     # 111 - 108 < 30; 150 - 113 >= 30
    assert on["hand_preceded"].isna().all()             # no episodes given


# --------------------------------------------------------------------------- points table


def test_points_schema_grid_and_flags(world, truth):
    for sc in world:
        p = labels.points(sc.class_id, truth[sc.class_id])
        schemas.validate(p, "points", strict=True)
        assert np.allclose(np.diff(p["t_s"]), config.DELTA_S) and p["t_s"].iloc[0] == 0
        assert p["t_s"].iloc[-1] <= sc.T_end_s
        assert (p["at_risk"] == p["excl_reason"].isna()).all()
        assert set(p["excl_reason"].dropna()) <= set(config.EXCL_REASONS)
        assert (p["in_vault"] == config.in_vault(sc.class_id)).all()
        assert p["fold"].isna().all()
        head = p[config.HEADLINE_LABEL]
        assert head[p["at_risk"]].notna().all()          # at risk => defined
        causal = p["excl_reason"].isin(config.EXCL_CAUSAL)
        for col in config.LABEL_COLUMNS:
            if col != labels.SPEECH_LABEL:               # T-speech has its own at-risk set
                assert p.loc[causal & (p["excl_reason"] != "end"), col].isna().all(), col
        expect = np.where(head.isna(), "undef", np.where(head == 1, "pos", "neg"))
        assert (p["label_state"].to_numpy() == expect).all()
        assert (p["in_G"] == ~p["excl_reason"].isin(config.EXCL_CAUSAL)).all()


def test_each_bid_gives_about_six_and_a_half_positives(world, truth):
    pos, bids = 0, 0
    for sc in world:
        p = labels.points(sc.class_id, truth[sc.class_id])
        pos += int((p[config.HEADLINE_LABEL] == 1).sum())
        bids += int((sc.episodes["ep_type"] == "bid").sum())
    assert 5.5 <= pos / bids <= 7.0      # (H - g) / Δ = 6.5 when the run-up is at risk


def test_next_bid_is_the_first_bid_after_t():
    tb = scenario(episodes=[{"onset": 300, "offset": 306}, {"onset": 420, "offset": 425}])
    p = labels.points(CID, tb)
    assert at(p, 200)["next_bid_s"] == 300 and at(p, 300)["next_bid_s"] == 420
    assert np.isnan(at(p, 430)["next_bid_s"])


# --------------------------------------------------------------------------- exclusion codes


def test_every_exclusion_code_fires_where_expected():
    eps_f = 0.6
    tb = scenario(
        T_end=900.0, eps_f=eps_f, restarts=(700.0,),
        segments=(("intro", 3.0), ("breakout", 200.0), ("debrief", 260.0), ("break", 380.0),
                  ("activity", 400.0)),
        episodes=[{"onset": 520, "offset": 530}, {"onset": 601.8, "offset": 605},
                  {"onset": 650, "offset": 650.5, "ep_type": "glitch", "eligible": False},
                  {"onset": 810, "offset": 816, "ep_type": "vote"},
                  {"onset": 860, "offset": 866}],
        na=((750, 752),),
        voice=[{"role": "me", "start": 430, "end": 432, "main": False},
               {"role": "me", "start": 470, "end": 478}, {"role": "prof", "start": 10, "end": 20}],
        words=[{"role": "me", "start": 470.5, "end": 472}, {"role": "me", "start": 474, "end": 477.5}])
    p = labels.points(CID, tb)
    assert reason(p, 100) == "opening"
    assert reason(p, 890) == "end"                       # t + 15 > T_end
    assert reason(p, 702) == "rec_gap" and reason(p, 730) != "rec_gap"
    assert reason(p, 230) == "breakout"                  # breakout segment
    assert reason(p, 200) is None or reason(p, 200) != "breakout"   # not yet available
    assert reason(p, 440) == "breakout"                  # Brady's non-main voice in [t-60, t]
    assert reason(p, 390) == "break"
    assert reason(p, 476) == "me_speaking"               # ongoing word-bearing Brady event
    assert reason(p, 480) == "me_speaking"               # Brady word ended at 477.5 > t - 3
    assert reason(p, 482) is None or reason(p, 482) != "me_speaking"
    assert reason(p, 522) == "hand_up"
    assert reason(p, 532) == "post_lowering" and reason(p, 540) == "post_lowering"
    assert reason(p, 542) != "post_lowering"
    assert reason(p, 602) == "onset_pending"             # onset 601.8, not yet live
    assert reason(p, 652) == "post_glitch"
    assert reason(p, 640) in ("glitch_window", "guard")
    assert reason(p, 644) == "glitch_window"
    assert reason(p, 740) == "unobservable"              # NA samples in (t, t + H]
    assert reason(p, 858) == "guard"                     # bid at 860 in (t, t + 2]
    assert reason(p, 800) == "vote_or_censored"
    assert reason(p, 846) is None and at(p, 846)[config.HEADLINE_LABEL] == 1
    q = labels.points(CID, scenario(T_end=900.0, qa_pass=False))
    assert reason(q, 300) == "class_qa" and reason(q, 100) == "opening"
    assert not q["at_risk"].any()
    assert q.loc[q["t_s"] == 300, config.HEADLINE_LABEL].iloc[0] == 0   # y ignores class QA


def test_first_matching_rule_wins():
    tb = scenario(segments=(("intro", 3.0), ("breakout", 50.0)))
    p = labels.points(CID, tb)
    assert reason(p, 100) == "opening"                   # opening precedes breakout


def test_left_censored_run_counts_as_hand_up_and_makes_windows_undefined():
    tb = scenario(episodes=[{"onset": 300, "offset": 310, "left_censored": True}])
    p = labels.points(CID, tb)
    assert reason(p, 304) == "hand_up"
    assert reason(p, 290) == "vote_or_censored"
    assert np.isnan(at(p, 290)[config.HEADLINE_LABEL])


# --------------------------------------------------------------------------- label rules


def test_asymmetric_observability():
    # NA samples after the onset: positives need (t, o*] only; negatives need (t, t + H]
    tb = scenario(episodes=[{"onset": 300, "offset": 306}], na=((301, 303), (405, 407)))
    p = labels.points(CID, tb)
    sym = config.y_col(15, 2, sym=True)
    for t in (286, 290, 296):
        assert at(p, t)[config.HEADLINE_LABEL] == 1
        assert np.isnan(at(p, t)[sym])                   # symmetric rule: undefined
    assert np.isnan(at(p, 396)[config.HEADLINE_LABEL])   # negative window has NA
    assert reason(p, 396) == "unobservable"
    assert at(p, 380)[config.HEADLINE_LABEL] == 0 and at(p, 380)[sym] == 0


def test_tile_not_found_at_t_is_unobservable():
    tb = scenario(na=((349.5, 350.5),))
    p = labels.points(CID, tb)
    assert reason(p, 350) == "unobservable"


def test_short_na_gaps_are_bridged():
    obs = labels.observability(scenario(na=((300.5, 301.0),))["hand_frames"])
    assert obs.observed(np.array([290.0]), np.array([305.0]))[0]     # 2 samples = 1 s: bridged
    obs = labels.observability(scenario(na=((300.5, 302.0),))["hand_frames"])
    assert not obs.observed(np.array([290.0]), np.array([305.0]))[0]


def test_guard_sweep_g_columns():
    tb = scenario(episodes=[{"onset": 301, "offset": 306}])
    p = labels.points(CID, tb)
    for g in config.G_LABELS:
        col = config.y_col(15, g)
        pos = p.loc[p[col] == 1, "t_s"].to_numpy()
        lo, hi = 301 - 15, 301 - g                       # t in [o - H, o - g)
        expect = p["t_s"][(p["t_s"] >= lo) & (p["t_s"] < hi)].to_numpy()
        assert np.array_equal(pos, expect), g
        guard = p.loc[(p["t_s"] >= hi) & (p["t_s"] < 301), col]
        assert guard.isna().all(), g
    assert config.y_col(15, 1) in p and config.y_col(15, 3) in p       # A1 columns


def test_horizon_sweep_columns():
    tb = scenario(episodes=[{"onset": 401, "offset": 406}])
    p = labels.points(CID, tb)
    for H in config.H_SWEEP:
        col = config.y_col(H, 2)
        n = int((p[col] == 1).sum())
        assert n == len(np.arange(np.ceil((401 - H) / 2) * 2, 401 - 2, 2)), H


def test_glitch_window_is_undefined_and_refractory():
    tb = scenario(episodes=[{"onset": 400, "offset": 400.5, "ep_type": "glitch", "eligible": False}])
    p = labels.points(CID, tb)
    for t in (386, 390, 396):
        assert np.isnan(at(p, t)[config.HEADLINE_LABEL])
    assert reason(p, 390) == "glitch_window"
    assert reason(p, 402) == "post_glitch" and reason(p, 410) == "post_glitch"
    assert reason(p, 412) is None


def test_votes_are_undefined_not_positive_and_sensitivities():
    tb = scenario(episodes=[{"onset": 400, "offset": 406, "ep_type": "vote", "contested": True}])
    p = labels.points(CID, tb)
    assert not (p[config.HEADLINE_LABEL] == 1).any()
    sets = labels.episode_sets(tb["hand_episodes"], votes_as_bids=True)
    assert list(sets.bids) == [400] and not len(sets.votes)
    sets = labels.episode_sets(tb["hand_episodes"], contested_as_bids=True)
    assert list(sets.bids) == [400]


def test_retimed_onsets_are_used_unless_retiming_failed():
    eps = scenario(episodes=[{"onset": 400, "offset": 406}])["hand_episodes"]
    eps["onset_retimed_s"] = 399.7
    assert labels.onset_times(eps)[0] == 399.7
    eps["retime_failed"] = True
    assert labels.onset_times(eps)[0] == 400


def test_onset_pending_without_live_state():
    tb = scenario(episodes=[{"onset": 401.2, "offset": 410}])
    p = labels.points(CID, tb)
    # samples UP from 401.5; live at 402.0; t = 402 sees live UP -> hand_up
    assert reason(p, 402) == "hand_up"
    tb = scenario(episodes=[{"onset": 401.8, "offset": 410}])
    p = labels.points(CID, tb)
    assert reason(p, 402) == "onset_pending"


# --------------------------------------------------------------------------- T-speech


def onsets(rows):
    return pd.DataFrame(rows, columns=["onset_s", "turn_end_s", "fresh"])


def test_tspeech_target_and_30s_rule():
    on = onsets([(300.0, 330.0, True), (400.0, 410.0, True), (420.0, 425.0, False)])
    t = np.arange(280.0, 440.0, 2.0)
    y = pd.Series(labels.speech_labels(t, on), index=t)
    # fresh onset at 400: positive for t in [385, 397)
    assert (y.loc[386:396] == 1).all()
    assert y.loc[[398.0, 400.0]].isna().all()        # onset in (t, t + 3]: guard
    # negatives need t + 15 - last_me_end >= 30 (last turn ended at 330)
    assert np.isnan(y.loc[340.0])                    # 355 - 330 < 30
    assert y.loc[346.0] == 0 and y.loc[370.0] == 0   # 361 - 330 >= 30
    # a non-fresh onset in the window makes it undefined
    assert np.isnan(y.loc[410.0])
    assert np.isnan(y.loc[302.0])                    # ongoing turn: 30 s rule fails


def test_tspeech_has_no_hand_rules():
    tb = scenario(episodes=[{"onset": 300, "offset": 320}], na=((330, 360),))
    p = labels.points(CID, tb)
    col = labels.SPEECH_LABEL
    assert at(p, 310)[col] == 0                      # hand up, NA tile: speech target defined
    assert np.isnan(at(p, 310)[config.HEADLINE_LABEL])


def test_cli_writes_tables(tmp_path, monkeypatch, truth):
    from types import SimpleNamespace

    from oncue import causal

    tables_dir = tmp_path / "tables"
    tables_dir.mkdir()
    cid, tb = next(iter(truth.items()))
    for name in ("classes", "words", "voice", "segments", "hand_frames", "hand_episodes"):
        tb[name].to_parquet(tables_dir / f"{name}.parquet", index=False)
    monkeypatch.setattr(config, "TABLES_DIR", tables_dir)
    monkeypatch.setattr(causal.config, "TABLES_DIR", tables_dir)
    assert labels.cli(SimpleNamespace(class_ids=None, extra=[])) == 0
    pts = pd.read_parquet(tables_dir / "points.parquet")
    schemas.validate(pts, "points", strict=True)
    assert set(pts["class_id"]) == {cid}
    assert (tables_dir / "speech_onsets.parquet").exists()


@pytest.mark.parametrize("H", [5, 60])
def test_end_rule_depends_on_horizon(H):
    tb = scenario(T_end=600.0)
    p = labels.points(CID, tb)
    col = config.y_col(H, 2)
    assert p.loc[p["t_s"] + H > 600, col].isna().all()


def test_class_without_hand_frames_is_unobservable_but_keeps_tspeech():
    tb = scenario(T_end=400.0)
    tb.pop("hand_frames")
    tb["hand_episodes"] = tb["hand_episodes"].iloc[0:0]
    p = labels.points(CID, tb)
    schemas.validate(p, "points", strict=True)
    late = p["t_s"] >= config.T_OPEN_S
    assert (p.loc[late & (p["t_s"] + 15 <= 400), "excl_reason"] == "unobservable").all()
    assert p[config.HEADLINE_LABEL].isna().all()
    assert (p.loc[late & (p["t_s"] + 15 <= 400), labels.SPEECH_LABEL] == 0).all()


def test_missing_t_end_uses_the_data_extent():
    tb = scenario(T_end=300.0)
    tb["classes"]["T_end_s"] = np.nan
    p = labels.points(CID, tb)
    assert p["t_s"].iloc[-1] == 300.0


# --------------------------------------------------------------------------- hand rules from samples <= t


def coded(codes: str, *, T_end: float | None = None, found=None, times=None) -> dict:
    """Tables for class CID from a 2 fps hand-state string (U/D/N); episodes are the
    detector's own (hands.episodes)."""
    from oncue import hands

    states = {"U": UP, "D": DOWN, "N": NA}
    raw = np.array([states[c] for c in codes], dtype=object)
    t = np.arange(len(raw)) * 0.5 if times is None else np.asarray(times, float)
    hf = pd.DataFrame({"class_id": CID, "t_s": t, "n_tiles": 10,
                       "tile_found": (raw != NA) if found is None else np.asarray(found, bool),
                       "tile_idx": pd.array([0] * len(t), "Int64"), "shape_ok": raw == UP,
                       "n_others": 0, "raw_state": raw, "live_state": live_states(raw)})
    tb = scenario(T_end=float(t[-1]) if T_end is None else T_end)
    tb["hand_frames"] = hf
    tb["hand_episodes"] = hands.episodes(hf)
    return tb


def test_hand_rules_do_not_depend_on_a_later_reraise():
    """Identical samples up to t = 311: whether the hand goes up again at 312.5 (a toggle
    merged label-side into one episode) must not change post_lowering, me_hand_* or the
    at-risk status at t (they come from live exits, not merged episodes)."""
    from oncue import features as F

    base = "D" * 600 + "U" * 20 + "D" * 5
    a = coded(base + "U" * 25 + "D" * 300)
    b = coded(base + "D" * 325)
    assert a["hand_episodes"]["n_toggles"].iloc[0] == 1          # label side: merged
    ts = np.array([310.5, 311.0, 312.0])
    ra, rb = labels.causal_rules(ts, a), labels.causal_rules(ts, b)
    pd.testing.assert_frame_equal(ra, rb)
    assert ra["post_lowering"].all()
    enc = F.HashEncoder(8)
    cols = list(F.GROUP_COLUMNS["me_hand"])
    pd.testing.assert_frame_equal(F.build(CID, ts, a, encoder=enc)[cols],
                                  F.build(CID, ts, b, encoder=enc)[cols])
    pa, pb = labels.points(CID, a, t=ts), labels.points(CID, b, t=ts)
    assert pa["excl_reason"].tolist() == pb["excl_reason"].tolist() == ["post_lowering"] * 3


def test_inner_lowering_of_a_toggle_starts_r_low():
    """A lowering inside a toggle-merged episode is a live exit too: the point before the
    re-raise is excluded, never a forced negative."""
    tb = coded("D" * 600 + "U" * 40 + "D" * 5 + "U" * 35 + "D" * 300)
    p = labels.points(CID, tb)
    assert reason(p, 322) == "post_lowering"
    assert np.isnan(at(p, 322)[config.HEADLINE_LABEL])


@pytest.mark.parametrize("tracked", [True, False])
def test_one_second_of_na_inside_a_raise_keeps_hand_up(tracked):
    """A 1 s NA (poll tick: tracked; or a tracking flicker) inside an UP run holds the live
    state: the point is excluded as hand_up, not an at-risk negative."""
    codes = "D" * 600 + "U" * 20 + "NN" + "U" * 20 + "D" * 300
    found = np.ones(len(codes), bool) if tracked else None
    tb = coded(codes, found=found)
    p = labels.points(CID, tb, t=np.arange(309.0, 312.5, 0.5))
    assert (p["excl_reason"] == "hand_up").all() and not p["at_risk"].any()


def test_lowering_followed_by_na_still_starts_r_low():
    """Lowered, then the tile is lost (pagination) before the lowering is confirmed: the
    live state exits UP into NA after 1 s and R_low starts there."""
    tb = coded("D" * 600 + "U" * 40 + "D" + "N" * 8 + "D" * 300)
    p = labels.points(CID, tb, t=np.arange(318.0, 334.0, 2.0))
    assert [reason(p, t) for t in (322, 326, 330)] == ["post_lowering"] * 3
    assert reason(p, 332) is None


def test_a_gap_in_hand_frames_is_not_observed():
    """Rows missing for 400-460 s: windows inside the gap have no samples, so no confident
    negative; and the last sample before the gap is not reused at t."""
    t = np.r_[np.arange(0, 400, 0.5), np.arange(460, 900, 0.5)]
    tb = coded("D" * len(t), times=t, T_end=900.0)
    p = labels.points(CID, tb)
    inside = p[(p["t_s"] > 400) & (p["t_s"] + 15 < 460)]
    assert len(inside) and inside[config.HEADLINE_LABEL].isna().all()
    assert (inside["excl_reason"] == "unobservable").all()
    obs = labels.observability(tb["hand_frames"])
    assert not obs.tile_found_at(np.array([420.0]))[0] and obs.tile_found_at(np.array([399.6]))[0]
    assert obs.observed(np.array([380.0]), np.array([395.0]))[0]           # before the gap
    assert not obs.observed(np.array([395.0]), np.array([410.0]))[0]       # reaches into it
    assert reason(p, 470) is None and at(p, 470)[config.HEADLINE_LABEL] == 0


def _random_codes(rng, n=2400) -> str:
    """Raises, toggles, blips, short and long NA runs at 2 fps (invented)."""
    out = []
    while len(out) < n:
        kind = rng.choice(["D", "U", "blip", "na", "na_long", "toggle"],
                          p=[0.45, 0.2, 0.1, 0.1, 0.05, 0.1])
        if kind == "D":
            out += ["D"] * int(rng.integers(4, 60))
        elif kind == "U":
            out += ["U"] * int(rng.integers(2, 40))
        elif kind == "blip":
            out += ["U"] + ["D"] * int(rng.integers(1, 10))
        elif kind == "na":
            out += ["N"] * int(rng.integers(1, 3))
        elif kind == "na_long":
            out += ["N"] * int(rng.integers(3, 20))
        else:
            out += ["U"] * int(rng.integers(2, 10)) + ["D"] * int(rng.integers(1, 6)) \
                + ["U"] * int(rng.integers(2, 10))
    return "".join(out[:n])


def test_hand_rules_unchanged_when_frames_and_episodes_are_rebuilt_from_truncated_samples():
    """Mask causality (§4.4 rule 4) with nothing precomputed: for random t, the hand rules
    from the full tables equal those from hand_frames truncated at t, with the episodes
    re-detected from the truncated samples only."""
    from oncue import hands

    rng = np.random.default_rng(156)
    for _ in range(3):
        tb = coded(_random_codes(rng))
        hf = tb["hand_frames"]
        full = labels.causal_rules(hf["t_s"].to_numpy(), tb)
        for t in rng.uniform(5.0, float(hf["t_s"].max()), 60):
            part = hf[hf["t_s"] <= t].copy()
            part["live_state"] = live_states(part["raw_state"])
            tr = dict(tb, hand_frames=part, hand_episodes=hands.episodes(part))
            got = labels.causal_rules(np.array([t]), tr)
            want = labels.causal_rules(np.array([t]), tb)
            pd.testing.assert_frame_equal(got[["hand_up", "post_lowering", "post_glitch"]],
                                          want[["hand_up", "post_lowering", "post_glitch"]])
        assert full["post_lowering"].any() and full["post_glitch"].any()


def test_assign_folds_once_over_dev_classes_with_twins_together():
    """§8.1 / L18: points.fold is assigned once over all dev classes; same-section
    same-date twins (shared cv_group) always share a fold; vault rows stay <NA>."""
    rows = []
    for c in range(12):
        rows.append({"class_id": 100 + c, "course_code": f"CS{c % 2}",
                     "in_vault": c == 11, "cv_group": 100 if c in (0, 1) else 100 + c})
    classes = pd.DataFrame(rows)
    pts = pd.DataFrame({"class_id": np.repeat(classes["class_id"], 3),
                        "t_s": np.tile([120.0, 122.0, 124.0], 12),
                        "fold": pd.array([pd.NA] * 36, dtype="Int64")})
    out = labels.assign_folds(pts, classes)
    per = out.groupby("class_id")["fold"].first()
    assert per.loc[100] == per.loc[101]
    assert pd.isna(per.loc[111]) and per.drop(111).notna().all()
    assert sorted(per.drop(111).unique()) == list(range(config.N_OUTER_FOLDS))
    tiny = labels.assign_folds(pts[pts["class_id"] < 103], classes)    # < 5 dev classes
    assert tiny["fold"].isna().all()
