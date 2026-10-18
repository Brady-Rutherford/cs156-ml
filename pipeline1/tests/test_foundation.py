"""Foundation tests: config constants and rules, schemas, the synthetic world, the CLI."""

from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
import sys

import cv2
import numpy as np
import pandas as pd
import pytest
from oncue import __main__ as cli
from oncue import config as C
from oncue import schemas, synthetic

# --------------------------------------------------------------------------- config


def test_constants_match_spec():
    assert (C.DELTA_S, C.H_S, C.G_S, C.G_SPEECH_S) == (2.0, 15, 2, 3)
    assert C.H_SWEEP == (5, 10, 15, 30, 60)
    assert (C.EPS_W_S, C.T_OPEN_S, C.R_LOW_S, C.S_ME_S) == (0.25, 120.0, 10.0, 3.0)
    assert (C.TURN_MERGE_GAP_S, C.FRESH_SILENCE_S, C.F_SCAN) == (2.0, 30.0, 2.0)
    assert (C.THETA_ON, C.THETA_OFF, C.K_CONFIRM, C.D_PRE_S) == (0.20, 0.08, 2, 2.0)
    assert (C.SHAPE_CIRCULARITY_MIN, C.SHAPE_GLYPH_FRAC) == (0.6, (0.08, 0.45))
    assert (C.DOWN_GAP_MERGE_S, C.MIN_EPISODE_S, C.ME_HAND_MIN_S) == (3.0, 1.0, 2.0)
    assert (C.NCC_MIN, C.NCC_GAP, C.GLYPH_H, C.GLYPH_W) == (0.75, 0.10, 12, 48)
    assert (C.FRAME_PAUSE_S, C.FRAME_MAX_S, C.KAPPA) == (0.4, 5.0, 2.0)
    assert C.EMB == "BAAI/bge-small-en-v1.5"
    assert C.FAR_BUDGETS == (0.1, 0.5)
    assert (C.SEED, C.VAULT_SALT) == (156, "oncue-vault")
    assert C.alarm_refractory() == 13.0
    assert C.HAND_H_RANGE == (26, 40) and C.TICK_H_RANGE == (60, 82) and C.HAND_SV_MIN == 115


def test_amendments_applied():
    assert C.G_SWEEP == (1, 2, 3, 5)                                       # A1
    spec_labels = {"y_H15_g2", "y_H5_g2", "y_H10_g2", "y_H30_g2", "y_H60_g2", "y_H15_g0",
                   "y_H15_g1", "y_H15_g5", "y_H15_g2_sym", "y_speech_H15_g3"}
    assert spec_labels | {"y_H15_g3"} == set(C.LABEL_COLUMNS)              # A1
    assert C.HEADLINE_LABEL == "y_H15_g2" and C.LABEL_COLUMNS[0] == C.HEADLINE_LABEL
    assert all(C.feature_allowed(c) for c in C.HIST_COLUMNS)              # A2
    assert len(C.EDU_CHECKPOINTS) == 5 and all(                            # A3
        v.startswith("StanfordSCALE/assertion_sentence_") and C.feature_allowed(k)
        for k, v in C.EDU_CHECKPOINTS.items())
    assert all(C.feature_allowed(c) for c in C.STACK_COLUMNS)              # A4
    assert "text" not in C.KNN_OUTPUT_COLUMNS                               # A5
    assert C.asr_provider("CS156") == "deepgram"                            # A8
    assert C.asr_provider("SS999") == "local"                               # A8
    assert C.CALIBRATION_METHOD == "sigmoid"                                # A9
    assert C.GRU_PARAM_RANGE == (10_000, 50_000)


def test_vote_threshold_and_eps_f():
    assert [C.vote_threshold(n) for n in (2, 11, 15, 21, 31)] == [4, 4, 5, 6, 9]
    assert C.eps_f("s") == 1.0 and C.eps_f("ms") == 0.5
    assert C.eps_f("ms", 0.8, 0.3) == 0.8 and C.eps_f("ms", float("nan"), 1.2) == 1.2
    assert C.eps_f_poll(0.5) == 1.0 and C.eps_f_poll(1.4) == 1.4


def _cal(**kw):
    cal = {"status": "ok", "applied": True, "method": "windowed", "windows": [[0, 0, 1]] * 8,
           "max_residual_s": 0.4}
    cal.update(kw)
    return cal


@pytest.mark.parametrize("cal, med, p95, ok, reason", [
    (_cal(), 0.2, 0.9, True, "ok"),
    (_cal(status="piecewise"), 0.2, 0.9, True, "ok"),
    (_cal(method="global", max_residual_s=0.0, windows=[]), 0.1, 0.2, False, "cal_method_global"),
    (_cal(windows=[[0, 0, 1]] * 5), 0.2, 0.9, False, "cal_few_windows"),
    (_cal(applied=False), 0.2, 0.9, False, "cal_not_applied"),
    (_cal(status="unverified"), 0.2, 0.9, False, "cal_status_unverified"),
    (_cal(max_residual_s=2.6), 0.2, 0.9, False, "cal_max_residual"),
    (_cal(), 0.2, 2.6, False, "resid_p95"),
    (_cal(), 0.6, 0.9, False, "resid_median"),
    (_cal(), 0.2, None, False, "resid_p95"),
    (None, 0.2, 0.9, False, "no_calibration"),
])
def test_calibration_qa(cal, med, p95, ok, reason):
    assert C.calibration_qa(cal, med, p95) == (ok, reason)


def test_calibration_qa_sensitivity_admits_3s():
    assert C.calibration_qa(_cal(max_residual_s=2.5), 0.3, 2.5, sensitivity=True)[0]
    assert not C.calibration_qa(_cal(max_residual_s=3.5), 0.3, 2.5, sensitivity=True)[0]


@pytest.mark.parametrize("title, section, expected", [
    ("Breakout instructions", "Part 2", "breakout_instr"),
    ("Breakouts", "Part 2", "breakout"),
    ("Eye rest break", "", "break"),
    ("Poll: first impressions", "", "poll"),
    ("Debrief", "", "debrief"),
    ("Discussion of the readings", "", "debrief"),
    ("Warm-up and recap", "Opening", "intro"),
    ("Wrap-up and reflection", "Closing", "wrapup"),
    ("Case activity", "", "activity"),
    ("Something else", "Misc", "other"),
    ("Debrief", "Breakout room", "breakout"),       # section title is matched first
])
def test_seg_type(title, section, expected):
    assert C.seg_type_of(title, section) == expected


def test_lexicons():
    assert C.INVITE_RE.search("does anyone want to share")
    assert C.INVITE_RE.search("whod like to start") and C.INVITE_RE.search("who'd like to go")
    assert C.INVITE_RE.search("any thoughts on the model")
    assert not C.INVITE_RE.search("the gradient is steep")
    assert C.VOTE_RE.search("raise your hand if you agree")
    assert C.VOTE_RE.search("lets vote") and C.VOTE_RE.search("show of hands please")
    assert C.CALLON_RE.search("go ahead") and C.CALLON_RE.search("youre up")
    assert C.CALLON_RE.search("i see your hand") and not C.CALLON_RE.search("go a head")
    assert C.CHECK_RE.search("does that make sense")
    assert C.Q_FIRST_RE.search("why does this matter") and not C.Q_FIRST_RE.search("so why")
    assert C.Q_TAG_RE.search("that is the point right") and C.Q_TAG_RE.search("isnt it")


def test_feature_allowlist():
    allowed = ["ctx_min", "ctx_min_spl_0", "spk_silence_s", "me_turns_so_far",
               "me_hand_n_so_far", "poll_res_my", "q_open", "q_vote_lex", "pcw_res_Q",
               "pcw_unspent", "lo_sim", "sg_sim", "n_words_15", "wpm_60", "topic_shift",
               "own_sim", "hist_bid_rate_course", "edu_invites", "stack_m4"]
    for col in allowed:
        assert C.feature_allowed(col), col
    blocked = ["y_H15_g2", "next_bid_s", "excl_reason", "label_state", "fold", "in_vault",
               "tile_found", "coverage", "pcw_edited_after_start", "pcw_status", "cal_offset",
               "eps_F_s", "av_lag_s", "hand_log_lag", "participation", "analytics_x",
               "class_id", "t_s", "date", "emb_row", "n_others", "border_any", "in_G",
               "at_risk", "random_col", "pcw_mystery"]
    for col in blocked:
        assert not C.feature_allowed(col), col
    assert C.feature_allowed("has_qmark", "F-punct") and not C.feature_allowed("has_qmark")
    assert C.feature_group("me_hand_since_down_s") == "me_hand"
    assert C.feature_group("me_since_turn_s") == "me"


def test_raw_path_whitelist():
    w = C.whitelist_use
    assert w("events", "[].event-data.duration") == C.USE_FEATURE
    assert w("events", "[].event-data.message-text") == C.USE_FEATURE
    assert w("events", "[].actor.first-name") is None              # names never via events
    assert w("class", "recording-sessions[].recording-ended") == C.USE_LABEL
    assert w("class", "class-users[].user.first-name") == C.USE_MASK
    assert w("class", "actual-length-seconds") is None
    assert w("class", "class-users[].absent") == C.USE_META
    assert w("pcw_doc", "answers.9b1c-77aa.value") == C.USE_FEATURE
    assert w("pcw_doc", "answers.9b1c-77aa.files[].url") is None
    assert w("pcw_doc", "answers.9b1c-77aa.updated-at") is None    # per-block timestamps
    assert w("pcw_doc", "comments[].text") is None
    assert w("pcw_template", "blocks[].block-components[].value") == C.USE_FEATURE
    assert w("asr", "words[].text") == C.USE_FEATURE and w("asr", "words[].cluster") == "meta"
    assert w("transcript", "events[].type") is None
    assert w("transcript", "participation") is None
    assert w("transcript", "alignment.calibration.offset") == C.USE_TIMING
    for blocked in ("analytics", "outcome_assessments", "focused_outcomes"):
        assert w(blocked, "[].id") is None
    assert set(C.PATH_USES) >= {u for paths in C.RAW_PATH_WHITELIST.values()
                                for u in paths.values()}
    assert not C.EVENT_TYPES & C.BLOCKED_EVENT_TYPES
    assert C.EVENT_TYPES == {"voice", "message", "timeline-segment"}


def test_paths_are_inside_pipeline1():
    for p in (C.DATA_DIR, C.TABLES_DIR, C.CACHE_DIR, C.LOGS_DIR, C.TRANSCRIPTS_DIR,
              C.RESULTS_DIR, C.FIGURES_DIR, C.AUDIT_DIR, C.RAW_DIR, C.EMB_CACHE_DIR):
        assert C.PIPELINE_DIR in p.parents
    assert C.DATA_DIR == C.PIPELINE_DIR / "data"
    assert C.table_path("words").name == "words.parquet"
    assert C.raw_class_dir(900101) == C.RAW_DIR / "900101"


def test_id_salt_read_at_call_time(monkeypatch):
    with pytest.raises(RuntimeError):
        C.id_salt()
    with pytest.raises(RuntimeError):
        C.hash_id(900101)
    monkeypatch.setenv("ONCUE_ID_SALT", "salt-one")
    h1 = C.hash_id(900101)
    assert len(h1) == 12 and int(h1, 16) >= 0 and h1 == C.hash_id("900101")
    monkeypatch.setenv("ONCUE_ID_SALT", "salt-two")
    assert C.hash_id(900101) != h1                       # not cached
    monkeypatch.delenv("ONCUE_ID_SALT")
    monkeypatch.setenv("ID_SALT", "salt-one")
    assert C.hash_id(900101) == h1                       # fallback name
    source = inspect.getsource(C)
    assert "salt-one" not in source and "ID_SALT =" not in source


def test_vault_rule_deterministic_and_about_20_percent():
    ids = np.arange(100_000, 105_000)
    flags = np.array([C.in_vault(i) for i in ids])
    assert np.array_equal(flags, np.array([C.in_vault(i) for i in ids]))
    expected = [int(hashlib.sha1(f"{i}:oncue-vault".encode()).hexdigest(), 16) % 5 == 0
                for i in ids[:200]]
    assert list(flags[:200]) == expected
    assert 0.18 < flags.mean() < 0.22
    assert C.in_vault(123456) == C.in_vault("123456")


def test_g2_rules():
    ok, detail = C.g2_pass(285, 300, 50, 55)
    assert ok, detail
    assert not C.g2_pass(285, 300, 34, 38)[0]                # recall n < 40
    assert not C.g2_pass(260, 300, 50, 55)[0]                # precision point < 0.90
    assert not C.g2_pass(285, 300, 50, 55, wrong_tile_share=0.02)[0]
    assert not C.g2_pass(285, 300, 50, 55, left_censored_share=0.11)[0]
    lo, hi = C.wilson_interval(45, 50)
    assert 0.78 < lo < 0.79 and 0.95 < hi < 0.97
    assert C.wilson_interval(0, 0) == (0.0, 1.0)


def test_asr_provider_map():
    assert C.asr_provider("SS110") == "local"
    assert C.asr_provider("CS156", {"CS156": "deepgram"}) == "deepgram"
    with pytest.raises(ValueError):
        C.asr_provider("CS156", {"CS156": "groq"})


def test_term_and_area():
    assert C.term_code("Fall 2023") == "F23" and C.term_code("Spring 2026") == "Sp26"
    assert C.term_code("F25") == "F25" and C.term_code("Summer 2024") is None
    assert [C.SEMESTER_IDX[t] for t in ("F23", "F25", "Sp26", "F26")] == [1, 5, 6, 7]
    assert [C.course_area(c) for c in ("CS156", "SS51", "NS110", "AH112", "B154", "XX1")] == \
        ["CS", "SS", "NS", "AH", "B", "other"]


# --------------------------------------------------------------------------- schemas

SPEC_TABLES = ("classes", "words", "frames", "voice", "segments", "chat", "polls", "pcw_blocks",
               "hand_tiles", "hand_frames", "hand_episodes", "hand_log", "speech_onsets",
               "points", "features", "oof", "results")


def test_every_spec_table_has_a_schema():
    assert set(SPEC_TABLES) <= set(schemas.TABLE_NAMES)
    assert "end_s" not in schemas.columns("segments")              # no segment end (L10)
    assert set(C.LABEL_COLUMNS) <= set(schemas.columns("points"))
    assert {"brier_skill", "recall_random_far01"} <= set(schemas.columns("results"))


@pytest.mark.parametrize("name", schemas.TABLE_NAMES)
def test_empty_tables_validate(name):
    df = schemas.empty(name)
    assert list(df.columns) == schemas.columns(name)
    schemas.validate(df, name, strict=True)


def test_validate_reports_missing_and_bad_columns():
    df = schemas.empty("voice").drop(columns=["word_bearing"])
    with pytest.raises(schemas.SchemaError, match="word_bearing"):
        schemas.validate(df, "voice")
    bad = schemas.empty("voice")
    bad["start_s"] = pd.Series(["x"], dtype=object).iloc[:0]
    bad = pd.concat([bad, pd.DataFrame({"start_s": ["not a time"]})], ignore_index=True)
    with pytest.raises(schemas.SchemaError, match="start_s"):
        schemas.validate(bad, "voice")
    with pytest.raises(schemas.SchemaError, match="unexpected"):
        schemas.validate(schemas.empty("chat").assign(extra=1), "chat", strict=True)
    with pytest.raises(schemas.SchemaError, match="unknown table"):
        schemas.validate(pd.DataFrame(), "nope")


def test_features_table_enforces_allowlist():
    ok = pd.DataFrame({"class_id": [1], "t_s": [120.0], "emb_row": [0], "ctx_min": [2.0],
                       "q_open": [1.0]})
    schemas.validate(ok, "features")
    assert schemas.feature_columns(ok) == ["ctx_min", "q_open"]
    with pytest.raises(schemas.SchemaError, match="pcw_status"):
        schemas.validate(ok.assign(pcw_status=1.0), "features")
    with pytest.raises(schemas.SchemaError, match="y_H15_g2"):
        schemas.validate(ok.assign(y_H15_g2=1.0), "features")


def test_truth_tables_validate(truth, tmp_path):
    for cid, tables in truth.items():
        for name, df in tables.items():
            schemas.validate(df, name, strict=True)
            assert (df["class_id"] == cid).all()
    tables = next(iter(truth.values()))
    for name in ("words", "hand_episodes", "classes", "hand_frames"):
        path = tmp_path / f"{name}.parquet"
        tables[name].to_parquet(path)
        back = pd.read_parquet(path)
        schemas.validate(back, name, strict=True)
        assert len(back) == len(tables[name])


# --------------------------------------------------------------------------- synthetic world


def test_world_is_deterministic_and_prefix_stable(world):
    again = synthetic.world(6, C.SEED)
    for a, b in zip(world, again, strict=True):
        pd.testing.assert_frame_equal(a.words, b.words)
        assert a.raw["events"] == b.raw["events"]
        assert a.raw["class"] == b.raw["class"]
    small = synthetic.world(2, C.SEED)
    pd.testing.assert_frame_equal(small.classes[0].words, world.classes[0].words)
    assert synthetic.world(2, C.SEED + 5).classes[0].raw["events"] != \
        world.classes[0].raw["events"]


def test_world_variety(world):
    meta = [sc.meta for sc in world]
    assert len({m["class_id"] for m in meta}) == len(world)
    assert any(m["multi_session"] for m in meta) and any(not m["precise"] for m in meta)
    assert {m["pcw_status"] for m in meta} >= {"verified", "none", "unknown"}
    assert any(m["calibration"]["method"] == "global" for m in meta)
    assert len({m["section_id"] for m in meta}) >= 2


def _key(ev, kebab_key):
    return ev.get(kebab_key, ev.get(kebab_key.replace("-", "_")))


def test_raw_class_and_events_shape(world):
    sc = world.classes[0]
    k = sc.raw["class"]
    assert {"id", "section", "class-users", "recording-sessions", "teacher-user-id"} <= set(k)
    assert k["section"]["course"]["course-code"] == sc.meta["course_code"]
    rs = k["recording-sessions"][0]
    assert rs["recording-started"].endswith("Z") and rs["recording-ended"].endswith("Z")
    events = sc.raw["events"]
    types = {_key(e, "event-type") for e in events}
    assert {"voice", "message", "timeline-segment", "bookmark-note"} <= types
    assert any("event-type" in e for e in events) and any("event_type" in e for e in events)
    sigs = {_key(e, "room-signature") for e in events if _key(e, "event-type") == "voice"}
    assert C.MAIN_ROOM_SIGNATURE in sigs and any(s.startswith("Breakout Room") for s in sigs)
    # system notice with no actor, and a name trap in chat
    assert any(_key(e, "event-type") == "message" and _key(e, "actor-id") is None
               for e in events)
    whole = world.classes[1]
    assert not whole.meta["precise"]
    assert all(_key(e, "timestamp") % 1000 == 0 for e in whole.raw["events"]
               if _key(e, "event-type") == "voice")


def test_voice_timestamp_is_end_and_scribe_recovers_truth(world):
    from scribe_core.calibrate import calibrate
    from scribe_core.forum_events import parse_voice_events, recording_timeline
    from scribe_core.fuse import calibration_words

    for sc in world:
        segs = recording_timeline(sc.raw["class"])
        voice = parse_voice_events(sc.raw["events"], segs)
        truth = sc.voice[sc.voice["room_main"] & (sc.voice["role"] != C.ROLE_PROF)]
        truth = truth.sort_values("start_s")   # Forum never logs the professor's mic (A12)
        assert len(voice) == len(truth)
        got = np.sort([v["start"] for v in voice]) + sc.meta["cal_offset_s"]
        tol = 1e-3 if sc.meta["precise"] else 1.0
        assert np.abs(got - truth["start_s"].to_numpy()).max() <= tol
        if sc.meta["precise"]:
            words = [{"start": w["start"], "end": w["end"], "cluster": w.get("cluster")}
                     for w in sc.asr["words"]]
            cal = calibrate(calibration_words(words, voice), voice, sc.T_end_s)
            assert cal["applied"] and abs(cal["offset"] - sc.meta["cal_offset_s"]) < 0.15


def test_asr_and_truth_words(world):
    sc = world.classes[0]
    words = sc.asr["words"]
    assert {"start", "end", "text", "punctuated_word", "conf", "cluster", "filler"} <= set(words[0])
    starts = [w["start"] for w in words]
    assert starts == sorted(starts)
    assert all(w["end"] > w["start"] for w in words)
    w = sc.words
    assert set(w["role"]) == {C.ROLE_PROF, C.ROLE_ME, C.ROLE_STUDENT}
    assert w["is_name"].any() and w["is_callon"].any() and w["text"].str.endswith("?").any()
    assert ((w["start_s"] >= 0) & (w["end_s"] <= sc.T_end_s)).all()
    tr = synthetic.scribe_transcript(sc)
    assert tr["schema"] == "scribe.transcript/v2" and tr["privacy"] == "ids"
    assert all("name" not in s for s in tr["speakers"].values())
    assert any(e["type"] == "hand-raise-key" for e in tr["events"])     # trap for loaders
    assert tr["participation"]                                          # trap for loaders


def test_episodes_with_questions_before_bids(world):
    eps = pd.concat([sc.episodes for sc in world])
    assert set(eps["ep_type"]) == set(C.EP_TYPES)
    bids = eps[eps["ep_type"] == C.EP_BID]
    assert len(bids) >= 3 * len(world)
    assert bids["has_question"].mean() >= 0.6
    q = bids[bids["has_question"]]
    gap = q["onset_s"] - q["q_end_s"]
    assert ((gap > 1.0) & (gap < 9.0)).all()
    assert set(bids["outcome"]) <= {"called", "withdrawn"}
    assert (bids["onset_s"] >= C.T_OPEN_S).all() and (bids["dur_s"] > 1.0).all()
    for sc in world:
        e = sc.episodes
        assert (e.loc[e["ep_type"] == C.EP_GLITCH, "dur_s"] < C.MIN_EPISODE_S).all()
        votes = e[e["ep_type"] == C.EP_VOTE]
        assert (votes["n_others_up"] >= C.vote_threshold(sc.meta["n_tiles"])).all()
        called = e[e["outcome"] == "called"]
        me_words = sc.words[sc.words["role"] == C.ROLE_ME]
        for o, lat in zip(called["onset_s"], called["call_latency_s"], strict=True):
            assert ((me_words["start_s"] >= o) & (me_words["start_s"] <= o + lat + 0.01)).any()


def test_pcw_template_and_doc(world):
    sc = world.classes[0]
    tmpl, doc = sc.raw["pcw_template"], sc.raw["pcw_doc"]
    block_ids = {b["id"] for b in tmpl["blocks"]}
    for b in tmpl["blocks"]:
        for comp in b["block_components"]:
            ops = json.loads(comp["value"])["ops"]
            assert all("insert" in op for op in ops)
    assert set(doc["answers"]) == block_ids
    assert {a["type"] for a in doc["answers"].values()} >= {"text"}
    by_status = {s.meta["pcw_status"]: s for s in world}
    assert by_status["none"].raw["pcw_template"] is None
    late = by_status["unknown"]
    assert late.raw["pcw_doc"]["updated_at"] > late.raw["class"]["start-time"]


def _hsv_masks(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype(int)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    sv = (s > C.HAND_SV_MIN) & (v > C.HAND_SV_MIN)
    hand = (h >= C.HAND_H_RANGE[0]) & (h <= C.HAND_H_RANGE[1]) & sv
    tick = (h >= C.TICK_H_RANGE[0]) & (h <= C.TICK_H_RANGE[1]) & sv
    white = (s < C.WHITE_S_MAX) & (v > C.WHITE_V_MIN)
    return hand, tick, white


def _roi(img, lay, k, s):
    side = round(s * lay["tile_h"])
    x0, y0 = lay["x0"][k] + C.ROI_INSET_PX, lay["tile_y0"] + C.ROI_INSET_PX
    return img[y0:y0 + side, x0:x0 + side]


def _largest_blob(mask):
    _n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    contours, _ = cv2.findContours((lab == k).astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    c = max(contours, key=cv2.contourArea)
    circularity = 4 * np.pi * cv2.contourArea(c) / cv2.arcLength(c, True) ** 2
    return circularity, cv2.minEnclosingCircle(c)


def test_frames_have_the_right_hues(world):
    sc = world.classes[0]
    lay = synthetic.strip_layout(sc)
    bid = sc.episodes[sc.episodes["ep_type"] == C.EP_BID].iloc[0]
    my_tick = sc.ticks[sc.ticks["user_id"] == sc.meta["me_id"]].iloc[0]
    t_up, t_down, t_tick = bid.onset_s + 1.0, bid.onset_s - 30.0, my_tick.start_s + 1.0
    assert synthetic.hand_state_at(sc, t_up)["me_up"]
    assert not synthetic.hand_state_at(sc, t_down)["me_up"]
    up, down, tick = synthetic.frames_for(sc, [t_up, t_down, t_tick])
    assert up.shape == (synthetic.FRAME_H, synthetic.FRAME_W, 3) and up.dtype == np.uint8
    me = lay["me_tile"]
    for s in (C.ROI_S1, C.ROI_S2):
        hand_up, tick_up, _ = _hsv_masks(_roi(up, lay, me, s))
        assert hand_up.mean() >= C.THETA_ON and tick_up.mean() == 0
        hand_down, _, _ = _hsv_masks(_roi(down, lay, me, s))
        assert hand_down.mean() < C.THETA_OFF / 2
        hand_t, tick_t, _ = _hsv_masks(_roi(tick, lay, me, s))
        assert tick_t.mean() >= C.THETA_ON and hand_t.mean() == 0
    # shape test: the icon is circular with a white glyph of the right share
    hand, _, white = _hsv_masks(_roi(up, lay, me, C.ROI_S1))
    circularity, ((cx, cy), r) = _largest_blob(hand)
    assert circularity >= C.SHAPE_CIRCULARITY_MIN
    yy, xx = np.mgrid[: hand.shape[0], : hand.shape[1]]
    inside = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
    lo, hi = C.SHAPE_GLYPH_FRAC
    assert lo <= (white & inside).sum() / inside.sum() <= hi
    # the lime distractor passes the hue test but fails the shape test
    dist, _, _ = _hsv_masks(_roi(down, lay, lay["distractor_tile"], C.ROI_S1))
    assert dist.mean() >= C.THETA_ON
    assert _largest_blob(dist)[0] < C.SHAPE_CIRCULARITY_MIN


def test_strip_gutters_and_layout(world):
    for sc in world:
        lay = synthetic.strip_layout(sc)
        strip = synthetic.frames_for(sc, [500.0], full=False)[0]
        assert strip.shape == (lay["strip_h"], synthetic.FRAME_W, 3)
        v = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)[..., 2].mean(axis=0) / 255.0
        dark = np.r_[0, (v < C.GUTTER_V_MAX).astype(int), 0]
        starts, ends = np.flatnonzero(np.diff(dark) == 1), np.flatnonzero(np.diff(dark) == -1)
        assert len(starts) == lay["n_tiles"] + 1
        assert ((ends - starts) >= C.GUTTER_MIN_PX).all()
        assert lay["users"][lay["me_tile"]] == sc.meta["me_id"]
        assert lay["labels"][lay["me_tile"]].startswith("Brady")
        assert lay["tile_w"] >= 30


def test_frames_deterministic_and_border(world):
    sc = world.classes[0]
    a = synthetic.frames_for(sc, [321.5, 900.0])
    b = synthetic.frames_for(sc, [321.5, 900.0])
    assert np.array_equal(a, b)
    v = sc.voice[sc.voice["room_main"] & (sc.voice["role"] == C.ROLE_STUDENT)].iloc[3]
    t = (v.start_s + v.end_s) / 2 - sc.meta["cal_offset_s"]
    state = synthetic.hand_state_at(sc, t)
    k = synthetic.strip_layout(sc)["users"].index(v.user_id)
    assert k in state["border"]


def test_truth_tables_are_consistent(truth, world):
    for sc in world:
        t = truth[sc.class_id]
        assert len(t["words"]) == len(sc.words)
        names = t["words"]["text_masked"].str.contains(r"\[NAME\]", regex=True)
        assert names.sum() == sc.words["is_name"].sum()
        assert not t["words"]["text_norm"].str.contains("brady|go ahead", regex=True).any()
        assert (t["words"]["role_avail_s"] >= t["words"]["end_s"] + C.EPS_W_S - 1e-9).all()
        assert t["classes"]["n_bids"].iloc[0] == (sc.episodes["ep_type"] == C.EP_BID).sum()
        assert not t["chat"]["text_masked"].str.contains(
            "|".join(sc.roster["first"])).any()
        frames = t["hand_frames"]
        assert set(frames["live_state"]) <= {C.STATE_UP, C.STATE_DOWN}
    qa = {cid: bool(t["classes"]["qa_pass"].iloc[0]) for cid, t in truth.items()}
    assert not qa[world.classes[2].class_id] and not qa[world.classes[-1].class_id]
    assert qa[world.classes[0].class_id]


def test_whitelist_drops_exactly_the_planted_traps(world):
    import re

    from scribe_core.forum_events import kebab

    def matches(rules, source, path):
        return any(src == source and re.fullmatch(rx, path) for src, rx in rules)

    for sc in world:
        sources = {**sc.raw, "asr": sc.asr}
        for source, obj in sources.items():
            if obj is None:
                continue
            for path in synthetic.flatten_paths(kebab(obj)):
                trap = matches(synthetic.TRAP_PATHS, source, path)
                kept = C.whitelist_use(source, path) is not None
                assert not (trap and kept), (source, path)
                if not kept:
                    assert trap or matches(synthetic.BOOKKEEPING_PATHS, source, path), \
                        (source, path)


def test_short_classes_build_and_validate(small_world):
    for sc in small_world:
        assert sc.T_end_s == 720.0 and (sc.episodes["ep_type"] == C.EP_BID).sum() >= 2
        for name, df in synthetic.truth_tables(sc).items():
            schemas.validate(df, name, strict=True)


def test_write_world_layout(data_dir, world):
    raw = data_dir["raw"]
    for sc in world:
        cdir = raw / str(sc.class_id)
        assert (cdir / "class.json").exists() and (cdir / "events.json").exists()
        assert (cdir / "_complete.json").exists() and (cdir / "class_edit_page.json").exists()
        base = synthetic.output_basename(sc)
        assert (data_dir["transcripts"] / f"{base}_asr.json").exists()
        assert (data_dir["transcripts"] / f"{base}_ids.json").exists()
    manifest = pd.read_csv(data_dir["manifest"])
    assert list(manifest.columns) == synthetic.MANIFEST_COLUMNS
    assert set(manifest["class_id"]) == set(world.class_ids)
    assert (data_dir["scribe"] / "me.json").exists()
    assert len(list((data_dir["scribe"] / "schedule").glob("*.json"))) == len(world.sections)


# --------------------------------------------------------------------------- CLI

SPEC_SUBCOMMANDS = ["harvest", "harvest-import", "select", "transcribe", "words", "events",
                    "scan", "track", "episodes", "retime", "labels", "frames", "embed",
                    "features", "train", "evaluate", "controls", "figures", "hand-log-import",
                    "results-bundle", "privacy"]


INTEGRATION_SUBCOMMANDS = {"pcw", "qa", "e2e", "ingest", "ingest-report", "audit", "pdf",
                           "ringcheck", "vectors", "flex"}     # added after the spec


def test_cli_lists_every_spec_subcommand():
    names = list(cli.COMMANDS)
    assert [n for n in names if n in SPEC_SUBCOMMANDS] == SPEC_SUBCOMMANDS
    assert set(names) - set(SPEC_SUBCOMMANDS) == INTEGRATION_SUBCOMMANDS


def test_cli_missing_stage_exits_2(capsys, monkeypatch):
    monkeypatch.setitem(cli.COMMANDS, "select", ("not_written_yet_xyz", "cli", "test"))
    assert cli.main(["select", "--phase", "A"]) == cli.EXIT_NOT_IMPLEMENTED
    assert "not implemented yet" in capsys.readouterr().err
    assert cli.main([]) == cli.EXIT_NOT_IMPLEMENTED


def test_cli_dispatches_lazily(monkeypatch):
    seen = {}

    def fake_cli(args):
        seen.update(vars(args))
        return 0

    module = type(sys)("oncue.fake_stage")
    module.cli = fake_cli
    monkeypatch.setitem(sys.modules, "oncue.fake_stage", module)
    monkeypatch.setitem(cli.COMMANDS, "select", ("fake_stage", "cli", "test"))
    rc = cli.main(["select", "--phase", "B", "--workers", "3", "--class", "9", "--class", "10",
                   "--extra-flag", "x"])
    assert rc == 0
    assert seen["phase"] == "B" and seen["workers"] == 3 and seen["class_ids"] == ["9", "10"]
    assert seen["extra"] == ["--extra-flag", "x"] and seen["command"] == "select"


def test_cli_missing_function_exits_2(monkeypatch, capsys):
    module = type(sys)("oncue.half_stage")
    monkeypatch.setitem(sys.modules, "oncue.half_stage", module)
    monkeypatch.setitem(cli.COMMANDS, "figures", ("half_stage", "cli", "test"))
    assert cli.main(["figures"]) == cli.EXIT_NOT_IMPLEMENTED
    assert "oncue.half_stage.cli" in capsys.readouterr().err


def test_python_dash_m_runs():
    proc = subprocess.run([sys.executable, "-m", "oncue", "--help"], capture_output=True,
                          text=True, cwd=C.PIPELINE_DIR, timeout=60, check=False)
    assert proc.returncode == 0 and "hand-log-import" in proc.stdout


# --------------------------------------------------------------------------- repo setup


def test_gitignore_and_pyproject():
    lines = (C.PIPELINE_DIR / ".gitignore").read_text().splitlines()
    for needed in ("data/*", ".env", "audit/", "*.parquet", "*.npy", "!results/*.parquet"):
        assert needed in lines
    # only the cleaning and split outputs come back out of data/ (A11)
    assert [x for x in lines if x.startswith("!data/")] == ["!data/02_clean/", "!data/03_split/"]
    assert lines.index("*.parquet") < lines.index("!results/*.parquet")
    pyproject = (C.REPO_DIR / "pyproject.toml").read_text()
    assert 'packages = ["src/cs156", "pipeline1/oncue"]' in pyproject
    assert "Minervas_scribe" not in pyproject and "file://" not in pyproject
