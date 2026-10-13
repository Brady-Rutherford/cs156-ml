"""Features (§5.7 + A2, A3): groups, feature sets, provenance and the meaning of each group.

The causal contract itself (fast == reference under truncation, sentinels) is in
test_causal.py; here each group is checked against hand-computed values on small invented
scenarios and on the synthetic world.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from oncue import causal, config, labels, schemas, synthetic
from oncue import features as F

CID = 7
EPS_W = config.EPS_W_S
EPS_F = 0.6


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def enc():
    return F.Memo(F.HashEncoder(48))


@pytest.fixture(scope="module")
def znull(enc):
    rng = np.random.default_rng(7)
    vocab = " ".join(synthetic.TOPICS.values()).split()
    return F.ZNull(enc([" ".join(rng.choice(vocab, 12)) for _ in range(200)]))


@pytest.fixture(scope="module")
def world_tables(world, truth):
    out = {}
    for sc in world:
        tb = dict(truth[sc.class_id])
        tb["prep"] = F.prep_table(sc.class_id, sc.raw["class_edit_page"])
        out[sc.class_id] = tb
    return out


@pytest.fixture(scope="module")
def world_points(world, world_tables):
    return {sc.class_id: labels.points(sc.class_id, world_tables[sc.class_id]) for sc in world}


def say(start: float, text: str, role: str, *, step: float = 0.35, dur: float = 0.3,
        role_delay: float = EPS_W) -> list[dict]:
    """Words of `text` spoken by `role` from `start` (one word per `step` s)."""
    return [{"start": start + k * step, "end": start + k * step + dur, "text": tok,
             "role": role, "role_delay": role_delay} for k, tok in enumerate(text.split())]


def mini(words=(), voice=(), episodes=(), polls=(), segments=(), T_end=900.0, **row):
    """Schema-shaped tables for one invented class."""
    classes = pd.DataFrame([{"class_id": CID, "T_end_s": T_end, "eps_F_s": EPS_F,
                             "rec_offset_s": 60.0, "rec_segments_json": "[[0, 1, 0.0]]",
                             "timing_precision": "ms", "pcw_status": "none",
                             "course_code": "CS901", "course_area": "CS", "session_no": 3,
                             "n_sessions": 12, "weekday": 2, "hour_utc": 14.0,
                             "enrolled": 11, "semester_idx": 7, **row}])
    hashes = {"prof": "u_prof", "me": "u_me", "student": "u_stud"}
    w = pd.DataFrame([{"class_id": CID, "w_idx": i, "start_s": x["start"], "end_s": x["end"],
                       "text_masked": x["text"], "text_norm": x["text"], "has_qmark": False,
                       "conf": 0.9, "filler": False, "spk_hash": hashes[x["role"]],
                       "role": x["role"], "role_event_start_s": x["start"] - 0.1,
                       "role_avail_s": x["end"] + x.get("role_delay", EPS_W),
                       "role_fused": x["role"], "how": "forum"} for i, x in enumerate(words)],
                     columns=schemas.columns("words"))
    v = pd.DataFrame([{"class_id": CID, "spk_hash": x.get("spk", hashes[x["role"]]),
                       "role": x["role"], "start_s": x["start"], "end_s": x["end"],
                       "room_main": x.get("main", True), "precise": True, "word_bearing": True,
                       "first_word_avail_s": np.nan} for x in voice],
                     columns=schemas.columns("voice"))
    e = pd.DataFrame([{"class_id": CID, "ep_id": i, "onset_s": x["onset"],
                       "onset_retimed_s": x["onset"], "retime_failed": False,
                       "offset_s": x["offset"], "offset_live_s": x["offset"] + 0.5,
                       "dur_s": x["offset"] - x["onset"], "n_toggles": 0,
                       "ep_type": x.get("ep_type", "bid"), "delta_n_others": 0,
                       "vote_lex": False, "contested": False, "left_censored": False,
                       "me_hand_eligible": x.get("eligible", True),
                       "outcome": x.get("outcome"), "call_latency_s": np.nan,
                       "seg_type": "debrief"} for i, x in enumerate(episodes)],
                     columns=schemas.columns("hand_episodes"))
    p = pd.DataFrame([{"class_id": CID, **x} for x in polls], columns=schemas.columns("polls"))
    s = pd.DataFrame([{"class_id": CID, "seg_idx": i, "start_s": st, "title_masked": typ,
                       "section_masked": "", "seg_type": typ} for i, (typ, st) in enumerate(segments)],
                     columns=schemas.columns("segments"))
    return {"classes": classes, "words": schemas.coerce(w, "words"),
            "voice": schemas.coerce(v, "voice"), "hand_episodes": schemas.coerce(e, "hand_episodes"),
            "polls": schemas.coerce(p, "polls"), "segments": schemas.coerce(s, "segments"),
            "hand_frames": hand_frames_for(episodes, T_end)}


def hand_frames_for(episodes, T_end: float) -> pd.DataFrame:
    """2 fps hand samples: UP on [onset, offset) of each episode, DOWN elsewhere."""
    t = np.arange(0.0, T_end + 1e-9, 1.0 / config.F_SCAN)
    raw = np.full(len(t), config.STATE_DOWN, dtype=object)
    for x in episodes:
        raw[(t >= x["onset"]) & (t < x["offset"])] = config.STATE_UP
    return pd.DataFrame({"class_id": CID, "t_s": t, "n_tiles": 9, "tile_found": True,
                         "shape_ok": raw == config.STATE_UP, "raw_state": raw,
                         "live_state": causal.live_states(raw)})


def feat(tables, ts, enc, **kw) -> pd.DataFrame:
    return F.build(CID, np.asarray(ts, float), tables, encoder=enc, **kw)


# --------------------------------------------------------------------------- groups, sets


def test_groups_match_the_config_allowlist():
    for group, cols in F.GROUPS.items():
        assert group in config.FEATURE_ALLOWLIST
        for c in cols:
            assert config.feature_group(c) == group, c
    stored = set(F.STORED_COLUMNS) | set(F.EDU_COLUMNS)
    assert all(config.feature_allowed(c, "F-punct") for c in stored)
    assert set(F.GROUP_COLUMNS["me_hand"]) == {"me_hand_n_so_far", "me_hand_since_down_s",
                                               "me_hand_since_up_s", "me_hand_last_outcome"}


def test_featuresets():
    fs = F.FEATURESETS
    need = {"F-struct", "F-struct-me_hand", "F-text", "F-all", "F-all-me_hand", "F-all-pcw",
            "F-all+pcwA", "F-pcw", "F-all-me", "F-name", "F-punct", "F-min", "F-all+hist",
            "F-all+edu", "F-all+stack"}
    assert need <= set(fs)
    assert set(fs["F-struct"]["groups"]) == {"ctx", "spk", "me", "me_hand", "poll"}
    assert set(fs["F-struct"]["groups"]) <= set(fs["F-all"]["groups"])
    assert "me_hand" not in fs["F-all-me_hand"]["groups"]
    assert "me_hand" not in fs["F-struct-me_hand"]["groups"]
    assert not {"me", "me_hand"} & set(fs["F-all-me"]["groups"])
    assert not {"pcwT", "pcwA"} & set(fs["F-all-pcw"]["groups"])
    assert "pcwA" not in fs["F-all"]["groups"] and "pcwA" in fs["F-all+pcwA"]["groups"]
    assert "pcwA" in F.featuresets("H")["F-all"]["groups"]           # Branch H
    assert fs["F-text"]["columns"] == ("ctx_min",)
    assert fs["F-min"]["groups"] == () and len(fs["F-min"]["columns"]) == 5
    assert "hist" in fs["F-all+hist"]["groups"] and "edu" in fs["F-all+edu"]["groups"]
    assert "stack" in fs["F-all+stack"]["groups"]
    assert fs["F-name"]["text"] == "name" and fs["F-punct"]["text"] == "punct"
    assert [n for n, v in fs.items() if "punct" in v["groups"]] == ["F-punct"]


def test_columns_for(world, world_tables, enc):
    sc = world.classes[0]
    df = F.build(sc.class_id, [600.0], world_tables[sc.class_id], encoder=enc)
    df = F.add_hist(df, pd.DataFrame({"class_id": [sc.class_id], "hist_bid_rate_course": [0.1],
                                      "hist_bid_rate_all": [0.2]}))
    struct = F.columns_for("F-struct", df.columns)
    assert "ctx_min" in struct and "me_hand_n_so_far" in struct and "q_open" not in struct
    assert F.columns_for("F-min", df.columns) == [f"ctx_min_spl_{k}" for k in range(5)]
    assert set(F.columns_for("F-text", df.columns)) >= {"ctx_min", "q_open", "n_words_60"}
    assert "hist_bid_rate_all" in F.columns_for("F-all+hist", df.columns)
    assert "hist_bid_rate_all" not in F.columns_for("F-all", df.columns)
    assert "has_qmark" in F.columns_for("F-punct", df.columns)
    for fs in F.FEATURESETS:
        assert not {"class_id", "t_s", "emb_row"} & set(F.columns_for(fs, df.columns))


def test_provenance_covers_every_column_and_is_whitelisted():
    cols = set(F.STORED_COLUMNS) | set(F.EDU_COLUMNS) | set(config.HIST_COLUMNS) \
        | set(config.STACK_COLUMNS) | {"pcw_new_block"}
    assert cols <= set(F.PROVENANCE)
    try:
        from oncue import loaders
    except ImportError:                          # pragma: no cover - loaders is another stage
        loaders = None
    if loaders is not None and hasattr(loaders, "feature_provenance_problems"):
        assert loaders.feature_provenance_problems(F.PROVENANCE) == []
    else:                                        # pragma: no cover
        for refs in F.PROVENANCE.values():
            for ref in refs:
                if isinstance(ref, tuple):
                    assert config.whitelist_use(*ref) in (config.USE_FEATURE, config.USE_TIMING)
    assert set(F.DERIVED_INPUTS) >= {"me_hand", "hist", "edu", "stack"}


# --------------------------------------------------------------------------- build output


def test_build_schema_and_dtypes(world, world_tables, world_points, enc, znull):
    sc = world.classes[0]
    pts = world_points[sc.class_id]
    df, emb = F.build(sc.class_id, pts, world_tables[sc.class_id], encoder=enc, znull=znull,
                      return_emb=True)
    schemas.validate(df, "features")
    assert len(df) == len(pts) and np.allclose(df["t_s"], pts["t_s"])
    assert list(df.columns[:3]) == ["class_id", "t_s", "emb_row"]
    assert list(df["emb_row"]) == list(range(len(df)))
    assert emb.shape == (len(df), 4 * 48) and emb.dtype == np.float32
    for c in schemas.feature_columns(df):
        if c in F.CATEGORICAL:
            assert pd.api.types.is_string_dtype(df[c]), c
        else:
            assert df[c].dtype == np.float32, c
    assert df["ctx_min"].notna().all()
    assert not [c for c in df.columns if c.startswith("edu_")]       # no scorer given
    multi = pd.concat([pts, world_points[world.classes[1].class_id]])
    assert len(F.build(sc.class_id, multi, world_tables[sc.class_id], encoder=enc)) == len(pts)


def test_ctx_group(world, world_tables, enc):
    sc = world.classes[0]
    tb = world_tables[sc.class_id]
    info = causal.class_info(tb["classes"], sc.class_id)
    seg = tb["segments"].iloc[3]
    ts = [seg["start_s"] + info.eps_f - 0.01, seg["start_s"] + info.eps_f, 120.0]
    df = F.build(sc.class_id, ts, tb, encoder=enc)
    np.testing.assert_allclose(df["ctx_min"], (np.array(ts) + sc.meta["rec_offset_s"]) / 60,
                               rtol=1e-6)
    assert df["ctx_seg_type"].iloc[0] == tb["segments"].iloc[2]["seg_type"]
    assert df["ctx_seg_type"].iloc[1] == seg["seg_type"] and df["ctx_seg_elapsed_s"].iloc[1] \
        == pytest.approx(info.eps_f, abs=1e-4)
    assert df["ctx_course"].iloc[0] == "CS901" and df["ctx_area"].iloc[0] == "CS"
    present = tb["presence"]["enter_s"] + info.eps_f <= 120.0
    assert df["ctx_n_present"].iloc[2] == present.sum()
    poll = tb["polls"].iloc[0]
    inside = F.build(sc.class_id, [poll["open_s"] + info.eps_p + 1], tb, encoder=enc)
    assert inside["ctx_in_poll"].iloc[0] == 1.0 and df["ctx_in_poll"].iloc[2] == 0.0
    after = F.build(sc.class_id, [float(tb["segments"].iloc[5]["start_s"]) + 30], tb, encoder=enc)
    assert after["ctx_after_breakout"].iloc[0] == 1.0
    assert after["ctx_since_breakout_s"].iloc[0] == pytest.approx(30.0, abs=1e-4)
    spl = F.ctx_splines([5.0, 30.0, 70.0])
    assert spl.shape == (3, 5) and np.isfinite(spl).all()


def test_spk_group_uses_elapsed_time_and_ignores_blips(enc):
    voice = [{"role": "prof", "start": 100.0, "end": 110.0}, {"role": "student", "start": 111.0,
                                                               "end": 115.0},
             {"role": "me", "start": 120.0, "end": 120.4}]            # Brady blip, no words
    tb = mini(voice=voice)
    df = feat(tb, [105.0, 116.0, 121.5], enc)
    assert df["spk_cur_role"].iloc[0] == "prof"
    assert df["spk_cur_elapsed_s"].iloc[0] == pytest.approx(5.0)
    assert df["spk_silence_s"].iloc[0] == 0.0
    assert df["spk_cur_role"].iloc[1] == "none"
    assert df["spk_silence_s"].iloc[1] == pytest.approx(1.0)
    assert df["spk_silence_s"].iloc[2] == pytest.approx(6.5)     # blip is invisible
    assert df["spk_prof_share_60"].iloc[1] == pytest.approx(10 / 14, rel=1e-5)
    assert df["spk_stud_turns_60"].iloc[1] == 1 and df["spk_turns_since_prof"].iloc[1] == 1
    # with a Brady word inside the event, it counts as Brady speaking
    tb = mini(voice=voice, words=say(120.05, "yes", "me", dur=0.2))
    df = feat(tb, [120.7], enc)
    assert df["spk_cur_role"].iloc[0] == "me"


def test_me_group(enc):
    words = say(200.0, "i think the gradient explains the variance here", "me")
    voice = [{"role": "me", "start": 199.9, "end": 203.0}, {"role": "me", "start": 600, "end": 640,
                                                          "main": False}]
    chat = pd.DataFrame({"class_id": CID, "t_s": [150.0, 250.0], "spk_hash": "u_me",
                         "role": ["me", "student"], "text_masked": ["hi", "yo"]})
    tb = {**mini(words=words, voice=voice), "chat": chat}
    df = feat(tb, [199.0, 210.0, 700.0], enc)
    assert df["me_words_so_far"].tolist() == [0, 8, 8]
    assert np.isnan(df["me_since_turn_s"].iloc[0])
    assert df["me_since_turn_s"].iloc[1] == pytest.approx(210.0 - (200.0 + 7 * 0.35 + 0.3))
    assert df["me_turns_so_far"].tolist() == [0, 1, 1]
    assert df["me_chat_n"].tolist() == [1, 1, 1]
    assert df["me_since_chat_s"].iloc[1] == pytest.approx(60.0)
    assert df["me_breakout_talk_s"].tolist() == [0, 0, 40]


def test_me_hand_group_history_and_outcome_timing(enc):
    """me_hand_* come from Brady's live UP runs (samples <= t): a run counts once its live
    lowering is confirmed; its outcome ("called": a word-bearing Brady turn starts within
    5 s of the lowering) is known at lowering + 5 s + eps_F, whatever the episode's type."""
    eps = [{"onset": 300.0, "offset": 306.0},
           {"onset": 400.0, "offset": 405.0, "ep_type": "vote"},
           {"onset": 450.0, "offset": 450.5, "ep_type": "glitch", "eligible": False}]
    tb = mini(episodes=eps, voice=[{"role": "me", "start": 308.0, "end": 310.0}],
              words=say(308.2, "i think so", "me"))
    known = 306.0 + config.CALLED_AFTER_OFFSET_S + EPS_F       # first DOWN sample + 5 + eps_F
    ts = [306.4, 306.5, known - 0.01, known, 406.0, 460.0, 405.0 + 5 + EPS_F]
    df = feat(tb, ts, enc)
    assert df["me_hand_n_so_far"].tolist() == [0, 1, 1, 1, 2, 2, 2]
    assert df["me_hand_since_down_s"].iloc[1] == pytest.approx(0.5)
    assert df["me_hand_since_up_s"].iloc[1] == pytest.approx(6.5)
    assert df["me_hand_last_outcome"].tolist()[1:4] == ["none", "none", "called"]
    assert df["me_hand_last_outcome"].iloc[4] == "called"            # vote's not known yet
    assert df["me_hand_since_down_s"].iloc[5] == pytest.approx(55.0)  # a blip never counts
    assert df["me_hand_last_outcome"].iloc[6] == "withdrawn"          # typed vote: same rule


def test_me_hand_outcome_ignores_vote_typing(enc):
    """Other tiles' hands (n_others / Δn vote typing) never reach a feature: the same raise
    gives the same me_hand_* whether it was typed bid or vote."""
    voice = [{"role": "me", "start": 193.0, "end": 195.0}]
    words = say(193.2, "my answer", "me")
    ts = [194.0, 200.0, 220.0]
    a = feat(mini(episodes=[{"onset": 180.0, "offset": 190.0}], voice=voice, words=words),
             ts, enc)
    b = feat(mini(episodes=[{"onset": 180.0, "offset": 190.0, "ep_type": "vote"}],
                  voice=voice, words=words), ts, enc)
    cols = list(F.GROUP_COLUMNS["me_hand"])
    pd.testing.assert_frame_equal(a[cols], b[cols])
    assert a["me_hand_last_outcome"].iloc[1] == "called"


def test_poll_group(world, world_tables, enc, znull):
    sc = world.classes[0]
    tb = world_tables[sc.class_id]
    info = causal.class_info(tb["classes"], sc.class_id)
    poll = tb["polls"].iloc[0]
    before = poll["my_answer_avail_s"] - 0.01
    ts = [poll["open_s"] + info.eps_p - 0.01, before, poll["my_answer_avail_s"],
          poll["close_s"] + info.eps_p + 5]
    df = F.build(sc.class_id, ts, tb, encoder=enc, znull=znull)
    assert df["poll_n_so_far"].tolist() == [0, 1, 1, 1]
    assert np.isnan(df["poll_res_my"].iloc[1]) and np.isfinite(df["poll_res_my"].iloc[2])
    assert np.isnan(df["poll_my_ans_words"].iloc[1]) and df["poll_my_ans_words"].iloc[2] == 1
    assert df["poll_since_close_s"].iloc[3] == pytest.approx(info.eps_p + 5, abs=1e-4)


def test_q_group_open_question_kind_and_answer(enc):
    q = say(100.0, "why does the gradient shrink at this point", "prof")
    end = q[-1]["end"]
    student = say(110.0, "i think the gradient shrinks", "student")
    inv = say(200.0, "does anyone want to share their answer", "prof")
    vote = say(300.0, "raise your hand if you agree", "prof")
    tb = mini(words=q + student + inv + vote)
    ts = [end + 0.5, end + config.Q_SILENCE_S + EPS_W + 0.01, 109.0, 115.0,
          inv[-1]["end"] + 1.1, vote[-1]["end"] + 1.0]
    df = feat(tb, ts, enc)
    # "does anyone ..." opens with an inverted auxiliary: an open invitation question
    assert df["q_open"].tolist() == [0, 1, 1, 0, 1, 0]
    assert df["q_kind"].iloc[1] == "plain" and df["q_age_s"].iloc[1] == pytest.approx(
        ts[1] - end, abs=1e-4)
    assert df["q_kind"].iloc[4] == "invite"
    assert df["q_inv_age_s"].iloc[4] == pytest.approx(1.1, abs=1e-4)
    assert df["q_inv_age_s"].iloc[1] == config.Q_AGE_CAP_S
    assert df["q_count_300"].iloc[4] == 2
    assert df["q_vote_lex"].tolist() == [0, 0, 0, 0, 0, 1]


def test_q_rhetorical_needs_the_resume_to_be_known(enc):
    q = say(100.0, "why does the loss matter here", "prof")
    resume = say(q[-1]["end"] + 1.5, "so the answer is the bias", "prof")
    tb = mini(words=q + resume)
    first = resume[0]
    ts = [first["start"] - 0.1, first["end"] + EPS_W - 0.01, first["end"] + EPS_W + 0.01]
    df = feat(tb, ts, enc)
    assert df["q_kind"].tolist() == ["plain", "plain", "rhetorical"]


def test_text_windows_and_has_qmark(enc):
    words = say(100.0, "one two three four five six", "prof") + \
        say(150.0, "um seven eight", "student")
    tb = mini(words=words)
    tb["words"].loc[tb["words"]["text_norm"] == "um", "filler"] = True
    tb["words"].loc[tb["words"]["text_norm"] == "eight", "has_qmark"] = True
    w3 = tb["words"].iloc[3]
    df = feat(tb, [w3["end_s"] + EPS_W, 151.5, 157.0], enc)
    assert df["n_words_15"].iloc[0] == 4 and df["n_words_60"].iloc[0] == 4
    assert df["n_words_15"].iloc[2] == 3 and df["n_words_60"].iloc[2] == 9
    assert df["filler_rate_60"].iloc[2] == pytest.approx(1 / 9)
    assert df["wpm_60"].iloc[0] == pytest.approx(60 * 4 / (4 * 0.3))
    assert df["frac_unk_60"].iloc[2] == 0
    assert df["has_qmark"].iloc[2] == 0          # student word: not a professor utterance


def test_pcw_status_gates_the_answer_side(world, world_tables, enc, znull):
    rows = {}
    for k in (0, 3, 4):
        sc = world.classes[k]
        df = F.build(sc.class_id, [900.0, 1500.0], world_tables[sc.class_id], encoder=enc,
                     znull=znull)
        rows[sc.meta["pcw_status"]] = df
    ok, none, unknown = rows["verified"], rows["none"], rows["unknown"]
    assert (ok["pcw_has"] == 1).all() and ok["pcw_res_A"].notna().all()
    assert ok["pcw_my_words"].iloc[0] > 0 and ok["pcw_frac_answered"].iloc[0] > 0
    assert (none["pcw_tmpl_has"] == 0).all() and (none["pcw_has"] == 0).all()
    assert none["pcw_res_Q"].isna().all() and none["pcw_res_A"].isna().all()
    assert (unknown["pcw_tmpl_has"] == 1).all() and unknown["pcw_res_Q"].notna().all()
    pcwa = [c for c in F.GROUP_COLUMNS["pcwA"]]
    # an unknown doc (edited after class start) looks exactly like no usable doc: neither
    # the values nor the NaN pattern of pcwA can encode pcw_status (§4.2, L12)
    pd.testing.assert_frame_equal(unknown[pcwa].reset_index(drop=True),
                                  none[pcwa].reset_index(drop=True))
    assert (unknown["pcw_has"] == 0).all() and unknown["pcw_res_A"].isna().all()


def test_pcwA_is_not_a_function_of_pcw_status(enc, znull):
    """Same blocks (answers present), every non-usable status: identical pcwA values and NaN
    pattern, so HGB's NaN branch cannot learn "Brady edited the PCW during class"."""
    answer = "boosting combines weak learners into a strong one"
    blocks = pd.DataFrame({"class_id": CID, "block_uuid": ["b1"], "block_order": [0],
                           "prompt_text": ["Explain boosting"], "answer_text": [answer],
                           "answer_words": [len(answer.split())], "has_code": [False],
                           "n_files": [0]})
    words = say(300.0, "boosting combines weak learners", "prof")
    pcwa = list(F.GROUP_COLUMNS["pcwA"])
    out = {}
    for status in ("empty", "none", "unknown"):
        tb = mini(words=words, pcw_status=status)
        tb["pcw_blocks"] = blocks
        out[status] = feat(tb, [320.0], enc, znull=znull)[pcwa]
    for status in ("none", "unknown"):
        pd.testing.assert_frame_equal(out[status], out["empty"])
    tb = mini(words=words, pcw_status="verified")
    tb["pcw_blocks"] = blocks
    assert feat(tb, [320.0], enc, znull=znull)["pcw_has"].iloc[0] == 1


def test_pcw_unspent_drops_chunks_brady_already_said(enc, znull):
    answer = "regularization shrinks variance while boosting reduces bias in the ensemble"
    blocks = pd.DataFrame({"class_id": CID, "block_uuid": ["b1"], "block_order": [0],
                           "prompt_text": ["Explain boosting"], "answer_text": [answer],
                           "answer_words": [len(answer.split())], "has_code": [False],
                           "n_files": [0]})
    words = say(100.0, "boosting reduces bias", "prof") + say(200.0, answer, "me") + \
        say(400.0, "boosting reduces bias", "prof")
    tb = {**mini(words=words, pcw_status="verified"), "pcw_blocks": blocks}
    df = feat(tb, [110.0, 410.0], enc, znull=znull)
    assert np.isfinite(df["pcw_unspent"].iloc[0])
    assert np.isnan(df["pcw_unspent"].iloc[1])              # the only chunk is spent
    assert np.isfinite(df["pcw_bm25_gap"].iloc[0])


def test_similarities_are_nan_without_a_reference_set(world, world_tables, enc):
    sc = world.classes[0]
    small = F.ZNull(enc([f"t{k}" for k in range(10)]))
    df = F.build(sc.class_id, [1500.0], world_tables[sc.class_id], encoder=enc, znull=small)
    for c in ("own_sim", "poll_res_my", "pcw_res_Q", "lo_sim", "sg_sim", "pcw_res_A"):
        assert np.isnan(df[c].iloc[0]), c
    assert np.isfinite(df["topic_shift"].iloc[0])           # cosine only, no z-scoring


def test_features_carry_signal_on_the_synthetic_world(world, world_tables, world_points, enc):
    rows = []
    for sc in world:
        p = world_points[sc.class_id]
        p = p[p[config.HEADLINE_LABEL].notna()]
        df = F.build(sc.class_id, p, world_tables[sc.class_id], encoder=enc)
        rows.append(df.assign(y=p[config.HEADLINE_LABEL].to_numpy()))
    d = pd.concat(rows, ignore_index=True)
    inv = (d["q_kind"] == "invite").groupby(d["y"]).mean()
    assert inv[1.0] > inv[0.0] + 0.05
    assert d.groupby("y")["q_open"].mean()[1.0] > d.groupby("y")["q_open"].mean()[0.0]


# --------------------------------------------------------------------------- hist (A2)


def test_hist_is_fold_aware_and_uses_earlier_training_classes_only(world, world_points):
    classes = pd.concat([synthetic.truth_tables(sc)["classes"] for sc in world],
                        ignore_index=True)
    pts = pd.concat(world_points.values(), ignore_index=True)
    ids = [sc.class_id for sc in world]
    train, test = ids[:4], ids[4:]
    h = F.hist_features(classes, pts, train).set_index("class_id")
    start = pd.to_datetime(classes["sched_start_utc"], utc=True)
    order = classes.assign(s=start).sort_values("s")["class_id"].tolist()
    assert h.loc[order[0]].isna().all()                     # nothing earlier
    minutes = pts[pts["class_id"].isin(train)].groupby("class_id")["at_risk"].sum() * 2 / 60
    bids = classes.set_index("class_id")["n_bids"].astype(float)
    assert (minutes == 0).any()                  # a QA-failed training class: no rate
    for cid in ids:
        mine = start[classes["class_id"] == cid].iloc[0]
        prior = [c for c in train if c != cid and minutes[c] > 0
                 and start[classes["class_id"] == c].iloc[0] < mine]
        if prior:
            expect = bids[prior].sum() / minutes[prior].sum()
            assert h.loc[cid, "hist_bid_rate_all"] == pytest.approx(expect)
        course = classes.set_index("class_id")["course_code"]
        same = [c for c in prior if course[c] == course[cid]]
        if same:
            assert h.loc[cid, "hist_bid_rate_course"] == pytest.approx(
                bids[same].sum() / minutes[same].sum())
        else:
            assert np.isnan(h.loc[cid, "hist_bid_rate_course"])
    # test-fold labels never enter: rewriting them changes nothing
    pts2 = pts.copy()
    pts2.loc[pts2["class_id"].isin(test), "at_risk"] = True
    classes2 = classes.copy()
    classes2.loc[classes2["class_id"].isin(test), "n_bids"] = 99
    h2 = F.hist_features(classes2, pts2, train).set_index("class_id")
    pd.testing.assert_frame_equal(h, h2)
    out = F.add_hist(pd.DataFrame({"class_id": ids, "t_s": 0.0}), h.reset_index())
    assert out["hist_bid_rate_all"].dtype == np.float32


# --------------------------------------------------------------------------- edu (A3)


class FakeModel:
    def __init__(self, p):
        self.p = p

    def predict_proba(self, texts):
        return np.array([[1 - self.p, self.p]] * len(texts))


def test_edu_scorer_from_oncue_edu_plugs_in_and_skipped_columns_stay_absent(tmp_path, enc,
                                                                              caplog):
    edu = pytest.importorskip("oncue.edu")

    def loader(repo, device):
        if repo.endswith("invites_participation"):
            return FakeModel(0.8)
        raise OSError("offline")
    scorer = edu.EduScorer(loader=loader, cache_dir=None, log_path=tmp_path / "skipped.csv",
                           device="cpu")
    words = say(100.0, "any thoughts on this", "prof")
    tb = mini(words=words)
    close = causal.build_frames(causal.prep_words(tb["words"])).close[0]
    with caplog.at_level(logging.WARNING):
        df = feat(tb, [close + EPS_W + 1.0], enc, edu=scorer)
    assert [c for c in df.columns if c.startswith("edu_")] == ["edu_invites"]
    assert df["edu_invites"].iloc[0] == pytest.approx(0.8 * np.exp(-(EPS_W + 1.0) / 10), rel=1e-5)
    assert len(scorer.skipped) == 4 and (tmp_path / "skipped.csv").exists()


def test_edu_join_decays_professor_frame_scores(enc):
    words = say(100.0, "any thoughts on this", "prof") + say(130.0, "great question", "prof")
    tb = mini(words=words)
    table = F.EduTable(pd.DataFrame({"text_norm": ["any thoughts on this", "great question"],
                                     "edu_invites": [0.9, 0.5]}))
    fr = causal.build_frames(causal.prep_words(tb["words"]))
    c0, c1 = fr.close
    ts = [c0 + EPS_W - 0.01, c0 + EPS_W + 5.0, c1 + EPS_W + 2.0]
    df = feat(tb, ts, enc, edu=table)
    assert df["edu_invites"].iloc[0] == 0                     # closed <= t - eps_w only
    assert df["edu_invites"].iloc[1] == pytest.approx(0.9 * np.exp(-(ts[1] - c0) / 10), rel=1e-5)
    expect = max(0.9 * np.exp(-(ts[2] - c0) / 10), 0.5 * np.exp(-(ts[2] - c1) / 10))
    assert df["edu_invites"].iloc[2] == pytest.approx(expect, rel=1e-5)
    assert not [c for c in df.columns if c.startswith("edu_") and c != "edu_invites"]
    assert F.edu_decayed(np.array([10.0, 20.0]), np.array([[0.9], [0.5]]), 25.0)[0] == \
        pytest.approx(max(0.9 * np.exp(-1.5), 0.5 * np.exp(-0.5)))


# --------------------------------------------------------------------------- helpers


def test_pcw_new_block_is_in_fold_and_first_switch_only():
    df = pd.DataFrame({"class_id": [1] * 6, "t_s": np.arange(6.0),
                       "pcw_block_pos_Q": [0.5, 0.5, 1.0, 0.5, 1.0, 1.0],
                       "pcw_res_Q": [3.0, 3.0, 2.5, 0.5, 2.6, 2.6]})
    train = pd.DataFrame({"pcw_res_Q": np.linspace(0, 2.5, 11)})
    thr = F.pcw_new_block_threshold(train)
    assert thr == pytest.approx(2.25)
    out = F.pcw_new_block(df, thr).tolist()
    assert out == [1.0, 0.0, 1.0, 0.0, 0.0, 0.0]


def test_text_helpers_and_encoders():
    assert F.norm_text("<p>Fit the <b>gradientBoost</b> model_v2!</p>") == \
        "fit the gradient boost model v2"
    assert F.norm_text("[NAME] said hi") == "said hi" and F.norm_text(None) == ""
    ch = F.chunks(" ".join(str(i) for i in range(250)))
    assert [len(c.split()) for c in ch] == [120, 120, 70]
    assert ch[1].split()[0] == "90"                          # 30-word overlap
    assert F.is_question("what do you think about it") and not F.is_question("what is")
    assert F.is_question("that is the idea right") and not F.is_question("so the idea is")
    assert F.lexical_kind("does anyone want to try") == "invite"
    assert F.lexical_kind("does that make sense") == "check"
    e1, e2 = F.HashEncoder(16), F.HashEncoder(16)
    a = e1(["alpha beta", ""])
    assert np.allclose(a, e2(["alpha beta", ""])) and np.allclose(np.linalg.norm(a[0]), 1)
    assert not a[1].any()
    m = F.Memo(e1)
    m(["alpha beta"])
    assert "alpha beta" in m.cache and F.Memo(m).cache is m.cache
    assert not F.ZNull(np.ones((10, 16))).ok and F.ZNull(np.random.rand(60, 16)).ok


def test_znull_and_ref_sets_use_only_earlier_classes(world, enc):
    tabs = [synthetic.truth_tables(sc) for sc in world]
    classes = pd.concat([t["classes"] for t in tabs], ignore_index=True)
    blocks = pd.concat([t["pcw_blocks"] for t in tabs], ignore_index=True)
    first = world.classes[1]                                 # the earliest class (F23)
    z = F.znull_reference(first.class_id, classes, blocks, enc)
    assert z.R is None or len(z.R) == 0
    last = world.classes[3]                                  # CS901, F26 week 2
    z = F.znull_reference(last.class_id, classes, blocks, enc)
    assert z.R is not None and 0 < len(z.R) < config.ZNULL_MIN and not z.ok
    refs = F.pcw_ref_sets(last.class_id, classes, blocks, enc)
    assert len(refs) == 1                                    # class 0: same section, earlier
    assert F.pcw_ref_sets(world.classes[0].class_id, classes, blocks, enc) == []


def test_prep_table_reads_only_whitelisted_paths(world):
    sc = world.classes[0]
    prep = F.prep_table(sc.class_id, sc.raw["class_edit_page"])
    assert set(prep["kind"]) == {"lo", "sg"}
    text = " ".join(prep["text"])
    assert "graded" not in text and "Analyze how" in text
    snake = F.prep_table(sc.class_id, synthetic._snake(sc.raw["class_edit_page"]))
    assert len(snake) == len(prep)


def test_cli_writes_features(tmp_path, monkeypatch, world, world_tables, world_points):
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir()
    sc = world.classes[0]
    tb = world_tables[sc.class_id]
    for name in ("classes", "words", "voice", "segments", "chat", "polls", "presence",
                 "pcw_blocks", "hand_episodes", "prep"):
        tb[name].to_parquet(tables_dir / f"{name}.parquet", index=False)
    world_points[sc.class_id].to_parquet(tables_dir / "points.parquet", index=False)
    monkeypatch.setattr(config, "TABLES_DIR", tables_dir)
    monkeypatch.setattr(F, "default_encoder", lambda: F.Memo(F.HashEncoder(16)))
    rc = F.cli(SimpleNamespace(class_ids=None, extra=["--no-edu"]))
    assert rc == 0
    out = pd.read_parquet(tables_dir / "features.parquet")
    schemas.validate(out, "features")
    emb = np.load(tables_dir / "features_emb.npy")
    assert emb.shape == (len(out), 64) and out["emb_row"].tolist() == list(range(len(out)))


def test_sparse_classes_do_not_crash(enc, znull):
    """Real classes can lack words, voice, polls, PCW or a located tile."""
    empty = mini()
    df = feat(empty, [0.0, 130.0, 800.0], enc, znull=znull)
    schemas.validate(df, "features")
    assert df["n_words_60"].tolist() == [0, 0, 0] and (df["q_kind"] == "none").all()
    assert (df["spk_cur_role"] == "none").all() and df["pcw_tmpl_has"].eq(0).all()
    # words whose speaker Forum never saw: NA spk_hash, role "unk"
    tb = mini(words=say(100.0, "hello there everyone", "prof"))
    tb["words"]["spk_hash"] = pd.NA
    tb["words"]["role"] = "unk"
    out = feat(tb, [102.5], enc)
    assert out["frac_unk_60"].iloc[0] == 1 and out["n_words_60"].iloc[0] == 3
    view = causal.CausalView(CID, 102.5, tb, encoder=enc)
    pd.testing.assert_frame_equal(out, view.features(), check_exact=False, rtol=1e-5, atol=1e-6)
    # no classes row at all
    tb.pop("classes")
    assert len(feat(tb, [102.5], enc)) == 1
