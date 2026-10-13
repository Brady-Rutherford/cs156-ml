"""Forum event tables, the causal words table and Scribe runs (events.py, transcribe.py).

Spec: §2.1 (unclamped clock, gap events dropped), §4 (c) (raw class-events only, type
allowlist), §4.1 A1-A3, A6, A8-A11, §4.2, §4.3, §5.1, §5.3, §13, L5, L10, L26, L27, D7.
Synthetic data only; every Scribe run here is offline (replayed ASR, no Forum client).
"""

from __future__ import annotations

import json
import re
from argparse import Namespace

import numpy as np
import pandas as pd
import pytest
from oncue import causal, config, events, schemas, synthetic, transcribe
from scribe_core.asr_base import ASRProvider, ASRResult
from scribe_core.calibrate import make_shift
from scribe_core.forum_events import timeline_segments
from scribe_core.fuse import EDGE, HOW_RANK, teacher_cluster
from scribe_core.privacy import pseudonym

SALT = "synthetic-test-salt"


def _kind(ev):
    return ev.get("event-type", ev.get("event_type"))


def _planted(sc) -> dict:
    """The synthetic world's true calibration (audio = forum video + offset)."""
    off = sc.meta["cal_offset_s"]
    return {"applied": True, "status": "ok", "method": "windowed",
            "pieces": [{"t_from": None, "offset": off, "drift": 0.0}]}


def _clock(sc, cal=None):
    return events.video_clock(None, _planted(sc) if cal is None else cal, sc.raw["class"])


def _wall(sc, audio_t: float) -> int:
    """Inverse of the planted clock (single session, or the session holding audio_t)."""
    v = audio_t - sc.meta["cal_offset_s"]
    for a, _b, v0 in reversed(sc.meta["sessions"]):
        if v >= v0:
            return round(a + (v - v0) * 1000)
    a, _b, v0 = sc.meta["sessions"][0]
    return round(a + (v - v0) * 1000)


@pytest.fixture(scope="module")
def built(world):
    """{class_id: (WordsBuild, EventTables)} with the calibration Scribe's calibrate() finds."""
    out = {}
    for sc in world:
        b = transcribe.build_words(sc.class_id, sc.asr, sc.raw["class"], sc.raw["events"],
                                   me_id=sc.meta["me_id"], salt=SALT)
        et = events.build_event_tables(
            sc.class_id, klass=sc.raw["class"], events=sc.raw["events"], me_id=sc.meta["me_id"],
            eps_f_s=b.timing["eps_F_s"], cal=b.calibration.cal, words=b.words,
            polls=sc.raw["polls"], poll_responses=sc.raw["poll_responses"], salt=SALT,
            timing_precision=b.timing["timing_precision"])
        out[sc.class_id] = (b, et)
    return out


# =========================================================================== raw allowlist


def test_whitelist_filter_drops_traps(world):
    for sc in world:
        for source in ("class", "events", "polls", "poll_responses", "class_edit_page"):
            kept = events.whitelist_filter(source, sc.raw[source])
            for path in synthetic.flatten_paths(kept):
                assert config.whitelist_use(source, path) is not None, (source, path)
        evs = events.allowed_events(sc.raw["events"])
        assert {e["event-type"] for e in evs} <= config.EVENT_TYPES
        assert not any("first-name" in (e.get("actor") or {}) for e in evs)
        assert not any("created-on" in e for e in evs)
        klass = events.whitelist_filter("class", sc.raw["class"])
        assert "actual-length-seconds" not in klass and "assessment" not in klass
        polls = events.whitelist_filter("polls", sc.raw["polls"])
        assert all("correct-choice-id" not in p and "results" not in p for p in polls)


def test_blocked_sources_and_types(world, data_dir):
    sc = world.classes[0]
    for source in config.BLOCKED_SOURCES:
        with pytest.raises(ValueError):
            events.whitelist_filter(source, {"x": 1})
        with pytest.raises(ValueError):
            events.read_raw(sc.class_id, source, data_dir["raw"])
    traps = [{"event-type": t, "timestamp": 1, "actor-id": 1} for t in config.BLOCKED_EVENT_TYPES]
    kept = events.allowed_events(list(sc.raw["events"]) + traps)
    assert len(kept) == sum(_kind(e) in config.EVENT_TYPES for e in sc.raw["events"])


def test_read_raw_from_disk(world, data_dir):
    sc = world.classes[0]
    assert events.read_raw(sc.class_id, "events", data_dir["raw"]) == \
        events.allowed_events(sc.raw["events"])
    assert events.read_raw(sc.class_id, "pcw_doc", data_dir["raw"]) is not None
    assert events.read_raw(999, "class", data_dir["raw"]) is None
    assert events.load_me_id(data_dir["scribe"]) == str(synthetic.ME_ID)


# =========================================================================== clock


def test_clock_maps_segments_onto_the_audio(world):
    for sc in world:
        clock = _clock(sc)
        got = clock.masters(sc.segments["wall_ms"].to_numpy(float))
        assert np.abs(got - sc.segments["start_s"].to_numpy()).max() < 1e-6


def test_clock_is_unclamped_and_drops_gap_times(world):
    sc = world.classes[-1]
    assert sc.meta["multi_session"]
    clock = _clock(sc)
    gap = sc.meta["gap_wall_ms"]
    mid = (gap[0] + gap[1]) / 2
    assert clock.status(mid) == "gap" and np.isnan(clock.master(mid))
    first = sc.meta["sessions"][0][0]
    assert clock.status(first - 5000) == "pre"
    assert clock.master(first - 5000) == pytest.approx(-5.0 + sc.meta["cal_offset_s"])
    restart = sc.meta["sessions"][1][2] + sc.meta["cal_offset_s"]
    assert clock.restarts() == [pytest.approx(restart)]
    assert clock.gaps_wall() == [(gap[0], gap[1])]
    # Scribe's chapter mapping clamps a gap-time segment to the restart; ours drops it.
    seg = {"event-type": "timeline-segment", "timestamp": int(mid), "actor-id": 100,
           "room-id": 1, "room-signature": config.MAIN_ROOM_SIGNATURE,
           "event-data": {"timeline-segment-title": "Gap slide",
                          "timeline-section-title": "Gap"}}
    scribe_t = [c["t"] for c in timeline_segments([seg], sc.raw["class"])]
    assert scribe_t == [pytest.approx(sc.meta["sessions"][1][2])]
    segs, counts = events.segments_table(sc.class_id, events.allowed_events([seg]), clock)
    assert len(segs) == 0 and counts["gap"] == 1


def test_clock_shift_matches_scribe_make_shift():
    cal = {"applied": True, "pieces": [{"t_from": None, "offset": 0.3, "drift": 0.001},
                                       {"t_from": 900.0, "offset": -1.2, "drift": 0.0}]}
    v = np.array([-5.0, 0.0, 450.0, 899.99, 900.0, 2000.0])
    f = make_shift(cal)
    assert np.allclose(events._shift_array(v, cal), [f(x) for x in v])
    assert np.allclose(events._shift_array(v, {**cal, "applied": False}), v)
    assert np.allclose(events._shift_array(v, None), v)
    seg = [events.RecordingSegment(1_000_000, 1_600_000, 0.0)]
    clock = events.video_clock({"asr_origin_s": 0.0}, None, seg)
    assert clock.master(1_010_000) == pytest.approx(10.0)
    assert clock.master(None) != clock.master(None)                # NaN


# =========================================================================== voice


def test_voice_table_matches_truth(built, world):
    for sc in world:
        _, et = built[sc.class_id]
        schemas.validate(et.voice, "voice", strict=True)
        v = et.voice[~et.voice["inferred"]].reset_index(drop=True)       # Forum's own rows
        truth = sc.voice[sc.voice["role"] != config.ROLE_PROF]           # Forum logs no prof
        truth = truth.sort_values(["start_s", "end_s"]).reset_index(drop=True)
        assert len(v) == len(truth)
        planted = events.parse_voice(events.allowed_events(sc.raw["events"]), _clock(sc))
        tol = 1e-3 if sc.meta["precise"] else 1.0
        assert np.abs(np.sort(planted["start_s"]) - np.sort(truth["start_s"])).max() <= tol
        # rows that start at nearly the same time may sort differently: compare multisets
        got = sorted(zip(v["spk_hash"], v["role"], v["room_main"], v["word_bearing"],
                         strict=True))
        want = sorted(zip(truth["user_id"].map(lambda u: pseudonym(u, SALT)), truth["role"],
                          truth["room_main"], truth["word_bearing"], strict=True))
        assert got == want
        assert (v["precise"] == sc.meta["precise"]).all()
        assert v["start_avail_s"].isna().all() and v["end_avail_s"].isna().all()


def _covered(spans_a, spans_b) -> float:
    """Share of the total length of spans_a that some span of spans_b covers (10 ms grid)."""
    hz = 100
    n = int(max([e for _, e in spans_a] + [e for _, e in spans_b] + [0]) * hz) + 2
    a, b = np.zeros(n, bool), np.zeros(n, bool)
    for s, e in spans_a:
        a[int(s * hz):int(e * hz)] = True
    for s, e in spans_b:
        b[int(s * hz):int(e * hz)] = True
    return (a & b).sum() / max(a.sum(), 1)


def test_professor_turns_stand_in_for_the_missing_voice_events(built, world):
    """A12: the raw events hold no professor mic, yet the voice table has the professor's
    turns, built from words, each one knowable only after it happened."""
    for sc in world:
        assert not any(e.get("actor-id", e.get("actor_id")) == int(sc.meta["prof_id"])
                       for e in sc.raw["events"]
                       if e.get("event-type", e.get("event_type")) == "voice")
        _, et = built[sc.class_id]
        p = et.voice[et.voice["inferred"]]
        assert len(p) and (p["role"] == config.ROLE_PROF).all() and p["room_main"].all()
        assert p["spk_hash"].nunique() == 1
        assert p["spk_hash"].iloc[0] == pseudonym(sc.meta["prof_id"], SALT)
        assert (p["start_avail_s"] >= p["start_s"] + EDGE).all()
        assert (p["end_avail_s"] > p["end_s"]).all() and (p["end_s"] >= p["start_s"]).all()
        assert p["word_bearing"].all()
        assert np.allclose(p["first_word_avail_s"], p["start_avail_s"])
        true_prof = sc.voice[(sc.voice["role"] == config.ROLE_PROF) & sc.voice["room_main"]]
        assert _covered(list(zip(true_prof["start_s"], true_prof["end_s"])),
                        list(zip(p["start_s"], p["end_s"]))) >= 0.9


def test_professor_turn_rules():
    def words(rows):   # (start, end, role, has_event)
        return pd.DataFrame({
            "w_idx": range(len(rows)), "start_s": [r[0] for r in rows],
            "end_s": [r[1] for r in rows], "role": [r[2] for r in rows],
            "role_event_start_s": [r[0] if r[3] else np.nan for r in rows],
            "role_avail_s": [r[1] + 1.25 for r in rows], "spk_hash": "p"})
    P, S = config.ROLE_PROF, config.ROLE_STUDENT
    t = events.professor_turns(words([
        (0.0, 0.3, P, False), (0.5, 0.8, P, False),        # one turn (gap 0.2 s)
        (3.0, 3.3, P, False),                               # gap 2.2 s: a new turn ...
        (3.6, 3.9, S, True),                                # ... cut short by a student
        (10.0, 10.3, P, True),                              # a staff mic event: not inferred
    ]))
    assert t[["start_s", "end_s"]].values.tolist() == [[0.0, 0.8], [3.0, 3.3]]
    assert t["start_avail_s"].tolist() == pytest.approx([1.55, 4.55])
    # turn 1 is over once 1.2 s pass with no new word; turn 2 once the student's word is known
    assert t["end_avail_s"].tolist() == pytest.approx([0.8 + 1.2 + config.EPS_W_S, 3.9 + 1.25])
    assert len(events.professor_turns(None)) == 0


def test_word_bearing_flags_blips_and_availability(built, world):
    total_blips = 0
    for sc in world:
        b, et = built[sc.class_id]
        v = et.voice
        eps = b.timing["eps_F_s"]
        n_blips = int((sc.voice["blip"] & sc.voice["room_main"]).sum())   # Brady's mic blips
        total_blips += n_blips
        me = v[(v["role"] == config.ROLE_ME) & v["room_main"]]
        assert int((~me["word_bearing"]).sum()) == n_blips
        assert me["word_bearing"].any()
        assert not v.loc[~v["room_main"], "word_bearing"].any()     # breakout audio unrecorded
        wb = v[v["word_bearing"]]
        assert (wb["first_word_avail_s"] >= wb["start_s"] + eps - 1e-9).all()
        assert v.loc[~v["word_bearing"], "first_word_avail_s"].isna().all()
    assert total_blips > 0


def test_voice_events_in_and_across_gaps(world):
    sc = world.classes[-1]
    clock = _clock(sc)
    gap = sc.meta["gap_wall_ms"]
    base = {"event-type": "voice", "actor-id": 100, "room-id": 5000 + sc.meta["index"],
            "room-signature": config.MAIN_ROOM_SIGNATURE}
    inside = {**base, "id": "in-gap", "timestamp": gap[0] + 9000, "event-data": {"duration": 3000}}
    across = {**base, "id": "across", "timestamp": gap[0] + 4000, "event-data": {"duration": 6000}}
    v = events.parse_voice([inside, across], clock)
    assert list(v["event_id"]) == ["across"]
    assert v["end_s"].iloc[0] == pytest.approx(clock.session_end_master(gap[0] - 2000))
    assert v["start_s"].iloc[0] == pytest.approx(clock.master(gap[0] - 2000))


# =========================================================================== chat


def test_chat_keeps_human_messages_only(built, world):
    common = events.common_words()
    for sc in world:
        _, et = built[sc.class_id]
        chat = et.chat
        schemas.validate(chat, "chat", strict=True)
        want = sc.chat[~sc.chat["system"] & ~sc.chat["private"] & ~sc.chat["in_gap"]]
        assert len(chat) == len(want)
        assert et.counts["chat_null_actor"] == 1 and et.counts["chat_private"] == 1
        assert et.counts.get("chat_gap", 0) == int(sc.meta["multi_session"])
        planted, _ = events.chat_table(sc.class_id, events.allowed_events(sc.raw["events"]),
                                       _clock(sc), events.roster_from_class(
                                           sc.raw["class"], sc.meta["me_id"]), salt=SALT)
        assert np.allclose(np.sort(planted["t_s"]), np.sort(want["t_s"].to_numpy(float)))
        text = " ".join(chat["text_masked"])
        for name in pd.concat([sc.roster["first"], sc.roster["last"]]):
            if name.lower() not in common:
                assert not re.search(rf"\b{name}\b", text), name
        assert set(chat["role"]) <= {config.ROLE_ME, config.ROLE_STUDENT}


def test_chat_filters_notices_strangers_and_gaps(world):
    sc = world.classes[-1]
    clock = _clock(sc)
    roster = events.roster_from_class(sc.raw["class"], sc.meta["me_id"])
    student = sc.roster.loc[sc.roster["role"] == config.ROLE_STUDENT, "user_id"].iloc[0]
    t_ok = sc.meta["sessions"][0][0] + 200_000
    msg = lambda actor, text, ts: {"event-type": "message", "actor-id": actor,
                                   "timestamp": ts, "event-data": {"message-text": text}}
    raw = [msg(int(student), "Ada Lovelace raised their hand", t_ok),
           msg(99999, "hello from a stranger", t_ok),
           msg(int(student), "a real point about kernels", t_ok),
           msg(int(student), "said during the break", (sum(sc.meta["gap_wall_ms"]) // 2)),
           msg(int(student), "   ", t_ok)]
    chat, counts = events.chat_table(sc.class_id, events.allowed_events(raw), clock, roster,
                                     salt=SALT)
    assert list(chat["text_masked"]) == ["a real point about kernels"]
    assert (counts["notice"], counts["non_roster"], counts["gap"], counts["empty"]) == (1, 1, 1, 1)


# =========================================================================== segments


def test_segments_have_types_and_no_end(built, world):
    common = events.common_words()
    for sc in world:
        _, et = built[sc.class_id]
        segs = et.segments
        schemas.validate(segs, "segments", strict=True)
        assert not any("end" in c for c in segs.columns)
        assert list(segs["seg_type"]) == list(sc.segments["seg_type"])
        assert list(segs["seg_idx"]) == list(range(len(segs)))
        planted, _ = events.segments_table(sc.class_id, events.allowed_events(sc.raw["events"]),
                                           _clock(sc))
        assert np.allclose(planted["start_s"], sc.segments["start_s"], atol=1e-6)
        titles = " ".join(segs["title_masked"])
        for name in pd.concat([sc.roster["first"], sc.roster["last"]]):
            if name.lower() not in common:
                assert not re.search(rf"\b{name}\b", titles), name
        if sc.meta["index"] % 2 == 1:
            assert segs["title_masked"].str.contains(r"\[NAME\]", regex=True).any()


# =========================================================================== polls


def test_parse_clock_string():
    assert events.parse_clock_string("started at 0:32:39") == 1959.0
    assert events.parse_clock_string("Started at 12:05") == 725.0
    assert events.parse_clock_string("closed at 1:02:03", events._POLL_CLOSE_RE) == 3723.0
    assert events.parse_clock_string("started at 1:02:03, ended at 1:05:00",
                                     events._POLL_CLOSE_RE) == 3900.0
    assert events.parse_clock_string(None) is None
    assert events.parse_clock_string("no time here") is None


def test_polls_table_parses_reference_and_my_answer(built, world):
    for sc in world:
        _, et = built[sc.class_id]
        schemas.validate(et.polls, "polls", strict=True)
        truth = sc.polls.iloc[0]
        clock = _clock(sc)
        roster = events.roster_from_class(sc.raw["class"], sc.meta["me_id"])
        segs, _ = events.segments_table(sc.class_id, events.allowed_events(sc.raw["events"]),
                                        clock)
        other = {"id": 1, "poll-id": int(truth["poll_id"]), "user-id": 2002,
                 "choice-id": 2, "answer": "SOMEONE ELSE", "submitted-at": "2030-01-01T00:00:00Z"}
        responses = events.whitelist_filter("poll_responses",
                                             list(sc.raw["poll_responses"]) + [other])
        p = events.polls_table(sc.class_id, events.whitelist_filter("polls", sc.raw["polls"]),
                               responses, clock, roster, segs, 0.5).iloc[0]
        assert p["poll_id"] == truth["poll_id"]
        assert -1.0 - 1e-6 <= p["open_s"] - truth["open_s"] <= 1e-6      # whole-second string
        assert p["close_s"] == pytest.approx(truth["close_s"], abs=1e-6)
        assert p["my_answer_text"] == truth["my_answer"]
        assert p["my_answer_avail_s"] == pytest.approx(
            truth["my_submit_s"] + config.eps_f_poll(0.5), abs=1e-3)
        assert "Which matters more" in p["prompt_masked"]
        ref = events.poll_reference_check(et.polls, et.segments)
        assert (ref["delta_s"].abs() < 6.0).all()


def test_poll_answer_without_submit_time_is_available_at_close(world):
    sc = world.classes[0]
    clock = _clock(sc)
    roster = events.roster_from_class(sc.raw["class"], sc.meta["me_id"])
    segs, _ = events.segments_table(sc.class_id, events.allowed_events(sc.raw["events"]), clock)
    resp = [{k: v for k, v in r.items() if k != "submitted-at"}
            for r in events.whitelist_filter("poll_responses", sc.raw["poll_responses"])]
    p = events.polls_table(sc.class_id, events.whitelist_filter("polls", sc.raw["polls"]), resp,
                           clock, roster, segs, 1.2).iloc[0]
    assert p["my_answer_avail_s"] == pytest.approx(p["close_s"] + config.eps_f_poll(1.2))
    video = events.polls_table(sc.class_id, events.whitelist_filter("polls", sc.raw["polls"]),
                               resp, clock, roster, segs, 1.2, reference="video").iloc[0]
    assert video["open_s"] == pytest.approx(p["open_s"])            # one session: same thing
    with pytest.raises(ValueError):
        events.polls_table(sc.class_id, sc.raw["polls"], resp, clock, roster, segs, 1.0,
                           reference="bogus")


# =========================================================================== presence


def test_presence_from_enter_times(built, world, truth):
    for sc in world:
        _, et = built[sc.class_id]
        schemas.validate(et.presence, "presence", strict=True)
        want = truth[sc.class_id]["presence"]
        assert len(et.presence) == len(want) == int(sc.roster["present"].sum())
        planted = events.presence_table(sc.class_id, events.roster_from_class(
            sc.raw["class"], sc.meta["me_id"]), _clock(sc), salt=SALT)
        assert np.allclose(np.sort(planted["enter_s"]), np.sort(want["enter_s"]), atol=1e-3)
        assert (planted["enter_s"] < 0).any()                        # joined before recording


def test_presence_join_during_gap_counts_from_restart(world):
    sc = world.classes[-1]
    klass = json.loads(json.dumps(sc.raw["class"]))
    mid = sum(sc.meta["gap_wall_ms"]) / 2
    klass["class-users"][2]["enter-class-time"] = synthetic._iso(mid)
    roster = events.roster_from_class(klass, sc.meta["me_id"])
    clock = _clock(sc)
    pres = events.presence_table(sc.class_id, roster, clock, salt=SALT)
    uid = str(klass["class-users"][2]["user"]["id"])
    row = pres[pres["spk_hash"] == pseudonym(uid, SALT)]
    assert row["enter_s"].iloc[0] == pytest.approx(clock.restarts()[0])


def test_presence_ignores_the_post_class_absent_flag(world):
    """`absent` is post-class attendance bookkeeping: a user with an enter-class-time is
    present from that time on (ctx_n_present uses no future information)."""
    sc = world.classes[0]
    klass = json.loads(json.dumps(sc.raw["class"]))
    users = [u for u in klass["class-users"] if u.get("enter-class-time")]
    users[0]["absent"] = True                                     # marked absent after class
    roster = events.roster_from_class(klass, sc.meta["me_id"])
    pres = events.presence_table(sc.class_id, roster, _clock(sc), salt=SALT)
    assert len(pres) == len(users)
    uid = str(users[0]["user"]["id"])
    assert (pres["spk_hash"] == pseudonym(uid, SALT)).any()


# =========================================================================== hand log (QA)


def test_hand_log_import_matches_onsets(world, truth):
    sc = world.classes[0]
    eps = truth[sc.class_id]["hand_episodes"]
    bids = eps.loc[eps["ep_type"] == config.EP_BID, "onset_s"].to_numpy()
    entries = [{"t": _wall(sc, o - 0.4), "classId": sc.class_id} for o in bids]
    entries += [{"t": _wall(sc, 300.0), "classId": 1},                  # another class
                {"t": sc.meta["sessions"][0][0] - 3_600_000},           # untagged, not recorded
                {"t": synthetic._iso(_wall(sc, 400.0))}]                # untagged, recorded
    clock = _clock(sc)
    log = events.hand_log_import({"entries": entries}, sc.class_id, clock, episodes=eps)
    schemas.validate(log, "hand_log", strict=True)
    assert len(log) == len(bids) + 1
    matched = log.dropna(subset=["matched_onset_s"])
    assert np.allclose(np.sort(matched["matched_onset_s"]), np.sort(bids), atol=1e-6)
    assert np.allclose(matched["lag_s"], 0.4, atol=2e-3)
    bare = events.hand_log_import(entries, sc.class_id, clock)
    assert bare["matched_onset_s"].isna().all() and len(bare) == len(log)
    assert len(events.hand_log_import([], sc.class_id, clock)) == 0


# =========================================================================== words


def test_words_table_schema_roles_and_availability(built, world):
    for sc in world:
        b, _ = built[sc.class_id]
        w = b.words
        schemas.validate(w, "words", strict=True)
        truth = sc.words.sort_values(["start_s", "end_s"]).reset_index(drop=True)
        assert len(w) == len(truth) and list(w["w_idx"]) == list(range(len(w)))
        assert (np.diff(w["start_s"]) >= 0).all()
        assert (w["role"].to_numpy() == truth["role"].to_numpy()).mean() >= 0.99
        eps = b.timing["eps_F_s"]
        expect = np.maximum(w["end_s"] + config.EPS_W_S,
                            (w["role_event_start_s"] + eps).fillna(-np.inf))
        # no mic on -> professor (A12), known once a covering mic event would have shown up
        quiet = w["role_event_start_s"].isna()
        assert (w.loc[quiet, "role"] == config.ROLE_PROF).all()
        expect = np.where(quiet, w["end_s"] + EDGE + eps, expect)
        assert np.allclose(w["role_avail_s"], expect)
        unk = w["role"] == config.ROLE_UNK
        assert w.loc[unk, "role_event_start_s"].isna().all() == bool(
            w.loc[unk, "spk_hash"].isna().all())
        known = w["spk_hash"].notna()
        ids = truth.loc[known.to_numpy(), "user_id"].map(lambda u: pseudonym(u, SALT))
        assert (w.loc[known, "spk_hash"].to_numpy() == ids.to_numpy()).mean() >= 0.99
        assert set(w["role_fused"]) <= set(config.ROLES)
        assert set(w["how"]) <= set(HOW_RANK)


def test_words_masking_callon_and_normalisation(built, world):
    common = events.common_words()
    for sc in world:
        b, _ = built[sc.class_id]
        w = b.words
        truth = sc.words.sort_values(["start_s", "end_s"]).reset_index(drop=True)
        masked = w["text_masked"].str.contains(r"\[(?:NAME|ME)\]", regex=True).to_numpy()
        assert (masked == truth["is_name"].to_numpy()).all()
        me_tok = w["text_masked"].str.contains(r"\[ME\]", regex=True)
        assert me_tok.any()
        assert (truth.loc[me_tok.to_numpy(), "word"] == "brady").all()
        norm = w["text_norm"]
        assert norm.str.fullmatch(r"[a-z0-9_]*").all()
        # per word, no look-ahead: names and one-word CALLONs ("unmute") are deleted; the
        # words of a multi-word phrase ("go", "ahead") stay until text is joined
        one_word = truth["word"].map(lambda x: bool(config.CALLON_RE.fullmatch(x))).to_numpy()
        gone = truth["is_name"].to_numpy() | (truth["is_callon"].to_numpy() & one_word)
        assert (norm[gone] == "").all() and (norm[~gone] != "").all()
        joined = " ".join(t for t in norm if t)
        assert "brady" not in joined
        frames = causal.frames_table(sc.class_id, w)          # joined feature text
        assert not frames["text_norm"].str.contains(config.CALLON_RE).any()
        for name in pd.concat([sc.roster["first"], sc.roster["last"]]):
            if name.lower() not in common:
                assert not re.search(rf"\b{name.lower()}\b", joined), name
        assert (w["has_qmark"].to_numpy() == truth["text"].str.endswith("?").to_numpy()).all()
        assert (w["filler"].to_numpy() == truth["filler"].to_numpy()).all()


def test_role_does_not_depend_on_clusters(world):
    """Roles come from Forum alone. Clusters reach them only through the clock fit, which
    leaves out the professor's cluster (A12), so that cluster is kept and every other word's
    cluster is scrambled."""
    sc = world.classes[0]
    cal = transcribe.calibrate_class(sc.asr, sc.raw["class"], sc.raw["events"])
    tc = teacher_cluster(cal.words, cal.voice_raw, min_share=0.5)
    assert tc is not None
    rng = np.random.default_rng(0)
    scrambled = {**sc.asr, "words": [
        x if x.get("cluster") == tc else {**x, "cluster": 100 + int(rng.integers(0, 3))}
        for x in sc.asr["words"]]}
    a = transcribe.build_words(sc.class_id, sc.asr, sc.raw["class"], sc.raw["events"],
                               me_id=sc.meta["me_id"], salt=SALT).words
    c = transcribe.build_words(sc.class_id, scrambled, sc.raw["class"], sc.raw["events"],
                               me_id=sc.meta["me_id"], salt=SALT).words
    cols = ["role", "spk_hash", "role_event_start_s", "role_avail_s", "text_norm"]
    pd.testing.assert_frame_equal(a[cols], c[cols])
    assert (a["role_fused"] != c["role_fused"]).any() or (a["how"] != c["how"]).any()


def test_forum_roles_most_overlap_edge_and_ties():
    # words: [10, 10.3], [20, 20.4], [30, 30.2], [40, 40.2]
    ws, we = [10.0, 20.0, 30.0, 40.0], [10.3, 20.4, 30.2, 40.2]
    # ev0 barely touches word 1; ev1 covers it; ev2 starts 0.2 s after word 3 ends (inside the
    # 0.25 s edge, L5); ev3 starts 0.3 s after word 4 ends (outside)
    vs = [9.0, 20.1, 30.4, 40.5]
    ve = [20.05, 25.0, 33.0, 42.0]
    best = transcribe.forum_roles(ws, we, vs, ve)
    assert list(best) == [0, 1, 2, -1]
    tie = transcribe.forum_roles([5.0], [5.2], [4.0, 4.5], [6.0, 6.0])
    assert list(tie) == [0]                                   # equal overlap: earlier event
    assert list(transcribe.forum_roles([], [], vs, ve)) == []
    assert list(transcribe.forum_roles(ws, we, [], [])) == [-1] * 4


def test_per_word_text_never_looks_ahead():
    """A word's text_norm depends on that word alone: "go" is the same whether "ahead" or
    "home" follows (multi-word CALLONs are removed from joined available text later)."""
    assert [transcribe.norm_word(x, False) for x in ["Go", "ahead."]] == ["go", "ahead"]
    assert [transcribe.norm_word(x, False) for x in ["Go", "home."]] == ["go", "home"]
    assert transcribe.norm_word("Unmute,", False) == ""            # one-word CALLON
    assert transcribe.norm_word("Ada,", True) == ""                # masked name
    assert causal.join_tokens(["so", "go", "ahead", "please"]) == "so please"
    assert causal.join_tokens(["so", "go"]) == "so go"             # "ahead" not heard yet


def test_transcript_events_are_never_read(world):
    sc = world.classes[0]

    class Guarded(dict):
        def __getitem__(self, key):
            if key in ("events", "participation"):
                raise AssertionError(f"read transcript {key!r}")
            return super().__getitem__(key)

        def get(self, key, default=None):
            if key in ("events", "participation"):
                raise AssertionError(f"read transcript {key!r}")
            return super().get(key, default)

    tr = Guarded(synthetic.scribe_transcript(sc, salt=SALT))
    assert any(e["type"] == "hand-raise-key" for e in dict.__getitem__(tr, "events"))
    b = transcribe.build_words(sc.class_id, sc.asr, sc.raw["class"], sc.raw["events"],
                               me_id=sc.meta["me_id"], salt=SALT, transcript=tr)
    assert b.timing["n_words_transcript"] == len(b.words)
    assert set(transcribe.read_transcript_meta(tr)) == {
        "calibration", "n_words", "timing_precision", "source_res", "provider"}


# =========================================================================== masker


def test_roster_masker_rules():
    m = events.RosterMasker(["Ada Lovelace", "Will Herschel", "Grace Hopper", "Al Kindi"],
                            ["Brady Rutherford"])
    assert m.mask_text("Thanks Will for that.") == "Thanks [NAME] for that."
    assert m.mask_text("We will see.") == "We will see."
    assert m.mask_text("Will you explain?") == "Will you explain?"        # sentence-initial verb
    assert m.mask_text("Will, go ahead.") == "[NAME], go ahead."          # vocative
    assert m.mask_text("I agree with Lovelase here.") == "I agree with [NAME] here."
    assert m.mask_text("Lovelace's point was good.") == "[NAME] point was good."
    assert m.mask_text("so Hopper, then.") == "so [NAME], then."
    assert m.mask_text("brady, can you") == "[ME], can you"
    assert m.mask_text("Thanks, Brady.") == "Thanks, [ME]."
    assert m.mask_text("I asked Alx about it.") == "I asked Alx about it."  # names < 4: exact only
    assert m.mask_text("I asked Al about it.") == "I asked [NAME] about it."
    assert m.mask_text(None) is None
    toks, kinds = m.mask_tokens(["Ada,", "go", "ahead"], [False, False, False])
    assert toks == ["[NAME],", "go", "ahead"] and kinds == ["name", None, None]


# =========================================================================== transcribe_many


def test_me_id_is_required(world):
    sc = world.classes[0]
    with pytest.raises(ValueError, match="me.json"):
        transcribe.build_words(sc.class_id, sc.asr, sc.raw["class"], sc.raw["events"],
                               me_id=None, salt=SALT)
    with pytest.raises(ValueError, match="me.json"):
        events.build_event_tables(sc.class_id, klass=sc.raw["class"], events=sc.raw["events"],
                                  me_id="", eps_f_s=0.5, salt=SALT)


def test_course_code_falls_back_to_listing(world):
    sc = world.classes[0]
    klass = {k: v for k, v in sc.raw["class"].items() if k != "section"}
    assert transcribe.course_code_of({"class": sc.raw["class"]}) == "CS901"
    assert transcribe.course_code_of({"class": klass, "listing": sc.raw["listing"]}) == "CS901"
    assert transcribe.course_code_of({"class": klass}) is None


def test_provider_per_course_and_consent():
    assert transcribe.provider_for("CS901", mapping={}) == "local"
    cleared = {"CS901": "deepgram"}
    assert transcribe.provider_for("CS901", mapping=cleared) == "deepgram"
    assert transcribe.provider_for("CS901", "local", mapping=cleared) == "local"
    with pytest.raises(ValueError):
        transcribe.provider_for("SS902", "deepgram", mapping=cleared)
    with pytest.raises(ValueError):
        transcribe.provider_for("CS901", "groq", mapping=cleared)


class _Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, cid, client, **kwargs):
        self.calls.append((cid, client, kwargs))
        return {"summary": {"n_words": 10, "calibration_status": "ok"}}


def test_transcribe_many_options_and_never_hand_log(world, data_dir, tmp_path):
    rec = _Recorder()
    rows = transcribe.transcribe_many(
        world.class_ids, workers=3, out_dir=tmp_path / "t", raw_dir=data_dir["raw"],
        scribe_dir=data_dir["scribe"], salt=SALT, require_strict_res=False,
        provider_map={"CS901": "deepgram"}, transcribe_fn=rec, client_factory=lambda: None,
        run_log=tmp_path / "runs.csv")
    assert [r["status"] for r in rows] == ["ok"] * len(world)
    assert len(rec.calls) == len(world)
    for cid, client, kw in rec.calls:
        sc = world.by_id(cid)
        assert client is None and "hand_log" not in kw
        opts = kw["options"]
        assert opts.privacy == "ids" and tuple(opts.formats) == ("json", "jsonl")
        assert opts.language == "en" and opts.debug_html is False and opts.save_asr is True
        assert opts.res == config.SRC_RES and opts.salt == SALT
        assert opts.provider == ("deepgram" if sc.meta["course_code"] == "CS901" else "local")
        if transcribe.scribe_has_strict_res():
            assert opts.strict_res is True
        bundle = kw["bundle"]
        assert set(bundle) >= {"class", "events"} and bundle["me"]["id"] == str(synthetic.ME_ID)
        assert not ({"analytics", "outcome_assessments", "focused_outcomes"} & set(bundle))
    assert len(pd.read_csv(tmp_path / "runs.csv")) == len(world)


def test_transcribe_many_isolates_failures_and_strict_res(world, data_dir, tmp_path):
    rec = _Recorder()
    rows = transcribe.transcribe_many(
        [world.class_ids[0], 4242], out_dir=tmp_path, raw_dir=data_dir["raw"],
        scribe_dir=data_dir["scribe"], salt=SALT, require_strict_res=False, transcribe_fn=rec,
        client_factory=lambda: None, run_log=False)
    assert [r["status"] for r in rows] == ["ok", "error"]
    assert "FileNotFoundError" in rows[1]["error"]
    rows = transcribe.transcribe_many(
        [world.class_ids[1]], provider="deepgram", out_dir=tmp_path, raw_dir=data_dir["raw"],
        scribe_dir=data_dir["scribe"], salt=SALT, require_strict_res=False, transcribe_fn=rec,
        client_factory=lambda: None, run_log=False)
    assert rows[0]["status"] == "error" and "third-party" in rows[0]["error"]
    if not transcribe.scribe_has_strict_res():
        before = len(rec.calls)
        rows = transcribe.transcribe_many(
            [world.class_ids[0]], out_dir=tmp_path, raw_dir=data_dir["raw"],
            scribe_dir=data_dir["scribe"], salt=SALT, transcribe_fn=rec,
            client_factory=lambda: None, run_log=False)
        assert rows[0]["status"] == "error" and "strict_res" in rows[0]["error"]
        assert len(rec.calls) == before


class _Replay(ASRProvider):
    """Replays a synthetic *_asr.json (no audio, no network)."""

    name = "replay"
    needs_audio = False
    diarizes = True

    def __init__(self, asr):
        super().__init__("replay-1")
        self.asr = asr

    def transcribe(self, **_):
        a = self.asr
        return ASRResult(words=a["words"], provider="replay", model="replay-1", language="en",
                         duration=a["duration"], diarized=True, mode="saved")


def test_offline_scribe_run_then_words_rebuild_matches(small_world, tmp_path):
    """D7: run the REAL Scribe pipeline offline, then rebuild words from its outputs."""
    from scribe_core.pipeline import transcribe_class

    paths = synthetic.write_world(small_world, tmp_path / "data")
    out = tmp_path / "transcripts"
    asr = {sc.class_id: sc.asr for sc in small_world}

    def run(cid, client, **kw):
        assert "hand_log" not in kw
        return transcribe_class(cid, client, **{**kw, "provider": _Replay(asr[cid])})

    rows = transcribe.transcribe_many(
        small_world.class_ids, workers=2, out_dir=out, raw_dir=paths["raw"],
        scribe_dir=paths["scribe"], salt=SALT, require_strict_res=False, transcribe_fn=run,
        client_factory=lambda: None, workdir_root=tmp_path / "work", run_log=False)
    assert [r["status"] for r in rows] == ["ok", "ok"]
    for sc in small_world:
        ids_path = transcribe.find_transcript(sc.class_id, "ids", out)
        assert ids_path and transcribe.find_transcript(sc.class_id, "asr", out)
        tr = json.loads(ids_path.read_text())
        assert not any(e.get("type") == "hand-raise-key" for e in tr["events"])
        b = transcribe.load_words_build(sc.class_id, raw_dir=paths["raw"], transcripts_dir=out,
                                        scribe_dir=paths["scribe"], salt=SALT)
        assert b.timing["n_words"] == b.timing["n_words_transcript"] == len(sc.words)
        assert b.timing["cal_matches_transcript"] is True
        schemas.validate(b.words, "words", strict=True)
    again = transcribe.transcribe_many(
        small_world.class_ids, out_dir=out, raw_dir=paths["raw"], scribe_dir=paths["scribe"],
        salt=SALT, require_strict_res=False, transcribe_fn=run, client_factory=lambda: None,
        run_log=False)
    assert [r["status"] for r in again] == ["skipped", "skipped"]


def test_find_transcript_does_not_confuse_prefixes(tmp_path):
    (tmp_path / "CS_2026-01-01_class12_asr.json").write_text("{}")
    (tmp_path / "CS_2026-01-01_class123_asr.json").write_text("{}")
    assert transcribe.find_transcript(12, "asr", tmp_path).name.endswith("class12_asr.json")
    assert transcribe.find_transcript(123, "asr", tmp_path).name.endswith("class123_asr.json")
    assert transcribe.find_transcript(1, "asr", tmp_path) is None
    assert transcribe.find_transcript(12, "ids", tmp_path) is None


# =========================================================================== CLI


@pytest.fixture
def sandbox(data_dir, tmp_path, monkeypatch, id_salt):
    """Point every config path at the synthetic world; never read a real .env."""
    monkeypatch.setattr(config, "RAW_DIR", data_dir["raw"])
    monkeypatch.setattr(config, "TRANSCRIPTS_DIR", data_dir["transcripts"])
    monkeypatch.setattr(config, "ME_PATH", data_dir["scribe"] / "me.json")
    monkeypatch.setattr(config, "TABLES_DIR", tmp_path / "tables")
    monkeypatch.setattr(config, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(config, "load_env", lambda *a, **k: False)
    return tmp_path


def test_cli_words_events_and_hand_log(sandbox, world, truth):
    from oncue.__main__ import main

    cids = [str(world.classes[0].class_id), str(world.classes[-1].class_id)]
    argv = [arg for c in cids for arg in ("--class", c)]
    assert main(["words", *argv]) == 0
    words = pd.read_parquet(config.table_path("words"))
    schemas.validate(words, "words", strict=True)
    assert set(words["class_id"].astype(str)) == set(cids)
    timing = pd.read_parquet(config.table_path(transcribe.TIMING_TABLE))
    assert len(timing) == 2 and timing["cal_qa_pass"].all()
    assert main(["words", "--class", cids[0]]) == 0                 # upsert, not duplicate
    assert len(pd.read_parquet(config.table_path("words"))) == len(words)

    assert main(["events", *argv]) == 0
    for name in ("voice", "chat", "segments", "polls", "presence"):
        df = pd.read_parquet(config.table_path(name))
        schemas.validate(df, name, strict=True)
        assert set(df["class_id"].astype(str)) == set(cids), name
    voice = pd.read_parquet(config.table_path("voice"))
    assert voice["word_bearing"].any()
    assert (sandbox / "logs" / "event_counts.csv").exists()

    sc = world.classes[0]
    eps = truth[sc.class_id]["hand_episodes"]
    eps.to_parquet(config.table_path("hand_episodes"))
    bids = eps.loc[eps["ep_type"] == config.EP_BID, "onset_s"]
    log_path = sandbox / "hand_log.json"
    log_path.write_text(json.dumps({"entries": [{"t": _wall(sc, o - 0.5), "classId": sc.class_id}
                                                for o in bids]}))
    assert main(["hand-log-import", str(log_path), "--class", cids[0]]) == 0
    hl = pd.read_parquet(config.table_path("hand_log"))
    schemas.validate(hl, "hand_log", strict=True)
    assert len(hl) == len(bids) and hl["matched_onset_s"].notna().all()


def test_cli_words_discovers_transcripts_and_transcribe_needs_classes(sandbox, world, capsys):
    ids = transcribe._class_ids(Namespace(class_ids=None, phase=None))
    assert ids == sorted(world.class_ids)
    assert transcribe.cli_transcribe(Namespace(class_ids=None, phase=None, workers=1,
                                               extra=[])) == 2
    assert "pass --class" in capsys.readouterr().out
