"""§4.4 test_loader_allowlist: per-source JSON-path whitelists, blocked sources, keys and event
types, the human-actor chat rule, and column provenance. Synthetic data only."""

from __future__ import annotations

import copy
import importlib
import json
import re

import pytest
from oncue import config, loaders, synthetic
from scribe_core.forum_events import kebab

RAW_SOURCES = [s for s in config.RAW_SOURCE_FILES if s not in config.BLOCKED_SOURCES]


@pytest.fixture
def log_path(tmp_path, monkeypatch):
    """Route the dropped-keys log to a tmp file (never data/logs) and reset the dedupe set."""
    path = tmp_path / "dropped_keys.csv"
    monkeypatch.setattr(config, "DROPPED_KEYS_LOG", path)
    loaders._LOGGED.clear()
    return path


def _leaves(obj) -> set[str]:
    """Leaf paths, ignoring empty containers (kept by design, they carry no data)."""
    return {p for p in synthetic.flatten_paths(obj) if p and not p.endswith("[]")}


def _matches(rules, source, path) -> bool:
    return any(src == source and re.fullmatch(rx, path) for src, rx in rules)


def _events_of_allowed_types(events):
    return [e for e in kebab(events) if e.get("event-type") in config.EVENT_TYPES]


# --------------------------------------------------------------------------- whitelists


def test_kept_leaves_are_exactly_the_whitelisted_ones(world, data_dir, log_path):
    for sc in world:
        for source in RAW_SOURCES:
            if sc.raw.get(source) is None:
                continue
            got = loaders.load_raw(sc.class_id, source, raw_dir=data_dir["raw"])
            original = kebab(sc.raw[source])
            if source == "events":
                original = _events_of_allowed_types(original)
            orig_leaves = _leaves(original)
            expected = {p for p in orig_leaves if config.whitelist_use(source, p)}
            assert _leaves(got) == expected, (sc.class_id, source)
            for p in orig_leaves - expected:          # every drop is a trap or bookkeeping
                assert (_matches(synthetic.TRAP_PATHS, source, p)
                        or _matches(synthetic.BOOKKEEPING_PATHS, source, p)), (source, p)
            assert not any(_matches(synthetic.TRAP_PATHS, source, p) for p in _leaves(got))


def test_values_survive_unchanged(world, data_dir, log_path):
    sc = world.classes[0]
    klass = loaders.load_raw(sc.class_id, "class", raw_dir=data_dir["raw"])
    assert klass["section"]["course"]["course-code"] == sc.meta["course_code"]
    assert klass["teacher-user-id"] == int(sc.meta["prof_id"])
    assert [s["recording-started"] for s in klass["recording-sessions"]] == \
        [s["recording-started"] for s in sc.raw["class"]["recording-sessions"]]
    up = next(c for c in world if "upload" in c.pcw["answer_kinds"])
    doc = loaders.load_raw(up.class_id, "pcw_doc", raw_dir=data_dir["raw"])
    for uuid, ans in up.raw["pcw_doc"]["answers"].items():
        assert doc["answers"][uuid].get("value") == ans.get("value")
        assert "updated-at" not in doc["answers"][uuid]
    files = [f for a in doc["answers"].values() for f in a.get("files", [])]
    assert files and all(set(f) == {"name"} for f in files)       # signed url dropped


def test_events_keep_only_allowed_types_and_kebab_keys(world, data_dir, log_path):
    for sc in world:
        events = loaders.load_raw(sc.class_id, "events", raw_dir=data_dir["raw"])
        types = {e["event-type"] for e in events}
        assert types <= config.EVENT_TYPES and "bookmark-note" not in types
        assert all("event_type" not in e and "actor_id" not in e for e in events)
        want = len(_events_of_allowed_types(sc.raw["events"]))
        assert len(events) == want
        assert all("first-name" not in (e.get("actor") or {}) for e in events)


def test_blocked_event_types_and_unknown_keys_are_dropped_and_logged(log_path):
    events = [
        {"event_type": "voice", "actor_id": 7, "timestamp": 1000, "event_data": {"duration": 500},
         "event_data_extra": 1},
        {"event-type": "hand-raise-key", "actor-id": 7, "timestamp": 1200},
        {"event-type": "participation", "actor-id": 7},
        {"event-type": "bookmark-note", "actor-id": 7, "event-data": {"note": "x"}},
        {"event-type": "message", "actor-id": 7, "timestamp": 1300,
         "event-data": {"message-text": "hi", "reactions": [{"user-id": 9}]}},
    ]
    doc, dropped = loaders.filter_doc(events, "events")
    assert [e["event-type"] for e in doc] == ["voice", "message"]
    assert "event-data-extra" not in doc[0] and "reactions" not in doc[1]["event-data"]
    reasons = {path: why for path, why in dropped}
    assert reasons["[]#event-type=hand-raise-key"] == loaders.REASON_EVENT_TYPE
    assert reasons["[]#event-type=participation"] == loaders.REASON_EVENT_TYPE
    assert "[].event-data.reactions[].user-id" in reasons
    assert loaders.log_dropped(dropped, class_id=1, source="events") == len(dropped)
    assert loaders.log_dropped(dropped, class_id=1, source="events") == 0    # once per process
    log = loaders.read_dropped_log()
    assert set(log.columns) == set(loaders.DROPPED_LOG_COLUMNS)
    assert set(log["path"]) == set(reasons)


def test_dropped_log_holds_paths_not_values(world, data_dir, log_path):
    for sc in world:
        for source in RAW_SOURCES:
            loaders.load_raw(sc.class_id, source, raw_dir=data_dir["raw"])
    text = log_path.read_text()
    assert "actual-length-seconds" in text and "blocked_key" in text
    names = {n for sc in world for n in sc.roster["first"].tolist() + sc.roster["last"].tolist()}
    assert not any(re.search(rf"\b{re.escape(n)}\b", text) for n in names)
    assert "Good start" not in text and "graded" not in text
    log = loaders.read_dropped_log()
    # uuid-keyed paths are generalised, so the log stays small
    assert not log["path"].str.contains(r"[0-9a-f]{8}-[0-9a-f]{4}-", regex=True).any()
    assert log.duplicated(["class_id", "source", "path", "reason"]).sum() == 0


@pytest.mark.parametrize("source", sorted(config.BLOCKED_SOURCES))
def test_blocked_sources_raise_even_when_the_file_exists(world, data_dir, source, log_path):
    sc = world.classes[0]
    if source in config.RAW_SOURCE_FILES and sc.raw.get(source) is not None:
        assert (data_dir["raw"] / str(sc.class_id) / config.RAW_SOURCE_FILES[source]).exists()
    with pytest.raises(loaders.BlockedSourceError):
        loaders.load_raw(sc.class_id, source, raw_dir=data_dir["raw"])
    with pytest.raises(loaders.BlockedSourceError):
        loaders.filter_doc({"x": 1}, source)


def test_transcript_events_and_participation_never_load(world, data_dir, log_path):
    sc = world.classes[0]
    path = loaders.find_transcript(sc.class_id, "ids", data_dir["transcripts"])
    raw = json.loads(path.read_text())
    assert any(e["type"] == "hand-raise-key" for e in raw["events"])      # the trap is there
    assert raw["participation"]
    t = loaders.load_transcript(sc.class_id, transcripts_dir=data_dir["transcripts"])
    for key in ("events", "participation", "utterances", "speakers", "attendance"):
        assert key not in t
    with pytest.raises(loaders.BlockedKeyError):
        t["events"]
    with pytest.raises(loaders.BlockedKeyError):
        t.get("participation")
    with pytest.raises(loaders.BlockedKeyError):
        t["alignment"]["calibration"].get("events", [])          # nested dicts are guarded too
    assert t["asr"]["source-res"] == config.SRC_RES
    cal = t["alignment"]["calibration"]
    assert cal["method"] == sc.meta["calibration"]["method"]
    assert "hand-raise-key" not in json.dumps(t)


def test_calibration_windows_survive_so_qa_can_count_them(world, data_dir, log_path):
    for sc in world:
        cal = loaders.load_calibration(sc.class_id, transcripts_dir=data_dir["transcripts"])
        assert cal["windows"] == sc.meta["calibration"]["windows"]
        assert cal["max_residual_s"] == sc.meta["calibration"]["max_residual_s"]
        ok, reason = config.calibration_qa(cal, 0.2, 0.6)
        if sc.meta["calibration"]["method"] == "global":
            assert (ok, reason) == (False, "cal_method_global")
        else:
            assert ok, reason
    # an array is kept whole only when it holds no objects
    doc, dropped = loaders.filter_doc(
        {"alignment": {"calibration": {"windows": [[1, 0.5, 0.9], {"x": 1}]}}}, "transcript")
    assert "alignment" not in doc                       # nothing whitelisted left in the branch
    assert ("alignment.calibration.windows[].x", loaders.REASON_NOT_WHITELISTED) in dropped


def test_asr_words_only_and_use_filter(world, data_dir, log_path):
    sc = world.classes[0]
    asr = loaders.load_asr(sc.class_id, transcripts_dir=data_dir["transcripts"])
    assert len(asr["words"]) == len(sc.asr["words"])
    assert set(asr["words"][0]) <= {"start", "end", "text", "punctuated-word", "conf", "filler",
                                    "cluster", "cluster-conf"}
    feat = loaders.load_asr(sc.class_id, transcripts_dir=data_dir["transcripts"],
                            uses={config.USE_FEATURE})
    assert "cluster" not in feat["words"][0] and "provider" not in feat


def test_uses_filter_on_class(world, data_dir, log_path):
    sc = world.classes[0]
    klass = loaders.load_raw(sc.class_id, "class", raw_dir=data_dir["raw"],
                             uses={config.USE_FEATURE, config.USE_TIMING})
    users = klass["class-users"]
    assert users and all("first-name" not in (u.get("user") or {}) for u in users)
    assert "title" not in klass and "teacher-user-id" in klass
    assert "recording-ended" not in klass["recording-sessions"][0]          # label use


def test_leaf_semantics_nulls_and_empty_containers(log_path):
    doc = {"study_guide": {"description": "Key terms", "updated_at": "2026-01-01"},
           "preparatory_assignment": None, "assessment": {"focused_outcomes": []},
           "grades": [{"score": 3}]}
    out, dropped = loaders.filter_doc(doc, "class_edit_page")
    assert out["study-guide"] == {"description": "Key terms"}
    assert out["preparatory-assignment"] is None
    assert out["assessment"] == {"focused-outcomes": []}
    assert "grades" not in out
    assert {p for p, _ in dropped} == {"study-guide.updated-at", "grades[].score"}
    klass, _ = loaders.filter_doc({"class-users": [], "teacher-user-id": None}, "class")
    assert klass == {"class-users": [], "teacher-user-id": None}


def test_guarded_dict_behaves_like_a_dict():
    g, _ = loaders.filter_doc({"id": 1, "date": "2026-09-09"}, "class")
    assert isinstance(g, dict) and json.loads(json.dumps(g)) == {"id": 1, "date": "2026-09-09"}
    assert copy.deepcopy(g) == g and "events" not in g
    with pytest.raises(loaders.BlockedKeyError):
        g["participation"]
    with pytest.raises(loaders.BlockedKeyError):
        g.get("hand_raise_key")


def test_unknown_source_and_missing_files(tmp_path, log_path):
    with pytest.raises(ValueError):
        loaders.filter_doc({}, "nonsense")
    with pytest.raises(ValueError):
        loaders.raw_path(1, "asr")
    assert loaders.load_raw(123, "class", raw_dir=tmp_path) is None
    with pytest.raises(FileNotFoundError):
        loaders.load_raw(123, "class", raw_dir=tmp_path, missing_ok=False)


def test_schedule_me_and_manifest_loaders(world, data_dir, log_path):
    sid = world.classes[0].meta["section_id"]
    sched = loaders.load_schedule(sid, schedule_dir=data_dir["scribe"] / "schedule")
    assert [r["id"] for r in sched["results"]] == \
        [r["id"] for r in world.schedules[sid]["results"]]
    assert loaders.me_id(data_dir["scribe"] / "me.json") == str(synthetic.ME_ID)
    man = loaders.load_manifest(data_dir["manifest"])
    assert list(man["class_id"]) == world.class_ids
    assert "n_voice" not in man and "n_my_voice" not in man       # whole-class aggregates


# --------------------------------------------------------------------------- chat actors


def test_every_chat_actor_is_a_human_in_class_users(world, data_dir, log_path):
    for sc in world:
        klass = loaders.load_raw(sc.class_id, "class", raw_dir=data_dir["raw"])
        events = loaders.load_raw(sc.class_id, "events", raw_dir=data_dir["raw"])
        humans = loaders.human_actor_ids(klass)
        assert humans == set(sc.roster["user_id"])
        msgs = loaders.human_messages(events, klass)
        assert msgs and all(str(m["actor-id"]) in humans for m in msgs)
        assert not any("raised their hand" in m["event-data"]["message-text"] for m in msgs)
        rep = loaders.message_actor_report(events, klass)
        assert rep["n_null_actor"] >= 1 and rep["n_human"] == len(msgs)
        truth = sc.chat[~sc.chat["system"] & ~sc.chat["private"]]
        assert len(msgs) == len(truth)          # includes the in-gap message (dropped later)


def test_non_roster_actor_is_dropped():
    klass = {"class-users": [{"role": "student", "user": {"id": 1}}]}
    events = [{"event-type": "message", "actor-id": 1, "event-data": {"message-text": "a"}},
              {"event-type": "message", "actor-id": 99, "event-data": {"message-text": "b"}},
              {"event-type": "message", "actor-id": None, "event-data": {"message-text": "c"}},
              {"event-type": "voice", "actor-id": 1}]
    assert [m["event-data"]["message-text"] for m in loaders.human_messages(events, klass)] == \
        ["a"]
    assert loaders.message_actor_report(events, klass) == \
        {"n_messages": 3, "n_human": 1, "n_non_roster": 1, "n_null_actor": 1}


# --------------------------------------------------------------------------- provenance


def test_provenance_paths_are_all_whitelisted():
    assert loaders.provenance_problems() == []
    for key in loaders.PROVENANCE:
        table, _, col = key.partition(".")
        assert table and col


def test_feature_safety_of_provenance_keys():
    for key in ("voice.start_s", "voice.role", "chat.text_masked", "segments.title_masked",
                "words.text_norm", "classes.course_code", "classes.session_no",
                "pcw_blocks.answer_text", "presence.enter_s"):
        assert loaders.feature_safe(key), key
    for key in ("classes.T_end_s", "classes.in_vault", "words.role_fused"):
        assert not loaders.feature_safe(key), key


def test_feature_provenance_checker():
    good = {"spk_cur_elapsed_s": ["voice.start_s"], "ctx_course": ["classes.course_code"],
            "q_open": ["words.text_norm", ("events", "[].event-data.timeline-segment-title")]}
    assert loaders.feature_provenance_problems(good) == []
    bad = {"ctx_left": ["classes.T_end_s"], "x": [("class", "actual-length-seconds")],
           "y": ["nope.col"]}
    problems = loaders.feature_provenance_problems(bad)
    assert any("ctx_left" in p and "label" in p for p in problems)
    assert any(p.startswith("x:") for p in problems) and any(p.startswith("y:") for p in problems)


def test_features_provenance_traces_to_whitelisted_paths():
    try:
        features = importlib.import_module("oncue.features")
    except ModuleNotFoundError:
        pytest.skip("oncue.features not written yet")
    prov = getattr(features, "PROVENANCE", None)
    if not prov:
        pytest.skip("features.PROVENANCE not defined yet")
    assert loaders.feature_provenance_problems(prov) == []
