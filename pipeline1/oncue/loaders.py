"""Whitelisted raw loaders (spec §4 rule (c), §4.4, trap L11/L26/L27).

Every stage reads harvested JSON through here, never with a bare `json.load`:

* `load_raw(cid, source)` reads `data/scribe/raw/{cid}/{file}`, normalises keys with
  `scribe_core.forum_events.kebab()`, and keeps only leaf paths listed in
  `config.RAW_PATH_WHITELIST[source]`. Everything else is dropped and logged (generic path
  and count only, never values) to `config.DROPPED_KEYS_LOG`.
* `class-events` keep only the event types in `config.EVENT_TYPES`; blocked sources
  (`config.BLOCKED_SOURCES`) raise `BlockedSourceError`.
* Dicts come back as `GuardedDict`: reading a blocked key such as a transcript's `events` or
  `participation` raises `BlockedKeyError` instead of returning None, so a feature build that
  tries it fails loudly.
* `PROVENANCE` maps each raw-derived table column ("table.column") to the whitelisted
  (source, path) pairs it is computed from; `features.PROVENANCE` chains onto it.

Leaf semantics: a value is kept only if its own full path is whitelisted. A whitelisted path
that turns out to hold an object keeps only those of its leaves that are listed too; one that
holds an array with no objects inside (e.g. calibration `windows`, [[t, offset, score], ...])
is kept whole. Empty containers and nulls on the way to a whitelisted path are kept (they
carry no data).
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from functools import cache, lru_cache
from pathlib import Path

import pandas as pd

from . import config


class BlockedSourceError(PermissionError):
    """A whole source is off limits (post-class grading, staff analytics)."""


class BlockedKeyError(KeyError):
    """A blocked key (transcript `events`, `participation`, ...) was read."""


class GuardedDict(dict):
    """A dict that raises `BlockedKeyError` when a blocked key is read.

    Loaded documents never contain blocked keys, so a read can only come from code that
    expects them (e.g. a feature reading transcript `events`): that must fail, not return
    None or an empty list.
    """

    @staticmethod
    def _check(key) -> None:
        if isinstance(key, str) and key.replace("_", "-") in config.BLOCKED_KEYS:
            raise BlockedKeyError(f"{key!r} is a blocked key (spec §4.2); it is never loaded")

    def __getitem__(self, key):
        self._check(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        self._check(key)
        return super().get(key, default)


# --------------------------------------------------------------------------- whitelist lookup

RAW_FILE_SOURCES = tuple(config.RAW_SOURCE_FILES)          # sources living in raw/{cid}/
DROPPED_LOG_COLUMNS = ("class_id", "source", "path", "reason", "count")
REASON_NOT_WHITELISTED = "not_whitelisted"
REASON_BLOCKED_KEY = "blocked_key"
REASON_EVENT_TYPE = "event_type"
_GENERIC_KEY = re.compile(r"^(?:\d+|[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}"
                          r"-[0-9a-fA-F]{12}|[0-9a-fA-F]{24,})$")


@lru_cache(maxsize=200_000)
def _use(source: str, path: str) -> str | None:
    return config.whitelist_use(source, path)


@cache
def _prefix_res(source: str) -> tuple[re.Pattern, ...]:
    """Regexes for every container prefix of every whitelisted path of `source`."""
    out = set()
    for path in config.RAW_PATH_WHITELIST.get(source, {}):
        segs = path.split(".")
        for i in range(1, len(segs) + 1):
            prefix = ".".join(segs[:i])
            out.add(prefix)
            if prefix.endswith("[]"):
                out.add(prefix[:-2])
    out.discard("")
    return tuple(config._path_regex(p) for p in sorted(out))


@lru_cache(maxsize=50_000)
def _is_prefix(source: str, path: str) -> bool:
    return path == "" or any(rx.match(path) for rx in _prefix_res(source))


def _generic(path: str) -> str:
    """Replace id-like dict keys (digits, uuids, hashes) with '*' for the dropped-keys log."""
    return ".".join("*" if _GENERIC_KEY.match(seg) else seg for seg in path.split("."))


def _reason(path: str) -> str:
    keys = {seg.removesuffix("[]") for seg in path.split(".")}
    return REASON_BLOCKED_KEY if keys & config.BLOCKED_KEYS else REASON_NOT_WHITELISTED


_DROP = object()


def _dict_free(node) -> bool:
    """True for a list (possibly nested) holding only scalars."""
    return all(_dict_free(v) if isinstance(v, list) else not isinstance(v, dict) for v in node)


class _Walker:
    def __init__(self, source: str, uses):
        self.source = source
        self.uses = None if uses is None else frozenset(uses)
        self.dropped: Counter = Counter()

    def walk(self, node, path: str):
        if isinstance(node, dict):
            out = GuardedDict()
            for key, value in node.items():
                child = self.walk(value, f"{path}.{key}" if path else str(key))
                if child is not _DROP:
                    dict.__setitem__(out, key, child)
            if out or path == "" or (not node and _is_prefix(self.source, path)):
                return out
            return _DROP
        if isinstance(node, list):
            use = _use(self.source, path) if path else None
            if use is not None and _dict_free(node):     # e.g. calibration windows [[t, o, s]]
                return node if self.uses is None or use in self.uses else _DROP
            lp = path + "[]"
            out = [c for c in (self.walk(v, lp) for v in node) if c is not _DROP]
            if out or path == "" or (not node and _is_prefix(self.source, lp)):
                return out
            return _DROP
        use = _use(self.source, path)
        if use is None:
            if node is None and _is_prefix(self.source, path):
                return None
            self.dropped[(_generic(path), _reason(path))] += 1
            return _DROP
        if self.uses is not None and use not in self.uses:
            return _DROP
        return node


def _event_type(ev: dict) -> str | None:
    et = ev.get("event-type")
    return None if et is None else str(et)


def filter_doc(doc, source: str, *, uses=None) -> tuple[object, Counter]:
    """(whitelisted copy of `doc`, Counter{(generic path, reason): n dropped}).

    `doc` may use snake_case or kebab-case keys; the result is kebab-case. `uses` restricts
    the kept leaves to those whitelisted for these uses (e.g. {"feature", "timing"}); leaves
    filtered by `uses` are not counted as dropped. Raises `BlockedSourceError` for blocked
    sources and ValueError for sources without a whitelist.
    """
    from scribe_core.forum_events import kebab

    if source in config.BLOCKED_SOURCES:
        raise BlockedSourceError(f"source {source!r} is blocked (spec §4.2); never load it")
    if source not in config.RAW_PATH_WHITELIST:
        raise ValueError(f"no whitelist for source {source!r}")
    doc = kebab(doc)
    dropped: Counter = Counter()
    if source == "events" and isinstance(doc, list):
        kept = []
        for ev in doc:
            et = _event_type(ev) if isinstance(ev, dict) else None
            if et in config.EVENT_TYPES:
                kept.append(ev)
            else:
                dropped[(f"[]#event-type={et}", REASON_EVENT_TYPE)] += 1
        doc = kept
    walker = _Walker(source, uses)
    out = walker.walk(doc, "")
    dropped.update(walker.dropped)
    return out, dropped


# --------------------------------------------------------------------------- dropped-keys log

_LOGGED: set = set()


def log_dropped(dropped: Counter, *, class_id=None, source: str, log_path=None) -> int:
    """Append new (class_id, source, path, reason) rows to the dropped-keys CSV.

    Each row is written once per process; returns the number of rows written. `log_path`
    False disables logging; None means `config.DROPPED_KEYS_LOG` (read at call time).
    """
    if log_path is False or not dropped:
        return 0
    path = Path(log_path) if log_path is not None else Path(config.DROPPED_KEYS_LOG)
    rows = []
    for (p, reason), n in sorted(dropped.items()):
        key = (str(path), str(class_id), source, p, reason)
        if key in _LOGGED:
            continue
        _LOGGED.add(key)
        rows.append({"class_id": "" if class_id is None else class_id, "source": source,
                     "path": p, "reason": reason, "count": n})
    if not rows:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=DROPPED_LOG_COLUMNS)
        if new:
            writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def read_dropped_log(log_path=None) -> pd.DataFrame:
    path = Path(log_path) if log_path is not None else Path(config.DROPPED_KEYS_LOG)
    if not path.exists():
        return pd.DataFrame(columns=list(DROPPED_LOG_COLUMNS))
    return pd.read_csv(path, dtype={"class_id": str})


# --------------------------------------------------------------------------- loaders


def _read_json(path: Path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_json(path, source: str, *, class_id=None, uses=None, log_path=None,
              missing_ok: bool = True):
    """Read any JSON file as `source` through the whitelist (None if missing and allowed)."""
    path = Path(path)
    if source in config.BLOCKED_SOURCES:
        raise BlockedSourceError(f"source {source!r} is blocked (spec §4.2); never load it")
    if not path.exists():
        if missing_ok:
            return None
        raise FileNotFoundError(path)
    doc, dropped = filter_doc(_read_json(path), source, uses=uses)
    log_dropped(dropped, class_id=class_id, source=source, log_path=log_path)
    return doc


def raw_path(class_id, source: str, raw_dir=None) -> Path:
    if source not in config.RAW_SOURCE_FILES:
        raise ValueError(f"{source!r} is not a raw/{{cid}}/ source; use load_json")
    base = Path(raw_dir) if raw_dir is not None else config.RAW_DIR
    return base / str(class_id) / config.RAW_SOURCE_FILES[source]


def load_raw(class_id, source: str, *, raw_dir=None, uses=None, log_path=None,
             missing_ok: bool = True):
    """Whitelisted contents of `raw/{class_id}/{source file}` (kebab-case keys).

    Returns None when the file is missing (unless `missing_ok=False`). Blocked sources raise
    `BlockedSourceError` before any file is opened.
    """
    if source in config.BLOCKED_SOURCES:
        raise BlockedSourceError(f"source {source!r} is blocked (spec §4.2); never load it")
    return load_json(raw_path(class_id, source, raw_dir), source, class_id=class_id, uses=uses,
                     log_path=log_path, missing_ok=missing_ok)


def load_me(path=None, *, log_path=None) -> dict | None:
    """`me.json` from `scribe sync` ({"id", "role"})."""
    return load_json(path if path is not None else config.ME_PATH, "me", log_path=log_path)


def me_id(path=None) -> str | None:
    me = load_me(path, log_path=False)
    return None if not me or me.get("id") is None else str(me["id"])


def load_schedule(section_id, *, schedule_dir=None, log_path=None):
    """The full section schedule (`paginated-classes`, incl. unrecorded classes)."""
    base = Path(schedule_dir) if schedule_dir is not None else config.SCHEDULE_DIR
    return load_json(base / f"{section_id}.json", "schedule", class_id=None, log_path=log_path)


def find_transcript(class_id, kind: str = "asr", transcripts_dir=None) -> Path | None:
    """Scribe output `{course}_{date}_class{cid}_{kind}.json` (kind: asr | ids)."""
    base = Path(transcripts_dir) if transcripts_dir is not None else config.TRANSCRIPTS_DIR
    hits = sorted(base.glob(f"*class{class_id}_{kind}.json"))
    return hits[-1] if hits else None


def load_asr(class_id, *, transcripts_dir=None, uses=None, log_path=None):
    """`*_asr.json`: only `words[]` (and provider metadata) survive."""
    path = find_transcript(class_id, "asr", transcripts_dir)
    if path is None:
        return None
    return load_json(path, "asr", class_id=class_id, uses=uses, log_path=log_path)


def load_transcript(class_id, *, transcripts_dir=None, log_path=None):
    """Scribe `*_ids.json`, reduced to calibration/QA metadata. `events`, `participation`,
    utterances and speakers never survive, and reading them raises `BlockedKeyError`."""
    path = find_transcript(class_id, "ids", transcripts_dir)
    if path is None:
        return None
    return load_json(path, "transcript", class_id=class_id, log_path=log_path)


def snake(obj):
    """kebab-case keys -> snake_case, recursively (for APIs written with snake keys)."""
    if isinstance(obj, dict):
        return {(k.replace("-", "_") if isinstance(k, str) else k): snake(v)
                for k, v in obj.items()}
    if isinstance(obj, list):
        return [snake(v) for v in obj]
    return obj


def load_calibration(class_id, *, transcripts_dir=None, log_path=None) -> dict | None:
    """The transcript's `calibrate()` result with snake_case keys, ready for
    `config.calibration_qa` (None when there is no transcript or no calibration)."""
    t = load_transcript(class_id, transcripts_dir=transcripts_dir, log_path=log_path)
    cal = ((t or {}).get("alignment") or {}).get("calibration")
    return snake(cal) if cal else None


def load_hand_log(path, *, log_path=None):
    """Live-logger export (QA only: the `hand_log` table, never features)."""
    data = _read_json(Path(path))
    if isinstance(data, dict):
        data = data.get("entries") or []
    doc, dropped = filter_doc(data, "hand_log")
    log_dropped(dropped, source="hand_log", log_path=log_path)
    return doc


# Manifest columns other stages may read. n_voice / n_my_voice / n_events_by_type are
# whole-class aggregates (§4.2) and are not loaded.
MANIFEST_KEEP = ("class_id", "section_id", "course_code", "term", "title", "date",
                 "recording_started", "recording_ended")


def load_manifest(path=None) -> pd.DataFrame:
    """`scribe sync` manifest.csv, restricted to MANIFEST_KEEP (one row per class)."""
    path = Path(path) if path is not None else config.MANIFEST_PATH
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    keep = [c for c in MANIFEST_KEEP if c in df.columns]
    df = df[keep].replace({"": None})
    df["class_id"] = pd.to_numeric(df["class_id"], errors="coerce").astype("Int64")
    if "section_id" in df:
        df["section_id"] = pd.to_numeric(df["section_id"], errors="coerce").astype("Int64")
    return df.dropna(subset=["class_id"]).reset_index(drop=True)


# --------------------------------------------------------------------------- roster helpers


def class_user_id(entry: dict) -> str | None:
    """The Forum user id of one `class-users[]` entry (user.id, then user-id, then id)."""
    if not isinstance(entry, dict):
        return None
    user = entry.get("user") if isinstance(entry.get("user"), dict) else {}
    uid = user.get("id", entry.get("user-id", entry.get("id")))
    return None if uid is None else str(uid)


def human_actor_ids(klass: dict | None) -> set[str]:
    """User ids of the humans in `class-users` (the only allowed chat actors, A8)."""
    users = (klass or {}).get("class-users") or []
    return {u for u in (class_user_id(e) for e in users) if u is not None}


def _actor(ev: dict) -> str | None:
    uid = ev.get("actor-id")
    if uid is None and isinstance(ev.get("actor"), dict):
        uid = ev["actor"].get("id")
    return None if uid is None else str(uid)


def human_messages(events, klass, *, include_private: bool = False) -> list[dict]:
    """`message` events whose actor is a human in `class-users` (L27: system notices such as
    "... raised their hand" have no or a non-roster actor and are dropped)."""
    humans = human_actor_ids(klass)
    out = []
    for ev in events or []:
        if not isinstance(ev, dict) or _event_type(ev) != "message":
            continue
        if _actor(ev) not in humans:
            continue
        if ev.get("private") and not include_private:
            continue
        out.append(ev)
    return out


def message_actor_report(events, klass) -> dict:
    """Counts for the D1 funnel: messages kept vs dropped by reason (no text, no ids)."""
    humans = human_actor_ids(klass)
    rep = Counter()
    for ev in events or []:
        if not isinstance(ev, dict) or _event_type(ev) != "message":
            continue
        rep["n_messages"] += 1
        actor = _actor(ev)
        if actor is None:
            rep["n_null_actor"] += 1
        elif actor not in humans:
            rep["n_non_roster"] += 1
        elif ev.get("private"):
            rep["n_private"] += 1
        else:
            rep["n_human"] += 1
    return dict(rep)


# --------------------------------------------------------------------------- provenance

_E = "events"
_CAL = (("transcript", "alignment.calibration.offset"),
        ("transcript", "alignment.calibration.drift"),
        ("transcript", "alignment.calibration.pieces[].t-from"),
        ("transcript", "alignment.calibration.pieces[].offset"),
        ("transcript", "alignment.calibration.pieces[].drift"))
_REC = (("class", "recording-sessions[].recording-started"),)
_VOICE_T = (((_E, "[].timestamp"), (_E, "[].event-data.duration"), (_E, "[].event-type"))
            + _REC + _CAL)
_ROLE = ((_E, "[].actor-id"), (_E, "[].actor.id"), ("class", "teacher-user-id"),
         ("class", "section.teacher-user-id"), ("class", "class-users[].role"),
         ("class", "class-users[].user.id"), ("class", "class-users[].user-id"),
         ("class", "class-users[].id"), ("me", "id"))
_ROOM = ((_E, "[].room-id"), (_E, "[].room-signature"))
_POINT_T = ((_E, "[].timestamp"), (_E, "[].event-type")) + _REC + _CAL
_TERM = (("class", "section.course.term.name"), ("class", "section.course.term"),
         ("listing", "term"))
_SCHED = (("class", "start-time"), ("listing", "class.start-time"))
_ASR_TEXT = (("asr", "words[].punctuated-word"), ("asr", "words[].text"))

# "table.column" -> whitelisted (source, path) pairs it is computed from. Table builders own
# their columns; this map is what the provenance test checks (§4.4 test_loader_allowlist).
# Pure join keys whose whitelist use is "meta" (e.g. polls `[].choices[].id`, which only links
# a response's choice-id to the choice text) are omitted: they never become values.
PROVENANCE: dict[str, tuple[tuple[str, str], ...]] = {
    # classes (select.py)
    "classes.course_code": (("class", "section.course.course-code"), ("listing", "course-code")),
    "classes.course_area": (("class", "section.course.course-code"), ("listing", "course-code")),
    "classes.term": _TERM,
    "classes.semester_idx": _TERM,
    "classes.session_no": (("schedule", "results[].id"), ("schedule", "results[].date"),
                           ("schedule", "results[].start-time"), ("class", "date")),
    "classes.date": (("class", "date"), ("listing", "class.date")),
    "classes.weekday": _SCHED + (("class", "date"),) + _TERM,
    "classes.hour_local": _SCHED + _TERM,
    "classes.hour_utc": _SCHED,
    "classes.sched_start_utc": _SCHED,
    "classes.rec_offset_s": _SCHED + _REC,
    "classes.enrolled": (("class", "class-users[].role"), ("class", "class-users[].user.id"),
                         ("class", "class-users[].user-id"), ("class", "class-users[].id")),
    "classes.prof_hash": (("class", "teacher-user-id"), ("class", "section.teacher-user-id")),
    "classes.n_rec_sessions": _REC + (("class", "recording-sessions[].id"),),
    "classes.rec_segments_json": _REC + (("class", "recording-sessions[].recording-ended"),),
    "classes.T_end_s": _REC + (("class", "recording-sessions[].recording-ended"),),
    "classes.n_my_voice_main": ((_E, "[].event-type"), (_E, "[].actor-id"), ("me", "id"))
                               + _ROOM,
    "classes.timing_precision": ((_E, "[].timestamp"), (_E, "[].event-data.duration")),
    "classes.in_vault": (("class", "id"),),
    # voice / chat / segments / polls / presence (events.py)
    "voice.start_s": _VOICE_T,
    "voice.end_s": _VOICE_T,
    "voice.spk_hash": ((_E, "[].actor-id"), (_E, "[].actor.id")),
    "voice.role": _ROLE,
    "voice.room_main": _ROOM,
    "voice.precise": ((_E, "[].timestamp"),),
    "voice.word_bearing": _VOICE_T + (("asr", "words[].start"), ("asr", "words[].end")),
    "voice.first_word_avail_s": _VOICE_T + (("asr", "words[].start"), ("asr", "words[].end")),
    "chat.t_s": _POINT_T,
    "chat.spk_hash": ((_E, "[].actor-id"),),
    "chat.role": _ROLE,
    "chat.text_masked": ((_E, "[].event-data.message-text"), (_E, "[].event-data.message"),
                         ("class", "class-users[].first-name"),
                         ("class", "class-users[].last-name"),
                         ("class", "class-users[].user.first-name"),
                         ("class", "class-users[].user.last-name")),
    "segments.start_s": _POINT_T,
    "segments.seg_idx": _POINT_T,
    "segments.title_masked": ((_E, "[].event-data.timeline-segment-title"),),
    "segments.section_masked": ((_E, "[].event-data.timeline-section-title"),),
    "segments.seg_type": ((_E, "[].event-data.timeline-segment-title"),
                          (_E, "[].event-data.timeline-section-title")),
    "polls.poll_id": (("polls", "[].id"),),
    "polls.open_s": (("polls", "[].timing-text"),) + _REC + _CAL,
    "polls.close_s": (("polls", "[].timing-text"),) + _POINT_T,
    "polls.prompt_masked": (("polls", "[].prompt"), ("polls", "[].title"),
                            ("polls", "[].choices[].text")),
    "polls.my_answer_text": (("poll_responses", "[].answer"), ("poll_responses", "[].choice-id"),
                             ("poll_responses", "[].poll-id"), ("poll_responses", "[].user-id"),
                             ("polls", "[].choices[].text"), ("me", "id")),
    "polls.my_answer_avail_s": (("poll_responses", "[].submitted-at"),) + _POINT_T,
    "presence.enter_s": (("class", "class-users[].enter-class-time"),) + _REC + _CAL,
    "presence.role": _ROLE,
    # breakout membership (A7)
    "breakouts.my_group": (("breakouts", "[].room-id"), ("breakouts", "[].users[].id"),
                           ("breakouts", "[].start-time"), ("me", "id")),
    # words (transcribe.words_table)
    "words.start_s": (("asr", "words[].start"),),
    "words.end_s": (("asr", "words[].end"),),
    "words.text_norm": _ASR_TEXT,
    "words.text_masked": _ASR_TEXT,
    "words.has_qmark": (("asr", "words[].punctuated-word"),),
    "words.conf": (("asr", "words[].conf"),),
    "words.filler": (("asr", "words[].filler"),),
    "words.spk_hash": (("asr", "words[].start"), ("asr", "words[].end")) + _VOICE_T + _ROLE,
    "words.role": (("asr", "words[].start"), ("asr", "words[].end")) + _VOICE_T + _ROLE,
    "words.role_event_start_s": _VOICE_T,
    "words.role_avail_s": (("asr", "words[].end"),) + _VOICE_T,
    "words.role_fused": (("asr", "words[].cluster"), ("asr", "words[].cluster-conf")) + _ROLE,
    # PCW and class pages (pcw.py / embed.py)
    "pcw_blocks.block_uuid": (("pcw_template", "blocks[].id"), ("pcw_template", "blocks[].uuid")),
    "pcw_blocks.block_order": (("pcw_template", "blocks[].order"),),
    "pcw_blocks.prompt_text": (("pcw_template", "blocks[].block-components[].value"),
                               ("pcw_template", "blocks[].block-components[].type")),
    "pcw_blocks.answer_text": (("pcw_doc", "answers.*.value"), ("pcw_doc", "answers.*.type"),
                               ("pcw_doc", "answers.*.files[].name"),
                               ("pcw_doc", "answers.*.files[].filename")),
    "pcw_blocks.answer_words": (("pcw_doc", "answers.*.value"),),
    "pcw_blocks.has_code": (("pcw_doc", "answers.*.type"),),
    "pcw_blocks.n_files": (("pcw_doc", "answers.*.files[].name"),
                           ("pcw_doc", "answers.*.files[].filename")),
    "class_edit_page.lo_text": (
        ("class_edit_page", "assessment.focused-outcomes[].learning-outcome.name"),
        ("class_edit_page", "assessment.focused-outcomes[].learning-outcome.description")),
    "class_edit_page.sg_text": (("class_edit_page", "study-guide"),
                                ("class_edit_page", "study-guide.description"),
                                ("class_edit_page", "preparatory-assignment.description")),
}

FEATURE_SAFE_USES = frozenset({config.USE_FEATURE, config.USE_TIMING, config.USE_MASK})


def provenance_uses(key: str) -> set[str | None]:
    """The whitelist uses of every (source, path) behind a PROVENANCE key."""
    return {config.whitelist_use(src, p) for src, p in PROVENANCE[key]}


def provenance_problems(mapping: dict | None = None) -> list[str]:
    """Every (source, path) in `mapping` that is not whitelisted (empty list = all good)."""
    out = []
    for key, refs in (PROVENANCE if mapping is None else mapping).items():
        for src, p in refs:
            if config.whitelist_use(src, p) is None:
                out.append(f"{key}: {src}:{p} is not whitelisted")
    return out


def feature_safe(key: str) -> bool:
    """True if a table column may feed features: all inputs whitelisted for feature, timing
    or in-memory masking use (no label, meta-only or unlisted paths)."""
    uses = provenance_uses(key)
    return None not in uses and uses <= FEATURE_SAFE_USES


def resolve_provenance(refs) -> set[tuple[str, str]]:
    """Expand references (PROVENANCE keys "table.column" or (source, path) pairs) into the
    set of raw (source, path) pairs. Unknown keys raise KeyError."""
    out: set[tuple[str, str]] = set()
    for ref in refs:
        if isinstance(ref, str):
            out |= set(PROVENANCE[ref])
        else:
            src, p = ref
            out.add((src, p))
    return out


def feature_provenance_problems(feature_prov: dict) -> list[str]:
    """Check a `features.PROVENANCE`-style map {column or prefix: refs}: every ref must
    resolve to whitelisted (source, path) pairs whose use is feature, timing or mask."""
    out = []
    for col, refs in feature_prov.items():
        try:
            pairs = resolve_provenance(refs)
        except KeyError as exc:
            out.append(f"{col}: unknown provenance key {exc}")
            continue
        for src, p in sorted(pairs):
            use = config.whitelist_use(src, p)
            if use is None:
                out.append(f"{col}: {src}:{p} is not whitelisted")
            elif use not in FEATURE_SAFE_USES:
                out.append(f"{col}: {src}:{p} has use {use!r}")
    return out
