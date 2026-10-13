"""Forum event tables from raw `class-events.json` only (spec §2.1, §4 (c), §4.1 A3/A6/A8/A9/A11).

Every table here is built from the raw Forum bundle that `scribe sync` wrote, never from
Scribe's transcript JSON (whose `events` can carry `hand-raise-key` and `participation`).
Raw documents are kebab-cased (`scribe_core.forum_events.kebab`), restricted to the event-type
allowlist {voice, message, timeline-segment} and to the whitelisted JSON paths of
`config.RAW_PATH_WHITELIST` before anything reads them.

Clock. Forum wall-clock ms map to Forum video seconds with `wall_to_video` (NOT the clamped
variant), then onto the master clock with the class's Scribe calibration (`make_shift`).
Events whose wall time falls inside a recording gap are dropped (Scribe would move them to the
restart and create fake bursts). Events before the first session extrapolate to negative
master times, so a segment opened just before recording is still the current one at t = 0.

Tables (schemas in `oncue.schemas`; times in master-clock seconds; ids hashed):
* voice     calibrated voice events of every room; `word_bearing` / `first_word_avail_s` from
            the words table (only main-room events can carry words);
* chat      human class-users only; system, null-actor, non-roster, private and gap messages
            are dropped and counted;
* segments  timeline segments with `seg_type` and NO end column;
* polls     open time parsed from the "started at H:MM:SS" string against a reference
            (PROVISIONAL until D9); Brady's own answer only, available at
            submit (else close) + eps_F,poll;
* presence  enter-class-time per user (a state: a join inside a gap counts from the restart);
* hand_log  live-logger key presses, QA only (never a feature source).

Names are masked in memory before anything is written (§5.1); `RosterMasker` is the
spec-shaped fallback masker until `oncue.text` provides the canonical one (pass `masker=`).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scribe_core.forum_events import (
    RecordingSegment,
    event_start_ms,
    event_user_id,
    kebab,
    main_room_info,
    room_key,
    timeline_from,
    voice_wall_ms,
)
from scribe_core.fuse import EDGE, PROFESSOR
from scribe_core.privacy import pseudonym
from scribe_core.text import levenshtein
from scribe_core.timeutil import epoch_ms, iso_to_ms

from . import config, schemas

log = logging.getLogger(__name__)

# Poll "started at H:MM:SS" strings: what the clock reading is relative to. PROVISIONAL (D9).
POLL_REFERENCES = ("recording_started", "video", "scheduled_start")
POLL_REFERENCE = "recording_started"
_POLL_OPEN_RE = re.compile(r"\bstart(?:ed|s)?\s+at\s+(\d{1,2}(?::\d{1,2}){1,2})",
                           re.IGNORECASE)
_POLL_CLOSE_RE = re.compile(r"\b(?:ended|closed|stopped|ends|closes)\s+at\s+"
                            r"(\d{1,2}(?::\d{1,2}){1,2})", re.IGNORECASE)

# Forum system notices posted as chat (L27). Dropped and counted even with a human actor.
SYSTEM_NOTICE_RE = re.compile(
    r"\b(?:raised|lowered)\s+(?:their|his|her|a)\s+hand\b"
    r"|\bhas\s+(?:joined|left)\s+the\s+(?:class|room|call|meeting|breakout)\b",
    re.IGNORECASE)

# hand_log: a key press matches the bid/vote onset in [press - 2 s, press + 5 s] (QA only).
HAND_LOG_MATCH_S = (-2.0, 5.0)

_EVENT_TABLES = ("voice", "chat", "segments", "polls", "presence")


# =========================================================================== raw loading

_DROP = object()


@lru_cache(maxsize=65536)
def _use(source: str, path: str) -> str | None:
    return config.whitelist_use(source, path)


def _filter(source: str, obj, path: str):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            kept = _filter(source, v, f"{path}.{k}" if path else str(k))
            if kept is not _DROP:
                out[k] = kept
        return out if out else _DROP
    if isinstance(obj, list):
        p = path + "[]"
        out = [x for x in (_filter(source, v, p) for v in obj) if x is not _DROP]
        return out if out else _DROP
    return obj if _use(source, path) is not None else _DROP


def whitelist_filter(source: str, obj):
    """Kebab-case `obj` and keep only leaves on whitelisted paths of `source` (§4.4).

    Blocked sources and blocked keys return nothing. Containers emptied by the filter are
    dropped; the root comes back as `{}` / `[]` rather than disappearing.
    """
    if source in config.BLOCKED_SOURCES:
        raise ValueError(f"source {source!r} is blocked (post-class grading or analytics)")
    k = kebab(obj)
    kept = _filter(source, k, "")
    if kept is _DROP:
        return [] if isinstance(k, list) else {}
    return kept


def allowed_events(events) -> list[dict]:
    """Kebab-cased raw events restricted to the type allowlist and the events whitelist."""
    evs = [e for e in kebab(events or []) if isinstance(e, dict)
           and e.get("event-type") in config.EVENT_TYPES]
    return whitelist_filter("events", evs)


def read_raw(class_id, source: str, raw_dir: Path | str | None = None):
    """One whitelisted raw source for a class (None when the file is absent)."""
    if source not in config.RAW_SOURCE_FILES:
        raise KeyError(f"unknown raw source {source!r}")
    if source in config.BLOCKED_SOURCES:
        raise ValueError(f"source {source!r} is blocked")
    path = Path(raw_dir or config.RAW_DIR) / str(class_id) / config.RAW_SOURCE_FILES[source]
    if not path.exists():
        return None
    obj = json.loads(path.read_text(encoding="utf-8"))
    if obj is None:
        return None
    if source == "events":
        return allowed_events(obj)
    out = whitelist_filter(source, obj)
    if source == "class" and "teacher-user-id" not in out:
        # class_grader omits the professor; our sync's section listing has them (A12)
        tid = ((read_raw(class_id, "listing", raw_dir) or {}).get("class") or {}).get(
            "teacher-user-id")
        if tid is not None:
            out["teacher-user-id"] = tid
    return out


def require_me_id(me_id) -> str:
    """Brady's id is mandatory: without it every one of his words would read `student`."""
    if me_id is None or str(me_id).strip() == "":
        raise ValueError("Brady's Forum user id is required (data/scribe/me.json from "
                         "`scribe sync`, or pass me_id)")
    return str(me_id)


def load_me_id(scribe_dir: Path | str | None = None) -> str | None:
    """Brady's Forum user id from `scribe sync`'s me.json (users/self.id)."""
    path = Path(scribe_dir) / "me.json" if scribe_dir else config.ME_PATH
    if not path.exists():
        return None
    data = whitelist_filter("me", json.loads(path.read_text(encoding="utf-8")))
    uid = data.get("id") if isinstance(data, dict) else None
    return None if uid is None else str(uid)


# =========================================================================== ids, roster


def spk_hash(user_id, salt: str | None = None) -> str | None:
    """Scribe's HMAC pseudonym ("u_" + 10 hex) keyed with the On Cue ID salt."""
    if user_id is None or (isinstance(user_id, float) and math.isnan(user_id)):
        return None
    return pseudonym(str(user_id), salt if salt is not None else config.id_salt())


def _split_names(names: Iterable[str]) -> set[str]:
    out = set()
    for n in names:
        for part in re.split(r"[\s\-]+", str(n or "").strip()):
            part = part.strip(".,'\"")
            if len(part) >= 2:
                out.add(part.lower())
        whole = str(n or "").strip().lower()
        if len(whole) >= 2 and " " not in whole:
            out.add(whole)
    return out


@dataclass(frozen=True)
class Roster:
    """class-users mapped to causal speaker roles. Names are kept in memory for masking only."""

    me_id: str | None
    roles: dict = field(default_factory=dict)        # uid -> prof | me | student
    names: tuple = ()                                # first/last/preferred names (masking)
    me_names: tuple = ()
    enter_iso: dict = field(default_factory=dict)    # uid -> enter-class-time
    absent: frozenset = frozenset()
    teacher: str = PROFESSOR                         # teacher-user-id, or Scribe's stand-in

    def role(self, user_id) -> str:
        if user_id is None:
            return config.ROLE_UNK
        uid = str(user_id)
        if self.me_id is not None and uid == self.me_id:
            return config.ROLE_ME
        if uid == self.teacher:
            return config.ROLE_PROF
        return self.roles.get(uid, config.ROLE_UNK)

    def is_member(self, user_id) -> bool:
        return user_id is not None and (str(user_id) in self.roles
                                         or str(user_id) == self.me_id)


def roster_from_class(klass: dict | None, me_id=None) -> Roster:
    """prof = teacher-user-id or a staff role; me = users/self.id; other class-users = student."""
    k = kebab(klass or {})
    me = None if me_id is None else str(me_id)
    section = k.get("section") if isinstance(k.get("section"), dict) else {}
    teachers = {str(x) for x in (k.get("teacher-user-id"), section.get("teacher-user-id"))
                if x is not None}
    roles, names, me_names, enter, absent = {}, [], [], {}, set()
    for cu in k.get("class-users") or []:
        if not isinstance(cu, dict):
            continue
        user = cu.get("user") if isinstance(cu.get("user"), dict) else cu
        uid = user.get("id", user.get("user-id", cu.get("user-id")))
        if uid is None and user is cu:
            uid = cu.get("id")
        if uid is None:
            continue
        uid = str(uid)
        forum_role = str(cu.get("role") or user.get("role") or "").lower()
        if me is not None and uid == me:
            roles[uid] = config.ROLE_ME
        elif uid in teachers or forum_role in config.PROF_ROLES:
            roles[uid] = config.ROLE_PROF
        else:
            roles[uid] = config.ROLE_STUDENT
        person = [user.get(f) or cu.get(f) for f in ("first-name", "last-name", "preferred-name")]
        (me_names if roles[uid] == config.ROLE_ME else names).extend(p for p in person if p)
        enter[uid] = cu.get("enter-class-time") or user.get("enter-class-time")
        if cu.get("absent"):
            absent.add(uid)
    for t in teachers:
        roles.setdefault(t, config.ROLE_PROF)
    if me is not None:
        roles.setdefault(me, config.ROLE_ME)
    return Roster(me, roles, tuple(names), tuple(me_names), enter, frozenset(absent),
                  min(teachers) if teachers else PROFESSOR)


# =========================================================================== masking (§5.1)


@lru_cache(maxsize=1)
def common_words() -> frozenset[str]:
    """wordfreq top-20k English plus Scribe's fixup.COMMON_WORDS (lower case)."""
    from scribe_core.fixup import COMMON_WORDS
    from wordfreq import top_n_list

    return frozenset(w.lower() for w in top_n_list("en", config.COMMON_WORDS_TOPN)) \
        | frozenset(COMMON_WORDS)


_EDGE_RE = re.compile(r"^(\W*)(.*?)(\W*)$", re.DOTALL)
_SENT_END_RE = re.compile(r"[.?!][\"'”’)\]]*$")


class RosterMasker:
    """Spec §5.1 masking on cased, punctuated tokens (fallback until `oncue.text` lands).

    1. strip surrounding punctuation and a possessive 's / s';
    2. exact roster match (case-insensitive) -> [NAME]; a name that is a common word is
       masked only when capitalised and not sentence-initial, or (extension) when it is a
       capitalised vocative directly followed by a comma ("Grace, go ahead");
    3. Levenshtein <= 1 for capitalised mid-sentence tokens of >= 4 characters that are not
       common words (names under 4 characters match exactly only);
    4. Brady's explicit variants (and his roster names) -> `me_token` ([ME]) always, because
       "brady"/"rutherford" are wordfreq-common yet carry the call-on leak.
    Surrounding punctuation is kept: "Ada," -> "[NAME],".
    """

    def __init__(self, names: Iterable[str] = (), me_names: Iterable[str] = (), *,
                 common: frozenset[str] | None = None, name_token: str = config.NAME_TOKEN,
                 me_token: str = config.ME_TOKEN, vocative: bool = True):
        self.me = _split_names(list(me_names) + list(config.BRADY_VARIANTS))
        self.names = _split_names(names) - self.me
        self.common = common_words() if common is None else frozenset(common)
        self.name_token, self.me_token, self.vocative = name_token, me_token, vocative
        self._fuzzy = [(n, "me") for n in sorted(self.me)] + [(n, "name")
                                                            for n in sorted(self.names)]
        self._fuzzy = [(n, k) for n, k in self._fuzzy
                       if len(n) >= config.MASK_LEV_MIN_LEN and n not in self.common]

    def _token(self, kind: str) -> str:
        return self.me_token if kind == "me" else self.name_token

    def mask_token(self, tok: str, initial: bool = False) -> tuple[str, str | None]:
        """(masked token, kind) with kind None | "name" | "me"."""
        pre, core, post = _EDGE_RE.match(tok or "").groups()
        if not core:
            return tok, None
        cands = [core]
        low = core.lower()
        if len(core) > 2 and low.endswith(("'s", "’s")):
            cands.append(core[:-2])
        for cand in cands:
            key = cand.lower()
            kind = "me" if key in self.me else ("name" if key in self.names else None)
            if kind is None:
                continue
            cap = cand[:1].isupper()
            if kind == "name" and key in self.common:
                vocative = self.vocative and cap and post.startswith(",") and cand == core
                if not ((cap and not initial) or vocative):
                    return tok, None
            return pre + self._token(kind) + post, kind
        cand = cands[-1]
        key = cand.lower()
        if (len(key) >= config.MASK_LEV_MIN_LEN and cand[:1].isupper() and not initial
                and key.isalpha() and key not in self.common):
            for name, kind in self._fuzzy:
                if abs(len(name) - len(key)) <= config.MASK_LEV_MAX \
                        and levenshtein(key, name) <= config.MASK_LEV_MAX:
                    return pre + self._token(kind) + post, kind
        return tok, None

    def mask_tokens(self, tokens: Sequence[str], initial: Sequence[bool] | None = None
                    ) -> tuple[list[str], list[str | None]]:
        """Mask a token stream. Without `initial`, a token is sentence-initial when it is the
        first one or the previous token ends a sentence."""
        out, kinds = [], []
        prev = None
        for i, tok in enumerate(tokens):
            init = bool(initial[i]) if initial is not None else (
                prev is None or bool(_SENT_END_RE.search(prev)))
            m, kind = self.mask_token(str(tok), init)
            out.append(m)
            kinds.append(kind)
            prev = str(tok)
        return out, kinds

    def mask_text(self, text) -> str | None:
        if text is None or (isinstance(text, float) and math.isnan(text)):
            return None
        return " ".join(self.mask_tokens(str(text).split())[0])


def masker_for(roster: Roster, masker=None):
    """The masker to use for one class: `masker` if given, else a RosterMasker on the roster."""
    if masker is not None:
        return masker
    return RosterMasker(roster.names, roster.me_names)


# =========================================================================== clock


def _shift_array(v: np.ndarray, cal: dict | None) -> np.ndarray:
    """Vectorised `scribe_core.calibrate.make_shift` (identical piece selection)."""
    v = np.asarray(v, dtype=float)
    cal = cal or {}
    pieces = cal.get("pieces") or [{"t_from": None, "offset": 0.0, "drift": 0.0}]
    if not cal.get("applied", True):
        pieces = [{"t_from": None, "offset": 0.0, "drift": 0.0}]
    bounds = np.array([float(p["t_from"]) for p in pieces[1:]], dtype=float)
    idx = np.searchsorted(bounds, v, side="right") if len(bounds) else np.zeros(v.shape, int)
    idx = np.minimum(idx, len(pieces) - 1)
    off = np.array([float(p.get("offset") or 0.0) for p in pieces])[idx]
    drift = np.array([float(p.get("drift") or 0.0) for p in pieces])[idx]
    return v + off + drift * v


@dataclass
class VideoClock:
    """Forum wall-clock ms -> master-clock seconds (unclamped; NaN inside recording gaps)."""

    segs: list[RecordingSegment]
    cal: dict | None = None
    origin_s: float = 0.0

    def __post_init__(self):
        self.segs = list(self.segs or [])
        self._a = np.array([s.wall_start_ms for s in self.segs], dtype=float)
        self._b = np.array([np.inf if s.wall_end_ms is None else s.wall_end_ms
                            for s in self.segs], dtype=float)
        self._v0 = np.array([s.video_start for s in self.segs], dtype=float)

    # -- Forum video time --------------------------------------------------------------
    def videos(self, wall_ms) -> np.ndarray:
        """Forum video seconds; before the first session / after the last they extrapolate,
        inside a gap between sessions they are NaN."""
        w = np.asarray(wall_ms, dtype=float)
        out = np.full(w.shape, np.nan)
        if not len(self._a):
            return out
        for a, b, v0 in zip(self._a, self._b, self._v0, strict=True):
            hit = np.isnan(out) & (w >= a) & (w <= b)
            out[hit] = v0 + (w[hit] - a) / 1000.0
        pre = w < self._a[0]
        out[pre] = self._v0[0] + (w[pre] - self._a[0]) / 1000.0
        post = w > self._b[-1]
        out[post] = self._v0[-1] + (w[post] - self._a[-1]) / 1000.0
        return out

    def video(self, wall_ms) -> float:
        if wall_ms is None:
            return float("nan")
        return float(self.videos([wall_ms])[0])

    # -- master clock ------------------------------------------------------------------
    def from_video(self, video_s) -> np.ndarray:
        return _shift_array(np.asarray(video_s, dtype=float), self.cal) + self.origin_s

    def masters(self, wall_ms) -> np.ndarray:
        return self.from_video(self.videos(wall_ms))

    def master(self, wall_ms) -> float:
        if wall_ms is None:
            return float("nan")
        return float(self.masters([wall_ms])[0])

    def status(self, wall_ms) -> str:
        """"pre" | "in" | "gap" | "post" | "none" for one wall time."""
        if wall_ms is None or not len(self._a):
            return "none"
        w = float(wall_ms)
        if w < self._a[0]:
            return "pre"
        if ((w >= self._a) & (w <= self._b)).any():
            return "in"
        return "post" if w > self._b[-1] else "gap"

    def session_end_master(self, wall_ms) -> float:
        """Master time of the end of the session holding (or following) `wall_ms`."""
        if not len(self._a):
            return float("nan")
        k = int(np.clip(np.searchsorted(self._a, float(wall_ms), side="right") - 1,
                        0, len(self._a) - 1))
        if np.isinf(self._b[k]):
            return float("inf")
        return float(self.from_video([self._v0[k] + (self._b[k] - self._a[k]) / 1000.0])[0])

    def next_start_wall(self, wall_ms) -> float:
        """Wall ms of the first session start after `wall_ms` (NaN if none)."""
        later = self._a[self._a > float(wall_ms)]
        return float(later[0]) if len(later) else float("nan")

    def restarts(self) -> list[float]:
        """Master times of every session start after the first (rec_gap rule, §3)."""
        return [float(self.from_video([v])[0]) for v in self._v0[1:]]

    def gaps_wall(self) -> list[tuple[float, float]]:
        return [(float(b), float(a)) for b, a in zip(self._b[:-1], self._a[1:], strict=True)]


def video_clock(probe: dict | None = None, cal: dict | None = None, segs=None) -> VideoClock:
    """Spec `video_clock(probe, cal, segs)`.

    `segs`: a list of RecordingSegment, a class dict or a recording-started string;
    `cal`: Scribe's `calibrate()` result (None or not applied = identity);
    `probe`: optional ffprobe facts; only `asr_origin_s` (default 0) is used, because ASR word
    times already count from the first decoded audio sample, which is the master-clock zero.
    """
    origin = float((probe or {}).get("asr_origin_s", 0.0) or 0.0)
    return VideoClock(timeline_from(segs) if segs is not None else [], cal, origin)


# =========================================================================== voice


VOICE_COLUMNS = ["user_id", "start_s", "end_s", "room_main", "room_id", "precise", "event_id"]


def parse_voice(events: list[dict], clock: VideoClock, main_room: str | None = None
                ) -> pd.DataFrame:
    """Calibrated voice events of every room (raw user ids: in memory only, never written).

    start = timestamp - duration (voice `timestamp` is the END). An event that starts inside a
    recording gap is dropped; one that ends inside a gap is cut at its session's end.
    """
    evs = [e for e in kebab(events or []) if isinstance(e, dict)
           and e.get("event-type") == "voice"]
    if main_room is None:
        main_room = main_room_info(evs)[0]
    rows, seen = [], set()
    for ev in evs:
        eid = ev.get("id")
        if eid is not None:
            if eid in seen:
                continue
            seen.add(eid)
        timing = voice_wall_ms(ev)
        if timing is None:
            continue
        s_ms, e_ms, precise = timing
        if e_ms <= s_ms:
            continue
        rows.append((event_user_id(ev), s_ms, e_ms, room_key(ev), bool(precise), eid))
    if not rows:
        return pd.DataFrame(columns=VOICE_COLUMNS)
    uid, s_ms, e_ms, room, precise, eid = map(list, zip(*rows, strict=True))
    start = clock.masters(s_ms)
    end = clock.masters(e_ms)
    for i in np.flatnonzero(np.isnan(end) & ~np.isnan(start)):
        end[i] = clock.session_end_master(s_ms[i])
    df = pd.DataFrame({"user_id": uid, "start_s": start, "end_s": end,
                       "room_main": [r is not None and r == main_room for r in room],
                       "room_id": room, "precise": precise, "event_id": eid})
    df = df[np.isfinite(df["start_s"])]
    return df.sort_values(["start_s", "end_s"], kind="stable").reset_index(drop=True)


def word_bearing_flags(voice: pd.DataFrame, words: pd.DataFrame | None, eps_f_s: float,
                       *, edge: float = EDGE) -> tuple[np.ndarray, np.ndarray]:
    """(word_bearing, first_word_avail_s) per voice row (§2.6, §4.1 A3).

    A main-room event is word-bearing when it overlaps (edges padded by Scribe's EDGE) a word
    attributed to the same speaker. It is known to be word-bearing once both the event and that
    word's role are available: max(min role_avail_s, start + eps_F). Breakout-room events are
    never word-bearing (the recording only carries main-room audio).
    """
    n = len(voice)
    wb = np.zeros(n, dtype=bool)
    fwa = np.full(n, np.nan)
    if n == 0 or words is None or len(words) == 0:
        return wb, fwa
    w = words.dropna(subset=["spk_hash"])
    groups = {k: g.sort_values("start_s") for k, g in w.groupby("spk_hash", sort=False)}
    vs = voice["start_s"].to_numpy(float)
    ve = voice["end_s"].to_numpy(float)
    main = voice["room_main"].to_numpy(bool)
    spk = voice["spk_hash"].to_numpy(object)
    cache = {}
    for i in range(n):
        g = groups.get(spk[i])
        if g is None or not main[i]:
            continue
        if spk[i] not in cache:
            ws = g["start_s"].to_numpy(float)
            we = g["end_s"].to_numpy(float)
            cache[spk[i]] = (ws, we, np.maximum.accumulate(we), g["role_avail_s"].to_numpy(float))
        ws, we, cmax, avail = cache[spk[i]]
        hi = np.searchsorted(ws - edge, ve[i], side="left")
        lo = np.searchsorted(cmax + edge, vs[i], side="right")
        if hi <= lo:
            continue
        hit = (ws[lo:hi] - edge < ve[i]) & (we[lo:hi] + edge > vs[i])
        if hit.any():
            wb[i] = True
            fwa[i] = max(float(np.nanmin(avail[lo:hi][hit])), vs[i] + eps_f_s)
    return wb, fwa


def professor_turns(words: pd.DataFrame | None, *, gap_s: float = config.PROF_TURN_GAP_S
                    ) -> pd.DataFrame:
    """The professor's speaking turns, inferred from words (A12): Forum logs no professor mic,
    so these rows stand in for the voice events the professor never gets.

    A turn is a run of professor words (role prof with no voice event behind it) with gaps
    under `gap_s`. Each row says when it became knowable, so features stay causal:
      start_avail_s  the first word's role_avail_s
      end_avail_s    the turn is over once `gap_s` has passed with no new word (+ eps_w for
                     that to be visible), or once the next speaker's word has a known role
    Until end_avail_s the turn reads as ongoing, whatever the next word turns out to be.
    """
    cols = ["spk_hash", "start_s", "end_s", "start_avail_s", "end_avail_s"]
    if words is None or not len(words):
        return pd.DataFrame(columns=cols)
    w = words.sort_values(["start_s", "w_idx"], kind="stable")
    ws, we = w["start_s"].to_numpy(float), w["end_s"].to_numpy(float)
    ra = w["role_avail_s"].to_numpy(float)
    spk = w["spk_hash"].to_numpy(object)
    prof = ((w["role"].to_numpy(object) == config.ROLE_PROF)
            & np.isnan(w["role_event_start_s"].to_numpy(float)))
    rows, i, n = [], 0, len(w)
    while i < n:
        if not prof[i]:
            i += 1
            continue
        j, end = i, we[i]
        while j + 1 < n and prof[j + 1] and ws[j + 1] - end < gap_s:
            j += 1
            end = max(end, we[j])
        if j + 1 < n and ws[j + 1] - end < gap_s:      # someone else spoke within the gap
            end_av = max(ra[j + 1], ra[j])
        else:
            end_av = max(end + gap_s + config.EPS_W_S, ra[j])
        rows.append((spk[i], ws[i], end, ra[i], end_av))
        i = j + 1
    return pd.DataFrame(rows, columns=cols)


def voice_table(class_id: int, voice: pd.DataFrame, roster: Roster, words: pd.DataFrame | None,
                eps_f_s: float, *, timing_precision: str | None = None,
                salt: str | None = None) -> pd.DataFrame:
    """The `voice` table from `parse_voice` rows (+ the class's words for word-bearing flags),
    plus the professor's inferred turns (`inferred` True; see `professor_turns`). Forum rows
    leave start_avail_s / end_avail_s empty: they are known at start / end + eps_F."""
    if len(voice) == 0:
        return schemas.empty("voice")
    salt = salt if salt is not None else config.id_salt()
    hashes = {u: spk_hash(u, salt) for u in set(voice["user_id"].dropna())}
    df = pd.DataFrame({
        "class_id": int(class_id),
        "spk_hash": voice["user_id"].map(hashes),
        "role": [roster.role(u) for u in voice["user_id"]],
        "start_s": voice["start_s"].to_numpy(float),
        "end_s": voice["end_s"].to_numpy(float),
        "room_main": voice["room_main"].to_numpy(bool),
        "precise": voice["precise"].to_numpy(bool) if timing_precision is None
        else np.full(len(voice), timing_precision != "s"),
        "inferred": False, "start_avail_s": np.nan, "end_avail_s": np.nan,
    })
    turns = professor_turns(words)
    if len(turns):
        df = pd.concat([df, turns.assign(class_id=int(class_id), role=config.ROLE_PROF,
                                         room_main=True, precise=True, inferred=True)],
                       ignore_index=True)
        df = df.sort_values(["start_s", "end_s"], kind="stable").reset_index(drop=True)
    df["word_bearing"], df["first_word_avail_s"] = word_bearing_flags(df, words, eps_f_s)
    return schemas.coerce(df, "voice")


# =========================================================================== chat, segments


def chat_table(class_id: int, events: list[dict], clock: VideoClock, roster: Roster, *,
               masker=None, salt: str | None = None) -> tuple[pd.DataFrame, Counter]:
    """Human chat only (§4.1 A8, L27). Returns (table, counts of kept and dropped messages)."""
    masker = masker_for(roster, masker)
    salt = salt if salt is not None else config.id_salt()
    counts: Counter = Counter()
    rows = []
    for ev in events:
        if ev.get("event-type") != "message":
            continue
        counts["seen"] += 1
        uid = event_user_id(ev)
        data = ev.get("event-data") if isinstance(ev.get("event-data"), dict) else {}
        text = data.get("message-text", data.get("message"))
        if uid is None:
            counts["null_actor"] += 1
        elif not roster.is_member(uid):
            counts["non_roster"] += 1
        elif ev.get("private"):
            counts["private"] += 1
        elif not isinstance(text, str) or not text.strip():
            counts["empty"] += 1
        elif SYSTEM_NOTICE_RE.search(text):
            counts["notice"] += 1
        else:
            t = clock.master(event_start_ms(ev))
            if not np.isfinite(t):
                counts["gap"] += 1
                continue
            counts["kept"] += 1
            rows.append({"class_id": int(class_id), "t_s": t, "spk_hash": spk_hash(uid, salt),
                         "role": roster.role(uid), "text_masked": masker.mask_text(text)})
    if not rows:
        return schemas.empty("chat"), counts
    df = pd.DataFrame(rows).sort_values("t_s", kind="stable").reset_index(drop=True)
    return schemas.coerce(df, "chat"), counts


def segments_table(class_id: int, events: list[dict], clock: VideoClock, *, masker=None,
                   roster: Roster | None = None) -> tuple[pd.DataFrame, Counter]:
    """Timeline segments: start (unclamped, gap segments dropped), masked titles, seg_type.

    There is deliberately NO end column: a segment's end is the next segment's start, which
    is future class structure (§4.2, L10).
    """
    masker = masker_for(roster or Roster(None), masker)
    counts: Counter = Counter()
    rows, seen = [], set()
    for i, ev in enumerate(events):
        if ev.get("event-type") != "timeline-segment":
            continue
        data = ev.get("event-data") if isinstance(ev.get("event-data"), dict) else {}
        title = str(data.get("timeline-segment-title") or "").strip()
        section = str(data.get("timeline-section-title") or "").strip()
        t_ms = event_start_ms(ev)
        t = clock.master(t_ms)
        if not np.isfinite(t):
            counts["gap" if clock.status(t_ms) == "gap" else "no_time"] += 1
            continue
        key = (round(t, 3), title, section)
        if key in seen:
            counts["duplicate"] += 1
            continue
        seen.add(key)
        counts["kept"] += 1
        rows.append({"order": i, "start_s": t, "title_masked": masker.mask_text(title),
                     "section_masked": masker.mask_text(section),
                     "seg_type": config.seg_type_of(title, section)})
    if not rows:
        return schemas.empty("segments"), counts
    df = pd.DataFrame(rows).sort_values(["start_s", "order"], kind="stable")
    df = df.drop(columns="order").reset_index(drop=True)
    df.insert(0, "seg_idx", np.arange(len(df)))
    df.insert(0, "class_id", int(class_id))
    return schemas.coerce(df, "segments"), counts


# =========================================================================== polls


def parse_clock_string(text: str | None, pattern: re.Pattern = _POLL_OPEN_RE) -> float | None:
    """Seconds from strings like "started at 0:32:39" (H:MM:SS or M:SS)."""
    if not isinstance(text, str):
        return None
    m = pattern.search(text)
    if not m:
        return None
    parts = [int(p) for p in m.group(1).split(":")]
    secs = 0
    for p in parts:
        secs = secs * 60 + p
    return float(secs)


def _poll_time(secs: float | None, clock: VideoClock, reference: str, klass: dict | None
               ) -> float:
    if secs is None:
        return float("nan")
    if reference == "video":
        return float(clock.from_video([secs])[0])
    if reference == "recording_started":
        if not clock.segs:
            return float("nan")
        ref_ms = clock.segs[0].wall_start_ms
    elif reference == "scheduled_start":
        ref_ms = iso_to_ms(kebab(klass or {}).get("start-time"))
        if ref_ms is None:
            return float("nan")
    else:
        raise ValueError(f"poll reference must be one of {POLL_REFERENCES}")
    return clock.master(ref_ms + secs * 1000.0)


def polls_table(class_id: int, polls, poll_responses, clock: VideoClock, roster: Roster,
                segments: pd.DataFrame, eps_f_s: float, *, klass: dict | None = None,
                masker=None, reference: str = POLL_REFERENCE) -> pd.DataFrame:
    """Polls opened in class, with Brady's own answer only (§4.1 A9, A10).

    open_s: parsed "started at" clock string against `reference` (whole seconds, so
    availability uses eps_F,poll). close_s: a parsed "ended at" string if any, else the first
    segment start after the opening (perceptible when the slide changes). Brady's answer is
    available at its submit time (else close_s), plus eps_F,poll.
    """
    masker = masker_for(roster, masker)
    polls = [p for p in (polls or []) if isinstance(p, dict)]
    if not polls:
        return schemas.empty("polls")
    seg_starts = np.sort(segments["start_s"].to_numpy(float)) if len(segments) else np.array([])
    mine = [r for r in (poll_responses or []) if isinstance(r, dict) and roster.me_id is not None
            and str(r.get("user-id")) == roster.me_id]
    eps_poll = config.eps_f_poll(eps_f_s)
    rows = []
    for p in polls:
        pid = str(p.get("id"))
        timing = p.get("timing-text")
        open_s = _poll_time(parse_clock_string(timing), clock, reference, klass)
        close_s = _poll_time(parse_clock_string(timing, _POLL_CLOSE_RE), clock, reference, klass)
        if not np.isfinite(close_s) and np.isfinite(open_s):
            later = seg_starts[seg_starts > open_s]
            close_s = float(later[0]) if len(later) else float("nan")
        resp = [r for r in mine if str(r.get("poll-id")) == pid]
        answer, avail = None, float("nan")
        if resp:
            resp.sort(key=lambda r: iso_to_ms(r.get("submitted-at")) or 0)
            r = resp[-1]
            answer = r.get("answer") if isinstance(r.get("answer"), str) else None
            if answer is None:
                chosen = [c.get("text") for c in p.get("choices") or []
                          if isinstance(c, dict) and str(c.get("id")) == str(r.get("choice-id"))]
                answer = chosen[0] if chosen else None
            submit_ms = iso_to_ms(r.get("submitted-at"))
            submit = clock.master(submit_ms) if submit_ms is not None else float("nan")
            base = submit if np.isfinite(submit) else close_s
            avail = base + eps_poll if np.isfinite(base) else float("nan")
        prompt = p.get("prompt") if isinstance(p.get("prompt"), str) else p.get("title")
        rows.append({"class_id": int(class_id), "poll_id": pid, "open_s": open_s,
                     "close_s": close_s, "prompt_masked": masker.mask_text(prompt),
                     "my_answer_text": masker.mask_text(answer), "my_answer_avail_s": avail})
    df = pd.DataFrame(rows).sort_values("open_s", kind="stable").reset_index(drop=True)
    return schemas.coerce(df, "polls")


def poll_reference_check(polls: pd.DataFrame, segments: pd.DataFrame) -> pd.DataFrame:
    """D9 helper: distance from each parsed open_s to the nearest `poll` segment start."""
    starts = segments.loc[segments["seg_type"] == "poll", "start_s"].to_numpy(float)
    out = polls[["class_id", "poll_id", "open_s"]].copy()
    out["nearest_poll_seg_s"] = [starts[np.argmin(np.abs(starts - o))] if len(starts)
                                 and np.isfinite(o) else np.nan for o in out["open_s"]]
    out["delta_s"] = out["open_s"] - out["nearest_poll_seg_s"]
    return out


# =========================================================================== presence


def presence_table(class_id: int, roster: Roster, clock: VideoClock, *,
                   salt: str | None = None) -> pd.DataFrame:
    """enter-class-time per user who entered (ctx_n_present(t), §4.1 A11; available + eps_F).

    Presence is a state, not an event: someone who joined during a recording gap is present
    from the restart on, so that entry maps to the restart instant. Joins before recording
    have negative times. The class-users `absent` flag is post-class attendance bookkeeping
    (USE_META): it never filters presence, so a user who joined at x counts from x + eps_F
    whatever was recorded about them after class.
    """
    salt = salt if salt is not None else config.id_salt()
    rows = []
    for uid, iso in roster.enter_iso.items():
        if not iso:
            continue
        w = iso_to_ms(iso)
        if w is None:
            continue
        if clock.status(w) == "gap":
            w = clock.next_start_wall(w)
        t = clock.master(w)
        if not np.isfinite(t):                        # no recording timeline to place it on
            continue
        rows.append({"class_id": int(class_id), "spk_hash": spk_hash(uid, salt),
                     "role": roster.role(uid), "enter_s": t})
    if not rows:
        return schemas.empty("presence")
    df = pd.DataFrame(rows).sort_values("enter_s", kind="stable").reset_index(drop=True)
    return schemas.coerce(df, "presence")


# =========================================================================== hand log (QA)


def parse_hand_log(obj) -> list[dict]:
    """extension_v3 'my hand raises' export: {"entries": [...]} or a bare list."""
    entries = obj.get("entries") if isinstance(obj, dict) else obj
    out = []
    for e in entries or []:
        if not isinstance(e, dict):
            continue
        e = kebab(e)
        t = epoch_ms(e.get("t"))
        if t is None:
            continue
        cid = e.get("class-id", e.get("classId", e.get("classid")))
        out.append({"t_ms": int(t), "class_id": None if cid in (None, "") else str(cid)})
    return out


def hand_log_import(entries, class_id: int, clock: VideoClock, *,
                    episodes: pd.DataFrame | None = None,
                    window: tuple[float, float] = HAND_LOG_MATCH_S) -> pd.DataFrame:
    """Live-logger key presses for one class -> the QA-only `hand_log` table (§2.7).

    Entries tagged with another class are ignored; untagged entries count when they fall
    inside this class's recording. Each press is matched to the nearest bid/vote onset in
    [press + window[0], press + window[1]]; lag_s = onset - press (key press -> icon).
    This table never feeds features or transcripts (Scribe's --hand-log is never used).
    """
    parsed = entries if isinstance(entries, list) and all(
        isinstance(e, dict) and "t_ms" in e for e in entries) else parse_hand_log(entries)
    rows = []
    for e in parsed:
        if e["class_id"] is not None and e["class_id"] != str(class_id):
            continue
        if e["class_id"] is None and clock.status(e["t_ms"]) != "in":
            continue
        rows.append({"class_id": int(class_id), "press_wall_ms": int(e["t_ms"]),
                     "press_s": clock.master(e["t_ms"])})
    if not rows:
        return schemas.empty("hand_log")
    df = pd.DataFrame(rows).sort_values("press_wall_ms").reset_index(drop=True)
    onsets = np.array([])
    if episodes is not None and len(episodes):
        ep = episodes[episodes["ep_type"].isin([config.EP_BID, config.EP_VOTE])]
        if "class_id" in ep.columns:
            ep = ep[ep["class_id"] == int(class_id)]
        on = ep["onset_s"].to_numpy(float)
        if "onset_retimed_s" in ep.columns:
            re_t = ep["onset_retimed_s"].to_numpy(float)
            on = np.where(np.isfinite(re_t), re_t, on)
        onsets = np.sort(on[np.isfinite(on)])
    matched = []
    for p in df["press_s"].to_numpy(float):
        d = onsets - p
        ok = (d >= window[0]) & (d <= window[1]) if np.isfinite(p) else np.zeros(len(d), bool)
        matched.append(float(onsets[ok][np.argmin(np.abs(d[ok]))]) if ok.any() else np.nan)
    df["matched_onset_s"] = matched
    df["lag_s"] = df["matched_onset_s"] - df["press_s"]
    return schemas.coerce(df, "hand_log")


# =========================================================================== orchestration


@dataclass
class EventTables:
    voice: pd.DataFrame
    chat: pd.DataFrame
    segments: pd.DataFrame
    polls: pd.DataFrame
    presence: pd.DataFrame
    counts: dict
    clock: VideoClock

    def tables(self) -> dict[str, pd.DataFrame]:
        return {name: getattr(self, name) for name in _EVENT_TABLES}


def build_event_tables(class_id: int, *, klass: dict, events: list[dict], me_id,
                       eps_f_s: float, cal: dict | None = None, words: pd.DataFrame | None = None,
                       polls=None, poll_responses=None, salt: str | None = None, masker=None,
                       probe: dict | None = None, poll_reference: str = POLL_REFERENCE,
                       timing_precision: str | None = None) -> EventTables:
    """Every event table for one class from whitelisted raw documents (pure; no disk)."""
    require_me_id(me_id)
    salt = salt if salt is not None else config.id_salt()
    events = allowed_events(events)
    klass = whitelist_filter("class", klass or {})
    roster = roster_from_class(klass, me_id)
    masker = masker_for(roster, masker)
    clock = video_clock(probe, cal, klass)
    voice = voice_table(class_id, parse_voice(events, clock), roster, words, eps_f_s,
                        timing_precision=timing_precision, salt=salt)
    chat, chat_counts = chat_table(class_id, events, clock, roster, masker=masker, salt=salt)
    segs, seg_counts = segments_table(class_id, events, clock, masker=masker, roster=roster)
    polls_df = polls_table(class_id, whitelist_filter("polls", polls or []),
                           whitelist_filter("poll_responses", poll_responses or []), clock,
                           roster, segs, eps_f_s, klass=klass, masker=masker,
                           reference=poll_reference)
    presence = presence_table(class_id, roster, clock, salt=salt)
    counts = {f"chat_{k}": v for k, v in chat_counts.items()}
    counts.update({f"segments_{k}": v for k, v in seg_counts.items()})
    return EventTables(voice, chat, segs, polls_df, presence, counts, clock)


def write_table(name: str, df: pd.DataFrame, class_ids: Iterable, *,
                tables_dir: Path | str | None = None) -> Path:
    """Upsert one class-keyed table: replace the rows of `class_ids`, keep the others."""
    path = Path(tables_dir) / f"{name}.parquet" if tables_dir else config.table_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = {int(c) for c in class_ids}
    parts = []
    if path.exists():
        old = pd.read_parquet(path)
        parts.append(old[~old["class_id"].astype("int64").isin(ids)])
    parts.append(df)
    parts = [p for p in parts if len(p)]
    out = pd.concat(parts, ignore_index=True) if parts else df
    out = schemas.coerce(out.reset_index(drop=True), name) if name in schemas.TABLES else out
    if name in schemas.TABLES:
        schemas.validate(out, name)
    out.to_parquet(path, index=False)
    return path


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def event_tables(class_id, *, raw_dir=None, scribe_dir=None, tables_dir=None,
                 salt: str | None = None, masker=None) -> EventTables:
    """Event tables for one class from disk: raw bundle + the `words` stage's timing row and
    words (for word-bearing flags). Without them the clock is uncalibrated and eps_F falls
    back to EPS_F_FALLBACK_S (D5's declared fallback)."""
    from . import transcribe

    cid = int(class_id)
    klass = read_raw(cid, "class", raw_dir)
    if klass is None:
        raise FileNotFoundError(f"no class.json for class {cid}")
    events = read_raw(cid, "events", raw_dir) or []
    timing = transcribe.read_timing(cid, tables_dir=tables_dir)
    words = None
    wpath = Path(tables_dir) / "words.parquet" if tables_dir else config.table_path("words")
    if wpath.exists():
        words = pd.read_parquet(wpath, filters=[("class_id", "==", cid)])
    if timing is not None:
        cal, eps, precision = timing["cal"], timing["eps_F_s"], timing["timing_precision"]
    else:
        log.warning("class %s: no calibration from the words stage; identity clock, eps_F %.1f",
                    cid, config.EPS_F_FALLBACK_S)
        cal, eps, precision = None, config.EPS_F_FALLBACK_S, None
    return build_event_tables(
        cid, klass=klass, events=events, me_id=load_me_id(scribe_dir), eps_f_s=eps, cal=cal,
        words=words, polls=read_raw(cid, "polls", raw_dir),
        poll_responses=read_raw(cid, "poll_responses", raw_dir), salt=salt, masker=masker,
        timing_precision=precision)


def _class_ids(args, *, raw_dir=None) -> list[int]:
    if getattr(args, "class_ids", None):
        return [int(c) for c in args.class_ids]
    if getattr(args, "phase", None):
        cpath = config.table_path("classes")
        if not cpath.exists():
            raise SystemExit("--phase needs data/tables/classes.parquet (run `oncue select`)")
        classes = pd.read_parquet(cpath, columns=["class_id", "phase"])
        return [int(c) for c in classes.loc[classes["phase"] == args.phase, "class_id"]]
    root = Path(raw_dir or config.RAW_DIR)
    return sorted(int(p.name) for p in root.iterdir()
                  if p.is_dir() and p.name.isdigit() and (p / "class.json").exists()) \
        if root.exists() else []


def cli(args) -> int:
    """`python -m oncue events [--class CID ...] [--phase P]`."""
    config.load_env()
    ids = _class_ids(args)
    if not ids:
        print("oncue events: no classes to process")
        return 1
    built = {name: [] for name in _EVENT_TABLES}
    counts, failed = [], []
    for cid in ids:
        try:
            et = event_tables(cid)
        except Exception as exc:  # noqa: BLE001 - one bad class must not stop the batch
            log.error("class %s: %s", cid, exc)
            failed.append(cid)
            continue
        for name, df in et.tables().items():
            built[name].append(df)
        counts.append({"class_id": cid, **et.counts})
    done = [c for c in ids if c not in failed]
    for name, parts in built.items():
        df = pd.concat(parts, ignore_index=True) if parts else schemas.empty(name)
        write_table(name, df, done)
    if counts:
        config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(counts).to_csv(config.LOGS_DIR / "event_counts.csv", index=False)
    print(f"oncue events: {len(done)} classes written, {len(failed)} failed")
    return 0 if not failed else 1


def cli_hand_log_import(args) -> int:
    """`python -m oncue hand-log-import PATH [--class CID ...]` -> hand_log (QA only)."""
    p = argparse.ArgumentParser(prog="oncue hand-log-import")
    p.add_argument("path", type=Path)
    opts = p.parse_args(args.extra)
    from . import transcribe

    entries = parse_hand_log(json.loads(opts.path.read_text(encoding="utf-8")))
    eps_path = config.table_path("hand_episodes")
    episodes = pd.read_parquet(eps_path) if eps_path.exists() else None
    ids = _class_ids(args)
    parts = []
    for cid in ids:
        klass = read_raw(cid, "class")
        if klass is None:
            continue
        timing = transcribe.read_timing(cid)
        clock = video_clock(None, timing["cal"] if timing else None, klass)
        df = hand_log_import(entries, cid, clock, episodes=episodes)
        if len(df):
            parts.append(df)
    df = pd.concat(parts, ignore_index=True) if parts else schemas.empty("hand_log")
    write_table("hand_log", df, ids)
    print(f"oncue hand-log-import: {len(df)} presses in {len(parts)} classes (QA only)")
    return 0
