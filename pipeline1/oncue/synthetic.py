"""Deterministic synthetic world for every On Cue test (no real data, no real classmates).

`world(n_classes, seed)` builds N invented classes. Each `SynthClass` carries:

(a) `raw`: a bundle shaped like the real Forum data, keyed by `config.RAW_SOURCE_FILES`
    (class JSON with recording-sessions, section/course, class-users; class_edit_page with
    preparatory-assignment and focused outcomes; class-events mixing voice / message /
    timeline-segment / bookmark-note events in BOTH kebab-case and snake_case keys, voice
    `timestamp` = END in epoch ms with start = timestamp - duration, "Main Classroom" plus
    breakout rooms; polls, poll responses, breakouts) and trap fields that loaders must drop;
(b) `asr`: a Scribe `*_asr.json`-like dict (words with start/end/text/punctuated_word/conf/
    cluster/filler) and `words`, the truth word table with speaker ids and roles;
    `scribe_transcript(sc)` renders a real `scribe.transcript/v2` dict with scribe_core;
(c) SLE PCW template (blocks[].block_components[].value = Quill delta JSON string) and doc
    (answers keyed by block uuid), in `raw["pcw_template"]` / `raw["pcw_doc"]`;
(d) ground-truth hand `episodes` for Brady (bids, votes, glitches), with professor questions
    placed before most bids, plus other students' hands;
(e) video-strip frames on demand: `frames_for(sc, t_list)` returns BGR uint8 images with a
    dark strip of tiles, binarised-looking name glyphs, the green speaker border, the
    yellow-green hand icon (hue 52-80 deg, white glyph) or the green poll tick (hue 120-165).

All times in the truth tables are master-clock seconds (the audio of the recording).
Forum events are written in wall-clock ms such that `scribe_core` mapping plus the Scribe
calibration convention (audio = forum + offset, offset = `meta["cal_offset_s"]`) recovers
them. `truth_tables(sc)` returns oncue-schema tables built from the truth, for tests of
downstream stages; `write_world(w, root)` writes the `scribe sync` + transcripts layout.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import uuid
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from . import config

# --------------------------------------------------------------------------- invented people

ME_ID = 7777
ME_FIRST, ME_LAST = "Brady", "Rutherford"
STUDENT_POOL = (
    ("Ada", "Lovelace"), ("Alan", "Turing"), ("Grace", "Hopper"), ("Emmy", "Noether"),
    ("Carl", "Gauss"), ("Marie", "Curie"), ("Rosalind", "Franklin"), ("Hedy", "Lamarr"),
    ("Sophie", "Germain"), ("Blaise", "Pascal"), ("Will", "Herschel"), ("Mark", "Faraday"),
    ("Srinivasa", "Ramanujan"), ("Leonhard", "Euler"), ("Florence", "Nightingale"),
    ("Nikola", "Tesla"), ("Johannes", "Kepler"), ("Mary", "Somerville"),
)
PROF_POOL = (("Hypatia", "Alexandria"), ("Euclid", "Megara"), ("Galileo", "Pisa"),
             ("Avicenna", "Bukhara"), ("Archimedes", "Syracuse"))
SECTION_SPECS = (
    ("CS901", "Learning Systems", "Fall 2026"),
    ("SS902", "Decision Institutions", "Fall 2023"),   # whole-second voice timestamps
    ("NS903", "Living Networks", "Spring 2025"),
    ("B905", "Venture Strategy", "Fall 2025"),
    ("AH904", "Reading Arguments", "Spring 2026"),
)
TERM_START = {"F23": date(2023, 9, 4), "Sp24": date(2024, 1, 22), "F24": date(2024, 9, 2),
              "Sp25": date(2025, 1, 20), "F25": date(2025, 9, 1), "Sp26": date(2026, 1, 19),
              "F26": date(2026, 9, 7)}
TOPICS = {
    "CS901": "model gradient overfitting validation features loss regularization kernel "
             "boosting embedding variance bias pipeline dataset classifier threshold",
    "SS902": "incentive policy market equilibrium voting institution coalition cost benefit "
             "contract signal auction treaty norm sanction ballot",
    "NS903": "population energy network feedback ecosystem cell signal growth predator "
             "nutrient pathway mutation habitat entropy carbon gene",
    "B905": "customer pricing strategy competitor revenue margin channel funnel segment "
            "retention brand supplier valuation pivot runway",
    "AH904": "argument premise rhetoric claim evidence ethics narrative metaphor audience "
             "author irony thesis context genre tone fallacy",
}

# --------------------------------------------------------------------------- speech templates

LECTURE = (
    "so the key idea here is that {a} shapes {b} in most cases",
    "if we look at the {a} we can see how {b} changes over time",
    "remember from the reading that {a} and {b} are not the same thing",
    "one way to think about {a} is as a kind of {b}",
    "this is where {a} starts to matter for the {b}",
    "notice that the {a} depends on the {b} we choose",
    "let me put the {a} on the board next to the {b}",
)
QUESTIONS = (  # (template, kind)
    ("what do you think happens to the {a} when the {b} changes", "plain"),
    ("why does {a} matter for {b}", "plain"),
    ("how would you explain the link between {a} and {b}", "invite"),
    ("does anyone want to share how they handled the {a}", "invite"),
    ("any thoughts on {a} versus {b}", "invite"),
    ("can someone tell me what {a} means here", "invite"),
    ("who would like to start with the {a}", "invite"),
    ("does that make sense so far", "check"),
)
VOTES = ("raise your hand if you think {a} wins", "show of hands who agrees that {a} drives {b}",
         "hands up if you chose the {a}")
CALLONS = ("{name}, go ahead", "i see your hand {name}", "{name}, you're up")
STUDENT = (
    "i think {a} is related to {b} because the {c} keeps changing",
    "building on that the {a} seems to explain the {b}",
    "i would push back a little since {a} ignores the {b}",
    "for me the {a} was the hardest part of the {b}",
    "maybe the {a} only works when the {b} is stable",
)
BRADY_LEADS = ("i think", "in my notes i argued that", "so from my pre class work",
               "to add to that")
FOLLOWUP = ("okay thanks that is a helpful point about the {a}",
            "great so let us connect that to the {b}")

# --------------------------------------------------------------------------- frame geometry

FRAME_W, FRAME_H = 960, 540
TILE_H = 64
GUTTER = 4
LABEL_H = 13
ICON_R = 8
ICON_OFF = 4              # icon top-left inset from the tile corner
BORDER_PX = 2
L_UI_S = 0.15             # speaker-border latency after the Forum voice event


def _hsv_to_bgr(h: int, s: int, v: int) -> tuple[int, int, int]:
    px = cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(px[0]), int(px[1]), int(px[2])


HAND_BGR = _hsv_to_bgr(33, 220, 235)        # hue 66 deg
TICK_BGR = _hsv_to_bgr(72, 210, 225)        # hue 144 deg
BORDER_BGR = _hsv_to_bgr(55, 200, 210)      # hue 110 deg (outside both icon ranges)
LIME_BGR = _hsv_to_bgr(35, 200, 220)        # distractor stripe ("lime shirt")
WHITE = (255, 255, 255)
GUTTER_BGR = (8, 8, 8)

# --------------------------------------------------------------------------- data classes


@dataclass
class SynthClass:
    """One invented class. Truth tables are DataFrames on the master clock (audio seconds)."""

    class_id: int
    meta: dict
    raw: dict
    asr: dict
    words: pd.DataFrame
    turns: pd.DataFrame
    voice: pd.DataFrame
    segments: pd.DataFrame
    episodes: pd.DataFrame
    others_hands: pd.DataFrame
    polls: pd.DataFrame
    ticks: pd.DataFrame
    chat: pd.DataFrame
    roster: pd.DataFrame
    pcw: dict
    layout: dict
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def T_end_s(self) -> float:
        return float(self.meta["T_end_s"])

    @property
    def me_tile(self) -> int:
        return int(self.layout["me_tile"])


@dataclass
class World:
    seed: int
    classes: list[SynthClass]
    me: dict
    sections: list[dict]
    schedules: dict[int, dict]

    def __len__(self) -> int:
        return len(self.classes)

    def __iter__(self):
        return iter(self.classes)

    @property
    def class_ids(self) -> list[int]:
        return [c.class_id for c in self.classes]

    def by_id(self, class_id) -> SynthClass:
        for c in self.classes:
            if c.class_id == int(class_id):
                return c
        raise KeyError(class_id)


# --------------------------------------------------------------------------- small helpers


def _iso(ms: float) -> str:
    dt = datetime.fromtimestamp(ms / 1000.0, tz=UTC)
    return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _iso_s(ms: float) -> str:
    dt = datetime.fromtimestamp(math.floor(ms / 1000.0), tz=UTC)
    return dt.isoformat().replace("+00:00", "Z")


def _uuid(*parts) -> str:
    return str(uuid.UUID(hashlib.md5(":".join(map(str, parts)).encode()).hexdigest()))


def _snake(obj):
    """kebab-case keys -> snake_case keys, recursively (what Forum's web app receives)."""
    if isinstance(obj, dict):
        return {(k.replace("-", "_") if isinstance(k, str) else k): _snake(v)
                for k, v in obj.items()}
    if isinstance(obj, list):
        return [_snake(v) for v in obj]
    return obj


def spk_hash(user_id) -> str:
    """Stable synthetic pseudonym (tests only; real tables use Scribe's HMAC pseudonym)."""
    return "u_" + hashlib.sha1(f"synthetic:{user_id}".encode()).hexdigest()[:10]


def _quill(*parts: tuple[str, dict | None]) -> str:
    ops = [{"insert": text, **({"attributes": attrs} if attrs else {})} for text, attrs in parts]
    return json.dumps({"ops": ops})


def _norm(token: str) -> str:
    return re.sub(r"[^\w\s]", "", token.lower())


# --------------------------------------------------------------------------- text and timing


@dataclass
class _Sentence:
    tokens: list[str]          # lowercase bare tokens, names capitalised
    names: list[bool]
    end: str                   # "." or "?"
    commas: set               # token indexes followed by ","
    kind: str = "statement"


def _fill(rng, template: str, topic: list[str], name: str | None = None,
          end: str = ".", kind: str = "statement") -> _Sentence:
    a, b, c = rng.choice(topic, size=3, replace=False)
    toks, names, commas = [], [], set()
    for raw in template.split():
        comma = raw.endswith(",")
        raw = raw.rstrip(",")
        tok = raw.format(a=a, b=b, c=c, name=name or "")
        for piece in tok.split():
            toks.append(piece)
            names.append(raw == "{name}")
        if comma:
            commas.add(len(toks) - 1)
    return _Sentence(toks, names, end, commas, kind)


def _callon_mask(tokens: list[str]) -> list[bool]:
    """Tokens covered by a CALLON phrase (matched on the normalised sentence)."""
    text, spans = "", []
    for tok in tokens:
        if text:
            text += " "
        spans.append((len(text), len(text) + len(tok)))
        text += _norm(tok)
    hit = [False] * len(tokens)
    for m in config.CALLON_RE.finditer(text):
        for i, (a, b) in enumerate(spans):
            if a < m.end() and b > m.start():
                hit[i] = True
    return hit


def _emit(rng, start: float, sentences: list[_Sentence], speaker: dict, turn_id: int,
          *, max_end: float | None = None, split_p: float = 0.15) -> tuple[list[dict], float]:
    """Time the words of `sentences` from `start`. Returns (word dicts, end of last word)."""
    words: list[dict] = []
    t = start
    for si, sent in enumerate(sentences):
        if si:
            t += rng.uniform(1.3, 2.2) if rng.random() < split_p else rng.uniform(0.25, 0.85)
        toks, names = list(sent.tokens), list(sent.names)
        fill = rng.random() < 0.05 and speaker["role"] != config.ROLE_PROF
        callon = _callon_mask(toks)
        if fill:
            toks.insert(0, str(rng.choice(["um", "uh"])))
            names.insert(0, False)
            callon.insert(0, False)
        for i, tok in enumerate(toks):
            is_name = names[i]
            is_filler = fill and i == 0
            punct = tok if is_name else ("I" if tok == "i" else tok)
            if i == 0 or (fill and i == 1):
                punct = punct[:1].upper() + punct[1:]
            k = i - (1 if fill else 0)
            if is_filler or k in sent.commas:
                punct += ","
            if i == len(toks) - 1:
                punct += sent.end
            dur = float(np.clip(0.09 + 0.055 * len(tok), 0.12, 0.7))
            if is_filler:
                dur = 0.3
            if max_end is not None and t + dur > max_end:
                return words, (words[-1]["end_s"] if words else start)
            words.append({
                "start_s": round(t, 3), "end_s": round(t + dur, 3), "word": _norm(tok),
                "text": punct, "user_id": speaker["user_id"], "role": speaker["role"],
                "is_name": is_name, "is_callon": callon[i], "filler": is_filler,
                "turn_id": turn_id,
                "conf": round(float(rng.uniform(0.7, 1.0)), 3),
            })
            t += dur + rng.uniform(0.02, 0.12)
    return words, (words[-1]["end_s"] if words else start)


# --------------------------------------------------------------------------- section and roster


def _sections(n_classes: int, seed: int) -> list[dict]:
    n_sec = min(len(SECTION_SPECS), max(1, (n_classes + 1) // 2))
    out = []
    for k in range(n_sec):
        rng = np.random.default_rng([seed, 10_000 + k])
        code, title, term_name = SECTION_SPECS[k]
        pool = list(range(len(STUDENT_POOL)))
        n_students = int(rng.integers(9, 12))
        forced = [2, 10, 11] if k == 0 else []          # Grace, Will, Mark: common-word names
        rest = [i for i in rng.permutation(pool) if i not in forced]
        picks = forced + rest[: n_students - len(forced)]
        prof_first, prof_last = PROF_POOL[k % len(PROF_POOL)]
        roster = [{"user_id": str(100 + k), "first": prof_first, "last": prof_last,
                   "role": config.ROLE_PROF, "forum_role": "professor"}]
        roster.append({"user_id": str(ME_ID), "first": ME_FIRST, "last": ME_LAST,
                       "role": config.ROLE_ME, "forum_role": "student"})
        for i in picks:
            first, last = STUDENT_POOL[i]
            roster.append({"user_id": str(2000 + i), "first": first, "last": last,
                           "role": config.ROLE_STUDENT, "forum_role": "student"})
        term = config.term_code(term_name)
        out.append({"section_id": 7000 + k, "course_code": code, "course_title": title,
                    "term_name": term_name, "term": term, "roster": roster,
                    "topic": TOPICS[code].split(), "weekday": int(rng.integers(0, 4)),
                    "hour_utc": int(rng.choice([14, 16, 18]))})
    return out


SEGMENT_PLAN = (  # (fraction of the class at which it starts, section title, segment title)
    (0.0, "Opening", "Warm-up and recap"),
    (0.08, "Part 1", "Poll: first impressions"),
    (0.14, "Part 1", "Discussion of the readings"),
    (0.38, "Part 2", "Breakout instructions"),
    (0.41, "Part 2", "Breakouts"),
    (0.55, "Part 2", "Debrief"),
    (0.72, "Break", "Eye rest break"),
    (0.75, "Part 3", "Case activity"),
    (0.92, "Closing", "Wrap-up and reflection"),
)
TALK_TYPES = ("intro", "poll", "debrief", "activity", "wrapup", "breakout_instr")
BID_TYPES = ("debrief", "activity", "wrapup", "intro")


# --------------------------------------------------------------------------- one class


class _Builder:
    """Builds one class. Kept as a class so the many intermediate lists stay readable."""

    def __init__(self, i: int, sec: dict, seed: int, minutes: float, multi: bool, n_secs: int):
        self.rng = np.random.default_rng([seed, i])
        self.i, self.sec, self.seed = i, sec, seed
        self.L = float(minutes) * 60.0
        self.multi = multi
        self.session_no = i // n_secs + 1
        self.cid = 900_000 + 101 * (i + 1)
        self.topic = sec["topic"]
        self.words: list[dict] = []
        self.turns: list[dict] = []
        self.blocks: list[tuple[float, float]] = []
        self.voice_extra: list[dict] = []      # blips and breakout voice (no words)
        self.turn_id = 0

    # ---- people ------------------------------------------------------------------------
    def people(self):
        rng = self.rng
        roster = [dict(r) for r in self.sec["roster"]]
        students = [r for r in roster if r["role"] == config.ROLE_STUDENT]
        absent = students[int(rng.integers(len(students)))]["user_id"]
        late = [r["user_id"] for r in students if r["user_id"] != absent]
        late = late[int(rng.integers(len(late)))]
        tile_users = [r["user_id"] for r in roster if r["role"] != config.ROLE_PROF
                      and r["user_id"] != absent]
        order = [tile_users[j] for j in rng.permutation(len(tile_users))]
        for r in roster:
            r["present"] = r["user_id"] != absent
            r["tile_idx"] = order.index(r["user_id"]) if r["user_id"] in order else None
            r["enter_s"] = (float(rng.uniform(60, 300)) if r["user_id"] == late
                            else float(-rng.uniform(5, 120)))
        self.roster = roster
        self.by_uid = {r["user_id"]: r for r in roster}
        self.prof = roster[0]
        self.me = self.by_uid[str(ME_ID)]
        self.others = [r for r in roster if r["role"] == config.ROLE_STUDENT and r["present"]]
        self.tile_order = order

    # ---- clock -------------------------------------------------------------------------
    def clock(self):
        rng = self.rng
        d0 = TERM_START[self.sec["term"]] + timedelta(days=7 * (self.session_no - 1)
                                                      + self.sec["weekday"])
        sched = datetime(d0.year, d0.month, d0.day, self.sec["hour_utc"], 0, tzinfo=UTC)
        self.date = d0.isoformat()
        self.sched_ms = int(sched.timestamp() * 1000)
        self.rec_offset_s = float(round(rng.uniform(20, 180), 3))
        r0 = self.sched_ms + int(self.rec_offset_s * 1000)
        self.cal_offset = float(round(rng.uniform(-0.8, 0.8), 3))
        self.precise = self.sec["term"] != "F23"
        if self.multi:
            d1 = round(self.L * 0.47, 3)
            gap_ms = int(rng.uniform(120, 300) * 1000)
            s1 = (r0, r0 + int(d1 * 1000), 0.0)
            s2 = (s1[1] + gap_ms, s1[1] + gap_ms + int((self.L - d1) * 1000), d1)
            self.sessions = [s1, s2]
            self.gap_wall = (s1[1], s2[0])
        else:
            self.sessions = [(r0, r0 + int(self.L * 1000), 0.0)]
            self.gap_wall = None

    def wall(self, audio_t: float) -> int:
        """Wall ms of a Forum event that truly happened at audio time `audio_t`."""
        v = audio_t - self.cal_offset            # Forum video-time (audio = forum + offset)
        for a, b, v0 in reversed(self.sessions):
            if v >= v0:
                return round(a + (v - v0) * 1000.0)
        a, _, v0 = self.sessions[0]
        return round(a + (v - v0) * 1000.0)

    # ---- structure ---------------------------------------------------------------------
    def segments(self):
        rows = []
        for k, (frac, section, title) in enumerate(SEGMENT_PLAN):
            start = 3.0 if k == 0 else round(frac * self.L, 2)
            if title == "Case activity" and self.i % 2 == 1:
                first, last = self.others[0]["first"], self.others[0]["last"]
                title = f"Case activity: {first} {last}'s proposal"     # masking trap
            rows.append({"seg_idx": k, "start_s": start, "section": section, "title": title,
                         "seg_type": config.seg_type_of(title, section)})
        for k, r in enumerate(rows):
            r["end_s"] = rows[k + 1]["start_s"] if k + 1 < len(rows) else self.L
        self.segs = rows

    def seg_at(self, t: float) -> dict:
        for s in reversed(self.segs):
            if t >= s["start_s"]:
                return s
        return self.segs[0]

    def _free(self, a: float, b: float, taken: list[tuple[float, float]]) -> bool:
        return all(b <= x or a >= y for x, y in taken)

    def place_episodes(self):
        """Choose onset times for bids, one vote, one cold call and one glitch."""
        rng = self.rng
        spans = [(s["start_s"] + 25.0, s["end_s"] - 45.0) for s in self.segs
                 if s["seg_type"] in BID_TYPES]
        spans = [(max(a, config.T_OPEN_S + 20.0), b) for a, b in spans]
        spans = [(a, b) for a, b in spans if b - a > 10]
        self.plan, self.taken = [], []
        if not spans:
            return
        n_bids = max(2, int(np.rint(rng.integers(3, 6) * self.L / 1800.0)))
        plan, taken = [], []

        def draw(kind, pad_before, pad_after):
            for _ in range(400):
                a, b = spans[int(rng.integers(len(spans)))]
                if b <= a:
                    continue
                o = float(round(rng.uniform(a, b), 2))
                if self._free(o - pad_before, o + pad_after, taken):
                    taken.append((o - pad_before, o + pad_after))
                    plan.append((kind, o))
                    return
        for _ in range(n_bids):
            draw("bid", 25.0, 50.0)
        draw("vote", 20.0, 30.0)
        draw("coldcall", 20.0, 40.0)
        draw("glitch", 15.0, 15.0)
        self.plan = sorted(plan, key=lambda x: x[1])
        self.taken = taken

    # ---- speech ------------------------------------------------------------------------
    def say(self, start: float, sentences: list[_Sentence], who: dict, kind: str,
            *, end_at: float | None = None, max_end: float | None = None) -> tuple[float, float]:
        """Emit a turn. With `end_at`, the turn is shifted so its last word ends there."""
        self.turn_id += 1
        words, end = _emit(self.rng, start, sentences, who, self.turn_id, max_end=max_end)
        if not words:
            return start, start
        if end_at is not None:
            shift = end_at - end
            for w in words:
                w["start_s"] = round(w["start_s"] + shift, 3)
                w["end_s"] = round(w["end_s"] + shift, 3)
        self.words += words
        a, b = words[0]["start_s"], words[-1]["end_s"]
        q_kind = sentences[-1].kind if sentences[-1].end == "?" else None
        self.turns.append({"turn_id": self.turn_id, "user_id": who["user_id"],
                           "role": who["role"], "start_s": a, "end_s": b, "kind": kind,
                           "q_kind": q_kind})
        return a, b

    def lecture(self, n: int) -> list[_Sentence]:
        return [_fill(self.rng, str(self.rng.choice(LECTURE)), self.topic) for _ in range(n)]

    def question(self, kinds=("plain", "invite")) -> _Sentence:
        opts = [q for q in QUESTIONS if q[1] in kinds]
        tmpl, kind = opts[int(self.rng.integers(len(opts)))]
        return _fill(self.rng, tmpl, self.topic, end="?", kind=kind)

    def student_sents(self, n: int) -> list[_Sentence]:
        return [_fill(self.rng, str(self.rng.choice(STUDENT)), self.topic) for _ in range(n)]

    def brady_sents(self) -> list[_Sentence]:
        rng = self.rng
        answers = [a for a, k in zip(self.pcw["answers_text"], self.pcw["answer_kinds"],
                                     strict=True) if a and k == "text"]
        answers = answers or [" ".join(self.topic)]
        src = str(answers[int(rng.integers(len(answers)))]).split()
        out = []
        for _ in range(int(rng.integers(2, 4))):
            n = int(rng.integers(8, 15))
            j = int(rng.integers(0, max(1, len(src) - n)))
            lead = str(rng.choice(BRADY_LEADS)).split()
            toks = lead + [_norm(t) for t in src[j:j + n] if _norm(t)]
            out.append(_Sentence(toks, [False] * len(toks), ".", set()))
        return out

    def callon(self, who: dict) -> _Sentence:
        return _fill(self.rng, str(self.rng.choice(CALLONS)), self.topic, name=who["first"])

    def other_student(self) -> dict:
        return self.others[int(self.rng.integers(len(self.others)))]

    def speech(self):
        """Cue blocks around planned episodes, then filler turns in the remaining time."""
        rng = self.rng
        self.episodes, self.others_up = [], []
        for kind, o in self.plan:
            if kind == "bid":
                self.bid_block(o)
            elif kind == "vote":
                self.vote_block(o)
            elif kind == "coldcall":
                self.coldcall_block(o)
            elif kind == "glitch":
                self.episodes.append({"ep_type": config.EP_GLITCH, "onset_s": o,
                                      "offset_s": round(o + 0.5, 3), "outcome": None,
                                      "call_latency_s": np.nan, "q_end_s": np.nan,
                                      "has_question": False, "vote_lex": False,
                                      "n_others_up": 0})
        for s in self.segs:
            if s["seg_type"] in ("breakout", "break"):
                if s["seg_type"] == "breakout":
                    self.say(s["start_s"] + 1.0, [_fill(rng, "okay see you all in a few "
                                                        "minutes", self.topic)], self.prof,
                             "lecture")
                    self.breakout_voice(s)
                continue
            self.filler(s["start_s"] + 0.5, s["end_s"] - 0.5)
        # independent hands of other students (single tiles; never enough for a vote)
        for _ in range(int(rng.integers(1, 4))):
            a = float(rng.uniform(config.T_OPEN_S, self.L - 30))
            who = self.other_student()
            self.others_up.append({"user_id": who["user_id"], "onset_s": round(a, 2),
                                   "offset_s": round(a + rng.uniform(3, 10), 2),
                                   "kind": "bid"})

    def bid_block(self, o: float):
        rng = self.rng
        has_q = rng.random() < 0.85
        if has_q:
            q_end = o - rng.uniform(1.5, 8.0)
            sents = self.lecture(int(rng.integers(0, 3))) + [self.question()]
            a, q_end = self.say(0.0, sents, self.prof, "question", end_at=q_end)
        else:
            q_end = np.nan
            a, _ = self.say(0.0, self.student_sents(1), self.other_student(), "student",
                            end_at=o - rng.uniform(1.0, 4.0))
        if rng.random() < 0.4:   # pre-raise mic blip (no words)
            b0 = o - rng.uniform(0.6, 1.5)
            self.voice_extra.append({"user_id": self.me["user_id"], "start_s": b0,
                                     "end_s": b0 + rng.uniform(0.25, 0.45), "room": "main",
                                     "blip": True})
        called = rng.random() < 0.75
        c = o + rng.uniform(2.5, 10.0)
        target = self.me if called else self.other_student()
        self.say(0.0, [self.callon(target)], self.prof, "callon", end_at=c - 0.3)
        if called:
            ts, te = self.say(c, self.brady_sents(), self.me, "answer")
            off = c + rng.uniform(0.4, 1.6)
        else:
            ts, te = self.say(c, self.student_sents(int(rng.integers(1, 3))), target, "student")
            off = c + rng.uniform(3.0, 8.0)
        _, fe = self.say(te + rng.uniform(0.5, 1.5), [_fill(rng, str(rng.choice(FOLLOWUP)),
                                                            self.topic)], self.prof, "lecture")
        self.blocks.append((a - 0.5, fe + 0.5))
        self.episodes.append({"ep_type": config.EP_BID, "onset_s": round(o, 3),
                              "offset_s": round(off, 3),
                              "outcome": "called" if called else "withdrawn",
                              "call_latency_s": round(ts - o, 3) if called else np.nan,
                              "q_end_s": q_end, "has_question": has_q, "vote_lex": False,
                              "n_others_up": 0})

    def vote_block(self, o: float):
        rng = self.rng
        v_end = o - rng.uniform(0.8, 2.5)
        sent = _fill(rng, str(rng.choice(VOTES)), self.topic, kind="vote")
        a, _ = self.say(0.0, [sent], self.prof, "vote_q", end_at=v_end)
        down = o + rng.uniform(4.0, 8.0)
        n_up = min(len(self.others), config.vote_threshold(len(self.tile_order))
                   + int(rng.integers(1, 4)))
        for j in rng.permutation(len(self.others))[:n_up]:
            who = self.others[int(j)]
            self.others_up.append({"user_id": who["user_id"],
                                   "onset_s": round(o + rng.uniform(-0.5, 1.5), 2),
                                   "offset_s": round(down + rng.uniform(-0.5, 0.5), 2),
                                   "kind": "vote"})
        _, fe = self.say(down + 0.6, [_fill(rng, "okay thanks most of you agree about the {a}",
                                            self.topic)], self.prof, "lecture")
        self.blocks.append((a - 0.5, fe + 0.5))
        self.episodes.append({"ep_type": config.EP_VOTE, "onset_s": round(o, 3),
                              "offset_s": round(down, 3), "outcome": None,
                              "call_latency_s": np.nan, "q_end_s": v_end, "has_question": True,
                              "vote_lex": True, "n_others_up": n_up})

    def coldcall_block(self, o: float):
        rng = self.rng
        sent = _fill(rng, "{name}, what do you think about the {a}", self.topic,
                     name=self.me["first"], end="?", kind="plain")
        a, qe = self.say(o - 6.0, [sent], self.prof, "coldcall_q")
        _, te = self.say(qe + rng.uniform(1.0, 2.5), self.brady_sents(), self.me, "answer")
        self.blocks.append((a - 0.5, te + 0.5))

    def filler(self, a: float, b: float):
        rng = self.rng
        blocks = sorted(x for x in self.blocks if x[1] > a and x[0] < b)
        cuts, t = [], a
        for x, y in blocks:
            if x > t:
                cuts.append((t, x))
            t = max(t, y)
        if t < b:
            cuts.append((t, b))
        for lo, hi in cuts:
            t = lo + rng.uniform(0.3, 1.0)
            while t < hi - 3.0:
                if rng.random() < 0.55:
                    n = int(rng.integers(1, 4))
                    sents = self.lecture(n)
                    if rng.random() < 0.3:
                        sents.append(self.question(("plain", "invite", "check")))
                    _, e = self.say(t, sents, self.prof, "lecture", max_end=hi - 0.3)
                else:
                    _, e = self.say(t, self.student_sents(int(rng.integers(1, 3))),
                                    self.other_student(), "student", max_end=hi - 0.3)
                t = max(e, t + 0.5) + rng.uniform(0.4, 2.5)

    def breakout_voice(self, s: dict):
        rng = self.rng
        groups = [[self.me] + self.others[:2], self.others[2:5], self.others[5:8]]
        self.breakout_groups = groups
        for g, members in enumerate(groups):
            if not members:
                continue
            t = s["start_s"] + rng.uniform(3.0, 8.0)
            while t < s["end_s"] - 5.0:
                who = members[int(rng.integers(len(members)))]
                d = rng.uniform(3.0, 12.0)
                self.voice_extra.append({"user_id": who["user_id"], "start_s": t,
                                         "end_s": min(t + d, s["end_s"] - 1.0),
                                         "room": f"breakout{g + 1}", "blip": False})
                t += d + rng.uniform(0.5, 2.0)

    # ---- Forum voice events ------------------------------------------------------------
    def voice_table(self) -> pd.DataFrame:
        rng = self.rng
        w = pd.DataFrame(self.words).sort_values(["start_s", "end_s"]).reset_index(drop=True)
        rows = []
        w["v_idx"] = -1
        for _, turn in w.groupby("turn_id", sort=False):
            idx = turn.index.to_numpy()
            gaps = turn["start_s"].to_numpy()[1:] - turn["end_s"].to_numpy()[:-1]
            pieces = np.split(idx, np.flatnonzero(gaps > 1.0) + 1)
            for piece in pieces:
                first, last = w.loc[piece[0]], w.loc[piece[-1]]
                v = len(rows)
                rows.append({"user_id": first["user_id"],
                             "start_s": round(first["start_s"] - rng.uniform(0.05, 0.3), 3),
                             "end_s": round(last["end_s"] + rng.uniform(0.1, 0.5), 3),
                             "room": "main", "blip": False, "word_bearing": True})
                w.loc[piece, "v_idx"] = v
        for x in self.voice_extra:
            rows.append({**x, "start_s": round(x["start_s"], 3), "end_s": round(x["end_s"], 3),
                         "word_bearing": False})
        v = pd.DataFrame(rows)
        v["role"] = v["user_id"].map(lambda u: self.by_uid[u]["role"])
        v["room_main"] = v["room"] == "main"
        v["v_idx"] = np.arange(len(v))
        self.words_df = w
        return v

    # ---- raw Forum bundle --------------------------------------------------------------
    def _event(self, etype: str, audio_t: float, actor, data: dict, *, room="main",
               end_audio_t: float | None = None, private=False,
               wall_ms: int | None = None) -> dict:
        rng = self.rng
        room_id, sig = self._room(room)
        if etype == "voice":
            start_ms = self.wall(audio_t)
            end_ms = self.wall(end_audio_t)
            dur = end_ms - start_ms
            ts = end_ms if self.precise else (end_ms // 1000) * 1000
            ev = {"timestamp": ts, "start-time": _iso_s(start_ms), "end-time": _iso_s(end_ms),
                  "event-data": {"duration": dur}}
        else:
            t_ms = self.wall(audio_t) if wall_ms is None else int(wall_ms)
            ev = {"timestamp": t_ms, "start-time": _iso(t_ms), "event-data": data}
        ev.update({"id": f"ev-{self.cid}-{len(self.events)}", "event-type": etype,
                   "actor-id": None if actor is None else int(actor),
                   "klass-id": self.cid, "private": private, "room-id": room_id,
                   "room-signature": sig,
                   "created-on": _iso(ev["timestamp"])})
        if actor is not None:
            r = self.by_uid[str(actor)]
            ev["actor"] = {"id": int(actor), "first-name": r["first"], "last-name": r["last"]}
        return _snake(ev) if rng.random() < 0.5 else ev

    def _room(self, room: str) -> tuple[int, str]:
        if room == "main":
            return 5000 + self.i, config.MAIN_ROOM_SIGNATURE
        k = int(room.replace("breakout", ""))
        return 6000 + 10 * self.i + k, f"Breakout Room {k}"

    def forum_events(self, voice: pd.DataFrame):
        rng = self.rng
        self.events: list[dict] = []
        for s in self.segs:
            self.events.append(self._event(
                "timeline-segment", s["start_s"], self.prof["user_id"],
                {"timeline-segment-id": 300 + s["seg_idx"],
                 "timeline-segment-title": s["title"],
                 "timeline-section-title": s["section"]}))
        for row in voice.itertuples():
            ev = self._event("voice", row.start_s, row.user_id, {}, room=row.room,
                             end_audio_t=row.end_s)
            # Forum never logs the professor's mic (A12; 0 of 717 real classes). The event is
            # still drawn so the random stream, and so the rest of the world, is unchanged.
            if row.role != config.ROLE_PROF:
                self.events.append(ev)
        chat_rows = []
        talk = [s for s in self.segs if s["seg_type"] in TALK_TYPES]
        lines = ("+1 to that point", "i found the {a} part confusing",
                 "here is the reading on {a} I mentioned", "agree that {a} matters for {b}")
        for k in range(int(rng.integers(3, 6))):
            s = talk[int(rng.integers(len(talk)))]
            t = float(rng.uniform(s["start_s"] + 5, s["end_s"] - 5))
            who = self.me if k == 0 else self.other_student()
            a, b = rng.choice(self.topic, size=2, replace=False)
            text = str(rng.choice(lines)).format(a=a, b=b)
            if k == 1:
                text = f"{self.others[1]['first']} made a good point about {a}"  # masking trap
            private = k == 2
            chat_rows.append({"t_s": t, "user_id": who["user_id"], "text": text,
                              "system": False, "private": private, "in_gap": False})
            self.events.append(self._event("message", t, who["user_id"],
                                           {"message-text": text}, private=private))
        # system notice with no actor (L27 trap) and a bookmark-note (blocked type)
        t = float(self.episodes[0]["onset_s"]) + 0.8 if self.episodes else 200.0
        note = f"{self.others[0]['first']} {self.others[0]['last']} raised their hand"
        chat_rows.append({"t_s": t, "user_id": None, "text": note, "system": True,
                          "private": False, "in_gap": False})
        self.events.append(self._event("message", t, None, {"message-text": note}))
        self.events.append(self._event("bookmark-note", t + 1.0, self.me["user_id"],
                                       {"note": "I raised my hand here"}))
        if self.gap_wall is not None:   # a chat message while not recording (dropped)
            g_ms = (self.gap_wall[0] + self.gap_wall[1]) // 2
            who = self.other_student()
            self.events.append(self._event("message", 0.0, who["user_id"],
                                           {"message-text": "are we back"}, wall_ms=g_ms))
            chat_rows.append({"t_s": np.nan, "user_id": who["user_id"], "text": "are we back",
                              "system": False, "private": False, "in_gap": True})
        order = rng.permutation(len(self.events))
        self.events = [self.events[j] for j in order]       # Forum order is not guaranteed
        self.chat_rows = chat_rows

    def polls(self):
        rng = self.rng
        s = next(x for x in self.segs if x["seg_type"] == "poll")
        open_s = round(s["start_s"] + 4.0, 3)
        close_s = s["end_s"]
        a, b = rng.choice(self.topic, size=2, replace=False)
        prompt = f"Which matters more for this class, {a} or {b}?"
        my_submit = round(open_s + rng.uniform(8, 25), 3)
        ticks = [{"user_id": self.me["user_id"], "start_s": my_submit, "end_s": close_s}]
        for who in self.others:
            if rng.random() < 0.8:
                ticks.append({"user_id": who["user_id"],
                              "start_s": round(open_s + rng.uniform(5, 40), 3),
                              "end_s": close_s})
        pid = 40_000 + self.i
        v_open = (self.wall(open_s) - self.sessions[0][0]) / 1000.0
        hh, rem = divmod(int(v_open), 3600)
        mm, ss = divmod(rem, 60)
        self.raw_polls = [{
            "id": pid, "title": "Poll: first impressions", "prompt": prompt,
            "choices": [{"id": 1, "text": str(a)}, {"id": 2, "text": str(b)}],
            "timing-text": f"started at {hh}:{mm:02d}:{ss:02d}",
            "correct-choice-id": 2, "results": {"1": 4, "2": 6},          # traps
        }]
        self.raw_poll_responses = [{
            "id": 41_000 + self.i, "poll-id": pid, "user-id": ME_ID, "choice-id": 1,
            "answer": str(a), "submitted-at": _iso(self.wall(my_submit)),
            "score": 1, "feedback": "Nice reasoning",                      # traps
        }]
        self.poll_truth = [{"poll_id": str(pid), "open_s": open_s, "close_s": close_s,
                            "prompt": prompt, "my_answer": str(a), "my_submit_s": my_submit}]
        self.tick_rows = ticks

    def pcw_docs(self):
        rng = self.rng
        status = "verified"
        if self.i % 5 == 3:
            status = "none"
        elif self.i % 5 == 4:
            status = "unknown"
        t_uuid = _uuid("tmpl", self.seed, self.cid)
        d_uuid = _uuid("doc", self.seed, self.cid)
        n_blocks = int(rng.integers(3, 5))
        prompts, answers, kinds, blocks, doc_answers = [], [], [], [], {}
        start_ms = self.sched_ms
        for j in range(n_blocks):
            b_uuid = _uuid("block", self.seed, self.cid, j)
            a, b = rng.choice(self.topic, size=2, replace=False)
            prompt = f"Explain how {a} relates to {b} in this week's reading."
            kind = "code" if (j == n_blocks - 1 and self.sec["course_code"].startswith("CS")) \
                else ("upload" if j == n_blocks - 1 else "text")
            prompts.append(prompt)
            kinds.append(kind)
            blocks.append({"id": b_uuid, "order": j, "block_components": [
                {"type": "header", "value": _quill((f"Question {j + 1}", {"bold": True}),
                                                    ("\n", None))},
                {"type": "text", "value": _quill((prompt, None), ("\n", None))}]})
            if kind == "text":
                words = [str(w) for w in rng.choice(self.topic, size=int(rng.integers(40, 80)))]
                answer = "I argue that " + " ".join(words) + "."
                answers.append(answer)
                doc_answers[b_uuid] = {"type": "text", "value": _quill((answer, None)),
                                       "updated_at": _iso(start_ms - 3_600_000)}  # trap
            elif kind == "code":
                code = (f"def fit_{a}_model(train_data):\n    # estimate the {b} first\n"
                        f"    return train_data")
                answers.append(code)
                doc_answers[b_uuid] = {"type": "code", "value": code}
            else:
                answers.append("")
                doc_answers[b_uuid] = {"type": "upload", "files": [
                    {"name": f"{a}_notes.ipynb", "url": "https://example.invalid/signed?x=1"}]}
        updated = start_ms - int(rng.uniform(2, 48) * 3_600_000)
        if status == "unknown":
            updated = start_ms + int(rng.uniform(0.5, 30) * 3_600_000)
        template = {"id": t_uuid, "title": f"PCW for session {self.session_no}",
                    "updated_at": _iso(start_ms - 7 * 86_400_000), "blocks": blocks}
        doc = {"id": d_uuid, "template_id": t_uuid, "created_at": _iso(updated - 86_400_000),
               "updated_at": _iso(updated), "answers": doc_answers,
               "comments": [{"author": self.prof["first"], "text": "Good start"}]}  # trap
        meta = {"template_id": t_uuid, "doc_id": d_uuid, "doc_updated_at": doc["updated_at"],
                "template_updated_at": template["updated_at"], "created": False,
                "fetched_at": "2026-10-13T08:00:00Z"}
        if status == "none":
            template = doc = meta = None
            prompts, answers, kinds = [], [], []
        self.pcw = {"status": status, "template": template, "doc": doc, "meta": meta,
                    "prompts": prompts, "answers_text": answers, "answer_kinds": kinds}

    def class_json(self) -> dict:
        users = []
        for r in self.roster:
            entry = {"role": r["forum_role"],
                     "user": {"id": int(r["user_id"]), "first-name": r["first"],
                              "last-name": r["last"]},
                     "absent": not r["present"]}
            if r["present"]:
                entry["enter-class-time"] = _iso(self.sessions[0][0] + r["enter_s"] * 1000)
            users.append(entry)
        return {
            "id": self.cid, "title": f"{self.sec['course_title']}: session {self.session_no}",
            "type": "seminar", "date": self.date, "start-time": _iso(self.sched_ms),
            "teacher-user-id": int(self.prof["user_id"]),
            "actual-length-seconds": int(self.L),                         # forbidden
            "assessment": {"grade": 3},                                   # forbidden
            "section": {"id": self.sec["section_id"], "title": f"{self.sec['course_code']} A",
                        "course": {"id": 50 + self.sec["section_id"] % 50,
                                   "course-code": self.sec["course_code"],
                                   "title": self.sec["course_title"],
                                   "term": {"name": self.sec["term_name"]}}},
            "class-users": users,
            "recording-sessions": [
                {"id": 800_000 + 10 * self.i + k, "klass-id": self.cid,
                 "recording-started": _iso(a), "recording-ended": _iso(b),
                 "recording-state": "NOREC", "recording-uuid": _uuid("rec", self.cid, k)}
                for k, (a, b, _) in enumerate(self.sessions)],
        }

    def class_edit_page(self) -> dict:
        rng = self.rng
        los = []
        for k in range(int(rng.integers(2, 4))):
            a, b = rng.choice(self.topic, size=2, replace=False)
            los.append({"id": 60 + k, "learning-outcome": {
                "id": 600 + k, "name": f"#{a}{b}",
                "description": f"Analyze how {a} constrains {b} in realistic settings."},
                "score": 4, "comments": "graded"})                        # traps
        a, b = rng.choice(self.topic, size=2, replace=False)
        return {"id": self.cid,
                "preparatory-assignment": {"description": f"<p>Read the chapter on {a} and "
                                                          f"answer the PCW on {b}.</p>"},
                "assessment": {"focused-outcomes": los},
                "study-guide": {"description": f"Key terms: {a}, {b}."},
                "grades": [{"user-id": ME_ID, "score": 3}]}                # trap

    def breakouts_json(self) -> list[dict]:
        g = getattr(self, "breakout_groups", None)
        if not g:
            return []
        s = next(x for x in self.segs if x["seg_type"] == "breakout")
        room_id, sig = self._room("breakout1")
        return [{"id": 9000 + self.i, "name": sig, "room-id": room_id,
                 "users": [{"id": int(r["user_id"])} for r in g[0]],
                 "start-time": _iso(self.wall(s["start_s"])),
                 "end-time": _iso(self.wall(s["end_s"]))}]

    def calibration(self) -> dict:
        n_win = max(config.CAL_MIN_WINDOWS, int((self.L - 300) // 150) + 1)
        method = "windowed"
        if self.i == 2:                       # a global-fallback calibration (never headline)
            method, n_win = "global", 2
        wins = [[round(150 * k + 150, 2), self.cal_offset, 0.8] for k in range(n_win)]
        return {"status": "piecewise" if self.multi else "ok", "applied": True,
                "offset": self.cal_offset, "drift": 0.0,
                "pieces": [{"t_from": None, "offset": self.cal_offset, "drift": 0.0}],
                "score": 0.8, "score_before": 0.5,
                "max_residual_s": 0.0 if method == "global" else 0.3,
                "method": method, "windows": wins, "notes": []}

    # ---- assemble ----------------------------------------------------------------------
    def build(self) -> SynthClass:
        self.people()
        self.clock()
        self.segments()
        self.pcw_docs()
        self.place_episodes()
        self.speech()
        voice = self.voice_table()
        self.forum_events(voice)
        self.polls()
        words = self.words_df
        words["cluster"] = words["user_id"].map(
            {u: k for k, u in enumerate(dict.fromkeys(words["user_id"]))})
        words["cluster_conf"] = np.round(self.rng.uniform(0.6, 1.0, len(words)), 3)
        words["w_idx"] = np.arange(len(words))
        asr_words = [{"start": float(r.start_s), "end": float(r.end_s), "text": r.text,
                      "punctuated_word": r.text, "conf": float(r.conf),
                      "cluster": int(r.cluster), "cluster_conf": float(r.cluster_conf),
                      "filler": bool(r.filler)} for r in words.itertuples()]
        asr = {"provider": "synthetic", "model": "synthetic-asr-1", "language": "en",
               "duration": self.L, "diarized": True, "source_res": config.SRC_RES,
               "words": asr_words}
        segs = pd.DataFrame(self.segs)
        segs["wall_ms"] = segs["start_s"].map(self.wall)
        eps = pd.DataFrame(self.episodes).sort_values("onset_s").reset_index(drop=True)
        eps.insert(0, "ep_id", np.arange(len(eps)))
        eps["dur_s"] = eps["offset_s"] - eps["onset_s"]
        eps["seg_type"] = eps["onset_s"].map(lambda t: self.seg_at(t)["seg_type"])
        roster = pd.DataFrame(self.roster)
        tile_of = {u: k for k, u in enumerate(self.tile_order)}
        others = pd.DataFrame(self.others_up, columns=["user_id", "onset_s", "offset_s", "kind"])
        others["tile_idx"] = others["user_id"].map(tile_of).astype("Int64")
        ticks = pd.DataFrame(self.tick_rows)
        ticks["tile_idx"] = ticks["user_id"].map(tile_of).astype("Int64")
        chat = pd.DataFrame(self.chat_rows)
        chat["role"] = [None if pd.isna(u) else self.by_uid[u]["role"] for u in chat["user_id"]]
        raw = {
            "class": self.class_json(), "events": self.events, "polls": self.raw_polls,
            "poll_responses": self.raw_poll_responses,
            "outcome_assessments": [{"learning-outcome-id": 600, "score": 4,
                                     "comment": "trap: post-class grading"}],
            "focused_outcomes": [{"learning-outcome": {"id": 600, "name": "#trap"}}],
            "breakouts": self.breakouts_json(),
            "analytics": {"user-engagement": [{"user-id": ME_ID,
                                               "hand-raises": int((eps["ep_type"] == "bid").sum())}]},
            "class_edit_page": self.class_edit_page(),
            "pcw_template": self.pcw["template"], "pcw_doc": self.pcw["doc"],
            "pcw_meta": self.pcw["meta"],
        }
        raw["listing"] = {"section_id": self.sec["section_id"],
                          "course_code": self.sec["course_code"], "term": self.sec["term_name"],
                          "class": {"id": self.cid, "title": raw["class"]["title"],
                                    "date": self.date, "start-time": raw["class"]["start-time"]}}
        layout = _layout(self)
        meta = {
            "class_id": self.cid, "index": self.i, "section_id": self.sec["section_id"],
            "course_code": self.sec["course_code"], "course_title": self.sec["course_title"],
            "term_name": self.sec["term_name"], "term": self.sec["term"],
            "session_no": self.session_no, "date": self.date, "sched_start_ms": self.sched_ms,
            "rec_offset_s": self.rec_offset_s, "sessions": [list(s) for s in self.sessions],
            "gap_wall_ms": self.gap_wall, "multi_session": self.multi, "T_end_s": self.L,
            "cal_offset_s": self.cal_offset, "precise": self.precise,
            "timing_precision": "ms" if self.precise else "s", "me_id": str(ME_ID),
            "prof_id": self.prof["user_id"], "pcw_status": self.pcw["status"],
            "calibration": self.calibration(), "l_ui_s": L_UI_S,
            "main_room_id": str(5000 + self.i), "n_tiles": len(self.tile_order),
        }
        turns = pd.DataFrame(self.turns).sort_values("start_s").reset_index(drop=True)
        return SynthClass(
            class_id=self.cid, meta=meta, raw=raw, asr=asr, words=words, turns=turns,
            voice=voice, segments=segs, episodes=eps, others_hands=others,
            polls=pd.DataFrame(self.poll_truth), ticks=ticks, chat=chat, roster=roster,
            pcw=self.pcw, layout=layout)


# --------------------------------------------------------------------------- world


def world(n_classes: int = 6, seed: int = config.SEED, *, minutes: float = 30.0,
          multi_session: bool = True) -> World:
    """Build `n_classes` invented classes deterministically from `seed`.

    Class i depends only on (seed, i, minutes), so adding classes never changes earlier
    ones. With `multi_session` and n_classes >= 3, the last class has two recording
    sessions with a gap. Class index 2 reports a global-fallback calibration; indexes with
    i % 5 == 3 have no PCW and i % 5 == 4 a PCW doc edited after class start.
    """
    if n_classes < 1:
        raise ValueError("n_classes must be >= 1")
    secs = _sections(n_classes, seed)
    classes = []
    for i in range(n_classes):
        sec = secs[i % len(secs)]
        multi = multi_session and n_classes >= 3 and i == n_classes - 1
        classes.append(_Builder(i, sec, seed, minutes, multi, len(secs)).build())
    sections = [{"id": s["section_id"], "title": f"{s['course_code']} A", "_kind": "past",
                 "course": {"course-code": s["course_code"], "title": s["course_title"],
                            "term": {"name": s["term_name"]}}} for s in secs]
    schedules = {}
    for s in secs:
        mine = [c for c in classes if c.meta["section_id"] == s["section_id"]]
        rows = [{"id": c.class_id, "title": c.raw["class"]["title"], "date": c.meta["date"],
                 "start-time": c.raw["class"]["start-time"]} for c in mine]
        last = max(mine, key=lambda c: c.meta["session_no"])
        extra_date = date.fromisoformat(last.meta["date"]) + timedelta(days=3)
        rows.append({"id": 990_000 + s["section_id"], "title": "Unrecorded session",
                     "date": extra_date.isoformat(), "start-time": f"{extra_date}T16:00:00Z"})
        schedules[s["section_id"]] = {"count": len(rows), "next": None, "previous": None,
                                      "results": sorted(rows, key=lambda r: r["date"])}
    me = {"id": ME_ID, "role": "student", "first-name": ME_FIRST, "last-name": ME_LAST}
    return World(seed=seed, classes=classes, me=me, sections=sections, schedules=schedules)


# --------------------------------------------------------------------------- frames


def _layout(b: _Builder) -> dict:
    n = len(b.tile_order)
    tile_w = min(96, (FRAME_W - 40 - GUTTER * (n + 1)) // n)
    total = n * tile_w + (n + 1) * GUTTER
    x_start = (FRAME_W - total) // 2
    strip_y = (20, 28, 36)[b.sec["section_id"] % 3]
    strip_h = TILE_H + 2 * GUTTER
    x0 = [x_start + GUTTER + k * (tile_w + GUTTER) for k in range(n)]
    users = list(b.tile_order)
    labels = [f"{b.by_uid[u]['first']} {b.by_uid[u]['last'][0]}." for u in users]
    distractor = next(k for k, u in enumerate(users) if u != str(ME_ID))
    return {"frame_w": FRAME_W, "frame_h": FRAME_H, "strip_y": strip_y, "strip_h": strip_h,
            "tile_y0": strip_y + GUTTER, "tile_h": TILE_H, "tile_w": tile_w,
            "gutter": GUTTER, "x0": x0, "x1": [x + tile_w for x in x0], "users": users,
            "labels": labels, "me_tile": users.index(str(ME_ID)), "n_tiles": n,
            "label_h": LABEL_H, "icon_r": ICON_R, "icon_off": ICON_OFF,
            "distractor_tile": distractor}


def strip_layout(sc: SynthClass) -> dict:
    """Geometry of the synthetic strip: strip_y/h, tile x0/x1, users, labels, me_tile."""
    return dict(sc.layout)


def _base_frame(sc: SynthClass) -> np.ndarray:
    if "base" in sc._cache:
        return sc._cache["base"]
    lay = sc.layout
    rng = np.random.default_rng([sc.class_id, 77])
    img = np.full((FRAME_H, FRAME_W, 3), 205, np.uint8)            # slide area
    img[: lay["strip_y"]] = 60                                       # header bar
    for k in range(6):                                               # slide "text" lines
        y = lay["strip_y"] + lay["strip_h"] + 60 + 45 * k
        cv2.rectangle(img, (120, y), (120 + int(rng.integers(300, 700)), y + 14), (70, 70, 70), -1)
    y0, y1 = lay["strip_y"], lay["strip_y"] + lay["strip_h"]
    img[y0:y1] = GUTTER_BGR
    ty = lay["tile_y0"]
    for k, (x0, x1) in enumerate(zip(lay["x0"], lay["x1"], strict=True)):
        bg = tuple(int(c) for c in (52 + rng.integers(-6, 7), 46 + rng.integers(-6, 7),
                                    40 + rng.integers(-6, 7)))
        cv2.rectangle(img, (x0, ty), (x1 - 1, ty + TILE_H - 1), bg, -1)
        cx, cy = (x0 + x1) // 2, ty + TILE_H // 2 - 3
        cv2.ellipse(img, (cx, cy), (max(6, (x1 - x0) // 6), 13), 0, 0, 360, (90, 130, 190), -1)
        ly = ty + TILE_H - LABEL_H
        cv2.rectangle(img, (x0, ly), (x1 - 1, ty + TILE_H - 1), (25, 25, 25), -1)
        cv2.putText(img, lay["labels"][k], (x0 + 3, ty + TILE_H - 3), cv2.FONT_HERSHEY_PLAIN,
                    0.75, WHITE, 1, cv2.LINE_8)
        cv2.rectangle(img, (x1 - 7, ly + 3), (x1 - 4, ly + 9), WHITE, -1)     # mic icon
        if k == lay["distractor_tile"]:                                         # lime stripe
            cv2.rectangle(img, (x0 + 4, ty + 14), (x0 + 21, ty + 18), LIME_BGR, -1)
    sc._cache["base"] = img
    return img


def _draw_icon(img: np.ndarray, x0: int, y0: int, kind: str) -> None:
    cx, cy = x0 + ICON_OFF + ICON_R, y0 + ICON_OFF + ICON_R
    cv2.circle(img, (cx, cy), ICON_R, HAND_BGR if kind == "hand" else TICK_BGR, -1, cv2.LINE_8)
    if kind == "hand":                                               # palm + three fingers
        cv2.rectangle(img, (cx - 3, cy - 1), (cx + 3, cy + 4), WHITE, -1)
        for dx, top in ((-2, cy - 5), (0, cy - 6), (2, cy - 5)):
            cv2.line(img, (cx + dx, top), (cx + dx, cy - 1), WHITE, 1)
    else:                                                            # check mark
        cv2.line(img, (cx - 4, cy), (cx - 1, cy + 3), WHITE, 2)
        cv2.line(img, (cx - 1, cy + 3), (cx + 4, cy - 3), WHITE, 2)


def _intervals(sc: SynthClass) -> dict:
    if "intervals" in sc._cache:
        return sc._cache["intervals"]
    lay = sc.layout
    tile_of = {u: k for k, u in enumerate(lay["users"])}
    me = sc.episodes[["onset_s", "offset_s"]].to_numpy(float)
    oth = [(int(r.tile_idx), r.onset_s, r.offset_s) for r in sc.others_hands.itertuples()
           if not pd.isna(r.tile_idx)]
    tick = [(int(r.tile_idx), r.start_s, r.end_s) for r in sc.ticks.itertuples()
            if not pd.isna(r.tile_idx)]
    v = sc.voice[sc.voice["room_main"] & sc.voice["user_id"].isin(tile_of)]
    border = [(tile_of[r.user_id], r.start_s - sc.meta["cal_offset_s"] + L_UI_S,
               r.end_s - sc.meta["cal_offset_s"] + L_UI_S) for r in v.itertuples()]
    out = {"me": me, "others": oth, "tick": tick, "border": border}
    sc._cache["intervals"] = out
    return out


def hand_state_at(sc: SynthClass, t: float) -> dict:
    """Truth at media time t: Brady up?, tiles with hands up, ticks, bordered tiles."""
    iv = _intervals(sc)
    me_up = bool(((iv["me"][:, 0] <= t) & (t < iv["me"][:, 1])).any())
    up = {k for k, a, b in iv["others"] if a <= t < b}
    if me_up:
        up.add(sc.me_tile)
    return {"me_up": me_up, "hands": up,
            "ticks": {k for k, a, b in iv["tick"] if a <= t < b},
            "border": {k for k, a, b in iv["border"] if a <= t < b}}


def frames_for(sc: SynthClass, t_list, *, full: bool = True) -> np.ndarray:
    """BGR uint8 frames at media times `t_list`: (n, 540, 960, 3), or the strip band only
    (n, strip_h, 960, 3) with `full=False`. Deterministic for a given (class, t)."""
    lay = sc.layout
    base = _base_frame(sc)
    y0, y1 = lay["strip_y"], lay["strip_y"] + lay["strip_h"]
    ty = lay["tile_y0"]
    out = []
    for t in np.atleast_1d(np.asarray(t_list, dtype=float)):
        img = base.copy()
        st = hand_state_at(sc, float(t))
        for k in st["border"]:
            cv2.rectangle(img, (lay["x0"][k] + 1, ty + 1), (lay["x1"][k] - 2, ty + TILE_H - 2),
                          BORDER_BGR, BORDER_PX)     # drawn inside the tile, never on gutters
        for k in st["ticks"] - st["hands"]:
            _draw_icon(img, lay["x0"][k], ty, "tick")
        for k in st["hands"]:
            _draw_icon(img, lay["x0"][k], ty, "hand")
        rng = np.random.default_rng([sc.class_id, round(float(t) * 1000)])
        band = img[y0:y1].astype(np.int16) + rng.integers(-2, 3, img[y0:y1].shape,
                                                          dtype=np.int16)
        img[y0:y1] = np.clip(band, 0, 255).astype(np.uint8)
        out.append(img if full else img[y0:y1].copy())
    return np.stack(out)


# --------------------------------------------------------------------------- truth tables


def _mask_names(text: str, roster: pd.DataFrame) -> str:
    names = sorted({n for n in pd.concat([roster["first"], roster["last"]]) if n},
                   key=len, reverse=True)
    rx = re.compile(r"\b(" + "|".join(map(re.escape, names)) + r")('s)?\b")
    return rx.sub(config.NAME_TOKEN, text)


def truth_tables(sc: SynthClass) -> dict[str, pd.DataFrame]:
    """oncue-schema tables built from the truth (perfect masking, roles, detector)."""
    from . import schemas

    cid, m = sc.class_id, sc.meta
    eps_f = config.eps_f(m["timing_precision"], 0.6, m["calibration"]["max_residual_s"])
    hashed = {u: spk_hash(u) for u in sc.roster["user_id"]}
    w = sc.words
    v = sc.voice
    vstart = v.set_index("v_idx")["start_s"]
    masked = np.where(w["is_name"], config.NAME_TOKEN
                      + w["text"].str.extract(r"([^\w]*)$")[0].fillna(""), w["text"])
    # per word, no look-ahead: names and one-word CALLONs only (multi-word phrases are
    # removed from joined available text by the feature paths)
    one_word = w["word"].map(lambda x: bool(config.CALLON_RE.fullmatch(str(x)))).to_numpy()
    norm = np.where(w["is_name"].to_numpy() | (w["is_callon"].to_numpy() & one_word), "",
                    w["word"])
    role_ev = w["v_idx"].map(vstart).to_numpy(float)
    words = pd.DataFrame({
        "class_id": cid, "w_idx": w["w_idx"], "start_s": w["start_s"], "end_s": w["end_s"],
        "text_masked": masked, "text_norm": norm, "has_qmark": w["text"].str.endswith("?"),
        "conf": w["conf"], "filler": w["filler"], "spk_hash": w["user_id"].map(hashed),
        "role": w["role"], "role_event_start_s": role_ev,
        "role_avail_s": np.maximum(w["end_s"] + config.EPS_W_S, role_ev + eps_f),
        "role_fused": w["role"], "how": "forum"})
    first_end = w.groupby("v_idx")["end_s"].min()
    voice = pd.DataFrame({
        "class_id": cid, "spk_hash": v["user_id"].map(hashed), "role": v["role"],
        "start_s": v["start_s"], "end_s": v["end_s"], "room_main": v["room_main"],
        "precise": m["precise"], "word_bearing": v["word_bearing"],
        "first_word_avail_s": v["v_idx"].map(first_end) + config.EPS_W_S})
    segs = pd.DataFrame({
        "class_id": cid, "seg_idx": sc.segments["seg_idx"], "start_s": sc.segments["start_s"],
        "title_masked": [_mask_names(t, sc.roster) for t in sc.segments["title"]],
        "section_masked": sc.segments["section"], "seg_type": sc.segments["seg_type"]})
    ch = sc.chat[~sc.chat["system"] & ~sc.chat["private"] & ~sc.chat["in_gap"]]
    chat = pd.DataFrame({"class_id": cid, "t_s": ch["t_s"].astype(float),
                         "spk_hash": ch["user_id"].map(hashed), "role": ch["role"],
                         "text_masked": [_mask_names(t, sc.roster) for t in ch["text"]]})
    p = sc.polls
    polls = pd.DataFrame({"class_id": cid, "poll_id": p["poll_id"], "open_s": p["open_s"],
                          "close_s": p["close_s"], "prompt_masked": p["prompt"],
                          "my_answer_text": p["my_answer"],
                          "my_answer_avail_s": p["my_submit_s"] + config.eps_f_poll(eps_f)})
    present = sc.roster[sc.roster["present"]]
    presence = pd.DataFrame({"class_id": cid, "spk_hash": present["user_id"].map(hashed),
                             "role": present["role"],
                             "enter_s": present["enter_s"] + m["cal_offset_s"]})
    blocks = []
    tmpl = sc.pcw["template"]
    if tmpl is not None:
        for j, blk in enumerate(tmpl["blocks"]):
            ans = sc.pcw["doc"]["answers"].get(blk["id"], {})
            text = sc.pcw["answers_text"][j]
            blocks.append({"class_id": cid, "block_uuid": blk["id"], "block_order": j,
                           "prompt_text": sc.pcw["prompts"][j], "answer_text": text,
                           "answer_words": len(text.split()),
                           "has_code": ans.get("type") == "code",
                           "n_files": len(ans.get("files", []))})
    pcw_blocks = pd.DataFrame(blocks) if blocks else schemas.empty("pcw_blocks")
    e = sc.episodes
    episodes = pd.DataFrame({
        "class_id": cid, "ep_id": e["ep_id"], "onset_s": e["onset_s"],
        "onset_retimed_s": e["onset_s"], "retime_failed": False, "offset_s": e["offset_s"],
        "offset_live_s": e["offset_s"] + 0.5, "dur_s": e["dur_s"], "n_toggles": 0,
        "ep_type": e["ep_type"], "delta_n_others": e["n_others_up"].astype("Int64"),
        "vote_lex": e["vote_lex"].astype(bool), "contested": False, "left_censored": False,
        "me_hand_eligible": (e["ep_type"] != config.EP_GLITCH) & (e["dur_s"] >= 2.0),
        "outcome": e["outcome"], "call_latency_s": e["call_latency_s"].astype(float),
        "seg_type": e["seg_type"]})
    frames = _truth_hand_frames(sc)
    onsets = _truth_speech_onsets(sc)
    me_main = int(((v["user_id"] == m["me_id"]) & v["room_main"]).sum())
    cal = m["calibration"]
    qa_ok, qa_reason = config.calibration_qa(cal, 0.2, 0.6)
    if m["multi_session"] and qa_ok:
        qa_ok, qa_reason = False, "multi_session"
    sched = datetime.fromtimestamp(m["sched_start_ms"] / 1000, tz=UTC)
    classes = pd.DataFrame([{
        "class_id": cid, "section_id": m["section_id"], "course_code": m["course_code"],
        "course_area": config.course_area(m["course_code"]), "term": m["term"],
        "semester_idx": config.SEMESTER_IDX[m["term"]], "session_no": m["session_no"],
        "date": m["date"], "weekday": sched.weekday(), "hour_local": np.nan,
        "hour_utc": sched.hour + sched.minute / 60, "sched_start_utc": _iso(m["sched_start_ms"]),
        "rec_offset_s": m["rec_offset_s"], "rec_segments_json": json.dumps(m["sessions"]),
        "n_rec_sessions": len(m["sessions"]), "T_end_s": m["T_end_s"],
        "prof_hash": hashed[m["prof_id"]], "enrolled": len(sc.roster) - 1,
        "n_my_voice_main": me_main, "phase": "A", "in_vault": config.in_vault(cid),
        "src_res": config.SRC_RES, "a_start_s": 0.0, "v_start_s": 0.0, "url_lifetime_s": np.nan,
        "cal_status": cal["status"], "cal_applied": cal["applied"], "cal_method": cal["method"],
        "cal_n_windows": len(cal["windows"]), "cal_max_residual_s": cal["max_residual_s"],
        "resid_med_s": 0.2, "resid_p95_s": 0.6, "eps_F_s": eps_f,
        "timing_precision": m["timing_precision"], "av_lag_s": L_UI_S - m["cal_offset_s"],
        "av_lag_brady_s": L_UI_S - m["cal_offset_s"], "av_ok": True,
        "multi_session_ok": not m["multi_session"], "era": m["term"],
        "strip_y": float(sc.layout["strip_y"]), "strip_h": float(sc.layout["strip_h"]),
        "strip_found": True, "template_validated": True, "coverage": 1.0,
        "tile_status": "located", "down_baseline": 0.0, "flagged": False,
        "pcw_status": m["pcw_status"], "template_asof": "harvest",
        "n_bids": int((e["ep_type"] == config.EP_BID).sum()),
        "n_votes": int((e["ep_type"] == config.EP_VOTE).sum()), "n_contested": 0,
        "qa_pass": qa_ok, "qa_reason": qa_reason,
        "asr_provider": config.asr_provider(m["course_code"])}])
    out = {"classes": classes, "words": words, "voice": voice, "segments": segs, "chat": chat,
           "polls": polls, "presence": presence, "pcw_blocks": pcw_blocks,
           "hand_episodes": episodes, "hand_frames": frames, "speech_onsets": onsets}
    return {k: schemas.coerce(df.reset_index(drop=True), k) for k, df in out.items()}


def _truth_hand_frames(sc: SynthClass) -> pd.DataFrame:
    """Ideal detector output at 2 fps (live state = confirmed after K_CONFIRM samples)."""
    t = np.arange(0.0, sc.T_end_s, 1.0 / config.F_SCAN)
    iv = _intervals(sc)
    me = iv["me"]
    up = ((me[:, None, 0] <= t) & (t < me[:, None, 1])).any(axis=0)
    n_oth = np.zeros(len(t), int)
    for _, a, b in iv["others"]:
        n_oth += (a <= t) & (t < b)
    tick = np.zeros(len(t), bool)
    border = np.zeros(len(t), bool)
    for k, a, b in iv["tick"]:
        if k == sc.me_tile:
            tick |= (a <= t) & (t < b)
    for _, a, b in iv["border"]:
        border |= (a <= t) & (t < b)
    raw = np.where(up, config.STATE_UP, config.STATE_DOWN)
    live = raw.copy()
    k = config.K_CONFIRM
    for i in range(len(raw)):          # a change becomes live at the k-th agreeing sample
        prev = live[i - 1] if i else config.STATE_DOWN
        live[i] = raw[i] if i >= k - 1 and (raw[i - k + 1: i + 1] == raw[i]).all() else prev
    frac = 0.43
    return pd.DataFrame({
        "class_id": sc.class_id, "t_s": t, "n_tiles": sc.meta["n_tiles"], "tile_found": True,
        "tile_idx": sc.me_tile, "ncc": np.float32(0.95), "ncc_gap": np.float32(0.5),
        "frac_hand_adj": np.where(up, frac, 0.0).astype(np.float32), "shape_ok": up,
        "frac_tick": np.where(tick & ~up, frac, 0.0).astype(np.float32), "n_others": n_oth,
        "border_any": border, "raw_state": raw, "live_state": live})


def _truth_speech_onsets(sc: SynthClass) -> pd.DataFrame:
    v = sc.voice
    me = v[(v["user_id"] == sc.meta["me_id"]) & v["room_main"] & v["word_bearing"]]
    me = me.sort_values("start_s")
    turns = []
    for r in me.itertuples():
        if turns and r.start_s - turns[-1][1] < config.TURN_MERGE_GAP_S:
            turns[-1][1] = max(turns[-1][1], r.end_s)
        else:
            turns.append([r.start_s, r.end_s])
    bids = sc.episodes.loc[sc.episodes["ep_type"] == config.EP_BID, "onset_s"].to_numpy()
    rows, last_end = [], -np.inf
    for a, b in turns:
        silence = a - last_end
        rows.append({"class_id": sc.class_id, "onset_s": a, "turn_end_s": b,
                     "silence_before_s": silence if np.isfinite(silence) else np.nan,
                     "fresh": bool(silence >= config.FRESH_SILENCE_S),
                     "hand_preceded": bool(((bids >= a - 180) & (bids <= a)).any())})
        last_end = b
    return pd.DataFrame(rows, columns=["class_id", "onset_s", "turn_end_s", "silence_before_s",
                                       "fresh", "hand_preceded"])


# --------------------------------------------------------------------------- Scribe outputs


def output_basename(sc: SynthClass) -> str:
    """Scribe's `{course}_{date}_class{id}` transcript basename."""
    return f"{sc.meta['course_code']}_{sc.meta['date']}_class{sc.class_id}"


def scribe_transcript(sc: SynthClass, salt: str = "synthetic-salt") -> dict:
    """A real `scribe.transcript/v2` dict (privacy "ids") rendered by scribe_core.

    It deliberately contains what features must never read: `events` with a
    `hand-raise-key` entry per bid and `participation` built from the analytics trap.
    """
    from scribe_core.schema import build_transcript
    from scribe_core.turns import build_turns

    w = sc.words
    fused = [{"start": r.start_s, "end": r.end_s, "text": r.text, "conf": r.conf,
              "filler": bool(r.filler), "cluster": int(r.cluster), "how": "forum",
              "user_id": r.user_id} for r in w.itertuples()]
    voice = [{"start": r.start_s, "end": r.end_s, "user_id": r.user_id}
             for r in sc.voice[sc.voice["room_main"]].itertuples()]
    segs = sc.segments
    chapters = [{"t": r.start_s, "end": r.end_s, "title": r.title, "section": r.section}
                for r in segs.itertuples()]
    events = [{"t": round(float(r.t_s), 3), "type": "message",
               "user_id": None if pd.isna(r.user_id) else r.user_id,
               "room_id": sc.meta["main_room_id"], "text": r.text, "private": bool(r.private)}
              for r in sc.chat.itertuples() if not r.in_gap]
    for o in sc.episodes.loc[sc.episodes["ep_type"] == config.EP_BID, "onset_s"]:
        events.append({"t": round(float(o) + 0.3, 3), "type": "hand-raise-key",
                       "user_id": sc.meta["me_id"], "room_id": None, "text": None,
                       "private": False})
    asr_info = {"provider": sc.asr["provider"], "model": sc.asr["model"], "language": "en",
                "mode": "saved", "diarized": True, "duration": sc.T_end_s, "n_words": len(w),
                "keyterms": 0, "url_mode_error": None, "source_res": config.SRC_RES}
    t = build_transcript(
        bundle={"class": sc.raw["class"], "class_id": sc.class_id,
                "analytics": sc.raw["analytics"]},
        words=fused, turns=build_turns(fused), voice=voice, chapters=chapters, events=events,
        calibration=sc.meta["calibration"], mapping={}, asr_info=asr_info, privacy="ids",
        salt=salt, me_id=sc.meta["me_id"], main_room_id=sc.meta["main_room_id"],
        main_room_method="room-signature", timing_precision=sc.meta["timing_precision"])
    t["generated_at"] = "2026-10-13T00:00:00Z"
    return t


# Raw paths (after kebab()) that the synthetic world plants and loaders must DROP, as
# (source, regex on the flattened path). Everything else in `sc.raw` is whitelisted except
# harmless bookkeeping (BOOKKEEPING_PATHS), which is dropped and logged too.
TRAP_PATHS = (
    ("class", r"actual-length-seconds|assessment\..*"),
    ("class_edit_page", r"assessment\.focused-outcomes\[\]\.(score|comments)|grades\[\]\..*"),
    ("events", r"\[\]\.actor\.(first|last)-name|\[\]\.event-data\.note"),
    ("pcw_doc", r"answers\.[^.]+\.updated-at|answers\.[^.]+\.files\[\]\.url|comments\[\]\..*"),
    ("polls", r"\[\]\.(correct-choice-id|results\..*)"),
    ("poll_responses", r"\[\]\.(score|feedback)"),
    ("analytics", r".*"), ("outcome_assessments", r".*"), ("focused_outcomes", r".*"),
)
BOOKKEEPING_PATHS = (
    ("class", r"recording-sessions\[\]\.(klass-id|recording-state|recording-uuid)"),
    ("class_edit_page", r"assessment\.focused-outcomes\[\]\.id"),
    ("events", r"\[\]\.created-on"),
)


def flatten_paths(obj, prefix: str = "") -> set[str]:
    """Flattened leaf paths in RAW_PATH_WHITELIST syntax ("a.b[].c"); run kebab() first."""
    out: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            out |= flatten_paths(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, list):
        for v in obj:
            out |= flatten_paths(v, prefix + "[]")
        if not obj:
            out.add(prefix + "[]")
    else:
        out.add(prefix)
    return out


MANIFEST_COLUMNS = ["class_id", "section_id", "course_code", "term", "title", "date",
                    "recording_started", "recording_ended", "n_voice", "n_my_voice",
                    "n_events_by_type"]


def _dump(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    return path


def write_world(w: World, root: Path | str) -> dict:
    """Write the synthetic world in the real on-disk layout under `root` (a tmp dir):

    root/scribe/{me.json, sections.json, manifest.csv, schedule/{sid}.json,
                 raw/{cid}/{class,events,...,pcw_doc}.json + _complete.json}
    root/transcripts/{base}_asr.json and {base}_ids.json
    Returns {"root", "scribe", "raw", "transcripts", "manifest"} paths.
    """
    root = Path(root)
    scribe, transcripts = root / "scribe", root / "transcripts"
    _dump(scribe / "me.json", {"id": w.me["id"], "role": w.me["role"]})
    _dump(scribe / "sections.json", w.sections)
    for sid, sched in w.schedules.items():
        _dump(scribe / "schedule" / f"{sid}.json", sched)
    rows = []
    for sc in w.classes:
        cdir = scribe / "raw" / str(sc.class_id)
        for source, fname in config.RAW_SOURCE_FILES.items():
            if sc.raw.get(source) is not None:
                _dump(cdir / fname, sc.raw[source])
        _dump(cdir / "_complete.json", {"fetched_at": "2026-10-12T21:00:00Z",
                                        "status": {k: 200 for k in config.RAW_SOURCE_FILES}})
        base = output_basename(sc)
        _dump(transcripts / f"{base}_asr.json", sc.asr)
        _dump(transcripts / f"{base}_ids.json", scribe_transcript(sc))
        types = {}
        for ev in sc.raw["events"]:
            et = ev.get("event-type", ev.get("event_type"))
            types[et] = types.get(et, 0) + 1
        voice = [ev for ev in sc.raw["events"]
                 if ev.get("event-type", ev.get("event_type")) == "voice"]
        mine = sum(1 for ev in voice if str(ev.get("actor-id", ev.get("actor_id"))) == str(ME_ID))
        sess = sc.raw["class"]["recording-sessions"]
        rows.append({"class_id": sc.class_id, "section_id": sc.meta["section_id"],
                     "course_code": sc.meta["course_code"], "term": sc.meta["term_name"],
                     "title": sc.raw["class"]["title"], "date": sc.meta["date"],
                     "recording_started": sess[0]["recording-started"],
                     "recording_ended": sess[-1]["recording-ended"], "n_voice": len(voice),
                     "n_my_voice": mine,
                     "n_events_by_type": json.dumps(dict(sorted(types.items())),
                                                    separators=(",", ":"))})
    manifest = scribe / "manifest.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return {"root": root, "scribe": scribe, "raw": scribe / "raw", "transcripts": transcripts,
            "manifest": manifest}
