"""Causal primitives, physical truncation and the slow reference path (spec §4, §5.2).

The causal contract has two implementations that must agree to 1e-6:

* `oncue.features.build` is the fast path: it computes every feature for every decision
  point from the FULL tables, using `np.searchsorted` on availability times;
* `CausalView(cid, t)` here is the reference path: it physically truncates every table at
  `t` (`truncate`), clips ongoing intervals (end = NaN, `ongoing = True`), rebuilds the
  causal frames, the tail, Brady's frames, the word-bearing flags and the live hand state
  from the truncated rows, and recomputes each feature with plain per-`t` code.

`tests/test_causal.py` asserts `build(full, t) == CausalView(truncate(full, t), t)`.

Availability conventions shared by both paths (§4):

* words are ordered by (start_s, w_idx). A word's onset time is usable once
  `start + eps_w <= t`. Its text is used once `cummax(end + eps_w) <= t` and its role once
  `cummax(role_avail_s) <= t`: taking the running maximum makes "available words" a
  prefix, which is slightly conservative (a word waits for every earlier word) and keeps
  both paths exact;
* Forum rows are available at `start + eps_F` (polls at `+ eps_F,poll`); an interval's end
  is usable only once `end + eps_F <= t`;
* a Brady voice event counts (as Brady speaking) only once it overlaps an available Brady
  word (word-bearing so far; blips never count);
* CALLON phrases are deleted from *joined* available text only (frame texts, the tail,
  professor utterances, windows), never per word by looking at a later word: "go" is kept
  until "ahead" is available, then both go (`text.callon_mask` / `text.delete_callon`);
* the hand state at t uses samples <= t only: the live state (`live_codes`, k agreeing
  samples, holding the last confirmed state through NA runs <= NA_BRIDGE_S) and the live UP
  runs / raw blips derived from it (`hand_runs`) drive `hand_up`, `post_lowering`,
  `post_glitch` and every `me_hand_*` feature. Merged label-side episodes never do.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, schemas
from .text import callon_mask, delete_callon

try:  # Scribe's role edge (fuse.EDGE); the constant is all we need from it
    from scribe_core.fuse import EDGE as EDGE_S
except ImportError:  # pragma: no cover - scribe_core is installed in the project venv
    EDGE_S = 0.25

EPS_W = config.EPS_W_S
INF = np.inf

# Tables a class view may carry. `prep` (learning outcomes / study guide text) and
# `breakout_groups` (Brady's own breakout group) are small extras beyond §13.
CLASS_TABLES = ("classes", "words", "voice", "segments", "chat", "polls", "presence",
                "pcw_blocks", "prep", "breakout_groups", "hand_frames", "hand_episodes",
                "speech_onsets")
_NAME_TOKENS = frozenset({"[name]", "[me]"})


# --------------------------------------------------------------------------- table access


def load_tables(names=CLASS_TABLES, class_ids=None, tables_dir: Path | str | None = None
                ) -> dict[str, pd.DataFrame]:
    """Read `data/tables/{name}.parquet` for the names that exist, filtered to `class_ids`."""
    root = Path(tables_dir) if tables_dir is not None else config.TABLES_DIR
    out = {}
    wanted = None if class_ids is None else {int(c) for c in class_ids}
    for name in names:
        path = root / f"{name}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if wanted is not None and "class_id" in df.columns:
            df = df[df["class_id"].isin(wanted)].reset_index(drop=True)
        out[name] = df
    return out


def class_tables(tables: dict, cid) -> dict[str, pd.DataFrame]:
    """Every table restricted to one class (tables without `class_id` pass through)."""
    out = {}
    for name, df in (tables or {}).items():
        if isinstance(df, pd.DataFrame) and "class_id" in df.columns:
            mine = (df["class_id"] == int(cid)).to_numpy()
            out[name] = df if mine.all() else df[mine].reset_index(drop=True)
        else:
            out[name] = df
    return out


def table(tables: dict, name: str) -> pd.DataFrame:
    """`tables[name]`, or the schema's empty table (or an empty frame) when absent."""
    df = tables.get(name) if tables else None
    if isinstance(df, pd.DataFrame):
        return df
    return schemas.empty(name) if name in schemas.TABLES else pd.DataFrame()


def strs(series, fill: str = "") -> np.ndarray:
    """A string column as a numpy object array with NA replaced by `fill`."""
    if series is None:
        return np.array([], dtype=object)
    arr = pd.Series(series).to_numpy(dtype=object)
    if not len(arr):
        return arr
    out = np.where(pd.isna(arr), fill, arr)
    return out.astype(str).astype(object)


def floats(series) -> np.ndarray:
    """A numeric column as a float64 numpy array (NA -> NaN; unparseable -> NaN)."""
    s = series if isinstance(series, pd.Series) else pd.Series(series)
    try:
        return s.to_numpy(dtype=float, na_value=np.nan)
    except (TypeError, ValueError):
        return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float, na_value=np.nan)


def bools(series, fill: bool = False) -> np.ndarray:
    """A boolean column as a numpy bool array with NA replaced by `fill`."""
    arr = pd.Series(series).to_numpy(dtype=object)
    if not len(arr):
        return np.zeros(0, dtype=bool)
    return np.where(pd.isna(arr), fill, arr).astype(bool)


# --------------------------------------------------------------------------- class info


@dataclass(frozen=True)
class ClassInfo:
    """Per-class constants both paths need (from the `classes` row)."""

    cid: int
    T_end: float
    eps_f: float
    eps_p: float
    rec_offset_s: float
    restarts: tuple[float, ...]
    row: dict = field(default_factory=dict, compare=False, hash=False)


def class_info(classes: pd.DataFrame, cid) -> ClassInfo:
    rows = classes[classes["class_id"] == int(cid)] if len(classes) else classes
    row = rows.iloc[0].to_dict() if len(rows) else {"class_id": int(cid)}

    def num(key, default=np.nan):
        v = row.get(key, default)
        try:
            v = float(v)
        except (TypeError, ValueError):
            return default
        return default if math.isnan(v) else v

    eps = num("eps_F_s")
    if math.isnan(eps):
        eps = config.eps_f(row.get("timing_precision"), num("resid_p95_s", 0.0),
                           num("cal_max_residual_s", 0.0))
    restarts = ()
    raw = row.get("rec_segments_json")
    if isinstance(raw, str) and raw:
        try:
            segs = json.loads(raw)
            restarts = tuple(float(s[2]) for s in segs[1:] if s is not None and len(s) > 2)
        except (ValueError, TypeError):
            restarts = ()
    return ClassInfo(cid=int(cid), T_end=num("T_end_s", INF), eps_f=float(eps),
                     eps_p=config.eps_f_poll(eps), rec_offset_s=num("rec_offset_s", 0.0),
                     restarts=restarts, row=row)


# --------------------------------------------------------------------------- words


def clean_token(text: str) -> str:
    """Defensive feature-text cleaning of one word's `text_norm` (§5.3 step 3): drops
    `[name]`/`[me]` tokens and single-word CALLON phrases that slipped through."""
    tok = (text or "").strip().lower()
    if not tok or tok in _NAME_TOKENS or config.CALLON_RE.fullmatch(tok):
        return ""
    return tok


def join_tokens(tokens) -> str:
    """Feature text of a run of already-available word tokens: joined, with every CALLON
    phrase removed (§5.3 step 3). Both paths join exactly the same available words, so a
    phrase is only removed once all of its words are available."""
    return delete_callon(" ".join(x for x in tokens if x))


def keep_mask(tokens) -> np.ndarray:
    """Per available word token: False where it is part of a CALLON phrase in the joined
    sequence (window counts, the BM25 query). Empty tokens are False."""
    toks = list(tokens)
    out = np.array([bool(x) for x in toks], dtype=bool)
    idx = np.flatnonzero(out)
    if len(idx):
        out[idx[callon_mask([toks[i] for i in idx])]] = False
    return out


@dataclass
class Words:
    """Words in causal order (start_s, w_idx) as arrays. NaN `end`/`role_avail` mean
    "not yet known" (truncated data)."""

    start: np.ndarray
    end: np.ndarray
    tok: np.ndarray          # cleaned feature token ("" = deleted or unknown)
    role: np.ndarray         # "prof" | "me" | "student" | "unk" ("" = role not yet known)
    spk: np.ndarray          # speaker key ("" = no Forum speaker)
    role_avail: np.ndarray   # per-word role availability (NaN = unknown)
    filler: np.ndarray
    has_qmark: np.ndarray
    text_av: np.ndarray      # cummax(end + eps_w): text usable once <= t
    role_av: np.ndarray      # cummax(role_avail): role usable once <= t

    def __len__(self) -> int:
        return len(self.start)


def order_words(words: pd.DataFrame) -> pd.DataFrame:
    keys = ["start_s", "w_idx"] if "w_idx" in words.columns else ["start_s"]
    return words.sort_values(keys, kind="stable").reset_index(drop=True)


def prep_words(words: pd.DataFrame, eps_w: float = EPS_W) -> Words:
    """Arrays for `words` (full or truncated). Missing role availability on a known word
    defaults to `end + eps_w` (a word with no Forum speaker)."""
    w = order_words(words) if len(words) else words
    start = floats(w["start_s"]) if len(w) else np.zeros(0)
    end = floats(w["end_s"]) if len(w) else np.zeros(0)
    n = len(start)
    if n == 0:
        z = np.zeros(0)
        e = np.array([], dtype=object)
        return Words(z, z, e, e, e, z, np.zeros(0, bool), np.zeros(0, bool), z, z)
    # Truncated tables carry `role_known`; in full data every role is known (NA -> "unk").
    known_role = bools(w["role_known"], True) if "role_known" in w else np.ones(n, bool)
    role = strs(w["role"], config.ROLE_UNK) if "role" in w else np.full(n, config.ROLE_UNK,
                                                                         object)
    role = np.where(known_role, role, "")
    ra = floats(w["role_avail_s"]) if "role_avail_s" in w else np.full(n, np.nan)
    ra = np.where(np.isnan(ra) & known_role, end + eps_w, ra)
    ra = np.where(known_role, ra, np.nan)
    tok = np.array([clean_token(x) for x in strs(w["text_norm"])], dtype=object)
    tok = np.where(np.isnan(end), "", tok)
    text_av = np.maximum.accumulate(np.where(np.isnan(end), INF, end + eps_w))
    role_av = np.maximum.accumulate(np.where(np.isnan(ra), INF, ra))
    return Words(start=start, end=end, tok=tok, role=role,
                 spk=strs(w["spk_hash"]) if "spk_hash" in w else np.full(n, "", object),
                 role_avail=ra,
                 filler=bools(w["filler"]) if "filler" in w else np.zeros(n, bool),
                 has_qmark=bools(w["has_qmark"]) if "has_qmark" in w else np.zeros(n, bool),
                 text_av=text_av, role_av=role_av)


# --------------------------------------------------------------------------- frames (§5.2)


@dataclass
class Frames:
    """Closed causal frames (word spans i0..i1 inclusive) and the tail start index."""

    i0: np.ndarray
    i1: np.ndarray
    close: np.ndarray
    start: np.ndarray
    end: np.ndarray
    role: np.ndarray
    spk: np.ndarray
    text: np.ndarray
    n_words: np.ndarray
    tail_start: int

    def __len__(self) -> int:
        return len(self.close)


def frame_spans(w: Words, t: float = INF, eps_w: float = EPS_W
                ) -> tuple[list[tuple[int, int, float]], int]:
    """Cut words into frames; return (closed spans, index of the first unclosed word).

    A frame is a run of words with the same Forum speaker, cut after word j when
      * pause: no word starts in (end_j, end_j + 0.4]  -> observable at end_j + 0.4 + eps_w;
      * speaker change: the next word has another speaker -> observable at its role time;
      * length: end_j - frame start > 5 s              -> observable at end_j + eps_w.
    The close time is the earliest observable cut, no earlier than the role availability
    of every word in the frame, and no earlier than the previous frame's close (so closed
    frames always form a prefix). With truncated data (t finite) a frame whose cut is not
    yet decidable is not closed, and nothing after it is.
    """
    n = len(w)
    spans: list[tuple[int, int, float]] = []
    prev_close = -INF
    i = 0
    pause, fmax = config.FRAME_PAUSE_S, config.FRAME_MAX_S
    while i < n:
        fstart = w.start[i]
        j = i
        while True:
            if j >= n or not (w.role_av[j] <= t):
                return spans, i
            e = w.end[j]
            cand, unknown = INF, False
            if e - fstart > fmax:
                cand = e + eps_w
            if j + 1 < n:
                if w.start[j + 1] > e + pause:
                    cand = min(cand, e + pause + eps_w)
                if w.role_av[j + 1] <= t:
                    if w.spk[j + 1] != w.spk[j]:
                        cand = min(cand, w.role_av[j + 1])
                else:
                    unknown = True
            else:
                cp = e + pause + eps_w
                if cp <= t:
                    cand = min(cand, cp)
                else:
                    unknown = True
            if cand == INF:
                if unknown:
                    return spans, i
                j += 1
                continue
            close = max(cand, w.role_av[j], prev_close)
            if close > t:
                return spans, i
            spans.append((i, j, close))
            prev_close = close
            i = j + 1
            break
    return spans, i


def build_frames(w: Words, t: float = INF, eps_w: float = EPS_W) -> Frames:
    """Closed frames at `t` (all frames for full data with t = inf) plus the tail start."""
    spans, tail_start = frame_spans(w, t, eps_w)
    k = len(spans)
    i0 = np.array([s[0] for s in spans], dtype=int)
    i1 = np.array([s[1] for s in spans], dtype=int)
    text, nw = [], []
    for a, b in zip(i0, i1, strict=True):
        joined = join_tokens(w.tok[a:b + 1])           # a closed frame: all words available
        text.append(joined)
        nw.append(len(joined.split()))
    return Frames(
        i0=i0, i1=i1, close=np.array([s[2] for s in spans], dtype=float),
        start=w.start[i0] if k else np.zeros(0), end=w.end[i1] if k else np.zeros(0),
        role=w.role[i0] if k else np.array([], dtype=object),
        spk=w.spk[i0] if k else np.array([], dtype=object),
        text=np.array(text, dtype=object), n_words=np.array(nw, dtype=int),
        tail_start=tail_start)


def frames_table(cid, words: pd.DataFrame, eps_w: float = EPS_W) -> pd.DataFrame:
    """The `frames` table (schema §13) built by this module's frame rule (full data)."""
    w = prep_words(words, eps_w)
    fr = build_frames(w, INF, eps_w)
    if len(fr) == 0:
        return schemas.empty("frames")
    df = pd.DataFrame({"class_id": int(cid), "f_idx": np.arange(len(fr)), "start_s": fr.start,
                       "end_s": fr.end, "close_s": fr.close, "role": fr.role, "spk_hash": fr.spk,
                       "text_norm": fr.text, "n_words": fr.n_words, "emb_row": pd.NA})
    df["spk_hash"] = df["spk_hash"].replace("", pd.NA)
    return schemas.coerce(df, "frames")


# --------------------------------------------------------------------------- hand state


DOWN_C, UP_C, NA_C = 0, 1, 2                      # integer state codes (hands.py uses them)
STATE_NAMES = np.array([config.STATE_DOWN, config.STATE_UP, config.STATE_NA], dtype=object)
_CODE_OF = {config.STATE_DOWN: DOWN_C, config.STATE_UP: UP_C, config.STATE_NA: NA_C}
NA_HOLD = round(config.NA_BRIDGE_S * config.F_SCAN)        # NA samples the live state holds


def state_codes(raw) -> np.ndarray:
    """UP / DOWN / NA strings (anything else, or missing, = NA) as integer codes."""
    r = strs(raw, config.STATE_NA)
    return np.array([_CODE_OF.get(x, NA_C) for x in r], dtype=np.int64)


def live_codes(codes, k: int = config.K_CONFIRM, hold: int = NA_HOLD) -> np.ndarray:
    """Causal confirmed (live) state per sample from integer state codes (§2.4).

    A state becomes live at the k-th consecutive agreeing sample, using samples <= that
    sample only; before any confirmation the live state is NA. Through an NA run of at most
    `hold` samples (NA_BRIDGE_S at the scan rate: a poll tick, a tracking flicker) the last
    confirmed non-NA state is held, also while the next state is still being confirmed; a
    longer NA run makes the live state NA from its (hold + 1)-th sample on. Still causal: the
    length of the NA run so far is known at every sample.
    """
    c = np.asarray(codes, dtype=np.int64)
    out = np.empty(len(c), dtype=np.int64)
    state, last_good, hold_ok = NA_C, -1, False
    prev, run = -1, 0
    for i, x in enumerate(c.tolist()):
        run = run + 1 if x == prev else 1
        prev = x
        if run >= k and state != x:
            state = x
            if x != NA_C:
                last_good, hold_ok = x, True
        if state != NA_C:
            out[i] = state
            continue
        if x == NA_C and run > hold:
            hold_ok = False
        out[i] = last_good if (hold_ok and last_good >= 0) else NA_C
    return out


def live_states(raw, k: int = config.K_CONFIRM, initial: str = config.STATE_NA,
                hold: int = NA_HOLD) -> np.ndarray:
    """Confirmed (live) hand state per sample as UP / DOWN / NA strings (see `live_codes`).
    `initial` replaces NA before the first confirmation."""
    if raw is None or not len(raw):
        return np.array([], dtype=object)
    live = STATE_NAMES[live_codes(state_codes(raw), k, hold)]
    first = np.flatnonzero(live != config.STATE_NA)
    if initial != config.STATE_NA:
        live[: first[0] if len(first) else len(live)] = initial
    return live


def live_raw_column(hand_frames: pd.DataFrame) -> str:
    """The raw-state column the live state is built from: `raw_state_live` (hysteresis on a
    causal DOWN baseline, hands.track) when present, else `raw_state`."""
    if "raw_state_live" in hand_frames and hand_frames["raw_state_live"].notna().any():
        return "raw_state_live"
    return "raw_state"


@dataclass
class HandRuns:
    """Brady's live UP runs and raw UP blips, all decided from samples <= their times.

    Per live UP run: `up_s` (first raw UP sample), `live_up_s` (confirmation), `down_s`
    (first sample after its last raw UP sample), `exit_s` (live exit from UP, inf while up),
    `exit_to` (DOWN | NA | ""), `n_shape` (shape passes on its UP samples), `eligible`
    (§2.4 `me_hand_*`: raw duration down_s - up_s >= 2 s and >= 2 shape passes; known at
    exit). `blip_end_s`: for every UP run shorter than k samples that lies outside the live
    UP runs and ends in a DOWN sample, the time of that DOWN sample (`post_glitch`).
    """

    up_s: np.ndarray
    live_up_s: np.ndarray
    down_s: np.ndarray
    exit_s: np.ndarray
    exit_to: np.ndarray
    n_shape: np.ndarray
    eligible: np.ndarray
    blip_end_s: np.ndarray

    def __len__(self) -> int:
        return len(self.up_s)


def hand_runs(hand_frames: pd.DataFrame, *, k: int = config.K_CONFIRM, hold: int = NA_HOLD,
              min_s: float = config.ME_HAND_MIN_S,
              min_shape: int = config.ME_HAND_MIN_SHAPE_SAMPLES) -> HandRuns:
    """Live UP runs and blips of one class's `hand_frames` (full or truncated at t: every
    run that has exited by t is identical in both, so the fast and reference paths agree)."""
    z = np.zeros(0)
    if hand_frames is None or not len(hand_frames):
        return HandRuns(z, z, z, z, np.array([], dtype=object), np.zeros(0, int),
                        np.zeros(0, bool), z)
    hf = hand_frames.sort_values("t_s", kind="stable")
    t = floats(hf["t_s"])
    raw = state_codes(hf[live_raw_column(hf)])
    shape = bools(hf["shape_ok"]) if "shape_ok" in hf else (raw == UP_C)
    live = live_codes(raw, k, hold)
    n = len(t)
    is_up = live == UP_C
    d = np.diff(np.r_[0, is_up.astype(np.int8), 0])
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    rows = []
    for i0, i1 in zip(starts, ends, strict=True):
        j0 = i0
        while j0 > 0 and raw[j0 - 1] == UP_C:
            j0 -= 1
        ups = np.flatnonzero(raw[i0:i1] == UP_C) + i0
        jl = int(ups[-1]) if len(ups) else i0
        up_idx = np.arange(j0, jl + 1)
        up_idx = up_idx[raw[up_idx] == UP_C]
        down = float(t[jl + 1]) if jl + 1 < n else INF
        ex = float(t[i1]) if i1 < n else INF
        to = str(STATE_NAMES[live[i1]]) if i1 < n else ""
        n_sh = int(shape[up_idx].sum())
        elig = bool(np.isfinite(ex) and down - float(t[j0]) >= min_s - 1e-9 and n_sh >= min_shape)
        rows.append((float(t[j0]), float(t[i0]), down, ex, to, n_sh, elig))
    blips = []
    rr = np.flatnonzero(np.diff(np.r_[-1, raw]) != 0)            # raw run starts
    re_ = np.r_[rr[1:], n]
    for a, b in zip(rr, re_, strict=True):
        if raw[a] == UP_C and b - a < k and b < n and raw[b] == DOWN_C and \
                not is_up[a:b + 1].any():
            blips.append(float(t[b]))
    cols = list(zip(*rows)) if rows else [[]] * 7
    return HandRuns(up_s=np.asarray(cols[0], float), live_up_s=np.asarray(cols[1], float),
                    down_s=np.asarray(cols[2], float), exit_s=np.asarray(cols[3], float),
                    exit_to=np.asarray(cols[4], dtype=object),
                    n_shape=np.asarray(cols[5], int), eligible=np.asarray(cols[6], bool),
                    blip_end_s=np.asarray(blips, float))


def run_outcomes(runs: HandRuns, voice: pd.DataFrame, words: pd.DataFrame, eps_f: float
                 ) -> tuple[np.ndarray, np.ndarray]:
    """(outcome, available-at) per live UP run, decided causally at
    A = down_s + CALLED_AFTER_OFFSET_S + eps_F (§2.5 outcome, §4 `me_hand_last_outcome`):
    "called" when a Brady main-room voice event starting in [up_s - dt/2, down_s + 5 s] is
    word-bearing as of A (overlaps a Brady word whose role is known by A); else "open" when
    the run ended in NA (tile lost while up), else "withdrawn". Independent of vote typing,
    and the same from full tables or from tables truncated at any t >= A."""
    n = len(runs)
    out = np.full(n, "", dtype=object)
    avail = np.full(n, INF)
    if not n:
        return out, avail
    avail = np.where(np.isfinite(runs.down_s),
                     np.maximum(runs.down_s + config.CALLED_AFTER_OFFSET_S + eps_f, runs.exit_s),
                     INF)
    v = voice if voice is not None else pd.DataFrame()
    if len(v):
        v = v[np.isfinite(floats(v["start_s"]))].reset_index(drop=True)
    called = np.zeros(n, bool)
    if len(v):
        t1, t2 = brady_wb_times(v, words if words is not None else pd.DataFrame())
        me = (strs(v["role"]) == config.ROLE_ME) & bools(v["room_main"])
        vs, ve = floats(v["start_s"]), floats(v["end_s"])
        lo = runs.up_s - 0.5 / config.F_SCAN
        hi = runs.down_s + config.CALLED_AFTER_OFFSET_S
        for j in range(n):
            A = avail[j]
            if not np.isfinite(A):
                continue
            inwin = me & (vs >= lo[j]) & (vs <= hi[j]) & (vs + eps_f <= A)
            wb = (A >= t1) | ((A >= t2) & ~(ve + eps_f <= A))
            called[j] = bool((inwin & wb).any())
    out = np.where(called, "called", np.where(runs.exit_to == config.STATE_NA, "open",
                                              "withdrawn")).astype(object)
    out[~np.isfinite(avail)] = ""
    return out, avail


def me_hand_values(runs: HandRuns, outcome: np.ndarray, outcome_av: np.ndarray, t
                   ) -> dict[str, np.ndarray]:
    """`me_hand_*` at decision times t from eligible live runs that have exited by t
    (§4.1 A16): count, seconds since the last one went down / up, and the outcome of the
    latest run whose outcome is available (or "none")."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    n = len(t)
    out = {"me_hand_n_so_far": np.zeros(n), "me_hand_since_down_s": np.full(n, np.nan),
           "me_hand_since_up_s": np.full(n, np.nan),
           "me_hand_last_outcome": np.full(n, "none", object)}
    if not len(runs):
        return out
    tt = t[:, None]
    done = runs.eligible[None, :] & (runs.exit_s[None, :] <= tt)
    out["me_hand_n_so_far"] = done.sum(axis=1).astype(float)
    last = np.argmax(np.where(done, runs.exit_s[None, :], -INF), axis=1)
    has = done.any(axis=1)
    out["me_hand_since_down_s"] = np.where(has, t - runs.down_s[last], np.nan)
    out["me_hand_since_up_s"] = np.where(has, t - runs.up_s[last], np.nan)
    known = runs.eligible[None, :] & (outcome_av[None, :] <= tt) & (outcome != "")[None, :]
    lo = np.argmax(np.where(known, runs.up_s[None, :], -INF), axis=1)
    out["me_hand_last_outcome"] = np.where(known.any(axis=1), outcome[lo], "none")
    return out


# --------------------------------------------------------------------------- Brady events


def brady_wb_times(voice: pd.DataFrame, words: pd.DataFrame, edge: float = EDGE_S
                   ) -> tuple[np.ndarray, np.ndarray]:
    """(T1, T2) per voice row for Brady's word-bearing-so-far rule (full data).

    A Brady event e counts as word-bearing at t when some Brady word with role known by t
    overlaps it: truly (start - edge < end_e and end + edge > start_e; from T1 = the
    earliest such role time), or, while e's end is still unknown, open-endedly
    (end + edge > start_e; from T2, until end_e + eps_F). Rows that are not Brady's
    main-room events get T1 = T2 = inf.
    """
    n = len(voice)
    t1, t2 = np.full(n, INF), np.full(n, INF)
    if n == 0 or len(words) == 0:
        return t1, t2
    role_v = strs(voice["role"])
    main = bools(voice["room_main"])
    vs, ve = floats(voice["start_s"]), floats(voice["end_s"])
    wr = strs(words["role"])
    me = wr == config.ROLE_ME
    ws, we = floats(words["start_s"])[me], floats(words["end_s"])[me]
    ra = floats(words["role_avail_s"])[me]
    ra = np.where(np.isnan(ra), we + EPS_W, ra)
    for i in np.flatnonzero((role_v == config.ROLE_ME) & main):
        opened = we + edge > vs[i]
        true = opened & (ws - edge < ve[i])
        if true.any():
            t1[i] = ra[true].min()
        other = opened & ~true
        if other.any():
            t2[i] = ra[other].min()
    return t1, t2


# --------------------------------------------------------------------------- truncation


def _blank(df: pd.DataFrame, mask: np.ndarray, cols) -> None:
    """Set `cols` to missing where `mask` (NaN for floats, <NA> otherwise)."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return
    m = pd.Series(mask, index=df.index)
    for c in cols:
        if c not in df.columns:
            continue
        s = df[c]
        if s.dtype == bool:
            s = s.astype("boolean")
        df[c] = s.mask(m)


def truncate(tables: dict, t: float, *, info: ClassInfo | None = None
             ) -> dict[str, pd.DataFrame]:
    """Physically truncate one class's tables at `t` (§4.4).

    Rows not yet available are dropped; fields not yet known are blanked; intervals still
    open at t keep their start, lose their end (NaN) and get `ongoing = True`. Columns that
    only exist with hindsight (`word_bearing`, `first_word_avail_s`, `live_state`, `role_fused`,
    `how`, vote typing, `speech_onsets`) are removed or blanked so the reference path must
    rebuild them from the truncated rows.
    """
    t = float(t)
    classes = table(tables, "classes")
    info = info or class_info(classes, int(classes["class_id"].iloc[0]) if len(classes)
                              else 0)
    ef, ep = info.eps_f, info.eps_p
    out: dict[str, pd.DataFrame] = {"classes": classes.copy()}

    w = table(tables, "words").copy()
    if len(w):
        w = w[floats(w["start_s"]) + EPS_W <= t].copy()
        end = floats(w["end_s"])
        text_unknown = ~(end + EPS_W <= t)
        ra = floats(w["role_avail_s"])
        ra = np.where(np.isnan(ra), end + EPS_W, ra)
        role_unknown = ~(ra <= t)
        _blank(w, text_unknown, ["end_s", "text_masked", "text_norm", "has_qmark", "conf",
                                 "filler"])
        _blank(w, role_unknown, ["role", "spk_hash", "role_event_start_s", "role_avail_s"])
        _blank(w, np.ones(len(w), bool), ["role_fused", "how"])
        w["ongoing"] = text_unknown
        w["role_known"] = ~role_unknown
    out["words"] = w.reset_index(drop=True)

    v = table(tables, "voice").copy()
    if len(v):
        sa = floats(v["start_avail_s"]) if "start_avail_s" in v else np.full(len(v), np.nan)
        v = v[np.where(np.isnan(sa), floats(v["start_s"]) + ef, sa) <= t].copy()
        ea = floats(v["end_avail_s"]) if "end_avail_s" in v else np.full(len(v), np.nan)
        open_ = ~(np.where(np.isnan(ea), floats(v["end_s"]) + ef, ea) <= t)
        _blank(v, open_, ["end_s", "end_avail_s"])
        _blank(v, np.ones(len(v), bool), ["word_bearing", "first_word_avail_s"])
        v["ongoing"] = open_
    out["voice"] = v.reset_index(drop=True)

    s = table(tables, "segments")
    out["segments"] = (s[floats(s["start_s"]) + ef <= t].reset_index(drop=True) if len(s)
                       else s.copy())
    c = table(tables, "chat")
    out["chat"] = c[floats(c["t_s"]) + ef <= t].reset_index(drop=True) if len(c) else c.copy()
    p = table(tables, "polls").copy()
    if len(p):
        p = p[floats(p["open_s"]) + ep <= t].copy()
        _blank(p, ~(floats(p["close_s"]) + ep <= t), ["close_s"])
        _blank(p, ~(floats(p["my_answer_avail_s"]) <= t), ["my_answer_text", "my_answer_avail_s"])
    out["polls"] = p.reset_index(drop=True)
    pr = table(tables, "presence")
    out["presence"] = (pr[floats(pr["enter_s"]) + ef <= t].reset_index(drop=True) if len(pr)
                       else pr.copy())
    for name in ("pcw_blocks", "prep"):              # pre-class material
        out[name] = table(tables, name).copy()

    groups = table(tables, "breakout_groups")
    ended = breakout_ended_at(out["segments"])
    out["breakout_groups"] = groups.copy() if (len(groups) and ended is not None) \
        else groups.iloc[0:0].copy()

    hf = table(tables, "hand_frames").copy()
    if len(hf):
        hf = hf[floats(hf["t_s"]) <= t].copy()
        _blank(hf, np.ones(len(hf), bool), ["live_state"])
    out["hand_frames"] = hf.reset_index(drop=True)
    out["hand_episodes"] = _truncate_episodes(table(tables, "hand_episodes"), t)
    out["speech_onsets"] = table(tables, "speech_onsets").iloc[0:0].copy()
    return out


def _truncate_episodes(eps: pd.DataFrame, t: float) -> pd.DataFrame:
    if not len(eps):
        return eps.copy()
    e = eps.copy()
    onset, off_live = floats(e["onset_s"]), floats(e["offset_live_s"])
    keep = (onset <= t) | (np.isnan(onset) & (off_live <= t))
    e = e[keep].copy()
    onset, off_live, off = (floats(e["onset_s"]), floats(e["offset_live_s"]),
                            floats(e["offset_s"]))
    done = off_live <= t
    # an episode that never went live-DOWN (it ended in NA or at the recording end) has
    # offset_s = end of observation, known from that moment on
    obs_end = np.isnan(off_live) & (off <= t)
    glitch = strs(e["ep_type"]) == config.EP_GLITCH
    typed = done & (glitch | (onset + config.VOTE_DN_WINDOW_S[1] <= t))
    outcome_known = done & (np.fmax(off + config.ME_HAND_OUTCOME_DELAY_S,
                                    onset + config.VOTE_DN_WINDOW_S[1]) <= t)
    _blank(e, ~done & ~obs_end, ["offset_s"])
    _blank(e, ~done, ["offset_live_s", "dur_s", "n_toggles", "me_hand_eligible"])
    _blank(e, ~typed, ["ep_type"])
    _blank(e, ~outcome_known, ["outcome", "call_latency_s"])
    _blank(e, np.ones(len(e), bool), ["onset_retimed_s", "retime_failed", "delta_n_others",
                                      "vote_lex", "contested"])
    e["ongoing"] = ~done
    return e.reset_index(drop=True)


def breakout_ended_at(segments: pd.DataFrame) -> float | None:
    """Start time of the first segment that follows a breakout segment (= the time the
    first breakout ended and its membership is known, A7), or None."""
    if not len(segments):
        return None
    s = segments.sort_values("start_s", kind="stable")
    st, typ = floats(s["start_s"]), strs(s["seg_type"])
    for i in range(len(s) - 1):
        if typ[i] == "breakout" and typ[i + 1] != "breakout":
            return float(st[i + 1])
    return None


# --------------------------------------------------------------------------- reference path


class CausalView:
    """Everything observable at time `t` in class `cid`, rebuilt from truncated tables.

    Attributes: `t`, `info`, `tables` (truncated), `words` (Words), `frames` (closed),
    `tail` (word indexes of the tail), `me_frames` (indexes of Brady's closed frames),
    `live_state` (str), `brady_wb` (bool per truncated voice row).
    """

    def __init__(self, cid, t: float, tables: dict | None = None, **feature_kwargs):
        full = class_tables(tables if tables is not None else load_tables(class_ids=[cid]),
                            cid)
        self.cid = int(cid)
        self.t = float(t)
        self.info = class_info(table(full, "classes"), cid)
        self.tables = truncate(full, self.t, info=self.info)
        self.feature_kwargs = feature_kwargs
        self.words = prep_words(self.tables["words"])
        self.frames = build_frames(self.words, self.t)
        k_text = int(np.sum(self.words.text_av <= self.t))
        self.tail = np.arange(self.frames.tail_start, max(self.frames.tail_start, k_text))
        self.me_frames = np.flatnonzero(self.frames.role == config.ROLE_ME)
        hf = self.tables["hand_frames"]
        if len(hf):
            hf = hf.sort_values("t_s", kind="stable")
            live = live_states(hf[live_raw_column(hf)])
        else:
            live = np.array([], dtype=object)
        self.live_state = str(live[-1]) if len(live) else config.STATE_NA
        self.hand_runs = hand_runs(hf)                 # rebuilt from samples <= t only
        self.brady_wb = self._brady_word_bearing()

    # ---- rebuilt flags -------------------------------------------------------------------
    def _brady_word_bearing(self) -> np.ndarray:
        """Word-bearing so far, per truncated voice row (Brady main-room events only)."""
        v = self.tables["voice"]
        out = np.zeros(len(v), dtype=bool)
        if not len(v):
            return out
        w = self.tables["words"]
        known = strs(w["role"]) == config.ROLE_ME if len(w) else np.zeros(0, bool)
        ws = floats(w["start_s"])[known] if len(w) else np.zeros(0)
        we = floats(w["end_s"])[known] if len(w) else np.zeros(0)
        role_v, main = strs(v["role"]), bools(v["room_main"])
        vs, ve = floats(v["start_s"]), floats(v["end_s"])
        for i in range(len(v)):
            if role_v[i] != config.ROLE_ME or not main[i]:
                continue
            end_i = ve[i] if not np.isnan(ve[i]) else INF
            hit = (we + EDGE_S > vs[i]) & (ws - EDGE_S < end_i)
            out[i] = bool(hit.any())
        return out

    # ---- causal at-risk rules (§3) -------------------------------------------------------
    def causal_rules(self, H: float = config.H_S) -> dict[str, bool]:
        """The causal exclusion rules at t (plus `end`), recomputed from truncated rows."""
        t, info, tb = self.t, self.info, self.tables
        segs = tb["segments"].sort_values("start_s", kind="stable")
        cur_type = strs(segs["seg_type"])[-1] if len(segs) else ""
        v = tb["voice"]
        role_v = strs(v["role"]) if len(v) else np.array([], dtype=object)
        vs = floats(v["start_s"]) if len(v) else np.zeros(0)
        brady = role_v == config.ROLE_ME
        breakout = cur_type == "breakout"
        recent = brady & (vs >= t - config.BREAKOUT_VOICE_LOOKBACK_S)
        if recent.any():
            last = np.flatnonzero(recent)[np.argmax(vs[recent])]
            breakout = breakout or not bool(v["room_main"].iloc[last])
        w = tb["words"]
        me_w = strs(w["role"]) == config.ROLE_ME if len(w) else np.zeros(0, bool)
        we = floats(w["end_s"]) if len(w) else np.zeros(0)
        speaking = bool((me_w & (we > t - config.S_ME_S) & (we <= t - EPS_W)).any())
        if len(v):
            ongoing = bools(v["ongoing"])
            speaking = speaking or bool((brady & ongoing & self.brady_wb).any())
        # hand rules from the live state and the live runs / blips rebuilt from samples <= t
        hand_up = self.live_state == config.STATE_UP
        hr = self.hand_runs
        ex, bl = hr.exit_s, hr.blip_end_s
        post_low = bool(((ex <= t) & (t < ex + config.R_LOW_S)).any())
        post_glitch = bool(((bl <= t) & (t < bl + config.R_LOW_S)).any())
        return {
            "opening": t < config.T_OPEN_S,
            "end": t + H > info.T_end,
            "rec_gap": any(r <= t < r + config.REC_RESTART_SKIP_S for r in info.restarts),
            "breakout": bool(breakout),
            "break": cur_type == "break",
            "me_speaking": speaking,
            "hand_up": bool(hand_up),
            "post_lowering": post_low,
            "post_glitch": post_glitch,
        }

    def causal_reason(self, H: float = config.H_S) -> str | None:
        rules = self.causal_rules(H)
        return next((r for r in config.EXCL_CAUSAL if rules[r]), None)

    # ---- features ------------------------------------------------------------------------
    def feature_values(self) -> dict:
        """Every feature at t, recomputed from the truncated tables ({column: value})."""
        return reference_values(self, **self.feature_kwargs)

    def features(self) -> pd.DataFrame:
        """One row: every feature at t (same columns and dtypes as `features.build`)."""
        return reference_frame([self])


def reference_frame(views: list[CausalView]) -> pd.DataFrame:
    """Stack the reference rows of several views of ONE class into a features frame."""
    from . import features as F

    vals = [v.feature_values() for v in views]
    cols = {k: np.array([d[k] for d in vals], dtype=object if isinstance(vals[0][k], str)
                        else float) for k in vals[0]} if vals else {}
    return F.assemble(views[0].cid if views else 0, np.array([v.t for v in views]), cols,
                     [pd.NA] * len(views))


# --------------------------------------------------------------------------- reference features
# Plain per-t code over the truncated tables. It shares only time-invariant helpers with the
# fast path (text predicates, z-null maths, PCW embeddings of pre-class material); every
# availability decision is re-derived here from what survived `truncate`.


def _run_start(W: Words, p: int) -> int:
    """First word of the professor utterance ending at word p (split at a pause > 1.2 s or
    a non-professor word)."""
    i = p
    while i > 0 and W.role[i - 1] == config.ROLE_PROF and \
            W.start[i] - W.end[i - 1] <= config.LASTPROF_PAUSE_S:
        i -= 1
    return i


def _evaluated(W: Words, p: int, t: float) -> bool:
    """>= 0.8 s of silence observed after word p (from word onsets visible at t)."""
    nxt = W.start[p + 1] if p + 1 < len(W) else t - EPS_W
    return nxt - W.end[p] >= config.Q_SILENCE_S and t - EPS_W - W.end[p] >= config.Q_SILENCE_S


def reference_values(view: CausalView, encoder=None, znull=None, edu=None, pcw_ref_sets=None
                     ) -> dict:
    """Every stored feature at view.t, from the truncated tables only."""
    from . import features as F

    t, info, tb = view.t, view.info, view.tables
    nan = np.nan
    enc = F.Memo(encoder if encoder is not None else F.default_encoder())
    zn = F.as_znull(znull)
    if enc.dim is None:
        enc.one("dimension probe")
    dim = enc.dim or 0
    W, FR = view.words, view.frames
    k_t = int(np.sum(W.text_av <= t))
    k_r = int(np.sum(W.role_av <= t))
    c: dict = {}

    # ---- ctx
    m = (t + info.rec_offset_s) / 60.0
    c["ctx_min"] = m
    spl = F.ctx_splines([m])[0]
    c.update({f"ctx_min_spl_{k}": spl[k] for k in range(5)})
    c.update(F.ctx_static(info.row))
    pres = tb["presence"]
    c["ctx_n_present"] = float(np.isfinite(floats(pres["enter_s"])).sum()) if len(pres) else 0.0
    segs = tb["segments"].sort_values("start_s", kind="stable")
    st, typ = floats(segs["start_s"]), strs(segs["seg_type"])
    if len(segs):
        c.update({"ctx_seg_type": typ[-1], "ctx_seg_elapsed_s": t - st[-1],
                  "ctx_seg_idx": float(len(segs) - 1)})
    else:
        c.update({"ctx_seg_type": F.UNKNOWN_CAT, "ctx_seg_elapsed_s": nan, "ctx_seg_idx": nan})
    ends = [st[j] for j in range(1, len(st)) if typ[j] != "breakout" and typ[j - 1] == "breakout"]
    c["ctx_since_breakout_s"] = t - ends[-1] if ends else nan
    c["ctx_after_breakout"] = float(bool(ends))
    polls = tb["polls"]
    c["ctx_in_poll"] = float(bool(len(polls)) and bool(np.isnan(floats(polls["close_s"])).any()))

    # ---- spk (main-room Forum voice; Brady only while word-bearing so far)
    v = tb["voice"]
    order = np.argsort(floats(v["start_s"]), kind="stable") if len(v) else np.zeros(0, int)
    vs, ve = floats(v["start_s"])[order], floats(v["end_s"])[order]
    role = strs(v["role"])[order] if len(v) else np.array([], dtype=object)
    spk = strs(v["spk_hash"])[order] if len(v) else np.array([], dtype=object)
    main = bools(v["room_main"])[order] if len(v) else np.zeros(0, bool)
    wb = view.brady_wb[order] if len(v) else np.zeros(0, bool)
    ended = ~np.isnan(ve)
    vis = main & ((role != config.ROLE_ME) | wb)
    cur = np.flatnonzero(vis & ~ended)
    if len(cur):
        ci = cur[np.argmax(vs[cur])]
        c.update({"spk_cur_role": role[ci], "spk_cur_elapsed_s": t - vs[ci],
                  "spk_silence_s": 0.0})
    else:
        ci = None
        done = np.flatnonzero(vis & ended)
        c.update({"spk_cur_role": F.UNKNOWN_CAT, "spk_cur_elapsed_s": nan,
                  "spk_silence_s": t - ve[done].max() if len(done) else nan})
    ext = np.where(ended, ve, t)
    for Wd in config.SPK_WINDOWS_S:
        ov = np.clip(np.minimum(ext, t) - np.maximum(vs, t - Wd), 0, None) * vis
        tot, prof = ov.sum(), ov[role == config.ROLE_PROF].sum()
        c[f"spk_prof_share_{Wd}"] = prof / tot if tot > 0 else nan
        stud = vis & (role == config.ROLE_STUDENT) & (vs > t - Wd)
        c[f"spk_stud_turns_{Wd}"] = float(stud.sum())
        if Wd == 300:
            c["spk_distinct_stud_300"] = float(len(set(spk[stud])))
    pv = vis & (role == config.ROLE_PROF)
    if pv.any():
        last_prof = vs[pv].max()
        c["spk_turns_since_prof"] = float((vis & (role != config.ROLE_PROF)
                                           & (vs > last_prof)).sum())
    else:
        c["spk_turns_since_prof"] = nan
    groups = tb["breakout_groups"]
    if len(groups) and ci is not None:
        c["spk_cur_in_my_group"] = float(role[ci] != config.ROLE_ME
                                         and spk[ci] in set(strs(groups["spk_hash"])))
    else:
        c["spk_cur_in_my_group"] = nan

    # ---- me
    me_f = np.flatnonzero(FR.role == config.ROLE_ME)
    c["me_words_so_far"] = float(FR.n_words[me_f].sum())
    c["me_since_turn_s"] = t - FR.end[me_f[-1]] if len(me_f) else nan
    bvis = np.flatnonzero(main & (role == config.ROLE_ME) & wb)
    turns = 0
    for j, e in enumerate(bvis):
        prev = bvis[j - 1] if j else None
        if prev is None or (ended[prev] and vs[e] - ve[prev] >= config.TURN_MERGE_GAP_S):
            turns += 1
    c["me_turns_so_far"] = float(turns)
    chat = tb["chat"]
    ct = floats(chat["t_s"])[strs(chat["role"]) == config.ROLE_ME] if len(chat) else np.zeros(0)
    c["me_chat_n"] = float(len(ct))
    c["me_since_chat_s"] = t - ct.max() if len(ct) else nan
    bo = (role == config.ROLE_ME) & ~main & ended
    c["me_breakout_talk_s"] = float((ve[bo] - vs[bo]).sum())

    # ---- me_hand (eligible live raises that went down by t, rebuilt from samples <= t)
    hr = view.hand_runs
    outc, outc_av = run_outcomes(hr, tb["voice"], tb["words"], info.eps_f)
    mh = me_hand_values(hr, outc, outc_av, [t])
    c.update({k: (v[0] if k != "me_hand_last_outcome" else str(v[0])) for k, v in mh.items()})

    # ---- poll
    c["poll_n_so_far"] = float(len(polls))
    closes = floats(polls["close_s"]) if len(polls) else np.zeros(0)
    closes = closes[np.isfinite(closes)]
    c["poll_since_close_s"] = t - closes.max() if len(closes) else nan
    ans_av = floats(polls["my_answer_avail_s"]) if len(polls) else np.zeros(0)
    my_ans = None
    if np.isfinite(ans_av).any():
        j = int(np.nanargmax(ans_av))
        my_ans = F.norm_text(strs(polls["my_answer_text"])[j])
    c["poll_my_ans_words"] = float(len(my_ans.split())) if my_ans is not None else nan

    # ---- q (professor utterances, role-known words only)
    prof_idx = [i for i in range(k_r) if W.role[i] == config.ROLE_PROF]
    p = prof_idx[-1] if prof_idx else -1
    events = []                                    # (end, kind, rhetorical) of questions
    inv_ends = []
    for q in prof_idx:
        if not _evaluated(W, q, t):
            continue
        text = join_tokens(W.tok[_run_start(W, q):q + 1])
        kind = F.lexical_kind(text)
        if kind == "invite":
            inv_ends.append(W.end[q])
        if F.is_question(text):
            rh = (q + 1 < k_r and W.role[q + 1] == config.ROLE_PROF
                  and W.start[q + 1] - W.end[q] <= config.Q_RHETORICAL_RESUME_S)
            events.append((W.end[q], kind, rh))
    cap = config.Q_AGE_CAP_S
    c["q_age_s"] = min(cap, t - max(x[0] for x in events)) if events else cap
    c["q_inv_age_s"] = min(cap, t - max(inv_ends)) if inv_ends else cap
    c["q_count_300"] = float(sum(1 for x in events if x[0] > t - config.Q_COUNT_WINDOW_S))
    if events:
        last = max(events, key=lambda x: x[0])
        c["q_kind"] = "rhetorical" if (last[1] == "plain" and last[2]) else last[1]
    else:
        c["q_kind"] = F.UNKNOWN_CAT
    evaluated = p >= 0 and _evaluated(W, p, t)
    lp_text = join_tokens(W.tok[_run_start(W, p):p + 1]) if p >= 0 else ""
    answered = bool((np.isin(FR.role, [config.ROLE_STUDENT, config.ROLE_ME])
                     & (FR.n_words >= config.Q_OPEN_STUDENT_MIN_WORDS)
                     & (FR.start > (W.end[p] if p >= 0 else INF))).any())
    c["q_open"] = float(evaluated and F.is_question(lp_text) and not answered)
    vote_txt = join_tokens(W.tok[i] for i in range(k_r) if W.role[i] == config.ROLE_PROF
                           and W.end[i] > t - config.WIN_LONG_S and W.tok[i])
    c["q_vote_lex"] = float(bool(config.VOTE_RE.search(vote_txt)))
    c["has_qmark"] = float(evaluated and bool(W.has_qmark[p]))

    # ---- text windows (CALLON phrases among the window's available words removed)
    for Wd in (config.WIN_SHORT_S, config.WIN_LONG_S):
        sel = [i for i in range(k_t) if W.end[i] > t - Wd and W.tok[i]]
        sel = [i for i, ok in zip(sel, keep_mask(W.tok[sel]), strict=True) if ok]
        c[f"n_words_{int(Wd)}"] = float(len(sel))
        if Wd == config.WIN_LONG_S:
            query = " ".join(W.tok[i] for i in sel)
            if sel:
                dur = float(sum(W.end[i] - W.start[i] for i in sel))
                c["wpm_60"] = 60.0 * len(sel) / dur if dur > 0 else nan
                c["filler_rate_60"] = float(np.mean([W.filler[i] for i in sel]))
                c["frac_unk_60"] = float(np.mean([i >= k_r or W.role[i] == config.ROLE_UNK
                                                  for i in sel]))
            else:
                c.update({"wpm_60": nan, "filler_rate_60": nan, "frac_unk_60": nan})

    # ---- embeddings (closed frames rebuilt from truncated words, plus the tail)
    E = enc(list(FR.text)) if len(FR) else np.zeros((0, dim))
    nonme = (FR.role != config.ROLE_ME) & (FR.n_words > 0)
    tail_txt = join_tokens(W.tok[i] for i in view.tail
                           if W.tok[i] and not (i < k_r and W.role[i] == config.ROLE_ME))
    tail = enc.one(tail_txt) if tail_txt else None
    f = np.flatnonzero(nonme & (FR.close > t - config.POOL_HORIZON_S))
    v60 = F.pool(E[f], FR.close[f], t, config.EMB_TAUS_S[1], tail)
    x = v60
    mu, sd = zn.stats(x)
    mu, sd = float(mu[0]), float(sd[0])
    rec = np.flatnonzero(nonme & (FR.close > t - config.WIN_LONG_S))
    old = np.flatnonzero(nonme & (FR.close > t - 300.0) & (FR.close <= t - config.WIN_LONG_S))
    rows = [E[rec]] + ([tail[None, :]] if tail is not None else [])
    recent = np.concatenate(rows) if sum(len(r) for r in rows) else np.zeros((0, dim))
    c["topic_shift"] = 1.0 - F.cos(F.mean_unit(recent, dim), F.mean_unit(E[old], dim))
    mine = np.flatnonzero((FR.role == config.ROLE_ME) & (FR.n_words > 0))
    c["own_sim"] = (F.zsim(x, mu, sd, F.mean_unit(E[mine], dim)[None, :])[0] if len(mine)
                    else nan)
    c["poll_res_my"] = (F.zsim(x, mu, sd, enc.one(my_ans)[None, :])[0] if my_ans else nan)
    pcw = F.pcw_context(tb["pcw_blocks"], info.row.get("pcw_status"), tb.get("prep"), enc,
                        pcw_ref_sets)
    c.update(F.pcw_static(pcw))
    c.update(F.pcw_dynamic(pcw, x, mu, sd, F.spent_now(pcw, E[mine], zn), query, zn))

    # ---- edu (A3): professor frames closed by t - eps_w, decayed
    if edu is not None and getattr(edu, "columns", ()):
        pf = np.flatnonzero((FR.role == config.ROLE_PROF) & (FR.n_words > 0)
                            & (FR.close <= t - EPS_W))
        S = edu.scores(list(FR.text[pf])) if len(pf) else np.zeros((0, len(edu.columns)))
        vals = F.edu_decayed(FR.close[pf], S, t)
        c.update({col: vals[j] for j, col in enumerate(edu.columns)})
    return c
