"""Causal features, fast vectorised path (spec §5, §4.4; amendments A2, A3).

`build(cid, points, tables)` computes every stored feature group of §5.7 for every
decision point of one class from the FULL tables. Each feature reads rows through an
availability cutoff found with `np.searchsorted` (word text `cummax(end + eps_w)`, word role
`cummax(role_avail_s)`, frame `close_s`, Forum rows `start + eps_F`, polls `+ eps_F,poll`,
live exits of Brady's own UP runs from hand samples <= t, `causal.hand_runs`). Multi-word
CALLON phrases are removed from joined available text only (`causal.join_tokens`). The slow
reference path is `causal.CausalView`; `tests/test_causal.py` checks the two agree.

Groups stored here: ctx, spk, me, me_hand, poll, q, text (dense), pcwT, pcwA, punct
(`has_qmark`, F-punct only) and edu (A3, only when scores are available). Groups fitted
in-fold elsewhere: emb (PCA/SVD), the TF-IDF text matrix, `pcw_new_block` (training-fold
percentile; `pcw_new_block()` here), hist (A2; `hist_features()` here, needs the training
class set) and stack (A4, models). The pooled embeddings [v15, v60, e_last, e_lastprof]
that M4 uses come back from `build(..., return_emb=True)`; `features.emb_row` indexes them.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, fields

import numpy as np
import pandas as pd

from . import config, schemas
from .causal import (
    CLASS_TABLES,
    EPS_W,
    INF,
    ClassInfo,
    Frames,
    Words,
    bools,
    brady_wb_times,
    breakout_ended_at,
    build_frames,
    class_info,
    class_tables,
    floats,
    hand_runs,
    join_tokens,
    keep_mask,
    load_tables,
    me_hand_values,
    prep_words,
    run_outcomes,
    strs,
    table,
)

log = logging.getLogger(__name__)

PCW_BRANCH = "N"   # §5.6; D8 decides. Branch H adds pcwA to F-all.
UNKNOWN_CAT = "none"

# --------------------------------------------------------------------------- columns

GROUP_COLUMNS: dict[str, tuple[str, ...]] = {
    "ctx": ("ctx_min",) + tuple(f"ctx_min_spl_{k}" for k in range(5)) + (
        "ctx_course", "ctx_area", "ctx_sem_idx", "ctx_session_no", "ctx_session_frac",
        "ctx_dow", "ctx_hour_local", "ctx_hour_utc", "ctx_enrolled", "ctx_n_present",
        "ctx_seg_type", "ctx_seg_elapsed_s", "ctx_seg_idx", "ctx_since_breakout_s",
        "ctx_after_breakout", "ctx_in_poll"),
    "spk": ("spk_cur_role", "spk_cur_elapsed_s", "spk_silence_s", "spk_prof_share_60",
            "spk_prof_share_300", "spk_stud_turns_60", "spk_stud_turns_300",
            "spk_distinct_stud_300", "spk_turns_since_prof", "spk_cur_in_my_group"),
    "me": ("me_turns_so_far", "me_since_turn_s", "me_words_so_far", "me_chat_n",
           "me_since_chat_s", "me_breakout_talk_s"),
    "me_hand": ("me_hand_n_so_far", "me_hand_since_down_s", "me_hand_since_up_s",
                "me_hand_last_outcome"),
    "poll": ("poll_n_so_far", "poll_since_close_s", "poll_my_ans_words", "poll_res_my"),
    "q": ("q_open", "q_age_s", "q_inv_age_s", "q_kind", "q_count_300", "q_vote_lex"),
    "text": ("n_words_15", "n_words_60", "wpm_60", "filler_rate_60", "frac_unk_60",
             "topic_shift", "own_sim"),
    "pcwT": ("pcw_tmpl_has", "pcw_res_Q", "pcw_block_pos_Q", "lo_sim", "sg_sim"),
    "pcwA": ("pcw_has", "pcw_my_words", "pcw_frac_answered", "pcw_n_code", "pcw_res_A",
             "pcw_res_gap", "pcw_unspent", "pcw_res_ref", "pcw_bm25_gap", "pcw_invest",
             "pcw_block_pos"),
    "punct": ("has_qmark",),
    "edu": tuple(config.EDU_CHECKPOINTS),
}
# Columns fitted in-fold (never stored by build): listed so featuresets can name them.
IN_FOLD_COLUMNS: dict[str, tuple[str, ...]] = {
    "pcwT": ("pcw_new_block",), "hist": config.HIST_COLUMNS, "stack": config.STACK_COLUMNS}
GROUPS: dict[str, tuple[str, ...]] = {
    g: GROUP_COLUMNS.get(g, ()) + IN_FOLD_COLUMNS.get(g, ()) for g in config.FEATURE_ALLOWLIST}
CATEGORICAL = ("ctx_course", "ctx_area", "ctx_seg_type", "spk_cur_role", "q_kind",
               "me_hand_last_outcome")
STORED_COLUMNS = tuple(c for g in ("ctx", "spk", "me", "me_hand", "poll", "q", "text",
                                   "pcwT", "pcwA", "punct") for c in GROUP_COLUMNS[g])
EDU_COLUMNS = GROUP_COLUMNS["edu"]


def _check_groups() -> None:
    for g, cols in GROUPS.items():
        for c in cols:
            if config.feature_group(c) != g:
                raise AssertionError(f"{c} is listed in group {g} but config says "
                                     f"{config.feature_group(c)}")


_check_groups()

_STRUCT = ("ctx", "spk", "me", "me_hand", "poll")


def featuresets(branch: str = PCW_BRANCH) -> dict[str, dict]:
    """Feature sets (§5.7 + A2, A3, A4) as {name: {"groups", "columns", "text"}}.

    "columns" adds single columns; "text" is the text variant the in-fold text groups use
    (norm | name | punct). ASCII names: "F-all-me_hand" is the spec's F-all−me_hand.
    """
    all_g = _STRUCT + ("q", "text", "emb", "pcwT") + (("pcwA",) if branch == "H" else ())

    def fs(groups, columns=(), text="norm"):
        return {"groups": tuple(dict.fromkeys(groups)), "columns": tuple(columns),
                "text": text}
    drop = lambda gs, *out: tuple(g for g in gs if g not in out)
    return {
        "F-struct": fs(_STRUCT),
        "F-struct-me_hand": fs(drop(_STRUCT, "me_hand")),
        "F-text": fs(("text", "q", "emb"), ("ctx_min",)),
        "F-all": fs(all_g),
        "F-all-me_hand": fs(drop(all_g, "me_hand")),
        "F-all-pcw": fs(drop(all_g, "pcwT", "pcwA")),
        "F-all+pcwA": fs(all_g + ("pcwA",)),
        "F-pcw": fs(("ctx", "pcwT", "pcwA")),
        "F-all-me": fs(drop(all_g, "me", "me_hand")),
        "F-name": fs(all_g, text="name"),
        "F-punct": fs(all_g + ("punct",), text="punct"),
        "F-min": fs((), tuple(f"ctx_min_spl_{k}" for k in range(5))),
        "F-all+hist": fs(all_g + ("hist",)),
        "F-all+edu": fs(all_g + ("edu",)),
        "F-all+stack": fs(all_g + ("stack",)),
    }


FEATURESETS = featuresets()


def columns_for(featureset: str, available: Sequence[str]) -> list[str]:
    """The model-input columns of `featureset` among `available` (allowlist enforced)."""
    spec = FEATURESETS[featureset]
    out = []
    for col in available:
        if col in config.FEATURE_META_COLUMNS:
            continue
        group = config.feature_group(col)
        if (group in spec["groups"] or col in spec["columns"]) and \
                config.feature_allowed(col, featureset):
            out.append(col)
    return out


# --------------------------------------------------------------------------- provenance
# column -> refs ("table.column" keys of loaders.PROVENANCE, or (source, path) pairs).

_CLOCK = ("words.start_s", "words.end_s")
_WORDS = ("words.start_s", "words.end_s", "words.text_norm", "words.role", "words.spk_hash",
          "words.role_avail_s")
_VOICE = ("voice.start_s", "voice.end_s", "voice.role", "voice.room_main", "voice.spk_hash",
          "words.start_s", "words.end_s", "words.role", "words.role_avail_s")
_SEGS = ("segments.start_s", "segments.seg_type")
_POLLS = ("polls.open_s", "polls.close_s")
_PCW_Q = ("pcw_blocks.prompt_text", "pcw_blocks.block_order", "pcw_blocks.block_uuid")
_PCW_A = ("pcw_blocks.answer_text", "pcw_blocks.answer_words", "pcw_blocks.has_code",
          "pcw_blocks.n_files", "pcw_blocks.block_order")
_SCHEDULE = (("schedule", "results[].id"), ("schedule", "results[].date"))

PROVENANCE: dict[str, tuple] = {}
PROVENANCE.update({c: ("classes.rec_offset_s",) + _CLOCK for c in GROUP_COLUMNS["ctx"][:6]})
PROVENANCE.update({
    "ctx_course": ("classes.course_code",), "ctx_area": ("classes.course_area",),
    "ctx_sem_idx": ("classes.semester_idx",), "ctx_session_no": ("classes.session_no",),
    "ctx_session_frac": ("classes.session_no",) + _SCHEDULE, "ctx_dow": ("classes.weekday",),
    "ctx_hour_local": ("classes.hour_local",), "ctx_hour_utc": ("classes.hour_utc",),
    "ctx_enrolled": ("classes.enrolled",), "ctx_n_present": ("presence.enter_s",),
    "ctx_seg_type": _SEGS, "ctx_seg_elapsed_s": _SEGS, "ctx_seg_idx": _SEGS,
    "ctx_since_breakout_s": _SEGS, "ctx_after_breakout": _SEGS, "ctx_in_poll": _POLLS,
})
PROVENANCE.update({c: _VOICE for c in GROUP_COLUMNS["spk"]})
PROVENANCE["spk_cur_in_my_group"] = _VOICE + ("breakouts.my_group",) + _SEGS
PROVENANCE.update({"me_turns_so_far": _VOICE, "me_since_turn_s": _WORDS,
                   "me_words_so_far": _WORDS, "me_chat_n": ("chat.t_s", "chat.role"),
                   "me_since_chat_s": ("chat.t_s", "chat.role"),
                   "me_breakout_talk_s": _VOICE})
# me_hand_* is the label's own history (A16, video-derived; see DERIVED_INPUTS). Its only
# raw inputs are the master clock and, for `called`, Brady's words and voice.
PROVENANCE.update({c: _CLOCK for c in GROUP_COLUMNS["me_hand"]})
PROVENANCE["me_hand_last_outcome"] = _VOICE
PROVENANCE.update({c: _POLLS for c in GROUP_COLUMNS["poll"]})
PROVENANCE.update({"poll_my_ans_words": ("polls.my_answer_text", "polls.my_answer_avail_s"),
                   "poll_res_my": ("polls.my_answer_text", "polls.my_answer_avail_s")
                   + _WORDS})
PROVENANCE.update({c: _WORDS for c in GROUP_COLUMNS["q"] + GROUP_COLUMNS["text"]})
PROVENANCE["filler_rate_60"] = _WORDS + ("words.filler",)
PROVENANCE["has_qmark"] = _WORDS + ("words.has_qmark",)
PROVENANCE.update({"pcw_tmpl_has": _PCW_Q, "pcw_res_Q": _PCW_Q + _WORDS,
                   "pcw_block_pos_Q": _PCW_Q + _WORDS, "pcw_new_block": _PCW_Q + _WORDS,
                   "lo_sim": ("class_edit_page.lo_text",) + _WORDS,
                   "sg_sim": ("class_edit_page.sg_text",) + _WORDS})
PROVENANCE.update({c: _PCW_A + _PCW_Q + _WORDS for c in GROUP_COLUMNS["pcwA"]})
PROVENANCE.update({c: _WORDS for c in EDU_COLUMNS})
PROVENANCE.update({c: ("classes.course_code", "classes.date") for c in config.HIST_COLUMNS})
PROVENANCE.update({c: _WORDS for c in config.STACK_COLUMNS})

# Inputs that are not raw JSON paths, by group (documented, never whitelisted paths).
DERIVED_INPUTS = {
    "me_hand": "hand_frames: Brady's earlier eligible live raises, samples <= t (A16)",
    "hist": "points/hand_episodes of EARLIER classes in the TRAINING fold only (A2)",
    "edu": "StanfordSCALE EduBehaviors SetFit scores of professor frames (A3)",
    "stack": "inner-OOF M2/M4 probabilities (A4)",
    "pcwT": "pcw_new_block uses the training-fold 90th percentile of pcw_res_Q",
}


# --------------------------------------------------------------------------- text helpers

_TAG_RE = re.compile(r"<[^>]+>")
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NONWORD_RE = re.compile(r"[^0-9a-z\[\]]+")


def norm_text(text) -> str:
    """Normalise pre-class / poll text like feature text: drop HTML, split identifiers on
    '_' and camelCase, lowercase, strip punctuation, drop [name]/[me] tokens."""
    if text is None or (isinstance(text, float) and math.isnan(text)) or text is pd.NA:
        return ""
    s = _TAG_RE.sub(" ", str(text)).replace("_", " ")
    s = _CAMEL_RE.sub(" ", s).lower()
    toks = [x for x in _NONWORD_RE.sub(" ", s).split() if x not in ("[name]", "[me]")]
    return " ".join(x.strip("[]") for x in toks if x.strip("[]"))


def text_key(text: str) -> str:
    """sha1 key of a (normalised) text, for embedding and EduBehaviors caches."""
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()


def is_question(text: str) -> bool:
    """Headline question test (§5.5, no punctuation): wh-word / inverted auxiliary first and
    >= 4 words, or a tag ending."""
    if not text:
        return False
    if config.Q_FIRST_RE.match(text) and len(text.split()) >= config.Q_MIN_WORDS:
        return True
    return bool(config.Q_TAG_RE.search(text))


def lexical_kind(text: str) -> str:
    """invite | check | plain (rhetorical is decided by timing)."""
    if config.INVITE_RE.search(text or ""):
        return "invite"
    if config.CHECK_RE.search(text or ""):
        return "check"
    return "plain"


def chunks(text: str, size: int = config.PCW_CHUNK_WORDS,
           overlap: int = config.PCW_CHUNK_OVERLAP) -> list[str]:
    """Split normalised text into chunks of <= size words with `overlap` words shared."""
    words = (text or "").split()
    if not words:
        return []
    step = max(1, size - overlap)
    out = []
    for a in range(0, len(words), step):
        out.append(" ".join(words[a:a + size]))
        if a + size >= len(words):
            break
    return out


# --------------------------------------------------------------------------- encoders


class HashEncoder:
    """Deterministic bag-of-words hashing encoder (unit vectors).

    For tests and offline dry runs only: it has no semantics beyond word overlap. Results
    use `oncue.embed` (bge-small) instead.
    """

    def __init__(self, dim: int = 64, seed: int = config.SEED):
        self.dim, self.seed = int(dim), int(seed)
        self._tok: dict[str, np.ndarray] = {}

    def _vec(self, tok: str) -> np.ndarray:
        v = self._tok.get(tok)
        if v is None:
            h = int(hashlib.sha1(f"{self.seed}:{tok}".encode()).hexdigest()[:12], 16)
            v = np.random.default_rng(h).standard_normal(self.dim)
            self._tok[tok] = v
        return v

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim))
        for i, text in enumerate(texts):
            toks = (text or "").split()
            if toks:
                v = np.sum([self._vec(x) for x in toks], axis=0)
                out[i] = v / (np.linalg.norm(v) or 1.0)
        return out


class Memo:
    """Text -> unit vector cache around an encoder (consecutive points share tails)."""

    def __init__(self, encoder: Callable[[Sequence[str]], np.ndarray]):
        self.encoder = encoder.encoder if isinstance(encoder, Memo) else encoder
        self.cache: dict[str, np.ndarray] = encoder.cache if isinstance(encoder, Memo) else {}
        self.dim: int | None = None

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        new = [x for x in dict.fromkeys(texts) if x and x not in self.cache]
        if new:
            vecs = np.asarray(self.encoder(new), dtype=float)
            for x, v in zip(new, vecs, strict=True):
                n = np.linalg.norm(v)
                self.cache[x] = v / n if n > 0 else np.full(len(v), np.nan)
        if self.dim is None:
            some = next(iter(self.cache.values()), None)
            self.dim = len(some) if some is not None else 0
        out = np.full((len(texts), self.dim or 0), np.nan)
        for i, x in enumerate(texts):
            if x:
                out[i] = self.cache[x]
        return out

    def one(self, text: str) -> np.ndarray:
        return self([text])[0] if text else np.full(self.dim or 0, np.nan)


def default_encoder(model: str | None = None):
    """bge-small (or `model`, e.g. embed.HASH_MODEL for offline dry runs) via oncue.embed,
    memoised, with the disk cache at config.EMB_CACHE_DIR as it is now."""
    try:
        from . import embed
    except ImportError as exc:  # pragma: no cover - depends on the embed stage
        raise RuntimeError("features need an encoder: pass encoder=... or add oncue.embed"
                           ) from exc
    model = model or config.EMB
    if model == embed.HASH_MODEL:
        return Memo(embed.Embedder(model, cache_dir=None).encode)
    return Memo(embed.Embedder.with_fallback(model, config.EMB_FALLBACK,
                                             cache_dir=config.EMB_CACHE_DIR).encode)


# --------------------------------------------------------------------------- z-null and math


@dataclass
class ZNull:
    """Reference set R_c for z-scoring similarities (§5.6); z is NaN with < ZNULL_MIN rows."""

    R: np.ndarray | None = None

    def __post_init__(self):
        if self.R is not None:
            R = np.asarray(self.R, dtype=float)
            R = R[np.isfinite(R).all(axis=1)] if R.ndim == 2 else np.zeros((0, 0))
            n = np.linalg.norm(R, axis=1)
            self.R = R[n > 0] / n[n > 0, None] if len(R) else R

    @property
    def ok(self) -> bool:
        return self.R is not None and len(self.R) >= config.ZNULL_MIN

    def stats(self, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(mu, sd) of cos(u, r) over R for each row of u (NaN if unusable)."""
        u = np.atleast_2d(u)
        if not self.ok or u.shape[1] != self.R.shape[1]:
            return np.full(len(u), np.nan), np.full(len(u), np.nan)
        s = unit(u) @ self.R.T
        return s.mean(axis=1), s.std(axis=1)


def as_znull(obj) -> ZNull:
    """A ZNull from a ZNull, an `oncue.embed.ZNull` (`.ref`), a reference matrix or None."""
    if isinstance(obj, ZNull):
        return obj
    return ZNull(getattr(obj, "ref", obj))


def unit(v: np.ndarray) -> np.ndarray:
    """Row-normalise (NaN rows stay NaN, zero rows become NaN)."""
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, v / np.where(n > 0, n, 1.0), np.nan)


def zsim(x: np.ndarray, mu: float, sd: float, V: np.ndarray) -> np.ndarray:
    """z(x, v) = (cos(x, v) - mu_x) / sd_x for each row of V (NaN when undefined).

    Rows of V and x are unit vectors (or NaN): every embedding here comes from `Memo`,
    `pool` or `mean_unit`, which all normalise."""
    V = np.atleast_2d(V)
    if V.size == 0 or not np.isfinite(mu) or not (sd > 0) or not np.isfinite(x).all():
        return np.full(len(V), np.nan)
    with np.errstate(invalid="ignore"):
        return (V @ x - mu) / sd


def pool(E: np.ndarray, closes: np.ndarray, t: float, tau: float,
         tail: np.ndarray | None) -> np.ndarray:
    """Recency-pooled unit vector v_tau(t) of frames (rows of E closed at `closes`) plus the
    tail (weight 1). NaN when nothing is pooled."""
    dim = E.shape[1] if E.ndim == 2 and E.shape[1] else (len(tail) if tail is not None else 0)
    acc = np.zeros(dim)
    have = False
    if len(closes):
        wts = np.exp(-(t - closes) / tau)
        acc = acc + wts @ E
        have = True
    if tail is not None and np.isfinite(tail).all():
        acc = acc + tail
        have = True
    return unit(acc) if have else np.full(dim, np.nan)


def mean_unit(E: np.ndarray, dim: int) -> np.ndarray:
    return unit(E.mean(axis=0)) if len(E) else np.full(dim, np.nan)


def cos(a: np.ndarray, b: np.ndarray) -> float:
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return np.nan
    return float(unit(a) @ unit(b))


_SPLINE = []


def ctx_splines(ctx_min) -> np.ndarray:
    """ctx_min_spl_0..4: quadratic B-splines with knots at 10/25/45/65/80 min, no bias
    column, linear extrapolation. The knots are fixed, so nothing is fitted on data."""
    if not _SPLINE:
        from sklearn.preprocessing import SplineTransformer

        knots = np.asarray(config.CTX_SPLINE_KNOTS, dtype=float).reshape(-1, 1)
        _SPLINE.append(SplineTransformer(knots=knots, degree=2, include_bias=False,
                                         extrapolation="linear").fit(knots))
    return _SPLINE[0].transform(np.asarray(ctx_min, dtype=float).reshape(-1, 1))


# --------------------------------------------------------------------------- EduBehaviors (A3)


class EduTable:
    """EduBehaviors scores looked up by frame text (a cached score table).

    `df` has `text_key` (or `text_norm`) plus any of the edu_* columns. The live scorer is
    `oncue.edu.EduScorer` (A3); `build` accepts either: anything with `.columns` and
    `.scores(texts) -> [n, len(columns)]`.
    """

    def __init__(self, df: pd.DataFrame):
        df = df.copy()
        if "text_key" not in df:
            df["text_key"] = [text_key(x) for x in strs(df["text_norm"])]
        self.columns = tuple(c for c in EDU_COLUMNS if c in df.columns)
        self._map = {k: row for k, row in zip(df["text_key"],
                                              df[list(self.columns)].to_numpy(float),
                                              strict=True)}

    def scores(self, texts: Sequence[str]) -> np.ndarray:
        out = np.full((len(texts), len(self.columns)), np.nan)
        for i, x in enumerate(texts):
            row = self._map.get(text_key(x))
            if row is not None:
                out[i] = row
        return out


def edu_decayed(closes: np.ndarray, S: np.ndarray, t: float,
                tau: float = config.EDU_DECAY_TAU_S) -> np.ndarray:
    """max_f s_f * exp(-(t - close_f) / tau) over the frames given (already closed);
    0 when there is none or no frame has a score (as `oncue.edu.edu_features`)."""
    k = S.shape[1] if S.ndim == 2 else 0
    if not len(closes):
        return np.zeros(k)
    w = np.exp(-(t - closes) / tau)[:, None]
    return (np.where(np.isfinite(S), S, 0.0) * w).max(axis=0)


# --------------------------------------------------------------------------- PCW context


class BM25:
    """rank_bm25.BM25Okapi over one class's PCW docs [Q_1..Q_B, A_1..A_B, QA_1..QA_B],
    scored with repeated query tokens grouped (same sums as BM25Okapi.get_scores)."""

    def __init__(self, docs: list[str]):
        from rank_bm25 import BM25Okapi

        self.model = BM25Okapi([d.split() or ["<empty>"] for d in docs])
        m = self.model
        dl = np.asarray(m.doc_len, dtype=float)
        self._den = m.k1 * (1 - m.b + m.b * dl / m.avgdl)
        self._tf = {}
        for j, freqs in enumerate(m.doc_freqs):
            for tok, f in freqs.items():
                self._tf.setdefault(tok, np.zeros(len(dl)))[j] = f

    def scores(self, tokens: Sequence[str]) -> np.ndarray:
        m = self.model
        out = np.zeros(len(self._den))
        counts: dict[str, int] = {}
        for tok in tokens:
            counts[tok] = counts.get(tok, 0) + 1
        for tok in sorted(counts):
            tf = self._tf.get(tok)
            idf = m.idf.get(tok) or 0.0
            if tf is not None and idf:
                out += counts[tok] * idf * tf * (m.k1 + 1) / (tf + self._den)
        return out


@dataclass
class PCW:
    """Pre-class material of one class (time-invariant; shared by both paths)."""

    status: str
    usable: bool
    B: int
    q_text: list[str]
    a_text: list[str]
    answered: np.ndarray
    words: np.ndarray
    has_code: np.ndarray
    eQ: np.ndarray
    eA: np.ndarray
    eQA: np.ndarray
    eC: np.ndarray            # answer chunks a_j
    lo: np.ndarray            # learning-outcome embeddings
    sg: np.ndarray            # study-guide paragraph embeddings
    ref_sets: list[np.ndarray] = field(default_factory=list)
    bm25: BM25 | None = None


def pcw_context(pcw_blocks: pd.DataFrame, status: str | None, prep: pd.DataFrame | None,
                enc: Memo, ref_sets: list[np.ndarray] | None = None) -> PCW:
    """Embed prompts Q_b, answers A_b, Q_b + A_b, answer chunks, learning outcomes and
    study-guide paragraphs; build the BM25 index over this class's blocks."""
    status = status or "unknown"
    b = pcw_blocks.sort_values("block_order", kind="stable") if len(pcw_blocks) else pcw_blocks
    q = [norm_text(x) for x in strs(b["prompt_text"])] if len(b) else []
    a = [norm_text(x) for x in strs(b["answer_text"])] if len(b) else []
    words = np.array([len(x.split()) for x in a], dtype=float)
    files = floats(b["n_files"]) if len(b) else np.zeros(0)
    answered = (words > 0) | (np.nan_to_num(files) > 0)
    code = bools(b["has_code"]) if len(b) else np.zeros(0, bool)
    dim = enc.dim or len(enc.one("pcw"))
    emb = lambda texts: enc(texts) if texts else np.zeros((0, dim))
    qa = [f"{x} {y}".strip() for x, y in zip(q, a, strict=True)]
    c = [ch for x, ok in zip(a, answered, strict=True) if ok for ch in chunks(x)]
    lo = sg = []
    if prep is not None and len(prep):
        kind, txt = strs(prep["kind"]), [norm_text(x) for x in strs(prep["text"])]
        lo = [x for k, x in zip(kind, txt, strict=True) if k == "lo" and x]
        sg = [x for k, x in zip(kind, txt, strict=True) if k == "sg" and x]
    bm25 = BM25(q + a + qa) if len(q) else None
    return PCW(status=status, usable=status in config.PCW_USABLE_STATUSES, B=len(q),
               q_text=q, a_text=a, answered=answered, words=words, has_code=code,
               eQ=emb(q), eA=emb(a), eQA=emb(qa), eC=emb(c), lo=emb(lo), sg=emb(sg),
               ref_sets=list(ref_sets or []), bm25=bm25)


def pcw_static(p: PCW) -> dict[str, float]:
    """Class-level PCW columns (pre-class facts).

    Only a usable doc (`asof_start` / `verified` / `git`) contributes answers. Every other
    status (`empty`, `none`, `unknown`) gets the same "no usable answers" values, so no pcwA
    value (and no NaN pattern) can tell an `unknown` class (doc edited after class start)
    from the others: `pcw_status` is QA, never a feature (§4.2, §5.6; L12)."""
    has_t = float(p.B > 0)
    out = {"pcw_tmpl_has": has_t}
    if p.usable:
        nb = max(p.B, 1)
        out.update({"pcw_has": float(p.answered.any()), "pcw_my_words": float(p.words.sum()),
                    "pcw_frac_answered": float(p.answered.sum() / nb) if p.B else 0.0,
                    "pcw_n_code": float((p.has_code & p.answered).sum())})
    else:
        out.update({"pcw_has": 0.0, "pcw_my_words": 0.0, "pcw_frac_answered": 0.0,
                    "pcw_n_code": 0.0})
    return out


def pcw_dynamic(p: PCW, x: np.ndarray, mu: float, sd: float, spent: np.ndarray,
                query: str, znull: ZNull) -> dict[str, float]:
    """Similarity PCW columns at one t, given x = v60(t) and which chunks are spent."""
    nan = np.nan
    out = {"pcw_res_Q": nan, "pcw_block_pos_Q": nan, "lo_sim": nan, "sg_sim": nan,
           "pcw_res_A": nan, "pcw_res_gap": nan, "pcw_unspent": nan, "pcw_res_ref": nan,
           "pcw_bm25_gap": nan, "pcw_invest": nan, "pcw_block_pos": nan}
    if p.B:
        zQ = zsim(x, mu, sd, p.eQ)
        if np.isfinite(zQ).any():
            bq = int(np.nanargmax(zQ))
            out["pcw_res_Q"], out["pcw_block_pos_Q"] = float(zQ[bq]), (bq + 1) / p.B
    for col, E in (("lo_sim", p.lo), ("sg_sim", p.sg)):
        z = zsim(x, mu, sd, E) if len(E) else np.zeros(0)
        if np.isfinite(z).any():
            out[col] = float(np.nanmax(z))
    if not (p.usable and p.answered.any()):
        return out
    zQ = zsim(x, mu, sd, p.eQ)
    zQA = np.where(p.answered, zsim(x, mu, sd, p.eQA), np.nan)
    if np.isfinite(zQA).any():
        b = int(np.nanargmax(zQA))
        zA = zsim(x, mu, sd, p.eA[b:b + 1])[0]
        out.update({"pcw_res_A": float(zA), "pcw_res_gap": float(zA - zQ[b]),
                    "pcw_invest": math.log1p(p.words[b]), "pcw_block_pos": (b + 1) / p.B})
    zC = zsim(x, mu, sd, p.eC) if len(p.eC) else np.zeros(0)
    live = zC[~spent] if len(zC) else zC
    if np.isfinite(live).any():
        out["pcw_unspent"] = float(np.nanmax(live))
    if len(p.ref_sets) >= config.PCW_REF_MIN and np.isfinite(zC).any():
        refs = [np.nanmax(zsim(x, mu, sd, R)) for R in p.ref_sets if len(R)]
        if len(refs) >= config.PCW_REF_MIN:
            out["pcw_res_ref"] = float(np.nanmax(zC) - np.mean(refs))
    if p.bm25 is not None and query:
        s = p.bm25.scores(query.split())
        B = p.B
        sqa = np.where(p.answered, s[2 * B:], -INF)
        b = int(np.argmax(sqa))
        out["pcw_bm25_gap"] = float(s[B + b] - s[b])
    return out


def chunk_spent_times(p: PCW, me_E: np.ndarray, me_close: np.ndarray,
                      znull: ZNull) -> np.ndarray:
    """For each answer chunk a_j: the close time of the first Brady frame m with
    z(a_j, m) >= kappa (inf if never). unspent(t) = {j: spent_j > t}."""
    out = np.full(len(p.eC), INF)
    if not len(p.eC) or not len(me_E):
        return out
    mu, sd = znull.stats(p.eC)
    for j in range(len(p.eC)):
        z = zsim(p.eC[j], mu[j], sd[j], me_E)
        hit = np.flatnonzero(z >= config.KAPPA)
        if len(hit):
            out[j] = me_close[hit].min()
    return out


def spent_now(p: PCW, me_E: np.ndarray, znull: ZNull) -> np.ndarray:
    """Reference form: chunks already said by Brady's closed frames (rows of me_E)."""
    if not len(p.eC):
        return np.zeros(0, bool)
    if not len(me_E):
        return np.zeros(len(p.eC), bool)
    mu, sd = znull.stats(p.eC)
    return np.array([bool((zsim(p.eC[j], mu[j], sd[j], me_E) >= config.KAPPA).any())
                     for j in range(len(p.eC))])


# --------------------------------------------------------------------------- z-null builders


def _class_start(classes: pd.DataFrame) -> pd.Series:
    key = classes["sched_start_utc"] if "sched_start_utc" in classes else classes["date"]
    return pd.to_datetime(key.astype("string"), utc=True, errors="coerce")


def znull_reference(cid, classes: pd.DataFrame, pcw_blocks: pd.DataFrame, enc,
                    n: int = config.ZNULL_N, seed: int = config.SEED) -> ZNull:
    """R_c: up to n chunks (fixed seed) of OTHER courses' PCW prompts and verified answers,
    from classes that started before class c (§5.6). No labels, so no fold dependence."""
    enc = enc if isinstance(enc, Memo) else Memo(enc)
    start = _class_start(classes)
    me = classes["class_id"] == int(cid)
    if not me.any():
        return ZNull(None)
    c_start, c_course = start[me].iloc[0], strs(classes.loc[me, "course_code"])[0]
    earlier = classes[(start < c_start) & (strs(classes["course_code"]) != c_course)]
    status = dict(zip(earlier["class_id"].astype(int), strs(earlier["pcw_status"]), strict=True))
    texts = []
    blocks = pcw_blocks[pcw_blocks["class_id"].isin(list(status))]
    for row in blocks.itertuples():
        texts += chunks(norm_text(row.prompt_text))
        if status.get(int(row.class_id)) in config.PCW_USABLE_STATUSES:
            texts += chunks(norm_text(row.answer_text))
    texts = sorted({x for x in texts if x})
    if len(texts) > n:
        idx = np.random.default_rng(seed).choice(len(texts), n, replace=False)
        texts = [texts[i] for i in sorted(idx)]
    return ZNull(enc(texts) if texts else None)


def pcw_ref_sets(cid, classes: pd.DataFrame, pcw_blocks: pd.DataFrame, enc) -> list[np.ndarray]:
    """Answer-chunk embeddings of the 5 most recent earlier classes of the same section with
    a usable doc (for pcw_res_ref)."""
    enc = enc if isinstance(enc, Memo) else Memo(enc)
    start = _class_start(classes)
    me = classes["class_id"] == int(cid)
    if not me.any():
        return []
    sec = classes.loc[me, "section_id"].iloc[0]
    cand = classes[(classes["section_id"] == sec) & (start < start[me].iloc[0])
                   & np.isin(strs(classes["pcw_status"]), list(config.PCW_USABLE_STATUSES))]
    cand = cand.assign(_s=start[cand.index]).sort_values("_s").tail(config.PCW_REF_RECENT)
    out = []
    for c in cand["class_id"].astype(int):
        b = pcw_blocks[pcw_blocks["class_id"] == c]
        texts = [ch for x in strs(b["answer_text"]) for ch in chunks(norm_text(x))]
        if texts:
            out.append(enc(texts))
    return out


def prep_table(class_id, class_edit_page: dict | None) -> pd.DataFrame:
    """Learning outcomes ("lo") and study-guide / PCW-instruction paragraphs ("sg") from a
    class_edit_page document, reading only the whitelisted paths (A12). Snake or kebab keys.
    Uses `oncue.pcw.class_materials` (the PCW stage's parser) when it is available."""
    try:
        from .pcw import class_materials
    except ImportError:  # pragma: no cover - pcw is another stage's module
        class_materials = None
    if class_materials is not None:
        mat = class_materials(class_edit_page or {})
        rows = [("lo", x) for x in mat.get("learning_outcomes", []) if x]
        rows += [("sg", x) for x in mat.get("study_guide", []) if x]
        rows += [("sg", x) for x in str(mat.get("prep") or "").split("\n") if x.strip()]
        return pd.DataFrame({"class_id": int(class_id), "kind": [k for k, _ in rows],
                             "text": [x for _, x in rows]}, columns=["class_id", "kind", "text"])
    rows = []
    doc = class_edit_page or {}

    def get(d, key):
        if not isinstance(d, dict):
            return None
        return d.get(key, d.get(key.replace("-", "_")))
    for fo in get(get(doc, "assessment"), "focused-outcomes") or []:
        desc = get(get(fo, "learning-outcome"), "description")
        if desc:
            rows.append(("lo", str(desc)))
    for key in ("preparatory-assignment", "study-guide"):
        node = get(doc, key)
        text = get(node, "description") if isinstance(node, dict) else node
        if text:
            for para in re.split(r"</p>|<br\s*/?>|\n\s*\n", str(text)):
                if _TAG_RE.sub(" ", para).strip():
                    rows.append(("sg", para))
    return pd.DataFrame({"class_id": int(class_id), "kind": [k for k, _ in rows],
                         "text": [x for _, x in rows]}, columns=["class_id", "kind", "text"])


# --------------------------------------------------------------------------- prepared class


@dataclass
class Prepared:
    """Full-information arrays for one class (fast path)."""

    info: ClassInfo
    W: Words
    FR: Frames
    E: np.ndarray                     # frame embeddings (NaN rows for empty frames)
    dim: int
    tables: dict
    enc: Memo
    znull: ZNull
    pcw: PCW
    edu: object | None


def _ensure_class_row(tables: dict, cid) -> dict:
    if not len(table(tables, "classes")):
        tables = dict(tables)
        tables["classes"] = pd.DataFrame({"class_id": [int(cid)]})
    return tables


def prepare(cid, tables: dict, *, encoder=None, znull=None, edu=None,
            pcw_ref_sets=None) -> Prepared:
    full = _ensure_class_row(class_tables(tables, cid), cid)
    info = class_info(table(full, "classes"), cid)
    enc = Memo(encoder if encoder is not None else default_encoder())
    W = prep_words(table(full, "words"))
    FR = build_frames(W)
    if enc.dim is None:
        enc.one("dimension probe")
    dim = enc.dim or 0
    E = enc(list(FR.text)) if len(FR) else np.zeros((0, dim))
    zn = as_znull(znull)
    pcw = pcw_context(table(full, "pcw_blocks"), info.row.get("pcw_status"),
                      full.get("prep"), enc, pcw_ref_sets)
    return Prepared(info=info, W=W, FR=FR, E=E, dim=dim, tables=full, enc=enc, znull=zn,
                    pcw=pcw, edu=edu)


# --------------------------------------------------------------------------- fast groups


def _searched(sorted_vals: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.searchsorted(sorted_vals, t, side="right")


def row_str(row: dict, key: str, default: str | None = None) -> str | None:
    """A string field of the classes row (NA-safe)."""
    v = row.get(key)
    if v is None or pd.isna(v) or str(v) == "":
        return default
    return str(v)


def row_num(row: dict, key: str) -> float:
    v = pd.to_numeric(pd.Series([row.get(key)]), errors="coerce").iloc[0]
    return float(v) if pd.notna(v) else np.nan


def ctx_static(row: dict) -> dict:
    """Class-level ctx columns (known before class)."""
    sess = row_num(row, "session_no")
    code = row_str(row, "course_code")
    return {"ctx_course": code or UNKNOWN_CAT,
            "ctx_area": row_str(row, "course_area") or config.course_area(code),
            "ctx_sem_idx": row_num(row, "semester_idx"), "ctx_session_no": sess,
            "ctx_session_frac": sess / row_num(row, "n_sessions"),
            "ctx_dow": row_num(row, "weekday"), "ctx_hour_local": row_num(row, "hour_local"),
            "ctx_hour_utc": row_num(row, "hour_utc"), "ctx_enrolled": row_num(row, "enrolled")}


def _ctx(P: Prepared, t: np.ndarray) -> dict:
    info, row = P.info, P.info.row
    n = len(t)

    m = (t + info.rec_offset_s) / 60.0
    spl = ctx_splines(m)
    out = {"ctx_min": m}
    out.update({f"ctx_min_spl_{k}": spl[:, k] for k in range(5)})
    out.update({k: np.full(n, v, dtype=object if isinstance(v, str) else float)
                for k, v in ctx_static(row).items()})
    pr = table(P.tables, "presence")
    enter = np.sort(floats(pr["enter_s"])) if len(pr) else np.zeros(0)
    enter = enter[np.isfinite(enter)]
    out["ctx_n_present"] = _searched(enter + info.eps_f, t).astype(float)
    s = table(P.tables, "segments")
    s = s.sort_values("start_s", kind="stable") if len(s) else s
    st = floats(s["start_s"]) if len(s) else np.zeros(0)
    typ = strs(s["seg_type"]) if len(s) else np.array([], dtype=object)
    idx = _searched(st + info.eps_f, t) - 1
    ok = idx >= 0
    ci = np.clip(idx, 0, None)
    out["ctx_seg_type"] = np.where(ok, typ[ci] if len(typ) else UNKNOWN_CAT, UNKNOWN_CAT)
    out["ctx_seg_elapsed_s"] = np.where(ok, t - (st[ci] if len(st) else 0.0), np.nan)
    out["ctx_seg_idx"] = np.where(ok, idx, np.nan).astype(float)
    # breakout end = start of a non-breakout segment right after a breakout segment
    end_at = np.full(len(st), np.nan)
    last = np.nan
    for j in range(len(st)):
        if j and typ[j] != "breakout" and typ[j - 1] == "breakout":
            last = st[j]
        end_at[j] = last
    since = np.where(ok, end_at[ci] if len(st) else np.nan, np.nan)
    out["ctx_since_breakout_s"] = t - since
    out["ctx_after_breakout"] = np.isfinite(since).astype(float)
    p = table(P.tables, "polls")
    if len(p):
        op = floats(p["open_s"]) + info.eps_p
        cl = floats(p["close_s"]) + info.eps_p
        cl = np.where(np.isnan(cl), INF, cl)
        out["ctx_in_poll"] = ((op[None, :] <= t[:, None]) & (t[:, None] < cl[None, :])
                              ).any(axis=1).astype(float)
    else:
        out["ctx_in_poll"] = np.zeros(n)
    return out


@dataclass
class _Voice:
    vs: np.ndarray
    ve: np.ndarray
    role: np.ndarray
    spk: np.ndarray
    main: np.ndarray
    t1: np.ndarray
    t2: np.ndarray
    sa: np.ndarray   # start known at: inferred professor turns; NaN = start + eps_F (Forum)
    ea: np.ndarray   # end known at: inferred professor turns; NaN = end + eps_F (Forum)

    def subset(self, m: np.ndarray) -> _Voice:
        return _Voice(*(getattr(self, f.name)[m] for f in fields(self)))


def _avail(v: pd.DataFrame, col: str) -> np.ndarray:
    return floats(v[col]) if len(v) and col in v else np.full(len(v), np.nan)


def _voice(P: Prepared) -> _Voice:
    v = table(P.tables, "voice")
    if len(v):   # rows without a media time (recording-gap events) never count
        v = v[np.isfinite(floats(v["start_s"]))]
        v = v.sort_values("start_s", kind="stable").reset_index(drop=True)
    t1, t2 = brady_wb_times(v, table(P.tables, "words")) if len(v) else (np.zeros(0),) * 2
    return _Voice(vs=floats(v["start_s"]) if len(v) else np.zeros(0),
                  ve=floats(v["end_s"]) if len(v) else np.zeros(0),
                  role=strs(v["role"]) if len(v) else np.array([], dtype=object),
                  spk=strs(v["spk_hash"]) if len(v) else np.array([], dtype=object),
                  main=bools(v["room_main"]) if len(v) else np.zeros(0, bool), t1=t1, t2=t2,
                  sa=_avail(v, "start_avail_s"), ea=_avail(v, "end_avail_s"))


def _visible(V: _Voice, t: np.ndarray, ef: float) -> tuple[np.ndarray, np.ndarray]:
    """(visible, ended) matrices (points x main-room events)."""
    tt = t[:, None]
    ended = np.where(np.isnan(V.ea), V.ve + ef, V.ea)[None, :] <= tt
    started = np.where(np.isnan(V.sa), V.vs + ef, V.sa)[None, :] <= tt
    brady = (V.role == config.ROLE_ME)[None, :]
    wb = (tt >= V.t1[None, :]) | ((tt >= V.t2[None, :]) & ~ended)
    return started & (~brady | wb), ended


def _spk(P: Prepared, t: np.ndarray, V: _Voice) -> dict:
    ef = P.info.eps_f
    n = len(t)
    m = V.main
    vs, ve, role, spk = V.vs[m], V.ve[m], V.role[m], V.spk[m]
    Vm = V.subset(m)
    out = {c: np.full(n, np.nan) for c in GROUP_COLUMNS["spk"]}
    out["spk_cur_role"] = np.full(n, UNKNOWN_CAT, object)
    groups = table(P.tables, "breakout_groups")
    g_set = set(strs(groups["spk_hash"])) if len(groups) else set()
    g_end = breakout_ended_at(table(P.tables, "segments"))
    g_known_at = (g_end + ef) if (g_end is not None and g_set) else INF
    for a in range(0, n, 256):
        tc = t[a:a + 256]
        vis, ended = _visible(Vm, tc, ef)
        cur_m = vis & ~ended
        k = len(tc)
        sl = slice(a, a + k)
        if not len(vs):
            for c in ("spk_stud_turns_60", "spk_stud_turns_300", "spk_distinct_stud_300"):
                out[c][sl] = 0.0
            continue
        start_or_ninf = np.where(cur_m, vs[None, :], -INF)
        cur = np.argmax(start_or_ninf, axis=1)
        has = cur_m.any(axis=1)
        out["spk_cur_role"][sl] = np.where(has, role[cur], UNKNOWN_CAT)
        out["spk_cur_elapsed_s"][sl] = np.where(has, tc - vs[cur], np.nan)
        done = vis & ended
        last_end = np.where(done, ve[None, :], -INF).max(axis=1)
        out["spk_silence_s"][sl] = np.where(has, 0.0, np.where(done.any(axis=1),
                                                                tc - last_end, np.nan))
        ext_end = np.where(ended, ve[None, :], tc[:, None])
        for W in config.SPK_WINDOWS_S:
            ov = np.clip(np.minimum(ext_end, tc[:, None])
                         - np.maximum(vs[None, :], tc[:, None] - W), 0, None) * vis
            tot = ov.sum(axis=1)
            prof = (ov * (role == config.ROLE_PROF)[None, :]).sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                out[f"spk_prof_share_{W}"][sl] = np.where(tot > 0, prof / tot, np.nan)
            stud = vis & (role == config.ROLE_STUDENT)[None, :] & (vs[None, :] > tc[:, None] - W)
            out[f"spk_stud_turns_{W}"][sl] = stud.sum(axis=1)
            if W == 300:
                out["spk_distinct_stud_300"][sl] = [len(set(spk[r])) for r in stud]
        prof_vis = vis & (role == config.ROLE_PROF)[None, :]
        last_prof = np.where(prof_vis, vs[None, :], -INF).max(axis=1)
        since = (vis & (role != config.ROLE_PROF)[None, :]
                 & (vs[None, :] > last_prof[:, None])).sum(axis=1)
        out["spk_turns_since_prof"][sl] = np.where(prof_vis.any(axis=1), since, np.nan)
        known = tc >= g_known_at
        cur_spk, cur_role = spk[cur], role[cur]
        in_g = np.array([s in g_set for s in cur_spk]) & (cur_role != config.ROLE_ME)
        out["spk_cur_in_my_group"][sl] = np.where(known & has, in_g.astype(float), np.nan)
    return out


def _me(P: Prepared, t: np.ndarray, V: _Voice) -> dict:
    ef = P.info.eps_f
    n = len(t)
    FR = P.FR
    nc = _searched(FR.close, t)
    is_me = FR.role == config.ROLE_ME
    cum_words = np.concatenate([[0], np.cumsum(np.where(is_me, FR.n_words, 0))])
    last_me = np.full(len(FR) + 1, -1)
    for k in range(len(FR)):
        last_me[k + 1] = k if is_me[k] else last_me[k]
    lm = last_me[nc]
    out = {"me_words_so_far": cum_words[nc].astype(float),
           "me_since_turn_s": np.where(lm >= 0, t - FR.end[np.clip(lm, 0, None)]
                                       if len(FR) else np.nan, np.nan)}
    b = (V.role == config.ROLE_ME) & V.main
    vs, ve = V.vs[b], V.ve[b]
    turns = np.zeros(n)
    if len(vs):
        Vb = V.subset(b)
        vis, ended = _visible(Vb, t, ef)
        for i in range(n):
            idx = np.flatnonzero(vis[i])
            cnt = 0
            for j, e in enumerate(idx):
                if j == 0:
                    cnt = 1
                    continue
                p = idx[j - 1]
                if ended[i, p] and vs[e] - ve[p] >= config.TURN_MERGE_GAP_S:
                    cnt += 1
            turns[i] = cnt
    out["me_turns_so_far"] = turns
    c = table(P.tables, "chat")
    ct = floats(c["t_s"])[strs(c["role"]) == config.ROLE_ME] if len(c) else np.zeros(0)
    ct = np.sort(ct[np.isfinite(ct)])
    k = _searched(ct + ef, t)
    out["me_chat_n"] = k.astype(float)
    out["me_since_chat_s"] = np.where(k > 0, t - (ct[np.clip(k - 1, 0, None)] if len(ct)
                                                  else 0.0), np.nan)
    bo = (V.role == config.ROLE_ME) & ~V.main
    dur = (V.ve - V.vs)[bo]
    end_av = V.ve[bo] + ef
    out["me_breakout_talk_s"] = ((end_av[None, :] <= t[:, None]) * dur[None, :]).sum(axis=1) \
        if len(dur) else np.zeros(n)
    return out


def _me_hand(P: Prepared, t: np.ndarray) -> dict:
    """Brady's earlier raises this class (§4.1 A16) from his live UP runs: every value is
    decided by hand samples <= t (live exits, never merged label-side episodes or vote
    typing); the outcome is known at down + 5 s + eps_F (`causal.run_outcomes`)."""
    runs = hand_runs(table(P.tables, "hand_frames"))
    outc, av = run_outcomes(runs, table(P.tables, "voice"), table(P.tables, "words"),
                            P.info.eps_f)
    return me_hand_values(runs, outc, av, t)


@dataclass
class _Q:
    """Professor-utterance events (precomputed over the full word list)."""

    run_start: np.ndarray
    last_prof: np.ndarray      # last prof word index <= i (-1 if none)
    ev: np.ndarray             # evaluated utterance ends (gap >= 0.8 s)
    e_av: np.ndarray           # end + 0.8 + eps_w
    is_q: np.ndarray
    invite: np.ndarray
    kind: np.ndarray
    rh: np.ndarray             # professor resumes within 3 s
    rh_av: np.ndarray


def utterance_text(W: Words, a: int, b: int) -> str:
    """X_lastprof text of words a..b (all available), CALLON phrases removed."""
    return join_tokens(W.tok[a:b + 1])


def _q_prep(W: Words) -> _Q:
    n = len(W)
    prof = W.role == config.ROLE_PROF
    run_start = np.arange(n)
    last_prof = np.full(n, -1)
    for i in range(n):
        if prof[i] and i and prof[i - 1] and W.start[i] - W.end[i - 1] <= config.LASTPROF_PAUSE_S:
            run_start[i] = run_start[i - 1]
        last_prof[i] = i if prof[i] else (last_prof[i - 1] if i else -1)
    nxt = np.concatenate([W.start[1:], [INF]]) if n else np.zeros(0)
    gap = nxt - W.end
    ev = prof & (gap >= config.Q_SILENCE_S)
    is_q = np.zeros(n, bool)
    invite = np.zeros(n, bool)
    kind = np.full(n, "plain", object)
    for p in np.flatnonzero(ev):
        text = utterance_text(W, run_start[p], p)
        is_q[p] = is_question(text)
        kind[p] = lexical_kind(text)
        invite[p] = kind[p] == "invite"
    nxt_prof = np.concatenate([prof[1:], [False]]) if n else np.zeros(0, bool)
    rh = nxt_prof & (gap <= config.Q_RHETORICAL_RESUME_S)
    rh_av = np.concatenate([W.role_av[1:], [INF]]) if n else np.zeros(0)
    return _Q(run_start, last_prof, ev, W.end + config.Q_SILENCE_S + EPS_W, is_q, invite,
              kind, rh, rh_av)


def _q(P: Prepared, t: np.ndarray, Q: _Q) -> tuple[dict, np.ndarray]:
    """q_* columns plus p(t) (the last role-known prof word, -1 if none)."""
    W, FR = P.W, P.FR
    n = len(t)
    k_r = _searched(W.role_av, t)
    p = np.where(k_r > 0, Q.last_prof[np.clip(k_r - 1, 0, None)] if len(W) else -1, -1)
    pc = np.clip(p, 0, None)
    has_p = p >= 0
    evaluated = has_p & (Q.ev[pc] if len(W) else False) & (Q.e_av[pc] <= t if len(W) else False)
    nc = _searched(FR.close, t)
    stud = np.isin(FR.role, [config.ROLE_STUDENT, config.ROLE_ME]) & \
        (FR.n_words >= config.Q_OPEN_STUDENT_MIN_WORDS)
    cum_stud = np.concatenate([[0], np.cumsum(stud)])
    lo = np.searchsorted(FR.start, W.end[pc], side="right") if len(W) else np.zeros(n, int)
    answered = cum_stud[nc] - cum_stud[np.minimum(lo, nc)] > 0
    out = {"q_open": (evaluated & Q.is_q[pc] & ~answered).astype(float) if len(W)
           else np.zeros(n)}
    av = np.maximum(Q.e_av, W.role_av) if len(W) else np.zeros(0)
    qev = np.flatnonzero(Q.ev & Q.is_q)
    iev = np.flatnonzero(Q.ev & Q.invite)
    cap = config.Q_AGE_CAP_S

    def age(events):
        if not len(events):
            return np.full(n, cap), np.full(n, -1)
        order = events[np.argsort(av[events], kind="stable")]
        av_s, end_s = av[order], W.end[order]
        run = np.maximum.accumulate(end_s)
        arg, best = np.empty(len(order), int), 0      # running argmax (first maximum)
        for k in range(len(order)):
            if end_s[k] > end_s[best]:
                best = k
            arg[k] = order[best]
        k = _searched(av_s, t) - 1
        ok = k >= 0
        last_end = np.where(ok, run[np.clip(k, 0, None)], -INF)
        return np.minimum(cap, t - last_end), np.where(ok, arg[np.clip(k, 0, None)], -1)
    out["q_age_s"], last_q = age(qev)
    out["q_inv_age_s"], _ = age(iev)
    if len(qev):
        cnt = (av[qev][None, :] <= t[:, None]) & (W.end[qev][None, :] > t[:, None]
                                                    - config.Q_COUNT_WINDOW_S)
        out["q_count_300"] = cnt.sum(axis=1).astype(float)
    else:
        out["q_count_300"] = np.zeros(n)
    lq = np.clip(last_q, 0, None)
    kinds = np.where(last_q >= 0, Q.kind[lq] if len(W) else UNKNOWN_CAT, UNKNOWN_CAT)
    rhet = (last_q >= 0) & (kinds == "plain") & (Q.rh[lq] if len(W) else False) & \
        (Q.rh_av[lq] <= t if len(W) else False)
    out["q_kind"] = np.where(rhet, "rhetorical", kinds).astype(object)
    cm_end = np.maximum.accumulate(W.end) if len(W) else np.zeros(0)
    vote = np.zeros(n)
    prof = W.role == config.ROLE_PROF
    for i in range(n):
        lo_i = np.searchsorted(cm_end, t[i] - config.WIN_LONG_S, side="right")
        hi = k_r[i]
        if hi > lo_i:
            sel = np.arange(lo_i, hi)
            sel = sel[prof[sel] & (W.end[sel] > t[i] - config.WIN_LONG_S) & (W.tok[sel] != "")]
            vote[i] = float(bool(config.VOTE_RE.search(join_tokens(W.tok[sel]))))
    out["q_vote_lex"] = vote
    out["has_qmark"] = (evaluated & (W.has_qmark[pc] if len(W) else False)).astype(float)
    return out, p


def _text_windows(P: Prepared, t: np.ndarray) -> dict:
    W = P.W
    n = len(t)
    k_t, k_r = _searched(W.text_av, t), _searched(W.role_av, t)
    cm_end = np.maximum.accumulate(W.end) if len(W) else np.zeros(0)
    out = {c: np.full(n, np.nan) for c in ("n_words_15", "n_words_60", "wpm_60",
                                           "filler_rate_60", "frac_unk_60")}
    query = np.full(n, "", object)
    for i in range(n):
        for Wd in (config.WIN_SHORT_S, config.WIN_LONG_S):
            lo = np.searchsorted(cm_end, t[i] - Wd, side="right")
            sel = np.arange(lo, max(lo, k_t[i]))
            sel = sel[(W.end[sel] > t[i] - Wd) & (W.tok[sel] != "")]
            sel = sel[keep_mask(W.tok[sel])]              # CALLON among available words
            if Wd == config.WIN_SHORT_S:
                out["n_words_15"][i] = len(sel)
                continue
            out["n_words_60"][i] = len(sel)
            query[i] = " ".join(W.tok[sel])
            if len(sel):
                dur = float(np.sum(W.end[sel] - W.start[sel]))
                out["wpm_60"][i] = 60.0 * len(sel) / dur if dur > 0 else np.nan
                out["filler_rate_60"][i] = float(W.filler[sel].mean())
                unk = (sel >= k_r[i]) | (W.role[sel] == config.ROLE_UNK)
                out["frac_unk_60"][i] = float(unk.mean())
    out["_query"] = query
    return out


def _embeddings(P: Prepared, t: np.ndarray, Q: _Q, p: np.ndarray, query: np.ndarray
                ) -> dict:
    """Pooled vectors per point and the embedding-based features."""
    W, FR, E, dim, enc, zn = P.W, P.FR, P.E, P.dim, P.enc, P.znull
    n = len(t)
    nc = _searched(FR.close, t)
    k_t, k_r = _searched(W.text_av, t), _searched(W.role_av, t)
    nonme = (FR.role != config.ROLE_ME) & (FR.n_words > 0)
    is_me = (FR.role == config.ROLE_ME) & (FR.n_words > 0)
    me_idx = np.flatnonzero(is_me)
    spent_at = chunk_spent_times(P.pcw, E[me_idx], FR.close[me_idx], zn)
    emb = np.full((n, 4 * dim), np.nan)
    out = {c: np.full(n, np.nan) for c in ("topic_shift", "own_sim", "poll_res_my")}
    dyn_cols = ("pcw_res_Q", "pcw_block_pos_Q", "lo_sim", "sg_sim", "pcw_res_A",
                "pcw_res_gap", "pcw_unspent", "pcw_res_ref", "pcw_bm25_gap", "pcw_invest",
                "pcw_block_pos")
    out.update({c: np.full(n, np.nan) for c in dyn_cols})
    polls = table(P.tables, "polls")
    ans_av = floats(polls["my_answer_avail_s"]) if len(polls) else np.zeros(0)
    if len(polls):
        ans_av = np.where(np.isnan(ans_av), np.nan,
                          np.fmax(ans_av, floats(polls["open_s"]) + P.info.eps_p))
    ans_txt = [norm_text(x) for x in strs(polls["my_answer_text"])] if len(polls) else []
    last_nonme = np.full(len(FR) + 1, -1)
    for j in range(len(FR)):
        last_nonme[j + 1] = j if nonme[j] else last_nonme[j]
    # pass 1: the tail and X_lastprof text of every point, encoded in one batch
    tail_txts, lp_txts = [], []
    for i in range(n):
        k = nc[i]
        tail_lo = FR.i1[k - 1] + 1 if k else 0
        tsel = np.arange(tail_lo, max(tail_lo, k_t[i]))
        tsel = tsel[(W.tok[tsel] != "") & ~((tsel < k_r[i]) & (W.role[tsel] == config.ROLE_ME))]
        tail_txts.append(join_tokens(W.tok[tsel]))
        lp_txts.append(utterance_text(W, Q.run_start[p[i]], p[i]) if p[i] >= 0 else "")
    enc(sorted({x for x in tail_txts + lp_txts if x}))
    for i in range(n):
        ti, k = t[i], nc[i]
        tail_txt = tail_txts[i]
        tail = enc.one(tail_txt) if tail_txt else None
        lo180 = np.searchsorted(FR.close, ti - config.POOL_HORIZON_S, side="right")
        f = np.arange(lo180, k)
        f = f[nonme[f]]
        v15 = pool(E[f], FR.close[f], ti, config.EMB_TAUS_S[0], tail)
        v60 = pool(E[f], FR.close[f], ti, config.EMB_TAUS_S[1], tail)
        lnm = last_nonme[k]
        e_last = tail if tail is not None else (E[lnm] if lnm >= 0 else np.full(dim, np.nan))
        lp_txt = lp_txts[i]
        e_lp = enc.one(lp_txt) if lp_txt else np.full(dim, np.nan)
        emb[i] = np.concatenate([v15, v60, e_last, e_lp])
        x = v60
        mu, sd = zn.stats(x)
        mu, sd = float(mu[0]), float(sd[0])
        lo60 = np.searchsorted(FR.close, ti - config.WIN_LONG_S, side="right")
        lo300 = np.searchsorted(FR.close, ti - 300.0, side="right")
        rec = np.arange(lo60, k)
        rec = rec[nonme[rec]]
        old = np.arange(lo300, min(lo60, k))
        old = old[nonme[old]]
        rows = [E[rec]] + ([tail[None, :]] if tail is not None else [])
        recent = np.concatenate(rows) if sum(len(r) for r in rows) else np.zeros((0, dim))
        out["topic_shift"][i] = 1.0 - cos(mean_unit(recent, dim), mean_unit(E[old], dim))
        mine = me_idx[me_idx < k]
        if len(mine):
            out["own_sim"][i] = zsim(x, mu, sd, mean_unit(E[mine], dim)[None, :])[0]
        av = np.flatnonzero(ans_av <= ti)
        if len(av):
            j = av[np.argmax(ans_av[av])]
            if ans_txt[j]:
                out["poll_res_my"][i] = zsim(x, mu, sd, enc.one(ans_txt[j])[None, :])[0]
        dyn = pcw_dynamic(P.pcw, x, mu, sd, spent_at <= ti, query[i], zn)
        for c in dyn_cols:
            out[c][i] = dyn[c]
    out["_emb"] = emb
    return out


def _poll(P: Prepared, t: np.ndarray) -> dict:
    n = len(t)
    p = table(P.tables, "polls")
    out = {"poll_n_so_far": np.zeros(n), "poll_since_close_s": np.full(n, np.nan),
           "poll_my_ans_words": np.full(n, np.nan)}
    if not len(p):
        return out
    ep = P.info.eps_p
    op = np.sort(floats(p["open_s"]) + ep)
    out["poll_n_so_far"] = _searched(op, t).astype(float)
    cl = floats(p["close_s"])
    cl_av = cl + ep
    done = cl_av[None, :] <= t[:, None]
    last = np.where(done, cl[None, :], -INF).max(axis=1)
    out["poll_since_close_s"] = np.where(done.any(axis=1), t - last, np.nan)
    av = np.fmax(floats(p["my_answer_avail_s"]), floats(p["open_s"]) + ep)
    av = np.where(np.isnan(floats(p["my_answer_avail_s"])), np.nan, av)
    words = np.array([len(norm_text(x).split()) for x in strs(p["my_answer_text"])], float)
    known = av[None, :] <= t[:, None]
    j = np.argmax(np.where(known, av[None, :], -INF), axis=1)
    out["poll_my_ans_words"] = np.where(known.any(axis=1), words[j], np.nan)
    return out


def _edu(P: Prepared, t: np.ndarray) -> dict:
    src = P.edu
    if src is None or not getattr(src, "columns", ()):
        return {}
    FR = P.FR
    prof = np.flatnonzero((FR.role == config.ROLE_PROF) & (FR.n_words > 0))
    S = src.scores(list(FR.text[prof])) if len(prof) else np.zeros((0, len(src.columns)))
    closes = FR.close[prof]
    k = _searched(closes, t - EPS_W)
    cols = tuple(src.columns)                  # read after scoring (a checkpoint may drop)
    out = {c: np.zeros(len(t)) for c in cols}
    for i in range(len(t)):
        vals = edu_decayed(closes[:k[i]], S[:k[i]], t[i])
        for j, c in enumerate(cols):
            out[c][i] = vals[j]
    return out


# --------------------------------------------------------------------------- build


def assemble(cid, t: np.ndarray, cols: dict, emb_rows) -> pd.DataFrame:
    """The features frame: keys, then stored columns in group order (float32 /
    string), then any edu_* columns."""
    df = pd.DataFrame({"class_id": np.full(len(t), int(cid), dtype=np.int64), "t_s": t,
                       "emb_row": pd.array(emb_rows, dtype="Int64")})
    order = [c for c in STORED_COLUMNS if c in cols] + [c for c in EDU_COLUMNS if c in cols]
    data = {}
    for c in order:
        if c in CATEGORICAL:
            data[c] = pd.array(np.asarray(cols[c], dtype=object).astype(str), dtype="string")
        else:
            data[c] = np.asarray(cols[c], dtype=float).astype(np.float32)
    return pd.concat([df, pd.DataFrame(data)], axis=1)


def build(cid, points, tables: dict | None = None, *, encoder=None, znull=None, edu=None,
          pcw_ref_sets=None, return_emb: bool = False):
    """Every stored feature at every point of class `cid` (fast path).

    `points`: a points table (rows of this class are used) or an array of t.
    `encoder`: texts -> vectors (default: oncue.embed bge-small); `znull`: a ZNull, an
    `oncue.embed.ZNull` or the reference matrix R_c (see `znull_reference`); `edu`: an
    `oncue.edu.EduScorer` or `EduTable` (A3; the group is omitted when None and a skipped
    checkpoint's column is absent); `pcw_ref_sets`: earlier same-section answer-chunk
    embeddings (`pcw_ref_sets()`).
    Returns the features frame (and the (n, 4*dim) pooled-embedding matrix with
    `return_emb`, indexed by `emb_row`).
    """
    if isinstance(points, pd.DataFrame):
        pts = points[points["class_id"] == int(cid)] if "class_id" in points else points
        t = floats(pts["t_s"])
    else:
        t = np.asarray(points, dtype=float)
    tables = tables if tables is not None else load_tables(class_ids=[cid])
    P = prepare(cid, tables, encoder=encoder, znull=znull, edu=edu, pcw_ref_sets=pcw_ref_sets)
    V = _voice(P)
    Q = _q_prep(P.W)
    cols: dict = {}
    cols.update(_ctx(P, t))
    cols.update(_spk(P, t, V))
    cols.update(_me(P, t, V))
    cols.update(_me_hand(P, t))
    cols.update(_poll(P, t))
    q, p_t = _q(P, t, Q)
    cols.update(q)
    tw = _text_windows(P, t)
    query = tw.pop("_query")
    cols.update(tw)
    em = _embeddings(P, t, Q, p_t, query)
    emb = em.pop("_emb")
    cols.update(em)
    cols.update({k: np.full(len(t), v) for k, v in pcw_static(P.pcw).items()})
    cols.update(_edu(P, t))
    df = assemble(cid, t, cols, np.arange(len(t)) if return_emb else [pd.NA] * len(t))
    return (df, emb.astype(np.float32)) if return_emb else df


# --------------------------------------------------------------------------- in-fold helpers


def pcw_new_block_threshold(train: pd.DataFrame) -> float:
    """Training-fold 90th percentile of pcw_res_Q (fit in-fold only)."""
    vals = floats(train["pcw_res_Q"])
    vals = vals[np.isfinite(vals)]
    return float(np.percentile(vals, config.PCW_NEW_BLOCK_PCTL)) if len(vals) else np.nan


def pcw_new_block(features: pd.DataFrame, threshold: float) -> pd.Series:
    """1 the first time b*_Q switches to a block whose rho_Q >= threshold (per class, in
    time order over the full grid); 0 otherwise; NaN where pcw_res_Q is NaN. Uses only the
    current and earlier points of the class."""
    out = pd.Series(np.nan, index=features.index, dtype=float)
    for _, g in features.sort_values(["class_id", "t_s"]).groupby("class_id", sort=False):
        pos, rho = floats(g["pcw_block_pos_Q"]), floats(g["pcw_res_Q"])
        seen, prev = set(), np.nan
        vals = np.full(len(g), np.nan)
        for i in range(len(g)):
            if not np.isfinite(rho[i]):
                prev = pos[i]
                continue
            new = (pos[i] != prev) and rho[i] >= threshold and pos[i] not in seen
            if new:
                seen.add(pos[i])
            vals[i] = float(new)
            prev = pos[i]
        out.loc[g.index] = vals
    return out


def hist_features(classes: pd.DataFrame, points: pd.DataFrame, train_class_ids,
                  episodes: pd.DataFrame | None = None) -> pd.DataFrame:
    """A2 cross-class history, fold-aware: for class c, Brady's bid rate per at-risk minute
    over TRAINING-fold classes that started before c (same course, and overall). Test-fold
    labels never enter; NaN when there is no earlier training class (or no at-risk time)."""
    train = {int(c) for c in train_class_ids}
    start = _class_start(classes)
    cls = classes.assign(start_utc=start.to_numpy())
    pts = points[points["class_id"].isin(train)]
    minutes = (pts.groupby("class_id")["at_risk"].sum() * config.DELTA_S / 60.0)
    if episodes is not None and len(episodes):
        e = episodes[episodes["class_id"].isin(train)]
        is_bid = (strs(e["ep_type"]) == config.EP_BID) & ~bools(e["left_censored"])
        bids = e[is_bid].groupby("class_id").size()
    else:
        bids = cls.set_index("class_id")["n_bids"].astype(float)
    stats = pd.DataFrame({"minutes": minutes}).join(bids.rename("bids"), how="left")
    stats["bids"] = stats["bids"].fillna(0.0)
    stats = stats.join(cls.set_index("class_id")[["start_utc", "course_code"]], how="left")
    stats = stats[stats["minutes"] > 0]          # no at-risk time (e.g. failed QA): no rate
    rows = []
    for c in cls.itertuples():
        prior = stats[(stats["start_utc"] < c.start_utc) & (stats.index != c.class_id)]
        same = prior[strs(prior["course_code"]) == str(c.course_code)]

        def rate(df):
            m = float(df["minutes"].sum())
            return float(df["bids"].sum()) / m if m > 0 else np.nan
        rows.append({"class_id": int(c.class_id), "hist_bid_rate_course": rate(same),
                     "hist_bid_rate_all": rate(prior)})
    return pd.DataFrame(rows, columns=["class_id", *config.HIST_COLUMNS])


def add_hist(features: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    """Join hist_* onto a features frame by class_id (float32)."""
    out = features.drop(columns=[c for c in config.HIST_COLUMNS if c in features],
                        errors="ignore").merge(hist, on="class_id", how="left")
    for c in config.HIST_COLUMNS:
        out[c] = out[c].astype(np.float32)
    return out


# --------------------------------------------------------------------------- CLI


def _znull_builder(classes: pd.DataFrame, blocks: pd.DataFrame, enc):
    """cid -> ZNull, via oncue.pcw.reference_pool + oncue.embed.znull when available."""
    try:
        from . import embed as E
        from . import pcw as P

        pool = P.reference_pool(blocks, classes)
        return lambda cid: as_znull(E.znull(cid, pool, enc, classes=classes))
    except (ImportError, AttributeError):  # pragma: no cover - other stages' modules
        return lambda cid: znull_reference(cid, classes, blocks, enc)


def _prep_from_raw(cid) -> pd.DataFrame:
    """prep rows from the whitelisted raw class_edit_page (empty if unavailable)."""
    try:
        from .loaders import load_raw

        page = load_raw(cid, "class_edit_page")
    except (ImportError, OSError, ValueError) as exc:  # pragma: no cover
        log.warning("features: class %s: no class_edit_page (%s)", cid, exc)
        page = None
    return prep_table(cid, page)


def cli(args) -> int:
    """`python -m oncue features [--class CID ...]`: features.parquet + pooled embeddings."""
    import argparse

    ap = argparse.ArgumentParser(prog="oncue features")
    ap.add_argument("--no-edu", action="store_true", help="skip EduBehaviors scores (A3)")
    ap.add_argument("--model", default=None,
                    help=f"encoder (default {config.EMB}; oncue/hash-bow = offline dry run)")
    opts = ap.parse_args(args.extra or [])
    tables = load_tables(CLASS_TABLES + ("points",), class_ids=args.class_ids)
    pts = table(tables, "points")
    classes = table(tables, "classes")
    if not len(pts) or not len(classes):
        log.error("features: need classes and points tables (run `oncue labels`)")
        return 1
    enc = default_encoder(opts.model) if opts.model else default_encoder()
    edu = None
    if not opts.no_edu:
        from .edu import EduScorer

        edu = EduScorer()
    all_pcw = load_tables(("classes", "pcw_blocks"))
    all_classes, all_blocks = table(all_pcw, "classes"), table(all_pcw, "pcw_blocks")
    znull_for = _znull_builder(all_classes, all_blocks, enc)
    frames, embs, offset = [], [], 0
    with_points = set(pts["class_id"].astype(int))       # labels wrote the population
    for cid in classes["class_id"].astype(int):
        if cid not in with_points:
            continue
        ct = class_tables(tables, cid)
        if not len(table(ct, "prep")):
            ct["prep"] = _prep_from_raw(cid)
        refs = pcw_ref_sets(cid, all_classes, all_blocks, enc)
        df, emb = build(cid, pts, ct, encoder=enc, znull=znull_for(cid), edu=edu,
                        pcw_ref_sets=refs, return_emb=True)
        df["emb_row"] = pd.array(np.arange(len(df)) + offset, dtype="Int64")
        offset += len(df)
        frames.append(df)
        embs.append(emb)
    out = schemas.validate(pd.concat(frames, ignore_index=True), "features")
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(config.table_path("features"), index=False)
    np.save(config.TABLES_DIR / "features_emb.npy", np.concatenate(embs))
    print(f"features: {len(out):,} rows x {len(schemas.feature_columns(out))} columns")
    return 0
