"""Text: name masking, normalisation, causal frames, windows at t, question detector, rule R.

Spec §5.1 (masking), §5.2 (causal frames and `close_s`), §5.3 (normalisation and windows),
§5.5 (professor-question detector and the rule baseline R). Every function here reads only
what its arguments hold; nothing is written to disk except by `cli_frames`.

Causality conventions (§4):
* word **text** is available at `end_s + ε_w`; a word **onset** (no text) at `start_s + ε_w`;
* a word's speaker/role is known at `role_avail_s` (it reads "unk" before);
* frames are usable once `close_s <= t`; the tail is the available words of unclosed frames.

CALLON phrases are deleted from *joined* text only (frames, windows, utterances), never by
looking at a later word: per-word `text_norm` deletes names and one-word CALLONs ("unmute"),
and every join runs `delete_callon` over text that is already available. Deleting "go" because
"ahead" follows would hide a word before the phrase is observable.

Roster names exist only in memory (`RosterPatterns` never prints them). The D11 audit output
(`mask_audit`) contains surface forms and must stay local (never results/, figures/, README).
"""

from __future__ import annotations

import argparse
import functools
import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, schemas

# --------------------------------------------------------------------------- tokens

NAME_FEATURE = "_name_"     # F-name feature-text form of [NAME] (survives \w tokenisers)
ME_FEATURE = "_me_"         # F-name feature-text form of [ME]
_MASK_TOKEN_RE = re.compile(r"\[(?:name|me)\]", re.IGNORECASE)
_ME_TOKEN_RE = re.compile(r"\[me\]", re.IGNORECASE)
_APOS = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "´": "'"})
_EDGE = re.compile(r"^(\W*)(.*?)(\W*)$", re.DOTALL)
_POSSESSIVE = re.compile(r"'s$", re.IGNORECASE)
_SENT_END = re.compile(r"[.?!…][\"')\]”]*$")
_NONWORD = re.compile(r"[^\w\s]|_")
_PUNCT_ONLY_TOKEN = re.compile(r"(?<!\S)[^\w\s]+(?!\S)")
_WS = re.compile(r"\s+")


class MaskingError(RuntimeError):
    """The D11 build check failed: a common English word would be masked as a name."""


@functools.lru_cache(maxsize=4)
def common_words(n: int = config.COMMON_WORDS_TOPN) -> frozenset[str]:
    """wordfreq's top-n English words plus Scribe's `fixup.COMMON_WORDS` (lowercase)."""
    from scribe_core.fixup import COMMON_WORDS
    from wordfreq import top_n_list

    return frozenset(top_n_list("en", n)) | frozenset(COMMON_WORDS)


def _split_token(token: str) -> tuple[str, str, str]:
    """(leading punctuation, core without a possessive 's, trailing punctuation)."""
    m = _EDGE.match(str(token).translate(_APOS))
    lead, core, trail = (m.group(1), m.group(2), m.group(3)) if m else ("", token, "")
    return lead, _POSSESSIVE.sub("", core), trail


def _sentence_initial(prev: str | None) -> bool:
    return prev is None or bool(_SENT_END.search(str(prev).translate(_APOS)))


# --------------------------------------------------------------------------- roster (§5.1)


def _name_keys(name: str | None) -> set[str]:
    """Lowercase match keys for one name field: each word, hyphen parts, apostrophe-free."""
    keys: set[str] = set()
    for word in str(name or "").translate(_APOS).split():
        core = _split_token(word)[1].lower()
        for k in {core, core.replace("'", ""), *core.split("-")}:
            if len(k) >= 2:
                keys.add(k)
    return keys


@dataclass(frozen=True)
class RosterPatterns:
    """Name keys for masking one class. `repr` never shows names (they are USE_MASK only)."""

    names: frozenset[str] = field(repr=False)
    me_names: frozenset[str] = field(repr=False)
    common: frozenset[str] = field(repr=False)
    lev_names: tuple[str, ...] = field(repr=False)

    @property
    def all_names(self) -> frozenset[str]:
        return self.names | self.me_names

    def __repr__(self) -> str:   # counts only
        return f"RosterPatterns(n_names={len(self.names)}, n_me={len(self.me_names)})"


def _roster_entries(roster) -> list[dict]:
    """Normalise a class JSON, a scribe roster list, a DataFrame or plain names to dicts."""
    if roster is None:
        return []
    if isinstance(roster, pd.DataFrame):
        return roster.to_dict("records")
    if isinstance(roster, dict):
        from scribe_core.asr_deepgram import roster as scribe_roster

        return scribe_roster(roster.get("class", roster))
    out = []
    for r in roster:
        out.append({"name": r} if isinstance(r, str) else dict(r))
    return out


def roster_patterns(roster=None, *, me_id=None, me_names: Iterable[str] = config.BRADY_VARIANTS,
                    extra_names: Iterable[str] = (), common: frozenset[str] | None = None
                    ) -> RosterPatterns:
    """Masking patterns from `class-users` (class JSON), roster dicts or a DataFrame.

    * every first, last and preferred name (and Scribe keyterm spellings via `extra_names`);
    * Brady's variants (`config.BRADY_VARIANTS`) plus the roster entry whose id is `me_id`
      (or whose role is "me") form `me_names`;
    * Levenshtein targets are names of >= 4 characters (the token side must be capitalised,
      mid-sentence and not a common word; see `mask`).
    """
    common = common_words() if common is None else frozenset(common)
    names: set[str] = set()
    mine: set[str] = set()
    for name in me_names:
        mine |= _name_keys(name)
    for r in _roster_entries(roster):
        keys: set[str] = set()
        for k in ("first", "last", "preferred", "name", "first_name", "last_name"):
            keys |= _name_keys(r.get(k))
        uid = r.get("user_id", r.get("id"))
        is_me = (me_id is not None and uid is not None and str(uid) == str(me_id)) \
            or r.get("role") == config.ROLE_ME
        (mine if is_me else names).update(keys)
    for name in extra_names:
        names |= _name_keys(name)
    names -= mine
    lev = tuple(sorted(k for k in names | mine if len(k) >= config.MASK_LEV_MIN_LEN))
    return RosterPatterns(frozenset(names), frozenset(mine), common, lev)


# --------------------------------------------------------------------------- masking


def _lev_match(key: str, pats: RosterPatterns) -> str | None:
    from scribe_core.text import levenshtein

    best = None
    for name in pats.lev_names:
        if abs(len(name) - len(key)) <= config.MASK_LEV_MAX \
                and levenshtein(key, name) <= config.MASK_LEV_MAX:
            if name in pats.me_names:
                return name
            best = best or name
    return best


def _mask_one(tok: str, pats: RosterPatterns, common: frozenset[str], initial: bool,
              me_token: str, vocative: bool) -> tuple[str, str | None]:
    """(masked token, kind) with kind "me" | "name" | None for one token."""
    lead, core, trail = _split_token(tok)
    key = core.lower()
    if not key or _MASK_TOKEN_RE.search(tok):
        return tok, None
    cap = core[:1].isupper()
    bare = key.replace("'", "")
    if key in pats.me_names or bare in pats.me_names:
        return f"{lead}{me_token}{trail}", "me"
    if key in pats.names or bare in pats.names:
        if key not in common or (cap and not initial) \
                or (vocative and cap and initial and trail.startswith(",")):
            return f"{lead}{config.NAME_TOKEN}{trail}", "name"
        return tok, None
    if cap and not initial and len(key) >= config.MASK_LEV_MIN_LEN and key not in common:
        hit = _lev_match(key, pats)
        if hit is not None:
            if hit in pats.me_names:
                return f"{lead}{me_token}{trail}", "me"
            return f"{lead}{config.NAME_TOKEN}{trail}", "name"
    return tok, None


def mask_with_kinds(tokens: Sequence[str], roster, common_words: frozenset[str] | None = None,
                    *, me_token: str = config.NAME_TOKEN, vocative: bool = True,
                    initial: Sequence[bool] | None = None) -> tuple[list[str], list[str | None]]:
    """`mask` plus, per token, the kind of mask applied ("me", "name" or None)."""
    pats = roster if isinstance(roster, RosterPatterns) else roster_patterns(roster)
    common = pats.common if common_words is None else frozenset(common_words)
    out: list[str] = []
    kinds: list[str | None] = []
    prev: str | None = None
    for i, tok in enumerate(tokens):
        tok = "" if tok is None else str(tok)
        init = bool(initial[i]) if initial is not None else _sentence_initial(prev)
        m, kind = _mask_one(tok, pats, common, init, me_token, vocative)
        out.append(m)
        kinds.append(kind)
        prev = tok
    return out, kinds


def mask(tokens: Sequence[str], roster, common_words: frozenset[str] | None = None, *,
         me_token: str = config.NAME_TOKEN, vocative: bool = True,
         initial: Sequence[bool] | None = None) -> list[str]:
    """Mask names in cased, punctuated tokens (Deepgram `punctuated_word`), per §5.1.

    1. strip surrounding punctuation and a possessive 's / s';
    2. exact roster match (case-insensitive) -> [NAME]; a name that is a common word is
       masked only when capitalised and not sentence-initial (or, with `vocative`, when it
       is sentence-initial, capitalised and directly followed by a comma: "Grace, go ahead");
    3. Levenshtein <= 1 only for capitalised, mid-sentence, non-common tokens of >= 4 chars;
    4. Brady's variants are always masked, as `me_token` ([NAME] by default, [ME] in F-name),
       so the call-on "Brady, go ahead" never survives even though "brady" is a common word.

    A token is sentence-initial when it is first or the previous token ends in . ? ! ...;
    `initial` (one bool per token) overrides that, e.g. to also mark speaker changes.
    The mask keeps surrounding punctuation ("Grace's," -> "[NAME],") and drops possessives.
    """
    return mask_with_kinds(tokens, roster, common_words, me_token=me_token,
                           vocative=vocative, initial=initial)[0]


class TextMasker:
    """`mask_tokens(tokens, initial) -> (masked, kinds)` for `events.masker_for` and
    `transcribe.words_table` (kinds: "me" | "name" | None per token)."""

    def __init__(self, roster, *, me_id=None, me_token: str = config.NAME_TOKEN,
                 extra_names: Iterable[str] = (), vocative: bool = True):
        self.patterns = roster if isinstance(roster, RosterPatterns) else roster_patterns(
            roster, me_id=me_id, extra_names=extra_names)
        self.me_token, self.vocative = me_token, vocative

    def __repr__(self) -> str:
        return f"TextMasker({self.patterns!r}, me_token={self.me_token!r})"

    def mask_tokens(self, tokens: Sequence[str], initial: Sequence[bool] | None = None
                    ) -> tuple[list[str], list[str | None]]:
        return mask_with_kinds(tokens, self.patterns, me_token=self.me_token,
                               vocative=self.vocative, initial=initial)

    def mask_text(self, text: str | None) -> str:
        return mask_text(text, self.patterns, me_token=self.me_token)


def mask_text(text: str | None, roster, common_words: frozenset[str] | None = None, *,
              me_token: str = config.NAME_TOKEN) -> str:
    """`mask` for free text (chat, timeline titles, poll prompts); whitespace is collapsed."""
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""
    return " ".join(mask(str(text).split(), roster, common_words, me_token=me_token))


def mask_audit(original: Sequence[str], masked: Sequence[str], roster, *,
               top_n: int = config.MASK_FAIL_TOPN) -> dict:
    """D11 numbers for one class. LOCAL ONLY: `top_forms` holds real surface forms.

    Returns n_tokens, n_masked, masked_rate, top_forms (20 most frequent masked forms) and
    bad_forms: masked forms that are not exact roster names but are top-`top_n` words.
    """
    from wordfreq import top_n_list

    pats = roster if isinstance(roster, RosterPatterns) else roster_patterns(roster)
    forms: Counter = Counter()
    for o, m in zip(original, masked, strict=True):
        if _MASK_TOKEN_RE.search(str(m)) and not _MASK_TOKEN_RE.search(str(o)):
            forms[_split_token(str(o))[1].lower()] += 1
    top = frozenset(top_n_list("en", top_n))
    bad = sorted(f for f in forms if f not in pats.all_names and f in top)
    n = len(original)
    return {"n_tokens": n, "n_masked": int(sum(forms.values())),
            "masked_rate": (sum(forms.values()) / n) if n else 0.0,
            "top_forms": forms.most_common(20), "bad_forms": bad}


def assert_mask_safe(original: Sequence[str], masked: Sequence[str], roster) -> dict:
    """Fail the build (MaskingError) if a non-roster masked form is a top-1,000 word."""
    audit = mask_audit(original, masked, roster)
    if audit["bad_forms"]:
        raise MaskingError(f"{len(audit['bad_forms'])} common-word surface form(s) masked "
                           "as names (inspect mask_audit locally)")
    return audit


# --------------------------------------------------------------------------- normalisation (§5.3)


def delete_callon(text: str | None) -> str:
    """Remove every CALLON phrase (`config.CALLON_RE`) and collapse whitespace."""
    if not text:
        return ""
    return _WS.sub(" ", config.CALLON_RE.sub(" ", str(text))).strip()


def normalise(text: str | None, *, keep_names: bool = False, punct: bool = False,
              callon: bool = True) -> str:
    """Feature text from masked text: lowercase, no punctuation, names and CALLON deleted.

    * `keep_names` (F-name): [NAME] -> "_name_", [ME] -> "_me_" instead of deleting them;
    * `punct` (F-punct): keep Deepgram's casing and punctuation (names/CALLON still deleted);
    * `callon=False`: keep CALLON phrases (diagnostics only, never feature text).
    """
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""
    toks: list[str] = []
    for tok in str(text).translate(_APOS).split():
        if _MASK_TOKEN_RE.search(tok):
            if keep_names:
                toks.append(ME_FEATURE if _ME_TOKEN_RE.search(tok) else NAME_FEATURE)
            continue
        if punct:
            toks.append(tok)
            continue
        t = _NONWORD.sub(" ", tok.lower().replace("'", "")).strip()
        if t:
            toks.append(t)
    s = " ".join(toks)
    if callon:
        s = delete_callon(s)
    if punct:
        s = _WS.sub(" ", _PUNCT_ONLY_TOKEN.sub(" ", s)).strip()
    return s


def normalise_words(masked_tokens: Sequence[str], **kw) -> list[str]:
    """Per-word `text_norm`: `normalise` each token on its own (names and one-word CALLONs
    go; multi-word CALLON phrases are removed later, from joined available text only)."""
    return [normalise(t, **kw) for t in masked_tokens]


def has_qmark(token: str | None) -> bool:
    """True when the punctuated token ends in '?' (F-punct only)."""
    return bool(token) and str(token).rstrip("\"')]”").endswith("?")


def text_columns(punctuated: Sequence[str], roster, *, me_token: str = config.NAME_TOKEN
                 ) -> dict[str, list]:
    """`words` columns from one class's punctuated ASR tokens (in time order):
    {"text_masked", "text_norm", "has_qmark"}. Masking happens here, in memory."""
    masked = mask(list(punctuated), roster, me_token=me_token)
    return {"text_masked": masked, "text_norm": normalise_words(masked),
            "has_qmark": [has_qmark(t) for t in punctuated]}


# --------------------------------------------------------------------------- causal frames (§5.2)


@dataclass
class _Words:
    """One class's words as arrays, sorted by (start_s, w_idx)."""

    order: np.ndarray        # positions in the input frame
    start: np.ndarray
    end: np.ndarray
    spk: np.ndarray          # object: hashed speaker, "" when unknown
    role: np.ndarray         # object
    ravail: np.ndarray       # role availability (end + eps_w when missing)
    text: np.ndarray         # object: per-word normalised text
    w_idx: np.ndarray


def _col(words: pd.DataFrame, name: str, fill, dtype=object) -> np.ndarray:
    if name not in words.columns:
        return np.full(len(words), fill, dtype=dtype)
    s = words[name].astype(object)
    return s.where(s.notna(), fill).to_numpy(dtype)


def _words_arrays(words: pd.DataFrame, eps_w: float = config.EPS_W_S, *,
                  text_col: str = "text_norm") -> _Words:
    """Arrays sorted by (start_s, w_idx); `order[i]` is the input row position of word i."""
    n = len(words)
    start0 = words["start_s"].to_numpy(float)
    w_idx0 = words["w_idx"].to_numpy() if "w_idx" in words.columns else np.arange(n)
    order = np.lexsort((w_idx0, start0))
    end0 = words["end_s"].to_numpy(float)
    ravail0 = (words["role_avail_s"].to_numpy(float) if "role_avail_s" in words.columns
               else np.full(n, np.nan))
    ravail0 = np.where(np.isnan(ravail0), end0 + eps_w, ravail0)
    return _Words(order, start0[order], end0[order], _col(words, "spk_hash", "")[order],
                  _col(words, "role", config.ROLE_UNK)[order], ravail0[order],
                  _col(words, text_col, "")[order], w_idx0[order])


def _segment(W: _Words, eps_w: float = config.EPS_W_S) -> tuple[list[tuple[int, int]],
                                                                 np.ndarray]:
    """Greedy frame cuts on full data. Returns ([(first, last)], raw close times).

    A frame [a..l] is cut after l when (in this order): its span end_l - start_a > 5 s
    (length cut, the crossing word l included, closed at end_l + ε_w); no word starts in
    (end_l, end_l + 0.4] (pause cut, closed at end_l + 0.4 + ε_w); or the next word has
    another speaker (closed when that word and its speaker are available:
    max(end_next + ε_w, role_avail_next)). A frame is never closed before every one of its
    words' roles is available. Boundaries are strict, as in `causal.frame_spans`.
    """
    n = len(W.start)
    bounds: list[tuple[int, int]] = []
    closes: list[float] = []
    a = 0
    pause, max_len = config.FRAME_PAUSE_S, config.FRAME_MAX_S
    while a < n:
        lst = a
        while True:
            if W.end[lst] - W.start[a] > max_len:
                close = W.end[lst] + eps_w
                break
            nxt = lst + 1
            if nxt >= n or W.start[nxt] - W.end[lst] > pause:
                close = W.end[lst] + pause + eps_w
                break
            if W.spk[nxt] != W.spk[lst]:
                close = max(W.end[nxt] + eps_w, W.ravail[nxt])
                break
            lst = nxt
        bounds.append((a, lst))
        closes.append(max(close, float(W.ravail[a:lst + 1].max())))
        a = lst + 1
    return bounds, np.asarray(closes, float)


def _frame_rows(W: _Words, bounds, close, class_id) -> pd.DataFrame:
    rows = []
    for f, (a, b) in enumerate(bounds):
        txt = delete_callon(" ".join(t for t in W.text[a:b + 1] if t))
        rows.append({"class_id": class_id, "f_idx": f, "start_s": float(W.start[a]),
                     "end_s": float(W.end[a:b + 1].max()), "close_s": float(close[f]),
                     "role": W.role[a], "spk_hash": W.spk[a] or None, "text_norm": txt,
                     "n_words": len(txt.split())})
    return pd.DataFrame(rows, columns=["class_id", "f_idx", "start_s", "end_s", "close_s",
                                       "role", "spk_hash", "text_norm", "n_words"])


def _finish_frames(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return schemas.empty("frames")
    df = df.assign(emb_row=pd.array([pd.NA] * len(df), dtype="Int64"))
    return schemas.coerce(df.reset_index(drop=True), "frames")


def _one_class_id(words: pd.DataFrame):
    return int(words["class_id"].iloc[0]) if "class_id" in words.columns and len(words) else 0


def frames(words: pd.DataFrame, *, eps_w: float = config.EPS_W_S) -> pd.DataFrame:
    """Causal frames (§5.2) for one or more classes -> a `frames` table.

    `close_s` is the time the cut became observable, made non-decreasing within a class
    (a frame never closes before its predecessor), so closed frames at t are always a
    prefix found with `np.searchsorted(close_s, t, "right")`. `emb_row` is <NA> until
    `oncue embed` fills it.
    """
    if words.empty:
        return schemas.empty("frames")
    parts = []
    groups = words.groupby("class_id", sort=True) if "class_id" in words.columns \
        else [(0, words)]
    for cid, w in groups:
        W = _words_arrays(w, eps_w)
        bounds, close = _segment(W, eps_w)
        parts.append(_frame_rows(W, bounds, np.maximum.accumulate(close), cid))
    return _finish_frames(pd.concat(parts, ignore_index=True))


def assign_frames(words: pd.DataFrame, *, eps_w: float = config.EPS_W_S
                  ) -> tuple[np.ndarray, np.ndarray]:
    """(f_idx per input row, close_s per frame) for ONE class's words."""
    W = _words_arrays(words, eps_w)
    bounds, close = _segment(W, eps_w)
    f_of = np.empty(len(W.start), int)
    for f, (a, b) in enumerate(bounds):
        f_of[a:b + 1] = f
    out = np.empty(len(words), int)
    out[W.order] = f_of
    return out, (np.maximum.accumulate(close) if len(close) else close)


def truncate_words(words: pd.DataFrame, t: float, *, eps_w: float = config.EPS_W_S
                   ) -> pd.DataFrame:
    """What exists of `words` at time t (reference path): rows whose onset is perceptible
    (start <= t - ε_w); text and end hidden (NaN / "") until end <= t - ε_w; speaker and
    role hidden ("unk") until role_avail_s <= t."""
    w = words[words["start_s"] <= t - eps_w].copy()
    hidden = ~(w["end_s"] <= t - eps_w)
    ravail = w["role_avail_s"] if "role_avail_s" in w.columns else w["end_s"] + eps_w
    ravail = ravail.fillna(w["end_s"] + eps_w)
    unknown = hidden | ~(ravail <= t)
    w.loc[hidden, "end_s"] = np.nan
    for col in ("text_norm", "text_masked"):
        if col in w.columns:
            w[col] = w[col].astype(object)
            w.loc[hidden, col] = ""
    if "spk_hash" in w.columns:
        w["spk_hash"] = w["spk_hash"].astype(object)
        w.loc[unknown, "spk_hash"] = None
    w["role"] = w["role"].astype(object) if "role" in w.columns else config.ROLE_UNK
    w.loc[unknown, "role"] = config.ROLE_UNK
    w["role_avail_s"] = np.where(unknown, np.nan, ravail)
    w["_known"] = ~unknown
    return w


def frames_at(words: pd.DataFrame, t: float, *, eps_w: float = config.EPS_W_S) -> pd.DataFrame:
    """Reference path: the frames closed at t, rebuilt from `truncate_words(words, t)` only.

    Must equal `frames(words)` restricted to `close_s <= t` (tests check this).
    """
    if words.empty:
        return schemas.empty("frames")
    cid = _one_class_id(words)
    tw = truncate_words(words, t, eps_w=eps_w)
    W = _words_arrays(tw, eps_w)
    known = tw["_known"].to_numpy(bool)[W.order] if len(tw) else np.zeros(0, bool)
    n = len(W.start)
    pause, max_len = config.FRAME_PAUSE_S, config.FRAME_MAX_S
    bounds, closes = [], []
    prev_close, a = -np.inf, 0
    while a < n and known[a]:
        lst, close = a, None
        while True:
            if W.end[lst] - W.start[a] > max_len:
                close = W.end[lst] + eps_w
                break
            nxt = lst + 1
            if nxt >= n:                       # no later onset perceptible yet
                c = W.end[lst] + pause + eps_w
                close = c if t >= c else None
                break
            if W.start[nxt] - W.end[lst] > pause:
                close = W.end[lst] + pause + eps_w
                break
            if not known[nxt]:                 # next word or its speaker not known yet
                break
            if W.spk[nxt] != W.spk[lst]:
                close = max(W.end[nxt] + eps_w, W.ravail[nxt])
                break
            lst = nxt
        if close is None:
            break
        close = max(close, float(W.ravail[a:lst + 1].max()), prev_close)
        if close > t:
            break
        bounds.append((a, lst))
        closes.append(close)
        prev_close, a = close, lst + 1
    return _finish_frames(_frame_rows(W, bounds, np.asarray(closes, float), cid))


# --------------------------------------------------------------------------- windows at t (§5.3)


def callon_mask(tokens: Sequence[str]) -> np.ndarray:
    """Bool per normalised token: covered by a CALLON phrase in the joined sequence."""
    text, spans = "", []
    for tok in tokens:
        if text:
            text += " "
        spans.append((len(text), len(text) + len(tok)))
        text += tok
    hit = np.zeros(len(tokens), bool)
    for m in config.CALLON_RE.finditer(text):
        for i, (a, b) in enumerate(spans):
            if a < m.end() and b > m.start():
                hit[i] = True
    return hit


def _runs(roles: Sequence[str], spks: Sequence[str]) -> list[tuple[int, int]]:
    """[start, stop) index runs of constant (role, speaker)."""
    out, a = [], 0
    for i in range(1, len(roles) + 1):
        if i == len(roles) or roles[i] != roles[i - 1] or spks[i] != spks[i - 1]:
            out.append((a, i))
            a = i
    return out


def _run_tokens(texts: Sequence[str]) -> list[str]:
    toks = " ".join(t for t in texts if t).split()
    if not toks:
        return []
    cover = callon_mask(toks)
    return [t for t, c in zip(toks, cover, strict=True) if not c]


def _render(texts, roles, spks) -> tuple[str, str, list[tuple[str, list[str]]]]:
    """(role-tagged text, role-prefixed tokens, [(role, tokens)] per run)."""
    tagged, pref, runs = [], [], []
    for a, b in _runs(roles, spks):
        toks = _run_tokens(texts[a:b])
        if not toks:
            continue
        role = roles[a]
        runs.append((role, toks))
        tagged.append(config.ROLE_TAGS.get(role, config.ROLE_TAGS[config.ROLE_UNK]))
        tagged.extend(toks)
        p = config.ROLE_PREFIX.get(role, config.ROLE_PREFIX[config.ROLE_UNK])
        pref.extend(p + t for t in toks)
    return " ".join(tagged), " ".join(pref), runs


WINDOW_KEYS = ("X_all15", "X_all60", "T_all15", "T_all60", "X_prof60", "X_stud60",
               "X_lastprof", "lastprof_end_s", "X_chat60", "T_chat60", "X_tail", "tail_s",
               "me_sofar")


class WindowBuilder:
    """Causal text windows for one class (§5.3), for any t.

    Only words with `end_s <= t - ε_w` contribute text; a word's role is the Forum-causal
    role once `role_avail_s <= t`, else "unk". Chat rows count once `t_s + ε_F <= t`.

    `at(t)` returns a dict with:
    * X_all15 / X_all60: role-tagged text (<P> <S> <M> <U> at speaker changes);
    * T_all15 / T_all60 / T_chat60: role-prefixed tokens (p_ s_ m_ u_ c_) for TF-IDF;
    * X_prof60 / X_stud60: professor / other-student words; X_chat60: "<C> msg" per message;
    * X_lastprof (+ lastprof_end_s): the professor's latest utterance, split at a pause
      > 1.2 s or a non-professor word, truncated at t - ε_w;
    * X_tail (+ tail_s): available non-me words of frames not closed by t (and the time the
      last of them became available), for on-demand tail embeddings;
    * me_sofar: f_idx of Brady's frames with close_s <= t.

    `variant`: "norm" (headline, `text_norm`), "name" (F-name, keeps _name_/_me_ tokens) or
    "punct" (F-punct, cased and punctuated) — the last two re-normalise `text_masked`.
    """

    def __init__(self, words: pd.DataFrame, chat: pd.DataFrame | None = None, *,
                 eps_w: float = config.EPS_W_S, eps_f: float | None = None,
                 variant: str = "norm"):
        if chat is not None and len(chat) and eps_f is None:
            raise ValueError("eps_f (the class's ε_F) is required when chat is given")
        if variant not in ("norm", "name", "punct"):
            raise ValueError(f"unknown variant {variant!r}")
        self.eps_w, self.eps_f = float(eps_w), (None if eps_f is None else float(eps_f))
        w = words
        if variant != "norm":
            kw = {"keep_names": True} if variant == "name" else {"punct": True}
            w = words.assign(_txt=normalise_words(words["text_masked"].tolist(), **kw))
        self.W = _words_arrays(w, self.eps_w, text_col="_txt" if variant != "norm"
                               else "text_norm")
        W = self.W
        bounds, close = _segment(W, self.eps_w)
        close = np.maximum.accumulate(close) if len(close) else close
        self.f_close = close
        self.f_of = np.empty(len(W.start), int)
        for f, (a, b) in enumerate(bounds):
            self.f_of[a:b + 1] = f
        self.f_role = np.array([W.role[a] for a, _ in bounds], dtype=object)
        self.max_dur = float(np.max(W.end - W.start)) if len(W.start) else 0.0
        prof = np.flatnonzero(W.role == config.ROLE_PROF)
        self.prof_idx = prof
        self.prof_start = W.start[prof]
        self.prof_avail = np.maximum(W.end[prof] + self.eps_w, W.ravail[prof])
        if chat is not None and len(chat):
            c = chat.sort_values("t_s", kind="stable")
            self.c_t = c["t_s"].to_numpy(float)
            txt = c["text_masked"].tolist()
            kw = {"keep_names": True} if variant == "name" else (
                {"punct": True} if variant == "punct" else {})
            self.c_txt = [normalise(x, **kw) for x in txt]
        else:
            self.c_t, self.c_txt = np.zeros(0), []

    def _roles_at(self, idx: np.ndarray, t: float) -> tuple[list[str], list[str]]:
        W = self.W
        known = W.ravail[idx] <= t
        roles = np.where(known, W.role[idx], config.ROLE_UNK).tolist()
        spks = np.where(known, W.spk[idx], "").tolist()
        return roles, spks

    def _available(self, lo: float, t: float) -> np.ndarray:
        """Indices (start order) of words with lo < end <= t - ε_w."""
        W, hi = self.W, t - self.eps_w
        i0 = np.searchsorted(W.start, lo - self.max_dur, "right")
        i1 = np.searchsorted(W.start, hi, "right")
        e = W.end[i0:i1]
        return i0 + np.flatnonzero((e <= hi) & (e > lo))

    def lastprof(self, t: float) -> tuple[str, float]:
        """(X_lastprof text, end of its last word) at t; ("", nan) if none."""
        W, hi = self.W, t - self.eps_w
        k = np.searchsorted(self.prof_start, hi, "right") - 1
        while k >= 0 and not self.prof_avail[k] <= t:
            k -= 1
        if k < 0:
            return "", float("nan")
        j = int(self.prof_idx[k])
        utt = [j]
        i = j - 1
        while i >= 0:
            if not W.end[i] <= hi:                 # not available: treat as absent
                i -= 1
                continue
            if W.role[i] != config.ROLE_PROF or not W.ravail[i] <= t:
                break
            if W.start[utt[-1]] - W.end[i] > config.LASTPROF_PAUSE_S:
                break
            utt.append(i)
            i -= 1
        utt.reverse()
        return " ".join(_run_tokens(W.text[utt].tolist())), float(W.end[j])

    def at(self, t: float) -> dict:
        t = float(t)
        W = self.W
        idx60 = self._available(t - config.WIN_LONG_S, t)
        roles, spks = self._roles_at(idx60, t)
        texts = W.text[idx60].tolist()
        x60, t60, runs60 = _render(texts, roles, spks)
        k15 = np.flatnonzero(W.end[idx60] > t - config.WIN_SHORT_S)
        x15, t15, _ = _render([texts[i] for i in k15], [roles[i] for i in k15],
                              [spks[i] for i in k15])
        prof = [" ".join(toks) for r, toks in runs60 if r == config.ROLE_PROF]
        stud = [" ".join(toks) for r, toks in runs60 if r == config.ROLE_STUDENT]
        lp, lp_end = self.lastprof(t)
        # chat: available (t_s + eps_f <= t) and t_s in (t - 60, t]
        if len(self.c_t):
            c0 = np.searchsorted(self.c_t, t - config.WIN_LONG_S, "right")
            c1 = np.searchsorted(self.c_t, t - self.eps_f, "right")
            msgs = [m for m in self.c_txt[c0:c1] if m]
        else:
            msgs = []
        x_chat = " ".join(f"{config.ROLE_TAGS['chat']} {m}" for m in msgs)
        t_chat = " ".join(config.ROLE_PREFIX["chat"] + tok for m in msgs for tok in m.split())
        # tail: available words of frames not closed by t, excluding Brady's
        n_closed = int(np.searchsorted(self.f_close, t, "right"))
        hi = t - self.eps_w
        first_open = np.searchsorted(self.f_of, n_closed, "left")
        cand = np.arange(first_open, len(W.start))
        cand = cand[W.end[cand] <= hi]
        troles, tspks = self._roles_at(cand, t)
        keep = [i for i, r in enumerate(troles) if r != config.ROLE_ME]
        tail_toks = []
        for a, b in _runs([troles[i] for i in keep], [tspks[i] for i in keep]):
            tail_toks += _run_tokens(W.text[cand[keep[a:b]]].tolist())
        tail_s = float(W.end[cand[keep]].max() + self.eps_w) if keep and tail_toks \
            else float("nan")
        me_sofar = np.flatnonzero(self.f_role[:n_closed] == config.ROLE_ME)
        return {"X_all15": x15, "X_all60": x60, "T_all15": t15, "T_all60": t60,
                "X_prof60": " ".join(prof), "X_stud60": " ".join(stud),
                "X_lastprof": lp, "lastprof_end_s": lp_end, "X_chat60": x_chat,
                "T_chat60": t_chat, "X_tail": " ".join(tail_toks), "tail_s": tail_s,
                "me_sofar": me_sofar}

    def many(self, ts: Iterable[float]) -> pd.DataFrame:
        """One row per t (column t_s plus every WINDOW_KEYS entry)."""
        rows = [{"t_s": float(t), **self.at(t)} for t in ts]
        return pd.DataFrame(rows, columns=["t_s", *WINDOW_KEYS])


def windows(words: pd.DataFrame, t: float, chat: pd.DataFrame | None = None, **kw) -> dict:
    """One-off `WindowBuilder(words, chat, **kw).at(t)` (reference-path convenience)."""
    return WindowBuilder(words, chat, **kw).at(t)


# --------------------------------------------------------------------------- questions (§5.5)


def is_question(text: str | None, *, punct: bool = False, qmark: bool = False) -> bool:
    """Headline question test on normalised text: first token is a wh-word or inverted
    auxiliary and >= 4 words, or the text ends in a tag. With `punct` (F-punct only), a
    final '?' (`qmark`, or a literal '?' in `text`) also counts."""
    s = str(text or "").strip()
    if not s:
        return False
    if config.Q_FIRST_RE.match(s) and len(s.split()) >= config.Q_MIN_WORDS:
        return True
    if config.Q_TAG_RE.search(s):
        return True
    return bool(punct and (qmark or s.rstrip("\"')]").endswith("?")))


def question_kind(text: str | None) -> str:
    """"check" ("make sense", "any questions"), "invite" (INVITE lexicon) or "plain"."""
    s = str(text or "")
    if config.CHECK_RE.search(s):
        return "check"
    if config.INVITE_RE.search(s):
        return "invite"
    return "plain"


QDET_COLUMNS = ("class_id", "w_last", "start_s", "end_s", "avail_s", "n_words", "is_q",
                "new_q", "is_invite", "is_check", "vote_lex", "kind", "rhetorical",
                "rhetorical_avail_s")


def _qdet_one(w: pd.DataFrame, *, punct: bool, eps_w: float, split_pause_s: float | None
              ) -> pd.DataFrame:
    W = _words_arrays(w, eps_w)
    qm = None
    if punct and "has_qmark" in w.columns:
        qm = _col(w, "has_qmark", False, bool)[W.order]
    n = len(W.start)
    is_prof = W.role == config.ROLE_PROF
    utt0 = np.full(n, -1)
    for i in range(n):
        if is_prof[i]:
            cont = i > 0 and is_prof[i - 1] \
                and W.start[i] - W.end[i - 1] <= config.LASTPROF_PAUSE_S
            utt0[i] = utt0[i - 1] if cont else i
    rows, prev_start, prev_q = [], None, False
    cid = _one_class_id(w)
    for j in np.flatnonzero(is_prof):
        k1 = int(np.searchsorted(W.start, W.end[j], "right"))   # first onset after end_j
        if k1 < n and not W.start[k1] - W.end[j] > config.Q_SILENCE_S:
            continue                                            # no 0.8 s silence
        a = int(utt0[j])
        toks = _run_tokens(W.text[a:j + 1].tolist())
        full = " ".join(toks)
        last = full
        if split_pause_s is not None:
            gaps = W.start[a + 1:j + 1] - W.end[a:j]
            cut = np.flatnonzero(gaps >= split_pause_s)
            if len(cut):
                last = " ".join(_run_tokens(W.text[a + int(cut[-1]) + 1:j + 1].tolist()))
        q = is_question(full) or is_question(last) or bool(
            punct and qm is not None and qm[j])
        resumed = k1 < n and is_prof[k1] \
            and W.start[k1] - W.end[j] <= config.Q_RHETORICAL_RESUME_S
        rows.append({
            "class_id": cid, "w_last": int(W.w_idx[j]), "start_s": float(W.start[a]),
            "end_s": float(W.end[j]),
            "avail_s": max(W.end[j] + config.Q_SILENCE_S + eps_w, float(W.ravail[a:j + 1].max())),
            "n_words": len(toks), "is_q": q, "new_q": q and not (prev_q and prev_start == a),
            "is_invite": bool(config.INVITE_RE.search(full)),
            "is_check": bool(config.CHECK_RE.search(full)),
            "vote_lex": bool(config.VOTE_RE.search(full)),
            "kind": question_kind(full) if q else None,
            "rhetorical": bool(q and resumed),
            "rhetorical_avail_s": (max(W.end[j] + config.Q_RHETORICAL_RESUME_S + eps_w,
                                       float(W.ravail[k1])) if q and resumed else np.nan)})
        prev_start, prev_q = a, q
    return pd.DataFrame(rows, columns=list(QDET_COLUMNS))


def question_detector(words: pd.DataFrame, *, punct: bool = False,
                      eps_w: float = config.EPS_W_S,
                      split_pause_s: float | None = config.FRAME_PAUSE_S) -> pd.DataFrame:
    """Evaluate the professor's utterance at every observed >= 0.8 s silence (§5.5).

    For each professor word j with no word onset (any speaker) in (end_j, end_j + 0.8], the
    utterance ending at j (split at a pause > 1.2 s or a non-professor word) is tested.
    `avail_s` = max(end_j + 0.8 + ε_w, role availability of its words). `is_q` uses the
    headline test (no punctuation) on the whole utterance or, with `split_pause_s`, on its
    last pause-delimited segment (questions often end a longer utterance); `punct=True`
    (F-punct) adds the final-'?' rule from `has_qmark`. `rhetorical`: the next onset is the
    professor resuming within 3 s, known at `rhetorical_avail_s` (>= end + 3 s + ε_w).
    `new_q` marks the first question evaluation of an utterance (for counts).
    """
    if words.empty:
        return pd.DataFrame(columns=list(QDET_COLUMNS))
    groups = words.groupby("class_id", sort=True) if "class_id" in words.columns \
        else [(0, words)]
    parts = [_qdet_one(w, punct=punct, eps_w=eps_w, split_pause_s=split_pause_s)
             for _, w in groups]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=list(QDET_COLUMNS))


def _latest(avail: np.ndarray, end: np.ndarray, ts: np.ndarray) -> np.ndarray:
    """Per t: index (into the inputs) of the max-end row among rows with avail <= t; -1."""
    if not len(avail):
        return np.full(len(ts), -1)
    order = np.argsort(avail, kind="stable")
    e = end[order]
    run = np.zeros(len(e), int)
    for i in range(1, len(e)):
        run[i] = i if e[i] >= e[run[i - 1]] else run[i - 1]
    k = np.searchsorted(avail[order], ts, "right") - 1
    return np.where(k >= 0, order[run[np.maximum(k, 0)]], -1)


def _take(values, idx: np.ndarray, fill) -> np.ndarray:
    """values[idx] with `fill` where idx < 0 (safe for empty `values`)."""
    values = np.asarray(values)
    if not len(values):
        return np.array([fill] * len(idx))
    return np.where(idx >= 0, values[np.maximum(idx, 0)], fill)


def _latest_in(mask: np.ndarray, avail: np.ndarray, end: np.ndarray, ts: np.ndarray
               ) -> np.ndarray:
    """`_latest` over the rows where `mask` holds, as indices into the full arrays."""
    sub = np.flatnonzero(mask)
    return _take(sub, _latest(avail[sub], end[sub], ts), -1).astype(int)


Q_FEATURE_COLUMNS = ("q_open", "q_age_s", "q_inv_age_s", "q_kind", "q_count_300", "q_vote_lex")


def q_features(qdet: pd.DataFrame, words: pd.DataFrame, frames_df: pd.DataFrame,
               ts: Sequence[float], *, eps_w: float = config.EPS_W_S) -> pd.DataFrame:
    """The §5.5 `q` columns for ONE class at decision times `ts` (vectorised, causal).

    * q_open: the latest evaluated professor utterance is a question, no newer professor
      word is known, it is not (yet known to be) rhetorical, and no student/me frame of
      >= 3 words that started after it has closed;
    * q_age_s / q_inv_age_s: seconds since the latest question / invitation (INVITE
      lexicon in an evaluated utterance, question or not), capped at 300;
    * q_kind: kind of the latest question (rhetorical once known; "none" before any);
    * q_count_300: new questions available in (t - 300, t]; q_vote_lex: a VOTE-lexicon
      utterance became available in (t - 60, t].
    """
    ts = np.asarray(ts, float)
    cap = config.Q_AGE_CAP_S
    q = qdet.reset_index(drop=True)
    avail, end = q["avail_s"].to_numpy(float), q["end_s"].to_numpy(float)
    isq = q["is_q"].to_numpy(bool)
    rh_av = q["rhetorical_avail_s"].to_numpy(float)
    out = pd.DataFrame({"t_s": ts})
    # latest evaluation, latest question, latest invitation (by end, among available)
    li = _latest(avail, end, ts)
    lq = _latest_in(isq, avail, end, ts)
    lv = _latest_in(q["is_invite"].to_numpy(bool), avail, end, ts)
    out["q_age_s"] = np.minimum(np.nan_to_num(ts - _take(end, lq, np.nan), nan=cap), cap)
    out["q_inv_age_s"] = np.minimum(np.nan_to_num(ts - _take(end, lv, np.nan), nan=cap), cap)
    # newest professor word known at t; newest student/me frame (>= 3 words) closed by t
    W = _words_arrays(words, eps_w)
    p = np.flatnonzero(W.role == config.ROLE_PROF)
    pa = np.maximum(W.end[p] + eps_w, W.ravail[p])
    last_prof_end = _take(W.end[p], _latest(pa, W.end[p], ts), -np.inf)
    f = frames_df
    sf = f[f["role"].isin([config.ROLE_STUDENT, config.ROLE_ME])
           & (f["n_words"] >= config.Q_OPEN_STUDENT_MIN_WORDS)]
    s_start = sf["start_s"].to_numpy(float)
    stud_start = _take(s_start, _latest(sf["close_s"].to_numpy(float), s_start, ts), -np.inf)
    last_end = _take(end, li, np.nan)
    last_isq = _take(isq, li, False).astype(bool)
    rh_known = _take(rh_av, li, np.nan) <= ts
    out["q_open"] = ((li >= 0) & last_isq & (last_end >= last_prof_end - 1e-9)
                     & (stud_start <= last_end) & ~rh_known).astype(float)
    kind = _take(q["kind"].to_numpy(object), lq, "none")
    out["q_kind"] = np.where(_take(rh_av, lq, np.nan) <= ts, "rhetorical", kind).astype(object)
    nq = q["new_q"].to_numpy(bool)
    a_n, e_n = avail[nq], end[nq]
    out["q_count_300"] = ((a_n[None, :] <= ts[:, None])
                          & (e_n[None, :] > ts[:, None] - cap)).sum(axis=1)
    vl = np.sort(avail[q["vote_lex"].to_numpy(bool)])
    out["q_vote_lex"] = (np.searchsorted(vl, ts, "right")
                         - np.searchsorted(vl, ts - config.WIN_LONG_S, "right") > 0).astype(float)
    return out


def rule_R(q_open, q_age_s=None, q_inv_age_s=None) -> np.ndarray:
    """Rule baseline R (no training): (0.5 + 0.5 q_open) * exp(-min(q_inv_age, 2 q_age) / 10 s).

    Accepts the three arrays, or one DataFrame with the `q_features` columns.
    """
    if isinstance(q_open, pd.DataFrame):
        df = q_open
        q_open, q_age_s, q_inv_age_s = df["q_open"], df["q_age_s"], df["q_inv_age_s"]
    qo = np.asarray(q_open, float)
    age = np.minimum(np.asarray(q_age_s, float), config.Q_AGE_CAP_S)
    inv = np.minimum(np.asarray(q_inv_age_s, float), config.Q_AGE_CAP_S)
    return (0.5 + 0.5 * qo) * np.exp(-np.minimum(inv, 2.0 * age) / config.R_DECAY_S)


def callon_events(words: pd.DataFrame, *, eps_w: float = config.EPS_W_S) -> pd.DataFrame:
    """CALLON phrases in the professor's words, for the CALLON check (§9.2) ONLY.

    Uses `text_masked` with CALLON kept. Returns class_id, start_s, end_s, avail_s per hit.
    """
    cols = ["class_id", "start_s", "end_s", "avail_s"]
    rows = []
    groups = words.groupby("class_id", sort=True) if "class_id" in words.columns \
        else [(0, words)]
    for cid, w in groups:
        w = w.assign(_raw=[normalise(x, callon=False) for x in
                           w.get("text_masked", w.get("text_norm", pd.Series(dtype=object)))])
        W = _words_arrays(w, eps_w, text_col="_raw")
        roles = W.role.tolist()
        for a, b in _runs(roles, W.spk.tolist()):
            if roles[a] != config.ROLE_PROF:
                continue
            toks, owner = [], []
            for i in range(a, b):
                for tok in str(W.text[i]).split():
                    toks.append(tok)
                    owner.append(i)
            cover = callon_mask(toks)
            if not cover.any():
                continue
            # group consecutive covered tokens into phrases
            idx = np.flatnonzero(cover)
            for grp in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
                i0, i1 = owner[grp[0]], owner[grp[-1]]
                rows.append({"class_id": cid, "start_s": float(W.start[i0]),
                             "end_s": float(W.end[i1]),
                             "avail_s": max(float(W.end[i1]) + eps_w,
                                            float(W.ravail[i0:i1 + 1].max()))})
    return pd.DataFrame(rows, columns=cols)


# --------------------------------------------------------------------------- CLI


def _parse_extra(extra: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m oncue frames", add_help=False)
    p.add_argument("--tables", type=Path, default=config.TABLES_DIR)
    return p.parse_args(list(extra or []))


def cli_frames(args) -> int:
    """`python -m oncue frames`: words.parquet -> frames.parquet (per class, causal close_s).

    Masking and per-word normalisation happen when `words` is built (transcribe.words_table
    calls `text_columns`); this step only cuts frames. `--tables DIR` overrides the table dir.
    """
    opts = _parse_extra(getattr(args, "extra", []))
    src, dst = opts.tables / "words.parquet", opts.tables / "frames.parquet"
    if not src.exists():
        print(f"oncue frames: {src} not found (run `oncue words` first)")
        return 1
    words = pd.read_parquet(src)
    class_ids = getattr(args, "class_ids", None)
    if class_ids:
        words = words[words["class_id"].astype(str).isin([str(c) for c in class_ids])]
    out = schemas.validate(frames(words), "frames")
    if class_ids and dst.exists():
        old = pd.read_parquet(dst)
        old = old[~old["class_id"].astype(str).isin([str(c) for c in class_ids])]
        out = pd.concat([old, out], ignore_index=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dst, index=False)
    print(f"oncue frames: {len(out)} frames for {out['class_id'].nunique()} classes -> {dst}")
    return 0
