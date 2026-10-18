"""Labels: speech onsets, the decision grid, the at-risk set and every y_* column.

Spec §2.4-§2.6 and §3, with amendment A1 (g in {1, 2, 3, 5}; `y_H15_g1`, `y_H15_g3`).

* `speech_onsets(voice, words, episodes)` -> Brady's main-room speech turns (2 s merge,
  fresh after >= 30 s of silence), the T-speech target and the label cross-checks.
* `causal_rules(t, tables)` -> the causal exclusion rules on a grid, vectorised. Each rule
  reads only rows available at t (the reference re-derivation is `causal.CausalView`).
* `label_values(...)` -> y_{H,g} with asymmetric observability: positives need the tile
  observed only up to their onset, negatives over the whole window (`sym=True` for the
  symmetric sensitivity).
* `points(cid)` -> the `points` table: grid t = 0, Δ, ... <= T_end; `excl_reason` (first
  matching rule in config.EXCL_REASONS order); `in_G`; `at_risk`; `label_state`; all
  config.LABEL_COLUMNS; `next_bid_s`; `fold` (<NA> here; `assign_folds` fills it once over
  all dev classes in `oncue labels`); `in_vault`.

Conventions: every y_* column is NaN where its own target is undefined, and also where a
causal rule (or right-censoring `t + H > T_end`) removes the point. Class QA is not baked
into y_*: it only enters `excl_reason`/`at_risk` (rule `class_qa`, fired when
`classes.qa_pass` is False), so sensitivity populations can re-admit classes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config, schemas
from .causal import (
    EDGE_S,
    EPS_W,
    INF,
    ClassInfo,
    bools,
    brady_wb_times,
    class_info,
    class_tables,
    floats,
    hand_runs,
    live_raw_column,
    live_states,
    load_tables,
    strs,
    table,
)

log = logging.getLogger(__name__)

SPEECH_CAUSAL = ("opening", "end", "rec_gap", "breakout", "break", "me_speaking")
STEP_S = 1.0 / config.F_SCAN                 # scan step
STEP_TOL_S = 0.25 * STEP_S                   # pts jitter tolerated before a sample is stale
LABEL_SPECS = {  # column -> (H, g, symmetric)
    config.y_col(h, config.G_S): (h, config.G_S, False) for h in config.H_SWEEP}
LABEL_SPECS.update({config.y_col(config.H_S, g): (config.H_S, g, False)
                    for g in config.G_LABELS})
LABEL_SPECS[config.y_col(config.H_S, config.G_S, sym=True)] = (config.H_S, config.G_S, True)
SPEECH_LABEL = config.y_col(config.H_S, config.G_SPEECH_S, speech=True)


# --------------------------------------------------------------------------- speech onsets


def _word_bearing(voice: pd.DataFrame, words: pd.DataFrame | None) -> np.ndarray:
    """Brady events overlapping (edges padded by EDGE) at least one Brady-attributed word."""
    if words is None or not len(words):
        return bools(voice["word_bearing"]) if "word_bearing" in voice else np.zeros(len(voice),
                                                                                     bool)
    me = strs(words["role"]) == config.ROLE_ME
    ws, we = floats(words["start_s"])[me], floats(words["end_s"])[me]
    vs, ve = floats(voice["start_s"]), floats(voice["end_s"])
    out = np.zeros(len(voice), dtype=bool)
    for i in range(len(voice)):
        out[i] = bool(((ws - EDGE_S < ve[i]) & (we + EDGE_S > vs[i])).any())
    return out


def speech_onsets(voice: pd.DataFrame, words: pd.DataFrame | None = None,
                  episodes: pd.DataFrame | None = None, *, class_id=None) -> pd.DataFrame:
    """Brady's speech turns (§2.6): main-room, word-bearing voice events merged when the gap
    is < 2 s. `fresh` = silent >= 30 s before (the first turn of the recording is fresh,
    with unknown silence). `hand_preceded` = a bid in [o - 180, o] (NA without episodes)."""
    if not len(voice):
        return schemas.empty("speech_onsets")
    cid = int(class_id if class_id is not None else voice["class_id"].iloc[0])
    sel = (strs(voice["role"]) == config.ROLE_ME) & bools(voice["room_main"])
    sel &= _word_bearing(voice, words)
    vs, ve = floats(voice["start_s"])[sel], floats(voice["end_s"])[sel]
    order = np.argsort(vs, kind="stable")
    turns: list[list[float]] = []
    for a, b in zip(vs[order], ve[order], strict=True):
        if turns and a - turns[-1][1] < config.TURN_MERGE_GAP_S:
            turns[-1][1] = max(turns[-1][1], b)
        else:
            turns.append([a, b])
    if not turns:
        return schemas.empty("speech_onsets")
    on = np.array([x[0] for x in turns])
    end = np.array([x[1] for x in turns])
    silence = np.concatenate([[np.nan], on[1:] - end[:-1]])
    fresh = np.isnan(silence) | (silence >= config.FRESH_SILENCE_S)
    if episodes is not None:
        bids = episode_sets(episodes).bids
        lo = np.searchsorted(bids, on - 180.0, side="left")
        hi = np.searchsorted(bids, on, side="right")
        preceded = pd.array(hi > lo, dtype="boolean")
    else:
        preceded = pd.array([pd.NA] * len(on), dtype="boolean")
    df = pd.DataFrame({"class_id": cid, "onset_s": on, "turn_end_s": end,
                       "silence_before_s": silence, "fresh": fresh, "hand_preceded": preceded})
    return schemas.coerce(df, "speech_onsets")


# --------------------------------------------------------------------------- episodes


@dataclass
class EpisodeSets:
    """Sorted onset times by kind (label side; full information)."""

    bids: np.ndarray
    votes: np.ndarray
    censored: np.ndarray
    glitches: np.ndarray

    @property
    def any(self) -> np.ndarray:
        return np.sort(np.concatenate([self.bids, self.votes, self.censored, self.glitches]))

    @property
    def vote_or_censored(self) -> np.ndarray:
        return np.sort(np.concatenate([self.votes, self.censored]))


def onset_times(episodes: pd.DataFrame, retimed: bool = True) -> np.ndarray:
    """Best onset per episode: re-timed when available and not failed, else coarse."""
    on = floats(episodes["onset_s"])
    if retimed and "onset_retimed_s" in episodes:
        rt = floats(episodes["onset_retimed_s"])
        failed = bools(episodes["retime_failed"]) if "retime_failed" in episodes \
            else np.zeros(len(on), bool)
        on = np.where(np.isfinite(rt) & ~failed, rt, on)
    return on


def episode_sets(episodes: pd.DataFrame, *, retimed: bool = True,
                 votes_as_bids: bool = False, contested_as_bids: bool = False) -> EpisodeSets:
    """Bids, votes, left-censored runs and glitches. Defaults are the headline rules (votes
    undefined); the flags are the §2.5 sensitivities."""
    if episodes is None or not len(episodes):
        z = np.zeros(0)
        return EpisodeSets(z, z, z, z)
    on = onset_times(episodes, retimed)
    typ = strs(episodes["ep_type"])
    lc = bools(episodes["left_censored"]) if "left_censored" in episodes \
        else np.zeros(len(on), bool)
    contested = bools(episodes["contested"]) if "contested" in episodes \
        else np.zeros(len(on), bool)
    bid = (typ == config.EP_BID) & ~lc
    vote = (typ == config.EP_VOTE) & ~lc
    if votes_as_bids:
        bid, vote = bid | vote, np.zeros_like(vote)
    elif contested_as_bids:
        bid, vote = bid | (vote & contested), vote & ~contested
    glitch = (typ == config.EP_GLITCH) & ~lc

    def pick(mask):
        x = on[mask]
        return np.sort(x[np.isfinite(x)])
    return EpisodeSets(pick(bid), pick(vote), pick(lc), pick(glitch))


def retime_summary(hand_episodes: pd.DataFrame) -> dict:
    """A1: share of bid onsets (non-censored bids) whose re-timing failed or never ran.
    {n_bids, n_retimed, n_retime_failed, retime_failed_share, g1_approximate}; g = 1 results
    are reported as approximate when the share is >= G1_RETIME_FAILED_MAX."""
    e = hand_episodes
    if e is None or not len(e):
        return {"n_bids": 0, "n_retimed": 0, "n_retime_failed": 0,
                "retime_failed_share": float("nan"), "g1_approximate": pd.NA}
    bid = (e["ep_type"].astype(str) == config.EP_BID).to_numpy() \
        & ~e["left_censored"].fillna(False).astype(bool).to_numpy()
    rf = e["retime_failed"] if "retime_failed" in e else pd.Series(pd.NA, index=e.index)
    rf = pd.Series(pd.array(rf, dtype="boolean"))[bid]
    n = int(bid.sum())
    ok = int((~rf.fillna(True)).sum())                    # never re-timed counts as failed
    share = (n - ok) / n if n else float("nan")
    return {"n_bids": n, "n_retimed": int(rf.notna().sum()), "n_retime_failed": n - ok,
            "retime_failed_share": share,
            "g1_approximate": (pd.NA if not n else bool(share >= config.G1_RETIME_FAILED_MAX))}


def _count(sorted_x: np.ndarray, a, b) -> np.ndarray:
    """Number of values in (a, b] (vectorised over a and b)."""
    return (np.searchsorted(sorted_x, b, side="right")
            - np.searchsorted(sorted_x, a, side="right"))


# --------------------------------------------------------------------------- observability


@dataclass
class Observability:
    """Per-sample observation flags after NA bridging (label side)."""

    t: np.ndarray
    bad_cum: np.ndarray     # bad_cum[i] = number of unobserved samples among the first i
    found: np.ndarray       # tile_found per sample
    t_last: float

    def tile_found_at(self, t) -> np.ndarray:
        """`tile_found` of the last sample at or before t, and only if that sample is at most
        one scan step old (inside a gap in hand_frames nothing is seen)."""
        t = np.asarray(t, float)
        idx = np.searchsorted(self.t, t, side="right") - 1
        ok = idx >= 0
        ok[ok] &= t[ok] - self.t[idx[ok]] <= STEP_S + STEP_TOL_S
        out = np.zeros(np.shape(t), dtype=bool)
        out[ok] = self.found[idx[ok]]
        return out

    def observed(self, a, b) -> np.ndarray:
        """True where (a, b] is fully covered by observed samples: no bad sample in it, about
        one sample per scan step (a time gap with no rows is not observed), and inside the
        scan."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        if not len(self.t):
            return np.zeros(np.broadcast(a, b).shape, dtype=bool)
        hi = np.searchsorted(self.t, b, side="right")
        lo = np.searchsorted(self.t, a, side="right")
        clean = (self.bad_cum[hi] - self.bad_cum[lo]) == 0
        dense = (hi - lo) >= (b - a) * config.F_SCAN - 1.0 - 1e-9
        inside = b <= self.t_last + 1.0 / config.F_SCAN
        return clean & dense & inside


def observability(hand_frames: pd.DataFrame) -> Observability:
    """Observed samples: tile found and a raw state, with NA gaps <= NA_BRIDGE_S between
    samples of the same state bridged (§2.4). Missing samples count as NA."""
    if hand_frames is None or not len(hand_frames):
        return Observability(np.zeros(0), np.zeros(1, int), np.zeros(0, bool), -INF)
    hf = hand_frames.sort_values("t_s", kind="stable")
    t = floats(hf["t_s"])
    raw = strs(hf["raw_state"], config.STATE_NA)
    found = bools(hf["tile_found"]) if "tile_found" in hf else np.ones(len(t), bool)
    bad = (raw == config.STATE_NA) | ~found
    step = 1.0 / config.F_SCAN
    gaps = np.concatenate([[False], np.diff(t) > 1.5 * step])
    bad = bad | gaps
    # bridge short NA runs with the same state on both sides
    i, n = 0, len(t)
    while i < n:
        if not bad[i]:
            i += 1
            continue
        j = i
        while j < n and bad[j]:
            j += 1
        run_s = (j - i) * step
        if (0 < i and j < n and run_s <= config.NA_BRIDGE_S and not gaps[i:j].any()
                and raw[i - 1] == raw[j] and raw[j] != config.STATE_NA):
            bad[i:j] = False
        i = j
    return Observability(t, np.concatenate([[0], np.cumsum(bad)]), found, float(t[-1]))


# --------------------------------------------------------------------------- causal rules


def _seg_types_at(segments: pd.DataFrame, t: np.ndarray, eps_f: float) -> np.ndarray:
    if not len(segments):
        return np.full(len(t), "", dtype=object)
    s = segments.sort_values("start_s", kind="stable")
    idx = np.searchsorted(floats(s["start_s"]) + eps_f, t, side="right") - 1
    typ = strs(s["seg_type"])
    return np.where(idx >= 0, typ[np.clip(idx, 0, None)], "")


def causal_rules(t, tables: dict, info: ClassInfo | None = None, *, H: float = config.H_S
                 ) -> pd.DataFrame:
    """Boolean frame (one column per config.EXCL_CAUSAL rule) on the grid `t`.

    Every rule uses rows available at t only: segments at start + eps_F; Brady's voice at
    start + eps_F (end only once end + eps_F <= t); Brady's words at role_avail_s. The hand
    rules read hand samples <= t only (`causal.hand_runs` on `raw_state_live`): `hand_up` =
    the live state (short NA held), `post_lowering` = R_low after every live exit from UP
    (also the inner lowering of a toggle, and an exit into a long NA run), `post_glitch` =
    R_low after every raw UP blip that ended in a DOWN sample. Label-side `hand_episodes`
    (merged toggles, bridged NA, typing) are never read here. `end` is the label-side
    right-censoring rule t + H > T_end (it also removes points from G).
    """
    t = np.asarray(t, dtype=float)
    info = info or class_info(table(tables, "classes"), _cid(tables))
    ef = info.eps_f
    out = {"opening": t < config.T_OPEN_S, "end": t + H > info.T_end}
    rec = np.zeros(len(t), bool)
    for r in info.restarts:
        rec |= (t >= r) & (t < r + config.REC_RESTART_SKIP_S)
    out["rec_gap"] = rec
    seg = _seg_types_at(table(tables, "segments"), t, ef)

    v = table(tables, "voice")
    if len(v):   # rows without a media time (recording-gap events) never count
        v = v[np.isfinite(floats(v["start_s"]))].reset_index(drop=True)
    role_v = strs(v["role"]) if len(v) else np.array([], dtype=object)
    brady = role_v == config.ROLE_ME
    vs = floats(v["start_s"]) if len(v) else np.zeros(0)
    ve = floats(v["end_s"]) if len(v) else np.zeros(0)
    main = bools(v["room_main"]) if len(v) else np.zeros(0, bool)
    # breakout: current segment, or Brady's latest visible voice event in [t - 60, t]
    breakout = seg == "breakout"
    bi = np.flatnonzero(brady)
    if len(bi):
        bi = bi[np.argsort(vs[bi], kind="stable")]
        last = np.searchsorted(vs[bi] + ef, t, side="right") - 1
        ok = last >= 0
        li = bi[np.clip(last, 0, None)]
        breakout |= ok & (vs[li] >= t - config.BREAKOUT_VOICE_LOOKBACK_S) & ~main[li]
    out["breakout"] = breakout
    out["break"] = seg == "break"

    # me_speaking: a Brady word ending in (t - S_me, t - eps_w] whose role is known, or an
    # ongoing Brady main-room event that already overlaps an available Brady word
    w = table(tables, "words")
    speaking = np.zeros(len(t), bool)
    if len(w):
        me = strs(w["role"]) == config.ROLE_ME
        we = floats(w["end_s"])[me]
        ra = floats(w["role_avail_s"])[me]
        ra = np.where(np.isnan(ra), we + EPS_W, ra)
        for a in range(0, len(t), 512):
            tt = t[a:a + 512, None]
            hit = (we > tt - config.S_ME_S) & (we <= tt - EPS_W) & (ra <= tt)
            speaking[a:a + 512] = hit.any(axis=1)
    if len(v):
        t1, t2 = brady_wb_times(v, w)
        ev = np.flatnonzero(brady & main)
        if len(ev):
            wb_from = np.minimum(t1[ev], t2[ev])[None, :]
            tt = t[:, None]
            ongoing = (vs[ev][None, :] + ef <= tt) & ~(ve[ev][None, :] + ef <= tt)
            speaking |= (ongoing & (wb_from <= tt)).any(axis=1)
    out["me_speaking"] = speaking

    # hand rules: the live state (samples <= t, short NA held), R_low after every live exit
    # from UP (inner lowerings of toggles and exits into NA included) and after every raw
    # blip. Merged label-side episodes never enter: they depend on samples after t.
    hf = table(tables, "hand_frames")
    hand_up = np.zeros(len(t), bool)
    post_low = np.zeros(len(t), bool)
    post_glitch = np.zeros(len(t), bool)
    if len(hf):
        hf = hf.sort_values("t_s", kind="stable")
        live = live_states(hf[live_raw_column(hf)])
        idx = np.searchsorted(floats(hf["t_s"]), t, side="right") - 1
        hand_up = (idx >= 0) & (live[np.clip(idx, 0, None)] == config.STATE_UP)
        runs = hand_runs(hf)
        post_low = _within(t, runs.exit_s, config.R_LOW_S)
        post_glitch = _within(t, runs.blip_end_s, config.R_LOW_S)
    out["hand_up"] = hand_up
    out["post_lowering"] = post_low
    out["post_glitch"] = post_glitch
    return pd.DataFrame({r: np.asarray(out[r], dtype=bool) for r in config.EXCL_CAUSAL})


def _within(t: np.ndarray, starts: np.ndarray, span: float) -> np.ndarray:
    """True where some start s (finite) has s <= t < s + span."""
    s = np.sort(starts[np.isfinite(starts)])
    if not len(s):
        return np.zeros(len(t), bool)
    k = np.searchsorted(s, t, side="right") - 1          # latest start <= t
    return (k >= 0) & (t < s[np.clip(k, 0, None)] + span)


def first_reason(flags: pd.DataFrame, order) -> np.ndarray:
    """The first rule (in `order`) that fires per row, or None."""
    out = np.full(len(flags), None, dtype=object)
    for rule in reversed(order):
        if rule in flags:
            out = np.where(flags[rule].to_numpy(bool), rule, out)
    return out


def _cid(tables: dict) -> int:
    for name in ("classes", "words", "voice", "hand_frames"):
        df = table(tables, name)
        if len(df) and "class_id" in df:
            return int(df["class_id"].iloc[0])
    return 0


# --------------------------------------------------------------------------- label rules


def label_values(t, sets: EpisodeSets, obs: Observability, *, H: float, g: float,
                 sym: bool = False) -> tuple[np.ndarray, pd.DataFrame]:
    """y_{H,g}(t) before causal masking, plus the label-side rule flags (§3).

    Positive: the first bid o* in (t + g, t + H]; no onset or glitch in (t, t + g]; no vote,
    left-censored onset or glitch in (t, o*]; (t, o*] observed (and (t, t + H] if `sym`).
    Negative: no onset, vote, censored onset or glitch in (t, t + H]; (t, t + H] observed.
    Observation also needs `tile_found` at t. Anything else is undefined (NaN).
    """
    t = np.asarray(t, dtype=float)
    b = sets.bids
    k = np.searchsorted(b, t + g, side="right")
    has = k < len(b)
    o_star = np.where(has, b[np.clip(k, 0, max(len(b) - 1, 0))] if len(b) else np.nan, np.nan)
    pos = has & (o_star <= t + H)
    span_end = np.where(pos, o_star, t + H)
    allon = sets.any
    flags = pd.DataFrame({
        "unobservable": ~obs.tile_found_at(t) | ~obs.observed(t, span_end)
        | (sym & ~obs.observed(t, t + H)),
        "guard": _count(allon, t, t + g) > 0,
        "onset_pending": _count(allon, t - config.ONSET_PENDING_S, t) > 0,
        "vote_or_censored": _count(sets.vote_or_censored, t, span_end) > 0,
        "glitch_window": _count(sets.glitches, t, span_end) > 0,
    })
    undefined = flags.to_numpy(bool).any(axis=1)
    y = np.where(undefined, np.nan, np.where(pos, 1.0, 0.0))
    return y, flags


def speech_labels(t, onsets: pd.DataFrame, *, H: float = config.H_S,
                  gs: float = config.G_SPEECH_S) -> np.ndarray:
    """T-speech target y_s(t) (§2.6, frozen spec), before causal masking."""
    t = np.asarray(t, dtype=float)
    if onsets is None or not len(onsets):
        return np.zeros(len(t))
    o = onsets.sort_values("onset_s", kind="stable")
    on, end = floats(o["onset_s"]), floats(o["turn_end_s"])
    fresh = bools(o["fresh"])
    guard = _count(on, t, t + gs) > 0
    k = np.searchsorted(on, t + gs, side="right")
    nxt_ok = k < len(on)
    nxt = on[np.clip(k, 0, len(on) - 1)]
    in_win = nxt_ok & (nxt <= t + H)
    pos = in_win & fresh[np.clip(k, 0, len(on) - 1)]
    last = np.searchsorted(on, t, side="right") - 1
    last_end = np.where(last >= 0, end[np.clip(last, 0, None)], -INF)
    neg_ok = (t + H - last_end) >= config.SPEECH_FRESH_SILENCE_S
    y = np.where(guard, np.nan, np.where(pos, 1.0, np.where(in_win, np.nan,
                                                             np.where(neg_ok, 0.0, np.nan))))
    return y


# --------------------------------------------------------------------------- points


def grid(info: ClassInfo, tables: dict | None = None) -> np.ndarray:
    """Decision grid t = 0, Δ, 2Δ, ... <= T_end (all codes, including `opening` and `end`,
    are kept as rows so the GRU sees every step)."""
    T = info.T_end
    if not np.isfinite(T):
        ends = [floats(table(tables, n)[c]).max() for n, c in
                (("hand_frames", "t_s"), ("words", "end_s")) if len(table(tables, n))]
        T = float(np.nanmax(ends)) if ends else 0.0
    return np.arange(0.0, T + 1e-9, config.DELTA_S)


def points(cid, tables: dict | None = None, *, retimed: bool = True,
           onsets: pd.DataFrame | None = None, t=None) -> pd.DataFrame:
    """The `points` table for one class (§3, §13)."""
    full = class_tables(tables if tables is not None else load_tables(class_ids=[cid]), cid)
    info = class_info(table(full, "classes"), cid)
    t = grid(info, full) if t is None else np.asarray(t, dtype=float)
    eps = table(full, "hand_episodes")
    sets = episode_sets(eps, retimed=retimed)
    obs = observability(table(full, "hand_frames"))
    flags_h = causal_rules(t, full, info, H=config.H_S)
    causal_any = flags_h.drop(columns="end").to_numpy(bool).any(axis=1)
    cols = {}
    for col in config.LABEL_COLUMNS:
        if col == SPEECH_LABEL:
            continue
        H, g, sym = LABEL_SPECS[col]
        y, _ = label_values(t, sets, obs, H=H, g=g, sym=sym)
        cols[col] = np.where(causal_any | (t + H > info.T_end), np.nan, y).astype(np.float32)
    y_head, label_flags = label_values(t, sets, obs, H=config.H_S, g=config.G_S)
    if onsets is None:
        onsets = speech_onsets(table(full, "voice"), table(full, "words"), eps, class_id=cid)
    ys = speech_labels(t, onsets)
    speech_mask = flags_h[list(SPEECH_CAUSAL)].to_numpy(bool).any(axis=1)
    cols[SPEECH_LABEL] = np.where(speech_mask, np.nan, ys).astype(np.float32)

    qa = info.row.get("qa_pass", pd.NA)
    qa_fail = not pd.isna(qa) and not bool(qa)          # NA: not decided yet -> no rule
    flags = pd.concat([flags_h, label_flags], axis=1)
    flags["class_qa"] = qa_fail
    excl = first_reason(flags, config.EXCL_REASONS)
    in_g = ~flags_h.to_numpy(bool).any(axis=1)
    state = np.where(np.isnan(y_head), config.LABEL_UNDEF,
                     np.where(y_head == 1.0, config.LABEL_POS, config.LABEL_NEG))
    state = np.where(causal_any | (t + config.H_S > info.T_end), config.LABEL_UNDEF, state)
    b = sets.bids
    k = np.searchsorted(b, t, side="right")
    nxt = np.where(k < len(b), b[np.clip(k, 0, max(len(b) - 1, 0))] if len(b) else np.nan,
                   np.nan)
    df = pd.DataFrame({"class_id": int(cid), "t_s": t, "in_G": in_g,
                       "at_risk": pd.isna(pd.Series(excl)).to_numpy(),
                       "excl_reason": pd.array(excl, dtype="string"), "label_state": state})
    for col in config.LABEL_COLUMNS:
        df[col] = cols[col]
    df["next_bid_s"] = nxt
    df["fold"] = pd.array([pd.NA] * len(t), dtype="Int64")
    df["in_vault"] = config.in_vault(cid)
    return schemas.coerce(df, "points")


def assign_folds(points: pd.DataFrame, classes: pd.DataFrame) -> pd.DataFrame:
    """`points.fold`: S-group outer folds assigned ONCE over all dev classes of the
    population (`splits.fold_column`, stratified by course, same-date twins grouped), so
    every condition, population subset and variant uses the same folds (§8.1) and a class
    tested in fold k is never in fold k's tuning data. Vault rows stay <NA>. With fewer dev
    classes than folds (pilot runs) the column stays <NA> and experiments assigns folds
    over the whole dev set at run time."""
    from . import splits

    cls = classes[classes["class_id"].isin(points["class_id"].unique())]
    if "course_code" not in cls:
        cls = cls.assign(course_code="other")
    out = points.copy()
    try:
        out["fold"] = splits.fold_column(out, cls).to_numpy()
    except ValueError as exc:                  # fewer classes than folds
        log.warning("labels: points.fold left <NA> (%s)", exc)
    out["fold"] = out["fold"].astype("Int64")
    return out


def at_risk(cid, tables: dict | None = None, **kwargs) -> pd.DataFrame:
    """The headline at-risk view of `points`: t_s, in_G, at_risk, excl_reason."""
    p = points(cid, tables, **kwargs)
    return p[["class_id", "t_s", "in_G", "at_risk", "excl_reason"]]


# --------------------------------------------------------------------------- CLI


def selected(classes: pd.DataFrame, args=None) -> pd.DataFrame:
    """The modelling population: classes `select` assigned a phase (vault included), or the
    explicit `--class` ids; `--phase P` narrows to one phase."""
    if getattr(args, "class_ids", None):
        return classes[classes["class_id"].isin([int(c) for c in args.class_ids])]
    if getattr(args, "split", None):               # the step-3 split (A11): train|validation|test|all
        from .splits import split_ids

        return classes[classes["class_id"].isin(split_ids(args.split))].reset_index(drop=True)
    if "phase" in classes:
        classes = classes[classes["phase"].notna()]
        if getattr(args, "phase", None):
            classes = classes[classes["phase"] == args.phase]
    return classes.reset_index(drop=True)


def cli(args) -> int:
    """`python -m oncue labels [--class CID ...] [--phase P]`: speech_onsets + points
    parquet tables for the selected classes."""
    tables = load_tables(("classes", "words", "voice", "segments", "hand_frames",
                          "hand_episodes"), class_ids=args.class_ids)
    classes = selected(table(tables, "classes"), args)
    if not len(classes):
        log.error("labels: no selected classes (run `oncue select` first)")
        return 1
    on_all, pts_all = [], []
    for cid in classes["class_id"].astype(int):
        ct = class_tables(tables, cid)
        on = speech_onsets(table(ct, "voice"), table(ct, "words"),
                           table(ct, "hand_episodes"), class_id=cid)
        on_all.append(on)
        pts_all.append(points(cid, ct, onsets=on))
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    on_df = schemas.validate(pd.concat(on_all, ignore_index=True), "speech_onsets")
    pts = assign_folds(pd.concat(pts_all, ignore_index=True), classes)
    pts = schemas.validate(pts, "points", strict=True)
    on_df.to_parquet(config.table_path("speech_onsets"), index=False)
    pts.to_parquet(config.table_path("points"), index=False)
    print(f"labels: {len(pts):,} points over {pts['class_id'].nunique()} classes; "
          f"{int(pts['at_risk'].sum()):,} at risk; "
          f"{int((pts[config.HEADLINE_LABEL] == 1).sum()):,} positive")
    return 0
