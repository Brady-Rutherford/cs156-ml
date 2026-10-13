"""Positive and negative controls (spec §9.2, §7; L7, L8, L9 in §9.1).

Positive controls: P-clock (offset-sensitive clock check, stop rule), P-plant (token `zqxw`
planted 8 s before 30% of bids, stop rule), IH+ (information horizon past t, including
Brady's words: `cutoff_shift`). Negative controls: CALLON check (are labels late against the
audio?), A-mismatch (deranged PCW, similarity columns only, x5), N-name (descriptive), label
shift. Audits: A-stream (streaming re-transcription, injected transcriber only), A-jitter.

Every function here works on in-memory tables (dev classes only) and returns aggregates;
nothing returns words. Steps that need another stage (feature rebuilds, re-transcription)
take that stage as an injected callable, so tests stay offline and no network path is
reachable by default.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold

from . import config

N_BOOT_DEFAULT = config.N_BOOT
# A-mismatch: only these columns are recomputed against the donor class's PCW (§9.2);
# class-level columns (pcw_has, pcw_my_words, pcw_frac_answered, pcw_n_code) are kept.
MISMATCH_SIM_RE = re.compile(r"^(?:pcw_res_\w+|pcw_unspent|pcw_bm25_gap|pcw_new_block"
                             r"|pcw_block_pos\w*|pcw_invest|lo_sim|sg_sim)$")
MISMATCH_KEEP = ("pcw_has", "pcw_my_words", "pcw_frac_answered", "pcw_n_code", "pcw_tmpl_has")
PLANT_WORD_DUR_S = 0.3
_TEXT_CLEAN_RE = re.compile(r"[^a-z0-9'\[\] ]+")


# --------------------------------------------------------------------------- helpers


def assert_dev(df: pd.DataFrame) -> pd.DataFrame:
    """Controls run on dev classes only (§8.1 L32)."""
    if "in_vault" in df and df["in_vault"].fillna(False).astype(bool).any():
        raise AssertionError("vault rows reached a dev control")
    return df


def dev_only(points: pd.DataFrame) -> pd.DataFrame:
    if "in_vault" not in points:
        return points
    return assert_dev(points[~points["in_vault"].fillna(False).astype(bool)])


def cluster_ci(values_by_class: pd.DataFrame, stat: Callable[[pd.DataFrame], float], *,
               n_boot: int = N_BOOT_DEFAULT, seed: int = config.SEED,
               alpha: float = config.ALPHA) -> tuple[float, float, float]:
    """(estimate, lo, hi): percentile CI from resampling classes with replacement."""
    est = stat(values_by_class)
    groups = values_by_class.groupby("class_id").indices
    keys = list(groups)
    if not keys:
        return est, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx_lists = [groups[k] for k in keys]
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        rows = np.concatenate([idx_lists[i] for i in pick])
        boots.append(stat(values_by_class.iloc[rows]))
    b = np.asarray(boots, float)
    b = b[np.isfinite(b)]
    if not len(b):
        return est, float("nan"), float("nan")
    return est, float(np.quantile(b, alpha / 2)), float(np.quantile(b, 1 - alpha / 2))


def bid_onsets(points: pd.DataFrame | None = None, episodes: pd.DataFrame | None = None
               ) -> pd.DataFrame:
    """(class_id, onset_s) of bids: from hand_episodes when given, else from points'
    `next_bid_s` (eval-only column)."""
    if episodes is not None:
        e = episodes[episodes["ep_type"] == config.EP_BID]
        e = e[~e["left_censored"].fillna(False).astype(bool)] if "left_censored" in e else e
        return e[["class_id", "onset_s"]].dropna().drop_duplicates().reset_index(drop=True)
    if points is None or "next_bid_s" not in points:
        raise KeyError("need hand_episodes or points.next_bid_s for bid onsets")
    o = points[["class_id", "next_bid_s"]].dropna().drop_duplicates()
    return o.rename(columns={"next_bid_s": "onset_s"}).reset_index(drop=True)


def _clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", _TEXT_CLEAN_RE.sub(" ", str(s).lower())).strip()


def _window_text(words: pd.DataFrame, a: float, b: float, *, col: str = "text_masked"
                 ) -> str:
    """Joined text of words starting in [a, b)."""
    w = words[(words["start_s"] >= a) & (words["start_s"] < b)]
    return " ".join(_clean_text(t) for t in w[col].astype(str))


def seg_type_at(segments: pd.DataFrame | None, class_id, t) -> np.ndarray:
    """Segment type current at times t (label-side use only)."""
    t = np.atleast_1d(np.asarray(t, float))
    if segments is None or not len(segments):
        return np.full(len(t), "other", dtype=object)
    s = segments[segments["class_id"] == class_id].sort_values("start_s")
    if not len(s):
        return np.full(len(t), "other", dtype=object)
    i = np.searchsorted(s["start_s"].to_numpy(float), t, side="right") - 1
    types = s["seg_type"].astype(object).to_numpy()
    return np.where(i >= 0, types[np.clip(i, 0, None)], "other")


# --------------------------------------------------------------------------- P-clock (§9.2)


def _coverage(starts: np.ndarray, ends: np.ndarray):
    """C(x) = total word time in (-inf, x], vectorised over x (words may overlap)."""
    s = np.sort(np.asarray(starts, float))
    e = np.sort(np.asarray(ends, float))
    cs = np.concatenate([[0.0], np.cumsum(s)])
    ce = np.concatenate([[0.0], np.cumsum(e)])

    def C(x):
        x = np.asarray(x, float)
        a = np.searchsorted(s, x, side="right")
        b = np.searchsorted(e, x, side="right")
        return (a * x - cs[a]) - (b * x - ce[b])
    return C


def speech_in_window(t, starts, ends, delta: float, width: float = 1.0) -> np.ndarray:
    """s_δ(f): seconds of word time in (t_f + δ - width, t_f + δ]."""
    C = _coverage(starts, ends)
    t = np.asarray(t, float)
    return C(t + delta) - C(t + delta - width)


def _ap(y, s) -> float:
    y = np.asarray(y)
    return float(average_precision_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def p_clock_curve(t, z, starts, ends, deltas=config.P_CLOCK_DELTAS_S) -> pd.DataFrame:
    """AP(δ) of the speech score s_δ for the border indicator z."""
    return pd.DataFrame({"delta_s": list(deltas),
                         "ap": [_ap(z, speech_in_window(t, starts, ends, d)) for d in deltas]})


def p_clock(samples: pd.DataFrame, words: pd.DataFrame, *,
            deltas=config.P_CLOCK_DELTAS_S, peak_range=config.P_CLOCK_PEAK_RANGE_S,
            sharpness_min: float = config.P_CLOCK_SHARPNESS,
            class_tol: float = config.AV_LAG_TOL_S, width: float = 1.0) -> dict:
    """P-clock (MVP, stop rule).

    `samples`: class_id, t_s, z (1 when the tracked tile is bordered: Brady's tile for the
    Brady version, any tile for the any-speaker version), restricted by the caller to
    tracked main-room samples. `words`: class_id, start_s, end_s of the matching speakers
    (Brady-attributed words, or all words). Pass when the peak lies in [-1.0, +0.25] s,
    AP(δ*) - max(AP(δ* - 2), AP(δ* + 2)) >= 0.1, and every class's δ*_c is within ±0.5 s of
    the median δ*_c. A failure stops work until alignment is fixed.

    Peak location: with the trailing window (t + δ - 1, t + δ], AP(δ) peaks at
    δ* = width/2 - L for a border lagging speech by L (the window's centre, not its end, meets
    the speech that caused the border). The pre-registered range is therefore applied to
    -L = δ* - width/2 (`lag_s` = L is reported); a perfectly aligned clock with zero UI lag
    gives δ* = +0.5 and passes."""
    samples = assert_dev(samples)
    t_all, z_all, s_by_delta = [], [], {d: [] for d in deltas}
    extra = {}
    per_class = []
    for cid, smp in samples.groupby("class_id"):
        w = words[words["class_id"] == cid]
        t, z = smp["t_s"].to_numpy(float), smp["z"].astype(float).to_numpy()
        st, en = w["start_s"].to_numpy(float), w["end_s"].to_numpy(float)
        cur = p_clock_curve(t, z, st, en, deltas)
        if cur["ap"].notna().any():
            per_class.append({"class_id": cid,
                              "delta_star": float(cur.loc[cur["ap"].idxmax(), "delta_s"])})
        t_all.append(t)
        z_all.append(z)
        for d in deltas:
            s_by_delta[d].append(speech_in_window(t, st, en, d))
        extra[cid] = (t, st, en)
    if not t_all:
        return {"pass": False, "reason": "no samples"}
    z = np.concatenate(z_all)
    curve = pd.DataFrame({"delta_s": list(deltas),
                          "ap": [_ap(z, np.concatenate(s_by_delta[d])) for d in deltas]})
    d_star = float(curve.loc[curve["ap"].idxmax(), "delta_s"])

    def pooled_ap(delta):
        s = np.concatenate([speech_in_window(t, st, en, delta) for t, st, en in extra.values()])
        return _ap(z, s)
    ap_star = float(curve["ap"].max())
    sharp = ap_star - np.nanmax([pooled_ap(d_star - 2.0), pooled_ap(d_star + 2.0)])
    pc = pd.DataFrame(per_class, columns=["class_id", "delta_star"])
    med = float(pc["delta_star"].median()) if len(pc) else float("nan")
    off = pc[(pc["delta_star"] - med).abs() > class_tol]
    lag = width / 2 - d_star
    checks = {"peak_in_range": peak_range[0] <= -lag <= peak_range[1],
              "sharp": bool(sharp >= sharpness_min),
              "classes_consistent": len(off) == 0}
    return {"pass": all(checks.values()), "checks": checks, "delta_star": d_star,
            "lag_s": lag, "ap_star": ap_star, "sharpness": float(sharp), "curve": curve,
            "per_class": pc, "n_classes": len(pc), "n_off_classes": len(off),
            "median_class_delta_star": med}


def p_clock_inputs(hand_frames: pd.DataFrame, words: pd.DataFrame, *, any_speaker: bool,
                   me_border: pd.Series | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(samples, words) for `p_clock`. Any-speaker: z = border_any on tracked samples, all
    words. Brady version: z = `me_border` (his tracked tile's border, from hand_tiles,
    aligned with hand_frames rows), Brady-role words only."""
    hf = hand_frames[hand_frames["tile_found"].fillna(False).astype(bool)]
    if any_speaker:
        z = hf["border_any"].astype(float)
        w = words
    else:
        if me_border is None:
            raise ValueError("the Brady version needs his tile's border per sample")
        z = me_border.loc[hf.index].astype(float)
        w = words[words["role"] == config.ROLE_ME]
    samples = pd.DataFrame({"class_id": hf["class_id"].to_numpy(), "t_s": hf["t_s"].to_numpy(),
                            "z": z.to_numpy()})
    return samples, w[["class_id", "start_s", "end_s"]]


# --------------------------------------------------------------------------- P-plant (§9.2)


def choose_planted(onsets: pd.DataFrame, frac: float = config.PLANT_FRAC,
                   seed: int = config.SEED) -> pd.DataFrame:
    """A seeded `frac` share of bids (training and test alike) that get the planted token."""
    o = onsets.sort_values(["class_id", "onset_s"]).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    k = round(frac * len(o))
    pick = np.sort(rng.choice(len(o), size=k, replace=False)) if k else np.array([], int)
    return o.iloc[pick].reset_index(drop=True)


def plant_words(words: pd.DataFrame, planted: pd.DataFrame, *,
                lead_s: float = config.PLANT_LEAD_S, token: str = config.PLANT_TOKEN,
                dur_s: float = PLANT_WORD_DUR_S) -> pd.DataFrame:
    """Insert a professor word `token` at o - lead for each planted bid (full-rebuild path:
    frames, windows and every feature are then recomputed from these words)."""
    rows = []
    for r in planted.itertuples():
        start = float(r.onset_s) - lead_s
        rows.append({"class_id": r.class_id, "start_s": start, "end_s": start + dur_s,
                     "text_masked": token, "text_norm": token, "has_qmark": False,
                     "conf": 1.0, "filler": False, "spk_hash": "planted",
                     "role": config.ROLE_PROF, "role_event_start_s": start,
                     "role_avail_s": start + dur_s + config.EPS_W_S, "role_fused":
                     config.ROLE_PROF, "how": "planted"})
    if not rows:
        return words.copy()
    out = pd.concat([words, pd.DataFrame(rows)], ignore_index=True)
    out = out.sort_values(["class_id", "start_s"], kind="stable").reset_index(drop=True)
    if "w_idx" in out:
        out["w_idx"] = out.groupby("class_id").cumcount()
    return out


def plant_visible(t_s, onset_s, *, lead_s: float = config.PLANT_LEAD_S,
                  dur_s: float = PLANT_WORD_DUR_S, window_s: float = config.WIN_LONG_S,
                  eps_w: float = config.EPS_W_S) -> np.ndarray:
    """True where a planted word ending at o - lead + dur is inside (t - window, t - ε_w]."""
    end = np.asarray(onset_s, float) - lead_s + dur_s
    t = np.asarray(t_s, float)
    return (end > t - window_s) & (end <= t - eps_w)


def plant_design(X: pd.DataFrame, planted: pd.DataFrame, *,
                 lead_s: float = config.PLANT_LEAD_S, token: str = config.PLANT_TOKEN
                 ) -> tuple[pd.DataFrame, np.ndarray]:
    """Fast path for M1/M2: append `p_<token>` to every text window (all variants) whose
    (t - w, t - ε_w] contains the planted word. Returns (design, visible-in-60s mask).
    Only bag-of-n-gram inputs are exact under this path; embeddings are not re-encoded."""
    X = X.copy()
    tok = config.ROLE_PREFIX[config.ROLE_PROF] + token
    vis60 = np.zeros(len(X), bool)
    by_class = {c: g["onset_s"].to_numpy(float) for c, g in planted.groupby("class_id")}
    cids = X["class_id"].to_numpy()
    t = X["t_s"].to_numpy(float)
    for window, w_s in (("all60", config.WIN_LONG_S), ("all15", config.WIN_SHORT_S)):
        hit = np.zeros(len(X), bool)
        for cid, ons in by_class.items():
            m = cids == cid
            if m.any():
                hit[m] = plant_visible(t[m][:, None], ons[None, :], lead_s=lead_s,
                                       window_s=w_s).any(axis=1)
        if window == "all60":
            vis60 = hit
        for col in [c for c in X.columns if c.startswith(f"txt_{window}")]:
            s = X[col].fillna("").astype(str).to_numpy(object)
            s[hit] = [(v + " " + tok).strip() for v in s[hit]]
            X[col] = s
    return X, vis60


def planted_ap(y, p, planted_visible) -> float:
    """P-plant score: AP on planted positives (positives whose window holds the token) vs all
    negatives. Must reach >= 0.9 (§9.2) for M1 and M2, or work stops."""
    y = np.asarray(y, float)
    keep = (y == 0) | ((y == 1) & np.asarray(planted_visible, bool))
    return _ap(y[keep], np.asarray(p, float)[keep])


# --------------------------------------------------------------------------- CALLON (§9.2)


def callon_hits(text: str) -> int:
    return len(config.CALLON_RE.findall(text))


def matched_points(points: pd.DataFrame, onsets: pd.DataFrame,
                   segments: pd.DataFrame | None = None, *, k: int = config.CALLON_MATCHED,
                   dist=config.CALLON_MATCH_DIST_S, target: str = config.HEADLINE_LABEL,
                   seed: int = config.SEED) -> pd.DataFrame:
    """Up to k matched non-bid points per bid: same class, same seg_type,
    |b - o| in [60, 600] s, at risk, y = 0."""
    rng = np.random.default_rng(seed)
    pts = dev_only(points)
    ok = pts[target].eq(0)
    if "at_risk" in pts:
        ok &= pts["at_risk"].fillna(False).astype(bool)
    elif "in_G" in pts:
        ok &= pts["in_G"].fillna(False).astype(bool)
    cand = pts[ok]
    rows = []
    for cid, ons in onsets.groupby("class_id"):
        c = cand[cand["class_id"] == cid]
        if not len(c):
            continue
        ct = c["t_s"].to_numpy(float)
        cseg = seg_type_at(segments, cid, ct)
        for o in ons["onset_s"].to_numpy(float):
            oseg = seg_type_at(segments, cid, [o])[0]
            d = np.abs(ct - o)
            pool = np.flatnonzero((d >= dist[0]) & (d <= dist[1]) & (cseg == oseg))
            pick = rng.choice(pool, size=min(k, len(pool)), replace=False) if len(pool) else []
            rows += [{"class_id": cid, "onset_s": o, "t_s": ct[i]} for i in pick]
    return pd.DataFrame(rows, columns=["class_id", "onset_s", "t_s"])


def callon_check(words: pd.DataFrame, onsets: pd.DataFrame, points: pd.DataFrame,
                 segments: pd.DataFrame | None = None, *,
                 window_s: float = config.CALLON_WINDOW_S, n_boot: int = N_BOOT_DEFAULT,
                 seed: int = config.SEED) -> dict:
    """CALLON check (MVP): rate of CALLON phrases in the professor's words in [o - 5, o)
    against 3 matched points per bid, as a rate ratio with a cluster-bootstrap CI. Pass if the
    CI includes 1 or lies below it; a failure means labels run late against the audio.

    Reads `text_masked` (CALLON phrases are deleted from `text_norm`)."""
    prof = words[words["role"] == config.ROLE_PROF]
    match = matched_points(points, onsets, segments, seed=seed)
    rows = []
    for cid, ons in onsets.groupby("class_id"):
        pw = prof[prof["class_id"] == cid]
        for o in ons["onset_s"].to_numpy(float):
            rows.append({"class_id": cid, "kind": "bid",
                         "hits": callon_hits(_window_text(pw, o - window_s, o))})
    for r in match.itertuples():
        pw = prof[prof["class_id"] == r.class_id]
        rows.append({"class_id": r.class_id, "kind": "matched",
                     "hits": callon_hits(_window_text(pw, r.t_s - window_s, r.t_s))})
    df = pd.DataFrame(rows, columns=["class_id", "kind", "hits"])

    def ratio(d):
        b = d.loc[d["kind"] == "bid", "hits"].mean()
        m = d.loc[d["kind"] == "matched", "hits"].mean()
        if not np.isfinite(b) or not np.isfinite(m):
            return float("nan")
        if m == 0:
            return float("nan") if b == 0 else float("inf")
        return float(b / m)
    est, lo, hi = cluster_ci(df, ratio, n_boot=n_boot, seed=seed)
    rate_b = float(df.loc[df["kind"] == "bid", "hits"].mean()) if len(df) else float("nan")
    rate_m = float(df.loc[df["kind"] == "matched", "hits"].mean()) if len(df) else float("nan")
    no_phrases = (rate_b == 0 and rate_m == 0)
    passed = bool(no_phrases or (np.isfinite(lo) and lo <= 1.0) or
                  (not np.isfinite(lo) and rate_b <= rate_m))
    return {"pass": passed, "rate_ratio": est, "ci_lo": lo, "ci_hi": hi,
            "rate_bid": rate_b, "rate_matched": rate_m,
            "n_bids": int((df["kind"] == "bid").sum()),
            "n_matched": int((df["kind"] == "matched").sum()),
            "n_classes": int(df["class_id"].nunique())}


# --------------------------------------------------------------------------- N-name (§9.2)


def _name_count(text: str) -> int:
    """Name tokens in masked text. Brady's own name counts too: `words.text_masked` stores it
    as [ME] (kept apart for F-name), but under §1.3 it is a name token like any other."""
    text = str(text)
    return text.count(config.NAME_TOKEN) + text.count(config.ME_TOKEN)


def name_rates(words: pd.DataFrame, onsets: pd.DataFrame, matched: pd.DataFrame | None = None,
               *, n_boot: int = N_BOOT_DEFAULT, seed: int = config.SEED) -> pd.DataFrame:
    """[NAME] tokens per minute of professor speech in [o - 15, o - 2] vs [o, o + 10] vs
    matched non-bid points ([b - 15, b - 2]), pooled with cluster-bootstrap CIs. Descriptive
    only: professors also name previous speakers."""
    prof = words[words["role"] == config.ROLE_PROF]
    prof = prof.assign(dur=(prof["end_s"] - prof["start_s"]).clip(lower=0),
                       names=prof["text_masked"].map(_name_count))
    rows = []

    def add(kind, cid, a, b):
        w = prof[(prof["class_id"] == cid) & (prof["start_s"] >= a) & (prof["start_s"] < b)]
        rows.append({"class_id": cid, "kind": kind, "names": int(w["names"].sum()),
                     "minutes": float(w["dur"].sum()) / 60})
    for r in onsets.itertuples():
        add("pre_bid", r.class_id, r.onset_s - 15, r.onset_s - 2)
        add("post_bid", r.class_id, r.onset_s, r.onset_s + 10)
    if matched is not None:
        for r in matched.itertuples():
            add("matched", r.class_id, r.t_s - 15, r.t_s - 2)
    df = pd.DataFrame(rows, columns=["class_id", "kind", "names", "minutes"])
    out = []
    for kind, d in df.groupby("kind"):
        est, lo, hi = cluster_ci(d, lambda x: x["names"].sum() / max(x["minutes"].sum(), 1e-9),
                                 n_boot=n_boot, seed=seed)
        out.append({"window": kind, "rate_per_min": est, "ci_lo": lo, "ci_hi": hi,
                    "n_windows": len(d), "n_classes": d["class_id"].nunique()})
    return pd.DataFrame(out)


def name_counts_at(points: pd.DataFrame, words: pd.DataFrame,
                   windows=(config.WIN_SHORT_S, config.WIN_LONG_S),
                   eps_w: float = config.EPS_W_S) -> pd.DataFrame:
    """Causal [NAME] counts of professor words with end in (t - w, t - ε_w] at each point."""
    prof = words[words["role"] == config.ROLE_PROF]
    out = {f"name_n_{int(w)}": np.zeros(len(points)) for w in windows}
    for cid, idx in points.groupby("class_id").indices.items():
        pw = prof[prof["class_id"] == cid].sort_values("end_s")
        end = pw["end_s"].to_numpy(float)
        cs = np.concatenate([[0], np.cumsum(pw["text_masked"].map(_name_count).to_numpy())])
        t = points["t_s"].to_numpy(float)[idx]
        hi = np.searchsorted(end, t - eps_w, side="right")
        for w in windows:
            lo = np.searchsorted(end, t - w, side="right")
            out[f"name_n_{int(w)}"][idx] = cs[hi] - cs[lo]
    return pd.DataFrame(out, index=points.index)


def name_lr_lift(counts: pd.DataFrame, y, groups, *, n_boot: int = N_BOOT_DEFAULT,
                 seed: int = config.SEED) -> dict:
    """Grouped-CV AP and lift of an LR on [NAME] counts, with a cluster-bootstrap CI."""
    y = np.asarray(y, float)
    groups = np.asarray(groups)
    p = np.full(len(y), np.nan)
    k = min(config.N_OUTER_FOLDS, len(np.unique(groups)))
    for tr, te in GroupKFold(k).split(counts, y, groups):
        if len(np.unique(y[tr])) < 2:
            p[te] = y[tr].mean()
            continue
        lr = LogisticRegression(max_iter=1000).fit(np.log1p(counts.iloc[tr]), y[tr])
        p[te] = lr.predict_proba(np.log1p(counts.iloc[te]))[:, 1]
    d = pd.DataFrame({"class_id": groups, "y": y, "p": p})
    est, lo, hi = cluster_ci(d, lambda x: _ap(x["y"], x["p"]) / max(x["y"].mean(), 1e-12),
                             n_boot=n_boot, seed=seed)
    return {"ap": _ap(y, p), "lift": est, "lift_ci_lo": lo, "lift_ci_hi": hi,
            "base_rate": float(y.mean())}


def name_flag(delta_ap_ci_lo: float, threshold: float = config.N_NAME_FLAG_DAP) -> bool:
    """N-name's one flag: concerning if the CI lower bound of ΔAP(F-name − F-all) > 0.02."""
    return bool(np.isfinite(delta_ap_ci_lo) and delta_ap_ci_lo > threshold)


# --------------------------------------------------------------------------- A-mismatch (§9.2)


def derangement(items, rng) -> list:
    """A uniformly random derangement (no fixed points) by rejection sampling."""
    items = list(items)
    if len(items) < 2:
        raise ValueError("a derangement needs at least 2 items")
    while True:
        perm = list(rng.permutation(len(items)))
        if all(i != j for i, j in enumerate(perm)):
            return [items[j] for j in perm]


def derangements(classes: pd.DataFrame, *, n: int = config.MISMATCH_N,
                 seed: int = config.SEED, section_col: str = "section_id"
                 ) -> list[dict]:
    """`n` donor maps {class_id: donor_class_id}, deranging PCW within each section (among the
    classes given, i.e. the PCW subset). Classes alone in their section get no donor and drop
    out of A-mismatch (reported by the caller)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        mapping = {}
        for _, g in classes.groupby(section_col):
            cids = list(g["class_id"])
            if len(cids) >= 2:
                mapping.update(dict(zip(cids, derangement(cids, rng))))
        out.append(mapping)
    return out


def mismatch_columns(columns) -> list[str]:
    return [c for c in columns if MISMATCH_SIM_RE.match(c)]


def mismatch_pcw(features: pd.DataFrame, mapping: dict,
                 recompute: Callable[[object, object, pd.DataFrame], pd.DataFrame]
                 ) -> pd.DataFrame:
    """Features with only the PCW *similarity* columns recomputed against the donor class's
    PCW (`recompute(class_id, donor_id, rows) -> DataFrame` of similarity columns aligned with
    `rows`, supplied by the features/pcw stage). Class-level PCW columns stay the class's own.
    Rows of classes without a donor are dropped."""
    sim = mismatch_columns(features.columns)
    keep = features["class_id"].isin(list(mapping))
    out = features[keep].copy()
    for cid, donor in mapping.items():
        m = out["class_id"] == cid
        if not m.any():
            continue
        new = recompute(cid, donor, out.loc[m])
        bad = [c for c in new.columns if c not in sim]
        if bad:
            raise ValueError(f"A-mismatch may recompute only similarity columns; got {bad}")
        out.loc[m, list(new.columns)] = new.to_numpy()
    return out


def mismatch_summary(runs: pd.DataFrame, metrics=("ap_within", "ap")) -> dict:
    """Mean ± sd over the derangement runs (results rows of the A-mismatch conditions)."""
    return {m: {"mean": float(runs[m].mean()), "sd": float(runs[m].std(ddof=1)),
                "n": int(runs[m].notna().sum())} for m in metrics if m in runs}


# --------------------------------------------------------------------------- IH / cut-off shift


def cutoff_shift(points: pd.DataFrame, delta_s: float, *, include_me: bool = False
                 ) -> pd.DataFrame:
    """Query times for the information horizon: features at t + δ, labels at t. δ > 0 is the
    illegal side (shaded in the IH figure); IH+ (+30, +60) includes Brady's own words."""
    out = points[["class_id", "t_s"]].copy()
    out["t_feat_s"] = out["t_s"] + float(delta_s)
    out["include_me"] = bool(include_me)
    out["illegal"] = float(delta_s) > 0
    return out


# --------------------------------------------------------------------------- label shift, jitter


def label_shift(hand_episodes: pd.DataFrame, shift_s: float) -> pd.DataFrame:
    """LS (should): every onset moved `shift_s` earlier; labels.py then recomputes y."""
    e = hand_episodes.copy()
    for c in ("onset_s", "onset_retimed_s"):
        if c in e:
            e[c] = e[c] - float(shift_s)
    return e


JITTER_TIME_COLUMNS = {
    "words": (("start_s", "end_s", "role_event_start_s", "role_avail_s"),),
    "frames": (("start_s", "end_s", "close_s"),),
    "voice": (("start_s", "end_s", "first_word_avail_s", "start_avail_s", "end_avail_s"),),
    "segments": (("start_s",),),
    "chat": (("t_s",),),
    "polls": (("open_s", "close_s", "my_answer_avail_s"),),
    "presence": (("enter_s",),),
}


def jitter_times(tables: dict[str, pd.DataFrame], sd_s: float = config.A_JITTER_SD_S,
                 seed: int = config.SEED) -> dict[str, pd.DataFrame]:
    """A-jitter (could): add N(0, sd) to every time; one draw per row, shared by that row's
    time columns so durations are kept."""
    rng = np.random.default_rng(seed)
    out = {}
    for name, df in tables.items():
        df = df.copy()
        for cols in JITTER_TIME_COLUMNS.get(name, ()):
            present = [c for c in cols if c in df]
            if present:
                noise = rng.normal(0.0, sd_s, len(df))
                for c in present:
                    df[c] = df[c] + noise
        out[name] = df
    return out


# --------------------------------------------------------------------------- A-stream (§9.1 L7)


def stream_sample(points: pd.DataFrame, n: int = config.A_STREAM_N, *,
                  target: str = config.HEADLINE_LABEL, seed: int = config.SEED
                  ) -> pd.DataFrame:
    """n/2 positive and n/2 negative at-risk dev points (seeded) for the A-stream audit."""
    pts = dev_only(points)
    ok = pts[target].notna()
    if "in_G" in pts:
        ok &= pts["in_G"].fillna(False).astype(bool)
    pts = pts[ok]
    rng = np.random.default_rng(seed)
    parts = []
    for val in (1.0, 0.0):
        pool = pts[pts[target] == val]
        k = min(n // 2, len(pool))
        parts.append(pool.iloc[np.sort(rng.choice(len(pool), k, replace=False))])
    return pd.concat(parts)[["class_id", "t_s", target]].reset_index(drop=True)


def splice_stream_words(batch: pd.DataFrame, stream: pd.DataFrame, t0: float, t1: float
                        ) -> pd.DataFrame:
    """Batch words outside [t0, t1] plus the stream words inside it (one class)."""
    keep = batch[(batch["end_s"] < t0) | (batch["start_s"] > t1)]
    ins = stream[(stream["start_s"] >= t0) & (stream["end_s"] <= t1)]
    out = pd.concat([keep, ins], ignore_index=True).sort_values("start_s", kind="stable")
    out = out.reset_index(drop=True)
    if "w_idx" in out:
        out["w_idx"] = np.arange(len(out))
    return out


def punct_case_disagreement(batch: pd.DataFrame, stream: pd.DataFrame, *,
                            col: str = "text_masked", tol_s: float = 0.25) -> dict:
    """Share of time-matched words whose trailing punctuation or casing differ."""
    if not len(batch) or not len(stream):
        return {"n_matched": 0, "punct": float("nan"), "case": float("nan")}
    b = batch.sort_values("start_s")
    s = stream.sort_values("start_s")
    bs = b["start_s"].to_numpy(float)
    j = np.clip(np.searchsorted(bs, s["start_s"].to_numpy(float)), 0, len(bs) - 1)
    near = np.abs(bs[j] - s["start_s"].to_numpy(float)) <= tol_s
    bw = b[col].astype(str).to_numpy()[j][near]
    sw = s[col].astype(str).to_numpy()[near]

    def punct(w):
        return re.sub(r"^.*?([^\w\]]*)$", r"\1", w)
    return {"n_matched": int(near.sum()),
            "punct": float(np.mean([punct(x) != punct(y) for x, y in zip(bw, sw)]))
            if near.any() else float("nan"),
            "case": float(np.mean([x[:1].isupper() != y[:1].isupper() for x, y in zip(bw, sw)]))
            if near.any() else float("nan")}


def stream_audit(sample: pd.DataFrame, *, transcribe: Callable | None = None,
                 rebuild: Callable | None = None, score: Callable | None = None,
                 words_by_class: dict | None = None,
                 span_s: float = config.A_STREAM_SPAN_S) -> pd.DataFrame:
    """A-stream: for each sampled point, re-transcribe [t - 180, t] with the injected
    `transcribe(class_id, t0, t1) -> words` (consented courses only; there is deliberately no
    default, so nothing here can reach a network), splice the stream words into the batch
    history, `rebuild(class_id, t, words) -> design row`, and `score(row) -> p` with the
    frozen fold model. Returns the sample with `p_stream` added."""
    if transcribe is None or rebuild is None or score is None:
        raise RuntimeError("stream_audit needs injected transcribe, rebuild and score "
                           "callables (no default ASR path exists)")
    p = []
    for r in sample.itertuples():
        t0, t1 = float(r.t_s) - span_s, float(r.t_s)
        batch = (words_by_class or {}).get(r.class_id)
        stream = transcribe(r.class_id, t0, t1)
        words = splice_stream_words(batch, stream, t0, t1) if batch is not None else stream
        p.append(float(score(rebuild(r.class_id, float(r.t_s), words))))
    return sample.assign(p_stream=p)


def stream_summary(df: pd.DataFrame, *, target: str = config.HEADLINE_LABEL,
                   n_boot: int = N_BOOT_DEFAULT, seed: int = config.SEED) -> dict:
    """Spearman(p_full, p_stream) and AP both ways with a paired cluster-bootstrap CI of the
    difference (full - stream)."""
    from scipy.stats import spearmanr

    rho = spearmanr(df["p_full"], df["p_stream"]).statistic if len(df) > 2 else float("nan")
    d = df.rename(columns={target: "y"})
    est, lo, hi = cluster_ci(d, lambda x: _ap(x["y"], x["p_full"]) - _ap(x["y"], x["p_stream"]),
                             n_boot=n_boot, seed=seed)
    return {"spearman": float(rho), "ap_full": _ap(d["y"], d["p_full"]),
            "ap_stream": _ap(d["y"], d["p_stream"]), "delta_ap": est, "ci_lo": lo,
            "ci_hi": hi, "n": len(df)}


# --------------------------------------------------------------------------- CLI


CONTROL_NAMES = ("p-clock", "callon", "n-name")


def _out_path(name: str):
    path = config.DATA_DIR / "controls" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _jsonable(obj):
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def cli(args) -> int:
    """`python -m oncue controls [--control p-clock|callon|n-name ...]` on local tables.

    Model-based controls (P-plant, A-mismatch, IH, LS) are conditions in experiments.py and
    run through `oncue train`. Outputs are aggregate JSON under data/controls/."""
    import argparse

    p = argparse.ArgumentParser(prog="oncue controls")
    p.add_argument("--control", action="append", choices=CONTROL_NAMES)
    p.add_argument("--n-boot", type=int, default=N_BOOT_DEFAULT)
    opts = p.parse_args(getattr(args, "extra", []) or [])
    names = opts.control or list(CONTROL_NAMES)

    def table(name):
        path = config.table_path(name)
        if not path.exists():
            raise SystemExit(f"oncue controls: missing {path} (run the earlier stages first)")
        return pd.read_parquet(path)
    points = dev_only(table("points"))
    dev = set(points["class_id"])
    words = table("words")
    words = words[words["class_id"].isin(dev)]
    episodes = table("hand_episodes")
    onsets = bid_onsets(episodes=episodes[episodes["class_id"].isin(dev)])
    segments = table("segments")
    for name in names:
        if name == "p-clock":
            hf = table("hand_frames")
            res = p_clock(*p_clock_inputs(hf[hf["class_id"].isin(dev)], words,
                                          any_speaker=True))
        elif name == "callon":
            res = callon_check(words, onsets, points, segments, n_boot=opts.n_boot)
        else:
            match = matched_points(points, onsets, segments)
            res = {"rates": name_rates(words, onsets, match, n_boot=opts.n_boot)}
        _out_path(name).write_text(json.dumps(_jsonable(res), indent=1, default=float))
        print(f"{name}: pass={res.get('pass', 'n/a')}")
    return 0
