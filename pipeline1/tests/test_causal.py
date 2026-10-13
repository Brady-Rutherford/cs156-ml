"""The causal contract (§4.4). Blocking: training must not start while this fails.

1. Truncation: `features.build(full, t)` (fast, vectorised) equals the reference
   `CausalView(cid, t)` recomputed from `truncate(full, t)`, for random (class, t) plus
   points sitting exactly on availability boundaries.
2. Sentinels: future words, a professor voice event at t - eps_F/2, a segment at t + 1, a
   Brady frame at t + 5 that repeats the current window, CALLON / [NAME] tokens at t - 5, a
   recording-gap event, a Brady poll answer submitted at t + 3 and a future raise change
   no feature.
3. Straddle: a word with start < t - eps_w < end never shows its text; a voice event
   spanning t exposes only its elapsed time.
4. Mask causality: the causal at-risk rules are unchanged under truncation.
5. Allowlist: feature columns are allowlisted; has_qmark only in F-punct.

Synthetic world only. ONCUE_CAUSAL_N sets the number of random points (default 500, the
spec's blocking run size; lower it for quick local runs). ONCUE_CAUSAL_REAL=1 also runs the
truncation test on data/tables (>= 5 dev classes).
"""

from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd
import pytest
from oncue import causal, config, labels, schemas, synthetic
from oncue import features as F

N_RANDOM = int(os.environ.get("ONCUE_CAUSAL_N", "500"))
EPS_W = config.EPS_W_S


# --------------------------------------------------------------------------- fixtures


class FakeEdu:
    """Deterministic stand-in for the EduBehaviors scorers (scores in [0, 1] from text)."""

    columns = tuple(config.EDU_CHECKPOINTS)

    def scores(self, texts):
        out = np.zeros((len(texts), len(self.columns)))
        for i, x in enumerate(texts):
            h = int(F.text_key(x)[:8], 16)
            out[i] = [((h >> (4 * k)) & 15) / 15.0 for k in range(len(self.columns))]
        return out


@pytest.fixture(scope="module")
def enc():
    return F.Memo(F.HashEncoder(48))


@pytest.fixture(scope="module")
def znull(enc):
    rng = np.random.default_rng(7)
    vocab = " ".join(synthetic.TOPICS.values()).split()
    return F.ZNull(enc([" ".join(rng.choice(vocab, 12)) for _ in range(200)]))


def class_tables_for(sc, truth_tables) -> dict:
    """Truth tables plus the small extras features read (prep text, breakout group)."""
    tb = dict(truth_tables)
    tb["classes"] = tb["classes"].assign(n_sessions=4)
    tb["prep"] = F.prep_table(sc.class_id, sc.raw["class_edit_page"])
    users = sc.raw["breakouts"][0]["users"] if sc.raw["breakouts"] else []
    tb["breakout_groups"] = pd.DataFrame({
        "class_id": sc.class_id, "spk_hash": [synthetic.spk_hash(str(u["id"])) for u in users]})
    return tb


@pytest.fixture(scope="module")
def tables(world, truth):
    return {sc.class_id: class_tables_for(sc, truth[sc.class_id]) for sc in world}


@pytest.fixture(scope="module")
def kwargs(world, enc, znull):
    rng = np.random.default_rng(11)
    vocab = " ".join(synthetic.TOPICS.values()).split()
    refs = [enc([" ".join(rng.choice(vocab, 30)) for _ in range(3)]) for _ in range(3)]
    out = {}
    for k, sc in enumerate(world):
        out[sc.class_id] = {"encoder": enc, "znull": znull, "edu": FakeEdu(),
                            "pcw_ref_sets": refs if k == 0 else None}
    return out


def boundary_times(tb: dict, info: causal.ClassInfo, rng, n: int = 14) -> np.ndarray:
    """Times sitting exactly on availability boundaries (the <= conventions)."""
    w = causal.prep_words(tb["words"])
    fr = causal.build_frames(w)
    v, e, p = tb["voice"], tb["hand_episodes"], tb["polls"]
    cands = np.concatenate([
        fr.close, w.end + EPS_W, w.role_av, w.start + EPS_W,
        causal.floats(v["start_s"]) + info.eps_f, causal.floats(v["end_s"]) + info.eps_f,
        causal.floats(tb["segments"]["start_s"]) + info.eps_f,
        causal.floats(e["offset_live_s"]), causal.floats(e["onset_s"]) + 10.0,
        causal.floats(e["offset_s"]) + 5.0, causal.floats(p["my_answer_avail_s"]),
        causal.floats(p["open_s"]) + info.eps_p, fr.close + 3.0])
    cands = cands[np.isfinite(cands) & (cands > 0) & (cands < info.T_end)]
    return rng.choice(cands, min(n, len(cands)), replace=False)


def sample_times(world, tables, n_total: int, seed: int = 156) -> dict:
    rng = np.random.default_rng(seed)
    per = max(1, n_total // len(world))
    out = {}
    for sc in world:
        info = causal.class_info(tables[sc.class_id]["classes"], sc.class_id)
        rand = rng.uniform(0.0, info.T_end, per)
        out[sc.class_id] = np.sort(np.concatenate(
            [rand, boundary_times(tables[sc.class_id], info, rng)]))
    return out


def reference(cid, ts, tb, kw) -> pd.DataFrame:
    return causal.reference_frame([causal.CausalView(cid, t, tb, **kw) for t in ts])


def assert_same(fast: pd.DataFrame, ref: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(fast.reset_index(drop=True), ref.reset_index(drop=True),
                                  check_exact=False, rtol=1e-5, atol=1e-6)


# --------------------------------------------------------------------------- truncation


def test_truncate_hides_the_future(world, tables):
    sc = world.classes[0]
    tb = tables[sc.class_id]
    info = causal.class_info(tb["classes"], sc.class_id)
    t = 600.0
    tr = causal.truncate(tb, t, info=info)
    w = tr["words"]
    assert (w["start_s"] + EPS_W <= t).all()
    late = ~(tb["words"]["end_s"] + EPS_W <= t) & (tb["words"]["start_s"] + EPS_W <= t)
    assert w["text_norm"].isna().sum() == late.sum() and w["end_s"].isna().sum() == late.sum()
    assert w["role_fused"].isna().all() and w["how"].isna().all()
    v = tr["voice"]
    assert (v["start_s"] + info.eps_f <= t).all()
    assert (v["ongoing"] == v["end_s"].isna()).all()
    assert v["word_bearing"].isna().all() and v["first_word_avail_s"].isna().all()
    assert (tr["segments"]["start_s"] + info.eps_f <= t).all()
    assert (tr["hand_frames"]["t_s"] <= t).all() and tr["hand_frames"]["live_state"].isna().all()
    assert (tr["hand_episodes"]["onset_s"] <= t).all()
    assert len(tr["speech_onsets"]) == 0
    p = causal.truncate(tb, 150.0, info=info)["polls"]       # poll open, answer not yet
    assert len(p) == 1 and p["my_answer_text"].isna().all()


def test_frames_rebuilt_from_truncated_words_match(world, tables):
    rng = np.random.default_rng(5)
    for sc in world.classes[:3]:
        tb = tables[sc.class_id]
        info = causal.class_info(tb["classes"], sc.class_id)
        full = causal.build_frames(causal.prep_words(tb["words"]))
        assert np.all(np.diff(full.close) >= 0)                  # closed frames = prefix
        for t in np.concatenate([rng.uniform(0, info.T_end, 25), full.close[::40]]):
            w = causal.prep_words(causal.truncate(tb, t, info=info)["words"])
            tr = causal.build_frames(w, t)
            k = int(np.sum(full.close <= t))
            assert len(tr) == k, t
            assert np.array_equal(tr.i1, full.i1[:k]) and np.array_equal(tr.close, full.close[:k])
            assert list(tr.text) == list(full.text[:k])
            assert tr.tail_start == (full.i1[k - 1] + 1 if k else 0)


def test_text_stage_frames_match_the_causal_frame_rule(world, tables):
    """text.frames (frames.parquet, embedded by oncue.embed) must cut and close frames exactly
    like causal.build_frames, or emb_row alignment and close_s would drift."""
    text = pytest.importorskip("oncue.text")
    if not hasattr(text, "frames"):
        pytest.skip("oncue.text.frames not available")
    for sc in world:
        mine = causal.frames_table(sc.class_id, tables[sc.class_id]["words"])
        theirs = text.frames(tables[sc.class_id]["words"])
        assert len(mine) == len(theirs)
        np.testing.assert_allclose(theirs["close_s"].to_numpy(float), mine["close_s"].to_numpy(float))
        assert list(theirs["text_norm"].astype(str)) == list(mine["text_norm"].astype(str))


def _words(rows):
    return pd.DataFrame([{"w_idx": i, "start_s": a, "end_s": b, "text_norm": f"w{i}",
                          "spk_hash": s, "role": "prof" if s == "A" else "student",
                          "role_avail_s": b + EPS_W if ra is None else ra, "filler": False,
                          "has_qmark": False} for i, (a, b, s, ra) in enumerate(rows)])


def test_frame_close_times_fixture():
    rows = [(0.0, 0.3, "A", None), (0.5, 0.8, "A", None), (1.2, 1.5, "A", None),
            (1.91, 2.2, "A", None),                       # pause 0.41 after word 2: cut
            (2.3, 2.6, "B", 3.4),                         # speaker change; role known late
            (2.7, 3.0, "B", None)]
    rows += [(3.1 + 0.4 * k, 3.4 + 0.4 * k, "B", None) for k in range(14)]   # long run
    w = causal.prep_words(_words(rows))
    fr = causal.build_frames(w)
    # frame 0: words 0-2 (gap 0.4 exactly is not a pause); closed at 1.5 + 0.4 + eps_w
    assert (fr.i0[0], fr.i1[0]) == (0, 2) and np.isclose(fr.close[0], 2.15)
    # frame 1: word 3; speaker change seen when word 4's role is known (3.4)
    assert (fr.i0[1], fr.i1[1]) == (3, 3) and np.isclose(fr.close[1], 3.4)
    # frame 2 starts at 2.3; the first word ending > 7.3 s is cut by length at end + eps_w
    k = 2
    assert fr.start[k] == 2.3 and fr.end[k] - 2.3 > 5 and fr.end[k] - 0.4 - 2.3 <= 5
    assert np.isclose(fr.close[k], fr.end[k] + EPS_W)
    # truncated builds agree on both sides of every close time
    for c in fr.close:
        for t in (c - 1e-9, c):
            tw = _words(rows)
            keep = tw["start_s"] + EPS_W <= t
            tw = tw[keep].copy()
            tw.loc[~(tw["end_s"] + EPS_W <= t), ["end_s", "text_norm"]] = np.nan
            tw["role_known"] = tw["role_avail_s"] <= t
            tr = causal.build_frames(causal.prep_words(tw), t)
            assert len(tr) == int(np.sum(fr.close <= t))


def hand_times(tb: dict, info: causal.ClassInfo, rng, n: int = 10) -> np.ndarray:
    """Times right around raises, live lowerings and glitch ends (mask boundaries)."""
    e = tb["hand_episodes"]
    c = np.concatenate([causal.floats(e["onset_s"]) + d for d in (0.2, 0.5, 0.6, 1.0)]
                       + [causal.floats(e["offset_live_s"]) + d
                          for d in (-0.01, 0.0, 9.99, 10.0)])
    c = c[np.isfinite(c) & (c > 0) & (c < info.T_end)]
    return rng.choice(c, min(n, len(c)), replace=False)


def test_truncation_fast_equals_reference_and_mask_causality(world, tables, kwargs):
    """THE truncation test: fast(full, t) == reference(truncate(full, t), t) for every
    feature; and the causal at-risk rules (live hand state, post_glitch, word-bearing)
    computed from full tables equal the ones rebuilt from the truncated tables."""
    times = sample_times(world, tables, N_RANDOM)
    rng = np.random.default_rng(4)
    n = 0
    for sc in world:
        cid, tb, kw = sc.class_id, tables[sc.class_id], kwargs[sc.class_id]
        info = causal.class_info(tb["classes"], cid)
        ts = np.sort(np.concatenate([times[cid], hand_times(tb, info, rng)]))
        views = [causal.CausalView(cid, t, tb, **kw) for t in ts]
        fast = F.build(cid, ts, tb, **kw)
        ref = causal.reference_frame(views)
        assert_same(fast, ref)
        rules = labels.causal_rules(ts, tb, info)
        for i, v in enumerate(views):
            assert v.causal_rules() == {r: bool(rules[r].iloc[i]) for r in config.EXCL_CAUSAL}, \
                (cid, ts[i])
        n += len(ts)
    assert n >= N_RANDOM


# --------------------------------------------------------------------------- sentinels


def _append(df: pd.DataFrame, rows: list[dict]) -> pd.DataFrame:
    add = pd.DataFrame(rows, columns=df.columns)
    out = pd.concat([df, add], ignore_index=True)
    for c in df.columns:
        if df[c].dtype != out[c].dtype:
            try:
                out[c] = out[c].astype(df[c].dtype)
            except (TypeError, ValueError):
                pass
    return out


def _word(tb, **kw) -> dict:
    base = tb["words"].iloc[0].to_dict()
    base.update({"w_idx": int(tb["words"]["w_idx"].max()) + 1 + kw.pop("k", 0),
                 "has_qmark": False, "conf": 0.9, "filler": False, "how": "forum"})
    base.update(kw)
    base.setdefault("text_masked", base.get("text_norm"))
    base["role_fused"] = base["role"]
    return base


def _role_hash(tb, role):
    w = tb["words"]
    return str(w.loc[w["role"] == role, "spk_hash"].iloc[0])


def sentinel_future_words(tb, t, info):
    ph = _role_hash(tb, "prof")
    rows = [_word(tb, k=0, start_s=t - 0.2, end_s=t - 0.05, text_norm="zqxsentinel",
                  role="prof", spk_hash=ph, role_event_start_s=t - 0.3,
                  role_avail_s=t - 0.05 + EPS_W),
            _word(tb, k=1, start_s=t + 0.1, end_s=t + 0.4, text_norm="zqxsentinel",
                  role="prof", spk_hash=ph, role_event_start_s=t, role_avail_s=t + 0.65)]
    return {**tb, "words": _append(tb["words"], rows)}


def sentinel_prof_voice(tb, t, info):
    row = tb["voice"].iloc[0].to_dict()
    row.update({"spk_hash": _role_hash(tb, "prof"), "role": "prof",
                "start_s": t - info.eps_f / 2, "end_s": t + 5.0, "room_main": True})
    return {**tb, "voice": _append(tb["voice"], [row])}


def sentinel_segment(tb, t, info):
    row = tb["segments"].iloc[-1].to_dict()
    row.update({"seg_idx": 99, "start_s": t + 1.0, "title_masked": "zqx breakout",
                "seg_type": "breakout"})
    return {**tb, "segments": _append(tb["segments"], [row])}


def _window_tokens(tb, t):
    w = tb["words"]
    sel = (w["end_s"] > t - 60) & (w["end_s"] + EPS_W <= t)
    return [x for x in w.loc[sel, "text_norm"].astype(str) if x] or ["quiet"]


def sentinel_brady_frame(tb, t, info):
    """A Brady frame at t + 5 whose text equals the current 60 s window."""
    mh = _role_hash(tb, "me")
    toks = _window_tokens(tb, t)
    rows = [_word(tb, k=i, start_s=t + 5 + 0.3 * i, end_s=t + 5.25 + 0.3 * i, text_norm=x,
                  role="me", spk_hash=mh, role_event_start_s=t + 4.9,
                  role_avail_s=max(t + 5.5 + 0.3 * i, t + 4.9 + info.eps_f))
            for i, x in enumerate(toks)]
    ev = tb["voice"].iloc[0].to_dict()
    ev.update({"spk_hash": mh, "role": "me", "start_s": t + 4.9,
               "end_s": t + 5.5 + 0.3 * len(toks), "room_main": True})
    return {**tb, "words": _append(tb["words"], rows), "voice": _append(tb["voice"], [ev])}


def sentinel_callon_name(tb, t, info):
    """A CALLON phrase and a [NAME] token at about t - 5, inside an already-closed frame (so
    the only change is the deleted tokens themselves). The raw tokens go through the real
    per-word normalisation (`transcribe.norm_word`): "go" and "ahead" keep their per-word
    text, and must vanish from every joined feature text (frames, windows, utterances)."""
    from oncue import transcribe

    w = causal.order_words(tb["words"])
    pw = causal.prep_words(w)
    fr = causal.build_frames(pw)
    plain = np.array([" ".join(x for x in pw.tok[a:b + 1] if x) == fr.text[j]
                      for j, (a, b) in enumerate(zip(fr.i0, fr.i1, strict=True))], bool)
    ok = np.flatnonzero((fr.close <= t) & (fr.i1 > fr.i0) & (fr.end - fr.start <= 4.0)
                        & plain)                   # never split a phrase already there
    if not len(ok):
        return None
    k = ok[np.argmin(np.abs(fr.end[ok] - (t - 5)))]
    a = w.iloc[fr.i0[k]].to_dict()
    keep = {key: a[key] for key in ("spk_hash", "role", "role_event_start_s", "role_avail_s")}
    rows = [dict(_word(tb, k=i), **keep, start_s=a["end_s"], end_s=a["end_s"],
                 text_norm=transcribe.norm_word(raw, raw == "[NAME]"), text_masked=raw)
            for i, raw in enumerate(["go", "ahead", "[NAME]"])]
    assert [r["text_norm"] for r in rows] == ["go", "ahead", ""]
    return {**tb, "words": _append(tb["words"], rows)}


def sentinel_gap_event(tb, t, info):
    """Forum events inside a recording gap have no media time; they must be ignored."""
    chat = tb["chat"].iloc[0].to_dict() if len(tb["chat"]) else {"class_id": info.cid}
    chat.update({"t_s": np.nan, "role": "me", "text_masked": "are we back"})
    ev = tb["voice"].iloc[0].to_dict()
    ev.update({"start_s": np.nan, "end_s": np.nan, "role": "me"})
    return {**tb, "chat": _append(tb["chat"], [chat]), "voice": _append(tb["voice"], [ev])}


def sentinel_future_raise(tb, t, info):
    """A raise starting at t + 1 (hand samples and its episode row) changes no feature."""
    row = tb["hand_episodes"].iloc[0].to_dict()
    row.update({"ep_id": 99, "onset_s": t + 1.0, "onset_retimed_s": t + 1.0,
                "offset_s": t + 6.0, "offset_live_s": t + 6.5, "dur_s": 5.0,
                "ep_type": "bid", "me_hand_eligible": True, "outcome": "called"})
    hf = tb["hand_frames"].copy()
    up = (hf["t_s"] > t + 1.0) & (hf["t_s"] < t + 6.0)
    hf.loc[up, ["raw_state", "live_state"]] = config.STATE_UP
    hf.loc[up, "shape_ok"] = True
    return {**tb, "hand_episodes": _append(tb["hand_episodes"], [row]), "hand_frames": hf}


SENTINELS = {
    "future_words": sentinel_future_words, "prof_voice": sentinel_prof_voice,
    "segment": sentinel_segment, "brady_frame": sentinel_brady_frame,
    "callon_name": sentinel_callon_name, "gap_event": sentinel_gap_event,
    "future_raise": sentinel_future_raise,
}


@pytest.fixture(scope="module")
def sentinel_points(world, tables, kwargs):
    """(class, t, baseline features) for the sentinel tests, computed once."""
    rng = np.random.default_rng(31)
    out = []
    for sc in world.classes[:3]:
        tb = tables[sc.class_id]
        info = causal.class_info(tb["classes"], sc.class_id)
        for t in rng.uniform(200.0, info.T_end - 30.0, 4):
            out.append((sc.class_id, float(t), info,
                        F.build(sc.class_id, [t], tb, **kwargs[sc.class_id])))
    return out


@pytest.mark.parametrize("name", list(SENTINELS))
def test_sentinels_change_no_feature(tables, kwargs, sentinel_points, name):
    for cid, t, info, base in sentinel_points:
        mod = SENTINELS[name](tables[cid], t, info)
        if mod is None:
            continue
        assert_same(F.build(cid, [t], mod, **kwargs[cid]), base)


def test_sentinel_poll_answer_after_t(tables, kwargs, sentinel_points):
    """A Brady poll answer submitted at t + 3 changes no answer-based feature."""
    cols = ["poll_my_ans_words", "poll_res_my"]
    for cid, t, info, base in sentinel_points:
        tb = tables[cid]
        row = tb["polls"].iloc[0].to_dict()
        row.update({"poll_id": "zqx", "open_s": t - 20.0, "close_s": np.nan,
                    "my_answer_text": " ".join(_window_tokens(tb, t)),
                    "my_answer_avail_s": t + 3.0 + info.eps_p})
        mod = {**tb, "polls": _append(tb["polls"], [row])}
        assert_same(F.build(cid, [t], mod, **kwargs[cid])[cols], base[cols])


def test_callon_phrase_completed_after_t_changes_no_feature(world, tables, kwargs):
    """No look-ahead in CALLON deletion: "go" (heard by t) followed by a word that ends
    after t gives the same features at t whether that word is "ahead" (a CALLON phrase) or
    "home"; once "ahead" is available, the phrase is deleted from the joined text."""
    from oncue import text, transcribe

    sc = world.classes[0]
    tb = tables[sc.class_id]
    words = tb["words"]
    st, en = words["start_s"].to_numpy(float), words["end_s"].to_numpy(float)
    order = np.argsort(st)
    gaps = st[order][1:] - en[order][:-1]
    i = int(np.flatnonzero((gaps > 4.0) & (st[order][1:] > 300))[0])
    t = float(en[order][i] + 2.0)
    ph = _role_hash(tb, "prof")

    def world_with(second):
        rows = []
        for k, (a, b, raw) in enumerate([(t - 0.9, t - 0.6, "go"),
                                         (t - 0.3, t + 0.2, second)]):
            rows.append(_word(tb, k=k, start_s=a, end_s=b, text_masked=raw,
                              text_norm=transcribe.norm_word(raw, False), role="prof",
                              spk_hash=ph, role_event_start_s=t - 1.0,
                              role_avail_s=b + EPS_W))
        return {**tb, "words": _append(words, rows)}
    a, b = world_with("ahead"), world_with("home")
    kw = kwargs[sc.class_id]
    assert_same(F.build(sc.class_id, [t], a, **kw), F.build(sc.class_id, [t], b, **kw))
    assert_same(causal.CausalView(sc.class_id, t, a, **kw).features(),
                causal.CausalView(sc.class_id, t, b, **kw).features())
    eps_f = float(tb["classes"]["eps_F_s"].iloc[0])
    wa = text.WindowBuilder(a["words"], a["chat"], eps_f=eps_f).at(t)
    wb = text.WindowBuilder(b["words"], b["chat"], eps_f=eps_f).at(t)
    assert {k: wa[k] for k in ("T_all15", "T_all60", "X_tail")} == \
        {k: wb[k] for k in ("T_all15", "T_all60", "X_tail")}
    later = t + 5.0                                          # both words now available
    fa = causal.CausalView(sc.class_id, later, a, **kw)
    joined = " ".join(fa.frames.text) + " " + " ".join(
        fa.words.tok[i] for i in fa.tail if fa.words.tok[i])
    assert "go ahead" not in causal.join_tokens(joined.split())
    assert_same(F.build(sc.class_id, [later], a, **kw), fa.features())


def test_text_columns_path_never_lets_callon_into_feature_text():
    """`text.text_columns` (per-word text) + the causal frame rule: "go ahead" spoken in one
    frame never reaches the frame text that features embed."""
    from oncue import text

    cols = text.text_columns(["So", "Ada,", "go", "ahead."], roster=["Ada Lovelace"])
    assert cols["text_norm"] == ["so", "", "go", "ahead"]
    w = pd.DataFrame({"w_idx": range(4), "start_s": [0.0, 0.3, 0.6, 0.9],
                      "end_s": [0.2, 0.5, 0.8, 1.1], "text_norm": cols["text_norm"],
                      "spk_hash": "A", "role": "prof", "role_avail_s": np.nan,
                      "filler": False, "has_qmark": False})
    fr = causal.build_frames(causal.prep_words(w))
    assert list(fr.text) == ["so"] and fr.n_words.tolist() == [1]
    assert list(text.frames(w.assign(class_id=1))["text_norm"]) == ["so"]


# --------------------------------------------------------------------------- straddle


def test_straddling_word_text_never_appears(world, tables, kwargs):
    rng = np.random.default_rng(21)
    for sc in world.classes[:3]:
        tb = tables[sc.class_id]
        info = causal.class_info(tb["classes"], sc.class_id)
        for t in rng.uniform(200.0, info.T_end - 30.0, 3):
            w = _word(tb, start_s=t - 0.5, end_s=t + 0.5, text_norm="zqxstraddle",
                      role="prof", spk_hash=_role_hash(tb, "prof"),
                      role_event_start_s=t - 0.6, role_avail_s=t + 0.75)
            mod = {**tb, "words": _append(tb["words"], [w])}
            view = causal.CausalView(sc.class_id, t, mod, **kwargs[sc.class_id])
            assert "zqxstraddle" not in " ".join(view.frames.text)
            assert "zqxstraddle" not in " ".join(view.words.tok)
            assert "zqxstraddle" not in " ".join(view.words.tok[view.tail])
            # the fast path agrees with a reference that cannot see the text
            assert_same(F.build(sc.class_id, [t], mod, **kwargs[sc.class_id]),
                        view.features())


def test_voice_event_spanning_t_exposes_only_elapsed_time(world, tables, kwargs):
    for sc in world.classes[:3]:
        tb = tables[sc.class_id]
        info = causal.class_info(tb["classes"], sc.class_id)
        v = tb["voice"]
        long = v[(v["end_s"] - v["start_s"] > 6) & v["room_main"]].head(2)
        for idx, ev in long.iterrows():
            t = float(ev["start_s"] + info.eps_f + 2.0)
            base = F.build(sc.class_id, [t], tb, **kwargs[sc.class_id])
            for new_end in (t - info.eps_f + 0.01, t + 1.0, t + 120.0):
                mod_v = v.copy()
                mod_v.loc[idx, "end_s"] = new_end
                mod = {**tb, "voice": mod_v}
                got = F.build(sc.class_id, [t], mod, **kwargs[sc.class_id])
                spk = [c for c in got.columns if c.startswith(("spk_", "me_"))]
                assert_same(got[spk], base[spk])


# --------------------------------------------------------------------------- allowlist


def test_feature_columns_are_allowlisted(world, tables, kwargs):
    sc = world.classes[0]
    df = F.build(sc.class_id, [300.0, 900.0], tables[sc.class_id], **kwargs[sc.class_id])
    schemas.validate(df, "features")
    forbidden = re.compile(r"^(y_|next_|excl|label_|fold|in_vault|tile_|coverage|pcw_edited"
                           r"|pcw_status|cal_|eps_|av_|hand_log|participation|analytics)")
    cols = schemas.feature_columns(df)
    assert cols and not [c for c in cols if forbidden.match(c)]
    assert all(config.feature_allowed(c, "F-punct") for c in cols)
    for fs in F.FEATURESETS:
        chosen = F.columns_for(fs, df.columns)
        assert ("has_qmark" in chosen) == (fs == "F-punct"), fs
        assert all(config.feature_allowed(c, fs) for c in chosen)
    assert not config.feature_allowed("has_qmark", "F-all")


def test_features_code_reads_no_future_structure():
    """L10 grep: no actual-length / recording-ended / segment end / transcript events."""
    import inspect

    src = inspect.getsource(F) + inspect.getsource(causal)
    for bad in ("actual-length", "actual_length", "recording-ended", "recording_ended",
                "participation", "hand-raise-key", "\"events\"]"):
        assert bad not in src, bad
    assert not re.search(r"segments\b[^\n]*\[\"end", src)


# --------------------------------------------------------------------------- real data


@pytest.mark.skipif(os.environ.get("ONCUE_CAUSAL_REAL") != "1",
                    reason="set ONCUE_CAUSAL_REAL=1 to run on data/tables")
def test_truncation_on_real_tables():  # pragma: no cover - needs harvested data
    tables = causal.load_tables()
    classes = tables.get("classes")
    assert classes is not None and len(classes), "no classes table"
    dev = classes[~classes["in_vault"].astype(bool)]["class_id"].astype(int).tolist()
    assert len(dev) >= 5
    rng = np.random.default_rng(config.SEED)
    picks = rng.choice(dev, min(len(dev), 10), replace=False)
    kw = {"encoder": F.Memo(F.HashEncoder(48))}
    per = max(50, int(os.environ.get("ONCUE_CAUSAL_N", "500")) // len(picks))
    for cid in picks:
        tb = causal.class_tables(tables, cid)
        info = causal.class_info(tb["classes"], cid)
        ts = rng.uniform(0, info.T_end, per)
        assert_same(F.build(cid, ts, tb, **kw), reference(cid, ts, tb, kw))
