"""Tests for oncue.hands (spec §2, amendment A1) on synthetic frames only.

Covers strip detection, glyph patches and NCC tracking (including a reordered strip and an
off-strip Brady), hand vs poll tick vs lime distractor, the episode state machine (onsets
within 0.5 s at 2 fps, votes, glitches, left-censoring, merges, outcomes), re-timing to
within 0.05 s, the A/V lag, the ffmpeg plumbing (restarts, pts, URL redaction) and the
audit helpers (pixelation, audit/ only, dev classes only).
"""

from __future__ import annotations

import dataclasses
import io
import logging
import math
import shutil
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from oncue import config, hands, schemas, synthetic

C = config
FPS = config.F_SCAN


# --------------------------------------------------------------------------- helpers


def _strips(sc, ts):
    ts = np.asarray(ts, dtype=float)
    return list(zip(ts, synthetic.frames_for(sc, ts, full=False), strict=True))


def _full_reader(sc):
    def reader(times):
        times = [float(t) for t in times]
        return list(zip(times, synthetic.frames_for(sc, times), strict=True))
    return reader


def _geometry(sc) -> hands.StripGeometry:
    times = np.linspace(0, sc.T_end_s, C.PASS1_N_FRAMES + 2)[1:-1]
    return hands.detect_strip(list(synthetic.frames_for(sc, times)))


def _reordered(sc, perm, *, relabel_me: str | None = None):
    """A copy of `sc` whose strip shows the same people in the order `perm` (old indexes)."""
    lay = dict(sc.layout)
    lay["users"] = [lay["users"][j] for j in perm]
    lay["labels"] = [lay["labels"][j] for j in perm]
    lay["me_tile"] = lay["users"].index(sc.meta["me_id"])
    lay["distractor_tile"] = list(perm).index(lay["distractor_tile"])
    if relabel_me is not None:
        lay["labels"][lay["me_tile"]] = relabel_me
    new_of_old = {old: new for new, old in enumerate(perm)}
    oh, tk = sc.others_hands.copy(), sc.ticks.copy()
    oh["tile_idx"] = oh["tile_idx"].map(new_of_old).astype("Int64")
    tk["tile_idx"] = tk["tile_idx"].map(new_of_old).astype("Int64")
    return dataclasses.replace(sc, layout=lay, others_hands=oh, ticks=tk, _cache={})


def _hf(codes, *, dt=0.5, n_others=None, n_tiles=10, shape=None, cid=900101):
    """A minimal hand_frames table from a state string like "DDDDUUUDD" (U/D/N)."""
    states = {"U": C.STATE_UP, "D": C.STATE_DOWN, "N": C.STATE_NA}
    raw = [states[c] for c in codes]
    n = len(raw)
    return pd.DataFrame({
        "class_id": cid, "t_s": np.arange(n) * dt, "n_tiles": n_tiles,
        "tile_found": [r != C.STATE_NA for r in raw], "tile_idx": pd.array([0] * n, "Int64"),
        "ncc": np.float32(0.9), "ncc_gap": np.float32(0.4), "frac_hand_adj": np.float32(0),
        "shape_ok": [r == C.STATE_UP for r in raw] if shape is None else shape,
        "frac_tick": np.float32(0),
        "n_others": np.zeros(n, int) if n_others is None else np.asarray(n_others),
        "border_any": False, "raw_state": raw, "live_state": raw})


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def pass1s(world, truth):
    """Pass 1 on every class of the 30-minute world."""
    return {sc.class_id: hands.pass1(sc.class_id, _full_reader(sc), T_end_s=sc.T_end_s,
                                     voice=truth[sc.class_id]["voice"],
                                     n_present=sc.layout["n_tiles"]) for sc in world}


@pytest.fixture(scope="module")
def template(pass1s):
    """Brady's glyph template from the world's anchors (none of them is small_world's)."""
    eras = {cid: "all" for cid in pass1s} | {-1: "all"}
    tmpl = hands.build_era_templates(pass1s, eras)[-1]
    assert tmpl is not None and tmpl.shape == (C.GLYPH_H, C.GLYPH_W)
    return tmpl


@pytest.fixture(scope="module")
def scanned(small_world, template):
    """small_world class 0 (12 min) scanned at 2 fps, tracked and turned into episodes."""
    sc = small_world.classes[0]
    tt = synthetic.truth_tables(sc)
    geom = _geometry(sc)
    ts = np.arange(0.0, sc.T_end_s, 1.0 / FPS)
    ht = hands.scan(sc.class_id, FPS, _strips(sc, ts), geometry=geom, template=template)
    hf = hands.track(ht, template)
    ep = hands.episodes(hf, words=tt["words"], voice=tt["voice"], segments=tt["segments"])
    return SimpleNamespace(sc=sc, tt=tt, geom=geom, ht=ht, hf=hf, ep=ep)


# --------------------------------------------------------------------------- strip and tiles


def test_detect_strip_matches_layout(world):
    for sc in world:
        lay = sc.layout
        g = _geometry(sc)
        assert g.found and g.n_frames_found == C.PASS1_N_FRAMES
        assert (g.strip_y, g.strip_h) == (lay["strip_y"], lay["strip_h"])
        assert (g.tile_y0, g.tile_h) == (lay["tile_y0"] - lay["strip_y"], lay["tile_h"])
        assert g.n_tiles == lay["n_tiles"]
        tiles = hands.strip_tiles(synthetic.frames_for(sc, [500.0], full=False)[0])
        assert tiles[:, 0].tolist() == lay["x0"] and tiles[:, 1].tolist() == lay["x1"]


def test_detect_strip_checks_presence_height_and_blank_frames(world):
    sc = world.classes[0]
    frames = list(synthetic.frames_for(sc, [400.0, 800.0]))
    assert hands.detect_strip(frames, n_present=sc.layout["n_tiles"] + 1).found
    bad = hands.detect_strip(frames, n_present=sc.layout["n_tiles"] + 5)
    assert not bad.found and bad.reason == "n_present"
    assert hands.detect_strip(frames, expected_h=72).found
    assert hands.detect_strip(frames, expected_h=200).reason in ("height", "no_band")
    blank = [np.full((540, 960, 3), 128, np.uint8)] * 3
    assert not hands.detect_strip(blank).found
    assert not hands.detect_strip([]).found
    with pytest.raises(ValueError):
        hands.detect_strip(frames, n_present=[9, 9, 9])


def test_frames_without_tiles_keep_a_placeholder_and_read_na(world, template):
    sc = world.classes[0]
    geom = _geometry(sc)
    blank = np.zeros((geom.strip_h, 960, 3), np.uint8) + 128
    src = [(0.0, blank), (0.5, blank)] + _strips(sc, [1.0, 1.5])
    ht = hands.scan(sc.class_id, FPS, src, geometry=geom, template=template)
    assert (ht.loc[ht["pts_s"] < 1.0, "tile_idx"] == -1).all()
    hf = hands.track(ht, template)
    assert len(hf) == 4
    assert hf["tile_found"].tolist() == [False, False, True, True]
    assert hf["raw_state"].iloc[0] == C.STATE_NA and hf["n_tiles"].iloc[0] == 0
    only_blank = hands.track(ht[ht["pts_s"] < 1.0], template)
    assert not only_blank["tile_found"].any()
    with pytest.raises(ValueError):
        hands.scan(sc.class_id, FPS, src, geometry=hands.StripGeometry(found=False))


# --------------------------------------------------------------------------- glyphs and NCC


def test_glyph_bits_roundtrip_and_ncc_basics():
    rng = np.random.default_rng(0)
    patch = rng.random((C.GLYPH_H, C.GLYPH_W)) > 0.5
    bits = hands.pack_glyph(patch)
    assert isinstance(bits, bytes) and len(bits) == 72
    back = hands.unpack_glyphs([bits, b"", b"short"])
    assert np.array_equal(back[0], patch) and not back[1:].any()
    assert hands.ncc(patch, patch)[0] == pytest.approx(1.0)
    assert hands.ncc(patch, ~patch)[0] == pytest.approx(-1.0)
    assert hands.ncc(np.zeros_like(patch), patch)[0] == 0.0
    assert hands.pick_tile(np.array([0.9, 0.85]))[0] == -1          # gap too small
    assert hands.pick_tile(np.array([0.7, 0.1]))[0] == -1           # below NCC_min
    assert hands.pick_tile(np.array([0.2, 0.95, 0.5]))[0] == 1
    assert hands.pick_tile(np.array([]))[0] == -1


def test_glyph_patch_ignores_state_and_tile_width(world, template):
    """Brady's patch matches the template whether his tile is bordered, his hand is up or
    his poll tick shows, at every tile width; no other name gets close (§2.2, D3)."""
    widths = set()
    for sc in world:
        geom = _geometry(sc)
        me = sc.me_tile
        bid = sc.episodes[sc.episodes["ep_type"] == C.EP_BID].iloc[0]
        tick = sc.ticks[sc.ticks["user_id"] == sc.meta["me_id"]].iloc[0]
        v = sc.voice[sc.voice["room_main"] & (sc.voice["user_id"] == sc.meta["me_id"])
                     & sc.voice["word_bearing"]].iloc[0]
        t_border = (v.start_s + v.end_s) / 2 - sc.meta["cal_offset_s"] + synthetic.L_UI_S
        times = [bid.onset_s + 1.0, bid.onset_s - 30.0, tick.start_s + 1.0, t_border]
        assert me in synthetic.hand_state_at(sc, t_border)["border"]
        for t, strip in _strips(sc, times):
            st = hands.frame_stats(strip, geom)
            score = hands.ncc(st["glyph"], template)
            assert score[me] >= 0.95, (sc.class_id, t)
            assert np.delete(score, me).max() < C.NCC_MIN
            assert hands.pick_tile(score)[0] == me
        widths.add(sc.layout["tile_w"])
    assert len(widths) >= 2


def test_hand_tick_and_distractor_are_told_apart(world):
    sc = world.classes[0]
    geom = _geometry(sc)
    me, lime = sc.me_tile, sc.layout["distractor_tile"]
    bid = sc.episodes[sc.episodes["ep_type"] == C.EP_BID].iloc[0]
    tick = sc.ticks[sc.ticks["user_id"] == sc.meta["me_id"]].iloc[0]
    (_, up), (_, down), (_, tk) = _strips(sc, [bid.onset_s + 1.0, bid.onset_s - 30.0,
                                               tick.start_s + 1.0])
    s_up, s_dn, s_tk = (hands.frame_stats(x, geom) for x in (up, down, tk))
    for k in range(sc.layout["n_tiles"]):                 # integral-image sums == direct means
        for s, side in (("s1", C.ROI_S1), ("s2", C.ROI_S2)):
            rs, cs = hands._roi_slices(geom.tile_y0, geom.tile_h, sc.layout["x0"][k],
                                       sc.layout["x1"][k], side, hands.UI)
            m = hands.hsv_masks(up)
            assert s_up[f"frac_hand_{s}"][k] == pytest.approx(m["hand"][rs, cs].mean())
            assert s_up[f"frac_tick_{s}"][k] == pytest.approx(m["tick"][rs, cs].mean())
    for s in ("s1", "s2"):
        assert s_up[f"frac_hand_{s}"][me] >= C.THETA_ON and s_up[f"frac_tick_{s}"][me] == 0
        assert hands.shape_ok(s_up[f"shape_{s}"][me], s_up[f"white_{s}"][me])
        assert s_dn[f"frac_hand_{s}"][me] < C.THETA_OFF / 2
        assert s_tk[f"frac_tick_{s}"][me] >= C.THETA_ON and s_tk[f"frac_hand_{s}"][me] == 0
        assert not hands.shape_ok(s_tk[f"shape_{s}"][me], s_tk[f"white_{s}"][me])
    # the lime stripe passes the hue test but fails the shape test
    assert s_dn["frac_hand_s1"][lime] >= C.THETA_ON
    assert not hands.shape_ok(s_dn["shape_s1"][lime], s_dn["white_s1"][lime])
    # border score: only the speaking tile's ring is in the border colour
    v = sc.voice[sc.voice["room_main"] & (sc.voice["role"] == C.ROLE_STUDENT)].iloc[3]
    t = (v.start_s + v.end_s) / 2 - sc.meta["cal_offset_s"]
    k = sc.layout["users"].index(v.user_id)
    st = hands.frame_stats(_strips(sc, [t])[0][1], geom)
    assert st["border"][k] >= 0.9 and np.delete(st["border"], k).max() < 0.05
    tile = _strips(sc, [t])[0][1][geom.tile_y0:geom.tile_y0 + geom.tile_h,
                                  sc.layout["x0"][k]:sc.layout["x1"][k]]
    assert hands.border_score(tile) == pytest.approx(st["border"][k])
    assert hands.shape_score(np.zeros((5, 5), bool), np.zeros((5, 5), bool)) == (0.0, 0.0)


def test_hand_over_a_poll_tick_reads_up(world):
    """When Brady has answered the poll and raises his hand, the hand wins (synthetic D2
    answer) and the sample is UP; a tick alone is NA under the tick rule, DOWN without."""
    sc = world.classes[0]
    geom = _geometry(sc)
    bid = sc.episodes[sc.episodes["ep_type"] == C.EP_BID].iloc[0]
    ticks = sc.ticks.copy()
    mine = ticks["user_id"] == sc.meta["me_id"]
    ticks.loc[mine, "start_s"], ticks.loc[mine, "end_s"] = 0.0, sc.T_end_s
    both = dataclasses.replace(sc, ticks=ticks, _cache={})
    (_, s_both), (_, s_tick) = _strips(both, [bid.onset_s + 1.0, bid.onset_s - 30.0])
    a, b = hands.frame_stats(s_both, geom), hands.frame_stats(s_tick, geom)
    me = sc.me_tile
    assert a["frac_hand_s1"][me] >= C.THETA_ON and a["frac_tick_s1"][me] == 0
    assert b["frac_tick_s1"][me] >= C.THETA_ON
    frac = np.array([a["frac_hand_s1"][me], b["frac_hand_s1"][me]])
    shp = hands.shape_ok([a["shape_s1"][me], b["shape_s1"][me]],
                         [a["white_s1"][me], b["white_s1"][me]])
    tick = np.array([a["frac_tick_s1"][me], b["frac_tick_s1"][me]])
    found = np.array([True, True])
    na_rule = hands.raw_states(frac, shp, found, tick)
    no_rule = hands.raw_states(frac, shp, found, tick, hands.Detector(tick_na=False))
    assert hands._STATE_NAMES[na_rule].tolist() == [C.STATE_UP, C.STATE_NA]
    assert hands._STATE_NAMES[no_rule].tolist() == [C.STATE_UP, C.STATE_DOWN]


# --------------------------------------------------------------------------- pass 1 and templates


def test_pass1_anchors_are_bradys_bordered_tile(world, pass1s):
    for sc in world:
        res = pass1s[sc.class_id]
        assert res.geometry.found
        assert res.n_anchor_events >= 1 and res.anchors
        assert all(a.tile_idx == sc.me_tile for a in res.anchors)
        assert res.anchors[0].patches.shape[1:] == (C.GLYPH_H, C.GLYPH_W)
        back = hands.Pass1.from_json(res.to_json())
        assert back.geometry == res.geometry and back.n_anchor_events == res.n_anchor_events
        assert np.array_equal(back.anchors[0].patches, res.anchors[0].patches)


def test_anchor_times_need_isolated_word_bearing_events():
    voice = pd.DataFrame({
        "role": ["me", "me", "me", "prof", "me"],
        "start_s": [10.0, 30.0, 50.0, 52.5, 70.0], "end_s": [13.0, 31.5, 54.0, 60.0, 74.0],
        "room_main": [True, True, True, True, True],
        "word_bearing": [True, True, True, True, False]})
    times = hands.anchor_times(voice)
    assert times == [[11.0, 11.5, 12.0]]          # short, overlapped and wordless are dropped
    many = pd.DataFrame({"role": "me", "start_s": np.arange(30) * 10.0,
                         "end_s": np.arange(30) * 10.0 + 3, "room_main": True,
                         "word_bearing": True})
    assert len(hands.anchor_times(many)) == C.ANCHOR_MAX_EVENTS


def test_loco_templates_never_use_the_class_itself(world, pass1s):
    eras = {sc.class_id: sc.meta["term"] for sc in world}
    loco = hands.build_era_templates(pass1s, eras)
    full = hands.build_era_templates(pass1s, eras, loco=False)
    for sc in world:
        mates = [c for c in eras if eras[c] == eras[sc.class_id] and c != sc.class_id]
        own = np.stack([a.patch for a in pass1s[sc.class_id].anchors]).astype(np.float32)
        others = np.stack([a.patch for c in mates for a in pass1s[c].anchors])
        assert np.array_equal(loco[sc.class_id], np.median(others.astype(np.float32), 0))
        both = np.concatenate([own, others.astype(np.float32)])
        assert np.array_equal(full[sc.class_id], np.median(both, 0))
        share, n = hands.validate_tracking(pass1s[sc.class_id].anchors, loco[sc.class_id])
        assert share == 1.0 and n == len(pass1s[sc.class_id].anchors)
    alone = hands.build_era_templates(pass1s, {world.classes[0].class_id: "solo"})
    assert alone[world.classes[0].class_id] is None
    assert np.isnan(hands.validate_tracking([], None)[0])


def test_tile_status_rules():
    ok = hands.StripGeometry(found=True, strip_h=72, tile_h=64)
    assert hands.tile_status(ok, 0.9, 5, 0.95) == ("located", True)
    assert hands.tile_status(ok, 0.9, 2, float("nan")) == ("located", False)
    assert hands.tile_status(ok, 0.9, 5, 0.5) == ("tracking_failed", False)
    assert hands.tile_status(ok, 0.2, 5, 0.95)[0] == "unlocated"
    assert hands.tile_status(hands.StripGeometry(found=False), 1.0, 5, 1.0)[0] == "unlocated"


# --------------------------------------------------------------------------- scan, track


def test_scan_writes_schema_valid_hand_tiles(scanned):
    ht, sc = scanned.ht, scanned.sc
    schemas.validate(ht, "hand_tiles")
    n_frames = ht["pts_s"].nunique()
    assert n_frames == int(sc.T_end_s * FPS)
    assert len(ht) == n_frames * sc.layout["n_tiles"]
    assert ht["glyph_bits"].map(len).eq(72).all()
    assert ht["ncc_me"].notna().all()
    me = ht[ht["tile_idx"] == sc.me_tile]
    assert me["ncc_me"].min() >= 0.95


def test_track_follows_brady_and_counts_others(scanned):
    hf, sc = scanned.hf, scanned.sc
    schemas.validate(hf, "hand_frames")
    assert hf["tile_found"].all()
    assert (hf["tile_idx"] == sc.me_tile).all()
    assert hf.attrs["down_baseline"] == 0.0 and not hf.attrs["flagged"]
    assert hands.coverage(hf) == 1.0
    # n_others never counts the lime distractor or Brady, and peaks during the vote
    vote = sc.episodes[sc.episodes["ep_type"] == C.EP_VOTE].iloc[0]
    during = hf[(hf["t_s"] > vote.onset_s + 2) & (hf["t_s"] < vote.offset_s - 1)]
    assert during["n_others"].max() == vote.n_others_up
    quiet = hf[(hf["t_s"] > 20) & (hf["t_s"] < 60)]
    assert (quiet["n_others"] == 0).all()
    assert hf["border_any"].any()


def test_onsets_within_half_a_second_at_2fps(scanned):
    ep, truth = scanned.ep, scanned.sc.episodes
    schemas.validate(ep, "hand_episodes")
    assert len(ep) == len(truth)
    for (_, e), (_, t) in zip(ep.iterrows(), truth.iterrows(), strict=True):
        assert e["ep_type"] == t["ep_type"]
        assert abs(e["onset_s"] - t["onset_s"]) <= 0.5
        assert abs(e["onset_s"] - t["onset_s"]) <= 0.25 + 1e-9      # midpoint rule
        assert abs(e["offset_s"] - t["offset_s"]) <= 0.5
        assert not e["left_censored"]
        # live exit: the 2nd DOWN sample; a blip (glitch) is known to be over at its end
        lag = 0.0 if e["ep_type"] == C.EP_GLITCH else 0.5
        assert e["offset_live_s"] == pytest.approx(e["offset_s"] + lag)
        assert e["seg_type"] == t["seg_type"]


def test_votes_detected_by_count_or_lexicon(scanned):
    ep, hf, tt = scanned.ep, scanned.hf, scanned.tt
    vote = ep[ep["ep_type"] == C.EP_VOTE]
    assert len(vote) == 1
    v = vote.iloc[0]
    assert v["delta_n_others"] >= C.vote_threshold(scanned.sc.layout["n_tiles"])
    assert v["vote_lex"] and not v["contested"] and pd.isna(v["outcome"])
    # Δn alone (no words) still types the vote
    no_words = hands.episodes(hf)
    assert (no_words["ep_type"] == C.EP_VOTE).sum() == 1
    # the lexicon alone types it when nobody else's hand rises
    flat = hf.assign(n_others=0)
    lex = hands.episodes(flat, words=tt["words"])
    assert lex.loc[lex["onset_s"] == v["onset_s"], "ep_type"].item() == C.EP_VOTE
    assert lex.loc[lex["onset_s"] == v["onset_s"], "delta_n_others"].item() == 0
    # with neither, it is a bid
    neither = hands.episodes(flat)
    assert neither.loc[neither["onset_s"] == v["onset_s"], "ep_type"].item() == C.EP_BID


def test_contested_votes_and_threshold_sensitivity():
    n = 80
    others = np.zeros(n, int)
    others[44:60] = 5                                   # 5 others rise with Brady at t = 22 s
    hf = _hf("D" * 44 + "U" * 8 + "D" * 28, n_others=others, n_tiles=11)
    words = pd.DataFrame({"role": "prof", "start_s": [15.0, 15.5], "end_s": [15.4, 15.9],
                          "text_norm": ["anyone", "else"]})
    ep = hands.episodes(hf, words=words)
    assert ep["ep_type"].item() == C.EP_VOTE and ep["contested"].item()
    assert ep["delta_n_others"].item() == 5
    strict = hands.episodes(hf, words=words, vote_frac=0.5)       # max(4, ceil(5)) = 5
    assert strict["ep_type"].item() == C.EP_VOTE
    assert hands.episodes(hf.assign(n_tiles=21), vote_frac=0.5)["ep_type"].item() == C.EP_BID


def test_glitch_and_outcomes(scanned):
    ep, truth = scanned.ep, scanned.sc.episodes
    g = ep[ep["ep_type"] == C.EP_GLITCH].iloc[0]
    assert g["dur_s"] < C.MIN_EPISODE_S and not g["me_hand_eligible"]
    bids = ep[ep["ep_type"] == C.EP_BID].reset_index(drop=True)
    tb = truth[truth["ep_type"] == C.EP_BID].reset_index(drop=True)
    assert bids["outcome"].tolist() == tb["outcome"].tolist()
    called = bids["outcome"] == "called"
    assert np.allclose(bids.loc[called, "call_latency_s"], tb.loc[called, "call_latency_s"],
                       atol=0.5)
    assert bids["me_hand_eligible"].all()
    summary = hands.class_summary(scanned.sc.class_id, geometry=scanned.geom,
                                  hand_frames=scanned.hf, hand_episodes=ep)
    assert summary["n_bids"] == len(tb) and summary["n_votes"] == 1
    assert summary["tile_status"] == "located" and summary["coverage"] == 1.0


def test_tracking_and_vote_diagnostics(scanned):
    ep, hf = scanned.ep, scanned.hf
    assert hands.left_censored_share(ep) == 0.0
    assert np.isnan(hands.left_censored_share(ep[ep["ep_type"] == C.EP_GLITCH]))
    onsets = scanned.tt["speech_onsets"]
    fresh = onsets.loc[onsets["fresh"], "onset_s"]
    assert len(fresh) and hands.na_rate_before(hf, fresh) <= 0.1
    tick = scanned.sc.ticks
    t_tick = tick.loc[tick["user_id"] == scanned.sc.meta["me_id"], "end_s"].iloc[0]
    assert hands.na_rate_before(hf, [t_tick], span_s=20.0) == pytest.approx(1.0, abs=0.03)
    assert np.isnan(hands.na_rate_before(hf, []))
    d = hands.vote_diagnostics(ep, hf)
    assert d["n_votes"] == 1 and d["n_dn_typed"] == 1 and d["n_lex_typed"] == 1
    assert d["share_dn_with_lex"] == 1.0 and d["share_lex_with_dn2"] == 1.0
    assert hands.vote_diagnostics(ep.head(0))["n_votes"] == 0


def test_poll_tick_is_never_read_as_a_hand(scanned):
    sc, hf = scanned.sc, scanned.hf
    tick = sc.ticks[sc.ticks["user_id"] == sc.meta["me_id"]].iloc[0]
    during = hf[(hf["t_s"] >= tick.start_s + 0.5) & (hf["t_s"] < tick.end_s - 0.5)]
    assert len(during) > 10
    assert (during["raw_state"] == C.STATE_NA).all()               # tick rule (PROVISIONAL)
    assert (during["frac_tick"] >= C.THETA_ON).all() and (during["frac_hand_adj"] == 0).all()
    # without the tick rule (D2 finds the hand always wins) the tick reads DOWN, never UP;
    # no template here, so tracking uses the ncc_me stored at scan time
    no_rule = hands.track(scanned.ht, det=hands.Detector(tick_na=False))
    d2 = no_rule[(no_rule["t_s"] >= tick.start_s + 0.5) & (no_rule["t_s"] < tick.end_s - 0.5)]
    assert (d2["raw_state"] == C.STATE_DOWN).all()


def test_live_state_is_causal_and_confirmed_after_k(scanned):
    hf = scanned.hf
    raw = pd.Series(hf["raw_state"].astype(str)).map(
        {C.STATE_DOWN: 0, C.STATE_UP: 1, C.STATE_NA: 2}).to_numpy()
    full = hands.live_states(raw)
    for cut in (100, 421, 850, 939):
        assert np.array_equal(hands.live_states(raw[:cut]), full[:cut])
    first_up = int(np.flatnonzero(raw == 1)[0])
    live = hf["live_state"].astype(str).to_numpy()
    assert live[first_up] != C.STATE_UP and live[first_up + 1] == C.STATE_UP
    assert hands._STATE_NAMES[hands.live_states(np.array([1, 0, 1, 1, 0, 0]))].tolist() == [
        C.STATE_NA, C.STATE_NA, C.STATE_NA, C.STATE_UP, C.STATE_UP, C.STATE_DOWN]
    tr = hands.live_transitions(hf)
    lowered = tr[(tr["from_state"] == C.STATE_UP) & (tr["to_state"] == C.STATE_DOWN)]
    assert np.allclose(lowered["t_s"].to_numpy(),
                       scanned.ep.loc[scanned.ep["ep_type"] != C.EP_GLITCH,
                                      "offset_live_s"].to_numpy())


def test_tracking_survives_strip_reorder_and_off_strip(small_world, template):
    sc = small_world.classes[0]
    geom = _geometry(sc)
    n = sc.layout["n_tiles"]
    moved = _reordered(sc, list(range(n))[::-1])
    gone = _reordered(sc, list(range(n)), relabel_me="Ada L.")     # Brady not on the strip
    assert moved.me_tile != sc.me_tile
    src = (_strips(sc, np.arange(300, 320, 0.5)) + _strips(moved, np.arange(320, 340, 0.5))
           + _strips(gone, np.arange(340, 350, 0.5)))
    hf = hands.track(hands.scan(sc.class_id, FPS, src, geometry=geom), template)
    first, second, third = hf[hf.t_s < 320], hf[(hf.t_s >= 320) & (hf.t_s < 340)], hf[hf.t_s >= 340]
    assert first["tile_found"].all() and (first["tile_idx"] == sc.me_tile).all()
    assert second["tile_found"].all() and (second["tile_idx"] == moved.me_tile).all()
    assert not third["tile_found"].any() and (third["raw_state"] == C.STATE_NA).all()


def test_baseline_is_subtracted_and_flagged(scanned, template):
    ht = scanned.ht.copy()
    me = ht["tile_idx"] == scanned.sc.me_tile
    ht.loc[me, "frac_hand_s1"] = ht.loc[me, "frac_hand_s1"] + np.float32(0.06)
    hf = hands.track(ht, template)
    assert hf.attrs["down_baseline"] == pytest.approx(0.06, abs=1e-6)
    assert hf.attrs["flagged"]
    assert hf["raw_state"].tolist() == scanned.hf["raw_state"].tolist()
    b, flag = hands.baseline(np.array([0.01, 0.02, 0.5, np.nan]), np.array([0, 0, 1, 0], bool))
    assert b == pytest.approx(0.015) and not flag


# --------------------------------------------------------------------------- state machine units


def test_hysteresis_keeps_the_previous_state():
    det = hands.Detector()
    frac = np.array([0.0, 0.25, 0.15, 0.1, 0.05, 0.15, 0.3, 0.3])
    shp = np.array([0, 1, 1, 1, 0, 1, 0, 1], bool)
    found = np.array([1, 1, 1, 1, 1, 1, 1, 0], bool)
    got = hands._STATE_NAMES[hands.raw_states(frac, shp, found, det=det)].tolist()
    U, D, N = C.STATE_UP, C.STATE_DOWN, C.STATE_NA
    assert got == [D, U, U, U, D, D, D, N]
    assert hands._STATE_NAMES[hands.raw_states([0.15], [True], [True])].tolist() == [N]


def test_d_pre_and_left_censoring():
    ok = hands.episodes(_hf("DDDD" + "UUUU" + "DDDD"))
    assert not ok["left_censored"].item() and ok["onset_s"].item() == pytest.approx(1.75)
    short = hands.episodes(_hf("DDD" + "UUUU" + "DDDD"))          # 1.5 s from recording start
    assert short["left_censored"].item()
    after_na = hands.episodes(_hf("DDDDDD" + "NNN" + "DD" + "UUUU" + "DDDD"))
    assert after_na["left_censored"].item()
    at_start = hands.episodes(_hf("UUUU" + "DDDD"))
    assert at_start["left_censored"].item() and at_start["onset_s"].item() == 0.0


def test_na_bridging_merges_and_toggles():
    bridged = hands.episodes(_hf("DDDDD" + "UUU" + "NN" + "UUU" + "DDDDD"))
    assert len(bridged) == 1 and bridged["n_toggles"].item() == 0
    broken = hands.episodes(_hf("DDDDD" + "UUU" + "NNN" + "UUU" + "DDDDD"))
    assert len(broken) == 2 and broken["left_censored"].tolist() == [False, True]
    assert broken["outcome"].iloc[0] == "open"                    # tile lost while up
    merged = hands.episodes(_hf("DDDDD" + "UUUU" + "DDDDD" + "UUUU" + "DDDDDDD"))
    assert len(merged) == 1 and merged["n_toggles"].item() == 1
    assert merged["dur_s"].item() == pytest.approx(9.0 - 2.25)
    split = hands.episodes(_hf("DDDDD" + "UUUU" + "DDDDDD" + "UUUU" + "DDDDD"))
    assert len(split) == 2 and split["n_toggles"].tolist() == [0, 0]
    codes = np.array([0, 2, 0, 1, 2, 2, 2, 1])
    assert hands.bridge_na(codes, 0.5).tolist() == [0, 0, 0, 1, 2, 2, 2, 1]


def test_glitches_open_episodes_and_eligibility():
    blip = hands.episodes(_hf("DDDDDDDD" + "U" + "DDDDDDDD"))      # isolated: a glitch
    assert blip["ep_type"].item() == C.EP_GLITCH and blip["dur_s"].item() == pytest.approx(0.75)
    assert blip["offset_live_s"].item() == pytest.approx(blip["offset_s"].item())
    to_end = hands.episodes(_hf("DDDDDD" + "UUUUUU"))
    assert to_end["outcome"].item() == "open" and np.isnan(to_end["offset_live_s"].item())
    short = hands.episodes(_hf("DDDDDD" + "UUU" + "DDDDDD"))       # 1.75 s < 2 s
    assert short["ep_type"].item() == C.EP_BID and not short["me_hand_eligible"].item()
    no_shape = hands.episodes(_hf("DDDDDD" + "UUUUUU" + "DDDD", shape=[False] * 16))
    assert not no_shape["me_hand_eligible"].item()
    assert len(hands.episodes(_hf("DDDDDDDD"))) == 0
    schemas.validate(hands.episodes(_hf("DDDD")), "hand_episodes")


def test_outcomes_from_bradys_turns():
    hf = _hf("D" * 10 + "U" * 10 + "D" * 30)                       # onset 4.75, offset 10
    voice = pd.DataFrame({"role": ["me", "me", "me"], "start_s": [2.0, 13.0, 14.0],
                          "end_s": [3.0, 13.5, 16.0], "room_main": True,
                          "word_bearing": [True, True, True]})
    called = hands.episodes(hf, voice=voice)
    assert called["outcome"].item() == "called"
    assert called["call_latency_s"].item() == pytest.approx(13.0 - 4.75)
    late = voice.assign(start_s=[2.0, 15.5, 16.0], end_s=[3.0, 15.8, 17.0])
    assert hands.episodes(hf, voice=late)["outcome"].item() == "withdrawn"
    blips = voice.assign(word_bearing=False)
    assert hands.episodes(hf, voice=blips)["outcome"].item() == "withdrawn"
    assert hands.me_turn_starts(voice).tolist() == [2.0, 13.0]


# --------------------------------------------------------------------------- re-timing


def test_retime_to_within_50_ms(scanned, template):
    sc, geom = scanned.sc, scanned.geom

    def reader(t0, dur):
        ts = np.arange(math.ceil(t0 * 30 - 1e-6) / 30, t0 + dur, 1 / 30)
        return _strips(sc, ts)

    out = hands.retime(scanned.ep, reader, geometry=geom, template=template)
    schemas.validate(out, "hand_episodes")
    assert not out["retime_failed"].any()
    err = np.abs(out["onset_retimed_s"].to_numpy() - sc.episodes["onset_s"].to_numpy())
    assert err.max() <= 0.05
    # without a template, the tracked x-span is used
    spans = {int(e): (sc.layout["x0"][sc.me_tile], sc.layout["x1"][sc.me_tile])
             for e in out["ep_id"]}
    by_span = hands.retime(scanned.ep, reader, geometry=geom, spans=spans)
    assert np.allclose(by_span["onset_retimed_s"], out["onset_retimed_s"])


def test_retime_failures_fall_back_to_the_midpoint(scanned, template):
    def broken(t0, dur):
        raise RuntimeError("403 Forbidden")

    def late_seek(t0, dur):
        return _strips(scanned.sc, np.arange(t0 + 0.3, t0 + dur, 1 / 30))

    for reader in (broken, late_seek, lambda t0, dur: []):
        out = hands.retime(scanned.ep, reader, geometry=scanned.geom, template=template)
        assert out["retime_failed"].all()
        assert np.allclose(out["onset_retimed_s"], out["onset_s"])
    censored = scanned.ep.assign(left_censored=True)
    skipped = hands.retime(censored, broken, geometry=scanned.geom, template=template)
    assert skipped["retime_failed"].isna().all()


# --------------------------------------------------------------------------- A/V lag


def test_av_lag_recovers_the_planted_lag(scanned):
    truth = synthetic.L_UI_S - scanned.sc.meta["cal_offset_s"]
    lag = hands.av_lag(scanned.hf, scanned.tt["words"])
    assert abs(lag - truth) <= 0.2
    assert C.av_lag_ok(lag, truth)
    hf = scanned.hf.set_index(scanned.hf.index + 1000)              # any index is kept
    mb = hands.me_border(hf, scanned.ht)
    assert mb.index.equals(hf.index) and mb.dtype == bool
    assert mb.any() and not (mb & ~hf["border_any"]).any()
    me = hands.av_lag_brady(scanned.hf, scanned.ht, scanned.tt["words"])
    words = scanned.tt["words"]
    me_speech = (words.loc[words.role == C.ROLE_ME, "end_s"]
                 - words.loc[words.role == C.ROLE_ME, "start_s"]).sum()
    if me_speech >= C.AV_BRADY_MIN_SPEECH_S:
        assert abs(me - truth) <= 0.3
    else:
        assert np.isnan(me)
    assert np.isnan(hands.av_lag(scanned.hf, words.head(0)))
    assert np.isnan(hands.av_lag(scanned.hf.assign(border_any=False), words))


# --------------------------------------------------------------------------- ffmpeg plumbing

SIGNED = "https://cdn.example.invalid/rec/lowres.mp4?Expires=1&Signature=SECRET{n}&Key-Pair-Id=K"


class _Proc:
    def __init__(self, stdout: bytes, stderr: bytes, rc: int):
        self.stdout, self.stderr, self.returncode = io.BytesIO(stdout), io.BytesIO(stderr), rc

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        pass


class FakeFfmpeg:
    """Stands in for subprocess.Popen: serves synthetic strips with showinfo pts lines.
    `fail_at` (media seconds) makes the first `n_fail` runs die there with a 403."""

    def __init__(self, sc, *, fail_at=None, n_fail=1, a_start=0.0, end_s=60.0):
        self.sc, self.fail_at, self.n_fail = sc, fail_at, n_fail
        self.a_start, self.end_s, self.calls = a_start, end_s, []

    def __call__(self, cmd, stdout=None, stderr=None):
        self.calls.append(list(cmd))
        url = cmd[cmd.index("-i") + 1]
        seek = float(cmd[cmd.index("-ss") + 1]) if "-ss" in cmd else 0.0
        end = seek + float(cmd[cmd.index("-t") + 1]) if "-t" in cmd else self.end_s
        vf = cmd[cmd.index("-vf") + 1]
        fps = float(vf.split("fps=")[1].split(",")[0].split(":")[0]) if "fps=" in vf else 25.0
        ts = np.arange(math.ceil(seek * fps - 1e-9) / fps, end - 1e-9, 1 / fps)
        rc = 0
        if self.fail_at is not None and len(self.calls) <= self.n_fail:
            ts, rc = ts[ts < self.fail_at], 1
        frames = synthetic.frames_for(self.sc, ts, full=False) if len(ts) else []
        err = [f"Input #0, mov,mp4,m4a,3gp,3g2,mj2, from '{url}':"]
        err += [f"[Parsed_showinfo_2 @ 0x600001] n:{i:4d} pts:{int(t * 90000)} "
                f"pts_time:{t + self.a_start:.6g} duration:45000 fmt:bgr24"
                for i, t in enumerate(ts)]
        if rc:
            err.append(f"[https @ 0x6000] HTTP error 403 Forbidden ({url})")
        return _Proc(b"".join(f.tobytes() for f in frames), "\n".join(err).encode() + b"\n", rc)


def _url_fn():
    n = {"i": 0}

    def fresh():
        n["i"] += 1
        return SIGNED.format(n=n["i"])
    fresh.count = n
    return fresh


def test_ffmpeg_cmd_shape():
    cmd = hands.ffmpeg_cmd(SIGNED.format(n=1), crop=(28, 72), fps=2, seek=12.5,
                           hwaccel="videotoolbox")
    assert cmd.index("-copyts") < cmd.index("-ss") < cmd.index("-i")
    # round=up: a sample labelled t never shows content from after t (see the real-file test)
    assert cmd[cmd.index("-vf") + 1] == "crop=iw:72:0:28,fps=2:round=up,showinfo"
    assert cmd[cmd.index("-reconnect") + 1] == "1" and "-reconnect_delay_max" in cmd
    assert cmd[cmd.index("-hwaccel") + 1] == "videotoolbox"
    assert cmd[-5:] == ["-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    local = hands.ffmpeg_cmd("/tmp/x.mp4", frames=1)
    assert "-reconnect" not in local and local[local.index("-frames:v") + 1] == "1"
    assert hands.redact(f"failed for {SIGNED.format(n=3)}") == \
        "failed for https://cdn.example.invalid/rec/lowres.mp4?<redacted>"


def test_strip_reader_restarts_with_fresh_url_and_dedups(small_world, caplog):
    sc = small_world.classes[0]
    lay = sc.layout
    fake = FakeFfmpeg(sc, fail_at=20.0, a_start=1.5, end_s=60.0)
    url_fn = _url_fn()
    reader = hands.FfmpegStripReader(url_fn, width=960, strip_y=0, strip_h=lay["strip_h"],
                                     fps=2, a_start=1.5, end_s=60.0, popen=fake, hwaccel=None)
    with caplog.at_level(logging.DEBUG, logger="oncue.hands"):
        got = list(reader)
    times = np.array([t for t, _ in got])
    # media time = pts_time - a_start; every 2 fps sample exactly once, none duplicated
    assert times.tolist() == pytest.approx(np.arange(0.0, 60.0, 0.5).tolist())
    assert reader.restarts == 1 and url_fn.count["i"] == 2
    second = fake.calls[1]
    assert second[second.index("-i") + 1] == SIGNED.format(n=2)     # fresh URL on restart
    resume = float(second[second.index("-ss") + 1])
    assert resume == pytest.approx(19.5 - 2.0)                      # last pts - 2 s
    assert got[0][1].shape == (lay["strip_h"], 960, 3)
    assert np.array_equal(got[45][1], synthetic.frames_for(sc, [22.5], full=False)[0])
    assert "Signature" not in caplog.text and "SECRET" not in caplog.text


def test_strip_reader_times_and_failure_message(small_world, caplog):
    sc = small_world.classes[0]
    fake = FakeFfmpeg(sc, a_start=0.0, end_s=10.0)
    reader = hands.FfmpegStripReader(_url_fn(), width=960, strip_y=0,
                                     strip_h=sc.layout["strip_h"], fps=2, end_s=10.0,
                                     popen=fake, hwaccel=None)
    got = list(reader)
    assert [t for t, _ in got] == pytest.approx(np.arange(0.0, 10.0, 0.5).tolist())
    assert np.array_equal(got[3][1], synthetic.frames_for(sc, [1.5], full=False)[0])
    dead = FakeFfmpeg(sc, fail_at=0.0, n_fail=99)
    bad = hands.FfmpegStripReader(_url_fn(), width=960, strip_y=0, strip_h=72, fps=2,
                                  end_s=10.0, popen=dead, hwaccel=None, max_restarts=2)
    with caplog.at_level(logging.DEBUG, logger="oncue.hands"), \
            pytest.raises(hands.ScanError) as exc:
        list(bad)
    msg = str(exc.value)
    assert "403" in msg and "Signature" not in msg and "SECRET" not in msg
    assert "SECRET" not in caplog.text and len(dead.calls) == 3


def test_scan_builds_the_ffmpeg_reader(small_world, template):
    sc = small_world.classes[0]
    geom = _geometry(sc)
    fake = FakeFfmpeg(sc, end_s=5.0)
    ht = hands.scan(sc.class_id, FPS, None, geometry=geom, template=template,
                    url_fn=_url_fn(), media={"width": 960, "a_start": 0.0}, popen=fake)
    assert ht["pts_s"].nunique() == 10 and len(fake.calls) == 1
    cmd = fake.calls[0]
    assert cmd[cmd.index("-hwaccel") + 1] == "videotoolbox"
    assert cmd[cmd.index("-vf") + 1] == f"crop=iw:{geom.strip_h}:0:{geom.strip_y},fps=2:round=up,showinfo"
    with pytest.raises(ValueError):
        hands.scan(sc.class_id, FPS, None, geometry=geom)


def test_clip_reader_and_probe_with_fakes(small_world):
    sc = small_world.classes[0]
    fake = FakeFfmpeg(sc)
    clip = hands.FfmpegClipReader(_url_fn(), width=960, height=540, crop=(0, 72), fps=30,
                                  a_start=0.0, popen=fake)
    got = clip(100.0, 1.25)
    assert got[0][0] == pytest.approx(100.0) and len(got) == 38
    assert float(fake.calls[0][fake.calls[0].index("-ss") + 1]) == pytest.approx(100.0)
    failing = hands.FfmpegClipReader(_url_fn(), width=960, height=540, crop=(0, 72), fps=2,
                                     popen=FakeFfmpeg(sc, fail_at=0.0, n_fail=9))
    with pytest.raises(hands.ScanError) as exc:
        failing(5.0, 1.0)
    assert "SECRET" not in str(exc.value)

    def runner(cmd, **kw):
        assert cmd[-1] == SIGNED.format(n=1)
        out = ('{"streams": [{"codec_type": "video", "start_time": "0.066", "width": 960,'
               ' "height": 540}, {"codec_type": "audio", "start_time": "0.021"}],'
               ' "format": {"start_time": "0.021", "duration": "5400.5"}}')
        return SimpleNamespace(returncode=0, stdout=out, stderr="")

    info = hands.probe(SIGNED.format(n=1), runner=runner)
    assert info == {"a_start": 0.021, "v_start": 0.066, "fmt_start": 0.021, "width": 960,
                    "height": 540, "duration": 5400.5}

    def refused(cmd, **kw):
        return SimpleNamespace(returncode=1, stdout="", stderr=f"403 Forbidden {cmd[-1]}")

    with pytest.raises(hands.ScanError) as exc:
        hands.probe(SIGNED.format(n=1), runner=refused)
    assert "SECRET" not in str(exc.value) and "Signature" not in str(exc.value)


def test_sources_use_the_asr_rendition_only(monkeypatch):
    """Real-source factory with a fake Forum client: the lowres URL only (no silent highres
    fallback, §2.1.1), one Forum call for probing, a fresh call per url_fn()."""
    calls = []

    class Client:
        def recording_urls(self, cid):
            calls.append(cid)
            return {"lowres": SIGNED.format(n=len(calls)),
                    "highres": "https://cdn.example.invalid/hi.mp4?Signature=HI"}

    class HighOnly:
        def recording_urls(self, cid):
            return {"highres": "https://cdn.example.invalid/hi.mp4?Signature=HI"}

    probed = []
    monkeypatch.setattr(hands, "probe", lambda url, **kw: probed.append(url) or {
        "width": 960, "height": 540, "a_start": 0.0, "fmt_start": 0.0})
    src = hands._sources(900101, Client())
    assert probed == [SIGNED.format(n=1)] and calls == [900101]
    assert src["url_fn"]() == SIGNED.format(n=2) and len(calls) == 2
    assert src["media"]["width"] == 960 and callable(src["frame_reader"])
    with pytest.raises(hands.ScanError) as exc:
        hands._sources(900101, HighOnly())
    assert "Signature" not in str(exc.value) and "lowres" in str(exc.value)


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                    reason="ffmpeg not installed")
def test_real_ffmpeg_on_a_local_file(small_world, tmp_path, template):
    """End to end through the real ffmpeg binary on a local synthetic mp4 (no network):
    pts-based times under -copyts, strip crop, and the detector after compression."""
    import cv2

    sc = small_world.classes[0]
    bid = sc.episodes[sc.episodes["ep_type"] == C.EP_BID].iloc[0]
    ts = np.arange(0.0, 6.0, 1 / 25)
    path = tmp_path / "clip.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (960, 540))
    for f in synthetic.frames_for(sc, ts + bid.onset_s - 3.0):
        w.write(f)
    w.release()
    info = hands.probe(str(path))
    assert info["width"] == 960 and info["height"] == 540
    lay = sc.layout
    reader = hands.FfmpegStripReader(lambda: str(path), width=960, strip_y=lay["strip_y"],
                                     strip_h=lay["strip_h"], fps=2, a_start=info["a_start"],
                                     fmt_start=info["fmt_start"], hwaccel=None, timeout_s=30)
    got = list(reader)
    assert [t for t, _ in got] == pytest.approx(np.arange(0.0, 6.0, 0.5).tolist(), abs=0.02)
    geom = _geometry(sc)
    up = hands.frame_stats(got[-2][1], geom, template=template)
    assert up["ncc_me"][sc.me_tile] >= C.NCC_MIN
    assert up["frac_hand_s1"][sc.me_tile] >= C.THETA_ON
    clip = hands.FfmpegClipReader(lambda: str(path), width=960, height=540,
                                  crop=(lay["strip_y"], lay["strip_h"]),
                                  a_start=info["a_start"], fmt_start=info["fmt_start"])
    frames = clip(2.0, 1.25)
    assert abs(frames[0][0] - 2.0) <= 0.05 and len(frames) >= 30


# --------------------------------------------------------------------------- audit helpers


def test_render_strip_pixelates_everyone_but_brady(scanned):
    sc, geom = scanned.sc, scanned.geom
    t = 300.0
    strip = _strips(sc, [t])[0][1]
    tiles = hands.strip_tiles(strip)
    me = sc.me_tile
    img = hands.render_strip(strip, tiles, geometry=geom, me_tile=me, state=C.STATE_DOWN,
                             t_s=t, scale=1)
    body = img[hands.AUDIT_HEADER_PX:]
    ty0, th = geom.tile_y0, geom.tile_h
    corner = C.ROI_INSET_PX + round(C.ROI_S2 * th)
    label = round(hands.UI.label_h_frac * th)
    x0, x1 = tiles[me]
    assert np.array_equal(body[ty0:ty0 + th, x0:x1], strip[ty0:ty0 + th, x0:x1])
    for k, (a, b) in enumerate(tiles):
        if k == me:
            continue
        assert np.array_equal(body[ty0:ty0 + corner, a:a + corner],
                              strip[ty0:ty0 + corner, a:a + corner])      # corner sharp
        assert np.array_equal(body[ty0 + th - label:ty0 + th, a:b],
                              strip[ty0 + th - label:ty0 + th, a:b])      # name label sharp
        face = body[ty0 + corner:ty0 + th - label, a:b].reshape(-1, 3)
        assert len(np.unique(face, axis=0)) <= C.G2_SAMPLING["pixelate_blocks"] ** 2
        assert not np.array_equal(face, strip[ty0 + corner:ty0 + th - label, a:b]
                                  .reshape(-1, 3))
    na = hands.render_strip(strip, tiles, geometry=geom, me_tile=None, t_s=t, scale=2)
    assert na.shape == (2 * (geom.strip_h + hands.AUDIT_HEADER_PX), 2 * strip.shape[1], 3)
    face = na[2 * hands.AUDIT_HEADER_PX:][2 * (ty0 + corner):2 * (ty0 + th - label),
                                          2 * x0:2 * x1].reshape(-1, 3)
    assert len(np.unique(face, axis=0)) <= C.G2_SAMPLING["pixelate_blocks"] ** 2


def test_audit_outputs_stay_under_audit_and_dev_only(scanned, template, tmp_path, world):
    sc, geom = scanned.sc, scanned.geom
    out = tmp_path / "audit"

    def source(t0, t1, fps):
        return _strips(sc, np.arange(t0, t1, 1 / fps))

    clip = hands.audit_clip(sc.class_id, source, scanned.hf, geometry=geom, t0=460.0,
                            t1=480.0, out_dir=out)
    assert clip.parent == out.resolve() and clip.suffix == ".mp4" and clip.stat().st_size > 0
    sample = scanned.ep[scanned.ep["ep_type"] != C.EP_GLITCH].assign(stratum="per_class")
    sheet = hands.contact_sheet(sample, lambda cid, t: _strips(sc, [t])[0][1],
                                templates={sc.class_id: template},
                                geometries={sc.class_id: geom}, out_dir=out)
    pngs = [p for p in sheet if p.suffix == ".png"]
    assert pngs and all(p.parent == out.resolve() and p.stat().st_size > 0 for p in pngs)
    manifest = pd.read_csv(sheet[-1])
    assert len(manifest) == len(sample) and manifest["verdict"].isna().all()
    onsets = pd.DataFrame({"class_id": [sc.class_id], "onset_s": [480.0]})
    paths = hands.onset_clips(onsets, {sc.class_id: source}, {sc.class_id: scanned.hf},
                              {sc.class_id: geom}, span_s=10.0, out_dir=out)
    assert len(paths) == 1 and paths[0].exists()
    with pytest.raises(ValueError):
        hands.audit_clip(sc.class_id, source, scanned.hf, geometry=geom, t0=0, t1=2,
                         out_dir=tmp_path / "elsewhere")
    vault = next(c.class_id for c in world if C.in_vault(c.class_id))
    n_files = len(list(out.iterdir()))
    with pytest.raises(AssertionError, match="vault"):             # select.VaultError
        hands.audit_clip(vault, source, scanned.hf, geometry=geom, t0=0, t1=2, out_dir=out)
    with pytest.raises(AssertionError, match="vault"):
        hands.contact_sheet(sample.assign(class_id=vault), lambda cid, t: None,
                            templates={}, geometries={vault: geom}, out_dir=out)
    assert len(list(out.iterdir())) == n_files                     # nothing was written
    assert str(vault) not in " ".join(p.name for p in out.iterdir())


def test_precision_and_onset_samples_are_stratified_and_dev_only():
    dev = [c for c in range(900_000, 900_200) if not C.in_vault(c)][:12]
    rng = np.random.default_rng(1)
    rows = []
    for cid in dev:
        for k in range(6):
            rows.append({"class_id": cid, "ep_id": k, "onset_s": 100.0 * k,
                         "ep_type": C.EP_VOTE if k == 5 else C.EP_BID,
                         "seg_type": "poll" if k == 4 else "debrief",
                         "left_censored": bool(k == 0 and cid == dev[0])})
    eps = pd.DataFrame(rows)
    s = hands.precision_sample(eps, flagged=[dev[1]], seed=3)
    per = s[s["stratum"] == "per_class"].groupby("class_id").size()
    assert per.max() <= C.G2_SAMPLING["precision_per_class"]
    assert set(s.loc[s["stratum"] == "vote", "class_id"]) <= set(dev)
    assert not s.duplicated(["class_id", "ep_id"]).any()
    assert not ((s["class_id"] == dev[0]) & (s["ep_id"] == 0)).any()   # left-censored out
    flagged = eps[(eps["class_id"] == dev[1]) & (eps["ep_type"] == C.EP_BID)]
    got = s[s["class_id"] == dev[1]]
    assert set(flagged["ep_id"]) <= set(got["ep_id"])
    vault = next(c for c in range(900_000, 900_200) if C.in_vault(c))
    with pytest.raises(AssertionError, match="vault"):
        hands.precision_sample(eps.assign(class_id=vault))
    onsets = pd.DataFrame({"class_id": dev * 10 + [vault] * 10,
                           "onset_s": rng.uniform(0, 5000, 130), "fresh": True})
    pick = hands.onset_sample(onsets, n=60, seed=2)
    assert len(pick) == 60 and not pick["class_id"].eq(vault).any()


# --------------------------------------------------------------------------- CLI


def test_cli_scan_track_episodes_offline(small_world, tmp_path, monkeypatch, caplog):
    """The CLI stages on the synthetic world, with the network-facing source factory and
    Forum client replaced (nothing here touches the network or real data)."""
    from oncue import __main__ as cli

    monkeypatch.setattr(config, "TABLES_DIR", tmp_path / "tables")
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    config.TABLES_DIR.mkdir(parents=True)
    tts = {sc.class_id: synthetic.truth_tables(sc) for sc in small_world}
    for name in ("classes", "voice", "presence", "words", "segments"):
        pd.concat([t[name] for t in tts.values()], ignore_index=True).to_parquet(
            config.table_path(name), index=False)

    def fake_sources(cid, client):
        sc = small_world.by_id(cid)

        def clip_reader(t0, dur):
            return _strips(sc, np.arange(math.ceil(t0 * 30 - 1e-6) / 30, t0 + dur, 1 / 30))
        return {"frame_reader": _full_reader(sc),
                "strip_source": iter(_strips(sc, np.arange(380.0, 500.0, 0.5))),
                "clip_reader": clip_reader, "url_fn": None, "media": None}

    monkeypatch.setattr(hands, "_forum_client", lambda: None)
    monkeypatch.setattr(hands, "_sources", fake_sources)
    monkeypatch.setattr(hands, "SCANNER", "glyph")       # synthetic frames: the glyph-era UI
    assert cli.main(["scan", "--workers", "2"]) == 0
    for sc in small_world:
        assert hands.hand_tiles_path(sc.class_id).exists()
        assert hands.load_pass1(sc.class_id).geometry.found
    assert cli.main(["track"]) == 0
    assert cli.main(["episodes"]) == 0
    hf = pd.read_parquet(config.table_path("hand_frames"))
    schemas.validate(hf, "hand_frames")
    qc = pd.read_parquet(config.table_path("hand_qc")).set_index("class_id")
    ep = pd.read_parquet(config.table_path("hand_episodes"))
    schemas.validate(ep, "hand_episodes")
    # LOCO: each class is tracked only with the *other* class's anchors. Only class 0 has
    # anchor events, so class 1 is tracked (with class 0's template) and class 0 is not.
    a, b = (sc.class_id for sc in small_world)
    assert hands.load_pass1(a).n_anchor_events > 0 and hands.load_pass1(b).n_anchor_events == 0
    assert qc.loc[b, "coverage"] == 1.0 and qc.loc[b, "tile_status"] == "located"
    assert not qc.loc[b, "template_validated"]                     # < 3 own anchors
    assert qc.loc[a, "coverage"] == 0.0 and qc.loc[a, "tile_status"] == "unlocated"
    truth = small_world.by_id(b).episodes
    inside = truth[(truth.onset_s > 383) & (truth.offset_s < 499)]
    got = ep[ep["class_id"] == b]
    assert len(inside) == 1 and len(got) == 1
    assert abs(got["onset_s"].item() - inside["onset_s"].item()) <= 0.25 + 1e-9
    assert got["ep_type"].item() == inside["ep_type"].item()
    assert abs(qc.loc[b, "av_lag_s"] - (synthetic.L_UI_S
                                        - small_world.by_id(b).meta["cal_offset_s"])) <= 0.3
    assert qc.loc[b, "n_bids"] == 1 and qc.loc[a, "n_bids"] == 0   # updated by `episodes`
    # a --class run replaces only that class's rows
    assert cli.main(["track", "--class", str(b)]) == 0
    hf2 = pd.read_parquet(config.table_path("hand_frames"))
    assert set(hf2["class_id"]) == set(hf["class_id"]) and len(hf2) == len(hf)
    assert set(pd.read_parquet(config.table_path("hand_qc"))["class_id"]) == {a, b}
    assert cli.main(["retime", "--class", str(b)]) == 0
    re = pd.read_parquet(config.table_path("hand_episodes"))
    schemas.validate(re, "hand_episodes")
    mine = re[re["class_id"] == b]
    assert not mine["retime_failed"].any()
    assert abs(mine["onset_retimed_s"].item() - inside["onset_s"].item()) <= 0.05
    assert len(re) == len(ep)
    # A1: the retime share lands in hand_qc; re-running `episodes` warns that it wipes the
    # re-timed onsets (g = 1 would silently fall back to coarse onsets)
    qc = pd.read_parquet(config.table_path("hand_qc")).set_index("class_id")
    assert qc.loc[b, "retime_failed_share"] == 0.0 and not qc.loc[b, "g1_approximate"]
    with caplog.at_level(logging.WARNING, logger="oncue.hands"):
        assert cli.main(["episodes", "--class", str(b)]) == 0
    assert any("re-timed onsets" in r.getMessage() for r in caplog.records)


# --------------------------------------------------------------------------- review fixes


def _tiles(T, n_tiles, fn):
    """hand_tiles rows from fn(t, k) -> dict(frac, shape, white, ncc) (no glyph bits)."""
    rows = []
    for t in T:
        for k in range(n_tiles):
            d = fn(t, k)
            if d is None:
                continue
            rows.append({"class_id": 1, "pts_s": float(t), "tile_idx": k, "x0": 100 * k,
                         "x1": 100 * k + 90, "frac_hand_s1": d["frac"],
                         "frac_hand_s2": d["frac"], "frac_tick_s1": 0.0, "frac_tick_s2": 0.0,
                         "shape_s1": d["shape"], "shape_s2": d["shape"],
                         "white_s1": d["white"], "white_s2": d["white"], "border": 0.0,
                         "ncc_me": d["ncc"], "glyph_bits": bytes(hands.GLYPH_BYTES)})
    df = pd.DataFrame(rows)
    for c in df.columns:
        if c.startswith(("frac", "shape", "white", "border", "ncc")):
            df[c] = df[c].astype(np.float32)
    return df


UPD = {"frac": 0.40, "shape": 0.9, "white": 0.2}
DN = {"frac": 0.01, "shape": 0.0, "white": 0.0}


def test_blips_never_start_extend_or_end_an_episode():
    """Episodes are built from confirmed runs: a one-sample UP blip 2.5 s before a raise
    neither moves its onset nor adds a toggle; a one-sample DOWN inside a raise is not a
    toggle; an isolated blip (>= 3 s from any confirmed raise) is a glitch, decided apart
    from the raise next to it."""
    near = hands.episodes(_hf("D" * 600 + "U" + "D" * 5 + "U" * 40 + "D" * 20))
    clean = hands.episodes(_hf("D" * 606 + "U" * 40 + "D" * 20))
    assert len(near) == 1 and near["ep_type"].item() == C.EP_BID
    assert near["onset_s"].item() == pytest.approx(clean["onset_s"].item())   # 302.75
    assert near["n_toggles"].item() == 0 and near["dur_s"].item() == clean["dur_s"].item()
    dip = hands.episodes(_hf("D" * 10 + "U" * 8 + "D" + "U" * 8 + "D" * 10))
    assert len(dip) == 1 and dip["n_toggles"].item() == 0
    far = hands.episodes(_hf("D" * 600 + "U" + "D" * 7 + "U" * 40 + "D" * 20))
    assert far["ep_type"].tolist() == [C.EP_GLITCH, C.EP_BID]
    assert far["onset_s"].iloc[1] == pytest.approx(303.75) and not far["left_censored"].iloc[1]
    off = hands.episodes(_hf("D" * 600 + "U" + "D" * 20), det=hands.Detector(
        blips_as_glitches=False))
    assert len(off) == 0


def test_live_state_holds_short_na_and_exits_into_long_na():
    U, D, N = C.STATE_UP, C.STATE_DOWN, C.STATE_NA
    from oncue import causal

    flicker = causal.live_states([D, D, U, U, U, N, N, U, U])
    assert list(flicker[3:]) == [U] * 6                      # 1 s of NA: still UP
    lost = causal.live_states([D, D, U, U, U, N, N, N, N, D, D])
    assert list(lost[3:]) == [U, U, U, U, N, N, N, D]       # NA from the 3rd NA sample
    blip = causal.live_states([D, D, U, N, N, D, D])
    assert U not in list(blip)                               # NA never confirms a change
    full = causal.live_states(lost)
    for k in range(1, len(lost)):
        assert list(causal.live_states(lost[:k])) == list(full[:k])


def test_live_state_and_bids_ignore_frames_after_t():
    """The DOWN baseline behind the live state is a running median of samples <= t: a late
    lime object (frac 0.16, shape fails) cannot erase an earlier live raise. The whole-class
    baseline stays label side, flags the class and lists the erased raise for the audit."""
    def fn(t, k):
        if 300 <= t < 320:
            return {"frac": 0.30, "shape": 0.9, "white": 0.2, "ncc": 0.95}
        if t >= 400:
            return {"frac": 0.16, "shape": 0.1, "white": 0.0, "ncc": 0.95}
        return {"frac": 0.02, "shape": 0.0, "white": 0.0, "ncc": 0.95}
    T = np.arange(0, 1200, 0.5)
    ht = _tiles(T, 1, fn)
    full = hands.track(ht)
    part = hands.track(ht[ht["pts_s"] <= 350])
    m = full["t_s"] <= 350
    assert full.loc[m, "live_state"].tolist() == part["live_state"].tolist()
    assert full.loc[full["t_s"] == 310, "live_state"].item() == C.STATE_UP
    assert full.loc[m, "raw_state_live"].tolist() == part["raw_state_live"].tolist()
    assert full.attrs["flagged"] and full.loc[full["t_s"] == 310, "raw_state"].item() != C.STATE_UP
    audit = hands.flagged_audit_runs(full)
    assert len(audit) and audit["start_s"].min() <= 300.5 and audit["end_s"].max() >= 319
    # every prefix: the live state at t never reads a later frame
    for cut in (150.0, 305.0, 401.0, 800.0):
        p = hands.track(ht[ht["pts_s"] <= cut])
        assert full.loc[full["t_s"] <= cut, "live_state"].tolist() == p["live_state"].tolist()


def test_n_others_never_counts_bradys_tile_when_he_is_na():
    """§2.5: Brady's own tile is never in Δn. A 1 s label dropout (NCC 0.4) while three
    classmates raise must not turn his bid into a vote."""
    def make(drop):
        def fn(t, k):
            if k == 0:
                d = dict(UPD if 300 <= t < 330 else DN)
                d["ncc"] = 0.4 if (drop and 303.0 <= t <= 303.5) else 0.95
                return d
            d = dict(UPD if (k in (1, 2, 3) and 302 <= t < 330) else DN)
            d["ncc"] = 0.2
            return d
        return fn
    T = np.arange(0, 600, 0.5)
    for drop in (False, True):
        hf = hands.track(_tiles(T, 10, make(drop)))
        assert hf["n_others"].max() == 3
        ep = hands.episodes(hf)
        assert ep.loc[ep["onset_s"].between(299, 301), "ep_type"].item() == C.EP_BID


def test_lookalike_name_is_rejected_by_the_loco_ncc_floor():
    """Brady off the strip, a classmate whose label shares his first glyphs (NCC 0.86, no
    close runner-up once Brady is gone) must read NA, not Brady: picks must reach the LOCO
    anchor floor."""
    def fn(t, k):
        if k == 0:
            return None if t >= 300 else {**DN, "ncc": 0.98}
        return {**DN, "ncc": 0.86 if k == 1 else 0.2}
    T = np.arange(0, 600, 0.5)
    ht = _tiles(T, 5, fn)
    loose = hands.track(ht)
    assert loose.loc[loose["t_s"] >= 300, "tile_found"].all()           # wrong tile
    rng = np.random.default_rng(0)
    eras = {1: "e", 2: "e", 3: "e"}
    tmpl = rng.random((C.GLYPH_H, C.GLYPH_W)) > 0.5
    anchors = {}
    for cid in (2, 3):                       # other classes' anchors of the same era
        frames = []
        for _ in range(4):
            noisy = tmpl ^ (rng.random(tmpl.shape) < 0.01)
            frames.append(hands.AnchorFrame(0.0, 0, np.stack([noisy, ~tmpl])))
        anchors[cid] = frames
    floors = hands.ncc_floors(anchors, {c: tmpl.astype(np.float32) for c in eras}, eras)
    assert floors[1] is not None and 0.86 < floors[1] < 0.98
    strict = hands.track(ht, ncc_floor=floors[1])
    assert strict.loc[strict["t_s"] < 300, "tile_found"].all()
    assert not strict.loc[strict["t_s"] >= 300, "tile_found"].any()
    assert strict.attrs["n_floor_rejected"] == int((loose["t_s"] >= 300).sum())
    assert strict.attrs["n_lookalike_frames"] >= strict.attrs["n_floor_rejected"]
    assert hands.pick_tile(np.array([0.9, 0.2]), ncc_floor=0.95)[0] == -1
    assert hands.pick_tile(np.array([0.97, 0.2]), ncc_floor=0.95)[0] == 0


def test_lookalike_label_rendered_with_the_world_template(small_world, template, pass1s):
    """Same check on rendered strips: Brady's label replaced by an invented lookalike
    ("Brady S.") while he is off the strip; the floor comes from the world's anchors."""
    sc = small_world.classes[0]
    geom = _geometry(sc)
    gone = _reordered(sc, list(range(sc.layout["n_tiles"])), relabel_me="Brady S.")
    ht = hands.scan(sc.class_id, FPS, _strips(sc, np.arange(300, 310, 0.5))
                    + _strips(gone, np.arange(310, 320, 0.5)), geometry=geom)
    eras = {cid: "all" for cid in pass1s} | {-1: "all"}
    templates = hands.build_era_templates(pass1s, eras)
    templates[-1] = template
    floor = hands.ncc_floors(pass1s, templates, eras)[-1]
    hf = hands.track(ht, template, ncc_floor=floor)
    assert hf.loc[hf["t_s"] < 310, "tile_found"].all()
    assert not hf.loc[hf["t_s"] >= 310, "tile_found"].any()


def _index_mp4(path, n=300, fps=30, w=64, h=32, offset=0.37):
    """A local mp4 whose frame i shows i in two grey blocks; video starts `offset` s after
    the audio (a_start != v_start). Returns (path, frame pts on the audio clock)."""
    import json
    import subprocess

    raw = path.with_suffix(".raw.mp4")
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
                          "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps), "-i",
                          "pipe:0", "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv444p",
                          str(raw)], stdin=subprocess.PIPE)
    for i in range(n):
        f = np.zeros((h, w, 3), np.uint8)
        f[:, : w // 2] = 4 + 8 * (i // 32)
        f[:, w // 2:] = 4 + 8 * (i % 32)
        p.stdin.write(f.tobytes())
    p.stdin.close()
    p.wait()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-itsoffset", str(offset), "-i",
                    str(raw), "-f", "lavfi", "-i", f"sine=frequency=440:duration={n / fps + 1}",
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", str(path)],
                   check=True)
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_frames",
                          "-show_entries", "frame=best_effort_timestamp_time", "-of", "json",
                          str(path)], capture_output=True, text=True, check=True).stdout
    return path, np.array([float(x["best_effort_timestamp_time"])
                           for x in json.loads(out)["frames"]])


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                    reason="needs local ffmpeg/ffprobe")
def test_scan_samples_never_show_content_from_after_their_label(tmp_path):
    """§2.1 clock, through the real reader on a local file (no network): with ffmpeg's
    default fps rounding a 2 fps sample labelled t showed the frame of t + 0.24 s; with
    round=up the content is never later than the label (at most one source frame earlier),
    also after a resume with input -ss."""
    path, vpts = _index_mp4(tmp_path / "idx.mp4")
    info = hands.probe(str(path))

    def decode(img):
        hi = round((float(img[:, :32].mean()) - 4) / 8)
        lo = round((float(img[:, 32:].mean()) - 4) / 8)
        return int(hi) * 32 + int(lo)

    for start in (0.0, 3.3):
        reader = hands.FfmpegStripReader(lambda: str(path), width=64, strip_y=0, strip_h=32,
                                         fps=2, a_start=info["a_start"],
                                         fmt_start=info["fmt_start"], start_s=start,
                                         hwaccel=None, timeout_s=30)
        got = [(t, decode(f)) for t, f in reader]
        assert len(got) >= 10
        content = np.array([vpts[i] - info["a_start"] for _, i in got])
        label = np.array([t for t, _ in got])
        assert (content <= label + 1e-3).all(), (content - label).max()
        assert (content >= label - 1 / 30 - 1e-3).all()


def test_retime_window_fits_a_late_onset_and_reports_the_failed_share():
    """The re-timing read covers [t_up - 1.0, t_up + 0.5]: an onset right at t_up still has
    its 3 confirming frames inside; A1 share of failed / never re-timed bid onsets."""
    assert config.RETIME_PRE_S == 1.0 and config.RETIME_SPAN_S - config.RETIME_PRE_S >= 0.5
    ep = pd.DataFrame({"class_id": 1, "ep_id": [0, 1, 2, 3, 4], "onset_s": 1.0,
                       "ep_type": [C.EP_BID, C.EP_BID, C.EP_BID, C.EP_VOTE, C.EP_BID],
                       "left_censored": [False, False, False, False, True],
                       "retime_failed": pd.array([False, True, pd.NA, True, True],
                                                 dtype="boolean")})
    r = hands.retime_summary(ep)
    assert r["n_bids"] == 3 and r["n_retime_failed"] == 2 and r["n_retimed"] == 2
    assert r["retime_failed_share"] == pytest.approx(2 / 3) and r["g1_approximate"] is True
    ok = hands.retime_summary(ep.assign(retime_failed=False))
    assert ok["retime_failed_share"] == 0.0 and ok["g1_approximate"] is False


def test_contested_uses_open_questions_in_the_last_minute():
    """§2.5: a Δn-typed vote with no VOTE hit is contested when an INVITE hit OR a detected
    open question (`q_open`) falls in the professor's last 60 s."""
    n = 80
    others = np.zeros(n, int)
    others[44:60] = 5
    hf = _hf("D" * 44 + "U" * 8 + "D" * 28, n_others=others, n_tiles=11)
    words = pd.DataFrame({"class_id": 900101, "w_idx": range(5), "role": "prof",
                          "start_s": [10.0, 10.3, 10.6, 10.9, 11.2],
                          "end_s": [10.2, 10.5, 10.8, 11.1, 11.4],
                          "text_norm": ["what", "is", "the", "main", "idea"]})
    plain = hands.episodes(hf, words=words)
    assert plain["ep_type"].item() == C.EP_VOTE and not plain["contested"].item()
    q = hands.q_open_window(words)
    assert q(22.0) and not q(80.0) and not q(5.0)
    got = hands.episodes(hf, words=words, q_open=q)
    assert got["contested"].item()


def test_retiming_succeeds_at_every_sub_sample_phase(small_world, template):
    """With round=up sampling (a sample never shows content after its label; here it shows
    content one source frame older), every onset phase relative to the 0.5 s grid is
    re-timed to within one frame, and the coarse onset is within [-0.5, 0] s + a frame."""
    sc0 = small_world.classes[0]
    rng = np.random.default_rng(5)
    on = 30.0 + 15.0 * np.arange(20) + rng.uniform(0, 0.5, 20)
    ep = pd.DataFrame({c: [np.nan] * 20 for c in sc0.episodes.columns})
    ep["onset_s"], ep["offset_s"] = on, on + 8.0
    sc = dataclasses.replace(sc0, episodes=ep, others_hands=sc0.others_hands.iloc[:0],
                             ticks=sc0.ticks.iloc[:0],
                             _cache={})
    geom = _geometry(sc)
    ts = np.arange(0.0, 340.0, 0.5)
    lag = 1 / 30                                       # content one source frame older
    strips = list(zip(ts, synthetic.frames_for(sc, np.maximum(ts - lag, 0.0), full=False),
                      strict=True))
    hf = hands.track(hands.scan(sc.class_id, FPS, strips, geometry=geom, template=template),
                     template)
    bids = hands.episodes(hf)
    bids = bids[(bids["ep_type"] == C.EP_BID) & ~bids["left_censored"]].reset_index(drop=True)
    assert len(bids) == 20

    def reader(t0, dur):
        tt = np.arange(math.ceil(t0 * 30 - 1e-6) / 30, t0 + dur, 1 / 30)
        return _strips(sc, tt)
    rt = hands.retime(bids, reader, geometry=geom, template=template)
    assert not rt["retime_failed"].any()
    truth = on[np.abs(on[:, None] - rt["onset_s"].to_numpy()[None, :]).argmin(axis=0)]
    assert np.abs(rt["onset_retimed_s"].to_numpy() - truth).max() <= 1 / 30 + 1e-6
    coarse = rt["onset_s"].to_numpy() - truth
    assert coarse.min() >= -0.5 - 1e-6 and coarse.max() <= lag + 0.25 + 1e-6
