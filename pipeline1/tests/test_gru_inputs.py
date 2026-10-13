"""M5 causality (spec §4.4, §6, §9 L29): the GRU/TCN input tensors are built from causal
inputs only, over every grid step, and labels and exclusions enter only the loss mask.

* Inputs for steps <= t are bit-identical when labels are recomputed with every onset after t
  deleted (and with the label history's own causal features recomputed from the truncated
  episodes). For F-all−me_hand they are bit-identical with every onset deleted.
* The networks are causal: changing inputs after step k never changes outputs <= k.
* Changing labels at masked-out steps never changes the trained weights.

Synthetic world only (invented classes), CPU only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from oncue import config, gru

EMB_DIM = 24


# --------------------------------------------------------------------------- test-local stages
# Stand-ins for text.frames / features.build / labels.points, written to the same causal
# rules, so the test exercises gru.py exactly as the pipeline feeds it.


def _frames(words: pd.DataFrame) -> pd.DataFrame:
    w = words.sort_values("start_s").reset_index(drop=True)
    role = w["role"].astype(object).to_numpy()
    start, end = w["start_s"].to_numpy(float), w["end_s"].to_numpy(float)
    cut = np.r_[True, (role[1:] != role[:-1]) | (start[1:] - end[:-1] >= 0.4)]
    fid = np.cumsum(cut)
    f = w.groupby(fid).agg(start_s=("start_s", "min"), end_s=("end_s", "max"),
                           role=("role", "first"), ravail=("role_avail_s", "max"))
    f["close_s"] = np.maximum(f["end_s"] + config.FRAME_PAUSE_S + config.EPS_W_S, f["ravail"])
    return f.reset_index(drop=True)


def _frame_emb(frames: pd.DataFrame, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(len(frames), EMB_DIM)).astype(np.float32)


def _last(sorted_vals: np.ndarray, n: np.ndarray) -> np.ndarray:
    """The n-th smallest value (the latest one before the cut), NaN when n == 0."""
    vals = np.r_[np.nan, sorted_vals]
    return vals[n]


def _features(t: np.ndarray, words: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    """Causal scalars at every grid step: ctx, me (Brady's words so far), me_hand (eligible
    episodes with offset_live_s < t), q (a professor question in the last 20 s)."""
    eps = config.EPS_W_S
    me_end = np.sort(words.loc[words["role"] == config.ROLE_ME, "end_s"].to_numpy(float))
    prof = words[words["role"] == config.ROLE_PROF]
    q_end = np.sort(prof.loc[prof["text_masked"].astype(str).str.endswith("?"),
                             "end_s"].to_numpy(float))
    ep = episodes[episodes["me_hand_eligible"].astype(bool)]
    off = np.sort(ep["offset_live_s"].to_numpy(float))
    n_off = np.searchsorted(off, t, side="left")              # offset_live_s < t
    last_off = _last(off, n_off)
    n_q = np.searchsorted(q_end, t - eps, side="right")
    last_q = _last(q_end, n_q)
    q_age = np.where(np.isfinite(last_q), t - last_q, np.inf)
    return pd.DataFrame({
        "class_id": 1, "t_s": t, "ctx_min": t / 60.0,
        "me_words_so_far": np.searchsorted(me_end, t - eps, side="right").astype(float),
        "me_hand_n_so_far": n_off.astype(float), "me_hand_since_down_s": t - last_off,
        "q_open": (q_age < 20).astype(float),
        "q_age_s": np.minimum(q_age, config.Q_AGE_CAP_S),
        "q_kind": np.where(q_age < 20, "plain", "none")})


def _labels(t: np.ndarray, episodes: pd.DataFrame, H=15.0, g=2.0) -> pd.DataFrame:
    """y_H15_g2 and a mask with the label-side and hand-derived exclusions (§3)."""
    bids = np.sort(episodes.loc[episodes["ep_type"] == config.EP_BID, "onset_s"].to_numpy())
    y = np.full(len(t), np.nan)
    for i, ti in enumerate(t):
        if ((bids > ti) & (bids <= ti + g)).any():
            continue
        y[i] = float(((bids > ti + g) & (bids <= ti + H)).any())
    up = np.zeros(len(t), bool)
    for r in episodes.itertuples():
        up |= (t >= r.onset_s) & (t < r.offset_live_s + config.R_LOW_S)
    return pd.DataFrame({"t_s": t, config.HEADLINE_LABEL: y.astype(np.float32),
                         "in_G": ~up, "excl_reason": np.where(up, "hand_up", None)})


def _class_inputs(sc, words, episodes, featureset="F-all"):
    t = np.arange(config.T_OPEN_S, sc.T_end_s - config.H_S + 1e-9, config.DELTA_S)
    fr = _frames(words)
    feats = _features(t, words, episodes)
    raw = gru.raw_steps(sc.class_id, t, features=feats, frames=fr,
                        frame_emb=_frame_emb(fr, sc.class_id % 1000), words=words,
                        featureset=featureset)
    return t, raw, _labels(t, episodes)


@pytest.fixture(scope="module")
def encoders(world, truth):
    """Step encoders fit on OTHER classes' raw inputs (no labels), one per feature set."""
    out = {}
    for fs in ("F-all", "F-all-me_hand"):
        raws = [_class_inputs(sc, truth[sc.class_id]["words"],
                              truth[sc.class_id]["hand_episodes"], fs)[1]
                for sc in world.classes[1:4]]
        out[fs] = gru.StepEncoder(pca_dim=8).fit(raws)
    return out


@pytest.fixture(scope="module")
def cls0(world, truth):
    sc = world.classes[0]
    return sc, truth[sc.class_id]["words"], truth[sc.class_id]["hand_episodes"]


# --------------------------------------------------------------------------- the §4.4 test


def test_inputs_up_to_t_ignore_onsets_after_t(cls0, encoders):
    sc, words, ep = cls0
    t, raw, lab = _class_inputs(sc, words, ep)
    X = gru.build_inputs(raw, encoders["F-all"])
    assert X.shape == (len(t), encoders["F-all"].transform(raw).shape[1])
    bids = ep.loc[ep["ep_type"] == config.EP_BID, "onset_s"].to_numpy()
    assert len(bids) >= 2
    changed_labels = 0
    for o in bids:
        t_cut = o - 5.0                                     # positives before it will flip
        kept = ep[ep["onset_s"] <= t_cut]
        _, raw2, lab2 = _class_inputs(sc, words, kept)
        X2 = gru.build_inputs(raw2, encoders["F-all"])
        k = int(np.searchsorted(t, t_cut, side="right"))   # steps <= t_cut
        assert np.array_equal(X[:k].view(np.uint32), X2[:k].view(np.uint32))
        y1, m1 = gru.loss_targets(lab, config.HEADLINE_LABEL)
        y2, m2 = gru.loss_targets(lab2, config.HEADLINE_LABEL)
        changed_labels += int(((y1 != y2) | (m1 != m2))[:k].sum())
    assert changed_labels > 0                    # the labels really changed before t


def test_me_hand_variant_ignores_every_onset(cls0, encoders):
    sc, words, ep = cls0
    none = ep.iloc[:0]
    _, raw, _ = _class_inputs(sc, words, ep, "F-all-me_hand")
    _, raw0, _ = _class_inputs(sc, words, none, "F-all-me_hand")
    enc = encoders["F-all-me_hand"]
    assert not any(c.startswith("me_hand_") for c in raw.scalars.columns)
    assert np.array_equal(gru.build_inputs(raw, enc).view(np.uint32),
                          gru.build_inputs(raw0, enc).view(np.uint32))
    # ... while F-all does read the label history (so the check above is not vacuous)
    _, a, _ = _class_inputs(sc, words, ep)
    _, b, _ = _class_inputs(sc, words, none)
    assert not np.array_equal(gru.build_inputs(a, encoders["F-all"]),
                              gru.build_inputs(b, encoders["F-all"]))


def test_inputs_never_read_label_or_mask_columns(cls0, encoders):
    sc, words, ep = cls0
    t, raw, lab = _class_inputs(sc, words, ep)
    feats = _features(t, words, ep)
    leaky = feats.assign(**{config.HEADLINE_LABEL: lab[config.HEADLINE_LABEL].to_numpy(),
                            "in_G": lab["in_G"].to_numpy(), "at_risk": True,
                            "excl_reason": lab["excl_reason"].to_numpy(), "next_bid_s": 1.0,
                            "fold": 0})
    fr = _frames(words)
    raw2 = gru.raw_steps(sc.class_id, t, features=leaky, frames=fr,
                         frame_emb=_frame_emb(fr, sc.class_id % 1000), words=words)
    assert list(raw2.scalars.columns) == list(raw.scalars.columns)
    assert np.array_equal(gru.build_inputs(raw, encoders["F-all"]),
                          gru.build_inputs(raw2, encoders["F-all"]))


def test_every_grid_step_no_compaction(cls0):
    sc, words, ep = cls0
    t = np.arange(config.T_OPEN_S, 400.0, config.DELTA_S)
    feats = _features(t, words, ep)
    raw = gru.raw_steps(sc.class_id, t, features=feats)
    assert len(raw) == len(t)
    with pytest.raises(ValueError):                         # a dropped step = compaction
        gru.raw_steps(sc.class_id, np.delete(t, 5), features=feats.drop(index=5))


def test_step_components_are_causal():
    words = pd.DataFrame({"start_s": [8.0, 9.5, 9.9], "end_s": [9.0, 10.5, 12.0],
                          "role": ["prof", "student", "me"],
                          "role_avail_s": [9.5, 11.0, 12.5]})
    s = gru.speech_seconds([10.0], words)[0]
    roles = dict(zip(gru.SPEECH_ROLES, s))
    assert roles["prof"] == pytest.approx(1.0)             # (8, 9], role known at 9.5
    assert roles["unk"] == pytest.approx(0.25)              # 9.5 -> 9.75, role not yet known
    assert roles["me"] == 0.0                               # starts after t - eps_w
    m, has = gru.frame_means([10.0, 11.0], [8.5, 9.0, 10.5], np.array([[1.0], [3.0], [100.0]]))
    assert has.tolist() == [1.0, 1.0] and m[0, 0] == 2.0 and m[1, 0] == 100.0


# --------------------------------------------------------------------------- networks


@pytest.mark.parametrize("kind", gru.HAZARD_KINDS)
def test_networks_are_causal(kind):
    torch.manual_seed(0)
    net = gru.make_net(12, kind, 16).eval()
    x = torch.randn(1, 120, 12)
    y0, _ = net(x)
    k = 70
    x2 = x.clone()
    x2[:, k + 1:] = torch.randn(1, 120 - k - 1, 12) * 5
    y1, _ = net(x2)
    assert torch.allclose(y0[0, :k + 1], y1[0, :k + 1], atol=1e-6)
    assert not torch.allclose(y0[0, k + 1:], y1[0, k + 1:])


def test_gru_chunked_streaming_matches_full_sequence():
    torch.manual_seed(0)
    net = gru.make_net(10, "gru", 16)
    x = np.random.default_rng(0).normal(size=(250, 10)).astype(np.float32)
    assert np.allclose(gru.stream_logits(net, x), gru.stream_logits(net, x, chunk=37),
                       atol=1e-5)


@pytest.mark.parametrize("kind", gru.HAZARD_KINDS)
@pytest.mark.parametrize("in_dim", [6, 80, 140, 260])
def test_parameter_budget(kind, in_dim):
    size = gru.choose_size(in_dim, kind)
    n = gru.n_params(gru.make_net(in_dim, kind, size))
    assert config.GRU_PARAM_RANGE[0] <= n <= config.GRU_PARAM_RANGE[1]


@pytest.mark.parametrize("kind", gru.HAZARD_KINDS)
def test_labels_at_masked_steps_never_change_training(kind):
    rng = np.random.default_rng(1)
    xs = [rng.normal(size=(90, 7)).astype(np.float32) for _ in range(3)]
    ys = [(rng.random(90) < 0.1).astype(np.float32) for _ in range(3)]
    ms = [rng.random(90) < 0.6 for _ in range(3)]
    flipped = [np.where(m, y, 1 - y).astype(np.float32) for y, m in zip(ys, ms)]
    a = gru.train_net(xs, ys, ms, kind=kind, size=8, epochs=2, chunk=40, batch_classes=2)
    b = gru.train_net(xs, flipped, ms, kind=kind, size=8, epochs=2, chunk=40,
                      batch_classes=2)
    for (ka, va), (kb, vb) in zip(a.net.state_dict().items(), b.net.state_dict().items()):
        assert ka == kb and torch.equal(va, vb), ka


def test_gru_hazard_fit_predict(world, truth):
    raws, ys, ms = [], [], []
    for sc in world.classes[:4]:
        _, raw, lab = _class_inputs(sc, truth[sc.class_id]["words"],
                                    truth[sc.class_id]["hand_episodes"])
        y, m = gru.loss_targets(lab, config.HEADLINE_LABEL)
        raws.append(raw)
        ys.append(y)
        ms.append(m)
    model = gru.GRUHazard(max_epochs=2, n_inner=2, chunk=100, patience=1).fit(raws, ys, ms)
    assert config.GRU_PARAM_RANGE[0] <= model.n_params_ <= config.GRU_PARAM_RANGE[1]
    preds = model.predict(raws[:2])
    for raw, (p_raw, p_cal) in zip(raws, preds):
        assert p_raw.shape == p_cal.shape == (len(raw),)
        assert np.all((p_cal >= 0) & (p_cal <= 1))
    inner = model.inner_predictions()
    assert inner is not None and len(inner) == 4 and np.isfinite(inner[0][0]).all()
    assert model.history_ and "val_ap" in model.history_[0]["history"][0]
