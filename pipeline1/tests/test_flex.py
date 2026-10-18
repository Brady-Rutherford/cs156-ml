"""Part 3 (oncue.flex) and the guesses-only scoring (oncue.baselines): inputs, causality of
"my earlier words", grouped folds, the threshold rule, and the MLP's shape."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from oncue import baselines, config, flex


def test_gbt_inputs_leave_out_lookahead_empty_and_spline_columns():
    cols = ["class_id", "t_s", "emb_row", "split", "y", "has_qmark", "ctx_min", "ctx_min_spl_0",
            "spk_cur_in_my_group", "ctx_n_present", "q_open"]
    assert flex.gbt_inputs(cols) == ["ctx_min", "q_open"]


def _vec(i):
    v = np.zeros(config.EMB_DIM, np.float32)
    v[i] = 1.0
    return v


def test_my_words_so_far_reads_only_earlier_classes_and_closed_frames():
    # class 1 starts first, then 2; 3 starts at the same time as 2; 4 starts last
    start = {1: 100, 2: 200, 3: 200, 4: 300}
    items = pd.DataFrame({"class_id": [1, 1, 2, 2, 3, 4], "kind": "my_utterance",
                          "avail_s": [10.0, 50.0, 20.0, 100.0, 5.0, 1.0], "row": range(6)})
    emb = np.stack([_vec(i) for i in range(6)])
    df = pd.DataFrame({"class_id": [1, 2, 2, 2], "t_s": [5.0, 10.0, 20.0, 99.0]})
    M, n = flex.my_words_so_far(df, items, emb, start)
    assert list(n) == [0, 2, 3, 3]                      # class 1 at 5 s: nothing yet
    assert not M[0].any()
    # class 2 at 10 s: class 1's two utterances only (not 3: same start; not 4: later)
    assert np.allclose(M[1][:2], 1 / np.sqrt(2)) and not M[1][2:].any()
    # at 20 s its own first utterance (avail 20 s) has arrived; at 99 s the second (100 s) not
    assert M[2][2] > 0 and M[3][3] == 0 and not M[3][4:].any()


def test_fold_labels_keep_each_class_in_one_fold():
    cid = np.repeat(np.arange(12), 5)
    f = flex.fold_labels(cid)
    assert set(f) == set(range(flex.N_FOLDS))
    assert all(len(set(f[cid == c])) == 1 for c in range(12))


def test_tau_like_truth_gives_as_many_guesses_as_positives():
    rng = np.random.default_rng(0)
    p = rng.random(1000)
    y = (rng.random(1000) < 0.03).astype(int)
    tau = baselines.tau_like_truth(y, p)
    assert (p >= tau).sum() == y.sum()


def test_guess_scores_right_wrong_and_raises_warned():
    df = pd.DataFrame({"class_id": 7, "t_s": np.arange(0, 60, 2.0)})
    df["y"] = ((df["t_s"] >= 23) & (df["t_s"] < 38)).astype(int)        # a raise at 40 s
    raises = pd.DataFrame({"class_id": [7], "onset_s": [40.0]})
    p = np.zeros(len(df))
    p[df.index[df["t_s"] == 30]] = 1                                   # right, 10 s ahead
    p[df.index[df["t_s"] == 50]] = 1                                   # wrong
    s = baselines.guess_scores(df, p, 0.5, raises)
    assert (s["guesses"], s["right"], s["wrong"], s["raises_warned"], s["raises"]) == (2, 1, 1, 1, 1)
    assert (s["warned_2_5"], s["warned_5_10"], s["warned_10_15"]) == (0, 1, 0)   # 10 s ahead
    assert (s["perfect_warned"], s["perfect_2_5"], s["perfect_10_15"]) == (1, 1, 1)
    assert (s["warnings"], s["warnings_right"]) == (2, 1)
    p[df.index[df["t_s"].isin([26, 28, 32])]] = 1                      # 26-32 s: one run with 30
    s = baselines.guess_scores(df, p, 0.5, raises)
    assert (s["guesses"], s["warnings"], s["warnings_right"], s["warnings_wrong"]) == (5, 2, 1, 1)


def test_align_net_shape_and_size():
    torch = pytest.importorskip("torch")
    net = flex.align_net()
    b = 5
    x = [torch.randn(b, config.EMB_DIM) for _ in range(5)]
    out = net(*x, torch.randn(b, len(flex.COSINES) + 5))
    assert out.shape == (b,)
    assert flex.n_params(net) == 25_473


def test_tcn_output_at_t_depends_only_on_its_past_window():
    torch = pytest.importorskip("torch")
    from oncue import gru

    net = gru.make_net(6, "tcn", 8, 0.0, flex.TCN_SETTINGS["dilations"]).eval()
    x = torch.randn(1, 700, 6, requires_grad=True)
    t, R = 600, flex.tcn_receptive_field()
    assert R == 255                                           # 255 steps x 2 s = 8.5 min
    net(x)[0][0, t].backward()
    g = x.grad[0].abs().sum(1)
    assert g[t + 1:].sum() == 0                               # nothing after t
    assert g[:t - R + 1].sum() == 0                           # nothing older than the window
    assert g[t - R + 1:t + 1].sum() > 0


def test_val_folds_give_each_session_one_fold_round_robin():
    cid = np.repeat([30, 10, 20, 40, 50], 3)
    f = flex.val_folds(cid)
    assert all(len(set(f[cid == c])) == 1 for c in set(cid))
    assert {c: int(f[cid == c][0]) for c in (10, 20, 30, 40, 50)} == {10: 0, 20: 1, 30: 2, 40: 3, 50: 0}


def test_session_bootstrap_is_paired_and_ranks_a_perfect_model_first():
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"class_id": np.repeat(np.arange(20), 50)})
    frame["y"] = (rng.random(len(frame)) < 0.1).astype(int)
    preds = {"perfect": frame["y"].to_numpy(float), "noise": rng.random(len(frame))}
    preds["same"] = preds["noise"].copy()
    ci, diff = flex.session_bootstrap(frame, preds, ref="perfect", B=50)
    assert (ci.loc["perfect", ["AP low", "AP_w low"]] == 1.0).all()
    assert diff.loc["noise", "AP_w: ref better in"] == 1.0
    _, d2 = flex.session_bootstrap(frame, preds, ref="noise", B=50)
    assert d2.loc["same", "AP gain"] == 0 and d2.loc["same", "AP_w gain"] == 0


def test_train_net_calls_on_epoch_once_per_epoch_and_it_does_not_change_training():
    pytest.importorskip("torch")
    from oncue import gru

    rng = np.random.default_rng(0)
    xs = [rng.normal(size=(40, 5)).astype(np.float32) for _ in range(4)]
    ys = [(rng.random(40) < 0.2).astype(np.float32) for _ in range(4)]
    ms = [np.ones(40, bool) for _ in range(4)]
    seen = []
    kw = dict(kind="tcn", size=4, epochs=3, dropout=0.0, batch_classes=2, device="cpu", dilations=(1, 2))
    a = gru.train_net(xs, ys, ms, on_epoch=lambda ep, net: seen.append((ep, gru.stream_logits(net, xs[0]))), **kw)
    b = gru.train_net(xs, ys, ms, **kw)
    assert [ep for ep, _ in seen] == [1, 2, 3]
    assert np.allclose(seen[-1][1], gru.stream_logits(b.net, xs[0]), atol=1e-5)    # scoring mid-run is harmless
