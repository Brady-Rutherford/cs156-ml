"""oncue.embed and oncue.edu: sha1-cached embeddings, causal recency pooling, the z-null and
the EduBehaviors scores (A3). Every model is a fake injected through `loader`: nothing is
downloaded (the conftest also blocks sockets and forces Hugging Face offline)."""

from __future__ import annotations

import hashlib
import types

import numpy as np
import pandas as pd
import pytest
from oncue import config, edu, embed, pcw, text

DIM = 16


class FakeST:
    """Deterministic bag-of-words hashing 'sentence transformer' (similar text -> similar)."""

    def __init__(self, name="fake", dim=DIM):
        self.name, self.dim, self.calls, self.texts = name, dim, 0, []

    def encode(self, texts, batch_size=32, normalize_embeddings=False, convert_to_numpy=True,
               show_progress_bar=False):
        self.calls += 1
        self.texts += list(texts)
        out = np.zeros((len(texts), self.dim), np.float32)
        for i, t in enumerate(texts):
            for tok in t.split():
                h = hashlib.sha1(f"{self.name}:{tok}".encode()).digest()
                out[i, h[0] % self.dim] += 1.0 + h[1] / 255
        if normalize_embeddings:
            out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
        return out


def make_loader(models: dict):
    def loader(name, device):
        if name not in models:
            raise OSError(f"no such model {name}")
        return models[name]
    return loader


@pytest.fixture(autouse=True)
def _no_default_loaders(monkeypatch):
    """Any accidental use of the real (downloading) loaders fails loudly."""
    def boom(*a, **k):
        raise AssertionError("a real model loader was called in a test")
    monkeypatch.setattr(embed, "default_loader", boom)
    monkeypatch.setattr(edu, "default_loader", boom)


def make_embedder(tmp_path, model=None, name="fake-model", **kw):
    model = model or FakeST(name)
    return embed.Embedder(name, cache_dir=tmp_path / "emb", dim=DIM, device="cpu",
                          loader=make_loader({name: model}), **kw), model


# --------------------------------------------------------------------------- cache


def test_vector_cache_roundtrip_and_persistence(tmp_path):
    c = embed.VectorCache(tmp_path / "c", 4)
    keys = [embed.sha1_key("m", t) for t in ("a", "b")]
    c.put(keys, np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32))
    got, found = embed.VectorCache(tmp_path / "c", 4).get(keys + [embed.sha1_key("m", "z")])
    assert found.tolist() == [True, True, False]
    assert got.dtype == np.float16 and got[1, 1] == 1 and not got[2].any()
    assert len(embed.VectorCache(tmp_path / "c", 4)) == 2
    with pytest.raises(ValueError):
        len(embed.VectorCache(tmp_path / "c", 5))
    assert embed.sha1_key("m1", "a") != embed.sha1_key("m2", "a")


# --------------------------------------------------------------------------- embedder


def test_embedder_shapes_norms_and_cache(tmp_path):
    emb, model = make_embedder(tmp_path)
    texts = ["the bias of the model", "", "the bias of the model", "variance  matters", None]
    v = emb.encode(texts)
    assert v.shape == (5, DIM) and v.dtype == np.float32
    assert np.allclose(np.linalg.norm(v[[0, 2, 3]], axis=1), 1, atol=1e-6)
    assert not v[1].any() and not v[4].any()           # empty -> zeros, never sent
    assert sorted(model.texts) == ["the bias of the model", "variance matters"]
    assert np.array_equal(v[0], v[2])
    # a new instance with the same cache dir never calls the model and is bit-identical
    emb2, model2 = make_embedder(tmp_path)
    assert np.array_equal(emb2.encode(texts), v)
    assert model2.calls == 0 and emb2.n_encoded == 0
    # in-memory reuse: a repeat on the same instance does not hit the model either
    calls = model.calls
    emb.encode(["variance matters"])
    assert model.calls == calls


def test_cache_namespace_is_per_model(tmp_path):
    emb_a, _ = make_embedder(tmp_path, name="model-a")
    emb_b, model_b = make_embedder(tmp_path, name="model-b")
    emb_a.encode(["shared text"])
    emb_b.encode(["shared text"])
    assert model_b.calls == 1                         # not served from model-a's cache


def test_load_failure_and_fallback(tmp_path):
    bad = embed.Embedder("missing", cache_dir=tmp_path, dim=DIM, device="cpu",
                         loader=make_loader({}))
    with pytest.raises(embed.EmbedderError):
        bad.encode(["x"])
    fb = FakeST("fb")
    emb = embed.Embedder.with_fallback("missing", "fb", cache_dir=tmp_path, dim=DIM,
                                       device="cpu", loader=make_loader({"fb": fb}))
    assert emb.model_name == "fb"
    emb.encode(["x y"])
    assert fb.calls == 1
    wrong = embed.Embedder("w", cache_dir=None, dim=DIM + 1, device="cpu",
                           loader=make_loader({"w": FakeST("w")}))
    with pytest.raises(embed.EmbedderError):
        wrong.encode(["dimension mismatch"])


# --------------------------------------------------------------------------- pooling


def _unit_rows(n, d=DIM, seed=0):
    x = np.random.default_rng(seed).normal(size=(n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_pooled_matches_formula_and_is_causal():
    E = _unit_rows(5)
    close = np.array([10.0, 20.0, 30.0, 200.0, 205.0])
    inc = np.array([True, True, False, True, True])           # frame 2 is Brady's
    t, tau = 31.0, 15.0
    want = sum(np.exp(-(t - close[i]) / tau) * E[i] for i in (0, 1))
    got = embed.pooled(E, close, t, tau, include=inc)[0]
    assert np.allclose(got, want / np.linalg.norm(want), atol=1e-6)
    # frames closing after t, or at/before t - 180, never count
    assert np.allclose(embed.pooled(E[:2], close[:2], t, tau)[0], got, atol=1e-6)
    late = embed.pooled(E, close, 201.0, tau, include=inc)[0]
    assert np.allclose(late, E[3], atol=1e-6)
    assert np.isnan(embed.pooled(E, close, 5.0, tau)).all()
    with pytest.raises(ValueError):
        embed.pooled(E, close[::-1], t, tau)


def test_pooled_tail_and_latest_and_mean_window():
    E = _unit_rows(3)
    close = np.array([10.0, 20.0, 30.0])
    tail = _unit_rows(2, seed=1)
    ts = np.array([25.0, 40.0])
    tail_s = np.array([24.0, np.nan])
    p = embed.pooled(E, close, ts, 15.0, tail_emb=tail, tail_s=tail_s)
    want0 = (np.exp(-15 / 15) * E[0] + np.exp(-5 / 15) * E[1] + np.exp(-1 / 15) * tail[0])
    assert np.allclose(p[0], want0 / np.linalg.norm(want0), atol=1e-6)
    last = embed.latest(E, close, ts, tail_emb=tail, tail_s=tail_s)
    assert np.allclose(last[0], tail[0]) and np.allclose(last[1], E[2])
    assert np.isnan(embed.latest(E, close, [5.0])).all()
    mw = embed.mean_window(E, close, [40.0], start_ago=0, end_ago=25)
    want = E[1] + E[2]
    assert np.allclose(mw[0], want / np.linalg.norm(want), atol=1e-6)
    assert np.isnan(embed.mean_window(E, close, [5.0])).all()


def test_pooling_on_synthetic_frames_is_unchanged_by_later_frames(truth, synth_class,
                                                                  tmp_path):
    emb, _ = make_embedder(tmp_path)
    f = text.frames(truth[synth_class.class_id]["words"])
    E = emb.encode(f["text_norm"].tolist())
    close = f["close_s"].to_numpy()
    inc = (f["role"] != config.ROLE_ME).to_numpy()
    t = 900.0
    a = embed.pooled(E, close, t, 60.0, include=inc)
    keep = close <= t
    b = embed.pooled(E[keep], close[keep], t, 60.0, include=inc[keep])
    assert np.array_equal(a, b)


# --------------------------------------------------------------------------- z-null


def test_znull_stats_and_minimum():
    ref = _unit_rows(80, seed=3)
    zn = embed.ZNull(ref)
    u = _unit_rows(2, seed=4)
    z = zn.z(u, ref)
    assert np.allclose(z.mean(axis=1), 0, atol=1e-5) and np.allclose(z.std(axis=1), 1, atol=1e-4)
    assert np.allclose(zn.z_pairs(u, ref[:2]), np.diag(zn.z(u, ref[:2])), atol=1e-5)
    small = embed.ZNull(ref[: config.ZNULL_MIN - 1])
    assert not small.ok and np.isnan(small.z(u, ref)).all()


def _pool(n_classes=8, per_class=10):
    rows = []
    for c in range(n_classes):
        for j in range(per_class):
            rows.append({"chunk_key": f"{c}:b:prompt:{j}", "class_id": 100 + c,
                         "course_code": ("CS901", "SS902")[c % 2],
                         "class_start": f"2025-0{1 + c % 9}-10T14:00:00+00:00",
                         "kind": "prompt", "text": f"topic{c} word{j} shared"})
    return pd.DataFrame(rows)


def test_reference_set_is_earlier_other_course_and_stable():
    pool = _pool()
    keys = embed.reference_keys(pool, 999, class_start="2025-05-01T00:00:00Z",
                                course_code="CS901", n=200)
    chosen = pool.set_index("chunk_key").loc[keys]
    assert len(keys) == 20                              # classes 101 and 103 only
    assert (chosen["course_code"] != "CS901").all()
    assert (pd.to_datetime(chosen["class_start"]) < pd.Timestamp("2025-05-01", tz="UTC")).all()
    # adding later classes never changes the set; the ranking is deterministic
    later = _pool(12).iloc[80:].assign(class_start="2026-01-10T14:00:00+00:00")
    more = pd.concat([pool, later], ignore_index=True)
    assert embed.reference_keys(more, 999, class_start="2025-05-01T00:00:00Z",
                                course_code="CS901", n=200) == keys
    few = embed.reference_keys(pool, 999, class_start="2025-05-01T00:00:00Z",
                               course_code="CS901", n=5)
    assert few == keys[:5]
    classes = pd.DataFrame({"class_id": [999], "sched_start_utc": ["2025-05-01T00:00:00Z"],
                            "course_code": ["CS901"]})
    assert embed.reference_keys(pool, 999, classes=classes) == keys[:config.ZNULL_N]
    with pytest.raises(ValueError):
        embed.reference_keys(pool, 12345)


def test_znull_embeds_only_the_reference_chunks(tmp_path):
    emb, model = make_embedder(tmp_path)
    pool = _pool(40, 5)
    zn = embed.znull(999, pool, emb.encode, class_start="2025-12-31T00:00:00Z",
                     course_code="CS901")
    assert zn.ok and len(zn.ref) == len(zn.keys)
    texts = pool.set_index("chunk_key").loc[list(zn.keys), "text"]
    assert set(model.texts) == set(texts)
    empty = embed.znull(999, pool, emb.encode, class_start="2000-01-01T00:00:00Z",
                        course_code="CS901")
    assert not empty.ok and len(empty.ref) == 0


def test_znull_with_synthetic_pcw_pool(world, truth, tmp_path):
    classes = pd.concat([t["classes"] for t in truth.values()], ignore_index=True)
    blocks = pd.concat([pcw.blocks(sc.raw["pcw_template"], sc.raw["pcw_doc"], sc.class_id)
                        for sc in world], ignore_index=True)
    pool = pcw.reference_pool(blocks, classes)
    latest = classes.sort_values("sched_start_utc").iloc[-1]
    keys = embed.reference_keys(pool, latest["class_id"], classes=classes)
    ref_classes = set(pool.set_index("chunk_key").loc[keys, "class_id"])
    starts = classes.set_index("class_id")["sched_start_utc"]
    assert all(starts[c] < latest["sched_start_utc"] for c in ref_classes)


# --------------------------------------------------------------------------- EduBehaviors (A3)


class FakeSetFit:
    def __init__(self, word: str, fail_predict=False):
        self.word, self.fail_predict, self.calls, self.texts = word, fail_predict, 0, []

    def predict_proba(self, texts):
        self.calls += 1
        self.texts += list(texts)
        if self.fail_predict:
            raise RuntimeError("bad head")
        p = np.array([0.9 if self.word in t.split() else 0.1 for t in texts])
        return np.stack([1 - p, p], axis=1)


CKPTS = {"edu_invites": "fake/invites", "edu_question": "fake/question",
         "edu_rhetorical": "fake/rhetorical"}


def _scorer(tmp_path, models, **kw):
    return edu.EduScorer(CKPTS, loader=make_loader(models), cache_dir=tmp_path / "edu",
                         log_path=tmp_path / "logs" / "edu_skipped.csv", device="cpu", **kw)


def test_edu_scores_cache_and_skip(tmp_path):
    models = {"fake/invites": FakeSetFit("anyone"), "fake/question": FakeSetFit("what"),
              "fake/rhetorical": FakeSetFit("x", fail_predict=True)}
    s = _scorer(tmp_path, models)
    out = s.score_texts(["does anyone know", "what is it", ""])
    assert list(out.columns) == ["edu_invites", "edu_question"]          # failed one absent
    assert out["edu_invites"].tolist()[:2] == pytest.approx([0.9, 0.1])
    assert np.isnan(out.loc[2, "edu_invites"])
    assert "edu_rhetorical" in s.skipped
    log = (tmp_path / "logs" / "edu_skipped.csv").read_text()
    assert "edu_rhetorical" in log and "fake/rhetorical" in log
    # the cache serves a new scorer without loading any model
    calls = []
    s2 = edu.EduScorer({k: v for k, v in CKPTS.items() if k != "edu_rhetorical"},
                       loader=lambda name, dev: calls.append(name), cache_dir=tmp_path / "edu",
                       log_path=None, device="cpu")
    again = s2.score_texts(["does anyone know", "what is it"])
    assert calls == [] and np.allclose(again.to_numpy(), out.iloc[:2].to_numpy())


def test_edu_load_failure_is_skipped_and_logged(tmp_path):
    s = _scorer(tmp_path, {"fake/invites": FakeSetFit("anyone")})
    out = s.score_texts(["anyone here"])
    assert list(out.columns) == ["edu_invites"]
    assert set(s.skipped) == {"edu_question", "edu_rhetorical"}
    assert all(r.startswith("load:") for r in s.skipped.values())
    s.score_texts(["more text"])                     # never retried
    assert set(s.skipped) == {"edu_question", "edu_rhetorical"}


def test_edu_array_interface_for_features(tmp_path):
    s = _scorer(tmp_path, {"fake/invites": FakeSetFit("anyone"),
                           "fake/question": FakeSetFit("what")})
    assert s.columns == tuple(CKPTS)                 # nothing loaded yet
    S = s.scores(["anyone", "what now", ""])
    assert s.columns == ("edu_invites", "edu_question")
    assert S.shape == (3, 2) and np.isnan(S[2]).all()
    assert S[0].tolist() == pytest.approx([0.9, 0.1])


def test_edu_scores_only_professor_frames(truth, synth_class, tmp_path):
    f = text.frames(truth[synth_class.class_id]["words"])
    model = FakeSetFit("anyone")
    s = _scorer(tmp_path, {"fake/invites": model})
    sc = s.score_frames(f)
    prof = f[f["role"] == config.ROLE_PROF]
    assert len(sc) == len(prof) and list(sc.columns[:3]) == ["class_id", "f_idx", "close_s"]
    assert set(model.texts) <= set(prof["text_norm"])
    empty = prof["text_norm"].fillna("").str.strip().eq("").to_numpy()
    assert sc["edu_invites"].isna().to_numpy().tolist() == empty.tolist()
    assert sc["edu_invites"].dropna().between(0, 1).all()


def test_edu_features_decay_and_causality():
    scores = pd.DataFrame({"class_id": 1, "f_idx": [0, 1, 2], "close_s": [10.0, 20.0, 30.0],
                           "edu_invites": [0.9, 0.2, np.nan], "edu_question": [0.0, 0.5, 0.5]})
    ts = [5.0, 10.0 + config.EPS_W_S, 25.0, 30.1]
    f = edu.edu_features(scores, ts, tau=10.0)
    assert f["edu_invites"].iloc[0] == 0.0
    assert f["edu_invites"].iloc[1] == pytest.approx(0.9 * np.exp(-0.025))
    assert f["edu_invites"].iloc[2] == pytest.approx(max(0.9 * np.exp(-1.5), 0.2 * np.exp(-0.5)))
    # frame 2 closed at 30.0 is not joined before 30.0 + eps_w
    assert f["edu_question"].iloc[3] == pytest.approx(0.5 * np.exp(-1.01))
    f2 = edu.edu_features(scores.iloc[:2], ts, tau=10.0)
    assert np.allclose(f["edu_question"].iloc[:3], f2["edu_question"].iloc[:3])
    assert "edu_rhetorical" not in f.columns
    assert all(config.feature_allowed(c) for c in f.columns if c != "t_s")


# --------------------------------------------------------------------------- CLI


def test_embed_cli_fills_emb_rows_and_scores(tmp_path, truth, monkeypatch):
    frames = pd.concat([text.frames(t["words"]) for t in truth.values()], ignore_index=True)
    tables = tmp_path / "tables"
    tables.mkdir()
    frames.to_parquet(tables / "frames.parquet", index=False)
    fake = FakeST(config.EMB, dim=config.EMB_DIM)
    monkeypatch.setattr(embed, "default_loader", make_loader({config.EMB: fake}))
    monkeypatch.setattr(edu, "default_loader", make_loader(
        {repo: FakeSetFit("anyone") for repo in config.EDU_CHECKPOINTS.values()}))
    args = types.SimpleNamespace(extra=["--tables", str(tables), "--cache", str(tmp_path / "c"),
                                        "--edu-cache", str(tmp_path / "e"), "--edu"],
                                 class_ids=None)
    assert embed.cli(args) == 0
    mat = np.load(tables / embed.FRAMES_EMB_FILE)
    out = pd.read_parquet(tables / "frames.parquet")
    assert mat.shape == (len(frames), config.EMB_DIM) and mat.dtype == np.float16
    assert out["emb_row"].tolist() == list(range(len(out)))
    sc = pd.read_parquet(tables / "edu_scores.parquet")
    assert set(config.EDU_CHECKPOINTS) <= set(sc.columns)
    assert len(sc) == int((frames["role"] == config.ROLE_PROF).sum())
