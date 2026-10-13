"""Embeddings (§5.4): bge-small with a sha1 disk cache, causal recency pooling, z-null.

* `Embedder` encodes masked, normalised text with `BAAI/bge-small-en-v1.5` (CLS pooling,
  L2-normalised) through sentence-transformers on `mps` (cpu if MPS is absent). Vectors are
  cached as float16 under `data/cache/emb/{model}/`, keyed by sha1(model, text), and every
  returned vector goes through float16 so cached and fresh calls are bit-identical.
* `pooled` implements v_τ(t): exponentially weighted frames with close_s in (t - 180, t],
  Brady's frames excluded, plus an optional tail, L2-normalised.
* `ZNull` / `znull` z-score similarities against 200 reference chunks from other courses'
  PCW prompts and usable answers of classes that started BEFORE the class (§5.6).

The model is loaded lazily through an injectable `loader(model_name, device)`, so tests
never download anything. Nothing here makes network calls unless the default loader runs.
`HASH_MODEL` names an offline bag-of-words hashing stand-in (dry runs, the synthetic
end-to-end run and tests): it has no semantics beyond word overlap and is never a result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

log = logging.getLogger("oncue.embed")

FRAMES_EMB_FILE = "frames_emb.npy"        # in the tables dir: float16, row = frames.emb_row
FRAMES_EMB_META = "frames_emb.json"
_SHARD_RE = re.compile(r"^shard_\d+_[0-9a-f]+\.npz$")
HASH_MODEL = "oncue/hash-bow"             # offline stand-in model (never used for results)
_CONFIG = object()                         # default: the config path read at construction


class EmbedderError(RuntimeError):
    """The embedding model could not be loaded."""


def model_slug(name: str) -> str:
    """Filesystem-safe cache namespace for a model id."""
    return re.sub(r"[^A-Za-z0-9._-]+", "__", name)


def sha1_key(namespace: str, text: str) -> bytes:
    """20-byte sha1 digest of (namespace, text): the cache key."""
    return hashlib.sha1(f"{namespace}\x00{text}".encode()).digest()


# --------------------------------------------------------------------------- disk cache


class VectorCache:
    """Append-only sha1 -> vector store: `root/shard_*.npz` files (keys uint8[n, 20],
    vecs[n, dim]). No pickle; each `put` writes one new shard atomically."""

    def __init__(self, root: Path | str, dim: int, dtype: str = "float16"):
        self.root, self.dim, self.dtype = Path(root), int(dim), np.dtype(dtype)
        self._index: dict[bytes, tuple[int, int]] | None = None
        self._shards: list[np.ndarray] = []

    def _load(self) -> None:
        if self._index is not None:
            return
        self._index = {}
        if not self.root.exists():
            return
        for p in sorted(x for x in self.root.iterdir() if _SHARD_RE.match(x.name)):
            with np.load(p, allow_pickle=False) as z:
                keys, vecs = z["keys"], z["vecs"]
            if vecs.ndim != 2 or vecs.shape[1] != self.dim:
                raise ValueError(f"{p.name}: dim {vecs.shape} != {self.dim}")
            s = len(self._shards)
            self._shards.append(vecs.astype(self.dtype, copy=False))
            for r, k in enumerate(keys):
                self._index[k.tobytes()] = (s, r)

    def __len__(self) -> int:
        self._load()
        return len(self._index)

    def get(self, keys: Sequence[bytes]) -> tuple[np.ndarray, np.ndarray]:
        """(vectors[n, dim] in the cache dtype, found mask[n]); missing rows are zero."""
        self._load()
        out = np.zeros((len(keys), self.dim), self.dtype)
        found = np.zeros(len(keys), bool)
        for i, k in enumerate(keys):
            hit = self._index.get(k)
            if hit is not None:
                out[i] = self._shards[hit[0]][hit[1]]
                found[i] = True
        return out, found

    def put(self, keys: Sequence[bytes], vecs: np.ndarray) -> None:
        self._load()
        if not len(keys):
            return
        vecs = np.asarray(vecs, self.dtype).reshape(len(keys), self.dim)
        karr = np.frombuffer(b"".join(keys), np.uint8).reshape(len(keys), 20)
        self.root.mkdir(parents=True, exist_ok=True)
        name = f"shard_{time.time_ns():020d}_{os.urandom(4).hex()}.npz"
        fd, tmp = tempfile.mkstemp(dir=self.root, suffix=".tmp")
        with os.fdopen(fd, "wb") as fh:
            np.savez(fh, keys=karr, vecs=vecs)
        os.replace(tmp, self.root / name)
        s = len(self._shards)
        self._shards.append(vecs)
        for r, k in enumerate(keys):
            self._index[bytes(k)] = (s, r)


# --------------------------------------------------------------------------- embedder


def default_device() -> str:
    try:
        import torch

        return "mps" if torch.backends.mps.is_available() else "cpu"
    except (ImportError, AttributeError, RuntimeError):   # torch missing or broken
        return "cpu"


class HashModel:
    """Offline stand-in with the sentence-transformers `encode` API: a deterministic sum of
    per-token Gaussian vectors (seeded by sha1 of the token), unit-normalised. Shared
    words give similar vectors; nothing else does. Dry runs and tests only."""

    def __init__(self, dim: int = config.EMB_DIM, seed: int = config.SEED):
        self.dim, self.seed = int(dim), int(seed)
        self._tok: dict[str, np.ndarray] = {}

    def _vec(self, tok: str) -> np.ndarray:
        v = self._tok.get(tok)
        if v is None:
            h = int(hashlib.sha1(f"{self.seed}:{tok}".encode()).hexdigest()[:12], 16)
            v = self._tok[tok] = np.random.default_rng(h).standard_normal(self.dim)
        return v

    def encode(self, texts, batch_size: int = 0, normalize_embeddings: bool = True,
               convert_to_numpy: bool = True, show_progress_bar: bool = False) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), np.float32)
        for i, text in enumerate(texts):
            toks = str(text or "").split()
            if toks:
                v = np.sum([self._vec(t) for t in toks], axis=0)
                out[i] = v / (np.linalg.norm(v) or 1.0)
        return out


def default_loader(model_name: str, device: str):
    """sentence-transformers model (downloads on first use; never called by tests), or the
    offline `HashModel` for `HASH_MODEL`."""
    if model_name == HASH_MODEL:
        return HashModel()
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=device)


def _clean_text(t) -> str:
    if t is None or (isinstance(t, float) and math.isnan(t)):
        return ""
    return " ".join(str(t).split())


class Embedder:
    """Cached sentence embeddings. `encode(texts)` -> float32 [n, dim], unit norm rows
    (empty texts give zero rows and are never sent to the model)."""

    def __init__(self, model: str = config.EMB, *, device: str | None = None,
                 cache_dir: Path | str | None = _CONFIG,
                 batch_size: int = config.EMB_BATCH, dim: int = config.EMB_DIM,
                 loader: Callable | None = None, chunk: int = 4096):
        """`cache_dir`: None disables the disk cache; the default is `config.EMB_CACHE_DIR`
        as it is when the Embedder is built (so a redirected data dir is honoured). The
        offline HASH_MODEL is never cached (it is cheaper than the cache)."""
        if cache_dir is _CONFIG:
            cache_dir = None if model == HASH_MODEL else config.EMB_CACHE_DIR
        self.model_name, self.dim = model, int(dim)
        self.device = device
        self.batch_size, self.chunk = int(batch_size), int(chunk)
        self._loader = loader or default_loader
        self._model = None
        self.cache = (VectorCache(Path(cache_dir) / model_slug(model), self.dim)
                      if cache_dir is not None else None)
        self._mem: dict[bytes, np.ndarray] = {}
        self.n_encoded = 0          # texts actually sent to the model (for tests and logs)

    @classmethod
    def with_fallback(cls, model: str = config.EMB, fallback: str = config.EMB_FALLBACK,
                      **kw) -> Embedder:
        """Load `model` now; if it fails, use `fallback` for the WHOLE run (never mixed)."""
        emb = cls(model, **kw)
        try:
            emb.load()
            return emb
        except EmbedderError as exc:
            log.warning("embedding model %s failed (%s); using %s", model, exc, fallback)
            emb = cls(fallback, **kw)
            emb.load()
            return emb

    def load(self):
        if self._model is None:
            dev = self.device or ("cpu" if self.model_name == HASH_MODEL else default_device())
            try:
                self._model = self._loader(self.model_name, dev)
            except Exception as exc:          # any loader failure -> EmbedderError
                raise EmbedderError(f"{self.model_name}: {type(exc).__name__}: {exc}") from exc
            self.device = dev
        return self._model

    def _compute(self, texts: list[str]) -> np.ndarray:
        model = self.load()
        out = model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=False)
        out = np.asarray(out, np.float32).reshape(len(texts), -1)
        if out.shape[1] != self.dim:
            raise EmbedderError(f"{self.model_name} gave dim {out.shape[1]}, want {self.dim}")
        self.n_encoded += len(texts)
        return out.astype(np.float16)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        clean = [_clean_text(t) for t in texts]
        out = np.zeros((len(clean), self.dim), np.float16)
        uniq = sorted({t for t in clean if t})
        keys = {t: sha1_key(self.model_name, t) for t in uniq}
        vec: dict[str, np.ndarray] = {}
        todo = []
        for t in uniq:
            m = self._mem.get(keys[t])
            if m is not None:
                vec[t] = m
            else:
                todo.append(t)
        if todo and self.cache is not None:
            got, found = self.cache.get([keys[t] for t in todo])
            for t, v, f in zip(todo, got, found, strict=True):
                if f:
                    vec[t] = v
            todo = [t for t, f in zip(todo, found, strict=True) if not f]
        for i in range(0, len(todo), self.chunk):
            part = todo[i:i + self.chunk]
            new = self._compute(part)
            if self.cache is not None:
                self.cache.put([keys[t] for t in part], new)
            vec.update(zip(part, new, strict=True))
        for t, v in vec.items():
            self._mem[keys[t]] = v
        for i, t in enumerate(clean):
            if t:
                out[i] = vec[t]
        return _unit(out.astype(np.float32))


def _unit(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return np.divide(x, n, out=np.zeros_like(x), where=n > 0)


# --------------------------------------------------------------------------- pooling (§5.4)


def pooled(emb: np.ndarray, close_s: np.ndarray, t, tau: float, *,
           horizon_s: float = config.POOL_HORIZON_S, include: np.ndarray | None = None,
           tail_emb: np.ndarray | None = None, tail_s: np.ndarray | None = None) -> np.ndarray:
    """v_τ(t) for each t: Σ exp(-(t - close_f)/τ) e_f over frames with close_f in
    (t - horizon, t] (and `include`, e.g. role != me), plus the tail weighted by its own
    availability time, L2-normalised. `close_s` must be non-decreasing (frames() ensures
    it). Rows with nothing to pool are NaN."""
    ts = np.atleast_1d(np.asarray(t, float))
    close = np.asarray(close_s, float)
    E = np.asarray(emb, np.float32)
    if len(close) and np.any(np.diff(close) < 0):
        raise ValueError("close_s must be non-decreasing")
    inc = np.ones(len(close), bool) if include is None else np.asarray(include, bool)
    lo = np.searchsorted(close, ts - horizon_s, "right")
    hi = np.searchsorted(close, ts, "right")
    out = np.full((len(ts), E.shape[1] if E.ndim == 2 else config.EMB_DIM), np.nan, np.float32)
    for i, (a, b, tt) in enumerate(zip(lo, hi, ts, strict=True)):
        w = np.exp(-(tt - close[a:b]) / tau) * inc[a:b]
        v = w @ E[a:b] if b > a else np.zeros(out.shape[1], np.float32)
        if tail_emb is not None and tail_s is not None and np.isfinite(tail_s[i]) \
                and tail_s[i] <= tt:
            v = v + np.exp(-(tt - tail_s[i]) / tau) * tail_emb[i]
        n = float(np.linalg.norm(v))
        if n > 0:
            out[i] = v / n
    return out


def latest(emb: np.ndarray, close_s: np.ndarray, t, *, include: np.ndarray | None = None,
           tail_emb: np.ndarray | None = None, tail_s: np.ndarray | None = None) -> np.ndarray:
    """e_last: the most recent included non-empty frame with close <= t, or the tail when
    it is non-empty at t (it is always newer). NaN rows when there is neither."""
    ts = np.atleast_1d(np.asarray(t, float))
    close = np.asarray(close_s, float)
    E = np.asarray(emb, np.float32)
    inc = np.ones(len(close), bool) if include is None else np.asarray(include, bool)
    pos = np.flatnonzero(inc & (np.abs(E).sum(axis=1) > 0))     # empty-text frames skipped
    k = np.searchsorted(close[pos], ts, "right") - 1
    out = np.full((len(ts), E.shape[1]), np.nan, np.float32)
    has = k >= 0
    out[has] = E[pos[k[has]]]
    if tail_emb is not None and tail_s is not None:
        use = np.isfinite(tail_s) & (np.asarray(tail_s) <= ts) & (np.abs(tail_emb).sum(1) > 0)
        out[use] = tail_emb[use]
    return out


def mean_window(emb: np.ndarray, close_s: np.ndarray, t, *, start_ago: float = 0.0,
                end_ago: float = np.inf, include: np.ndarray | None = None) -> np.ndarray:
    """Normalised mean of included frames with close in (t - end_ago, t - start_ago]
    (topic_shift: 0-60 s vs 60-300 s; own_sim: Brady's frames so far). NaN when empty."""
    ts = np.atleast_1d(np.asarray(t, float))
    close = np.asarray(close_s, float)
    E = np.asarray(emb, np.float32)
    inc = np.ones(len(close), bool) if include is None else np.asarray(include, bool)
    cs = np.vstack([np.zeros((1, E.shape[1]), np.float64),
                    np.cumsum(E * inc[:, None], axis=0, dtype=np.float64)])
    lo = np.searchsorted(close, ts - end_ago, "right")
    hi = np.searchsorted(close, ts - start_ago, "right")
    v = cs[np.maximum(hi, lo)] - cs[lo]
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return np.where(n > 0, v / np.where(n > 0, n, 1.0), np.nan).astype(np.float32)


# --------------------------------------------------------------------------- z-null (§5.6)


@dataclass
class ZNull:
    """Reference set R_c (unit vectors) for z(u, v) = (s(u, v) - μ_u) / σ_u."""

    ref: np.ndarray
    keys: tuple = ()

    @property
    def ok(self) -> bool:
        return len(self.ref) >= config.ZNULL_MIN

    def stats(self, U: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(μ_u, σ_u) per row of U; NaN when |R_c| < ZNULL_MIN (or u is NaN)."""
        U = np.atleast_2d(np.asarray(U, np.float32))
        if not self.ok:
            nan = np.full(len(U), np.nan)
            return nan, nan
        s = U @ self.ref.T
        return s.mean(axis=1), s.std(axis=1)

    def z(self, U: np.ndarray, V: np.ndarray) -> np.ndarray:
        """z-scores [n_u, n_v]; the null is the query u's (row-wise)."""
        U = np.atleast_2d(np.asarray(U, np.float32))
        V = np.atleast_2d(np.asarray(V, np.float32))
        mu, sd = self.stats(U)
        with np.errstate(invalid="ignore", divide="ignore"):
            return (U @ V.T - mu[:, None]) / sd[:, None]

    def z_pairs(self, U: np.ndarray, V: np.ndarray) -> np.ndarray:
        """Row-wise z(u_i, v_i)."""
        U = np.atleast_2d(np.asarray(U, np.float32))
        V = np.atleast_2d(np.asarray(V, np.float32))
        mu, sd = self.stats(U)
        with np.errstate(invalid="ignore", divide="ignore"):
            return ((U * V).sum(axis=1) - mu) / sd


def _start_of(classes: pd.DataFrame | None, class_id, class_start, course_code):
    from .pcw import to_utc

    if (class_start is None or course_code is None) and classes is not None:
        row = classes.loc[classes["class_id"] == class_id]
        if len(row):
            class_start = class_start if class_start is not None \
                else row["sched_start_utc"].iloc[0]
            course_code = course_code if course_code is not None else row["course_code"].iloc[0]
    start = to_utc(class_start)
    if start is None:
        raise ValueError(f"class {class_id}: start time unknown for the z-null")
    return start, course_code


def reference_keys(pool: pd.DataFrame, class_id, *, class_start=None, course_code=None,
                   classes: pd.DataFrame | None = None, n: int = config.ZNULL_N,
                   seed: int = config.SEED) -> list[str]:
    """Chunk keys of R_c: chunks of OTHER courses from classes that started strictly before
    class c, ranked by sha1(seed, c, key) (fixed seed; adding later classes or unrelated
    chunks never reshuffles the set), first `n`. Labels are never used."""
    from .pcw import to_utc

    start, course = _start_of(classes, class_id, class_start, course_code)
    if pool.empty:
        return []
    starts = pool["class_start"].map(to_utc)
    ok = starts.notna() & (pool["course_code"] != course) & (pool["class_id"] != class_id)
    ok &= starts.map(lambda s: s is not None and s < start)
    keys = pool.loc[ok, "chunk_key"].astype(str).tolist()
    rank = sorted(keys, key=lambda k: hashlib.sha1(f"{seed}:{class_id}:{k}".encode()).digest())
    return rank[:n]


def znull(class_id, pool: pd.DataFrame, embed_fn: Callable[[Sequence[str]], np.ndarray], *,
          classes: pd.DataFrame | None = None, class_start=None, course_code=None,
          n: int = config.ZNULL_N, seed: int = config.SEED) -> ZNull:
    """ZNull for class c from `pcw.reference_pool` rows (earlier classes, other courses).
    `embed_fn` is usually `Embedder.encode`. |R_c| < 50 gives NaN z-scores."""
    keys = reference_keys(pool, class_id, class_start=class_start, course_code=course_code,
                          classes=classes, n=n, seed=seed)
    if not keys:
        return ZNull(np.zeros((0, config.EMB_DIM), np.float32), ())
    text = pool.set_index("chunk_key").loc[keys, "text"].tolist()
    ref = np.asarray(embed_fn(text), np.float32)
    return ZNull(ref, tuple(keys))


# --------------------------------------------------------------------------- CLI


def _parse_extra(extra: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m oncue embed", add_help=False)
    p.add_argument("--tables", type=Path, default=config.TABLES_DIR)
    p.add_argument("--cache", type=Path, default=config.EMB_CACHE_DIR)
    p.add_argument("--edu-cache", type=Path, default=config.EDU_CACHE_DIR)
    p.add_argument("--model", default=config.EMB,
                   help=f"encoder (default {config.EMB}; {HASH_MODEL} = offline dry run)")
    p.add_argument("--batch", type=int, default=config.EMB_BATCH)
    p.add_argument("--edu", action="store_true", help="also score professor frames (A3)")
    p.add_argument("--edu-only", action="store_true")
    p.add_argument("--no-pcw", action="store_true", help="skip warming the PCW chunk cache")
    return p.parse_args(list(extra or []))


def embed_frames(frames: pd.DataFrame, embedder: Embedder) -> tuple[pd.DataFrame, np.ndarray]:
    """(frames with emb_row = row, float16 matrix) for every frame (empty text -> zeros)."""
    mat = embedder.encode(frames["text_norm"].fillna("").tolist()).astype(np.float16)
    out = frames.reset_index(drop=True).copy()
    out["emb_row"] = pd.array(np.arange(len(out)), dtype="Int64")
    return out, mat


def cli(args) -> int:
    """`python -m oncue embed [--edu|--edu-only] [--model M] [--tables DIR] [--cache DIR]`.

    Embeds every frame once (frames_emb.npy aligned with frames.emb_row), warms the cache
    with PCW prompts and answer chunks, and with --edu writes edu_scores.parquet (A3; local,
    never committed). Model weights download at runtime only.
    """
    opts = _parse_extra(getattr(args, "extra", []))
    src = opts.tables / "frames.parquet"
    if not src.exists():
        print(f"oncue embed: {src} not found (run `oncue frames` first)")
        return 1
    frames = pd.read_parquet(src)
    if not opts.edu_only:
        cache = None if opts.model == HASH_MODEL else opts.cache
        emb = Embedder.with_fallback(opts.model, cache_dir=cache, batch_size=opts.batch)
        frames, mat = embed_frames(frames, emb)
        np.save(opts.tables / FRAMES_EMB_FILE, mat)
        (opts.tables / FRAMES_EMB_META).write_text(json.dumps(
            {"model": emb.model_name, "n": len(mat), "dim": int(mat.shape[1])}))
        frames.to_parquet(src, index=False)
        blocks_path = opts.tables / "pcw_blocks.parquet"
        if not opts.no_pcw and blocks_path.exists():
            from .pcw import answer_chunks, chunks

            b = pd.read_parquet(blocks_path)
            texts = [c for p in b["prompt_text"].fillna("") for c in chunks(p)]
            texts += answer_chunks(b)["text"].tolist()
            emb.encode(texts)
        print(f"oncue embed: {len(mat)} frames with {emb.model_name} "
              f"({emb.n_encoded} newly encoded)")
    if opts.edu or opts.edu_only:
        from .edu import EDU_LOG, EduScorer

        scorer = EduScorer(cache_dir=opts.edu_cache, log_path=config.LOGS_DIR / EDU_LOG)
        scores = scorer.score_frames(frames)
        scores.to_parquet(opts.tables / "edu_scores.parquet", index=False)
        print(f"oncue embed: EduBehaviors scores for {len(scores)} professor frames; "
              f"skipped {sorted(scorer.skipped) or 'none'}")
    return 0
