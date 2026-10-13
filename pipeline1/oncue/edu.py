"""EduBehaviors classroom-discourse scores for professor frames (amendment A3, should-tier).

Five Stanford SCALE SetFit classifiers (`config.EDU_CHECKPOINTS`) score the masked,
normalised text of every professor frame once. Rules from A3:

* weights are downloaded at RUNTIME only (licences unstated) and loaded lazily; scores are
  cached per frame text under `data/cache/edu/` and never committed;
* a checkpoint that fails to load (or to predict) is skipped and logged, never faked: its
  `edu_*` column is simply absent;
* features join on frame `close_s <= t - ε_w` and decay with time since close
  (`edu_features`), like the `q_*` ages.

The model loader is injectable (`loader(repo_id, device)`), so tests never download.
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .embed import VectorCache, default_device, model_slug, sha1_key

log = logging.getLogger("oncue.edu")

EDU_LOG = "edu_skipped.csv"                 # under config.LOGS_DIR
DEFAULT_LOG = config.LOGS_DIR / EDU_LOG
_CONFIG = object()                          # default: the config path read at construction


def default_loader(repo_id: str, device: str):
    """SetFit model from the Hugging Face hub (runtime download; never used by tests)."""
    from setfit import SetFitModel

    model = SetFitModel.from_pretrained(repo_id)
    try:
        model.to(device)
    except Exception as exc:  # noqa: BLE001 - any device failure: stay on cpu
        log.warning("%s: could not move to %s (%s); using cpu", repo_id, device, exc)
    return model


def _positive_proba(out, n: int) -> np.ndarray:
    """P(positive) from predict_proba output: [n, 2] -> column 1; [n] -> as is."""
    if hasattr(out, "detach"):
        out = out.detach().cpu().numpy()
    arr = np.asarray(out, dtype=float)
    if arr.ndim == 2:
        arr = arr[:, -1]
    if arr.shape != (n,):
        raise ValueError(f"predict_proba gave shape {arr.shape} for {n} texts")
    return arr


class EduScorer:
    """Lazy, cached EduBehaviors scoring. `skipped` maps column -> reason for failures."""

    def __init__(self, checkpoints: dict[str, str] | None = None, *,
                 loader: Callable | None = None,
                 cache_dir: Path | str | None = _CONFIG,
                 log_path: Path | str | None = _CONFIG, device: str | None = None,
                 batch_size: int = 64):
        """`cache_dir` / `log_path`: None disables; the defaults are config.EDU_CACHE_DIR and
        config.LOGS_DIR / EDU_LOG as they are when the scorer is built."""
        if cache_dir is _CONFIG:
            cache_dir = config.EDU_CACHE_DIR
        if log_path is _CONFIG:
            log_path = config.LOGS_DIR / EDU_LOG
        self.checkpoints = dict(config.EDU_CHECKPOINTS if checkpoints is None else checkpoints)
        self._loader = loader or default_loader
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
        self.log_path = None if log_path is None else Path(log_path)
        self.device, self.batch_size = device, int(batch_size)
        self._models: dict[str, object] = {}
        self.skipped: dict[str, str] = {}
        self.n_scored: dict[str, int] = {c: 0 for c in self.checkpoints}

    # ---- loading -----------------------------------------------------------------------
    def _skip(self, col: str, exc: BaseException, stage: str) -> None:
        reason = f"{stage}: {type(exc).__name__}: {str(exc)[:200]}"
        self.skipped[col] = reason
        self._models.pop(col, None)
        log.warning("EduBehaviors %s (%s) skipped: %s", col, self.checkpoints[col], reason)
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            new = not self.log_path.exists()
            with open(self.log_path, "a", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                if new:
                    w.writerow(["time_utc", "column", "checkpoint", "reason"])
                w.writerow([datetime.now(UTC).isoformat(timespec="seconds"), col,
                            self.checkpoints[col], reason])

    def model(self, col: str):
        """The loaded model for `col`, or None if it failed (then it is never retried)."""
        if col in self.skipped:
            return None
        if col not in self._models:
            try:
                self.device = self.device or default_device()
                self._models[col] = self._loader(self.checkpoints[col], self.device)
            except Exception as exc:  # noqa: BLE001 - A3: any load failure skips it
                self._skip(col, exc, "load")
                return None
        return self._models[col]

    def _cache(self, col: str) -> VectorCache | None:
        if self.cache_dir is None:
            return None
        return VectorCache(self.cache_dir / model_slug(self.checkpoints[col]), 1, "float32")

    # ---- scoring -----------------------------------------------------------------------
    def score_texts(self, texts: Sequence[str]) -> pd.DataFrame:
        """One column per usable checkpoint (P(positive)); empty texts score NaN.

        A checkpoint is only loaded when some text is not already cached.
        """
        clean = [" ".join(str(t or "").split()) for t in texts]
        uniq = sorted({t for t in clean if t})
        out = pd.DataFrame(index=range(len(clean)))
        for col, repo in self.checkpoints.items():
            if col in self.skipped:
                continue
            keys = {t: sha1_key(repo, t) for t in uniq}
            cache = self._cache(col)
            got: dict[str, float] = {}
            todo = list(uniq)
            if cache is not None and todo:
                vals, found = cache.get([keys[t] for t in todo])
                got = {t: float(v[0]) for t, v, f in zip(todo, vals, found, strict=True) if f}
                todo = [t for t in todo if t not in got]
            if todo:
                model = self.model(col)
                if model is None:
                    continue
                try:
                    probs = []
                    for i in range(0, len(todo), self.batch_size):
                        part = todo[i:i + self.batch_size]
                        probs.append(_positive_proba(model.predict_proba(part), len(part)))
                    p = np.concatenate(probs)
                except Exception as exc:  # noqa: BLE001 - A3: skip, never fake
                    self._skip(col, exc, "predict")
                    continue
                self.n_scored[col] += len(todo)
                if cache is not None:
                    cache.put([keys[t] for t in todo], p.reshape(-1, 1))
                got.update(zip(todo, p.tolist(), strict=True))
            out[col] = [got.get(t, np.nan) if t else np.nan for t in clean]
        return out

    @property
    def columns(self) -> tuple[str, ...]:
        """edu_* columns not (yet) skipped. Read it AFTER `scores` to match its output."""
        return tuple(c for c in self.checkpoints if c not in self.skipped)

    def scores(self, texts: Sequence[str]) -> np.ndarray:
        """[n, len(self.columns)] array (the interface features.py / causal.py expect)."""
        df = self.score_texts(texts)
        cols = self.columns
        if not cols:
            return np.zeros((len(df), 0))
        return np.column_stack([df[c].to_numpy(float) if c in df.columns
                                else np.full(len(df), np.nan) for c in cols])

    def score_frames(self, frames: pd.DataFrame) -> pd.DataFrame:
        """Scores for professor frames only: class_id, f_idx, close_s + edu_* columns."""
        prof = frames[frames["role"] == config.ROLE_PROF].reset_index(drop=True)
        scores = self.score_texts(prof["text_norm"].fillna("").tolist())
        base = prof[["class_id", "f_idx", "close_s"]].reset_index(drop=True)
        return pd.concat([base, scores], axis=1)


def edu_columns(scores: pd.DataFrame) -> list[str]:
    return [c for c in scores.columns if c.startswith("edu_")]


def edu_features(scores: pd.DataFrame, ts: Sequence[float], *, tau: float = config.EDU_DECAY_TAU_S,
                 eps_w: float = config.EPS_W_S) -> pd.DataFrame:
    """Decayed EduBehaviors features for ONE class at decision times `ts`.

    edu_x(t) = max over professor frames f with close_f <= t - ε_w of
    score_f * exp(-(t - close_f) / τ); 0 before any scored frame. Computed exactly with a
    running max in log space (vectorised). Columns absent from `scores` stay absent.
    """
    ts = np.asarray(ts, float)
    out = pd.DataFrame({"t_s": ts})
    s = scores.sort_values("close_s", kind="stable")
    close = s["close_s"].to_numpy(float)
    k = np.searchsorted(close, ts - eps_w, "right") - 1
    for col in edu_columns(s):
        v = s[col].to_numpy(float)
        with np.errstate(divide="ignore"):
            lg = np.where(np.isfinite(v) & (v > 0), np.log(np.where(v > 0, v, 1.0)), -np.inf)
        run = np.maximum.accumulate(lg + close / tau) if len(lg) else lg
        val = np.zeros(len(ts))
        has = k >= 0
        val[has] = np.exp(run[k[has]] - ts[has] / tau)
        out[col] = val
    return out
