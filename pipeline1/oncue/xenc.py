"""M6: fine-tuned cross-encoder (spec §6, could-tier).

Input text per decision point (§6):

    "[{area}] [{seg_type}] [min {m}] [since me {k}m] || {normalised X_all60, left-truncated
     to 224 tokens} || PCW: {best a_j, <= 64 tokens}"

`AutoModelForSequenceClassification("sentence-transformers/all-MiniLM-L6-v2", num_labels=1)`,
max_len 320, lr 3e-5, 2 epochs, batch 32, 6% linear warmup, on `mps`. Training keeps every
positive and a seeded r = 0.2 sample of negatives; predictions get the prior correction
logit += log r, so scores are on the full-data odds scale before Platt calibration. Nothing
here downloads at import time; the checkpoint loads only when `CrossEncoderModel.fit` runs
(and the tests inject a tiny local model, offline).
"""

from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

from . import config

CHECKPOINT = config.EMB_FALLBACK            # sentence-transformers/all-MiniLM-L6-v2
MAX_LEN = 320
WINDOW_TOKENS = 224
PCW_TOKENS = 64
LR = 3e-5
EPOCHS = 2
BATCH = 32
WARMUP = 0.06
NEG_RATE = config.M6_NEG_RATE
_ROLE_TAG = {p.rstrip("_"): config.ROLE_TAGS[r] for r, p in config.ROLE_PREFIX.items()}
_PREFIX_RE = re.compile(r"^(p|s|m|u|c)_(.+)$")


def tagged_from_prefixed(text: str | None) -> str:
    """Role-prefixed tokens ("p_what p_now s_i") -> tagged text ("<P> what now <S> i"), the
    §5.3 X_all60 form: a role tag at every speaker change."""
    out, cur = [], None
    for tok in str(text or "").split():
        m = _PREFIX_RE.match(tok)
        role, word = (m.group(1), m.group(2)) if m else (cur, tok)
        if role != cur and role is not None:
            out.append(_ROLE_TAG[role])
            cur = role
        out.append(word)
    return " ".join(out)


def header(area, seg_type, minute, since_me_s) -> str:
    m = "na" if minute is None or not np.isfinite(minute) else str(int(minute))
    if since_me_s is None or not np.isfinite(since_me_s):
        k = "never"
    else:
        k = f"{int(max(since_me_s, 0) // 60)}m"
    return f"[{area or 'other'}] [{seg_type or 'other'}] [min {m}] [since me {k}]"


def format_input(tokenizer, head: str, window: str, pcw: str | None = None, *,
                 window_tokens: int = WINDOW_TOKENS, pcw_tokens: int = PCW_TOKENS,
                 max_len: int = MAX_LEN) -> list[int]:
    """Token ids for one point: the window is LEFT-truncated (most recent words kept), the PCW
    chunk right-truncated, and the whole sequence capped at `max_len` with special tokens."""
    sep = " || "
    h = tokenizer.encode(head + sep, add_special_tokens=False)
    p = (tokenizer.encode(sep + "PCW:", add_special_tokens=False)
         + tokenizer.encode(pcw, add_special_tokens=False)[:pcw_tokens]) if pcw else []
    budget = max(0, min(window_tokens, max_len - 2 - len(h) - len(p)))
    w = tokenizer.encode(window, add_special_tokens=False) if window else []
    w = w[len(w) - budget:] if budget else []
    return _with_specials(tokenizer, (h + w + p)[: max_len - 2])


def _with_specials(tokenizer, ids: list[int]) -> list[int]:
    """[CLS] ids [SEP] (BOS/EOS for tokenizers without CLS/SEP)."""
    cls = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else \
        tokenizer.bos_token_id
    sep = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else \
        tokenizer.eos_token_id
    return ([cls] if cls is not None else []) + list(ids) + ([sep] if sep is not None else [])


def texts_from_design(X: pd.DataFrame, *, window_col: str = "txt_all60",
                      pcw_col: str = "txt_pcw_best") -> pd.DataFrame:
    """(head, window, pcw) strings per design row. Names are never present: the window is the
    normalised text with [NAME] tokens and CALLON phrases already deleted (§5.3)."""
    def col(c):
        return X[c] if c in X else pd.Series([None] * len(X), index=X.index)
    heads = [header(a, s, m, k) for a, s, m, k in zip(
        col("ctx_area"), col("ctx_seg_type"), pd.to_numeric(col("ctx_min"), errors="coerce"),
        pd.to_numeric(col("me_since_turn_s"), errors="coerce"))]
    return pd.DataFrame({"head": heads,
                         "window": [tagged_from_prefixed(t) for t in col(window_col)],
                         "pcw": col(pcw_col).astype(object).where(col(pcw_col).notna(), None)
                         .tolist()}, index=X.index)


def subsample_negatives(y, rate: float = NEG_RATE, seed: int = config.SEED) -> np.ndarray:
    """Indices of every positive plus a seeded `rate` share of negatives."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    neg = np.flatnonzero(y == 0)
    keep = rng.random(len(neg)) < rate
    return np.sort(np.concatenate([np.flatnonzero(y == 1), neg[keep]]))


def prior_correct(logits, rate: float = NEG_RATE) -> np.ndarray:
    """Undo negative subsampling: logit += log r (§6)."""
    return np.asarray(logits, float) + math.log(rate)


class CrossEncoderModel:
    """sklearn-like wrapper: fit(X, y) on design rows, predict_proba(X) with prior correction.

    `model_factory()` -> (tokenizer, model) lets tests inject a tiny offline model; by default
    the checkpoint is loaded with transformers (network only if it is not cached)."""

    def __init__(self, checkpoint: str = CHECKPOINT, *, neg_rate: float = NEG_RATE,
                 epochs: int = EPOCHS, batch_size: int = BATCH, lr: float = LR,
                 warmup: float = WARMUP, max_len: int = MAX_LEN, device: str | None = None,
                 seed: int = config.SEED, model_factory=None):
        self.checkpoint, self.neg_rate, self.epochs, self.batch_size = checkpoint, neg_rate, \
            epochs, batch_size
        self.lr, self.warmup, self.max_len, self.device, self.seed = lr, warmup, max_len, \
            device, seed
        self.model_factory = model_factory

    def _load(self):
        if self.model_factory is not None:
            return self.model_factory()
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(self.checkpoint)
        model = AutoModelForSequenceClassification.from_pretrained(self.checkpoint,
                                                                   num_labels=1)
        return tok, model

    def _device(self):
        import torch
        if self.device == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _encode(self, X: pd.DataFrame) -> list[list[int]]:
        t = texts_from_design(X)
        return [format_input(self.tok_, h, w, p, max_len=self.max_len)
                for h, w, p in zip(t["head"], t["window"], t["pcw"])]

    def _batches(self, ids: list[list[int]], order):
        import torch
        pad = self.tok_.pad_token_id or 0
        for a in range(0, len(order), self.batch_size):
            idx = order[a:a + self.batch_size]
            L = max(len(ids[i]) for i in idx)
            inp = torch.full((len(idx), L), pad, dtype=torch.long)
            att = torch.zeros((len(idx), L), dtype=torch.long)
            for j, i in enumerate(idx):
                inp[j, :len(ids[i])] = torch.as_tensor(ids[i])
                att[j, :len(ids[i])] = 1
            yield idx, inp, att

    def fit(self, X: pd.DataFrame, y) -> CrossEncoderModel:
        import torch
        from transformers import get_linear_schedule_with_warmup

        torch.manual_seed(self.seed)
        y = np.asarray(y, float)
        keep = subsample_negatives(y, self.neg_rate, self.seed)
        self.tok_, self.model_ = self._load()
        dev = self._device()
        self.model_.to(dev)
        ids = self._encode(X.iloc[keep])
        yk = torch.as_tensor(y[keep], dtype=torch.float32)
        steps = max(1, math.ceil(len(keep) / self.batch_size) * self.epochs)
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr)
        sched = get_linear_schedule_with_warmup(opt, int(self.warmup * steps), steps)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        rng = np.random.default_rng(self.seed)
        self.history_ = []
        for epoch in range(self.epochs):
            self.model_.train()
            tot = 0.0
            for idx, inp, att in self._batches(ids, rng.permutation(len(ids))):
                out = self.model_(input_ids=inp.to(dev), attention_mask=att.to(dev))
                loss = loss_fn(out.logits.squeeze(-1), yk[idx].to(dev))
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                tot += float(loss.detach()) * len(idx)
            self.history_.append({"epoch": epoch + 1, "train_loss": tot / max(len(ids), 1)})
        self.classes_ = np.array([0, 1])
        return self

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        """Prior-corrected logits (full-data odds scale)."""
        import torch

        dev = self._device()
        self.model_.eval()
        ids = self._encode(X)
        out = np.zeros(len(ids))
        with torch.no_grad():
            for idx, inp, att in self._batches(ids, np.arange(len(ids))):
                lg = self.model_(input_ids=inp.to(dev), attention_mask=att.to(dev)).logits
                out[idx] = lg.squeeze(-1).float().cpu().numpy()
        return prior_correct(out, self.neg_rate)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = 1.0 / (1.0 + np.exp(-self.decision_function(X)))
        return np.column_stack([1 - p, p])
