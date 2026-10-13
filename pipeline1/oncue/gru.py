"""M5: streaming GRU hazard (spec §6, §9 L29; A9), with a causal TCN as the allowed alternative.

One sequence per class over **every** decision-grid step: no compaction, no zero-filling of
excluded steps, and no mask or at-risk channel. Each step's input is built only from causal
inputs available at t (the same cut-offs as M3):

* PCA-64 of the mean embedding e_f of frames closing in (t - 2, t] (+ a has-frame flag);
* seconds of speech per role in (t - 2, t - ε_w], from word onsets that are perceptible
  (start <= t - ε_w), each word's role read only once `role_avail_s <= t` (else `unk`);
* `q_open` and a one-hot `q_kind`;
* the ctx/me (and, unless the feature set drops it, me_hand) scalars of the features table.

Causal and label-side exclusions enter **only the loss mask** (BCE on at-risk steps with a
defined label). `tests/test_gru_inputs.py` checks this instead of assuming it.

The network is a 1-layer `nn.GRU(in, 64)` (dropout 0.3 on its outputs, linear head) or a
causal dilated TCN, both held to 10-50k parameters (A9). Training is truncated BPTT over
300-step (10-min) chunks with the hidden state carried across chunks, AdamW (lr 1e-3, wd 1e-2),
at most 30 epochs, early stopped on grouped inner-fold AP; Platt calibration uses the inner
out-of-fold scores, as for every other model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from . import config, models

PCA_DIM = 64
STEP_WINDOW_S = config.DELTA_S
SPEECH_ROLES = config.ROLES                      # prof, me, student, unk
Q_KIND_LEVELS = config.Q_KINDS + ("none",)
SCALAR_GROUPS = frozenset({"ctx", "me", "me_hand"})
HAZARD_KINDS = ("gru", "tcn")
TCN_KERNEL = 3
TCN_DILATIONS = (1, 2, 4, 8)                     # ~1-min receptive field at 2-s steps


# --------------------------------------------------------------------------- causal step inputs


@dataclass
class StepRaw:
    """Raw causal inputs of one class on its full grid (before in-fold PCA and scaling).
    Built without any label, mask or exclusion column."""

    class_id: int
    t_s: np.ndarray
    emb_mean: np.ndarray        # (n, d) mean e_f of frames closing in (t - 2, t]; NaN if none
    has_frame: np.ndarray       # (n,) float32 0/1
    speech: np.ndarray          # (n, 4) seconds of speech per role in (t - 2, t - ε_w]
    scalars: pd.DataFrame       # (n, k) ctx/me/me_hand numeric columns + q_open, q_kind

    def __len__(self) -> int:
        return len(self.t_s)


def check_grid(t_s: np.ndarray, step_s: float = config.DELTA_S) -> None:
    """Every grid step must be present: missing steps would be compaction (§6 M5)."""
    t = np.asarray(t_s, float)
    if len(t) > 1 and not np.allclose(np.diff(t), step_s):
        raise ValueError("M5 needs every decision-grid step of the class (stride "
                         f"{step_s} s); got gaps or duplicates (compaction is forbidden)")


def frame_means(t_s, close_s, frame_emb, window_s: float = STEP_WINDOW_S
                ) -> tuple[np.ndarray, np.ndarray]:
    """Mean frame embedding over frames with close_s in (t - window, t], per step."""
    t = np.asarray(t_s, float)
    close = np.asarray(close_s, float)
    emb = np.asarray(frame_emb, np.float64)
    order = np.argsort(close, kind="stable")
    close, emb = close[order], emb[order]
    emb = emb.reshape(len(close), -1)
    cs = np.vstack([np.zeros((1, emb.shape[1])), np.cumsum(emb, axis=0)])
    lo = np.searchsorted(close, t - window_s, side="right")
    hi = np.searchsorted(close, t, side="right")
    n = (hi - lo).astype(float)
    out = np.full((len(t), cs.shape[1]), np.nan)
    ok = n > 0
    out[ok] = (cs[hi[ok]] - cs[lo[ok]]) / n[ok, None]
    return out.astype(np.float32), ok.astype(np.float32)


def speech_seconds(t_s, words: pd.DataFrame, *, window_s: float = STEP_WINDOW_S,
                   eps_w: float = config.EPS_W_S) -> np.ndarray:
    """Seconds of perceptible speech per role in (t - window, t - ε_w] (n x 4).

    A word counts from its start once start <= t - ε_w, up to min(end, t - ε_w). Its role is
    used only once role_avail_s <= t; before that the time counts as `unk`."""
    t = np.asarray(t_s, float)
    out = np.zeros((len(t), len(SPEECH_ROLES)), np.float32)
    if words is None or len(words) == 0:
        return out
    w = words.sort_values("start_s", kind="stable")
    start = w["start_s"].to_numpy(float)
    end = w["end_s"].to_numpy(float)
    avail = (w["role_avail_s"] if "role_avail_s" in w else w["end_s"] + eps_w).to_numpy(float)
    role_idx = w["role"].map({r: i for i, r in enumerate(SPEECH_ROLES)}).fillna(
        SPEECH_ROLES.index(config.ROLE_UNK)).to_numpy(int)
    max_dur = float(np.nanmax(end - start)) if len(start) else 0.0
    unk = SPEECH_ROLES.index(config.ROLE_UNK)
    for i, ti in enumerate(t):
        cut = ti - eps_w
        a = np.searchsorted(start, ti - window_s - max_dur, side="left")
        b = np.searchsorted(start, cut, side="right")
        if b <= a:
            continue
        s, e = start[a:b], np.minimum(end[a:b], cut)
        ov = np.clip(e - np.maximum(s, ti - window_s), 0.0, None)
        roles = np.where(avail[a:b] <= ti, role_idx[a:b], unk)
        np.add.at(out[i], roles, ov.astype(np.float32))
    return out


def scalar_columns(features: pd.DataFrame, featureset: str = "F-all",
                   branch: str = models.PCW_BRANCH) -> list[str]:
    """ctx/me(/me_hand) numeric scalars of `featureset`, plus q_open (and q_kind)."""
    num = models.select_columns(features, featureset, "num", branch)
    cols = [c for c in num if config.feature_group(c) in SCALAR_GROUPS]
    groups = models.featureset_groups(featureset, branch)
    if "q" in groups:
        cols += [c for c in ("q_open", "q_kind") if c in features]
    return cols


def raw_steps(class_id, t_s, *, features: pd.DataFrame, frames: pd.DataFrame | None = None,
              frame_emb: np.ndarray | None = None, words: pd.DataFrame | None = None,
              featureset: str = "F-all", branch: str = models.PCW_BRANCH,
              emb_dim: int | None = None) -> StepRaw:
    """Causal inputs of one class on its full grid.

    `features` must hold one row per grid step (sorted by t_s, same t_s as `t_s`); `frames`
    needs close_s, with `frame_emb` aligned row-for-row; `words` needs start_s, end_s, role
    and role_avail_s. Nothing here reads labels, `in_G`, `at_risk` or `excl_reason`.
    """
    t = np.asarray(t_s, float)
    check_grid(t)
    f = features.sort_values("t_s", kind="stable").reset_index(drop=True)
    if len(f) != len(t) or not np.allclose(f["t_s"].to_numpy(float), t):
        raise ValueError("features must have exactly one row per grid step")
    if frames is not None and frame_emb is not None and len(frames):
        emb_mean, has = frame_means(t, frames["close_s"], frame_emb)
    else:
        d = emb_dim or (frame_emb.shape[1] if frame_emb is not None else config.EMB_DIM)
        emb_mean, has = np.full((len(t), d), np.nan, np.float32), np.zeros(len(t), np.float32)
    return StepRaw(class_id=class_id, t_s=t, emb_mean=emb_mean, has_frame=has,
                   speech=speech_seconds(t, words),
                   scalars=f[scalar_columns(f, featureset, branch)].copy())


def loss_targets(points: pd.DataFrame, target: str = config.HEADLINE_LABEL
                 ) -> tuple[np.ndarray, np.ndarray]:
    """(y, mask) for one class's grid rows (sorted by t_s). This is the only place labels and
    exclusions enter M5: as BCE targets and the loss mask, never as inputs."""
    p = points.sort_values("t_s", kind="stable")
    y = p[target].to_numpy(float)
    mask = np.isfinite(y)
    if "in_G" in p and not target.startswith("y_speech"):
        mask &= p["in_G"].fillna(False).to_numpy(bool)
    if "excl_reason" in p:
        mask &= (p["excl_reason"].astype(object) != "class_qa").to_numpy(bool)
    return np.nan_to_num(y, nan=0.0).astype(np.float32), mask


class StepEncoder:
    """In-fold encoder of raw steps: PCA-64 of frame means, median-impute + indicator +
    standardise the scalars (durations log1p'd), one-hot q_kind. Fit on training classes'
    raw inputs only; it never sees a label."""

    def __init__(self, pca_dim: int = PCA_DIM):
        self.pca_dim = pca_dim

    def fit(self, raws: list[StepRaw]) -> StepEncoder:
        emb = np.vstack([r.emb_mean[r.has_frame > 0] for r in raws]) if raws else None
        k = 0 if emb is None or len(emb) < 2 else int(min(self.pca_dim, len(emb) - 1,
                                                          emb.shape[1]))
        self.pca_ = PCA(n_components=k, random_state=config.SEED).fit(emb) if k else None
        self.k_ = k
        sc = pd.concat([r.scalars for r in raws], ignore_index=True)
        self.num_cols_ = [c for c in sc.columns if c != "q_kind"]
        self.has_q_kind_ = "q_kind" in sc.columns
        self.num_ = Pipeline([
            ("float", models.ToFloat(log_durations=True)),
            ("impute", SimpleImputer(strategy="median", add_indicator=True,
                                     keep_empty_features=True)),
            ("scale", StandardScaler())]).fit(sc[self.num_cols_]) if self.num_cols_ else None
        return self

    def transform(self, raw: StepRaw) -> np.ndarray:
        n = len(raw)
        parts = []
        if self.pca_ is not None:
            z = np.zeros((n, self.k_), np.float32)
            ok = raw.has_frame > 0
            if ok.any():
                z[ok] = self.pca_.transform(raw.emb_mean[ok]).astype(np.float32)
            parts.append(z)
        parts += [raw.has_frame[:, None], raw.speech / np.float32(STEP_WINDOW_S)]
        if self.num_ is not None:
            parts.append(np.asarray(self.num_.transform(raw.scalars[self.num_cols_]),
                                    np.float32))
        if self.has_q_kind_:
            kind = raw.scalars["q_kind"].astype(object).where(raw.scalars["q_kind"].notna(),
                                                               "none")
            parts.append(np.stack([(kind == k).to_numpy(np.float32) for k in Q_KIND_LEVELS],
                                  axis=1))
        return np.concatenate([np.asarray(p, np.float32).reshape(n, -1) for p in parts],
                              axis=1).astype(np.float32)

    @property
    def in_dim(self) -> int | None:
        return getattr(self, "in_dim_", None)


def build_inputs(raw: StepRaw, encoder: StepEncoder) -> np.ndarray:
    """(n_steps, in_dim) float32 input tensor of one class: every grid step, unmasked."""
    return encoder.transform(raw)


# --------------------------------------------------------------------------- networks


class HazardGRU(nn.Module):
    """nn.GRU(in, hidden) -> dropout -> Linear(hidden, 1); unidirectional, so causal."""

    def __init__(self, in_dim: int, hidden: int = config.GRU_HIDDEN, dropout: float = 0.3):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, h: torch.Tensor | None = None):
        out, h = self.gru(x, h)
        return self.head(self.drop(out)).squeeze(-1), h


class _CausalConv(nn.Module):
    def __init__(self, c_in: int, c_out: int, dilation: int, dropout: float):
        super().__init__()
        self.pad = (TCN_KERNEL - 1) * dilation
        self.conv = nn.Conv1d(c_in, c_out, TCN_KERNEL, dilation=dilation)
        self.drop = nn.Dropout(dropout)
        self.res = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):                          # (B, C, T): left padding only
        y = self.conv(nn.functional.pad(x, (self.pad, 0)))
        return torch.relu(self.drop(torch.relu(y)) + self.res(x))


class CausalTCN(nn.Module):
    """Dilated causal convolutions (kernel 3, dilations 1-8): output t sees inputs <= t."""

    def __init__(self, in_dim: int, channels: int = 32, dropout: float = 0.3):
        super().__init__()
        layers, c = [], in_dim
        for d in TCN_DILATIONS:
            layers.append(_CausalConv(c, channels, d, dropout))
            c = channels
        self.blocks = nn.Sequential(*layers)
        self.head = nn.Conv1d(channels, 1, 1)

    def forward(self, x: torch.Tensor, h=None):   # x: (B, T, C)
        y = self.head(self.blocks(x.transpose(1, 2))).squeeze(1)
        return y, None


def n_params(net: nn.Module) -> int:
    return int(sum(p.numel() for p in net.parameters()))


def choose_size(in_dim: int, kind: str = "gru", lo_hi=config.GRU_PARAM_RANGE) -> int:
    """Largest hidden size (GRU, <= 64) or channel count (TCN, <= 64) whose network stays
    inside the A9 parameter range."""
    lo, hi = lo_hi
    for size in range(config.GRU_HIDDEN, 3, -1):
        n = n_params(make_net(in_dim, kind, size))
        if n <= hi:
            if n < lo:
                raise ValueError(f"{kind} with in_dim={in_dim} has {n} < {lo} parameters")
            return size
    raise ValueError(f"no {kind} size fits {lo_hi} parameters for in_dim={in_dim}")


def make_net(in_dim: int, kind: str = "gru", size: int | None = None,
             dropout: float = 0.3) -> nn.Module:
    if kind == "gru":
        return HazardGRU(in_dim, size or config.GRU_HIDDEN, dropout)
    if kind == "tcn":
        return CausalTCN(in_dim, size or 32, dropout)
    raise ValueError(f"kind must be one of {HAZARD_KINDS}")


@torch.no_grad()
def stream_logits(net: nn.Module, x: np.ndarray, chunk: int | None = None) -> np.ndarray:
    """Logits for every step of one sequence. GRU chunks carry the hidden state, so chunked
    and whole-sequence inference agree; the TCN always reads the whole sequence."""
    net.eval()
    dev = next(net.parameters()).device
    xt = torch.as_tensor(np.asarray(x, np.float32), device=dev)[None]
    if chunk is None or isinstance(net, CausalTCN):
        return net(xt)[0][0].float().cpu().numpy()
    out, h = [], None
    for a in range(0, xt.shape[1], chunk):
        y, h = net(xt[:, a:a + chunk], h)
        out.append(y[0])
    return torch.cat(out).float().cpu().numpy()


# --------------------------------------------------------------------------- training


def _device(device: str | None) -> torch.device:
    if device == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _pad_batch(xs, ys, ms):
    T = max(len(x) for x in xs)
    d = xs[0].shape[1]
    X = np.zeros((len(xs), T, d), np.float32)
    Y = np.zeros((len(xs), T), np.float32)
    M = np.zeros((len(xs), T), np.float32)
    for i, (x, y, m) in enumerate(zip(xs, ys, ms)):
        X[i, :len(x)], Y[i, :len(x)], M[i, :len(x)] = x, y, m
    return X, Y, M


def _ap_masked(logit_list, ys, masks) -> float:
    y = np.concatenate([yy[m] for yy, m in zip(ys, masks)]) if ys else np.array([])
    s = np.concatenate([lg[m] for lg, m in zip(logit_list, masks)]) if ys else np.array([])
    return float(average_precision_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


@dataclass
class TrainResult:
    net: nn.Module
    best_epoch: int
    history: list = field(default_factory=list)   # per epoch: train_loss, val_loss, val_ap


def train_net(xs, ys, masks, *, kind: str = "gru", size: int | None = None,
              epochs: int = config.GRU_MAX_EPOCHS, chunk: int = config.GRU_CHUNK_STEPS,
              lr: float = 1e-3, weight_decay: float = 1e-2, dropout: float = 0.3,
              batch_classes: int = 8, val: tuple | None = None, patience: int = 5,
              seed: int = config.SEED, device: str | None = None) -> TrainResult:
    """Train one network on class sequences. Loss = BCE over steps with mask = 1 only (no
    pos_weight, no resampling). With `val=(xs, ys, masks)`, keep the epoch with the best
    validation AP (early stopping); otherwise train for `epochs`."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = _device(device)
    in_dim = xs[0].shape[1]
    net = make_net(in_dim, kind, size or choose_size(in_dim, kind), dropout).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    bce = nn.BCEWithLogitsLoss(reduction="sum")
    best = (-np.inf, 0, {k: v.detach().clone() for k, v in net.state_dict().items()})
    history, since_best = [], 0
    for epoch in range(1, epochs + 1):
        net.train()
        order = rng.permutation(len(xs))
        tot, n_m = 0.0, 0.0
        for b in range(0, len(order), batch_classes):
            idx = order[b:b + batch_classes]
            X, Y, M = (torch.as_tensor(a, device=dev) for a in
                       _pad_batch([xs[i] for i in idx], [ys[i] for i in idx],
                                  [masks[i].astype(np.float32) for i in idx]))
            if kind == "tcn":
                spans = [(0, X.shape[1])]
            else:
                spans = [(a, min(a + chunk, X.shape[1])) for a in range(0, X.shape[1], chunk)]
            h = None
            for a, z in spans:
                logits, h = net(X[:, a:z], h)
                if h is not None:
                    h = h.detach()                     # truncated BPTT
                m = M[:, a:z]
                if m.sum() == 0:
                    continue
                loss = bce(logits[m > 0], Y[:, a:z][m > 0])
                opt.zero_grad()
                (loss / m.sum()).backward()
                opt.step()
                tot += float(loss.detach())
                n_m += float(m.sum())
        rec = {"epoch": epoch, "train_loss": tot / max(n_m, 1.0)}
        if val is not None:
            lg = [stream_logits(net, x, chunk) for x in val[0]]
            rec["val_ap"] = _ap_masked(lg, val[1], val[2])
            yv = np.concatenate([yy[m] for yy, m in zip(val[1], val[2])])
            sv = np.concatenate([g[m] for g, m in zip(lg, val[2])])
            rec["val_loss"] = float(np.mean(np.logaddexp(0, sv) - yv * sv)) if len(yv) \
                else float("nan")
            score = rec["val_ap"] if np.isfinite(rec["val_ap"]) else -rec["val_loss"]
            if score > best[0]:
                best = (score, epoch, {k: v.detach().clone() for k, v in
                                       net.state_dict().items()})
                since_best = 0
            else:
                since_best += 1
        history.append(rec)
        if val is not None and since_best >= patience:
            break
    if val is not None:
        net.load_state_dict(best[2])
        return TrainResult(net, best[1], history)
    return TrainResult(net, epochs, history)


class GRUHazard:
    """M5 for one outer fold: encoder fit on training classes, grouped inner folds for early
    stopping and Platt calibration (inner OOF kept), then a refit on every training class for
    the median best epoch count."""

    def __init__(self, kind: str = "gru", featureset: str = "F-all",
                 max_epochs: int = config.GRU_MAX_EPOCHS, chunk: int = config.GRU_CHUNK_STEPS,
                 n_inner: int = config.N_INNER_FOLDS, patience: int = 5,
                 batch_classes: int = 8, lr: float = 1e-3, weight_decay: float = 1e-2,
                 dropout: float = 0.3, device: str | None = None, seed: int = config.SEED):
        if kind not in HAZARD_KINDS:
            raise ValueError(f"kind must be one of {HAZARD_KINDS}")
        self.kind, self.featureset, self.max_epochs, self.chunk = kind, featureset, \
            max_epochs, chunk
        self.n_inner, self.patience, self.batch_classes = n_inner, patience, batch_classes
        self.lr, self.weight_decay, self.dropout = lr, weight_decay, dropout
        self.device, self.seed = device, seed

    def _train(self, xs, ys, ms, val=None, epochs=None) -> TrainResult:
        return train_net(xs, ys, ms, kind=self.kind, epochs=epochs or self.max_epochs,
                         chunk=self.chunk, lr=self.lr, weight_decay=self.weight_decay,
                         dropout=self.dropout, batch_classes=self.batch_classes, val=val,
                         patience=self.patience, seed=self.seed, device=self.device)

    def fit(self, raws: list[StepRaw], ys: list[np.ndarray], masks: list[np.ndarray]
            ) -> GRUHazard:
        self.encoder_ = StepEncoder().fit(raws)
        xs = [build_inputs(r, self.encoder_) for r in raws]
        self.in_dim_ = xs[0].shape[1]
        self.inner_scores_ = [np.full(len(x), np.nan, np.float32) for x in xs]
        self.history_, best_epochs = [], []
        if self.n_inner and len(raws) >= 2:
            k = min(self.n_inner, len(raws))
            groups = np.arange(len(raws))
            for fold, (tr, va) in enumerate(GroupKFold(k).split(groups, groups=groups)):
                res = self._train([xs[i] for i in tr], [ys[i] for i in tr],
                                  [masks[i] for i in tr],
                                  val=([xs[i] for i in va], [ys[i] for i in va],
                                       [masks[i] for i in va]))
                best_epochs.append(res.best_epoch)
                self.history_.append({"inner_fold": fold, "history": res.history})
                for i in va:
                    self.inner_scores_[i] = stream_logits(res.net, xs[i], self.chunk)
            s = np.concatenate([sc[m] for sc, m in zip(self.inner_scores_, masks)])
            y = np.concatenate([yy[m] for yy, m in zip(ys, masks)])
            self.platt_ = models.Platt.fit(s, y)
            self.n_epochs_ = max(1, int(np.median(best_epochs)))
        else:
            self.platt_, self.n_epochs_ = None, self.max_epochs
        self.net_ = self._train(xs, ys, masks, epochs=self.n_epochs_).net
        self.n_params_ = n_params(self.net_)
        return self

    def predict(self, raws: list[StepRaw]) -> list[tuple[np.ndarray, np.ndarray]]:
        """[(p_raw, p_cal)] per class, for every grid step."""
        out = []
        for r in raws:
            s = stream_logits(self.net_, build_inputs(r, self.encoder_), self.chunk)
            p_raw = models._sigmoid(s)
            out.append((p_raw, self.platt_(s) if self.platt_ is not None else p_raw))
        return out

    def inner_predictions(self) -> list[tuple[np.ndarray, np.ndarray]] | None:
        if self.platt_ is None:
            return None
        return [(models._sigmoid(s), self.platt_(s)) for s in self.inner_scores_]
