"""Part 3 of the notebook: the flexible models, stacked one step at a time.

Brady's pipeline (design/model_pipeline_proposal.md): gradient-boosted trees make the final
prediction; each specialist (content-alignment MLP, then a TCN, then a fine-tuned transformer)
adds its score to the trees' inputs, and each step is measured on validation. Built so far:

* 3.1 `GBT`: HistGradientBoosting on the structured inputs (`gbt_inputs`).
* 3.2 `MLP`: a small content-alignment network on frozen bge vectors. It compares what is
  being said now (the professor's latest words, the last 60 s) with the class's material, my
  pre-class work (as of class start) and my own earlier words. `GBT+MLP`: the trees with the
  MLP's score and the plain cosine similarities as extra inputs.

Protocol (Step 2's, plus out-of-fold scores): fit on train; every setting (tree size, number
of trees, MLP epochs) chosen on validation AP; train scores out-of-fold (OOF) from 4 folds
grouped by class, so the trees never learn from a specialist's in-sample score and the guess
threshold (`baselines.tau_like_truth`) is set on honest train scores. Validation scores are
the mean of the SAME 4 fold models, so train and validation scores come from the same models
and share one scale (a fifth model on all train classes only picks the number of trees /
epochs; its validation scores are kept as `val_full`). Test is not scored here: it is scored
once, for the final model.

Causal: every input is cut at t. The pooled vectors are the features' own (frames closed by
t); the class material and my pre-class work are as of class start; my earlier words are my
frames from earlier classes (by scheduled start) plus this class's with close_s <= t.

`python -m oncue flex [--step gbt|mlp|stack|tcn|stack_tcn|combiners|importance|calibrate|
bootstrap|learning|all]` writes data/results/flex/ (local, gitignored).
"""

from __future__ import annotations

import copy
import pickle
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold

from . import config

LABEL = "y_H15_g2"
KEYS = ("class_id", "t_s")
EXCLUDED = {"has_qmark": "punctuation may look ahead (F-punct only)",
            "spk_cur_in_my_group": "breakout groups not parsed (always missing)",
            "ctx_n_present": "presence events not parsed (Brady: ignore)"}
_SPLINE = re.compile(r"ctx_min_spl_\d+$")          # smooth copies of ctx_min: trees need none
N_FOLDS = 4
N_SEEDS = 5                                        # training runs averaged per fold (MLP, TCN)
DIM = config.EMB_DIM
POOLED = {"v15": 0, "v60": 1, "last": 2, "lastprof": 3}   # blocks of features_emb.npy

GBT_BASE = {"learning_rate": 0.05, "l2_regularization": 1.0, "max_features": 0.8,
            "categorical_features": "from_dtype", "early_stopping": False,
            "random_state": config.SEED}
GBT_GRID = ({"max_leaf_nodes": 15, "min_samples_leaf": 200},
            {"max_leaf_nodes": 31, "min_samples_leaf": 200},
            {"max_leaf_nodes": 63, "min_samples_leaf": 400})
GBT_MAX_TREES = 600
STAGE_EVERY = 20

COSINES = ("cos_q_material", "cos_q_pcw", "cos_q_mine", "cos_60_material", "cos_60_pcw",
           "cos_60_mine", "cos_material_mine")
MLP_SETTINGS = {"d": 32, "hidden": 64, "dropout": 0.2, "lr": 1e-3, "weight_decay": 1e-4,
                "batch": 4096, "max_epochs": 20}


def out_dir() -> Path:
    return config.DATA_DIR / "results" / "flex"


# --------------------------------------------------------------------------- data


def gbt_inputs(columns) -> list[str]:
    return [c for c in columns if c not in (*KEYS, "emb_row", "split", "y") and c not in EXCLUDED
            and not _SPLINE.match(c)]


def load(tables: Path | None = None, split_path: Path | None = None) -> pd.DataFrame:
    """Counting moments (the rows of `baselines.load`, same order) with y, split, emb_row and
    the trees' inputs; text columns become pandas categories."""
    tables = Path(tables or config.TABLES_DIR)
    split_path = Path(split_path or config.DATA_DIR / "03_split" / "split.csv")
    pts = pd.read_parquet(tables / "points.parquet", columns=[*KEYS, "at_risk", LABEL])
    pts = pts[pts["at_risk"].astype(bool) & pts[LABEL].notna()]
    feat = pd.read_parquet(tables / "features.parquet")
    split = pd.read_csv(split_path, usecols=["class_id", "split"])
    df = (pts.merge(feat, on=list(KEYS), how="left").merge(split, on="class_id", how="left"))
    df["y"] = df[LABEL].astype(int)
    df = df.drop(columns=["at_risk", LABEL]).sort_values(list(KEYS)).reset_index(drop=True)
    for c in gbt_inputs(df.columns):
        if not (pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c])):
            df[c] = df[c].astype("string").astype("category")
    return df


def fold_labels(class_ids, n_folds: int = N_FOLDS) -> np.ndarray:
    """Grouped folds over the train rows: every class's rows in one fold."""
    g = np.asarray(class_ids)
    out = np.empty(len(g), int)
    for k, (_, idx) in enumerate(GroupKFold(n_splits=n_folds).split(np.zeros(len(g)), groups=g)):
        out[idx] = k
    return out


def val_folds(class_ids, n_folds: int = N_FOLDS) -> np.ndarray:
    """Each validation (or test) session is scored by ONE of the fold models (round-robin over
    the sorted session ids), so its score is made exactly like an out-of-fold train score: by
    a model trained on 3/4 of the train sessions, on a session it never saw. Averaging the 4
    fold models instead narrows the spread of the scores, and a threshold set on the train
    scores then says yes too rarely."""
    g = np.asarray(class_ids)
    lab = {c: i % n_folds for i, c in enumerate(np.sort(np.unique(g)))}
    return np.array([lab[c] for c in g], int)


# --------------------------------------------------------------------------- vectors (MLP inputs)


def _unit(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(np.asarray(x, np.float32))
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return np.where(n > 0, x / np.maximum(n, 1e-8), 0.0).astype(np.float32)


def my_words_so_far(df: pd.DataFrame, items: pd.DataFrame, emb: np.ndarray,
                    start_of: dict) -> tuple[np.ndarray, np.ndarray]:
    """For each row (class c, moment t): the mean vector of my utterances in classes that
    started before c, plus c's own with avail_s <= t; and how many there were. Never reads an
    utterance of c after t or of a class that started at or after c."""
    mine = items[items["kind"] == "my_utterance"]
    by_class = {cid: g.sort_values("avail_s") for cid, g in mine.groupby("class_id")}
    order = sorted(start_of, key=lambda c: (start_of[c], c))
    running, n_run = np.zeros(DIM, np.float64), 0
    prior = {}
    i = 0
    while i < len(order):                                  # classes with the same start share
        j = i
        while j < len(order) and start_of[order[j]] == start_of[order[i]]:
            j += 1
        for c in order[i:j]:
            prior[c] = (running.copy(), n_run)
        for c in order[i:j]:
            g = by_class.get(c)
            if g is not None and len(g):
                running += np.asarray(emb[g["row"].to_numpy()], np.float64).sum(0)
                n_run += len(g)
        i = j
    out = np.zeros((len(df), DIM), np.float32)
    n_out = np.zeros(len(df), np.int32)
    t_all = df["t_s"].to_numpy(float)
    for cid, idx in df.groupby("class_id", sort=False).indices.items():
        s0, n0 = prior.get(cid, (np.zeros(DIM), 0))
        g = by_class.get(cid)
        if g is not None and len(g):
            v = np.asarray(emb[g["row"].to_numpy()], np.float64)
            cums = np.vstack([np.zeros(DIM), np.cumsum(v, 0)])
            k = np.searchsorted(g["avail_s"].to_numpy(float), t_all[idx], side="right")
            s, n = s0 + cums[k], n0 + k
        else:
            s, n = np.broadcast_to(s0, (len(idx), DIM)), np.full(len(idx), n0)
        out[idx] = np.where(np.asarray(n)[:, None] > 0, s / np.maximum(np.asarray(n), 1)[:, None], 0.0)
        n_out[idx] = n
    return _unit(out), n_out


def vector_inputs(df: pd.DataFrame, tables: Path | None = None,
                  vectors_dir: Path | None = None) -> dict:
    """The MLP's inputs for every row of `df`, all L2-normalised (zeros where missing):
    q = the professor's latest words, w = the last 60 s (pooled features), C = the class's
    material (learning outcomes + study guide), A = my pre-class answers as of class start,
    M = my earlier words (`my_words_so_far`); `has` flags and the plain cosines."""
    tables = Path(tables or config.TABLES_DIR)
    vectors_dir = Path(vectors_dir or config.DATA_DIR / "vectors")
    fe = np.load(tables / "features_emb.npy", mmap_mode="r")
    rows = pd.array(df["emb_row"], dtype="Int64")
    ok = ~np.asarray(rows.isna())
    r = rows[ok].to_numpy(int)
    q = np.zeros((len(df), DIM), np.float32)
    w = np.zeros((len(df), DIM), np.float32)
    b = POOLED["lastprof"] * DIM
    q[ok] = _unit(fe[r, b:b + DIM])
    b = POOLED["v60"] * DIM
    w[ok] = _unit(fe[r, b:b + DIM])
    items = pd.read_parquet(vectors_dir / "items.parquet",
                            columns=["class_id", "kind", "avail_s", "row", "sched_start_utc"])
    emb = np.load(vectors_dir / "embeddings.npy", mmap_mode="r")
    classes = sorted(df["class_id"].unique())
    cix = {c: i for i, c in enumerate(classes)}
    C = np.zeros((len(classes), DIM), np.float32)
    A = np.zeros((len(classes), DIM), np.float32)
    for kinds, tab in ((("learning_outcome", "study_guide"), C), (("pcw_answer",), A)):
        sel = items[items["kind"].isin(kinds) & items["class_id"].isin(cix)]
        for cid, g in sel.groupby("class_id"):
            tab[cix[cid]] = _unit(np.asarray(emb[g["row"].to_numpy()], np.float32).mean(0))
    cls = pd.read_parquet(tables / "classes.parquet", columns=["class_id", "sched_start_utc"])
    start = pd.to_datetime(cls.drop_duplicates("class_id").set_index("class_id")["sched_start_utc"],
                           utc=True, errors="coerce")
    start_of = {int(c): (s.value if pd.notna(s) else np.iinfo(np.int64).max)
                for c, s in start.items()}
    for c in classes:
        start_of.setdefault(int(c), np.iinfo(np.int64).max)
    M, n_mine = my_words_so_far(df, items, emb, start_of)
    ci = df["class_id"].map(cix).to_numpy()
    Cr, Ar = C[ci], A[ci]
    has = np.stack([q.any(1), w.any(1), Cr.any(1), Ar.any(1), M.any(1)], 1).astype(np.float32)

    def cos(a, b_, ha, hb):
        v = (a * b_).sum(1)
        return np.where((ha > 0) & (hb > 0), v, np.nan).astype(np.float32)

    cosines = np.stack([cos(q, Cr, has[:, 0], has[:, 2]), cos(q, Ar, has[:, 0], has[:, 3]),
                        cos(q, M, has[:, 0], has[:, 4]), cos(w, Cr, has[:, 1], has[:, 2]),
                        cos(w, Ar, has[:, 1], has[:, 3]), cos(w, M, has[:, 1], has[:, 4]),
                        cos(Cr, M, has[:, 2], has[:, 4])], 1)
    return {"q": q.astype(np.float16), "w": w.astype(np.float16), "M": M.astype(np.float16),
            "C": C, "A": A, "class_index": ci.astype(np.int32), "has": has,
            "cosines": cosines, "n_mine": n_mine}


# --------------------------------------------------------------------------- 3.1 trees


def _hgb(setting: dict, n_trees: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(**GBT_BASE, **setting, max_iter=int(n_trees))


def choose_gbt(Xtr, ytr, Xva, yva, *, sample_train: int = 60_000, log=print):
    """Fit each GBT_GRID setting once with GBT_MAX_TREES trees; the validation AP after every
    STAGE_EVERY trees picks (setting, number of trees). Also returns the learning curve
    (train AP on a fixed sample of train rows, validation AP) for the notebook."""
    rng = np.random.default_rng(config.SEED)
    sub = rng.choice(len(Xtr), min(sample_train, len(Xtr)), replace=False)
    rows, best = [], None
    for k, st in enumerate(GBT_GRID):
        t0 = time.time()
        m = _hgb(st, GBT_MAX_TREES).fit(Xtr, ytr)
        stages_tr = m.staged_predict_proba(Xtr.iloc[sub])
        for n, (p_va, p_tr) in enumerate(zip(m.staged_predict_proba(Xva), stages_tr), 1):
            if n % STAGE_EVERY:
                continue
            ap = average_precision_score(yva, p_va[:, 1])
            rows.append({"setting": k, **st, "trees": n, "val_AP": ap,
                         "train_AP": average_precision_score(np.asarray(ytr)[sub], p_tr[:, 1])})
            if best is None or ap > best[0]:
                best = (ap, k, n)
        log(f"  GBT setting {k} {st}: best val AP so far {best[0]:.4f} ({time.time() - t0:.0f} s)")
    return {"setting": GBT_GRID[best[1]], "setting_index": best[1], "trees": best[2],
            "val_AP": best[0]}, pd.DataFrame(rows)


def fit_gbt_oof(X: pd.DataFrame, y, folds: np.ndarray, setting: dict, n_trees: int,
                X_val: pd.DataFrame | None = None, vfolds: np.ndarray | None = None,
                log=print) -> tuple[np.ndarray, np.ndarray | None]:
    """Out-of-fold train scores, and validation scores from the fold model each validation
    session is assigned to (`val_folds`)."""
    oof = np.zeros(len(X))
    val = np.zeros(len(X_val)) if X_val is not None else None
    for k in range(folds.max() + 1):
        tr, te = folds != k, folds == k
        m = _hgb(setting, n_trees).fit(X[tr], np.asarray(y)[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
        if X_val is not None and (vfolds == k).any():
            val[vfolds == k] = m.predict_proba(X_val[vfolds == k])[:, 1]
        log(f"  OOF fold {k} done")
    return oof, val


def run_gbt(df: pd.DataFrame, extra: pd.DataFrame | None = None, *, name: str = "gbt",
            log=print) -> dict:
    """3.1 (and 3.2's stack, with `extra` columns): choose on validation, OOF on train, final
    fit on all train. Saves data/results/flex/{name}.pkl."""
    cols = gbt_inputs(df.columns)
    X = df[cols] if extra is None else pd.concat([df[cols], extra], axis=1)
    tr, va = (df["split"] == "train").to_numpy(), (df["split"] == "validation").to_numpy()
    ytr, yva = df["y"].to_numpy()[tr], df["y"].to_numpy()[va]
    log(f"{name}: {X.shape[1]} inputs, {tr.sum():,} train / {va.sum():,} validation moments")
    choice, curve = choose_gbt(X[tr], ytr, X[va], yva, log=log)
    log(f"  chosen: {choice}")
    folds = fold_labels(df.loc[tr, "class_id"])
    vf = val_folds(df.loc[va, "class_id"])
    oof, val = fit_gbt_oof(X[tr], ytr, folds, choice["setting"], choice["trees"], X[va], vf, log=log)
    model = _hgb(choice["setting"], choice["trees"]).fit(X[tr], ytr)
    res = {"name": name, "inputs": list(X.columns), "choice": choice, "curve": curve,
           "folds": folds, "oof": oof, "val": val,
           "val_full": model.predict_proba(X[va])[:, 1], "model": model, "excluded": EXCLUDED}
    _save(res, name)
    return res


def importance(df: pd.DataFrame, res: dict, *, n_repeats: int = 3, extra=None, log=print) -> pd.DataFrame:
    """Permutation importance on validation: the drop in validation AP when one input is
    shuffled (mean over `n_repeats`)."""
    from sklearn.inspection import permutation_importance

    va = (df["split"] == "validation").to_numpy()
    X = df[[c for c in res["inputs"] if c in df.columns]]
    if extra is not None:
        X = pd.concat([X, extra], axis=1)
    X = X[res["inputs"]][va]
    pi = permutation_importance(res["model"], X, df["y"].to_numpy()[va],
                                scoring="average_precision", n_repeats=n_repeats,
                                random_state=config.SEED, n_jobs=1)
    out = pd.DataFrame({"input": res["inputs"], "AP_drop": pi.importances_mean,
                        "AP_drop_sd": pi.importances_std}).sort_values("AP_drop", ascending=False)
    res["importance"] = out.reset_index(drop=True)
    _save(res, res["name"])
    log(f"  importance: top {out.head(5)['input'].tolist()}")
    return out


# --------------------------------------------------------------------------- 3.2 MLP


def _torch():
    import torch
    from torch import nn
    return torch, nn


def align_net(settings: dict = MLP_SETTINGS):
    """q, w (now) and C, A, M (material, my pre-class work, my words) are projected by ONE
    shared linear map P (384 -> d); the head sees the element-wise products of the "now"
    projections with each reference projection, plus the cosines and `has` flags."""
    torch, nn = _torch()

    class AlignNet(nn.Module):
        def __init__(self, d=settings["d"], hidden=settings["hidden"],
                     dropout=settings["dropout"], n_extra=len(COSINES) + 5):
            super().__init__()
            self.proj = nn.Linear(DIM, d, bias=False)
            self.head = nn.Sequential(nn.Linear(6 * d + n_extra, hidden), nn.ReLU(),
                                      nn.Dropout(dropout), nn.Linear(hidden, 1))

        def forward(self, q, w, c, a, m, extra):
            hq, hw, hc, ha, hm = (self.proj(x) for x in (q, w, c, a, m))
            z = torch.cat([hq * hc, hq * ha, hq * hm, hw * hc, hw * ha, hw * hm, extra], 1)
            return self.head(z).squeeze(1)

    return AlignNet()


def n_params(net) -> int:
    return int(sum(p.numel() for p in net.parameters()))


class _Batches:
    def __init__(self, V: dict, idx: np.ndarray, y: np.ndarray | None, device):
        torch, _ = _torch()
        self.torch, self.device = torch, device
        self.idx, self.y = idx, y
        self.V = V
        self.C = torch.as_tensor(V["C"], device=device)
        self.A = torch.as_tensor(V["A"], device=device)
        extra = np.concatenate([np.nan_to_num(V["cosines"]), V["has"]], 1).astype(np.float32)
        self.extra = extra

    def __call__(self, rows: np.ndarray):
        t, d = self.torch, self.device
        ci = t.as_tensor(self.V["class_index"][rows].astype(np.int64), device=d)
        f = lambda k: t.as_tensor(np.asarray(self.V[k][rows], np.float32), device=d)
        return (f("q"), f("w"), self.C[ci], self.A[ci], f("M"),
                t.as_tensor(self.extra[rows], device=d))


def _device():
    torch, _ = _torch()
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def predict_mlp(net, V: dict, rows: np.ndarray, batch: int = 16384) -> np.ndarray:
    torch, _ = _torch()
    dev = next(net.parameters()).device
    get = _Batches(V, rows, None, dev)
    net.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(rows), batch):
            out.append(torch.sigmoid(net(*get(rows[s:s + batch]))).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def train_mlp(V: dict, tr_rows: np.ndarray, y_tr: np.ndarray, *, epochs: int,
              va_rows: np.ndarray | None = None, y_va: np.ndarray | None = None,
              settings: dict = MLP_SETTINGS, seed: int = config.SEED, log=print, on_epoch=None):
    """Plain BCE, AdamW, shuffled mini-batches. With validation rows: AP after every epoch
    and the weights of the best epoch are kept. `on_epoch(epoch, net)` is called after every
    epoch. Returns (net, history)."""
    torch, nn = _torch()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = _device()
    net = align_net(settings).to(dev)
    with torch.no_grad():                    # start at the base rate, not at p = 0.5
        rate = float(np.clip(y_tr.mean(), 1e-4, 1 - 1e-4))
        net.head[-1].bias.fill_(float(np.log(rate / (1 - rate))))
    opt = torch.optim.AdamW(net.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()
    get = _Batches(V, tr_rows, y_tr, dev)
    yt = torch.as_tensor(y_tr.astype(np.float32), device=dev)
    hist, best = [], (-1.0, None, 0)
    for ep in range(1, epochs + 1):
        t0 = time.time()
        net.train()
        perm = rng.permutation(len(tr_rows))
        tot = 0.0
        for s in range(0, len(perm), settings["batch"]):
            pos = perm[s:s + settings["batch"]]
            logit = net(*get(tr_rows[pos]))
            loss = loss_fn(logit, yt[torch.as_tensor(pos, device=dev)])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(pos)
        row = {"epoch": ep, "train_loss": tot / len(perm)}
        if va_rows is not None:
            p = predict_mlp(net, V, va_rows)
            row["val_AP"] = average_precision_score(y_va, p)
            eps = 1e-7
            row["val_loss"] = float(-np.mean(y_va * np.log(p + eps) + (1 - y_va) * np.log(1 - p + eps)))
            if row["val_AP"] > best[0]:
                best = (row["val_AP"], copy.deepcopy(net.state_dict()), ep)
        hist.append(row)
        if on_epoch is not None:
            on_epoch(ep, net)
        log(f"  epoch {ep}: train loss {row['train_loss']:.5f}"
            + (f", val AP {row['val_AP']:.4f}" if "val_AP" in row else "") + f" ({time.time() - t0:.0f} s)")
    if best[1] is not None:
        net.load_state_dict(best[1])
    return net, pd.DataFrame(hist), best[2]


def run_mlp(df: pd.DataFrame, V: dict | None = None, *, log=print) -> dict:
    """3.2. Choosing the epoch: N_SEEDS training runs on all train sessions, validation AP
    after every epoch; E = the epoch with the best AP averaged over the runs (one run's curve
    is noisy). Scores: 4 grouped folds; in each, N_SEEDS runs of E epochs, averaged; they score
    the held-out train sessions (out-of-fold) and the validation sessions assigned to that fold
    (`val_folds`). Saves mlp.pkl."""
    V = V if V is not None else vector_inputs(df)
    tr = np.flatnonzero((df["split"] == "train").to_numpy())
    va = np.flatnonzero((df["split"] == "validation").to_numpy())
    y = df["y"].to_numpy()
    quiet = lambda *_: None
    hists = []
    for r in range(N_SEEDS):
        net, h, _ = train_mlp(V, tr, y[tr], epochs=MLP_SETTINGS["max_epochs"], va_rows=va,
                              y_va=y[va], seed=config.SEED + 100 * r, log=quiet)
        hists.append(h.assign(run=r))
        log(f"  run {r}: best validation AP {h['val_AP'].max():.4f} at epoch {int(h.loc[h['val_AP'].idxmax(), 'epoch'])}")
    H = pd.concat(hists, ignore_index=True)
    E = int(H.groupby("epoch")["val_AP"].mean().idxmax())
    log(f"  chosen: {E} epochs (best mean validation AP over {N_SEEDS} runs); {n_params(net):,} parameters")
    folds = fold_labels(df["class_id"].to_numpy()[tr])
    vf = val_folds(df["class_id"].to_numpy()[va])
    oof, val = np.zeros(len(tr)), np.zeros(len(va))
    for k in range(N_FOLDS):
        fit_rows, held, vrows = tr[folds != k], tr[folds == k], va[vf == k]
        for r in range(N_SEEDS):
            nk, _, _ = train_mlp(V, fit_rows, y[fit_rows], epochs=E, log=quiet,
                                 seed=config.SEED + 10 * k + 100 * r + 1)
            oof[folds == k] += predict_mlp(nk, V, held) / N_SEEDS
            val[vf == k] += predict_mlp(nk, V, vrows) / N_SEEDS
        log(f"  fold {k} done ({N_SEEDS} runs)")
    res = {"name": "mlp", "settings": MLP_SETTINGS, "history": H, "best_epoch": E,
           "n_seeds": N_SEEDS, "n_params": n_params(net), "folds": folds, "oof": oof, "val": val,
           "cosines": COSINES}
    _save(res, "mlp")
    return res


def stack_extra(df: pd.DataFrame, V: dict, mlp: dict) -> pd.DataFrame:
    """The trees' extra inputs for GBT+MLP: the MLP score (out-of-fold on train rows, the
    assigned fold models on validation rows; NaN elsewhere) and the plain cosines."""
    score = np.full(len(df), np.nan)
    score[(df["split"] == "train").to_numpy()] = mlp["oof"]
    score[(df["split"] == "validation").to_numpy()] = mlp["val"]
    extra = pd.DataFrame(V["cosines"], columns=list(COSINES), index=df.index)
    extra.insert(0, "mlp_score", score)
    return extra


# --------------------------------------------------------------------------- 3.3 TCN

TCN_SETTINGS = {"channels": 32, "dilations": (1, 2, 4, 8, 16, 32, 64), "dropout": 0.2,
                "lr": 1e-3, "weight_decay": 1e-2, "batch_classes": 8, "max_epochs": 15,
                "patience": 99, "pca_dim": 64, "pca_sample": 150_000}   # every run sees all 15 epochs


def tcn_receptive_field(settings: dict = TCN_SETTINGS) -> int:
    """Steps (2 s each) the TCN's output at t can see: 1 + (kernel - 1) * sum(dilations)."""
    from . import gru

    return 1 + (gru.TCN_KERNEL - 1) * int(sum(settings["dilations"]))


def _categories(df: pd.DataFrame) -> dict:
    return {c: list(df[c].cat.categories) for c in gbt_inputs(df.columns)
            if isinstance(df[c].dtype, pd.CategoricalDtype) and c != "q_kind"}


def tcn_steps(df: pd.DataFrame, tables: Path | None = None, log=print) -> dict:
    """Per class of `df`: (StepRaw, y, mask) on the class's FULL 2-s grid, every step kept.
    Step inputs (gru.py's causal builders): the mean sentence vector of frames closing in
    (t - 2, t], seconds of speech per role in (t - 2, t - eps_w], and the trees' numeric
    inputs + one-hot categories + q_kind at t. `mask` marks the class's counting moments (the
    rows of `df`); labels enter only as loss targets there, never as inputs."""
    from . import gru

    tables = Path(tables or config.TABLES_DIR)
    classes = sorted(df["class_id"].unique())
    feat = pd.read_parquet(tables / "features.parquet")
    feat = feat[feat["class_id"].isin(classes)].sort_values(list(KEYS)).reset_index(drop=True)
    num = [c for c in gbt_inputs(feat.columns)
           if pd.api.types.is_numeric_dtype(feat[c]) or pd.api.types.is_bool_dtype(feat[c])]
    dummies = []
    for c, levels in _categories(df).items():
        v = feat[c].astype("string")
        for lev in levels:
            name = f"{c}=={lev}"
            feat[name] = (v == lev).fillna(False).astype(np.float32)
            dummies.append(name)
    cols = num + dummies + ["q_kind"]
    fr = pd.read_parquet(tables / "frames.parquet", columns=["class_id", "close_s", "emb_row"])
    fr = fr[fr["class_id"].isin(classes) & fr["emb_row"].notna()]
    femb = np.load(tables / "frames_emb.npy", mmap_mode="r")
    words = pd.read_parquet(tables / "words.parquet",
                            columns=["class_id", "start_s", "end_s", "role", "role_avail_s"])
    words = words[words["class_id"].isin(classes)]
    fr_by, w_by = dict(tuple(fr.groupby("class_id"))), dict(tuple(words.groupby("class_id")))
    rows_by = df.groupby("class_id").indices
    out = {}
    t0 = time.time()
    for n, (cid, f) in enumerate(feat.groupby("class_id", sort=True), 1):
        t = f["t_s"].to_numpy(float)
        gru.check_grid(t)
        frc = fr_by.get(cid)
        if frc is not None and len(frc):
            emb_mean, has = gru.frame_means(t, frc["close_s"], np.asarray(
                femb[frc["emb_row"].astype(int).to_numpy()], np.float32))
        else:
            emb_mean, has = np.full((len(t), DIM), np.nan, np.float32), np.zeros(len(t), np.float32)
        raw = gru.StepRaw(class_id=int(cid), t_s=t, emb_mean=emb_mean.astype(np.float16), has_frame=has,
                          speech=gru.speech_seconds(t, w_by.get(cid)),
                          scalars=f[cols].reset_index(drop=True))
        r = rows_by[cid]
        pos = np.searchsorted(t, df["t_s"].to_numpy(float)[r])
        assert np.allclose(t[pos], df["t_s"].to_numpy(float)[r]), cid
        y = np.zeros(len(t), np.float32)
        mask = np.zeros(len(t), bool)
        y[pos], mask[pos] = df["y"].to_numpy(np.float32)[r], True
        out[int(cid)] = (raw, y, mask)
        if n % 100 == 0:
            log(f"  steps: {n}/{len(classes)} classes ({time.time() - t0:.0f} s)")
    return out


class _StepEncoder:
    """gru.StepEncoder with the PCA fitted on a fixed random sample of the training steps'
    sentence vectors (bounded memory); the scalars as there (log1p durations, median impute +
    missing flags, standardise), fitted on training classes only."""

    def __init__(self, pca_dim: int, sample: int):
        from . import gru

        self.inner = gru.StepEncoder(pca_dim)
        self.sample = sample

    def fit(self, raws):
        from sklearn.decomposition import PCA
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        from . import models

        inner = self.inner
        emb = np.vstack([r.emb_mean[r.has_frame > 0] for r in raws]).astype(np.float32)
        rng = np.random.default_rng(config.SEED)
        if len(emb) > self.sample:
            emb = emb[rng.choice(len(emb), self.sample, replace=False)]
        inner.k_ = int(min(inner.pca_dim, len(emb) - 1, emb.shape[1]))
        inner.pca_ = PCA(n_components=inner.k_, random_state=config.SEED).fit(emb)
        sc = pd.concat([r.scalars for r in raws], ignore_index=True)
        inner.num_cols_ = [c for c in sc.columns if c != "q_kind"]
        inner.has_q_kind_ = "q_kind" in sc.columns
        inner.num_ = Pipeline([("float", models.ToFloat(log_durations=True)),
                               ("impute", SimpleImputer(strategy="median", add_indicator=True,
                                                        keep_empty_features=True)),
                               ("scale", StandardScaler())]).fit(sc[inner.num_cols_])
        return self

    def transform(self, raw) -> np.ndarray:
        r = type(raw)(raw.class_id, raw.t_s, raw.emb_mean.astype(np.float32), raw.has_frame,
                      raw.speech, raw.scalars)
        return self.inner.transform(r)


def _tcn_fit(X: list, Y: list, M: list, *, epochs: int, val=None, seed: int = config.SEED,
             settings: dict = TCN_SETTINGS, on_epoch=None):
    from . import gru

    rate = float(np.clip(np.concatenate([y[m] for y, m in zip(Y, M)]).mean(), 1e-4, 1 - 1e-4))
    return gru.train_net(X, Y, M, kind="tcn", size=settings["channels"], epochs=epochs,
                         lr=settings["lr"], weight_decay=settings["weight_decay"],
                         dropout=settings["dropout"], batch_classes=settings["batch_classes"],
                         val=val, patience=settings["patience"], seed=seed, device="mps",
                         dilations=settings["dilations"], init_bias=float(np.log(rate / (1 - rate))),
                         on_epoch=on_epoch)


def _tcn_predict(net, X: list, M: list) -> np.ndarray:
    from . import gru

    out = [1 / (1 + np.exp(-gru.stream_logits(net, x)[m])) for x, m in zip(X, M)]
    return np.concatenate(out) if out else np.zeros(0)


def run_tcn(df: pd.DataFrame, steps: dict | None = None, *, log=print) -> dict:
    """3.3, as the MLP: choosing the epoch from N_SEEDS training runs on all train sessions
    (validation AP after every epoch, best mean over the runs); then 4 grouped folds, each with
    its own encoder fitted on its training sessions and N_SEEDS runs of E epochs, averaged,
    scoring the held-out train sessions and the validation sessions assigned to that fold.
    Saves tcn.pkl."""
    from . import gru

    st = TCN_SETTINGS
    steps = steps or tcn_steps(df, log=log)
    trm = (df["split"] == "train").to_numpy()
    tr_c = sorted(df.loc[trm, "class_id"].unique())
    va_c = sorted(df.loc[df["split"] == "validation", "class_id"].unique())
    get = lambda cs, k: [steps[c][k] for c in cs]
    t0 = time.time()
    enc = _StepEncoder(st["pca_dim"], st["pca_sample"]).fit([steps[c][0] for c in tr_c])
    X = {c: enc.transform(steps[c][0]) for c in tr_c + va_c}
    in_dim = X[tr_c[0]].shape[1]
    log(f"  TCN inputs: {in_dim} numbers per step; receptive field {tcn_receptive_field()} steps "
        f"({tcn_receptive_field() * 2 / 60:.1f} min); encoder {time.time() - t0:.0f} s")
    hists = []
    for r in range(N_SEEDS):
        t0 = time.time()
        rr = _tcn_fit([X[c] for c in tr_c], get(tr_c, 1), get(tr_c, 2), epochs=st["max_epochs"],
                      val=([X[c] for c in va_c], get(va_c, 1), get(va_c, 2)),
                      seed=config.SEED + 100 * r)
        h = pd.DataFrame(rr.history).assign(run=r)
        hists.append(h)
        n_par = gru.n_params(rr.net)
        log(f"  run {r}: best validation AP {h['val_ap'].max():.4f} at epoch "
            f"{int(h.loc[h['val_ap'].idxmax(), 'epoch'])} ({time.time() - t0:.0f} s)")
    del X
    H = pd.concat(hists, ignore_index=True)
    E = int(H.groupby("epoch")["val_ap"].mean().idxmax())
    log(f"  chosen: {E} epochs (best mean validation AP over {N_SEEDS} runs); {n_par:,} parameters")
    folds = fold_labels(df.loc[trm, "class_id"].to_numpy())
    fold_of = dict(zip(df.loc[trm, "class_id"].to_numpy(), folds))
    vfold_of = dict(zip(va_c, val_folds(va_c)))
    by = {}
    for k in range(N_FOLDS):
        fit_c = [c for c in tr_c if fold_of[c] != k]
        score_c = [c for c in tr_c if fold_of[c] == k] + [c for c in va_c if vfold_of[c] == k]
        ek = _StepEncoder(st["pca_dim"], st["pca_sample"]).fit([steps[c][0] for c in fit_c])
        Xf = [ek.transform(steps[c][0]) for c in fit_c]
        Xs = [ek.transform(steps[c][0]) for c in score_c]
        acc = [np.zeros(int(steps[c][2].sum())) for c in score_c]
        for r in range(N_SEEDS):
            rk = _tcn_fit(Xf, get(fit_c, 1), get(fit_c, 2), epochs=E,
                          seed=config.SEED + 10 * k + 100 * r + 1)
            for j, c in enumerate(score_c):
                acc[j] += _tcn_predict(rk.net, [Xs[j]], [steps[c][2]]) / N_SEEDS
        by.update(zip(score_c, acc))
        log(f"  fold {k} done ({N_SEEDS} runs)")
    res_out = {"name": "tcn", "settings": st, "history": H, "best_epoch": E, "n_seeds": N_SEEDS,
               "n_params": n_par, "in_dim": in_dim, "receptive_field_steps": tcn_receptive_field(),
               "oof": np.concatenate([by[c] for c in tr_c]),
               "val": np.concatenate([by[c] for c in va_c])}
    assert len(res_out["oof"]) == trm.sum() and len(res_out["val"]) == (df["split"] == "validation").sum()
    _save(res_out, "tcn")
    return res_out


def stack_extra_tcn(df: pd.DataFrame, V: dict, mlp: dict, tcn: dict) -> pd.DataFrame:
    """GBT+MLP's extra inputs plus the TCN's score (out-of-fold on train, the assigned fold
    models on validation)."""
    extra = stack_extra(df, V, mlp)
    score = np.full(len(df), np.nan)
    score[(df["split"] == "train").to_numpy()] = tcn["oof"]
    score[(df["split"] == "validation").to_numpy()] = tcn["val"]
    extra.insert(1, "tcn_score", score)
    return extra


# --------------------------------------------------------------------------- combiners without trees

LR_C_GRID = (0.01, 0.1, 1.0)


def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _lr_pipeline(X: pd.DataFrame, C: float):
    """Logistic regression on mixed inputs: numbers log-scaled if durations, median-filled
    with missing flags, standardised; categories one-hot."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    from . import models

    cat = [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]
    num = [c for c in X.columns if c not in cat]
    ct = ColumnTransformer([
        ("num", Pipeline([("float", models.ToFloat(log_durations=True)),
                          ("impute", SimpleImputer(strategy="median", add_indicator=True,
                                                   keep_empty_features=True)),
                          ("scale", StandardScaler())]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
    return Pipeline([("ct", ct), ("lr", LogisticRegression(C=C, max_iter=3000))])


def run_logistic(df: pd.DataFrame, X: pd.DataFrame, name: str, *, log=print) -> dict:
    """A logistic-regression combiner on the columns of X: C chosen on validation AP (fit on
    all train), then out-of-fold train scores and fold-assigned validation scores, as the
    trees. Saves {name}.pkl."""
    tr, va = (df["split"] == "train").to_numpy(), (df["split"] == "validation").to_numpy()
    y = df["y"].to_numpy()
    rows = []
    for C in LR_C_GRID:
        m = _lr_pipeline(X, C).fit(X[tr], y[tr])
        rows.append({"C": C, "val_AP": average_precision_score(y[va], m.predict_proba(X[va])[:, 1])})
    grid = pd.DataFrame(rows)
    C = float(grid.loc[grid["val_AP"].idxmax(), "C"])
    folds, vf = fold_labels(df.loc[tr, "class_id"]), val_folds(df.loc[va, "class_id"])
    Xtr, Xva = X[tr], X[va]
    oof, val = np.zeros(tr.sum()), np.zeros(va.sum())
    for k in range(N_FOLDS):
        m = _lr_pipeline(X, C).fit(Xtr[folds != k], y[tr][folds != k])
        oof[folds == k] = m.predict_proba(Xtr[folds == k])[:, 1]
        if (vf == k).any():
            val[vf == k] = m.predict_proba(Xva[vf == k])[:, 1]
    log(f"{name}: {X.shape[1]} inputs; C = {C} (validation AP {grid['val_AP'].max():.4f})")
    res = {"name": name, "inputs": list(X.columns), "grid": grid, "C": C, "folds": folds,
           "oof": oof, "val": val}
    _save(res, name)
    return res


def specialist_scores(df: pd.DataFrame, *names) -> pd.DataFrame:
    """The specialists' scores as logits, one column each (train: out-of-fold; validation:
    fold-assigned; NaN elsewhere)."""
    out = pd.DataFrame(index=df.index)
    for n in names:
        r = load_result(n)
        v = np.full(len(df), np.nan)
        v[(df["split"] == "train").to_numpy()] = r["oof"]
        v[(df["split"] == "validation").to_numpy()] = r["val"]
        out[f"{n}_logit"] = _logit(v)
    return out


# --------------------------------------------------------------------------- calibration


def calibrate(name: str, y_train: np.ndarray) -> dict:
    """Platt scaling, p' = sigma(a * logit(p) + b), fitted on the out-of-fold train scores.
    It is monotone, so it changes no ranking (AP, AP_w) and no warning under "yes as often as
    the truth"; it makes the probabilities mean what they say. Adds oof_cal / val_cal."""
    from sklearn.linear_model import LogisticRegression

    r = load_result(name)
    m = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(r["oof"])[:, None], y_train)
    r["platt"] = (float(m.coef_[0, 0]), float(m.intercept_[0]))
    r["oof_cal"] = m.predict_proba(_logit(r["oof"])[:, None])[:, 1]
    r["val_cal"] = m.predict_proba(_logit(r["val"])[:, None])[:, 1]
    _save(r, name)
    return r


def reliability(y, p, bins=(0, 0.01, 0.02, 0.04, 0.08, 1.0)) -> pd.DataFrame:
    """Mean predicted probability vs the observed share of positives, in probability bins."""
    b = pd.cut(np.asarray(p), bins=list(bins), include_lowest=True)
    t = pd.DataFrame({"bin": b, "p": p, "y": y}).groupby("bin", observed=True)
    return pd.DataFrame({"moments": t.size(), "mean predicted": t["p"].mean(),
                         "observed": t["y"].mean()})


# --------------------------------------------------------------------------- session bootstrap


def session_bootstrap(frame: pd.DataFrame, preds: dict, *, ref: str, B: int = 1000,
                      seed: int = config.SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resample the sessions of `frame` with replacement B times and recompute AP and AP_w for
    every model in `preds` on each resample (all models on the same resamples, so differences
    are paired). Returns (95% intervals per model; differences vs `ref`: mean, 95% interval and
    the share of resamples where `ref` is better)."""
    rng = np.random.default_rng(seed)
    y = frame["y"].to_numpy()
    idx = frame.groupby("class_id", sort=False).indices
    classes = list(idx)
    pos = np.array([y[idx[c]].sum() for c in classes], float)
    per_ap = {}
    for m, p in preds.items():                       # each session's own AP (for AP_w)
        per_ap[m] = np.array([average_precision_score(y[idx[c]], p[idx[c]]) if pos[i] > 0 else 0.0
                              for i, c in enumerate(classes)])
    draws = {m: {"AP": [], "AP_w": []} for m in preds}
    for _ in range(B):
        pick = rng.integers(0, len(classes), len(classes))
        rows = np.concatenate([idx[classes[i]] for i in pick])
        w = pos[pick]
        for m, p in preds.items():
            draws[m]["AP"].append(average_precision_score(y[rows], p[rows]))
            draws[m]["AP_w"].append(float((per_ap[m][pick] * w).sum() / w.sum()))
    ci, diff = [], []
    for m in preds:
        row = {"model": m}
        for k in ("AP", "AP_w"):
            d = np.asarray(draws[m][k])
            row[f"{k} low"], row[f"{k} high"] = np.quantile(d, [0.025, 0.975])
        ci.append(row)
        if m != ref:
            row = {"model": m}
            for k in ("AP", "AP_w"):
                d = np.asarray(draws[ref][k]) - np.asarray(draws[m][k])
                row[f"{k} gain"], row[f"{k} low"], row[f"{k} high"] = d.mean(), *np.quantile(d, [0.025, 0.975])
                row[f"{k}: ref better in"] = float((d > 0).mean())
            diff.append(row)
    return pd.DataFrame(ci).set_index("model"), pd.DataFrame(diff).set_index("model")


# --------------------------------------------------------------------------- the final test (once)

FINAL = "mlp_tcn_lr"            # declared before test (QUESTIONS.md, 18 Oct): MLP + TCN, no trees


def score_test(df: pd.DataFrame, *, log=print) -> dict:
    """The final pipeline, frozen on validation, scored ONCE on test. Only the fold models'
    scores were saved, so the same fold models are retrained with the same settings and seeds
    (MLP and TCN: 4 folds x N_SEEDS runs; the logistic combiner per fold on the saved
    out-of-fold specialist scores). Each test session is scored by one fold model (round-robin,
    as validation). Validation is rescored the same way as a reproducibility check. Saves
    test_final.pkl."""
    from . import gru

    split = df["split"].to_numpy()
    rows = {s: np.flatnonzero(split == s) for s in ("train", "validation", "test")}
    y = df["y"].to_numpy()
    cid = df["class_id"].to_numpy()
    folds = fold_labels(cid[rows["train"]])
    vf, tf = val_folds(cid[rows["validation"]]), val_folds(cid[rows["test"]])
    quiet = lambda *_: None
    mlp, tcn = load_result("mlp"), load_result("tcn")
    out = {m: {s: np.zeros(len(rows[s])) for s in ("train", "validation", "test")} for m in ("mlp", "tcn")}
    t0 = time.time()
    V = vector_inputs(df)
    for k in range(N_FOLDS):
        fit = rows["train"][folds != k]
        for r in range(N_SEEDS):
            nk, _, _ = train_mlp(V, fit, y[fit], epochs=mlp["best_epoch"], log=quiet,
                                 seed=config.SEED + 10 * k + 100 * r + 1)
            for s, sel in (("train", folds == k), ("validation", vf == k), ("test", tf == k)):
                out["mlp"][s][sel] += predict_mlp(nk, V, rows[s][sel]) / N_SEEDS
        log(f"  MLP fold {k} done ({time.time() - t0:.0f} s)")
    del V
    st = TCN_SETTINGS
    steps = tcn_steps(df, log=quiet)
    cls = {s: sorted(set(cid[rows[s]])) for s in rows}
    fold_of = dict(zip(cid[rows["train"]], folds))
    fold_of.update(zip(cls["validation"], val_folds(cls["validation"])))
    fold_of.update(zip(cls["test"], val_folds(cls["test"])))
    get = lambda cs, j: [steps[c][j] for c in cs]
    by = {}
    for k in range(N_FOLDS):
        fit_c = [c for c in cls["train"] if fold_of[c] != k]
        score_c = [c for s in ("train", "validation", "test") for c in cls[s] if fold_of[c] == k]
        ek = _StepEncoder(st["pca_dim"], st["pca_sample"]).fit([steps[c][0] for c in fit_c])
        Xf = [ek.transform(steps[c][0]) for c in fit_c]
        Xs = [ek.transform(steps[c][0]) for c in score_c]
        acc = [np.zeros(int(steps[c][2].sum())) for c in score_c]
        for r in range(N_SEEDS):
            rk = _tcn_fit(Xf, get(fit_c, 1), get(fit_c, 2), epochs=tcn["best_epoch"],
                          seed=config.SEED + 10 * k + 100 * r + 1)
            for j, c in enumerate(score_c):
                acc[j] += _tcn_predict(rk.net, [Xs[j]], [steps[c][2]]) / N_SEEDS
        by.update(zip(score_c, acc))
        log(f"  TCN fold {k} done ({time.time() - t0:.0f} s)")
    for s in rows:
        out["tcn"][s] = np.concatenate([by[c] for c in cls[s]])
    comb = load_result(FINAL)
    lg = lambda p: _logit(p)
    Xtr = pd.DataFrame({"mlp_logit": lg(mlp["oof"]), "tcn_logit": lg(tcn["oof"])})
    Xs_ = {s: pd.DataFrame({"mlp_logit": lg(out["mlp"][s]), "tcn_logit": lg(out["tcn"][s])})
           for s in ("validation", "test")}
    final = {s: np.zeros(len(rows[s])) for s in ("validation", "test")}
    for k in range(N_FOLDS):
        m = _lr_pipeline(Xtr, comb["C"]).fit(Xtr[folds != k], y[rows["train"]][folds != k])
        for s, sel in (("validation", vf == k), ("test", tf == k)):
            if sel.any():
                final[s][sel] = m.predict_proba(Xs_[s][sel])[:, 1]
    from . import metrics as _m

    yv, cv = y[rows["validation"]], cid[rows["validation"]]
    repro = {"val_AP_w_saved": _m.ap_within(yv, comb["val"], cv),
             "val_AP_w_rescored": _m.ap_within(yv, final["validation"], cv),
             "val_corr": float(np.corrcoef(comb["val"], final["validation"])[0, 1]),
             "mlp_oof_corr": float(np.corrcoef(mlp["oof"], out["mlp"]["train"])[0, 1]),
             "tcn_oof_corr": float(np.corrcoef(tcn["oof"], out["tcn"]["train"])[0, 1])}
    log(f"  reproducibility: {repro}")
    res = {"final": FINAL, "test": final["test"], "val_rescored": final["validation"],
           "mlp_test": out["mlp"]["test"], "tcn_test": out["tcn"]["test"], "repro": repro,
           "tau": None}
    from . import baselines as _b

    res["tau"] = _b.tau_like_truth(y[rows["train"]], comb["oof"])
    _save(res, "test_final")
    log(f"test scored ({time.time() - t0:.0f} s)")
    return res


def run_test_bootstrap(df: pd.DataFrame, *, B: int = 1000, log=print) -> dict:
    """Session bootstrap on TEST: the final pipeline against its two specialists and the
    frozen naive models (all scored on the same resamples)."""
    from . import baselines as _b, metrics as _m

    te = df[df["split"] == "test"].reset_index(drop=True)
    data = _b.load(config.TABLES_DIR, config.DATA_DIR / "03_split" / "split.csv")
    dte = data[data["split"] == "test"].reset_index(drop=True)
    assert (dte[["class_id", "t_s"]].to_numpy() == te[["class_id", "t_s"]].to_numpy()).all()
    with open(config.DATA_DIR / "results" / "naive_models.pkl", "rb") as fh:
        fitted = pickle.load(fh)["fitted"]
    tres = load_result("test_final")
    preds = {i: np.asarray(_b.proba(fitted[i][0], dte)) for i in ("R", "QT", "NB", "TFIDF")}
    preds.update({"mlp": tres["mlp_test"], "tcn": tres["tcn_test"], FINAL: tres["test"]})
    t0 = time.time()
    ci, diff = session_bootstrap(te, preds, ref=FINAL, B=B)
    out = {"B": B, "ref": FINAL, "ci": ci, "diff": diff,
           "point": {m: {"AP_w": _m.ap_within(te["y"].to_numpy(), p, te["class_id"].to_numpy()),
                         "AP": float(average_precision_score(te["y"], p))} for m, p in preds.items()},
           "random_AP_w": _m.ap_within(te["y"].to_numpy(), np.full(len(te), 0.5), te["class_id"].to_numpy()),
           "random_AP": float(te["y"].mean())}
    _save(out, "test_bootstrap")
    log(f"test bootstrap: {B} resamples ({time.time() - t0:.0f} s)")
    return out


# --------------------------------------------------------------------------- io / cli


def _save(res: dict, name: str) -> Path:
    out = out_dir()
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.pkl"
    with open(path, "wb") as fh:
        pickle.dump(res, fh)
    return path


def load_result(name: str) -> dict:
    with open(out_dir() / f"{name}.pkl", "rb") as fh:
        return pickle.load(fh)


# --------------------------------------------------------------------------- learning (validation)

LEARN_EPOCHS = 15


def run_learning(df: pd.DataFrame, *, epochs: int = LEARN_EPOCHS, log=print) -> dict:
    """3.6. How the final pipeline learned, on validation only. The fold models of the MLP and
    the TCN are retrained as in `run_mlp` / `run_tcn` (same folds, settings and seeds) for
    `epochs` epochs; after every epoch each run scores its held-out train sessions and its
    validation sessions. At each epoch the seed-averaged scores are combined as `FINAL` (a
    logistic regression per fold on the out-of-fold scores), so the row at the chosen epoch
    is the final pipeline itself (`repro` checks it). Saves learning.pkl: per epoch, train and
    validation log loss of single runs and validation AP / AP_w of the MLP, the TCN and the
    pipeline, plus the pipeline's validation scores after every epoch."""
    from . import metrics

    split = df["split"].to_numpy()
    tr, va = np.flatnonzero(split == "train"), np.flatnonzero(split == "validation")
    y, cid = df["y"].to_numpy(), df["class_id"].to_numpy()
    folds, vf = fold_labels(cid[tr]), val_folds(cid[va])
    quiet = lambda *_: None
    bce = lambda yy, p: float(-np.sum(yy * np.log(p + 1e-7) + (1 - yy) * np.log(1 - p + 1e-7)))
    score = {m: {"oof": np.zeros((epochs, len(tr)), np.float32), "val": np.zeros((epochs, len(va)), np.float32)}
             for m in ("mlp", "tcn")}
    loss = {m: {"train": np.zeros(epochs), "val": np.zeros(epochs)} for m in ("mlp", "tcn")}
    t0 = time.time()
    V = vector_inputs(df)
    for k in range(N_FOLDS):
        fit, held, vrows = tr[folds != k], tr[folds == k], va[vf == k]

        def snap(ep, net):
            score["mlp"]["oof"][ep - 1, folds == k] += predict_mlp(net, V, held) / N_SEEDS
            pv = predict_mlp(net, V, vrows)
            score["mlp"]["val"][ep - 1, vf == k] += pv / N_SEEDS
            loss["mlp"]["val"][ep - 1] += bce(y[vrows], pv) / (N_SEEDS * len(va))

        for r in range(N_SEEDS):
            _, h, _ = train_mlp(V, fit, y[fit], epochs=epochs, log=quiet, on_epoch=snap,
                                seed=config.SEED + 10 * k + 100 * r + 1)
            loss["mlp"]["train"] += h["train_loss"].to_numpy() / (N_SEEDS * N_FOLDS)
        log(f"  MLP fold {k} done ({time.time() - t0:.0f} s)")
    del V
    st = TCN_SETTINGS
    steps = tcn_steps(df, log=quiet)
    tr_c, va_c = sorted(set(cid[tr])), sorted(set(cid[va]))
    fold_of = dict(zip(cid[tr], folds))
    fold_of.update(zip(va_c, val_folds(va_c)))
    get = lambda cs, j: [steps[c][j] for c in cs]
    by = {}
    for k in range(N_FOLDS):
        fit_c = [c for c in tr_c if fold_of[c] != k]
        val_k = [c for c in va_c if fold_of[c] == k]
        score_c = [c for c in tr_c if fold_of[c] == k] + val_k
        ek = _StepEncoder(st["pca_dim"], st["pca_sample"]).fit([steps[c][0] for c in fit_c])
        Xf = [ek.transform(steps[c][0]) for c in fit_c]
        Xs = [ek.transform(steps[c][0]) for c in score_c]
        acc = {c: np.zeros((epochs, int(steps[c][2].sum())), np.float32) for c in score_c}
        is_val = set(val_k)

        def snap(ep, net):
            for x, c in zip(Xs, score_c):
                p = _tcn_predict(net, [x], [steps[c][2]])
                acc[c][ep - 1] += p / N_SEEDS
                if c in is_val:
                    loss["tcn"]["val"][ep - 1] += bce(steps[c][1][steps[c][2]], p) / (N_SEEDS * len(va))

        for r in range(N_SEEDS):
            rk = _tcn_fit(Xf, get(fit_c, 1), get(fit_c, 2), epochs=epochs, on_epoch=snap,
                          seed=config.SEED + 10 * k + 100 * r + 1)
            loss["tcn"]["train"] += np.array([h["train_loss"] for h in rk.history]) / (N_SEEDS * N_FOLDS)
        by.update(acc)
        log(f"  TCN fold {k} done ({time.time() - t0:.0f} s)")
    score["tcn"]["oof"] = np.concatenate([by[c] for c in tr_c], axis=1)
    score["tcn"]["val"] = np.concatenate([by[c] for c in va_c], axis=1)
    comb = load_result(FINAL)
    final_val = np.zeros((epochs, len(va)), np.float32)
    rows = []
    for e in range(epochs):
        Xtr = pd.DataFrame({"mlp_logit": _logit(score["mlp"]["oof"][e]), "tcn_logit": _logit(score["tcn"]["oof"][e])})
        Xva = pd.DataFrame({"mlp_logit": _logit(score["mlp"]["val"][e]), "tcn_logit": _logit(score["tcn"]["val"][e])})
        for k in range(N_FOLDS):
            m = _lr_pipeline(Xtr, comb["C"]).fit(Xtr[folds != k], y[tr][folds != k])
            final_val[e, vf == k] = m.predict_proba(Xva[vf == k])[:, 1]
        row = {"epoch": e + 1}
        for name, p in (("mlp", score["mlp"]["val"][e]), ("tcn", score["tcn"]["val"][e]), ("final", final_val[e])):
            row[f"{name}_AP"] = average_precision_score(y[va], p)
            row[f"{name}_AP_w"] = metrics.ap_within(y[va], p, cid[va])
        for m_ in ("mlp", "tcn"):
            row[f"{m_}_train_loss"], row[f"{m_}_val_loss"] = loss[m_]["train"][e], loss[m_]["val"][e]
        rows.append(row)
    curve = pd.DataFrame(rows)
    E = load_result("mlp")["best_epoch"]
    repro = {"epoch": E, "AP_w_saved": metrics.ap_within(y[va], comb["val"], cid[va])}
    if E <= epochs:
        repro.update(AP_w_here=float(curve.loc[E - 1, "final_AP_w"]),
                     corr=float(np.corrcoef(comb["val"], final_val[E - 1])[0, 1]))
    log(f"  learning: chosen epoch {E}; {repro}")
    res = {"epochs": epochs, "chosen_epoch": E, "tcn_chosen_epoch": load_result("tcn")["best_epoch"],
           "curve": curve, "final_val": final_val, "repro": repro}
    _save(res, "learning")
    log(f"learning curves saved ({time.time() - t0:.0f} s)")
    return res


STEPS = ("gbt", "mlp", "stack", "tcn", "stack_tcn", "combiners", "importance", "calibrate",
         "bootstrap", "learning")
TEST_STEP = "test"              # NOT part of "all": run once, after the final model is declared
FLEX_MODELS = ("gbt", "mlp", "gbt_mlp", "tcn", "gbt_mlp_tcn", "mlp_tcn_lr", "lr_all")


def all_inputs(df: pd.DataFrame, V: dict) -> pd.DataFrame:
    """The full stack's inputs for the logistic combiner: the trees' inputs, the cosines and
    the two specialists' scores (as logits)."""
    return pd.concat([df[gbt_inputs(df.columns)],
                      pd.DataFrame(V["cosines"], columns=list(COSINES), index=df.index),
                      specialist_scores(df, "mlp", "tcn")], axis=1)


def run_bootstrap(df: pd.DataFrame, *, B: int = 1000, log=print) -> dict:
    """Session bootstrap on validation for the naive and flexible models; the reference is the
    model with the best within-class AP."""
    from . import metrics

    va = df[df["split"] == "validation"].reset_index(drop=True)
    with open(config.DATA_DIR / "results" / "naive_models.pkl", "rb") as fh:
        fitted = pickle.load(fh)["fitted"]
    preds = {i: np.asarray(fitted[i][2]) for i in ("R", "QT", "NB", "TFIDF")}
    preds.update({m: load_result(m)["val"] for m in FLEX_MODELS})
    apw = {m: metrics.ap_within(va["y"].to_numpy(), p, va["class_id"].to_numpy()) for m, p in preds.items()}
    ref = max(apw, key=apw.get)
    t0 = time.time()
    ci, diff = session_bootstrap(va, preds, ref=ref, B=B)
    out = {"B": B, "ref": ref, "ci": ci, "diff": diff, "point_AP_w": apw}
    _save(out, "bootstrap")
    log(f"bootstrap: {B} resamples of {va['class_id'].nunique()} sessions, reference {ref} "
        f"({time.time() - t0:.0f} s)")
    return out


def cli(args) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="oncue flex")
    ap.add_argument("--step", default="all", choices=STEPS + ("all", TEST_STEP))
    opts = ap.parse_args(getattr(args, "extra", None) or [])
    todo = STEPS if opts.step == "all" else (opts.step,)
    log = lambda s: print(s, flush=True)
    if opts.step == TEST_STEP:
        df = load()
        score_test(df, log=log)
        run_test_bootstrap(df, log=log)
        return 0
    t0 = time.time()
    df = load()
    log(f"loaded {len(df):,} counting moments ({time.time() - t0:.0f} s)")
    V = None
    need_v = {"mlp", "stack", "stack_tcn", "combiners", "importance"}
    if need_v & set(todo):
        V = vector_inputs(df)
        log(f"vectors ready ({time.time() - t0:.0f} s)")
    if "gbt" in todo:
        run_gbt(df, name="gbt", log=log)
    if "mlp" in todo:
        run_mlp(df, V, log=log)
    if "stack" in todo:
        run_gbt(df, stack_extra(df, V, load_result("mlp")), name="gbt_mlp", log=log)
    if "tcn" in todo:
        run_tcn(df, log=log)
    if "stack_tcn" in todo:
        run_gbt(df, stack_extra_tcn(df, V, load_result("mlp"), load_result("tcn")),
                name="gbt_mlp_tcn", log=log)
    if "combiners" in todo:
        run_logistic(df, specialist_scores(df, "mlp", "tcn"), "mlp_tcn_lr", log=log)
        run_logistic(df, all_inputs(df, V), "lr_all", log=log)
    if "importance" in todo:
        importance(df, load_result("gbt"), log=log)
        importance(df, load_result("gbt_mlp"), extra=stack_extra(df, V, load_result("mlp")), log=log)
        importance(df, load_result("gbt_mlp_tcn"),
                   extra=stack_extra_tcn(df, V, load_result("mlp"), load_result("tcn")), log=log)
    if "calibrate" in todo:
        for m in FLEX_MODELS:
            calibrate(m, df.loc[df["split"] == "train", "y"].to_numpy())
        log("calibrated: " + ", ".join(FLEX_MODELS))
    if "bootstrap" in todo:
        run_bootstrap(df, log=log)
    if "learning" in todo:
        run_learning(df, log=log)
    log(f"oncue flex {opts.step}: done ({time.time() - t0:.0f} s)")
    return 0
