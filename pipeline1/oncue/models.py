"""Models (spec §6; amendments A4, A5, A9): AO, B-min, B0, R, B1, M1-M4, M3+stack, M4-MLP.

`make(model_id, featureset, hyper=...)` returns an unfitted sklearn estimator. Every model reads
one *design frame*: a DataFrame aligned row-for-row with `points`, holding

* `class_id`, `t_s` (keys: never model inputs; `class_id` is read only to count training
  classes for `min_df` and the course filter, and to build grouped inner folds);
* allowlisted feature columns (`config.FEATURE_ALLOWLIST`; categoricals as str/category);
* text windows `txt_all60`, `txt_all15`, `txt_chat60`: space-separated, role-prefixed
  normalised tokens (`p_what p_do s_i ...`); F-name / F-punct variants carry a suffix
  (`txt_all60_name`, `txt_all60_punct`);
* point embeddings `emb_{v15,v60,last,lastprof}_{000..383}` (the 1,536-d M4 block).

Hygiene (§6, §8.1): every transform (TF-IDF vocabulary/IDF, scalers, splines, the course
class-count filter, one-hot levels, PCA, SVD) is fit inside the estimator, i.e. inside the
training fold. Inner splits are `GroupKFold` by class, never random. Calibration is Platt on
grouped inner out-of-fold scores (A9: never isotonic). No resampling, no class weights.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from . import config

assert config.CALIBRATION_METHOD == "sigmoid", "A9: Platt only, never isotonic"

# --------------------------------------------------------------------------- design frame

KEY_COLUMNS = ("class_id", "t_s")
TEXT_WINDOWS = ("txt_all60", "txt_all15", "txt_chat60")
TEXT_VARIANTS = (None, "name", "punct")
EMB_BLOCKS = ("v15", "v60", "last", "lastprof")      # §5.4: [v_15, v_60, e_last, e_lastprof]
CTX_TOKEN_COLUMNS = ("ctx_seg_type", "ctx_min", "me_since_turn_s", "q_open")
RULE_COLUMNS = ("q_open", "q_age_s", "q_inv_age_s")
M4_C_GRID = (0.01, 0.03, 0.1)          # §6 says "C chosen once per outer fold" without a grid
_SPL_RE = re.compile(r"ctx_min_spl_\d+$")
_EMB_RE = re.compile(r"emb_(v15|v60|last|lastprof)_\d+$")


def text_column(window: str, variant: str | None = None) -> str:
    """`txt_all60` (headline) or `txt_all60_name` / `txt_all60_punct`."""
    return window if variant is None else f"{window}_{variant}"


def emb_columns_for(block: str, dim: int = config.EMB_DIM) -> list[str]:
    return [f"emb_{block}_{i:03d}" for i in range(dim)]


def emb_frame(matrix: np.ndarray, rows, dim: int = config.EMB_DIM) -> pd.DataFrame:
    """Expand a point-embedding cache [v15 | v60 | e_last | e_lastprof] (n x 4*dim) at `rows`
    (the features table's `emb_row`; <NA> -> NaN) into named `emb_*` columns (float32)."""
    rows = pd.array(rows, dtype="Int64")
    out = np.full((len(rows), len(EMB_BLOCKS) * dim), np.nan, dtype=np.float32)
    ok = ~rows.isna()
    if ok.any():
        out[np.asarray(ok)] = np.asarray(matrix, dtype=np.float32)[rows[ok].to_numpy(int)]
    cols = [c for b in EMB_BLOCKS for c in emb_columns_for(b, dim)]
    return pd.DataFrame(out, columns=cols)


# --------------------------------------------------------------------------- feature sets (§5.7)

PCW_BRANCH = "N"   # §5.6: Branch H adds pcwA to F-all; frozen in the prereg after D8
_STRUCT = ("ctx", "spk", "me", "me_hand", "poll")
_ALL = _STRUCT + ("q", "text", "emb", "pcwT")


def _drop(groups, *names):
    return tuple(g for g in groups if g not in names)


FEATURESETS: dict[str, dict] = {
    "F-none": {"groups": ()},
    "F-min": {"groups": (), "columns": ("ctx_min",)},
    "F-q": {"groups": ("q",)},
    "F-struct": {"groups": _STRUCT},
    "F-struct-me_hand": {"groups": _drop(_STRUCT, "me_hand")},
    "F-text": {"groups": ("text", "q", "emb"), "columns": ("ctx_min",)},
    "F-all": {"groups": _ALL, "branch_h": ("pcwA",)},
    "F-all-me_hand": {"groups": _drop(_ALL, "me_hand"), "branch_h": ("pcwA",)},
    "F-all-pcw": {"groups": _ALL[:-1]},
    "F-all-pcwT": {"groups": _ALL[:-1], "branch_h": ("pcwA",)},
    "F-all+pcwA": {"groups": _ALL + ("pcwA",)},
    "F-pcw": {"groups": ("ctx", "pcwT", "pcwA")},
    "F-all-me": {"groups": _drop(_ALL, "me", "me_hand"), "branch_h": ("pcwA",)},
    "F-name": {"groups": _ALL, "branch_h": ("pcwA",), "text_variant": "name"},
    "F-punct": {"groups": _ALL + ("punct",), "branch_h": ("pcwA",), "text_variant": "punct"},
    "F-all+hist": {"groups": _ALL + ("hist",), "branch_h": ("pcwA",)},
    "F-all+edu": {"groups": _ALL + ("edu",), "branch_h": ("pcwA",)},
    "F-all+stack": {"groups": _ALL + ("stack",), "branch_h": ("pcwA",)},
}


def canonical_featureset(name: str | None) -> str:
    """Accept the spec's Unicode minus ("F-all−me_hand") and return the registry key."""
    key = (name or "F-none").replace("−", "-").strip()
    if key not in FEATURESETS:
        raise KeyError(f"unknown feature set {name!r}; known: {', '.join(FEATURESETS)}")
    return key


def featureset_groups(name: str | None, branch: str = PCW_BRANCH) -> frozenset[str]:
    spec = FEATURESETS[canonical_featureset(name)]
    extra = spec.get("branch_h", ()) if branch == "H" else ()
    return frozenset(spec["groups"]) | frozenset(extra)


def featureset_text_variant(name: str | None) -> str | None:
    return FEATURESETS[canonical_featureset(name)].get("text_variant")


def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s.dtype) or pd.api.types.is_bool_dtype(s.dtype)


def select_columns(X: pd.DataFrame, featureset: str | None, kind: str,
                   branch: str = PCW_BRANCH) -> list[str]:
    """Columns of `X` that `featureset` feeds to one input block.

    kinds: num (dense numeric), cat (+ class_id), text (+ class_id), emb (all four blocks),
    emb_v60, emb_last, minute (ctx_min), ctx_tokens (NB pseudo-tokens), rule (R's q_*).
    """
    fs = canonical_featureset(featureset)
    groups = featureset_groups(fs, branch)
    explicit = set(FEATURESETS[fs].get("columns", ()))
    cols = list(X.columns)

    def allowed(c):
        if c in KEY_COLUMNS or not config.feature_allowed(c, fs):
            return False
        return config.feature_group(c) in groups or c in explicit

    if kind in ("num", "cat"):
        out = [c for c in cols if allowed(c) and not c.startswith(("emb_", "txt_"))
               and not _SPL_RE.match(c) and _is_numeric(X[c]) == (kind == "num")]
        return out + ["class_id"] if kind == "cat" and out else out
    if kind == "text":
        if "text" not in groups:
            return []
        variant = featureset_text_variant(fs)
        out = [text_column(w, variant) for w in TEXT_WINDOWS if text_column(w, variant) in X]
        if not out and variant is not None and any(w in X for w in TEXT_WINDOWS):
            raise KeyError(f"{fs} needs the {variant!r} text windows (e.g. "
                           f"{text_column(TEXT_WINDOWS[0], variant)}) in the design frame")
        return out + ["class_id"] if out else []
    if kind.startswith("emb"):
        if "emb" not in groups:
            return []
        blocks = {"emb": EMB_BLOCKS, "emb_v60": ("v60",), "emb_last": ("last",)}[kind]
        return [c for c in cols if (m := _EMB_RE.match(c)) and m.group(1) in blocks]
    if kind == "minute":
        return ["ctx_min"] if "ctx_min" in X and ("ctx" in groups or "ctx_min" in explicit) \
            else []
    if kind == "ctx_tokens":
        return [c for c in CTX_TOKEN_COLUMNS if c in X and allowed(c)]
    if kind == "rule":
        return [c for c in RULE_COLUMNS if c in X]
    raise ValueError(f"unknown column kind {kind!r}")


class Cols:
    """Picklable ColumnTransformer column selector, evaluated at fit time on the design."""

    def __init__(self, featureset: str | None, kind: str, branch: str = PCW_BRANCH):
        self.featureset, self.kind, self.branch = featureset, kind, branch

    def __call__(self, X: pd.DataFrame) -> list[str]:
        return select_columns(X, self.featureset, self.kind, self.branch)

    def __repr__(self) -> str:
        return f"Cols({self.featureset!r}, {self.kind!r})"


# --------------------------------------------------------------------------- transformers


class _NoWrap:
    """Mixin for transformers that may emit sparse output: accept `set_output` as a no-op so
    a pandas-output ColumnTransformer can hold them (a later step densifies)."""

    def set_output(self, *, transform=None):
        return self


class ToFloat(TransformerMixin, BaseEstimator):
    """Numeric/boolean/nullable columns -> float64 with NaN. With `log_durations`, columns
    ending in `_s` are clipped to [0, 3600] s and log1p'd (§5.7 encoding). With
    `drop_empty`, columns with no finite value in the training data are dropped (sklearn
    1.9's HistGradientBoosting binner raises on an all-NaN column, e.g. a feature whose
    input table does not exist yet)."""

    def __init__(self, log_durations: bool = False, drop_empty: bool = False):
        self.log_durations = log_durations
        self.drop_empty = drop_empty

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(list(X.columns), dtype=object)
        cols = list(X.columns)
        if self.drop_empty:
            cols = [c for c in cols if np.isfinite(self._col(X, c)).any()]
        self.columns_ = np.asarray(cols, dtype=object)
        return self

    def _col(self, X, c) -> np.ndarray:
        v = pd.to_numeric(X[c], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        if self.log_durations and c.endswith("_s"):
            v = np.log1p(np.clip(v, 0.0, config.DURATION_CAP_S))
        return v

    def transform(self, X):
        cols = getattr(self, "columns_", X.columns)
        return pd.DataFrame({c: self._col(X, c) for c in cols}, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(getattr(self, "columns_", self.feature_names_in_), dtype=object)


class ResetIndex(TransformerMixin, BaseEstimator):
    """Give the design a RangeIndex so pandas-output blocks (some built from sparse input,
    which carries no index) concatenate row-aligned."""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(list(X.columns), dtype=object)
        return self

    def transform(self, X):
        return X.reset_index(drop=True)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_in_, dtype=object)


class CategoryPrep(TransformerMixin, BaseEstimator):
    """Categorical columns (input includes `class_id`, which is dropped).

    `ctx_course` keeps only courses with >= `min_classes` *training classes* (not rows); the
    rest become "other" (§5.7 MinClassCountCategories). Other columns keep their training
    levels. `as_category=True` (HGB) returns pandas categoricals with the fitted levels,
    unknown/missing -> NaN; otherwise strings with "missing" for NaN (for one-hot)."""

    def __init__(self, min_classes: int = config.CTX_COURSE_MIN_CLASSES,
                 course_col: str = "ctx_course", as_category: bool = False):
        self.min_classes, self.course_col, self.as_category = min_classes, course_col, \
            as_category

    def _clean(self, X: pd.DataFrame, c: str) -> pd.Series:
        s = X[c].astype(object).map(lambda v: None if pd.isna(v) else str(v))
        if c == self.course_col and hasattr(self, "courses_"):
            s = s.map(lambda v: None if pd.isna(v) else (v if v in self.courses_ else "other"))
        return s

    def fit(self, X, y=None):
        self.columns_ = [c for c in X.columns if c != "class_id"]
        if self.course_col in self.columns_:
            per = (pd.DataFrame({"course": X[self.course_col].astype(object),
                                 "cid": X["class_id"].to_numpy()})
                   .dropna().drop_duplicates().groupby("course").size())
            self.courses_ = frozenset(str(k) for k, n in per.items() if n >= self.min_classes)
        self.levels_ = {c: sorted({v for v in self._clean(X, c) if not pd.isna(v)})
                        for c in self.columns_}
        if self.course_col in self.levels_ and "other" not in self.levels_[self.course_col]:
            self.levels_[self.course_col].append("other")
        return self

    def transform(self, X):
        out = {}
        for c in self.columns_:
            s = self._clean(X, c)
            if self.as_category:            # unseen levels -> NaN explicitly (pandas 4)
                known = s.where(s.isin(self.levels_[c]))
                out[c] = pd.Categorical(known, categories=self.levels_[c])
            else:
                out[c] = s.fillna("missing").astype(object).to_numpy()
        return pd.DataFrame(out, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columns_, dtype=object)


class MultiTextVectorizer(_NoWrap, TransformerMixin, BaseEstimator,
                          auto_wrap_output_keys=None):
    """TF-IDF (or counts) of whitespace tokens, one vocabulary per text column, hstacked.

    Input columns: the text windows plus `class_id`. A term is kept only if it occurs in at
    least `min_classes` distinct *training classes* (§5.7 min_df by class); the
    `max_features` most frequent survivors are kept per column. Output is sparse CSR.
    """

    def __init__(self, kind: str = "tfidf", min_classes: int = config.TFIDF_MIN_CLASSES,
                 ngram_range=config.TFIDF_PARAMS["ngram_range"],
                 max_features: int = config.TFIDF_PARAMS["max_features"],
                 sublinear_tf: bool = config.TFIDF_PARAMS["sublinear_tf"]):
        self.kind, self.min_classes, self.ngram_range = kind, min_classes, ngram_range
        self.max_features, self.sublinear_tf = max_features, sublinear_tf

    def _cv(self, vocabulary=None):
        return CountVectorizer(token_pattern=r"\S+", lowercase=False,
                               ngram_range=tuple(self.ngram_range), vocabulary=vocabulary,
                               dtype=np.float32)

    @staticmethod
    def _docs(s: pd.Series) -> list[str]:
        return s.fillna("").astype(str).tolist()

    def fit(self, X, y=None):
        self.text_cols_ = [c for c in X.columns if c != "class_id"]
        codes, _ = pd.factorize(X["class_id"])
        G = sparse.csr_matrix((np.ones(len(codes)), (codes, np.arange(len(codes)))))
        self.vocab_, self.tfidf_ = {}, {}
        for c in self.text_cols_:
            docs = self._docs(X[c])
            cv = self._cv()
            try:
                counts = cv.fit_transform(docs)
            except ValueError:                       # empty vocabulary
                self.vocab_[c] = {}
                continue
            n_cls = np.asarray(((G @ (counts > 0)) > 0).sum(axis=0)).ravel()
            freq = np.asarray(counts.sum(axis=0)).ravel()
            keep = np.flatnonzero(n_cls >= self.min_classes)
            if len(keep) > self.max_features:
                keep = keep[np.argsort(-freq[keep], kind="stable")[: self.max_features]]
            terms = cv.get_feature_names_out()[np.sort(keep)]
            self.vocab_[c] = {t: i for i, t in enumerate(terms)}
            if self.kind == "tfidf" and terms.size:
                self.tfidf_[c] = TfidfTransformer(sublinear_tf=self.sublinear_tf).fit(
                    counts[:, np.sort(keep)])
        return self

    def transform(self, X):
        blocks = []
        for c in self.text_cols_:
            if not self.vocab_[c]:
                continue
            m = self._cv(self.vocab_[c]).transform(self._docs(X[c]))
            if self.kind == "tfidf":
                m = self.tfidf_[c].transform(m)
            blocks.append(m.astype(np.float32))
        if not blocks:      # no term reached min_classes: one empty column keeps shapes valid
            return sparse.csr_matrix((len(X), 1), dtype=np.float32)
        return sparse.hstack(blocks, format="csr")

    def get_feature_names_out(self, input_features=None):
        names = [f"{c}:{t}" for c in self.text_cols_ for t in self.vocab_[c]]
        return np.asarray(names or ["__no_terms__"], dtype=object)


class ContextTokens(_NoWrap, TransformerMixin, BaseEstimator, auto_wrap_output_keys=None):
    """NB context pseudo-tokens (§5.7): ctx_seg_<type>, ctx_min_<bin>, ctx_sinceme_<bin>,
    ctx_qopen, as a fixed-vocabulary sparse count matrix (nothing is learned)."""

    MIN_BINS = (0, 15, 30, 45, 60, 75)
    SINCE_BINS = ((0, 60, "lt1m"), (60, 300, "1_5m"), (300, 600, "5_10m"),
                  (600, math.inf, "10m_plus"))

    def __init__(self):
        pass

    def vocabulary(self) -> list[str]:
        v = [f"ctx_seg_{s}" for s in config.SEG_TYPES]
        v += [f"ctx_min_{a}_{b}" for a, b in zip(self.MIN_BINS, self.MIN_BINS[1:] + (999,))]
        v += [f"ctx_sinceme_{n}" for *_, n in self.SINCE_BINS] + ["ctx_sinceme_never",
                                                                  "ctx_qopen"]
        return v

    def fit(self, X, y=None):
        self.vocab_ = {t: i for i, t in enumerate(self.vocabulary())}
        return self

    def _min_token(self, j: int) -> str:
        edges = self.MIN_BINS[1:] + (999,)
        return f"ctx_min_{self.MIN_BINS[j]}_{edges[j]}"

    def transform(self, X):
        n = len(X)
        rows, cols = [], []

        def add(mask, idx):
            r = np.flatnonzero(mask)
            rows.append(r)
            cols.append(np.asarray(idx)[r] if np.ndim(idx) else np.full(len(r), idx))
        if "ctx_seg_type" in X:
            seg = X["ctx_seg_type"].astype(object).map(
                lambda v: self.vocab_.get(f"ctx_seg_{v}", -1)).to_numpy(int)
            add(seg >= 0, seg)
        if "ctx_min" in X:
            m = pd.to_numeric(X["ctx_min"], errors="coerce").to_numpy(float)
            j = np.searchsorted(self.MIN_BINS[1:], np.nan_to_num(m), side="right")
            idx = np.array([self.vocab_[self._min_token(k)] for k in range(len(self.MIN_BINS))])
            add(np.isfinite(m), idx[j])
        if "me_since_turn_s" in X:
            v = pd.to_numeric(X["me_since_turn_s"], errors="coerce").to_numpy(float)
            edges = np.array([b for _, b, _ in self.SINCE_BINS[:-1]], float)
            k = np.searchsorted(edges, np.clip(np.nan_to_num(v), 0, None), side="right")
            idx = np.array([self.vocab_[f"ctx_sinceme_{name}"] for *_, name in
                            self.SINCE_BINS])[k]
            add(np.ones(n, bool), np.where(np.isfinite(v), idx,
                                           self.vocab_["ctx_sinceme_never"]))
        if "q_open" in X:
            q = pd.to_numeric(X["q_open"], errors="coerce").fillna(0).to_numpy(float)
            add(q > 0, self.vocab_["ctx_qopen"])
        r = np.concatenate(rows) if rows else np.array([], int)
        c = np.concatenate(cols) if cols else np.array([], int)
        return sparse.csr_matrix((np.ones(len(r), np.float32), (r, c)),
                                 shape=(n, len(self.vocab_)))

    def get_feature_names_out(self, input_features=None):
        return np.asarray(list(self.vocab_), dtype=object)


class SafePCA(TransformerMixin, BaseEstimator):
    """PCA (dense) or TruncatedSVD (sparse) with the component count capped by the data, so
    tiny folds and small vocabularies still fit. NaN rows (no embedding yet) -> 0 input."""

    def __init__(self, n_components: int = 32, prefix: str = "pc", sparse_input: bool = False):
        self.n_components, self.prefix, self.sparse_input = n_components, prefix, sparse_input

    def fit(self, X, y=None):
        A = self._arr(X)
        n, d = A.shape
        k = int(min(self.n_components, max(n - 1, 0), max(d - (1 if self.sparse_input else 0),
                                                           0)))
        self.k_ = k
        if k < 1:
            self.model_ = None
            return self
        if self.sparse_input:
            self.model_ = TruncatedSVD(n_components=k, random_state=config.SEED).fit(A)
        else:
            self.model_ = PCA(n_components=k, svd_solver="randomized",
                              random_state=config.SEED).fit(A)
        return self

    def _arr(self, X):
        if sparse.issparse(X):
            return X
        A = np.asarray(X, dtype=np.float32)
        return np.nan_to_num(A, nan=0.0)

    def transform(self, X):
        A = self._arr(X)
        if self.model_ is None:            # too little data: one constant column
            return np.zeros((A.shape[0], 1), dtype=np.float32)
        return self.model_.transform(A).astype(np.float32)

    def get_feature_names_out(self, input_features=None):
        return np.asarray([f"{self.prefix}{i}" for i in range(max(self.k_, 1))], dtype=object)


# --------------------------------------------------------------------------- simple models


class ConstantProba(ClassifierMixin, BaseEstimator):
    """p(x) = `p` for every row (AO: p = 1, the FROC reference point)."""

    def __init__(self, p: float = 1.0):
        self.p = p

    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = np.full(len(X), float(self.p))
        return np.column_stack([1 - p, p])


class PriorProba(ClassifierMixin, BaseEstimator):
    """B0: the training-fold prevalence π for every row."""

    def __init__(self):
        pass

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        self.prior_ = float(np.mean(np.asarray(y, float))) if len(y) else 0.0
        return self

    def predict_proba(self, X):
        p = np.full(len(X), self.prior_)
        return np.column_stack([1 - p, p])


def rule_r(q_open, q_inv_age_s, q_age_s, *, cap: float = config.Q_AGE_CAP_S,
           decay_s: float = config.R_DECAY_S) -> np.ndarray:
    """R (§5.5): (0.5 + 0.5 q_open) * exp(-min(q_inv_age, 2 q_age) / 10 s); ∞/NaN -> cap."""
    q = np.nan_to_num(np.asarray(q_open, float), nan=0.0)
    inv = np.clip(np.nan_to_num(np.asarray(q_inv_age_s, float), nan=cap, posinf=cap), 0, cap)
    age = np.clip(np.nan_to_num(np.asarray(q_age_s, float), nan=cap, posinf=cap), 0, cap)
    return (0.5 + 0.5 * (q > 0)) * np.exp(-np.minimum(inv, 2 * age) / decay_s)


class RuleR(ClassifierMixin, BaseEstimator):
    """R: the question rule, no fitting (reads q_open, q_inv_age_s, q_age_s)."""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        missing = [c for c in RULE_COLUMNS if c not in X]
        if missing:
            raise KeyError(f"rule R needs columns {missing}")
        p = rule_r(X["q_open"], X["q_inv_age_s"], X["q_age_s"])
        return np.column_stack([1 - p, p])


# --------------------------------------------------------------------------- pipelines


def _linear_dense() -> Pipeline:
    return Pipeline([("float", ToFloat(log_durations=True)),
                     ("impute", SimpleImputer(strategy="median", add_indicator=True,
                                              keep_empty_features=True)),
                     ("scale", StandardScaler())])


def _linear_cat() -> Pipeline:
    return Pipeline([("prep", CategoryPrep()),
                     ("onehot", OneHotEncoder(handle_unknown="ignore"))])


def _minute_splines() -> Pipeline:
    knots = np.asarray(config.CTX_SPLINE_KNOTS, float)[:, None]
    return Pipeline([("float", ToFloat()), ("impute", SimpleImputer(strategy="median")),
                     ("spline", SplineTransformer(knots=knots, degree=3,
                                                  extrapolation="linear"))])


def _linear_blocks(fs: str, *, text: bool, emb: bool, minute: bool = True) -> list:
    blocks = [("num", _linear_dense(), Cols(fs, "num")),
              ("cat", _linear_cat(), Cols(fs, "cat"))]
    if minute:
        blocks.append(("min", _minute_splines(), Cols(fs, "minute")))
    if text:
        blocks.append(("txt", MultiTextVectorizer("tfidf"), Cols(fs, "text")))
    if emb:
        blocks.append(("emb", Pipeline([("impute", SimpleImputer(strategy="constant",
                                                                 fill_value=0.0,
                                                                 keep_empty_features=True)),
                                        ("scale", StandardScaler())]), Cols(fs, "emb")))
    return blocks


def _lr(C: float, max_iter: int) -> LogisticRegression:
    return LogisticRegression(C=C, solver="lbfgs", max_iter=max_iter)


def _hgb(n_iter: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(**config.HGB_PARAMS, max_iter=int(n_iter),
                                          random_state=config.SEED)


def _m3_preprocessor(fs: str) -> ColumnTransformer:
    pca = config.PCA_DIMS
    ct = ColumnTransformer([
        ("num", ToFloat(drop_empty=True), Cols(fs, "num")),
        ("cat", CategoryPrep(as_category=True), Cols(fs, "cat")),
        ("v60", SafePCA(pca["v60"], "pca_v60_"), Cols(fs, "emb_v60")),
        ("elast", SafePCA(pca["e_last"], "pca_elast_"), Cols(fs, "emb_last")),
        ("svd", Pipeline([("tfidf", MultiTextVectorizer("tfidf")),
                          ("svd", SafePCA(pca["tfidf_svd"], "svd_", sparse_input=True))]),
         Cols(fs, "text")),
    ], remainder="drop", verbose_feature_names_out=True)
    return ct.set_output(transform="pandas")


@dataclass(frozen=True)
class ModelInfo:
    id: str
    name: str
    default_featureset: str
    tier: str
    in_class: bool
    calibrate: bool = True
    tuned: str | None = None          # hyperparameter tuned once per outer fold
    sequence: bool = False            # M5/M6 have their own fit loops (gru.py, xenc.py)


MODELS: dict[str, ModelInfo] = {m.id: m for m in (
    ModelInfo("AO", "Always-on (p = 1)", "F-none", "mvp", False, calibrate=False),
    ModelInfo("B-min", "Minute-only LR", "F-min", "mvp", True, tuned="C"),
    ModelInfo("B0", "Prevalence", "F-none", "mvp", False, calibrate=False),
    ModelInfo("R", "Question rule", "F-q", "mvp", False),
    ModelInfo("B1", "Structure-only LR", "F-struct", "mvp", True, tuned="C"),
    ModelInfo("M1", "Multinomial naive Bayes", "F-all", "mvp", True, tuned="alpha"),
    ModelInfo("M2", "Sparse TF-IDF LR", "F-all", "mvp", True, tuned="C"),
    ModelInfo("M3", "HistGradientBoosting", "F-all", "mvp", False, tuned="n_star"),
    ModelInfo("M4", "Frozen embeddings + LR", "F-all", "must", False, tuned="C"),
    ModelInfo("M3+stack", "M3 with grouped inner-OOF M2/M4 scores (A4)", "F-all", "should",
              False),
    ModelInfo("M4-MLP", "Frozen embeddings + MLP", "F-all", "could", False),
    ModelInfo("M5", "Streaming GRU hazard (gru.py)", "F-all", "should", False, sequence=True),
    ModelInfo("M6", "Fine-tuned cross-encoder (xenc.py)", "F-all", "could", False,
              sequence=True),
)}
MODEL_IDS = tuple(MODELS)
TUNED_MODELS = tuple(m for m, info in MODELS.items() if info.tuned)
GRIDS = {"B-min": config.B1_C_GRID, "B1": config.B1_C_GRID, "M1": config.M1_ALPHA_GRID,
         "M2": config.M2_C_GRID, "M4": M4_C_GRID}
DEFAULT_HYPER = {"B-min": {"C": 0.1}, "B1": {"C": 0.1}, "M1": {"alpha": 0.3},
                 "M2": {"C": 0.1}, "M3": {"n_star": config.HGB_MAX_ITER}, "M4": {"C": 0.03}}


def default_featureset(model_id: str) -> str:
    return MODELS[model_id].default_featureset


def make(model_id: str, featureset: str | None = None, *, hyper: dict | None = None,
         branch: str = PCW_BRANCH):
    """Unfitted estimator for `model_id` on `featureset` (default: the model's own, §6).

    `hyper` carries the per-outer-fold choices from `tune_once` (C, alpha, n_star; for
    M3+stack also m2_C and m4_C). M5/M6 live in gru.py / xenc.py and are not built here.
    """
    if model_id not in MODELS:
        raise KeyError(f"unknown model {model_id!r}; known: {', '.join(MODELS)}")
    info = MODELS[model_id]
    if info.sequence:
        raise ValueError(f"{model_id} is a sequence model: use oncue.gru / oncue.xenc")
    fs = canonical_featureset(featureset or info.default_featureset)
    h = {**DEFAULT_HYPER.get(model_id, {}), **(hyper or {})}
    if model_id == "AO":
        return ConstantProba(1.0)
    if model_id == "B0":
        return PriorProba()
    if model_id == "R":
        return RuleR()
    if model_id == "B-min":
        ct = ColumnTransformer([("min", _minute_splines(), Cols("F-min", "minute"))])
        return Pipeline([("ct", ct), ("lr", _lr(h["C"], config.LR_MAX_ITER["B1"]))])
    if model_id == "B1":
        ct = ColumnTransformer(_linear_blocks(fs, text=False, emb=False))
        return Pipeline([("ct", ct), ("lr", _lr(h["C"], config.LR_MAX_ITER["B1"]))])
    if model_id == "M1":
        ct = ColumnTransformer([("txt", MultiTextVectorizer("count"), Cols(fs, "text")),
                                ("ctx", ContextTokens(), Cols(fs, "ctx_tokens"))])
        return Pipeline([("ct", ct), ("nb", MultinomialNB(alpha=h["alpha"]))])
    if model_id == "M2":
        ct = ColumnTransformer(_linear_blocks(fs, text=True, emb=False))
        return Pipeline([("ct", ct), ("lr", _lr(h["C"], config.LR_MAX_ITER["M2"]))])
    if model_id == "M3":
        return Pipeline([("reset", ResetIndex()), ("ct", _m3_preprocessor(fs)),
                         ("hgb", _hgb(h["n_star"]))])
    if model_id == "M4":
        ct = ColumnTransformer(_linear_blocks(fs, text=False, emb=True))
        return Pipeline([("ct", ct), ("lr", _lr(h["C"], config.LR_MAX_ITER["M2"]))])
    if model_id == "M4-MLP":
        ct = ColumnTransformer(_linear_blocks(fs, text=False, emb=True))
        mlp = MLPClassifier((128,), alpha=1e-3, early_stopping=False, max_iter=200,
                            random_state=config.SEED)
        return Pipeline([("ct", ct), ("mlp", mlp)])
    if model_id == "M3+stack":
        return StackedM3(featureset=fs, n_star=h.get("n_star", config.HGB_MAX_ITER),
                         m2_C=h.get("m2_C", DEFAULT_HYPER["M2"]["C"]),
                         m4_C=h.get("m4_C", DEFAULT_HYPER["M4"]["C"]), branch=branch)
    raise AssertionError(model_id)


# --------------------------------------------------------------------------- fitting helpers


LOGIT_EPS = 1e-300                    # only for models without a log-odds method (AO, R, B0)


def _logit(p, eps: float = LOGIT_EPS) -> np.ndarray:
    p = np.clip(np.asarray(p, float), eps, 1 - 1e-16)
    return np.log(p) - np.log1p(-p)


def _sigmoid(s) -> np.ndarray:
    return expit(np.asarray(s, float))


def predict_p(model, X) -> np.ndarray:
    return np.asarray(model.predict_proba(X))[:, 1]


def decision_scores(model, X) -> np.ndarray:
    """Unclipped log-odds scores, the input to Platt scaling (the same transform at inner
    OOF and at prediction time).

    LR and HGB give `decision_function` (their raw log-odds); naive Bayes (and any model
    with `predict_log_proba`) gives log p1 - log p0, computed in log space, so a saturating
    MultinomialNB keeps its ranking instead of collapsing at a probability clip. Only models
    with nothing better fall back to the logit of predict_proba (clipped at 1e-300)."""
    if hasattr(model, "decision_function"):
        s = np.asarray(model.decision_function(X), dtype=float)
        if s.ndim == 1:
            return s
    if hasattr(model, "predict_log_proba"):
        lp = np.asarray(model.predict_log_proba(X), dtype=float)
        s = lp[:, 1] - lp[:, 0]
        if np.isfinite(s).all():
            return s
        fallback = _logit(predict_p(model, X))
        return np.where(np.isfinite(s), s, fallback)
    return _logit(predict_p(model, X))


def fit_safe(estimator, X, y):
    """Fit, or fall back to the training prior when a fold holds a single label."""
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return ConstantProba(float(y.mean()) if len(y) else 0.0).fit(X, y)
    return clone(estimator).fit(X, y)


def groups_of(X: pd.DataFrame) -> np.ndarray:
    """Inner-fold groups: `cv_group` when the design carries it (L18: same-section same-date
    twins share a group), else `class_id`."""
    if "cv_group" in X and X["cv_group"].notna().all():
        return X["cv_group"].to_numpy()
    if "class_id" not in X:
        raise KeyError("the design frame needs a class_id column for grouped inner folds")
    return X["class_id"].to_numpy()


def inner_splits(groups, n_splits: int = config.N_INNER_FOLDS):
    """GroupKFold by class (never random); fewer splits if there are fewer classes."""
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        raise ValueError("grouped inner CV needs at least 2 classes")
    k = min(n_splits, n_groups)
    return list(GroupKFold(n_splits=k).split(np.zeros(len(groups)), groups=groups))


def inner_oof(estimator, X: pd.DataFrame, y, *, n_splits: int = config.N_INNER_FOLDS,
              scores: bool = True, grid: pd.DataFrame | None = None):
    """Grouped inner out-of-fold scores (log-odds, or probabilities with scores=False).

    With `grid` (more rows of the same outer training fold, e.g. its whole causal grid G
    including undefined labels), also returns the inner models' scores on those rows: each
    row is scored by the inner model that did not train on its group (rows of a group that
    has no fitted row go to the first inner model, which never saw them either)."""
    y = np.asarray(y)
    out = np.full(len(X), np.nan)
    g_out = None if grid is None else np.full(len(grid), np.nan)
    g_groups = None if grid is None else groups_of(grid)
    groups = groups_of(X)
    assigned = np.zeros(0 if grid is None else len(grid), bool)
    for j, (tr, va) in enumerate(inner_splits(groups, n_splits)):
        m = fit_safe(estimator, X.iloc[tr], y[tr])
        f = decision_scores if scores else predict_p
        out[va] = f(m, X.iloc[va])
        if grid is not None:
            rows = np.isin(g_groups, np.unique(groups[va]))
            if j == 0:
                rows |= ~np.isin(g_groups, np.unique(groups))
            rows &= ~assigned
            if rows.any():
                g_out[rows] = f(m, grid.iloc[np.flatnonzero(rows)])
                assigned |= rows
    return out if grid is None else (out, g_out)


@dataclass
class Platt:
    """p = sigmoid(a * s + b), fit by maximum likelihood with Platt's smoothed targets."""

    a: float = 1.0
    b: float = 0.0

    @classmethod
    def fit(cls, s, y) -> Platt:
        s, y = np.asarray(s, float), np.asarray(y, float)
        ok = np.isfinite(s) & np.isfinite(y)
        s, y = s[ok], y[ok]
        n_pos, n_neg = y.sum(), len(y) - y.sum()
        if len(y) == 0:
            return cls(0.0, 0.0)
        t = np.where(y > 0, (n_pos + 1) / (n_pos + 2), 1 / (n_neg + 2))
        if n_pos == 0 or n_neg == 0 or np.ptp(s) == 0:
            return cls(0.0, float(_logit(t.mean())))
        # fit on standardised scores (log-odds of saturating models span thousands), then
        # map the coefficients back: a * s + b == a' * (s - mu) / sd + b'
        mu, sd = float(s.mean()), float(s.std()) or 1.0
        u = (s - mu) / sd

        def nll(w):
            z = w[0] * u + w[1]
            loss = np.logaddexp(0, z) - t * z
            p = _sigmoid(z)
            return loss.sum(), np.array([((p - t) * u).sum(), (p - t).sum()])

        res = minimize(nll, x0=np.array([1.0, 0.0]), jac=True, method="L-BFGS-B")
        a = float(res.x[0]) / sd
        return cls(a, float(res.x[1]) - a * mu)

    def __call__(self, s) -> np.ndarray:
        return _sigmoid(self.a * np.asarray(s, float) + self.b)


class CalibratedModel:
    """Fit `estimator` on the outer training fold and a Platt map on its grouped inner OOF
    scores; equivalent to `CalibratedClassifierCV(ensemble=False, cv=GroupKFold(3),
    method="sigmoid")` with groups = class. Platt reads unclipped log-odds
    (`decision_scores`) both at inner OOF and at prediction time.

    The inner OOF scores are kept for the OOF table and the rate-matched event thresholds
    (§8.3 step 1): `inner_scores_` on the fitted (defined-label) rows and, when `fit` gets
    `grid` (the outer training fold's whole causal grid G, undefined labels included),
    `grid_scores_` on those rows, so thresholds see the same grid FAR is measured on."""

    def __init__(self, estimator, *, calibrate: bool = True,
                 n_inner: int = config.N_INNER_FOLDS):
        self.estimator, self.calibrate, self.n_inner = estimator, calibrate, n_inner

    def fit(self, X: pd.DataFrame, y, grid: pd.DataFrame | None = None) -> CalibratedModel:
        y = np.asarray(y)
        self.grid_scores_ = None
        if self.calibrate:
            res = inner_oof(self.estimator, X, y, n_splits=self.n_inner, grid=grid)
            self.inner_scores_, self.grid_scores_ = res if grid is not None else (res, None)
            self.platt_ = Platt.fit(self.inner_scores_, y)
        else:
            self.inner_scores_, self.platt_ = None, None
        self.model_ = fit_safe(self.estimator, X, y)
        return self

    def predict(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """(p_raw, p_cal)."""
        p_raw = predict_p(self.model_, X)
        p_cal = (self.platt_(decision_scores(self.model_, X)) if self.platt_ is not None
                 else p_raw)
        return p_raw, p_cal

    def inner_predictions(self) -> tuple[np.ndarray, np.ndarray] | None:
        """(p_raw, p_cal) of the inner OOF rows, aligned with the training rows."""
        if self.inner_scores_ is None:
            return None
        return _sigmoid(self.inner_scores_), self.platt_(self.inner_scores_)

    def grid_predictions(self) -> tuple[np.ndarray, np.ndarray] | None:
        """(p_raw, p_cal) of the inner models on the `grid` rows given to `fit`."""
        if self.grid_scores_ is None:
            return None
        return _sigmoid(self.grid_scores_), self.platt_(self.grid_scores_)


# --------------------------------------------------------------------------- stacking (A4)


class StackedM3(ClassifierMixin, BaseEstimator):
    """M3 + stack (A4): grouped inner-OOF probabilities of M2 (TF-IDF LR) and M4 (embeddings
    LR) become `stack_m2`, `stack_m4` for M3. Inner OOF is GroupKFold(3) by class inside the
    training fold; at prediction time the stack columns come from M2/M4 refit on the whole
    training fold."""

    def __init__(self, featureset: str = "F-all", n_star: int = config.HGB_MAX_ITER,
                 m2_C: float = 0.1, m4_C: float = 0.03, n_inner: int = config.STACK_INNER_FOLDS,
                 branch: str = PCW_BRANCH):
        self.featureset, self.n_star, self.m2_C, self.m4_C = featureset, n_star, m2_C, m4_C
        self.n_inner, self.branch = n_inner, branch

    def _bases(self) -> dict:
        fs = canonical_featureset(self.featureset).replace("+stack", "")
        fs = fs if fs in FEATURESETS else "F-all"
        return {"stack_m2": make("M2", fs, hyper={"C": self.m2_C}, branch=self.branch),
                "stack_m4": make("M4", fs, hyper={"C": self.m4_C}, branch=self.branch)}

    def _m3_featureset(self) -> str:
        fs = canonical_featureset(self.featureset)
        return fs if "stack" in featureset_groups(fs, self.branch) else "F-all+stack"

    def fit(self, X: pd.DataFrame, y):
        y = np.asarray(y)
        oof = {name: inner_oof(est, X, y, n_splits=self.n_inner, scores=False)
               for name, est in self._bases().items()}
        self.m3_ = fit_safe(make("M3", self._m3_featureset(), hyper={"n_star": self.n_star},
                                 branch=self.branch), X.assign(**oof), y)
        self.bases_ = {name: fit_safe(est, X, y) for name, est in self._bases().items()}
        self.classes_ = np.array([0, 1])
        return self

    def stack_columns(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        return {name: predict_p(est, X) for name, est in self.bases_.items()}

    def predict_proba(self, X: pd.DataFrame):
        return self.m3_.predict_proba(X.assign(**self.stack_columns(X)))

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        """M3's raw log-odds on the stacked design (Platt input)."""
        return decision_scores(self.m3_, X.assign(**self.stack_columns(X)))


# --------------------------------------------------------------------------- tune_once (§6)


def _ap(y, p) -> float:
    y = np.asarray(y)
    return float(average_precision_score(y, p)) if 0 < y.sum() < len(y) else float("nan")


def _logloss_curve(y, staged) -> list[float]:
    y = np.asarray(y, float)
    out = []
    for proba in staged:
        p = np.clip(proba[:, 1], 1e-12, 1 - 1e-12)
        out.append(float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p))))
    return out


def tune_hgb(X: pd.DataFrame, y, *, featureset: str = "F-all",
             max_iter: int = config.HGB_MAX_ITER, n_inner: int = config.N_INNER_FOLDS,
             branch: str = PCW_BRANCH) -> dict:
    """n* = argmin of the mean inner-validation staged log-loss (§6). Returns n_star plus the
    per-inner-fold training and validation curves (the must-tier learning figure)."""
    y = np.asarray(y)
    train_curves, val_curves = [], []
    for tr, va in inner_splits(groups_of(X), n_inner):
        if len(np.unique(y[tr])) < 2:
            continue
        pipe = make("M3", featureset, hyper={"n_star": max_iter}, branch=branch)
        pipe.fit(X.iloc[tr], y[tr])
        pre, hgb = pipe[:-1], pipe[-1]
        train_curves.append(_logloss_curve(y[tr], hgb.staged_predict_proba(
            pre.transform(X.iloc[tr]))))
        val_curves.append(_logloss_curve(y[va], hgb.staged_predict_proba(
            pre.transform(X.iloc[va]))))
    if not val_curves:
        return {"n_star": max_iter, "curves": {"train": [], "val": []}}
    n = min(len(c) for c in val_curves)
    mean_val = np.mean([c[:n] for c in val_curves], axis=0)
    return {"n_star": int(np.argmin(mean_val)) + 1,
            "curves": {"train": [c[:n] for c in train_curves],
                       "val": [c[:n] for c in val_curves], "val_mean": mean_val.tolist()}}


def tune_grid(model_id: str, X: pd.DataFrame, y, *, featureset: str | None = None,
              grid=None, n_inner: int = config.N_INNER_FOLDS, branch: str = PCW_BRANCH) -> dict:
    """Pick C (or alpha) by mean inner-validation AP; ties go to the stronger penalty (the
    first grid value). Keeps training and validation AP per value (learning figure (b))."""
    y = np.asarray(y)
    param = MODELS[model_id].tuned
    grid = tuple(grid or GRIDS[model_id])
    splits = inner_splits(groups_of(X), n_inner)
    rows = []
    for value in grid:
        tr_ap, va_ap = [], []
        for tr, va in splits:
            m = fit_safe(make(model_id, featureset, hyper={param: value}, branch=branch),
                         X.iloc[tr], y[tr])
            tr_ap.append(_ap(y[tr], predict_p(m, X.iloc[tr])))
            va_ap.append(_ap(y[va], predict_p(m, X.iloc[va])))
        rows.append({param: value, "train_ap": float(np.nanmean(tr_ap)) if np.isfinite(
            tr_ap).any() else float("nan"), "val_ap": float(np.nanmean(va_ap)) if np.isfinite(
            va_ap).any() else float("nan"), "val_ap_folds": va_ap})
    vals = np.array([r["val_ap"] for r in rows], float)
    best = grid[int(np.nanargmax(vals))] if np.isfinite(vals).any() else \
        DEFAULT_HYPER[model_id][param]
    return {param: best, "curves": rows}


def tune_once(X: pd.DataFrame, y, *, models=TUNED_MODELS, featureset: str = "F-all",
              n_inner: int = config.N_INNER_FOLDS, hgb_max_iter: int = config.HGB_MAX_ITER,
              branch: str = PCW_BRANCH) -> dict:
    """Hyperparameters chosen once per outer fold on F-all (§6, §1.3) and reused for every
    ablation and sweep. B1 is tuned on F-struct and B-min on F-min (their only inputs).

    Returns {model: {param: value, "curves": ...}} (JSON-serialisable)."""
    out = {}
    for m in models:
        if m == "M3":
            out[m] = tune_hgb(X, y, featureset=featureset, max_iter=hgb_max_iter,
                              n_inner=n_inner, branch=branch)
        elif m in GRIDS:
            fs = {"B1": "F-struct", "B-min": "F-min"}.get(m, featureset)
            out[m] = tune_grid(m, X, y, featureset=fs, n_inner=n_inner, branch=branch)
    return out


def hyper_for(model_id: str, tuned: dict | None) -> dict:
    """The `hyper` dict for `make` from a `tune_once` result (defaults where untuned)."""
    tuned = tuned or {}
    if model_id == "M3+stack":
        return {"n_star": tuned.get("M3", {}).get("n_star", config.HGB_MAX_ITER),
                "m2_C": tuned.get("M2", {}).get("C", DEFAULT_HYPER["M2"]["C"]),
                "m4_C": tuned.get("M4", {}).get("C", DEFAULT_HYPER["M4"]["C"])}
    if model_id == "M4-MLP":
        return {}
    param = MODELS[model_id].tuned if model_id in MODELS else None
    if not param:
        return {}
    value = tuned.get(model_id, {}).get(param, DEFAULT_HYPER[model_id][param])
    return {param: value}


# --------------------------------------------------------------------------- kNN explainer (A5)


@dataclass
class KNNExplainer:
    """Nearest neighbours over pooled window embeddings of *training* classes. Output is
    metadata only (segment type, minute, whether a bid followed, distance): exemplars are
    classmates' words, so no text ever leaves this object."""

    k: int = config.KNN_K
    meta_columns: tuple = ("seg_type", "minute", "bid_followed")
    _nn: NearestNeighbors | None = field(default=None, repr=False)

    def fit(self, emb: np.ndarray, meta: pd.DataFrame) -> KNNExplainer:
        extra = [c for c in meta.columns if c not in self.meta_columns]
        if extra:
            raise ValueError(f"kNN metadata must be {self.meta_columns}; got extra {extra}")
        if len(meta) != len(emb):
            raise ValueError("emb and meta must align")
        self.meta_ = meta[list(self.meta_columns)].reset_index(drop=True)
        self._nn = NearestNeighbors(n_neighbors=min(self.k, len(emb)), metric="cosine")
        self._nn.fit(np.nan_to_num(np.asarray(emb, np.float32)))
        return self

    def explain(self, emb: np.ndarray) -> pd.DataFrame:
        dist, idx = self._nn.kneighbors(np.nan_to_num(np.asarray(emb, np.float32)))
        n, k = idx.shape
        out = self.meta_.iloc[idx.ravel()].reset_index(drop=True)
        out.insert(0, "rank", np.tile(np.arange(k), n))
        out.insert(0, "query", np.repeat(np.arange(n), k))
        out["distance"] = dist.ravel()
        return out[["query", "rank", *config.KNN_OUTPUT_COLUMNS]]


def knn_bid_share(explainer: KNNExplainer, emb_true_pos: np.ndarray,
                  emb_false_alarm: np.ndarray) -> dict:
    """A5's aggregate: the share of nearest neighbours that preceded a bid, for true
    positives vs false alarms."""
    def share(e):
        if len(e) == 0:
            return float("nan")
        return float(explainer.explain(e)["bid_followed"].astype(float).mean())
    return {"share_tp": share(emb_true_pos), "share_fa": share(emb_false_alarm),
            "n_tp": len(emb_true_pos), "n_fa": len(emb_false_alarm)}
