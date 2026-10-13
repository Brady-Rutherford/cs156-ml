"""Cross-validation splits (spec §8.1, §7 S2) and the in-fold course filter (§5.7).

Every split is by **class**, never by point, except S-rand, which exists only to measure the
leakage tax. Fold labels are integers: `k >= 0` means "tested in fold k"; `TRAIN_ONLY` (-1)
means "always in training, never tested" (S-time training semesters, S-loco courses with too
few bids). `iter_folds` turns any fold-label array into (k, train_idx, test_idx) triples.

* `group_folds`: S-group (headline), StratifiedGroupKFold(5, shuffle, 156) over classes,
  stratified on course code; `fold_column` maps it onto `points.fold` (vault -> <NA>).
* `random_folds`: S-rand, KFold(5, shuffle, 156) over points.
* `time_split`: S-time, train semesters <= 5 (F23-F25), test 6-7 (Sp26, F26).
* `loco`: S-loco, leave one course out for courses with >= 15 bids.
* `inner_groups`: GroupKFold(3) inside an outer training fold (hyperparameters, n*, Platt,
  thresholds, stacking); the result is a list of (train, val) index pairs usable as `cv=`.
* `MinClassCountCategories`: rare courses -> "other", counting training *classes*, not rows.
"""

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Iterator

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold

from . import config

TRAIN_ONLY = -1
LOCO_MIN_BIDS = 15


# --------------------------------------------------------------------------- outer folds


def _class_series(values, class_ids=None, name: str = "value") -> pd.Series:
    """A per-class Series indexed by class id (accepts a Series or parallel arrays)."""
    if isinstance(values, pd.Series) and class_ids is None:
        s = values.copy()
    else:
        if class_ids is None:
            raise ValueError("class_ids are required when values is not a Series")
        s = pd.Series(np.asarray(values), index=pd.Index(np.asarray(class_ids)), name=name)
    if s.index.has_duplicates:
        # several rows per class: they must agree
        agg = s.groupby(level=0).nunique(dropna=False)
        if (agg > 1).any():
            raise ValueError(f"{name} is not constant within class: {list(agg[agg > 1].index)}")
        s = s.groupby(level=0).first()
    return s.sort_index()


def cv_groups(class_ids, classes: pd.DataFrame | None = None) -> pd.Series:
    """class id -> CV group id. Classes flagged as same-section same-date twins by
    `select.dedupe` (L18: a re-recording under a new class id) share `classes.cv_group`, so
    they always land in the same outer and inner fold; every other class is its own group."""
    ids = pd.Index(pd.unique(np.asarray(class_ids)))
    out = pd.Series(ids.to_numpy(), index=ids, name="cv_group")
    if classes is not None and "cv_group" in classes and "class_id" in classes:
        m = classes.drop_duplicates("class_id").set_index("class_id")["cv_group"]
        mapped = pd.Series(ids, index=ids).map(m)
        out = mapped.where(mapped.notna(), out).astype(out.dtype)
    return out


def group_folds(class_ids, strata=None, *, groups=None, n_splits: int = config.N_OUTER_FOLDS,
                seed: int = config.SEED) -> pd.Series:
    """S-group outer folds: one fold per class, stratified on `strata` (course code).

    `class_ids` may repeat (e.g. one entry per point) as long as `strata` is constant within a
    class. Classes are sorted first, so the assignment does not depend on row order. `groups`
    (class id -> CV group, see `cv_groups`) keeps grouped classes in one fold; by default
    every class is its own group. Returns a Series `fold` (int) indexed by class id. One row
    per class means courses are balanced by class count across folds.
    """
    ids = np.asarray(class_ids)
    if strata is None:
        strata = np.zeros(len(ids), dtype=int)
    per_class = _class_series(strata, ids, "strata")
    n = len(per_class)
    grp = (per_class.index.to_numpy() if groups is None
           else pd.Series(per_class.index, index=per_class.index).map(groups)
           .fillna(pd.Series(per_class.index, index=per_class.index)).to_numpy())
    n_groups = len(pd.unique(grp))
    if n_groups < n_splits:
        raise ValueError(f"group_folds: {n_groups} class groups < n_splits={n_splits}")
    y = pd.factorize(per_class.astype(str), sort=True)[0]
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold = np.full(n, TRAIN_ONLY, dtype=int)
    with warnings.catch_warnings():
        # a course with fewer than n_splits classes is expected (and harmless)
        warnings.filterwarnings("ignore", message=".*least populated class.*")
        for k, (_, test) in enumerate(splitter.split(np.zeros(n), y, groups=grp)):
            fold[test] = k
    return pd.Series(fold, index=per_class.index, name="fold")


def point_folds(class_ids, folds: pd.Series) -> np.ndarray:
    """Map a per-class fold Series onto rows (e.g. points). Unknown classes -> TRAIN_ONLY."""
    mapped = pd.Series(np.asarray(class_ids)).map(folds)
    return mapped.fillna(TRAIN_ONLY).astype(int).to_numpy()


def fold_column(points: pd.DataFrame, classes: pd.DataFrame, *,
                strata_col: str = "course_code", n_splits: int = config.N_OUTER_FOLDS,
                seed: int = config.SEED) -> pd.Series:
    """`points.fold` (nullable Int64): S-group folds over dev classes; <NA> for vault classes.

    `classes` needs class_id, `strata_col` and in_vault (missing in_vault = config.in_vault);
    `cv_group`, when present, keeps same-section same-date twins in one fold (L18). Computed
    once over all dev classes (`oncue labels`), so every population and variant inherits the
    same assignment (§8.1: the same folds for every condition).
    """
    cls = classes.drop_duplicates("class_id").set_index("class_id")
    vault = (cls["in_vault"].astype(bool) if "in_vault" in cls
             else pd.Series([config.in_vault(c) for c in cls.index], index=cls.index))
    dev = cls.index[~vault.to_numpy()]
    folds = group_folds(dev, cls.loc[dev, strata_col].to_numpy(),
                        groups=cv_groups(dev, classes), n_splits=n_splits, seed=seed)
    out = points["class_id"].map(folds).astype("Int64")
    out.name = "fold"
    return out


def random_folds(n_points: int, *, n_splits: int = config.N_OUTER_FOLDS,
                 seed: int = config.SEED) -> np.ndarray:
    """S-rand: KFold(5, shuffle=True, random_state=156) labels over points (inflation only)."""
    fold = np.full(int(n_points), TRAIN_ONLY, dtype=int)
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for k, (_, test) in enumerate(splitter.split(np.zeros(int(n_points)))):
        fold[test] = k
    return fold


def time_split(semester_idx, *, train_max_idx: int = config.S_TIME_TRAIN_MAX_IDX):
    """S-time: semesters <= train_max_idx train only (TRAIN_ONLY), later ones test fold 0.

    Accepts per-class or per-row semester indices; returns the same shape (Series in, Series
    out). Missing semester -> TRAIN_ONLY is never silently used: it raises.
    """
    s = pd.Series(semester_idx) if not isinstance(semester_idx, pd.Series) else semester_idx
    if s.isna().any():
        raise ValueError("time_split: missing semester_idx")
    fold = np.where(s.astype(int).to_numpy() <= train_max_idx, TRAIN_ONLY, 0)
    if isinstance(semester_idx, pd.Series):
        return pd.Series(fold, index=semester_idx.index, name="fold")
    return fold


def loco(courses: pd.Series, n_bids: pd.Series, *, min_bids: int = LOCO_MIN_BIDS) -> pd.Series:
    """S-loco: leave one course out, for courses with >= `min_bids` bids in total.

    `courses` and `n_bids` are per-class Series indexed by class id. Eligible courses become
    folds 0..m-1 in sorted order (`.attrs["courses"]` lists them); other classes are TRAIN_ONLY.
    """
    bids = n_bids.reindex(courses.index).fillna(0)
    per_course = bids.groupby(courses).sum()
    eligible = sorted(per_course.index[per_course >= min_bids])
    code = {c: k for k, c in enumerate(eligible)}
    fold = courses.map(code).fillna(TRAIN_ONLY).astype(int)
    fold.name = "fold"
    fold.attrs["courses"] = eligible
    return fold


def iter_folds(fold) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Yield (k, train_idx, test_idx) for every fold k >= 0; TRAIN_ONLY rows always train.

    Rows with a missing fold (<NA>, e.g. vault) are in neither set.
    """
    f = pd.Series(fold).astype("Int64")
    present = f.notna().to_numpy()
    vals = f.fillna(TRAIN_ONLY).to_numpy(dtype=int)
    for k in sorted(set(vals[present]) - {TRAIN_ONLY}):
        test = np.flatnonzero(present & (vals == k))
        train = np.flatnonzero(present & (vals != k))
        yield int(k), train, test


def check_no_leak(train_idx, test_idx, groups) -> None:
    """Raise AssertionError if any class has rows on both sides of a split (L15)."""
    g = np.asarray(groups)
    shared = np.intersect1d(g[np.asarray(train_idx, int)], g[np.asarray(test_idx, int)])
    if shared.size:
        raise AssertionError(f"classes on both sides of the split: {shared[:5].tolist()}")


# --------------------------------------------------------------------------- inner folds


def inner_groups(groups, *, n_splits: int = config.N_INNER_FOLDS
                 ) -> list[tuple[np.ndarray, np.ndarray]]:
    """GroupKFold(3) by class over the rows of one outer training fold.

    Returns [(train_idx, val_idx), ...] positions into `groups`, usable directly as `cv=` in
    sklearn (CalibratedClassifierCV, cross_val_predict, ...). Never a random point split (L16).
    """
    g = np.asarray(groups)
    n_groups = len(np.unique(g))
    if n_groups < n_splits:
        raise ValueError(f"inner_groups: {n_groups} classes < n_splits={n_splits}")
    splitter = GroupKFold(n_splits=n_splits)
    return [(tr, va) for tr, va in splitter.split(np.zeros(len(g)), groups=g)]


def inner_fold_labels(groups, *, n_splits: int = config.N_INNER_FOLDS) -> np.ndarray:
    """The inner GroupKFold(3) assignment as one label per row."""
    labels = np.full(len(groups), TRAIN_ONLY, dtype=int)
    for k, (_, va) in enumerate(inner_groups(groups, n_splits=n_splits)):
        labels[va] = k
    return labels


# --------------------------------------------------------------------------- course filter


class MinClassCountCategories(TransformerMixin, BaseEstimator):
    """Map categories seen in fewer than `min_classes` *training classes* to `other`.

    Rows are not counted: a course with 1,000 points from one class is still rare. Fit inside
    each fold. The class id comes from `groups=` in `fit`, or from the `group_col` column of a
    DataFrame (dropped from the output when `drop_group`). Missing values stay missing; unseen
    categories become `other`. DataFrame output columns are pandas Categoricals with a fixed
    category list (kept + `other`), so HGB `categorical_features="from_dtype"` and
    `OneHotEncoder(handle_unknown="ignore")` both work.
    """

    def __init__(self, min_classes: int = config.CTX_COURSE_MIN_CLASSES,
                 group_col: str = "class_id", other: str = "other", drop_group: bool = True):
        self.min_classes = min_classes
        self.group_col = group_col
        self.other = other
        self.drop_group = drop_group

    def _split(self, X):
        if isinstance(X, pd.DataFrame):
            cols = [c for c in X.columns if c != self.group_col]
            grp = X[self.group_col] if self.group_col in X.columns else None
            return X[cols], grp
        if isinstance(X, pd.Series):
            return X.to_frame(), None
        arr = np.asarray(X, dtype=object)
        arr = arr.reshape(-1, 1) if arr.ndim == 1 else arr
        return pd.DataFrame(arr, columns=[f"x{i}" for i in range(arr.shape[1])]), None

    def fit(self, X, y=None, groups=None):
        cats, grp = self._split(X)
        if groups is None:
            groups = grp
        if groups is None:
            raise ValueError("MinClassCountCategories needs class ids: pass groups= or a "
                             f"{self.group_col!r} column")
        groups = np.asarray(groups)
        if len(groups) != len(cats):
            raise ValueError("groups and X have different lengths")
        self.feature_names_in_ = np.asarray(cats.columns, dtype=object)
        self.n_classes_ = {}
        self.categories_ = {}
        for col in cats.columns:
            df = pd.DataFrame({"cat": cats[col].to_numpy(dtype=object), "g": groups}).dropna()
            counts = df.groupby("cat")["g"].nunique()
            self.n_classes_[col] = counts
            keep = sorted(str(c) for c in counts.index[counts >= self.min_classes])
            self.categories_[col] = [c for c in keep if c != self.other] + [self.other]
        return self

    def transform(self, X):
        cats, _ = self._split(X)
        out = {}
        for col in self.feature_names_in_:
            keep = set(self.categories_[col][:-1])
            mapped = [None if pd.isna(v) else (str(v) if str(v) in keep else self.other)
                      for v in cats[col].to_numpy(dtype=object)]
            out[col] = pd.Categorical(mapped, categories=self.categories_[col])
        res = pd.DataFrame(out, index=cats.index)
        if isinstance(X, pd.DataFrame):
            if not self.drop_group and self.group_col in X.columns:
                res[self.group_col] = X[self.group_col]
            return res
        return res.to_numpy(dtype=object)

    def get_feature_names_out(self, input_features=None):
        names = list(self.feature_names_in_)
        if not self.drop_group:
            names.append(self.group_col)
        return np.asarray(names, dtype=object)


# --------------------------------------------------------------------------- course split
# Brady's rule (13 Oct): drop classes he missed, then split each course's remaining classes
# 60/20/20 so every course appears in train, validation and test.
SPLIT_FRACTIONS = {"validation": 0.2, "test": 0.2}   # train gets the rest
MIN_CLASSES_PER_COURSE = 5                           # fewer can't fill all three splits


def _rank_key(unit, seed: int) -> str:
    """A stable, seeded pseudo-random order (the same on every machine and every run)."""
    return hashlib.sha256(f"{seed}:{unit}".encode()).hexdigest()


def course_split(classes: pd.DataFrame, *, seed: int = config.SEED,
                 min_classes: int = MIN_CLASSES_PER_COURSE) -> pd.DataFrame:
    """Assign every class to train, validation or test, separately within each course.

    `classes` needs `class_id`, `course_code` and `usable` (bool; False for classes Brady was
    absent from, not enrolled in, or that have no recording). An optional `cv_group` keeps
    classes taught on the same day together. Within a course, units are put in a seeded
    random order; the first 20% go to test, the next 20% to validation, the rest to train
    (each at least 1). Courses with fewer than `min_classes` usable units are excluded.

    Returns `class_id, course_code, split, reason`, where split is one of
    train / validation / test / excluded.
    """
    df = classes[["class_id", "course_code", "usable"]].copy()
    df["unit"] = classes["cv_group"] if "cv_group" in classes else classes["class_id"]
    df["split"] = "excluded"
    df["reason"] = np.where(df["usable"], "", "not usable")

    for course, grp in df[df["usable"]].groupby("course_code", sort=True):
        units = sorted(grp["unit"].unique(), key=lambda u: _rank_key(u, seed))
        n = len(units)
        if n < min_classes:
            df.loc[grp.index, "reason"] = f"course has {n} usable classes (< {min_classes})"
            continue
        n_test = max(1, round(n * SPLIT_FRACTIONS["test"]))
        n_val = max(1, round(n * SPLIT_FRACTIONS["validation"]))
        split_of = {u: ("test" if i < n_test else "validation" if i < n_test + n_val else "train")
                    for i, u in enumerate(units)}
        df.loc[grp.index, "split"] = grp["unit"].map(split_of)
    return df[["class_id", "course_code", "split", "reason"]]



# --------------------------------------------------------------------------- the step-3 split


def split_table(path=None) -> pd.DataFrame:
    """`data/03_split/split.csv` (step 3): class_id, course_code, ..., split."""
    return pd.read_csv(path or config.SPLIT_PATH)


def split_ids(split: str | None = None, *, path=None) -> list[int]:
    """Class ids of one split (train / validation / test), or of all three when `split` is
    None or "all". Excluded classes are never returned."""
    s = split_table(path)
    s = s[s["split"].isin(config.SPLITS)]
    if split not in (None, "all"):
        if split not in config.SPLITS:
            raise ValueError(f"unknown split {split!r}; use one of {config.SPLITS} or 'all'")
        s = s[s["split"] == split]
    return sorted(int(c) for c in s["class_id"])


def split_of(path=None) -> dict[int, str]:
    """class id -> split name, for every class in split.csv (excluded ones included)."""
    s = split_table(path)
    return dict(zip(s["class_id"].astype(int), s["split"].astype(str), strict=True))
