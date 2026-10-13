"""Tests for oncue.splits (spec §8.1, §5.7 course filter). Synthetic class tables only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from oncue import config, splits
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def _classes(n_per_course=(10, 10, 5, 3, 2), seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    cid = 1000
    for k, n in enumerate(n_per_course):
        for _ in range(n):
            cid += int(rng.integers(1, 50))
            rows.append({"class_id": cid, "course_code": f"C{k}",
                         "semester_idx": int(rng.integers(1, 8))})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- outer folds


def test_group_folds_are_by_class_deterministic_and_stratified():
    cls = _classes()
    f = splits.group_folds(cls["class_id"], cls["course_code"])
    assert f.index.tolist() == sorted(cls["class_id"])
    assert sorted(f.unique()) == [0, 1, 2, 3, 4]
    shuffled = cls.sample(frac=1, random_state=3)
    f2 = splits.group_folds(shuffled["class_id"], shuffled["course_code"])
    assert f.equals(f2)                                       # row order does not matter
    by_course = pd.crosstab(cls.set_index("class_id").loc[f.index, "course_code"], f)
    assert (by_course.loc["C0"] == 2).all() and (by_course.loc["C1"] == 2).all()
    assert (by_course.loc["C2"] == 1).all()
    assert not splits.group_folds(cls["class_id"], cls["course_code"], seed=1).equals(f)


def test_group_folds_accept_point_rows_and_reject_inconsistent_strata():
    cls = _classes()
    pts = cls.loc[cls.index.repeat(7)]
    f = splits.group_folds(pts["class_id"], pts["course_code"])
    assert f.equals(splits.group_folds(cls["class_id"], cls["course_code"]))
    bad = pts["course_code"].to_numpy().copy()
    bad[0] = "C9"
    with pytest.raises(ValueError):
        splits.group_folds(pts["class_id"], bad)
    with pytest.raises(ValueError):
        splits.group_folds([1, 2, 3], ["a", "a", "b"])        # fewer classes than folds


def test_fold_column_marks_vault_na_and_never_splits_a_class():
    cls = _classes()
    cls["in_vault"] = [config.in_vault(c) for c in cls["class_id"]]
    pts = pd.DataFrame({"class_id": np.repeat(cls["class_id"], 5).to_numpy(),
                        "t_s": np.tile(np.arange(5) * 2.0, len(cls))})
    fold = splits.fold_column(pts, cls)
    assert fold.dtype == "Int64" and len(fold) == len(pts)
    vault_ids = set(cls.loc[cls["in_vault"], "class_id"])
    assert fold[pts["class_id"].isin(vault_ids)].isna().all()
    assert fold[~pts["class_id"].isin(vault_ids)].notna().all()
    assert (pts.assign(f=fold).groupby("class_id")["f"].nunique(dropna=False) == 1).all()
    folds = splits.group_folds(cls["class_id"], cls["course_code"])
    assert splits.point_folds([cls["class_id"].iloc[0], -5], folds).tolist() == [
        folds.iloc[0], splits.TRAIN_ONLY]


def test_iter_folds_and_no_leak():
    cls = _classes()
    f = splits.group_folds(cls["class_id"], cls["course_code"])
    groups = np.repeat(f.index.to_numpy(), 4)
    fold = splits.point_folds(groups, f)
    seen = []
    for k, tr, te in splits.iter_folds(fold):
        splits.check_no_leak(tr, te, groups)
        assert set(fold[te]) == {k} and k not in set(fold[tr])
        seen.append(te)
    assert np.array_equal(np.sort(np.concatenate(seen)), np.arange(len(groups)))
    with pytest.raises(AssertionError):
        splits.check_no_leak([0, 1], [1, 2], ["a", "b", "b"])
    # <NA> rows (vault) are in neither set; TRAIN_ONLY rows always train
    fold2 = pd.array([0, 1, splits.TRAIN_ONLY, None], dtype="Int64")
    parts = {k: (tr.tolist(), te.tolist()) for k, tr, te in splits.iter_folds(fold2)}
    assert parts == {0: ([1, 2], [0]), 1: ([0, 2], [1])}


def test_random_folds_match_sklearn_kfold():
    fold = splits.random_folds(103)
    want = np.empty(103, int)
    for k, (_, te) in enumerate(KFold(5, shuffle=True, random_state=156).split(np.zeros(103))):
        want[te] = k
    assert np.array_equal(fold, want)
    assert np.bincount(fold).min() >= 20


def test_time_split_and_loco():
    sem = pd.Series([1, 5, 6, 7, 3], index=[11, 12, 13, 14, 15])
    ts = splits.time_split(sem)
    assert ts.to_dict() == {11: -1, 12: -1, 13: 0, 14: 0, 15: -1}
    assert splits.time_split([2, 6]).tolist() == [-1, 0]
    with pytest.raises(ValueError):
        splits.time_split(pd.Series([1.0, np.nan]))
    courses = pd.Series(["A", "A", "B", "B", "C"], index=[1, 2, 3, 4, 5])
    bids = pd.Series({1: 10, 2: 6, 3: 3, 4: 4, 5: 40})
    lo = splits.loco(courses, bids)
    assert lo.attrs["courses"] == ["A", "C"]
    assert lo.to_dict() == {1: 0, 2: 0, 3: -1, 4: -1, 5: 1}
    parts = list(splits.iter_folds(lo.to_numpy()))
    assert [k for k, _, _ in parts] == [0, 1]
    assert parts[0][2].tolist() == [0, 1] and 2 in parts[0][1]   # course B always trains


def test_inner_groups_are_grouped_and_usable_as_cv():
    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(9), 30)
    cv = splits.inner_groups(groups)
    assert len(cv) == config.N_INNER_FOLDS
    for tr, va in cv:
        splits.check_no_leak(tr, va, groups)
    assert np.array_equal(np.sort(np.concatenate([va for _, va in cv])),
                          np.arange(groups.size))
    labels = splits.inner_fold_labels(groups)
    assert set(labels) == {0, 1, 2}
    X = rng.normal(size=(groups.size, 3))
    y = (X[:, 0] + rng.normal(size=groups.size) > 0).astype(int)
    pred = cross_val_predict(LogisticRegression(), X, y, cv=cv, method="predict_proba")
    assert pred.shape == (groups.size, 2)
    with pytest.raises(ValueError):
        splits.inner_groups([1, 1, 2])


# --------------------------------------------------------------------------- course filter


def _course_rows():
    # course "big": 1 class, 1000 rows; "wide": 3 classes x 1 row; "two": 2 classes
    course = ["big"] * 1000 + ["wide"] * 3 + ["two"] * 4
    cid = [1] * 1000 + [2, 3, 4] + [5, 5, 6, 6]
    return pd.DataFrame({"ctx_course": course, "class_id": cid})


def test_min_class_count_counts_classes_not_rows():
    df = _course_rows()
    tr = splits.MinClassCountCategories().fit(df)
    assert tr.n_classes_["ctx_course"].to_dict() == {"big": 1, "two": 2, "wide": 3}
    out = tr.transform(df)
    assert list(out.columns) == ["ctx_course"]                     # class_id dropped
    assert isinstance(out["ctx_course"].dtype, pd.CategoricalDtype)
    assert list(out["ctx_course"].cat.categories) == ["wide", "other"]
    assert out["ctx_course"].value_counts().to_dict() == {"other": 1004, "wide": 3}
    new = pd.DataFrame({"ctx_course": ["wide", "never_seen", None], "class_id": [9, 9, 9]})
    assert tr.transform(new)["ctx_course"].tolist()[:2] == ["wide", "other"]
    assert pd.isna(tr.transform(new)["ctx_course"].iloc[2])
    assert tr.get_feature_names_out().tolist() == ["ctx_course"]


def test_min_class_count_groups_kwarg_arrays_and_errors():
    df = _course_rows()
    tr = splits.MinClassCountCategories(min_classes=2).fit(df["ctx_course"].to_numpy(),
                                                           groups=df["class_id"])
    out = tr.transform(np.array(["two", "big", "wide"], dtype=object))
    assert out.ravel().tolist() == ["two", "other", "wide"]
    with pytest.raises(ValueError):
        splits.MinClassCountCategories().fit(df[["ctx_course"]])     # no class ids


def test_min_class_count_in_sklearn_pipelines():
    rng = np.random.default_rng(2)
    n = 600
    df = pd.DataFrame({"class_id": rng.integers(0, 40, n),
                       "x": rng.normal(size=n)})
    df["ctx_course"] = np.where(df["class_id"] < 30, "C" + (df["class_id"] % 3).astype(str),
                                "rare" + df["class_id"].astype(str))
    y = (df["x"] + (df["ctx_course"] == "C0") + rng.normal(size=n) > 0.5).astype(int)
    lin = Pipeline([
        ("cols", ColumnTransformer([
            ("course", Pipeline([("rare", splits.MinClassCountCategories()),
                                 ("oh", OneHotEncoder(handle_unknown="ignore"))]),
             ["ctx_course", "class_id"]),
            ("num", "passthrough", ["x"])])),
        ("lr", LogisticRegression())])
    lin.fit(df, y)
    oh = lin.named_steps["cols"].named_transformers_["course"].named_steps["oh"]
    assert list(oh.categories_[0]) == ["C0", "C1", "C2", "other"]
    assert lin.predict_proba(df).shape == (n, 2)
    rare = splits.MinClassCountCategories().fit(df[["ctx_course", "class_id"]])
    X = pd.concat([rare.transform(df[["ctx_course", "class_id"]]), df[["x"]]], axis=1)
    hgb = HistGradientBoostingClassifier(max_iter=5, categorical_features="from_dtype")
    hgb.fit(X, y)
    assert hgb.is_categorical_.tolist() == [True, False]
