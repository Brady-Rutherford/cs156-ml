"""Brady's course split: absences out, then 60/20/20 within every course."""

import pandas as pd
from oncue.splits import course_split


def _classes(counts: dict[str, int], absent: set[int] = frozenset()) -> pd.DataFrame:
    rows, cid = [], 1000
    for course, n in counts.items():
        for _ in range(n):
            cid += 1
            rows.append({"class_id": cid, "course_code": course, "usable": cid not in absent})
    return pd.DataFrame(rows)


def test_every_course_is_in_all_three_splits_at_60_20_20():
    out = course_split(_classes({"CX50": 25, "FA51": 21, "PR51": 8}))
    counts = out.groupby(["course_code", "split"]).size().unstack(fill_value=0)
    assert counts.loc["CX50"].to_dict() == {"test": 5, "train": 15, "validation": 5}
    assert counts.loc["FA51"].to_dict() == {"test": 4, "train": 13, "validation": 4}
    assert counts.loc["PR51"].to_dict() == {"test": 2, "train": 4, "validation": 2}


def test_absent_classes_are_removed_before_splitting():
    classes = _classes({"CX51": 25}, absent={1001, 1002, 1003})
    out = course_split(classes)
    assert set(out.loc[out["class_id"].isin([1001, 1002, 1003]), "split"]) == {"excluded"}
    kept = out[out["split"] != "excluded"]
    assert len(kept) == 22 and kept["split"].value_counts().to_dict() == {"train": 14, "test": 4, "validation": 4}


def test_tiny_courses_are_excluded_with_a_reason():
    out = course_split(_classes({"GL91": 2, "MC50": 25}))
    gl = out[out["course_code"] == "GL91"]
    assert set(gl["split"]) == {"excluded"} and gl["reason"].str.contains("2 usable").all()


def test_split_is_deterministic_and_seed_dependent():
    classes = _classes({"EA50": 23})
    a, b = course_split(classes), course_split(classes)
    assert a.equals(b)
    assert not a["split"].equals(course_split(classes, seed=7)["split"])


def test_same_day_classes_stay_together():
    classes = _classes({"MC51": 20})
    classes["cv_group"] = classes["class_id"]
    classes.loc[classes["class_id"].isin([1005, 1006]), "cv_group"] = 1005   # taught the same day
    out = course_split(classes)
    assert out.loc[out["class_id"] == 1005, "split"].item() == out.loc[out["class_id"] == 1006, "split"].item()
