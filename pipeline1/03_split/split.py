"""Step 3: split each course's usable classes 60/20/20 into train, validation and test.

The rule is `oncue.splits.course_split` (Brady's rule, amendment A10): within each course, a
seeded random order; first 20% test, next 20% validation, rest train; courses with fewer
than 5 usable classes are excluded (rare after step 2). Writes (committed with the assignment):
  data/03_split/split.csv            every class -> train / validation / test / excluded
  data/03_split/split_by_course.csv  counts per course

Run from pipeline1/, after step 2:  ../.venv/bin/python 03_split/split.py
"""

import sys
from pathlib import Path

import pandas as pd

PIPELINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE))
from oncue.splits import course_split

CLASSES = PIPELINE / "data" / "02_clean" / "classes_clean.csv"
OUT = PIPELINE / "data" / "03_split"


def main() -> None:
    classes = pd.read_csv(CLASSES).assign(usable=True)    # step 2 already removed the rest
    split = course_split(classes)
    out = classes.drop(columns="usable").merge(split[["class_id", "split", "reason"]], on="class_id")
    out = out.rename(columns={"reason": "why_excluded"})

    by_course = (out[out["split"] != "excluded"].groupby(["course_code", "split"]).size()
                 .unstack(fill_value=0).reindex(columns=["train", "validation", "test"], fill_value=0))
    by_course["total"] = by_course.sum(axis=1)
    too_small = out.loc[out["why_excluded"].str.startswith("course has", na=False), "course_code"].unique()

    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "split.csv", index=False)
    by_course.to_csv(OUT / "split_by_course.csv")
    print(by_course.to_string())
    print(f"\nsplit totals: {out['split'].value_counts().to_dict()}")
    print(f"courses too small to split (< 5 usable classes): {', '.join(sorted(too_small)) or 'none'}")
    print(f"\nwrote {OUT / 'split.csv'}\n      {OUT / 'split_by_course.csv'}")


if __name__ == "__main__":
    main()
