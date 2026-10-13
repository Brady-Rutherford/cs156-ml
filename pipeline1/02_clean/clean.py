"""Step 2: clean the class list. Every synced class ends up in exactly one of three files.

Rules, in order (the first one a class meets decides where it goes):
  1. Course-level, set aside in special_courses.csv:
     - "summer program"    every class of the course falls in June-August
     - "too few classes"   the course lists fewer than 5 classes (one-off sessions, workshops)
  2. Class-level, recorded in removed_classes.csv:
     - "TA section (not a student)"  sections Brady teaches as a TA (TA_SECTIONS)
     - "not recorded"      no recording exists
     - "not enrolled"      Brady isn't on the roster (e.g. sections he only visited)
     - "absent"            Forum marks him absent
     - "no attendance data"  Forum has no attendance flag for him
  3. Everything else is usable and goes to classes_clean.csv, the input to step 3.

Writes (committed with the assignment) to data/02_clean/:
  classes_clean.csv, removed_classes.csv, special_courses.csv, cleaning_report.csv

Run from pipeline1/:  ../.venv/bin/python 02_clean/clean.py
"""

import json
from pathlib import Path

import pandas as pd

PIPELINE = Path(__file__).resolve().parents[1]
RAW = PIPELINE / "data" / "scribe" / "raw"
ME = PIPELINE / "data" / "scribe" / "me.json"
OUT = PIPELINE / "data" / "02_clean"

MIN_CLASSES_PER_COURSE = 5
SUMMER_MONTHS = (6, 7, 8)
TA_SECTIONS = {13519: "B145 Fall 2026"}   # Brady is the TA here, not a student (13 Oct)


def term_of(date: str) -> str:
    """'2024-01-09' -> 'Spring 2024'. Forum stores past terms with dates but no name."""
    if not date:
        return ""
    year, month = int(date[:4]), int(date[5:7])
    season = "Spring" if month <= 5 else "Summer" if month in SUMMER_MONTHS else "Fall"
    return f"{season} {year}"


def class_row(klass: dict, listing: dict, me_id: int) -> dict:
    course = (klass.get("section") or {}).get("course") or {}
    sessions = klass.get("recording-sessions") or []
    started = min((s.get("recording-started") or "" for s in sessions), default="")
    date = (started or listing.get("date") or "")[:10]
    me = [cu for cu in klass.get("class-users") or [] if (cu.get("user") or {}).get("id") == me_id]
    if (klass.get("section") or {}).get("id") in TA_SECTIONS:
        why = "TA section (not a student)"
    elif not sessions:
        why = "not recorded"
    elif not me:
        why = "not enrolled"
    elif me[0].get("absent") is True:
        why = "absent"
    elif me[0].get("absent") is None:
        why = "no attendance data"
    else:
        why = ""
    return {
        "class_id": klass["id"],
        "course_code": course.get("course-code") or "?",
        "course_title": (course.get("title") or "").strip(),
        "term": term_of(date),
        "date": date,
        "title": klass.get("title", ""),
        "why_removed": why,
    }


def special_course_reasons(classes: pd.DataFrame) -> pd.Series:
    """Course code -> reason, for courses set aside as a whole (rule 1)."""
    reasons = {}
    for code, grp in classes.groupby("course_code"):
        months = pd.to_datetime(grp["date"], errors="coerce").dt.month.dropna()
        if len(months) and months.isin(SUMMER_MONTHS).all():
            reasons[code] = "summer program"
        elif len(grp) < MIN_CLASSES_PER_COURSE:
            reasons[code] = f"too few classes ({len(grp)} listed)"
    return pd.Series(reasons, dtype="object")


def main() -> None:
    me_id = json.loads(ME.read_text())["id"]
    rows = []
    for p in sorted(RAW.glob("*/class.json")):
        listing_path = p.parent / "listing.json"
        listing = json.loads(listing_path.read_text()) if listing_path.exists() else {}
        rows.append(class_row(json.loads(p.read_text()), listing, me_id))
    classes = pd.DataFrame(rows).sort_values(["course_code", "date", "class_id"])

    special = special_course_reasons(classes)
    is_special = classes["course_code"].isin(special.index)
    special_courses = classes[is_special].assign(reason=lambda d: d["course_code"].map(special))
    rest = classes[~is_special]
    removed = rest[rest["why_removed"] != ""].rename(columns={"why_removed": "reason"})
    clean = rest[rest["why_removed"] == ""].drop(columns="why_removed")

    report = pd.concat([
        special_courses.groupby("reason").size(),
        removed.groupby("reason").size(),
        pd.Series({"usable (to step 3)": len(clean), "total synced": len(classes)}),
    ]).rename_axis("reason").reset_index(name="classes")

    OUT.mkdir(parents=True, exist_ok=True)
    for stale in ("classes.csv",):                      # earlier single-file version
        (OUT / stale).unlink(missing_ok=True)
    clean.to_csv(OUT / "classes_clean.csv", index=False)
    removed.to_csv(OUT / "removed_classes.csv", index=False)
    special_courses.drop(columns="why_removed").to_csv(OUT / "special_courses.csv", index=False)
    report.to_csv(OUT / "cleaning_report.csv", index=False)

    print(report.to_string(index=False))
    print("\nset aside as special:", ", ".join(f"{c} ({r})" for c, r in special.sort_index().items()))
    print(f"\nwrote {OUT}/classes_clean.csv, removed_classes.csv, special_courses.csv, cleaning_report.csv")


if __name__ == "__main__":
    main()
