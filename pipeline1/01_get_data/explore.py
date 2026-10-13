"""Step 1 exploration: what exactly did the Forum sync download?

Reads every file in data/scribe/raw/{class_id}/ (9 JSON files per class), flattens each into
a table (one row per record; nested objects become dotted columns like `event-data.duration`;
nested lists such as the roster become their own tables), and writes to 01_get_data/exploration/:

  tables_overview.csv   every table: which file it comes from, rows, columns
  fields.csv            every field (header): type, how often it's filled, distinct values, a real example
  event_types.csv       what kinds of events the logs contain, overall and in your own events
  raw_tables.sql        CREATE TABLE for every table, with 3 real example rows each

Privacy: everyone's events may be shown, but not who they are. Every person id (actor,
user, grader, target ...) that belongs to a student other than Brady reads <student>; Brady's
and professors' ids stay. Text that identifies or quotes classmates (names, emails, avatars,
chat text, grade comments, poll answers) reads <masked>. Forum also returns classmates' grades
and poll answers to a student account: for those tables and the roster, examples come only
from Brady's own rows. Every field is still listed.

Run from pipeline1/:  ../.venv/bin/python 01_get_data/explore.py
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

PIPELINE = Path(__file__).resolve().parents[1]
SCRIBE = PIPELINE / "data" / "scribe"
OUT = Path(__file__).resolve().parent / "exploration"
MASK = re.compile(r"(^|[-_.])(first|last|official)[-_](first[-_]|last[-_])?name$|(^|\.)name$|email|avatar|"
                  r"message[-_]text$|pronunciation|calendar[-_]url|(^|\.)comment$|(^|\.)response$")
PERSON_ID = re.compile(r"(actor|user)[-.]id$")   # actor-id, user.id, grader-user-id, target-user-id ...


def flatten(record: dict, table: str, tables: dict, parent: dict, prefix: str = "") -> dict:
    """Nested dicts -> dotted columns; lists of dicts -> child tables keyed to the parent."""
    row = {}
    for key, value in record.items():
        col = f"{prefix}{key}"
        if isinstance(value, dict):
            row.update(flatten(value, table, tables, parent, prefix=f"{col}."))
        elif isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
            child = f"{table}.{col}"
            for item in value:
                tables[child].append({**parent, **flatten(item, child, tables, parent)})
            row[f"{col}.count"] = len(value)
        elif isinstance(value, list):
            row[col] = json.dumps(value)
        else:
            row[col] = value
    return row


def load_tables() -> dict[str, pd.DataFrame]:
    tables = defaultdict(list)
    for cdir in sorted(p for p in (SCRIBE / "raw").iterdir() if p.is_dir()):
        class_id = int(cdir.name)
        for f in sorted(cdir.glob("*.json")):
            if f.name.startswith("_"):
                continue
            name = f.stem
            data = json.loads(f.read_text())
            records = data if isinstance(data, list) else [data]
            for rec in records:
                if isinstance(rec, dict):
                    tables[name].append({"class_id": class_id, **flatten(rec, name, tables, {"class_id": class_id})})
    sections = json.loads((SCRIBE / "sections.json").read_text())
    for s in sections if isinstance(sections, list) else sections.get("sections", []):
        tables["sections (sections.json)"].append(flatten(s, "sections", tables, {}))
    out = {name: pd.DataFrame(rows) for name, rows in tables.items()}
    out["manifest (manifest.csv)"] = pd.read_csv(SCRIBE / "manifest.csv")
    return out


def own_rows(df: pd.DataFrame, me_id: int) -> pd.DataFrame:
    """Rows about one person (grades, poll answers, roster entries): keep only Brady's own."""
    for col in ("target-user-id", "user-id", "user.id"):
        if col in df.columns:
            return df[df[col] == me_id]
    return df


def is_student_id(value, field: str, shown: frozenset) -> bool:
    """A person id that isn't Brady's or a professor's/staff member's."""
    if not PERSON_ID.search(field):
        return False
    try:
        return int(float(value)) not in shown
    except (TypeError, ValueError):
        return False


def example(series: pd.Series, field: str, shown: frozenset) -> str:
    vals = series.dropna()
    vals = vals[vals.astype(str).str.len() > 0]
    if vals.empty:
        return ""
    if MASK.search(field):
        return "<masked>"
    if is_student_id(vals.iloc[len(vals) // 2], field, shown):
        return "<student>"
    text = str(vals.iloc[len(vals) // 2])
    return text if len(text) <= 80 else text[:77] + "…"


def sqltype(series: pd.Series) -> str:
    kind = pd.api.types.infer_dtype(series, skipna=True)
    return {"integer": "BIGINT", "floating": "DOUBLE", "mixed-integer-float": "DOUBLE",
            "boolean": "BOOLEAN"}.get(kind, "TEXT")


def sql_literal(value, field: str, shown: frozenset) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "NULL"
    if MASK.search(field):
        return "'<masked>'"
    if is_student_id(value, field, shown):
        return "'<student>'"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return repr(value)
    text = str(value)
    text = text if len(text) <= 60 else text[:57] + "…"
    return "'" + text.replace("'", "''") + "'"


def main() -> None:
    tables = load_tables()
    me_id = json.loads((SCRIBE / "me.json").read_text())["id"]
    roster = tables["class.class-users"]
    staff = roster.loc[roster["role"].fillna("student") != "student", "user.id"]
    shown = frozenset([me_id, *tables["listing"]["class.teacher-user-id"].dropna().astype(int),
                       *staff.dropna().astype(int)])
    OUT.mkdir(exist_ok=True)

    overview, fields, sql = [], [], ["-- What the Forum sync downloaded (real data, examples masked where they identify classmates).",
                                     "-- Generated by 01_get_data/explore.py from data/scribe/.", ""]
    for name in sorted(tables):
        df = tables[name]
        overview.append({"table": name, "rows": len(df), "columns": df.shape[1],
                         "classes": df["class_id"].nunique() if "class_id" in df else ""})
        for col in df.columns:
            s = df[col]
            filled = int(s.notna().sum())
            fields.append({"table": name, "field": col, "type": sqltype(s), "filled_rows": filled,
                           "filled_pct": round(100 * filled / max(len(df), 1), 1),
                           "distinct": int(s.astype(str).nunique()) if filled else 0,
                           "example": example(own_rows(df, me_id)[col], col, shown)})
        cols = list(df.columns)
        sql.append(f"-- {name}: {len(df):,} rows")
        sql.append(f'CREATE TABLE "{name}" (\n' + ",\n".join(f'    "{c}" {sqltype(df[c])}' for c in cols) + "\n);")
        mine = own_rows(df, me_id)
        sample = mine.sample(min(3, len(mine)), random_state=156) if len(mine) else mine
        if len(sample):
            rows = ",\n".join("    (" + ", ".join(sql_literal(r[c], c, shown) for c in cols) + ")" for _, r in sample.iterrows())
            sql.append(f'INSERT INTO "{name}" VALUES\n{rows};')
        sql.append("")

    ev = tables["events"]
    types = ev.groupby("event-type").size().rename("all_events")
    mine = ev[ev["actor-id"] == me_id].groupby("event-type").size().rename("my_events")
    event_types = pd.concat([types, mine], axis=1).fillna(0).astype(int).sort_values("all_events", ascending=False)

    pd.DataFrame(overview).to_csv(OUT / "tables_overview.csv", index=False)
    pd.DataFrame(fields).to_csv(OUT / "fields.csv", index=False)
    event_types.to_csv(OUT / "event_types.csv")
    (OUT / "raw_tables.sql").write_text("\n".join(sql))

    print(pd.DataFrame(overview).to_string(index=False))
    print()
    print(event_types.to_string())
    print(f"\nwrote {OUT}/tables_overview.csv, fields.csv, event_types.csv, raw_tables.sql")


if __name__ == "__main__":
    main()
