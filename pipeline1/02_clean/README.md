# Step 2 — Clean the data

Sorts every synced class into exactly one of three files. First, whole courses are set aside: summer-program courses (every class in June–August) and courses with fewer than 5 classes listed (one-off sessions, workshops). Then single classes are removed if they weren't recorded, you weren't enrolled, Forum marks you absent, or there's no attendance data. Everything left is usable. Later this step will also fix the event logs (key spellings, voice start times, breakout speech, system chat messages) and replace classmates with codes.

**Run** (from `pipeline1/`):

```
../.venv/bin/python 02_clean/clean.py
```

**Time:** a few seconds

**In:** `data/scribe/raw/*/class.json` from step 1

**Out (committed):** in `data/02_clean/`: `classes_clean.csv` (usable classes, the input to step 3), `removed_classes.csv` (with the reason), `special_courses.csv` (summer and one-off courses, with the reason), `cleaning_report.csv` (counts)

**Code:**

- `02_clean/clean.py`: the usability rules and the report
- `oncue/loaders.py, oncue/events.py, oncue/qa.py`: event-level cleaning, to be moved under this step

**When you review it, check that:**

- no class is removed because of whether you raised your hand; that would bias the data
- every removed class appears in the report with its reason
