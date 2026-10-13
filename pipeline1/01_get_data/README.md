# Step 1 — Get the data

Downloads every class's records from Forum with your login: the class record and roster (including attendance), the event log (who spoke when, chat, lesson segments), polls and your poll answers, and the recording times. It only reads, and it downloads no video.

**Run** (from `pipeline1/`):

```
../.venv/bin/python -m oncue harvest --sync-only
```

**Time:** about 37 minutes for all ~739 listed classes. It sends 3 requests a second on purpose, so Forum isn't hammered. If interrupted, it resumes where it stopped.

**In:** your Forum cookie in `~/.minerva/.env`

**Out:** `data/scribe/raw/{class_id}/` (9 JSON files per class), plus `manifest.csv`, `me.json` and `sections.json` in `data/scribe/`

**Explore what was downloaded** (writes `01_get_data/exploration/`, committed):

```
../.venv/bin/python 01_get_data/explore.py
```

- `tables_overview.csv`: the 20 tables inside the 9 files per class, with row and column counts
- `fields.csv`: every header in every table, how often it's filled, distinct values, a real example
- `event_types.csv`: voice / chat / lesson-segment events, overall and your own
- `raw_tables.sql`: `CREATE TABLE` for every table, with 3 real example rows each

Privacy: Forum also gives a student account classmates' grades (outcome assessments) and poll
answers. Examples for those tables and the roster come only from your own rows, and names,
chat text, grade comments and poll answers are masked. The pipeline never reads those files.

**Code:**

- `oncue/harvest.py`: the sync, the class page, the pre-class workbook
- `oncue/loaders.py`: later steps read only whitelisted fields from these files

**When you review it, check that:**

- it only reads (GET). The one exception is the POST Forum itself sends to open your workbook.
- it never calls `/api/v1/class-users/` (where pressing H is saved)
- nothing here is committed: `data/` is gitignored
