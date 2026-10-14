# Pipeline 1: process so far, and the next step

*Handoff note, updated 15 Oct 2026 (night, after the full run of steps 4-5). Due Fri 16 Oct (closes Sun 18 Oct). Start a new chat with:
"Read `pipeline1/PROCESS.md` and `pipeline1/design/amendments.md`, then run the next step."*

## The question

Can a model tell, from what is being said in class and the class context, that I am about to
raise my hand? Every 2 s of a class it predicts "hand goes up within H seconds" (headline
H = 15 s, guard g = 2 s). Inputs are strictly causal: the transcript so far, who is speaking,
chat, polls, lesson segments, the minute of class, the course, my pre-class work, my own history.
Labels (y) come from the raised-hand icon on my tile in the recordings, found with a simple
image check. There is no face recognition in this pipeline.

## Where each step stands

| Step | Folder | Status |
|---|---|---|
| 1 Get data | `01_get_data/` | **Done on real data.** 739 classes synced from Forum (0 failures). |
| 2 Clean | `02_clean/` | **Done.** 739 → 477 usable classes. Reasons in `data/02_clean/`. |
| 3 Split | `03_split/` | **Done.** 289 train / 94 validation / 94 test across 26 courses (`data/03_split/split.csv`). |
| 4 Words + events | `04_words_events/` | **Done on all 477 classes (14 Oct).** 477 transcripts (Deepgram, upload mode); 474 with words: 98590, 101578 and 102735 (CP191/CP192) were recorded without sound. 457 pass the clock rule. |
| 5 Hand raises | `05_hand_raises/` | **Done on all 477 (14-15 Oct, rented pod).** 2,774 raises with a known start, all re-timed to the frame and with an audit clip (`pipeline1/audit/`); 1,295 called. QA: 446/477 classes pass (train 274, validation 86, test 86); 2,674 raises in passing classes. |
| 6–9 Moments, snapshots, train, score | `06_…` to `09_…` | Built and tested on synthetic data. **Next:** run them on the real tables. |

The pilot's metrics and the decisions to approve: `QUALITY.md`. Every question asked during
the pilot, with its answer: `QUESTIONS.md`. The pilot itself lives in `data/pilot/`
(gitignored; delete it once its PDFs and clips have been looked at).

## What the data showed

- **Size.** The 477 usable classes hold 103,005 voice events, 5,192 of them mine in the main
  room. Those merge into 4,286 speaking turns, 2,811 of them "fresh" (after 30+ s of silence).
  Forum doesn't log hand raises. Fresh turns are **only a rough guess at the size**
  (perhaps 2,500–4,000 raises), **never labels**: a fresh turn can be a cold call, a raise can
  go uncalled, and a turn starts seconds after the hand went up. The labels and their times
  come only from the hand icon in the video (step 5). The 2-s decision grid gives about 1.3M
  moments; the positive share is known once the real raises are counted.
  Moments within a class are correlated, so the class is the unit of evaluation: test is
  94 classes (about 590 fresh turns; the real number of raises comes from step 5).
- **Forum never logs the professor's microphone** (0 of 717 classes; amendment A12). Voice
  events are students only. Three fixes follow:
  1. *Clock sync:* the clock is fitted without the professor's voice cluster. A stress test
     on 360 simulated classes (`04_words_events/stress_test/`) put the old method 1–14 s off
     or found no fit at all; the new one is 0.03 s off.
  2. *Roles:* a word heard while no student mic is on is the professor's.
  3. *Professor turns:* they are added to the voice table as `inferred` rows, each marked with
     when it became knowable, so the "professor talking" features work and stay causal.
- Lesson segments are started by professors (93%) or staff, never by me: they are context only.

## Rules we settled

- **Privacy (public repo).** Classmates' events may be shown. Their identities may not: names,
  emails, avatars and student ids are masked. Grades and poll answers only ever show my own
  rows. Raw Forum data, transcripts, video and features stay local under `data/` (gitignored).
  Only `data/02_clean/` and `data/03_split/` are committed.
- **Secrets** live in `~/.minerva/.env` (FORUM_COOKIE, DEEPGRAM_API_KEY, ONCUE_ID_SALT), linked
  as `pipeline1/.env` and `~/Minervas_scribe/.env`. Pre-commit hooks in both repos refuse
  secrets. To refresh the cookie, copy it from DevTools, then run `~/.minerva/set_secret`.
- **Scribe stays my own repo.** The transcriber (Scribe v3: Deepgram, clock calibration,
  speaker fusion, the A12 professor fix) is merged into `main` of
  `Brady-Rutherford/Minervas_scribe` (PR #1, 13 Oct). The pipeline imports it from the local
  clone `~/Minervas_scribe`. Never merge it with any other transcriber repo.

## Steps 4 + 5 on every class: how it was run (14-15 Oct)

Done on a rented Prime Intellect pod (5×A40), see `pod/README.md`. What changed on the way, all
recorded in `QUESTIONS.md`:

- **Two label fixes** (`oncue/strip.py`): a classmate's label must not pass as Brady's, neither by
  OCR (`ME_NAME_MIN = 0.75`) nor by a near look-alike patch (`ME_SAME_LABEL_NCC = 0.95`). Classes
  scanned before a fix with long unread stretches were scanned again (old results in
  `data/rescan_backup*/`).
- **GPU decoding** (`ONCUE_HWACCEL=cuda`): bit-identical frames, ~4x less CPU for ffmpeg.
- **QA rules (Brady, 14 Oct):** clock limits 2.5 s (max residual and p95; was 1.5 s); the ring
  check is judged over the frames where his tile is read (`oncue ringcheck`), so breakouts and
  unread stretches never drop a class, a wrong tile does; coverage >= 30% stays.

The original instructions follow.

## (Original) Next step: run steps 4 + 5 on every class

One command per split, train first. Each class: Deepgram transcript (upload mode), words,
events, the strip scan, tracking, episodes, re-timing, audit clips, QA:

```
cd ~/cs156-ml/pipeline1
caffeinate -i ../.venv/bin/python -m oncue ingest --split train --workers 6
caffeinate -i ../.venv/bin/python -m oncue ingest --split validation --workers 6
caffeinate -i ../.venv/bin/python -m oncue ingest --split test --workers 6
../.venv/bin/python -m oncue ingest-report      # checks A-E per class -> data/logs/ingest_report.csv
../.venv/bin/python -m oncue pdf                # readable PDF per class (local only)
```

- On this Mac: about 3–4 h of transcription and ~14 h of scanning (34 classes/hour). It must
  stay plugged in with the lid open; `caffeinate` stops idle sleep.
- On a rented CPU pod (e.g. Prime Intellect GH200: 72 Arm cores, $1.99/h): about 2–3 h. Test one
  class first: the Forum cookie may not work from a datacenter IP.
- A throwaway run: `--work-dir data/pilot` (everything it writes stays there); `--sample N`
  takes a seeded sample spread over terms.

## Still open

- **The project's name.** The code still says `oncue`. Renaming touches the package, the CLI,
  `ONCUE_ID_SALT`, the site and the READMEs.
- **The fixed split in `oncue/experiments.py`.** It still uses rotating folds and a vault;
  switch it (and `--phase`) to `data/03_split/split.csv`. This is a small separate diff.
- **Consent email** to Prof. Watson (Deepgram; data from other courses).
- **Left out of the PR on purpose:** local changes to the Session 2 notebook (they delete the
  overfitting analysis from `57fff38` and look like an accidental revert; restore them with
  `git checkout -- "PCW/Session 2 - Linear Algebra 1 - Tensors, Classification, and Regression/drawing-lines.ipynb"`)
  and to `.vscode/settings.json`.
- **Known limits:**
  - Whole-second (2023) timestamps sync to about ±0.5 s.
  - Professor/student labels drop to about 93% when mics are noisy (simulated rates).
  - Groq/local Whisper can't sync the clock.
