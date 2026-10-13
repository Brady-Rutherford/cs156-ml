# Pipeline 1: process so far, and the next step

*Handoff note, 13 Oct 2026. Due Fri 16 Oct (closes Sun 18 Oct). Start a new chat with:
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
| 1 Get data | `01_get_data/` | **Done on real data.** 739 classes synced from Forum (0 failures). `exploration/` lists every table and field with masked examples. Row counts were checked against an independent recount. |
| 2 Clean | `02_clean/` | **Done.** 739 → 477 usable. Set aside: summer program 77, too few classes 17, TA section (B145 Fall 2026) 10, absent 76, not enrolled 73, not recorded 8, no attendance data 1. Reasons are in `data/02_clean/`. |
| 3 Split | `03_split/` | **Done.** Each course split 60/20/20 by a seeded hash: 289 train / 94 validation / 94 test across 26 courses. `data/03_split/split.csv` |
| 4 Words + events | `04_words_events/` | Code built and tested on synthetic classes. **Not run on real classes yet: this is the next step.** |
| 5 Hand raises | `05_hand_raises/` | Code built and tested on synthetic video only. The detector's colour and size settings are placeholders until checked on real frames. **Runs together with step 4 on the first class.** |
| 6–9 Moments, snapshots, train, score | `06_…` to `09_…` | Code built and tested. Run after step 5. |

The code is one package, `oncue/` (to be renamed once the name is chosen); tests are in
`tests/` (600 pass). The design is `design/on_cue_spec.md`. `design/amendments.md` (A1–A12)
overrides the spec.

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
- **Scribe stays my own repo.** The transcriber upgrade (Scribe v3: Deepgram, clock
  calibration, speaker fusion, the A12 professor fix) is on the local `v3` branch of
  `~/Minervas_scribe` as new files. The pipeline imports it from there. Nothing there has been
  committed or pushed; it's a public repo, so it needs an explicit go-ahead.

## Next step: transcripts and exact hand-raise times on one clock (steps 4 + 5)

The data isn't complete until each class has three things on the **same clock**:
1. the transcript (Deepgram, step 4)
2. the Forum events (step 4)
3. the exact moment the hand icon appears on my tile in the video (step 5)

Getting that alignment right is the hardest part of the data work. Steps 4 and 5 are separate
folders, but on the first class they run together and are checked together.

**The clock is the video's own time.**

| Source | How it gets onto video time | What can go wrong |
|---|---|---|
| Transcript words | Deepgram reads the audio of the same recording file (lowres, `strict_res`), so its times are already video times | an A/V offset inside the recording: `av_lag` checks it by comparing the green speaking border on my tile with when the transcript has me talking |
| Hand icon | frames are read with their own timestamps (ffmpeg pts − audio start). Scanned at 2 fps, then each raise is re-read at full frame rate (`retime`) to find its first frame | missed or false detections (poll ticks, class-wide votes); the delay between my key press and the icon appearing; the placeholder settings |
| Forum events | wall clock → video time through the recording sessions, plus the calibrated offset (A12) | calibration error: ~0.03 s on ms timestamps, ~0.5 s on whole seconds (2023); recording restarts |


**0. Before the first run**

- The cookie must work. `~/Minervas_scribe/.venv/bin/scribe me` should print your name; it did
  on 13 Oct.
- **Choose which courses may use Deepgram.** The pipeline sends a course to Deepgram only if it
  appears in `ASR_PROVIDER_BY_COURSE` in `oncue/config.py`. That map is empty now, so
  everything would run on local Whisper. Don't batch-run local Whisper: it can't tell voices
  apart, and the stress test shows the clock can't be synced without that. Deepgram audio
  leaves the machine (EU endpoint, model-improvement opt-out on), so clear the course first.
  Emailing Prof. Watson covers CS156. Then set:
  ```python
  ASR_PROVIDER_BY_COURSE: dict[str, str] = {"CS156": "deepgram"}
  ```
- Cost is roughly $0.40 per 90-minute class (check the Deepgram console). Time is about a
  minute per class.

**1. One class end to end:** CS156 Session 2 (class 104619, train split)

```
cd ~/cs156-ml/pipeline1
../.venv/bin/python -m oncue transcribe --class 104619
../.venv/bin/python -m oncue words --class 104619
../.venv/bin/python -m oncue events --class 104619
```

Then scan the same class for the hand icon:

```
../.venv/bin/python -m oncue scan --class 104619
../.venv/bin/python -m oncue track --class 104619
../.venv/bin/python -m oncue episodes --class 104619
../.venv/bin/python -m oncue retime --class 104619
../.venv/bin/python -m oncue qa --class 104619
```

Then check that it worked:

```
../.venv/bin/python - <<'EOF'
import pandas as pd
t = pd.read_parquet("data/tables/class_timing.parquet")
print(t.T)                                     # cal_qa_pass True, method windowed, offset
w = pd.read_parquet("data/tables/words.parquet")
print(w["role"].value_counts(normalize=True))  # prof should be the largest share
print(w["how"].value_counts(normalize=True))   # teacher-cluster large, "none" small
v = pd.read_parquet("data/tables/voice.parquet")
print(v.groupby(["inferred", "role"]).size())  # inferred rows = professor turns
e = pd.read_parquet("data/tables/hand_episodes.parquet")
print(e[e.class_id == 104619])                 # every raise: onset, re-timed onset, outcome
EOF
```

**Check the hand times by eye before trusting any label.** For this class:
- Watch a short clip around every detected raise (`hands.onset_clips` writes them to
  `audit/`, with every tile except mine pixelated). Is the icon really mine, and does the
  re-timed onset land on the first frame it shows?
- Look for raises the scan missed: go through the transcript for moments you remember
  raising your hand.
- Check alignment against the transcript. After each raise, does the professor call on you
  (your name, "go ahead") and do your words follow a few seconds later? The gap should be
  steady from raise to raise; a drifting gap means a clock problem.
- Look at `av_lag` for this class. It should be small and steady.
- Count the raises against the fresh turns: how many raises went uncalled, and how many
  turns were cold calls? This replaces the rough size guess above.
- If the colour or size settings miss or misfire, fix the placeholder values in `hands.py`
  (`UiSpec`, `Detector`) and rerun the scan.


Scribe's readable transcript is `data/transcripts/*class104619_ids.json`. Skim it against your
memory of the class. Look for: the professor labelled as such, your own turns where you spoke,
"Unattributed" rare. This is the first real check of A12. It also shows the real rates of mic
noise and of speech with no mic event, which the stress test could only guess.

**2. The rest:** once the one class's transcript **and** hand times look right, and more courses
are cleared, run steps 4 and 5 on every class in `data/03_split/split.csv` whose split isn't
`excluded`. Each course uses the provider its map entry says.

```
ids=$(../.venv/bin/python -c "import pandas as pd; s = pd.read_csv('data/03_split/split.csv'); print(' '.join(f'--class {c}' for c in s.loc[s.split != 'excluded', 'class_id']))")
../.venv/bin/python -m oncue transcribe $ids --workers 4
../.venv/bin/python -m oncue words
../.venv/bin/python -m oncue events
for c in scan track episodes retime qa; do ../.venv/bin/python -m oncue $c $ids --workers 4; done
```

(`--phase` still selects the old rotating phases; use the class list above until it is switched
to `split.csv`.)

**3. After that:** steps 6–9 (moments, snapshots, training, scoring).

## Still open

- **The project's name.** The code still says `oncue`. Renaming touches the package, the CLI,
  `ONCUE_ID_SALT`, the site and the READMEs.
- **The fixed split in `oncue/experiments.py`.** It still uses rotating folds and a vault;
  switch it (and `--phase`) to `data/03_split/split.csv`. This is a small separate diff.
- **Consent email** to Prof. Watson (Deepgram; data from other courses).
- **PR history.** Commit `c252b62` still contains unmasked student ids in the exploration
  examples (fixed in `f90a93f`). Rewriting the branch before merging would remove them from
  GitHub.
- **Left out of the PR on purpose:** local changes to the Session 2 notebook (they delete the
  overfitting analysis from `57fff38` and look like an accidental revert; restore them with
  `git checkout -- "PCW/Session 2 - Linear Algebra 1 - Tensors, Classification, and Regression/drawing-lines.ipynb"`)
  and to `.vscode/settings.json`.
- **Known limits:**
  - Whole-second (2023) timestamps sync to about ±0.5 s.
  - Professor/student labels drop to about 93% when mics are noisy (simulated rates).
  - Groq/local Whisper can't sync the clock.
