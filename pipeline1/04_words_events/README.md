# Step 4 — Words and events on one clock

Transcribes each class with Scribe v3 (Deepgram), then builds a table of every word with its time and speaker. Speakers come from Forum's own voice log as known at the time. Forum's events (voice, chat, lesson segments, polls) become tables on the same video clock.

Forum logs students' microphones only, never the professor's (0 of 717 classes; amendment A12). So a word heard while no student mic is on is the professor's, and the professor's turns are added to the voice table as `inferred` rows. Each row records when it became knowable.

**Run** (from `pipeline1/`):

```
../.venv/bin/python -m oncue transcribe
../.venv/bin/python -m oncue words
../.venv/bin/python -m oncue events
```

**Time:** about a minute of Deepgram time per class, run several at once

**In:** steps 1–3, `DEEPGRAM_API_KEY`

**Out:** `data/transcripts/` (Scribe's `*_ids.json` and `*_asr.json`); `data/tables/words.parquet`, `voice`, `chat`, `segments`, `polls`

**Code:**

- `oncue/transcribe.py`: runs Scribe; builds the words table
- `oncue/events.py`: voice, chat, segment and poll tables
- `stress_test/`: clock-sync and professor-labelling stress tests on simulated classes (`stress_sync.py 40`, `stress_prof.py`)

**When you review it, check that:**

- a voice event's start = its timestamp − its duration (the timestamp marks the end)
- classes whose clocks can't be lined up are flagged, never used silently
- words with no student mic event are `prof`, and their role is available only at word end + 0.25 s + the Forum delay
- the clock fit leaves out the professor's voice cluster (`calibration_words`); `stress_test/stress_sync_summary.csv` shows why
