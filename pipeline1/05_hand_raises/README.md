# Step 5 — Find my hand raises (the labels)

Streams each recording, decodes only the strip of tiles, finds your tile, and checks its top-left corner for the yellow-green hand at 2 frames a second. A state machine turns frames into hand raises, separates them from class-wide votes, and re-times each one at full frame rate.

**Run** (from `pipeline1/`):

```
../.venv/bin/python -m oncue scan
../.venv/bin/python -m oncue track
../.venv/bin/python -m oncue episodes
../.venv/bin/python -m oncue retime
../.venv/bin/python -m oncue qa
```

**Time:** about 3–5 minutes per class per worker; no video is stored

**In:** signed recording links (fetched per class), step 4's voice table

**Out:** `data/tables/hand_tiles`, `hand_frames`, `hand_episodes.parquet`

**Code:**

- `oncue/hands.py`: the strip, tile tracking, the colour test, the state machine, re-timing
- `oncue/qa.py`: per-class quality checks

**When you review it, check that:**

- hand (hue 52–80°) vs poll tick (120–165°) in the same corner
- you hand-check a sample of detected raises before trusting the labels
