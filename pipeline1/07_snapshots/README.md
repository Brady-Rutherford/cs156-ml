# Step 7 — Describe each moment from the past only

Builds each moment's snapshot: the last 60 s of words (names masked, punctuation removed), the class state, your own history, and how close the talk is to your workbook answers. A test deletes everything after each moment and checks the features don't change.

**Run** (from `pipeline1/`):

```
../.venv/bin/python -m oncue pcw
../.venv/bin/python -m oncue frames
../.venv/bin/python -m oncue embed
../.venv/bin/python -m oncue features
```

**Time:** minutes; sentence embeddings use the M4's GPU

**In:** steps 1, 4 and 6

**Out:** `data/tables/frames.parquet`, `features.parquet`; `data/cache/emb/`

**Code:**

- `oncue/text.py`: masking, cleaning, 60-second windows, question detector
- `oncue/pcw.py`: workbook questions and your answers
- `oncue/embed.py, oncue/edu.py`: embeddings; Stanford's classroom classifiers
- `oncue/causal.py, oncue/features.py`: the past-only cut and the feature builder

**When you review it, check that:**

- the truncation test passes on real data
- names are masked before anything is saved
