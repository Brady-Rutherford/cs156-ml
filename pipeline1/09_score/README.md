# Step 9 — Score it honestly

Turns predictions into numbers that mean something: average precision against the base rate, how many raises it warned about at a fixed false-alarm rate and how early, with uncertainty from resampling whole classes. It writes the committed results (hashed ids, no words), and a scan refuses to finish if any name appears.

**Run** (from `pipeline1/`):

```
../.venv/bin/python -m oncue evaluate
../.venv/bin/python -m oncue figures
../.venv/bin/python -m oncue results-bundle
../.venv/bin/python -m oncue privacy
```

**Time:** minutes

**In:** step 8 predictions

**Out:** `results/` and `figures/` (the only things meant to be committed)

**Code:**

- `oncue/metrics.py`: the scores, the alarm rule, the bootstrap
- `oncue/plots.py, oncue/report.py`: figures and tables
- `oncue/privacy.py`: the results bundle and the name scan

**When you review it, check that:**

- an always-on model can't game the alarm score (there's a test)
- nothing committed contains a word or a real id
