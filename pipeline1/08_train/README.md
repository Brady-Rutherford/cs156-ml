# Step 8 — Train and compare models

Fits the models from simplest to richest (base rate, question rule, logistic regression, naive Bayes, TF-IDF, gradient boosting, embeddings) and runs the comparisons: words vs no words, with vs without your workbook, every H and g.

**Run** (from `pipeline1/`):

```
../.venv/bin/python -m oncue train
../.venv/bin/python -m oncue controls
```

**Time:** minutes per comparison

**In:** step 7 features, step 6 answers, step 3 split

**Out:** `data/oof/{comparison}.parquet`: every predicted probability

**Code:**

- `oncue/models.py, oncue/experiments.py`: the models and the comparison runner
- `oncue/splits.py`: train / validation / test
- `oncue/gru.py, oncue/xenc.py, oncue/controls.py`: optional extras

**When you review it, check that:**

- it still uses rotating groups; switching it to step 3's fixed split is the next change for review
- settings are tuned only on validation classes, never test
