# Step 6 — Cut each class into moments

Every 2 seconds becomes one row. Its answer is 1 if your hand goes up between g and H seconds later, 0 if not. The row is dropped if it can't count: your hand is already up, you're talking, or it's a breakout.

**Run** (from `pipeline1/`):

```
../.venv/bin/python -m oncue labels
```

**Time:** seconds

**In:** steps 4 and 5

**Out:** `data/tables/points.parquet`, with an answer column for every H and g

**Code:**

- `oncue/labels.py`: the grid, the exclusion rules, the answer columns

**When you review it, check that:**

- every rule uses only what was known at that moment
- answer columns exist for g = 1, 2, 3, 5 s
