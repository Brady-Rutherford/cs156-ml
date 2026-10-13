# Step 3 — Split into train, validation and test

Splits whole classes, never single moments. Within each course, usable classes are put in a fixed random order (a seeded hash of the class id): the first 20% go to test, the next 20% to validation, the rest to train. Every course appears in all three. Courses with fewer than 5 usable classes are left out, and classes taught the same day stay together.

**Run** (from `pipeline1/`):

```
../.venv/bin/python 03_split/split.py
```

**Time:** a few seconds

**In:** `data/02_clean/classes.csv` from step 2

**Out (committed):** `data/03_split/split.csv` (each class → train / validation / test / excluded) and `data/03_split/split_by_course.csv`

**Code:**

- `03_split/split.py`: reads step 2's table and writes the split
- `oncue/splits.py → course_split`: the rule itself, with tests in tests/test_course_split.py

**When you review it, check that:**

- every course is in all three splits
- the test classes are touched only once, at the very end
- the same seed always gives the same split
