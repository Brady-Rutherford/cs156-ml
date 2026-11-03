# Rerun plan: more data, a reshuffled split, better embeddings (planned, not started)

Planned 3 Nov 2026, after the Pipeline 1 grade (4, #cs156-MLFlexibility). The professor's
future-direction note: re-run the classifiers on a pre-trained embedding for a much better
representation of the conversation, and combine it with knowledge evidence from other sources
(assignments, PCW) to see if it is predictive.

## 1. Data

- Add the classes recorded since 14 Oct with the existing steps 1-5 (Forum sync, Deepgram
  transcript, hand scan, QA). Needs a fresh Forum cookie. Likely fewer than 10 classes, from two
  courses: nothing special, just pooled with the rest.
- Pool them with the 446 classes that pass QA.

## 2. Split: reshuffle, whole classes, stratified by course

- Random reshuffle with a fixed seed. Whole classes only: a class's moments never sit in both
  train and test.
- **Test: exactly 100 classes**, allocated proportionally to course size (largest remainder),
  at least 1 per course with >= 5 classes. Courses with < 5 classes stay entirely in train.
  With today's 446 classes: test 100, train 346, 23 of 26 courses in test, every course in train.
- No separate validation split: model and threshold choices use **grouped 5-fold cross-validation
  on train** (folds by whole class, stratified by course). Test is scored once at the end of the
  round.

| Course | CX50 | MC50 | FA50 | B110 | EA50 | CS110 | CS114 | MC51 | B145 | B155 | CX51 | B111 | CS142 | B112 | CS111 | EA51 | B165 | FA51 | B113 | CS113 | CP192 | CS156 | CP191 | IL18102 | PR50 | PR51 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| classes | 25 | 24 | 24 | 22 | 22 | 22 | 21 | 21 | 21 | 21 | 21 | 21 | 20 | 20 | 19 | 19 | 19 | 18 | 18 | 16 | 8 | 7 | 6 | 4 | 4 | 3 |
| test | 6 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 4 | 4 | 4 | 4 | 4 | 4 | 2 | 2 | 1 | 0 | 0 | 0 |

(Recomputed by the same rule once the new classes are in.)

## 3. Embeddings (the professor's suggestion)

- Replace `BAAI/bge-small-en-v1.5` (384-d) with a large open embedder, run locally on a GPU pod
  (class text never goes to an embedding API). `oncue/embed.py` already caches vectors on disk by
  text hash, so this is a model swap plus one GPU pass; save the vectors and reuse them for every
  model.
- Use them two ways: (a) **directly as model inputs**: the last ~60 s of talk, reduced to 64-128
  dimensions, not only cosine similarities; (b) **knowledge evidence**: similarity between the
  current discussion and my own PCW answers and assignments (assignments need the Forum harvest).
- Optional: the fine-tuned 7B (`minerva-gpt`, variant B) hidden states as a me-specific embedding.

## 4. Rerun

Every model (naive baselines, trees, content MLP, TCN, combiner) on the new split, with
cross-validation for choices. Decision rule: **per-course thresholds** at my real raise rate.
Report the assignment's metrics (warnings right, raises warned, AP_w with session bootstrap) and
the single-instant scoring next to them, with time in warning.

Heavy steps (embedding, MLP/TCN training) on a GPU pod, not the Mac.

## What the 3 Nov analysis already settled

Decision rules on the final pipeline's scores (scripts and results in
`data/results/dynamic/`, local only):

- **Per-course thresholds** at the same number of warnings as the global threshold: test 7.8%
  right (vs 6.7%) and 38 of 533 raises warned (vs 35); validation 6.3% (vs 5.5%). Same direction
  on both splits; bootstrap intervals include zero. Chosen as the decision rule.
- CUSUM, rise-vs-median, derivative triggers, a PID-style score, a second-stage model and
  arm-then-fire did not beat the plain threshold when scored fairly. CUSUM first looked 2x
  better only because its warnings lasted ~41 s (vs ~6 s): always report time in warning.
- XGBoost stacks (2 Nov, `xgb_full.py`, results in `data/results/xgb/`) tied the final pipeline on
  test (AP_w 0.075-0.076 vs 0.076; 6.5% vs 6.7% of warnings right): more model types on the same
  inputs do not help.
- Lowering the threshold catches more raises at about the same % right; it trades false alarms
  for recall, it does not improve precision.
- About a third of false alarms are followed by a classmate speaking within 15 s: the model
  detects floor openings more than *my* turn. Training on everyone's turns (about 100,000 labels)
  is the next model idea after the embeddings.
