# Proposed model pipeline (draft for Brady's decision, 18 Oct 2026)

Question: every 2 s of a class, will Brady raise his hand in the next 15 s (H = 15 s, guard 2 s)?
Only information available up to that moment may be used. Train on the train split, choose
settings on validation, score test once.

```
  inputs at moment t (causal)
  ├── structured: 76 numbers (minute, segment, who talks, questions, my history, polls ...)
  ├── text: what was said in the last 15/60 s (names masked)
  ├── class content: learning outcomes + study guide (class page), my pre-class work (as of class start)
  └── timeline: the class so far, one step per 2 s

  layer 1: specialist models, each gives a score (or a small vector)
  ├── (1) content-alignment MLP   "how close is what's being asked to the class material and to me?"
  ├── (2) TCN                     "what has happened so far, and where is the class heading?"
  └── (3) fine-tuned transformer  "reading the last 60 s of text"

  layer 2: gradient-boosted trees  "weigh everything"  -> calibrated P(raise in next 15 s)

  baselines shown first: question-timing model -> Naive Bayes -> TF-IDF + logistic regression
```

## Baselines (the story before the main pipeline)

1. **Question-timing model.** From training classes: after each professor question, the chance
   Brady raises in the next 15 s as a function of seconds since the question (a hazard curve or
   a logistic regression on time since question and "question still open"). It already knows
   that he does not raise at every question.
2. **Naive Bayes** on the words of the last 60 s.
3. **TF-IDF + logistic regression** on the same words. Better, and still "not good enough":
   it sees words, not meaning, timing, or Brady.

## (1) Content-alignment MLP

- **Role.** Turn meaning into numbers the trees can use: is the professor asking about the
  class's learning outcomes, about something Brady prepared, about something he has said before?
- **Inputs.** bge-small sentence vectors (384 numbers each) of: the professor's latest question;
  the class's learning outcomes and study guide (class page); Brady's pre-class work answers as
  of class start; Brady's own earlier contributions. Their cosine similarities. Optionally a
  tone score of the professor's words (encouraging, open, rhetorical).
- **Model.** A small multilayer perceptron (one or two hidden layers, ~10-30k parameters) on top
  of the frozen vectors. The language model is not trained; only the small head is.
- **Output.** A score, plus the similarities themselves as features.
- **Cost.** Minutes on the laptop. Needs the pre-class work rebuilt as of class start (from its
  revision history) so later edits never leak in.
- **Risk.** Low. Note: "sentiment" is best read as *meaning* (topic match); a tone score can be an
  extra input, but topic match is the stronger signal.

## (2) TCN (temporal convolutional network)

- **Role.** A "memory of the class so far": a sequence of moments (2 s each) read with causal
  1-D convolutions, so it only ever looks backward. Learns patterns over time, e.g. a question,
  silence, two students answer, the professor rephrases.
- **Inputs per step.** A 64-number summary of what was said in that step, seconds of speech per
  role (professor, student, Brady), question open and its kind, minute and segment, Brady's
  participation and raise history.
- **Model.** Dilated causal convolutions with a receptive field of several minutes, 10-50k
  parameters, a few epochs, early stopping on validation classes.
- **Output.** A score per step, or its last hidden layer (16-32 numbers) as features for the trees.
- **Cost.** Minutes to tens of minutes on the laptop.
- **Risk.** Medium: easy to overfit with ~2,800 raises; keep it small.

## (3) Fine-tuned transformer

- **Role.** The strongest reader of the text: a small language model (MiniLM, ~22M parameters)
  fine-tuned to answer "will he raise in the next 15 s?" from the last 60 s of words.
- **Input.** "[subject] [segment type] [minute] [minutes since I spoke] || last 60 s of text
  (up to 224 tokens) || best-matching part of my pre-class work (up to 64 tokens)".
- **Training.** All positive moments plus a 20% sample of negatives (corrected afterwards),
  2 epochs, on the Mac GPU (MPS).
- **Output.** A score.
- **Cost.** The expensive part: roughly 30-60 min per training; stacking needs 3-4 trainings
  (see below), so a few hours.
- **Risk.** Medium-high: overfitting, time. Cut first if time is short.

## (4) Gradient-boosted trees (the final model)

- **Role.** The decider. Combines the 76 structured numbers with the scores and vectors from
  (1)-(3), and handles mixed inputs, missing values and interactions (e.g. "a question about my
  pre-class work, early in a segment, and I have not spoken yet").
- **Model.** scikit-learn HistGradientBoosting, early stopping on validation classes, then
  calibrated (Platt) so its output is a real probability.
- **Cost.** Minutes. **Risk.** Low.

## Do they make sense together? Yes, as a stacked ensemble

Each part has a different job (meaning, time, reading, combining), and the trees decide how much
to trust each one. Rules that keep it honest:

1. **Out-of-fold scores.** The trees must be trained on scores that (1)-(3) produced for classes
   they were *not* trained on: split the train classes into 3-5 groups, train each specialist on
   the others, score the held-out group. Otherwise the trees learn to trust overfit scores.
2. **Validation decides, test scores once.** Every choice (which parts to keep, sizes, epochs)
   on validation classes; the test split is scored a single time at the end.
3. **Show that each part earns its place.** Report the trees with and without each specialist
   (ablation), next to the three baselines. That is the exploration part of the write-up.
4. **Causality everywhere.** Every input is cut at the moment t (as the features already are).

## If time is short, the order to build

1. Baselines (question-timing, NB, TF-IDF).
2. Gradient-boosted trees on the 76 numbers + similarity features from (1).
3. + the MLP score (1).
4. + the TCN (2).
5. + the transformer (3).

Each step is a complete pipeline on its own, so stopping early still leaves a finished result.
