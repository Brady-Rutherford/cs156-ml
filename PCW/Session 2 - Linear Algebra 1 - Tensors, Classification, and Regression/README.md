# Session 2 — Linear Algebra 1: Tensors, Classification, and Regression

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Brady-Rutherford/cs156-ml/main?urlpath=lab/tree/PCW/Session%202%20-%20Linear%20Algebra%201%20-%20Tensors%2C%20Classification%2C%20and%20Regression)

Pre-class work for the second session. Implements the two models the session is built on —
linear regression and logistic regression — on the same 150 iris flowers, using scikit-learn.

| # | Question | Deliverable | Status |
|---|---|---|---|
| Drawing Lines | Replicate the two reading guides: pull Iris, fit a linear regression and a logistic regression, visualize the output | [`drawing-lines.ipynb`](drawing-lines.ipynb) — Code Cell 1 and Code Cell 2 | Done |

## The two code cells

The booklet has one code cell per model. Both live in
[`drawing-lines.ipynb`](drawing-lines.ipynb) and **each is self-contained** — its own imports,
its own `load_iris()` call, nothing imported from this repo — so either one can be pasted into
the booklet's cell runner and executed on its own.

- **Code Cell 1 — Linear Regression.** Predicts petal *width* from petal *length*, the two most
  strongly related columns in Iris. Load → choose X and y → 80/20 split → fit → score → plot.
- **Code Cell 2 — Logistic Regression.** Classifies all three species from the two petal
  measurements, fitting both **multinomial (softmax)** and **one-vs-rest**, then drawing the
  decision boundary and the learned hyperplanes side by side.

Both models are the same machine underneath: a weighted sum $z = \mathbf{w}\cdot\mathbf{x} + b$.
Linear regression returns $z$ as a quantity; logistic regression squashes it through a softmax
into class probabilities. "Drawing lines" is meant literally — Cell 1 draws one line through a
cloud of points, Cell 2 draws three that carve the plane into species territories.

## Code Cell 1 — what it shows

The fitted line is

$$\text{petal width} = 0.41 \times \text{petal length} - 0.36$$

with **test $R^2 = 0.928$** (train 0.927 — no gap, so nothing is overfitted) and test
MSE 0.046 cm². The slope carries the biology: **petal width grows at about four tenths the
rate of petal length**, so iris petals keep roughly the same proportions as they get bigger.

![Least-squares fit of petal width on petal length, with test residuals](assets/drawing-lines-linear-fit.png)

Two things the picture says that the metrics do not. The line runs through **three separate
clouds** — setosa alone in the bottom left, versicolor and virginica stacked along the top
right — even though species is never an input to the model. And the intercept of $-0.36$ is a
visible flaw: a petal of length 0 would be predicted to have *negative* width. The fit is a good
local description over the 1–7 cm range the data covers, and nonsense outside it.

## Code Cell 2 — what it shows

Both multinomial and one-vs-rest score **96.7% — 145 of 150 flowers**. The confusion matrix
says where the five misses are, and it is the same story Cell 1 told: **setosa is perfect
(50/50, precision and recall 1.00) and every single error is versicolor confused with
virginica.**

![Decision boundaries, multinomial vs one-vs-rest](assets/drawing-lines-decision-boundary.png)

The two boundaries in that plot are not equally trustworthy. The setosa boundary sits in a wide
**empty gap** — no flower is anywhere near it, so its exact position is arbitrary and the model
is very confident there. The versicolor/virginica boundary is driven straight through a region
where points of both colours sit on the wrong side of it. Same model, same figure, completely
different confidence.

![The hyperplanes each model learned](assets/drawing-lines-hyperplanes.png)

The hyperplane panels are the interesting comparison, and they are *not* the boundaries above —
each dashed line is where one class's own score crosses zero. Setosa and virginica land in
nearly the same place either way, but **the versicolor line is completely different**: under
one-vs-rest it cuts diagonally through the middle of the data, under multinomial it leaves the
plotted range entirely. That is the argument for the joint fit. One-vs-rest has to answer
"versicolor vs everything else" with a single line, and versicolor is sandwiched *between* the
other two — a question one line cannot answer. Multinomial fits all three at once, so versicolor
simply wins where the other two lose.

**Caveat, stated in the notebook too:** Cell 2 scores on the same 150 flowers it was fitted on,
matching the scikit-learn example it adapts. That makes 96.7% an optimistic ceiling rather than
an estimate on new flowers. Adding `train_test_split` — exactly as in Cell 1 — is the fix.

## Readings behind this

| Reading | How it is used |
|---|---|
| [Basic Linear Regression using the Iris Data set](https://medium.com/analytics-vidhya/linear-regression-using-iris-dataset-hello-world-of-machine-learning-b0feecac9cc1) | Code Cell 1: the load → split → fit → score → plot pipeline on Iris |
| [Basic Logistic Regression using the Iris Data set](https://scikit-learn.org/stable/auto_examples/linear_model/plot_logistic_multinomial.html) | Code Cell 2: multinomial vs one-vs-rest, `DecisionBoundaryDisplay`, the hyperplane plot |
| StatQuest: Linear Regression / Logistic Regression | Intuition for least squares and for maximum likelihood |
| [Extension] Murphy, *Probabilistic Machine Learning* (2022), linear and logistic regression sections | The model / loss / optimizer framing both cells follow |

One note on faithfulness: the scikit-learn page fits its example on **synthetic blobs**, not on
Iris. Code Cell 2 keeps that page's structure and plotting code and substitutes Iris for the
blobs, which is what the assignment asks for.

## Running this notebook

**In your browser, nothing to install** —
[run it on Binder](https://mybinder.org/v2/gh/Brady-Rutherford/cs156-ml/main?urlpath=lab/tree/PCW/Session%202%20-%20Linear%20Algebra%201%20-%20Tensors%2C%20Classification%2C%20and%20Regression/drawing-lines.ipynb).
The first launch takes a few minutes while the environment builds. Or just click the notebook in
this folder to read it on GitHub with all its output already rendered.

**Locally**, from the repository root:

```bash
uv sync
uv run python -m ipykernel install --user --name cs156-ml --display-name "Python (cs156-ml)"
```

Then open the notebook in VS Code or Cursor and run the cells. Iris needs no download — it
ships inside scikit-learn.

## Files

```
drawing-lines.ipynb       Code Cell 1 (linear) and Code Cell 2 (logistic), both executed
assets/                   figures the notebook saves
  drawing-lines-linear-fit.png
  drawing-lines-decision-boundary.png
  drawing-lines-hyperplanes.png
```
