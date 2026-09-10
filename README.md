# CS156 — Finding Patterns in Data with Machine Learning

Coursework for CS156. Every class session has its own folder holding all the work for it:
notebooks, written answers, and photos of any handwritten work.

**If you are grading this, start with the session folder you want and read its README.**
Each one lists every question in the assignment, links the file that answers it, and says
what that file shows. You should never have to open a notebook to find out whether a
question was answered.

**Two ways to see the code.** Clicking any `.ipynb` in this repo renders it on GitHub
complete with its plots and printed output — nothing to install, nothing to run. To edit
and re-run it, clone the repo and follow [Running the code](#running-the-code) below:
`uv sync` builds the pinned environment, and every notebook already records the kernel it
needs.

## Sessions

| Session | Title | Work |
|---|---|---|
| 1 | Introduction and overview of machine learning concepts | [PCW/Session 1](<PCW/Session 1 - Introduction and overview of machine learning concepts/README.md>) |
| 2 | Linear Algebra 1: Tensors, Classification, and Regression | [PCW/Session 2](<PCW/Session 2 - Linear Algebra 1 - Tensors, Classification, and Regression/README.md>) |

## How the repo is organised

```
PCW/                              pre-class work, one folder per class session
  Session 1 - <session title>/
    README.md                     index of every question, with answers linked
    Notes/                        reading notes for the session
    2.1.md, 2.2.md, 2.3.md        written answers, named by their PCW question number
    3.1-iris.ipynb                notebooks, prefixed with their question number
    3.1-mnist.ipynb
    3.1-conversation-log.md       the LLM conversation behind that question
    assets/                       photos of handwritten work, diagrams, exports
  Session 2 - <session title>/
    README.md                     index of every question, with answers linked
    drawing-lines.ipynb           booklet Code Cells 1 and 2, plus analysis
    extension-question.ipynb      booklet Code Cells 4 and 5, models from scratch
    assets/

src/cs156/                        shared code the notebooks import
  data.py                         dataset loaders (Iris, MNIST)
  plotting.py                     plotting helpers (image grids)

.vscode/settings.json             pins the .venv interpreter so notebooks just open

data/                             local datasets, gitignored
outputs/                          figures and exports, gitignored
```

Two conventions make this navigable:

- **Files are named after the question number they answer.** `3.1-iris.ipynb` answers
  question 3.1. Everything for one question sorts together.
- **Session folders are named after the session title**, so the repo reads in the same
  order as the course schedule.

## Running the code

Everything is pinned and reproducible. From the repository root:

```bash
uv sync                          # creates .venv, installs the pinned dependencies
uv run nbstripout --install      # strip notebook outputs on commit (once per clone)

# register the venv as a named Jupyter kernel (once per clone)
uv run python -m ipykernel install --user --name cs156-ml --display-name "Python (cs156-ml)"
```

Then open the repo in VS Code or Cursor and open any notebook — it selects the
`Python (cs156-ml)` kernel on its own, no prompt. `.vscode/settings.json` pins the
interpreter to `.venv`, and every notebook records that kernel in its metadata.

For the classic notebook UI instead: `uv run jupyter lab`.

Notebooks import shared code as `from cs156.data import iris`, which works from any
folder because the project installs itself into the environment.

## Notes on the setup

- **Notebook outputs are committed**, so the plots and printed results show up when you
  click a notebook on GitHub. That is deliberate: the output *is* the deliverable here, and
  a reader should not have to run anything to see it. It costs slightly noisier diffs.
  `nbstripout` is still installed if that trade ever stops being worth it
  (`uv run nbstripout --install` turns stripping back on).
- **Datasets are never committed.** Iris ships inside scikit-learn. MNIST downloads ~15MB
  from OpenML on first use and caches to `~/scikit_learn_data`, outside the repo.
- **Dependencies are pinned** in `uv.lock`. Add one with `uv add <package>`.
