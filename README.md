# cs156-ml

Coursework for CS156 (Machine Learning). Each lesson lives in its own folder under `notebooks/`
with the notebook(s) and the LLM conversation log that accompanied the work.

## Setup

```bash
uv sync                          # creates .venv and installs pinned deps from uv.lock
uv run nbstripout --install      # strip notebook outputs on commit (once per clone)
uv run jupyter lab               # open JupyterLab
```

## Layout

```
notebooks/          one folder per lesson: NN-topic/
  01-data-pipeline/
    conversation-log.md
src/cs156/          reusable code imported by notebooks (loaders, plotting)
data/               local datasets, gitignored
outputs/            figures and exports, gitignored
```

## Conventions

- Notebook outputs are stripped by `nbstripout` before commit, so diffs stay readable.
  To show a finished notebook on GitHub, export it: `uv run jupyter nbconvert --to html <nb>.ipynb`.
- Datasets are never committed. Iris and MNIST load through `cs156.data`.
- Dependencies are pinned in `uv.lock`. Add one with `uv add <package>`.
