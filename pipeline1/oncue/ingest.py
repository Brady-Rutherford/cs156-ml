"""Steps 4-5 for one split: `python -m oncue ingest --split S [--sample N] [--work-dir DIR]`.

Runs, in order, for the classes of one step-3 split (or `--class` ids):

    transcribe (Deepgram, per course) -> words -> events -> select (classes.parquet)
    -> scan -> track -> episodes -> retime -> audit (clips of every raise) -> qa

Ingest train first, then validation, then test. A class that fails a stage is logged and
the batch goes on (`--strict` stops at the first failing stage instead).

`--sample N` takes a seeded random sample of the split, spread evenly over terms (UI eras
and timestamp precision differ by term). With `--work-dir data/pilot` that is the throwaway
pilot: everything it writes stays under data/pilot/, and deleting that folder discards it.
Every class ingested in a work dir is listed in `{logs}/ingest_classes.csv`.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from . import config
from .splits import split_table

STAGES = ("transcribe", "words", "events", "select", "scan", "track", "episodes", "retime",
          "audit", "qa")
PARALLEL = ("transcribe", "scan")          # stages that take --workers
WHOLE_TABLE = ("select", "qa")             # stages that work on the whole classes table


def sample_by_term(split: pd.DataFrame, n: int, *, seed: int = config.SEED) -> pd.DataFrame:
    """`n` classes of `split`, round-robin over terms (oldest first), random within a term."""
    rng = np.random.default_rng(seed)
    queues = []
    for _, grp in split.sort_values("date").groupby("term", sort=False):
        ids = grp["class_id"].to_numpy()
        queues.append(list(ids[rng.permutation(len(ids))]))
    picked = []
    while len(picked) < n and any(queues):
        for q in queues:
            if q and len(picked) < n:
                picked.append(int(q.pop(0)))
    return split[split["class_id"].isin(picked)].sort_values(["term", "date"])


def stage_argvs(class_ids: list[int], stages=STAGES, workers: int = 1) -> list[list[str]]:
    """The CLI calls of one ingest run, in order."""
    sel = [a for c in class_ids for a in ("--class", str(c))]
    out = []
    for st in stages:
        argv = [st] if st in WHOLE_TABLE else [st, *sel]
        if st in PARALLEL:
            argv += ["--workers", str(workers)]
        out.append(argv)
    return out


def cli(args) -> int:
    """`python -m oncue ingest --split S | --class CID ... [--sample N] [--stages a,b]
    [--workers N] [--work-dir DIR] [--strict]`."""
    from .driver import StageError, run_stages

    p = argparse.ArgumentParser(prog="python -m oncue ingest")
    p.add_argument("--sample", type=int, default=None, help="random N classes, spread over terms")
    p.add_argument("--stages", default=",".join(STAGES), help="comma-separated subset, in order")
    p.add_argument("--strict", action="store_true", help="stop at the first failing stage")
    o = p.parse_args(getattr(args, "extra", None) or [])
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    stages = [s.strip() for s in o.stages.split(",") if s.strip()]
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        print(f"oncue ingest: unknown stages {unknown}; choose from {', '.join(STAGES)}")
        return 2

    split = split_table()
    split = split[split["split"].isin(config.SPLITS)]
    if getattr(args, "class_ids", None):
        chosen = split[split["class_id"].isin([int(c) for c in args.class_ids])]
    elif getattr(args, "split", None):
        chosen = split if args.split == "all" else split[split["split"] == args.split]
    else:
        print("oncue ingest: pass --split train|validation|test|all or --class CID")
        return 2
    if o.sample:
        chosen = sample_by_term(chosen, o.sample)
    if not len(chosen):
        print("oncue ingest: no classes selected")
        return 1
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = config.LOGS_DIR / "ingest_classes.csv"     # every class ingested in this dir
    if log_path.exists():
        old = pd.read_csv(log_path)
        chosen_all = pd.concat([old[~old["class_id"].isin(chosen["class_id"])], chosen])
    else:
        chosen_all = chosen
    chosen_all.to_csv(log_path, index=False)
    print(f"oncue ingest: {len(chosen)} classes "
          f"({', '.join(f'{k} {v}' for k, v in chosen['split'].value_counts().items())}); "
          f"writing under {config.TABLES_DIR.parent}")

    ids = [int(c) for c in chosen["class_id"]]
    workers = max(1, int(getattr(args, "workers", 1) or 1))
    try:
        runs = run_stages(stage_argvs(ids, stages, workers), strict=o.strict)
    except StageError as exc:
        print(exc)
        return 1
    bad = [r.name for r in runs if not r.ok]
    print(f"oncue ingest: {len(runs) - len(bad)}/{len(runs)} stages ok"
          + (f"; with failures: {', '.join(bad)} (see the log)" if bad else ""))
    return 0 if not bad else 1
