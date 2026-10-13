"""Synthetic end-to-end driver: every stage, in order, on the invented world, in a sandbox.

`run_synthetic(root)` writes `synthetic.world(...)` in the real on-disk layout under `root`
(Scribe sync `raw/` plus transcripts with a saved `*_asr.json`), points every `config` path
at `root` and runs the stages through the real CLI (`oncue.__main__.main`), in the order a
real run uses. There are three substitutions, all confined to the sandbox:

* `transcribe` is skipped: the synthetic `*_asr.json` / `*_ids.json` stand in for the saved
  ASR, so no audio and no ASR provider are involved;
* the hand scan reads synthetic frames (`synthetic.frames_for`) instead of signed recording
  URLs (`hands._sources` and `hands._forum_client` are swapped);
* embeddings use the offline `embed.HASH_MODEL` and EduBehaviors is skipped (no downloads).

Inside the sandbox `config.load_env` is a no-op and a synthetic ID salt is set, so nothing
reads `.env`, touches the network, or writes to the real `pipeline1/data`, `results/` or
`figures/`. `python -m oncue e2e [--out DIR]` runs it from the command line.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import tempfile
import time
import traceback
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config

SYNTH_SALT = "synthetic-e2e-salt"          # never a real salt
STRIP_CHUNK = 200                          # synthetic frames rendered per batch

# Model conditions of the synthetic run: the MVP rows (§7) plus M4 on the stub embedder.
DEFAULT_CONDITIONS = ("K0-AO", "K0-B0", "K0-Bmin", "K0-R", "K0-B1", "K0-M1", "K0-M2",
                      "K0-M3", "K1-M4", "F0-Fstruct", "F0-Fall-me_hand",
                      "F0-Fstruct-me_hand", "S-rand-M3", "P-plant-M1", "P-plant-M2")


def sandbox_paths(root: Path | str) -> dict[str, Path]:
    """config attribute -> path under `root` (the layout of `pipeline1/data`, plus the
    committed-output dirs results/ and figures/ and the audit dir)."""
    root = Path(root)
    scribe = root / "scribe"
    return {
        "DATA_DIR": root, "SCRIBE_DIR": scribe, "RAW_DIR": scribe / "raw",
        "MANIFEST_PATH": scribe / "manifest.csv", "ME_PATH": scribe / "me.json",
        "SECTIONS_PATH": scribe / "sections.json", "SCHEDULE_DIR": scribe / "schedule",
        "TRANSCRIPTS_DIR": root / "transcripts", "TABLES_DIR": root / "tables",
        "CACHE_DIR": root / "cache", "EMB_CACHE_DIR": root / "cache" / "emb",
        "EDU_CACHE_DIR": root / "cache" / "edu", "LOGS_DIR": root / "logs",
        "DROPPED_KEYS_LOG": root / "logs" / "dropped_keys.csv", "OOF_DIR": root / "oof",
        "RESULTS_DIR": root / "results", "FIGURES_DIR": root / "figures",
        "AUDIT_DIR": root / "audit",
    }


def synthetic_sources(world, *, fps: float = config.F_SCAN, chunk: int = STRIP_CHUNK
                      ) -> Callable:
    """A stand-in for `hands._sources(cid, client)` that renders the synthetic recording:
    full frames for pass 1, a 2 fps strip stream for pass 2, a 30 fps clip reader for
    re-timing. Times are master-clock seconds (synthetic media time)."""
    from . import synthetic

    def sources(cid, client=None) -> dict:
        sc = world.by_id(cid)

        def frame_reader(times):
            times = [float(t) for t in times]
            return list(zip(times, synthetic.frames_for(sc, times), strict=True))

        def strip_stream():
            ts = np.arange(0.0, sc.T_end_s, 1.0 / fps)
            for i in range(0, len(ts), chunk):
                part = ts[i:i + chunk]
                yield from zip(part, synthetic.frames_for(sc, part, full=False), strict=True)

        def clip_reader(t0, dur):
            ts = np.arange(np.ceil(t0 * 30 - 1e-6) / 30, t0 + dur, 1 / 30)
            return list(zip(ts, synthetic.frames_for(sc, ts, full=False), strict=True))

        return {"frame_reader": frame_reader, "strip_source": strip_stream(),
                "clip_reader": clip_reader, "url_fn": None, "media": None}
    return sources


@contextlib.contextmanager
def sandbox(root: Path | str, *, world=None, salt: str = SYNTH_SALT) -> Iterator[Path]:
    """Point every config path at `root`, make `config.load_env` a no-op, set a synthetic ID
    salt and offline Hugging Face flags, and (with `world`) swap the hand scan's frame
    sources for synthetic ones. Everything is restored on exit."""
    from . import hands

    root = Path(root)
    saved = {k: getattr(config, k) for k in sandbox_paths(root)}
    saved_env = {k: os.environ.get(k) for k in (*config.ID_SALT_ENV, "HF_HUB_OFFLINE",
                                                "TRANSFORMERS_OFFLINE")}
    saved_hooks = (config.load_env, hands._sources, hands._forum_client, hands.SCANNER)
    try:
        for k, v in sandbox_paths(root).items():
            setattr(config, k, v)
        config.load_env = lambda *a, **k: False
        for k in config.ID_SALT_ENV:
            os.environ.pop(k, None)
        os.environ[config.ID_SALT_ENV[0]] = salt
        os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
        if world is not None:
            hands._sources = synthetic_sources(world)
            hands._forum_client = lambda: None
            hands.SCANNER = "glyph"               # synthetic frames draw the glyph-era UI
        yield root
    finally:
        for k, v in saved.items():
            setattr(config, k, v)
        config.load_env, hands._sources, hands._forum_client, hands.SCANNER = saved_hooks
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def stage_argvs(world, root: Path, *, conditions: Sequence[str] = DEFAULT_CONDITIONS,
                hgb_max_iter: int = 60, n_boot: int = 100, workers: int = 4
                ) -> list[list[str]]:
    """The CLI calls of a full run, in order. `select` gets `--always` for every synthetic
    section so the whole invented world is the population (the real §10 rules would drop
    the low-participation and current-term sections)."""
    from .embed import HASH_MODEL

    always = []
    for s in world.sections:
        course = s["course"]
        always += ["--always", f"{course['course-code']}:{config.term_code(course['term']['name'])}"]
    conds = [a for c in conditions for a in ("--cond", c)]
    return [
        ["select", *always],
        ["words"], ["events"], ["pcw"], ["frames"],
        ["embed", "--model", HASH_MODEL],
        ["scan", "--workers", str(workers)], ["track"], ["episodes"], ["retime"],
        ["qa"], ["labels"],
        ["features", "--no-edu", "--model", HASH_MODEL],
        ["train", *conds, "--hgb-max-iter", str(hgb_max_iter)],
        ["evaluate", *conds],
        ["controls", "--n-boot", str(n_boot)],
        ["figures", "--n-boot", str(n_boot)],
        ["results-bundle"],
        ["privacy", "--path", str(root / "results"), "--path", str(root / "figures")],
    ]


@dataclass
class StageRun:
    argv: list[str]
    rc: int | None
    seconds: float
    error: str | None = None

    @property
    def name(self) -> str:
        return self.argv[0]

    @property
    def ok(self) -> bool:
        return self.rc == 0 and self.error is None


@dataclass
class E2EResult:
    root: Path
    class_ids: list[int]
    stages: list[StageRun] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(s.ok for s in self.stages)

    def table(self, name: str):
        import pandas as pd

        return pd.read_parquet(Path(self.root) / "tables" / f"{name}.parquet")


class StageError(RuntimeError):
    """A stage of the synthetic run failed (strict mode)."""


def run_stages(argvs: Sequence[Sequence[str]], *, strict: bool = True,
               echo: Callable[[str], None] | None = print) -> list[StageRun]:
    """Run CLI stages in order through `oncue.__main__.main`. Strict: stop at the first
    stage that fails (non-zero exit or exception) and raise StageError."""
    from . import __main__ as cli

    runs = []
    for argv in argvs:
        argv = list(argv)
        t0 = time.perf_counter()
        rc, err = None, None
        try:
            rc = int(cli.main(argv))
        except SystemExit as exc:            # stages signal missing inputs this way
            rc, err = 1, f"SystemExit: {exc}"
        except Exception:                    # noqa: BLE001 - reported with the stage name
            err = traceback.format_exc()
        run = StageRun(argv, rc, time.perf_counter() - t0, err)
        runs.append(run)
        if echo is not None:
            echo(f"[e2e] {run.name:<15s} rc={rc} {run.seconds:6.1f}s"
                 + ("" if run.ok else "  FAILED"))
        if strict and not run.ok:
            raise StageError(f"stage {argv} failed (rc={rc}):\n{err or ''}")
    return runs


def check_root(root: Path) -> None:
    """Refuse a sandbox inside pipeline1/ (real data/, results/, figures/ live there)."""
    real = config.PIPELINE_DIR.resolve()
    r = Path(root).resolve()
    if r == real or real in r.parents:
        raise ValueError(f"oncue e2e: the sandbox must be outside {real} (got {r})")


def run_synthetic(root: Path | str, *, n_classes: int = 6, minutes: float = 20.0,
                  seed: int = config.SEED, conditions: Sequence[str] = DEFAULT_CONDITIONS,
                  hgb_max_iter: int = 60, n_boot: int = 100, workers: int = 4,
                  strict: bool = True, echo: Callable[[str], None] | None = print
                  ) -> E2EResult:
    """Write a synthetic world under `root` and run every stage on it in a sandbox. `root`
    must lie outside pipeline1/ (its results/ and figures/ are committed paths)."""
    from . import synthetic

    root = Path(root)
    check_root(root)
    root.mkdir(parents=True, exist_ok=True)
    world = synthetic.world(n_classes, seed, minutes=minutes)
    synthetic.write_world(world, root)
    result = E2EResult(root=root, class_ids=list(world.class_ids))
    with sandbox(root, world=world):
        argvs = stage_argvs(world, root, conditions=conditions, hgb_max_iter=hgb_max_iter,
                            n_boot=n_boot, workers=workers)
        result.stages = run_stages(argvs, strict=strict, echo=echo)
    return result


def cli(args) -> int:
    """`python -m oncue e2e [--out DIR] [--classes N] [--minutes M] [--cond ID ...]
    [--hgb-max-iter N] [--keep-going] [--workers N]` (scan workers, default 4)."""
    p = argparse.ArgumentParser(prog="python -m oncue e2e")
    p.add_argument("--out", type=Path, default=None,
                   help="sandbox root (default: a new temp dir; never pipeline1/data)")
    p.add_argument("--classes", type=int, default=6)
    p.add_argument("--minutes", type=float, default=20.0)
    p.add_argument("--cond", action="append", default=None)
    p.add_argument("--hgb-max-iter", type=int, default=60)
    p.add_argument("--keep-going", action="store_true", help="run later stages after a failure")
    o = p.parse_args(getattr(args, "extra", None) or [])
    root = o.out or Path(tempfile.mkdtemp(prefix="oncue_e2e_"))
    try:
        check_root(root)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    try:
        workers = int(getattr(args, "workers", 1) or 1)
        res = run_synthetic(root, n_classes=o.classes, minutes=o.minutes,
                            conditions=tuple(o.cond or DEFAULT_CONDITIONS),
                            hgb_max_iter=o.hgb_max_iter, strict=not o.keep_going,
                            workers=workers if workers > 1 else 4)
    except StageError as exc:
        print(exc)
        print(f"sandbox kept at {root}")
        return 1
    print(f"oncue e2e: {sum(s.ok for s in res.stages)}/{len(res.stages)} stages ok in "
          f"{sum(s.seconds for s in res.stages):.0f}s; sandbox at {root}")
    return 0 if res.ok else 1
