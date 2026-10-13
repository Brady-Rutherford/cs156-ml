"""CLI: `python -m oncue <subcommand> [--split S] [--workers N] [--class CID ...] [--work-dir D]`.

`--split train|validation|test|all` picks the classes of step 3 (`data/03_split/split.csv`);
`--class` overrides it. `--work-dir DIR` writes every output under DIR instead of data/ (a
throwaway pilot), reading the Scribe sync and the split from data/ as usual.

Each subcommand is dispatched lazily to `oncue.<module>.<function>(args) -> int | None`, so a
stage module can be written later and plug in without touching this file. Options a stage
needs beyond the common ones arrive unparsed in `args.extra` (a list of strings) for the
stage to parse itself. A stage that does not exist yet prints "not implemented yet" and the
process exits with status 2.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

# subcommand -> (module under oncue, function, one-line help). Order follows spec §13.
COMMANDS: dict[str, tuple[str, str, str]] = {
    "harvest": ("harvest", "cli_harvest",
                "class_edit_page + PCW template/doc, GET-only on sle, cached once"),
    "harvest-import": ("harvest", "cli_import",
                       "import Route C console exports into data/scribe/raw/{cid}/"),
    "select": ("select", "cli", "build classes.parquet: population, sections, vault, phases"),
    "ingest": ("ingest", "cli",
               "steps 4-5 for one split: transcribe, words, events, classes, hand scan, QA"),
    "ingest-report": ("ingest_qa", "cli",
                      "per-class timing checks of an ingest run -> logs/ingest_report.csv"),
    "audit": ("strip", "cli_audit", "grey-view clip + frame sheet around every raise onset"),
    "pdf": ("transcript_pdf", "cli", "readable PDF transcript + class summary (local only)"),
    "transcribe": ("transcribe", "cli_transcribe",
                   "Scribe transcription per class (provider per course, never --hand-log)"),
    "words": ("transcribe", "cli_words", "words.parquet from *_asr.json + raw voice events"),
    "events": ("events", "cli", "voice/chat/segments/polls/presence tables from raw events"),
    "pcw": ("pcw", "cli", "pcw_blocks + class-level pcw_status from raw PCW (QA only)"),
    "scan": ("hands", "cli_scan", "hand scan pass 1 + 2 -> hand_tiles (frames never stored)"),
    "track": ("hands", "cli_track", "offline glyph tracking -> hand_frames"),
    "episodes": ("hands", "cli_episodes", "state machine, votes, outcomes -> hand_episodes"),
    "retime": ("hands", "cli_retime", "sub-second onset re-timing (should-tier)"),
    "qa": ("qa", "cli", "merge timing/detector/PCW QA into classes; qa_pass, qa_reason"),
    "labels": ("labels", "cli", "speech onsets, at-risk grid and y_* -> points.parquet"),
    "frames": ("text", "cli_frames", "masking, normalisation, causal frames -> frames.parquet"),
    "embed": ("embed", "cli", "frame/tail/PCW embeddings with sha1 cache"),
    "features": ("features", "cli", "vectorised causal features -> features.parquet"),
    "train": ("experiments", "cli_train", "run comparison conditions -> oof/{cond}.parquet"),
    "evaluate": ("experiments", "cli_evaluate", "metrics, bootstrap -> results.parquet"),
    "controls": ("controls", "cli", "positive/negative controls (P-clock, P-plant, CALLON ...)"),
    "figures": ("plots", "cli", "aggregate-only figures -> figures/"),
    "hand-log-import": ("events", "cli_hand_log_import",
                        "live-logger export -> hand_log (QA only, never features)"),
    "results-bundle": ("privacy", "cli_results_bundle",
                       "write the committed results/ bundle with HMAC ids"),
    "privacy": ("privacy", "cli_scan", "scan committed paths for roster names and ids"),
    "e2e": ("driver", "cli", "synthetic end-to-end dry run of every stage in a sandbox dir"),
}

EXIT_NOT_IMPLEMENTED = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m oncue",
        description="On Cue pipeline stages (pipeline1/design/on_cue_spec.md).")
    sub = parser.add_subparsers(dest="command", metavar="<subcommand>")
    for name, (module, func, help_text) in COMMANDS.items():
        p = sub.add_parser(name, help=help_text, description=f"{help_text} "
                           f"(oncue.{module}.{func})")
        p.add_argument("--split", choices=("train", "validation", "test", "all"),
                       default=None, help="classes of this step-3 split (split.csv)")
        p.add_argument("--phase", choices=("A", "B", "C"), default=None,
                       help="old selection phase (§10); superseded by --split")
        p.add_argument("--work-dir", type=Path, default=None, metavar="DIR",
                       help="write every output under DIR instead of data/ (pilot runs)")
        p.add_argument("--workers", type=int, default=1, help="parallel workers")
        p.add_argument("--class", dest="class_ids", action="append", default=None,
                       metavar="CID", help="restrict to this class id (repeatable)")
    return parser


def resolve(command: str):
    """Import and return the function behind `command`, or None if it does not exist yet.

    Only a missing `oncue.<module>` (or a missing function) counts as "not implemented":
    an ImportError raised *inside* an existing module is re-raised so real bugs surface.
    """
    module, func = COMMANDS[command][:2]
    full = f"oncue.{module}"
    try:
        mod = importlib.import_module(full)
    except ModuleNotFoundError as exc:
        if exc.name in (full, "oncue"):
            return None
        raise
    return getattr(mod, func, None)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)
    if not args.command:
        parser.print_help()
        return EXIT_NOT_IMPLEMENTED
    args.extra = extra
    if args.work_dir is not None:
        from . import config

        config.use_work_dir(args.work_dir)
    fn = resolve(args.command)
    if fn is None:
        module, func = COMMANDS[args.command][:2]
        print(f"oncue {args.command}: not implemented yet (expected oncue.{module}.{func}).",
              file=sys.stderr)
        return EXIT_NOT_IMPLEMENTED
    result = fn(args)
    return int(result or 0)


if __name__ == "__main__":
    sys.exit(main())
