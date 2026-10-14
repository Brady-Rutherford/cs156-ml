"""Progress board of the ingestion: `python pod/progress.py` (from pipeline1/).

Per split: classes transcribed, scanned, with raises re-timed; the stage running now (from
~/ingest.log); the last classes finished. Writes data/logs/live/progress.txt too. No names.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    import pandas as pd

    from oncue import config, hands

    s = pd.read_csv(config.SPLIT_PATH)
    s = s[s["split"].isin(config.SPLITS)]
    tr = config.TRANSCRIPTS_DIR
    asr = {int(m.group(1)): p for p in tr.rglob("*_asr.json")
           if (m := re.search(r"class(\d+)_asr\.json$", p.name))}
    scanned = {int(p.stem) for p in (config.TABLES_DIR / "hand_tiles").glob("*.parquet")}
    eps_p = config.table_path("hand_episodes")
    retimed = set()
    if eps_p.exists():
        e = pd.read_parquet(eps_p, columns=["class_id", "onset_retimed_s"])
        retimed = set(e.loc[e["onset_retimed_s"].notna(), "class_id"].astype(int))
    lines = [f"Ingestion progress   {time.strftime('%Y-%m-%d %H:%M:%S %Z')}", ""]
    lines.append(f"{'split':<11}{'classes':>8}{'transcribed':>13}{'scanned':>9}{'re-timed':>10}")
    for sp in config.SPLITS:
        ids = set(s.loc[s["split"] == sp, "class_id"].astype(int))
        lines.append(f"{sp:<11}{len(ids):>8}{len(ids & set(asr)):>13}{len(ids & scanned):>9}"
                     f"{len(ids & retimed):>10}")
    log_p = Path.home() / "ingest.log"
    if log_p.exists():
        stage = None
        with open(log_p, errors="ignore") as fh:        # stage markers anywhere in the log
            for x in fh:
                if x.startswith("== "):
                    stage = x[3:].rstrip()
        size = log_p.stat().st_size
        with open(log_p, "rb") as fh:                    # recent lines: the last 4 MB
            fh.seek(max(0, size - 4_000_000))
            tail = fh.read().decode("utf-8", "ignore").splitlines()
        tail = [x for x in tail if "[downloading]" not in x]
        lines += ["", "now: " + (stage or "starting")]
        done = [x for x in tail if x.startswith("scan ")][-5:]
        trs = [x for x in tail if "[complete]" in x][-5:]
        if trs:
            lines += ["", "last transcribed:"] + [
                "  " + re.sub(r".*class (\d+) .*", r"class \1", x) for x in trs]
        if done:
            lines += ["", "last scanned (frames, strip found, my tile frames, ring check):"] + [
                "  " + x[5:120] for x in done]
        errs = [x for x in tail if " ERROR " in x or "failed" in x.lower()][-3:]
        if errs:
            lines += ["", "recent errors:"] + ["  " + hands.redact(x)[:160] for x in errs]
    live = config.LOGS_DIR / "live"
    live.mkdir(parents=True, exist_ok=True)
    snaps = sorted(live.glob("scan_*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    if snaps:
        lines += ["", f"live scan snapshots: {len(snaps)} (newest: {snaps[0].stem})"]
    text = "\n".join(lines) + "\n"
    (live / "progress.txt").write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
