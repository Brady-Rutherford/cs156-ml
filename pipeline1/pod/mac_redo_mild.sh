#!/usr/bin/env bash
# On the LAPTOP, overnight 14-15 Oct, after pod/pull.sh: re-scan the "milder" classes (3-10% of
# the strip time with Brady's tile unread, scanned before label fix 2) with both label fixes,
# then re-time, clip and QA them, then the report and the PDFs.
#   caffeinate -i bash pipeline1/pod/mac_redo_mild.sh
# The class list comes from the pod (data/logs/rescan_label_fix2_mild.txt, via pull.sh).
# Old scans, caches and clips of those classes are kept in data/rescan_backup2_mild/.
set -uo pipefail
cd "$HOME/cs156-ml/pipeline1"
PY=../.venv/bin/python
LOG=data/logs/mac_redo_mild.log
LIST=data/logs/rescan_label_fix2_mild.txt
say() { echo "== $(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
[ -s "$LIST" ] || { echo "no $LIST: run pod/pull.sh first"; exit 1; }
if pgrep -f "[o]ncue ingest" >/dev/null; then echo "an oncue ingest is already running here"; exit 1; fi
export ONCUE_LIVE=1
say "set aside the old scans of $(wc -l < "$LIST" | tr -d ' ') classes"
$PY - "$LIST" <<'PYEOF' 2>&1 | tee -a "$LOG"
import shutil, sys
from pathlib import Path
from oncue import config, hands, strip
ids = [int(x) for x in Path(sys.argv[1]).read_text().split()]
bk = config.TABLES_DIR.parent / "rescan_backup2_mild"
bk.mkdir(parents=True, exist_ok=True)
n = 0
for c in ids:
    for p in (hands.hand_tiles_path(c), config.CACHE_DIR / "hands" / f"{c}.json", strip.labels_path(c),
              config.AUDIT_DIR / str(c)):
        if p.exists():
            shutil.move(str(p), bk / (p.name if p.is_file() else f"audit_{c}"))
            n += 1
print(f"moved {n} files/folders of {len(ids)} classes to {bk}")
PYEOF
ids=$(sed 's/^/--class /' "$LIST" | tr '\n' ' ')
say "scan (6 workers, both label fixes)"
caffeinate -i $PY -u -m oncue ingest $ids --workers 6 --stages scan 2>&1 | tee -a "$LOG"
say "track, episodes, retime, audit for these classes"
caffeinate -i $PY -u -m oncue ingest $ids --workers 6 --stages track,episodes,retime,audit 2>&1 | tee -a "$LOG"
say "QA over all classes, report"
$PY -m oncue qa 2>&1 | tail -3 | tee -a "$LOG"
$PY -m oncue ingest-report --split all 2>&1 | tail -3 | tee -a "$LOG"
say "PDFs (local only)"
caffeinate -i $PY -m oncue pdf --split all 2>&1 | tail -2 | tee -a "$LOG"
say "done: milder classes re-scanned"
