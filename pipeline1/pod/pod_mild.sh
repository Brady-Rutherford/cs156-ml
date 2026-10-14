#!/usr/bin/env bash
# On the POD (14 Oct, Brady: "lets just do it on pod everything"): re-scan the "milder" classes
# (3-10% of the strip time with Brady's tile unread, scanned before label fix 2) with both
# label fixes, then re-time, clip and QA them.
#   tmux new -d -s mild "ONCUE_HWACCEL=cuda ONCUE_GPUS=5 bash ~/cs156-ml/pipeline1/pod/pod_mild.sh 30"
# 1. copies their current scans, caches and clips to data/rescan_backup2_mild/;
# 2. waits until every train class has a scan (the GPUs are then free of the train scan);
# 3. re-scans them, each new scan replacing the old one in one step (the old file stays
#    readable until then, so no other run sees a gap or scans them twice);
# 4. waits for run.sh to finish (its re-timing ran on the old scans), moves their old clips
#    aside, then track, episodes, retime, audit for them, QA over all classes, the report.
set -uo pipefail
W="${1:-30}"
cd "$HOME/cs156-ml/pipeline1"
PY=../.venv/bin/python
export ONCUE_FFMPEG_THREADS="${ONCUE_FFMPEG_THREADS:-2}" ONCUE_LIVE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_THREAD_LIMIT=1
LOG="$HOME/mild_scan.log"
LIST=data/logs/rescan_label_fix2_mild.txt
say() { echo "== $(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
[ -s "$LIST" ] || { say "no $LIST"; exit 1; }
say "milder classes: $(wc -l < "$LIST" | tr -d ' '); back up their scans"
$PY - "$LIST" <<'PYEOF' 2>&1 | tee -a "$LOG"
import shutil, sys
from pathlib import Path
from oncue import config, hands, strip
ids = [int(x) for x in Path(sys.argv[1]).read_text().split()]
bk = config.TABLES_DIR.parent / "rescan_backup2_mild"; bk.mkdir(parents=True, exist_ok=True)
n = 0
for c in ids:
    for p in (hands.hand_tiles_path(c), config.CACHE_DIR / "hands" / f"{c}.json", strip.labels_path(c)):
        if p.exists():
            shutil.copy2(p, bk / p.name); n += 1
print(f"copied {n} files of {len(ids)} classes to {bk}")
PYEOF
say "waiting for the train scan to finish"
until $PY - <<'PYEOF'
import sys, pandas as pd
from oncue import config, hands
s = pd.read_csv(config.SPLIT_PATH)
sys.exit(0 if all(hands.hand_tiles_path(c).exists() for c in s[s.split == "train"].class_id) else 1)
PYEOF
do sleep 60; done
say "re-scan ($W workers, label fixes 1 + 2)"
$PY -u pod/rescan.py "$LIST" "$W" 2>&1 | tee -a "$LOG"
tail -3 "$LOG" | grep -q "^re-scan: " || { say "MILD RE-SCAN BROKE (no summary line): stopping"; exit 1; }
say "MILD SCAN DONE; waiting for run.sh to finish"
until grep -q "done. On the laptop" "$HOME/ingest.log"; do sleep 60; done
$PY - "$LIST" <<'PYEOF' 2>&1 | tee -a "$LOG"
import shutil, sys
from pathlib import Path
from oncue import config
ids = [int(x) for x in Path(sys.argv[1]).read_text().split()]
bk = config.TABLES_DIR.parent / "rescan_backup2_mild"; n = 0
for c in ids:
    d = config.AUDIT_DIR / str(c)
    if d.exists():
        shutil.move(str(d), bk / f"audit_{c}"); n += 1
print(f"moved the old clips of {n} classes to {bk}")
PYEOF
ids=$(sed 's/^/--class /' "$LIST" | tr '\n' ' ')
say "track, episodes, retime, audit for the milder classes"
$PY -u -m oncue ingest $ids --workers "$W" --stages track,episodes,retime,audit 2>&1 | tee -a "$LOG"
say "QA over all classes, report"
$PY -m oncue qa 2>&1 | tail -3 | tee -a "$LOG"
$PY -m oncue ingest-report --split all > data/logs/ingest_report.txt 2>&1
tail -1 data/logs/ingest_report.txt | tee -a "$LOG"
say "MILD ALL DONE"
