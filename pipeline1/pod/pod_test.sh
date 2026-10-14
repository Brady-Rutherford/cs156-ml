#!/usr/bin/env bash
# On the POD (option B rebalance, 14 Oct): scan the test split here, alongside the train scan
# in run.sh, while the laptop keeps validation (pod/mac_share.sh).
#   tmux new -d -s test "bash ~/cs156-ml/pipeline1/pod/pod_test.sh 10"
# Waits for the validation + test transcripts (tmux session transcribe_vt), drops this pod's
# empty test placeholders (except for the classes listed in data/logs/mac_test_ids.txt, which
# the laptop scans), then runs words, events, select and the scan for test. run.sh does
# track, episodes, retime, audit and QA for test after the train split, as for any split.
set -uo pipefail
W="${1:-10}"
cd "$HOME/cs156-ml/pipeline1"
PY=../.venv/bin/python
export ONCUE_FFMPEG_THREADS="${ONCUE_FFMPEG_THREADS:-2}" ONCUE_LIVE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_THREAD_LIMIT=1
LOG="$HOME/test_scan.log"
say() { echo "== $(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
say "test on this pod: waiting for the transcripts"
until grep -q "VT TRANSCRIBE DONE" "$HOME/transcribe_vt.log" 2>/dev/null; do sleep 60; done
$PY - <<'PYEOF' 2>&1 | tee -a "$LOG"
import pandas as pd
from oncue import config, hands
s = pd.read_csv(config.SPLIT_PATH)
keep_f = config.LOGS_DIR / "mac_test_ids.txt"
keep = {int(x) for x in keep_f.read_text().split()} if keep_f.exists() else set()
n = 0
for c in s[s.split == "test"].class_id.astype(int):
    p = hands.hand_tiles_path(c)
    if c not in keep and p.exists() and p.stat().st_size == 0:
        p.unlink(); n += 1
print(f"removed {n} empty test placeholders; {len(keep)} test classes left to the laptop")
PYEOF
say "test: words, events, select"
$PY -u -m oncue ingest --split test --stages words,events,select 2>&1 | tee -a "$LOG"
say "test: hand scan ($W workers)"
$PY -u -m oncue ingest --split test --workers "$W" --stages scan 2>&1 | tee -a "$LOG"
say "TEST SCAN DONE"
