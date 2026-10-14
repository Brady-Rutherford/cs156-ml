#!/usr/bin/env bash
# On the POD (14 Oct, evening): when pod_test.sh has scanned test, scan validation with the same
# workers, so they are not idle until run.sh reaches validation after train.
#   tmux new -d -s val "ONCUE_HWACCEL=cuda ONCUE_GPUS=5 bash ~/cs156-ml/pipeline1/pod/pod_val.sh 20"
# run.sh later finds these scans done and runs track, episodes, retime, audit and QA for them.
set -uo pipefail
W="${1:-20}"
cd "$HOME/cs156-ml/pipeline1"
PY=../.venv/bin/python
export ONCUE_FFMPEG_THREADS="${ONCUE_FFMPEG_THREADS:-2}" ONCUE_LIVE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_THREAD_LIMIT=1
LOG="$HOME/val_scan.log"
say() { echo "== $(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
if [ -z "${NO_WAIT:-}" ]; then              # NO_WAIT=1: start now (the test pool's tail is idle)
  say "validation on this pod: waiting for the test scan"
  until grep -q "TEST SCAN DONE" "$HOME/test_scan.log" 2>/dev/null; do sleep 60; done
fi
say "validation: words, events, select"
$PY -u -m oncue ingest --split validation --stages words,events,select 2>&1 | tee -a "$LOG"
say "validation: hand scan ($W workers)"
$PY -u -m oncue ingest --split validation --workers "$W" --stages scan 2>&1 | tee -a "$LOG"
say "VALIDATION SCAN DONE"
