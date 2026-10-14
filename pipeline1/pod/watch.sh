#!/usr/bin/env bash
# On the LAPTOP while run.sh works: every 20 s, print the progress board and copy the live
# scan snapshots to pipeline1/data/pod_live/ (open that folder in Cursor: images refresh).
#   bash pipeline1/pod/watch.sh root@HOST PORT [SSH_KEY]
set -uo pipefail
HOST="${1:?usage: watch.sh user@host port [ssh_key]}"
PORT="${2:-22}"
KEY="${3:-}"
SSH="ssh -p $PORT -o BatchMode=yes -o StrictHostKeyChecking=accept-new${KEY:+ -i $KEY}"
OUT="$HOME/cs156-ml/pipeline1/data/pod_live"
mkdir -p "$OUT"
while true; do
  board=$($SSH "$HOST" 'cd ~/cs156-ml/pipeline1 && ../.venv/bin/python pod/progress.py' 2>&1)
  clear
  echo "$board"
  rsync -az --delete -e "$SSH" "$HOST:cs156-ml/pipeline1/data/logs/live/" "$OUT/" 2>/dev/null
  echo; echo "(snapshots in $OUT; refreshed $(date +%H:%M:%S); Ctrl-c stops watching, not the run)"
  sleep 20
done
