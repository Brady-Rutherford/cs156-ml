#!/usr/bin/env bash
# On the LAPTOP, in the background: copy the pod's live scan snapshots and progress board to
# pipeline1/data/pod_live/ every 20 s, until the run's tmux session has ended. No output.
#   bash pipeline1/pod/sync_live.sh root@HOST PORT [SSH_KEY]
HOST="${1:?usage: sync_live.sh user@host port [ssh_key]}"; PORT="${2:-22}"; KEY="${3:-}"
SSH="ssh -p $PORT -o BatchMode=yes -o ConnectTimeout=20${KEY:+ -i $KEY}"
OUT="$HOME/cs156-ml/pipeline1/data/pod_live"; mkdir -p "$OUT"
gone=0
while [ "$gone" -lt 3 ]; do                 # stop only after the session is missing 3 times
  $SSH "$HOST" 'cd ~/cs156-ml/pipeline1 && ../.venv/bin/python pod/progress.py >/dev/null 2>&1; true' 2>/dev/null
  rsync -az --delete -e "$SSH" "$HOST:cs156-ml/pipeline1/data/logs/live/" "$OUT/" 2>/dev/null
  if $SSH "$HOST" 'tmux has-session -t ingest' 2>/dev/null; then gone=0; else gone=$((gone + 1)); fi
  sleep 20
done
