#!/usr/bin/env bash
# On the LAPTOP, after run.sh: copy the results back into ~/cs156-ml/pipeline1/data/.
#   bash pipeline1/pod/pull.sh root@HOST [PORT] [SSH_KEY] [--no-audit]
# ~1.3 GB without the audit clips (transcripts, tables, caches, logs), ~4 GB with them.
set -euo pipefail
HOST="${1:?usage: pull.sh user@host [port] [ssh_key] [--no-audit]}"
PORT="${2:-22}"
KEY="${3:-}"
SSH="ssh -p $PORT -o StrictHostKeyChecking=accept-new${KEY:+ -i $KEY}"
DIRS=(transcripts tables cache logs)
for d in "${DIRS[@]}"; do
  echo "== $d"
  rsync -az -e "$SSH" "$HOST:cs156-ml/pipeline1/data/$d/" \
      "$HOME/cs156-ml/pipeline1/data/$d/"
done
if [ "${4:-}" != "--no-audit" ]; then           # the clips live in pipeline1/audit/, not data/
  echo "== audit"
  rsync -az -e "$SSH" "$HOST:cs156-ml/pipeline1/audit/" "$HOME/cs156-ml/pipeline1/audit/"
fi
rsync -az -e "$SSH" "$HOST:ingest.log" "$HOME/cs156-ml/pipeline1/data/logs/pod_ingest.log" || true
echo "done. Then on the pod: bash ~/cs156-ml/pipeline1/pod/wipe.sh, and terminate the pod."
