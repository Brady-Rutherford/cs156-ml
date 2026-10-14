#!/usr/bin/env bash
# On the LAPTOP (option B, 14 Oct): scan validation + test here while the pod scans train.
#   bash pipeline1/pod/mac_share.sh root@HOST PORT [SSH_KEY]
# Each round: pull the pod's finished transcripts, then run words, events, select and the scan
# for up to 12 validation/test classes that have a transcript and no scan yet (6 workers, as in
# the pilot). When all of them are scanned: track, episodes, retime, audit and QA for both
# splits. Resumable: run it again and finished classes are skipped.
set -uo pipefail
HOST="${1:?usage: mac_share.sh user@host port [ssh_key]}"; PORT="${2:-22}"; KEY="${3:-}"
SSH="ssh -p $PORT -o BatchMode=yes -o ConnectTimeout=20${KEY:+ -i $KEY}"
cd "$HOME/cs156-ml/pipeline1"
PY=../.venv/bin/python
export ONCUE_LIVE=1
LOG=data/logs/mac_share.log
mkdir -p data/logs
say() { echo "== $(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
say "start: validation + test on this laptop"
while true; do
  rsync -az -e "$SSH" "$HOST:cs156-ml/pipeline1/data/transcripts/" data/transcripts/ 2>/dev/null
  read -r left batch <<<"$($PY - <<'PYEOF'
import re, pandas as pd
from oncue import config, hands
s = pd.read_csv(config.SPLIT_PATH)
vt = set(s[s.split.isin(["validation", "test"])].class_id.astype(int))
have = {int(m.group(1)) for p in config.TRANSCRIPTS_DIR.rglob("*_asr.json")
        if (m := re.search(r"class(\d+)_asr\.json$", p.name))}
todo = sorted(c for c in vt if not hands.hand_tiles_path(c).exists())
ready = [c for c in todo if c in have][:12]
print(len(todo), ",".join(map(str, ready)) or "-")
PYEOF
)"
  if [ "$left" -eq 0 ]; then break; fi
  if [ "$batch" = "-" ]; then say "waiting for transcripts ($left classes left)"; sleep 120; continue; fi
  ids=$(echo "$batch" | tr ',' '\n' | sed 's/^/--class /' | tr '\n' ' ')
  say "batch: $(echo "$batch" | tr ',' ' ') ($left left)"
  caffeinate -i $PY -u -m oncue ingest $ids --workers 6 --stages words,events,select,scan 2>&1 | tee -a "$LOG"
done
for split in validation test; do
  say "$split: track, episodes, retime, audit, QA"
  caffeinate -i $PY -u -m oncue ingest --split "$split" --workers 6 \
      --stages track,episodes,retime,audit,qa 2>&1 | tee -a "$LOG"
done
say "done: validation + test"
