#!/usr/bin/env bash
# On the LAPTOP, when the pod is cut (14 Oct, Brady: "1 more hour on pod then cut it and move
# over to mac"): stop the pod's work, bring every finished result here, and finish steps 4-5 for
# all 477 classes on this laptop.
#   bash pipeline1/pod/cut_to_mac.sh root@HOST PORT [SSH_KEY]
# First stop pod/mac_share.sh (and its batch) on this laptop: this script refuses to start while
# another `oncue ingest` runs here. Then terminate the pod (wipe.sh, `prime pods terminate`).
# Brings back: transcripts, finished scans (hand_tiles of size > 0, their caches), logs. The
# tables are rebuilt here from those (words, events, select take minutes), so nothing is merged.
set -uo pipefail
HOST="${1:?usage: cut_to_mac.sh user@host port [ssh_key]}"
PORT="${2:-22}"
KEY="${3:-}"
SSH="ssh -p $PORT -o BatchMode=yes -o ConnectTimeout=20${KEY:+ -i $KEY}"
REMOTE="$HOST:cs156-ml/pipeline1/data"
cd "$HOME/cs156-ml/pipeline1"
PY=../.venv/bin/python
LOG=data/logs/mac_all.log
mkdir -p data/logs/pod
say() { echo "== $(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
if pgrep -f "[o]ncue ingest" >/dev/null; then
  echo "an oncue ingest is still running on this laptop (mac_share.sh?): stop it first"; exit 1
fi

say "stop the pod's runs"
$SSH "$HOST" 'for s in ingest test transcribe_vt retry1; do tmux kill-session -t $s 2>/dev/null; done
  sleep 3; pkill -f "[m]ultiprocessing.spawn"; pkill -f "[o]ncue ingest"; pkill -x ffmpeg; sleep 2
  echo "left on the pod: $(pgrep -fc "[m]ultiprocessing.spawn") workers, $(pgrep -xc ffmpeg) ffmpeg"' \
  | tee -a "$LOG"

say "bring the pod's results here"
rsync -az -e "$SSH" "$REMOTE/transcripts/" data/transcripts/
rsync -az --min-size=1 --exclude="*.tmp" -e "$SSH" "$REMOTE/tables/hand_tiles/" data/tables/hand_tiles/
rsync -az --ignore-existing -e "$SSH" "$REMOTE/cache/" data/cache/
rsync -az -e "$SSH" "$REMOTE/logs/" data/logs/pod/
for f in ingest.log test_scan.log transcribe_vt.log; do rsync -az -e "$SSH" "$HOST:$f" data/logs/pod/ || true; done
rsync -az -e "$SSH" "$HOST:cs156-ml/pipeline1/data/rescan_backup/" data/rescan_backup/pod/ || true

$PY - <<'PY' 2>&1 | tee -a "$LOG"
import re, pandas as pd
from oncue import config, hands
s = pd.read_csv(config.SPLIT_PATH); s = s[s.split.isin(config.SPLITS)]
gone = 0
for c in s.class_id.astype(int):                      # the other machine's placeholders
    p = hands.hand_tiles_path(c)
    if p.exists() and p.stat().st_size == 0:
        p.unlink(); gone += 1
asr = {int(m.group(1)) for p in config.TRANSCRIPTS_DIR.rglob("*_asr.json")
       if (m := re.search(r"class(\d+)_asr\.json$", p.name))}
done = {c for c in s.class_id.astype(int) if hands.hand_tiles_path(c).exists()}
print(f"placeholders removed {gone}; transcribed {len(asr & set(s.class_id))}/{len(s)}; "
      f"scanned {len(done)}/{len(s)}: " + ", ".join(
          f"{k} {len(done & set(g.class_id))}/{len(g)}" for k, g in s.groupby("split")))
PY

export ONCUE_LIVE=1
say "transcripts still missing (if any), words, events, select: all classes"
caffeinate -i $PY -u -m oncue ingest --split all --workers 6 \
    --stages transcribe,words,events,select 2>&1 | tee -a "$LOG"
say "hand scan: every class not yet scanned (one pool, 6 workers)"
caffeinate -i $PY -u -m oncue ingest --split all --workers 6 --stages scan 2>&1 | tee -a "$LOG"
say "track, episodes, retime, audit, QA: all classes"
caffeinate -i $PY -u -m oncue ingest --split all --workers 6 \
    --stages track,episodes,retime,audit,qa 2>&1 | tee -a "$LOG"
$PY -m oncue ingest-report --split all 2>&1 | tail -3 | tee -a "$LOG"
say "done: all classes. PDFs (local only): $PY -m oncue pdf --split all"
