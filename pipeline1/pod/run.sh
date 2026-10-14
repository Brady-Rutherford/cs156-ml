#!/usr/bin/env bash
# On the POD, inside tmux: steps 4-5 for every class, train -> validation -> test.
#   tmux new -s ingest
#   bash ~/cs156-ml/pipeline1/pod/run.sh          (detach: Ctrl-b then d; back: tmux attach -t ingest)
# Resumable: run it again after a stop (or after refreshing the Forum cookie) and finished
# classes are skipped (transcripts and scans already on disk are reused).
# Workers: TRANSCRIBE_WORKERS (default 24: network-bound) and SCAN_WORKERS (default 40% of the
# cores: each scan is one ffmpeg with ONCUE_FFMPEG_THREADS=2 plus one Python process).
set -uo pipefail
cd "$HOME/cs156-ml/pipeline1"
PY=../.venv/bin/python
# CPUs this pod may really use: a container's CPU quota (RunPod reports the whole host in
# nproc), else nproc.
cpus() {
  if [ -r /sys/fs/cgroup/cpu.max ]; then
    read -r q p < /sys/fs/cgroup/cpu.max
    [ "$q" != "max" ] && { echo $(( q / p )); return; }
  elif [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]; then
    q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us); p=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)
    [ "$q" -gt 0 ] && { echo $(( q / p )); return; }
  fi
  nproc
}
CPUS="$(cpus)"
TW="${TRANSCRIBE_WORKERS:-24}"
SW="${SCAN_WORKERS:-$(( CPUS * 2 / 5 ))}"
[ "$SW" -ge 1 ] || SW=1
export ONCUE_FFMPEG_THREADS="${ONCUE_FFMPEG_THREADS:-2}"
export ONCUE_LIVE=1                       # scan snapshots for pod/watch.sh
# one thread per library in every worker process: in a container they would otherwise each
# start one per visible host core (96 here) and swamp the 40 CPUs actually allowed
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_THREAD_LIMIT=1
LOG="$HOME/ingest.log"
echo "== $(date -u +%FT%TZ) start: $CPUS CPUs, transcribe workers $TW, scan workers $SW" | tee -a "$LOG"
$PY pod/preflight.py 2>&1 | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "preflight failed: fix it before running" | tee -a "$LOG"; exit 1; }
for split in train validation test; do
  echo "== $(date -u +%FT%TZ) $split: transcripts, words, events" | tee -a "$LOG"
  $PY -u -m oncue ingest --split "$split" --workers "$TW" \
      --stages transcribe,words,events,select 2>&1 | tee -a "$LOG"
  echo "== $(date -u +%FT%TZ) $split: hand scan, raises, clips, QA" | tee -a "$LOG"
  $PY -u -m oncue ingest --split "$split" --workers "$SW" \
      --stages scan,track,episodes,retime,audit,qa 2>&1 | tee -a "$LOG"
done
echo "== $(date -u +%FT%TZ) timing report" | tee -a "$LOG"
$PY -m oncue ingest-report --split all > data/logs/ingest_report.txt 2>&1
tail -1 data/logs/ingest_report.txt | tee -a "$LOG"
$PY - <<'PY' 2>&1 | tee -a "$LOG"
import pandas as pd
c = pd.read_parquet("data/tables/classes.parquet")
s = pd.read_csv("data/03_split/split.csv"); c = c[c.class_id.isin(s.loc[s.split != "excluded", "class_id"])]
e = pd.read_parquet("data/tables/hand_episodes.parquet")
b = e[(e.ep_type == "bid") & ~e.left_censored.astype(bool)]
print(f"classes {len(c)}: QA pass {int(c.qa_pass.fillna(False).sum())}; raises {len(b)} "
      f"({int((b.outcome == 'called').sum())} called); failures by reason "
      f"{c.loc[~c.qa_pass.fillna(False).astype(bool), 'qa_reason'].value_counts().head(8).to_dict()}")
PY
echo "== $(date -u +%FT%TZ) done. On the laptop: bash pipeline1/pod/pull.sh ..." | tee -a "$LOG"
