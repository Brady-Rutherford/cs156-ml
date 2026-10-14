#!/usr/bin/env bash
# On the LAPTOP, after option B (14 Oct: the pod did train + test, this laptop validation).
# Brings the pod's results in without touching the laptop's own validation results, then redoes
# QA over all classes (the A/V lag rule compares each class with the median of all of them).
#   bash pipeline1/pod/pull_merge.sh root@HOST PORT [SSH_KEY] [--no-audit]
# Per-class files: transcripts and caches are added (files already here are kept), hand_tiles of
# size > 0 only (the pod's empty validation placeholders never land here; the pod's real test
# files replace this laptop's empty test placeholders). Tables: the pod's rows for train + test,
# this laptop's rows for validation and the test classes in data/logs/mac_test_ids.txt.
set -euo pipefail
HOST="${1:?usage: pull_merge.sh user@host port [ssh_key] [--no-audit]}"
PORT="${2:-22}"
KEY="${3:-}"
SSH="ssh -p $PORT -o BatchMode=yes${KEY:+ -i $KEY}"
REMOTE="$HOST:cs156-ml/pipeline1/data"
cd "$HOME/cs156-ml/pipeline1"
PY=../.venv/bin/python
echo "== transcripts"
rsync -az -e "$SSH" "$REMOTE/transcripts/" data/transcripts/
echo "== hand_tiles (non-empty only)"
rsync -az --min-size=1 -e "$SSH" "$REMOTE/tables/hand_tiles/" data/tables/hand_tiles/
echo "== caches"
rsync -az --ignore-existing -e "$SSH" "$REMOTE/cache/" data/cache/
if [ "${4:-}" != "--no-audit" ]; then
  echo "== audit clips"
  rsync -az --ignore-existing -e "$SSH" "$REMOTE/audit/" data/audit/
fi
echo "== pod tables and logs (kept apart, then merged)"
mkdir -p data/pod_tables data/logs/pod
rsync -az -e "$SSH" --include="*.parquet" --exclude="*" "$REMOTE/tables/" data/pod_tables/
rsync -az -e "$SSH" "$REMOTE/logs/" data/logs/pod/
for f in ingest.log test_scan.log transcribe_vt.log; do
  rsync -az -e "$SSH" "$HOST:$f" data/logs/pod/ || true
done
mkdir -p data/tables_before_merge && cp data/tables/*.parquet data/tables_before_merge/
$PY - <<'PY'
import pandas as pd
from pathlib import Path
from oncue import config

s = pd.read_csv(config.SPLIT_PATH)
mine = set(s.loc[s.split == "validation", "class_id"].astype(int))   # done on this laptop
extra = config.LOGS_DIR / "mac_test_ids.txt"                        # test classes done here too
mine |= {int(x) for x in extra.read_text().split()} if extra.exists() else set()
here, pod = config.TABLES_DIR, Path("data/pod_tables")
for name in sorted({p.stem for p in here.glob("*.parquet")} | {p.stem for p in pod.glob("*.parquet")}):
    a = pd.read_parquet(here / f"{name}.parquet") if (here / f"{name}.parquet").exists() else None
    b = pd.read_parquet(pod / f"{name}.parquet") if (pod / f"{name}.parquet").exists() else None
    parts = []
    if b is not None:
        parts.append(b[~b["class_id"].isin(mine)])
    if a is not None:
        parts.append(a[a["class_id"].isin(mine)] if b is not None else a)
    out = pd.concat([p for p in parts if len(p)], ignore_index=True) if any(len(p) for p in parts) \
        else parts[0]
    out.sort_values("class_id", kind="stable").to_parquet(here / f"{name}.parquet", index=False)
    print(f"{name}: {out['class_id'].nunique()} classes, {len(out)} rows")
PY
echo "== QA over all classes, report"
$PY -m oncue qa
$PY -m oncue ingest-report --split all | tail -3
echo "done. PDFs (local only): $PY -m oncue pdf --split all"
echo "Then on the pod: bash ~/cs156-ml/pipeline1/pod/wipe.sh, and terminate the pod."
