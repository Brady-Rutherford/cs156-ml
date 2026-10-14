#!/usr/bin/env bash
# On the LAPTOP: copy the code, the Forum sync and the secrets to the pod.
#   bash pipeline1/pod/push.sh root@HOST [PORT] [SSH_KEY]
# Code = the files git knows about in ~/cs156-ml and ~/Minervas_scribe (no .venv, no data,
# no .git). Data = pipeline1/data/scribe (Forum metadata only, ~160 MB; no video, no
# transcripts). Secrets = ~/.minerva/.env, written straight to pipeline1/.env on the pod
# (mode 600), never through git.
set -euo pipefail
HOST="${1:?usage: push.sh user@host [port] [ssh_key]}"
PORT="${2:-22}"
KEY="${3:-}"
SSH="ssh -p $PORT -o StrictHostKeyChecking=accept-new${KEY:+ -i $KEY}"
CS="$HOME/cs156-ml"
SC="$HOME/Minervas_scribe"
LIST="$(mktemp)"
trap 'rm -f "$LIST"' EXIT

echo "== rsync on the pod (some images lack it)"
$SSH "$HOST" 'command -v rsync >/dev/null 2>&1 || { S=""; [ "$(id -u)" -ne 0 ] && S=sudo; \
  $S apt-get update -qq >/dev/null && $S env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq rsync >/dev/null; }'
echo "== code: cs156-ml"
(cd "$CS" && git ls-files -co --exclude-standard) > "$LIST"
rsync -az --files-from="$LIST" -e "$SSH" "$CS/" "$HOST:cs156-ml/"
echo "== code: Minervas_scribe"
(cd "$SC" && git ls-files -co --exclude-standard) > "$LIST"
rsync -az --files-from="$LIST" -e "$SSH" "$SC/" "$HOST:Minervas_scribe/"
echo "== data: the Forum sync (pipeline1/data/scribe)"
rsync -az -e "$SSH" "$CS/pipeline1/data/scribe/" "$HOST:cs156-ml/pipeline1/data/scribe/"
echo "== secrets -> pipeline1/.env (mode 600)"
$SSH "$HOST" 'umask 077; cat > cs156-ml/pipeline1/.env' < "$HOME/.minerva/.env"
echo "done. Next, on the pod: bash ~/cs156-ml/pipeline1/pod/setup.sh"
