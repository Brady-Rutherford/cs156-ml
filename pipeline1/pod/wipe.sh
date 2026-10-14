#!/usr/bin/env bash
# On the POD, after pull.sh: delete the secrets and everything about classes, then terminate
# the pod in Prime's dashboard (that also destroys its disk).
#   bash ~/cs156-ml/pipeline1/pod/wipe.sh
set -euo pipefail
shred -u "$HOME/cs156-ml/pipeline1/.env" 2>/dev/null || rm -f "$HOME/cs156-ml/pipeline1/.env"
rm -rf "$HOME/cs156-ml" "$HOME/Minervas_scribe" "$HOME/fixcode" "$HOME"/*.log "$HOME/.cache/uv" \
    /tmp/scribe_* /tmp/*.py /tmp/*.csv /tmp/*.txt /tmp/*.png 2>/dev/null || true
history -c 2>/dev/null || true
echo "wiped. Now terminate the pod in the dashboard."
