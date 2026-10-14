#!/usr/bin/env bash
# On the POD, once: system tools, uv, the project's Python environment and Scribe.
#   bash ~/cs156-ml/pipeline1/pod/setup.sh
# Works on x86-64 and Arm (e.g. the GH200's Grace CPU), Ubuntu/Debian images.
set -euo pipefail
SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
echo "== system packages"
$SUDO apt-get update -y -qq
$SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  ffmpeg tesseract-ocr tesseract-ocr-eng tmux rsync curl ca-certificates poppler-utils >/dev/null
echo "== uv"
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/cs156-ml"
echo "== Python environment (uv.lock)"
uv sync --frozen || { echo "frozen sync failed on this platform; resolving again"; uv sync; }
echo "== Scribe (scribe_core) into the same environment"
uv pip install --python .venv/bin/python -e "$HOME/Minervas_scribe"
.venv/bin/python -c "import oncue, scribe_core, cv2, pandas, pyarrow, reportlab; print('imports ok')"
chmod 600 pipeline1/.env 2>/dev/null || true
echo "== $(nproc) CPU cores, $(free -g | awk '/Mem:/{print $2}') GB RAM, $(df -h ~ | awk 'NR==2{print $4}') disk free"
echo "done. Next: cd ~/cs156-ml/pipeline1 && ../.venv/bin/python pod/preflight.py"
