#!/usr/bin/env bash
# On the POD, once: system tools, uv, the project's Python environment and Scribe.
#   bash ~/cs156-ml/pipeline1/pod/setup.sh
# Works on x86-64 and Arm (e.g. the GH200's Grace CPU), Ubuntu/Debian images.
set -euo pipefail
SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
echo "== system packages"
$SUDO apt-get update -y -qq
$SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  ffmpeg tesseract-ocr tesseract-ocr-eng tmux rsync curl ca-certificates gnupg poppler-utils >/dev/null
echo "== Tesseract 5 (the pilot used 5.5; Ubuntu 22.04 ships 4.1)"
if ! tesseract --version 2>&1 | head -1 | grep -q "tesseract 5"; then
  if grep -q "jammy" /etc/os-release 2>/dev/null; then
    FP=$(curl -fsSL https://api.launchpad.net/1.0/~alex-p/+archive/ubuntu/tesseract-ocr5 \
         | grep -o '"signing_key_fingerprint": *"[0-9A-F]*"' | grep -o '[0-9A-F]\{40\}')
    curl -fsSL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x$FP" \
      | gpg --dearmor | $SUDO tee /etc/apt/trusted.gpg.d/tesseract-ocr5.gpg >/dev/null
    echo "deb https://ppa.launchpadcontent.net/alex-p/tesseract-ocr5/ubuntu jammy main" \
      | $SUDO tee /etc/apt/sources.list.d/tesseract-ocr5.list >/dev/null
    $SUDO apt-get update -qq >/dev/null
    $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq tesseract-ocr tesseract-ocr-eng >/dev/null
  fi
fi
tesseract --version 2>&1 | head -1
echo "== uv"
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/cs156-ml"
echo "== Python environment (uv.lock)"
# Python 3.12, as on the laptop: uv would otherwise take the newest it finds (3.15), for
# which torch has no wheels yet.
rm -rf .venv
uv sync --frozen --python 3.12 || { echo "frozen sync failed; resolving again"; uv sync --python 3.12; }
echo "== Scribe (scribe_core) into the same environment"
uv pip install --python .venv/bin/python -e "$HOME/Minervas_scribe"
.venv/bin/python -c "import oncue, scribe_core, cv2, pandas, pyarrow, reportlab; print('imports ok')"
chmod 600 pipeline1/.env 2>/dev/null || true
Q=$(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo max); echo "== cgroup CPU limit: $Q (nproc says $(nproc)); $(free -g | awk '/Mem:/{print $2}') GB RAM, $(df -h ~ | awk 'NR==2{print $4}') disk free"
echo "done. Next: cd ~/cs156-ml/pipeline1 && ../.venv/bin/python pod/preflight.py"
