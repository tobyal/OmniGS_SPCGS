#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
REVISION=628784299fd5c01394be9beeae052e63bc533ee5
BASE_PYTHON=${WAN360_BOOTSTRAP_PYTHON:-/mnt/sda/ubuntu/anaconda3/envs/jxy_sam2/bin/python}
mkdir -p "$ROOT/vendor" "$ROOT/weights/Wan360" "$ROOT/logs"
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$ROOT/weights/pip_wan360_cache}
if [ ! -d "$ROOT/vendor/MUGEN/.git" ]; then
    git clone https://github.com/AlayaLab/MUGEN.git "$ROOT/vendor/MUGEN"
fi
git -C "$ROOT/vendor/MUGEN" checkout --detach "$REVISION"
if [ ! -x "$ROOT/.venv_wan360/bin/python" ]; then
    "$BASE_PYTHON" -m venv "$ROOT/.venv_wan360"
fi
PY="$ROOT/.venv_wan360/bin/python"
"$PY" -m pip install --upgrade pip
if [ -n "${WAN360_WHEELHOUSE:-}" ]; then
    # Explicit local paths prevent pip from choosing an equivalent remote wheel
    # ahead of --find-links and downloading a large runtime a second time.
    "$PY" -m pip install --no-index --no-deps "$WAN360_WHEELHOUSE/"*.whl
fi
"$PY" -m pip install torch==2.6.0 torchvision==0.21.0 --index-url "${WAN360_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
"$PY" -m pip install -e "$ROOT/vendor/MUGEN/model"
"$PY" -m pip check
"$PY" -m pip freeze > "$ROOT/weights/Wan360/python_packages.txt"
if [ "${1:-}" != '--env-only' ]; then
    "$PY" "$ROOT/tools/download_wan360.py"
fi
