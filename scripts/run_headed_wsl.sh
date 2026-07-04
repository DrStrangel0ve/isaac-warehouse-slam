#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${ISAAC_VENV_PATH:-$HOME/isaacsim6-env}"

cd "$PROJECT_DIR"
source "$VENV_PATH/bin/activate"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"

python scripts/run_isaac_warehouse_slam.py \
  --headed \
  --window-width 1280 \
  --window-height 720 \
  "$@"

