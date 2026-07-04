#!/usr/bin/env bash
set -euo pipefail

VENV_PATH="${ISAAC_VENV_PATH:-$HOME/isaacsim6-env}"
UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"

if [[ ! -x "$UV_BIN" ]]; then
  curl -LsSf https://astral.sh/uv/install.sh -o /tmp/uv-install.sh
  sh /tmp/uv-install.sh
fi

"$UV_BIN" python install 3.12
"$UV_BIN" venv --python 3.12 "$VENV_PATH"
"$VENV_PATH/bin/python" -m ensurepip --upgrade || true
"$VENV_PATH/bin/python" -m pip install --upgrade pip

"$VENV_PATH/bin/python" -m pip install \
  torch==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu130

"$VENV_PATH/bin/python" -m pip install \
  "isaacsim[all,extscache]==6.0.0.1" \
  --extra-index-url https://pypi.nvidia.com

"$VENV_PATH/bin/python" -m pip install -e ".[dev]"

cat <<EOF
Isaac Sim environment is ready at:
  $VENV_PATH

For headed Isaac Sim runs from WSLg:
  export OMNI_KIT_ACCEPT_EULA=YES
  python scripts/run_isaac_warehouse_slam.py --headed
EOF

