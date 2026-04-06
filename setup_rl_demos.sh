#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

python3 -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/pip" install -r "${ROOT_DIR}/cartpole-rl/requirements.txt" -r "${ROOT_DIR}/car-track-rl/requirements.txt"

cat <<'EOF'
Setup complete.

Run the demos with:
  ./run_cartpole_demo.sh
  ./run_car_track_demo.sh
EOF
