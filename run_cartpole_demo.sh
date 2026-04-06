#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing ${PYTHON_BIN}. Run ./setup_rl_demos.sh first." >&2
  exit 1
fi

exec "${PYTHON_BIN}" "${ROOT_DIR}/cartpole-rl/evaluate_dqn.py" \
  --checkpoint "${ROOT_DIR}/cartpole-rl/dqn_gpu_tuned/dqn_best_model.pt" \
  --episodes 3 \
  --device auto \
  --render human \
  --delay 0.03
