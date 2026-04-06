#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing ${PYTHON_BIN}. Run ./setup_rl_demos.sh first." >&2
  exit 1
fi

exec "${PYTHON_BIN}" "${ROOT_DIR}/car-track-rl/evaluate.py" \
  --checkpoint "${ROOT_DIR}/car-track-rl/artifacts_run13_curriculum_fromscratch_race/ppo_best_model.pt" \
  --episodes 5 \
  --device auto \
  --render human \
  --delay 0.02
