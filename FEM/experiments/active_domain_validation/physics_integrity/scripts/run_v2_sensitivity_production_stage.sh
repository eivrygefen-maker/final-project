#!/usr/bin/env bash
# Phase-2 production-parameter validation only (no phase-1 radius/depth rerun).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/FEM/scripts:${SCRIPT_DIR}:${PYTHONPATH:-}"
python "${SCRIPT_DIR}/run_v2_sensitivity_production_stage.py" --resume "$@"
