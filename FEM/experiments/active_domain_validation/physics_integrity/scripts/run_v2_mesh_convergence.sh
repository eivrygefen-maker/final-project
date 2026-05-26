#!/usr/bin/env bash
# Resumable v2 mesh convergence (L0 ingest + L_mid/L_prod/L_check solves + post).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/FEM/scripts:${SCRIPT_DIR}:${PYTHONPATH:-}"
python "${SCRIPT_DIR}/run_v2_mesh_convergence.py" --resume "$@"
