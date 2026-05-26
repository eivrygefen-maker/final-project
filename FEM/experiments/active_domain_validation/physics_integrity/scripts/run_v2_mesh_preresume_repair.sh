#!/usr/bin/env bash
# Pre-resume repair: L_prod finite-area gate revalidation + L_mid acoustic branch rescue.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/FEM/scripts:${SCRIPT_DIR}:${PYTHONPATH:-}"
python "${SCRIPT_DIR}/run_v2_mesh_preresume_repair.py" "$@"
