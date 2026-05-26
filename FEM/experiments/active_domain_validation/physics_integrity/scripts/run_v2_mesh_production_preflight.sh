#!/usr/bin/env bash
# L_prod production-mesh gates-only preflight + L_mid solve-log diagnosis (no eigen solves).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/FEM/scripts:${SCRIPT_DIR}:${PYTHONPATH:-}"
python "${SCRIPT_DIR}/run_v2_mesh_production_preflight.py" "$@"
