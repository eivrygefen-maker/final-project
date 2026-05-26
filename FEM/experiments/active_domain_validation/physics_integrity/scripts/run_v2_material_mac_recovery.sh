#!/usr/bin/env bash
# Report-only: fix reduced u_to_W MAC indexing and refresh material structural validation status.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/FEM/scripts:${SCRIPT_DIR}:${PYTHONPATH:-}"
python "${SCRIPT_DIR}/run_v2_material_mac_recovery.py" "$@"
