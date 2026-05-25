#!/usr/bin/env bash
# Resume Phase-2 missing work only: baseline MAC reference (post-only), geometry coupled solves, material MAC refresh.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/FEM/scripts:${SCRIPT_DIR}:${PYTHONPATH:-}"

echo "[v2_production] Step 1: baseline structural MAC reference (no material re-solve)" >&2
mpiexec -n 1 python "${SCRIPT_DIR}/capture_baseline_structural_mac_reference.py"

echo "[v2_production] Step 2: geometry samples (locator + coupled; reuse meshes/gates)" >&2
python "${SCRIPT_DIR}/run_v2_sensitivity_production_stage.py" --resume --geometry-only

echo "[v2_production] Step 3: material MAC refresh only (no material re-solve)" >&2
python "${SCRIPT_DIR}/run_v2_sensitivity_production_stage.py" --resume --material-mac-only
