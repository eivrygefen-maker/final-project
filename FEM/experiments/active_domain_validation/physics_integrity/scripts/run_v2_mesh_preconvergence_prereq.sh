#!/usr/bin/env bash
# (1) L_prod finite-area gate revalidation on repaired meshes; (2) L_mid acoustic locator rescue.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/FEM/scripts:${SCRIPT_DIR}:${PYTHONPATH:-}"

echo "[prereq] Step 1: L_prod gates-only revalidation (finite area_ratio, fail-closed)" >&2
python "${SCRIPT_DIR}/run_v2_mesh_production_preflight.py" --gates-only-revalidate

echo "[prereq] Step 2: L_mid acoustic locator + targeted coupled capture" >&2
python "${SCRIPT_DIR}/run_v2_l_mid_acoustic_locator_rescue.py"
