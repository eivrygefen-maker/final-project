#!/usr/bin/env bash
# PGSM STK guitar demo v3 — stronger perceptual differentiation (STK guitar tone preserved).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${REPO_ROOT}/cpp/stk_pgsm_guitar_demo/build"
PARAMS="${REPO_ROOT}/audio/debug_reports/pgsm_stk_demo_parameters_v3.json"
BINARY="${BUILD_DIR}/stk_pgsm_guitar_demo"

echo "== PGSM STK demo v3 (perceptual differentiation) =="
cd "${REPO_ROOT}"

echo "Step 1: v3 parameter export (Python, no audio)"
python3 gui/pgsm_stk_parameter_export.py --demo-version v3 --output "${PARAMS}"

if [[ ! -x "${BINARY}" ]]; then
  echo "Binary not found: ${BINARY}"
  echo "Run tools/build_stk_pgsm_demo.sh first."
  exit 1
fi

echo "Step 2: STK/C++ render v3"
"${BINARY}" --params "${PARAMS}" --repo-root "${REPO_ROOT}"

echo "Done."
echo "WAV dir: ${REPO_ROOT}/audio/pgsm_stk_guitar_demo_v3"
echo "Report:  ${REPO_ROOT}/audio/debug_reports/pgsm_stk_guitar_demo_v3_report.json"
