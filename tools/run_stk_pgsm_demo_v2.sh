#!/usr/bin/env bash
# PGSM STK guitar demo v2 — physical audit path + differentiated render.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${REPO_ROOT}/cpp/stk_pgsm_guitar_demo/build"
PARAMS="${REPO_ROOT}/audio/debug_reports/pgsm_stk_demo_parameters_v2.json"
BINARY="${BUILD_DIR}/stk_pgsm_guitar_demo"

echo "== PGSM STK demo v2 (physical difference audit) =="
cd "${REPO_ROOT}"

echo "Step 1: v2 parameter export (Python, no audio)"
python3 gui/pgsm_stk_parameter_export.py --demo-version v2 --output "${PARAMS}"

if [[ ! -x "${BINARY}" ]]; then
  echo "Binary not found: ${BINARY}"
  echo "Run tools/build_stk_pgsm_demo.sh first."
  exit 1
fi

echo "Step 2: STK/C++ render v2"
"${BINARY}" --params "${PARAMS}" --repo-root "${REPO_ROOT}"

echo "Done."
echo "WAV dir: ${REPO_ROOT}/audio/pgsm_stk_guitar_demo_v2"
echo "Report:  ${REPO_ROOT}/audio/debug_reports/pgsm_stk_guitar_demo_v2_report.json"
