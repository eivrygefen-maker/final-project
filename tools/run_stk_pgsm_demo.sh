#!/usr/bin/env bash
# Export PGSM parameters (Python) then render 9 demo WAVs via STK/C++ on VM.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${REPO_ROOT}/cpp/stk_pgsm_guitar_demo/build"
PARAMS="${REPO_ROOT}/audio/debug_reports/pgsm_stk_demo_parameters.json"
BINARY="${BUILD_DIR}/stk_pgsm_guitar_demo"

echo "== PGSM STK demo run =="
cd "${REPO_ROOT}"

echo "Step 1: parameter export (Python, no audio)"
python3 gui/pgsm_stk_parameter_export.py --output "${PARAMS}"

if [[ ! -x "${BINARY}" ]]; then
  echo "Binary not found: ${BINARY}"
  echo "Run tools/build_stk_pgsm_demo.sh first."
  exit 1
fi

echo "Step 2: STK/C++ render"
"${BINARY}" --params "${PARAMS}" --repo-root "${REPO_ROOT}"

echo "Done."
echo "WAV dir: ${REPO_ROOT}/audio/pgsm_stk_guitar_demo"
echo "Report:  ${REPO_ROOT}/audio/debug_reports/pgsm_stk_guitar_demo_report.json"
