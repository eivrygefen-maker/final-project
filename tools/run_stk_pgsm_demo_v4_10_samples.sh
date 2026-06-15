#!/usr/bin/env bash
# PGSM STK guitar demo v4 — 10 LHS samples × 3 notes (30 WAVs), STK/C++ renderer.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${REPO_ROOT}/cpp/stk_pgsm_guitar_demo/build"
PARAMS="${REPO_ROOT}/audio/debug_reports/pgsm_stk_demo_parameters_v4_10_samples.json"
BINARY="${BUILD_DIR}/stk_pgsm_guitar_demo"
OUT_DIR="${REPO_ROOT}/audio/pgsm_stk_guitar_demo_v4_10_samples"
REPORT="${REPO_ROOT}/audio/debug_reports/pgsm_stk_guitar_demo_v4_10_samples_report.json"

echo "== PGSM STK demo v4_10_samples (10 LHS samples, physical-factor audit) =="
cd "${REPO_ROOT}"

echo "Step 1: v4_10_samples parameter export (Python, no audio)"
python3 gui/pgsm_stk_parameter_export.py --demo-version v4_10_samples --output "${PARAMS}"

if [[ ! -x "${BINARY}" ]]; then
  echo "Binary not found: ${BINARY}"
  echo "Run tools/build_stk_pgsm_demo.sh first."
  exit 1
fi

echo "Step 2: STK/C++ render v4_10_samples (expected 30 WAVs)"
"${BINARY}" --params "${PARAMS}" --repo-root "${REPO_ROOT}"

WAV_COUNT="$(find "${OUT_DIR}" -maxdepth 1 -name '*.wav' 2>/dev/null | wc -l | tr -d ' ')"
echo "Done."
echo "WAV dir:   ${OUT_DIR}"
echo "WAV count: ${WAV_COUNT} (expected 30)"
echo "Report:    ${REPORT}"

if [[ "${WAV_COUNT}" -ne 30 ]]; then
  echo "ERROR: expected 30 WAV files, found ${WAV_COUNT}"
  exit 1
fi
