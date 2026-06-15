#!/usr/bin/env bash
# PGSM STK guitar demo v4 — 10 LHS samples × 3 notes (30 WAVs) + stitched listening files.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${REPO_ROOT}/cpp/stk_pgsm_guitar_demo/build"
MAIN_CPP="${REPO_ROOT}/cpp/stk_pgsm_guitar_demo/main.cpp"
PARAMS="${REPO_ROOT}/audio/debug_reports/pgsm_stk_demo_parameters_v4_10_samples.json"
BINARY="${BUILD_DIR}/stk_pgsm_guitar_demo"
OUT_DIR="${REPO_ROOT}/audio/pgsm_stk_guitar_demo_v4_10_samples"
STITCHED_DIR="${REPO_ROOT}/audio/pgsm_stk_guitar_demo_v4_10_samples_stitched"
REPORT="${REPO_ROOT}/audio/debug_reports/pgsm_stk_guitar_demo_v4_10_samples_report.json"
STITCH_REPORT="${REPO_ROOT}/audio/debug_reports/pgsm_stk_guitar_demo_v4_10_samples_stitched_report.json"

SAMPLES=(
  sample_000 sample_001 sample_002 sample_003 sample_004
  sample_005 sample_006 sample_007 sample_008 sample_009
)

require_fresh_binary() {
  if [[ ! -f "${MAIN_CPP}" ]]; then
    echo "ERROR: missing source file: ${MAIN_CPP}"
    exit 1
  fi
  if [[ ! -x "${BINARY}" ]]; then
    echo "ERROR: renderer binary missing: ${BINARY}"
    echo "Run: ${REPO_ROOT}/tools/build_stk_pgsm_demo.sh"
    exit 1
  fi
  if [[ "${MAIN_CPP}" -nt "${BINARY}" ]]; then
    echo "ERROR: renderer binary is stale (main.cpp newer than binary)."
    echo "Rebuild required: ${REPO_ROOT}/tools/build_stk_pgsm_demo.sh"
    exit 1
  fi
}

echo "== PGSM STK demo v4_10_samples (10 LHS samples, physical-factor audit) =="
cd "${REPO_ROOT}"

echo "Step 1: v4_10_samples parameter export (Python, no audio)"
python3 gui/pgsm_stk_parameter_export.py --demo-version v4_10_samples --output "${PARAMS}"

require_fresh_binary

echo "Step 2: STK/C++ render v4_10_samples (expected 30 WAVs)"
"${BINARY}" --params "${PARAMS}" --repo-root "${REPO_ROOT}"

WAV_COUNT="$(find "${OUT_DIR}" -maxdepth 1 -name '*.wav' 2>/dev/null | wc -l | tr -d ' ')"
echo "WAV dir:   ${OUT_DIR}"
echo "WAV count: ${WAV_COUNT} (expected 30)"
echo "Report:    ${REPORT}"

if [[ "${WAV_COUNT}" -ne 30 ]]; then
  echo "ERROR: expected 30 WAV files, found ${WAV_COUNT}"
  exit 1
fi

echo "Step 3: stitch listening WAVs (3 per-note comparison files)"
python3 "${REPO_ROOT}/tools/stitch_stk_listening_wavs.py" \
  --input-dir "${OUT_DIR}" \
  --output-dir "${STITCHED_DIR}" \
  --samples "${SAMPLES[@]}" \
  --notes A2 A4 E5 \
  --segment-seconds 3.5 \
  --gap-seconds 0.3 \
  --report-json "${STITCH_REPORT}" \
  --report-md "${REPO_ROOT}/audio/debug_reports/pgsm_stk_guitar_demo_v4_10_samples_stitched_report.md"

STITCH_COUNT="$(find "${STITCHED_DIR}" -maxdepth 1 -name '*_stitched.wav' 2>/dev/null | wc -l | tr -d ' ')"
echo "Done."
echo "Stitched dir:   ${STITCHED_DIR}"
echo "Stitched count: ${STITCH_COUNT} (expected 3)"
echo "Stitch report:  ${STITCH_REPORT}"

if [[ "${STITCH_COUNT}" -ne 3 ]]; then
  echo "ERROR: expected 3 stitched WAV files, found ${STITCH_COUNT}"
  exit 1
fi
