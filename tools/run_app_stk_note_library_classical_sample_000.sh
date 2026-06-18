#!/usr/bin/env bash
# Build classical STK note library for sample_000 (E2:E5 chromatic).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_SCRIPT="${REPO_ROOT}/tools/build_stk_pgsm_demo.sh"
MAIN_CPP="${REPO_ROOT}/cpp/stk_pgsm_guitar_demo/main.cpp"
BINARY="${REPO_ROOT}/cpp/stk_pgsm_guitar_demo/build/stk_pgsm_guitar_demo"
OUT_DIR="${REPO_ROOT}/audio/app_stk_note_cache/classical/sample_000"

echo "== APP STK note library — classical / sample_000 =="
cd "${REPO_ROOT}"

echo "Step 1: build/check STK renderer"
"${BUILD_SCRIPT}"
if [[ ! -x "${BINARY}" ]]; then
  echo "ERROR: STK binary missing after build: ${BINARY}"
  exit 1
fi
if [[ "${MAIN_CPP}" -nt "${BINARY}" ]]; then
  echo "ERROR: STK binary stale after build"
  exit 1
fi

echo "Step 2: generate note library (E2:E5 chromatic)"
START_TS="$(date +%s)"
python3 tools/build_app_stk_note_library.py \
  --sample-id sample_000 \
  --shape-type Classical \
  --note-range E2:E5 \
  --output-root "${REPO_ROOT}/audio/app_stk_note_cache" \
  --duration-s 2.5
END_TS="$(date +%s)"
ELAPSED="$((END_TS - START_TS))"

NOTE_COUNT="$(find "${OUT_DIR}" -maxdepth 1 -name '*.wav' 2>/dev/null | wc -l | tr -d ' ')"
EXPECTED=37

echo ""
echo "Total wall time: ${ELAPSED} s"
echo "Note WAV count: ${NOTE_COUNT} (expected ${EXPECTED})"
echo "Output dir: ${OUT_DIR}"

if [[ "${NOTE_COUNT}" -lt "${EXPECTED}" ]]; then
  echo "ERROR: missing note WAVs (expected at least ${EXPECTED})"
  exit 1
fi

echo "Done."
