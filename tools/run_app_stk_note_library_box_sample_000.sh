#!/usr/bin/env bash
# Build BOX STK note library for box_sample_000 (E2:E5 chromatic, isolated from CLASSIC).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_SCRIPT="${REPO_ROOT}/tools/build_stk_pgsm_demo.sh"
MAIN_CPP="${REPO_ROOT}/cpp/stk_pgsm_guitar_demo/main.cpp"
BINARY="${REPO_ROOT}/cpp/stk_pgsm_guitar_demo/build/stk_pgsm_guitar_demo"
SAMPLE_ID="box_sample_000"
INSTRUMENT="box"
OUT_DIR="${REPO_ROOT}/audio/app_stk_note_cache/box/${SAMPLE_ID}"
REPORT_DIR="${REPO_ROOT}/audio/debug_reports/box"
SHARED_AUDIO="${SHARED_HOST_DIR:-/media/sf_gmar}/box/audio"

echo "== APP STK note library — box / ${SAMPLE_ID} =="
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

echo "Step 2: verify BOX LHS pool"
LHS_POOL="${REPO_ROOT}/ROM/box/lhs_pool.json"
if [[ ! -f "${LHS_POOL}" ]]; then
  echo "ERROR: missing BOX LHS pool: ${LHS_POOL}"
  exit 1
fi

echo "Step 3: generate note library (E2:E5 chromatic, parallel_batch x3)"
START_TS="$(date +%s)"
python3 tools/build_app_stk_note_library.py \
  --sample-id "${SAMPLE_ID}" \
  --instrument "${INSTRUMENT}" \
  --note-range E2:E5 \
  --output-root "${REPO_ROOT}/audio/app_stk_note_cache" \
  --duration-s 2.5
END_TS="$(date +%s)"
ELAPSED="$((END_TS - START_TS))"

NOTE_COUNT="$(find "${OUT_DIR}" -maxdepth 1 -name '*.wav' 2>/dev/null | wc -l | tr -d ' ')"
POSITION_WAV="$(find "${OUT_DIR}" -maxdepth 1 -name 'S6_f2.wav' 2>/dev/null | head -1 || true)"
REPORT_COUNT="$(find "${REPORT_DIR}" -maxdepth 1 -type f -name '*box*' 2>/dev/null | wc -l | tr -d ' ')"
EXPECTED=37

echo ""
echo "Total wall time: ${ELAPSED} s"
echo "Note WAV count: ${NOTE_COUNT} (expected ${EXPECTED})"
echo "Position alias S6_f2.wav: ${POSITION_WAV:-missing}"
echo "BOX debug reports: ${REPORT_COUNT} file(s) under ${REPORT_DIR}"
echo "Output dir: ${OUT_DIR}"
echo "Expected shared export root: ${SHARED_AUDIO}/"

if [[ "${NOTE_COUNT}" -lt "${EXPECTED}" ]]; then
  echo "ERROR: missing note WAVs (expected at least ${EXPECTED})"
  exit 1
fi

if [[ -z "${POSITION_WAV}" ]]; then
  echo "ERROR: missing position WAV alias S6_f2.wav"
  exit 1
fi

if [[ -d "${OUT_DIR}" ]] && [[ "$(realpath "${OUT_DIR}")" == *"/app_stk_note_cache/classical/"* ]]; then
  echo "ERROR: BOX output landed under classical cache (path isolation failed)"
  exit 1
fi

echo "Done."
