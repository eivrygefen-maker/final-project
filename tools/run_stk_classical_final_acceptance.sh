#!/usr/bin/env bash
# Classical-guitar STK final acceptance: v4 render + stitch + factor acceptance audit.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${REPO_ROOT}/cpp/stk_pgsm_guitar_demo/build"
MAIN_CPP="${REPO_ROOT}/cpp/stk_pgsm_guitar_demo/main.cpp"
BINARY="${BUILD_DIR}/stk_pgsm_guitar_demo"
BUILD_SCRIPT="${REPO_ROOT}/tools/build_stk_pgsm_demo.sh"
V4_RUN="${REPO_ROOT}/tools/run_stk_pgsm_demo_v4_10_samples.sh"
ACCEPTANCE_JSON="${REPO_ROOT}/audio/debug_reports/pgsm_stk_classical_final_acceptance_report.json"
ACCEPTANCE_MD="${REPO_ROOT}/audio/debug_reports/pgsm_stk_classical_final_acceptance_report.md"
RENDER_REPORT="${REPO_ROOT}/audio/debug_reports/pgsm_stk_guitar_demo_v4_10_samples_report.json"

echo "== PGSM STK classical final acceptance audit =="
cd "${REPO_ROOT}"

echo "Step 1: build STK renderer (fail fast on compile errors)"
if [[ ! -x "${BUILD_SCRIPT}" ]]; then
  echo "ERROR: missing build script: ${BUILD_SCRIPT}"
  exit 1
fi
"${BUILD_SCRIPT}"

if [[ ! -x "${BINARY}" ]]; then
  echo "ERROR: build succeeded but binary missing: ${BINARY}"
  exit 1
fi
if [[ "${MAIN_CPP}" -nt "${BINARY}" ]]; then
  echo "ERROR: binary still stale after build — refusing to run old renderer"
  exit 1
fi

echo "Step 2: v4_10_samples export, render (30 WAVs), stitch (3 listening WAVs)"
if [[ ! -x "${V4_RUN}" ]]; then
  echo "ERROR: missing v4 run helper: ${V4_RUN}"
  exit 1
fi
"${V4_RUN}"

echo "Step 3: validate pluck amplitude handling in render report"
if ! python3 - <<'PY'
import json, sys
from pathlib import Path
p = Path("audio/debug_reports/pgsm_stk_guitar_demo_v4_10_samples_report.json")
doc = json.loads(p.read_text(encoding="utf-8"))
if doc.get("pluck_amplitude_handling") != "clamped_to_stk_0_1_range":
    print("ERROR: pluck_amplitude_handling not set in render report")
    sys.exit(1)
audit = doc.get("pluck_amplitude_audit") or []
if not audit:
    print("ERROR: pluck_amplitude_audit missing from render report")
    sys.exit(1)
for row in audit:
    raw = float(row.get("raw_pluck_amplitude") or 0)
    clamped = float(row.get("clamped_pluck_amplitude") or 0)
    was = bool(row.get("was_clamped"))
    sid = row.get("sample_id")
    note = row.get("note_name")
    if raw < 0 or raw > 1.0:
        if not was:
            print(f"ERROR: unhandled out-of-range pluck: {sid} {note} raw={raw}")
            sys.exit(1)
    if clamped < 0 or clamped > 1.0:
        print(f"ERROR: clamped pluck still illegal: {sid} {note} clamped={clamped}")
        sys.exit(1)
print(f"pluck_amplitude_audit: {len(audit)} entries OK")
PY
then
  exit 1
fi

echo "Step 4: write final classical acceptance report"
python3 "${REPO_ROOT}/tools/write_stk_classical_final_acceptance.py" \
  --render-report "${RENDER_REPORT}" \
  --output-json "${ACCEPTANCE_JSON}" \
  --output-md "${ACCEPTANCE_MD}"

if [[ ! -f "${ACCEPTANCE_JSON}" ]]; then
  echo "ERROR: acceptance report not written: ${ACCEPTANCE_JSON}"
  exit 1
fi

DECISION="$(python3 - <<'PY'
import json
from pathlib import Path
doc = json.loads(Path("audio/debug_reports/pgsm_stk_classical_final_acceptance_report.json").read_text())
print(doc.get("classical_stk_acceptance_decision") or "")
PY
)"

if [[ -z "${DECISION}" ]]; then
  echo "ERROR: classical_stk_acceptance_decision missing from acceptance report"
  exit 1
fi

echo "Done."
echo "Acceptance report: ${ACCEPTANCE_JSON}"
echo "Decision: ${DECISION}"

if [[ "${DECISION}" == "not_accepted_missing_factor_activation" ]]; then
  echo "ERROR: classical STK path not accepted — see acceptance report"
  exit 1
fi
