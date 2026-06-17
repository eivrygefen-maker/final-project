#!/usr/bin/env bash
# Unified overnight FOM/M4 batch — one script for classic / box / acoustic.
# No STK, no WAV, no audio cache.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

SHAPE="${SHAPE:-box}"
COUNT="${COUNT:-100}"
START="${START:-0}"
STOP_AFTER_FAILURES="${STOP_AFTER_FAILURES:-3}"
WORKERS="${WORKERS:-3}"
RUN_ID_SUFFIX="${RUN_ID_SUFFIX:-}"
MESH_PROFILE="${MESH_PROFILE:-rom}"
SHARED_ROOT="${SHARED_ROOT:-/media/sf_gmar}"
ALLOW_CLASSIC_REGEN="${ALLOW_CLASSIC_REGEN:-0}"

M4_SCRIPT="${REPO_ROOT}/FEM/experiments/active_domain_validation/physics_integrity/scripts/run_m4_production_pipeline.py"
GEN_SCRIPT="${REPO_ROOT}/tools/generate_shape_lhs_pool.py"
FOM_RUNS_ROOT="${REPO_ROOT}/FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars"

case "${SHAPE}" in
  classic|box|acoustic) ;;
  *)
    echo "SHAPE_FOM_FAIL unknown_shape=${SHAPE} (expected classic|box|acoustic)"
    exit 2
    ;;
esac

SHAPE_LOWER="$(printf '%s' "${SHAPE}" | tr '[:upper:]' '[:lower:]')"
LHS_JSON="${LHS_JSON:-ROM/${SHAPE_LOWER}/lhs_pool.json}"
LHS_PATH="${REPO_ROOT}/${LHS_JSON}"
REPORT_ROOT="${REPO_ROOT}/FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/index/${SHAPE_LOWER}"
SHARED_EXPORT_ROOT="${SHARED_ROOT}/${SHAPE_LOWER}"
ROM_ROOT="${REPO_ROOT}/ROM/${SHAPE_LOWER}"

if [[ -z "${RUN_ID_SUFFIX}" ]]; then
  case "${SHAPE_LOWER}" in
    classic) RUN_ID_SUFFIX="m4prod1" ;;
    box) RUN_ID_SUFFIX="box_fom_v1" ;;
    acoustic) RUN_ID_SUFFIX="acoustic_fom_v1" ;;
  esac
fi

mkdir -p "${REPORT_ROOT}"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${REPORT_ROOT}/${SHAPE_LOWER}_fom_overnight_batch_${TS}.log"
SUMMARY_JSON="${REPORT_ROOT}/${SHAPE_LOWER}_fom_overnight_batch_${TS}_summary.json"
RESULTS_JSONL="${REPORT_ROOT}/${SHAPE_LOWER}_fom_overnight_batch_${TS}_results.jsonl"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

exec > >(tee -a "${LOG_FILE}") 2>&1

log() { echo "$*"; }

append_result() {
  local sample_id="$1" index="$2" status="$3" elapsed_s="${4:-0}" exit_code="${5:-0}" run_root="$6"
  SAMPLE_ID="${sample_id}" INDEX="${index}" STATUS="${status}" ELAPSED_S="${elapsed_s}" EXIT_CODE="${exit_code}" \
    RUN_ROOT="${run_root}" \
    python3 - <<'PY' >> "${RESULTS_JSONL}"
import json, os
print(json.dumps({
    "sample_id": os.environ["SAMPLE_ID"],
    "index": int(os.environ["INDEX"]),
    "status": os.environ["STATUS"],
    "elapsed_s": float(os.environ.get("ELAPSED_S") or 0),
    "exit_code": int(os.environ.get("EXIT_CODE") or 0),
    "run_root": os.environ["RUN_ROOT"],
}))
PY
}

write_summary() {
  local end_ts="${1:-$(date +%s)}"
  local batch_exit="${2:-0}"
  python3 - <<PY
import json
from datetime import datetime, timezone
from pathlib import Path

def utc_ts(ts):
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

results = []
results_path = Path("${RESULTS_JSONL}")
if results_path.is_file():
    for line in results_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            results.append(json.loads(line))

doc = {
    "pipeline": "m4_fom",
    "shape_name": "${SHAPE_LOWER}",
    "started_at": utc_ts(${BATCH_START_TS}),
    "finished_at": utc_ts(${end_ts}),
    "elapsed_s": ${end_ts} - ${BATCH_START_TS},
    "requested_count": ${COUNT},
    "start_index": ${START},
    "workers_per_sample": ${WORKERS},
    "run_id_suffix": "${RUN_ID_SUFFIX}",
    "mesh_profile": "${MESH_PROFILE}",
    "completed_count": ${COMPLETED},
    "skipped_count": ${SKIPPED},
    "failed_count": ${FAILED},
    "stop_after_failures": ${STOP_AFTER_FAILURES},
    "lhs_pool": "${LHS_PATH}",
    "rom_root": "${ROM_ROOT}",
    "fom_runs_root": "${FOM_RUNS_ROOT}",
    "report_root": "${REPORT_ROOT}",
    "shared_export_root": "${SHARED_EXPORT_ROOT}",
    "log_file": "${LOG_FILE}",
    "summary_json": "${SUMMARY_JSON}",
    "results_jsonl": "${RESULTS_JSONL}",
    "sample_results": results,
    "exit_code": ${batch_exit},
}
Path("${SUMMARY_JSON}").write_text(json.dumps(doc, indent=2) + "\\n", encoding="utf-8")
PY
}

BATCH_START_TS="$(date +%s)"
COMPLETED=0
SKIPPED=0
FAILED=0
FAILURES=0
BATCH_EXIT=1

finish_batch() {
  local code="${1:-1}"
  write_summary "$(date +%s)" "${code}"
  log "SHAPE_FOM_NIGHT_END shape=${SHAPE_LOWER} exit=${code} completed=${COMPLETED} skipped=${SKIPPED} failed=${FAILED} summary=${SUMMARY_JSON}"
  exit "${code}"
}

trap 'finish_batch "${BATCH_EXIT}"' EXIT

log "SHAPE_FOM_NIGHT_START shape=${SHAPE_LOWER} count=${COUNT} start=${START} workers=${WORKERS}"
log "SHAPE_FOM_PATH lhs=${LHS_PATH} rom=${ROM_ROOT} shared=${SHARED_EXPORT_ROOT}"

if [[ ! -f "${M4_SCRIPT}" ]]; then
  log "SHAPE_FOM_FAIL missing_m4_script=${M4_SCRIPT}"
  BATCH_EXIT=2
  exit 2
fi

if [[ "${SHAPE_LOWER}" == "classic" && "${ALLOW_CLASSIC_REGEN}" != "1" ]]; then
  if [[ ! -f "${LHS_PATH}" ]]; then
    log "SHAPE_FOM_FAIL classic_lhs_missing=${LHS_PATH} (set ALLOW_CLASSIC_REGEN=1 to generate)"
    BATCH_EXIT=2
    exit 2
  fi
  log "SHAPE_FOM_LHS_SKIP classic pool preserved path=${LHS_PATH}"
else
  if [[ -f "${LHS_PATH}" ]]; then
    log "SHAPE_FOM_LHS_ENSURE preserve_existing path=${LHS_PATH}"
    python3 "${GEN_SCRIPT}" --shape "${SHAPE_LOWER}" --pool-path "${LHS_PATH}" --ensure-existing
  else
    log "SHAPE_FOM_LHS_CREATE path=${LHS_PATH}"
    python3 "${GEN_SCRIPT}" --shape "${SHAPE_LOWER}" --pool-path "${LHS_PATH}"
  fi
fi

SAMPLE_PREFIX="$(python3 - <<PY
import sys
sys.path.insert(0, "${REPO_ROOT}/FEM/scripts")
from m4_shape_registry import resolve_shape_config
print(resolve_shape_config("${SHAPE_LOWER}").sample_id_prefix)
PY
)"

SAMPLE_COUNT="$(python3 - <<PY
import json
from pathlib import Path
pool = json.loads(Path("${LHS_PATH}").read_text(encoding="utf-8"))
prefix = "${SAMPLE_PREFIX}"
print(sum(1 for e in pool.get("entries") or [] if str(e.get("id","")).startswith(prefix)))
PY
)"
if [[ "${SAMPLE_COUNT}" -lt "$((START + COUNT))" ]]; then
  log "SHAPE_FOM_FAIL lhs_count=${SAMPLE_COUNT} expected_at_least=$((START + COUNT))"
  BATCH_EXIT=2
  exit 2
fi

END_IDX=$((START + COUNT))
idx="${START}"
while [[ "${idx}" -lt "${END_IDX}" ]]; do
  sample_id="$(printf '%s%03d' "${SAMPLE_PREFIX}" "${idx}")"
  run_root="${FOM_RUNS_ROOT}/${sample_id}/runs/${sample_id}_${RUN_ID_SUFFIX}"

  if python3 "${GEN_SCRIPT}" \
      --shape "${SHAPE_LOWER}" \
      --check-fom-ready "${sample_id}" \
      --run-id-suffix "${RUN_ID_SUFFIX}" \
      --pool-path "${LHS_PATH}" >/dev/null 2>&1; then
    log "SHAPE_FOM_SAMPLE_SKIP_READY sample_id=${sample_id} index=${idx}"
    SKIPPED=$((SKIPPED + 1))
    append_result "${sample_id}" "${idx}" "skipped_ready" 0 0 "${run_root}"
    idx=$((idx + 1))
    continue
  fi

  log "SHAPE_FOM_SAMPLE_START sample_id=${sample_id} index=${idx} workers=${WORKERS}"
  sample_start_ts="$(date +%s)"

  set +e
  python3 "${M4_SCRIPT}" \
    --shape "${SHAPE_LOWER}" \
    --force-sample "${sample_id}" \
    --max-samples 1 \
    --workers "${WORKERS}" \
    --execute \
    --run-id-suffix "${RUN_ID_SUFFIX}" \
    --mesh-profile "${MESH_PROFILE}" \
    --shared-root "${SHARED_ROOT}" \
    --continue-on-fail
  sample_exit=$?
  set -e

  sample_elapsed="$(( $(date +%s) - sample_start_ts ))"

  if [[ "${sample_exit}" -eq 0 ]]; then
    log "SHAPE_FOM_SAMPLE_READY sample_id=${sample_id} elapsed_s=${sample_elapsed}"
    COMPLETED=$((COMPLETED + 1))
    append_result "${sample_id}" "${idx}" "completed" "${sample_elapsed}" 0 "${run_root}"
  else
    log "SHAPE_FOM_SAMPLE_FAIL sample_id=${sample_id} exit_code=${sample_exit}"
    FAILED=$((FAILED + 1))
    FAILURES=$((FAILURES + 1))
    append_result "${sample_id}" "${idx}" "failed" "${sample_elapsed}" "${sample_exit}" "${run_root}"
    if [[ "${FAILURES}" -ge "${STOP_AFTER_FAILURES}" ]]; then
      log "SHAPE_FOM_STOP reason=stop_after_failures limit=${STOP_AFTER_FAILURES}"
      break
    fi
  fi

  idx=$((idx + 1))
done

if [[ "${COMPLETED}" -ge 1 ]]; then
  BATCH_EXIT=0
  exit 0
fi
if [[ "${SKIPPED}" -ge "${COUNT}" ]] && [[ "${FAILED}" -eq 0 ]]; then
  BATCH_EXIT=0
  exit 0
fi
BATCH_EXIT=1
exit 1
