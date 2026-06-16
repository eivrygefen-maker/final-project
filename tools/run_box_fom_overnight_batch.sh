#!/usr/bin/env bash
# Overnight BOX FOM/M4 batch — LHS → full FEM per sample (3 workers inside each sample).
# No STK, no WAV, no audio cache.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

SHAPE_NAME="${SHAPE_NAME:-box}"
LHS_JSON="${LHS_JSON:-ROM/box/lhs_pool.json}"
COUNT="${COUNT:-40}"
START="${START:-0}"
STOP_AFTER_FAILURES="${STOP_AFTER_FAILURES:-3}"
WORKERS="${WORKERS:-3}"
RUN_ID_SUFFIX="${RUN_ID_SUFFIX:-box_fom_v1}"
MESH_PROFILE="${MESH_PROFILE:-rom}"
SHARED_ROOT="${SHARED_ROOT:-/media/sf_gmar}"
M4_SCRIPT="${REPO_ROOT}/FEM/experiments/active_domain_validation/physics_integrity/scripts/run_m4_production_pipeline.py"

BOX_LHS_PATH="${REPO_ROOT}/${LHS_JSON}"
BOX_CACHE_ROOT=""  # not used — FOM only
BOX_REPORT_ROOT="${REPO_ROOT}/FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/index/box"
BOX_FOM_RUNS_ROOT="${REPO_ROOT}/FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars"
SHARED_EXPORT_ROOT="${SHARED_ROOT}/box"
ROM_BOX_ROOT="${REPO_ROOT}/ROM/box"

mkdir -p "${BOX_REPORT_ROOT}"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${BOX_REPORT_ROOT}/box_fom_overnight_batch_${TS}.log"
SUMMARY_JSON="${BOX_REPORT_ROOT}/box_fom_overnight_batch_${TS}_summary.json"
RESULTS_JSONL="${BOX_REPORT_ROOT}/box_fom_overnight_batch_${TS}_results.jsonl"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

exec > >(tee -a "${LOG_FILE}") 2>&1

log() {
  echo "$*"
}

append_result() {
  local sample_id="$1" index="$2" status="$3" elapsed_s="${4:-0}" exit_code="${5:-0}" run_root="$6"
  SAMPLE_ID="${sample_id}" INDEX="${index}" STATUS="${status}" ELAPSED_S="${elapsed_s}" EXIT_CODE="${exit_code}" \
    RUN_ROOT="${run_root}" \
    python3 - <<'PY' >> "${RESULTS_JSONL}"
import json
import os
print(
    json.dumps(
        {
            "sample_id": os.environ["SAMPLE_ID"],
            "index": int(os.environ["INDEX"]),
            "status": os.environ["STATUS"],
            "elapsed_s": float(os.environ.get("ELAPSED_S") or 0),
            "exit_code": int(os.environ.get("EXIT_CODE") or 0),
            "run_root": os.environ["RUN_ROOT"],
        }
    )
)
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
    "shape_name": "${SHAPE_NAME}",
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
    "lhs_pool": "${BOX_LHS_PATH}",
    "rom_root": "${ROM_BOX_ROOT}",
    "fom_runs_root": "${BOX_FOM_RUNS_ROOT}",
    "report_root": "${BOX_REPORT_ROOT}",
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
  local end_ts
  end_ts="$(date +%s)"
  write_summary "${end_ts}" "${code}"
  log "BOX_FOM_NIGHT_BATCH_END exit=${code} completed=${COMPLETED} skipped=${SKIPPED} failed=${FAILED} summary=${SUMMARY_JSON}"
  exit "${code}"
}

trap 'finish_batch "${BATCH_EXIT}"' EXIT

log "BOX_FOM_NIGHT_BATCH_START timestamp=${TS} count=${COUNT} start=${START} workers=${WORKERS}"
log "BOX_FOM_ISOLATION_OK shape=${SHAPE_NAME}"
log "BOX_FOM_PATH lhs=${BOX_LHS_PATH}"
log "BOX_FOM_PATH rom=${ROM_BOX_ROOT}"
log "BOX_FOM_PATH fom_runs=${BOX_FOM_RUNS_ROOT}"
log "BOX_FOM_PATH reports=${BOX_REPORT_ROOT}"
log "BOX_FOM_PATH shared_export=${SHARED_EXPORT_ROOT}"

if [[ "${SHAPE_NAME}" != "box" ]]; then
  log "BOX_FOM_ISOLATION_FAIL shape=${SHAPE_NAME} (must be box)"
  BATCH_EXIT=2
  exit 2
fi

if [[ ! -f "${M4_SCRIPT}" ]]; then
  log "BOX_FOM_ISOLATION_FAIL missing_m4_script=${M4_SCRIPT}"
  BATCH_EXIT=2
  exit 2
fi

if [[ "${BOX_LHS_PATH}" == *"/ROM/classic/"* ]]; then
  log "BOX_FOM_ISOLATION_FAIL refused_classic_lhs=${BOX_LHS_PATH}"
  BATCH_EXIT=2
  exit 2
fi

python3 tools/generate_box_lhs_pool.py --count "${COUNT}" --pool-path "${BOX_LHS_PATH}"

BOX_SAMPLE_COUNT="$(python3 - <<PY
import json
from pathlib import Path
pool = json.loads(Path("${BOX_LHS_PATH}").read_text(encoding="utf-8"))
print(sum(1 for e in pool.get("entries") or [] if str(e.get("id","")).startswith("box_sample_")))
PY
)"
if [[ "${BOX_SAMPLE_COUNT}" -lt "${COUNT}" ]]; then
  log "BOX_FOM_ISOLATION_FAIL lhs_count=${BOX_SAMPLE_COUNT} expected_at_least=${COUNT}"
  BATCH_EXIT=2
  exit 2
fi

idx="${START}"
while [[ "${idx}" -lt "${COUNT}" ]]; do
  sample_id="$(printf 'box_sample_%03d' "${idx}")"
  run_root="${BOX_FOM_RUNS_ROOT}/${sample_id}/runs/${sample_id}_${RUN_ID_SUFFIX}"

  if python3 tools/generate_box_lhs_pool.py \
      --check-fom-ready "${sample_id}" \
      --run-id-suffix "${RUN_ID_SUFFIX}" \
      --pool-path "${BOX_LHS_PATH}" >/dev/null 2>&1; then
    log "BOX_FOM_SAMPLE_SKIP_READY sample_id=${sample_id} index=${idx} run_root=${run_root}"
    SKIPPED=$((SKIPPED + 1))
    append_result "${sample_id}" "${idx}" "skipped_ready" 0 0 "${run_root}"
    idx=$((idx + 1))
    continue
  fi

  log "BOX_FOM_SAMPLE_START sample_id=${sample_id} index=${idx} workers=${WORKERS} run_id_suffix=${RUN_ID_SUFFIX}"
  sample_start_ts="$(date +%s)"

  set +e
  python3 "${M4_SCRIPT}" \
    --lhs-json "${LHS_JSON}" \
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
    log "BOX_FOM_SAMPLE_READY sample_id=${sample_id} elapsed_s=${sample_elapsed} run_root=${run_root}"
    COMPLETED=$((COMPLETED + 1))
    append_result "${sample_id}" "${idx}" "completed" "${sample_elapsed}" 0 "${run_root}"
  else
    log "BOX_FOM_SAMPLE_FAIL sample_id=${sample_id} exit_code=${sample_exit} elapsed_s=${sample_elapsed}"
    FAILED=$((FAILED + 1))
    FAILURES=$((FAILURES + 1))
    append_result "${sample_id}" "${idx}" "failed" "${sample_elapsed}" "${sample_exit}" "${run_root}"
    if [[ "${FAILURES}" -ge "${STOP_AFTER_FAILURES}" ]]; then
      log "BOX_FOM_STOP reason=stop_after_failures limit=${STOP_AFTER_FAILURES}"
      break
    fi
  fi

  idx=$((idx + 1))
done

if [[ "${COMPLETED}" -ge 1 ]]; then
  BATCH_EXIT=0
  exit 0
fi
if [[ "${SKIPPED}" -ge "$((COUNT - START))" ]] && [[ "${FAILED}" -eq 0 ]]; then
  BATCH_EXIT=0
  exit 0
fi
BATCH_EXIT=1
exit 1
