#!/usr/bin/env bash
# Overnight BOX STK note-library batch — sequential samples, 3 workers inside each sample.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

INSTRUMENT="${INSTRUMENT:-box}"
COUNT="${COUNT:-40}"
START="${START:-0}"
STOP_AFTER_FAILURES="${STOP_AFTER_FAILURES:-3}"
STK_PARALLEL_WORKERS="${STK_PARALLEL_WORKERS:-3}"
RENDER_MODE="${RENDER_MODE:-parallel_batch}"
DURATION_S="${DURATION_S:-2.5}"

if [[ "${INSTRUMENT}" != "box" ]]; then
  echo "BOX_NIGHT_ISOLATION_FAIL instrument=${INSTRUMENT} (must be box)"
  exit 2
fi

LHS_POOL="${REPO_ROOT}/ROM/box/lhs_pool.json"
BOX_CACHE_ROOT="${REPO_ROOT}/audio/app_stk_note_cache/box"
BOX_REPORT_ROOT="${REPO_ROOT}/audio/debug_reports/box"
SHARED_EXPORT_ROOT="${SHARED_HOST_DIR:-/media/sf_gmar}/box/audio"
BUILD_SCRIPT="${REPO_ROOT}/tools/build_stk_pgsm_demo.sh"
BINARY="${REPO_ROOT}/cpp/stk_pgsm_guitar_demo/build/stk_pgsm_guitar_demo"

mkdir -p "${BOX_REPORT_ROOT}"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${BOX_REPORT_ROOT}/box_overnight_batch_${TS}.log"
SUMMARY_JSON="${BOX_REPORT_ROOT}/box_overnight_batch_${TS}_summary.json"
RESULTS_JSONL="${BOX_REPORT_ROOT}/box_overnight_batch_${TS}_results.jsonl"

exec > >(tee -a "${LOG_FILE}") 2>&1

log() {
  echo "$*"
}

append_result() {
  local sample_id="$1" index="$2" status="$3" elapsed_s="${4:-0}" exit_code="${5:-0}" cache_dir="$6" report_json="$7"
  SAMPLE_ID="${sample_id}" INDEX="${index}" STATUS="${status}" ELAPSED_S="${elapsed_s}" EXIT_CODE="${exit_code}" \
    CACHE_DIR="${cache_dir}" REPORT_JSON="${report_json}" \
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
            "cache_dir": os.environ["CACHE_DIR"],
            "report_json": os.environ["REPORT_JSON"],
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
    "instrument": "${INSTRUMENT}",
    "started_at": utc_ts(${BATCH_START_TS}),
    "finished_at": utc_ts(${end_ts}),
    "elapsed_s": ${end_ts} - ${BATCH_START_TS},
    "requested_count": ${COUNT},
    "start_index": ${START},
    "parallel_workers_per_sample": ${STK_PARALLEL_WORKERS},
    "render_mode": "${RENDER_MODE}",
    "completed_count": ${COMPLETED},
    "skipped_count": ${SKIPPED},
    "failed_count": ${FAILED},
    "stop_after_failures": ${STOP_AFTER_FAILURES},
    "log_file": "${LOG_FILE}",
    "summary_json": "${SUMMARY_JSON}",
    "results_jsonl": "${RESULTS_JSONL}",
    "lhs_pool": "${LHS_POOL}",
    "cache_root": "${BOX_CACHE_ROOT}",
    "report_root": "${BOX_REPORT_ROOT}",
    "shared_export_root": "${SHARED_EXPORT_ROOT}",
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
  log "BOX_NIGHT_BATCH_END exit=${code} completed=${COMPLETED} skipped=${SKIPPED} failed=${FAILED} summary=${SUMMARY_JSON}"
  exit "${code}"
}

trap 'finish_batch "${BATCH_EXIT}"' EXIT

log "BOX_NIGHT_BATCH_START timestamp=${TS} count=${COUNT} start=${START} workers=${STK_PARALLEL_WORKERS}"
log "BOX_NIGHT_ISOLATION_OK instrument=${INSTRUMENT}"
log "BOX_NIGHT_PATH lhs=${LHS_POOL}"
log "BOX_NIGHT_PATH cache_root=${BOX_CACHE_ROOT}"
log "BOX_NIGHT_PATH report_root=${BOX_REPORT_ROOT}"
log "BOX_NIGHT_PATH shared_export=${SHARED_EXPORT_ROOT}"

if [[ ! -f "${LHS_POOL}" ]]; then
  log "BOX_NIGHT_ISOLATION_FAIL missing_lhs_pool=${LHS_POOL}"
  BATCH_EXIT=2
  exit 2
fi

if [[ "${BOX_CACHE_ROOT}" != *"/app_stk_note_cache/box" ]]; then
  log "BOX_NIGHT_ISOLATION_FAIL bad_cache_root=${BOX_CACHE_ROOT}"
  BATCH_EXIT=2
  exit 2
fi

if [[ "${BOX_REPORT_ROOT}" != *"/debug_reports/box" ]]; then
  log "BOX_NIGHT_ISOLATION_FAIL bad_report_root=${BOX_REPORT_ROOT}"
  BATCH_EXIT=2
  exit 2
fi

if [[ "${SHARED_EXPORT_ROOT}" != *"/box/audio" ]]; then
  log "BOX_NIGHT_ISOLATION_FAIL bad_shared_export=${SHARED_EXPORT_ROOT}"
  BATCH_EXIT=2
  exit 2
fi

python3 tools/generate_box_lhs_pool.py --count "${COUNT}"

BOX_SAMPLE_COUNT="$(python3 - <<'PY'
import json
from pathlib import Path
pool = json.loads(Path("ROM/box/lhs_pool.json").read_text(encoding="utf-8"))
ids = [str(e.get("id")) for e in pool.get("entries") or [] if str(e.get("id", "")).startswith("box_sample_")]
print(len(ids))
PY
)"
if [[ "${BOX_SAMPLE_COUNT}" -lt "${COUNT}" ]]; then
  log "BOX_NIGHT_ISOLATION_FAIL lhs_count=${BOX_SAMPLE_COUNT} expected_at_least=${COUNT}"
  BATCH_EXIT=2
  exit 2
fi

log "Step: build/check STK renderer"
"${BUILD_SCRIPT}"
if [[ ! -x "${BINARY}" ]]; then
  log "BOX_NIGHT_FAIL reason=stk_binary_missing path=${BINARY}"
  BATCH_EXIT=2
  exit 2
fi

idx="${START}"
while [[ "${idx}" -lt "${COUNT}" ]]; do
  sample_id="$(printf 'box_sample_%03d' "${idx}")"
  cache_dir="${BOX_CACHE_ROOT}/${sample_id}"
  report_json="${BOX_REPORT_ROOT}/app_stk_note_library_box_${sample_id}_report.json"

  if python3 tools/generate_box_lhs_pool.py --check-ready "${sample_id}" >/dev/null 2>&1; then
    log "BOX_NIGHT_SAMPLE_SKIP_READY sample_id=${sample_id} index=${idx} cache_dir=${cache_dir}"
    SKIPPED=$((SKIPPED + 1))
    append_result "${sample_id}" "${idx}" "skipped_ready" 0 0 "${cache_dir}" "${report_json}"
    idx=$((idx + 1))
    continue
  fi

  log "BOX_NIGHT_SAMPLE_START sample_id=${sample_id} index=${idx} workers=${STK_PARALLEL_WORKERS} render_mode=${RENDER_MODE}"
  sample_start_ts="$(date +%s)"

  set +e
  python3 tools/build_app_stk_note_library.py \
    --sample-id "${sample_id}" \
    --instrument "${INSTRUMENT}" \
    --output-root "${REPO_ROOT}/audio/app_stk_note_cache" \
    --render-mode "${RENDER_MODE}" \
    --parallel-workers "${STK_PARALLEL_WORKERS}" \
    --duration-s "${DURATION_S}"
  sample_exit=$?
  set -e

  sample_elapsed="$(( $(date +%s) - sample_start_ts ))"

  if [[ "${sample_exit}" -eq 0 ]]; then
    log "BOX_NIGHT_SAMPLE_READY sample_id=${sample_id} elapsed_s=${sample_elapsed} cache_dir=${cache_dir}"
    COMPLETED=$((COMPLETED + 1))
    append_result "${sample_id}" "${idx}" "completed" "${sample_elapsed}" 0 "${cache_dir}" "${report_json}"
  else
    log "BOX_NIGHT_SAMPLE_FAIL sample_id=${sample_id} exit_code=${sample_exit} elapsed_s=${sample_elapsed}"
    FAILED=$((FAILED + 1))
    FAILURES=$((FAILURES + 1))
    append_result "${sample_id}" "${idx}" "failed" "${sample_elapsed}" "${sample_exit}" "${cache_dir}" "${report_json}"
    if [[ "${FAILURES}" -ge "${STOP_AFTER_FAILURES}" ]]; then
      log "BOX_NIGHT_STOP reason=stop_after_failures limit=${STOP_AFTER_FAILURES}"
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
