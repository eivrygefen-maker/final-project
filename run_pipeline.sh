#!/usr/bin/env bash
# LHS FEM pipeline wrapper: merges LHS JSON, runs fem_master_dynamic → tuner → package_rom → pool update.
#
# Usage:
#   ./run_pipeline.sh 15                    # single sample (--sample-id 15)
#   ./run_pipeline.sh sample_015            # single sample (explicit key)
#   ./run_pipeline.sh --marathon 16         # samples 16, 17, 18, … until Ctrl+C
#   ./run_pipeline.sh --marathon 16 --force # on failure: log and continue to next ID
#
# Overrides (optional):
#   RUN_PIPELINE_LHS, RUN_PIPELINE_POOL, RUN_PIPELINE_CONFIG — env vars
#   --lhs-samples PATH, --pool PATH, --config PATH — CLI (override env)
#
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Defaults (override with env or CLI)
LHS="${RUN_PIPELINE_LHS:-$ROOT/FEM/configs/lhs_samples.json}"
POOL="${RUN_PIPELINE_POOL:-}"
CONFIG="${RUN_PIPELINE_CONFIG:-$ROOT/FEM/configs/guitar_3d.json}"
MARATHON=0
START_RAW=""
FORCE_CONTINUE=0
POSITIONAL=()

usage() {
  cat <<'EOF'
LHS FEM pipeline (wraps FEM/scripts/run_pipeline.py).

  ./run_pipeline.sh 15                 Single run (--sample-id 15)
  ./run_pipeline.sh sample_015         Single run (explicit pool key)
  ./run_pipeline.sh -m 16                Marathon: 16, 17, 18, … until Ctrl+C
  ./run_pipeline.sh --marathon 16 --force   On failure: log and continue

Options: --marathon|-m START, --force|-f, --lhs-samples PATH, --pool PATH, --config PATH

Env defaults: RUN_PIPELINE_LHS, RUN_PIPELINE_POOL, RUN_PIPELINE_CONFIG

Logs: failures appended to logs/run_pipeline_errors.log
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --marathon|-m)
      MARATHON=1
      if [[ $# -lt 2 ]]; then echo "error: --marathon requires a starting sample index" >&2; exit 2; fi
      START_RAW="$2"
      shift 2
      ;;
    --force|-f)
      FORCE_CONTINUE=1
      shift
      ;;
    --lhs-samples)
      if [[ $# -lt 2 ]]; then echo "error: --lhs-samples requires a path" >&2; exit 2; fi
      LHS="$2"
      shift 2
      ;;
    --pool)
      if [[ $# -lt 2 ]]; then echo "error: --pool requires a path" >&2; exit 2; fi
      POOL="$2"
      shift 2
      ;;
    --config)
      if [[ $# -lt 2 ]]; then echo "error: --config requires a path" >&2; exit 2; fi
      CONFIG="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "error: unknown option: $1" >&2
      usage
      exit 2
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done

if [[ $MARATHON -eq 1 ]]; then
  if [[ ${#POSITIONAL[@]} -gt 0 ]]; then
    echo "error: extra arguments with --marathon: ${POSITIONAL[*]}" >&2
    exit 2
  fi
  if [[ -z "$START_RAW" ]]; then
    echo "error: --marathon needs a starting id" >&2
    exit 2
  fi
else
  if [[ ${#POSITIONAL[@]} -ne 1 ]]; then
    echo "error: provide exactly one sample id (e.g. 15 or sample_015), or use --marathon START" >&2
    usage
    exit 2
  fi
  START_RAW="${POSITIONAL[0]}"
fi

# --- Python from venv (activate if needed) ---
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  if [[ -f "$ROOT/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$ROOT/.venv/bin/activate"
  elif [[ -f "$ROOT/.venv/Scripts/activate" ]]; then
    # shellcheck disable=SC1091
    source "$ROOT/.venv/Scripts/activate"
  else
    echo "error: no project venv at $ROOT/.venv (expected bin/activate or Scripts/activate)" >&2
    exit 1
  fi
fi

if command -v python >/dev/null 2>&1; then
  PY=(python)
elif command -v python3 >/dev/null 2>&1; then
  PY=(python3)
else
  echo "error: python not found after venv activation" >&2
  exit 1
fi

PIPELINE_PY="$ROOT/FEM/scripts/run_pipeline.py"
if [[ ! -f "$PIPELINE_PY" ]]; then
  echo "error: missing $PIPELINE_PY" >&2
  exit 1
fi

LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"
ERR_LOG="$LOG_DIR/run_pipeline_errors.log"

parse_marathon_counter() {
  local s="$1"
  if [[ "$s" =~ ^sample_0*([0-9]+)$ ]]; then
    echo "${BASH_REMATCH[1]}"
  elif [[ "$s" =~ ^[0-9]+$ ]]; then
    echo "$s"
  else
    echo ""
  fi
}

run_one_sample() {
  local sample_arg="$1"
  echo ""
  echo ">>> $(date '+%Y-%m-%d %H:%M:%S')  sample-id=$sample_arg"
  local -a cmd=("${PY[@]}" "$PIPELINE_PY" "--sample-id" "$sample_arg" "--config" "$CONFIG")
  if [[ -f "$LHS" ]]; then
    cmd+=("--lhs-samples" "$LHS")
  else
    echo "[warn] LHS file not found: $LHS — continuing without --lhs-samples." >&2
  fi
  if [[ -n "$POOL" ]]; then
    cmd+=("--pool" "$POOL")
  fi
  echo "\$ ${cmd[*]}"
  if "${cmd[@]}"; then
    return 0
  fi
  return 1
}

if [[ $MARATHON -eq 0 ]]; then
  if ! run_one_sample "$START_RAW"; then
    ts="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[$ts] FAILED sample-id=$START_RAW" >>"$ERR_LOG"
    exit 1
  fi
  exit 0
fi

# --- Marathon: integer counter from START_RAW ---
counter="$(parse_marathon_counter "$START_RAW")"
if [[ -z "$counter" || ! "$counter" =~ ^[0-9]+$ ]]; then
  echo "error: --marathon start must be an integer or sample_NNN (got: $START_RAW)" >&2
  exit 2
fi

echo "Marathon from sample index $counter (Ctrl+C to stop). force-continue=$FORCE_CONTINUE"
trap 'echo ""; echo "Marathon interrupted."; exit 130' INT TERM

while true; do
  if run_one_sample "$counter"; then
    counter=$((counter + 1))
    continue
  fi
  ec=$?
  ts="$(date '+%Y-%m-%d %H:%M:%S')"
  {
    echo "[$ts] FAILED sample-id=$counter (exit=$ec)"
    echo "  command: ${PY[*]} $PIPELINE_PY --sample-id $counter --config $CONFIG ..."
  } >>"$ERR_LOG"
  echo "[error] Sample $counter failed — see $ERR_LOG"
  if [[ $FORCE_CONTINUE -eq 1 ]]; then
    echo "[info] --force: advancing to $((counter + 1))"
    counter=$((counter + 1))
    continue
  fi
  exit 1
done
