#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
EXP_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate
export OMP_NUM_THREADS=1 OMP_PROC_BIND=false OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
CONFIG="$EXP_ROOT/configs/sample_000_active_domain.json"
mkdir -p "$EXP_ROOT/active_domain/logs" "$EXP_ROOT/active_domain/timing"
/usr/bin/time -v -o "$EXP_ROOT/active_domain/timing/time_verbose.txt" \
  mpiexec -n 1 python "$EXP_ROOT/scripts/run_coupled_solve.py" \
    --variant active_domain \
    --config "$CONFIG" \
    --target-hz 202 \
    --num-modes 8 \
    --harvest-lo-hz 156 \
    --harvest-hi-hz 248 \
    --eps-broad-search-hz 46 \
  2>&1 | tee "$EXP_ROOT/active_domain/logs/solve_202hz.log"
python3 "$EXP_ROOT/scripts/parse_time_stats.py" \
  --verbose "$EXP_ROOT/active_domain/timing/time_verbose.txt" \
  --out "$EXP_ROOT/active_domain/timing/time_stats.json"
