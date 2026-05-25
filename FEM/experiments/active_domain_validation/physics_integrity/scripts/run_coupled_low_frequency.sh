#!/usr/bin/env bash
# TEST 4 — coupled solve targeted for low-frequency / air-sensitive band
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
cd "$ROOT"
source .venv/bin/activate 2>/dev/null || true
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
python FEM/experiments/active_domain_validation/physics_integrity/scripts/prepare_physics_configs.py
# After TEST 3, set shift_invert_target_hz in configs/coupled_low_frequency.json to the
# first acoustic candidate if needed, then run:
mpiexec -n 1 python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_physics_case.py \
  --case coupled_low_frequency \
  --config FEM/experiments/active_domain_validation/physics_integrity/configs/coupled_low_frequency.json \
  2>&1 | tee FEM/experiments/active_domain_validation/physics_integrity/coupled_low_frequency/logs/run.log
