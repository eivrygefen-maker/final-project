#!/usr/bin/env bash
# TEST 5 — coupled FSI solve near validated acoustic cavity mode (~244.39 Hz)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
cd "$ROOT"
source .venv/bin/activate 2>/dev/null || true
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
python FEM/experiments/active_domain_validation/physics_integrity/scripts/prepare_physics_configs.py
mpiexec -n 1 python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_physics_case.py \
  --case coupled_near_acoustic \
  --config FEM/experiments/active_domain_validation/physics_integrity/configs/coupled_near_acoustic_244hz.json \
  2>&1 | tee FEM/experiments/active_domain_validation/physics_integrity/coupled_near_acoustic/logs/run.log
