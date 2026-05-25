#!/usr/bin/env bash
# Diagnostic coupled solve: all u + air-supported p only (algebraic restriction; ~244 Hz band)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
cd "$ROOT"
source .venv/bin/activate 2>/dev/null || true
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
python FEM/experiments/active_domain_validation/physics_integrity/scripts/prepare_physics_configs.py
mpiexec -n 1 python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_physics_case.py \
  --case coupled_near_acoustic_air_p \
  --config FEM/experiments/active_domain_validation/physics_integrity/configs/coupled_near_acoustic_air_p_restricted.json \
  2>&1 | tee FEM/experiments/active_domain_validation/physics_integrity/coupled_near_acoustic_air_p/logs/run.log
