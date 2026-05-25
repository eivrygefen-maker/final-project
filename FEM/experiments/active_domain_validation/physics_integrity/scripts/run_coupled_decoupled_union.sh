#!/usr/bin/env bash
# Block-diagonal decoupled-union diagnostic on reduced mixed domain (u + air p only)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
cd "$ROOT"
source .venv/bin/activate 2>/dev/null || true
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
python FEM/experiments/active_domain_validation/physics_integrity/scripts/prepare_physics_configs.py
mkdir -p FEM/experiments/active_domain_validation/physics_integrity/coupled_decoupled_union/logs
mpiexec -n 1 python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_physics_case.py \
  --case coupled_decoupled_union \
  --config FEM/experiments/active_domain_validation/physics_integrity/configs/coupled_decoupled_union.json \
  2>&1 | tee FEM/experiments/active_domain_validation/physics_integrity/coupled_decoupled_union/logs/run.log
