#!/usr/bin/env bash
# Physical FSI + Nitsche nit_pu only (reduced domain). Run manually after participation audit PASS.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
cd "$ROOT"
source .venv/bin/activate 2>/dev/null || true
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
python FEM/experiments/active_domain_validation/physics_integrity/scripts/prepare_physics_configs.py
mkdir -p FEM/experiments/active_domain_validation/physics_integrity/coupled_physical_fsi_nit_pu/logs
mpiexec -n 1 python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_physics_case.py \
  --case coupled_physical_fsi_nit_pu \
  --config FEM/experiments/active_domain_validation/physics_integrity/configs/coupled_physical_fsi_nit_pu.json \
  2>&1 | tee FEM/experiments/active_domain_validation/physics_integrity/coupled_physical_fsi_nit_pu/logs/run.log
