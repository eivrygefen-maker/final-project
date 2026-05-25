#!/usr/bin/env bash
# coupled_physical_core_v2: coupling_disabled + physical_coupling_enabled (one validation runner).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
cd "$ROOT"
source .venv/bin/activate 2>/dev/null || true
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
python FEM/experiments/active_domain_validation/physics_integrity/scripts/prepare_physics_configs.py
mkdir -p FEM/experiments/active_domain_validation/physics_integrity/coupled_physical_core_v2/logs
mpiexec -n 1 python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_physical_core_v2_validation.py \
  --config FEM/experiments/active_domain_validation/physics_integrity/configs/coupled_physical_core_v2.json \
  2>&1 | tee FEM/experiments/active_domain_validation/physics_integrity/coupled_physical_core_v2/logs/validation.log
