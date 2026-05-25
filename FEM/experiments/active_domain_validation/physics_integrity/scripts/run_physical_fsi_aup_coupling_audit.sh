#!/usr/bin/env bash
# No-eigensolve A_up coupling-strength / structural-response audit (alpha sweep).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
cd "$ROOT"
source .venv/bin/activate 2>/dev/null || true
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
mkdir -p FEM/experiments/active_domain_validation/physics_integrity/coupled_physical_fsi_alpha_pilot/logs
mpiexec -n 1 python FEM/experiments/active_domain_validation/physics_integrity/scripts/physical_fsi_aup_coupling_audit.py \
  --config FEM/experiments/active_domain_validation/physics_integrity/configs/coupled_physical_fsi_alpha_pilot.json \
  2>&1 | tee FEM/experiments/active_domain_validation/physics_integrity/coupled_physical_fsi_alpha_pilot/logs/aup_coupling_audit.log
