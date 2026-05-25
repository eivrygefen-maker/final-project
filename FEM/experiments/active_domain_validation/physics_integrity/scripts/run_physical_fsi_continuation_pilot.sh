#!/usr/bin/env bash
# Pilot: one new solve at alpha_fsi=0.01 + post-process (reuse alpha=0 decoupled, alpha=1 physical-FSI-only).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
cd "$ROOT"
source .venv/bin/activate 2>/dev/null || true
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
python FEM/experiments/active_domain_validation/physics_integrity/scripts/prepare_physics_configs.py
mkdir -p FEM/experiments/active_domain_validation/physics_integrity/coupled_physical_fsi_alpha_pilot/logs
mpiexec -n 1 python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_physics_case.py \
  --case coupled_physical_fsi_alpha_pilot \
  --config FEM/experiments/active_domain_validation/physics_integrity/configs/coupled_physical_fsi_alpha_pilot.json \
  2>&1 | tee FEM/experiments/active_domain_validation/physics_integrity/coupled_physical_fsi_alpha_pilot/logs/run.log
python FEM/experiments/active_domain_validation/physics_integrity/scripts/physical_fsi_continuation_post.py \
  --pilot \
  2>&1 | tee FEM/experiments/active_domain_validation/physics_integrity/coupled_physical_fsi_alpha_pilot/logs/continuation_post.log
