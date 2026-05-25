#!/usr/bin/env bash
# Coupled pressure-domain audit: full vs air-supported p DOFs (no eigen solve)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
cd "$ROOT"
source .venv/bin/activate 2>/dev/null || true
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
python FEM/experiments/active_domain_validation/physics_integrity/scripts/prepare_physics_configs.py
mpiexec -n 1 python FEM/experiments/active_domain_validation/physics_integrity/scripts/coupled_pressure_domain_audit.py \
  --config FEM/experiments/active_domain_validation/physics_integrity/configs/coupled_near_acoustic_244hz.json \
  2>&1 | tee FEM/experiments/active_domain_validation/physics_integrity/diagnostics/coupled_pressure_domain/audit.log
