#!/usr/bin/env bash
# Minimal v2 sensitivity pilot: soundhole radius small/large (+ ingest frozen baseline).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
cd "$ROOT"
source .venv/bin/activate 2>/dev/null || true
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
mkdir -p FEM/experiments/active_domain_validation/physics_integrity/v2_sensitivity_validation/logs
mpiexec -n 1 python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_v2_sensitivity_validation.py \
  --pilot \
  2>&1 | tee FEM/experiments/active_domain_validation/physics_integrity/v2_sensitivity_validation/logs/pilot.log
