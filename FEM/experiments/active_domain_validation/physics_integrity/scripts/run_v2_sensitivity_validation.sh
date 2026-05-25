#!/usr/bin/env bash
# Full v2 sensitivity suite (all manifest samples). Run pilot first.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
cd "$ROOT"
source .venv/bin/activate 2>/dev/null || true
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
mkdir -p FEM/experiments/active_domain_validation/physics_integrity/v2_sensitivity_validation/logs
mpiexec -n 1 python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_v2_sensitivity_validation.py \
  2>&1 | tee FEM/experiments/active_domain_validation/physics_integrity/v2_sensitivity_validation/logs/full.log
