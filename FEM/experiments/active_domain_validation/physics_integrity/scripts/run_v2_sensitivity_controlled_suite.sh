#!/usr/bin/env bash
# Controlled non-random v2 sensitivity: depth, top thickness, E_L (preserves radius pilot).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
cd "$ROOT"
source .venv/bin/activate 2>/dev/null || true
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
mkdir -p FEM/experiments/active_domain_validation/physics_integrity/v2_sensitivity_validation/logs
python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_v2_sensitivity_controlled_suite.py \
  2>&1 | tee FEM/experiments/active_domain_validation/physics_integrity/v2_sensitivity_validation/logs/controlled_suite.log
