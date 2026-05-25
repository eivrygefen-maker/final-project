#!/usr/bin/env bash
# Targeted hole_radius_large acoustic capture (255–300 Hz) + pilot summary merge.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
cd "$ROOT"
source .venv/bin/activate 2>/dev/null || true
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
mkdir -p FEM/experiments/active_domain_validation/physics_integrity/v2_sensitivity_validation/logs
python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_v2_sensitivity_pilot_large_capture.py \
  2>&1 | tee FEM/experiments/active_domain_validation/physics_integrity/v2_sensitivity_validation/logs/pilot_large_capture.log
