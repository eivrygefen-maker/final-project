#!/usr/bin/env bash
# TEST 3 — acoustic-cavity-only reference (mpiexec -n 1)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
cd "$ROOT"
source .venv/bin/activate 2>/dev/null || true
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
python FEM/experiments/active_domain_validation/physics_integrity/scripts/prepare_physics_configs.py
mpiexec -n 1 python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_physics_case.py \
  --case acoustic_only \
  --config FEM/experiments/active_domain_validation/physics_integrity/configs/acoustic_only.json \
  2>&1 | tee FEM/experiments/active_domain_validation/physics_integrity/acoustic_only/logs/run.log
