#!/usr/bin/env bash
# FSI operator assembly audit (no eigen solve)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
cd "$ROOT"
source .venv/bin/activate 2>/dev/null || true
python FEM/experiments/active_domain_validation/physics_integrity/scripts/prepare_physics_configs.py
mpiexec -n 1 python FEM/experiments/active_domain_validation/physics_integrity/scripts/operator_audit.py \
  --config FEM/experiments/active_domain_validation/physics_integrity/configs/operator_audit.json \
  2>&1 | tee FEM/experiments/active_domain_validation/physics_integrity/coupled_nominal/logs/operator_audit.log
