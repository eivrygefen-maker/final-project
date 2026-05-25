#!/usr/bin/env bash
# No-eigensolve audit: physical-FSI-only ~245.30 Hz vs decoupled-union ~244.39 Hz (participation + MAC).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
cd "$ROOT"
source .venv/bin/activate 2>/dev/null || true
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
mkdir -p FEM/experiments/active_domain_validation/physics_integrity/coupled_physical_fsi_only/logs
mpiexec -n 1 python FEM/experiments/active_domain_validation/physics_integrity/scripts/physical_fsi_participation_audit.py \
  2>&1 | tee FEM/experiments/active_domain_validation/physics_integrity/coupled_physical_fsi_only/logs/physical_fsi_participation_audit.log
