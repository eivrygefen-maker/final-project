#!/usr/bin/env bash
# Soundhole facet (tag 2) ↔ air volume (tag 10) adjacency audit — no solve.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
cd "$ROOT"
source .venv/bin/activate 2>/dev/null || true
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
mkdir -p FEM/experiments/active_domain_validation/physics_integrity/diagnostics/soundhole_air_audit
mpiexec -n 1 python FEM/experiments/active_domain_validation/physics_integrity/scripts/audit_soundhole_air_adjacency.py \
  2>&1 | tee FEM/experiments/active_domain_validation/physics_integrity/diagnostics/soundhole_air_audit/run.log
