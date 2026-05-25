#!/usr/bin/env bash
# Geometric tag-2 aperture audit (area, radius, planarity) — no solve.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
cd "$ROOT"
source .venv/bin/activate 2>/dev/null || true
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
mkdir -p FEM/experiments/active_domain_validation/physics_integrity/diagnostics/soundhole_aperture_audit
python FEM/experiments/active_domain_validation/physics_integrity/scripts/audit_soundhole_aperture_geometry.py \
  2>&1 | tee FEM/experiments/active_domain_validation/physics_integrity/diagnostics/soundhole_aperture_audit/run.log
