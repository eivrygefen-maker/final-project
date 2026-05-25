#!/usr/bin/env bash
# Validation mesh construction only (CAD audit on failure; no eigen solve, no post-build audit).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
cd "$ROOT"
source .venv/bin/activate 2>/dev/null || true
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export FEM_SKIP_POST_MESH_AUDIT=1
python FEM/experiments/active_domain_validation/scripts/prepare_validation_mesh.py
