#!/usr/bin/env bash
# Validation CAD volume audit only (cavity probes + reports; no mesh, no eigen solve).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
cd "$ROOT"
source .venv/bin/activate 2>/dev/null || true
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export FEM_VALIDATION_MESH=1
export FEM_VALIDATION_CAD_AUDIT_ONLY=1
export FEM_SKIP_POST_MESH_AUDIT=1
export FEM_MESH_OUT="$ROOT/FEM/experiments/active_domain_validation/mesh/validation_tiny_guitar_3d.msh"
CONFIG="$ROOT/FEM/experiments/active_domain_validation/configs/sample_000_validation_base.json"
if [[ -f "$CONFIG" ]]; then
  export FEM_MESH_CONFIG="$CONFIG"
fi
python FEM/experiments/active_domain_validation/scripts/prepare_validation_mesh.py
