#!/usr/bin/env bash
# Rebuild validation_tiny_guitar_3d.msh (air-connected soundhole tag 2) and run adjacency audit.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
cd "$ROOT"
source .venv/bin/activate 2>/dev/null || true
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export FEM_SKIP_POST_MESH_AUDIT=1
python FEM/experiments/active_domain_validation/scripts/prepare_validation_mesh.py
bash FEM/experiments/active_domain_validation/physics_integrity/scripts/run_soundhole_air_audit.sh
python FEM/experiments/active_domain_validation/scripts/print_validation_soundhole_gate.py
