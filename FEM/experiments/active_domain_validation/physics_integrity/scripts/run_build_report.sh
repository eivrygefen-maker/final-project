#!/usr/bin/env bash
# Aggregate comparison report (no MPI)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
cd "$ROOT"
source .venv/bin/activate 2>/dev/null || true
python FEM/experiments/active_domain_validation/physics_integrity/scripts/build_physics_integrity_report.py
