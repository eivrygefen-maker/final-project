#!/usr/bin/env bash
# TEST 1 — reuse existing ../baseline outputs (no solve)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
cd "$ROOT"
source .venv/bin/activate 2>/dev/null || true
python FEM/experiments/active_domain_validation/physics_integrity/scripts/prepare_physics_configs.py
python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_physics_case.py \
  --case coupled_nominal \
  --config FEM/experiments/active_domain_validation/physics_integrity/configs/coupled_nominal_202hz.json \
  --ingest-baseline
python FEM/experiments/active_domain_validation/physics_integrity/scripts/analyze_modes.py \
  --case-dir FEM/experiments/active_domain_validation/physics_integrity/coupled_nominal \
  --config FEM/experiments/active_domain_validation/physics_integrity/configs/coupled_nominal_202hz.json \
  --target-hz 202
