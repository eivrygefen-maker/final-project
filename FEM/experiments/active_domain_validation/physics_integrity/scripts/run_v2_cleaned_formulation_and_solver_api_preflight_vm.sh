#!/usr/bin/env bash
# Report-only VM preflight: cleaned formulation contract + solver API availability (no EPS solve).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/FEM/scripts:${REPO_ROOT}/FEM/experiments/active_domain_validation/physics_integrity/scripts:${PYTHONPATH:-}"

python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_v2_cleaned_formulation_and_solver_api_preflight.py
