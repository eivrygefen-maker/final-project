#!/usr/bin/env bash
# Report-only B2 vs B3 cleaned formulation architecture decision (no eigensolve).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/FEM/scripts:${REPO_ROOT}/FEM/experiments/active_domain_validation/physics_integrity/scripts:${PYTHONPATH:-}"

mpiexec -n 1 python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_v2_cleaned_formulation_B2_vs_B3_architecture_decision.py
