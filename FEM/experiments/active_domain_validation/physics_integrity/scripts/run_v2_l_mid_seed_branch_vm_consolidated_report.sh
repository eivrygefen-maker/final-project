#!/usr/bin/env bash
# Consolidated VM report-only bundle: filtered candidate evaluation + root-cause/closure audit.
# Requires VM-local artifacts under seed_branch_recovery_diagnostic_filtered/ (not in git).
# No eigensolve.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/FEM/scripts:${REPO_ROOT}/FEM/experiments/active_domain_validation/physics_integrity/scripts:${PYTHONPATH:-}"
python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_v2_l_mid_seed_branch_filtered_evaluation.py
python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_v2_solver_root_cause_and_forward_risk_audit.py
