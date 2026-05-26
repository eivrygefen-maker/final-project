#!/usr/bin/env bash
# Report-only evaluation of completed unregularized-offset baseline diagnostic (no eigensolve).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/FEM/scripts:${REPO_ROOT}/FEM/experiments/active_domain_validation/physics_integrity/scripts:${PYTHONPATH:-}"
python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_v2_l_mid_seed_branch_unregularized_offset_diagnostic.py --evaluate-only
python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_v2_solver_root_cause_and_forward_risk_audit.py
