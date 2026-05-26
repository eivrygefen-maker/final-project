#!/usr/bin/env bash
# Final report-only: saved-vector persistence and mass-norm audit (no eigensolve).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/FEM/scripts:${REPO_ROOT}/FEM/experiments/active_domain_validation/physics_integrity/scripts:${PYTHONPATH:-}"
python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_v2_l_mid_unregularized_saved_vector_mass_norm_audit.py
python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_v2_solver_root_cause_and_forward_risk_audit.py
