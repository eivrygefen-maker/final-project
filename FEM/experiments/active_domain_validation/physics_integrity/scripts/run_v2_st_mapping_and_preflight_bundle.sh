#!/usr/bin/env bash
# Report-only: mapping impact inventory, PASS replay recertification, ST preflight (no eigensolve).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/FEM/scripts:${REPO_ROOT}/FEM/experiments/active_domain_validation/physics_integrity/scripts:${PYTHONPATH:-}"
python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_v2_eps_mapping_impact_inventory.py
python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_v2_existing_pass_replay_recertification.py
python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_v2_st_singular_mass_preflight.py
python FEM/experiments/active_domain_validation/physics_integrity/scripts/write_v2_st_singular_mass_rehabilitation_plan.py
python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_v2_solver_root_cause_and_forward_risk_audit.py
