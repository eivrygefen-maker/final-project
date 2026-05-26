#!/usr/bin/env bash
# No-eigensolve L_mid coupled mixed-mode continuation audit (mpiexec -n 1).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/FEM/scripts:${REPO_ROOT}/FEM/experiments/active_domain_validation/physics_integrity/scripts:${PYTHONPATH:-}"
mpiexec -n 1 python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_v2_l_mid_coupled_mixed_mode_audit.py "$@"
