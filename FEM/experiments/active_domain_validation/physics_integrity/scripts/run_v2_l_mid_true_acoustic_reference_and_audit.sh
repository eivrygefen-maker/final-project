#!/usr/bin/env bash
# Build coupled-W seeds from archived true acoustic locator vectors + no-eigensolve audit.
# Does not rerun acoustic locators or coupled EPS.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/FEM/scripts:${REPO_ROOT}/FEM/experiments/active_domain_validation/physics_integrity/scripts:${PYTHONPATH:-}"
python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_v2_l_mid_true_acoustic_reference_recovery.py "$@"
