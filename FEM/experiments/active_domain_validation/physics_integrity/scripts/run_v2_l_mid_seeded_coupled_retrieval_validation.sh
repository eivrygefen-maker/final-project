#!/usr/bin/env bash
# L_mid seeded coupled-v2 EPS retrieval validation only (no locator, no L_prod/L_check).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/FEM/scripts:${REPO_ROOT}/FEM/experiments/active_domain_validation/physics_integrity/scripts:${PYTHONPATH:-}"
python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_v2_l_mid_seeded_coupled_retrieval_validation.py --baseline-only "$@"
