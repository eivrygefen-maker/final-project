#!/usr/bin/env bash
# Baseline-only post-processing sigma/BC spurious-mode audit (no eigensolve).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/FEM/scripts:${REPO_ROOT}/FEM/experiments/active_domain_validation/physics_integrity/scripts:${PYTHONPATH:-}"
python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_v2_l_mid_seed_branch_spurious_mode_audit.py
