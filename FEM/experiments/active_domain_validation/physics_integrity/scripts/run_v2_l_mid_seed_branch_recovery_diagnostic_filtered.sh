#!/usr/bin/env bash
# Baseline filtered seed-branch recovery diagnostic (EPS rerun) — run only after reviewing
# v2_l_mid_seed_branch_candidate_filter_audit.{json,md} when no saved candidate passes.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/FEM/scripts:${REPO_ROOT}/FEM/experiments/active_domain_validation/physics_integrity/scripts:${PYTHONPATH:-}"
python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_v2_l_mid_seed_branch_recovery_diagnostic.py --filtered-harvest
