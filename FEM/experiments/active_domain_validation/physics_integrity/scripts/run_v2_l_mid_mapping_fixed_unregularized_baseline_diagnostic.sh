#!/usr/bin/env bash
# Exactly one authorized mapping-corrected unregularized baseline ST diagnostic:
# mpiexec solve (preserve all nconv candidates) + immediate report-only evaluation.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/FEM/scripts:${REPO_ROOT}/FEM/experiments/active_domain_validation/physics_integrity/scripts:${PYTHONPATH:-}"
python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_v2_l_mid_mapping_fixed_unregularized_baseline_diagnostic.py
