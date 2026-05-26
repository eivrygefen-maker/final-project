#!/usr/bin/env bash
# 1) No-EVP persistence self-test (required gate).
# 2) Replacement mapping-corrected baseline solve + report-only evaluation (only if self-test passes).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/FEM/scripts:${REPO_ROOT}/FEM/experiments/active_domain_validation/physics_integrity/scripts:${PYTHONPATH:-}"

echo "[vm] Step 1/2: mapping-fixed candidate persistence self-test (no EPS)" >&2
mpiexec -n 1 python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_v2_mapping_fixed_candidate_persistence_self_test.py

echo "[vm] Step 2/2: persistence-fixed replacement baseline + evaluation" >&2
python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_v2_l_mid_mapping_fixed_unregularized_persistence_fixed_baseline_diagnostic.py
