#!/usr/bin/env bash
# Report-only: full pipeline audit over existing persistence-fixed replacement baseline.
# Does not call eps.solve() or regenerate candidates.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/FEM/scripts:${REPO_ROOT}/FEM/experiments/active_domain_validation/physics_integrity/scripts:${PYTHONPATH:-}"
mpiexec -n 1 python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_v2_mapping_fixed_persistence_fixed_full_pipeline_audit.py
