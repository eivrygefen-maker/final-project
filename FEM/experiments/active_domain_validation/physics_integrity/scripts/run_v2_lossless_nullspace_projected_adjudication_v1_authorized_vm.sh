#!/usr/bin/env bash
# One authorized nullspace-projected lossless ST adjudication (final ST attempt).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/FEM/scripts:${REPO_ROOT}/FEM/experiments/active_domain_validation/physics_integrity/scripts:${PYTHONPATH:-}"

python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_v2_lossless_nullspace_projected_adjudication_v1_gated_runner.py \
  --authorize-single-projected-eps-run
