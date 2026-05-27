#!/usr/bin/env bash
# Report-only: empirical M_uu null-basis certification + projection preflight (no EPS).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/FEM/scripts:${REPO_ROOT}/FEM/experiments/active_domain_validation/physics_integrity/scripts:${PYTHONPATH:-}"

mpiexec -n 1 python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_v2_lossless_adjudication_v1_Muu_null_basis_certification_and_projection_preflight.py
