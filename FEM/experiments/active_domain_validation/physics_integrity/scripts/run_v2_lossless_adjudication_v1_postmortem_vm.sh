#!/usr/bin/env bash
# Report-only: mass-null postmortem + missing full audit for lossless adjudication v1 (no EPS).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/FEM/scripts:${REPO_ROOT}/FEM/experiments/active_domain_validation/physics_integrity/scripts:${PYTHONPATH:-}"

mpiexec -n 1 python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_v2_lossless_adjudication_v1_mass_null_postmortem.py
