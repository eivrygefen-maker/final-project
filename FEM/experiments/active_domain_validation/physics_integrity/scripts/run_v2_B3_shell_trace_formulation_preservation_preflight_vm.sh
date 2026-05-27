#!/usr/bin/env bash
# Report-only B3 shell/trace formulation preservation preflight (no eigensolve).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/FEM/scripts:${REPO_ROOT}/FEM/experiments/active_domain_validation/physics_integrity/scripts:${PYTHONPATH:-}"

mpiexec -n 1 python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_v2_B3_shell_trace_formulation_preservation_preflight.py
