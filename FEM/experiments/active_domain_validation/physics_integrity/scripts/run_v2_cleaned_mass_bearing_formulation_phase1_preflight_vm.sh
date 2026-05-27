#!/usr/bin/env bash
# Phase-1 no-EPS: cleaned formulation preflight + SLEPc API probe (no eps.solve).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/FEM/scripts:${REPO_ROOT}/FEM/experiments/active_domain_validation/physics_integrity/scripts:${PYTHONPATH:-}"

mpiexec -n 1 python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_v2_cleaned_mass_bearing_formulation_phase1_preflight.py
