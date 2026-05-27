#!/usr/bin/env bash
# Report-only bundle: pipeline audit, architecture audit, provenance inventory, lossless preflight.
# Does not call eps.solve() or regenerate candidates.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/FEM/scripts:${REPO_ROOT}/FEM/experiments/active_domain_validation/physics_integrity/scripts:${PYTHONPATH:-}"
python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_v2_coupled_physical_core_report_only_bundle.py
