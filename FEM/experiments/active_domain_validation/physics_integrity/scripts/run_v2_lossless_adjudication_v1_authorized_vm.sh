#!/usr/bin/env bash
# One authorized isolated lossless adjudication EPS (L_mid baseline_coupled_v2 only).
# 1) Regenerates report-only preflight gate contract
# 2) Runs exactly one EPS in seed_branch_recovery_diagnostic_mapping_fixed_unregularized_lossless_adjudication_v1/
# 3) Evaluates authoritative lossless vectors and writes diagnostic + audit reports
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/FEM/scripts:${REPO_ROOT}/FEM/experiments/active_domain_validation/physics_integrity/scripts:${PYTHONPATH:-}"

python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_v2_lossless_adjudication_v1_gated_runner.py \
  --authorize-single-eps-run
