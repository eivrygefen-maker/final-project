#!/usr/bin/env bash
# Regenerate v2 sensitivity summary + structural trend report (no solves).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
cd "$ROOT"
source .venv/bin/activate 2>/dev/null || true
python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_v2_sensitivity_report_post.py \
  2>&1 | tee FEM/experiments/active_domain_validation/physics_integrity/v2_sensitivity_validation/logs/report_post.log
