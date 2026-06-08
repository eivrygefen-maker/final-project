#!/usr/bin/env bash
# Guarded wrapper for m4_geometry_corrected_v1 dataset reset (dry-run by default).
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../../../../.." && pwd)}"
SCRIPT="$REPO_ROOT/FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_m4_geometry_corrected_dataset_reset.py"

if [[ ! -f "$SCRIPT" ]]; then
  echo "error: reset tool not found: $SCRIPT" >&2
  exit 2
fi

if [[ "${1:-}" == "--execute" ]]; then
  shift
  exec python "$SCRIPT" --execute "$@"
fi

if [[ "${1:-}" == "--verify" ]]; then
  shift
  exec python "$SCRIPT" --verify "$@"
fi

exec python "$SCRIPT" "$@"
