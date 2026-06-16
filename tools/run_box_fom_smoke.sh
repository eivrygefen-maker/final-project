#!/usr/bin/env bash
# Backward-compatible wrapper — delegates to unified shape smoke.
set -euo pipefail
export SHAPE="${SHAPE:-box}"
exec bash "$(dirname "$0")/run_shape_fom_smoke.sh" "$@"
