#!/usr/bin/env bash
# Backward-compatible wrapper — delegates to unified shape overnight runner.
set -euo pipefail
export SHAPE="${SHAPE:-box}"
export COUNT="${COUNT:-100}"
export RUN_ID_SUFFIX="${RUN_ID_SUFFIX:-box_fom_v1}"
exec bash "$(dirname "$0")/run_shape_fom_overnight_batch.sh" "$@"
