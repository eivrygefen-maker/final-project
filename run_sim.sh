#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <shape_name>"
  echo
  echo "Available shapes in ROM/:"
  if [[ -d "ROM" ]]; then
    ls -1 ROM | while read -r d; do
      [[ -d "ROM/$d" ]] && echo "  - $d"
    done
  else
    echo "  (ROM directory not found yet)"
  fi
  exit 1
fi

SHAPE_NAME="$1"

if [[ ! -d ".venv" ]]; then
  echo "[ERROR] .venv not found in current directory: $(pwd)"
  exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate
python3 -m FEM.scripts.rom_pipeline offline --shape "$SHAPE_NAME"
