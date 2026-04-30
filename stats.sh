#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <snapshot.npz>"
  echo "Example: $0 ROM_DATA/classic/snapshots/snapshot_0001.npz"
  exit 1
fi

SNAPSHOT_PATH="$1"

if [[ ! -d ".venv" ]]; then
  echo "[ERROR] .venv not found in current directory: $(pwd)"
  exit 1
fi

if [[ ! -f "$SNAPSHOT_PATH" ]]; then
  echo "[ERROR] Snapshot file not found: $SNAPSHOT_PATH"
  exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate
python analyze_snapshot.py "$SNAPSHOT_PATH"
