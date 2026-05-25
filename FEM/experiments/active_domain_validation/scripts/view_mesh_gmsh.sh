#!/usr/bin/env bash
set -euo pipefail
EXP_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MSH="$EXP_ROOT/mesh/validation_tiny_guitar_3d.msh"
if [[ ! -f "$MSH" ]]; then
  echo "Mesh not found: $MSH — run prepare_validation_mesh.py first." >&2
  exit 1
fi
gmsh "$MSH" &
