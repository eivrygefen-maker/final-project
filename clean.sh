#!/usr/bin/env bash
set -u

echo "[clean] Starting project cleanup..."

# 1) Mesh artifacts
echo "[clean] Removing mesh files in FEM/mesh (*.msh, *.xdmf)..."
rm -f FEM/mesh/*.msh FEM/mesh/*.xdmf

echo "[clean] Removing mesh cache directory FEM/mesh/_xdmf_cache/ ..."
if [ -d "FEM/mesh/_xdmf_cache" ]; then
  rm -rf FEM/mesh/_xdmf_cache
else
  echo "[clean] FEM/mesh/_xdmf_cache not found (already clean)."
fi

# 2) ROM classic artifacts
echo "[clean] Removing ROM pool file ROM_DATA/classic/lhs_pool_classic.json ..."
rm -f ROM_DATA/classic/lhs_pool_classic.json

echo "[clean] Removing all snapshot files in ROM_DATA/classic/snapshots/ ..."
if [ -d "ROM_DATA/classic/snapshots" ]; then
  rm -f ROM_DATA/classic/snapshots/*
else
  echo "[clean] ROM_DATA/classic/snapshots not found (already clean)."
fi

echo "[clean] Cleanup complete."
