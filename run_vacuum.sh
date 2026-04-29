#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status

echo "========================================"
echo "🚀 Starting Vacuum Test Pipeline..."
echo "========================================"

echo "[1/4] Forcing clean state and pulling latest changes..."
git checkout FEM/configs/guitar_3d.json # Discard local changes to config
git pull origin main

echo "[2/4] Cleaning stale mesh files..."
rm -f FEM/mesh/guitar_3d.msh

echo "[3/4] Building 3D Geometry and Mesh (1.5mm/7mm/15mm)..."
python3 FEM/geometry/build_3d_guitar.py

echo "[4/4] Running Single-Core Solver..."
PYTHONPATH=. python3 FEM/scripts/rom_pipeline.py offline --shape classic --max-runs 1 --num-modes 10 --force-pool-rebuild 2>&1 | tee simulation.log

echo "========================================"
echo "✅ Pipeline completed successfully!"
echo "========================================"
