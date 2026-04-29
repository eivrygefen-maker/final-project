#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status

echo "========================================"
echo "🚀 Starting Vacuum Test Pipeline..."
echo "========================================"

echo "[1/4] Enforcing coupled-run config (no structural-only diagnosis)..."
python3 - <<'PY'
import json
from pathlib import Path

cfg_path = Path("FEM/configs/guitar_3d.json")
cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
solver = cfg.setdefault("solver", {})
materials = cfg.setdefault("materials", {})
air = materials.setdefault("air", {})

solver["structural_only_diagnosis"] = False
solver["pressure_gauge"] = "soundhole"
solver["st_ksp_type"] = "preonly"
solver["st_pc_type"] = "lu"
solver["st_pc_factor_mat_solver_type"] = "mumps"
solver["st_factor_solver_type"] = "mumps"
solver["mat_mumps_icntl_14"] = 5000
solver["mat_mumps_icntl_23"] = 0
solver["mat_mumps_icntl_22"] = 1
solver["shift_invert_target_hz"] = 105.0
solver["structural_shift_target_hz"] = 105.0
solver["st_shift_target_hz"] = 105.0
solver["sifter_start_hz"] = 105.0
air["density"] = 1.21
air["speed_of_sound"] = 343.0

cfg_path.write_text(json.dumps(cfg, indent=4), encoding="utf-8")
print("[config] structural_only_diagnosis=False, pressure_gauge=soundhole, air=(rho=1.21, c=343.0)")
PY

echo "[2/4] Cleaning stale mesh files..."
rm -f FEM/mesh/guitar_3d.msh

echo "[3/4] Building 3D Geometry and Mesh (1.5mm/7mm/15mm)..."
python3 FEM/geometry/build_3d_guitar.py

echo "[4/4] Running Single-Core Solver..."
PYTHONPATH=. python3 FEM/scripts/rom_pipeline.py offline --shape classic --max-runs 1 --num-modes 10 --force-pool-rebuild 2>&1 | tee simulation.log

echo "========================================"
echo "✅ Pipeline completed successfully!"
echo "========================================"
