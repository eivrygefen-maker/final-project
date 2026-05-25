#!/usr/bin/env python3
"""Build the shared validation mesh and experiment JSON configs (isolated under FEM/experiments/)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
MESH_DIR = EXPERIMENT_ROOT / "mesh"
CONFIG_DIR = EXPERIMENT_ROOT / "configs"
MESH_PATH = MESH_DIR / "validation_tiny_guitar_3d.msh"
BASE_CONFIG = CONFIG_DIR / "sample_000_validation_base.json"
SOURCE_CONFIG = REPO_ROOT / "FEM" / "configs" / "guitar_3d.json"


def _load_base_solver() -> dict:
    with open(SOURCE_CONFIG, encoding="utf-8") as f:
        cfg = json.load(f)
    geom = dict(cfg.get("geometry", {}))
    geom.update(
        {
            "shape_type": "Classical",
            "length": 0.48,
            "width": 0.325,
            "depth": 0.10,
            "top_thickness": 0.003,
            "back_thickness": 0.0033,
            "hole_radius": 0.047,
            "mesh_mode": "fom",
        }
    )
    solver = dict(cfg.get("solver", {}))
    solver.update(
        {
            "mesh_file": str(MESH_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
            "soundhole_bc": "pressure_release",
            "pressure_gauge": "none",
            "couple_fluid": True,
            "clamp_ribs": False,
            "adaptive_mode_sifter": False,
            "num_modes": 8,
            "eps_worker_num_modes_cap": 8,
            "eps_ncv_max": 32,
            "eps_rigid_mode_buffer": 4,
            "st_use_fieldsplit": False,
            "st_fieldsplit": False,
            "st_pc_type": "lu",
            "gnhep_block_frobenius_normalize": True,
            "eps_pin_fix_tag5": True,
            "eps_algebraic_bc_zero_columns": True,
            "eps_band_solver": "shift_invert",
            "eps_broad_search_hz": 46.0,
            "eps_reject_sigma_spurious": False,
            "eps_reject_target_locked": False,
            "eps_reject_decoupled_u_only": False,
            "eps_harvest_allow_weak_coupling": True,
            "eigs_maxiter": 3000,
            "eps_max_it": 3000,
        }
    )
    solver.pop("active_domain_experiment", None)
    return {"geometry": geom, "materials": cfg.get("materials", {}), "solver": solver}


def _write_configs(base: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    BASE_CONFIG.write_text(json.dumps(base, indent=2), encoding="utf-8")

    baseline = json.loads(json.dumps(base))
    baseline["solver"]["active_domain_experiment"] = {
        "enabled": False,
        "label": "baseline_full_volume",
    }
    (CONFIG_DIR / "sample_000_baseline.json").write_text(
        json.dumps(baseline, indent=2), encoding="utf-8"
    )

    active = json.loads(json.dumps(base))
    active["solver"]["active_domain_experiment"] = {
        "enabled": True,
        "method": "algebraic_restriction",
        "bypass_worker_mode_cap": True,
        "label": "active_domain_algebraic",
    }
    (CONFIG_DIR / "sample_000_active_domain.json").write_text(
        json.dumps(active, indent=2), encoding="utf-8"
    )


def _build_mesh() -> None:
    MESH_DIR.mkdir(parents=True, exist_ok=True)
    log_path = MESH_DIR / "mesh_build.log"
    env = os.environ.copy()
    env["FEM_VALIDATION_MESH"] = "1"
    env["FEM_MESH_OUT"] = str(MESH_PATH.resolve())
    env["FEM_MESH_CONFIG"] = str(BASE_CONFIG.resolve())
    cmd = [sys.executable, str(REPO_ROOT / "FEM" / "geometry" / "build_3d_guitar.py")]
    print("[prepare] Building validation mesh:", MESH_PATH)
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"Mesh build failed (exit {proc.returncode}); see {log_path}")
    if not MESH_PATH.is_file():
        raise FileNotFoundError(f"Expected mesh not written: {MESH_PATH}")


def main() -> int:
    base = _load_base_solver()
    _write_configs(base)
    _build_mesh()
    audit_script = EXPERIMENT_ROOT / "scripts" / "inspect_mesh_and_tags.py"
    proc = subprocess.run(
        [sys.executable, str(audit_script), "--mesh", str(MESH_PATH)],
        cwd=str(REPO_ROOT),
        check=False,
    )
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
