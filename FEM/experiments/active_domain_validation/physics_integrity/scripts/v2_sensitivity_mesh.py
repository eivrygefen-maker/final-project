#!/usr/bin/env python3
"""Build validation-guitar meshes for v2 sensitivity samples (experiment-only)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[5]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
PHYSICS_ROOT = Path(__file__).resolve().parents[1]
SENS_ROOT = PHYSICS_ROOT / "v2_sensitivity_validation"
MESH_DIR = SENS_ROOT / "mesh"
CONFIG_DIR = SENS_ROOT / "configs"
SOURCE_CONFIG = REPO_ROOT / "FEM" / "configs" / "guitar_3d.json"

NOMINAL_GEOMETRY_BY_SHAPE: Dict[str, Dict[str, Any]] = {
    "Classical": {
        "shape_type": "Classical",
        "length": 0.48,
        "width": 0.325,
        "depth": 0.10,
        "top_thickness": 0.003,
        "back_thickness": 0.0033,
        "hole_radius": 0.047,
        "mesh_mode": "fom",
    },
    "Box": {
        "shape_type": "Box",
        "length": 0.46,
        "width": 0.36,
        "depth": 0.10,
        "top_thickness": 0.003,
        "back_thickness": 0.0033,
        "hole_radius": 0.042,
        "mesh_mode": "fom",
    },
    "Acoustic": {
        "shape_type": "Acoustic",
        "length": 0.50,
        "width": 0.40,
        "depth": 0.12,
        "top_thickness": 0.003,
        "back_thickness": 0.0033,
        "hole_radius": 0.045,
        "mesh_mode": "fom",
    },
}

NOMINAL_GEOMETRY = dict(NOMINAL_GEOMETRY_BY_SHAPE["Classical"])


def sample_geometry(sample: Dict[str, Any], *, shape_type: Optional[str] = None) -> Dict[str, Any]:
    geom_in = dict(sample.get("geometry") or {})
    st = (
        shape_type
        or geom_in.get("shape_type")
        or sample.get("shape_type")
        or sample.get("geometry_shape_type")
        or "Classical"
    )
    base = dict(NOMINAL_GEOMETRY_BY_SHAPE.get(str(st), NOMINAL_GEOMETRY))
    base["shape_type"] = str(st)
    geom = dict(base)
    geom.update(geom_in)
    if "back_thickness" not in geom_in and "back_thickness" not in geom:
        geom["back_thickness"] = float(geom["top_thickness"]) * 1.1
    return geom


def sample_mesh_path(sample_id: str) -> Path:
    return MESH_DIR / f"{sample_id}.msh"


def sample_config_path(sample_id: str) -> Path:
    return CONFIG_DIR / f"{sample_id}.json"


def build_sample_mesh(sample: Dict[str, Any]) -> Path:
    """Run FEM_VALIDATION_MESH build for one sensitivity sample."""
    sample_id = str(sample["id"])
    geom = sample_geometry(sample)
    mesh_path = sample_mesh_path(sample_id)
    cfg_path = sample_config_path(sample_id)

    MESH_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
    cfg["geometry"] = geom
    cfg.setdefault("solver", {})
    cfg["solver"]["mesh_file"] = str(mesh_path.resolve())
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    log_path = MESH_DIR / f"{sample_id}_build.log"
    env = os.environ.copy()
    env["FEM_VALIDATION_MESH"] = "1"
    env["FEM_MESH_OUT"] = str(mesh_path.resolve())
    env["FEM_MESH_CONFIG"] = str(cfg_path.resolve())
    cmd = [sys.executable, str(REPO_ROOT / "FEM" / "geometry" / "build_3d_guitar.py")]
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
        raise RuntimeError(
            f"Mesh build failed for {sample_id} (exit {proc.returncode}); see {log_path}"
        )
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Expected mesh not written: {mesh_path}")
    return mesh_path.resolve()
