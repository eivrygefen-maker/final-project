#!/usr/bin/env python3
"""Build and audit meshes for v2_mesh_convergence levels."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[5]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_CONFIG = REPO_ROOT / "FEM" / "configs" / "guitar_3d.json"

from v2_mesh_convergence_common import (
    CONV_MESH,
    mesh_audit_path,
    mesh_path,
    write_json,
)
from v2_sensitivity_mesh import sample_geometry


def _mesh_audit(msh: Path, out_json: Path) -> Dict[str, Any]:
    py = sys.executable
    audit_script = EXPERIMENT_ROOT / "scripts" / "inspect_mesh_and_tags.py"
    subprocess.run(
        [py, str(audit_script), "--mesh", str(msh.resolve()), "--out", str(out_json.resolve())],
        cwd=str(REPO_ROOT),
        check=False,
    )
    if out_json.is_file():
        return json.loads(out_json.read_text(encoding="utf-8"))
    return {"mesh_file": str(msh), "error": "audit_failed"}


def build_level_mesh(
    case: Dict[str, Any],
    level_id: str,
    level_def: Dict[str, Any],
    *,
    config_dir: Path,
) -> Dict[str, Any]:
    case_id = str(case["id"])
    out_msh = mesh_path(level_id, case_id)
    out_audit = mesh_audit_path(level_id, case_id)
    level_env = dict(level_def.get("build_env") or {})
    lc_scale = float(level_def.get("lc_scale", 1.0))

    if out_msh.is_file() and out_audit.is_file():
        audit = json.loads(out_audit.read_text(encoding="utf-8"))
        audit["reused_existing_mesh"] = True
        return audit

    config_dir.mkdir(parents=True, exist_ok=True)
    CONV_MESH.joinpath(level_id).mkdir(parents=True, exist_ok=True)
    cfg_path = config_dir / f"{level_id}_{case_id}.json"
    cfg = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
    cfg["geometry"] = sample_geometry({"geometry": case.get("geometry") or {}})
    cfg.setdefault("solver", {})["mesh_file"] = str(out_msh.resolve())
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    env = os.environ.copy()
    env.pop("FEM_VALIDATION_MESH", None)
    env.pop("FEM_ALLOW_FOM", None)
    for k, v in level_env.items():
        env[str(k)] = str(v)
    env["FEM_MESH_LC_SCALE"] = str(lc_scale)
    env["FEM_MESH_OUT"] = str(out_msh.resolve())
    env["FEM_MESH_CONFIG"] = str(cfg_path.resolve())

    log_path = CONV_MESH / level_id / f"{case_id}_build.log"
    cmd = [sys.executable, str(REPO_ROOT / "FEM" / "geometry" / "build_3d_guitar.py")]
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(
            cmd, cwd=str(REPO_ROOT), env=env, stdout=logf, stderr=subprocess.STDOUT, check=False
        )
    if proc.returncode != 0 or not out_msh.is_file():
        return {
            "mesh_file": str(out_msh),
            "build_exit_code": proc.returncode,
            "build_failed": True,
            "log": str(log_path),
        }

    audit = _mesh_audit(out_msh, out_audit)
    audit["build_exit_code"] = proc.returncode
    audit["mesh_level"] = level_id
    audit["case_id"] = case_id
    audit["lc_scale"] = lc_scale
    audit["build_env"] = level_env
    manifest_controls = {}
    if "FEM_VALIDATION_MESH" in level_env:
        manifest_controls = {
            "wood_surface_size_m": 0.014 * lc_scale,
            "wood_thickness_size_m": 0.003 * lc_scale,
            "air_threshold_size_min_m": 0.009 * lc_scale,
            "air_threshold_size_max_m": 0.04 * lc_scale,
        }
    elif "FEM_ALLOW_FOM" in level_env:
        manifest_controls = {
            "wood_surface_size_m": 0.007 * lc_scale,
            "wood_thickness_size_m": 0.001 * lc_scale,
            "air_threshold_size_min_m": 0.004 * lc_scale,
            "air_threshold_size_max_m": 0.05 * lc_scale,
        }
    audit["effective_controls_m"] = manifest_controls
    write_json(out_audit, audit)
    return audit
