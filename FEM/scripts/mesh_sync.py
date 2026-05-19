#!/usr/bin/env python3
"""Build ``FEM/mesh/guitar_3d.msh`` from a specific FEM case JSON (LHS-merged sample config)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from paths import REPO_ROOT


def build_mesh_for_config(config_path: Path, repo_root: Path | None = None) -> Path:
    """
    Run ``build_3d_guitar.py`` with ``--config`` so Gmsh geometry matches the sample LHS file.

    Returns the path to the generated ``guitar_3d.msh``.
    """
    root = Path(repo_root).resolve() if repo_root is not None else REPO_ROOT
    cfg = Path(config_path).resolve()
    if not cfg.is_file():
        raise FileNotFoundError(f"FEM config for mesh build not found: {cfg}")

    script = root / "FEM" / "geometry" / "build_3d_guitar.py"
    if not script.is_file():
        raise FileNotFoundError(f"Mesh generator not found: {script}")

    mesh_out = root / "FEM" / "mesh" / "guitar_3d.msh"
    cmd = [sys.executable, str(script), "-nopopup", "--config", str(cfg)]
    proc = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Gmsh mesh build failed for config {cfg}.\nstdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )
    if not mesh_out.is_file():
        raise FileNotFoundError(f"Mesh build succeeded but {mesh_out} was not created.")
    print(f"[mesh_sync] Built {mesh_out} from config {cfg}")
    return mesh_out
