#!/usr/bin/env python3
"""Build ``FEM/mesh/guitar_3d.msh`` from a specific FEM case JSON (LHS-merged sample config)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from paths import REPO_ROOT

REQUIRED_FACET_TAGS = (1, 3, 4)
MERGED_CONFIG_DIR = REPO_ROOT / "FEM" / "SORTING" / "pipeline_merged_configs"


def _mesh_sidecar_paths(mesh_out: Path) -> list[Path]:
    mesh_dir = mesh_out.parent
    return [
        mesh_out,
        mesh_dir / "guitar_3d.h5",
        mesh_dir / "guitar_3d.xdmf",
        mesh_dir / "_xdmf_cache",
    ]


def _remove_stale_mesh_artifacts(mesh_out: Path) -> None:
    import shutil

    for path in _mesh_sidecar_paths(mesh_out):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
            print(f"[mesh_sync] Removed cache dir {path}")
        elif path.is_file():
            path.unlink()
            print(f"[mesh_sync] Removed stale {path}")


def verify_msh_physical_tags(msh_path: Path, required_facet_tags: tuple[int, ...] = REQUIRED_FACET_TAGS) -> None:
    """Raise if triangle facets lack Top/Back/Ribs physical tags (1, 3, 4)."""
    try:
        import meshio
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("meshio is required to verify guitar_3d.msh physical tags.") from exc

    mesh = meshio.read(str(msh_path))
    phys = mesh.cell_data_dict.get("gmsh:physical")
    if not phys:
        raise RuntimeError(f"{msh_path}: no gmsh:physical cell_data (mesh is physically unlabelled).")

    tri_counts: dict[int, int] = {}
    if isinstance(phys, dict):
        tri_raw = phys.get("triangle")
        if tri_raw is not None:
            arr = np.asarray(tri_raw, dtype=np.int32).ravel()
            uniq, cnt = np.unique(arr, return_counts=True)
            tri_counts = {int(t): int(c) for t, c in zip(uniq, cnt)}
    else:
        for i, block in enumerate(mesh.cells):
            if block.type != "triangle" or i >= len(phys):
                continue
            arr = np.asarray(phys[i], dtype=np.int32).ravel()
            uniq, cnt = np.unique(arr, return_counts=True)
            for t, c in zip(uniq, cnt):
                tri_counts[int(t)] = tri_counts.get(int(t), 0) + int(c)

    missing = [t for t in required_facet_tags if tri_counts.get(int(t), 0) <= 0]
    if missing:
        present = ", ".join(f"{k}({v})" for k, v in sorted(tri_counts.items()))
        raise RuntimeError(
            f"{msh_path}: triangle facets missing required physical tags {missing}. "
            f"Present triangle tags: {present or '(none)'}"
        )

    summary = ", ".join(f"tag{t}={tri_counts[t]}" for t in required_facet_tags)
    print(f"[mesh_sync] Physical tag audit OK ({summary})")


def build_mesh_for_config(config_path: Path, repo_root: Path | None = None) -> Path:
    """
    Run ``build_3d_guitar.py`` with ``--config`` so Gmsh geometry matches the sample LHS file.

    Deletes stale ``guitar_3d.msh`` / ``guitar_3d.h5`` (and XDMF cache) before building, then
    verifies facet tags 1 (Top), 3 (Back), and 4 (Ribs) are present in the written mesh.

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
    _remove_stale_mesh_artifacts(mesh_out)

    cmd = [sys.executable, str(script), "-nopopup", "--config", str(cfg)]
    # Exclusive mesh mode (same as gui/app.py _run_gmsh): leaked FEM_ALLOW_PREVIEW breaks FOM.
    clean = {
        k: v
        for k, v in os.environ.items()
        if k not in ("FEM_ALLOW_PREVIEW", "FEM_ALLOW_DISPLAY", "FEM_ALLOW_FOM")
    }
    env = {**clean, "FEM_ALLOW_FOM": "1"}
    proc = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True, env=env)
    if proc.stdout:
        print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
    if proc.returncode != 0:
        raise RuntimeError(
            f"Gmsh mesh build failed for config {cfg}.\nstdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )
    if not mesh_out.is_file():
        raise FileNotFoundError(f"Mesh build succeeded but {mesh_out} was not created.")

    verify_msh_physical_tags(mesh_out)
    print(f"[mesh_sync] Built {mesh_out} from config {cfg}")
    return mesh_out


def write_merged_sample_config(
    sample_id: str,
    *,
    base_config: Path | None = None,
    pool_path: Path | None = None,
    repo_root: Path | None = None,
) -> Path:
    """Merge LHS pool entry into base FEM JSON; write ``pipeline_merged_configs/sample_XXX.json``."""
    import copy
    import json

    from run_pipeline import (
        _atomic_write_json,
        _default_pool_path,
        _find_pool_entry,
        _parse_sample_index,
        _pool_sample_id,
        _resolve_sample_parameters,
    )
    from wood_library import apply_lhs_parameters_to_config, finalize_plate_thickness_geometry

    root = Path(repo_root).resolve() if repo_root is not None else REPO_ROOT
    n = _parse_sample_index(sample_id)
    sample_key = _pool_sample_id(n)
    pool = pool_path.resolve() if pool_path is not None else _default_pool_path()
    base = (base_config or (root / "FEM" / "configs" / "guitar_3d.json")).resolve()
    if not base.is_file():
        raise FileNotFoundError(f"Base FEM config not found: {base}")
    if not pool.is_file():
        raise FileNotFoundError(f"LHS pool not found: {pool}")

    pool_entry = _find_pool_entry(pool, sample_key)
    parameters = _resolve_sample_parameters(sample_key, pool_entry, None)
    merged = copy.deepcopy(json.loads(base.read_text(encoding="utf-8")))
    if parameters:
        apply_lhs_parameters_to_config(merged, parameters)
    geom = merged.get("geometry")
    if isinstance(geom, dict):
        finalize_plate_thickness_geometry(geom)
    out = MERGED_CONFIG_DIR / f"{sample_key}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(out, merged)
    print(f"[mesh_sync] Wrote merged config -> {out}")
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build guitar_3d.msh from a FEM case JSON.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="FEM case JSON passed to build_3d_guitar.py (default: FEM/configs/guitar_3d.json)",
    )
    parser.add_argument(
        "--sample-id",
        type=str,
        default=None,
        help='LHS sample (e.g. "0" or "sample_000"): merge pool into base config, then mesh.',
    )
    parser.add_argument(
        "--pool",
        type=Path,
        default=None,
        help="lhs_pool.json (default: FEM/configs/lhs_pool.json or ROM/classic/lhs_pool.json)",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=REPO_ROOT / "FEM" / "configs" / "guitar_3d.json",
        help="Base FEM JSON when using --sample-id",
    )
    args = parser.parse_args()
    if args.sample_id is not None:
        cfg = write_merged_sample_config(
            args.sample_id,
            base_config=args.base_config,
            pool_path=args.pool,
        )
    else:
        cfg = (args.config or args.base_config).resolve()
    build_mesh_for_config(cfg, REPO_ROOT)
