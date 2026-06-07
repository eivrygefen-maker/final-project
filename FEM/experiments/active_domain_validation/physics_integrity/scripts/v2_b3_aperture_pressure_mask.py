#!/usr/bin/env python3
"""Experimental sample-specific aperture pressure probe mask (not wired to production)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
TAG_SOUNDHOLE = 2
AIR_VOLUME_TAG = 10
DEFAULT_PROBE_RADIUS_M = 0.025
DEFAULT_SOUNDHOLE_FROM_NECK = 0.5


def _sha256_indices(indices: Sequence[int]) -> str:
    payload = ",".join(str(int(i)) for i in sorted(indices))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def soundhole_center_from_geometry(geometry: Mapping[str, float], *, from_neck: float = DEFAULT_SOUNDHOLE_FROM_NECK) -> Tuple[float, float, float]:
    """Classical guitar hole centre in body frame (matches build_3d_guitar convention)."""
    length = float(geometry["length"])
    width = float(geometry["width"])
    depth = float(geometry["depth"])
    hole_x = (0.5 - float(from_neck)) * length - 0.5 * length
    hole_y = 0.0
    hole_z = 0.5 * depth
    return float(hole_x), float(hole_y), float(hole_z)


def build_aperture_pressure_mask(
    mesh_file: Path,
    *,
    geometry: Mapping[str, float],
    built_meta: Mapping[str, Any],
    probe_radius_m: float = DEFAULT_PROBE_RADIUS_M,
) -> Dict[str, Any]:
    """
    Build non-empty pressure-DOF mask near the soundhole aperture in air volume.

    Uses interior air pressure DOFs within probe_radius of the geometry-derived hole centre.
    Maps volume DOF indices to global W rows via built_metadata p_idx layout.
    """
    repo_scripts = SCRIPT_DIR.parent.parent.parent / "scripts"
    import sys

    if str(repo_scripts) not in sys.path:
        sys.path.insert(0, str(repo_scripts))
    import fem_main_3d as fem3d  # noqa: WPS433
    from basix.ufl import element as basix_element  # noqa: WPS433
    from dolfinx import fem  # noqa: WPS433

    mesh_file = Path(mesh_file).expanduser().resolve()
    msh, cell_tags, facet_tags = fem3d._load_mesh_and_tags(mesh_file)
    coords = np.asarray(msh.geometry.x, dtype=np.float64)
    hx, hy, hz = soundhole_center_from_geometry(geometry)

    V_p = fem.functionspace(msh, basix_element("Lagrange", msh.basix_cell(), 1))
    p_air_v = fem3d._locate_air_volume_pressure_dofs(V_p, msh, cell_tags)

    # Prefer DOFs near hole centre; fallback to soundhole facet-adjacent air DOFs.
    dists = np.linalg.norm(coords[p_air_v] - np.array([hx, hy, hz], dtype=np.float64), axis=1)
    probe_v = p_air_v[dists <= float(probe_radius_m)]
    method = "air_volume_radius_probe_v1"
    if probe_v.size == 0:
        soundhole_facets = np.asarray(facet_tags.find(TAG_SOUNDHOLE), dtype=np.int32)
        p_sh = fem3d._locate_soundhole_pressure_release_dofs(V_p, soundhole_facets)
        probe_v = np.intersect1d(p_air_v, p_sh)
        method = "soundhole_facet_pressure_intersection_v1"
    if probe_v.size == 0:
        # Last resort: nearest 16 air DOFs to hole centre.
        order = np.argsort(dists)[: min(16, p_air_v.size)]
        probe_v = p_air_v[order]
        method = "nearest_air_dofs_fallback_v1"

    p_idx_all = np.asarray(built_meta.get("p_idx") or [], dtype=np.int32).ravel()
    n_u_b3 = int(built_meta.get("n_u_b3") or 0)
    if p_idx_all.size == 0:
        raise ValueError("built_metadata missing p_idx")

    # Volume pressure DOF i maps to W row p_idx_all[i] when layouts align.
    p_idx_aperture = p_idx_all[np.asarray(probe_v, dtype=np.int32)]
    p_idx_aperture = np.unique(p_idx_aperture.astype(np.int32, copy=False))

    probe_coords = coords[np.asarray(probe_v, dtype=np.int32)]
    return {
        "mask_method": method,
        "probe_radius_m": float(probe_radius_m),
        "soundhole_center_m": [hx, hy, hz],
        "n_p_aperture_dofs": int(p_idx_aperture.size),
        "p_idx_aperture": p_idx_aperture,
        "aperture_index_sha256": _sha256_indices(p_idx_aperture.tolist()),
        "coordinate_bbox_min": probe_coords.min(axis=0).tolist() if probe_coords.size else None,
        "coordinate_bbox_max": probe_coords.max(axis=0).tolist() if probe_coords.size else None,
        "mesh_file": str(mesh_file),
        "n_u_b3": n_u_b3,
        "geometry_fingerprint_source": dict(geometry),
    }


def aperture_mask_summary(mask: Mapping[str, Any]) -> Dict[str, Any]:
    """JSON-serializable summary without large index arrays."""
    out = {k: v for k, v in mask.items() if k != "p_idx_aperture"}
    arr = mask.get("p_idx_aperture")
    if arr is not None:
        out["p_idx_aperture_count"] = int(np.asarray(arr).size)
    return out


def write_aperture_mask_npz(path: Path, mask: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = aperture_mask_summary(mask)
    np.savez_compressed(
        path,
        p_idx_aperture=np.asarray(mask["p_idx_aperture"], dtype=np.int32),
        mask_method=np.asarray([str(mask.get("mask_method") or "")]),
        probe_radius_m=np.asarray([float(mask.get("probe_radius_m") or 0.0)]),
        soundhole_center_m=np.asarray(mask.get("soundhole_center_m") or [0.0, 0.0, 0.0], dtype=np.float64),
        metadata_json=np.asarray([json.dumps(meta, sort_keys=True)]),
    )
