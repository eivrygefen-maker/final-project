#!/usr/bin/env python3
"""Experimental sample-specific aperture pressure probe mask (validation-only, not production)."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
TAG_SOUNDHOLE = 2
TAG_TOP = 1
AIR_VOLUME_TAG = 10
DEFAULT_SOUNDHOLE_FROM_NECK = 0.5
MIN_APERTURE_DOFS = 8
MAX_APERTURE_DOFS = 32
RADIUS_FACTORS = (1.0, 1.25, 1.5)
AXIAL_CELL_MULTS = (1, 2, 3)


def _as_int32_index_map(value: Any) -> np.ndarray:
    """Coerce index map without NumPy truth-value tests (never use ``value or []``)."""
    if value is None:
        return np.asarray([], dtype=np.int32)
    return np.asarray(value, dtype=np.int32).ravel()


def _as_float64_coords(value: Any) -> np.ndarray:
    """Coerce coordinate rows without ndarray ``or`` fallbacks."""
    if value is None:
        return np.asarray([], dtype=np.float64)
    return np.asarray(value, dtype=np.float64)


def _optional_float(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    return float(value)


def _sha256_indices(indices: Sequence[int]) -> str:
    payload = ",".join(str(int(i)) for i in sorted(indices))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _import_fem_main_3d() -> Any:
    from v2_b3_synthesis_export import import_fem_main_3d  # noqa: WPS433

    fem3d, _diag = import_fem_main_3d(start=SCRIPT_DIR)
    return fem3d


def soundhole_center_from_geometry(
    geometry: Mapping[str, float],
    *,
    from_neck: float = DEFAULT_SOUNDHOLE_FROM_NECK,
) -> Tuple[float, float, float]:
    """Body frame: x=+L/2 neck, x=-L/2 tail (matches build_3d_guitar)."""
    length = float(geometry["length"])
    depth = float(geometry["depth"])
    hole_x = (0.5 - float(from_neck)) * length
    hole_y = 0.0
    top_t = float(geometry.get("top_thickness") or 0.003)
    hole_z = float(depth) - float(top_t)
    return float(hole_x), float(hole_y), float(hole_z)


def _facet_centroid(msh: Any, facet_index: int) -> np.ndarray:
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, 0)
    f2v = msh.topology.connectivity(fdim, 0)
    verts = f2v.links(int(facet_index))
    return np.mean(np.asarray(msh.geometry.x[np.asarray(verts, dtype=np.int32)], dtype=np.float64), axis=0)


def _soundhole_facet_geometry(
    msh: Any,
    facet_tags: Any,
    *,
    geometry: Mapping[str, float],
) -> Dict[str, Any]:
    from dolfinx import fem  # noqa: WPS433

    fem3d = _import_fem_main_3d()
    soundhole_facets = np.asarray(facet_tags.find(TAG_SOUNDHOLE), dtype=np.int32)
    if soundhole_facets.size == 0:
        hx, hy, hz = soundhole_center_from_geometry(geometry)
        hr = float(geometry.get("hole_radius") or 0.047)
        return {
            "n_soundhole_facets": 0,
            "center_m": [hx, hy, hz],
            "radius_m": hr,
            "plane_z_m": hz,
            "source": "geometry_fallback_no_tag2_facets",
        }

    centroids = np.asarray([_facet_centroid(msh, int(fi)) for fi in soundhole_facets], dtype=np.float64)
    center = np.mean(centroids, axis=0)
    xy = centroids[:, :2] - center[:2]
    radii = np.linalg.norm(xy, axis=1)
    hr_geom = float(geometry.get("hole_radius") or 0.047)
    radius = float(max(np.max(radii) if radii.size else hr_geom, hr_geom * 0.5))
    return {
        "n_soundhole_facets": int(soundhole_facets.size),
        "center_m": center.tolist(),
        "radius_m": radius,
        "plane_z_m": float(center[2]),
        "facet_centroid_bbox_min": centroids.min(axis=0).tolist(),
        "facet_centroid_bbox_max": centroids.max(axis=0).tolist(),
        "source": "mesh_facet_tag2_centroids",
        "geometry_center_m": list(soundhole_center_from_geometry(geometry)),
    }


def _estimate_cell_length_m(msh: Any, *, near_point: np.ndarray) -> float:
    coords = np.asarray(msh.geometry.x, dtype=np.float64)
    if coords.shape[0] < 2:
        return 0.01
    d = np.linalg.norm(coords - near_point.reshape(1, 3), axis=1)
    order = np.argsort(d)[: min(32, coords.shape[0])]
    near = coords[order]
    if near.shape[0] < 2:
        return 0.01
    diffs = near[1:] - near[:-1]
    lens = np.linalg.norm(diffs, axis=1)
    positive = lens[lens > 1.0e-9]
    return float(np.median(positive)) if positive.size else 0.01


def _load_pressure_layout(
    mesh_file: Path,
    *,
    core_config_path: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, Any, Any, Any, Any]:
    """
    Return (p_air_collapsed, p_dof_coords, msh, cell_tags, facet_tags, V_p_collapsed).

    Uses coupled replay (solve_evp=False) to match checkpoint pressure DOF numbering.
    """
    fem3d = _import_fem_main_3d()
    from basix.ufl import element as basix_element  # noqa: WPS433
    from dolfinx import fem  # noqa: WPS433

    mesh_file = Path(mesh_file).expanduser().resolve()
    msh, cell_tags, facet_tags = fem3d._load_mesh_and_tags(mesh_file)

    if core_config_path and core_config_path.is_file():
        cfg = json.loads(core_config_path.read_text(encoding="utf-8"))
    else:
        cfg = json.loads(
            (SCRIPT_DIR.parent / "configs" / "coupled_physical_core_v2.json").read_text(encoding="utf-8")
        )
    sc = cfg.setdefault("solver", {})
    sc["mesh_file"] = str(mesh_file)
    sc["coupled_air_pressure_restriction_diagnosis"] = True
    sc["coupled_air_pressure_restriction_replay_audit"] = True

    _msh, W, A, M = fem3d._solve_coupled_evp(
        mesh_file=mesh_file,
        config=cfg,
        num_modes=0,
        solve_evp=False,
    )
    try:
        p_air_cfg = _as_int32_index_map(cfg.get("_coupled_air_p_air_collapsed_indices"))
        V_p, _ = W.sub(1).collapse()
        p_air_v = np.asarray(
            fem3d._locate_air_volume_pressure_dofs(V_p, msh, cell_tags),
            dtype=np.int32,
        ).ravel()
        if p_air_cfg.size and p_air_v.size and not np.array_equal(np.sort(p_air_cfg), np.sort(p_air_v)):
            raise RuntimeError(
                f"pressure layout mismatch: replay cfg n={p_air_cfg.size} locate n={p_air_v.size}"
            )
        p_air = p_air_cfg if p_air_cfg.size else p_air_v
        dof_coords = np.asarray(V_p.tabulate_dof_coordinates(), dtype=np.float64)
        if dof_coords.shape[0] <= int(p_air.max() if p_air.size else 0):
            raise RuntimeError(
                f"pressure dof coords too small: {dof_coords.shape[0]} for max index {p_air.max()}"
            )
        return p_air, dof_coords, msh, cell_tags, facet_tags, V_p
    finally:
        try:
            A.destroy()
            M.destroy()
        except Exception:
            pass


def _adjacent_air_cells_to_soundhole(msh: Any, cell_tags: Any, facet_tags: Any) -> np.ndarray:
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)
    f2c = msh.topology.connectivity(fdim, tdim)
    soundhole_facets = np.asarray(facet_tags.find(TAG_SOUNDHOLE), dtype=np.int32)
    air_cells: List[int] = []
    for fi in soundhole_facets:
        try:
            adj = np.asarray(f2c.links(int(fi)), dtype=np.int32)
        except Exception:
            adj = np.array([], dtype=np.int32)
        for c in adj:
            if int(cell_tags.values[int(c)]) == AIR_VOLUME_TAG:
                air_cells.append(int(c))
    return np.unique(np.asarray(air_cells, dtype=np.int32))


def _pressure_positions_near_aperture(
    *,
    p_air: np.ndarray,
    dof_coords: np.ndarray,
    aperture: Mapping[str, Any],
    msh: Any,
    cell_tags: Any,
    facet_tags: Any,
    fem3d: Any,
    V_p: Any,
) -> Tuple[np.ndarray, str, Dict[str, Any]]:
    from dolfinx import fem  # noqa: WPS433

    center = np.asarray(aperture["center_m"], dtype=np.float64)
    radius = float(aperture["radius_m"])
    plane_z = float(aperture["plane_z_m"])
    cell_h = _estimate_cell_length_m(msh, near_point=center)

    inv_air = {int(d): k for k, d in enumerate(p_air.tolist())}
    meta: Dict[str, Any] = {
        "estimated_cell_length_m": cell_h,
        "radius_factors_tried": [],
        "axial_cell_mults_tried": [],
    }

    # Method 1: facet tag -> adjacent cavity-air cells -> pressure DOFs
    adj_air_cells = _adjacent_air_cells_to_soundhole(msh, cell_tags, facet_tags)
    if adj_air_cells.size:
        tdim = msh.topology.dim
        dofs_adj = np.unique(
            np.asarray(fem.locate_dofs_topological(V_p, tdim, adj_air_cells), dtype=np.int32).ravel()
        )
        pos_adj = np.asarray([inv_air[int(d)] for d in dofs_adj if int(d) in inv_air], dtype=np.int32)
        if pos_adj.size >= MIN_APERTURE_DOFS:
            meta.update(
                {
                    "selection_method": "facet_adjacent_air_cell_dofs_v1",
                    "n_adjacent_air_cells": int(adj_air_cells.size),
                    "radius_factor": 1.0,
                    "axial_cell_mult": 0,
                }
            )
            return pos_adj, "facet_adjacent_air_cell_dofs_v1", meta

    # Method 2: nearfield slab/cylinder inside cavity with adaptive expansion
    coords_air = dof_coords[p_air]
    xy_dist = np.linalg.norm(coords_air[:, :2] - center[:2], axis=1)
    z_dist = np.abs(coords_air[:, 2] - plane_z)

    for rf in RADIUS_FACTORS:
        for am in AXIAL_CELL_MULTS:
            axial_th = float(am) * cell_h
            radial_th = float(rf) * radius
            sel = np.asarray(
                [k for k, d in enumerate(p_air.tolist()) if xy_dist[k] <= radial_th and z_dist[k] <= axial_th],
                dtype=np.int32,
            )
            meta["radius_factors_tried"].append({"factor": rf, "count": int(sel.size)})
            meta["axial_cell_mults_tried"].append({"mult": am, "count": int(sel.size)})
            if MIN_APERTURE_DOFS <= sel.size <= MAX_APERTURE_DOFS * 4:
                meta.update(
                    {
                        "selection_method": "aperture_nearfield_pressure_slab_v1",
                        "radius_factor": rf,
                        "axial_cell_mult": am,
                        "probe_radius_m": radial_th,
                        "probe_axial_half_thickness_m": axial_th,
                    }
                )
                return sel, "aperture_nearfield_pressure_slab_v1", meta
            if sel.size > 0 and sel.size < MIN_APERTURE_DOFS:
                continue
            if sel.size > MAX_APERTURE_DOFS * 4:
                # keep closest subset
                order = np.argsort(xy_dist[sel] + 0.5 * z_dist[sel])[:MAX_APERTURE_DOFS]
                trimmed = sel[order]
                meta.update(
                    {
                        "selection_method": "aperture_nearfield_pressure_slab_trimmed_v1",
                        "radius_factor": rf,
                        "axial_cell_mult": am,
                        "probe_radius_m": radial_th,
                        "probe_axial_half_thickness_m": axial_th,
                        "trimmed_from": int(sel.size),
                    }
                )
                return trimmed, "aperture_nearfield_pressure_slab_trimmed_v1", meta

    # Controlled expansion: nearest MIN_APERTURE_DOFS to center (never global cavity max)
    dist = xy_dist + 0.25 * z_dist
    order = np.argsort(dist)[: min(max(MIN_APERTURE_DOFS, 1), p_air.size)]
    meta.update({"selection_method": "nearest_air_pressure_dofs_minimum_count_v1", "radius_factor": None})
    return np.asarray(order, dtype=np.int32), "nearest_air_pressure_dofs_minimum_count_v1", meta


def diagnose_aperture_pressure_mask(
    mesh_file: Path,
    *,
    geometry: Mapping[str, float],
    built_meta: Mapping[str, Any],
    core_config_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Task 1 diagnostic report for empty-mask root cause analysis."""
    fem3d = _import_fem_main_3d()
    from basix.ufl import element as basix_element  # noqa: WPS433
    from dolfinx import fem  # noqa: WPS433

    mesh_file = Path(mesh_file).expanduser().resolve()
    p_air, dof_coords, msh, cell_tags, facet_tags, _V_p = _load_pressure_layout(
        mesh_file, core_config_path=core_config_path
    )
    aperture = _soundhole_facet_geometry(msh, facet_tags, geometry=geometry)
    center = np.asarray(aperture["center_m"], dtype=np.float64)
    radius = float(aperture["radius_m"])

    coords = np.asarray(msh.geometry.x, dtype=np.float64)
    air_coords = dof_coords[p_air]
    dist = np.linalg.norm(air_coords - center.reshape(1, 3), axis=1)

    n_u_b3 = int(built_meta.get("n_u_b3") or 0)
    p_idx = _as_int32_index_map(built_meta.get("p_idx"))
    active_local = _as_int32_index_map(built_meta.get("active_local"))
    free_rows = _as_int32_index_map(built_meta.get("free_rows"))
    bc_rows = _as_int32_index_map(built_meta.get("bc_rows"))

    # Legacy bug demo: vertex indexing mistake
    V_p = fem.functionspace(msh, basix_element("Lagrange", msh.basix_cell(), 1))
    p_air_v = fem3d._locate_air_volume_pressure_dofs(V_p, msh, cell_tags)
    legacy_wrong = 0
    try:
        legacy_wrong = int(np.sum(np.linalg.norm(coords[p_air_v] - center, axis=1) <= radius))
    except Exception:
        legacy_wrong = -1

    cell_h = _estimate_cell_length_m(msh, near_point=center)
    radii_counts = {}
    for factor, label in (
        (0.5, "0.5r"),
        (1.0, "1.0r"),
        (1.25, "1.25r"),
        (1.5, "1.5r"),
        (2.0, "2.0r"),
    ):
        radii_counts[label] = int(np.sum(dist <= factor * radius))

    xy_dist = np.linalg.norm(air_coords[:, :2] - center[:2], axis=1)
    plane_z = _optional_float(aperture.get("plane_z_m"), float(center[2]))
    z_dist = np.abs(air_coords[:, 2] - plane_z)
    thin_slab_counts = {}
    for am in (1, 2, 3):
        axial_th = float(am) * cell_h
        thin_slab_counts[f"r<=1.0_axial<={am}_cells"] = int(
            np.sum((xy_dist <= radius) & (z_dist <= axial_th))
        )

    facet_tag_counts = {}
    for tag in (1, 2, 3, 4, 5, 10):
        try:
            if tag == 10:
                facet_tag_counts[str(tag)] = int(np.asarray(cell_tags.find(tag)).size)
            else:
                facet_tag_counts[str(tag)] = int(np.asarray(facet_tags.find(tag)).size)
        except Exception:
            facet_tag_counts[str(tag)] = 0

    return {
        "soundhole_geometry": aperture,
        "geometry_hole_radius_m": float(geometry.get("hole_radius") or 0.0),
        "mesh_bbox_min": coords.min(axis=0).tolist(),
        "mesh_bbox_max": coords.max(axis=0).tolist(),
        "air_pressure_dof_count": int(p_air.size),
        "air_pressure_coord_shape": list(air_coords.shape),
        "air_pressure_coord_bbox_min": air_coords.min(axis=0).tolist() if air_coords.size else None,
        "air_pressure_coord_bbox_max": air_coords.max(axis=0).tolist() if air_coords.size else None,
        "min_distance_any_air_pressure_dof_to_center_m": float(dist.min()) if dist.size else None,
        "counts_within_radius": radii_counts,
        "thin_aperture_slab_counts": thin_slab_counts,
        "estimated_cell_length_m": cell_h,
        "legacy_vertex_indexing_bug_count_at_1r": legacy_wrong,
        "root_cause_notes": [
            "Previous mask used coords[p_air_v] treating pressure DOF indices as mesh vertex indices.",
            "p_idx maps pressure position k to B3 W row n_u_b3+k, not parent vertex ids.",
            "Soundhole Dirichlet DOFs may be eliminated from active solve; prefer interior air DOFs adjacent to tag-2 facets.",
        ],
        "facet_tag_counts": facet_tag_counts,
        "n_u_b3": n_u_b3,
        "p_idx_len": int(p_idx.size),
        "active_local_len": int(active_local.size),
        "bc_rows_len": int(bc_rows.size),
        "mapping_contract": {
            "p_position_to_W": "w_row = n_u_b3 + position_k",
            "active_mapping": "active_index = index in active_local where free_rows[active_local] == w_row",
        },
    }


def build_aperture_pressure_mask(
    mesh_file: Path,
    *,
    geometry: Mapping[str, float],
    built_meta: Mapping[str, Any],
    core_config_path: Optional[Path] = None,
    probe_radius_m: float = 0.025,
) -> Dict[str, Any]:
    """
    Build non-empty aperture pressure mask as B3 W-row indices (p_idx_aperture).

    Methods (in order): facet-adjacent air cell DOFs, nearfield slab, controlled nearest fallback.
    Never falls back to global cavity maximum.
    """
    del probe_radius_m  # geometry-relative radii used instead
    fem3d = _import_fem_main_3d()

    mesh_file = Path(mesh_file).expanduser().resolve()
    p_air, dof_coords, msh, cell_tags, facet_tags, V_p = _load_pressure_layout(
        mesh_file, core_config_path=core_config_path
    )
    aperture = _soundhole_facet_geometry(msh, facet_tags, geometry=geometry)
    positions, method, sel_meta = _pressure_positions_near_aperture(
        p_air=p_air,
        dof_coords=dof_coords,
        aperture=aperture,
        msh=msh,
        cell_tags=cell_tags,
        facet_tags=facet_tags,
        fem3d=fem3d,
        V_p=V_p,
    )

    n_u_b3 = int(built_meta.get("n_u_b3") or 0)
    p_idx = _as_int32_index_map(built_meta.get("p_idx"))
    bc_rows = set(int(x) for x in _as_int32_index_map(built_meta.get("bc_rows")).tolist())
    active_local = _as_int32_index_map(built_meta.get("active_local"))
    free_rows = _as_int32_index_map(built_meta.get("free_rows"))

    w_rows = np.asarray([n_u_b3 + int(k) for k in positions.tolist()], dtype=np.int32)
    w_rows = w_rows[(w_rows >= 0) & (w_rows < int(built_meta.get("n_w") or (n_u_b3 + p_idx.size)))]
    w_rows = np.asarray([w for w in w_rows.tolist() if int(w) not in bc_rows], dtype=np.int32)
    w_rows = np.unique(w_rows)

    if w_rows.size == 0:
        raise RuntimeError("aperture_pressure_mask_empty_after_W_mapping")

    if p_idx.size and not np.all(np.isin(w_rows, p_idx)):
        bad = w_rows[~np.isin(w_rows, p_idx)]
        raise RuntimeError(f"aperture indices outside pressure block: {bad[:8].tolist()}")

    sel_coords = dof_coords[p_air[positions]]
    active_indices: List[int] = []
    for w in w_rows.tolist():
        free_pos = np.where(free_rows == int(w))[0]
        if free_pos.size == 0:
            continue
        act = np.where(active_local == int(free_pos[0]))[0]
        if act.size:
            active_indices.append(int(act[0]))

    dist_stats = {
        "min_m": float(np.min(np.linalg.norm(sel_coords - np.asarray(aperture["center_m"]), axis=1))),
        "max_m": float(np.max(np.linalg.norm(sel_coords - np.asarray(aperture["center_m"]), axis=1))),
        "mean_m": float(np.mean(np.linalg.norm(sel_coords - np.asarray(aperture["center_m"]), axis=1))),
    }

    mic_method = (
        "aperture_nearfield_pressure_rms_proxy_v1"
        if "nearfield" in method or "slab" in method
        else "aperture_pressure_rms_proxy_v1"
    )

    return {
        "mask_method": method,
        "mic_output_method": mic_method,
        "soundhole_center_m": aperture["center_m"],
        "soundhole_radius_m": aperture["radius_m"],
        "selection_meta": sel_meta,
        "n_p_aperture_dofs": int(w_rows.size),
        "p_idx_aperture": w_rows,
        "p_active_indices": active_indices,
        "aperture_index_sha256": _sha256_indices(w_rows.tolist()),
        "coordinate_bbox_min": sel_coords.min(axis=0).tolist(),
        "coordinate_bbox_max": sel_coords.max(axis=0).tolist(),
        "distance_to_center_stats_m": dist_stats,
        "mesh_file": str(mesh_file),
        "n_u_b3": n_u_b3,
        "geometry_fingerprint_source": dict(geometry),
        "selected_coordinates": sel_coords.tolist(),
    }


def validate_aperture_mask_contract(mask: Mapping[str, Any], built_meta: Mapping[str, Any]) -> None:
    p_idx = _as_int32_index_map(built_meta.get("p_idx"))
    n_w = int(built_meta.get("n_w") or 0)
    n_u = int(built_meta.get("n_u_b3") or 0)
    active_local = _as_int32_index_map(built_meta.get("active_local"))
    active_dim = int(built_meta.get("active_dimension") or active_local.size or 0)
    arr = _as_int32_index_map(mask.get("p_idx_aperture"))
    if arr.size == 0:
        raise ValueError("contract_fail: empty p_idx_aperture")
    if arr.size != len(np.unique(arr)):
        raise ValueError("contract_fail: duplicate indices")
    if np.any(arr < n_u) or np.any(arr >= n_w):
        raise ValueError("contract_fail: indices outside W pressure block")
    if p_idx.size and not np.all(np.isin(arr, p_idx)):
        raise ValueError("contract_fail: indices not in p_idx pressure rows")
    act = _as_int32_index_map(mask.get("p_active_indices"))
    if act.size and (np.any(act < 0) or np.any(act >= active_dim)):
        raise ValueError("contract_fail: active indices out of bounds")
    center_raw = mask.get("soundhole_center_m")
    center = _as_float64_coords(center_raw if center_raw is not None else [0.0, 0.0, 0.0])
    radius = _optional_float(mask.get("soundhole_radius_m"), 0.0)
    coords = _as_float64_coords(mask.get("selected_coordinates"))
    if coords.size:
        dist = np.linalg.norm(coords - center.reshape(1, 3), axis=1)
        max_allowed = max(2.5 * radius, 0.05)
        if float(np.max(dist)) > max_allowed:
            raise ValueError(
                f"contract_fail: selected coords too far from soundhole (max_dist={float(np.max(dist)):.4f}m)"
            )


def aperture_mask_summary(mask: Mapping[str, Any]) -> Dict[str, Any]:
    out = {
        k: v
        for k, v in mask.items()
        if k not in ("p_idx_aperture", "selected_coordinates", "p_active_indices")
    }
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
        mic_output_method=np.asarray([str(mask.get("mic_output_method") or "aperture_pressure_rms_proxy_v1")]),
        soundhole_center_m=_as_float64_coords(
            mask.get("soundhole_center_m")
            if mask.get("soundhole_center_m") is not None
            else [0.0, 0.0, 0.0]
        ),
        metadata_json=np.asarray([json.dumps(meta, sort_keys=True)]),
    )


def write_aperture_diagnostic_json(path: Path, diag: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(diag, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_aperture_coordinates_csv(path: Path, mask: Mapping[str, Any]) -> None:
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    coords_raw = mask.get("selected_coordinates")
    coords = coords_raw if coords_raw is not None else []
    w_rows = _as_int32_index_map(mask.get("p_idx_aperture"))
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["w_row_index", "x_m", "y_m", "z_m"])
        for i, c in enumerate(coords):
            w.writerow([int(w_rows[i]) if i < w_rows.size else "", c[0], c[1], c[2]])
