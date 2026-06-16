#!/usr/bin/env python3
"""Mesh sidecar manifests and shape-aware reuse validation for v2_mesh_convergence caches."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from v2_b3_m4_lprod_interfaces import extract_geometry_dict, extract_run_metadata  # noqa: E402
from v2_b3_m4_mesh_profile_lib import production_mesh_levels_for_cleanup  # noqa: E402
from v2_b3_petsc_util import write_json_atomic  # noqa: E402

MESH_MANIFEST_SCHEMA = "m4_mesh_manifest_v1"
MESH_GENERATOR_VERSION = "m4_mesh_manifest_v1"
MESH_REUSE_REJECTED = "MESH_REUSE_REJECTED"

GEOMETRY_TOLERANCE = 1.0e-9


def mesh_manifest_path(mesh_path: Path) -> Path:
    return mesh_path.with_suffix(".mesh_manifest.json")


def mesh_audit_path_for(mesh_path: Path) -> Path:
    return mesh_path.with_name(f"{mesh_path.stem}_mesh_audit.json")


def resolve_case_shape_metadata(
    case: Mapping[str, Any],
    *,
    sample_input: Optional[Mapping[str, Any]] = None,
) -> Dict[str, str]:
    shape_name = str(case.get("shape_name") or "classic")
    repo_scripts = Path(__file__).resolve().parents[5] / "FEM" / "scripts"
    import sys

    scripts_str = str(repo_scripts)
    if scripts_str not in sys.path:
        sys.path.insert(0, scripts_str)
    from m4_shape_registry import resolve_geometry_shape_type, resolve_shape_config  # noqa: WPS433

    cfg = resolve_shape_config(shape_name)
    geometry_shape_type = str(
        case.get("geometry_shape_type")
        or case.get("shape_type")
        or (case.get("geometry") or {}).get("shape_type")
        or (sample_input or {}).get("geometry_shape_type")
        or resolve_geometry_shape_type(sample_input=sample_input or case)
        or cfg.geometry_shape_type
    )
    gmsh_shape_type = str(
        case.get("gmsh_shape_type")
        or (sample_input or {}).get("gmsh_shape_type")
        or cfg.gmsh_shape_type
    )
    return {
        "shape_name": shape_name,
        "geometry_shape_type": geometry_shape_type,
        "gmsh_shape_type": gmsh_shape_type,
    }


def build_mesh_manifest(
    *,
    sample_id: str,
    shape_name: str,
    geometry_shape_type: str,
    gmsh_shape_type: str,
    mesh_level: str,
    mesh_path: Path,
    geometry: Mapping[str, Any],
    lhs_path: str = "",
    generator_version: str = MESH_GENERATOR_VERSION,
) -> Dict[str, Any]:
    return {
        "schema": MESH_MANIFEST_SCHEMA,
        "sample_id": sample_id,
        "shape_name": shape_name,
        "geometry_shape_type": geometry_shape_type,
        "gmsh_shape_type": gmsh_shape_type,
        "mesh_level": mesh_level,
        "mesh_path": str(mesh_path),
        "geometry_parameters": {k: float(v) for k, v in geometry.items() if _is_numeric(v)},
        "lhs_path": lhs_path or None,
        "mesh_generator_version": generator_version,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def write_mesh_manifest(manifest_path: Path, body: Mapping[str, Any]) -> Path:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(manifest_path, dict(body))
    return manifest_path


def _is_numeric(val: Any) -> bool:
    try:
        float(val)
        return True
    except (TypeError, ValueError):
        return False


def _geometry_matches(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    keys = sorted(set(expected.keys()) | set(actual.keys()))
    for key in keys:
        if key not in expected:
            continue
        if key not in actual:
            errors.append(f"missing_geometry:{key}")
            continue
        try:
            exp = float(expected[key])
            got = float(actual[key])
        except (TypeError, ValueError):
            errors.append(f"non_numeric_geometry:{key}")
            continue
        if not math.isclose(exp, got, rel_tol=0.0, abs_tol=GEOMETRY_TOLERANCE):
            errors.append(f"geometry_mismatch:{key}:expected={exp}:actual={got}")
    return len(errors) == 0, errors


def validate_mesh_reuse(
    mesh_path: Path,
    *,
    sample_id: str,
    mesh_level: str,
    shape_name: str,
    geometry_shape_type: str,
    gmsh_shape_type: str,
    geometry: Mapping[str, Any],
    lhs_path: str = "",
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    mesh_path = mesh_path.expanduser().resolve()
    manifest_path = mesh_manifest_path(mesh_path)
    if not mesh_path.is_file():
        return False, "mesh_missing", None
    if not manifest_path.is_file():
        return False, "missing_manifest", None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False, "manifest_unreadable", None

    if str(manifest.get("sample_id") or "") != sample_id:
        return False, "sample_id_mismatch", manifest
    if str(manifest.get("mesh_level") or "") != mesh_level:
        return False, "mesh_level_mismatch", manifest
    if str(manifest.get("shape_name") or "") != shape_name:
        return False, "shape_name_mismatch", manifest

    existing_geom = str(manifest.get("geometry_shape_type") or "")
    existing_gmsh = str(manifest.get("gmsh_shape_type") or "")
    if existing_geom != geometry_shape_type:
        return False, f"shape_mismatch:existing={existing_geom}:expected={geometry_shape_type}", manifest
    if existing_gmsh != gmsh_shape_type:
        return False, f"gmsh_shape_mismatch:existing={existing_gmsh}:expected={gmsh_shape_type}", manifest

    geom_ok, geom_errors = _geometry_matches(
        {k: float(v) for k, v in geometry.items() if _is_numeric(v)},
        manifest.get("geometry_parameters") or {},
    )
    if not geom_ok:
        return False, f"geometry_parameters_mismatch:{';'.join(geom_errors)}", manifest

    manifest_lhs = str(manifest.get("lhs_path") or "")
    if lhs_path and manifest_lhs and manifest_lhs != lhs_path:
        return False, "lhs_path_mismatch", manifest

    gen = str(manifest.get("mesh_generator_version") or "")
    if gen and gen != MESH_GENERATOR_VERSION:
        return False, f"generator_version_incompatible:{gen}", manifest

    return True, "ok", manifest


def format_mesh_reuse_rejected(
    *,
    reason: str,
    existing_shape: str,
    expected_shape: str,
    mesh_path: Path,
) -> str:
    return (
        f"{MESH_REUSE_REJECTED} reason={reason} existing={existing_shape} "
        f"expected={expected_shape} mesh={mesh_path}"
    )


def invalidate_stale_mesh_files(mesh_path: Path) -> List[str]:
    removed: List[str] = []
    for path in (
        mesh_path,
        mesh_manifest_path(mesh_path),
        mesh_audit_path_for(mesh_path),
        mesh_path.parent / f"{mesh_path.stem}_mesh_build_summary.json",
        mesh_path.parent / f"{mesh_path.stem}_build.log",
    ):
        if path.is_file():
            path.unlink()
            removed.append(str(path))
    return removed


def collect_global_mesh_cache_paths(repo_root: Path, sample_id: str) -> List[Path]:
    return collect_global_mesh_cache_paths_resolved(repo_root, sample_id)


def collect_global_mesh_cache_paths_resolved(repo_root: Path, sample_id: str) -> List[Path]:
    from v2_mesh_convergence_common import CONV_MESH, mesh_audit_path  # noqa: WPS433

    found: List[Path] = []
    for level_id in production_mesh_levels_for_cleanup():
        msh = CONV_MESH / level_id / f"{sample_id}.msh"
        for path in (
            msh,
            mesh_manifest_path(msh),
            mesh_audit_path(level_id, sample_id),
            CONV_MESH / level_id / f"{sample_id}_mesh_build_summary.json",
            CONV_MESH / level_id / f"{sample_id}_build.log",
        ):
            if path.is_file():
                found.append(path)
    return found


def load_mesh_manifest(mesh_path: Path) -> Optional[Dict[str, Any]]:
    manifest_path = mesh_manifest_path(mesh_path)
    if not manifest_path.is_file():
        return None
    try:
        doc = json.loads(manifest_path.read_text(encoding="utf-8"))
        return doc if isinstance(doc, dict) else None
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def assert_scout_mesh_shape_gate(
    *,
    mesh_path: Path,
    sample: Mapping[str, Any],
) -> Tuple[bool, str]:
    sample_id = str(sample.get("sample_id") or sample.get("id") or mesh_path.stem)
    meta = extract_run_metadata(sample)
    shape_name = str(meta.get("shape_name") or sample.get("shape_name") or "classic")
    shape_meta = resolve_case_shape_metadata(
        {"shape_name": shape_name, **meta},
        sample_input=sample,
    )
    geometry = extract_geometry_dict(sample)
    ok, reason, manifest = validate_mesh_reuse(
        mesh_path,
        sample_id=sample_id,
        mesh_level="L_scout_coarse",
        shape_name=shape_name,
        geometry_shape_type=shape_meta["geometry_shape_type"],
        gmsh_shape_type=shape_meta["gmsh_shape_type"],
        geometry=geometry,
        lhs_path=str(meta.get("lhs_path") or sample.get("lhs_path") or ""),
    )
    manifest_shape = str((manifest or {}).get("geometry_shape_type") or "missing")
    line = (
        f"SCOUT_MESH_SHAPE_ASSERT shape={shape_name} "
        f"gmsh_shape_type={shape_meta['gmsh_shape_type']} "
        f"mesh_manifest_shape={manifest_shape} status={'PASS' if ok else 'FAIL'}"
    )
    if not ok:
        existing = manifest_shape
        rejected = format_mesh_reuse_rejected(
            reason=reason,
            existing_shape=existing,
            expected_shape=shape_meta["geometry_shape_type"],
            mesh_path=mesh_path,
        )
        return False, f"{line} {rejected}"
    return True, line


def assert_scout_lprod_shape_consistency(
    *,
    scout_mesh_path: Path,
    lprod_mesh_path: Path,
) -> Tuple[bool, str]:
    scout_manifest = load_mesh_manifest(scout_mesh_path)
    lprod_manifest = load_mesh_manifest(lprod_mesh_path)
    if not scout_manifest or not lprod_manifest:
        return False, "SCOUT_LPROD_SHAPE_CONSISTENCY_FAIL missing_mesh_manifest"

    scout_shape = str(scout_manifest.get("geometry_shape_type") or "")
    lprod_shape = str(lprod_manifest.get("geometry_shape_type") or "")
    shape_name = str(lprod_manifest.get("shape_name") or scout_manifest.get("shape_name") or "")
    if scout_shape != lprod_shape:
        return (
            False,
            f"SCOUT_LPROD_SHAPE_CONSISTENCY_FAIL shape={shape_name} scout={scout_shape} lprod={lprod_shape}",
        )

    geom_ok, geom_errors = _geometry_matches(
        scout_manifest.get("geometry_parameters") or {},
        lprod_manifest.get("geometry_parameters") or {},
    )
    if not geom_ok:
        return False, f"SCOUT_LPROD_SHAPE_CONSISTENCY_FAIL geometry:{';'.join(geom_errors)}"

    return True, f"SCOUT_LPROD_SHAPE_CONSISTENCY_PASS shape={shape_name} scout={scout_shape} lprod={lprod_shape}"


def audit_stale_mesh_cache_for_sample(
    repo_root: Path,
    *,
    sample_id: str,
    expected_shape_name: str,
    expected_geometry_shape_type: str,
) -> Tuple[int, List[str], int]:
    stale_paths: List[str] = []
    mismatch_count = 0
    for path in collect_global_mesh_cache_paths_resolved(repo_root, sample_id):
        rel = str(path)
        if path.suffix == ".msh":
            manifest = load_mesh_manifest(path)
            if manifest is None:
                stale_paths.append(rel)
                continue
            if str(manifest.get("geometry_shape_type") or "") != expected_geometry_shape_type:
                mismatch_count += 1
                stale_paths.append(rel)
            elif str(manifest.get("shape_name") or "") != expected_shape_name:
                mismatch_count += 1
                stale_paths.append(rel)
        elif path.name.endswith(".mesh_manifest.json") and not (path.with_suffix(".msh").is_file()):
            stale_paths.append(rel)
    return len(stale_paths), stale_paths, mismatch_count
