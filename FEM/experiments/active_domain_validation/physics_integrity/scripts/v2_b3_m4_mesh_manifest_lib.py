#!/usr/bin/env python3
"""Mesh sidecar manifests and shape-aware reuse validation for v2_mesh_convergence caches."""
from __future__ import annotations

import json
import math
import shutil
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
    manifest_path: Optional[Path] = None,
) -> str:
    line = (
        f"{MESH_REUSE_REJECTED} reason={reason} existing={existing_shape} "
        f"expected={expected_shape} mesh={mesh_path}"
    )
    if reason == "missing_manifest" and manifest_path is not None:
        line += f" expected_manifest={manifest_path}"
    return line


def _mesh_install_sidecar_sources(src_msh: Path, sample_id: str) -> List[Path]:
    parent = src_msh.parent
    stem = src_msh.stem
    candidates = (
        parent / f"{sample_id}_mesh_build_summary.json",
        parent / f"{stem}_mesh_build_summary.json",
        mesh_audit_path_for(src_msh),
        parent / f"{sample_id}_mesh_audit.json",
        parent / f"{sample_id}_build.log",
    )
    seen: set[Path] = set()
    found: List[Path] = []
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen or not path.is_file():
            continue
        seen.add(resolved)
        found.append(path)
    return found


def install_mesh_with_sidecars(*, src_msh: Path, dst_msh: Path, sample_id: str) -> Dict[str, Any]:
    src_msh = src_msh.expanduser().resolve()
    dst_msh = dst_msh.expanduser().resolve()
    if not src_msh.is_file():
        raise FileNotFoundError(f"mesh source missing: {src_msh}")

    src_manifest = mesh_manifest_path(src_msh)
    if not src_manifest.is_file():
        raise FileNotFoundError(
            f"mesh manifest missing at source: {src_manifest} (mesh={src_msh})"
        )

    dst_msh.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_msh, dst_msh)

    manifest = json.loads(src_manifest.read_text(encoding="utf-8"))
    manifest["canonical_mesh_path"] = str(src_msh)
    manifest["run_dir_mesh_path"] = str(dst_msh)
    manifest["installed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    dst_manifest = mesh_manifest_path(dst_msh)
    write_mesh_manifest(dst_manifest, manifest)

    copied = ["mesh", "mesh_manifest"]
    for sidecar_src in _mesh_install_sidecar_sources(src_msh, sample_id):
        sidecar_dst = dst_msh.parent / sidecar_src.name
        shutil.copy2(sidecar_src, sidecar_dst)
        copied.append(sidecar_src.name)

    return {
        "src_msh": str(src_msh),
        "dst_msh": str(dst_msh),
        "dst_manifest": str(dst_manifest),
        "copied": copied,
    }


def resolve_mesh_validation_context(
    mesh_path: Path,
    *,
    sample_id: str,
    mesh_level: str = "L_scout_coarse",
) -> Tuple[Path, Path, str, List[str]]:
    """Resolve mesh + manifest for validation; repair run-dir manifest from global when possible."""
    from v2_mesh_convergence_common import CONV_MESH  # noqa: WPS433

    diag: List[str] = []
    run_mesh = mesh_path.expanduser().resolve()
    global_mesh = (CONV_MESH / mesh_level / f"{sample_id}.msh").resolve()
    run_manifest = mesh_manifest_path(run_mesh)
    global_manifest = mesh_manifest_path(global_mesh)

    def _log_exists(label: str, path: Path) -> None:
        diag.append(f"SCOUT_MESH_{label} path={path} exists={path.is_file()}")

    _log_exists("EXISTS", run_mesh)
    _log_exists("MANIFEST_EXISTS", run_manifest)

    if run_mesh.is_file() and run_manifest.is_file():
        diag.append("SCOUT_MESH_VALIDATED_SOURCE run_dir")
        return run_mesh, run_manifest, "run_dir", diag

    if run_mesh.is_file() and global_mesh.is_file() and global_manifest.is_file():
        manifest = json.loads(global_manifest.read_text(encoding="utf-8"))
        manifest["canonical_mesh_path"] = str(global_mesh)
        manifest["run_dir_mesh_path"] = str(run_mesh)
        write_mesh_manifest(run_manifest, manifest)
        diag.append(f"SCOUT_MESH_MANIFEST_EXISTS path={run_manifest} exists=True")
        diag.append("SCOUT_MESH_VALIDATED_SOURCE run_dir_repaired_from_global")
        return run_mesh, run_manifest, "run_dir_repaired_from_global", diag

    if not run_mesh.is_file() and global_mesh.is_file():
        _log_exists("EXISTS", global_mesh)
        _log_exists("MANIFEST_EXISTS", global_manifest)
        if global_manifest.is_file():
            diag.append("SCOUT_MESH_VALIDATED_SOURCE global")
            return global_mesh, global_manifest, "global", diag

    target = run_mesh if run_mesh.is_file() else global_mesh
    expected_manifest = mesh_manifest_path(target) if target.is_file() else run_manifest
    diag.append("SCOUT_MESH_VALIDATED_SOURCE missing")
    return target, expected_manifest, "missing", diag


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
    validated_mesh, expected_manifest, _source, diag = resolve_mesh_validation_context(
        mesh_path,
        sample_id=sample_id,
        mesh_level="L_scout_coarse",
    )
    ok, reason, manifest = validate_mesh_reuse(
        validated_mesh,
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
            mesh_path=validated_mesh,
            manifest_path=expected_manifest if reason == "missing_manifest" else None,
        )
        return False, "\n".join(diag + [line, rejected])
    return True, "\n".join(diag + [line])


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
