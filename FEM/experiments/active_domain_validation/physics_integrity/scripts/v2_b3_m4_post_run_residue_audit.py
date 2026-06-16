#!/usr/bin/env python3
"""Strict post-run forbidden residue audit for M4 production samples."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from v2_b3_m4_physics_identity_lib import count_forbidden_heavy_artifacts  # noqa: E402
from v2_b3_m4_mesh_manifest_lib import audit_stale_mesh_cache_for_sample  # noqa: E402
from v2_b3_m4_sample_cleanup_barrier import collect_shared_sample_artifact_paths  # noqa: E402
from v2_b3_petsc_util import write_json_atomic  # noqa: E402

AUDIT_SCHEMA = "m4_post_run_residue_audit_v1"
AUDIT_REL = "cleanup/post_run_residue_audit.json"


def run_post_run_residue_audit(
    repo_root: Path,
    run_root: Path,
    sample_id: str,
    run_id: str,
    *,
    require_zero_forbidden: bool = True,
    write_report: bool = True,
    shape_name: str = "",
    geometry_shape_type: str = "",
) -> Dict[str, Any]:
    run_root = run_root.expanduser().resolve()
    forbidden_count, forbidden_paths = count_forbidden_heavy_artifacts(run_root)
    shared_paths = collect_shared_sample_artifact_paths(
        repo_root=repo_root,
        sample_id=sample_id,
        run_id=run_id,
    )
    shared_present = [str(p) for p in shared_paths if p.exists()]

    stale_mesh_count = 0
    stale_mesh_paths: List[str] = []
    shape_mismatch_count = 0
    if shape_name and geometry_shape_type:
        stale_mesh_count, stale_mesh_paths, shape_mismatch_count = audit_stale_mesh_cache_for_sample(
            repo_root,
            sample_id=sample_id,
            expected_shape_name=shape_name,
            expected_geometry_shape_type=geometry_shape_type,
        )

    report: Dict[str, Any] = {
        "schema": AUDIT_SCHEMA,
        "sample_id": sample_id,
        "run_id": run_id,
        "run_root": str(run_root),
        "audited_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "forbidden_heavy_artifact_count": forbidden_count,
        "forbidden_heavy_artifacts_present": forbidden_paths,
        "shared_sample_artifact_count": len(shared_present),
        "shared_sample_artifacts_present": shared_present,
        "stale_mesh_count": stale_mesh_count,
        "stale_mesh_paths": stale_mesh_paths,
        "shape_mismatch_count": shape_mismatch_count,
        "pass": (
            forbidden_count == 0
            and len(shared_present) == 0
            and stale_mesh_count == 0
            and shape_mismatch_count == 0
        ),
    }
    errors: List[str] = []
    if require_zero_forbidden and forbidden_count > 0:
        errors.append(f"forbidden_heavy_artifacts:{forbidden_paths}")
    if shared_present:
        errors.append(f"shared_sample_artifacts:{shared_present}")
    if stale_mesh_count > 0:
        errors.append(f"stale_mesh:{stale_mesh_paths}")
    if shape_mismatch_count > 0:
        errors.append(f"shape_mismatch_count={shape_mismatch_count}")
    report["errors"] = errors

    if write_report:
        out_path = run_root / AUDIT_REL
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(out_path, report)

    return report


def format_post_run_residue_audit_line(report: Dict[str, Any]) -> str:
    sample = report.get("sample_id") or "unknown"
    forbidden = int(report.get("forbidden_heavy_artifact_count") or 0)
    heavy = report.get("forbidden_heavy_artifacts_present") or []
    stale_mesh = int(report.get("stale_mesh_count") or 0)
    shape_mismatch = int(report.get("shape_mismatch_count") or 0)
    return (
        f"POST_RUN_RESIDUE_AUDIT sample={sample} forbidden={forbidden} "
        f"stale_mesh={stale_mesh} shape_mismatch={shape_mismatch} "
        f"heavy_paths_present={heavy}"
    )


def assert_post_run_residue_clean(
    repo_root: Path,
    run_root: Path,
    sample_id: str,
    run_id: str,
) -> Tuple[bool, str]:
    report = run_post_run_residue_audit(
        repo_root=repo_root,
        run_root=run_root,
        sample_id=sample_id,
        run_id=run_id,
    )
    line = format_post_run_residue_audit_line(report)
    if report.get("pass"):
        return True, line
    errors = report.get("errors") or []
    return False, f"{line} errors={errors}"
