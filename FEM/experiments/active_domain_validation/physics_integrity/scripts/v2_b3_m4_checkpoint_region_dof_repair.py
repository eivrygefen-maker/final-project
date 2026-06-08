#!/usr/bin/env python3
"""Repair region-DOF / aperture mask export on an existing L_prod checkpoint (no A/M rebuild)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_aperture_pressure_mask import attach_aperture_mask_to_region_dof_build  # noqa: E402
from v2_b3_checkpoint_pipeline_lib import PIPELINE_EXPORT_MANIFEST  # noqa: E402
from v2_b3_m4_production_contracts import (  # noqa: E402
    DATASET_VERSION,
    geometry_from_core_config,
    resolve_operator_mesh_file,
    validate_post_export_region_dof_contract,
)
from v2_b3_m4_run_status_repair import (  # noqa: E402
    STALE_RUNNING_REPAIR_REASON,
    detect_active_worker_lock,
    promote_checkpoint_ready_terminal,
)
from v2_b3_m4_worker_run_lib import detect_repo_root, load_json  # noqa: E402
from v2_b3_petsc_util import write_json_atomic  # noqa: E402
from v2_b3_rich_modal_lib import SYNTHESIS_METADATA_JSON, load_region_dof_bundle  # noqa: E402
from v2_b3_synthesis_export import (  # noqa: E402
    REGION_DOF_STATUS_PASS,
    capture_region_dof_build_from_mesh,
    export_region_dof_indices_from_operator_build,
    region_dof_status_is_pass,
    write_stage_a_synthesis_artifacts,
)


def _load_json_optional(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        doc = load_json(path)
        return doc if isinstance(doc, dict) else {}
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def assess_checkpoint_region_dof_repair(run_root: Path, *, repo_root: Path) -> Dict[str, Any]:
    run_root = run_root.expanduser().resolve()
    ckpt = run_root / "lprod" / "checkpoint"
    core_config = run_root / "lprod" / "resolved_core_config.json"
    built_path = ckpt / "built_metadata.json"
    export_manifest = ckpt / PIPELINE_EXPORT_MANIFEST
    active_lock, lock_detail = detect_active_worker_lock(run_root)

    out: Dict[str, Any] = {
        "run_root": str(run_root),
        "checkpoint_dir": str(ckpt),
        "worker_active": active_lock,
        "worker_lock_detail": lock_detail,
        "eligible": False,
        "reasons": [],
    }
    if active_lock:
        out["reasons"].append(f"worker_active:{lock_detail}")
        return out
    for name in ("A_active_csr.npz", "M_active_csr.npz", "built_metadata.json"):
        if not (ckpt / name).is_file():
            out["reasons"].append(f"missing:{name}")
    if not core_config.is_file():
        out["reasons"].append("missing:lprod/resolved_core_config.json")
    if not export_manifest.is_file():
        out["reasons"].append("missing:checkpoint_export_manifest.json")
    if out["reasons"]:
        return out

    built_meta = _load_json_optional(built_path)
    contract_errors = validate_post_export_region_dof_contract(ckpt, core_config_path=core_config)
    out["region_dof_contract_errors"] = contract_errors
    if not contract_errors:
        out["eligible"] = False
        out["reasons"].append("region_dof_contract_already_pass")
        return out

    out["eligible"] = True
    out["mesh_file"] = str(
        resolve_operator_mesh_file(
            operator_mesh_arg=None,
            core_config_path=core_config,
            repo_root=repo_root,
        )
    )
    return out


def repair_checkpoint_region_dof_export(
    *,
    repo_root: Path,
    run_root: Path,
    force: bool = False,
) -> Tuple[int, str]:
    """
    Replay region-DOF + aperture mask export from existing checkpoint A/M + mesh.
    Does not rebuild Scout, mesh, or operator matrices.
    """
    run_root = run_root.expanduser().resolve()
    repo_root = repo_root.resolve()
    assessment = assess_checkpoint_region_dof_repair(run_root, repo_root=repo_root)
    if not assessment.get("eligible") and not force:
        return 2, ";".join(assessment.get("reasons") or ["not_eligible"])

    ckpt = run_root / "lprod" / "checkpoint"
    core_config = run_root / "lprod" / "resolved_core_config.json"
    built_path = ckpt / "built_metadata.json"
    built_meta = _load_json_optional(built_path)
    mesh_file = resolve_operator_mesh_file(
        operator_mesh_arg=None,
        core_config_path=core_config,
        repo_root=repo_root,
    )
    geom = geometry_from_core_config(core_config)
    if not geom:
        return 2, "geometry_missing_from_resolved_core_config"

    region_dof_build, err = capture_region_dof_build_from_mesh(
        mesh_file=mesh_file,
        built_meta=built_meta,
    )
    if region_dof_build is None:
        return 2, str(err or "capture_region_dof_build_failed")

    region_dof_build = attach_aperture_mask_to_region_dof_build(
        region_dof_build,
        mesh_file=mesh_file,
        geometry=geom,
        built_meta=built_meta,
        core_config_path=core_config,
    )
    status, export_err = export_region_dof_indices_from_operator_build(
        ckpt,
        region_dof_build=region_dof_build,
    )
    if not region_dof_status_is_pass(status):
        return 2, str(export_err or f"export_status={status}")

    p_count = int(region_dof_build.get("p_idx_aperture_count") or 0)
    built_meta = dict(built_meta)
    built_meta["p_idx_aperture_count"] = p_count
    built_meta["aperture_selection_method"] = str(region_dof_build.get("aperture_selection_method") or "")
    built_meta["dataset_version"] = built_meta.get("dataset_version") or DATASET_VERSION
    write_json_atomic(built_path, built_meta)

    built_stub = {"region_dof_build": region_dof_build, "n_u_b3": built_meta.get("n_u_b3"), "p_idx": built_meta.get("p_idx")}
    synthesis_export = write_stage_a_synthesis_artifacts(
        ckpt,
        built=built_stub,
        built_meta=built_meta,
        mesh_level="L_prod",
        compose_backend=str(built_meta.get("compose_backend") or "csr_bulk"),
        region_dofs_mode="best_effort",
        core_config_path=core_config,
    )

    export_manifest_path = ckpt / PIPELINE_EXPORT_MANIFEST
    manifest = _load_json_optional(export_manifest_path)
    manifest["synthesis_region_dofs_mode"] = "best_effort"
    manifest["synthesis_export"] = synthesis_export
    manifest["region_dof_repair"] = {
        "status": "PASS",
        "p_idx_aperture_count": p_count,
        "aperture_selection_method": region_dof_build.get("aperture_selection_method"),
    }
    write_json_atomic(export_manifest_path, manifest)

    contract_errors = validate_post_export_region_dof_contract(ckpt, core_config_path=core_config)
    if contract_errors:
        return 2, ";".join(contract_errors)

    load_region_dof_bundle(ckpt, built_meta, validate_aperture=True)

    manifest_path = run_root / "pipeline_run_manifest.json"
    terminal = str(_load_json_optional(manifest_path).get("terminal_status") or "")
    if terminal == "RUNNING":
        repair = promote_checkpoint_ready_terminal(
            run_root,
            repair_reason=STALE_RUNNING_REPAIR_REASON,
            previous_status=terminal,
        )
        if repair.get("status") != "PASS":
            return 2, f"terminal_repair_failed:{repair}"

    return 0, (
        f"region_dof_repaired p_idx_aperture_count={p_count} "
        f"method={region_dof_build.get('aperture_selection_method')}"
    )
