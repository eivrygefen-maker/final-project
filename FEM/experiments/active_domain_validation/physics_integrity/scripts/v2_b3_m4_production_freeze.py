#!/usr/bin/env python3
"""Production freeze + acceptance finalization (no workers / Stage A)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from v2_b3_m4_freeze_first_e2e_run import (  # noqa: E402
    AGG_STATUS_PASS,
    build_freeze_payload,
    freeze_outputs_present,
    resolve_freeze_config,
    write_freeze_outputs,
    _validate_milestone,
)
from v2_b3_m4_production_contracts import (  # noqa: E402
    DATASET_VERSION,
    PRODUCTION_MIC_METHOD,
    evaluate_production_acceptance,
)
from v2_b3_m4_worker_run_lib import load_json, rel, utc_now  # noqa: E402
from v2_b3_petsc_util import write_json_atomic  # noqa: E402

PRODUCTION_FREEZE_MANIFEST = "freeze_manifest.json"
PRODUCTION_FREEZE_SCHEMA = "m4_production_freeze_v1"
TERMINAL_PRODUCTION_COMPLETED = "COMPLETED"
FREEZE_DIR_NAME = "freeze"


def production_freeze_manifest_path(run_root: Path) -> Path:
    return run_root / FREEZE_DIR_NAME / PRODUCTION_FREEZE_MANIFEST


def load_sample_input(run_root: Path) -> Dict[str, Any]:
    path = run_root / "sample" / "sample_input.json"
    if path.is_file():
        try:
            doc = load_json(path)
            if isinstance(doc, dict):
                return doc
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    return {"sample_id": run_root.parent.parent.name}


def production_freeze_complete(run_root: Path) -> bool:
    manifest_path = production_freeze_manifest_path(run_root)
    if not manifest_path.is_file():
        return False
    try:
        doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if not bool(doc.get("production_acceptance_pass")):
        return False
    pipeline = run_root / "pipeline_run_manifest.json"
    if not pipeline.is_file():
        return False
    try:
        pm = json.loads(pipeline.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return str(pm.get("terminal_status") or "") == TERMINAL_PRODUCTION_COMPLETED


def assess_production_completion(run_root: Path) -> Dict[str, Any]:
    failures: list[str] = []
    manifest_path = production_freeze_manifest_path(run_root)
    if not manifest_path.is_file():
        failures.append("freeze/freeze_manifest.json does not exist")
    pipeline_path = run_root / "pipeline_run_manifest.json"
    terminal = ""
    if pipeline_path.is_file():
        try:
            terminal = str(json.loads(pipeline_path.read_text(encoding="utf-8")).get("terminal_status") or "")
        except (OSError, ValueError, json.JSONDecodeError):
            failures.append("pipeline_run_manifest.json unreadable")
    if terminal != TERMINAL_PRODUCTION_COMPLETED:
        failures.append(f"terminal_status={terminal or 'missing'}")
    if manifest_path.is_file():
        try:
            fm = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not bool(fm.get("production_acceptance_pass")):
                failures.extend(list(fm.get("production_acceptance_failures") or ["production_acceptance_pass!=true"]))
        except (OSError, ValueError, json.JSONDecodeError):
            failures.append("freeze/freeze_manifest.json unreadable")
    return {
        "complete": len(failures) == 0,
        "failures": failures,
        "terminal_status": terminal or None,
        "freeze_manifest": str(manifest_path) if manifest_path.is_file() else None,
    }


def promote_production_completed(
    run_root: Path,
    *,
    acceptance: Mapping[str, Any],
) -> None:
    manifest_path = run_root / "pipeline_run_manifest.json"
    manifest: Dict[str, Any] = {}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            manifest = {}
    manifest["terminal_status"] = TERMINAL_PRODUCTION_COMPLETED
    manifest["updated_utc"] = utc_now()
    manifest["production_acceptance_pass"] = bool(acceptance.get("acceptance_pass"))
    manifest["production_acceptance_failures"] = list(acceptance.get("failures") or [])
    manifest["mic_output_method"] = PRODUCTION_MIC_METHOD
    manifest["dataset_version"] = acceptance.get("dataset_version") or DATASET_VERSION
    for key in ("mesh_profile", "mesh_level_id", "effective_controls_m"):
        if acceptance.get(key) is not None:
            manifest[key] = acceptance.get(key)
    stages = manifest.setdefault("stages", {})
    for key in ("stage5_workers", "stage6_aggregate"):
        st = stages.setdefault(key, {})
        if str(st.get("status")) not in ("PASS",):
            st["status"] = "PASS"
        st["updated_utc"] = utc_now()
    agg_st = stages.setdefault("stage6_aggregate", {})
    agg_st["aggregation_status"] = AGG_STATUS_PASS
    freeze_st = stages.setdefault("stage6_freeze", {})
    freeze_st["status"] = "PASS"
    freeze_st.pop("warning", None)
    freeze_st["updated_utc"] = utc_now()
    freeze_st["artifact_paths"] = [
        f"{FREEZE_DIR_NAME}/{PRODUCTION_FREEZE_MANIFEST}",
    ]
    write_json_atomic(manifest_path, manifest)


def write_production_freeze_manifest(
    *,
    repo_root: Path,
    run_root: Path,
    acceptance: Mapping[str, Any],
    sample_input: Mapping[str, Any],
) -> Path:
    freeze_dir = run_root / FREEZE_DIR_NAME
    freeze_dir.mkdir(parents=True, exist_ok=True)
    agg_path = run_root / "aggregation" / "aggregation_result.json"
    agg: Dict[str, Any] = {}
    if agg_path.is_file():
        try:
            agg = json.loads(agg_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            agg = {}
    built_path = run_root / "lprod" / "checkpoint" / "built_metadata.json"
    built: Dict[str, Any] = {}
    if built_path.is_file():
        try:
            built = json.loads(built_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            built = {}
    body: Dict[str, Any] = {
        "schema": PRODUCTION_FREEZE_SCHEMA,
        "generated_utc": utc_now(),
        "sample_id": str(sample_input.get("sample_id") or agg.get("sample_id") or run_root.parent.parent.name),
        "run_id": str(agg.get("run_id") or run_root.name),
        "terminal_status": TERMINAL_PRODUCTION_COMPLETED,
        "aggregation_status": agg.get("status") or AGG_STATUS_PASS,
        "final_aggregation_ready": bool(agg.get("final_aggregation_ready")),
        "dataset_version": built.get("dataset_version") or acceptance.get("dataset_version") or DATASET_VERSION,
        "mesh_profile": acceptance.get("mesh_profile") or built.get("mesh_profile"),
        "mesh_level_id": acceptance.get("mesh_level_id") or built.get("mesh_level_id") or built.get("mesh_level"),
        "effective_controls_m": acceptance.get("effective_controls_m") or built.get("effective_controls_m"),
        "generated_mesh_sha256": built.get("generated_mesh_sha256"),
        "operator_mesh_sha256": built.get("operator_mesh_sha256"),
        "operator_mesh_matches_generated": bool(built.get("operator_mesh_matches_generated")),
        "p_idx_aperture_count": int(acceptance.get("p_idx_aperture_count") or built.get("p_idx_aperture_count") or 0),
        "production_acceptance_pass": bool(acceptance.get("acceptance_pass")),
        "production_acceptance_failures": list(acceptance.get("failures") or []),
        "mic_output_method": PRODUCTION_MIC_METHOD,
        "deduped_mode_count": agg.get("deduped_mode_count"),
        "completed_chunk_count": agg.get("completed_chunk_count"),
        "planned_chunk_count": agg.get("planned_chunk_count"),
        "freeze_dir": rel(freeze_dir, repo_root=repo_root),
    }
    out_path = production_freeze_manifest_path(run_root)
    write_json_atomic(out_path, body)
    return out_path


def write_physics_identity_manifest(
    *,
    repo_root: Path,
    run_root: Path,
    acceptance: Mapping[str, Any],
    sample_input: Mapping[str, Any],
) -> Path:
    from v2_b3_m4_physics_identity_lib import (  # noqa: WPS433
        PHYSICS_IDENTITY_MANIFEST,
        build_physics_identity_manifest,
        validate_physics_identity_manifest,
    )

    manifest = build_physics_identity_manifest(
        run_root=run_root,
        repo_root=repo_root,
        sample_input=sample_input,
        acceptance=acceptance,
    )
    ok, errors = validate_physics_identity_manifest(manifest)
    manifest["manifest_validation_pass"] = ok
    manifest["manifest_validation_errors"] = errors
    out_path = run_root / PHYSICS_IDENTITY_MANIFEST
    write_json_atomic(out_path, manifest)
    return out_path


def replay_production_freeze(
    *,
    repo_root: Path,
    run_root: Path,
    sample_input: Optional[Mapping[str, Any]] = None,
    force: bool = False,
) -> Tuple[int, str]:
    """
    Freeze + production acceptance + terminal promotion from existing aggregation artifacts.
    Does not rerun workers or Stage A.
    """
    run_root = run_root.resolve()
    errors = _validate_milestone(run_root=run_root)
    if errors:
        return 2, "; ".join(errors)

    sample_doc = dict(sample_input or load_sample_input(run_root))
    acceptance = evaluate_production_acceptance(run_root=run_root, sample_input=sample_doc)
    if not acceptance.get("acceptance_pass"):
        return 2, "; ".join(acceptance.get("failures") or ["production_acceptance_failed"])

    if production_freeze_complete(run_root) and not force:
        return 0, "production freeze already complete"

    sample_id = str(sample_doc.get("sample_id") or run_root.parent.parent.name)
    if not freeze_outputs_present(run_root) or force:
        cfg = resolve_freeze_config(sample_id)
        payload = build_freeze_payload(repo_root=repo_root, run_root=run_root, freeze_cfg=cfg)
        try:
            write_freeze_outputs(
                repo_root=repo_root,
                run_root=run_root,
                payload=payload,
                force=force,
                idempotent=not force,
            )
        except FileExistsError as exc:
            if not freeze_outputs_present(run_root):
                return 2, str(exc)

    write_production_freeze_manifest(
        repo_root=repo_root,
        run_root=run_root,
        acceptance=acceptance,
        sample_input=sample_doc,
    )
    identity_path = write_physics_identity_manifest(
        repo_root=repo_root,
        run_root=run_root,
        acceptance=acceptance,
        sample_input=sample_doc,
    )
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if (identity.get("fallback_flags") or {}).get("cross_sample_reuse"):
        return 2, "cross_sample_reuse=true_in_physics_identity_manifest"
    if not bool(identity.get("production_acceptance_pass")):
        return 2, "; ".join(identity.get("production_acceptance_failures") or ["production_acceptance_failed"])
    promote_production_completed(run_root, acceptance=acceptance)
    return 0, "production freeze finalized"
