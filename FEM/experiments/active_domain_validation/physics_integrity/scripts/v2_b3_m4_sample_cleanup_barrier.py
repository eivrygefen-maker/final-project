#!/usr/bin/env python3
"""Mandatory per-sample cleanup barrier for strict M4 production batches."""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from v2_b3_m4_lhs_pool_bridge import specs_generated_dir  # noqa: E402
from v2_b3_m4_physics_identity_lib import (  # noqa: E402
    count_forbidden_heavy_artifacts,
    verify_post_compaction_contract,
)
from v2_b3_m4_worker_run_lib import utc_now  # noqa: E402
from v2_b3_petsc_util import write_json_atomic  # noqa: E402

try:
    from compact_completed_m4_runs import (  # noqa: E402
        PRODUCTION_PASS_OUTCOMES,
        _collect_archivable_paths,
        _delete_archived_paths,
        compact_one_completed_run,
    )
except ImportError:
    PRODUCTION_PASS_OUTCOMES = frozenset({"pass", "reused_complete", "pass_freeze_warning"})
    compact_one_completed_run = None  # type: ignore[misc, assignment]
    _collect_archivable_paths = None  # type: ignore[misc, assignment]
    _delete_archived_paths = None  # type: ignore[misc, assignment]

TOOL_VERSION = "m4_sample_cleanup_barrier_v1"
BARRIER_MANIFEST_SCHEMA = "m4_sample_cleanup_barrier_v1"
FAILURE_REPORT_REL = "cleanup/sample_cleanup_failure_report.json"
BARRIER_MANIFEST_REL = "cleanup/sample_cleanup_barrier.json"
FAILURE_RETENTION_REL = "cleanup/sample_failure_retention.json"
FAILURE_DIAGNOSTIC_REL = "logs/sample_failure_diagnostic.log"

from v2_b3_m4_mesh_profile_lib import (  # noqa: E402
    DURABLE_VALIDATION_INPUT_REL,
    VALIDATION_INPUT_PACKAGE_REL,
    preserve_target_plan_before_cleanup,
    production_mesh_levels_for_cleanup,
)
from v2_b3_m4_shared_export import APPROVED_SHARED_PLOT_NAMES  # noqa: E402
from v2_b3_m4_mesh_profile_provenance_lib import (  # noqa: E402
    preserve_comparison_provenance_before_cleanup,
)
from v2_b3_m4_scout_discovery_diagnostics import (  # noqa: E402
    preserve_scout_discovery_failure_diagnostics,
)

MESH_LEVELS: Tuple[str, ...] = production_mesh_levels_for_cleanup()

DURABLE_REQUIRED_REL: Tuple[str, ...] = (
    "aggregation/modes_catalog.jsonl",
    "aggregation/modes_catalog_deduped.jsonl",
    "aggregation/mode_provenance.jsonl",
    "aggregation/aggregation_result.json",
    "aggregation/modes_summary.json",
    "freeze/freeze_manifest.json",
    "freeze/physics_identity_manifest.json",
    "sample/sample_input.json",
    "pipeline_run_manifest.json",
    "compaction/compaction_manifest.json",
)

DURABLE_RUNTIME_WARNINGS_REL: Tuple[str, ...] = (
    "aggregation/runtime_summary.json",
    "aggregation/warnings_and_failures.json",
)

FREEZE_MANIFEST_FALLBACKS: Tuple[str, ...] = (
    "freeze/sample_e2e_run_manifest.json",
    "freeze/first_end_to_end_run_manifest.json",
)


def physics_root(repo_root: Path) -> Path:
    return repo_root / "FEM/experiments/active_domain_validation/physics_integrity"


def pipeline_runs_root(repo_root: Path) -> Path:
    return physics_root(repo_root) / "pipeline_runs"


def mesh_convergence_root(repo_root: Path) -> Path:
    return physics_root(repo_root) / "v2_mesh_convergence"


def mesh_build_config_dir(repo_root: Path) -> Path:
    return physics_root(repo_root) / "scripts" / "configs" / "v2_mesh_convergence_build"


def _mesh_sidecar_paths(mesh_dir: Path, level_id: str, sample_id: str) -> List[Path]:
    base = mesh_dir / level_id
    names = (
        f"{sample_id}.msh",
        f"{sample_id}_mesh_build_summary.json",
        f"{sample_id}_mesh_audit.json",
        f"{sample_id}_build.log",
    )
    return [base / name for name in names]


def collect_shared_sample_artifact_paths(
    *,
    repo_root: Path,
    sample_id: str,
    run_id: str,
) -> List[Path]:
    """Sample-specific artifacts outside run_root that must not survive the barrier."""
    found: List[Path] = []
    mesh_root = mesh_convergence_root(repo_root) / "mesh"
    for level_id in MESH_LEVELS:
        found.extend(_mesh_sidecar_paths(mesh_root, level_id, sample_id))

    cfg_dir = mesh_build_config_dir(repo_root)
    for level_id in MESH_LEVELS:
        cfg = cfg_dir / f"{level_id}_{sample_id}.json"
        if cfg.exists():
            found.append(cfg)

    overlay_dir = pipeline_runs_root(repo_root) / "config_overlays" / sample_id
    if overlay_dir.exists():
        found.append(overlay_dir)

    generated_spec = specs_generated_dir(repo_root) / f"{run_id}.json"
    if generated_spec.is_file():
        found.append(generated_spec)

    lock_globs = (
        pipeline_runs_root(repo_root) / "logs" / f"*{sample_id}*",
        pipeline_runs_root(repo_root) / "logs" / f"*{run_id}*",
    )
    for pattern in lock_globs:
        for path in pattern.parent.glob(pattern.name):
            if path.is_file() and (
                path.suffix in {".lock", ".lock.json"} or path.name.endswith(".lock.json")
            ):
                found.append(path)

    deduped: List[Path] = []
    seen: Set[str] = set()
    for path in found:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return sorted(deduped, key=lambda p: str(p))


def collect_run_tree_lock_paths(run_root: Path) -> List[Path]:
    locks: List[Path] = []
    if not run_root.is_dir():
        return locks
    for path in run_root.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        if name.endswith(".lock") or name.endswith(".lock.json") or "lock" in name and path.suffix == ".json":
            if "lock" in name:
                locks.append(path)
    return sorted(locks, key=lambda p: str(p))


def _aggregation_plots_present(run_root: Path) -> bool:
    agg = run_root / "aggregation"
    if not agg.is_dir():
        return False
    return any((agg / name).is_file() for name in APPROVED_SHARED_PLOT_NAMES)


def _freeze_manifest_present(run_root: Path) -> bool:
    if (run_root / "freeze" / "freeze_manifest.json").is_file():
        return True
    return any((run_root / rel_path).is_file() for rel_path in FREEZE_MANIFEST_FALLBACKS)


def verify_success_durable_outputs(
    run_root: Path,
    *,
    require_compaction_manifest: bool = True,
) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    for rel_path in DURABLE_REQUIRED_REL:
        if rel_path == "compaction/compaction_manifest.json" and not require_compaction_manifest:
            continue
        if rel_path == "freeze/freeze_manifest.json":
            if not _freeze_manifest_present(run_root):
                errors.append(f"missing:{rel_path}")
            continue
        if not (run_root / rel_path).is_file():
            errors.append(f"missing:{rel_path}")

    runtime_or_warnings = [
        rel_path
        for rel_path in DURABLE_RUNTIME_WARNINGS_REL
        if (run_root / rel_path).is_file()
    ]
    if not runtime_or_warnings:
        errors.append("missing:aggregation/runtime_or_warnings_summary")

    if not _aggregation_plots_present(run_root):
        errors.append("missing:aggregation_plots")

    return len(errors) == 0, errors


def load_cleanup_barrier_manifest(run_root: Path) -> Optional[Dict[str, Any]]:
    path = run_root / BARRIER_MANIFEST_REL
    if not path.is_file():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return doc if isinstance(doc, dict) else None


def require_cleanup_barrier_passed_for_validation(
    *,
    repo_root: Path,
    run_root: Path,
    label: str,
) -> Tuple[bool, Dict[str, Any], List[str]]:
    """
    Hard precondition for mesh-profile validation / comparison.
    Both forbidden_heavy_artifact_count and shared_sample_artifact_count must be 0.
    """
    errors: List[str] = []
    run_root = run_root.resolve()
    sample_id = run_root.parent.parent.name
    run_id = run_root.name

    meta: Dict[str, Any] = {
        "label": label,
        "sample_id": sample_id,
        "run_id": run_id,
        "run_root": str(run_root),
    }

    failure_report = run_root / FAILURE_REPORT_REL
    if failure_report.is_file():
        errors.append(f"{label}:cleanup_failure_report_present")
        meta["cleanup_failure_report"] = str(failure_report)

    barrier = load_cleanup_barrier_manifest(run_root)
    if barrier is None:
        errors.append(f"{label}:missing_cleanup_barrier_manifest")
        meta["barrier_manifest_present"] = False
        return False, meta, errors

    meta["barrier_manifest_present"] = True
    meta["barrier_status"] = barrier.get("status")
    if str(barrier.get("status")) != "completed":
        errors.append(f"{label}:cleanup_barrier_status={barrier.get('status')!r}")

    sample_success = bool(barrier.get("sample_success"))
    meta["sample_success"] = sample_success

    live = verify_cleanup_barrier(
        repo_root=repo_root,
        run_root=run_root,
        sample_id=sample_id,
        run_id=run_id,
        sample_success=sample_success,
    )
    forbidden = int(live.get("forbidden_heavy_artifact_count") or 0)
    shared = int(live.get("shared_sample_artifact_count") or 0)
    meta.update(
        {
            "forbidden_heavy_artifact_count": forbidden,
            "shared_sample_artifact_count": shared,
            "verification_pass": bool(live.get("pass")),
            "live_verification_errors": list(live.get("errors") or []),
        }
    )

    if forbidden != 0:
        errors.append(
            f"{label}:forbidden_heavy_artifact_count={forbidden} "
            f"({live.get('forbidden_heavy_artifacts_present')})"
        )
    if shared != 0:
        errors.append(
            f"{label}:shared_sample_artifact_count={shared} "
            f"({live.get('shared_sample_artifacts_present')})"
        )
    if not live.get("pass"):
        errors.append(f"{label}:cleanup_barrier_verification_failed")

    ok = len(errors) == 0
    meta["cleanup_barrier_passed"] = ok
    return ok, meta, errors


def verify_cleanup_barrier(
    *,
    repo_root: Path,
    run_root: Path,
    sample_id: str,
    run_id: str,
    sample_success: bool,
) -> Dict[str, Any]:
    """Blocking post-cleanup verification."""
    run_root = run_root.resolve()
    forbidden_count, forbidden_paths = count_forbidden_heavy_artifacts(run_root)
    shared_paths = [
        p
        for p in collect_shared_sample_artifact_paths(
            repo_root=repo_root,
            sample_id=sample_id,
            run_id=run_id,
        )
        if p.exists()
    ]
    shared_count = len(shared_paths)

    report: Dict[str, Any] = {
        "schema": BARRIER_MANIFEST_SCHEMA,
        "tool_version": TOOL_VERSION,
        "generated_utc": utc_now(),
        "sample_id": sample_id,
        "run_id": run_id,
        "run_root": str(run_root),
        "sample_success": sample_success,
        "forbidden_heavy_artifact_count": forbidden_count,
        "forbidden_heavy_artifacts_present": forbidden_paths,
        "shared_sample_artifact_count": shared_count,
        "shared_sample_artifacts_present": [str(p) for p in shared_paths],
        "errors": [],
        "pass": False,
    }

    for rel in DURABLE_VALIDATION_INPUT_REL:
        if (run_root / rel).is_file():
            report.setdefault("durable_validation_inputs_present", []).append(rel)

    if sample_success:
        durable_ok, durable_errors = verify_success_durable_outputs(run_root)
        report["durable_outputs_ok"] = durable_ok
        if not durable_ok:
            report["errors"].extend(durable_errors)
        compaction_report = verify_post_compaction_contract(run_root)
        report["compaction_verify"] = compaction_report
        if not compaction_report.get("pass"):
            report["errors"].extend(compaction_report.get("errors") or [])
    else:
        retention = run_root / FAILURE_RETENTION_REL
        diagnostic = run_root / FAILURE_DIAGNOSTIC_REL
        if not retention.is_file():
            report["errors"].append(f"missing:{FAILURE_RETENTION_REL}")
        if not diagnostic.is_file():
            report["errors"].append(f"missing:{FAILURE_DIAGNOSTIC_REL}")

    if forbidden_count > 0:
        report["errors"].append(f"forbidden_heavy_artifacts:{forbidden_paths}")
    if shared_count > 0:
        report["errors"].append(
            f"shared_sample_artifacts:{[str(p) for p in shared_paths]}"
        )

    report["pass"] = len(report["errors"]) == 0
    return report


def _delete_paths(paths: Sequence[Path]) -> Tuple[List[str], List[str]]:
    deleted: List[str] = []
    errors: List[str] = []
    for path in sorted(paths, key=lambda p: len(str(p)), reverse=True):
        if not path.exists():
            continue
        try:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            elif path.is_file() and not path.is_symlink():
                path.unlink()
            else:
                continue
            deleted.append(str(path))
        except OSError as exc:
            errors.append(f"{path}:{exc}")
    return deleted, errors


def _write_failure_retention(
    *,
    run_root: Path,
    sample_id: str,
    run_id: str,
    row: Mapping[str, Any],
) -> None:
    payload = {
        "schema": "m4_sample_failure_retention_v1",
        "generated_utc": utc_now(),
        "sample_id": sample_id,
        "run_id": run_id,
        "outcome": row.get("outcome"),
        "error_message": row.get("error_message"),
        "return_code": row.get("return_code"),
        "aggregation_status": row.get("aggregation_status"),
        "terminal_status": row.get("terminal_status"),
    }
    write_json_atomic(run_root / FAILURE_RETENTION_REL, payload)
    diagnostic_lines = [
        f"sample_id={sample_id}",
        f"run_id={run_id}",
        f"outcome={row.get('outcome')}",
        f"error_message={row.get('error_message')}",
        f"return_code={row.get('return_code')}",
        f"aggregation_status={row.get('aggregation_status')}",
    ]
    diag_path = run_root / FAILURE_DIAGNOSTIC_REL
    diag_path.parent.mkdir(parents=True, exist_ok=True)
    diag_path.write_text("\n".join(diagnostic_lines) + "\n", encoding="utf-8")


def _delete_failed_run_heavy(
    *,
    repo_root: Path,
    run_root: Path,
    sample_id: str,
    run_id: str,
) -> Tuple[List[str], List[str]]:
    if _collect_archivable_paths is None or _delete_archived_paths is None:
        return [], ["compact_module_unavailable"]

    preserve_scout_discovery_failure_diagnostics(
        run_root,
        reason="failed_sample_heavy_cleanup",
    )

    try:
        from v2_b3_m4_reuse_integrity_lib import (  # noqa: WPS433
            has_downstream_stale_evidence,
            quarantine_stale_downstream_artifacts,
            remove_stale_worker_plan_outputs,
            repair_inconsistent_reuse_state,
        )

        repair_inconsistent_reuse_state(run_root, production_mode=True)
        if has_downstream_stale_evidence(run_root):
            quarantine_stale_downstream_artifacts(
                run_root,
                reason="failed_sample_heavy_cleanup",
            )
        remove_stale_worker_plan_outputs(run_root)
    except ImportError:
        pass

    deleted: List[str] = []
    errors: List[str] = []
    archivable = _collect_archivable_paths(run_root)
    discovery_root = run_root / "scout" / "discovery"
    archivable = [p for p in archivable if p != discovery_root]
    extra_dirs = (
        run_root / "scout" / "mesh",
        run_root / "lprod" / "mesh",
    )
    for extra in extra_dirs:
        if extra.exists() and extra not in archivable:
            archivable.append(extra)
    lock_paths = collect_run_tree_lock_paths(run_root)
    archivable.extend(lock_paths)

    if archivable:
        deleted.extend(_delete_archived_paths(run_root, archivable))

    shared_paths = collect_shared_sample_artifact_paths(
        repo_root=repo_root,
        sample_id=sample_id,
        run_id=run_id,
    )
    shared_deleted, shared_errors = _delete_paths(shared_paths)
    deleted.extend(shared_deleted)
    errors.extend(shared_errors)
    return deleted, errors


def _delete_shared_only(
    *,
    repo_root: Path,
    sample_id: str,
    run_id: str,
) -> Tuple[List[str], List[str]]:
    shared_paths = collect_shared_sample_artifact_paths(
        repo_root=repo_root,
        sample_id=sample_id,
        run_id=run_id,
    )
    return _delete_paths(shared_paths)


@dataclass
class CleanupBarrierOutcome:
    sample_id: str
    run_id: str
    status: str
    sample_success: bool
    deleted_shared_paths: List[str] = field(default_factory=list)
    deleted_run_paths: List[str] = field(default_factory=list)
    forbidden_heavy_artifact_count: int = 0
    shared_sample_artifact_count: int = 0
    verification_pass: bool = False
    errors: List[str] = field(default_factory=list)
    remaining_paths: List[str] = field(default_factory=list)
    runtime_s: float = 0.0
    compaction: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _write_cleanup_failure_report(
    *,
    run_root: Path,
    sample_id: str,
    run_id: str,
    verify_report: Mapping[str, Any],
    delete_errors: Sequence[str],
) -> Path:
    remaining = list(verify_report.get("shared_sample_artifacts_present") or [])
    for rel_dir in verify_report.get("forbidden_heavy_artifacts_present") or []:
        remaining.append(str(run_root / rel_dir))
    payload = {
        "schema": "m4_sample_cleanup_failure_report_v1",
        "tool_version": TOOL_VERSION,
        "generated_utc": utc_now(),
        "sample_id": sample_id,
        "run_id": run_id,
        "verification": dict(verify_report),
        "delete_errors": list(delete_errors),
        "remaining_paths": remaining,
        "status": "FAILED",
    }
    out = run_root / FAILURE_REPORT_REL
    write_json_atomic(out, payload)
    return out


def run_sample_cleanup_barrier(
    *,
    repo_root: Path,
    run_root: Path,
    sample_id: str,
    run_id: str,
    row: Mapping[str, Any],
    pool: Mapping[str, Any],
    keep_full: bool = False,
    run_rom_compare: bool = False,
    blocking: bool = True,
) -> CleanupBarrierOutcome:
    """Run compaction (success), delete shared artifacts, and verify before next sample."""
    t0 = time.perf_counter()
    outcome = str(row.get("outcome") or "fail")
    sample_success = outcome in PRODUCTION_PASS_OUTCOMES
    deleted_shared: List[str] = []
    deleted_run: List[str] = []
    delete_errors: List[str] = []
    compaction_payload: Optional[Dict[str, Any]] = None

    if sample_success:
        preserve_target_plan_before_cleanup(
            run_root=run_root,
            sample_id=sample_id,
            run_id=run_id,
        )
        preserve_comparison_provenance_before_cleanup(
            run_root=run_root,
            sample_id=sample_id,
            run_id=run_id,
        )
        if not keep_full:
            if compact_one_completed_run is None:
                delete_errors.append("compact_module_unavailable")
            else:
                compact_out = compact_one_completed_run(
                    repo_root=repo_root,
                    pool=pool,
                    sample_id=sample_id,
                    run_id=run_id,
                    keep_full=False,
                    dry_run=False,
                    production_row=dict(row),
                    run_rom_compare=bool(run_rom_compare),
                    production_trigger=True,
                )
                compaction_payload = compact_out.to_dict()
                if compact_out.status == "failed":
                    delete_errors.append(compact_out.error or "compaction_failed")
        shared_deleted, shared_errors = _delete_shared_only(
            repo_root=repo_root,
            sample_id=sample_id,
            run_id=run_id,
        )
        deleted_shared.extend(shared_deleted)
        delete_errors.extend(shared_errors)
    else:
        _write_failure_retention(
            run_root=run_root,
            sample_id=sample_id,
            run_id=run_id,
            row=row,
        )
        run_deleted, run_errors = _delete_failed_run_heavy(
            repo_root=repo_root,
            run_root=run_root,
            sample_id=sample_id,
            run_id=run_id,
        )
        deleted_run.extend(run_deleted)
        delete_errors.extend(run_errors)

    verify = verify_cleanup_barrier(
        repo_root=repo_root,
        run_root=run_root,
        sample_id=sample_id,
        run_id=run_id,
        sample_success=sample_success,
    )

    result = CleanupBarrierOutcome(
        sample_id=sample_id,
        run_id=run_id,
        status="completed" if verify.get("pass") else "failed",
        sample_success=sample_success,
        deleted_shared_paths=deleted_shared,
        deleted_run_paths=deleted_run,
        forbidden_heavy_artifact_count=int(verify.get("forbidden_heavy_artifact_count") or 0),
        shared_sample_artifact_count=int(verify.get("shared_sample_artifact_count") or 0),
        verification_pass=bool(verify.get("pass")),
        errors=list(verify.get("errors") or []) + delete_errors,
        remaining_paths=list(verify.get("shared_sample_artifacts_present") or [])
        + [
            str(run_root / rel_path)
            for rel_path in (verify.get("forbidden_heavy_artifacts_present") or [])
        ],
        runtime_s=round(time.perf_counter() - t0, 4),
        compaction=compaction_payload,
    )

    barrier_doc = {
        "schema": BARRIER_MANIFEST_SCHEMA,
        "tool_version": TOOL_VERSION,
        "generated_utc": utc_now(),
        "sample_id": sample_id,
        "run_id": run_id,
        "status": result.status,
        "sample_success": sample_success,
        "blocking": blocking,
        "verification": verify,
        "deleted_shared_paths": deleted_shared,
        "deleted_run_paths": deleted_run,
        "delete_errors": delete_errors,
        "compaction": compaction_payload,
    }
    write_json_atomic(run_root / BARRIER_MANIFEST_REL, barrier_doc)

    if verify.get("pass"):
        failure_report = run_root / FAILURE_REPORT_REL
        if failure_report.is_file():
            try:
                failure_report.unlink()
            except OSError as exc:
                delete_errors.append(f"{failure_report}:{exc}")

    if not verify.get("pass"):
        _write_cleanup_failure_report(
            run_root=run_root,
            sample_id=sample_id,
            run_id=run_id,
            verify_report=verify,
            delete_errors=delete_errors,
        )

    return result
