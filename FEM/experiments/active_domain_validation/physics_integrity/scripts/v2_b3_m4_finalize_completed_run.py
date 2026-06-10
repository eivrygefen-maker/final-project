#!/usr/bin/env python3
"""Finalization-only recovery for a numerically completed M4 production run (no FEM)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compact_completed_m4_runs import (  # noqa: E402
    compact_one_completed_run,
    probe_compaction_eligibility,
    production_compaction_preconditions,
)
from v2_b3_m4_lhs_pool_bridge import (  # noqa: E402
    classify_batch_sample_outcome,
    classify_sample_outcome,
    load_lhs_pool,
    reconcile_run_bookkeeping,
)
from v2_b3_m4_lhs_production_batch import _read_sample_summary  # noqa: E402
from v2_b3_m4_physics_identity_lib import (  # noqa: E402
    FORBIDDEN_HEAVY_REL_DIRS,
    count_forbidden_heavy_artifacts,
)
from v2_b3_m4_production_freeze import (  # noqa: E402
    ensure_production_acceptance_for_finalization,
    read_production_acceptance_status,
)
from v2_b3_m4_sample_cleanup_barrier import run_sample_cleanup_barrier  # noqa: E402
from v2_b3_m4_shared_export import (  # noqa: E402
    detect_shared_root,
    try_export_sample_to_shared,
    verify_numerical_success,
)
from v2_b3_m4_worker_run_lib import detect_repo_root, load_json, rel, utc_now  # noqa: E402
from v2_b3_petsc_util import write_json_atomic  # noqa: E402

DEFAULT_LHS_REL = "ROM/classic/lhs_pool.json"
GUITARS_REL = "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars"
COMPACTION_MANIFEST_REL = "compaction/compaction_manifest.json"
FINALIZATION_STAGE_KEYS: Tuple[str, ...] = (
    "numerical_completion_verified",
    "shared_export_status",
    "acceptance_repair_attempted",
    "production_acceptance_pass",
    "production_acceptance_failures",
    "compaction_invoked",
    "compaction_status",
    "compaction_skip_reason",
    "compaction_deleted_paths",
    "compaction_manifest_present",
    "cleanup_invoked",
    "cleanup_status",
)

class CompactionNotCompletedError(RuntimeError):
    """Raised when compaction did not complete before cleanup is allowed."""


def resolve_run_root(repo_root: Path, sample_id: str, run_id: str) -> Path:
    return repo_root / GUITARS_REL / sample_id / "runs" / run_id


def _new_stage_report() -> Dict[str, Any]:
    return {key: None for key in FINALIZATION_STAGE_KEYS}


def _print_stage_report(stages: Mapping[str, Any]) -> None:
    for key in FINALIZATION_STAGE_KEYS:
        value = stages.get(key)
        if isinstance(value, (list, dict)):
            print(f"{key}={json.dumps(value, sort_keys=True)}")
        else:
            print(f"{key}={value}")


def build_compaction_production_row(
    *,
    sample_id: str,
    run_id: str,
    acceptance_report: Mapping[str, Any],
    export_manifest: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build the compaction row after acceptance repair — do not rely on stale manifest reads."""
    acceptance_pass = bool(acceptance_report.get("production_acceptance_pass"))
    return {
        "sample_id": sample_id,
        "run_id": run_id,
        "outcome": "pass",
        "return_code": 0,
        "aggregation_status": summary.get("aggregation_status") or "AGGREGATION_PASS",
        "final_aggregation_ready": bool(summary.get("final_aggregation_ready")),
        "terminal_status": summary.get("terminal_status") or "COMPLETED",
        "production_acceptance_pass": acceptance_pass,
        "production_acceptance_failures": list(acceptance_report.get("production_acceptance_failures") or []),
        "shared_export": export_manifest,
        "production_acceptance": dict(acceptance_report),
    }


def _heavy_paths_present(run_root: Path) -> List[str]:
    present: List[str] = []
    for rel in FORBIDDEN_HEAVY_REL_DIRS:
        if (run_root / rel).exists():
            present.append(rel)
    return present


def _read_compaction_manifest(run_root: Path) -> Optional[Dict[str, Any]]:
    path = run_root / COMPACTION_MANIFEST_REL
    if not path.is_file():
        return None
    try:
        doc = load_json(path)
        return doc if isinstance(doc, dict) else None
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def require_compaction_completed(
    *,
    run_root: Path,
    compact_out: Mapping[str, Any],
    stages: Dict[str, Any],
) -> None:
    status = str(compact_out.get("status") or "")
    skip_reason = str(compact_out.get("skip_reason") or "")
    deleted_bytes = int(compact_out.get("deleted_bytes") or 0)
    heavy_present = _heavy_paths_present(run_root)
    manifest = _read_compaction_manifest(run_root)
    manifest_present = manifest is not None

    stages["compaction_status"] = status
    stages["compaction_skip_reason"] = skip_reason or None
    stages["compaction_manifest_present"] = manifest_present
    if manifest:
        stages["compaction_deleted_paths"] = list(manifest.get("deleted_paths") or manifest.get("archived_paths") or [])

    acceptable_statuses = {"completed", "already_compacted"}
    blocked_statuses = {"planned", "skipped", "failed", "keep_full", "dry_run_planned_delete"}
    if status in blocked_statuses or status not in acceptable_statuses:
        raise CompactionNotCompletedError(
            f"COMPACTION_NOT_COMPLETED: status={status!r} skip_reason={skip_reason!r} "
            f"deleted_bytes={deleted_bytes} heavy_paths_present={heavy_present}"
        )
    if not manifest_present:
        raise CompactionNotCompletedError(
            f"COMPACTION_NOT_COMPLETED: missing {COMPACTION_MANIFEST_REL} "
            f"heavy_paths_present={heavy_present}"
        )
    if heavy_present:
        raise CompactionNotCompletedError(
            f"COMPACTION_NOT_COMPLETED: heavy_paths_still_present={heavy_present} "
            f"deleted_bytes={deleted_bytes}"
        )
    if deleted_bytes == 0 and heavy_present:
        raise CompactionNotCompletedError(
            f"COMPACTION_NOT_COMPLETED: deleted_bytes=0 heavy_paths_present={heavy_present}"
        )


def diagnose_finalization_state(
    *,
    repo_root: Path,
    sample_id: str,
    run_id: str,
    lhs_path: Path,
    shared_root: Optional[Path] = None,
    workers_requested: int = 3,
) -> Dict[str, Any]:
    """Read-only diagnostic: acceptance row, compaction gates, heavy paths (no deletion)."""
    run_root = resolve_run_root(repo_root, sample_id, run_id)
    stages = _new_stage_report()
    report: Dict[str, Any] = {
        "schema": "m4_finalize_diagnostic_v1",
        "generated_utc": utc_now(),
        "sample_id": sample_id,
        "run_id": run_id,
        "run_root": rel(run_root, repo_root=repo_root),
        "stages": stages,
        "fem_stages_executed": False,
    }

    ok, summary = verify_numerical_success(run_root)
    stages["numerical_completion_verified"] = ok

    local_export = run_root / "aggregation" / "shared_export_manifest.json"
    if local_export.is_file():
        try:
            export_manifest = load_json(local_export)
            stages["shared_export_status"] = export_manifest.get("export_status")
        except (OSError, ValueError, json.JSONDecodeError):
            export_manifest = {}
            stages["shared_export_status"] = "local_manifest_unreadable"
    else:
        export_manifest = {}
        stages["shared_export_status"] = "NOT_EXPORTED"

    recorded = read_production_acceptance_status(run_root)
    stages["acceptance_repair_attempted"] = False
    stages["production_acceptance_pass"] = recorded.get("production_acceptance_pass")
    stages["production_acceptance_failures"] = list(recorded.get("production_acceptance_failures") or [])

    full_summary = _read_sample_summary(run_root, workers_requested=workers_requested)
    acceptance_report = {
        "acceptance_pass": bool(recorded.get("production_acceptance_pass")),
        "production_acceptance_pass": recorded.get("production_acceptance_pass"),
        "production_acceptance_failures": stages["production_acceptance_failures"],
    }
    compaction_row = build_compaction_production_row(
        sample_id=sample_id,
        run_id=run_id,
        acceptance_report=acceptance_report,
        export_manifest=export_manifest or {},
        summary=full_summary,
    )
    report["compaction_row"] = compaction_row
    report["summary_after_read"] = {
        "terminal_status": full_summary.get("terminal_status"),
        "production_acceptance_pass": full_summary.get("production_acceptance_pass"),
    }

    pool = load_lhs_pool(lhs_path)
    entry = next((e for e in pool.get("entries") or [] if str(e.get("id")) == sample_id), {})
    gate_ok, gate_reason, gate_warnings = production_compaction_preconditions(
        row=compaction_row,
        pool_entry=entry,
        run_rom_compare=False,
    )
    report["production_compaction_gate"] = {
        "ok": gate_ok,
        "reason": gate_reason,
        "warnings": gate_warnings,
    }
    eligibility = probe_compaction_eligibility(
        repo_root=repo_root,
        pool=pool,
        sample_id=sample_id,
        run_id=run_id,
        production_row=compaction_row,
        allow_transitional_lhs=True,
    )
    report["lhs_eligibility"] = eligibility
    report["lhs_entry_status"] = eligibility.get("lhs_entry_status") or entry.get("status")
    report["lhs_entry_last_run_id"] = eligibility.get("lhs_entry_last_run_id")
    report["finalizing_run_id"] = run_id
    report["lhs_run_ownership_match"] = eligibility.get("lhs_run_ownership_match")
    report["transitional_lhs_allowed"] = eligibility.get("transitional_lhs_allowed")
    report["compaction_eligible"] = eligibility.get("compaction_eligible")
    stages["compaction_skip_reason"] = eligibility.get("compaction_skip_reason")

    heavy_count, heavy_paths = count_forbidden_heavy_artifacts(run_root)
    report["forbidden_heavy_artifact_count"] = heavy_count
    report["forbidden_heavy_artifacts_present"] = heavy_paths
    manifest = _read_compaction_manifest(run_root)
    stages["compaction_manifest_present"] = manifest is not None
    stages["compaction_status"] = (manifest or {}).get("status")
    stages["compaction_skip_reason"] = None
    stages["compaction_invoked"] = False
    stages["cleanup_invoked"] = False
    stages["cleanup_status"] = None
    if manifest:
        stages["compaction_deleted_paths"] = list(manifest.get("deleted_paths") or manifest.get("archived_paths") or [])

    barrier_path = run_root / "cleanup" / "sample_cleanup_barrier.json"
    if barrier_path.is_file():
        try:
            barrier = load_json(barrier_path)
            stages["cleanup_status"] = barrier.get("status")
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    return report


def finalize_completed_run(
    *,
    repo_root: Path,
    sample_id: str,
    run_id: str,
    lhs_path: Path,
    shared_root: Path,
    batch_id: Optional[str] = None,
    reconcile_bookkeeping: bool = True,
    workers_requested: int = 3,
    diagnose_only: bool = False,
) -> Dict[str, Any]:
    run_root = resolve_run_root(repo_root, sample_id, run_id)
    if not run_root.is_dir():
        raise FileNotFoundError(f"run_root missing: {run_root}")

    if diagnose_only:
        return diagnose_finalization_state(
            repo_root=repo_root,
            sample_id=sample_id,
            run_id=run_id,
            lhs_path=lhs_path,
            shared_root=shared_root,
            workers_requested=workers_requested,
        )

    stages = _new_stage_report()

    ok, summary = verify_numerical_success(run_root)
    stages["numerical_completion_verified"] = ok
    if not ok:
        _print_stage_report(stages)
        raise RuntimeError(f"numerical_success_check_failed: {summary}")

    export_manifest, export_warn = try_export_sample_to_shared(
        run_root=run_root,
        sample_id=sample_id,
        run_id=run_id,
        shared_root=shared_root,
        repo_root=repo_root,
        cleanup_stale_exports=True,
    )
    stages["shared_export_status"] = (export_manifest or {}).get("export_status")
    if export_manifest is None or str(export_manifest.get("export_status") or "") != "EXPORTED":
        _print_stage_report(stages)
        raise RuntimeError(f"shared_export_failed: {export_warn or export_manifest}")

    acceptance_report = ensure_production_acceptance_for_finalization(
        repo_root=repo_root,
        run_root=run_root,
    )
    stages["acceptance_repair_attempted"] = bool(acceptance_report.get("manifests_repaired"))
    stages["production_acceptance_pass"] = bool(acceptance_report.get("production_acceptance_pass"))
    stages["production_acceptance_failures"] = list(acceptance_report.get("production_acceptance_failures") or [])
    if not bool(acceptance_report.get("acceptance_pass")):
        _print_stage_report(stages)
        failures = acceptance_report.get("production_acceptance_failures") or []
        raise RuntimeError(f"production_acceptance_failed: {failures or acceptance_report}")

    full_summary = _read_sample_summary(run_root, workers_requested=workers_requested)
    compaction_row = build_compaction_production_row(
        sample_id=sample_id,
        run_id=run_id,
        acceptance_report=acceptance_report,
        export_manifest=export_manifest,
        summary=full_summary,
    )

    pool = load_lhs_pool(lhs_path)
    stages["compaction_invoked"] = True
    compact_out = compact_one_completed_run(
        repo_root=repo_root,
        pool=pool,
        sample_id=sample_id,
        run_id=run_id,
        keep_full=False,
        dry_run=False,
        production_row=compaction_row,
        run_rom_compare=False,
        production_trigger=True,
        allow_transitional_lhs=True,
    )
    compact_dict = compact_out.to_dict()
    stages["compaction_deleted_paths"] = compact_dict.get("deleted_paths")
    try:
        require_compaction_completed(run_root=run_root, compact_out=compact_dict, stages=stages)
    except CompactionNotCompletedError as exc:
        _print_stage_report(stages)
        raise RuntimeError(str(exc)) from exc

    stages["cleanup_invoked"] = True
    barrier = run_sample_cleanup_barrier(
        repo_root=repo_root,
        run_root=run_root,
        sample_id=sample_id,
        run_id=run_id,
        row=compaction_row,
        pool=pool,
        keep_full=True,
        run_rom_compare=False,
        blocking=True,
    )
    stages["cleanup_status"] = barrier.status

    row: Dict[str, Any] = {
        **compaction_row,
        "shared_export": export_manifest,
        "production_acceptance": acceptance_report,
        "compaction": compact_dict,
        "cleanup_barrier": barrier.to_dict(),
    }

    outcome, err_msg = classify_batch_sample_outcome(
        return_code=0,
        summary=full_summary,
        cleanup_barrier=row["cleanup_barrier"],
        require_cleanup_barrier=True,
        shared_export=export_manifest,
        require_graph_export=True,
    )
    row["outcome"] = outcome
    if err_msg:
        row["error_message"] = err_msg
    if outcome != "pass":
        _print_stage_report(stages)
        raise RuntimeError(f"finalization_classification_failed: {err_msg}")

    reconcile_report: Optional[Dict[str, Any]] = None
    if reconcile_bookkeeping:
        sample_input = load_json(run_root / "sample" / "sample_input.json") if (
            run_root / "sample" / "sample_input.json"
        ).is_file() else {}
        lhs_row_index = int(sample_input.get("lhs_row_index") or 0)
        reconcile_report = reconcile_run_bookkeeping(
            repo_root=repo_root,
            pool=pool,
            lhs_path=lhs_path,
            sample_id=sample_id,
            run_id=run_id,
            lhs_row_index=lhs_row_index,
            batch_id=batch_id,
            require_cleanup_barrier=True,
            return_code=0,
        )

    _print_stage_report(stages)

    return {
        "schema": "m4_finalize_completed_run_v1",
        "generated_utc": utc_now(),
        "sample_id": sample_id,
        "run_id": run_id,
        "run_root": rel(run_root, repo_root=repo_root),
        "outcome": outcome,
        "stages": stages,
        "compaction_row": compaction_row,
        "shared_export": export_manifest,
        "production_acceptance": acceptance_report,
        "production_acceptance_pass": bool(acceptance_report.get("production_acceptance_pass")),
        "compaction": compact_dict,
        "cleanup_barrier_status": barrier.status,
        "compaction_status": stages.get("compaction_status"),
        "reconcile_report": reconcile_report,
        "fem_stages_executed": False,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Finalize a numerically completed run: export, compact, cleanup, reconcile (no FEM)."
    )
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--lhs-json", type=Path, default=Path(DEFAULT_LHS_REL))
    parser.add_argument("--batch-id", help="Optional batch id for bookkeeping reconcile.")
    parser.add_argument("--shared-root", type=Path, help="Shared export root (default: /media/sf_gmar).")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--no-reconcile", action="store_true")
    parser.add_argument("--diagnose", action="store_true", help="Read-only diagnostic; no deletion.")
    parser.add_argument("--report-path", type=Path)
    args = parser.parse_args(argv)

    repo_root = detect_repo_root(SCRIPT_DIR)
    lhs_path = args.lhs_json if args.lhs_json.is_absolute() else repo_root / args.lhs_json
    shared_root = detect_shared_root(args.shared_root)

    if args.diagnose:
        report = diagnose_finalization_state(
            repo_root=repo_root,
            sample_id=str(args.sample_id),
            run_id=str(args.run_id),
            lhs_path=lhs_path,
            shared_root=shared_root,
            workers_requested=int(args.workers),
        )
        report_path = args.report_path
        if report_path is None:
            report_path = (
                repo_root
                / "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/specs/generated"
                / f"finalize_diagnose_{args.sample_id}_{args.run_id}.json"
            )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(report_path, report)
        _print_stage_report(report.get("stages") or {})
        print(f"lhs_entry_status={report.get('lhs_entry_status')}")
        print(f"lhs_entry_last_run_id={report.get('lhs_entry_last_run_id')}")
        print(f"finalizing_run_id={report.get('finalizing_run_id')}")
        print(f"lhs_run_ownership_match={report.get('lhs_run_ownership_match')}")
        print(f"transitional_lhs_allowed={report.get('transitional_lhs_allowed')}")
        print(f"compaction_eligible={report.get('compaction_eligible')}")
        print(f"compaction_skip_reason={report.get('stages', {}).get('compaction_skip_reason')}")
        print(f"production_compaction_gate={json.dumps(report.get('production_compaction_gate') or {})}")
        print(f"forbidden_heavy_artifact_count={report.get('forbidden_heavy_artifact_count')}")
        print(f"compaction_row_production_acceptance_pass={(report.get('compaction_row') or {}).get('production_acceptance_pass')}")
        print(f"report={rel(report_path, repo_root=repo_root)}")
        print("fem_stages_executed=false")
        return 0

    if shared_root is None:
        print("error: shared root not found", file=sys.stderr)
        return 2

    try:
        report = finalize_completed_run(
            repo_root=repo_root,
            sample_id=str(args.sample_id),
            run_id=str(args.run_id),
            lhs_path=lhs_path,
            shared_root=shared_root,
            batch_id=args.batch_id,
            reconcile_bookkeeping=not bool(args.no_reconcile),
            workers_requested=int(args.workers),
        )
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    report_path = args.report_path
    if report_path is None:
        report_path = (
            repo_root
            / "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/specs/generated"
            / f"finalize_{args.sample_id}_{args.run_id}.json"
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(report_path, report)
    print(f"sample_id={args.sample_id}")
    print(f"run_id={args.run_id}")
    print(f"outcome={report.get('outcome')}")
    print(f"report={rel(report_path, repo_root=repo_root)}")
    print("fem_stages_executed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
