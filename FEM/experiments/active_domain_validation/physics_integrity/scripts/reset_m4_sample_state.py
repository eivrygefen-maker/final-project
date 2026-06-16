#!/usr/bin/env python3
"""Reset one M4 sample run to a safe retry state without deleting completed peer samples."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_m4_freeze_first_e2e_run import CHECKPOINT_TERMINAL_READY, SCOUT_TERMINAL_READY  # noqa: E402
from v2_b3_m4_lhs_pool_bridge import (  # noqa: E402
    STATUS_PENDING,
    lhs_pool_status_path,
    load_lhs_pool_status,
    update_sample_status,
    write_lhs_pool_status,
)
from v2_b3_m4_production_freeze import TERMINAL_PRODUCTION_COMPLETED  # noqa: E402
from v2_b3_m4_reuse_integrity_lib import (  # noqa: E402
    checkpoint_artifact_contract_pass,
    quarantine_stale_downstream_artifacts,
    read_manifest,
    remove_stale_worker_plan_outputs,
    repair_inconsistent_reuse_state,
    scout_artifact_contract_pass,
    terminal_status_rank,
)
from v2_b3_m4_sample_cleanup_barrier import collect_shared_sample_artifact_paths  # noqa: E402
from v2_b3_m4_mesh_manifest_lib import collect_global_mesh_cache_paths_resolved  # noqa: E402
from v2_b3_m4_worker_run_lib import detect_repo_root, rel, utc_now  # noqa: E402
from v2_b3_petsc_util import write_json_atomic  # noqa: E402

GUITARS_REL = Path(
    "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars"
)

FULL_CLEAN_RUN_DIRS: Tuple[str, ...] = (
    "scout",
    "lprod",
    "worker_results",
    "aggregation",
    "freeze",
    "compaction",
    "mesh_inspection",
)

FULL_CLEAN_RUN_FILES: Tuple[str, ...] = (
    "pipeline_run_manifest.m4_4_full_aggregation_preview.json",
    "pipeline_run_manifest.m4_4_partial_aggregation_preview.json",
    "m4_run_one_sample_plan.json",
    "m4_sample_runtime_provenance.json",
    "stale_running_repair.json",
)


def resolve_run_root(repo_root: Path, *, sample_id: str, run_id: str) -> Path:
    return repo_root / GUITARS_REL / sample_id / "runs" / run_id


def _target_terminal_after_reset(run_root: Path, *, production_mode: bool) -> str:
    if checkpoint_artifact_contract_pass(run_root, production_mode=production_mode):
        return CHECKPOINT_TERMINAL_READY
    if scout_artifact_contract_pass(run_root):
        return SCOUT_TERMINAL_READY
    return "PLANNED"


def _delete_paths(paths: Sequence[Path]) -> List[str]:
    deleted: List[str] = []
    for path in paths:
        if not path.exists():
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        deleted.append(str(path))
    return deleted


def full_clean_sample_run(
    *,
    repo_root: Path,
    run_root: Path,
    sample_id: str,
    run_id: str,
    keep_failure_diagnostics: bool = True,
    dry_run: bool = False,
) -> Dict[str, Any]:
    run_root = run_root.expanduser().resolve()
    manifest = read_manifest(run_root)
    terminal = str(manifest.get("terminal_status") or "")
    if terminal == TERMINAL_PRODUCTION_COMPLETED:
        return {
            "status": "SKIP",
            "reason": "run_production_completed_refusing_full_clean",
            "terminal_status": terminal,
        }

    report: Dict[str, Any] = {
        "mode": "full-clean",
        "sample_id": sample_id,
        "run_id": run_id,
        "previous_terminal_status": terminal,
        "dry_run": dry_run,
        "deleted": [],
    }
    if dry_run:
        report["status"] = "DRY_RUN"
        report["would_delete_run_dirs"] = list(FULL_CLEAN_RUN_DIRS)
        return report

    for rel_dir in FULL_CLEAN_RUN_DIRS:
        path = run_root / rel_dir
        if path.exists():
            shutil.rmtree(path)
            report["deleted"].append(rel_dir)

    for rel_file in FULL_CLEAN_RUN_FILES:
        path = run_root / rel_file
        if path.is_file():
            path.unlink()
            report["deleted"].append(rel_file)

    shared = collect_shared_sample_artifact_paths(
        repo_root=repo_root,
        sample_id=sample_id,
        run_id=run_id,
    )
    global_mesh = collect_global_mesh_cache_paths_resolved(repo_root, sample_id)
    for path in global_mesh:
        if path not in shared:
            shared.append(path)
    report["deleted"].extend(_delete_paths(shared))

    manifest_path = run_root / "pipeline_run_manifest.json"
    if manifest_path.is_file():
        body = {
            "schema": "m4_pipeline_run_manifest_v1",
            "sample_id": sample_id,
            "run_id": run_id,
            "terminal_status": "PLANNED",
            "updated_utc": utc_now(),
            "reset_mode": "full-clean",
            "stages": {},
        }
        write_json_atomic(manifest_path, body)
        report["deleted"].append("pipeline_run_manifest.json:reset_to_planned")

    if not keep_failure_diagnostics:
        for rel_path in (
            "cleanup/sample_failure_retention.json",
            "logs/sample_failure_diagnostic.log",
        ):
            path = run_root / rel_path
            if path.is_file():
                path.unlink()
                report["deleted"].append(rel_path)

    report["status"] = "PASS"
    report["terminal_status"] = "PLANNED"
    return report


def reset_sample_run_state(
    *,
    repo_root: Path,
    run_root: Path,
    sample_id: str,
    run_id: str,
    mode: str = "repair",
    keep_failure_diagnostics: bool = True,
    production_mode: bool = True,
    reset_pool_status: bool = True,
    dry_run: bool = False,
) -> Dict[str, Any]:
    run_root = run_root.expanduser().resolve()
    report: Dict[str, Any] = {
        "schema": "m4_reset_sample_state_v1",
        "mode": mode,
        "sample_id": sample_id,
        "run_id": run_id,
        "run_root": str(run_root),
        "dry_run": dry_run,
        "actions": [],
    }

    if not run_root.is_dir():
        report["status"] = "FAIL"
        report["error"] = f"run_root missing: {run_root}"
        return report

    manifest = read_manifest(run_root)
    report["previous_terminal_status"] = str(manifest.get("terminal_status") or "")

    if mode == "full-clean":
        full = full_clean_sample_run(
            repo_root=repo_root,
            run_root=run_root,
            sample_id=sample_id,
            run_id=run_id,
            keep_failure_diagnostics=keep_failure_diagnostics,
            dry_run=dry_run,
        )
        report.update(full)
        if dry_run or full.get("status") == "SKIP":
            if reset_pool_status and not dry_run and full.get("status") != "SKIP":
                pass
            elif full.get("status") == "SKIP":
                return report
        elif reset_pool_status:
            _reset_pool_status(repo_root, sample_id=sample_id, run_id=run_id)
            report["actions"].append("reset_lhs_pool_status")
        if report.get("status") != "SKIP":
            report["status"] = full.get("status", "PASS")
        return report

    if dry_run:
        report["status"] = "DRY_RUN"
        report["would_repair"] = True
        report["would_target_terminal"] = _target_terminal_after_reset(
            run_root, production_mode=production_mode
        )
        return report

    repair = repair_inconsistent_reuse_state(run_root, production_mode=production_mode)
    if repair.get("repaired"):
        report["actions"].append("repair_inconsistent_reuse_state")
        report["repair"] = repair

    quarantine = quarantine_stale_downstream_artifacts(
        run_root,
        reason="reset_m4_sample_state",
    )
    if quarantine.get("moved_paths"):
        report["actions"].append("quarantine_downstream")
        report["quarantine"] = quarantine

    removed = remove_stale_worker_plan_outputs(run_root)
    if removed:
        report["actions"].append("remove_stale_worker_plan_outputs")
        report["removed_worker_plan_outputs"] = removed

    target_terminal = _target_terminal_after_reset(run_root, production_mode=production_mode)
    manifest = read_manifest(run_root)
    prev_rank = terminal_status_rank(str(manifest.get("terminal_status") or ""))
    target_rank = terminal_status_rank(target_terminal)
    if target_rank < prev_rank:
        manifest["terminal_status"] = target_terminal
        manifest["updated_utc"] = utc_now()
        stages = manifest.setdefault("stages", {})
        if target_terminal == SCOUT_TERMINAL_READY:
            for key in ("stage4_lprod_mesh", "stage4_lprod_export", "stage5_workers", "stage6_aggregate"):
                st = stages.setdefault(key, {})
                st["status"] = "PLANNED_READY"
                st["updated_utc"] = utc_now()
        elif target_terminal == CHECKPOINT_TERMINAL_READY:
            for key in ("stage5_workers", "stage6_aggregate"):
                st = stages.setdefault(key, {})
                if str(st.get("status") or "") == "PASS":
                    st["status"] = "PLANNED_READY"
                st["updated_utc"] = utc_now()
        manifest.pop("failure_reason", None)
        write_json_atomic(run_root / "pipeline_run_manifest.json", manifest)
        report["actions"].append("reset_manifest_terminal")
        report["terminal_status"] = target_terminal

    if not keep_failure_diagnostics:
        for rel_path in (
            "cleanup/sample_failure_retention.json",
            "logs/sample_failure_diagnostic.log",
        ):
            path = run_root / rel_path
            if path.is_file():
                path.unlink()
                report["actions"].append(f"deleted:{rel_path}")

    if reset_pool_status:
        _reset_pool_status(repo_root, sample_id=sample_id, run_id=run_id)
        report["actions"].append("reset_lhs_pool_status")

    report["status"] = "PASS"
    return report


def _reset_pool_status(repo_root: Path, *, sample_id: str, run_id: str) -> None:
    status_path = lhs_pool_status_path(repo_root)
    lhs_default = repo_root / "ROM/classic/lhs_pool.json"
    status_doc = load_lhs_pool_status(
        status_path,
        lhs_path=lhs_default,
        run_id_suffix=run_id.removeprefix(f"{sample_id}_"),
        repo_root=repo_root,
    )
    update_sample_status(
        status_doc,
        sample_id=sample_id,
        patch={
            "sample_id": sample_id,
            "status": STATUS_PENDING,
            "run_id": run_id,
            "updated_utc": utc_now(),
            "reset_reason": "reset_m4_sample_state",
        },
    )
    write_lhs_pool_status(status_path, status_doc)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Reset one M4 sample run for safe retry.")
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--shape", default="", help="Optional shape label for logging only.")
    parser.add_argument(
        "--mode",
        choices=("repair", "full-clean"),
        default="repair",
        help="repair=quarantine stale downstream; full-clean=delete runtime dirs and start from zero",
    )
    parser.add_argument(
        "--keep-failure-diagnostics",
        action="store_true",
        default=True,
        help="Retain cleanup/logs failure diagnostics in the run dir (default).",
    )
    parser.add_argument(
        "--no-keep-failure-diagnostics",
        action="store_false",
        dest="keep_failure_diagnostics",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-pool-status-reset", action="store_true")
    args = parser.parse_args(argv)

    repo_root = (args.repo_root or detect_repo_root(SCRIPT_DIR)).resolve()
    sample_id = str(args.sample_id)
    run_id = str(args.run_id)
    run_root = resolve_run_root(repo_root, sample_id=sample_id, run_id=run_id)

    report = reset_sample_run_state(
        repo_root=repo_root,
        run_root=run_root,
        sample_id=sample_id,
        run_id=run_id,
        mode=str(args.mode),
        keep_failure_diagnostics=bool(args.keep_failure_diagnostics),
        reset_pool_status=not bool(args.no_pool_status_reset),
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.shape:
        print(f"shape={args.shape} run_dir={rel(run_root, repo_root=repo_root)}")
    return 0 if report.get("status") in ("PASS", "DRY_RUN", "SKIP") else 2


if __name__ == "__main__":
    raise SystemExit(main())
