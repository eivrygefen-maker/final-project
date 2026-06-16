#!/usr/bin/env python3
"""Reset one M4 sample run to a safe retry state without deleting completed peer samples."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from v2_b3_m4_reuse_integrity_lib import (  # noqa: E402
    checkpoint_artifact_contract_pass,
    quarantine_stale_downstream_artifacts,
    read_manifest,
    remove_stale_worker_plan_outputs,
    repair_inconsistent_reuse_state,
    scout_artifact_contract_pass,
    terminal_status_rank,
)
from v2_b3_m4_worker_run_lib import detect_repo_root, rel, utc_now  # noqa: E402
from v2_b3_petsc_util import write_json_atomic  # noqa: E402

GUITARS_REL = Path(
    "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars"
)


def resolve_run_root(repo_root: Path, *, sample_id: str, run_id: str) -> Path:
    return repo_root / GUITARS_REL / sample_id / "runs" / run_id


def _target_terminal_after_reset(run_root: Path, *, production_mode: bool) -> str:
    if checkpoint_artifact_contract_pass(run_root, production_mode=production_mode):
        return CHECKPOINT_TERMINAL_READY
    if scout_artifact_contract_pass(run_root):
        return SCOUT_TERMINAL_READY
    return "PLANNED"


def reset_sample_run_state(
    *,
    repo_root: Path,
    run_root: Path,
    sample_id: str,
    run_id: str,
    keep_failure_diagnostics: bool = True,
    production_mode: bool = True,
    reset_pool_status: bool = True,
    dry_run: bool = False,
) -> Dict[str, Any]:
    run_root = run_root.expanduser().resolve()
    report: Dict[str, Any] = {
        "schema": "m4_reset_sample_state_v1",
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
        report["actions"].append("reset_lhs_pool_status")

    report["status"] = "PASS"
    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Reset one M4 sample run for safe retry.")
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--shape", default="", help="Optional shape label for logging only.")
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
        keep_failure_diagnostics=bool(args.keep_failure_diagnostics),
        reset_pool_status=not bool(args.no_pool_status_reset),
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.shape:
        print(f"shape={args.shape} run_dir={rel(run_root, repo_root=repo_root)}")
    return 0 if report.get("status") in ("PASS", "DRY_RUN") else 2


if __name__ == "__main__":
    raise SystemExit(main())
