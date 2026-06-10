#!/usr/bin/env python3
"""Finalization-only recovery for a numerically completed M4 production run (no FEM)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_m4_lhs_pool_bridge import (  # noqa: E402
    classify_batch_sample_outcome,
    classify_sample_outcome,
    load_lhs_pool,
    reconcile_run_bookkeeping,
)
from v2_b3_m4_lhs_production_batch import _read_sample_summary  # noqa: E402
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


def resolve_run_root(repo_root: Path, sample_id: str, run_id: str) -> Path:
    return repo_root / GUITARS_REL / sample_id / "runs" / run_id


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
) -> Dict[str, Any]:
    run_root = resolve_run_root(repo_root, sample_id, run_id)
    if not run_root.is_dir():
        raise FileNotFoundError(f"run_root missing: {run_root}")

    ok, summary = verify_numerical_success(run_root)
    if not ok:
        raise RuntimeError(f"numerical_success_check_failed: {summary}")

    export_manifest, export_warn = try_export_sample_to_shared(
        run_root=run_root,
        sample_id=sample_id,
        run_id=run_id,
        shared_root=shared_root,
        repo_root=repo_root,
        cleanup_stale_exports=True,
    )
    if export_manifest is None or str(export_manifest.get("export_status") or "") != "EXPORTED":
        raise RuntimeError(f"shared_export_failed: {export_warn or export_manifest}")

    full_summary = _read_sample_summary(run_root, workers_requested=workers_requested)
    prelim_outcome, _ = classify_sample_outcome(return_code=0, summary=full_summary)
    row: Dict[str, Any] = {
        "sample_id": sample_id,
        "run_id": run_id,
        "outcome": prelim_outcome,
        "return_code": 0,
        "shared_export": export_manifest,
        **full_summary,
    }

    pool = load_lhs_pool(lhs_path)
    barrier = run_sample_cleanup_barrier(
        repo_root=repo_root,
        run_root=run_root,
        sample_id=sample_id,
        run_id=run_id,
        row=row,
        pool=pool,
        keep_full=False,
        run_rom_compare=False,
        blocking=True,
    )
    row["cleanup_barrier"] = barrier.to_dict()
    if barrier.compaction:
        row["compaction"] = barrier.compaction

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

    return {
        "schema": "m4_finalize_completed_run_v1",
        "generated_utc": utc_now(),
        "sample_id": sample_id,
        "run_id": run_id,
        "run_root": rel(run_root, repo_root=repo_root),
        "outcome": outcome,
        "shared_export": export_manifest,
        "cleanup_barrier_status": barrier.status,
        "compaction_status": (barrier.compaction or {}).get("status"),
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
    parser.add_argument("--report-path", type=Path)
    args = parser.parse_args(argv)

    repo_root = detect_repo_root(SCRIPT_DIR)
    lhs_path = args.lhs_json if args.lhs_json.is_absolute() else repo_root / args.lhs_json
    shared_root = detect_shared_root(args.shared_root)
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
    print(f"graph_export_status={(report.get('shared_export') or {}).get('export_status')}")
    print(f"cleanup_barrier_status={report.get('cleanup_barrier_status')}")
    print(f"compaction_status={report.get('compaction_status')}")
    print(f"report={rel(report_path, repo_root=repo_root)}")
    print("fem_stages_executed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
