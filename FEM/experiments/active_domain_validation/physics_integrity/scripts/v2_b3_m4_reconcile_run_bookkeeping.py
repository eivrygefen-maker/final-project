#!/usr/bin/env python3
"""Read-only bookkeeping reconcile for one completed M4 production run (no FEM stages)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_m4_lhs_pool_bridge import reconcile_run_bookkeeping, specs_generated_dir  # noqa: E402
from v2_b3_m4_worker_run_lib import detect_repo_root, load_json, rel, utc_now  # noqa: E402
from v2_b3_petsc_util import write_json_atomic  # noqa: E402

DEFAULT_LHS_REL = "ROM/classic/lhs_pool.json"


def _lhs_row_index(pool: Dict[str, Any], sample_id: str) -> int:
    for index, entry in enumerate(pool.get("entries") or []):
        if str(entry.get("id")) == sample_id:
            return index
    raise SystemExit(f"error: sample_id not found in lhs pool: {sample_id}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Reconcile lhs_pool_status.json and lhs_production_runs_index.jsonl for one "
            "completed run without rerunning FEM stages or mutating the run tree."
        ),
    )
    parser.add_argument(
        "--lhs-json",
        type=Path,
        default=Path(DEFAULT_LHS_REL),
        help=f"LHS design pool JSON (default: {DEFAULT_LHS_REL}).",
    )
    parser.add_argument("--sample-id", required=True, help="Sample id, e.g. sample_002.")
    parser.add_argument("--run-id", required=True, help="Run id, e.g. sample_002_rom_prod_004.")
    parser.add_argument(
        "--batch-id",
        help="Optional batch id to patch pipeline_runs/batches/<batch_id>/batch_execution_summary.json.",
    )
    parser.add_argument(
        "--return-code",
        type=int,
        default=0,
        help="Pipeline subprocess return code to classify against (default: 0).",
    )
    parser.add_argument(
        "--no-require-cleanup-barrier",
        action="store_true",
        help="Do not require cleanup/sample_cleanup_barrier.json for pass classification.",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        help="Optional path for reconcile report JSON (default: specs/generated/).",
    )
    args = parser.parse_args(argv)

    repo_root = detect_repo_root(SCRIPT_DIR)
    lhs_path = (repo_root / args.lhs_json).resolve()
    if not lhs_path.is_file():
        print(f"error: lhs pool not found: {lhs_path}", file=sys.stderr)
        return 2

    pool = load_json(lhs_path)
    sample_id = str(args.sample_id)
    run_id = str(args.run_id)
    lhs_row_index = _lhs_row_index(pool, sample_id)

    report = reconcile_run_bookkeeping(
        repo_root=repo_root,
        pool=pool,
        lhs_path=lhs_path,
        sample_id=sample_id,
        run_id=run_id,
        lhs_row_index=lhs_row_index,
        batch_id=args.batch_id,
        require_cleanup_barrier=not args.no_require_cleanup_barrier,
        return_code=int(args.return_code),
    )

    out_path = args.report_out
    if out_path is None:
        out_path = (
            specs_generated_dir(repo_root)
            / f"bookkeeping_reconcile_{sample_id}_{run_id}_{utc_now()[:10].replace('-', '')}.json"
        )
    out_path = out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out_path, report)

    print(f"outcome={report.get('outcome')}")
    print(f"run_root={report.get('run_root')}")
    print(f"lhs_pool_updated={report.get('lhs_pool_updated')}")
    print(f"status_path={report.get('status_path')}")
    print(f"index_path={report.get('index_path')}")
    print(f"report={rel(out_path, repo_root=repo_root)}")
    batch_patch = report.get("batch_execution_summary") or {}
    if batch_patch.get("patched"):
        print(
            f"batch_summary_patched completed_count={batch_patch.get('completed_count')} "
            f"failed_count={batch_patch.get('failed_count')}"
        )
    elif args.batch_id:
        print(f"batch_summary_patch_skipped reason={batch_patch.get('reason')}")
    if report.get("error_message"):
        print(f"error_message={report.get('error_message')}")
    return 0 if report.get("outcome") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
