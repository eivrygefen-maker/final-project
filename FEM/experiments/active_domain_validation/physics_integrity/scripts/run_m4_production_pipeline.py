#!/usr/bin/env python3
"""Permanent M4 production runner — LHS pool → auto specs → scout/L_prod/workers/aggregation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_m4_lhs_pool_bridge import (  # noqa: E402
    DEFAULT_RUN_ID_SUFFIX,
    REFERENCE_SAMPLE_ID,
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_RUNNING,
    STATUS_SKIPPED,
    append_runs_index_row,
    build_batch_sample_entry,
    build_lhs_batch_spec,
    lhs_pool_status_path,
    lhs_runs_index_path,
    load_lhs_pool,
    load_lhs_pool_status,
    make_batch_id,
    select_lhs_samples,
    specs_generated_dir,
    status_row_from_run_summary,
    update_sample_status,
    write_lhs_pool_status,
    write_per_sample_spec,
)
from v2_b3_m4_lhs_production_batch import run_production_batch  # noqa: E402
from v2_b3_m4_worker_run_lib import detect_repo_root, rel, utc_now  # noqa: E402
from v2_b3_petsc_util import write_json_atomic  # noqa: E402

DEFAULT_LHS_REL = "ROM/classic/lhs_pool.json"


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "M4 permanent production runner: read LHS pool, auto-generate specs, "
            "run scout → L_prod → workers → aggregation per sample."
        ),
        epilog=(
            "Example:\n"
            "  python FEM/experiments/active_domain_validation/physics_integrity/scripts/"
            "run_m4_production_pipeline.py --lhs-json ROM/classic/lhs_pool.json "
            "--max-samples 10 --workers 3 --execute --continue-on-fail"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--lhs-json",
        type=Path,
        default=Path(DEFAULT_LHS_REL),
        help=f"LHS design pool JSON (default: {DEFAULT_LHS_REL}).",
    )
    parser.add_argument("--max-samples", type=int, default=1, help="Max samples to run this invocation.")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--batch-id", help="Override auto batch_id.")
    parser.add_argument("--run-id-suffix", default=DEFAULT_RUN_ID_SUFFIX, help="Run id suffix, e.g. m4prod1.")
    parser.add_argument("--start-index", type=int, default=0, help="0-based LHS entries[] start index.")
    parser.add_argument("--end-index", type=int, help="Inclusive LHS entries[] end index.")
    parser.add_argument("--force-sample", help="Run only this sample_id (e.g. sample_005).")
    parser.add_argument("--dry-run", action="store_true", help="Plan only; write specs/status previews.")
    parser.add_argument("--execute", action="store_true", help="Run M4 pipeline for selected samples.")
    parser.add_argument(
        "--continue-on-fail",
        action="store_true",
        help="Continue batch after a sample failure.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse PASS stages on partial runs (default behavior).",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        default=True,
        help="Skip samples with pass status for current run_id_suffix (default: true).",
    )
    parser.add_argument(
        "--no-skip-completed",
        action="store_false",
        dest="skip_completed",
        help="Include samples already marked pass (use with --force to re-run).",
    )
    parser.add_argument(
        "--exclude-reference",
        action="store_true",
        help=f"Exclude frozen reference {REFERENCE_SAMPLE_ID} from selection.",
    )
    parser.add_argument(
        "--include-reference",
        action="store_true",
        help=f"Explicitly allow {REFERENCE_SAMPLE_ID} in production (new run_id).",
    )
    parser.add_argument("--force", action="store_true", help="Re-run even when prior PASS exists.")
    parser.add_argument("--force-checkpoint", action="store_true")
    parser.add_argument("--force-workers", action="store_true")
    parser.add_argument("--force-aggregation", action="store_true")
    parser.add_argument(
        "--stop-after",
        choices=("scout", "checkpoint", "workers"),
        help="Stop each sample after this stage.",
    )
    return parser.parse_args(argv)


def _resolve_lhs_path(repo_root: Path, arg: Path) -> Path:
    return arg if arg.is_absolute() else repo_root / arg


def _build_selected_batch(
    *,
    repo_root: Path,
    pool: Dict[str, Any],
    selection: Sequence[Dict[str, Any]],
    batch_id: str,
    lhs_source_path: str,
    run_id_suffix: str,
    exclude_reference: bool,
) -> tuple[Dict[str, Any], Path]:
    sample_entries: List[Dict[str, Any]] = []
    for row in selection:
        sample_entries.append(
            build_batch_sample_entry(
                pool=pool,
                entry=row["entry"],
                lhs_row_index=int(row["lhs_row_index"]),
                run_id_suffix=run_id_suffix,
                batch_id=batch_id,
                lhs_source_path=lhs_source_path,
            )
        )
    batch_spec = build_lhs_batch_spec(
        pool=pool,
        samples=sample_entries,
        batch_id=batch_id,
        lhs_source_path=lhs_source_path,
        run_id_suffix=run_id_suffix,
        exclude_reference=exclude_reference,
    )
    gen_dir = specs_generated_dir(repo_root)
    gen_dir.mkdir(parents=True, exist_ok=True)
    spec_path = gen_dir / f"{batch_id}.json"
    write_json_atomic(spec_path, batch_spec)
    for entry in sample_entries:
        write_per_sample_spec(
            repo_root=repo_root,
            batch_spec=batch_spec,
            sample_entry=entry,
            lhs_source_path=lhs_source_path,
        )
    return batch_spec, spec_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    if args.dry_run and args.execute:
        print("error: use --dry-run or --execute, not both", file=sys.stderr)
        return 2
    if not args.dry_run and not args.execute:
        print("error: specify --dry-run or --execute", file=sys.stderr)
        return 2
    if args.max_samples < 1:
        print("error: --max-samples must be >= 1", file=sys.stderr)
        return 2
    if args.workers < 1:
        print("error: --workers must be >= 1", file=sys.stderr)
        return 2
    if args.include_reference and args.exclude_reference:
        print("error: use only one of --include-reference / --exclude-reference", file=sys.stderr)
        return 2

    repo_root = detect_repo_root(SCRIPT_DIR)
    lhs_path = _resolve_lhs_path(repo_root, args.lhs_json)
    if not lhs_path.is_file():
        print(f"error: missing --lhs-json: {lhs_path}", file=sys.stderr)
        return 2

    try:
        pool = load_lhs_pool(lhs_path)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    run_id_suffix = str(args.run_id_suffix).strip() or DEFAULT_RUN_ID_SUFFIX
    batch_id = args.batch_id or make_batch_id()
    lhs_rel = rel(lhs_path, repo_root=repo_root)
    exclude_reference = bool(args.exclude_reference) and not bool(args.include_reference)

    status_path = lhs_pool_status_path(repo_root)
    index_path = lhs_runs_index_path(repo_root)
    status_doc = load_lhs_pool_status(
        status_path,
        lhs_path=lhs_path,
        run_id_suffix=run_id_suffix,
        repo_root=repo_root,
    )
    status_doc["lhs_json"] = lhs_rel
    status_doc["run_id_suffix_default"] = run_id_suffix

    selection, skipped = select_lhs_samples(
        pool,
        status_doc,
        max_samples=int(args.max_samples),
        start_index=int(args.start_index),
        end_index=args.end_index,
        force_sample=args.force_sample,
        skip_completed=bool(args.skip_completed) and not bool(args.force),
        exclude_reference=exclude_reference,
        run_id_suffix=run_id_suffix,
    )

    if not selection:
        print("no samples selected (all completed or filtered out)")
        if skipped:
            for row in skipped[:10]:
                print(f"  skip {row.get('sample_id')}: {row.get('reason')}")
        return 0

    batch_spec, spec_path = _build_selected_batch(
        repo_root=repo_root,
        pool=pool,
        selection=selection,
        batch_id=batch_id,
        lhs_source_path=lhs_rel,
        run_id_suffix=run_id_suffix,
        exclude_reference=exclude_reference,
    )

    print(f"batch_id={batch_id}")
    print(f"lhs_json={lhs_rel}")
    print(f"spec_path={rel(spec_path, repo_root=repo_root)}")
    print(f"selected_count={len(selection)} skipped_count={len(skipped)}")
    for row in selection:
        print(f"  {row['sample_id']} -> {row['run_id']} (lhs_index={row['lhs_row_index']})")

    if args.dry_run:
        plan = {
            "schema": "m4_production_pipeline_plan_v1",
            "will_execute": False,
            "generated_utc": utc_now(),
            "batch_id": batch_id,
            "spec_path": rel(spec_path, repo_root=repo_root),
            "status_path": rel(status_path, repo_root=repo_root),
            "index_path": rel(index_path, repo_root=repo_root),
            "selected": selection,
            "skipped": skipped,
        }
        write_json_atomic(specs_generated_dir(repo_root) / f"{batch_id}_plan.json", plan)
        print("will_execute=false")
        print("no solver executed")
        return 0

    force_stages: Optional[Set[str]] = None
    if args.force_checkpoint or args.force_workers or args.force_aggregation:
        force_stages = set()
        if args.force_checkpoint:
            force_stages.add("checkpoint")
        if args.force_workers:
            force_stages.add("workers")
        if args.force_aggregation:
            force_stages.add("aggregate")

    allow_reference = bool(args.include_reference) or (
        not exclude_reference and any(r["sample_id"] == REFERENCE_SAMPLE_ID for r in selection)
    )

    def _on_sample_start(sid: str, run_id: str, lhs_row_index: int) -> None:
        update_sample_status(
            status_doc,
            sample_id=sid,
            patch={
                "lhs_row_index": lhs_row_index,
                "run_id": run_id,
                "status": STATUS_RUNNING,
                "batch_id": batch_id,
                "started_at": utc_now(),
                "error_message": None,
            },
        )
        write_lhs_pool_status(status_path, status_doc)
        append_runs_index_row(
            index_path,
            {
                "event": "sample_start",
                "sample_id": sid,
                "run_id": run_id,
                "lhs_row_index": lhs_row_index,
                "batch_id": batch_id,
            },
        )

    def _on_sample_finish(row: Dict[str, Any]) -> None:
        sid = str(row["sample_id"])
        outcome = str(row.get("outcome") or STATUS_FAIL)
        status = STATUS_PASS if outcome == "pass" else STATUS_FAIL
        if outcome == "reused_complete":
            status = STATUS_PASS
        patch = status_row_from_run_summary(
            sample_id=sid,
            lhs_row_index=int(row.get("lhs_row_index") or 0),
            run_id=str(row.get("run_id") or ""),
            batch_id=batch_id,
            run_root=Path(str(row.get("run_root_abs") or "")),
            outcome=outcome,
            elapsed_s=float(row.get("elapsed_s") or 0.0),
            summary=row,
            error_message=row.get("error_message"),
        )
        patch["status"] = status
        update_sample_status(status_doc, sample_id=sid, patch=patch)
        write_lhs_pool_status(status_path, status_doc)
        append_runs_index_row(index_path, {"event": "sample_finish", **patch})

    samples_for_batch = batch_spec.get("samples") or []
    lhs_index_by_sid = {str(r["sample_id"]): int(r["lhs_row_index"]) for r in selection}

    summary = run_production_batch(
        repo_root=repo_root,
        spec_path=spec_path,
        batch_id=batch_id,
        samples=samples_for_batch,
        spec=batch_spec,
        workers=int(args.workers),
        execute=True,
        continue_on_fail=bool(args.continue_on_fail),
        force=bool(args.force),
        stop_after=args.stop_after,
        resume=bool(args.resume),
        force_stages=force_stages,
        production_mode=True,
        exclude_reference=exclude_reference,
        allow_reference_mutation=allow_reference,
        skip_completed=bool(args.skip_completed) and not bool(args.force),
        lhs_index_by_sid=lhs_index_by_sid,
        on_sample_start=_on_sample_start,
        on_sample_finish=_on_sample_finish,
    )

    write_json_atomic(specs_generated_dir(repo_root) / f"{batch_id}_summary.json", summary)
    print(
        f"completed={summary['completed_count']} failed={summary['failed_count']} "
        f"skipped={summary['skipped_count']}"
    )
    print(f"status={rel(status_path, repo_root=repo_root)}")
    print(f"index={rel(index_path, repo_root=repo_root)}")
    return 1 if summary.get("failed_count") else 0


if __name__ == "__main__":
    raise SystemExit(main())
