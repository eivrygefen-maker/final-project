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
    LHS_RUNNING,
    OUTCOME_PASS_FREEZE_WARNING,
    REFERENCE_SAMPLE_ID,
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_RUNNING,
    append_runs_index_row,
    build_batch_sample_entry,
    build_lhs_batch_spec,
    lhs_pool_entry_patch_from_run,
    lhs_pool_status_path,
    lhs_runs_index_path,
    load_lhs_pool,
    load_lhs_pool_status,
    make_batch_id,
    reconcile_existing_runs,
    select_lhs_samples,
    specs_generated_dir,
    status_row_from_run_summary,
    sync_lhs_pool_entry,
    update_sample_status,
    write_lhs_pool_status,
    write_lhs_pool_with_backup,
    write_per_sample_spec,
)
from v2_b3_m4_lhs_production_batch import run_production_batch  # noqa: E402
from v2_b3_m4_mesh_profile_lib import (  # noqa: E402
    MESH_PROFILE_REFERENCE,
    MeshProfileError,
    resolve_mesh_profile,
)
from v2_b3_m4_production_contracts import DATASET_VERSION, is_strict_production_mode  # noqa: E402
from v2_b3_m4_rom_fom_compare_lib import (  # noqa: E402
    DEFAULT_MAX_MATCH_DISTANCE_HZ,
    DEFAULT_ROM_NEV,
    sync_lhs_pool_rom_fields,
)
from v2_b3_m4_production_control import (  # noqa: E402
    clear_stop_after_current,
    is_stop_after_current_requested,
    request_stop_after_current,
    stop_after_current_path,
)
from v2_b3_m4_shared_export import detect_shared_root  # noqa: E402
from v2_b3_m4_worker_run_lib import detect_repo_root, rel, utc_now  # noqa: E402
from v2_b3_petsc_util import write_json_atomic  # noqa: E402

try:
    from compact_completed_m4_runs import compact_runs_for_samples  # noqa: E402
except ImportError:
    compact_runs_for_samples = None  # type: ignore[misc, assignment]

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
            "--max-samples 10 --workers 3 --execute --continue-on-fail\n\n"
            "Reconcile existing runs without re-solving:\n"
            "  python FEM/.../run_m4_production_pipeline.py --lhs-json ROM/classic/lhs_pool.json "
            "--reconcile-existing-runs"
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
    parser.add_argument(
        "--reconcile-existing-runs",
        action="store_true",
        help="Scan run trees, repair freeze/terminal, update lhs_pool.json + sidecar (no workers).",
    )
    parser.add_argument(
        "--repair-stale-running",
        action="store_true",
        help="With --reconcile-existing-runs: repair terminal_status RUNNING -> LPROD_CHECKPOINT_READY when safe.",
    )
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
        help="Skip samples with COMPLETED status in lhs_pool.json (default: true).",
    )
    parser.add_argument(
        "--no-skip-completed",
        action="store_false",
        dest="skip_completed",
        help="Include samples already marked COMPLETED (use with --force to re-run).",
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
    parser.add_argument(
        "--shared-root",
        type=Path,
        default=None,
        help="Shared export root (default: auto-detect /media/sf_gmar).",
    )
    parser.add_argument(
        "--request-stop",
        action="store_true",
        help="Request graceful stop after the current sample finishes (writes control file).",
    )
    parser.add_argument(
        "--clear-stop",
        action="store_true",
        help="Clear STOP_AFTER_CURRENT_SAMPLE before/without running a batch.",
    )
    parser.add_argument(
        "--run-rom-prepredict",
        action="store_true",
        help="Run ROM online prediction before each sample FOM pipeline.",
    )
    parser.add_argument(
        "--run-rom-compare",
        action="store_true",
        help="Compare ROM vs M4 FOM after AGGREGATION_PASS (frequency matching).",
    )
    parser.add_argument(
        "--rom-nonblocking",
        action="store_true",
        default=True,
        help="ROM failures do not fail FOM sample (default: true).",
    )
    parser.add_argument(
        "--rom-blocking",
        action="store_false",
        dest="rom_nonblocking",
        help="Propagate ROM failures as sample failures (not recommended).",
    )
    parser.add_argument(
        "--rom-max-match-distance-hz",
        type=float,
        default=DEFAULT_MAX_MATCH_DISTANCE_HZ,
        help="Greedy ROM/FOM frequency match tolerance in Hz.",
    )
    parser.add_argument(
        "--rom-nev",
        type=int,
        default=DEFAULT_ROM_NEV,
        help="ROM modes to request (0 = all basis modes).",
    )
    parser.add_argument(
        "--compact-after-sample",
        action="store_true",
        default=False,
        help="After each completed sample (post ROM compare), delete heavy artifacts (no archive).",
    )
    parser.add_argument(
        "--compact-after-batch",
        action="store_true",
        default=False,
        help="After batch completes, compact completed samples (applies --compact-keep-full-latest).",
    )
    parser.add_argument(
        "--compact-keep-full-latest",
        type=int,
        default=0,
        help="Keep N most recent completed samples fully local (batch-end compaction only).",
    )
    parser.add_argument(
        "--compact-keep-full-samples",
        default="",
        help="Comma-separated samples to never compact (e.g. sample_000,sample_001).",
    )
    parser.add_argument(
        "--compact-nonblocking",
        action="store_true",
        default=True,
        help="Compaction failure does not stop batch / fail FOM sample (default).",
    )
    parser.add_argument(
        "--compact-blocking",
        action="store_false",
        dest="compact_nonblocking",
        help="Stop batch if compaction fails.",
    )
    parser.add_argument(
        "--strict-production",
        action="store_true",
        default=None,
        help="Strict fail-fast for m4_geometry_corrected_v1 (default: on for corrected dataset).",
    )
    parser.add_argument(
        "--no-strict-production",
        action="store_false",
        dest="strict_production",
        help="Disable strict production (not allowed for corrected-dataset FOM).",
    )
    parser.add_argument(
        "--isolated-subprocess",
        action="store_true",
        default=None,
        help="Run each sample in a fresh Python subprocess (default: on in strict mode).",
    )
    parser.add_argument(
        "--no-isolated-subprocess",
        action="store_false",
        dest="isolated_subprocess",
        help="Run samples in-process (not recommended for corrected production).",
    )
    parser.add_argument(
        "--mesh-profile",
        choices=("reference", "rom"),
        default=MESH_PROFILE_REFERENCE,
        help="Production mesh profile (default: reference = full fidelity mesh).",
    )
    parser.add_argument(
        "--dataset-version",
        help="Dataset version paired with --mesh-profile (default: canonical per profile).",
    )
    parser.add_argument(
        "--target-plan-file",
        type=Path,
        help="Validation-only: explicit frozen target plan JSON (SHA256 recorded; same sample only).",
    )
    return parser.parse_args(argv)


def _parse_compact_keep_full_samples(text: str) -> List[str]:
    return [s.strip() for s in str(text).split(",") if s.strip()]


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
    mesh_profile: str,
    dataset_version: Optional[str],
    target_plan_file: Optional[Path],
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
                mesh_profile=mesh_profile,
                dataset_version=dataset_version,
            )
        )
    target_rel: Optional[str] = None
    if target_plan_file is not None:
        target_rel = rel(target_plan_file if target_plan_file.is_absolute() else repo_root / target_plan_file, repo_root=repo_root)
    batch_spec = build_lhs_batch_spec(
        pool=pool,
        samples=sample_entries,
        batch_id=batch_id,
        lhs_source_path=lhs_source_path,
        run_id_suffix=run_id_suffix,
        exclude_reference=exclude_reference,
        mesh_profile=mesh_profile,
        dataset_version=dataset_version,
        target_plan_file=target_rel,
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


def _sync_lhs_running(pool: Dict[str, Any], *, sample_id: str, run_id: str, batch_id: str) -> None:
    sync_lhs_pool_entry(
        pool,
        sample_id=sample_id,
        patch={
            "status": LHS_RUNNING,
            "last_run_id": run_id,
            "last_batch_id": batch_id,
            "last_started_at": utc_now(),
            "last_error": None,
            "error": None,
        },
    )


def _sync_lhs_from_finish(
    pool: Dict[str, Any],
    *,
    row: Dict[str, Any],
    batch_id: str,
) -> None:
    sid = str(row["sample_id"])
    outcome = str(row.get("outcome") or "fail")
    patch = lhs_pool_entry_patch_from_run(
        run_id=str(row.get("run_id") or ""),
        run_dir=str(row.get("run_root_abs") or row.get("run_root") or ""),
        batch_id=batch_id,
        outcome=outcome,
        summary=row,
        elapsed_s=float(row.get("elapsed_s") or 0.0),
        started_at=row.get("started_at"),
        error_message=row.get("error_message"),
    )
    sync_lhs_pool_entry(pool, sample_id=sid, patch=patch)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    if args.request_stop and args.clear_stop:
        print("error: use only one of --request-stop / --clear-stop", file=sys.stderr)
        return 2
    if args.reconcile_existing_runs:
        if args.execute or args.dry_run:
            print("error: --reconcile-existing-runs cannot combine with --execute/--dry-run", file=sys.stderr)
            return 2
    elif args.dry_run and args.execute:
        print("error: use --dry-run or --execute, not both", file=sys.stderr)
        return 2
    elif (
        not args.dry_run
        and not args.execute
        and not args.reconcile_existing_runs
        and not args.request_stop
        and not args.clear_stop
    ):
        print(
            "error: specify --dry-run, --execute, --reconcile-existing-runs, "
            "--request-stop, or --clear-stop",
            file=sys.stderr,
        )
        return 2
    if not args.reconcile_existing_runs and args.max_samples < 1:
        print("error: --max-samples must be >= 1", file=sys.stderr)
        return 2
    if args.workers < 1:
        print("error: --workers must be >= 1", file=sys.stderr)
        return 2
    try:
        mesh_resolved = resolve_mesh_profile(
            mesh_profile=args.mesh_profile,
            dataset_version=args.dataset_version,
        )
    except MeshProfileError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.include_reference and args.exclude_reference:
        print("error: use only one of --include-reference / --exclude-reference", file=sys.stderr)
        return 2
    if args.compact_after_sample and args.compact_after_batch:
        print(
            "warning: --compact-after-sample uses keep-full-samples only at sample time; "
            "--compact-after-batch applies keep-full-latest at batch end",
            flush=True,
        )
    if (args.compact_after_sample or args.compact_after_batch) and args.dry_run:
        print("warning: compaction flags ignored during --dry-run", flush=True)

    repo_root = detect_repo_root(SCRIPT_DIR)

    if args.target_plan_file and not args.dry_run and not args.reconcile_existing_runs:
        tp = args.target_plan_file if args.target_plan_file.is_absolute() else repo_root / args.target_plan_file
        if not tp.is_file():
            print(f"error: --target-plan-file not found: {tp}", file=sys.stderr)
            return 2

    if args.request_stop:
        path = request_stop_after_current(repo_root)
        print(f"stop_requested=true path={rel(path, repo_root=repo_root)}")
        print("Current sample (if any) will finish workers/aggregation/freeze/export before exit.")
        if not args.clear_stop and not args.execute and not args.dry_run and not args.reconcile_existing_runs:
            return 0

    if args.clear_stop:
        cleared = clear_stop_after_current(repo_root)
        print(f"stop_cleared={str(cleared).lower()} path={rel(stop_after_current_path(repo_root), repo_root=repo_root)}")
        if not args.execute and not args.dry_run and not args.reconcile_existing_runs and not args.request_stop:
            return 0

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
    lhs_rel = rel(lhs_path, repo_root=repo_root)
    status_path = lhs_pool_status_path(repo_root)
    index_path = lhs_runs_index_path(repo_root)

    if args.reconcile_existing_runs:
        shared_root = detect_shared_root(args.shared_root)
        if shared_root is None:
            print("warning: shared export skipped during reconcile: shared root not found", flush=True)
        else:
            print(f"shared_root={rel(shared_root, repo_root=repo_root)}", flush=True)
        report = reconcile_existing_runs(
            repo_root=repo_root,
            pool=pool,
            lhs_path=lhs_path,
            run_id_suffix=run_id_suffix,
            repair_freeze=True,
            repair_stale_running=bool(args.repair_stale_running),
            shared_root=shared_root,
        )
        status_doc = load_lhs_pool_status(
            status_path,
            lhs_path=lhs_path,
            run_id_suffix=run_id_suffix,
            repo_root=repo_root,
        )
        for row in report.get("samples") or []:
            if row.get("action") != "reconciled_completed":
                continue
            sid = str(row["sample_id"])
            outcome = str(row.get("outcome") or "pass")
            patch = status_row_from_run_summary(
                sample_id=sid,
                lhs_row_index=int(row.get("lhs_row_index") or 0),
                run_id=str(row.get("run_id") or ""),
                batch_id=report.get("generated_utc", "reconcile"),
                run_root=repo_root / str(row.get("run_root") or ""),
                outcome=outcome,
                elapsed_s=0.0,
                summary=row,
                error_message=row.get("last_error"),
            )
            update_sample_status(status_doc, sample_id=sid, patch=patch)
            append_runs_index_row(index_path, {"event": "reconcile", **patch})
        write_lhs_pool_status(status_path, status_doc)
        out_path = specs_generated_dir(repo_root) / f"reconcile_{utc_now()[:10].replace('-', '')}.json"
        write_json_atomic(out_path, report)
        print(f"reconciled_completed={report.get('reconciled_completed_count')}")
        print(f"freeze_repaired={report.get('freeze_repaired_count')}")
        print(f"stale_running_repaired={report.get('stale_running_repaired_count')}")
        print(f"lhs_pool={lhs_rel}")
        print(f"report={rel(out_path, repo_root=repo_root)}")
        return 0

    batch_id = args.batch_id or make_batch_id()
    exclude_reference = bool(args.exclude_reference) and not bool(args.include_reference)

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
        mesh_profile=mesh_resolved.mesh_profile,
        dataset_version=mesh_resolved.dataset_version,
        target_plan_file=args.target_plan_file,
    )

    print(f"batch_id={batch_id}")
    print(f"mesh_profile={mesh_resolved.mesh_profile}")
    print(f"mesh_level_id={mesh_resolved.mesh_level_id}")
    print(f"dataset_version={mesh_resolved.dataset_version}")
    print(f"effective_controls_m={mesh_resolved.effective_controls_m}")
    print(f"lhs_json={lhs_rel}")
    print(f"spec_path={rel(spec_path, repo_root=repo_root)}")
    print(f"selected_count={len(selection)} skipped_count={len(skipped)}")
    for row in selection:
        print(f"  {row['sample_id']} -> {row['run_id']} (lhs_index={row['lhs_row_index']})")

    if args.dry_run:
        from v2_mesh_convergence_common import mesh_path  # noqa: WPS433

        example_sid = str(selection[0]["sample_id"]) if selection else "sample_000"
        mesh_out = mesh_path(mesh_resolved.mesh_level_id, example_sid)
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
            **mesh_resolved.provenance_fields(),
            "expected_mesh_output_path": rel(mesh_out, repo_root=repo_root),
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

    if is_stop_after_current_requested(repo_root):
        print(
            "warning: STOP_AFTER_CURRENT_SAMPLE is already set; "
            "batch will not start new samples until --clear-stop",
            flush=True,
        )
        return 0

    shared_root = detect_shared_root(args.shared_root)
    if shared_root is None:
        print("warning: shared export skipped: shared root not found", flush=True)
    else:
        print(f"shared_root={rel(shared_root, repo_root=repo_root)}", flush=True)

    def _on_sample_start(sid: str, run_id: str, lhs_row_index: int) -> None:
        _sync_lhs_running(pool, sample_id=sid, run_id=run_id, batch_id=batch_id)
        write_lhs_pool_with_backup(lhs_path, pool)
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
        _sync_lhs_from_finish(pool, row=row, batch_id=batch_id)
        rom_patch = row.get("rom_lhs_patch")
        if isinstance(rom_patch, dict) and rom_patch:
            sync_lhs_pool_rom_fields(pool, sample_id=sid, lhs_patch=rom_patch, lhs_path=lhs_path)
        else:
            write_lhs_pool_with_backup(lhs_path, pool)
        status = STATUS_PASS if outcome in ("pass", "reused_complete", OUTCOME_PASS_FREEZE_WARNING) else STATUS_FAIL
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
    compact_keep_full_samples = set(_parse_compact_keep_full_samples(args.compact_keep_full_samples))

    pool_ds = str(pool.get("dataset_version") or DATASET_VERSION)
    strict_production = (
        bool(args.strict_production)
        if args.strict_production is not None
        else is_strict_production_mode(dataset_version=pool_ds)
    )
    isolated_subprocess = (
        bool(args.isolated_subprocess)
        if args.isolated_subprocess is not None
        else strict_production
    )
    compact_after_sample = bool(args.compact_after_sample) or strict_production
    compact_nonblocking = bool(args.compact_nonblocking) and not strict_production
    if strict_production and args.execute and not args.dry_run:
        print(
            f"strict_production=1 isolated_subprocess={int(isolated_subprocess)} "
            f"compact_after_sample=1 compact_blocking={int(not compact_nonblocking)}",
            flush=True,
        )
        if args.strict_production is False:
            print("error: --no-strict-production not allowed for corrected dataset", file=sys.stderr)
            return 2

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
        shared_root=shared_root,
        run_rom_prepredict=bool(args.run_rom_prepredict),
        run_rom_compare=bool(args.run_rom_compare),
        rom_nonblocking=bool(args.rom_nonblocking),
        rom_nev=int(args.rom_nev),
        rom_max_match_distance_hz=float(args.rom_max_match_distance_hz),
        pool=pool,
        compact_after_sample=compact_after_sample and not bool(args.compact_after_batch),
        compact_keep_full_samples=compact_keep_full_samples,
        compact_nonblocking=compact_nonblocking,
        isolated_subprocess=isolated_subprocess,
        strict_production=strict_production,
        mesh_profile=mesh_resolved.mesh_profile,
        dataset_version=mesh_resolved.dataset_version,
        target_plan_file=args.target_plan_file,
    )

    if (
        args.compact_after_batch
        and args.execute
        and not args.dry_run
        and compact_runs_for_samples is not None
    ):
        completed_rows = list(summary.get("completed") or [])
        sample_specs = [(str(r["sample_id"]), str(r["run_id"])) for r in completed_rows]
        rows_by_sid = {str(r["sample_id"]): r for r in completed_rows}
        compact_summary = compact_runs_for_samples(
            repo_root=repo_root,
            pool=pool,
            sample_specs=sample_specs,
            keep_full_samples=sorted(compact_keep_full_samples),
            keep_full_latest=int(args.compact_keep_full_latest),
            production_rows_by_sid=rows_by_sid,
            run_rom_compare=bool(args.run_rom_compare),
            production_trigger=True,
        )
        summary["compaction"] = compact_summary
        summary["compaction_runtime_s"] = compact_summary.get("compaction_runtime_s")
        summary["compaction_status"] = compact_summary.get("compaction_status")
        summary["compaction_bytes_freed"] = compact_summary.get("compaction_bytes_freed")
        summary["compaction_sample_count"] = compact_summary.get("compaction_sample_count")
        summary["compaction_failed_count"] = compact_summary.get("compaction_failed_count")
        if not args.compact_nonblocking and int(compact_summary.get("compaction_failed_count") or 0) > 0:
            summary["failed_count"] = int(summary.get("failed_count") or 0) + int(
                compact_summary.get("compaction_failed_count") or 0
            )

    write_json_atomic(specs_generated_dir(repo_root) / f"{batch_id}_summary.json", summary)
    print(
        f"completed={summary['completed_count']} failed={summary['failed_count']} "
        f"skipped={summary['skipped_count']}"
    )
    if summary.get("compaction_status") and summary.get("compaction_status") != "not_run":
        print(
            f"compaction_status={summary.get('compaction_status')} "
            f"compaction_sample_count={summary.get('compaction_sample_count')} "
            f"compaction_bytes_freed={summary.get('compaction_bytes_freed')} "
            f"compaction_runtime_s={summary.get('compaction_runtime_s')}",
            flush=True,
        )
    print(f"lhs_pool={lhs_rel}")
    print(f"status={rel(status_path, repo_root=repo_root)}")
    print(f"index={rel(index_path, repo_root=repo_root)}")
    return 1 if summary.get("failed_count") else 0


if __name__ == "__main__":
    raise SystemExit(main())
