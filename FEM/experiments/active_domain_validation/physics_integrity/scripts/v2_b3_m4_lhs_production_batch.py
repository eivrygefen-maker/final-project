#!/usr/bin/env python3
"""M4 production LHS batch runner — scout → L_prod → workers → aggregation (+ freeze)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_m4_lhs_pool_bridge import classify_sample_outcome  # noqa: E402
from v2_b3_m4_shared_export import try_export_sample_to_shared  # noqa: E402
from v2_b3_m4_run_one_sample import (  # noqa: E402
    REFERENCE_SAMPLE_ID,
    run_pipeline,
)
from v2_b3_m4_small_batch_dry_run import (  # noqa: E402
    _build_sample_plan,
    _classify_run_status,
    _load_batch_spec,
)
from v2_b3_m4_runtime_provenance import collect_m4_runtime_provenance  # noqa: E402
from v2_b3_m4_worker_run_lib import detect_repo_root, load_json, rel, utc_now  # noqa: E402
from v2_b3_petsc_util import write_json_atomic  # noqa: E402

PIPELINE_RUNS = SCRIPT_DIR.parent / "pipeline_runs"
GUITARS_ROOT = PIPELINE_RUNS / "guitars"
AGG_PASS = "AGGREGATION_PASS"
TERMINAL_E2E = "LPROD_WORKERS_AND_AGGREGATION_PASS"


def _frequency_policy(spec: Dict[str, Any]) -> Dict[str, Any]:
    return spec.get("frequency_policy") or {}


def _select_samples(
    spec: Dict[str, Any],
    *,
    start_index: int,
    max_samples: Optional[int],
) -> List[Dict[str, Any]]:
    exclude = set(spec.get("exclude_from_batch") or [])
    rows: List[Dict[str, Any]] = []
    for entry in spec.get("samples") or []:
        sid = str(entry.get("sample_id") or "").strip()
        if not sid or sid in exclude:
            continue
        rows.append(entry)
    if start_index < 0:
        raise ValueError("--start-index must be >= 0")
    rows = rows[start_index:]
    if max_samples is not None:
        if max_samples < 1:
            raise ValueError("--max-samples must be >= 1")
        rows = rows[:max_samples]
    return rows


def _sample_run_root(entry: Dict[str, Any]) -> Path:
    return GUITARS_ROOT / str(entry["sample_id"]) / "runs" / str(entry["run_id"])


def _ensure_run_tree(
    *,
    repo_root: Path,
    spec: Dict[str, Any],
    entry: Dict[str, Any],
    batch_id: str,
    force: bool,
) -> None:
    run_root = _sample_run_root(entry)
    status = _classify_run_status(run_root)
    if status == "already_complete_reuse" and not force:
        return
    if status in ("planned_new_run", "resume_possible", "requires_review") or force:
        _build_sample_plan(
            repo_root=repo_root,
            spec=spec,
            entry=entry,
            batch_id=batch_id,
            force=force or status == "planned_new_run",
        )


def _read_sample_summary(run_root: Path, *, workers_requested: int = 1) -> Dict[str, Any]:
    manifest_path = run_root / "pipeline_run_manifest.json"
    manifest = load_json(manifest_path) if manifest_path.is_file() else {}
    agg_path = run_root / "aggregation" / "aggregation_result.json"
    agg: Dict[str, Any] = {}
    if agg_path.is_file():
        try:
            agg = load_json(agg_path)
        except (OSError, ValueError, json.JSONDecodeError):
            agg = {}
    freeze_manifest = run_root / "freeze" / "sample_e2e_run_manifest.json"
    if not freeze_manifest.is_file():
        freeze_manifest = run_root / "freeze" / "first_end_to_end_run_manifest.json"
    prov_path = run_root / "m4_sample_runtime_provenance.json"
    prov: Dict[str, Any] = {}
    if prov_path.is_file():
        try:
            prov = load_json(prov_path)
        except (OSError, ValueError, json.JSONDecodeError):
            prov = {}
    if not prov:
        try:
            prov = collect_m4_runtime_provenance(
                run_root=run_root, workers_requested=workers_requested
            )
        except Exception:
            prov = {}

    rt_path = run_root / "aggregation" / "runtime_summary.json"
    runtime: Dict[str, Any] = {}
    if rt_path.is_file():
        try:
            runtime = load_json(rt_path)
        except (OSError, ValueError, json.JSONDecodeError):
            runtime = {}

    ms_path = run_root / "aggregation" / "modes_summary.json"
    modes_summary: Dict[str, Any] = {}
    if ms_path.is_file():
        try:
            modes_summary = load_json(ms_path)
        except (OSError, ValueError, json.JSONDecodeError):
            modes_summary = {}
    audio_summary = modes_summary.get("audio_coupling_summary") or {}

    return {
        "terminal_status": manifest.get("terminal_status"),
        "aggregation_status": agg.get("status"),
        "planned_chunks": agg.get("planned_chunk_count"),
        "completed_chunks": agg.get("completed_chunk_count"),
        "missing_chunks": agg.get("missing_chunk_count"),
        "failed_chunks": agg.get("failed_chunk_count"),
        "raw_modes": agg.get("raw_mode_count") or prov.get("raw_mode_count"),
        "deduped_modes": agg.get("deduped_mode_count") or prov.get("deduped_mode_count"),
        "final_aggregation_ready": agg.get("final_aggregation_ready"),
        "pipeline_version": prov.get("pipeline_version") or runtime.get("pipeline_version"),
        "model_version": prov.get("model_version") or runtime.get("model_version"),
        "operator_version": prov.get("operator_version") or runtime.get("operator_version"),
        "mesh_level": prov.get("mesh_level") or runtime.get("mesh_level"),
        "target_policy": prov.get("target_policy") or runtime.get("target_policy"),
        "chunk_policy": prov.get("chunk_policy") or runtime.get("chunk_policy"),
        "solver_backend": prov.get("solver_backend") or runtime.get("solver_backend"),
        "workers_requested": prov.get("workers_requested") or runtime.get("workers_requested"),
        "workers_actual_parallel": prov.get("workers_actual_parallel")
        or runtime.get("workers_actual_parallel"),
        "worker_thread_settings": prov.get("worker_thread_settings")
        or runtime.get("worker_thread_settings"),
        "stage_wall_times_s": prov.get("stage_wall_times_s") or runtime.get("stage_wall_times_s"),
        "chunk_wall_times": prov.get("chunk_wall_times") or runtime.get("chunk_wall_times"),
        "participation_computed_count": prov.get("participation_computed_count")
        or runtime.get("participation_computed_count")
        or modes_summary.get("participation_computed_count"),
        "audio_coupling_computed_count": (
            prov.get("audio_coupling_computed_count")
            or runtime.get("audio_coupling_computed_count")
            or modes_summary.get("audio_coupling_computed_count")
            or audio_summary.get("audio_coupling_computed_count")
        ),
        "dominant_region_counts": prov.get("dominant_region_counts")
        or runtime.get("dominant_region_counts"),
        "freeze_manifest": rel(freeze_manifest, repo_root=detect_repo_root(SCRIPT_DIR))
        if freeze_manifest.is_file()
        else None,
    }


def _run_dry_run_batch(
    *,
    repo_root: Path,
    spec_path: Path,
    spec: Dict[str, Any],
    batch_id: str,
    samples: List[Dict[str, Any]],
    workers: int,
    force: bool,
) -> Dict[str, Any]:
    fp = _frequency_policy(spec)
    rows: List[Dict[str, Any]] = []
    for entry in samples:
        _ensure_run_tree(
            repo_root=repo_root,
            spec=spec,
            entry=entry,
            batch_id=batch_id,
            force=force,
        )
        run_root = _sample_run_root(entry)
        rows.append(
            {
                "sample_id": entry["sample_id"],
                "run_id": entry["run_id"],
                "run_root": rel(run_root, repo_root=repo_root),
                "reuse_status": _classify_run_status(run_root),
                "will_execute": False,
                "command_preview": (
                    "python FEM/experiments/active_domain_validation/physics_integrity/scripts/"
                    f"v2_b3_m4_run_one_sample.py --run-dir {rel(run_root, repo_root=repo_root)} "
                    f"--execute --workers {workers} --production-mode "
                    f"--production-samples-json {rel(spec_path, repo_root=repo_root)}"
                ),
            }
        )
    return {
        "schema": "m4_lhs_production_batch_plan_v1",
        "will_execute": False,
        "generated_utc": utc_now(),
        "batch_id": batch_id,
        "spec_path": rel(spec_path, repo_root=repo_root),
        "sample_count": len(rows),
        "samples": rows,
        "safety": {
            "no_stage_c": True,
            "no_rich_modal_export": True,
            "no_audio_stk": True,
            "no_cleanup": True,
            "runtime_not_committed": True,
        },
    }


def run_production_batch(
    *,
    repo_root: Path,
    spec_path: Path,
    batch_id: str,
    samples: List[Dict[str, Any]],
    spec: Dict[str, Any],
    workers: int,
    execute: bool,
    continue_on_fail: bool,
    force: bool,
    stop_after: Optional[str],
    resume: bool,
    force_stages: Optional[Set[str]] = None,
    production_mode: bool = True,
    exclude_reference: bool = False,
    allow_reference_mutation: bool = False,
    skip_completed: bool = True,
    lhs_index_by_sid: Optional[Mapping[str, int]] = None,
    on_sample_start: Optional[Callable[[str, str, int], None]] = None,
    on_sample_finish: Optional[Callable[[Dict[str, Any]], None]] = None,
    shared_root: Optional[Path] = None,
) -> Dict[str, Any]:
    fp = _frequency_policy(spec)
    band = fp.get("band_hz", [60.0, 550.0])
    batch_dir = PIPELINE_RUNS / "batches" / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    log_path = batch_dir / "batch_execution.log"

    t_batch = time.perf_counter()
    completed: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    for entry in samples:
        sid = str(entry["sample_id"])
        rid = str(entry["run_id"])
        run_root = _sample_run_root(entry)
        reuse = _classify_run_status(run_root)

        if sid == REFERENCE_SAMPLE_ID and exclude_reference and not allow_reference_mutation:
            skipped.append(
                {
                    "sample_id": sid,
                    "run_id": rid,
                    "reason": "frozen_reference_excluded",
                }
            )
            print(f"[skip] {sid}: reference excluded (--exclude-reference)", flush=True)
            continue

        if reuse == "already_complete_reuse" and skip_completed and not force:
            summary = _read_sample_summary(run_root, workers_requested=workers)
            row = {
                "sample_id": sid,
                "run_id": rid,
                "lhs_row_index": int(
                    (lhs_index_by_sid or {}).get(sid) or entry.get("lhs_row_index") or 0
                ),
                "run_root": rel(run_root, repo_root=repo_root),
                "run_root_abs": str(run_root.resolve()),
                "outcome": "reused_complete",
                "elapsed_s": 0.0,
                **summary,
            }
            export_manifest, export_warn = try_export_sample_to_shared(
                run_root=run_root,
                sample_id=sid,
                run_id=rid,
                shared_root=shared_root,
                repo_root=repo_root,
            )
            if export_manifest:
                row["shared_export"] = export_manifest
            if export_warn:
                row["shared_export_warning"] = export_warn
                print(f"[warn] {sid}: {export_warn}", flush=True)
            completed.append(row)
            if on_sample_finish is not None:
                on_sample_finish(row)
            print(f"[skip] {sid}: already complete (reuse)", flush=True)
            continue

        if not execute:
            continue

        _ensure_run_tree(
            repo_root=repo_root,
            spec=spec,
            entry=entry,
            batch_id=batch_id,
            force=force,
        )

        lhs_row_index = int(
            (lhs_index_by_sid or {}).get(sid)
            or entry.get("lhs_row_index")
            or 0
        )
        if on_sample_start is not None:
            on_sample_start(sid, rid, lhs_row_index)

        print(f"[run] {sid} / {rid} ...", flush=True)
        t0 = time.perf_counter()
        stage_force = force and not resume
        rc = run_pipeline(
            repo_root=repo_root,
            run_root=run_root.resolve(),
            workers=workers,
            force=stage_force,
            force_stages=force_stages,
            execute=True,
            stop_after=stop_after,
            m45_batch_mode=False,
            m45_batch_spec=spec_path,
            production_mode=production_mode,
            production_samples_json=spec_path,
            allow_unlisted_sample=not production_mode,
            allow_reference_mutation=allow_reference_mutation,
            freq_min=float(band[0]),
            freq_max=float(band[1]),
            scout_spacing=float(fp.get("scout_spacing_hz", 7.5)),
            scout_half_width=float(fp.get("scout_half_width_hz", 3.75)),
            zone_dense=float(fp.get("zone_spacing_hz", {}).get("ZONE_1_dense", 6.0)),
            zone_medium=float(fp.get("zone_spacing_hz", {}).get("ZONE_2_medium", 9.0)),
            zone_sparse=float(fp.get("zone_spacing_hz", {}).get("ZONE_3_sparse", 12.5)),
        )
        elapsed = time.perf_counter() - t0
        summary = _read_sample_summary(run_root, workers_requested=workers)
        row = {
            "sample_id": sid,
            "run_id": rid,
            "lhs_row_index": lhs_row_index,
            "run_root": rel(run_root, repo_root=repo_root),
            "run_root_abs": str(run_root.resolve()),
            "return_code": rc,
            "elapsed_s": round(elapsed, 2),
            "started_at": None,
            **summary,
        }
        outcome, err_msg = classify_sample_outcome(return_code=rc, summary=summary)
        row["outcome"] = outcome
        if err_msg:
            row["error_message"] = err_msg
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(
                f"{utc_now()} sample={sid} rc={rc} outcome={outcome} elapsed_s={elapsed:.1f}\n"
            )

        if on_sample_finish is not None:
            on_sample_finish(row)

        if outcome in ("pass", "reused_complete", "pass_freeze_warning"):
            export_manifest, export_warn = try_export_sample_to_shared(
                run_root=run_root,
                sample_id=sid,
                run_id=rid,
                shared_root=shared_root,
                repo_root=repo_root,
            )
            if export_manifest:
                row["shared_export"] = export_manifest
            if export_warn:
                row["shared_export_warning"] = export_warn
                print(f"[warn] {sid}: {export_warn}", flush=True)
            elif export_manifest and export_manifest.get("shared_plot_path"):
                print(f"[export] {sid}: {export_manifest.get('shared_plot_path')}", flush=True)
            completed.append(row)
            label = "AGGREGATION_PASS" if outcome == "pass" else outcome.upper()
            print(f"[pass] {sid}: {label} elapsed_s={elapsed:.1f}", flush=True)
        else:
            failed.append(row)
            print(f"[fail] {sid}: rc={rc} aggregation={summary.get('aggregation_status')}", flush=True)
            if not continue_on_fail:
                print("error: stopping batch (--continue-on-fail not set)", file=sys.stderr)
                break

    batch_summary = {
        "schema": "m4_lhs_production_batch_summary_v1",
        "generated_utc": utc_now(),
        "will_execute": execute,
        "batch_id": batch_id,
        "batch_dir": rel(batch_dir, repo_root=repo_root),
        "spec_path": rel(spec_path, repo_root=repo_root),
        "elapsed_s": round(time.perf_counter() - t_batch, 2),
        "pipeline_version": "M4 production",
        "model_version": "V2",
        "operator_version": "B3",
        "mesh_level": "L_prod",
        "target_policy": "adaptive_lprod_zones_v1",
        "chunk_policy": "lprod_target_plan_fcfs",
        "solver_backend": "mkl_pardiso",
        "workers": workers,
        "workers_requested": workers,
        "continue_on_fail": continue_on_fail,
        "force": force,
        "resume": resume,
        "stop_after": stop_after,
        "completed_count": len(completed),
        "failed_count": len(failed),
        "skipped_count": len(skipped),
        "completed": completed,
        "failed": failed,
        "skipped": skipped,
    }
    write_json_atomic(batch_dir / "batch_execution_summary.json", batch_summary)
    return batch_summary


_BATCH_HINT = (
    "Tip: for LHS pool runs, prefer run_m4_production_pipeline.py "
    "(auto specs + status index from ROM/classic/lhs_pool.json)."
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="M4 production LHS batch: scout → adaptive L_prod → workers → aggregation."
    )
    parser.add_argument("--samples-json", type=Path, required=True, help="Batch spec JSON (samples[]).")
    parser.add_argument("--batch-id", help="Override spec batch_id.")
    parser.add_argument("--start-index", type=int, default=0, help="0-based index into samples[] after excludes.")
    parser.add_argument("--max-samples", type=int, help="Limit number of samples to process.")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true", help="Plan batch; ensure run trees; no solvers.")
    parser.add_argument("--execute", action="store_true", help="Run M4 pipeline per sample.")
    parser.add_argument(
        "--continue-on-fail",
        action="store_true",
        help="Continue to next sample after a failure.",
    )
    parser.add_argument("--force", action="store_true", help="Re-run PASS stages / overwrite outputs.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse PASS stages (default); with --force, re-run even if PASS.",
    )
    parser.add_argument(
        "--stop-after",
        choices=("scout", "checkpoint", "workers"),
        help="Stop each sample after this stage (execute mode).",
    )
    parser.add_argument("--force-checkpoint", action="store_true", help="Re-run checkpoint only.")
    parser.add_argument("--force-workers", action="store_true", help="Re-run workers only.")
    parser.add_argument("--force-aggregation", action="store_true", help="Re-run aggregation only.")
    args = parser.parse_args(argv)
    if not any(a in ("-h", "--help") for a in (argv or sys.argv[1:])):
        print(_BATCH_HINT, file=sys.stderr)

    if args.dry_run and args.execute:
        print("error: use --dry-run or --execute, not both", file=sys.stderr)
        return 2
    if not args.dry_run and not args.execute:
        print("error: specify --dry-run or --execute", file=sys.stderr)
        return 2
    if args.workers < 1:
        print("error: --workers must be >= 1", file=sys.stderr)
        return 2

    repo_root = detect_repo_root(SCRIPT_DIR)
    spec_path = args.samples_json if args.samples_json.is_absolute() else repo_root / args.samples_json
    if not spec_path.is_file():
        print(f"error: missing --samples-json: {spec_path}", file=sys.stderr)
        return 2

    try:
        spec = _load_batch_spec(spec_path)
        bid = args.batch_id or str(spec.get("batch_id") or "m4_lhs_production")
        if spec.get("batch_id") and args.batch_id and spec["batch_id"] != args.batch_id:
            raise ValueError(f"--batch-id {args.batch_id!r} does not match spec {spec['batch_id']!r}")
        samples = _select_samples(spec, start_index=args.start_index, max_samples=args.max_samples)
        if not samples:
            raise ValueError("no samples selected (check --start-index / --max-samples / excludes)")
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        plan = _run_dry_run_batch(
            repo_root=repo_root,
            spec_path=spec_path,
            spec=spec,
            batch_id=bid,
            samples=samples,
            workers=args.workers,
            force=bool(args.force),
        )
        batch_dir = PIPELINE_RUNS / "batches" / bid
        batch_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(batch_dir / "batch_execution_plan.json", plan)
        print("will_execute=false")
        print(f"batch_id={bid}")
        print(f"sample_count={plan['sample_count']}")
        for row in plan["samples"]:
            print(f"  {row['sample_id']}: {row['reuse_status']} -> {row['run_root']}")
        print(f"wrote {rel(batch_dir / 'batch_execution_plan.json', repo_root=repo_root)}")
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

    summary = run_production_batch(
        repo_root=repo_root,
        spec_path=spec_path,
        batch_id=bid,
        samples=samples,
        spec=spec,
        workers=args.workers,
        execute=True,
        continue_on_fail=bool(args.continue_on_fail),
        force=bool(args.force),
        stop_after=args.stop_after,
        resume=bool(args.resume),
        force_stages=force_stages,
        production_mode=True,
        exclude_reference=REFERENCE_SAMPLE_ID in set(spec.get("exclude_from_batch") or []),
        allow_reference_mutation=False,
        skip_completed=not bool(args.force),
    )
    print(f"batch_id={bid}")
    print(f"completed={summary['completed_count']} failed={summary['failed_count']} skipped={summary['skipped_count']}")
    print(f"summary={rel(PIPELINE_RUNS / 'batches' / bid / 'batch_execution_summary.json', repo_root=repo_root)}")
    return 1 if summary["failed_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
