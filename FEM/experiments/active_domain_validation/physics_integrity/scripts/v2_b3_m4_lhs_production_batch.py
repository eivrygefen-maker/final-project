#!/usr/bin/env python3
"""M4 production LHS batch runner — scout → L_prod → workers → aggregation (+ freeze)."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_m4_lhs_pool_bridge import (  # noqa: E402
    classify_batch_sample_outcome,
    classify_sample_outcome,
    is_run_usably_complete,
)
from v2_b3_m4_production_control import is_stop_after_current_requested  # noqa: E402
from v2_b3_m4_rom_fom_compare_lib import (  # noqa: E402
    DEFAULT_MAX_MATCH_DISTANCE_HZ,
    DEFAULT_ROM_NEV,
    maybe_run_rom_compare,
    maybe_run_rom_prepredict,
    resolve_sample_context,
)
from v2_b3_m4_rom_shadow_pipeline_lib import (  # noqa: E402
    DEFAULT_RETRAIN_EVERY_N_NEW_SAMPLES,
    RetrainPolicy,
    attempt_register_and_retrain_after_cleanup,
    diagnose_shadow_rom_stages,
    mark_fom_pipeline_started,
    print_shadow_rom_stages,
    prune_rom_directory_to_durable,
    run_shadow_rom_compare_nonblocking,
    run_shadow_rom_prepredict_nonblocking,
)
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

from compact_completed_m4_runs import compact_one_completed_run  # noqa: E402
from v2_b3_m4_physics_identity_lib import count_forbidden_heavy_artifacts  # noqa: E402
from v2_b3_m4_sample_cleanup_barrier import (  # noqa: E402
    PRODUCTION_PASS_OUTCOMES,
    run_sample_cleanup_barrier,
)

PIPELINE_RUNS = SCRIPT_DIR.parent / "pipeline_runs"
GUITARS_ROOT = PIPELINE_RUNS / "guitars"
AGG_PASS = "AGGREGATION_PASS"
TERMINAL_E2E = "LPROD_WORKERS_AND_AGGREGATION_PASS"
RUN_ONE_SAMPLE_SCRIPT = SCRIPT_DIR / "v2_b3_m4_run_one_sample.py"


def _isolated_sample_subprocess_env() -> Dict[str, str]:
    """Fresh interpreter env: strict production cannot be disabled via inherited flags."""
    env = dict(os.environ)
    for key in (
        "B3_ALLOW_CAVITY_MAX_MIC_FALLBACK",
        "B3_DIAGNOSTIC_MIC_FALLBACK_ONLY",
        "B3_REQUIRE_APERTURE_MASK",
    ):
        env.pop(key, None)
    env["M4_STRICT_PRODUCTION"] = "1"
    return env


def _run_pipeline_isolated_subprocess(
    *,
    repo_root: Path,
    run_root: Path,
    spec_path: Path,
    workers: int,
    force: bool,
    force_stages: Optional[Set[str]],
    stop_after: Optional[str],
    allow_reference_mutation: bool,
    freq_min: float,
    freq_max: float,
    scout_spacing: float,
    scout_half_width: float,
    zone_dense: float,
    zone_medium: float,
    zone_sparse: float,
    mesh_profile: Optional[str] = None,
    dataset_version: Optional[str] = None,
    target_plan_file: Optional[Path] = None,
) -> int:
    cmd = [
        sys.executable,
        str(RUN_ONE_SAMPLE_SCRIPT),
        "--run-dir",
        str(run_root),
        "--execute",
        "--workers",
        str(workers),
        "--production-mode",
        "--production-samples-json",
        str(spec_path),
        "--freq-min-hz",
        str(freq_min),
        "--freq-max-hz",
        str(freq_max),
        "--scout-spacing-hz",
        str(scout_spacing),
        "--scout-half-width-hz",
        str(scout_half_width),
        "--zone-spacing-dense-hz",
        str(zone_dense),
        "--zone-spacing-medium-hz",
        str(zone_medium),
        "--zone-spacing-sparse-hz",
        str(zone_sparse),
    ]
    if force:
        cmd.append("--force")
    if allow_reference_mutation:
        cmd.append("--allow-reference-mutation")
    if stop_after:
        cmd.extend(["--stop-after", stop_after])
    if force_stages:
        if "checkpoint" in force_stages:
            cmd.append("--force-checkpoint")
        if "workers" in force_stages:
            cmd.append("--force-workers")
        if "aggregate" in force_stages:
            cmd.append("--force-aggregation")
    if mesh_profile:
        cmd.extend(["--mesh-profile", str(mesh_profile)])
    if dataset_version:
        cmd.extend(["--dataset-version", str(dataset_version)])
    if target_plan_file is not None:
        cmd.extend(["--target-plan-file", str(target_plan_file)])
    proc = subprocess.run(
        cmd,
        cwd=str(repo_root),
        env=_isolated_sample_subprocess_env(),
        check=False,
    )
    return int(proc.returncode)


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


def _guitars_root_for_repo(repo_root: Path) -> Path:
    return (
        repo_root
        / "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars"
    )


def _sample_run_root(entry: Dict[str, Any], *, repo_root: Optional[Path] = None) -> Path:
    root = _guitars_root_for_repo(repo_root) if repo_root is not None else GUITARS_ROOT
    return root / str(entry["sample_id"]) / "runs" / str(entry["run_id"])


def _resolve_run_root_from_row(row: Mapping[str, Any], *, repo_root: Path) -> Path:
    abs_path = row.get("run_root_abs")
    if abs_path:
        return Path(str(abs_path))
    rel_path = row.get("run_root")
    if rel_path:
        candidate = Path(str(rel_path))
        return candidate if candidate.is_absolute() else repo_root / candidate
    return _guitars_root_for_repo(repo_root) / str(row.get("sample_id")) / "runs" / str(row.get("run_id"))


def _ensure_run_tree(
    *,
    repo_root: Path,
    spec: Dict[str, Any],
    entry: Dict[str, Any],
    batch_id: str,
    force: bool,
) -> None:
    run_root = _sample_run_root(entry, repo_root=repo_root)
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


def _assert_compaction_ready_before_cleanup(
    *,
    run_root: Path,
    compact_after_sample: bool,
    compact_blocking: bool,
) -> tuple[bool, List[str]]:
    """Hard gate: compaction manifest present and heavy artifacts removed before cleanup."""
    if not (compact_after_sample and compact_blocking):
        return True, []
    errors: List[str] = []
    manifest_path = run_root / "compaction" / "compaction_manifest.json"
    if not manifest_path.is_file():
        errors.append("missing:compaction/compaction_manifest.json")
    forbidden_count, forbidden_paths = count_forbidden_heavy_artifacts(run_root)
    if forbidden_count > 0:
        errors.append(f"forbidden_heavy_artifacts:{forbidden_paths}")
    return len(errors) == 0, errors


def _run_sample_compaction_for_batch(
    *,
    row: Dict[str, Any],
    repo_root: Path,
    pool: Dict[str, Any],
    compact_after_sample: bool,
    compact_keep_full_samples: Set[str],
    compact_blocking: bool,
) -> bool:
    """Explicit per-sample compaction before cleanup barrier. Returns False to stop batch."""
    if not compact_after_sample:
        return True

    sid = str(row.get("sample_id") or "")
    rid = str(row.get("run_id") or "")
    run_root = _resolve_run_root_from_row(row, repo_root=repo_root)
    outcome = str(row.get("outcome") or "")
    if outcome not in PRODUCTION_PASS_OUTCOMES:
        return True

    compact_out = compact_one_completed_run(
        repo_root=repo_root,
        pool=pool,
        sample_id=sid,
        run_id=rid,
        keep_full=sid in compact_keep_full_samples,
        dry_run=False,
        production_row=row,
        run_rom_compare=False,
        production_trigger=True,
        allow_transitional_lhs=True,
    )
    compact_dict = compact_out.to_dict()
    row["compaction"] = compact_dict
    print(
        f"[compaction] {sid}: status={compact_out.status} "
        f"deleted_bytes={compact_out.deleted_bytes} runtime_s={compact_out.runtime_s}",
        flush=True,
    )

    if not compact_blocking:
        return True

    from v2_b3_m4_finalize_completed_run import (  # noqa: WPS433
        CompactionNotCompletedError,
        require_compaction_completed,
    )

    try:
        require_compaction_completed(
            run_root=run_root,
            compact_out=compact_dict,
            stages={},
        )
    except CompactionNotCompletedError as exc:
        row["compaction_error"] = str(exc)
        print(f"error: compaction blocking failure for {sid}: {exc}", file=sys.stderr)
        return False

    ready, pre_errors = _assert_compaction_ready_before_cleanup(
        run_root=run_root,
        compact_after_sample=True,
        compact_blocking=True,
    )
    if not ready:
        row["compaction_pre_cleanup_errors"] = pre_errors
        print(
            f"error: compaction pre-cleanup gate failed for {sid}: {pre_errors}",
            file=sys.stderr,
        )
        return False
    return True


def _run_sample_post_export_finalization(
    *,
    row: Dict[str, Any],
    repo_root: Path,
    pool: Dict[str, Any],
    compact_after_sample: bool,
    compact_keep_full_samples: Set[str],
    compact_nonblocking: bool,
    run_rom_compare: bool,
    use_shadow_rom: bool,
    strict_production: bool,
) -> bool:
    """Compaction → verify manifest → cleanup barrier. Returns False to stop batch."""
    compact_blocking = bool(strict_production) or not bool(compact_nonblocking)
    sid = str(row.get("sample_id") or "")

    if not _run_sample_compaction_for_batch(
        row=row,
        repo_root=repo_root,
        pool=pool,
        compact_after_sample=bool(compact_after_sample),
        compact_keep_full_samples=compact_keep_full_samples,
        compact_blocking=compact_blocking,
    ):
        return False

    outcome = str(row.get("outcome") or "")
    if outcome not in PRODUCTION_PASS_OUTCOMES:
        return _run_sample_cleanup_barrier_for_batch(
            row=row,
            repo_root=repo_root,
            pool=pool,
            compact_after_sample=bool(compact_after_sample),
            compact_keep_full_samples=compact_keep_full_samples,
            compact_nonblocking=compact_nonblocking,
            run_rom_compare=bool(run_rom_compare) and not bool(use_shadow_rom),
            strict_production=bool(strict_production),
            compaction_already_done=False,
        )

    run_root = _resolve_run_root_from_row(row, repo_root=repo_root)
    ready, pre_errors = _assert_compaction_ready_before_cleanup(
        run_root=run_root,
        compact_after_sample=bool(compact_after_sample),
        compact_blocking=compact_blocking,
    )
    if compact_after_sample and compact_blocking and not ready:
        row["compaction_pre_cleanup_errors"] = pre_errors
        print(
            f"error: refusing cleanup barrier for {sid}; compaction not ready: {pre_errors}",
            file=sys.stderr,
        )
        return False

    compaction_status = str((row.get("compaction") or {}).get("status") or "")
    compaction_already_done = compaction_status in {"completed", "already_compacted"}

    return _run_sample_cleanup_barrier_for_batch(
        row=row,
        repo_root=repo_root,
        pool=pool,
        compact_after_sample=bool(compact_after_sample),
        compact_keep_full_samples=compact_keep_full_samples,
        compact_nonblocking=compact_nonblocking,
        run_rom_compare=bool(run_rom_compare) and not bool(use_shadow_rom),
        strict_production=bool(strict_production),
        compaction_already_done=compaction_already_done,
    )


def _run_sample_cleanup_barrier_for_batch(
    *,
    row: Dict[str, Any],
    repo_root: Path,
    pool: Dict[str, Any],
    compact_after_sample: bool,
    compact_keep_full_samples: Set[str],
    compact_nonblocking: bool,
    run_rom_compare: bool,
    strict_production: bool,
    compaction_already_done: bool = False,
) -> bool:
    """Per-sample cleanup barrier. Returns False to stop batch before the next sample."""
    if not compact_after_sample and not strict_production:
        return True

    sid = str(row.get("sample_id") or "")
    rid = str(row.get("run_id") or "")
    run_root = _resolve_run_root_from_row(row, repo_root=repo_root)
    blocking = bool(strict_production) or not bool(compact_nonblocking)
    keep_full = sid in compact_keep_full_samples or compaction_already_done
    outcome = run_sample_cleanup_barrier(
        repo_root=repo_root,
        run_root=run_root,
        sample_id=sid,
        run_id=rid,
        row=row,
        pool=pool,
        keep_full=keep_full,
        run_rom_compare=bool(run_rom_compare) and not compaction_already_done,
        blocking=blocking,
    )
    row["cleanup_barrier"] = outcome.to_dict()
    if outcome.compaction and not row.get("compaction"):
        row["compaction"] = outcome.compaction
    print(
        f"[cleanup-barrier] {sid}: status={outcome.status} "
        f"forbidden={outcome.forbidden_heavy_artifact_count} "
        f"shared={outcome.shared_sample_artifact_count} "
        f"runtime_s={outcome.runtime_s}",
        flush=True,
    )
    if outcome.status == "failed" and blocking:
        print(
            f"error: cleanup barrier blocking failure for {sid}: {outcome.errors}",
            file=sys.stderr,
        )
        return False
    return True


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
    freeze_manifest = run_root / "freeze" / "freeze_manifest.json"
    if not freeze_manifest.is_file():
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
        run_root = _sample_run_root(entry, repo_root=repo_root)
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
    run_rom_prepredict: bool = False,
    run_rom_compare: bool = False,
    run_rom_shadow: bool = False,
    rom_nonblocking: bool = True,
    rom_nev: int = DEFAULT_ROM_NEV,
    rom_max_match_distance_hz: float = DEFAULT_MAX_MATCH_DISTANCE_HZ,
    rom_retrain_every_n: int = DEFAULT_RETRAIN_EVERY_N_NEW_SAMPLES,
    pool: Optional[Dict[str, Any]] = None,
    compact_after_sample: bool = False,
    compact_keep_full_samples: Optional[Set[str]] = None,
    compact_nonblocking: bool = True,
    isolated_subprocess: bool = False,
    strict_production: bool = False,
    mesh_profile: Optional[str] = None,
    dataset_version: Optional[str] = None,
    target_plan_file: Optional[Path] = None,
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
    stopped_early = False
    stop_reason: Optional[str] = None

    for entry in samples:
        if is_stop_after_current_requested(repo_root):
            stopped_early = True
            stop_reason = "STOP_AFTER_CURRENT_SAMPLE"
            skipped.append(
                {
                    "sample_id": str(entry.get("sample_id")),
                    "run_id": str(entry.get("run_id")),
                    "reason": "not_started_stop_after_current",
                }
            )
            print(
                "[stop] STOP_AFTER_CURRENT_SAMPLE set; not starting further samples",
                flush=True,
            )
            break

        sid = str(entry["sample_id"])
        rid = str(entry["run_id"])
        run_root = _sample_run_root(entry, repo_root=repo_root)
        reuse = _classify_run_status(run_root, production_mode=production_mode)

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
            lhs_row_index = int(
                (lhs_index_by_sid or {}).get(sid) or entry.get("lhs_row_index") or 0
            )
            row = {
                "sample_id": sid,
                "run_id": rid,
                "lhs_row_index": lhs_row_index,
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

            if run_rom_compare and pool is not None and str(summary.get("aggregation_status") or "") == AGG_PASS:
                try:
                    rom_context = resolve_sample_context(
                        pool=pool,
                        sample_id=sid,
                        run_id=rid,
                        run_root=run_root,
                        repo_root=repo_root,
                    )
                    cmp_result = maybe_run_rom_compare(
                        repo_root=repo_root,
                        run_root=run_root,
                        context=rom_context,
                        nev=int(rom_nev),
                        max_match_distance_hz=float(rom_max_match_distance_hz),
                        nonblocking=bool(rom_nonblocking),
                        copy_to_project=True,
                        write_csv=False,
                        rerun_rom_if_missing=True,
                    )
                    row["rom_compare"] = {
                        "status": (cmp_result.get("comparison") or {}).get("status"),
                        "error": cmp_result.get("error"),
                        "paths": cmp_result.get("paths"),
                    }
                    if cmp_result.get("lhs_patch"):
                        row["rom_lhs_patch"] = cmp_result["lhs_patch"]
                except Exception as exc:
                    print(f"[warn] {sid}: ROM compare on reuse failed: {exc}", flush=True)

            if on_sample_finish is not None:
                on_sample_finish(row)

            if not _run_sample_post_export_finalization(
                row=row,
                repo_root=repo_root,
                pool=pool or {},
                compact_after_sample=bool(compact_after_sample),
                compact_keep_full_samples=set(compact_keep_full_samples or ()),
                compact_nonblocking=bool(compact_nonblocking) and not strict_production,
                run_rom_compare=bool(run_rom_compare),
                use_shadow_rom=False,
                strict_production=bool(strict_production),
            ):
                failed.append(row)
                print("error: stopping batch after cleanup barrier failure", file=sys.stderr)
                break

            completed.append(row)
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

        use_shadow_rom = bool(run_rom_shadow)
        use_legacy_rom = (bool(run_rom_prepredict) or bool(run_rom_compare)) and not use_shadow_rom
        rom_context = None
        if (use_shadow_rom or use_legacy_rom) and pool is not None:
            try:
                rom_context = resolve_sample_context(
                    pool=pool,
                    sample_id=sid,
                    run_id=rid,
                    run_root=run_root,
                    repo_root=repo_root,
                )
            except Exception as exc:
                rom_context = None
                print(f"[warn] {sid}: ROM context unavailable: {exc}", flush=True)

        if use_shadow_rom and rom_context is not None:
            prep = run_shadow_rom_prepredict_nonblocking(
                repo_root=repo_root,
                run_root=run_root,
                context=rom_context,
                nev=int(rom_nev),
                dataset_version=str(dataset_version or ""),
            )
            print(
                f"[rom-shadow-prepredict] {sid}: status={prep.get('status')} "
                f"modes={prep.get('predicted_mode_count')}",
                flush=True,
            )
            if prep.get("blocking") and not rom_nonblocking:
                failed.append(
                    {
                        "sample_id": sid,
                        "run_id": rid,
                        "outcome": "fail",
                        "error_message": f"rom_shadow_prepredict:{prep.get('error')}",
                    }
                )
                print(f"error: ROM shadow prepredict blocking failure for {sid}", file=sys.stderr)
                break
        elif run_rom_prepredict and rom_context is not None:
            prep = maybe_run_rom_prepredict(
                repo_root=repo_root,
                run_root=run_root,
                context=rom_context,
                nev=int(rom_nev),
                nonblocking=bool(rom_nonblocking),
            )
            print(
                f"[rom-prepredict] {sid}: status={prep.get('status')} "
                f"modes={len(prep.get('frequencies_hz') or [])}",
                flush=True,
            )

        if use_shadow_rom and rom_context is not None:
            mark_fom_pipeline_started(run_root)

        print(f"[run] {sid} / {rid} ...", flush=True)
        if strict_production:
            print(f"[strict] isolated_subprocess={isolated_subprocess} compact_blocking=True", flush=True)
        t0 = time.perf_counter()
        stage_force = force and not resume
        run_root_resolved = run_root.resolve()
        if isolated_subprocess:
            rc = _run_pipeline_isolated_subprocess(
                repo_root=repo_root,
                run_root=run_root_resolved,
                spec_path=spec_path,
                workers=workers,
                force=stage_force,
                force_stages=force_stages,
                stop_after=stop_after,
                allow_reference_mutation=allow_reference_mutation,
                freq_min=float(band[0]),
                freq_max=float(band[1]),
                scout_spacing=float(fp.get("scout_spacing_hz", 7.5)),
                scout_half_width=float(fp.get("scout_half_width_hz", 3.75)),
                zone_dense=float(fp.get("zone_spacing_hz", {}).get("ZONE_1_dense", 6.0)),
                zone_medium=float(fp.get("zone_spacing_hz", {}).get("ZONE_2_medium", 9.0)),
                zone_sparse=float(fp.get("zone_spacing_hz", {}).get("ZONE_3_sparse", 12.5)),
            )
        else:
            rc = run_pipeline(
                repo_root=repo_root,
                run_root=run_root_resolved,
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
                mesh_profile=mesh_profile,
                dataset_version=dataset_version,
                target_plan_file=target_plan_file,
            )
        elapsed = time.perf_counter() - t0
        summary = _read_sample_summary(run_root, workers_requested=workers)
        sample_doc = entry.get("sample_input")
        if not isinstance(sample_doc, dict) or not str(sample_doc.get("sample_id") or ""):
            try:
                from v2_b3_m4_production_freeze import load_sample_input  # noqa: WPS433

                sample_doc = load_sample_input(run_root)
            except Exception:
                sample_doc = entry if isinstance(entry, dict) else {}
        acceptance: Dict[str, Any] = {}
        if production_mode:
            try:
                from v2_b3_m4_production_contracts import evaluate_production_acceptance  # noqa: WPS433

                acceptance = evaluate_production_acceptance(
                    run_root=run_root,
                    sample_input=sample_doc,
                )
                summary["production_acceptance_pass"] = bool(acceptance.get("acceptance_pass"))
                summary["production_acceptance_failures"] = list(acceptance.get("failures") or [])
            except Exception as exc:
                acceptance = {"acceptance_pass": False, "failures": [f"acceptance_eval_error:{exc}"]}
        row = {
            "sample_id": sid,
            "run_id": rid,
            "lhs_row_index": lhs_row_index,
            "run_root": rel(run_root, repo_root=repo_root),
            "run_root_abs": str(run_root.resolve()),
            "return_code": rc,
            "elapsed_s": round(elapsed, 2),
            "started_at": None,
            "production_acceptance": acceptance,
            **summary,
        }
        require_cleanup_barrier = bool(compact_after_sample) or bool(strict_production)
        require_graph_export = bool(strict_production) and shared_root is not None
        prelim_outcome, _ = classify_sample_outcome(return_code=rc, summary=summary)
        row["outcome"] = prelim_outcome

        rom_shadow_blocking = False
        if (
            use_shadow_rom
            and rom_context is not None
            and is_run_usably_complete(summary)
            and int(rc) == 0
            and str(summary.get("aggregation_status") or "") == AGG_PASS
        ):
            cmp_result = run_shadow_rom_compare_nonblocking(
                repo_root=repo_root,
                run_root=run_root,
                context=rom_context,
                max_match_distance_hz=float(rom_max_match_distance_hz),
            )
            row["rom_shadow_compare"] = cmp_result
            if cmp_result.get("blocking"):
                rom_shadow_blocking = not bool(rom_nonblocking)
                print(f"[warn] {sid}: ROM shadow compare integrity issue: {cmp_result.get('error')}", flush=True)
            else:
                print(
                    f"[rom-shadow-compare] {sid}: status={cmp_result.get('status')} "
                    f"matched={cmp_result.get('matched_mode_count')}",
                    flush=True,
                )

        if rom_shadow_blocking:
            row["outcome"] = "fail"
            row["error_message"] = "rom_shadow_integrity_failure"
            failed.append(row)
            print("error: stopping batch after ROM shadow integrity failure", file=sys.stderr)
            break

        if require_graph_export and is_run_usably_complete(summary) and int(rc) == 0:
            export_manifest, export_warn = try_export_sample_to_shared(
                run_root=run_root,
                sample_id=sid,
                run_id=rid,
                shared_root=shared_root,
                repo_root=repo_root,
                mesh_profile=mesh_profile,
            )
            if export_manifest:
                row["shared_export"] = export_manifest
            if export_warn:
                row["shared_export_warning"] = export_warn
                print(f"[warn] {sid}: {export_warn}", flush=True)
            elif export_manifest and export_manifest.get("exported_png_paths"):
                print(f"[export] {sid}: {export_manifest.get('exported_png_paths')[0]}", flush=True)

        if not _run_sample_post_export_finalization(
            row=row,
            repo_root=repo_root,
            pool=pool or {},
            compact_after_sample=bool(compact_after_sample),
            compact_keep_full_samples=set(compact_keep_full_samples or ()),
            compact_nonblocking=bool(compact_nonblocking) and not strict_production,
            run_rom_compare=bool(run_rom_compare),
            use_shadow_rom=bool(use_shadow_rom),
            strict_production=bool(strict_production),
        ):
            row["outcome"] = "fail"
            row["error_message"] = row.get("compaction_error") or row.get(
                "compaction_pre_cleanup_errors"
            ) or "cleanup_barrier_blocking_failure"
            failed.append(row)
            print("error: stopping batch after cleanup barrier failure", file=sys.stderr)
            break

        cleanup_passed = str((row.get("cleanup_barrier") or {}).get("status") or "") == "completed"
        if use_shadow_rom and rom_context is not None and cleanup_passed:
            reg = attempt_register_and_retrain_after_cleanup(
                repo_root=repo_root,
                run_root=run_root,
                sample_id=sid,
                run_id=rid,
                shape_name=str(rom_context.get("shape_name") or "classic"),
                production_acceptance_pass=bool(summary.get("production_acceptance_pass")),
                policy=RetrainPolicy(retrain_every_n_new_samples=int(rom_retrain_every_n)),
            )
            row["rom_dataset_registration"] = reg
            shadow_stages = reg.get("shadow_stages") or diagnose_shadow_rom_stages(run_root)
            row["rom_shadow_stages"] = shadow_stages
            print_shadow_rom_stages(shadow_stages)
            if reg.get("retrained"):
                print(f"[rom-retrain] {sid}: official model rebuilt", flush=True)
            elif reg.get("registered"):
                print(f"[rom-register] {sid}: added to official ROM dataset", flush=True)
            prune_rom_directory_to_durable(run_root)
        elif use_shadow_rom and rom_context is not None:
            shadow_stages = diagnose_shadow_rom_stages(run_root)
            shadow_stages["dataset_registration_attempted"] = False
            shadow_stages["dataset_registration_status"] = "skipped_cleanup_not_completed"
            row["rom_shadow_stages"] = shadow_stages
            print_shadow_rom_stages(shadow_stages)

        outcome, err_msg = classify_batch_sample_outcome(
            return_code=rc,
            summary=summary,
            cleanup_barrier=row.get("cleanup_barrier"),
            require_cleanup_barrier=require_cleanup_barrier,
            shared_export=row.get("shared_export") if isinstance(row.get("shared_export"), dict) else None,
            require_graph_export=require_graph_export,
        )
        row["outcome"] = outcome
        if err_msg:
            row["error_message"] = err_msg
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(
                f"{utc_now()} sample={sid} rc={rc} outcome={outcome} elapsed_s={elapsed:.1f}\n"
            )

        if outcome == "pass":
            if (
                use_legacy_rom
                and run_rom_compare
                and rom_context is not None
                and str(summary.get("aggregation_status") or "") == AGG_PASS
            ):
                cmp_result = maybe_run_rom_compare(
                    repo_root=repo_root,
                    run_root=run_root,
                    context=rom_context,
                    nev=int(rom_nev),
                    max_match_distance_hz=float(rom_max_match_distance_hz),
                    nonblocking=bool(rom_nonblocking),
                    copy_to_project=True,
                    write_csv=False,
                    rerun_rom_if_missing=not run_rom_prepredict,
                )
                row["rom_compare"] = {
                    "status": (cmp_result.get("comparison") or {}).get("status"),
                    "error": cmp_result.get("error"),
                    "paths": cmp_result.get("paths"),
                    "matched_mode_count": (cmp_result.get("comparison") or {}).get(
                        "matched_mode_count"
                    ),
                    "mean_abs_error_hz": (cmp_result.get("comparison") or {}).get(
                        "mean_abs_error_hz"
                    ),
                }
                if cmp_result.get("lhs_patch"):
                    row["rom_lhs_patch"] = cmp_result["lhs_patch"]
                if cmp_result.get("error"):
                    print(f"[warn] {sid}: ROM compare failed: {cmp_result['error']}", flush=True)
                else:
                    print(
                        f"[rom-compare] {sid}: status={row['rom_compare'].get('status')} "
                        f"matched={row['rom_compare'].get('matched_mode_count')}",
                        flush=True,
                    )

            if on_sample_finish is not None:
                on_sample_finish(row)

            completed.append(row)
            print(f"[pass] {sid}: AGGREGATION_PASS elapsed_s={elapsed:.1f}", flush=True)
        else:
            failed.append(row)
            print(f"[fail] {sid}: rc={rc} aggregation={summary.get('aggregation_status')}", flush=True)
            if not continue_on_fail:
                print("error: stopping batch (--continue-on-fail not set)", file=sys.stderr)
                break

    all_compaction_rows = [
        r.get("compaction")
        for r in (completed + failed)
        if isinstance(r.get("compaction"), dict)
    ]
    compaction_outcomes = [
        c
        for c in all_compaction_rows
        if str(c.get("status") or "") in {"completed", "already_compacted"}
    ]
    barrier_outcomes = [
        r.get("cleanup_barrier") for r in (completed + failed) if isinstance(r.get("cleanup_barrier"), dict)
    ]
    compaction_runtime_s = round(
        sum(float(c.get("runtime_s") or 0.0) for c in compaction_outcomes),
        4,
    )
    compaction_bytes_freed = sum(int(c.get("deleted_bytes") or 0) for c in compaction_outcomes)
    compaction_failed_count = sum(
        1
        for r in (completed + failed)
        if r.get("compaction_error")
        or r.get("compaction_pre_cleanup_errors")
        or (
            isinstance(r.get("compaction"), dict)
            and str(r["compaction"].get("status") or "") in {"failed", "skipped", "planned"}
        )
    )
    compaction_sample_count = len(compaction_outcomes)
    cleanup_barrier_failed_count = sum(
        1 for b in barrier_outcomes if str(b.get("status")) == "failed"
    )
    cleanup_barrier_sample_count = sum(
        1 for b in barrier_outcomes if str(b.get("status")) == "completed"
    )

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
        "mesh_profile": spec.get("mesh_profile"),
        "mesh_level_id": spec.get("mesh_level_id"),
        "mesh_level": spec.get("mesh_level_id") or spec.get("mesh_profile"),
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
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "compaction_runtime_s": compaction_runtime_s,
        "compaction_status": (
            "completed"
            if compact_after_sample
            and compaction_sample_count > 0
            and compaction_failed_count == 0
            else ("partial_failed" if compact_after_sample and compaction_failed_count else "not_run")
        ),
        "compaction_bytes_freed": compaction_bytes_freed,
        "compaction_sample_count": compaction_sample_count,
        "compaction_failed_count": compaction_failed_count,
        "cleanup_barrier_status": (
            "completed"
            if cleanup_barrier_failed_count == 0 and cleanup_barrier_sample_count
            else ("partial_failed" if cleanup_barrier_failed_count else "not_run")
        ),
        "cleanup_barrier_sample_count": cleanup_barrier_sample_count,
        "cleanup_barrier_failed_count": cleanup_barrier_failed_count,
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
