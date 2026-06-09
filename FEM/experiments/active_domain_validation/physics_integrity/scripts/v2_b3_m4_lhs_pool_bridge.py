#!/usr/bin/env python3
"""Bridge ROM/classic LHS pool → M4 production specs + persistent run status index."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from v2_b3_m4_mesh_profile_lib import (  # noqa: E402
    MESH_PROFILE_REFERENCE,
    MeshProfileError,
    MeshProfileResolved,
    apply_mesh_profile_to_sample_input,
    resolve_mesh_profile,
)
from v2_b3_m4_worker_run_lib import detect_repo_root, load_json, rel, utc_now  # noqa: E402
from v2_b3_petsc_util import write_json_atomic  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_RUNS = SCRIPT_DIR.parent / "pipeline_runs"
INDEX_DIR = PIPELINE_RUNS / "index"
SPECS_GENERATED = PIPELINE_RUNS / "specs" / "generated"
GUITARS_ROOT = PIPELINE_RUNS / "guitars"

STATUS_SCHEMA = "lhs_pool_status_v1"
INDEX_ROW_SCHEMA = "lhs_production_run_index_v1"
SAMPLE_SPEC_SCHEMA = "m4_lhs_sample_production_spec_v1"
BATCH_SPEC_SCHEMA = "m4_lhs_production_batch_spec_v1"

PIPELINE_VERSION = "M4 production v1"
DEFAULT_RUN_ID_SUFFIX = "m4prod1"
DEFAULT_BATCH_ID_PREFIX = "lhs_prod_m4"

REFERENCE_SAMPLE_ID = "sample_001"
AGG_PASS = "AGGREGATION_PASS"

# Sidecar status (lowercase legacy + uppercase LHS-aligned)
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_PASS = "pass"
STATUS_FAIL = "fail"
STATUS_SKIPPED = "skipped"

LHS_PENDING = "PENDING"
LHS_RUNNING = "RUNNING"
LHS_COMPLETED = "COMPLETED"
LHS_FAILED = "FAILED"
LHS_FAILED_RETRYABLE = "FAILED_RETRYABLE"

OUTCOME_PASS_FREEZE_WARNING = "pass_freeze_warning"

DEFAULT_FREQUENCY_POLICY: Dict[str, Any] = {
    "band_hz": [60.0, 550.0],
    "scout_spacing_hz": 7.5,
    "scout_half_width_hz": 3.75,
    "zone_spacing_hz": {
        "ZONE_1_dense": 6.0,
        "ZONE_2_medium": 9.0,
        "ZONE_3_sparse": 12.5,
    },
    "chunk_policy_version": "v1_1",
    "workers": 3,
}


def _index_dir(repo_root: Path) -> Path:
    return repo_root / "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/index"


def lhs_pool_status_path(repo_root: Path) -> Path:
    return _index_dir(repo_root) / "lhs_pool_status.json"


def lhs_runs_index_path(repo_root: Path) -> Path:
    return _index_dir(repo_root) / "lhs_production_runs_index.jsonl"


def specs_generated_dir(repo_root: Path) -> Path:
    return (
        repo_root
        / "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/specs/generated"
    )


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_lhs_pool(path: Path) -> Dict[str, Any]:
    data = load_json(path)
    entries = data.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path}: entries[] required")
    return data


def lhs_entry_index(pool: Mapping[str, Any], sample_id: str) -> Optional[int]:
    for i, row in enumerate(pool.get("entries") or []):
        if str(row.get("id")) == sample_id:
            return i
    return None


def build_sample_input(
    *,
    pool: Mapping[str, Any],
    entry: Mapping[str, Any],
    lhs_row_index: int,
    batch_id: str,
    lhs_source_path: str,
    mesh_profile: Optional[str] = None,
    dataset_version: Optional[str] = None,
) -> Dict[str, Any]:
    sid = str(entry["id"])
    params = dict(entry.get("parameters") or {})
    body = {
        "schema": "m4_sample_input_v1",
        "sample_id": sid,
        "shape_name": str(pool.get("shape_name") or "classic"),
        "parameters": params,
        "top_wood_id": params.get("top_wood_id"),
        "back_wood_id": params.get("back_wood_id"),
        "requires_mesh_regeneration": True,
        "lhs_row_index": int(lhs_row_index),
        "lhs_source_path": lhs_source_path,
        "batch_id": batch_id,
        "selection_reason": "lhs_pool_auto",
        "lhs_row_note": f"auto from {lhs_source_path}",
    }
    resolved = resolve_mesh_profile(
        mesh_profile=mesh_profile or MESH_PROFILE_REFERENCE,
        dataset_version=dataset_version,
    )
    return apply_mesh_profile_to_sample_input(body, resolved)


def build_batch_sample_entry(
    *,
    pool: Mapping[str, Any],
    entry: Mapping[str, Any],
    lhs_row_index: int,
    run_id_suffix: str,
    batch_id: str,
    lhs_source_path: str,
    mesh_profile: Optional[str] = None,
    dataset_version: Optional[str] = None,
) -> Dict[str, Any]:
    sid = str(entry["id"])
    run_id = f"{sid}_{run_id_suffix}"
    return {
        "sample_id": sid,
        "run_id": run_id,
        "lhs_source_id": f"lhs_row_{lhs_row_index:03d}",
        "lhs_row_index": int(lhs_row_index),
        "selection_reason": "lhs_pool_auto",
        "sample_input": build_sample_input(
            pool=pool,
            entry=entry,
            lhs_row_index=lhs_row_index,
            batch_id=batch_id,
            lhs_source_path=lhs_source_path,
            mesh_profile=mesh_profile,
            dataset_version=dataset_version,
        ),
    }


def build_lhs_batch_spec(
    *,
    pool: Mapping[str, Any],
    samples: Sequence[Dict[str, Any]],
    batch_id: str,
    lhs_source_path: str,
    run_id_suffix: str,
    exclude_reference: bool = False,
    reference_sample_id: str = REFERENCE_SAMPLE_ID,
    reference_run_id: Optional[str] = None,
    frequency_policy: Optional[Mapping[str, Any]] = None,
    mesh_profile: Optional[str] = None,
    dataset_version: Optional[str] = None,
    target_plan_file: Optional[str] = None,
) -> Dict[str, Any]:
    fp = dict(DEFAULT_FREQUENCY_POLICY)
    if frequency_policy:
        fp.update(frequency_policy)
    exclude: List[str] = []
    if exclude_reference:
        exclude.append(reference_sample_id)
    ref_run = reference_run_id or f"{reference_sample_id}_{run_id_suffix}"
    resolved = resolve_mesh_profile(
        mesh_profile=mesh_profile or MESH_PROFILE_REFERENCE,
        dataset_version=dataset_version,
    )
    body: Dict[str, Any] = {
        "schema": BATCH_SPEC_SCHEMA,
        "batch_id": batch_id,
        "description": f"Auto-generated M4 production batch from {lhs_source_path}",
        "lhs_source_path": lhs_source_path,
        "run_id_suffix": run_id_suffix,
        "pipeline_version": PIPELINE_VERSION,
        "reference_sample_id": reference_sample_id,
        "reference_run_id": ref_run,
        "exclude_from_batch": exclude,
        "frequency_policy": fp,
        "samples": list(samples),
        **resolved.provenance_fields(),
    }
    if target_plan_file:
        body["target_plan_file"] = str(target_plan_file)
    return body


def write_per_sample_spec(
    *,
    repo_root: Path,
    batch_spec: Mapping[str, Any],
    sample_entry: Mapping[str, Any],
    lhs_source_path: str,
) -> Path:
    gen_dir = specs_generated_dir(repo_root)
    gen_dir.mkdir(parents=True, exist_ok=True)
    sid = str(sample_entry["sample_id"])
    run_id = str(sample_entry["run_id"])
    out = gen_dir / f"{run_id}.json"
    body = {
        "schema": SAMPLE_SPEC_SCHEMA,
        "sample_id": sid,
        "run_id": run_id,
        "lhs_row_index": sample_entry.get("lhs_row_index"),
        "lhs_source_id": sample_entry.get("lhs_source_id"),
        "lhs_source_path": lhs_source_path,
        "batch_id": batch_spec.get("batch_id"),
        "pipeline_version": batch_spec.get("pipeline_version"),
        "frequency_policy": batch_spec.get("frequency_policy"),
        "worker_policy": {"workers": (batch_spec.get("frequency_policy") or {}).get("workers", 3)},
        "mesh_profile": batch_spec.get("mesh_profile"),
        "mesh_level_id": batch_spec.get("mesh_level_id"),
        "dataset_version": batch_spec.get("dataset_version"),
        "effective_controls_m": batch_spec.get("effective_controls_m"),
        "target_plan_file": batch_spec.get("target_plan_file"),
        "sample_input": sample_entry.get("sample_input"),
        "generated_utc": utc_now(),
        "result_paths": {
            "run_dir": f"FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars/{sid}/runs/{run_id}",
            "aggregation_dir": f"FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars/{sid}/runs/{run_id}/aggregation",
        },
    }
    write_json_atomic(out, body)
    return out


def load_lhs_pool_status(path: Path, *, lhs_path: Path, run_id_suffix: str, repo_root: Path) -> Dict[str, Any]:
    if path.is_file():
        data = load_json(path)
        if isinstance(data.get("samples"), dict):
            return data
    return {
        "schema": STATUS_SCHEMA,
        "lhs_json": rel(lhs_path, repo_root=repo_root),
        "lhs_pool_sha256": _sha256_file(lhs_path) if lhs_path.is_file() else None,
        "run_id_suffix_default": run_id_suffix,
        "pipeline_version": PIPELINE_VERSION,
        "updated_utc": utc_now(),
        "samples": {},
    }


def get_sample_status(status_doc: Mapping[str, Any], sample_id: str) -> Dict[str, Any]:
    samples = status_doc.get("samples") or {}
    row = samples.get(sample_id)
    if isinstance(row, dict):
        return dict(row)
    return {"sample_id": sample_id, "status": STATUS_PENDING}


def _is_completed_status(row: Mapping[str, Any], *, run_id: str) -> bool:
    return (
        str(row.get("status")) == STATUS_PASS
        and str(row.get("run_id") or "") == run_id
        and str(row.get("aggregation_status") or "") == AGG_PASS
    )


def select_lhs_samples(
    pool: Mapping[str, Any],
    status_doc: Mapping[str, Any],
    *,
    max_samples: int,
    start_index: int = 0,
    end_index: Optional[int] = None,
    force_sample: Optional[str] = None,
    skip_completed: bool = True,
    exclude_reference: bool = False,
    run_id_suffix: str = DEFAULT_RUN_ID_SUFFIX,
    include_only_pending: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Return (selected_sample_entries, skipped_rows) from LHS pool order.
    skipped_rows describe why samples were not selected.
    """
    entries = list(pool.get("entries") or [])
    selected: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    for i, entry in enumerate(entries):
        if i < start_index:
            continue
        if end_index is not None and i > end_index:
            break
        sid = str(entry.get("id") or "").strip()
        if not sid:
            continue
        if force_sample and sid != force_sample:
            continue
        if exclude_reference and sid == REFERENCE_SAMPLE_ID:
            skipped.append({"sample_id": sid, "reason": "exclude_reference"})
            continue

        run_id = f"{sid}_{run_id_suffix}"
        st = get_sample_status(status_doc, sid)
        cur_status = str(st.get("status") or STATUS_PENDING)
        entry_status = normalize_lhs_entry_status(entry.get("status"))

        if skip_completed and is_lhs_entry_completed(entry, run_id=run_id):
            skipped.append({"sample_id": sid, "reason": "lhs_pool_completed", "run_id": run_id})
            continue

        if skip_completed and _is_completed_status(st, run_id=run_id):
            skipped.append({"sample_id": sid, "reason": "sidecar_already_pass", "run_id": run_id})
            continue

        if include_only_pending and entry_status == LHS_COMPLETED and not force_sample:
            skipped.append({"sample_id": sid, "reason": "lhs_pool_completed", "run_id": run_id})
            continue

        if include_only_pending and cur_status == STATUS_PASS and not force_sample:
            if str(st.get("run_id") or "") == run_id:
                skipped.append({"sample_id": sid, "reason": "sidecar_status_pass", "run_id": run_id})
                continue

        if len(selected) >= max_samples:
            break

        selected.append(
            {
                "lhs_row_index": i,
                "entry": entry,
                "sample_id": sid,
                "run_id": run_id,
                "prior_status": cur_status,
            }
        )

    return selected, skipped


def append_runs_index_row(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": INDEX_ROW_SCHEMA, "recorded_utc": utc_now(), **dict(row)}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, sort_keys=True) + "\n")


def update_sample_status(
    status_doc: Dict[str, Any],
    *,
    sample_id: str,
    patch: Mapping[str, Any],
) -> None:
    samples = status_doc.setdefault("samples", {})
    cur = dict(samples.get(sample_id) or {"sample_id": sample_id})
    cur.update(patch)
    cur["sample_id"] = sample_id
    cur["updated_utc"] = utc_now()
    samples[sample_id] = cur
    status_doc["updated_utc"] = utc_now()


def write_lhs_pool_status(path: Path, status_doc: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, dict(status_doc))


def sample_run_root(sample_id: str, run_id: str) -> Path:
    return GUITARS_ROOT / sample_id / "runs" / run_id


def status_row_from_run_summary(
    *,
    sample_id: str,
    lhs_row_index: int,
    run_id: str,
    batch_id: str,
    run_root: Path,
    outcome: str,
    elapsed_s: float,
    summary: Mapping[str, Any],
    error_message: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "sample_id": sample_id,
        "lhs_row_index": lhs_row_index,
        "run_id": run_id,
        "status": (
            STATUS_PASS
            if outcome in ("pass", "reused_complete", OUTCOME_PASS_FREEZE_WARNING)
            else (STATUS_FAIL if outcome == "fail" else outcome)
        ),
        "batch_id": batch_id,
        "pipeline_version": summary.get("pipeline_version") or PIPELINE_VERSION,
        "run_dir": str(run_root),
        "started_at": summary.get("started_at"),
        "finished_at": utc_now(),
        "elapsed_s": round(float(elapsed_s), 2),
        "aggregation_status": summary.get("aggregation_status"),
        "raw_mode_count": summary.get("raw_modes"),
        "deduped_mode_count": summary.get("deduped_modes"),
        "participation_computed_count": summary.get("participation_computed_count"),
        "audio_coupling_computed_count": summary.get("audio_coupling_computed_count"),
        "workers_actual_parallel": summary.get("workers_actual_parallel"),
        "failed_chunks": summary.get("failed_chunks"),
        "error_message": error_message,
    }


def make_batch_id(*, prefix: str = DEFAULT_BATCH_ID_PREFIX) -> str:
    return f"{prefix}_{utc_now()[:10].replace('-', '')}"


def normalize_lhs_entry_status(raw: Any) -> str:
    value = str(raw or LHS_PENDING).strip().upper()
    aliases = {
        "PASS": LHS_COMPLETED,
        "COMPLETED": LHS_COMPLETED,
        "PENDING": LHS_PENDING,
        "RUNNING": LHS_RUNNING,
        "FAILED": LHS_FAILED,
        "FAIL": LHS_FAILED,
        "FAILED_RETRYABLE": LHS_FAILED_RETRYABLE,
    }
    return aliases.get(value, value if value in aliases.values() else LHS_PENDING)


def is_lhs_entry_completed(entry: Mapping[str, Any], *, run_id: str) -> bool:
    status = normalize_lhs_entry_status(entry.get("status"))
    if status != LHS_COMPLETED:
        return False
    last_run = str(entry.get("last_run_id") or "")
    return not last_run or last_run == run_id


def read_run_production_summary(run_root: Path, *, workers_requested: int = 3) -> Dict[str, Any]:
    """Lightweight run summary from aggregation/manifest (no solver execution)."""
    manifest_path = run_root / "pipeline_run_manifest.json"
    manifest = load_json(manifest_path) if manifest_path.is_file() else {}
    agg_path = run_root / "aggregation" / "aggregation_result.json"
    agg: Dict[str, Any] = {}
    if agg_path.is_file():
        try:
            agg = load_json(agg_path)
        except (OSError, ValueError, json.JSONDecodeError):
            agg = {}
    ms_path = run_root / "aggregation" / "modes_summary.json"
    modes_summary: Dict[str, Any] = {}
    if ms_path.is_file():
        try:
            modes_summary = load_json(ms_path)
        except (OSError, ValueError, json.JSONDecodeError):
            modes_summary = {}
    audio_summary = modes_summary.get("audio_coupling_summary") or {}
    prov_path = run_root / "m4_sample_runtime_provenance.json"
    prov: Dict[str, Any] = {}
    if prov_path.is_file():
        try:
            prov = load_json(prov_path)
        except (OSError, ValueError, json.JSONDecodeError):
            prov = {}
    built_path = run_root / "lprod" / "checkpoint" / "built_metadata.json"
    built_meta: Dict[str, Any] = {}
    if built_path.is_file():
        try:
            built_meta = load_json(built_path)
        except (OSError, ValueError, json.JSONDecodeError):
            built_meta = {}
    return {
        "terminal_status": manifest.get("terminal_status"),
        "dataset_version": built_meta.get("dataset_version") or manifest.get("dataset_version"),
        "operator_mesh_matches_generated": built_meta.get("operator_mesh_matches_generated"),
        "p_idx_aperture_count": built_meta.get("p_idx_aperture_count"),
        "aperture_selection_method": built_meta.get("aperture_selection_method"),
        "generated_mesh_sha256": built_meta.get("generated_mesh_sha256"),
        "aggregation_status": agg.get("status"),
        "planned_chunks": agg.get("planned_chunk_count"),
        "completed_chunks": agg.get("completed_chunk_count"),
        "missing_chunks": agg.get("missing_chunk_count"),
        "failed_chunks": agg.get("failed_chunk_count"),
        "raw_modes": agg.get("raw_mode_count"),
        "deduped_modes": agg.get("deduped_mode_count"),
        "final_aggregation_ready": agg.get("final_aggregation_ready"),
        "participation_computed_count": modes_summary.get("participation_computed_count")
        or prov.get("participation_computed_count"),
        "audio_coupling_computed_count": modes_summary.get("audio_coupling_computed_count")
        or audio_summary.get("audio_coupling_computed_count"),
        "workers_actual_parallel": prov.get("workers_actual_parallel"),
        "pipeline_version": prov.get("pipeline_version") or PIPELINE_VERSION,
    }


def is_run_usably_complete(summary: Mapping[str, Any]) -> bool:
    return (
        str(summary.get("aggregation_status") or "") == AGG_PASS
        and int(summary.get("failed_chunks") or 0) == 0
        and int(summary.get("missing_chunks") or 0) == 0
        and bool(summary.get("final_aggregation_ready"))
    )


def classify_sample_outcome(
    *,
    return_code: int,
    summary: Mapping[str, Any],
) -> Tuple[str, Optional[str]]:
    """Return (outcome, error_message). Usable aggregation pass is never a hard fail."""
    if is_run_usably_complete(summary):
        if return_code == 0:
            return "pass", None
        return (
            OUTCOME_PASS_FREEZE_WARNING,
            f"return_code={return_code} aggregation_status={summary.get('aggregation_status')} "
            f"(freeze/terminal repairable)",
        )
    if str(summary.get("aggregation_status") or "") == AGG_PASS:
        return (
            OUTCOME_PASS_FREEZE_WARNING,
            f"return_code={return_code} chunks_failed={summary.get('failed_chunks')} "
            f"missing={summary.get('missing_chunks')}",
        )
    return (
        "fail",
        f"return_code={return_code} aggregation_status={summary.get('aggregation_status')}",
    )


def patch_lhs_pool_entry(entry: Dict[str, Any], *, patch: Mapping[str, Any]) -> None:
    entry.update(patch)


def lhs_pool_entry_patch_from_run(
    *,
    run_id: str,
    run_dir: str,
    batch_id: Optional[str],
    outcome: str,
    summary: Mapping[str, Any],
    elapsed_s: float,
    started_at: Optional[str],
    error_message: Optional[str] = None,
) -> Dict[str, Any]:
    if outcome in ("pass", "reused_complete", OUTCOME_PASS_FREEZE_WARNING):
        lhs_status = LHS_COMPLETED
    elif outcome == "fail":
        lhs_status = LHS_FAILED
    else:
        lhs_status = LHS_PENDING
    return {
        "status": lhs_status,
        "last_run_id": run_id,
        "last_run_dir": run_dir,
        "last_batch_id": batch_id,
        "last_started_at": started_at,
        "last_finished_at": utc_now(),
        "last_elapsed_s": round(float(elapsed_s), 2),
        "last_aggregation_status": summary.get("aggregation_status"),
        "last_deduped_mode_count": summary.get("deduped_modes"),
        "last_participation_computed_count": summary.get("participation_computed_count"),
        "last_audio_coupling_computed_count": summary.get("audio_coupling_computed_count"),
        "last_error": error_message,
        "error": error_message,
    }


def write_lhs_pool_with_backup(lhs_path: Path, pool: Mapping[str, Any]) -> Path:
    lhs_path = lhs_path.expanduser().resolve()
    lhs_path.parent.mkdir(parents=True, exist_ok=True)
    stamp = utc_now().replace(":", "").replace("-", "")
    backup = lhs_path.parent / f"lhs_pool.backup_{stamp}.json"
    if lhs_path.is_file():
        shutil.copy2(lhs_path, backup)
    write_json_atomic(lhs_path, dict(pool))
    return backup


def sync_lhs_pool_entry(
    pool: Dict[str, Any],
    *,
    sample_id: str,
    patch: Mapping[str, Any],
) -> None:
    for entry in pool.get("entries") or []:
        if str(entry.get("id")) == sample_id:
            patch_lhs_pool_entry(entry, patch=patch)
            break


def reconcile_existing_runs(
    *,
    repo_root: Path,
    pool: Dict[str, Any],
    lhs_path: Path,
    run_id_suffix: str = DEFAULT_RUN_ID_SUFFIX,
    repair_freeze: bool = True,
    repair_stale_running: bool = False,
    batch_id: Optional[str] = None,
    shared_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Scan run trees for valid AGGREGATION_PASS outputs; repair freeze/terminal; update LHS pool.
    Does not rerun workers.
    """
    from v2_b3_m4_production_contracts import DATASET_VERSION  # noqa: E402
    from v2_b3_m4_production_freeze import load_sample_input, replay_production_freeze  # noqa: E402
    from v2_b3_m4_freeze_first_e2e_run import repair_run_freeze_and_terminal  # noqa: E402
    from v2_b3_m4_run_status_repair import (  # noqa: E402
        STALE_RUNNING_REPAIR_REASON,
        promote_checkpoint_ready_terminal,
    )

    guitars_root = (
        repo_root
        / "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars"
    )
    rows: List[Dict[str, Any]] = []
    repaired = 0
    stale_running_repaired = 0
    completed = 0
    failed = 0
    bid = batch_id or f"reconcile_{utc_now()[:10].replace('-', '')}"

    for i, entry in enumerate(pool.get("entries") or []):
        sid = str(entry.get("id") or "").strip()
        if not sid:
            continue
        run_id = f"{sid}_{run_id_suffix}"
        run_root = guitars_root / sid / "runs" / run_id
        row: Dict[str, Any] = {
            "sample_id": sid,
            "lhs_row_index": i,
            "run_id": run_id,
            "run_root": rel(run_root, repo_root=repo_root) if run_root.is_dir() else None,
            "action": "skip_no_run_dir",
        }
        if not run_root.is_dir():
            rows.append(row)
            continue

        summary = read_run_production_summary(run_root)
        row.update(summary)
        if repair_stale_running:
            stale_result = promote_checkpoint_ready_terminal(
                run_root,
                repair_reason=STALE_RUNNING_REPAIR_REASON,
            )
            row["stale_running_repair"] = stale_result
            if stale_result.get("status") == "PASS":
                stale_running_repaired += 1
                summary = read_run_production_summary(run_root)
                row.update(summary)
                row["terminal_status"] = summary.get("terminal_status")
        if not is_run_usably_complete(summary):
            row["action"] = (
                "stale_running_repaired"
                if row.get("stale_running_repair", {}).get("status") == "PASS"
                else "skip_not_usable"
            )
            failed += 1
            rows.append(row)
            continue

        freeze_rc = 0
        freeze_msg = "aggregation_pass"
        if repair_freeze:
            built_path = run_root / "lprod" / "checkpoint" / "built_metadata.json"
            production_run = False
            if built_path.is_file():
                try:
                    built_doc = load_json(built_path)
                    production_run = str(built_doc.get("dataset_version") or "") == DATASET_VERSION
                except (OSError, ValueError, json.JSONDecodeError):
                    production_run = False
            if production_run:
                freeze_rc, freeze_msg = replay_production_freeze(
                    repo_root=repo_root,
                    run_root=run_root,
                    sample_input=load_sample_input(run_root),
                    force=False,
                )
            else:
                freeze_rc, freeze_msg = repair_run_freeze_and_terminal(
                    repo_root=repo_root,
                    run_root=run_root,
                    sample_id=sid,
                    force=False,
                )
        summary = read_run_production_summary(run_root)
        outcome = "pass" if freeze_rc == 0 else OUTCOME_PASS_FREEZE_WARNING
        if freeze_rc == 0:
            repaired += 1

        shared_export = None
        shared_export_warning = None
        try:
            from v2_b3_m4_shared_export import try_export_sample_to_shared  # noqa: E402

            shared_export, shared_export_warning = try_export_sample_to_shared(
                run_root=run_root,
                sample_id=sid,
                run_id=run_id,
                shared_root=shared_root,
                repo_root=repo_root,
            )
        except Exception as exc:
            shared_export_warning = f"shared export skipped: {exc}"

        lhs_patch = lhs_pool_entry_patch_from_run(
            run_id=run_id,
            run_dir=str(run_root),
            batch_id=bid,
            outcome=outcome,
            summary=summary,
            elapsed_s=0.0,
            started_at=None,
            error_message=None if freeze_rc == 0 else freeze_msg,
        )
        sync_lhs_pool_entry(pool, sample_id=sid, patch=lhs_patch)
        row["action"] = "reconciled_completed"
        row["outcome"] = outcome
        row["freeze_repair"] = freeze_msg
        row["freeze_rc"] = freeze_rc
        if shared_export:
            row["shared_export"] = shared_export
        if shared_export_warning:
            row["shared_export_warning"] = shared_export_warning
        completed += 1
        rows.append(row)

    backup = write_lhs_pool_with_backup(lhs_path, pool)
    return {
        "schema": "m4_lhs_reconcile_report_v1",
        "generated_utc": utc_now(),
        "run_id_suffix": run_id_suffix,
        "lhs_json": rel(lhs_path, repo_root=repo_root),
        "lhs_backup": rel(backup, repo_root=repo_root),
        "reconciled_completed_count": completed,
        "freeze_repaired_count": repaired,
        "stale_running_repaired_count": stale_running_repaired,
        "not_usable_count": failed,
        "samples": rows,
    }
