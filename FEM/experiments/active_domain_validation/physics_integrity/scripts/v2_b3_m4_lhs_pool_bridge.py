#!/usr/bin/env python3
"""Bridge ROM/classic LHS pool → M4 production specs + persistent run status index."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_PASS = "pass"
STATUS_FAIL = "fail"
STATUS_SKIPPED = "skipped"

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
) -> Dict[str, Any]:
    sid = str(entry["id"])
    params = dict(entry.get("parameters") or {})
    return {
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


def build_batch_sample_entry(
    *,
    pool: Mapping[str, Any],
    entry: Mapping[str, Any],
    lhs_row_index: int,
    run_id_suffix: str,
    batch_id: str,
    lhs_source_path: str,
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
) -> Dict[str, Any]:
    fp = dict(DEFAULT_FREQUENCY_POLICY)
    if frequency_policy:
        fp.update(frequency_policy)
    exclude: List[str] = []
    if exclude_reference:
        exclude.append(reference_sample_id)
    ref_run = reference_run_id or f"{reference_sample_id}_{run_id_suffix}"
    return {
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
    }


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

        if skip_completed and _is_completed_status(st, run_id=run_id):
            skipped.append({"sample_id": sid, "reason": "already_pass", "run_id": run_id})
            continue

        if include_only_pending and cur_status == STATUS_PASS and not force_sample:
            if str(st.get("run_id") or "") == run_id:
                skipped.append({"sample_id": sid, "reason": "status_pass", "run_id": run_id})
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
        "status": STATUS_PASS if outcome == "pass" else (STATUS_FAIL if outcome == "fail" else outcome),
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
