#!/usr/bin/env python3
"""M4 production runtime provenance fields for sample/batch summaries."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

PIPELINE_VERSION = "M4 production"
MODEL_VERSION = "V2"
OPERATOR_VERSION = "B3"
MESH_LEVEL = "L_prod"
SOLVER_BACKEND = "mkl_pardiso"
CHUNK_POLICY = "lprod_target_plan_fcfs"
TARGET_POLICY = "adaptive_lprod_zones_v1"

PRODUCTION_WORKER_THREAD_VARS = (
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
)


def production_worker_thread_settings(env: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    out = {k: "1" for k in PRODUCTION_WORKER_THREAD_VARS}
    out["MKL_DYNAMIC"] = "0"
    if env:
        for k in PRODUCTION_WORKER_THREAD_VARS:
            if k in env:
                out[k] = str(env[k])
        if "MKL_DYNAMIC" in env:
            out["MKL_DYNAMIC"] = str(env["MKL_DYNAMIC"])
    return out


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    return data if isinstance(data, dict) else {}


def _stage_wall_seconds(manifest: Dict[str, Any], stage_key: str) -> Optional[float]:
    st = (manifest.get("stages") or {}).get(stage_key) or {}
    for key in ("wall_seconds", "elapsed_s", "wall_time_s"):
        val = st.get(key)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
    return None


def collect_m4_runtime_provenance(
    *,
    run_root: Path,
    workers_requested: int,
    worker_remaining_manifest: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble production provenance from run tree artifacts (no solver execution)."""
    run_root = run_root.expanduser().resolve()
    manifest_path = run_root / "pipeline_run_manifest.json"
    manifest = _load_json(manifest_path) if manifest_path.is_file() else {}

    wr = worker_remaining_manifest
    if wr is None:
        wr_path = run_root / "worker_results" / "remaining_workers_m4_4_1b_4_manifest.json"
        if wr_path.is_file():
            try:
                wr = _load_json(wr_path)
            except (OSError, ValueError, json.JSONDecodeError):
                wr = None

    agg: Dict[str, Any] = {}
    agg_path = run_root / "aggregation" / "aggregation_result.json"
    if agg_path.is_file():
        try:
            agg = _load_json(agg_path)
        except (OSError, ValueError, json.JSONDecodeError):
            agg = {}

    modes_summary: Dict[str, Any] = {}
    ms_path = run_root / "aggregation" / "modes_summary.json"
    if ms_path.is_file():
        try:
            modes_summary = _load_json(ms_path)
        except (OSError, ValueError, json.JSONDecodeError):
            modes_summary = {}

    chunk_walls: List[Dict[str, Any]] = []
    if wr:
        for row in wr.get("chunk_results") or []:
            if not isinstance(row, dict):
                continue
            chunk_walls.append(
                {
                    "chunk_id": row.get("chunk_id"),
                    "wall_seconds": row.get("wall_seconds"),
                    "status": row.get("status"),
                    "action": row.get("action"),
                }
            )

    workers_actual = int(wr.get("workers_actual_parallel") or 0) if wr else 0
    if workers_actual <= 0 and wr:
        workers_actual = int(wr.get("workers_requested") or workers_requested)

    out: Dict[str, Any] = {
        "schema": "m4_runtime_provenance_v1",
        "pipeline_version": PIPELINE_VERSION,
        "model_version": MODEL_VERSION,
        "operator_version": OPERATOR_VERSION,
        "mesh_level": MESH_LEVEL,
        "target_policy": TARGET_POLICY,
        "chunk_policy": CHUNK_POLICY,
        "solver_backend": SOLVER_BACKEND,
        "workers_requested": int(workers_requested),
        "workers_actual_parallel": workers_actual,
        "worker_thread_settings": dict(wr.get("worker_thread_settings") or production_worker_thread_settings()),
        "stage_wall_times_s": {
            "stage4_lprod_checkpoint": _stage_wall_seconds(manifest, "stage4_lprod_export"),
            "stage5_workers": float(wr["wall_time_s"]) if wr and wr.get("wall_time_s") is not None else None,
            "stage6_aggregate": _stage_wall_seconds(manifest, "stage6_aggregate"),
        },
        "chunk_wall_times": chunk_walls,
        "raw_mode_count": agg.get("raw_mode_count") or modes_summary.get("raw_mode_count"),
        "deduped_mode_count": agg.get("deduped_mode_count") or modes_summary.get("deduped_mode_count"),
        "participation_computed_count": modes_summary.get("participation_computed_count"),
        "dominant_region_counts": modes_summary.get("dominant_region_counts"),
        "aggregation_status": agg.get("status"),
        "sample_id": manifest.get("sample_id"),
        "run_id": manifest.get("run_id"),
    }
    return out


def merge_runtime_summary(
    runtime_summary: Dict[str, Any],
    provenance: Mapping[str, Any],
) -> Dict[str, Any]:
    merged = dict(runtime_summary)
    merged.update(
        {
            "pipeline_version": provenance.get("pipeline_version"),
            "model_version": provenance.get("model_version"),
            "operator_version": provenance.get("operator_version"),
            "mesh_level": provenance.get("mesh_level"),
            "target_policy": provenance.get("target_policy"),
            "chunk_policy": provenance.get("chunk_policy"),
            "solver_backend": provenance.get("solver_backend"),
            "workers_requested": provenance.get("workers_requested"),
            "workers_actual_parallel": provenance.get("workers_actual_parallel"),
            "worker_thread_settings": provenance.get("worker_thread_settings"),
            "stage_wall_times_s": provenance.get("stage_wall_times_s"),
            "chunk_wall_times": provenance.get("chunk_wall_times"),
            "participation_computed_count": provenance.get("participation_computed_count"),
            "dominant_region_counts": provenance.get("dominant_region_counts"),
        }
    )
    return merged
