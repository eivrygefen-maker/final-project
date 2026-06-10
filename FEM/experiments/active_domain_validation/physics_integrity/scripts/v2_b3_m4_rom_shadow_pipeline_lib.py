#!/usr/bin/env python3
"""Shadow ROM prediction pipeline for official L_rom_prod dataset (no FOM mutation)."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from v2_b3_m4_lhs_pool_bridge import AGG_PASS, read_run_production_summary  # noqa: E402
from v2_b3_m4_mesh_profile_lib import DATASET_VERSION_ROM, MESH_PROFILE_ROM  # noqa: E402
from v2_b3_m4_modal_surrogate_lib import (  # noqa: E402
    DEFAULT_K_NEIGHBORS,
    MODEL_VERSION_V2_1,
    build_surrogate_from_training_rows,
    load_surrogate_model,
    predict_modal_catalog,
    save_surrogate_model,
)
from v2_b3_m4_official_rom_dataset_lib import (  # noqa: E402
    MATURITY_INTEGRATION_ONLY,
    OFFICIAL_INITIAL_RUN_IDS,
    build_initial_five_run_dataset_report,
    collect_official_rom_training_rows,
    evaluate_official_rom_run_eligibility,
    load_official_dataset_registry,
    official_dataset_registry_path,
    register_official_rom_dataset_entry,
    write_official_rom_model_manifest,
)
from v2_b3_m4_rom_fom_compare_lib import (  # noqa: E402
    DEFAULT_MAX_MATCH_DISTANCE_HZ,
    DEFAULT_ROM_NEV,
    ROM_COMPARE_COMPLETED,
    ROM_COMPARE_FAILED,
    ROM_STATUS_COMPLETED,
    ROM_STATUS_FAILED,
    build_rom_fom_comparison,
    load_fom_modes_catalog_deduped,
    resolve_validation_metadata,
    VALIDATION_HOLDOUT,
)
from v2_b3_m4_worker_run_lib import load_json, rel, utc_now  # noqa: E402
from v2_b3_petsc_util import write_json_atomic  # noqa: E402

ROM_PREDICTION_SUMMARY_NAME = "rom_prediction_summary.json"
ROM_VS_FOM_COMPARISON_NAME = "rom_vs_fom_comparison.json"
FOM_STARTED_MARKER_NAME = "fom_pipeline_started_utc.json"
FROZEN_PREDICTION_INTERNAL_NAME = "rom_prediction_frozen_internal.json"

DURABLE_ROM_JSON_NAMES: Tuple[str, ...] = (
    ROM_PREDICTION_SUMMARY_NAME,
    ROM_VS_FOM_COMPARISON_NAME,
)

PREDICTION_SUMMARY_SCHEMA = "m4_rom_prediction_summary_v1"
COMPARISON_SUMMARY_SCHEMA = "m4_rom_vs_fom_comparison_v1"

DEFAULT_RETRAIN_EVERY_N_NEW_SAMPLES = 5


class RomShadowIntegrityError(RuntimeError):
    """Blocking ROM shadow failure (missing/corrupt prediction, leakage, identity mismatch)."""


@dataclass(frozen=True)
class RetrainPolicy:
    retrain_every_n_new_samples: int = DEFAULT_RETRAIN_EVERY_N_NEW_SAMPLES

    def should_retrain(self, *, new_samples_since_last_train: int) -> bool:
        n = int(self.retrain_every_n_new_samples)
        if n <= 0:
            return False
        return new_samples_since_last_train >= n


def rom_dir_for_run(run_root: Path) -> Path:
    return run_root / "rom"


def rom_prediction_summary_path(run_root: Path) -> Path:
    return rom_dir_for_run(run_root) / ROM_PREDICTION_SUMMARY_NAME


def rom_vs_fom_comparison_path(run_root: Path) -> Path:
    return rom_dir_for_run(run_root) / ROM_VS_FOM_COMPARISON_NAME


def fom_started_marker_path(run_root: Path) -> Path:
    return rom_dir_for_run(run_root) / FOM_STARTED_MARKER_NAME


def frozen_prediction_internal_path(run_root: Path) -> Path:
    return rom_dir_for_run(run_root) / FROZEN_PREDICTION_INTERNAL_NAME


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _sha256_json(doc: Mapping[str, Any]) -> str:
    payload = json.dumps(doc, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(payload)


def _load_manifest(repo_root: Path, shape_name: str) -> Dict[str, Any]:
    path = repo_root / "ROM" / shape_name / "rom_model_manifest.json"
    if not path.is_file():
        return {}
    try:
        doc = load_json(path)
        return doc if isinstance(doc, dict) else {}
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def _encode_feature_vector(parameters: Mapping[str, Any]) -> List[float]:
    from v2_b3_m4_modal_surrogate_lib import encode_lhs_parameters  # noqa: WPS433

    return [float(x) for x in encode_lhs_parameters(parameters).tolist()]


def build_holdout_official_rom_model(
    *,
    repo_root: Path,
    shape_name: str,
    exclude_sample_ids: Sequence[str],
    exclude_run_ids: Optional[Sequence[str]] = None,
    k_neighbors: int = DEFAULT_K_NEIGHBORS,
    min_mode_count: int = 1,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    training, skipped = collect_official_rom_training_rows(
        repo_root=repo_root,
        exclude_sample_ids=exclude_sample_ids,
        exclude_run_ids=exclude_run_ids,
        min_mode_count=min_mode_count,
    )
    if not training:
        raise RomShadowIntegrityError(
            f"no_official_training_rows_after_exclude sample_ids={list(exclude_sample_ids)}"
        )
    for row in training:
        if str(row["sample_id"]) in {str(s) for s in exclude_sample_ids}:
            raise RomShadowIntegrityError(f"training_leakage_sample_included:{row['sample_id']}")
    model = build_surrogate_from_training_rows(
        shape_name=shape_name,
        training_rows=training,
        k_neighbors=k_neighbors,
    )
    model["holdout_validation"] = True
    model["excluded_sample_ids"] = list(exclude_sample_ids)
    model["official_rom_mesh_only"] = True
    return model, training, skipped


def build_official_rom_surrogate_from_runs(
    *,
    repo_root: Path,
    shape_name: str = "classic",
    require_initial_allowlist: bool = True,
    allowed_run_ids: Optional[Sequence[str]] = None,
    k_neighbors: int = DEFAULT_K_NEIGHBORS,
    min_mode_count: int = 10,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Build fresh official ROM surrogate; never loads a previous on-disk model."""
    run_ids = list(allowed_run_ids) if allowed_run_ids is not None else list(OFFICIAL_INITIAL_RUN_IDS)
    training, skipped = collect_official_rom_training_rows(
        repo_root=repo_root,
        require_initial_allowlist=require_initial_allowlist,
        allowed_run_ids=run_ids,
        min_mode_count=min_mode_count,
    )
    if not training:
        raise ValueError("no official ROM training rows collected")
    model = build_surrogate_from_training_rows(
        shape_name=shape_name,
        training_rows=training,
        k_neighbors=k_neighbors,
    )
    model["official_rom_mesh_only"] = True
    model["maturity"] = MATURITY_INTEGRATION_ONLY
    model["production_accuracy_validated"] = False
    paths = save_official_rom_surrogate_model(repo_root, model, training_rows=training)
    report = build_initial_five_run_dataset_report(
        repo_root=repo_root,
        training_rows=training,
        skipped_rows=skipped,
    )
    report["model_paths"] = {k: rel(v, repo_root=repo_root) for k, v in paths.items()}
    return model, training, skipped, report


def save_official_rom_surrogate_model(
    repo_root: Path,
    model: Mapping[str, Any],
    *,
    training_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Path]:
    paths = save_surrogate_model(repo_root, model)
    manifest = write_official_rom_model_manifest(
        repo_root,
        str(model["shape_name"]),
        model_version=str(model.get("model_version") or MODEL_VERSION_V2_1),
        training_rows=training_rows,
        surrogate_schema=str(model.get("schema") or ""),
        prediction_method=str(model.get("method") or ""),
    )
    paths["manifest"] = manifest
    retrain_state_path = repo_root / "ROM" / str(model["shape_name"]) / "rom_retrain_state.json"
    write_json_atomic(
        retrain_state_path,
        {
            "schema": "m4_rom_retrain_state_v1",
            "last_trained_utc": utc_now(),
            "training_run_ids": [str(r["run_id"]) for r in training_rows],
            "training_sample_ids": [str(r["sample_id"]) for r in training_rows],
            "new_samples_since_last_train": 0,
        },
    )
    return paths


def mark_fom_pipeline_started(run_root: Path) -> Path:
    rom_dir_for_run(run_root).mkdir(parents=True, exist_ok=True)
    path = fom_started_marker_path(run_root)
    write_json_atomic(
        path,
        {
            "schema": "m4_rom_fom_started_v1",
            "started_utc": utc_now(),
            "started_perf_s": time.perf_counter(),
        },
    )
    return path


def _verify_prediction_before_fom(
    run_root: Path,
    prediction_created_utc: str,
    prediction_recorded_perf_s: Optional[float] = None,
) -> None:
    marker = fom_started_marker_path(run_root)
    if not marker.is_file():
        return
    try:
        started_doc = load_json(marker)
        started = str(started_doc.get("started_utc") or "")
        started_perf = started_doc.get("started_perf_s")
    except (OSError, ValueError, json.JSONDecodeError):
        raise RomShadowIntegrityError("fom_started_marker_unreadable") from None
    if prediction_recorded_perf_s is not None and started_perf is not None:
        if float(prediction_recorded_perf_s) >= float(started_perf):
            raise RomShadowIntegrityError(
                "prediction_not_before_fom: prediction_perf_s >= fom_started_perf_s"
            )
        return
    if started and prediction_created_utc > started:
        raise RomShadowIntegrityError(
            f"prediction_not_before_fom: prediction={prediction_created_utc} fom_started={started}"
        )


def _compact_prediction_summary(
    *,
    context: Mapping[str, Any],
    prediction: Mapping[str, Any],
    model_manifest: Mapping[str, Any],
    training_sample_ids: Sequence[str],
    feature_vector: Sequence[float],
    prediction_created_utc: str,
    prediction_recorded_perf_s: float,
    frozen_sha256: str,
) -> Dict[str, Any]:
    predicted_modes = list(prediction.get("predicted_modes") or prediction.get("modes") or [])
    return {
        "schema": PREDICTION_SUMMARY_SCHEMA,
        "sample_id": context["sample_id"],
        "run_id": context["run_id"],
        "prediction_created_utc": prediction_created_utc,
        "prediction_recorded_perf_s": prediction_recorded_perf_s,
        "model_version": prediction.get("model_version") or model_manifest.get("model_version"),
        "model_manifest_sha256": model_manifest.get("model_manifest_sha256"),
        "training_sample_ids": list(training_sample_ids),
        "training_sample_count": len(training_sample_ids),
        "training_dataset_version": DATASET_VERSION_ROM,
        "mesh_profile": MESH_PROFILE_ROM,
        "input_feature_vector": list(feature_vector),
        "predicted_mode_count": len(prediction.get("frequencies_hz") or []),
        "predicted_frequencies_hz": list(prediction.get("frequencies_hz") or []),
        "predicted_modes": predicted_modes,
        "prediction_runtime_s": prediction.get("runtime_s"),
        "frozen_prediction_sha256": frozen_sha256,
        "validation_mode": VALIDATION_HOLDOUT,
        "maturity": model_manifest.get("maturity") or MATURITY_INTEGRATION_ONLY,
        "production_accuracy_validated": bool(model_manifest.get("production_accuracy_validated")),
    }


def run_shadow_rom_prepredict(
    *,
    repo_root: Path,
    run_root: Path,
    context: Mapping[str, Any],
    nev: int = DEFAULT_ROM_NEV,
    dataset_version: Optional[str] = None,
) -> Dict[str, Any]:
    """Predict before FOM using holdout official model; freeze durable summary."""
    shape_name = str(context.get("shape_name") or "classic")
    sample_id = str(context["sample_id"])
    run_id = str(context["run_id"])
    ds = str(dataset_version or DATASET_VERSION_ROM)
    if ds != DATASET_VERSION_ROM:
        raise RomShadowIntegrityError(f"dataset_version_mismatch:{dataset_version}")

    manifest = _load_manifest(repo_root, shape_name)
    if not manifest:
        raise RomShadowIntegrityError("official_rom_model_manifest_missing")

    model, training_rows, _skipped = build_holdout_official_rom_model(
        repo_root=repo_root,
        shape_name=shape_name,
        exclude_sample_ids=[sample_id],
        exclude_run_ids=[run_id],
    )
    train_ids = [str(r["sample_id"]) for r in training_rows]
    if sample_id in train_ids:
        raise RomShadowIntegrityError(f"training_leakage:{sample_id}")

    t0 = time.perf_counter()
    prediction = predict_modal_catalog(
        model,
        context.get("parameters") or {},
        nev=int(nev),
    )
    elapsed = round(time.perf_counter() - t0, 4)
    prediction.update(
        {
            "status": ROM_STATUS_COMPLETED,
            "runtime_s": elapsed,
            "model_version": model.get("model_version"),
            "training_sample_ids": train_ids,
            "training_run_ids": [str(r["run_id"]) for r in training_rows],
        }
    )

    prediction_created_utc = utc_now()
    prediction_recorded_perf_s = time.perf_counter()
    rom_dir = rom_dir_for_run(run_root)
    rom_dir.mkdir(parents=True, exist_ok=True)
    frozen_internal = {
        "schema": "m4_rom_prediction_frozen_internal_v1",
        "prediction_created_utc": prediction_created_utc,
        "context": {
            "sample_id": sample_id,
            "run_id": run_id,
            "shape_name": shape_name,
            "lhs_row_index": context.get("lhs_row_index"),
            "parameters": dict(context.get("parameters") or {}),
        },
        "prediction": prediction,
        "model_manifest": manifest,
    }
    frozen_path = frozen_prediction_internal_path(run_root)
    write_json_atomic(frozen_path, frozen_internal)
    frozen_sha = _sha256_file(frozen_path)

    summary = _compact_prediction_summary(
        context=context,
        prediction=prediction,
        model_manifest=manifest,
        training_sample_ids=train_ids,
        feature_vector=_encode_feature_vector(context.get("parameters") or {}),
        prediction_created_utc=prediction_created_utc,
        prediction_recorded_perf_s=prediction_recorded_perf_s,
        frozen_sha256=frozen_sha,
    )
    summary_path = rom_prediction_summary_path(run_root)
    write_json_atomic(summary_path, summary)

    return {
        "status": ROM_STATUS_COMPLETED,
        "summary_path": rel(summary_path, repo_root=repo_root),
        "frozen_sha256": frozen_sha,
        "prediction_created_utc": prediction_created_utc,
        "training_sample_ids": train_ids,
        "predicted_mode_count": len(prediction.get("frequencies_hz") or []),
        "prediction_runtime_s": elapsed,
    }


def _load_frozen_prediction(run_root: Path) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    summary_path = rom_prediction_summary_path(run_root)
    frozen_path = frozen_prediction_internal_path(run_root)
    if not summary_path.is_file():
        raise RomShadowIntegrityError(f"missing_prediction_summary:{summary_path.name}")
    if not frozen_path.is_file():
        raise RomShadowIntegrityError(f"missing_frozen_prediction:{frozen_path.name}")
    summary = load_json(summary_path)
    frozen = load_json(frozen_path)
    frozen_sha = _sha256_file(frozen_path)
    recorded_sha = str(summary.get("frozen_prediction_sha256") or "")
    if recorded_sha and recorded_sha != frozen_sha:
        raise RomShadowIntegrityError("frozen_prediction_sha256_mismatch")
    prediction = dict(frozen.get("prediction") or {})
    context = dict(frozen.get("context") or {})
    _verify_prediction_before_fom(
        run_root,
        str(summary.get("prediction_created_utc") or ""),
        prediction_recorded_perf_s=summary.get("prediction_recorded_perf_s"),
    )
    return summary, prediction, frozen_sha


def _compact_comparison_report(
    *,
    comparison: Mapping[str, Any],
    frozen_sha256: str,
    model_version: Optional[str],
) -> Dict[str, Any]:
    phase2 = comparison.get("phase2_scalar_metrics") or comparison.get("phase2_intensity_metrics_v2_1") or {}
    return {
        "schema": COMPARISON_SUMMARY_SCHEMA,
        "generated_utc": utc_now(),
        "sample_id": comparison.get("sample_id"),
        "run_id": comparison.get("run_id"),
        "model_version": model_version or comparison.get("model_version"),
        "frozen_prediction_sha256": frozen_sha256,
        "matched_mode_count": comparison.get("matched_mode_count"),
        "unmatched_prediction_count": comparison.get("unmatched_rom_count"),
        "unmatched_fom_count": comparison.get("unmatched_fom_count"),
        "recall": comparison.get("recall"),
        "median_abs_error_hz": comparison.get("median_abs_error_hz"),
        "p95_abs_error_hz": comparison.get("p90_abs_error_hz"),
        "max_abs_error_hz": comparison.get("max_abs_error_hz"),
        "median_relative_error": comparison.get("median_relative_error"),
        "p95_relative_error": comparison.get("p90_relative_error"),
        "per_band_results": comparison.get("per_band_results") or comparison.get("band_metrics"),
        "top_share_mae": phase2.get("top_share_mae"),
        "back_share_mae": phase2.get("back_share_mae"),
        "air_share_mae": phase2.get("air_share_mae"),
        "coupling_class_accuracy": phase2.get("coupling_class_accuracy"),
        "bridge_excitation_mae": phase2.get("bridge_excitation_abs_mae"),
        "mic_output_proxy_mae": phase2.get("mic_output_proxy_mae"),
        "radiation_proxy_mae": phase2.get("radiation_proxy_mae"),
        "comparison_runtime_s": comparison.get("rom_comparison_runtime_s"),
        "accuracy_meaningful": comparison.get("accuracy_meaningful"),
        "low_accuracy_recorded_only": True,
        "status": comparison.get("status"),
        "warnings": list(comparison.get("warnings") or []),
    }


def run_shadow_rom_compare(
    *,
    repo_root: Path,
    run_root: Path,
    context: Mapping[str, Any],
    max_match_distance_hz: float = DEFAULT_MAX_MATCH_DISTANCE_HZ,
) -> Dict[str, Any]:
    """Compare frozen shadow prediction to FOM catalog; low accuracy is non-blocking."""
    t_compare = time.perf_counter()
    summary, prediction, frozen_sha = _load_frozen_prediction(run_root)

    if str(summary.get("sample_id") or "") != str(context.get("sample_id") or ""):
        raise RomShadowIntegrityError("sample_id_mismatch")
    if str(summary.get("run_id") or "") != str(context.get("run_id") or ""):
        raise RomShadowIntegrityError("run_id_mismatch")

    catalog_path = run_root / "aggregation" / "modes_catalog_deduped.jsonl"
    if not catalog_path.is_file():
        catalog_path = run_root / "aggregation" / "modes_catalog.jsonl"
    _raw, fom_modes, fom_catalog_meta = load_fom_modes_catalog_deduped(catalog_path)
    fom_summary = read_run_production_summary(run_root)
    if str(fom_summary.get("aggregation_status") or "") != AGG_PASS:
        raise RomShadowIntegrityError(f"fom_aggregation_not_pass:{fom_summary.get('aggregation_status')}")

    train_ids = list(summary.get("training_sample_ids") or [])
    vmeta = resolve_validation_metadata(
        target_sample_id=str(context["sample_id"]),
        training_sample_ids=train_ids,
        excluded_sample_ids=[str(context["sample_id"])],
        validation_mode=VALIDATION_HOLDOUT,
    )
    comparison = build_rom_fom_comparison(
        context=context,
        rom_prediction={**prediction, "validation_meta": vmeta, "status": ROM_STATUS_COMPLETED},
        fom_summary=fom_summary,
        fom_modes=fom_modes,
        max_match_distance_hz=max_match_distance_hz,
        rom_prediction_path_rel=rel(rom_prediction_summary_path(run_root), repo_root=repo_root),
        fom_catalog_path_rel=rel(catalog_path, repo_root=repo_root),
        validation_meta=vmeta,
        fom_catalog_meta=fom_catalog_meta,
    )
    cmp_elapsed = round(time.perf_counter() - t_compare, 4)
    comparison["rom_comparison_runtime_s"] = cmp_elapsed
    compact = _compact_comparison_report(
        comparison=comparison,
        frozen_sha256=frozen_sha,
        model_version=str(summary.get("model_version") or ""),
    )
    out_path = rom_vs_fom_comparison_path(run_root)
    write_json_atomic(out_path, compact)
    return {
        "status": compact.get("status") or comparison.get("status"),
        "comparison_path": rel(out_path, repo_root=repo_root),
        "matched_mode_count": compact.get("matched_mode_count"),
        "median_abs_error_hz": compact.get("median_abs_error_hz"),
        "comparison_runtime_s": cmp_elapsed,
        "blocking": False,
    }


def maybe_register_and_retrain(
    *,
    repo_root: Path,
    run_root: Path,
    sample_id: str,
    run_id: str,
    shape_name: str = "classic",
    production_acceptance_pass: bool,
    policy: Optional[RetrainPolicy] = None,
    require_post_cleanup_eligibility: bool = True,
) -> Dict[str, Any]:
    """Register accepted run and optionally retrain. Requires post-cleanup eligibility by default."""
    result: Dict[str, Any] = {"registered": False, "retrained": False}
    if not production_acceptance_pass:
        return result

    if require_post_cleanup_eligibility:
        eligible, reasons, _evidence = evaluate_official_rom_run_eligibility(run_root, run_id=run_id)
        if not eligible:
            result["blocked_reasons"] = reasons
            return result

    try:
        entry = register_official_rom_dataset_entry(
            repo_root=repo_root,
            sample_id=sample_id,
            run_id=run_id,
            run_root=run_root,
            shape_name=shape_name,
        )
    except ValueError as exc:
        result["error"] = str(exc)
        return result

    result["registered"] = True
    result["registry_entry"] = entry

    policy = policy or RetrainPolicy()
    state_path = repo_root / "ROM" / shape_name / "rom_retrain_state.json"
    state = load_json(state_path) if state_path.is_file() else {}
    new_count = int(state.get("new_samples_since_last_train") or 0) + 1
    write_json_atomic(
        state_path,
        {
            "schema": "m4_rom_retrain_state_v1",
            "last_registered_utc": utc_now(),
            "new_samples_since_last_train": new_count,
            "last_registered_run_id": run_id,
        },
    )
    result["new_samples_since_last_train"] = new_count
    if not policy.should_retrain(new_samples_since_last_train=new_count):
        return result

    result["retrain_attempted"] = True
    try:
        registry_rows = load_official_dataset_registry(repo_root, shape_name)
        allowed_run_ids = [str(r["run_id"]) for r in registry_rows if r.get("run_id")]
        _model, _training, _skipped, report = build_official_rom_surrogate_from_runs(
            repo_root=repo_root,
            shape_name=shape_name,
            require_initial_allowlist=False,
            allowed_run_ids=allowed_run_ids or None,
        )
        result["retrained"] = True
        result["retrain_status"] = "completed"
        result["retrain_report"] = report
    except Exception as exc:
        result["retrain_status"] = f"failed:{exc}"
    return result


SHADOW_STAGE_KEYS: Tuple[str, ...] = (
    "rom_prediction_present",
    "rom_prediction_created_before_fom",
    "rom_comparison_present",
    "rom_comparison_status",
    "compaction_status",
    "cleanup_status",
    "dataset_registration_attempted",
    "dataset_registration_status",
    "retrain_attempted",
    "retrain_status",
)


def _read_barrier_and_compaction(run_root: Path) -> Tuple[Optional[str], Optional[str], bool]:
    compaction_status: Optional[str] = None
    manifest_path = run_root / "compaction" / "compaction_manifest.json"
    if manifest_path.is_file():
        try:
            compaction_status = str(load_json(manifest_path).get("status") or "") or None
        except (OSError, ValueError, json.JSONDecodeError):
            compaction_status = None
    cleanup_status: Optional[str] = None
    verification_pass = False
    barrier_path = run_root / "cleanup" / "sample_cleanup_barrier.json"
    if barrier_path.is_file():
        try:
            barrier = load_json(barrier_path)
            cleanup_status = str(barrier.get("status") or "") or None
            verify = barrier.get("verification") if isinstance(barrier.get("verification"), dict) else {}
            verification_pass = bool(verify.get("pass")) if verify else bool(barrier.get("verification_pass"))
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    return compaction_status, cleanup_status, verification_pass


def diagnose_shadow_rom_stages(run_root: Path) -> Dict[str, Any]:
    """Read-only shadow ROM stage diagnostics for a run tree."""
    pred_path = rom_prediction_summary_path(run_root)
    cmp_path = rom_vs_fom_comparison_path(run_root)
    pred_present = pred_path.is_file()
    cmp_present = cmp_path.is_file()

    pred_before_fom: Optional[bool] = None
    if pred_present:
        try:
            summary = load_json(pred_path)
            marker_path = fom_started_marker_path(run_root)
            if marker_path.is_file():
                started_doc = load_json(marker_path)
                pred_perf = summary.get("prediction_recorded_perf_s")
                started_perf = started_doc.get("started_perf_s")
                if pred_perf is not None and started_perf is not None:
                    pred_before_fom = float(pred_perf) < float(started_perf)
                else:
                    pred_utc = str(summary.get("prediction_created_utc") or "")
                    started_utc = str(started_doc.get("started_utc") or "")
                    pred_before_fom = bool(pred_utc) and bool(started_utc) and pred_utc <= started_utc
            else:
                pred_before_fom = True
        except (OSError, ValueError, json.JSONDecodeError):
            pred_before_fom = None

    cmp_status: Optional[str] = None
    if cmp_present:
        try:
            cmp_status = str(load_json(cmp_path).get("status") or "") or None
        except (OSError, ValueError, json.JSONDecodeError):
            cmp_status = None

    compaction_status, cleanup_status, _verify_pass = _read_barrier_and_compaction(run_root)
    return {
        "rom_prediction_present": pred_present,
        "rom_prediction_created_before_fom": pred_before_fom,
        "rom_comparison_present": cmp_present,
        "rom_comparison_status": cmp_status,
        "compaction_status": compaction_status,
        "cleanup_status": cleanup_status,
        "dataset_registration_attempted": None,
        "dataset_registration_status": None,
        "retrain_attempted": None,
        "retrain_status": None,
    }


def verify_rom_prediction_summary(run_root: Path) -> Tuple[bool, Dict[str, Any]]:
    path = rom_prediction_summary_path(run_root)
    if not path.is_file():
        return False, {"error": f"missing:{path.name}"}
    try:
        summary = load_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return False, {"error": f"unreadable:{exc}"}
    if str(summary.get("schema") or "") != PREDICTION_SUMMARY_SCHEMA:
        return False, {"error": "invalid_schema"}
    if not summary.get("frozen_prediction_sha256"):
        return False, {"error": "missing_frozen_prediction_sha256"}
    frozen_path = frozen_prediction_internal_path(run_root)
    if not frozen_path.is_file():
        return False, {"error": "missing_frozen_internal"}
    return True, summary


def ensure_durable_rom_comparison(
    *,
    repo_root: Path,
    run_root: Path,
    context: Optional[Mapping[str, Any]] = None,
    max_match_distance_hz: float = DEFAULT_MAX_MATCH_DISTANCE_HZ,
    reuse_existing: bool = True,
) -> Dict[str, Any]:
    """Reuse durable comparison or materialize from frozen prediction + FOM catalog."""
    cmp_path = rom_vs_fom_comparison_path(run_root)
    if reuse_existing and cmp_path.is_file():
        try:
            doc = load_json(cmp_path)
            if str(doc.get("schema") or "") == COMPARISON_SUMMARY_SCHEMA:
                return {
                    "status": doc.get("status"),
                    "reused": True,
                    "comparison_path": rel(cmp_path, repo_root=repo_root),
                    "matched_mode_count": doc.get("matched_mode_count"),
                    "blocking": False,
                }
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    if context is None:
        frozen_path = frozen_prediction_internal_path(run_root)
        if frozen_path.is_file():
            try:
                frozen = load_json(frozen_path)
                context = dict(frozen.get("context") or {})
            except (OSError, ValueError, json.JSONDecodeError):
                context = None
    if not context:
        raise RomShadowIntegrityError("rom_comparison_context_unavailable")

    return run_shadow_rom_compare(
        repo_root=repo_root,
        run_root=run_root,
        context=context,
        max_match_distance_hz=max_match_distance_hz,
    )


def print_shadow_rom_stages(stages: Mapping[str, Any]) -> None:
    for key in SHADOW_STAGE_KEYS:
        print(f"{key}={stages.get(key)!r}")


def attempt_register_and_retrain_after_cleanup(
    *,
    repo_root: Path,
    run_root: Path,
    sample_id: str,
    run_id: str,
    shape_name: str = "classic",
    production_acceptance_pass: bool,
    policy: Optional[RetrainPolicy] = None,
) -> Dict[str, Any]:
    """Register/retrain only after compaction + cleanup eligibility passes."""
    stages = diagnose_shadow_rom_stages(run_root)
    stages["dataset_registration_attempted"] = False
    stages["retrain_attempted"] = False
    stages["retrain_status"] = "not_attempted"

    if not production_acceptance_pass:
        stages["dataset_registration_status"] = "skipped_production_acceptance"
        return {"registered": False, "retrained": False, "shadow_stages": stages}

    compaction_ok = str(stages.get("compaction_status") or "") in ("completed", "already_compacted")
    cleanup_ok = str(stages.get("cleanup_status") or "") == "completed"
    if not compaction_ok or not cleanup_ok:
        stages["dataset_registration_attempted"] = True
        stages["dataset_registration_status"] = (
            f"blocked:compaction={stages.get('compaction_status')!r} "
            f"cleanup={stages.get('cleanup_status')!r}"
        )
        return {
            "registered": False,
            "retrained": False,
            "shadow_stages": stages,
            "blocked_reason": "post_cleanup_gates_not_met",
        }

    stages["dataset_registration_attempted"] = True
    reg = maybe_register_and_retrain(
        repo_root=repo_root,
        run_root=run_root,
        sample_id=sample_id,
        run_id=run_id,
        shape_name=shape_name,
        production_acceptance_pass=True,
        policy=policy,
        require_post_cleanup_eligibility=True,
    )
    if reg.get("registered"):
        stages["dataset_registration_status"] = "registered"
    elif reg.get("blocked_reasons"):
        stages["dataset_registration_status"] = f"blocked:{reg.get('blocked_reasons')}"
    elif reg.get("error"):
        stages["dataset_registration_status"] = f"failed:{reg.get('error')}"
    else:
        stages["dataset_registration_status"] = "not_registered"

    if reg.get("retrain_attempted"):
        stages["retrain_attempted"] = True
        stages["retrain_status"] = reg.get("retrain_status") or ("completed" if reg.get("retrained") else "skipped_policy")
    elif reg.get("retrained"):
        stages["retrain_attempted"] = True
        stages["retrain_status"] = "completed"

    return {**reg, "shadow_stages": stages}


def prune_rom_directory_to_durable(run_root: Path) -> List[str]:
    """Keep only durable ROM JSON artifacts; remove internal/debug files."""
    rom_dir = rom_dir_for_run(run_root)
    if not rom_dir.is_dir():
        return []
    removed: List[str] = []
    keep = set(DURABLE_ROM_JSON_NAMES)
    for child in rom_dir.iterdir():
        if child.is_file() and child.name not in keep:
            child.unlink(missing_ok=True)
            removed.append(child.name)
    return removed


def run_shadow_rom_prepredict_nonblocking(
    *,
    repo_root: Path,
    run_root: Path,
    context: Mapping[str, Any],
    nev: int = DEFAULT_ROM_NEV,
    dataset_version: str = DATASET_VERSION_ROM,
) -> Dict[str, Any]:
    try:
        return run_shadow_rom_prepredict(
            repo_root=repo_root,
            run_root=run_root,
            context=context,
            nev=nev,
            dataset_version=dataset_version,
        )
    except RomShadowIntegrityError as exc:
        return {"status": ROM_STATUS_FAILED, "error": str(exc), "blocking": True}
    except Exception as exc:
        return {"status": ROM_STATUS_FAILED, "error": str(exc), "blocking": False}


def run_shadow_rom_compare_nonblocking(
    *,
    repo_root: Path,
    run_root: Path,
    context: Mapping[str, Any],
    max_match_distance_hz: float = DEFAULT_MAX_MATCH_DISTANCE_HZ,
) -> Dict[str, Any]:
    try:
        return run_shadow_rom_compare(
            repo_root=repo_root,
            run_root=run_root,
            context=context,
            max_match_distance_hz=max_match_distance_hz,
        )
    except RomShadowIntegrityError as exc:
        return {"status": ROM_COMPARE_FAILED, "error": str(exc), "blocking": True}
    except Exception as exc:
        return {
            "status": ROM_COMPARE_FAILED,
            "error": str(exc),
            "blocking": False,
            "low_accuracy_recorded_only": True,
        }
