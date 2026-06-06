#!/usr/bin/env python3
"""Lightweight M4 modal frequency surrogate — trains from modes_catalog.jsonl, no mode shapes."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from v2_b3_m4_lhs_pool_bridge import (  # noqa: E402
    AGG_PASS,
    DEFAULT_RUN_ID_SUFFIX,
    is_lhs_entry_completed,
    load_lhs_pool,
)
from v2_b3_m4_rom_fom_compare_lib import load_fom_modes_catalog  # noqa: E402
from v2_b3_m4_rom_scalar_fields import (  # noqa: E402
    PHASE2_CATEGORICAL_FIELDS,
    PHASE2_NUMERIC_FIELDS,
    PHASE2_PREDICTION_METHOD,
    _encode_categorical,
    categorical_vocab_for_field,
    predict_mode_scalars_at_frequency,
)
from v2_b3_m4_worker_run_lib import detect_repo_root, load_json, rel, utc_now  # noqa: E402
from v2_b3_petsc_util import write_json_atomic  # noqa: E402

SURROGATE_SCHEMA_V1 = "m4_modal_surrogate_v1"
SURROGATE_SCHEMA_V2 = "m4_modal_surrogate_v2"
SURROGATE_SCHEMA = SURROGATE_SCHEMA_V2
MANIFEST_SCHEMA = "m4_rom_model_manifest_v1"

SURROGATE_JSON_NAME = "m4_modal_surrogate.json"
SURROGATE_NPZ_NAME = "m4_modal_surrogate.npz"
MANIFEST_JSON_NAME = "rom_model_manifest.json"
LEGACY_BASIS_NAME = "reduced_basis.npz"

DEFAULT_K_NEIGHBORS = 5
DEFAULT_PREDICTION_METHOD_V1 = "knn_idw_sorted_modes"
DEFAULT_PREDICTION_METHOD_V2 = "knn_idw_modal_surrogate_v2"
DEFAULT_PREDICTION_METHOD = DEFAULT_PREDICTION_METHOD_V2

GEOMETRY_KEYS: Tuple[str, ...] = (
    "geometry.length",
    "geometry.width",
    "geometry.depth",
    "geometry.top_thickness",
    "geometry.hole_radius",
    "geometry.back_thickness",
)

WOOD_IDS: Tuple[str, ...] = ("spruce", "cedar", "mahogany", "rosewood", "maple")
FEATURE_NAMES: Tuple[str, ...] = GEOMETRY_KEYS + ("top_wood_id", "back_wood_id")


def shape_rom_dir(repo_root: Path, shape_name: str) -> Path:
    return repo_root / "ROM" / shape_name


def surrogate_json_path(repo_root: Path, shape_name: str) -> Path:
    return shape_rom_dir(repo_root, shape_name) / SURROGATE_JSON_NAME


def surrogate_npz_path(repo_root: Path, shape_name: str) -> Path:
    return shape_rom_dir(repo_root, shape_name) / SURROGATE_NPZ_NAME


def manifest_path(repo_root: Path, shape_name: str) -> Path:
    return shape_rom_dir(repo_root, shape_name) / MANIFEST_JSON_NAME


def legacy_basis_path(repo_root: Path, shape_name: str) -> Path:
    return shape_rom_dir(repo_root, shape_name) / LEGACY_BASIS_NAME


def guitars_root(repo_root: Path) -> Path:
    return (
        repo_root
        / "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars"
    )


def _normalize_wood(wood_id: Any) -> str:
    return str(wood_id or "").strip().lower().replace(" ", "_")


def encode_lhs_parameters(parameters: Mapping[str, Any]) -> np.ndarray:
    geom = [float(parameters.get(k, 0.0)) for k in GEOMETRY_KEYS]
    top = _normalize_wood(parameters.get("top_wood_id"))
    back = _normalize_wood(parameters.get("back_wood_id"))
    top_idx = float(WOOD_IDS.index(top)) if top in WOOD_IDS else 0.0
    back_idx = float(WOOD_IDS.index(back)) if back in WOOD_IDS else 0.0
    return np.array(geom + [top_idx, back_idx], dtype=np.float64)


def surrogate_is_available(repo_root: Path, shape_name: str) -> bool:
    return surrogate_json_path(repo_root, shape_name).is_file() and surrogate_npz_path(
        repo_root, shape_name
    ).is_file()


def legacy_basis_is_available(repo_root: Path, shape_name: str) -> bool:
    return legacy_basis_path(repo_root, shape_name).is_file()


def resolve_active_rom_backend(repo_root: Path, shape_name: str) -> str:
    """Return 'm4_surrogate', 'legacy_basis', or 'none'."""
    manifest = load_rom_model_manifest(repo_root, shape_name)
    preferred = str(manifest.get("active_backend") or "auto")
    if preferred == "m4_surrogate" and surrogate_is_available(repo_root, shape_name):
        return "m4_surrogate"
    if preferred == "legacy_basis" and legacy_basis_is_available(repo_root, shape_name):
        return "legacy_basis"
    if preferred == "auto":
        if surrogate_is_available(repo_root, shape_name):
            return "m4_surrogate"
        if legacy_basis_is_available(repo_root, shape_name):
            return "legacy_basis"
    return "none"


def load_rom_model_manifest(repo_root: Path, shape_name: str) -> Dict[str, Any]:
    path = manifest_path(repo_root, shape_name)
    if path.is_file():
        try:
            data = load_json(path)
            if isinstance(data, dict):
                return data
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    return {
        "schema": MANIFEST_SCHEMA,
        "active_backend": "auto",
        "m4_surrogate_json": SURROGATE_JSON_NAME,
        "legacy_basis_npz": LEGACY_BASIS_NAME,
    }


def write_rom_model_manifest(
    repo_root: Path,
    shape_name: str,
    *,
    active_backend: str = "m4_surrogate",
    training_sample_count: int = 0,
) -> Path:
    out = manifest_path(repo_root, shape_name)
    body = {
        "schema": MANIFEST_SCHEMA,
        "generated_utc": utc_now(),
        "shape_name": shape_name,
        "active_backend": active_backend,
        "m4_surrogate_json": SURROGATE_JSON_NAME,
        "m4_surrogate_npz": SURROGATE_NPZ_NAME,
        "legacy_basis_npz": LEGACY_BASIS_NAME,
        "training_sample_count": int(training_sample_count),
        "surrogate_schema": SURROGATE_SCHEMA_V2,
        "phase2_scalar_fields": True,
        "notes": (
            "M4 surrogate trains from aggregation/modes_catalog.jsonl (frequencies + scalar metadata). "
            "Legacy reduced_basis.npz requires full eigenvector snapshots and is optional."
        ),
    }
    write_json_atomic(out, body)
    return out


def collect_completed_fom_training_rows(
    *,
    repo_root: Path,
    pool: Mapping[str, Any],
    run_id_suffix: str = DEFAULT_RUN_ID_SUFFIX,
    completed_only: bool = True,
    max_samples: Optional[int] = None,
    force_sample: Optional[str] = None,
    exclude_sample_ids: Optional[Sequence[str]] = None,
    min_mode_count: int = 1,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (training_rows, skipped_rows) from completed M4 FOM runs."""
    training: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    excluded = {str(s).strip() for s in (exclude_sample_ids or []) if str(s).strip()}
    pool_shape = str(pool.get("shape_name") or "classic")

    for i, entry in enumerate(pool.get("entries") or []):
        sid = str(entry.get("id") or "").strip()
        if not sid:
            continue
        if sid in excluded:
            skipped.append({"sample_id": sid, "reason": "excluded_from_training"})
            continue
        if force_sample and sid != force_sample:
            continue

        run_id = str(entry.get("last_run_id") or f"{sid}_{run_id_suffix}")
        status = str(entry.get("status") or "").upper()
        agg = str(entry.get("last_aggregation_status") or "")

        if completed_only and not force_sample:
            if not is_lhs_entry_completed(entry, run_id=run_id) and agg != AGG_PASS:
                if status not in ("COMPLETED", "PASS"):
                    skipped.append({"sample_id": sid, "reason": "not_completed"})
                    continue

        run_root = guitars_root(repo_root) / sid / "runs" / run_id
        catalog_path = run_root / "aggregation" / "modes_catalog.jsonl"
        if not catalog_path.is_file():
            skipped.append(
                {
                    "sample_id": sid,
                    "run_id": run_id,
                    "reason": "missing_modes_catalog",
                    "path": rel(catalog_path, repo_root=repo_root),
                }
            )
            continue

        try:
            modes = load_fom_modes_catalog(catalog_path)
            freqs = [float(m["frequency_hz"]) for m in modes]
        except (OSError, ValueError, FileNotFoundError) as exc:
            skipped.append({"sample_id": sid, "reason": "catalog_read_error", "error": str(exc)})
            continue

        if len(freqs) < int(min_mode_count):
            skipped.append(
                {
                    "sample_id": sid,
                    "reason": "insufficient_modes",
                    "mode_count": len(freqs),
                }
            )
            continue

        params = dict(entry.get("parameters") or {})
        sample_input = run_root / "sample" / "sample_input.json"
        if sample_input.is_file():
            try:
                si = load_json(sample_input)
                if isinstance(si.get("parameters"), dict):
                    params = dict(si["parameters"])
            except (OSError, ValueError, json.JSONDecodeError):
                pass

        training.append(
            {
                "sample_id": sid,
                "lhs_row_index": i,
                "run_id": run_id,
                "run_root": rel(run_root, repo_root=repo_root),
                "catalog_path": rel(catalog_path, repo_root=repo_root),
                "shape_name": pool_shape,
                "parameters": params,
                "frequencies_hz": freqs,
                "mode_catalog": modes,
                "mode_count": len(freqs),
            }
        )
        if max_samples is not None and len(training) >= max_samples:
            break

    return training, skipped


def build_surrogate_from_training_rows(
    *,
    shape_name: str,
    training_rows: Sequence[Mapping[str, Any]],
    k_neighbors: int = DEFAULT_K_NEIGHBORS,
) -> Dict[str, Any]:
    if not training_rows:
        raise ValueError("no training rows")

    features = np.vstack([encode_lhs_parameters(r["parameters"]) for r in training_rows])
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    x_norm = (features - mean) / std

    mode_counts = np.array([len(r["frequencies_hz"]) for r in training_rows], dtype=np.int32)
    max_modes = int(mode_counts.max())
    freq_matrix = np.full((len(training_rows), max_modes), np.nan, dtype=np.float64)
    scalar_arrays: Dict[str, np.ndarray] = {
        field: np.full((len(training_rows), max_modes), np.nan, dtype=np.float64)
        for field in PHASE2_NUMERIC_FIELDS
    }
    cat_arrays: Dict[str, np.ndarray] = {
        field: np.full((len(training_rows), max_modes), -1, dtype=np.int32)
        for field in PHASE2_CATEGORICAL_FIELDS
    }

    for i, row in enumerate(training_rows):
        freqs = row["frequencies_hz"]
        freq_matrix[i, : len(freqs)] = np.asarray(freqs, dtype=np.float64)
        catalog = list(row.get("mode_catalog") or [])
        for j, mode in enumerate(catalog[:max_modes]):
            for field in PHASE2_NUMERIC_FIELDS:
                val = mode.get(field)
                try:
                    scalar_arrays[field][i, j] = float(val) if val is not None else np.nan
                except (TypeError, ValueError):
                    scalar_arrays[field][i, j] = np.nan
            for field in PHASE2_CATEGORICAL_FIELDS:
                vocab = categorical_vocab_for_field(field)
                cat_arrays[field][i, j] = _encode_categorical(mode.get(field), vocab)

    arrays: Dict[str, Any] = {
        "feature_matrix_norm": x_norm,
        "frequencies": freq_matrix,
        "mode_counts": mode_counts,
        "feature_mean": mean,
        "feature_std": std,
    }
    for field, arr in scalar_arrays.items():
        arrays[f"scalar__{field}"] = arr
    for field, arr in cat_arrays.items():
        arrays[f"cat__{field}"] = arr

    return {
        "schema": SURROGATE_SCHEMA_V2,
        "generated_utc": utc_now(),
        "shape_name": shape_name,
        "method": DEFAULT_PREDICTION_METHOD_V2,
        "scalar_alignment": "nearest_frequency_per_neighbor",
        "k_neighbors": int(min(k_neighbors, len(training_rows))),
        "feature_names": list(FEATURE_NAMES),
        "wood_ids": list(WOOD_IDS),
        "training_sample_count": len(training_rows),
        "training_samples": [
            {
                "sample_id": r["sample_id"],
                "lhs_row_index": r["lhs_row_index"],
                "run_id": r["run_id"],
                "mode_count": r["mode_count"],
                "catalog_path": r.get("catalog_path"),
            }
            for r in training_rows
        ],
        "mode_count_min": int(mode_counts.min()),
        "mode_count_max": int(mode_counts.max()),
        "mode_count_median": float(np.median(mode_counts)),
        "phase2_numeric_fields": list(PHASE2_NUMERIC_FIELDS),
        "phase2_categorical_fields": list(PHASE2_CATEGORICAL_FIELDS),
        "arrays": arrays,
    }


def save_surrogate_model(repo_root: Path, model: Mapping[str, Any]) -> Dict[str, Path]:
    shape = str(model["shape_name"])
    out_dir = shape_rom_dir(repo_root, shape)
    out_dir.mkdir(parents=True, exist_ok=True)

    arrays = dict(model["arrays"])
    json_body = {k: v for k, v in model.items() if k != "arrays"}
    json_path = surrogate_json_path(repo_root, shape)
    npz_path = surrogate_npz_path(repo_root, shape)

    write_json_atomic(json_path, json_body)
    npz_payload = {
        "feature_matrix_norm": np.asarray(arrays["feature_matrix_norm"], dtype=np.float64),
        "frequencies": np.asarray(arrays["frequencies"], dtype=np.float64),
        "mode_counts": np.asarray(arrays["mode_counts"], dtype=np.int32),
        "feature_mean": np.asarray(arrays["feature_mean"], dtype=np.float64),
        "feature_std": np.asarray(arrays["feature_std"], dtype=np.float64),
        "k_neighbors": np.array([int(model.get("k_neighbors") or DEFAULT_K_NEIGHBORS)], dtype=np.int32),
    }
    for key, val in arrays.items():
        if key.startswith(("scalar__", "cat__")):
            npz_payload[key] = np.asarray(val)
    np.savez(npz_path, **npz_payload)
    manifest = write_rom_model_manifest(
        repo_root,
        shape,
        active_backend="m4_surrogate",
        training_sample_count=int(model.get("training_sample_count") or 0),
    )
    return {"json": json_path, "npz": npz_path, "manifest": manifest}


def load_surrogate_model(repo_root: Path, shape_name: str) -> Dict[str, Any]:
    json_path = surrogate_json_path(repo_root, shape_name)
    npz_path = surrogate_npz_path(repo_root, shape_name)
    if not json_path.is_file() or not npz_path.is_file():
        raise FileNotFoundError(
            f"M4 modal surrogate missing for shape {shape_name!r}: "
            f"expected {json_path.name} and {npz_path.name}"
        )
    meta = load_json(json_path)
    with np.load(npz_path, allow_pickle=False) as z:
        arrays: Dict[str, Any] = {
            "feature_matrix_norm": np.asarray(z["feature_matrix_norm"], dtype=np.float64),
            "frequencies": np.asarray(z["frequencies"], dtype=np.float64),
            "mode_counts": np.asarray(z["mode_counts"], dtype=np.int32),
            "feature_mean": np.asarray(z["feature_mean"], dtype=np.float64),
            "feature_std": np.asarray(z["feature_std"], dtype=np.float64),
        }
        for name in z.files:
            if name.startswith(("scalar__", "cat__")):
                arrays[name] = np.asarray(z[name])
        meta["arrays"] = arrays
        if "k_neighbors" in z.files:
            meta["k_neighbors"] = int(np.asarray(z["k_neighbors"]).reshape(-1)[0])
    return meta


def _neighbor_catalog_from_arrays(
    arrays: Mapping[str, Any],
    row_index: int,
    mode_count: int,
) -> List[Dict[str, Any]]:
    """Reconstruct sorted mode dicts from stored training arrays."""
    from v2_b3_m4_rom_scalar_fields import decode_categorical  # noqa: WPS433

    modes: List[Dict[str, Any]] = []
    for j in range(mode_count):
        rec: Dict[str, Any] = {
            "frequency_hz": float(arrays["frequencies"][row_index, j]),
        }
        for field in PHASE2_NUMERIC_FIELDS:
            key = f"scalar__{field}"
            if key in arrays:
                val = float(arrays[key][row_index, j])
                rec[field] = None if val != val else val
        for field in PHASE2_CATEGORICAL_FIELDS:
            key = f"cat__{field}"
            if key in arrays:
                idx = int(arrays[key][row_index, j])
                vocab = categorical_vocab_for_field(field)
                rec[field] = decode_categorical(idx, vocab)
        modes.append(rec)
    return modes


def _select_neighbors(
    model: Mapping[str, Any],
    parameters: Mapping[str, Any],
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    arrays = model["arrays"]
    x = encode_lhs_parameters(parameters)
    x_norm = (x - arrays["feature_mean"]) / arrays["feature_std"]
    x_train = arrays["feature_matrix_norm"]
    dists = np.linalg.norm(x_train - x_norm.reshape(1, -1), axis=1)
    k = int(min(model.get("k_neighbors") or DEFAULT_K_NEIGHBORS, len(dists)))
    k = max(1, k)
    nn_idx = np.argsort(dists)[:k]
    weights = 1.0 / (dists[nn_idx] + 1e-8) ** 2
    weights = weights / weights.sum()
    samples = list(model.get("training_samples") or [])
    neighbor_ids: List[str] = []
    for j in nn_idx:
        if j < len(samples):
            neighbor_ids.append(str(samples[j].get("sample_id") or f"train_{j}"))
        else:
            neighbor_ids.append(f"train_{j}")
    return nn_idx, weights, neighbor_ids, dists


def predict_modal_catalog(
    model: Mapping[str, Any],
    parameters: Mapping[str, Any],
    *,
    nev: int = 0,
) -> Dict[str, Any]:
    """Phase-2: predict frequencies (sorted-index IDW) + scalars (nearest-frequency IDW)."""
    arrays = model["arrays"]
    nn_idx, weights, neighbor_ids, dists = _select_neighbors(model, parameters)
    k = len(nn_idx)

    neighbor_freqs: List[np.ndarray] = []
    neighbor_catalogs: List[List[Dict[str, Any]]] = []
    for j in nn_idx:
        count = int(arrays["mode_counts"][j])
        freqs = np.asarray(arrays["frequencies"][j, :count], dtype=np.float64)
        neighbor_freqs.append(freqs)
        neighbor_catalogs.append(_neighbor_catalog_from_arrays(arrays, int(j), count))

    min_count = min(len(f) for f in neighbor_freqs)
    mode_out = min(int(nev), min_count) if nev > 0 else min_count

    pred_freqs = np.zeros(mode_out, dtype=np.float64)
    for m in range(mode_out):
        pred_freqs[m] = np.average([float(f[m]) for f in neighbor_freqs], weights=weights)

    predicted_modes: List[Dict[str, Any]] = []
    for mi, f_hz in enumerate(pred_freqs.tolist()):
        mode_rec = predict_mode_scalars_at_frequency(
            target_hz=float(f_hz),
            neighbor_catalogs=neighbor_catalogs,
            neighbor_weights=list(weights),
        )
        mode_rec["mode_index"] = int(mi)
        predicted_modes.append(mode_rec)

    schema = str(model.get("schema") or SURROGATE_SCHEMA_V1)
    method = (
        DEFAULT_PREDICTION_METHOD_V2
        if schema == SURROGATE_SCHEMA_V2
        else str(model.get("method") or DEFAULT_PREDICTION_METHOD_V1)
    )

    return {
        "frequencies_hz": [round(float(f), 6) for f in pred_freqs.tolist()],
        "predicted_modes": predicted_modes,
        "nev_returned": int(mode_out),
        "k_neighbors_used": int(k),
        "neighbor_sample_ids": neighbor_ids,
        "neighbor_distances": [round(float(dists[i]), 6) for i in nn_idx],
        "method": method,
        "scalar_alignment": "nearest_frequency_per_neighbor",
        "source_json": SURROGATE_JSON_NAME,
        "source_npz": SURROGATE_NPZ_NAME,
        "surrogate_schema": schema,
    }


def predict_modal_frequencies(
    model: Mapping[str, Any],
    parameters: Mapping[str, Any],
    *,
    nev: int = 0,
) -> Dict[str, Any]:
    out = predict_modal_catalog(model, parameters, nev=nev)
    return {
        "frequencies_hz": out["frequencies_hz"],
        "nev_returned": out["nev_returned"],
        "k_neighbors_used": out["k_neighbors_used"],
        "neighbor_sample_ids": out["neighbor_sample_ids"],
        "neighbor_distances": out.get("neighbor_distances"),
        "method": out["method"],
        "source_json": out["source_json"],
        "source_npz": out["source_npz"],
        "predicted_modes": out.get("predicted_modes"),
    }


def production_surrogate_training_sample_ids(
    repo_root: Path,
    shape_name: str,
) -> Tuple[List[str], Optional[Dict[str, Any]]]:
    """Return (training_sample_ids, model) from on-disk production surrogate."""
    try:
        model = load_surrogate_model(repo_root, shape_name)
    except FileNotFoundError:
        return [], None
    ids = [str(s.get("sample_id") or "") for s in (model.get("training_samples") or [])]
    ids = [s for s in ids if s]
    return ids, model


def build_holdout_surrogate_model(
    *,
    repo_root: Path,
    pool: Mapping[str, Any],
    shape_name: str,
    exclude_sample_ids: Sequence[str],
    run_id_suffix: str = DEFAULT_RUN_ID_SUFFIX,
    k_neighbors: int = DEFAULT_K_NEIGHBORS,
    min_mode_count: int = 1,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Build an in-memory surrogate excluding holdout sample(s). Does not write production files.
    """
    training, skipped = collect_completed_fom_training_rows(
        repo_root=repo_root,
        pool=pool,
        run_id_suffix=run_id_suffix,
        completed_only=True,
        max_samples=None,
        exclude_sample_ids=exclude_sample_ids,
        min_mode_count=min_mode_count,
    )
    if not training:
        raise ValueError(
            f"holdout surrogate has no training rows after excluding {list(exclude_sample_ids)}"
        )
    model = build_surrogate_from_training_rows(
        shape_name=shape_name,
        training_rows=training,
        k_neighbors=k_neighbors,
    )
    model["holdout_validation"] = True
    model["excluded_sample_ids"] = list(exclude_sample_ids)
    return model, training
