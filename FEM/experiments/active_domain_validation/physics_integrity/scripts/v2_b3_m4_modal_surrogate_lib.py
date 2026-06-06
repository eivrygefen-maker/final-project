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
from v2_b3_m4_worker_run_lib import detect_repo_root, load_json, rel, utc_now  # noqa: E402
from v2_b3_petsc_util import write_json_atomic  # noqa: E402

SURROGATE_SCHEMA = "m4_modal_surrogate_v1"
MANIFEST_SCHEMA = "m4_rom_model_manifest_v1"

SURROGATE_JSON_NAME = "m4_modal_surrogate.json"
SURROGATE_NPZ_NAME = "m4_modal_surrogate.npz"
MANIFEST_JSON_NAME = "rom_model_manifest.json"
LEGACY_BASIS_NAME = "reduced_basis.npz"

DEFAULT_K_NEIGHBORS = 5
DEFAULT_PREDICTION_METHOD = "knn_idw_sorted_modes"

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
        "notes": (
            "M4 surrogate trains from aggregation/modes_catalog.jsonl frequencies only. "
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
    min_mode_count: int = 1,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (training_rows, skipped_rows) from completed M4 FOM runs."""
    training: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    for i, entry in enumerate(pool.get("entries") or []):
        sid = str(entry.get("id") or "").strip()
        if not sid:
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
                "parameters": params,
                "frequencies_hz": freqs,
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
    for i, row in enumerate(training_rows):
        freqs = row["frequencies_hz"]
        freq_matrix[i, : len(freqs)] = np.asarray(freqs, dtype=np.float64)

    return {
        "schema": SURROGATE_SCHEMA,
        "generated_utc": utc_now(),
        "shape_name": shape_name,
        "method": DEFAULT_PREDICTION_METHOD,
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
        "arrays": {
            "feature_matrix_norm": x_norm,
            "frequencies": freq_matrix,
            "mode_counts": mode_counts,
            "feature_mean": mean,
            "feature_std": std,
        },
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
    np.savez(
        npz_path,
        feature_matrix_norm=np.asarray(arrays["feature_matrix_norm"], dtype=np.float64),
        frequencies=np.asarray(arrays["frequencies"], dtype=np.float64),
        mode_counts=np.asarray(arrays["mode_counts"], dtype=np.int32),
        feature_mean=np.asarray(arrays["feature_mean"], dtype=np.float64),
        feature_std=np.asarray(arrays["feature_std"], dtype=np.float64),
        k_neighbors=np.array([int(model.get("k_neighbors") or DEFAULT_K_NEIGHBORS)], dtype=np.int32),
    )
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
        meta["arrays"] = {
            "feature_matrix_norm": np.asarray(z["feature_matrix_norm"], dtype=np.float64),
            "frequencies": np.asarray(z["frequencies"], dtype=np.float64),
            "mode_counts": np.asarray(z["mode_counts"], dtype=np.int32),
            "feature_mean": np.asarray(z["feature_mean"], dtype=np.float64),
            "feature_std": np.asarray(z["feature_std"], dtype=np.float64),
        }
        if "k_neighbors" in z.files:
            meta["k_neighbors"] = int(np.asarray(z["k_neighbors"]).reshape(-1)[0])
    return meta


def predict_modal_frequencies(
    model: Mapping[str, Any],
    parameters: Mapping[str, Any],
    *,
    nev: int = 0,
) -> Dict[str, Any]:
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

    neighbor_freqs: List[np.ndarray] = []
    neighbor_ids: List[str] = []
    samples = list(model.get("training_samples") or [])
    for j in nn_idx:
        count = int(arrays["mode_counts"][j])
        freqs = np.asarray(arrays["frequencies"][j, :count], dtype=np.float64)
        neighbor_freqs.append(freqs)
        if j < len(samples):
            neighbor_ids.append(str(samples[j].get("sample_id") or f"train_{j}"))
        else:
            neighbor_ids.append(f"train_{j}")

    min_count = min(len(f) for f in neighbor_freqs)
    if nev > 0:
        mode_out = min(int(nev), min_count)
    else:
        mode_out = min_count

    pred = np.zeros(mode_out, dtype=np.float64)
    for m in range(mode_out):
        pred[m] = np.average([float(f[m]) for f in neighbor_freqs], weights=weights)

    return {
        "frequencies_hz": [round(float(f), 6) for f in pred.tolist()],
        "nev_returned": int(mode_out),
        "k_neighbors_used": int(k),
        "neighbor_sample_ids": neighbor_ids,
        "neighbor_distances": [round(float(dists[i]), 6) for i in nn_idx],
        "method": str(model.get("method") or DEFAULT_PREDICTION_METHOD),
        "source_json": SURROGATE_JSON_NAME,
        "source_npz": SURROGATE_NPZ_NAME,
    }
