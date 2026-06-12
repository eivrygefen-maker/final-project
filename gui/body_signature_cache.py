#!/usr/bin/env python3
"""Diagnostic body signature cache for STK V4 (per-sample transfer envelopes)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

DEFAULT_GRID_HZ = (60.0, 1000.0, 512)


def _params_hash(parameters: Mapping[str, Any]) -> str:
    blob = json.dumps(dict(parameters), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def cache_paths(repo_root: Path, sample_id: str) -> Tuple[Path, Path]:
    base = repo_root / "ROM" / "classic" / "body_signature_cache"
    return base / f"{sample_id}.json", base / f"{sample_id}.npz"


def write_body_signature_cache(
    repo_root: Path,
    sample_id: str,
    *,
    frequencies_hz: np.ndarray,
    G_sample: np.ndarray,
    logG_sample: np.ndarray,
    D_sample: np.ndarray,
    modal_weights: Sequence[float],
    metadata: Mapping[str, Any],
) -> None:
    json_path, npz_path = cache_paths(repo_root, sample_id)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        frequencies_hz=np.asarray(frequencies_hz, dtype=np.float64),
        G_sample=np.asarray(G_sample, dtype=np.float64),
        logG_sample=np.asarray(logG_sample, dtype=np.float64),
        D_sample=np.asarray(D_sample, dtype=np.float64),
        modal_weights=np.asarray(modal_weights, dtype=np.float64),
    )
    doc = {
        "sample_id": sample_id,
        "frequencies_hz_count": int(len(frequencies_hz)),
        "modal_weights_count": int(len(modal_weights)),
        **dict(metadata),
    }
    json_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")


def load_body_signature_cache(
    repo_root: Path,
    sample_id: str,
) -> Optional[Dict[str, Any]]:
    json_path, npz_path = cache_paths(repo_root, sample_id)
    if not npz_path.is_file():
        return None
    meta: Dict[str, Any] = {}
    if json_path.is_file():
        meta = json.loads(json_path.read_text(encoding="utf-8"))
    with np.load(npz_path) as z:
        return {
            **meta,
            "frequencies_hz": z["frequencies_hz"],
            "G_sample": z["G_sample"],
            "logG_sample": z["logG_sample"],
            "D_sample": z["D_sample"],
            "modal_weights": z["modal_weights"],
        }


def build_reference_logG(logG_stack: Sequence[np.ndarray]) -> np.ndarray:
    if not logG_stack:
        return np.zeros(0, dtype=np.float64)
    arr = np.stack([np.asarray(x, dtype=np.float64) for x in logG_stack], axis=0)
    return np.median(arr, axis=0)


def interpolate_D_at_frequency(D: np.ndarray, freqs: np.ndarray, f_query: float) -> float:
    fq = float(f_query)
    if D.size == 0 or freqs.size == 0:
        return 0.0
    idx = int(np.argmin(np.abs(freqs - fq)))
    return float(D[idx])
