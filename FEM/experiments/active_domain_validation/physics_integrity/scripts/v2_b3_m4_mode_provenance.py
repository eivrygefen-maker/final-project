#!/usr/bin/env python3
"""Per-mode solve provenance: eigenvector fingerprint, mic/bridge decomposition, strict audit fields."""
from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from v2_b3_rich_modal_lib import prolongate_active_to_W  # noqa: E402

PROVENANCE_FIELD_KEYS: Sequence[str] = (
    "lambda_real",
    "lambda_imag",
    "eps_compute_error_relative",
    "convergence_status",
    "solver_backend",
    "solver_fallback_used",
    "normalization_method",
    "normalization_scalar",
    "modal_mass_before_norm",
    "modal_mass_after_norm",
    "mic_numerator",
    "mic_denominator",
    "bridge_raw_coupling",
    "bridge_raw_coupling_abs",
    "participation_status",
    "eigenvector_fingerprint_sha256",
    "eigenvector_sketch_sha256",
    "physics_fallback_flags",
    "aperture_mask_sha256",
    "bridge_mask_sha256",
)

NORMALIZATION_METHOD = "full_W_l2_norm_v1"
EIGENVECTOR_QUANTIZE_DECIMALS = 8
SKETCH_SAMPLE_COUNT = 64


def _quantize_vector(vec: np.ndarray, *, decimals: int = EIGENVECTOR_QUANTIZE_DECIMALS) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float64).ravel()
    if v.size == 0:
        return v
    scale = float(np.max(np.abs(v)))
    if scale <= 0.0:
        return np.zeros_like(v)
    normed = v / scale
    return np.round(normed, decimals=decimals)


def eigenvector_fingerprint_sha256(x_active: np.ndarray) -> str:
    """Deterministic SHA256 of L2-normalized, quantized active eigenvector coefficients."""
    v = np.asarray(x_active, dtype=np.float64).ravel()
    if v.size == 0:
        return hashlib.sha256(b"empty").hexdigest()
    nrm = float(np.linalg.norm(v))
    if nrm <= 0.0:
        return hashlib.sha256(b"zero").hexdigest()
    q = _quantize_vector(v / nrm)
    return hashlib.sha256(q.tobytes()).hexdigest()


def eigenvector_sketch_sha256(x_active: np.ndarray, *, n_samples: int = SKETCH_SAMPLE_COUNT) -> str:
    """Compact deterministic sketch: evenly spaced quantized samples of normalized vector."""
    v = np.asarray(x_active, dtype=np.float64).ravel()
    if v.size == 0:
        return hashlib.sha256(b"sketch_empty").hexdigest()
    nrm = float(np.linalg.norm(v))
    if nrm <= 0.0:
        return hashlib.sha256(b"sketch_zero").hexdigest()
    normed = v / nrm
    if v.size <= n_samples:
        sketch = _quantize_vector(normed)
    else:
        idx = np.linspace(0, v.size - 1, num=n_samples, dtype=np.int64)
        sketch = _quantize_vector(normed[idx])
    return hashlib.sha256(sketch.tobytes()).hexdigest()


def _sha256_indices(indices: Sequence[int]) -> Optional[str]:
    if not indices:
        return None
    payload = ",".join(str(int(i)) for i in sorted(int(x) for x in indices))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _rms(vals: np.ndarray) -> float:
    v = np.asarray(vals, dtype=np.float64).ravel()
    if v.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(v * v)))


def attach_mode_provenance(
    entry: Dict[str, Any],
    *,
    x_active: np.ndarray,
    built: Mapping[str, Any],
    region_ctx: Mapping[str, Any],
    solver_backend: str = "mkl_pardiso",
    solver_fallback_used: bool = False,
) -> None:
    """Augment accepted-mode dict with strict audit / fingerprint fields."""
    region = region_ctx.get("region") or {}
    x_full = prolongate_active_to_W(np.asarray(x_active, dtype=np.float64), dict(built))
    modal_norm = float(np.linalg.norm(x_full))
    entry["normalization_method"] = NORMALIZATION_METHOD
    entry["normalization_scalar"] = modal_norm if modal_norm > 0.0 else None
    entry["modal_mass_before_norm"] = modal_norm
    entry["modal_mass_after_norm"] = 1.0 if modal_norm > 0.0 else None

    p_ap = np.asarray(region.get("p_idx_aperture", []), dtype=np.int32).ravel()
    bridge_idx = np.asarray(region.get("u_idx_bridge", region.get("u_idx_top", [])), dtype=np.int32).ravel()
    if bridge_idx.size == 0:
        bridge_idx = np.asarray(region.get("u_idx_top", []), dtype=np.int32).ravel()

    aperture_vals = x_full[p_ap] if p_ap.size else np.asarray([], dtype=np.float64)
    bridge_vals = x_full[bridge_idx] if bridge_idx.size else np.asarray([], dtype=np.float64)
    aperture_rms = _rms(aperture_vals)
    bridge_raw = float(np.mean(bridge_vals)) if bridge_vals.size else None
    bridge_raw_abs = _rms(bridge_vals) if bridge_vals.size else None

    entry["mic_numerator"] = aperture_rms if p_ap.size else None
    entry["mic_denominator"] = modal_norm if modal_norm > 0.0 else None
    entry["bridge_raw_coupling"] = bridge_raw
    entry["bridge_raw_coupling_abs"] = bridge_raw_abs
    entry["eigenvector_fingerprint_sha256"] = eigenvector_fingerprint_sha256(x_active)
    entry["eigenvector_sketch_sha256"] = eigenvector_sketch_sha256(x_active)
    entry["solver_backend"] = solver_backend
    entry["solver_fallback_used"] = bool(solver_fallback_used)
    entry["convergence_status"] = (
        "converged" if entry.get("eps_compute_error_relative") is not None else "unknown"
    )
    entry["aperture_mask_sha256"] = _sha256_indices(p_ap.tolist())
    entry["bridge_mask_sha256"] = _sha256_indices(bridge_idx.tolist())
    entry["physics_fallback_flags"] = {
        "solver_fallback_used": bool(solver_fallback_used),
        "participation_fallback": str(entry.get("participation_status") or "") == "fallback",
        "mic_method_fallback": str(entry.get("mic_output_method") or "") != "aperture_pressure_rms_proxy_v1",
        "audio_coupling_fallback": str(entry.get("audio_coupling_status") or "") == "not_available",
    }


def merge_provenance_into_catalog_record(record: Dict[str, Any], mode: Mapping[str, Any]) -> None:
    for key in PROVENANCE_FIELD_KEYS:
        if key in mode:
            record[key] = mode[key]
