#!/usr/bin/env python3
"""
Overlapping spectral bands for multi-shift FSI harvest (90–480 Hz).

Each band defines a worker ``target_hz`` (scheduler shift center), an ST σ offset
(12% in ``fem_main_3d``), and a **harvest window** wide enough to keep physical
modes on both sides of σ without relying on a single global narrow window.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

SWEEP_HZ_MIN = 90.0
SWEEP_HZ_MAX = 480.0

# Band centers every ~52 Hz with ±46 Hz harvest → ~40 Hz overlap between neighbors.
SPECTRAL_BAND_STEP_HZ = 52.0
SPECTRAL_BAND_HALF_HZ = 46.0


def hz_shift_quantize(hz: float, tol: float = 1e-4) -> float:
    return round(float(hz) / tol) * tol


def spectral_band_centers(
    hz_min: float = SWEEP_HZ_MIN,
    hz_max: float = SWEEP_HZ_MAX,
    *,
    step_hz: float = SPECTRAL_BAND_STEP_HZ,
    half_hz: float = SPECTRAL_BAND_HALF_HZ,
) -> List[float]:
    """Band center frequencies covering ``[hz_min, hz_max]`` with overlapping harvest."""
    lo = float(hz_min)
    hi = float(hz_max)
    half = max(8.0, float(half_hz))
    step = max(half, float(step_hz))
    centers: List[float] = []
    c = lo + half
    while c <= hi + 1e-9:
        centers.append(hz_shift_quantize(c))
        c += step
    if not centers:
        centers.append(hz_shift_quantize(0.5 * (lo + hi)))
    return centers


def nearest_band_center(
    target_hz: float,
    hz_min: float = SWEEP_HZ_MIN,
    hz_max: float = SWEEP_HZ_MAX,
    *,
    step_hz: float = SPECTRAL_BAND_STEP_HZ,
    half_hz: float = SPECTRAL_BAND_HALF_HZ,
) -> float:
    centers = spectral_band_centers(hz_min, hz_max, step_hz=step_hz, half_hz=half_hz)
    t = float(target_hz)
    return float(min(centers, key=lambda c: abs(c - t)))


def harvest_window_for_target(
    target_hz: float,
    hz_min: float = SWEEP_HZ_MIN,
    hz_max: float = SWEEP_HZ_MAX,
    *,
    step_hz: float = SPECTRAL_BAND_STEP_HZ,
    half_hz: float = SPECTRAL_BAND_HALF_HZ,
) -> Tuple[float, float, float]:
    """
    Return ``(lo_hz, hi_hz, eps_broad_half_hz)`` for a scheduler target.

    ``eps_broad_half_hz`` is passed to the worker as ``eps_broad_search_hz`` (half-width).
    """
    center = nearest_band_center(
        target_hz, hz_min, hz_max, step_hz=step_hz, half_hz=half_hz
    )
    half = max(8.0, float(half_hz))
    lo = max(float(hz_min), float(center) - half)
    hi = min(float(hz_max), float(center) + half)
    broad = max(float(target_hz) - lo, hi - float(target_hz), half * 0.85)
    return float(lo), float(hi), float(broad)


def spectral_harvest_worker_overrides(
    target_hz: float,
    hz_min: float = SWEEP_HZ_MIN,
    hz_max: float = SWEEP_HZ_MAX,
    *,
    step_hz: float = SPECTRAL_BAND_STEP_HZ,
    half_hz: float = SPECTRAL_BAND_HALF_HZ,
) -> Dict[str, Any]:
    """Keys merged into master worker ``params`` and worker solver cfg."""
    lo, hi, broad = harvest_window_for_target(
        target_hz, hz_min, hz_max, step_hz=step_hz, half_hz=half_hz
    )
    center = nearest_band_center(target_hz, hz_min, hz_max, step_hz=step_hz, half_hz=half_hz)
    return {
        "harvest_lo_hz": lo,
        "harvest_hi_hz": hi,
        "eps_broad_search_hz": broad,
        "spectral_band_center_hz": center,
    }


def build_spectral_band_task_list(
    hz_min: float,
    hz_max: float,
    band_params_fn,
) -> List[Tuple[float, Dict[str, Any]]]:
    """Static task list: one worker per band center (production multi-band pass)."""
    tasks: List[Tuple[float, Dict[str, Any]]] = []
    for hz in spectral_band_centers(hz_min, hz_max):
        p = dict(band_params_fn(hz))
        p.update(spectral_harvest_worker_overrides(hz, hz_min, hz_max))
        p["label"] = f"Spectral band @ {hz:.1f} Hz"
        tasks.append((float(hz), p))
    return tasks
