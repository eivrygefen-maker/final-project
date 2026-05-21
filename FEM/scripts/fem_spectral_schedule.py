#!/usr/bin/env python3
"""
Overlapping spectral bands for multi-shift FSI harvest (60–550 Hz).

Each band defines a worker ``target_hz`` (scheduler shift center), an ST σ placed
**outside** the harvest window ``[harvest_lo_hz, harvest_hi_hz]``, and overlapping
windows wide enough to avoid gaps on ``[hz_min, hz_max]``.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

# Default production sweep (override via JSON ``spectral_bands`` or master ``--hz-min/max``).
SWEEP_HZ_MIN = 60.0
SWEEP_HZ_MAX = 550.0

# Centers every 48 Hz with ±46 Hz harvest → 44 Hz overlap (2*46 - 48 = 44 >= step/2).
SPECTRAL_BAND_STEP_HZ = 48.0
SPECTRAL_BAND_HALF_HZ = 46.0

# High-frequency stability (MUMPS INFOG=-10 near ~450–550 Hz).
HF_STABILITY_THRESHOLD_HZ = 400.0
ULTRA_HF_STABILITY_THRESHOLD_HZ = 480.0

# SLEPc quota for dense HF spectrum (still capped by VM LU memory in master).
HF_WORKER_NUM_MODES = 40
HF_EPS_NCV_MAX = 56

# ST σ must sit outside the per-shift harvest window so σ-Ritz is not the only
# strongly coupled mode inside [harvest_lo_hz, harvest_hi_hz].
SIGMA_OUTSIDE_HARVEST_MARGIN_HZ = 12.0


def hz_shift_quantize(hz: float, tol: float = 1e-4) -> float:
    return round(float(hz) / tol) * tol


def resolve_spectral_band_params(
    solver_cfg: Optional[Mapping[str, Any]] = None,
    *,
    hz_min: Optional[float] = None,
    hz_max: Optional[float] = None,
) -> Tuple[float, float, float, float]:
    """
    Merge JSON ``spectral_bands`` block with defaults.

    Returns ``(hz_min, hz_max, step_hz, half_hz)``.
    """
    lo = float(hz_min if hz_min is not None else SWEEP_HZ_MIN)
    hi = float(hz_max if hz_max is not None else SWEEP_HZ_MAX)
    step = float(SPECTRAL_BAND_STEP_HZ)
    half = float(SPECTRAL_BAND_HALF_HZ)
    if solver_cfg:
        block = solver_cfg.get("spectral_bands") or {}
        if isinstance(block, dict):
            lo = float(block.get("hz_min", lo))
            hi = float(block.get("hz_max", hi))
            step = float(block.get("step_hz", step))
            half = float(block.get("half_hz", half))
    return lo, hi, step, half


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


def verify_spectral_coverage(
    hz_min: float = SWEEP_HZ_MIN,
    hz_max: float = SWEEP_HZ_MAX,
    *,
    step_hz: float = SPECTRAL_BAND_STEP_HZ,
    half_hz: float = SPECTRAL_BAND_HALF_HZ,
    probe_step_hz: float = 1.0,
) -> Tuple[bool, float, List[Tuple[float, float]]]:
    """
    Check union of harvest windows covers ``[hz_min, hz_max]``.

    Returns ``(ok, max_gap_hz, gap_intervals)``.
    """
    centers = spectral_band_centers(hz_min, hz_max, step_hz=step_hz, half_hz=half_hz)
    if not centers:
        return False, float(hz_max - hz_min), [(float(hz_min), float(hz_max))]
    intervals = [
        (max(float(hz_min), float(c) - float(half_hz)), min(float(hz_max), float(c) + float(half_hz)))
        for c in centers
    ]
    intervals.sort(key=lambda x: x[0])
    merged: List[Tuple[float, float]] = [intervals[0]]
    for lo, hi in intervals[1:]:
        pl, ph = merged[-1]
        if lo <= ph + 1e-9:
            merged[-1] = (pl, max(ph, hi))
        else:
            merged.append((lo, hi))
    gaps: List[Tuple[float, float]] = []
    if merged[0][0] > float(hz_min) + 1e-9:
        gaps.append((float(hz_min), merged[0][0]))
    for i in range(len(merged) - 1):
        if merged[i + 1][0] > merged[i][1] + 1e-9:
            gaps.append((merged[i][1], merged[i + 1][0]))
    if merged[-1][1] < float(hz_max) - 1e-9:
        gaps.append((merged[-1][1], float(hz_max)))
    max_gap = 0.0
    f = float(hz_min)
    while f <= float(hz_max) + 1e-9:
        covered = any(lo - 1e-9 <= f <= hi + 1e-9 for lo, hi in intervals)
        if not covered:
            g0 = f
            while f <= float(hz_max) + 1e-9 and not any(
                lo - 1e-9 <= f <= hi + 1e-9 for lo, hi in intervals
            ):
                f += float(probe_step_hz)
            max_gap = max(max_gap, f - g0)
            gaps.append((g0, f))
        else:
            f += float(probe_step_hz)
    ok = len(gaps) == 0 and 2.0 * float(half_hz) >= float(step_hz) - 1e-9
    return ok, float(max_gap), gaps


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


def _sigma_hz_from_offset(target_hz: float, offset_hz: float) -> float:
    return max(1.0, float(target_hz) + float(offset_hz))


def sigma_hz_outside_harvest_window(
    sigma_hz: float,
    harvest_lo_hz: float,
    harvest_hi_hz: float,
    *,
    margin_hz: float = SIGMA_OUTSIDE_HARVEST_MARGIN_HZ,
) -> bool:
    """True when ``sigma_hz`` is not inside the harvest band (with margin)."""
    s = float(sigma_hz)
    lo = float(harvest_lo_hz) - float(margin_hz)
    hi = float(harvest_hi_hz) + float(margin_hz)
    return s < lo - 1.0e-9 or s > hi + 1.0e-9


def primary_sigma_offset_hz_outside_harvest(
    target_hz: float,
    harvest_lo_hz: float,
    harvest_hi_hz: float,
    *,
    hz_min: float = SWEEP_HZ_MIN,
    hz_max: float = SWEEP_HZ_MAX,
    margin_hz: float = SIGMA_OUTSIDE_HARVEST_MARGIN_HZ,
) -> float:
    """
    Offset (Hz) for ``eps_st_sigma_primary_offset_hz`` so ST σ lies outside ``[lo, hi]``.

    Prefers σ just below ``harvest_lo_hz``; if that violates ``hz_min``, uses just above
    ``harvest_hi_hz`` (still outside the window).
    """
    target = float(target_hz)
    lo = float(harvest_lo_hz)
    hi = float(harvest_hi_hz)
    margin = max(8.0, float(margin_hz))
    f0 = max(1.0, float(hz_min))
    f1 = max(f0, float(hz_max))

    sigma_below = lo - margin
    off_below = sigma_below - target
    if sigma_below >= f0 + 1.0e-9:
        return float(off_below)

    sigma_above = hi + margin
    off_above = sigma_above - target
    if sigma_above <= f1 - 1.0e-9:
        return float(off_above)

    # Last resort: push σ to the sweep edge farthest from the harvest window centre.
    mid = 0.5 * (lo + hi)
    if mid >= 0.5 * (f0 + f1):
        return float((f0 + 1.0) - target)
    return float((f1 - 1.0) - target)


def solver_stability_overrides_for_target(target_hz: float) -> Dict[str, Any]:
    """
    Per-shift ST/LU stabilizers for high-frequency bands (reduces MUMPS -10).

    Applied by the master via ``FEM_WORKER_SOLVER_OVERRIDES`` env on spawn.
    """
    hz = float(target_hz)
    out: Dict[str, Any] = {}
    if hz >= HF_STABILITY_THRESHOLD_HZ:
        out["eps_st_sigma_frac_offset"] = 0.15
        out["eps_st_sigma_min_offset_hz"] = 12.0
        out["eps_st_a_diagonal_shift_frac"] = 0.002
        out["eps_st_a_diagonal_shift_frac_ladder"] = [0.002, 0.001, 0.005, 0.02]
        out["eps_st_mass_reg_frac"] = 0.05
        out["st_pc_factor_shift_amount"] = 0.10
    if hz >= ULTRA_HF_STABILITY_THRESHOLD_HZ:
        out["eps_st_sigma_frac_offset"] = 0.18
        out["eps_st_sigma_min_offset_hz"] = 15.0
        out["eps_st_a_diagonal_shift_frac"] = 0.003
        out["eps_st_a_diagonal_shift_frac_ladder"] = [0.003, 0.001, 0.008, 0.025]
        out["eps_st_mass_reg_frac"] = 0.08
        out["st_pc_factor_shift_amount"] = 0.12
    return out


def worker_slepc_quota_for_target(
    target_hz: float,
    base_num_modes: int,
    *,
    coupled_cap: int = 32,
) -> Dict[str, int]:
    """Raise ``nev`` / ``ncv`` for HF targets where modal density increases."""
    nm = int(base_num_modes)
    cap = int(coupled_cap)
    ncv_max = 48
    if float(target_hz) >= HF_STABILITY_THRESHOLD_HZ:
        hf_cap = min(HF_WORKER_NUM_MODES, 48)
        nm = max(nm, hf_cap)
        cap = max(cap, hf_cap)
        ncv_max = HF_EPS_NCV_MAX
    if float(target_hz) >= ULTRA_HF_STABILITY_THRESHOLD_HZ:
        hf_cap = min(HF_WORKER_NUM_MODES, 48)
        nm = max(nm, hf_cap)
        cap = max(cap, hf_cap)
        ncv_max = HF_EPS_NCV_MAX
    nm = min(nm, cap)
    return {
        "num_modes": nm,
        "eps_worker_num_modes_cap": cap,
        "eps_ncv_max": ncv_max,
    }


def spectral_harvest_worker_overrides(
    target_hz: float,
    hz_min: float = SWEEP_HZ_MIN,
    hz_max: float = SWEEP_HZ_MAX,
    *,
    step_hz: float = SPECTRAL_BAND_STEP_HZ,
    half_hz: float = SPECTRAL_BAND_HALF_HZ,
    solver_cfg: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Keys merged into master worker ``params`` and worker solver cfg."""
    if solver_cfg is not None:
        hz_min, hz_max, step_hz, half_hz = resolve_spectral_band_params(
            solver_cfg, hz_min=hz_min, hz_max=hz_max
        )
    lo, hi, broad = harvest_window_for_target(
        target_hz, hz_min, hz_max, step_hz=step_hz, half_hz=half_hz
    )
    center = nearest_band_center(target_hz, hz_min, hz_max, step_hz=step_hz, half_hz=half_hz)
    out: Dict[str, Any] = {
        "harvest_lo_hz": lo,
        "harvest_hi_hz": hi,
        "eps_broad_search_hz": broad,
        "spectral_band_center_hz": center,
        "spectral_hz_min": float(hz_min),
        "spectral_hz_max": float(hz_max),
    }
    so = solver_stability_overrides_for_target(float(target_hz))
    sigma_off = primary_sigma_offset_hz_outside_harvest(
        float(target_hz),
        lo,
        hi,
        hz_min=float(hz_min),
        hz_max=float(hz_max),
    )
    so["eps_st_sigma_primary_offset_hz"] = float(sigma_off)
    out["solver_overrides"] = so
    out["eps_st_sigma_primary_offset_hz"] = float(sigma_off)
    out["eps_st_sigma_hz_planned"] = float(_sigma_hz_from_offset(float(target_hz), sigma_off))
    return out


def sigma_retry_offset_candidates(
    target_hz: float,
    harvest_lo_hz: float,
    harvest_hi_hz: float,
    *,
    hz_min: float = SWEEP_HZ_MIN,
    hz_max: float = SWEEP_HZ_MAX,
    solver_cfg: Optional[Mapping[str, Any]] = None,
) -> List[float]:
    """
    Ordered ST σ offsets (Hz) for master retry after zero-yield merge.

    Every candidate places σ outside ``[harvest_lo_hz, harvest_hi_hz]`` (with margin).
    """
    target = float(target_hz)
    lo = float(harvest_lo_hz)
    hi = float(harvest_hi_hz)
    margin = float(SIGMA_OUTSIDE_HARVEST_MARGIN_HZ)
    f0 = float(hz_min)
    f1 = float(hz_max)

    primary = primary_sigma_offset_hz_outside_harvest(
        target, lo, hi, hz_min=f0, hz_max=f1, margin_hz=margin
    )
    extra_steps: List[float] = [18.0, 28.0, 38.0]
    if solver_cfg:
        raw_cfg = solver_cfg.get("eps_st_sigma_retry_offsets_hz")
        if isinstance(raw_cfg, (list, tuple)):
            for x in raw_cfg:
                try:
                    v = float(x)
                    if abs(v) not in {abs(e) for e in extra_steps}:
                        extra_steps.append(v)
                except (TypeError, ValueError):
                    pass

    candidates: List[float] = []
    seen_off: Set[float] = set()
    seen_sigma: set = set()

    def _try_offset(off: float) -> None:
        key = round(float(off), 2)
        if key in seen_off:
            return
        s_hz = _sigma_hz_from_offset(target, off)
        if not sigma_hz_outside_harvest_window(s_hz, lo, hi, margin_hz=margin):
            return
        sk = round(s_hz, 2)
        if sk in seen_sigma:
            return
        seen_sigma.add(sk)
        seen_off.add(key)
        candidates.append(float(off))

    _try_offset(primary)
    for step in extra_steps:
        if abs(float(step)) < 1.0e-9:
            continue
        sigma_below = lo - margin - abs(float(step))
        if sigma_below >= f0 + 1.0e-9:
            _try_offset(sigma_below - target)
        sigma_above = hi + margin + abs(float(step))
        if sigma_above <= f1 - 1.0e-9:
            _try_offset(sigma_above - target)
    return candidates


def sigma_retry_offset_ladder(
    solver_cfg: Optional[Mapping[str, Any]] = None,
) -> List[float]:
    """Legacy flat retry offsets (used only when harvest bounds are unavailable)."""
    default = (18.0, 28.0, -18.0, -28.0, 35.0, -35.0, 42.0, -42.0)
    if not solver_cfg:
        return list(default)
    raw = solver_cfg.get("eps_st_sigma_retry_offsets_hz", default)
    if not isinstance(raw, (list, tuple)):
        return list(default)
    out: List[float] = []
    for x in raw:
        try:
            v = float(x)
            if v not in out:
                out.append(v)
        except (TypeError, ValueError):
            continue
    return out if out else list(default)


def build_spectral_band_task_list(
    hz_min: float,
    hz_max: float,
    band_params_fn,
    *,
    solver_cfg: Optional[Mapping[str, Any]] = None,
) -> List[Tuple[float, Dict[str, Any]]]:
    """Static task list: one worker per band center (production multi-band pass)."""
    step_hz = SPECTRAL_BAND_STEP_HZ
    half_hz = SPECTRAL_BAND_HALF_HZ
    if solver_cfg is not None:
        hz_min, hz_max, step_hz, half_hz = resolve_spectral_band_params(
            solver_cfg, hz_min=hz_min, hz_max=hz_max
        )
    tasks: List[Tuple[float, Dict[str, Any]]] = []
    for hz in spectral_band_centers(hz_min, hz_max, step_hz=step_hz, half_hz=half_hz):
        p = dict(band_params_fn(hz))
        p.update(
            spectral_harvest_worker_overrides(
                hz, hz_min, hz_max, step_hz=step_hz, half_hz=half_hz, solver_cfg=solver_cfg
            )
        )
        quota = worker_slepc_quota_for_target(hz, int(p.get("num_modes", 32)))
        p["num_modes"] = quota["num_modes"]
        p["eps_worker_num_modes_cap"] = quota["eps_worker_num_modes_cap"]
        p["eps_ncv_max"] = quota["eps_ncv_max"]
        p["label"] = f"Spectral band @ {hz:.1f} Hz"
        tasks.append((float(hz), p))
    return tasks


def format_coverage_report(
    hz_min: float = SWEEP_HZ_MIN,
    hz_max: float = SWEEP_HZ_MAX,
    *,
    solver_cfg: Optional[Mapping[str, Any]] = None,
) -> str:
    """Human-readable band table for master startup logs."""
    lo, hi, step, half = resolve_spectral_band_params(solver_cfg, hz_min=hz_min, hz_max=hz_max)
    centers = spectral_band_centers(lo, hi, step_hz=step, half_hz=half)
    ok, max_gap, gaps = verify_spectral_coverage(lo, hi, step_hz=step, half_hz=half)
    lines = [
        f"Spectral bands [{lo:.0f}, {hi:.0f}] Hz: {len(centers)} centers, "
        f"step={step:.0f} half={half:.0f} overlap={2 * half - step:.0f} Hz, coverage_ok={ok}",
    ]
    for c in centers:
        wlo, whi, _ = harvest_window_for_target(c, lo, hi, step_hz=step, half_hz=half)
        lines.append(f"  center {c:.1f} -> harvest [{wlo:.1f}, {whi:.1f}] Hz")
    if not ok:
        lines.append(f"  WARNING: coverage gaps (max_gap={max_gap:.1f} Hz): {gaps}")
    return "\n".join(lines)
