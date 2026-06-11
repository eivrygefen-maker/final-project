#!/usr/bin/env python3
"""
Per-mode damping / Q / bandwidth for body-response synthesis (no FEM).

Each mode gets individual decay/resonance width from frequency, material,
geometry, participation shares, and radiation proxies.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional

Q_MIN = 22.0
Q_MAX = 75.0


def _safe_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        out = float(val)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _geom(parameters: Mapping[str, Any], key: str, default: float) -> float:
    geom = parameters.get("geometry") or {}
    if isinstance(geom, Mapping) and key in geom:
        v = _safe_float(geom.get(key))
        if v is not None:
            return v
    v = _safe_float(parameters.get(key))
    return v if v is not None else default


def infer_mode_category(mode: Mapping[str, Any]) -> str:
    top = _safe_float(mode.get("top_share")) or 0.0
    back = _safe_float(mode.get("back_share")) or 0.0
    air = _safe_float(mode.get("air_share")) or 0.0
    bridge = _safe_float(mode.get("bridge_excitation_abs")) or 0.0
    if air >= max(top, back) and air >= 0.28:
        return "air"
    if top >= back and top >= 0.30:
        return "top"
    if back > top and back >= 0.30:
        return "back"
    if bridge >= 0.02 and top + back >= 0.35:
        return "coupled"
    return "coupled"


def _base_mode_q(mode: Mapping[str, Any], f_hz: float) -> float:
    for key in ("Q", "q", "modal_q", "quality_factor"):
        q = _safe_float(mode.get(key))
        if q is not None and q > 0:
            return float(q)
    air = _safe_float(mode.get("air_share")) or 0.22
    wood = (_safe_float(mode.get("top_share")) or 0.33) + (_safe_float(mode.get("back_share")) or 0.33)
    f_norm = min(max((f_hz - 60.0) / 490.0, 0.0), 1.0)
    return (42.0 + 28.0 * air + 10.0 * wood) * (1.0 - 0.32 * f_norm)


def _material_geometry_q_scale(
    parameters: Mapping[str, Any],
    f_hz: float,
    category: str,
    *,
    strength: float,
) -> float:
    """>1 lowers Q (more damping)."""
    s = max(0.0, min(1.0, float(strength)))
    if s <= 0:
        return 1.0
    top = str(parameters.get("top_wood_id") or "").lower()
    back = str(parameters.get("back_wood_id") or "").lower()
    top_t = _geom(parameters, "top_thickness", 0.003)
    back_t = _geom(parameters, "back_thickness", 0.0033)
    depth = _geom(parameters, "depth", 0.1)
    width = _geom(parameters, "width", 0.37)
    length = _geom(parameters, "length", 0.48)

    scale = 1.0
    if top == "cedar":
        scale *= 0.92 if category in ("top", "coupled") else 0.96
    elif top == "maple":
        scale *= 1.08 if category in ("top", "coupled") else 1.04
    if back == "rosewood":
        scale *= 1.05 if category in ("back", "coupled") else 1.02
    elif back == "mahogany":
        scale *= 0.97
    scale *= 1.0 + (top_t - 0.003) * 42.0
    scale *= 1.0 + (back_t - 0.0033) * 32.0
    scale *= 1.0 + (depth - 0.1) * 2.0
    scale *= 1.0 + (width - 0.37) * 1.1
    scale *= 1.0 + (length - 0.48) * 0.8
    if f_hz > 280.0:
        scale *= 1.0 + 0.05 * s * min(1.0, (f_hz - 280.0) / 220.0)
    if category == "air":
        scale *= 1.0 + 0.06 * s
    return 1.0 + (scale - 1.0) * s


def _radiation_damping_scale(mode: Mapping[str, Any]) -> float:
    rad = _safe_float(mode.get("radiation_proxy")) or 0.0
    air = _safe_float(mode.get("air_share")) or 0.0
    mic = _safe_float(mode.get("mic_output_proxy")) or 0.0
    scale = 1.0
    if rad > 0:
        scale += 0.12 * rad
    if air > 0.2:
        scale += 0.08 * air
    if mic > 0:
        scale += 0.04 * mic
    return scale


def compute_per_mode_damping(
    mode: Mapping[str, Any],
    f_hz: float,
    parameters: Mapping[str, Any],
    *,
    strength: float = 1.0,
    rad_k: float = 0.08,
) -> Dict[str, Any]:
    """
    Per-mode Q, damping ratio, decay tau, and bandwidth.

    Returns dict with mode_q, mode_damping, mode_tau_s, mode_bandwidth_hz, mode_category.
    """
    fm = max(float(f_hz), 1.0)
    category = infer_mode_category(mode)
    q_base = _base_mode_q(mode, fm)
    mat_scale = _material_geometry_q_scale(parameters, fm, category, strength=strength)
    rad_scale = _radiation_damping_scale(mode)
    inv_q = (1.0 / max(q_base, 0.5)) * mat_scale * rad_scale
    inv_q += rad_k * (fm / 1000.0)
    rad_proxy = _safe_float(mode.get("radiation_proxy")) or 0.0
    if rad_proxy > 0:
        inv_q += 0.035 * rad_proxy * math.sqrt(fm / 200.0)
    q_total = max(0.5, 1.0 / max(inv_q, 1e-9))
    q_total = float(max(Q_MIN, min(Q_MAX, q_total)))
    mode_damping = 1.0 / (2.0 * q_total)
    mode_tau_s = math.pi * q_total / fm
    mode_bandwidth_hz = fm / (2.0 * q_total)
    return {
        "mode_index": int(mode.get("mode_index", -1)),
        "frequency_hz": round(fm, 4),
        "mode_category": category,
        "mode_q": round(q_total, 4),
        "mode_damping": round(mode_damping, 6),
        "mode_tau_s": round(mode_tau_s, 6),
        "mode_bandwidth_hz": round(mode_bandwidth_hz, 4),
        "q_material_scale": round(mat_scale, 6),
        "q_radiation_scale": round(rad_scale, 6),
    }


def summarize_mode_damping_records(records: List[Mapping[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {}
    qs = [float(r["mode_q"]) for r in records]
    bws = [float(r["mode_bandwidth_hz"]) for r in records]
    taus = [float(r["mode_tau_s"]) for r in records]
    cats: Dict[str, int] = {}
    for r in records:
        c = str(r.get("mode_category") or "coupled")
        cats[c] = cats.get(c, 0) + 1
    return {
        "mode_count": len(records),
        "mode_q_min": round(min(qs), 4),
        "mode_q_max": round(max(qs), 4),
        "mode_q_median": round(sorted(qs)[len(qs) // 2], 4),
        "mode_q_spread": round(max(qs) - min(qs), 4),
        "mode_bandwidth_hz_min": round(min(bws), 4),
        "mode_bandwidth_hz_max": round(max(bws), 4),
        "mode_bandwidth_hz_median": round(sorted(bws)[len(bws) // 2], 4),
        "mode_tau_s_median": round(sorted(taus)[len(taus) // 2], 6),
        "mode_category_counts": cats,
    }
