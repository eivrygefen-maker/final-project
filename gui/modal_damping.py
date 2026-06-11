#!/usr/bin/env python3
"""
Per-mode damping / Q / bandwidth for body-response synthesis (no FEM).

Material damping is weighted by per-mode top/back/air/coupled participation shares.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Tuple

Q_MIN = 22.0
Q_MAX = 75.0

# Relative damping coefficients (>1 = more damping / lower Q). Spruce reference = 1.0.
WOOD_DAMPING_COEFF: Dict[str, float] = {
    "spruce": 1.00,
    "cedar": 0.78,
    "maple": 1.28,
    "mahogany": 1.08,
    "rosewood": 1.22,
}
AIR_DAMPING_COEFF = 1.22
COUPLED_DAMPING_COEFF = 1.08
DEFAULT_TOP_WOOD = "spruce"
DEFAULT_BACK_WOOD = "mahogany"


def list_wood_damping_constants() -> Dict[str, float]:
    return dict(WOOD_DAMPING_COEFF)


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
    for candidate in (f"geometry.{key}", key):
        v = _safe_float(parameters.get(candidate))
        if v is not None:
            return v
    return default


def normalize_participation_shares(mode: Mapping[str, Any]) -> Tuple[float, float, float, float]:
    top = max(_safe_float(mode.get("top_share")) or 0.0, 0.0)
    back = max(_safe_float(mode.get("back_share")) or 0.0, 0.0)
    air = max(_safe_float(mode.get("air_share")) or 0.0, 0.0)
    total = top + back + air
    if total < 0.05:
        return 0.38, 0.34, 0.18, 0.10
    top /= total
    back /= total
    air /= total
    coupled = max(0.0, 1.0 - top - back - air)
    if coupled < 0.02:
        coupled = 0.10
        scale = 0.90 / max(top + back + air, 1e-9)
        top *= scale
        back *= scale
        air *= scale
    return top, back, air, coupled


def infer_mode_category(mode: Mapping[str, Any]) -> str:
    top, back, air, coupled = normalize_participation_shares(mode)
    if air >= max(top, back) and air >= 0.28:
        return "air"
    if top >= back and top >= 0.30:
        return "top"
    if back > top and back >= 0.30:
        return "back"
    if coupled >= 0.20:
        return "coupled"
    return "coupled"


def _base_mode_q(mode: Mapping[str, Any], f_hz: float) -> float:
    for key in ("Q", "q", "modal_q", "quality_factor"):
        q = _safe_float(mode.get(key))
        if q is not None and q > 0:
            return float(q)
    top, back, air, _ = normalize_participation_shares(mode)
    f_norm = min(max((f_hz - 60.0) / 490.0, 0.0), 1.0)
    return (42.0 + 28.0 * air + 10.0 * (top + back)) * (1.0 - 0.32 * f_norm)


def compute_material_damping_components(
    mode: Mapping[str, Any],
    parameters: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Weighted material damping from participation shares × wood/air/coupled coeffs.

    mode_material_damping =
        top_share * damping(top_wood) + back_share * damping(back_wood)
        + air_share * damping_air + coupled_share * damping_coupled
    """
    top_s, back_s, air_s, coupled_s = normalize_participation_shares(mode)
    top_wood = str(parameters.get("top_wood_id") or DEFAULT_TOP_WOOD).lower()
    back_wood = str(parameters.get("back_wood_id") or DEFAULT_BACK_WOOD).lower()
    top_coeff = WOOD_DAMPING_COEFF.get(top_wood, 1.0)
    back_coeff = WOOD_DAMPING_COEFF.get(back_wood, 1.0)

    top_component = top_s * top_coeff
    back_component = back_s * back_coeff
    air_component = air_s * AIR_DAMPING_COEFF
    coupled_component = coupled_s * COUPLED_DAMPING_COEFF
    material = top_component + back_component + air_component + coupled_component

    return {
        "top_share": round(top_s, 6),
        "back_share": round(back_s, 6),
        "air_share": round(air_s, 6),
        "coupled_share": round(coupled_s, 6),
        "top_wood_id": top_wood,
        "back_wood_id": back_wood,
        "top_wood_damping_component": round(top_component, 6),
        "back_wood_damping_component": round(back_component, 6),
        "air_damping_component": round(air_component, 6),
        "coupled_damping_component": round(coupled_component, 6),
        "mode_material_damping": round(material, 6),
    }


def _geometry_damping_scale(parameters: Mapping[str, Any], f_hz: float) -> float:
    top_t = _geom(parameters, "top_thickness", 0.003)
    back_t = _geom(parameters, "back_thickness", 0.0033)
    depth = _geom(parameters, "depth", 0.1)
    width = _geom(parameters, "width", 0.37)
    length = _geom(parameters, "length", 0.48)
    scale = 1.0
    scale *= 1.0 + (top_t - 0.003) * 38.0
    scale *= 1.0 + (back_t - 0.0033) * 30.0
    scale *= 1.0 + (depth - 0.1) * 1.8
    scale *= 1.0 + (width - 0.37) * 0.9
    scale *= 1.0 + (length - 0.48) * 0.7
    if f_hz > 280.0:
        scale *= 1.0 + 0.04 * min(1.0, (f_hz - 280.0) / 220.0)
    return scale


def _radiation_damping_scale(mode: Mapping[str, Any]) -> float:
    rad = _safe_float(mode.get("radiation_proxy")) or 0.0
    mic = _safe_float(mode.get("mic_output_proxy")) or 0.0
    scale = 1.0
    if rad > 0:
        scale += 0.14 * rad
    if mic > 0:
        scale += 0.05 * mic
    return scale


def compute_per_mode_damping(
    mode: Mapping[str, Any],
    f_hz: float,
    parameters: Mapping[str, Any],
    *,
    strength: float = 1.0,
    rad_k: float = 0.08,
) -> Dict[str, Any]:
    fm = max(float(f_hz), 1.0)
    category = infer_mode_category(mode)
    q_base = _base_mode_q(mode, fm)
    mat = compute_material_damping_components(mode, parameters)
    material_scale = mat["mode_material_damping"]
    geom_scale = _geometry_damping_scale(parameters, fm)
    rad_scale = _radiation_damping_scale(mode)

    s = max(0.0, min(1.0, float(strength)))
    combined_mat_geom = 1.0 + s * (material_scale * geom_scale - 1.0)

    inv_q = (1.0 / max(q_base, 0.5)) * combined_mat_geom * rad_scale
    inv_q += rad_k * (fm / 1000.0)
    rad_proxy = _safe_float(mode.get("radiation_proxy")) or 0.0
    if rad_proxy > 0:
        inv_q += 0.035 * rad_proxy * math.sqrt(fm / 200.0)

    q_total = float(max(Q_MIN, min(Q_MAX, 1.0 / max(inv_q, 1e-9))))
    mode_damping = 1.0 / (2.0 * q_total)
    mode_tau_s = math.pi * q_total / fm
    mode_bandwidth_hz = fm / (2.0 * q_total)

    out = {
        "mode_index": int(mode.get("mode_index", -1)),
        "frequency_hz": round(fm, 4),
        "mode_category": category,
        "final_mode_q": round(q_total, 4),
        "mode_q": round(q_total, 4),
        "mode_damping": round(mode_damping, 6),
        "final_mode_tau_s": round(mode_tau_s, 6),
        "mode_tau_s": round(mode_tau_s, 6),
        "final_mode_bandwidth_hz": round(mode_bandwidth_hz, 4),
        "mode_bandwidth_hz": round(mode_bandwidth_hz, 4),
        "geometry_damping_component": round(geom_scale, 6),
        "q_radiation_scale": round(rad_scale, 6),
        "material_damping_strength": round(s, 4),
    }
    out.update(mat)
    return out


def summarize_mode_damping_records(records: List[Mapping[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {}
    qs = [float(r["mode_q"]) for r in records]
    bws = [float(r["mode_bandwidth_hz"]) for r in records]
    taus = [float(r["mode_tau_s"]) for r in records]
    mats = [float(r.get("mode_material_damping") or 1.0) for r in records]
    top_sh = [float(r.get("top_share") or 0.0) for r in records]
    back_sh = [float(r.get("back_share") or 0.0) for r in records]
    air_sh = [float(r.get("air_share") or 0.0) for r in records]
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
        "material_damping_min": round(min(mats), 6),
        "material_damping_median": round(sorted(mats)[len(mats) // 2], 6),
        "material_damping_max": round(max(mats), 6),
        "material_damping_spread": round(max(mats) - min(mats), 6),
        "avg_top_share": round(sum(top_sh) / len(top_sh), 6),
        "avg_back_share": round(sum(back_sh) / len(back_sh), 6),
        "avg_air_share": round(sum(air_sh) / len(air_sh), 6),
        "mode_category_counts": cats,
    }
