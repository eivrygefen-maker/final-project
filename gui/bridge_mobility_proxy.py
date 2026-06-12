#!/usr/bin/env python3
"""
Geometry/material-derived bridge mobility proxy (diagnostic only).

Models how each guitar body responds to the same string force at the bridge.
Not a separate sound source — scales bridge coupling / modal amplitude.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Tuple

from modal_damping import normalize_participation_shares
from sample_parameters import normalize_sample_parameters

# Relative density kg/m³ reference (spruce ≈ 1.0)
WOOD_DENSITY_REL: Dict[str, float] = {
    "spruce": 1.00,
    "cedar": 0.78,
    "maple": 1.18,
    "mahogany": 1.05,
    "rosewood": 1.14,
}

DEFAULT_LENGTH = 0.45
DEFAULT_WIDTH = 0.35
DEFAULT_DEPTH = 0.10
DEFAULT_TOP_T = 0.003
DEFAULT_BACK_T = 0.0033
DEFAULT_HOLE_R = 0.045


def _g(params: Mapping[str, Any], key: str, default: float) -> float:
    for candidate in (f"geometry.{key}", key):
        v = params.get(candidate)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    geom = params.get("geometry")
    if isinstance(geom, Mapping) and key in geom:
        try:
            return float(geom[key])
        except (TypeError, ValueError):
            pass
    return default


def compute_body_mass_proxies(parameters: Mapping[str, Any] | None) -> Dict[str, float]:
    """Sample-level geometry/material mass proxies."""
    p = normalize_sample_parameters(parameters)
    length = _g(p, "length", DEFAULT_LENGTH)
    width = _g(p, "width", DEFAULT_WIDTH)
    depth = _g(p, "depth", DEFAULT_DEPTH)
    top_t = _g(p, "top_thickness", DEFAULT_TOP_T)
    back_t = _g(p, "back_thickness", DEFAULT_BACK_T)
    hole_r = _g(p, "hole_radius", DEFAULT_HOLE_R)

    top_wood = str(p.get("top_wood_id") or "spruce").lower()
    back_wood = str(p.get("back_wood_id") or "mahogany").lower()
    top_rho = WOOD_DENSITY_REL.get(top_wood, 1.0)
    back_rho = WOOD_DENSITY_REL.get(back_wood, 1.0)

    top_area = length * width * 0.92
    back_area = length * width * 0.88
    hole_area = math.pi * hole_r * hole_r

    top_effective_mass = top_area * top_t * top_rho
    back_effective_mass = back_area * back_t * back_rho
    air_volume = max(length * width * depth - hole_area * 0.35, 1e-6)
    mixed_body_mass = 0.55 * top_effective_mass + 0.45 * back_effective_mass

    ref_mass = DEFAULT_LENGTH * DEFAULT_WIDTH * DEFAULT_TOP_T * 1.0
    mobility = ref_mass / max(0.55 * top_effective_mass + 0.45 * back_effective_mass, 1e-9)
    mobility = max(0.72, min(1.28, mobility**0.35))

    return {
        "top_effective_mass_proxy": round(top_effective_mass, 8),
        "back_effective_mass_proxy": round(back_effective_mass, 8),
        "body_air_volume_proxy": round(air_volume, 8),
        "mixed_body_mass_proxy": round(mixed_body_mass, 8),
        "bridge_mobility_proxy": round(mobility, 6),
        "top_wood_density_rel": top_rho,
        "back_wood_density_rel": back_rho,
    }


def effective_modal_mass_proxy(
    mode: Mapping[str, Any],
    mass_proxies: Mapping[str, float],
) -> float:
    top_s, back_s, air_s, coupled_s = normalize_participation_shares(mode)
    air_factor = float(mass_proxies.get("body_air_volume_proxy") or 1.0) / (
        DEFAULT_LENGTH * DEFAULT_WIDTH * DEFAULT_DEPTH
    )
    air_factor = max(0.65, min(1.35, air_factor**0.25))
    return (
        top_s * float(mass_proxies.get("top_effective_mass_proxy") or 1.0)
        + back_s * float(mass_proxies.get("back_effective_mass_proxy") or 1.0)
        + air_s * air_factor * float(mass_proxies.get("body_air_volume_proxy") or 1.0) * 0.15
        + coupled_s * float(mass_proxies.get("mixed_body_mass_proxy") or 1.0)
    )


def bridge_body_coupling_factor(
    mode: Mapping[str, Any],
    parameters: Mapping[str, Any] | None,
    *,
    existing_bridge: float,
) -> Tuple[float, Dict[str, Any]]:
    """
    Scale bridge coupling by mobility / effective modal mass (conservative clamp).
    """
    mass = compute_body_mass_proxies(parameters)
    eff_mass = effective_modal_mass_proxy(mode, mass)
    ref = mass["top_effective_mass_proxy"] + mass["back_effective_mass_proxy"]
    mass_scale = ref / max(eff_mass, 1e-9)
    mass_scale = max(0.80, min(1.20, mass_scale**0.18))

    mobility = float(mass["bridge_mobility_proxy"])
    coupling = existing_bridge * mobility * mass_scale
    coupling = max(existing_bridge * 0.65, min(existing_bridge * 1.35, coupling))

    return coupling, {
        **mass,
        "effective_modal_mass_proxy": round(eff_mass, 8),
        "bridge_body_coupling_factor": round(coupling / max(existing_bridge, 1e-9), 6),
        "bridge_mobility_affects": "amplitude",
    }
