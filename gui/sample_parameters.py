#!/usr/bin/env python3
"""Normalize LHS / ROM sample parameter dicts for synthesis."""
from __future__ import annotations

from typing import Any, Dict, Mapping


def normalize_sample_parameters(parameters: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Ensure top_wood_id, back_wood_id, and geometry.* keys are present."""
    raw = dict(parameters or {})
    out: Dict[str, Any] = dict(raw)
    geom = raw.get("geometry")
    if isinstance(geom, Mapping):
        for key, val in geom.items():
            gk = f"geometry.{key}"
            if gk not in out:
                out[gk] = val
    for key in (
        "length",
        "width",
        "depth",
        "top_thickness",
        "back_thickness",
        "hole_radius",
    ):
        gk = f"geometry.{key}"
        if gk in raw and key not in out:
            out[key] = raw[gk]
    if "top_wood_id" not in out and "top_wood" in out:
        out["top_wood_id"] = out["top_wood"]
    if "back_wood_id" not in out and "back_wood" in out:
        out["back_wood_id"] = out["back_wood"]
    return out


def sample_has_real_woods(parameters: Mapping[str, Any] | None) -> bool:
    p = normalize_sample_parameters(parameters)
    return bool(p.get("top_wood_id")) and bool(p.get("back_wood_id"))
