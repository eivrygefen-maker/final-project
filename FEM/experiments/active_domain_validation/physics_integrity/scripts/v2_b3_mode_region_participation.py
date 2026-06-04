#!/usr/bin/env python3
"""Lightweight per-mode dominant-region metadata (solve-time, no rich modal export)."""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np

from v2_b3_rich_modal_lib import (  # noqa: E402
    REGION_PARTICIPATION_STATUS_AVAILABLE,
    UNAVAILABLE_REGION_INDICES_STATUS,
    participation_energy_fraction,
    prolongate_active_to_W,
)


def participation_fields_not_available(*, reason: str, method: str = "not_available") -> Dict[str, Any]:
    return {
        "dominant_region": "unknown",
        "top_participation": None,
        "back_participation": None,
        "air_participation": None,
        "participation_method": method,
        "participation_status": "not_available",
        "participation_detail": reason,
    }


def _pick_dominant_region(
    *,
    top: Optional[float],
    back: Optional[float],
    air: Optional[float],
    min_fraction: float = 1.0e-8,
) -> str:
    candidates: Dict[str, float] = {}
    if top is not None and top >= min_fraction:
        candidates["top"] = float(top)
    if back is not None and back >= min_fraction:
        candidates["back"] = float(back)
    if air is not None and air >= min_fraction:
        candidates["air"] = float(air)
    if not candidates:
        return "unknown"
    return max(candidates, key=candidates.get)


def compute_mode_dominant_region_metadata(
    *,
    x_active: np.ndarray,
    built: Mapping[str, Any],
    region_ctx: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Energy fractions ‖x[region]‖² / ‖x‖² on full W layout.
    Back includes ribs facet DOFs when structural indices are available.
    """
    region = region_ctx["region"]
    structural_ok = bool(region_ctx.get("structural_indices_available"))
    pressure_ok = bool(region_ctx.get("pressure_indices_available"))

    if not structural_ok and not pressure_ok:
        return participation_fields_not_available(
            reason=UNAVAILABLE_REGION_INDICES_STATUS,
            method="not_available",
        )

    x_full = prolongate_active_to_W(np.asarray(x_active, dtype=np.float64), dict(built))

    top_p: Optional[float] = None
    back_p: Optional[float] = None
    air_p: Optional[float] = None
    method = "not_available"

    if structural_ok:
        top_p = participation_energy_fraction(x_full, region.get("u_idx_top", []))
        back_raw = participation_energy_fraction(x_full, region.get("u_idx_back", []))
        ribs_p = participation_energy_fraction(x_full, region.get("u_idx_ribs", []))
        if back_raw is not None or ribs_p is not None:
            back_p = float((back_raw or 0.0) + (ribs_p or 0.0))
        method = "structural_pressure_energy_fraction_v1"
    if pressure_ok:
        air_p = participation_energy_fraction(x_full, region.get("p_idx_air", []))
        if method == "not_available":
            method = "pressure_energy_fraction_v1"

    if structural_ok and pressure_ok:
        method = "structural_pressure_energy_fraction_v1"
    elif structural_ok:
        method = "structural_energy_fraction_v1"
    elif pressure_ok:
        method = "pressure_energy_fraction_v1"

    dominant = _pick_dominant_region(top=top_p, back=back_p, air=air_p)
    return {
        "dominant_region": dominant,
        "top_participation": top_p,
        "back_participation": back_p,
        "air_participation": air_p,
        "participation_method": method,
        "participation_status": "computed",
        "participation_detail": str(region_ctx.get("region_dof_source") or ""),
    }


def compute_mode_dominant_region_from_norms(
    *,
    u_norm_W: float,
    p_norm_W: float,
    x_norm_W: float,
) -> Dict[str, Any]:
    """Fallback when only scalar norms were persisted (no vector recompute)."""
    xn = max(float(x_norm_W), 1.0e-30)
    u_frac = float(u_norm_W) ** 2 / (xn**2)
    p_frac = float(p_norm_W) ** 2 / (xn**2)
    air_p = p_frac if p_frac > 0.0 else None
    if p_frac > max(u_frac, 1.0e-8) and p_frac >= 0.25:
        dominant = "air"
    else:
        dominant = "unknown"
    return {
        "dominant_region": dominant,
        "top_participation": None,
        "back_participation": None,
        "air_participation": air_p,
        "participation_method": "pressure_displacement_norm_proxy_v1",
        "participation_status": "fallback",
        "participation_detail": "top/back require region_dof_indices.npz at checkpoint export",
    }


def attach_participation_to_accepted_mode(
    entry: Dict[str, Any],
    *,
    x_active: Optional[np.ndarray],
    built: Mapping[str, Any],
    region_ctx: Optional[Mapping[str, Any]],
) -> None:
    """Mutate accepted-mode dict with dominant-region fields."""
    if x_active is not None and region_ctx is not None:
        entry.update(
            compute_mode_dominant_region_metadata(
                x_active=x_active,
                built=built,
                region_ctx=region_ctx,
            )
        )
        return
    if entry.get("u_norm_W") is not None and entry.get("x_norm_W") is not None:
        entry.update(
            compute_mode_dominant_region_from_norms(
                u_norm_W=float(entry["u_norm_W"]),
                p_norm_W=float(entry.get("p_norm_W") or 0.0),
                x_norm_W=float(entry["x_norm_W"]),
            )
        )
        return
    entry.update(participation_fields_not_available(reason="no_vector_or_norms"))


def merge_participation_into_catalog_record(record: Dict[str, Any], mode: Mapping[str, Any]) -> None:
    """Copy participation fields from solver accepted_modes into aggregation record."""
    for key in (
        "dominant_region",
        "top_participation",
        "back_participation",
        "air_participation",
        "participation_method",
        "participation_status",
        "participation_detail",
    ):
        if key in mode:
            record[key] = mode[key]
