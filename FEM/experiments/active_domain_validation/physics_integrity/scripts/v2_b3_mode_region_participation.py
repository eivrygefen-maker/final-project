#!/usr/bin/env python3
"""Lightweight per-mode dominant-region metadata (solve-time, no rich modal export)."""
from __future__ import annotations

import math
import statistics
from typing import Any, Dict, List, Mapping, Optional, Sequence

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


PARTICIPATION_SCORES_SEMANTICS = (
    "non_partitioning_energy_fractions_v1: "
    "top/back/air scores are ‖x[region]‖²/‖x‖² and may overlap (not a partition of unity). "
    "Use top_share/back_share/air_share for STK damping weights."
)

STK_DAMPING_GUIDANCE = (
    "Do not use hard dominant_region alone for damping. "
    "Weight Q/damping by top_share, back_share, air_share (normalized over available scores)."
)


def _participation_score_value(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(score) or score < 0.0:
        return None
    return score


def compute_participation_shares(
    *,
    top_score: Optional[float],
    back_score: Optional[float],
    air_score: Optional[float],
) -> Dict[str, Optional[float]]:
    """Normalize participation scores to shares that sum to 1 (STK damping weights)."""
    raw = {
        "top": _participation_score_value(top_score),
        "back": _participation_score_value(back_score),
        "air": _participation_score_value(air_score),
    }
    present = {k: v for k, v in raw.items() if v is not None}
    total = float(sum(present.values()))
    if total <= 0.0:
        return {"top_share": None, "back_share": None, "air_share": None, "share_denominator": 0.0}
    return {
        "top_share": (present["top"] / total) if "top" in present else None,
        "back_share": (present["back"] / total) if "back" in present else None,
        "air_share": (present["air"] / total) if "air" in present else None,
        "share_denominator": total,
    }


def dominant_region_from_shares(
    *,
    top_share: Optional[float],
    back_share: Optional[float],
    air_share: Optional[float],
    min_share: float = 1.0e-12,
) -> str:
    candidates: Dict[str, float] = {}
    for name, share in (("top", top_share), ("back", back_share), ("air", air_share)):
        if share is not None and float(share) >= min_share:
            candidates[name] = float(share)
    if not candidates:
        return "unknown"
    return max(candidates, key=candidates.get)


def secondary_region_from_shares(
    *,
    top_share: Optional[float],
    back_share: Optional[float],
    air_share: Optional[float],
    dominant_region: str,
    min_share: float = 0.1,
) -> Optional[str]:
    shares = {
        "top": top_share,
        "back": back_share,
        "air": air_share,
    }
    ordered = sorted(
        ((name, float(s)) for name, s in shares.items() if s is not None and float(s) >= min_share),
        key=lambda row: row[1],
        reverse=True,
    )
    for name, _ in ordered:
        if name != dominant_region:
            return name
    return None


def coupling_class_from_shares(
    *,
    dominant_region: str,
    top_share: Optional[float],
    back_share: Optional[float],
    air_share: Optional[float],
    mixed_threshold: float = 0.25,
    air_dominant_threshold: float = 0.5,
) -> str:
    top_s = float(top_share or 0.0)
    back_s = float(back_share or 0.0)
    air_s = float(air_share or 0.0)
    if top_s >= mixed_threshold and back_s >= mixed_threshold:
        return "top_back_mixed"
    if air_s >= air_dominant_threshold:
        return "air_dominant"
    if dominant_region in ("top", "back", "air"):
        return f"{dominant_region}_dominant"
    return "weak_or_unknown"


def enrich_participation_catalog_metadata(record: Dict[str, Any]) -> None:
    """
    Add normalized shares + coupling metadata from raw participation scores.
    Safe to call at aggregation time on existing solver rows (no re-solve).
    """
    top_score = _participation_score_value(record.get("top_participation"))
    back_score = _participation_score_value(record.get("back_participation"))
    air_score = _participation_score_value(record.get("air_participation"))

    if top_score is not None:
        record["top_participation_score"] = top_score
    if back_score is not None:
        record["back_participation_score"] = back_score
    if air_score is not None:
        record["air_participation_score"] = air_score

    record["participation_scores_semantics"] = PARTICIPATION_SCORES_SEMANTICS

    shares = compute_participation_shares(
        top_score=top_score,
        back_score=back_score,
        air_score=air_score,
    )
    record.update(shares)

    dominant = dominant_region_from_shares(
        top_share=shares.get("top_share"),
        back_share=shares.get("back_share"),
        air_share=shares.get("air_share"),
    )
    record["dominant_region"] = dominant
    record["secondary_region"] = secondary_region_from_shares(
        top_share=shares.get("top_share"),
        back_share=shares.get("back_share"),
        air_share=shares.get("air_share"),
        dominant_region=dominant,
    )
    record["coupling_class"] = coupling_class_from_shares(
        dominant_region=dominant,
        top_share=shares.get("top_share"),
        back_share=shares.get("back_share"),
        air_share=shares.get("air_share"),
    )


def summarize_participation_shares(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Aggregate share statistics for modes_summary.json."""
    out: Dict[str, Any] = {}
    for key in ("top_share", "back_share", "air_share"):
        vals = [
            float(r[key])
            for r in records
            if r.get(key) is not None and math.isfinite(float(r[key]))
        ]
        if not vals:
            out[key] = {"count": 0}
            continue
        out[key] = {
            "count": len(vals),
            "median": float(statistics.median(vals)),
            "mean": float(statistics.mean(vals)),
            "max": float(max(vals)),
            "min": float(min(vals)),
        }
    out["top_share_ge_0.25_count"] = sum(
        1 for r in records if (r.get("top_share") is not None and float(r["top_share"]) >= 0.25)
    )
    return out


def _pick_dominant_region(
    *,
    top: Optional[float],
    back: Optional[float],
    air: Optional[float],
    min_fraction: float = 1.0e-8,
) -> str:
    """Legacy argmax on raw scores (solve-time); aggregation recomputes from shares."""
    shares = compute_participation_shares(top_score=top, back_score=back, air_score=air)
    return dominant_region_from_shares(
        top_share=shares.get("top_share"),
        back_share=shares.get("back_share"),
        air_share=shares.get("air_share"),
        min_share=min_fraction,
    )


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
    if record.get("participation_status") in ("computed", "fallback"):
        enrich_participation_catalog_metadata(record)
