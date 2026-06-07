#!/usr/bin/env python3
"""STK combined-gain targets derived from existing FOM audio coupling scalars (read-only)."""
from __future__ import annotations

import math
import statistics
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from v2_b3_m4_rom_scalar_fields import (  # noqa: E402
    ACCURACY_BAND_HZ_DEFAULT,
    INTENSITY_LOG_EPSILON,
    NORMALIZATION_PERCENTILE,
    _percentile,
    _safe_float,
    enrich_mode_intensity_derivatives,
)

STRENGTH_LABELS: Tuple[str, ...] = (
    "very_weak",
    "weak",
    "medium",
    "strong",
    "very_strong",
)

COMBINED_GAIN_FIELDS: Tuple[str, ...] = (
    "bridge_to_mic_gain_raw",
    "bridge_to_radiation_gain_raw",
    "signed_bridge_to_mic_gain_raw",
)

STK_TARGET_FIELDS: Tuple[str, ...] = (
    "mic_output_proxy",
    "radiation_proxy",
    "bridge_excitation_abs",
    *COMBINED_GAIN_FIELDS,
)


def amplitude_semantics_audit() -> Dict[str, Any]:
    """Document FOM field semantics from v2_b3_mode_audio_coupling.py (read-only audit)."""
    return {
        "source_module": "v2_b3_mode_audio_coupling.py",
        "modal_norm": {
            "definition": "L2 norm of full W-layout eigenvector x",
            "signed": False,
            "normalized": False,
            "eigenvector_scale": "linear with x; doubles when x doubles",
            "stored_in_catalog": True,
        },
        "bridge_excitation_coupling": {
            "definition": "mean_signed(bridge_or_top_dof_displacement) / modal_norm",
            "signed": True,
            "normalized": "divided_once_by_modal_norm",
            "eigenvector_invariant": True,
            "stk_role": "excitation coupling (signed)",
        },
        "bridge_excitation_abs": {
            "definition": "RMS(bridge_or_top_dof_displacement) / modal_norm",
            "signed": False,
            "normalized": "divided_once_by_modal_norm",
            "eigenvector_invariant": True,
            "stk_role": "excitation amplitude (unsigned)",
        },
        "mic_output_proxy": {
            "definition": "soundhole_RMS/modal_norm, else cavity_pressure/modal_norm, else radiation_proxy",
            "signed": False,
            "normalized": "divided_once_by_modal_norm",
            "eigenvector_invariant": True,
            "stk_role": "audible output proxy (not physical mic pressure)",
        },
        "radiation_proxy": {
            "definition": "weighted_blend(top_output_proxy, back_output_proxy, air_pressure_proxy); each already /modal_norm",
            "signed": False,
            "normalized": "divided_once_by_modal_norm",
            "eigenvector_invariant": True,
            "stk_role": "surface/air radiation efficiency proxy",
        },
        "combined_gain_formulas": {
            "bridge_to_mic_gain_raw": "bridge_excitation_abs * mic_output_proxy",
            "bridge_to_radiation_gain_raw": "bridge_excitation_abs * radiation_proxy",
            "signed_bridge_to_mic_gain_raw": "bridge_excitation_coupling * mic_output_proxy",
        },
        "normalization_analysis": {
            "each_factor_already_divided_by_modal_norm": True,
            "product_scales_as": "RMS_a * RMS_b / modal_norm^2",
            "double_normalization_same_quantity": False,
            "eigenvector_scale_invariant": True,
            "interpretation": (
                "Product is a dimensionless bilinear coupling proxy, invariant to x->alpha*x. "
                "It is NOT a re-division of the same raw DOF by modal_norm twice. "
                "It combines two distinct spatial projections of the same mode."
            ),
            "mass_normalized_alternative": {
                "formula": "bridge_coupling * output_coupling / modal_mass",
                "modal_mass_in_catalog": False,
                "recomputable_without_fom_rerun": False,
                "note": "modal_norm is L2 not mass; mass norm not in current artifacts",
            },
        },
        "promotion_gate": (
            "Combined gain is mathematically consistent as derived proxy but is not a measured "
            "transfer function. Promote only if LOO metrics beat mic-only."
        ),
    }


def _product(
    a: Optional[float],
    b: Optional[float],
) -> Optional[float]:
    if a is None or b is None:
        return None
    out = float(a) * float(b)
    return out if math.isfinite(out) and out >= 0.0 else (out if math.isfinite(out) else None)


def enrich_mode_combined_gains(mode: Dict[str, Any]) -> Dict[str, Any]:
    b_abs = _safe_float(mode.get("bridge_excitation_abs"))
    b_sig = _safe_float(mode.get("bridge_excitation_coupling"))
    mic = _safe_float(mode.get("mic_output_proxy"))
    rad = _safe_float(mode.get("radiation_proxy"))
    mode["bridge_to_mic_gain_raw"] = _product(b_abs, mic)
    mode["bridge_to_radiation_gain_raw"] = _product(b_abs, rad)
    if b_sig is not None and mic is not None:
        mode["signed_bridge_to_mic_gain_raw"] = round(float(b_sig) * float(mic), 12)
    else:
        mode["signed_bridge_to_mic_gain_raw"] = None
    return mode


def _strength_label_from_quantile(q: float) -> str:
    if q < 0.2:
        return "very_weak"
    if q < 0.4:
        return "weak"
    if q < 0.6:
        return "medium"
    if q < 0.8:
        return "strong"
    return "very_strong"


def _assign_strength_classes(
    modes: Sequence[Mapping[str, Any]],
    *,
    field: str,
    band: Tuple[float, float] = ACCURACY_BAND_HZ_DEFAULT,
) -> List[str]:
    vals: List[Tuple[float, int]] = []
    for i, m in enumerate(modes):
        f = _safe_float(m.get("frequency_hz"))
        v = _safe_float(m.get(field))
        if f is None or v is None or not (band[0] <= f <= band[1]):
            continue
        vals.append((float(v), i))
    if not vals:
        return [""] * len(modes)
    ordered = sorted(vals, key=lambda t: t[0])
    n = len(ordered)
    labels = [""] * len(modes)
    for rank, (_, idx) in enumerate(ordered):
        q = rank / max(n - 1, 1)
        labels[idx] = _strength_label_from_quantile(q)
    return labels


def enrich_catalog_stk_gains(
    modes: Sequence[Mapping[str, Any]],
    *,
    band: Tuple[float, float] = ACCURACY_BAND_HZ_DEFAULT,
    epsilon: float = INTENSITY_LOG_EPSILON,
    percentile: int = NORMALIZATION_PERCENTILE,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for m in modes:
        rec = dict(m)
        enrich_mode_combined_gains(rec)
        out.append(rec)

    all_fields = list(STK_TARGET_FIELDS)
    p95_map = compute_intensity_p95_map_extended(out, band=band, percentile=percentile, fields_override=all_fields)

    for rec in out:
        enrich_mode_intensity_derivatives(rec, p95_map=p95_map, epsilon=epsilon)
        for base in COMBINED_GAIN_FIELDS:
            stem = base.replace("_raw", "")
            v = _safe_float(rec.get(base))
            if v is not None:
                rec[f"{stem}_log10"] = round(math.log10(abs(v) + epsilon), 8)
            p95 = p95_map.get(base)
            if v is not None and p95 is not None and p95 > 0:
                rec[f"{stem}_p95_norm"] = round(abs(v) / p95, 8)

    strength_maps: Dict[str, List[str]] = {}
    for field in ("mic_output_proxy", "bridge_to_mic_gain_raw", "radiation_proxy", "bridge_to_radiation_gain_raw"):
        labels = _assign_strength_classes(out, field=field, band=band)
        strength_maps[field] = labels
        for i, rec in enumerate(out):
            rec[f"{field}_strength_class"] = labels[i] if i < len(labels) else ""

    meta = {
        "band_hz": list(band),
        "epsilon": epsilon,
        "percentile": percentile,
        "p95_map": p95_map,
        "strength_quantile_method": "within_guitar_quintiles",
        "strength_labels": list(STRENGTH_LABELS),
    }
    return out, meta


def compute_intensity_p95_map_extended(
    modes: Sequence[Mapping[str, Any]],
    *,
    band: Tuple[float, float],
    percentile: int,
    fields_override: Sequence[str],
) -> Dict[str, Optional[float]]:
    buckets: Dict[str, List[float]] = {f: [] for f in fields_override}
    for m in modes:
        f_hz = _safe_float(m.get("frequency_hz"))
        if f_hz is None or not (band[0] <= f_hz <= band[1]):
            continue
        for field in fields_override:
            v = _safe_float(m.get(field))
            if v is not None and (v >= 0.0 or field.startswith("signed")):
                buckets[field].append(abs(v) if field.startswith("signed") else v)
    return {field: _percentile(vals, float(percentile)) for field, vals in buckets.items()}


