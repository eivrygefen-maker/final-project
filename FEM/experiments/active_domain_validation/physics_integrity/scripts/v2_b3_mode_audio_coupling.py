#!/usr/bin/env python3
"""Lightweight per-mode ROM/STK/audio coupling scalars (solve-time, no mode-shape storage)."""
from __future__ import annotations

import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from v2_b3_rich_modal_lib import (  # noqa: E402
    prolongate_active_to_W,
    safe_float,
)

AUDIO_COUPLING_METHOD = "lightweight_modal_coupling_v1"
MODAL_NORM_METHOD = "full_W_l2_norm_v1"

RADIATION_PROXY_WEIGHTS = {"top": 0.45, "back": 0.15, "air": 0.40}

STK_ROM_GUIDANCE = (
    "Use frequency_hz, top_share/back_share/air_share for damping, "
    "bridge_excitation_coupling for excitation, radiation_proxy or mic_output_proxy for output, "
    "and modal_norm for cross-mode amplitude comparison. Not full mode shapes."
)

AUDIO_COUPLING_FIELD_KEYS = (
    "bridge_excitation_coupling",
    "bridge_excitation_abs",
    "bridge_excitation_region",
    "bridge_excitation_status",
    "top_output_proxy",
    "back_output_proxy",
    "air_pressure_proxy",
    "radiation_proxy",
    "mic_output_proxy",
    "mic_output_method",
    "mic_output_status",
    "modal_norm",
    "modal_norm_method",
    "audio_coupling_status",
    "audio_coupling_method",
    "audio_coupling_detail",
)


def _rms(values: np.ndarray) -> Optional[float]:
    if values.size == 0:
        return None
    return float(np.sqrt(np.mean(values.astype(np.float64) ** 2)))


def _mean_signed(values: np.ndarray) -> Optional[float]:
    if values.size == 0:
        return None
    return float(np.mean(values.astype(np.float64)))


def _safe_norm_ratio(value: Optional[float], norm: float) -> Optional[float]:
    if value is None:
        return None
    denom = max(float(norm), 1.0e-30)
    out = float(value) / denom
    return out if math.isfinite(out) else None


def _back_region_indices(region: Mapping[str, Any], *, back_includes_ribs: bool) -> np.ndarray:
    back = np.asarray(region.get("u_idx_back", []), dtype=np.int32).ravel()
    if not back_includes_ribs:
        return back
    ribs = np.asarray(region.get("u_idx_ribs", []), dtype=np.int32).ravel()
    if ribs.size == 0:
        return back
    if back.size == 0:
        return ribs
    return np.unique(np.concatenate([back, ribs]))


def _bridge_region_indices(region: Mapping[str, Any]) -> tuple[np.ndarray, str, str]:
    bridge = np.asarray(region.get("u_idx_bridge", []), dtype=np.int32).ravel()
    if bridge.size > 0:
        return bridge, "bridge_mask", "computed"
    top = np.asarray(region.get("u_idx_top", []), dtype=np.int32).ravel()
    if top.size > 0:
        return top, "top_plate_proxy", "proxy"
    return np.asarray([], dtype=np.int32), "none", "not_available"


def _weighted_radiation_proxy(
    *,
    top: Optional[float],
    back: Optional[float],
    air: Optional[float],
) -> Optional[float]:
    parts: List[tuple[float, float]] = []
    for name, val in (("top", top), ("back", back), ("air", air)):
        if val is not None and math.isfinite(float(val)):
            parts.append((RADIATION_PROXY_WEIGHTS[name], float(val)))
    if not parts:
        return None
    weight_sum = sum(w for w, _ in parts)
    if weight_sum <= 0.0:
        return None
    return float(sum(w * v for w, v in parts) / weight_sum)


def _mic_output_from_proxies(
    *,
    soundhole_rms: Optional[float],
    cavity_pressure: Optional[float],
    aperture_pressure: Optional[float],
    radiation_proxy: Optional[float],
    structural_available: bool,
    pressure_available: bool,
    aperture_available: bool,
) -> tuple[Optional[float], str, str]:
    if aperture_available and aperture_pressure is not None:
        return aperture_pressure, "aperture_pressure_rms_proxy_v1", "proxy"
    if structural_available and soundhole_rms is not None:
        return soundhole_rms, "soundhole_displacement_rms_proxy_v1", "proxy"
    if pressure_available and cavity_pressure is not None:
        return cavity_pressure, "cavity_pressure_max_proxy_v1", "proxy"
    if radiation_proxy is not None:
        return radiation_proxy, "radiation_proxy_blend_v1", "proxy"
    return None, "not_available", "not_available"


def audio_coupling_fields_not_available(*, reason: str, detail: str = "") -> Dict[str, Any]:
    return {
        "bridge_excitation_coupling": None,
        "bridge_excitation_abs": None,
        "bridge_excitation_region": "none",
        "bridge_excitation_status": "not_available",
        "top_output_proxy": None,
        "back_output_proxy": None,
        "air_pressure_proxy": None,
        "radiation_proxy": None,
        "mic_output_proxy": None,
        "mic_output_method": "not_available",
        "mic_output_status": "not_available",
        "modal_norm": None,
        "modal_norm_method": MODAL_NORM_METHOD,
        "audio_coupling_status": "not_available",
        "audio_coupling_method": AUDIO_COUPLING_METHOD,
        "audio_coupling_detail": detail or reason,
    }


def compute_lightweight_audio_coupling(
    *,
    x_active: np.ndarray,
    built: Mapping[str, Any],
    region_ctx: Mapping[str, Any],
) -> Dict[str, Any]:
    """Scalar ROM/STK/audio coupling from in-memory accepted mode vector."""
    region = region_ctx["region"]
    structural_ok = bool(region_ctx.get("structural_indices_available"))
    pressure_ok = bool(region_ctx.get("pressure_indices_available"))
    back_includes_ribs = bool(region_ctx.get("back_includes_ribs", True))
    detail = str(region_ctx.get("region_dof_source") or "")

    x_full = prolongate_active_to_W(np.asarray(x_active, dtype=np.float64), dict(built))
    modal_norm = float(np.linalg.norm(x_full))
    if modal_norm <= 0.0:
        return audio_coupling_fields_not_available(reason="zero_modal_norm", detail=detail)

    bridge_idx, bridge_region, bridge_status = _bridge_region_indices(region)
    if not structural_ok:
        bridge_status = "not_available"
        bridge_region = "none"

    bridge_vals = x_full[bridge_idx] if bridge_idx.size else np.asarray([], dtype=np.float64)
    bridge_signed = _mean_signed(bridge_vals)
    bridge_abs = _rms(bridge_vals)

    top_vals = (
        x_full[np.asarray(region.get("u_idx_top", []), dtype=np.int32)]
        if structural_ok
        else np.asarray([], dtype=np.float64)
    )
    back_vals = (
        x_full[_back_region_indices(region, back_includes_ribs=back_includes_ribs)]
        if structural_ok
        else np.asarray([], dtype=np.float64)
    )
    sh_vals = (
        x_full[np.asarray(region.get("u_idx_soundhole", []), dtype=np.int32)]
        if structural_ok
        else np.asarray([], dtype=np.float64)
    )
    p_vals = (
        x_full[np.asarray(region.get("p_idx_air", []), dtype=np.int32)]
        if pressure_ok
        else np.asarray([], dtype=np.float64)
    )
    p_aperture_idx = np.asarray(region.get("p_idx_aperture", []), dtype=np.int32).ravel()
    aperture_ok = pressure_ok and p_aperture_idx.size > 0
    aperture_vals = x_full[p_aperture_idx] if aperture_ok else np.asarray([], dtype=np.float64)

    top_rms = _rms(top_vals)
    back_rms = _rms(back_vals)
    sh_rms = _rms(sh_vals)
    air_max = float(np.max(np.abs(p_vals))) if p_vals.size else None
    air_rms = _rms(p_vals)

    top_proxy = _safe_norm_ratio(top_rms, modal_norm)
    back_proxy = _safe_norm_ratio(back_rms, modal_norm)
    air_proxy = _safe_norm_ratio(air_max if air_max is not None else air_rms, modal_norm)
    radiation_proxy = _weighted_radiation_proxy(top=top_proxy, back=back_proxy, air=air_proxy)

    aperture_rms = _rms(aperture_vals)
    mic_proxy, mic_method, mic_status = _mic_output_from_proxies(
        soundhole_rms=_safe_norm_ratio(sh_rms, modal_norm),
        cavity_pressure=air_proxy,
        aperture_pressure=_safe_norm_ratio(aperture_rms, modal_norm),
        radiation_proxy=radiation_proxy,
        structural_available=structural_ok,
        pressure_available=pressure_ok,
        aperture_available=aperture_ok,
    )

    import os

    from v2_b3_m4_production_contracts import require_aperture_mask_production  # noqa: WPS433

    if require_aperture_mask_production() and not aperture_ok:
        return audio_coupling_fields_not_available(
            reason="empty_p_idx_aperture",
            detail="production_requires_aperture_pressure_rms_proxy_v1",
        )
    if require_aperture_mask_production() and mic_method != "aperture_pressure_rms_proxy_v1":
        return audio_coupling_fields_not_available(
            reason="mic_output_method_not_aperture_proxy",
            detail=f"got {mic_method}",
        )

    if structural_ok and pressure_ok:
        audio_status = "computed"
    elif structural_ok or pressure_ok:
        audio_status = "partial"
    else:
        audio_status = "not_available"

    return {
        "bridge_excitation_coupling": _safe_norm_ratio(bridge_signed, modal_norm),
        "bridge_excitation_abs": _safe_norm_ratio(bridge_abs, modal_norm),
        "bridge_excitation_region": bridge_region,
        "bridge_excitation_status": bridge_status,
        "top_output_proxy": top_proxy,
        "back_output_proxy": back_proxy,
        "air_pressure_proxy": air_proxy,
        "radiation_proxy": radiation_proxy,
        "mic_output_proxy": mic_proxy,
        "mic_output_method": mic_method,
        "mic_output_status": mic_status,
        "modal_norm": safe_float(modal_norm),
        "modal_norm_method": MODAL_NORM_METHOD,
        "audio_coupling_status": audio_status,
        "audio_coupling_method": AUDIO_COUPLING_METHOD,
        "audio_coupling_detail": detail,
    }


def compute_audio_coupling_from_norms(
    *,
    u_norm_W: float,
    p_norm_W: float,
    x_norm_W: float,
) -> Dict[str, Any]:
    """Partial fallback when only scalar norms were persisted (no vector recompute)."""
    modal_norm = max(float(x_norm_W), 1.0e-30)
    air_proxy = _safe_norm_ratio(float(p_norm_W), modal_norm)
    radiation_proxy = air_proxy
    return {
        "bridge_excitation_coupling": None,
        "bridge_excitation_abs": None,
        "bridge_excitation_region": "none",
        "bridge_excitation_status": "not_available",
        "top_output_proxy": None,
        "back_output_proxy": None,
        "air_pressure_proxy": air_proxy,
        "radiation_proxy": radiation_proxy,
        "mic_output_proxy": air_proxy,
        "mic_output_method": "cavity_pressure_norm_proxy_v1",
        "mic_output_status": "partial",
        "modal_norm": safe_float(modal_norm),
        "modal_norm_method": MODAL_NORM_METHOD,
        "audio_coupling_status": "partial",
        "audio_coupling_method": AUDIO_COUPLING_METHOD,
        "audio_coupling_detail": "top/back/bridge require region_dof_indices.npz and mode vector",
    }


def attach_audio_coupling_to_accepted_mode(
    entry: Dict[str, Any],
    *,
    x_active: Optional[np.ndarray],
    built: Mapping[str, Any],
    region_ctx: Optional[Mapping[str, Any]],
) -> None:
    """Mutate accepted-mode dict with lightweight audio coupling scalars."""
    if x_active is not None and region_ctx is not None:
        import os

        if os.environ.get("B3_MIC_PROXY_MODE") == "aperture_pressure_rms_v1":
            mask_npz = os.environ.get("B3_EXPERIMENTAL_APERTURE_MASK_NPZ")
            if mask_npz and Path(mask_npz).is_file():
                try:
                    import numpy as np
                    from v2_b3_mode_audio_coupling_experimental import (  # noqa: WPS433
                        compute_experimental_audio_coupling,
                    )

                    with np.load(mask_npz, allow_pickle=False) as z:
                        p_idx_aperture = np.asarray(z["p_idx_aperture"], dtype=np.int32).ravel()
                    exp = compute_experimental_audio_coupling(
                        x_active=x_active,
                        built=built,
                        region_ctx=region_ctx,
                        p_idx_aperture=p_idx_aperture,
                    )
                    if "mic_output_method" in z.files:
                        exp["mic_output_method"] = str(np.asarray(z["mic_output_method"]).item())
                    entry.update(exp)
                    return
                except Exception as exc:
                    entry.update(
                        audio_coupling_fields_not_available(
                            reason="experimental_aperture_proxy_failed",
                            detail=f"{type(exc).__name__}:{exc}",
                        )
                    )
                    return
        entry.update(
            compute_lightweight_audio_coupling(
                x_active=x_active,
                built=built,
                region_ctx=region_ctx,
            )
        )
        return
    if entry.get("x_norm_W") is not None:
        entry.update(
            compute_audio_coupling_from_norms(
                u_norm_W=float(entry.get("u_norm_W") or 0.0),
                p_norm_W=float(entry.get("p_norm_W") or 0.0),
                x_norm_W=float(entry["x_norm_W"]),
            )
        )
        return
    entry.update(audio_coupling_fields_not_available(reason="no_vector_or_norms"))


def merge_audio_coupling_into_catalog_record(record: Dict[str, Any], mode: Mapping[str, Any]) -> None:
    """Copy audio coupling fields from solver accepted_modes into aggregation record."""
    for key in AUDIO_COUPLING_FIELD_KEYS:
        if key in mode:
            record[key] = mode[key]


def _summarize_scalar_field(records: Sequence[Mapping[str, Any]], key: str) -> Dict[str, Any]:
    vals = [
        float(r[key])
        for r in records
        if r.get(key) is not None and math.isfinite(float(r[key]))
    ]
    if not vals:
        return {"count": 0}
    return {
        "count": len(vals),
        "median": float(statistics.median(vals)),
        "mean": float(statistics.mean(vals)),
        "max": float(max(vals)),
        "min": float(min(vals)),
    }


def summarize_audio_coupling(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Aggregate audio coupling statistics for modes_summary.json."""
    status_counts: Dict[str, int] = {}
    bridge_status_counts: Dict[str, int] = {}
    mic_status_counts: Dict[str, int] = {}
    audio_computed = 0
    bridge_available = 0
    mic_proxy_available = 0

    for rec in records:
        ac_status = str(rec.get("audio_coupling_status") or "not_available")
        status_counts[ac_status] = status_counts.get(ac_status, 0) + 1
        if ac_status in ("computed", "partial", "proxy"):
            audio_computed += 1

        bs = str(rec.get("bridge_excitation_status") or "not_available")
        bridge_status_counts[bs] = bridge_status_counts.get(bs, 0) + 1
        if rec.get("bridge_excitation_coupling") is not None or rec.get("bridge_excitation_abs") is not None:
            bridge_available += 1
        elif bs in ("computed", "proxy"):
            bridge_available += 1

        ms = str(rec.get("mic_output_status") or "not_available")
        mic_status_counts[ms] = mic_status_counts.get(ms, 0) + 1
        if rec.get("mic_output_proxy") is not None:
            mic_proxy_available += 1

    return {
        "audio_coupling_computed_count": audio_computed,
        "bridge_coupling_available_count": bridge_available,
        "mic_proxy_available_count": mic_proxy_available,
        "audio_coupling_status_counts": status_counts,
        "bridge_excitation_status_counts": bridge_status_counts,
        "mic_output_status_counts": mic_status_counts,
        "radiation_proxy_summary": _summarize_scalar_field(records, "radiation_proxy"),
        "modal_norm_summary": _summarize_scalar_field(records, "modal_norm"),
        "bridge_excitation_abs_summary": _summarize_scalar_field(records, "bridge_excitation_abs"),
        "mic_output_proxy_summary": _summarize_scalar_field(records, "mic_output_proxy"),
    }
