#!/usr/bin/env python3
"""Experimental mic/output proxy variants — not imported by production workers."""
from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional

import numpy as np

from v2_b3_mode_audio_coupling import (  # noqa: E402
    AUDIO_COUPLING_METHOD,
    MODAL_NORM_METHOD,
    _rms,
    _safe_norm_ratio,
    _weighted_radiation_proxy,
    compute_lightweight_audio_coupling,
)
from v2_b3_rich_modal_lib import prolongate_active_to_W  # noqa: E402

EXPERIMENTAL_MIC_METHOD = "aperture_pressure_rms_proxy_v1"
MIC_PROXY_MODE_ENV = "B3_MIC_PROXY_MODE"
APERTURE_MASK_NPZ_ENV = "B3_EXPERIMENTAL_APERTURE_MASK_NPZ"


def _mic_from_aperture_pressure(
    *,
    x_full: np.ndarray,
    p_idx_aperture: np.ndarray,
    modal_norm: float,
) -> tuple[Optional[float], str, str]:
    if p_idx_aperture.size == 0:
        return None, "aperture_mask_empty", "not_available"
    vals = x_full[np.asarray(p_idx_aperture, dtype=np.int32)]
    rms = _rms(vals)
    proxy = _safe_norm_ratio(rms, modal_norm)
    if proxy is None:
        return None, EXPERIMENTAL_MIC_METHOD, "not_available"
    return proxy, EXPERIMENTAL_MIC_METHOD, "proxy"


def compute_experimental_audio_coupling(
    *,
    x_active: np.ndarray,
    built: Mapping[str, Any],
    region_ctx: Mapping[str, Any],
    p_idx_aperture: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Production coupling plus experimental aperture-pressure mic proxy override.

    When p_idx_aperture is non-empty, mic_output_proxy uses RMS over aperture probe DOFs
    instead of cavity max / empty soundhole displacement mask.
    """
    base = compute_lightweight_audio_coupling(
        x_active=x_active,
        built=built,
        region_ctx=region_ctx,
    )
    if p_idx_aperture is None or np.asarray(p_idx_aperture).size == 0:
        base["experimental_mic_note"] = "aperture_mask_missing; production fallback retained"
        return base

    region = region_ctx["region"]
    x_full = prolongate_active_to_W(np.asarray(x_active, dtype=np.float64), dict(built))
    modal_norm = float(np.linalg.norm(x_full))
    if modal_norm <= 0.0:
        base["experimental_mic_note"] = "zero_modal_norm"
        return base

    mic_new, mic_method, mic_status = _mic_from_aperture_pressure(
        x_full=x_full,
        p_idx_aperture=np.asarray(p_idx_aperture, dtype=np.int32).ravel(),
        modal_norm=modal_norm,
    )
    out = dict(base)
    out["mic_output_proxy_legacy"] = base.get("mic_output_proxy")
    out["mic_output_method_legacy"] = base.get("mic_output_method")
    out["mic_output_proxy"] = mic_new
    out["mic_output_method"] = mic_method
    out["mic_output_status"] = mic_status
    out["audio_coupling_method"] = f"{AUDIO_COUPLING_METHOD}+experimental_aperture_v1"
    out["experimental_aperture_dof_count"] = int(np.asarray(p_idx_aperture).size)
    if mic_new is not None and base.get("mic_output_proxy") is not None:
        legacy = float(base["mic_output_proxy"])
        out["experimental_mic_ratio_vs_legacy"] = (
            float(mic_new) / legacy if legacy > 0 and math.isfinite(legacy) else None
        )
    return out
