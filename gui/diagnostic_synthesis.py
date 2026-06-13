#!/usr/bin/env python3
"""
Diagnostic synthesis modes for A/B body-difference listening tests (no FEM).

Modes compare how normalization, broad body spectral color, and damping affect
per-guitar timbre when string excitation is identical.
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

BROAD_BODY_BANDS_HZ: Tuple[Tuple[float, float], ...] = (
    (60.0, 120.0),
    (120.0, 180.0),
    (180.0, 280.0),
    (280.0, 420.0),
    (420.0, 650.0),
    (650.0, 1000.0),
)


@dataclass(frozen=True)
class DiagnosticSynthesisConfig:
    name: str
    raw_body_variation_preserve: float = 0.0
    body_gain_normalization_strength: float = 1.0
    final_loudness_normalization_strength: float = 1.0
    wide_body_signature: bool = False
    wide_body_signature_strength: float = 0.35
    wide_body_signature_damping: bool = False
    damping_strength: float = 0.0
    per_mode_damping_strength: float = 1.0
    all_mode_broad_contribution: bool = False
    broad_all_mode_strength: float = 0.0
    near_modal_boost: float = 1.0
    near_modal_energy_target: float = 0.0
    far_broad_energy_target: float = 0.0
    far_mode_color_gain: float = 1.0
    high_note_string_direct_scale: float = 1.0
    high_note_body_color_boost: float = 1.0
    high_note_body_to_string_target_ratio: Optional[float] = None
    high_note_pitch_layer_scale: float = 1.0
    fundamental_anchor_scale: float = 1.0
    low_note_fundamental_harmonic_boost: float = 1.55
    description: str = ""

    def to_metadata_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def effective_body_gain_preserve(self) -> float:
        """Fraction of raw body gain kept (1 = no calibration)."""
        if self.raw_body_variation_preserve > 0:
            return self.raw_body_variation_preserve
        return max(0.0, min(1.0, 1.0 - self.body_gain_normalization_strength))

    def effective_loudness_preserve(self) -> float:
        if self.raw_body_variation_preserve > 0 and self.final_loudness_normalization_strength >= 0.999:
            return self.raw_body_variation_preserve * 0.5
        return max(0.0, min(1.0, 1.0 - self.final_loudness_normalization_strength))


DIAGNOSTIC_MODES: Dict[str, DiagnosticSynthesisConfig] = {
    "baseline_current": DiagnosticSynthesisConfig(
        name="baseline_current",
        description="Production synthesis — full body/string calibration and loudness normalize.",
    ),
    "preserve_raw_body_variation": DiagnosticSynthesisConfig(
        name="preserve_raw_body_variation",
        raw_body_variation_preserve=0.5,
        description="Partial preservation of raw body gain and loudness differences between guitars.",
    ),
    "wide_body_signature": DiagnosticSynthesisConfig(
        name="wide_body_signature",
        raw_body_variation_preserve=0.35,
        wide_body_signature=True,
        wide_body_signature_strength=0.42,
        description="Broad frequency-band body color from modal catalog (not only near-harmonic peaks).",
    ),
    "wide_body_signature_damping": DiagnosticSynthesisConfig(
        name="wide_body_signature_damping",
        raw_body_variation_preserve=0.35,
        wide_body_signature=True,
        wide_body_signature_strength=0.42,
        wide_body_signature_damping=True,
        damping_strength=0.55,
        description="Wide body signature plus sample-parameter mode damping / Q variation.",
    ),
    "modal_damping_body_signature_v1": DiagnosticSynthesisConfig(
        name="modal_damping_body_signature_v1",
        raw_body_variation_preserve=0.55,
        body_gain_normalization_strength=0.45,
        final_loudness_normalization_strength=0.42,
        wide_body_signature=True,
        wide_body_signature_strength=0.52,
        wide_body_signature_damping=True,
        damping_strength=1.0,
        per_mode_damping_strength=1.0,
        all_mode_broad_contribution=True,
        broad_all_mode_strength=0.50,
        near_modal_boost=1.28,
        high_note_string_direct_scale=0.58,
        high_note_body_color_boost=1.38,
        high_note_body_to_string_target_ratio=5.9,
        high_note_pitch_layer_scale=0.68,
        fundamental_anchor_scale=0.50,
        low_note_fundamental_harmonic_boost=1.22,
        description=(
            "Structural STK v1: per-mode damping, all-mode broad body color, "
            "partial normalization, reduced string dominance (esp. high notes), "
            "safer low-note fundamental handling."
        ),
    ),
    "modal_body_60_40_v1": DiagnosticSynthesisConfig(
        name="modal_body_60_40_v1",
        raw_body_variation_preserve=0.50,
        body_gain_normalization_strength=0.40,
        final_loudness_normalization_strength=0.38,
        wide_body_signature=True,
        wide_body_signature_strength=0.55,
        wide_body_signature_damping=True,
        damping_strength=1.0,
        per_mode_damping_strength=1.0,
        all_mode_broad_contribution=True,
        broad_all_mode_strength=0.78,
        near_modal_boost=1.10,
        near_modal_energy_target=0.60,
        far_broad_energy_target=0.40,
        far_mode_color_gain=1.42,
        high_note_string_direct_scale=0.62,
        high_note_body_color_boost=1.45,
        high_note_body_to_string_target_ratio=6.1,
        high_note_pitch_layer_scale=0.70,
        fundamental_anchor_scale=0.45,
        low_note_fundamental_harmonic_boost=1.18,
        description=(
            "60/40 near/far body-color target: stronger far-mode broad contribution, "
            "share-weighted material damping, partial normalization, f0-based string/body balance."
        ),
    ),
    "modal_radiation_color_v1": DiagnosticSynthesisConfig(
        name="modal_radiation_color_v1",
        raw_body_variation_preserve=0.52,
        body_gain_normalization_strength=0.38,
        final_loudness_normalization_strength=0.36,
        wide_body_signature=False,
        wide_body_signature_strength=0.0,
        wide_body_signature_damping=True,
        damping_strength=1.0,
        per_mode_damping_strength=1.0,
        all_mode_broad_contribution=True,
        broad_all_mode_strength=0.62,
        near_modal_boost=1.05,
        near_modal_energy_target=0.55,
        far_broad_energy_target=0.45,
        far_mode_color_gain=1.0,
        high_note_string_direct_scale=0.60,
        high_note_body_color_boost=1.50,
        high_note_body_to_string_target_ratio=6.0,
        high_note_pitch_layer_scale=0.68,
        fundamental_anchor_scale=0.42,
        low_note_fundamental_harmonic_boost=1.12,
        description=(
            "Stage 4.6 prototype: separate bridge mobility × radiation transmittance per mode, "
            "category/wood amplitude color, sample-specific far modes — no global broad EQ."
        ),
    ),
    "body_audibility_balance_probe_v1": DiagnosticSynthesisConfig(
        name="body_audibility_balance_probe_v1",
        raw_body_variation_preserve=0.62,
        body_gain_normalization_strength=0.28,
        final_loudness_normalization_strength=0.26,
        wide_body_signature=False,
        all_mode_broad_contribution=True,
        broad_all_mode_strength=0.55,
        high_note_string_direct_scale=0.52,
        high_note_body_color_boost=1.62,
        high_note_body_to_string_target_ratio=5.8,
        high_note_pitch_layer_scale=0.62,
        fundamental_anchor_scale=0.30,
        low_note_fundamental_harmonic_boost=1.02,
        description=(
            "Stage 4.8 probe: slightly reduce direct string, increase body audibility, "
            "preserve raw variation — continuous f0 scaling only."
        ),
    ),
    "modal_body_signature_v3_core": DiagnosticSynthesisConfig(
        name="modal_body_signature_v3_core",
        raw_body_variation_preserve=0.52,
        body_gain_normalization_strength=0.38,
        final_loudness_normalization_strength=0.36,
        wide_body_signature=False,
        wide_body_signature_damping=True,
        damping_strength=1.0,
        per_mode_damping_strength=1.0,
        all_mode_broad_contribution=True,
        broad_all_mode_strength=0.62,
        near_modal_boost=1.05,
        near_modal_energy_target=0.55,
        far_broad_energy_target=0.45,
        high_note_string_direct_scale=0.60,
        high_note_body_color_boost=1.50,
        high_note_body_to_string_target_ratio=6.0,
        high_note_pitch_layer_scale=0.68,
        fundamental_anchor_scale=0.42,
        low_note_fundamental_harmonic_boost=1.12,
        description="Stage 4.9 V3 core: radiation v1 base only (ablation control).",
    ),
    "modal_body_signature_v3_low_f0_imprint_only": DiagnosticSynthesisConfig(
        name="modal_body_signature_v3_low_f0_imprint_only",
        raw_body_variation_preserve=0.52,
        body_gain_normalization_strength=0.38,
        final_loudness_normalization_strength=0.36,
        wide_body_signature=False,
        wide_body_signature_damping=True,
        all_mode_broad_contribution=True,
        broad_all_mode_strength=0.62,
        near_modal_boost=1.05,
        high_note_string_direct_scale=0.60,
        high_note_body_color_boost=1.50,
        high_note_body_to_string_target_ratio=6.0,
        high_note_pitch_layer_scale=0.68,
        fundamental_anchor_scale=0.42,
        low_note_fundamental_harmonic_boost=1.12,
        description="Stage 4.9 V3 ablation: v1 + continuous low-f0 harmonic/body imprint.",
    ),
    "modal_body_signature_v3_mobility_only": DiagnosticSynthesisConfig(
        name="modal_body_signature_v3_mobility_only",
        raw_body_variation_preserve=0.54,
        body_gain_normalization_strength=0.36,
        final_loudness_normalization_strength=0.34,
        wide_body_signature=False,
        wide_body_signature_damping=True,
        all_mode_broad_contribution=True,
        broad_all_mode_strength=0.62,
        near_modal_boost=1.05,
        high_note_string_direct_scale=0.60,
        high_note_body_color_boost=1.50,
        high_note_body_to_string_target_ratio=6.0,
        high_note_pitch_layer_scale=0.68,
        fundamental_anchor_scale=0.42,
        low_note_fundamental_harmonic_boost=1.12,
        description="Stage 4.9 V3 ablation: v1 + bounded geometry bridge mobility.",
    ),
    "modal_body_signature_v3_far_color_only": DiagnosticSynthesisConfig(
        name="modal_body_signature_v3_far_color_only",
        raw_body_variation_preserve=0.52,
        body_gain_normalization_strength=0.38,
        final_loudness_normalization_strength=0.36,
        wide_body_signature=False,
        wide_body_signature_damping=True,
        all_mode_broad_contribution=True,
        broad_all_mode_strength=0.66,
        near_modal_boost=1.02,
        far_broad_energy_target=0.48,
        high_note_string_direct_scale=0.60,
        high_note_body_color_boost=1.50,
        high_note_body_to_string_target_ratio=6.0,
        high_note_pitch_layer_scale=0.68,
        fundamental_anchor_scale=0.42,
        low_note_fundamental_harmonic_boost=1.12,
        description="Stage 4.9 V3 ablation: v1 + smoothed far/background body color.",
    ),
    "modal_body_signature_v3_full": DiagnosticSynthesisConfig(
        name="modal_body_signature_v3_full",
        raw_body_variation_preserve=0.55,
        body_gain_normalization_strength=0.34,
        final_loudness_normalization_strength=0.32,
        wide_body_signature=False,
        wide_body_signature_damping=True,
        all_mode_broad_contribution=True,
        broad_all_mode_strength=0.64,
        near_modal_boost=1.04,
        far_broad_energy_target=0.46,
        high_note_string_direct_scale=0.62,
        high_note_body_color_boost=1.48,
        high_note_body_to_string_target_ratio=5.9,
        high_note_pitch_layer_scale=0.70,
        fundamental_anchor_scale=0.40,
        low_note_fundamental_harmonic_boost=1.10,
        description="Stage 4.9 V3 full: v1 + low-f0 imprint + mobility + far color.",
    ),
    "modal_body_signature_v3": DiagnosticSynthesisConfig(
        name="modal_body_signature_v3",
        raw_body_variation_preserve=0.55,
        body_gain_normalization_strength=0.34,
        final_loudness_normalization_strength=0.32,
        wide_body_signature=False,
        wide_body_signature_damping=True,
        all_mode_broad_contribution=True,
        broad_all_mode_strength=0.64,
        near_modal_boost=1.04,
        far_broad_energy_target=0.46,
        high_note_string_direct_scale=0.62,
        high_note_body_color_boost=1.48,
        high_note_body_to_string_target_ratio=5.9,
        high_note_pitch_layer_scale=0.70,
        fundamental_anchor_scale=0.40,
        low_note_fundamental_harmonic_boost=1.10,
        description="Stage 4.9 V3 alias — same as modal_body_signature_v3_full.",
    ),
    "modal_radiation_color_v3": DiagnosticSynthesisConfig(
        name="modal_radiation_color_v3",
        raw_body_variation_preserve=0.55,
        body_gain_normalization_strength=0.34,
        final_loudness_normalization_strength=0.32,
        wide_body_signature=False,
        wide_body_signature_damping=True,
        all_mode_broad_contribution=True,
        broad_all_mode_strength=0.64,
        near_modal_boost=1.04,
        far_broad_energy_target=0.46,
        high_note_string_direct_scale=0.62,
        high_note_body_color_boost=1.48,
        high_note_body_to_string_target_ratio=5.9,
        high_note_pitch_layer_scale=0.70,
        fundamental_anchor_scale=0.40,
        low_note_fundamental_harmonic_boost=1.10,
        description="Stage 4.9 V3 alias (radiation_color_v3) — same as v3_full.",
    ),
    "modal_radiation_color_v2": DiagnosticSynthesisConfig(
        name="modal_radiation_color_v2",
        raw_body_variation_preserve=0.58,
        body_gain_normalization_strength=0.32,
        final_loudness_normalization_strength=0.30,
        wide_body_signature=False,
        wide_body_signature_strength=0.0,
        wide_body_signature_damping=True,
        damping_strength=1.0,
        per_mode_damping_strength=1.0,
        all_mode_broad_contribution=True,
        broad_all_mode_strength=0.58,
        near_modal_boost=1.08,
        near_modal_energy_target=0.52,
        far_broad_energy_target=0.48,
        far_mode_color_gain=1.0,
        high_note_string_direct_scale=0.56,
        high_note_body_color_boost=1.58,
        high_note_body_to_string_target_ratio=5.5,
        high_note_pitch_layer_scale=0.64,
        fundamental_anchor_scale=0.34,
        low_note_fundamental_harmonic_boost=1.04,
        description=(
            "Stage 4.7: bridge-gated radiation transmittance — audible strength = "
            "bridge_gate × output_transmittance × category/material color; "
            "no independent radiation source; no global far EQ."
        ),
    ),
    "modal_body_hybrid_v4_core": DiagnosticSynthesisConfig(
        name="modal_body_hybrid_v4_core",
        raw_body_variation_preserve=0.50,
        body_gain_normalization_strength=0.36,
        final_loudness_normalization_strength=0.34,
        description="Stage 5.0 V4 core: f0-continuous baseline/v1 hybrid only.",
    ),
    "modal_body_hybrid_v4_contrast_imprint_only": DiagnosticSynthesisConfig(
        name="modal_body_hybrid_v4_contrast_imprint_only",
        raw_body_variation_preserve=0.50,
        body_gain_normalization_strength=0.36,
        final_loudness_normalization_strength=0.34,
        description="Stage 5.0 V4: hybrid + harmonic contrast imprint.",
    ),
    "modal_body_hybrid_v4_contrast_body_layer_only": DiagnosticSynthesisConfig(
        name="modal_body_hybrid_v4_contrast_body_layer_only",
        raw_body_variation_preserve=0.50,
        body_gain_normalization_strength=0.36,
        final_loudness_normalization_strength=0.34,
        description="Stage 5.0 V4: hybrid + small contrast body residual layer.",
    ),
    "modal_body_hybrid_v4_mobility_light_only": DiagnosticSynthesisConfig(
        name="modal_body_hybrid_v4_mobility_light_only",
        raw_body_variation_preserve=0.50,
        body_gain_normalization_strength=0.36,
        final_loudness_normalization_strength=0.34,
        description="Stage 5.0 V4: hybrid + light bounded mobility in envelope.",
    ),
    "modal_body_hybrid_v4_full": DiagnosticSynthesisConfig(
        name="modal_body_hybrid_v4_full",
        raw_body_variation_preserve=0.52,
        body_gain_normalization_strength=0.34,
        final_loudness_normalization_strength=0.32,
        description="Stage 5.0 V4 full: hybrid + contrast imprint + body layer + mobility.",
    ),
    "modal_body_hybrid_v4": DiagnosticSynthesisConfig(
        name="modal_body_hybrid_v4",
        raw_body_variation_preserve=0.52,
        body_gain_normalization_strength=0.34,
        final_loudness_normalization_strength=0.32,
        description="Stage 5.0 V4 alias — same as modal_body_hybrid_v4_full.",
    ),
    "stk_body_transfer_v4": DiagnosticSynthesisConfig(
        name="stk_body_transfer_v4",
        raw_body_variation_preserve=0.52,
        body_gain_normalization_strength=0.34,
        final_loudness_normalization_strength=0.32,
        description="Stage 5.0 V4 alias (stk_body_transfer_v4).",
    ),
    "modal_body_hybrid_v4_1_core": DiagnosticSynthesisConfig(
        name="modal_body_hybrid_v4_1_core",
        raw_body_variation_preserve=0.50,
        body_gain_normalization_strength=0.36,
        final_loudness_normalization_strength=0.34,
        description="Stage 5.1 V4.1 strict f0 hybrid — baseline/v1 endpoints only.",
    ),
    "modal_body_hybrid_v4_1_full": DiagnosticSynthesisConfig(
        name="modal_body_hybrid_v4_1_full",
        raw_body_variation_preserve=0.50,
        body_gain_normalization_strength=0.36,
        final_loudness_normalization_strength=0.34,
        description="Stage 5.1 V4.1 full (currently same as core).",
    ),
    "modal_body_hybrid_v4_1": DiagnosticSynthesisConfig(
        name="modal_body_hybrid_v4_1",
        raw_body_variation_preserve=0.50,
        body_gain_normalization_strength=0.36,
        final_loudness_normalization_strength=0.34,
        description="Stage 5.1 V4.1 alias — same as core.",
    ),
    "stk_body_transfer_v4_1": DiagnosticSynthesisConfig(
        name="stk_body_transfer_v4_1",
        raw_body_variation_preserve=0.50,
        body_gain_normalization_strength=0.36,
        final_loudness_normalization_strength=0.34,
        description="Stage 5.1 V4.1 alias (stk_body_transfer_v4_1).",
    ),
    "modal_body_hybrid_v4_1_identity_space": DiagnosticSynthesisConfig(
        name="modal_body_hybrid_v4_1_identity_space",
        raw_body_variation_preserve=0.50,
        body_gain_normalization_strength=0.36,
        final_loudness_normalization_strength=0.34,
        description="Stage 5.1C V4.1 + bounded continuous body-identity layer (diagnostic).",
    ),
    "stk_body_transfer_v4_1_identity_space": DiagnosticSynthesisConfig(
        name="stk_body_transfer_v4_1_identity_space",
        raw_body_variation_preserve=0.50,
        body_gain_normalization_strength=0.36,
        final_loudness_normalization_strength=0.34,
        description="Stage 5.1C identity-space alias.",
    ),
}

_active_diagnostic: Optional[DiagnosticSynthesisConfig] = None
_active_sample_parameters: Dict[str, Any] = {}


def list_diagnostic_modes() -> List[str]:
    return list(DIAGNOSTIC_MODES.keys())


def get_diagnostic_mode(name: str) -> DiagnosticSynthesisConfig:
    key = str(name or "baseline_current").strip().lower()
    if key not in DIAGNOSTIC_MODES:
        raise ValueError(f"unknown diagnostic mode: {name!r}")
    return DIAGNOSTIC_MODES[key]


def active_diagnostic() -> Optional[DiagnosticSynthesisConfig]:
    return _active_diagnostic


def active_sample_parameters() -> Dict[str, Any]:
    global _active_sample_parameters
    return dict(_active_sample_parameters or {})


@contextmanager
def use_diagnostic_mode(
    mode_name: Optional[str],
    *,
    sample_parameters: Optional[Mapping[str, Any]] = None,
) -> Iterator[Optional[DiagnosticSynthesisConfig]]:
    global _active_diagnostic, _active_sample_parameters
    saved_mode = _active_diagnostic
    saved_params = _active_sample_parameters
    from sample_parameters import normalize_sample_parameters

    if mode_name:
        _active_diagnostic = get_diagnostic_mode(mode_name)
    else:
        _active_diagnostic = None
    if sample_parameters is not None:
        _active_sample_parameters = normalize_sample_parameters(sample_parameters)
    try:
        yield _active_diagnostic
    finally:
        _active_diagnostic = saved_mode
        _active_sample_parameters = saved_params


def blend_toward_unity(gain: float, preserve: float) -> float:
    """preserve=1 → gain 1.0; preserve=0 → unchanged gain."""
    p = max(0.0, min(1.0, float(preserve)))
    return 1.0 + (float(gain) - 1.0) * (1.0 - p)


def compute_broad_body_band_gains(
    band_modes: Sequence[Mapping[str, Any]],
) -> Dict[str, float]:
    """Per-guitar band energy → gentle EQ curve (mean-normalized to 1.0)."""
    energies = [0.0] * len(BROAD_BODY_BANDS_HZ)
    for mode in band_modes:
        f_m = float(mode.get("frequency_hz") or 0.0)
        if f_m <= 0:
            continue
        w = float(mode.get("bridge_excitation_abs") or mode.get("mic_output_proxy") or 0.01)
        w += 0.35 * float(mode.get("radiation_proxy") or 0.0)
        w += 0.2 * float(mode.get("air_share") or 0.0)
        for i, (lo, hi) in enumerate(BROAD_BODY_BANDS_HZ):
            if lo <= f_m < hi:
                energies[i] += max(w, 1e-9)
                break
    if not any(e > 0 for e in energies):
        return {f"band_{int(lo)}_{int(hi)}": 1.0 for lo, hi in BROAD_BODY_BANDS_HZ}
    mean_e = sum(energies) / max(len(energies), 1)
    mean_e = max(mean_e, 1e-12)
    out: Dict[str, float] = {}
    for (lo, hi), e in zip(BROAD_BODY_BANDS_HZ, energies):
        raw = e / mean_e if e > 0 else 0.65
        # Soft compression — avoid extreme EQ
        gain = 0.72 + 0.28 * math.sqrt(max(raw, 0.0))
        out[f"band_{int(lo)}_{int(hi)}"] = round(gain, 6)
    return out


def broad_signature_curve(
    freqs_hz: np.ndarray,
    band_gains: Mapping[str, float],
    *,
    strength: float,
) -> np.ndarray:
    """Smooth multi-band color curve for H_body(f)."""
    s = max(0.0, min(1.0, float(strength)))
    if s <= 0:
        return np.ones_like(freqs_hz, dtype=np.float64)
    curve = np.ones_like(freqs_hz, dtype=np.float64)
    for (lo, hi), gain in zip(BROAD_BODY_BANDS_HZ, _ordered_band_gain_values(band_gains)):
        mask = (freqs_hz >= lo) & (freqs_hz < hi)
        if np.any(mask):
            blended = 1.0 + (gain - 1.0) * s
            curve[mask] *= blended
    return curve


def _ordered_band_gain_values(band_gains: Mapping[str, float]) -> List[float]:
    vals: List[float] = []
    for lo, hi in BROAD_BODY_BANDS_HZ:
        key = f"band_{int(lo)}_{int(hi)}"
        vals.append(float(band_gains.get(key, 1.0)))
    return vals


def sample_damping_q_scale(
    parameters: Mapping[str, Any],
    mode_hz: float,
    *,
    strength: float,
) -> float:
    """
    >1.0 increases damping (lowers effective Q) based on geometry/material proxies.
    """
    s = max(0.0, min(1.0, float(strength)))
    if s <= 0:
        return 1.0
    top = str(parameters.get("top_wood_id") or "").lower()
    back = str(parameters.get("back_wood_id") or "").lower()
    top_t = float(parameters.get("geometry.top_thickness") or 0.003)
    back_t = float(parameters.get("geometry.back_thickness") or 0.0033)
    depth = float(parameters.get("geometry.depth") or 0.1)
    width = float(parameters.get("geometry.width") or 0.37)

    scale = 1.0
    if top in ("cedar",):
        scale *= 0.94
    elif top in ("maple",):
        scale *= 1.06
    if back in ("rosewood",):
        scale *= 1.03
    scale *= 1.0 + (top_t - 0.003) * 45.0
    scale *= 1.0 + (back_t - 0.0033) * 35.0
    scale *= 1.0 + (depth - 0.1) * 1.8
    scale *= 1.0 + (width - 0.37) * 0.9
    if float(mode_hz) > 300.0:
        scale *= 1.0 + 0.04 * s
    # Blend toward unity
    return 1.0 + (scale - 1.0) * s


def compute_note_reward_score(
    *,
    frequency_hz: float,
    body_rms_before_mix: float,
    string_rms_before_mix: float,
    body_to_string_ratio_before_loudness: float,
    top_contributing_modes: Sequence[Mapping[str, Any]],
    late_to_early_rms_db: float,
    output_decay_slope_db_per_s: float,
    broad_body_energy_fraction: float = 0.0,
    near_modal_energy_fraction: float = 0.0,
    final_rms_dbfs: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Estimate how strongly the body supports a note (not final loudness alone).

    Higher score ⇒ more body projection / resonance support for this pitch.
    """
    f0 = max(40.0, float(frequency_hz))
    harmonics = [f0 * k for k in range(1, 6)]
    harmonic_support = 0.0
    broad_support = 0.0
    for row in top_contributing_modes:
        f_m = float(row.get("mode_frequency_hz") or row.get("frequency_hz") or 0.0)
        w = float(row.get("contribution_weight") or 0.0)
        if f_m <= 0 or w <= 0:
            continue
        nearest = min(harmonics, key=lambda h: abs(h - f_m))
        rel = abs(nearest - f_m) / max(nearest, 1.0)
        if rel <= 0.10:
            harmonic_support += w
        elif rel <= 0.35:
            broad_support += w * 0.45
    harmonic_norm = math.log1p(harmonic_support * 900.0)
    broad_norm = math.log1p(broad_support * 500.0 + broad_body_energy_fraction * 2.5)

    body_over_string = body_rms_before_mix / max(string_rms_before_mix, 1e-12)
    ratio_term = math.log(max(body_to_string_ratio_before_loudness, 1e-6))
    body_projection = math.log(max(body_over_string, 1e-6))
    sustain_term = max(0.0, min(1.0, (-float(late_to_early_rms_db) - 3.0) / 18.0))
    decay_term = max(0.0, min(1.0, (-float(output_decay_slope_db_per_s) - 5.0) / 35.0))
    near_term = max(0.0, min(1.0, near_modal_energy_fraction))

    score = (
        0.22 * harmonic_norm
        + 0.18 * broad_norm
        + 0.18 * body_projection
        + 0.14 * ratio_term
        + 0.12 * sustain_term
        + 0.08 * decay_term
        + 0.08 * near_term
    )
    loudness_penalty = 0.0
    if final_rms_dbfs is not None:
        loudness_penalty = max(0.0, (float(final_rms_dbfs) + 16.0) * 0.02)
        score -= loudness_penalty

    return {
        "note_reward_score": round(score, 6),
        "note_reward_harmonic_support": round(harmonic_norm, 6),
        "note_reward_broad_support": round(broad_norm, 6),
        "note_reward_body_projection": round(body_projection, 6),
        "note_reward_body_string_ratio_term": round(ratio_term, 6),
        "note_reward_sustain_term": round(sustain_term, 6),
        "note_reward_decay_term": round(decay_term, 6),
        "note_reward_near_modal_term": round(near_term, 6),
        "note_reward_loudness_penalty": round(loudness_penalty, 6),
        "note_reward_formula": (
            "0.22*harmonic+0.18*broad+0.18*body_proj+0.14*ratio+0.12*sustain"
            "+0.08*decay+0.08*near_modal-loudness_penalty"
        ),
    }


def _spectral_features(audio: np.ndarray, sample_rate: int) -> Dict[str, float]:
    x = np.asarray(audio, dtype=np.float64)
    if x.size < 8:
        return {"centroid_hz": 0.0, "low_energy": 0.0, "mid_energy": 0.0, "high_energy": 0.0}
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(x.size, d=1.0 / float(sample_rate))
    power = spec**2
    total = float(np.sum(power)) + 1e-18
    centroid = float(np.sum(freqs * power) / total)
    low = float(np.sum(power[(freqs >= 60) & (freqs < 250)])) / total
    mid = float(np.sum(power[(freqs >= 250) & (freqs < 1200)])) / total
    high = float(np.sum(power[freqs >= 1200])) / total
    return {
        "centroid_hz": centroid,
        "low_energy": low,
        "mid_energy": mid,
        "high_energy": high,
    }


def _magnitude_profile(audio: np.ndarray, *, bins: int = 48) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float64)
    if x.size < bins:
        return np.zeros(bins, dtype=np.float64)
    mag = np.abs(np.fft.rfft(x))
    if mag.size < bins:
        return np.pad(mag, (0, bins - mag.size))
    step = max(mag.size // bins, 1)
    chunks = [float(np.mean(mag[i : i + step])) for i in range(0, bins * step, step)]
    return np.asarray(chunks[:bins], dtype=np.float64)


def average_spectral_similarity(audios: Sequence[np.ndarray]) -> float:
    """Mean pairwise cosine similarity of coarse magnitude profiles (1.0 = identical)."""
    profiles = [_magnitude_profile(a) for a in audios if a is not None and len(a) > 0]
    if len(profiles) < 2:
        return 1.0
    sims: List[float] = []
    for i in range(len(profiles)):
        for j in range(i + 1, len(profiles)):
            a = profiles[i]
            b = profiles[j]
            denom = float(np.linalg.norm(a) * np.linalg.norm(b))
            if denom <= 1e-18:
                continue
            sims.append(float(np.dot(a, b) / denom))
    return round(sum(sims) / max(len(sims), 1), 6)


def summarize_comparison_note(segments: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Aggregate spread metrics across guitar segments for one note."""
    rms_vals = [float(s.get("final_rms_dbfs") or 0.0) for s in segments]
    raw_body = [float(s.get("raw_body_rms_before_normalization") or 0.0) for s in segments]
    rewards = [float(s.get("note_reward_score") or 0.0) for s in segments]
    body_ratios = [float(s.get("body_to_string_ratio") or 0.0) for s in segments]
    centroids = [float(s.get("spectral_centroid_hz") or 0.0) for s in segments]
    decays = [float(s.get("output_decay_slope_db_per_s") or 0.0) for s in segments]
    q_medians = [float(s.get("mode_q_median") or 0.0) for s in segments if s.get("mode_q_median") is not None]
    q_fingerprints = [
        float(s.get("sample_mode_q_fingerprint") or 0.0)
        for s in segments
        if s.get("sample_mode_q_fingerprint") is not None
    ]
    mat_medians = [
        float(s.get("material_damping_median") or 0.0) for s in segments if s.get("material_damping_median") is not None
    ]
    mat_fingerprints = [
        float(s.get("sample_material_damping_fingerprint") or 0.0)
        for s in segments
        if s.get("sample_material_damping_fingerprint") is not None
    ]
    q_spreads_within = [
        float(s.get("mode_q_spread_within_sample") or (s.get("damping_q_summary") or {}).get("mode_q_spread") or 0.0)
        for s in segments
    ]
    mat_spreads_within = [
        float(
            s.get("material_damping_spread_within_sample")
            or (s.get("damping_q_summary") or {}).get("material_damping_spread")
            or 0.0
        )
        for s in segments
    ]
    near_fracs = [float(s.get("near_modal_energy_fraction") or 0.0) for s in segments]
    far_fracs = [float(s.get("broad_body_energy_fraction") or 0.0) for s in segments]
    sims = [float(s.get("segment_spectral_similarity_baseline") or 1.0) for s in segments]
    ranked = sorted(segments, key=lambda s: float(s.get("note_reward_score") or 0.0), reverse=True)
    return {
        "rms_spread_db": round(max(rms_vals) - min(rms_vals), 4) if rms_vals else 0.0,
        "raw_body_rms_spread": round(max(raw_body) - min(raw_body), 8) if raw_body else 0.0,
        "body_to_string_ratio_spread": round(max(body_ratios) - min(body_ratios), 6) if body_ratios else 0.0,
        "spectral_centroid_spread_hz": round(max(centroids) - min(centroids), 4) if centroids else 0.0,
        "decay_slope_spread_db_per_s": round(max(decays) - min(decays), 4) if decays else 0.0,
        "mode_q_spread_mean": round(max(q_fingerprints or q_medians) - min(q_fingerprints or q_medians), 6)
        if len(q_fingerprints or q_medians) >= 2
        else 0.0,
        "cross_sample_mode_q_median_spread": round(max(q_medians) - min(q_medians), 6)
        if len(q_medians) >= 2
        else 0.0,
        "cross_sample_mode_q_fingerprint_spread": round(max(q_fingerprints) - min(q_fingerprints), 6)
        if len(q_fingerprints) >= 2
        else 0.0,
        "material_damping_spread_mean": round(
            max(mat_fingerprints or mat_medians) - min(mat_fingerprints or mat_medians), 6
        )
        if len(mat_fingerprints or mat_medians) >= 2
        else 0.0,
        "cross_sample_material_damping_median_spread": round(
            max(mat_fingerprints or mat_medians) - min(mat_fingerprints or mat_medians), 6
        )
        if len(mat_fingerprints or mat_medians) >= 2
        else 0.0,
        "cross_sample_material_fingerprint_spread": round(max(mat_fingerprints) - min(mat_fingerprints), 6)
        if len(mat_fingerprints) >= 2
        else 0.0,
        "within_sample_mode_q_spread_mean": round(sum(q_spreads_within) / max(len(q_spreads_within), 1), 4)
        if q_spreads_within
        else 0.0,
        "within_sample_material_damping_spread_mean": round(
            sum(mat_spreads_within) / max(len(mat_spreads_within), 1), 6
        )
        if mat_spreads_within
        else 0.0,
        "near_modal_energy_fraction_mean": round(sum(near_fracs) / max(len(near_fracs), 1), 4)
        if near_fracs
        else 0.0,
        "far_broad_energy_fraction_mean": round(sum(far_fracs) / max(len(far_fracs), 1), 4) if far_fracs else 0.0,
        "note_reward_spread": round(max(rewards) - min(rewards), 6) if rewards else 0.0,
        "top_note_reward_sample_ids": [r.get("sample_id") for r in ranked[:3]],
        "bottom_note_reward_sample_ids": [r.get("sample_id") for r in ranked[-3:]],
    }


def summarize_diagnostic_mode(
    note_rows: Sequence[Mapping[str, Any]],
    *,
    diagnostic_mode: str,
) -> Dict[str, Any]:
    """Per-mode summary across all comparison notes."""
    per_note: Dict[str, Any] = {}
    all_centroids: List[float] = []
    all_rms: List[float] = []
    all_rewards: List[float] = []
    for row in note_rows:
        note = str(row.get("note") or "")
        segs = list(row.get("segments") or [])
        note_summary = summarize_comparison_note(segs)
        centroids = [float(s.get("spectral_centroid_hz") or 0.0) for s in segs]
        if centroids:
            note_summary["centroid_spread_hz"] = round(max(centroids) - min(centroids), 4)
            all_centroids.extend(centroids)
        lows = [float(s.get("spectral_low_energy") or 0.0) for s in segs]
        mids = [float(s.get("spectral_mid_energy") or 0.0) for s in segs]
        highs = [float(s.get("spectral_high_energy") or 0.0) for s in segs]
        if lows:
            note_summary["low_energy_spread"] = round(max(lows) - min(lows), 6)
            note_summary["mid_energy_spread"] = round(max(mids) - min(mids), 6)
            note_summary["high_energy_spread"] = round(max(highs) - min(highs), 6)
        note_summary["average_spectral_similarity"] = float(row.get("average_spectral_similarity") or 1.0)
        note_summary["spectral_differentiation"] = round(1.0 - note_summary["average_spectral_similarity"], 6)
        per_note[note] = note_summary
        all_rms.extend(float(s.get("final_rms_dbfs") or 0.0) for s in segs)
        all_rewards.extend(float(s.get("note_reward_score") or 0.0) for s in segs)

    return {
        "diagnostic_mode": diagnostic_mode,
        "notes": per_note,
        "overall_rms_spread_db": round(max(all_rms) - min(all_rms), 4) if all_rms else 0.0,
        "overall_note_reward_spread": round(max(all_rewards) - min(all_rewards), 6) if all_rewards else 0.0,
        "overall_centroid_spread_hz": round(max(all_centroids) - min(all_centroids), 4) if all_centroids else 0.0,
    }


def compare_mode_summaries(
    summaries: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Contrast baseline vs structural (or any two modes) on differentiation metrics."""
    modes = list(summaries.keys())
    out: Dict[str, Any] = {"modes": modes, "contrasts": {}}
    if len(modes) < 2:
        return out
    base = summaries.get("baseline_current") or summaries[modes[0]]
    other_name = "modal_damping_body_signature_v1" if "modal_damping_body_signature_v1" in summaries else modes[-1]
    other = summaries.get(other_name) or summaries[modes[-1]]
    for key in (
        "overall_rms_spread_db",
        "overall_note_reward_spread",
        "overall_centroid_spread_hz",
    ):
        b = float(base.get(key) or 0.0)
        o = float(other.get(key) or 0.0)
        out["contrasts"][key] = {
            "baseline_current": b,
            other_name: o,
            "delta": round(o - b, 6),
        }
    return out


def flatten_geometry_parameters(parameters: Mapping[str, Any]) -> Dict[str, Any]:
    out = {
        "top_wood_id": parameters.get("top_wood_id"),
        "back_wood_id": parameters.get("back_wood_id"),
    }
    geom = parameters.get("geometry") or {}
    if isinstance(geom, Mapping):
        for key in ("length", "width", "depth", "top_thickness", "back_thickness", "hole_radius"):
            if key in geom:
                out[f"geometry.{key}"] = geom.get(key)
    for key in ("length", "width", "depth", "top_thickness", "back_thickness", "hole_radius"):
        if key in parameters and f"geometry.{key}" not in out:
            out[f"geometry.{key}"] = parameters.get(key)
    return out
