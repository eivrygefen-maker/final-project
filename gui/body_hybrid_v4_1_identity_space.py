#!/usr/bin/env python3
"""
Stage 5.1C — bounded continuous body-identity layer on top of V4.1 (diagnostic only).

y_final = y_v4_1 + epsilon * residual, with harmonic shaping on harmonics 2–8.
Does not replace or alter V4.1 endpoint delegation.
"""
from __future__ import annotations

import json
import math
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from body_hybrid_v4_1 import synthesize_hybrid_v4_1_note
from body_response_synth import ModalInput, _rms, read_wav_float_mono
from bridge_mobility_proxy import compute_body_mass_proxies, effective_modal_mass_proxy
from modal_damping import WOOD_DAMPING_COEFF, compute_per_mode_damping, normalize_participation_shares
from sample_parameters import normalize_sample_parameters

IDENTITY_SPACE_MODES: Tuple[str, ...] = (
    "modal_body_hybrid_v4_1_identity_space",
    "stk_body_transfer_v4_1_identity_space",
    "modal_body_hybrid_v4_1_identity_light",
    "modal_body_hybrid_v4_1_identity_medium",
    "modal_body_hybrid_v4_1_identity_strong",
    "stk_body_transfer_v4_1_identity_sweep",
    "modal_body_hybrid_v4_1_identity_contrast",
    "stk_body_transfer_v4_1_identity_contrast",
    "modal_body_hybrid_v4_1_identity_contrast_medium",
    "modal_body_hybrid_v4_1_identity_contrast_strong",
    "modal_body_hybrid_v4_1_identity_contrast_hybrid",
    "stk_body_transfer_v4_1_identity_contrast_hybrid",
    "modal_body_hybrid_v4_1_identity_contrast_hybrid_25_75",
    "modal_body_hybrid_v4_1_identity_contrast_hybrid_40_60",
    "modal_body_hybrid_v4_1_identity_contrast_hybrid_50_50",
    "modal_body_hybrid_v4_1_identity_contrast_g_20_80",
    "modal_body_hybrid_v4_1_identity_contrast_g_25_75",
    "modal_body_hybrid_v4_1_identity_contrast_g_30_70",
    "modal_body_hybrid_v4_1_identity_contrast_g_25_75_decay",
    "modal_body_hybrid_v4_1_identity_contrast_g_25_75_bridge",
    "modal_body_hybrid_v4_1_identity_contrast_g_25_75_full",
    "stk_body_transfer_v4_1_identity_contrast_g",
    # Stage 5.1H — current default STK body/identity candidate (alias → g_30_70)
    "stk_body_transfer_final_v1",
    "modal_body_hybrid_v4_1_identity_contrast_g_30_70_de_thump",
    "stk_body_transfer_final_v1_de_thump_candidate",
)

# Stage 5.1H — frozen final STK candidate (GUI/ROM default; canonical diagnostic name below)
STK_BODY_TRANSFER_FINAL_V1 = "stk_body_transfer_final_v1"
STK_FINAL_CANDIDATE_CANONICAL = "modal_body_hybrid_v4_1_identity_contrast_g_30_70"
STK_FINAL_GUI_LABEL = "Physical Body Identity v1"
STK_FINAL_DE_THUMP_CANONICAL = "modal_body_hybrid_v4_1_identity_contrast_g_30_70_de_thump"
STK_BODY_TRANSFER_FINAL_V1_DE_THUMP = "stk_body_transfer_final_v1_de_thump_candidate"

DE_THUMP_ONSET_MS = 60.0
DE_THUMP_BLEND_MS = 30.0
DE_THUMP_HP_HZ = 100.0
DE_THUMP_LF_ATTENUATION_DB = 6.0
DE_THUMP_ATTACK_SCALE = 0.72

DZ_BODY_CLIP = 2.5
IQR_FLOOR = 0.15
HYBRID_RMS_GUARD_DB = 2.75
HYBRID_AUDIBILITY_MIN_DB = -40.0
HYBRID_AUDIBILITY_TARGET_DB = -26.0
HYBRID_AUDIBILITY_MAX_DB = -18.0
G_DECAY_STRENGTH = 0.22
G_BRIDGE_BLEND = 0.65
G_BRIDGE_FUNDAMENTAL_MAX = 0.035
G_BRIDGE_HARMONIC_MAX = 0.35

HYBRID_BLEND_RATIOS: Dict[str, Tuple[float, float]] = {
    "25_75": (0.25, 0.75),
    "40_60": (0.40, 0.60),
    "50_50": (0.50, 0.50),
}

G_BLEND_RATIOS: Dict[str, Tuple[float, float]] = {
    "20_80": (0.20, 0.80),
    "25_75": (0.25, 0.75),
    "30_70": (0.30, 0.70),
}

# Default/light bounds (Stage 5.1C baseline — diagnostic only)
IDENTITY_EPSILON = 0.18
HARMONIC_GAIN_MAX = 0.12
FUNDAMENTAL_GAIN_MAX = 0.025
RESIDUAL_GAIN_MAX = 0.08
RMS_GUARD_MAX_DB = 1.5
PEAK_CLIP_DBFS = -0.5

PERCEPTUAL_AXIS_NAMES: Tuple[str, ...] = (
    "brightness_centroid",
    "low_mid_warmth",
    "high_freq_rolloff",
    "attack_bloom",
    "decay_sustain",
    "body_resonance_density",
)


@dataclass(frozen=True)
class IdentityStrengthProfile:
    name: str
    identity_epsilon: float
    harmonic_gain_max: float
    fundamental_gain_max: float
    residual_gain_max: float
    rms_guard_max_db: float
    axis_gain_scale: float
    band_eq_max_db: float


STRENGTH_PROFILES: Dict[str, IdentityStrengthProfile] = {
    "light": IdentityStrengthProfile(
        name="light",
        identity_epsilon=0.18,
        harmonic_gain_max=0.12,
        fundamental_gain_max=0.025,
        residual_gain_max=0.08,
        rms_guard_max_db=1.5,
        axis_gain_scale=1.0,
        band_eq_max_db=0.35,
    ),
    "medium": IdentityStrengthProfile(
        name="medium",
        identity_epsilon=0.35,
        harmonic_gain_max=0.20,
        fundamental_gain_max=0.035,
        residual_gain_max=0.14,
        rms_guard_max_db=2.0,
        axis_gain_scale=1.75,
        band_eq_max_db=0.85,
    ),
    "strong": IdentityStrengthProfile(
        name="strong",
        identity_epsilon=0.55,
        harmonic_gain_max=0.30,
        fundamental_gain_max=0.04,
        residual_gain_max=0.22,
        rms_guard_max_db=2.5,
        axis_gain_scale=2.6,
        band_eq_max_db=1.35,
    ),
    "contrast_medium": IdentityStrengthProfile(
        name="contrast_medium",
        identity_epsilon=0.45,
        harmonic_gain_max=0.25,
        fundamental_gain_max=0.04,
        residual_gain_max=0.18,
        rms_guard_max_db=2.5,
        axis_gain_scale=2.6,
        band_eq_max_db=1.2,
    ),
    "contrast_strong": IdentityStrengthProfile(
        name="contrast_strong",
        identity_epsilon=0.65,
        harmonic_gain_max=0.35,
        fundamental_gain_max=0.04,
        residual_gain_max=0.25,
        rms_guard_max_db=3.0,
        axis_gain_scale=3.0,
        band_eq_max_db=1.8,
    ),
}

MODE_TO_STRENGTH: Dict[str, str] = {
    "modal_body_hybrid_v4_1_identity_space": "light",
    "stk_body_transfer_v4_1_identity_space": "light",
    "modal_body_hybrid_v4_1_identity_light": "light",
    "modal_body_hybrid_v4_1_identity_medium": "medium",
    "modal_body_hybrid_v4_1_identity_strong": "strong",
    "stk_body_transfer_v4_1_identity_sweep": "medium",
    "modal_body_hybrid_v4_1_identity_contrast": "contrast_medium",
    "stk_body_transfer_v4_1_identity_contrast": "contrast_medium",
    "modal_body_hybrid_v4_1_identity_contrast_medium": "contrast_medium",
    "modal_body_hybrid_v4_1_identity_contrast_strong": "contrast_strong",
}


def strength_profile_for_mode(mode_name: Optional[str]) -> Optional[IdentityStrengthProfile]:
    key = MODE_TO_STRENGTH.get(str(mode_name or ""))
    if not key:
        return None
    return STRENGTH_PROFILES[key]


def is_contrast_identity_mode(mode_name: Optional[str]) -> bool:
    m = str(mode_name or "")
    return "identity_contrast" in m and "identity_contrast_hybrid" not in m


def is_hybrid_identity_mode(mode_name: Optional[str]) -> bool:
    return "identity_contrast_hybrid" in str(mode_name or "")


def is_g_identity_mode(mode_name: Optional[str]) -> bool:
    m = str(mode_name or "")
    return (
        "identity_contrast_g_" in m
        or m.endswith("identity_contrast_g")
        or m == STK_BODY_TRANSFER_FINAL_V1
        or m == STK_BODY_TRANSFER_FINAL_V1_DE_THUMP
    )


def is_de_thump_mode(mode_name: Optional[str]) -> bool:
    m = str(mode_name or "")
    return m.endswith("_de_thump") or m.endswith("_de_thump_candidate")


def is_final_stk_candidate_mode(mode_name: Optional[str]) -> bool:
    m = str(mode_name or "")
    return m in (
        STK_BODY_TRANSFER_FINAL_V1,
        STK_FINAL_CANDIDATE_CANONICAL,
        STK_BODY_TRANSFER_FINAL_V1_DE_THUMP,
        STK_FINAL_DE_THUMP_CANONICAL,
    )


def canonical_stk_final_mode(mode_name: Optional[str]) -> str:
    """Resolve final alias to canonical diagnostic mode name."""
    m = str(mode_name or "")
    if m == STK_BODY_TRANSFER_FINAL_V1:
        return STK_FINAL_CANDIDATE_CANONICAL
    if m == STK_BODY_TRANSFER_FINAL_V1_DE_THUMP:
        return STK_FINAL_DE_THUMP_CANONICAL
    return m


def requires_identity_contrast_context(mode_name: Optional[str]) -> bool:
    return (
        is_contrast_identity_mode(mode_name)
        or is_hybrid_identity_mode(mode_name)
        or is_g_identity_mode(mode_name)
    )


def hybrid_blend_for_mode(mode_name: Optional[str]) -> Tuple[float, float]:
    """Return (absolute_weight, contrast_weight) for hybrid modes."""
    m = str(mode_name or "")
    for tag, (a, b) in HYBRID_BLEND_RATIOS.items():
        if m.endswith(tag) or f"hybrid_{tag}" in m:
            return a, b
    if is_hybrid_identity_mode(m):
        return HYBRID_BLEND_RATIOS["40_60"]
    return 0.40, 0.60


def g_config_for_mode(mode_name: Optional[str]) -> Dict[str, Any]:
    """Stage 5.1G: blend ratio + optional decay/bridge physical components."""
    m = str(mode_name or "")
    if m == STK_BODY_TRANSFER_FINAL_V1:
        return {
            "absolute_weight": G_BLEND_RATIOS["30_70"][0],
            "contrast_weight": G_BLEND_RATIOS["30_70"][1],
            "decay_active": False,
            "bridge_active": False,
            "de_thump_active": False,
            "canonical_mode": STK_FINAL_CANDIDATE_CANONICAL,
            "final_alias": True,
        }
    if m == STK_BODY_TRANSFER_FINAL_V1_DE_THUMP:
        return {
            "absolute_weight": G_BLEND_RATIOS["30_70"][0],
            "contrast_weight": G_BLEND_RATIOS["30_70"][1],
            "decay_active": False,
            "bridge_active": False,
            "de_thump_active": True,
            "canonical_mode": STK_FINAL_DE_THUMP_CANONICAL,
            "final_alias": True,
        }
    if m.endswith("_de_thump") or m.endswith("_de_thump_candidate"):
        abs_w, contrast_w = G_BLEND_RATIOS["30_70"]
        for tag, (a, b) in G_BLEND_RATIOS.items():
            if f"g_{tag}" in m:
                abs_w, contrast_w = a, b
                break
        return {
            "absolute_weight": abs_w,
            "contrast_weight": contrast_w,
            "decay_active": False,
            "bridge_active": False,
            "de_thump_active": True,
        }
    if m == "stk_body_transfer_v4_1_identity_contrast_g":
        return {
            "absolute_weight": 0.25,
            "contrast_weight": 0.75,
            "decay_active": True,
            "bridge_active": True,
        }
    use_decay = "_decay" in m or "_full" in m
    use_bridge = "_bridge" in m or "_full" in m
    abs_w, contrast_w = G_BLEND_RATIOS["25_75"]
    for tag, (a, b) in G_BLEND_RATIOS.items():
        if f"g_{tag}" in m:
            abs_w, contrast_w = a, b
            break
    return {
        "absolute_weight": abs_w,
        "contrast_weight": contrast_w,
        "decay_active": use_decay,
        "bridge_active": use_bridge,
        "de_thump_active": False,
    }


def apply_residual_de_thump(
    residual: np.ndarray,
    *,
    sample_rate: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Attenuate artificial low thump on identity residual onset (30–80 ms), preserve sustain."""
    r = np.asarray(residual, dtype=np.float64)
    n = len(r)
    sr = float(sample_rate)
    if n < 32:
        return r.copy(), {"de_thump_active": False}

    onset_end = int(DE_THUMP_ONSET_MS * 1e-3 * sr)
    blend_end = min(n, onset_end + int(DE_THUMP_BLEND_MS * 1e-3 * sr))
    onset_end = min(onset_end, n)

    onset = r[:onset_end].copy()
    if len(onset) >= 8:
        spec = np.fft.rfft(onset)
        freqs = np.fft.rfftfreq(len(onset), d=1.0 / sr)
        hp = float(DE_THUMP_HP_HZ)
        for i, f_hz in enumerate(freqs):
            if f_hz < hp:
                att_db = DE_THUMP_LF_ATTENUATION_DB * (1.0 - f_hz / max(hp, 1e-6))
                spec[i] *= 10.0 ** (att_db / 20.0)
        onset = np.fft.irfft(spec, n=len(onset))
        onset = onset * float(DE_THUMP_ATTACK_SCALE)

    out = r.copy()
    out[:onset_end] = onset
    if blend_end > onset_end:
        fade = np.linspace(1.0, 0.0, blend_end - onset_end)
        out[onset_end:blend_end] = (
            onset[-len(fade) :] * fade + r[onset_end:blend_end] * (1.0 - fade)
        )

    return out, {
        "de_thump_active": True,
        "de_thump_onset_ms": DE_THUMP_ONSET_MS,
        "de_thump_hp_hz": DE_THUMP_HP_HZ,
        "de_thump_attack_scale": DE_THUMP_ATTACK_SCALE,
    }


def compute_decay_axis(z_body: Mapping[str, Any]) -> float:
    """Physical decay axis from Q/damping/bridge/modal density/participation."""
    f = z_body.get("features") or {}
    near_harm = (
        _feat(f, "modal_near_60_120")
        + _feat(f, "modal_near_120_180")
        + _feat(f, "modal_near_180_280")
    ) / 3.0
    mat_damp = 0.5 * (_feat(f, "mat_top_damping") + _feat(f, "mat_back_damping"))
    participation = (
        _feat(f, "share_back_mean") + _feat(f, "share_top_mean") + _feat(f, "share_air_mean")
    ) / 3.0
    raw = (
        0.35 * _feat(f, "q_spread")
        - 0.25 * mat_damp
        + 0.25 * _feat(f, "bridge_mobility")
        + 0.30 * near_harm
        + 0.20 * participation
    )
    return max(-1.0, min(1.0, raw))


def compute_bridge_axis(z_body: Mapping[str, Any]) -> float:
    """Body-aware bridge energy-transfer axis (no raw string/radiation gain)."""
    f = z_body.get("features") or {}
    near_harm = (
        _feat(f, "modal_near_60_120")
        + _feat(f, "modal_near_120_180")
        + _feat(f, "modal_near_180_280")
    ) / 3.0
    inv_mass = -_feat(f, "eff_mass_median")
    raw = (
        0.30 * _feat(f, "bridge_rank_median")
        + 0.25 * _feat(f, "bridge_mobility")
        + 0.20 * inv_mass
        + 0.25 * near_harm
        + 0.15 * 0.5 * (_feat(f, "share_top_mean") + _feat(f, "share_back_mean"))
        + 0.10 * 0.5 * (_feat(f, "rad_rank_median") + _feat(f, "mic_rank_median"))
    )
    return max(-1.0, min(1.0, raw))


def compute_bridge_harmonic_gains(bridge_axis: float) -> List[float]:
    """Harmonic gains for bridge coupling — h2–h5 dominant, fundamental minimal."""
    scale = max(-1.0, min(1.0, float(bridge_axis)))
    coeffs = [0.12, 0.85, 0.80, 0.75, 0.70, 0.45, 0.35, 0.30]
    gains: List[float] = []
    for k, coeff in enumerate(coeffs, start=1):
        cap = G_BRIDGE_FUNDAMENTAL_MAX if k == 1 else G_BRIDGE_HARMONIC_MAX
        val = scale * coeff * G_BRIDGE_HARMONIC_MAX
        gains.append(round(max(-cap, min(cap, val)), 6))
    return gains


def apply_decay_differentiation_to_residual(
    residual: np.ndarray,
    z_body: Mapping[str, Any],
    *,
    sample_rate: int,
    decay_strength: float = G_DECAY_STRENGTH,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Gentle time-envelope tilt on body residual only (early bloom / late sustain)."""
    del sample_rate  # envelope is normalized-time based
    r = np.asarray(residual, dtype=np.float64)
    n = len(r)
    if n < 8:
        return r.copy(), {"decay_axis": 0.0, "decay_active": False}
    decay_axis = compute_decay_axis(z_body)
    f = z_body.get("features") or {}
    early_gain = 1.0 + 0.06 * _feat(f, "bridge_mobility")
    late_gain = math.exp(decay_axis * decay_strength * 0.35)
    t = np.linspace(0.0, 1.0, n)
    env = early_gain * np.exp(-3.0 * t) + late_gain * (1.0 - np.exp(-3.0 * t))
    env = env / max(float(np.sqrt(np.mean(env**2))), 1e-9)
    shaped = r * env
    return shaped, {
        "decay_axis": round(decay_axis, 6),
        "decay_strength": decay_strength,
        "early_gain": round(early_gain, 6),
        "late_gain": round(late_gain, 6),
        "decay_active": True,
    }


def apply_bridge_coupling_to_residual(
    residual: np.ndarray,
    *,
    frequency_hz: float,
    sample_rate: int,
    z_body: Mapping[str, Any],
    blend: float = G_BRIDGE_BLEND,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Harmonic-dependent body coupling on residual — same string, guitar-dependent transfer."""
    r = np.asarray(residual, dtype=np.float64)
    bridge_axis = compute_bridge_axis(z_body)
    h_gains = compute_bridge_harmonic_gains(bridge_axis)
    if float(np.max(np.abs(r))) < 1e-15:
        return r.copy(), {"bridge_axis": round(bridge_axis, 6), "bridge_active": False}
    shaped = apply_harmonic_identity_shaping(
        r,
        frequency_hz=frequency_hz,
        sample_rate=sample_rate,
        harmonic_gains=h_gains,
    )
    alpha = max(0.0, min(1.0, float(blend)))
    out = (1.0 - alpha) * r + alpha * shaped
    return out, {
        "bridge_axis": round(bridge_axis, 6),
        "bridge_harmonic_gains": h_gains,
        "bridge_blend": alpha,
        "bridge_active": True,
    }


def compose_hybrid_contrast_residual(
    base_audio: np.ndarray,
    *,
    frequency_hz: float,
    sample_rate: int,
    z_body: Mapping[str, Any],
    dz_body: Mapping[str, Any],
    abs_w: float,
    contrast_w: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Blend identity_strong + contrast_strong layer residuals (5.1F/G core)."""
    strong_prof = STRENGTH_PROFILES["strong"]
    contrast_prof = STRENGTH_PROFILES["contrast_strong"]
    res_abs, meta_abs = compute_identity_layer_residual(
        base_audio,
        frequency_hz=frequency_hz,
        sample_rate=sample_rate,
        feature_source=z_body,
        profile=strong_prof,
        contrast=False,
    )
    res_contrast, meta_contrast = compute_identity_layer_residual(
        base_audio,
        frequency_hz=frequency_hz,
        sample_rate=sample_rate,
        feature_source=dz_body,
        profile=contrast_prof,
        contrast=True,
    )
    combined = abs_w * res_abs + contrast_w * res_contrast
    peak_base = float(np.max(np.abs(base_audio)) + 1e-12)
    cap = max(strong_prof.residual_gain_max, contrast_prof.residual_gain_max) * peak_base
    combined = np.clip(combined, -cap, cap)
    return combined, {
        "hybrid_blend_absolute": abs_w,
        "hybrid_blend_contrast": contrast_w,
        "absolute_layer": meta_abs,
        "contrast_layer": meta_contrast,
        "residual_cap": cap,
    }


def is_v4_1_identity_space_mode(mode_name: Optional[str]) -> bool:
    return str(mode_name or "") in IDENTITY_SPACE_MODES


def _feat(feats: Mapping[str, float], key: str, default: float = 0.0) -> float:
    return float(feats.get(key, default))


def compute_perceptual_axes(
    z_body: Mapping[str, Any],
    *,
    contrast: bool = False,
) -> Dict[str, float]:
    """Bounded projection from z_body (or dz_body) to six perceptual timbre axes."""
    f = z_body.get("features") or {}

    def _blend(keys: Sequence[str], weights: Optional[Sequence[float]] = None) -> float:
        ws = weights or [1.0] * len(keys)
        if contrast:
            num = sum((_feat(f, k) / DZ_BODY_CLIP) * w for k, w in zip(keys, ws))
        else:
            num = sum(_feat(f, k) * w for k, w in zip(keys, ws))
        den = sum(abs(w) for w in ws) or 1.0
        return max(-1.0, min(1.0, num / den))

    axes = {
        "brightness_centroid": _blend(
            ("high_body_color", "share_air_mean", "geom_hole_to_area", "mat_top_damping"),
            (1.1, 0.7, 0.5, -0.35),
        ),
        "low_mid_warmth": _blend(
            ("low_body_color", "mid_body_color", "share_back_mean", "geom_depth", "geom_air_volume"),
            (1.0, 0.8, 0.9, 0.6, 0.5),
        ),
        "high_freq_rolloff": _blend(
            ("q_fingerprint", "high_body_color", "mid_body_color", "geom_top_thickness"),
            (-0.9, 0.6, -0.5, 0.45),
        ),
        "attack_bloom": _blend(
            ("bridge_mobility", "bridge_rank_median", "eff_mass_median", "mass_mixed"),
            (1.0, 0.8, -0.7, -0.4),
        ),
        "decay_sustain": _blend(
            ("q_spread", "q_fingerprint", "mat_top_damping", "mat_back_damping"),
            (0.9, -0.8, 0.5, 0.5),
        ),
        "body_resonance_density": _blend(
            (
                "modal_density_60_120",
                "modal_density_120_180",
                "modal_density_180_280",
                "modal_near_60_120",
                "modal_near_180_280",
                "rad_rank_median",
            ),
            (1.0, 0.9, 0.8, 0.7, 0.6, 0.5),
        ),
    }
    return {k: round(v, 6) for k, v in axes.items()}


def compute_robust_identity_reference(
    z_bodies: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Robust batch reference: per-feature median and IQR across sample set."""
    keys = sorted({k for z in z_bodies for k in (z.get("features") or {}).keys()})
    median_features: Dict[str, float] = {}
    iqr_features: Dict[str, float] = {}
    for key in keys:
        vals = np.asarray(
            [float((z.get("features") or {}).get(key, 0.0)) for z in z_bodies],
            dtype=np.float64,
        )
        if len(vals) == 0:
            continue
        q25, q50, q75 = np.percentile(vals, [25, 50, 75])
        median_features[key] = round(float(q50), 6)
        iqr_features[key] = round(max(float(q75 - q25), IQR_FLOOR), 6)
    return {
        "median_features": median_features,
        "iqr_features": iqr_features,
        "field_names": keys,
        "sample_count": len(z_bodies),
    }


def compute_dz_body(
    z_body: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    clip: float = DZ_BODY_CLIP,
) -> Dict[str, Any]:
    """Sample-relative contrast vector: robust z-score clipped to ±clip."""
    feats = z_body.get("features") or {}
    med = reference.get("median_features") or {}
    iqr = reference.get("iqr_features") or {}
    keys = sorted(set(feats.keys()) | set(med.keys()))
    dz_feats: Dict[str, float] = {}
    for key in keys:
        num = float(feats.get(key, 0.0)) - float(med.get(key, 0.0))
        den = float(iqr.get(key, IQR_FLOOR))
        raw = num / den
        dz_feats[key] = round(max(-clip, min(clip, raw)), 6)
    field_names = sorted(dz_feats.keys())
    return {
        "features": dz_feats,
        "vector": [dz_feats[k] for k in field_names],
        "field_names": field_names,
        "clip_limit": clip,
    }


def build_batch_contrast_context(
    z_bodies_by_sample: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Per-sample dz_body + shared z_ref for a note batch."""
    z_list = list(z_bodies_by_sample.values())
    if not z_list:
        return {}
    z_ref = compute_robust_identity_reference(z_list)
    out: Dict[str, Dict[str, Any]] = {}
    for sid, z_body in z_bodies_by_sample.items():
        dz_body = compute_dz_body(z_body, z_ref)
        out[str(sid)] = {
            "z_ref": z_ref,
            "z_body": z_body,
            "dz_body": dz_body,
        }
    return out


def compute_harmonic_gains(
    z_body: Mapping[str, Any],
    *,
    frequency_hz: float,
    profile: Optional[IdentityStrengthProfile] = None,
    axes: Optional[Mapping[str, float]] = None,
    contrast: bool = False,
) -> List[float]:
    """Bounded log-domain gains per harmonic 1..8 from perceptual axes."""
    prof = profile or STRENGTH_PROFILES["light"]
    ax = dict(axes) if axes is not None else compute_perceptual_axes(z_body, contrast=contrast)
    b = ax["brightness_centroid"]
    w = ax["low_mid_warmth"]
    r = ax["high_freq_rolloff"]
    a = ax["attack_bloom"]
    d = ax["decay_sustain"]
    m = ax["body_resonance_density"]
    scale = prof.axis_gain_scale * prof.harmonic_gain_max

    axis_by_harmonic = [
        0.15 * w + 0.10 * m,  # h1 fundamental — minimal
        0.55 * w + 0.45 * m,
        0.50 * w + 0.40 * m,
        0.35 * b + 0.30 * a + 0.20 * w,
        0.40 * b + 0.35 * a,
        0.30 * b + 0.25 * r + 0.20 * d,
        0.25 * r + 0.35 * d,
        0.20 * r + 0.40 * d,
    ]
    gains: List[float] = []
    for k, raw in enumerate(axis_by_harmonic, start=1):
        cap = prof.fundamental_gain_max if k == 1 else prof.harmonic_gain_max
        val = max(-cap, min(cap, raw * scale))
        gains.append(round(val, 6))
    return gains


def apply_perceptual_band_shaping(
    audio: np.ndarray,
    *,
    frequency_hz: float,
    sample_rate: int,
    axes: Mapping[str, float],
    profile: IdentityStrengthProfile,
) -> np.ndarray:
    """Band EQ from perceptual axes — low/mid/high bands, bounded dB."""
    x = np.asarray(audio, dtype=np.float64)
    n = len(x)
    if n < 64:
        return x.copy()
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sample_rate))
    f0 = float(frequency_hz)
    low_gain_db = profile.band_eq_max_db * (
        0.55 * axes.get("low_mid_warmth", 0.0) + 0.45 * axes.get("body_resonance_density", 0.0)
    )
    mid_gain_db = profile.band_eq_max_db * (
        0.45 * axes.get("brightness_centroid", 0.0) + 0.35 * axes.get("attack_bloom", 0.0)
    )
    high_gain_db = profile.band_eq_max_db * (
        0.50 * axes.get("high_freq_rolloff", 0.0) + 0.35 * axes.get("decay_sustain", 0.0)
    )
    out = spec.copy()
    for i, f_hz in enumerate(freqs):
        if f_hz < f0 * 1.8:
            db = low_gain_db
        elif f_hz < f0 * 5.0:
            db = mid_gain_db
        else:
            db = high_gain_db
        out[i] *= 10.0 ** (db / 20.0)
    return np.asarray(np.fft.irfft(out, n=n), dtype=np.float64)


def compare_audio_to_reference(
    audio: np.ndarray,
    reference: np.ndarray,
) -> Dict[str, Any]:
    """RMS / peak difference vs V4.1 reference (diagnostic)."""
    a = np.asarray(audio, dtype=np.float64)
    b = np.asarray(reference, dtype=np.float64)
    n = min(len(a), len(b))
    if n < 8:
        return {"rms_diff_db_vs_reference": None, "max_abs_diff": None, "likely_audible": False}
    a, b = a[:n], b[:n]
    diff = a - b
    ref_rms = max(_rms(b), 1e-12)
    diff_rms = _rms(diff)
    rms_diff_db = 20.0 * math.log10(max(diff_rms, 1e-15) / ref_rms)
    max_abs = float(np.max(np.abs(diff)))
    max_abs_db = 20.0 * math.log10(max(max_abs, 1e-15) / ref_rms)
    return {
        "rms_diff_db_vs_reference": round(rms_diff_db, 4),
        "max_abs_diff": round(max_abs, 8),
        "max_abs_diff_db_vs_reference_rms": round(max_abs_db, 4),
        "likely_audible": rms_diff_db > -45.0,
        "layer_active_gate_pass": rms_diff_db > -45.0,
    }


# Typical LHS geometry ranges for feature normalization
_GEOM_RANGES: Dict[str, Tuple[float, float]] = {
    "length": (0.48, 0.58),
    "width": (0.24, 0.44),
    "depth": (0.09, 0.12),
    "top_thickness": (0.0028, 0.0036),
    "back_thickness": (0.0030, 0.0038),
    "hole_radius": (0.040, 0.055),
}

_FREQ_BANDS: Tuple[Tuple[float, float], ...] = (
    (60.0, 120.0),
    (120.0, 180.0),
    (180.0, 280.0),
    (280.0, 420.0),
    (420.0, 650.0),
)

_WOOD_EMBED: Dict[str, Tuple[float, float]] = {
    "spruce": (0.0, 1.0),
    "cedar": (-0.35, 0.85),
    "maple": (0.55, 1.15),
    "mahogany": (0.25, 1.05),
    "rosewood": (0.45, 1.12),
}


def _norm_range(val: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (float(val) - lo) / (hi - lo)))


def _to_centered(val01: float) -> float:
    return max(-1.0, min(1.0, 2.0 * float(val01) - 1.0))


def _wood_embed(wood_id: str) -> Tuple[float, float]:
    key = str(wood_id or "spruce").lower()
    return _WOOD_EMBED.get(key, (0.0, 1.0))


def _rank_norm(val: float, pool: Sequence[float]) -> float:
    pos = [float(v) for v in pool if v is not None and float(v) > 0]
    if not val or val <= 0 or not pos:
        return 0.12
    pos.sort()
    rank = sum(1 for v in pos if v <= float(val)) / max(len(pos), 1)
    return max(0.08, min(1.0, rank))


def _modes_from_modal(modal_data: ModalInput) -> List[Dict[str, Any]]:
    from body_response_synth import modes_in_validated_band, parse_modal_modes

    all_modes, _ = parse_modal_modes(modal_data)
    return modes_in_validated_band(all_modes)


def build_body_identity_vector(
    *,
    parameters: Optional[Mapping[str, Any]],
    modal_data: ModalInput,
    frequency_hz: float,
    repo_root: Optional[Path] = None,
    sample_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Construct z_body feature dict + flat vector for distance metrics."""
    p = normalize_sample_parameters(parameters)
    mass = compute_body_mass_proxies(p)
    modes = _modes_from_modal(modal_data)
    f0 = max(40.0, float(frequency_hz))

    from bridge_mobility_proxy import _g

    length = _g(p, "length", 0.52)
    width = _g(p, "width", 0.32)
    depth = _g(p, "depth", 0.10)
    top_t = _g(p, "top_thickness", 0.003)
    back_t = _g(p, "back_thickness", 0.0033)
    hole_r = _g(p, "hole_radius", 0.047)

    top_area = length * width * 0.92
    back_area = length * width * 0.88
    aspect = length / max(width, 1e-6)
    cavity = length * width * depth
    hole_area = math.pi * hole_r * hole_r

    top_w, top_d = _wood_embed(str(p.get("top_wood_id") or "spruce"))
    back_w, back_d = _wood_embed(str(p.get("back_wood_id") or "mahogany"))
    top_damp = WOOD_DAMPING_COEFF.get(str(p.get("top_wood_id") or "spruce").lower(), 1.0)
    back_damp = WOOD_DAMPING_COEFF.get(str(p.get("back_wood_id") or "mahogany").lower(), 1.0)

    rad_pool = [float(m.get("radiation_proxy") or 0) for m in modes]
    mic_pool = [float(m.get("mic_output_proxy") or 0) for m in modes]
    bridge_pool = [
        float(m.get("bridge_excitation_abs") or m.get("bridge_excitation_coupling") or 0) for m in modes
    ]

    band_density: Dict[str, float] = {}
    near_counts: Dict[str, int] = {}
    far_energy: Dict[str, float] = {f"band_{i}": 0.0 for i in range(len(_FREQ_BANDS))}
    q_vals: List[float] = []
    top_shares: List[float] = []
    back_shares: List[float] = []
    air_shares: List[float] = []
    eff_masses: List[float] = []
    dom_regions: Dict[str, int] = {}

    harmonics = [f0 * k for k in range(1, 9)]

    for bi, (lo, hi) in enumerate(_FREQ_BANDS):
        in_band = [m for m in modes if lo <= float(m.get("frequency_hz") or 0) <= hi]
        band_density[f"density_{int(lo)}_{int(hi)}"] = len(in_band) / max(len(modes), 1)
        near = 0
        for m in in_band:
            f_hz = float(m.get("frequency_hz") or 0)
            if any(abs(f_hz - h) / max(h, 1e-6) < 0.05 for h in harmonics):
                near += 1
            rad = float(m.get("radiation_proxy") or 0)
            far_energy[f"band_{bi}"] += rad
        near_counts[f"near_{int(lo)}_{int(hi)}"] = near

    for m in modes:
        top_s, back_s, air_s, _ = normalize_participation_shares(m)
        top_shares.append(top_s)
        back_shares.append(back_s)
        air_shares.append(air_s)
        eff_masses.append(effective_modal_mass_proxy(m, mass))
        q_rec = compute_per_mode_damping(m, float(m.get("frequency_hz") or f0), p)
        q = q_rec.get("mode_q")
        if q and float(q) > 0:
            q_vals.append(float(q))
        dr = str(m.get("dominant_region") or "unknown")
        dom_regions[dr] = dom_regions.get(dr, 0) + 1

    rad_sorted = sorted(rad_pool, reverse=True)
    k_top = max(1, int(math.ceil(len(rad_sorted) * 0.2))) if rad_sorted else 1
    top_rad_mean = float(np.mean(rad_sorted[:k_top])) if rad_sorted else 0.0

    cache_meta: Dict[str, Any] = {}
    if repo_root and sample_id:
        from body_signature_cache import load_body_signature_cache

        cache = load_body_signature_cache(Path(repo_root), str(sample_id))
        if cache:
            cache_meta = {k: cache.get(k) for k in (
                "bridge_mobility_proxy",
                "mixed_body_mass_proxy",
                "top_effective_mass_proxy",
                "back_effective_mass_proxy",
            ) if k in cache}

    features: Dict[str, float] = {
        "geom_length": _to_centered(_norm_range(length, *_GEOM_RANGES["length"])),
        "geom_width": _to_centered(_norm_range(width, *_GEOM_RANGES["width"])),
        "geom_depth": _to_centered(_norm_range(depth, *_GEOM_RANGES["depth"])),
        "geom_top_thickness": _to_centered(_norm_range(top_t, *_GEOM_RANGES["top_thickness"])),
        "geom_back_thickness": _to_centered(_norm_range(back_t, *_GEOM_RANGES["back_thickness"])),
        "geom_hole_radius": _to_centered(_norm_range(hole_r, *_GEOM_RANGES["hole_radius"])),
        "geom_top_area": _to_centered(_norm_range(top_area, 0.10, 0.26)),
        "geom_back_area": _to_centered(_norm_range(back_area, 0.09, 0.24)),
        "geom_air_volume": _to_centered(_norm_range(cavity, 0.008, 0.025)),
        "geom_aspect_ratio": _to_centered(_norm_range(aspect, 1.15, 2.0)),
        "geom_hole_to_area": _to_centered(_norm_range(hole_area / max(top_area, 1e-9), 0.01, 0.08)),
        "mat_top_wood_a": top_w,
        "mat_top_wood_b": top_d,
        "mat_back_wood_a": back_w,
        "mat_back_wood_b": back_d,
        "mat_top_damping": _to_centered(_norm_range(top_damp, 0.75, 1.30)),
        "mat_back_damping": _to_centered(_norm_range(back_damp, 0.75, 1.30)),
        "mass_top": _to_centered(_norm_range(mass["top_effective_mass_proxy"], 0.0003, 0.0007)),
        "mass_back": _to_centered(_norm_range(mass["back_effective_mass_proxy"], 0.0003, 0.0008)),
        "mass_mixed": _to_centered(_norm_range(mass["mixed_body_mass_proxy"], 0.00035, 0.00075)),
        "bridge_mobility": _to_centered(_norm_range(mass["bridge_mobility_proxy"], 0.72, 1.28)),
        "share_top_mean": _to_centered(float(np.mean(top_shares)) if top_shares else 0.0),
        "share_back_mean": _to_centered(float(np.mean(back_shares)) if back_shares else 0.0),
        "share_air_mean": _to_centered(float(np.mean(air_shares)) if air_shares else 0.0),
        "q_fingerprint": _to_centered(_norm_range(float(np.median(q_vals)) if q_vals else 40.0, 25.0, 70.0)),
        "q_spread": _to_centered(_norm_range(float(np.std(q_vals)) if len(q_vals) > 1 else 0.0, 0.0, 12.0)),
        "rad_rank_top": _rank_norm(top_rad_mean, rad_pool),
        "rad_rank_median": _rank_norm(float(np.median(rad_pool)) if rad_pool else 0.0, rad_pool),
        "mic_rank_median": _rank_norm(float(np.median(mic_pool)) if mic_pool else 0.0, mic_pool),
        "bridge_rank_median": _rank_norm(float(np.median(bridge_pool)) if bridge_pool else 0.0, bridge_pool),
        "eff_mass_median": _to_centered(_norm_range(float(np.median(eff_masses)) if eff_masses else 0.0, 0.0002, 0.001)),
        "low_body_color": _to_centered(_norm_range(far_energy.get("band_0", 0.0), 0.0, max(rad_pool or [1.0]))),
        "mid_body_color": _to_centered(_norm_range(far_energy.get("band_2", 0.0), 0.0, max(rad_pool or [1.0]))),
        "high_body_color": _to_centered(_norm_range(far_energy.get("band_4", 0.0), 0.0, max(rad_pool or [1.0]))),
    }
    for k, v in band_density.items():
        features[f"modal_{k}"] = _to_centered(min(1.0, float(v) * 4.0))
    for k, v in near_counts.items():
        features[f"modal_{k}"] = _to_centered(min(1.0, float(v) / 8.0))

    if cache_meta.get("bridge_mobility_proxy") is not None:
        features["cache_bridge_mobility"] = _to_centered(
            _norm_range(float(cache_meta["bridge_mobility_proxy"]), 0.72, 1.28)
        )

    flat = [float(features[k]) for k in sorted(features.keys())]
    return {
        "features": features,
        "vector": flat,
        "field_names": sorted(features.keys()),
        "mass_proxies": mass,
        "cache_meta": cache_meta,
        "dominant_region_histogram": dom_regions,
        "mode_count": len(modes),
    }


def compute_identity_layer_residual(
    base: np.ndarray,
    *,
    frequency_hz: float,
    sample_rate: int,
    feature_source: Mapping[str, Any],
    profile: IdentityStrengthProfile,
    contrast: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Full identity layer residual (layered - base) with bounded shaping."""
    axes = compute_perceptual_axes(feature_source, contrast=contrast)
    h_gains = compute_harmonic_gains(
        feature_source,
        frequency_hz=frequency_hz,
        profile=profile,
        axes=axes,
        contrast=contrast,
    )
    band_shaped = apply_perceptual_band_shaping(
        base,
        frequency_hz=frequency_hz,
        sample_rate=sample_rate,
        axes=axes,
        profile=profile,
    )
    harmonic_shaped = apply_harmonic_identity_shaping(
        band_shaped,
        frequency_hz=frequency_hz,
        sample_rate=sample_rate,
        harmonic_gains=h_gains,
    )
    layered = apply_identity_residual(
        base,
        harmonic_shaped,
        epsilon=profile.identity_epsilon,
        residual_gain_max=profile.residual_gain_max,
    )
    residual = layered - base
    return residual, {
        "perceptual_axes": axes,
        "harmonic_gains": h_gains,
        "profile": profile.name,
        "residual_rms": round(_rms(residual), 8),
    }


def apply_hybrid_audibility_floor(
    audio: np.ndarray,
    reference: np.ndarray,
    *,
    min_db: float = HYBRID_AUDIBILITY_MIN_DB,
    target_db: float = HYBRID_AUDIBILITY_TARGET_DB,
    max_db: float = HYBRID_AUDIBILITY_MAX_DB,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Scale hybrid residual toward audible -30..-22 dB band vs V4.1 base."""
    a = np.asarray(audio, dtype=np.float64)
    b = np.asarray(reference, dtype=np.float64)
    n = min(len(a), len(b))
    if n < 8:
        return a.copy(), {"audibility_adjusted": False}
    a, b = a[:n], b[:n]
    diff = a - b
    ref_rms = max(_rms(b), 1e-12)
    diff_rms = _rms(diff)
    rms_db = 20.0 * math.log10(max(diff_rms, 1e-15) / ref_rms)
    gain = 1.0
    if rms_db < min_db:
        gain = 10.0 ** ((target_db - rms_db) / 20.0)
    elif rms_db > max_db:
        gain = 10.0 ** ((max_db - rms_db) / 20.0)
    if abs(gain - 1.0) < 0.02:
        return a, {"audibility_adjusted": False, "rms_diff_db_before": round(rms_db, 4)}
    out = b + gain * diff
    out_rms_db = 20.0 * math.log10(max(_rms(out - b), 1e-15) / ref_rms)
    return out, {
        "audibility_adjusted": True,
        "audibility_gain": round(gain, 6),
        "rms_diff_db_before": round(rms_db, 4),
        "rms_diff_db_after": round(out_rms_db, 4),
    }


def synthesize_v4_1_identity_space_note(
    *,
    frequency_hz: float,
    note_name: str,
    duration_s: float,
    sample_rate: int,
    modal_data: ModalInput,
    output_wav: Path,
    output_metadata_json: Optional[Path],
    velocity: float,
    sample_parameters: Optional[Mapping[str, Any]],
    modal_source: Optional[str],
    diagnostic_mode: str,
    synthesis_preset: Optional[str],
    repo_root: Path,
    sample_id: str,
) -> Dict[str, Any]:
    """V4.1 full base + bounded identity-space layer (strength from diagnostic mode)."""
    hybrid_active = is_hybrid_identity_mode(diagnostic_mode)
    g_active = is_g_identity_mode(diagnostic_mode)
    profile = strength_profile_for_mode(diagnostic_mode) or STRENGTH_PROFILES["light"]
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        base_meta = synthesize_hybrid_v4_1_note(
            frequency_hz=frequency_hz,
            note_name=note_name,
            duration_s=duration_s,
            sample_rate=sample_rate,
            modal_data=modal_data,
            output_wav=tmp_path,
            output_metadata_json=None,
            velocity=velocity,
            sample_parameters=sample_parameters,
            modal_source=modal_source,
            diagnostic_mode="modal_body_hybrid_v4_1_full",
            synthesis_preset=synthesis_preset,
            repo_root=repo_root,
            sample_id=sample_id,
        )
        base_audio, _ = read_wav_float_mono(tmp_path)

        z_body = build_body_identity_vector(
            parameters=sample_parameters,
            modal_data=modal_data,
            frequency_hz=frequency_hz,
            repo_root=repo_root,
            sample_id=sample_id,
        )
        contrast_ctx = (sample_parameters or {}).get("identity_contrast_context") or {}
        dz_body = contrast_ctx.get("dz_body") or z_body

        if g_active:
            g_cfg = g_config_for_mode(diagnostic_mode)
            abs_w = float(g_cfg["absolute_weight"])
            contrast_w = float(g_cfg["contrast_weight"])
            combined_residual, hybrid_core = compose_hybrid_contrast_residual(
                base_audio,
                frequency_hz=frequency_hz,
                sample_rate=sample_rate,
                z_body=z_body,
                dz_body=dz_body,
                abs_w=abs_w,
                contrast_w=contrast_w,
            )
            g_physical: Dict[str, Any] = {}
            if g_cfg["decay_active"]:
                combined_residual, decay_meta = apply_decay_differentiation_to_residual(
                    combined_residual,
                    z_body,
                    sample_rate=sample_rate,
                )
                g_physical["decay"] = decay_meta
            if g_cfg["bridge_active"]:
                combined_residual, bridge_meta = apply_bridge_coupling_to_residual(
                    combined_residual,
                    frequency_hz=frequency_hz,
                    sample_rate=sample_rate,
                    z_body=z_body,
                )
                g_physical["bridge"] = bridge_meta
            if g_cfg.get("de_thump_active"):
                combined_residual, de_thump_meta = apply_residual_de_thump(
                    combined_residual,
                    sample_rate=sample_rate,
                )
                g_physical["de_thump"] = de_thump_meta
            blended = base_audio + combined_residual
            blended, audibility_info = apply_hybrid_audibility_floor(blended, base_audio)
            final, guard = apply_rms_guard(blended, base_audio, max_db=HYBRID_RMS_GUARD_DB)
            vs_ref = compare_audio_to_reference(final, base_audio)
            h_gains = hybrid_core.get("contrast_layer", {}).get("harmonic_gains")
            axes = hybrid_core.get("contrast_layer", {}).get("perceptual_axes")
            hybrid_meta = {
                **hybrid_core,
                "g_config": g_cfg,
                "g_physical": g_physical,
                "hybrid_audibility": audibility_info,
            }
            contrast_active = True
        elif hybrid_active:
            abs_w, contrast_w = hybrid_blend_for_mode(diagnostic_mode)
            combined_residual, hybrid_core = compose_hybrid_contrast_residual(
                base_audio,
                frequency_hz=frequency_hz,
                sample_rate=sample_rate,
                z_body=z_body,
                dz_body=dz_body,
                abs_w=abs_w,
                contrast_w=contrast_w,
            )
            blended = base_audio + combined_residual
            blended, audibility_info = apply_hybrid_audibility_floor(blended, base_audio)
            final, guard = apply_rms_guard(blended, base_audio, max_db=HYBRID_RMS_GUARD_DB)
            vs_ref = compare_audio_to_reference(final, base_audio)
            h_gains = hybrid_core.get("contrast_layer", {}).get("harmonic_gains")
            axes = hybrid_core.get("contrast_layer", {}).get("perceptual_axes")
            hybrid_meta = {
                **hybrid_core,
                "hybrid_audibility": audibility_info,
            }
            contrast_active = True
        else:
            contrast_active = is_contrast_identity_mode(diagnostic_mode)
            if contrast_active and contrast_ctx.get("dz_body"):
                axis_source = contrast_ctx["dz_body"]
                axes = compute_perceptual_axes(axis_source, contrast=True)
                h_gains = compute_harmonic_gains(
                    axis_source,
                    frequency_hz=frequency_hz,
                    profile=profile,
                    axes=axes,
                    contrast=True,
                )
            else:
                axis_source = z_body
                axes = compute_perceptual_axes(z_body, contrast=False)
                h_gains = compute_harmonic_gains(
                    z_body,
                    frequency_hz=frequency_hz,
                    profile=profile,
                    axes=axes,
                )
            band_shaped = apply_perceptual_band_shaping(
                base_audio,
                frequency_hz=frequency_hz,
                sample_rate=sample_rate,
                axes=axes,
                profile=profile,
            )
            harmonic_shaped = apply_harmonic_identity_shaping(
                band_shaped,
                frequency_hz=frequency_hz,
                sample_rate=sample_rate,
                harmonic_gains=h_gains,
            )
            blended = apply_identity_residual(
                base_audio,
                harmonic_shaped,
                epsilon=profile.identity_epsilon,
                residual_gain_max=profile.residual_gain_max,
            )
            final, guard = apply_rms_guard(blended, base_audio, max_db=profile.rms_guard_max_db)
            vs_ref = compare_audio_to_reference(final, base_audio)
            hybrid_meta = None

        peak_db = 20.0 * math.log10(max(float(np.max(np.abs(final))), 1e-12))
        _write_wav(Path(output_wav), final, sample_rate)

        meta = dict(base_meta)
        meta.update(
            {
                "diagnostic_mode": diagnostic_mode,
                "body_hybrid_v4_1_identity_space_active": True,
                "v4_1_base_preserved": True,
                "identity_hybrid_active": hybrid_active,
                "identity_g_active": g_active,
                "identity_final_stk_candidate": is_final_stk_candidate_mode(diagnostic_mode),
                "identity_final_canonical_mode": (
                    canonical_stk_final_mode(diagnostic_mode)
                    if is_final_stk_candidate_mode(diagnostic_mode)
                    else None
                ),
                "identity_strength_profile": (
                    profile.name if not (hybrid_active or g_active) else ("g_physical" if g_active else "hybrid")
                ),
                "identity_epsilon": profile.identity_epsilon if not (hybrid_active or g_active) else None,
                "harmonic_gains": h_gains,
                "perceptual_axes": axes,
                "harmonic_gain_bounds": {
                    "fundamental_max": profile.fundamental_gain_max,
                    "harmonic_2_8_max": profile.harmonic_gain_max,
                    "residual_max": profile.residual_gain_max,
                    "band_eq_max_db": profile.band_eq_max_db,
                    "axis_gain_scale": profile.axis_gain_scale,
                }
                if not (hybrid_active or g_active)
                else {
                    "absolute_profile": "strong",
                    "contrast_profile": "contrast_strong",
                    "hybrid_rms_guard_db": HYBRID_RMS_GUARD_DB,
                    "g_decay_strength": G_DECAY_STRENGTH if g_active else None,
                    "g_bridge_blend": G_BRIDGE_BLEND if g_active else None,
                },
                "body_identity_vector": z_body,
                "identity_contrast_active": contrast_active,
                "identity_contrast_context": contrast_ctx if requires_identity_contrast_context(diagnostic_mode) else None,
                "perceptual_axes_source": (
                    "g_hybrid_abs+contrast+physical"
                    if g_active
                    else (
                        "hybrid_abs+contrast"
                        if hybrid_active
                        else ("dz_body" if contrast_active else "z_body")
                    )
                ),
                "identity_hybrid": hybrid_meta if (hybrid_active or g_active) else None,
                "identity_g": hybrid_meta if g_active else None,
                "identity_rms_guard": guard,
                "identity_vs_v41_reference": vs_ref,
                "output_peak_dbfs": round(peak_db, 4),
                "clipping_avoided": peak_db < PEAK_CLIP_DBFS,
            }
        )
        if output_metadata_json:
            output_metadata_json.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        return meta
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def apply_harmonic_identity_shaping(
    audio: np.ndarray,
    *,
    frequency_hz: float,
    sample_rate: int,
    harmonic_gains: Sequence[float],
) -> np.ndarray:
    """Shape harmonics 2–8; fundamental nearly preserved."""
    x = np.asarray(audio, dtype=np.float64)
    n = len(x)
    if n < 64:
        return x.copy()
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sample_rate))
    f0 = float(frequency_hz)
    out = spec.copy()
    for k, gain in enumerate(harmonic_gains[:8], start=1):
        target = f0 * k
        if target >= freqs[-1]:
            continue
        idx = int(np.argmin(np.abs(freqs - target)))
        scale = math.exp(float(gain))
        if k == 1:
            scale = 1.0 + (scale - 1.0) * 0.15
        out[idx] *= scale
        if idx > 0:
            out[idx - 1] *= 1.0 + (scale - 1.0) * 0.25
        if idx + 1 < len(out):
            out[idx + 1] *= 1.0 + (scale - 1.0) * 0.25
    y = np.fft.irfft(out, n=n)
    return np.asarray(y, dtype=np.float64)


def apply_identity_residual(
    base: np.ndarray,
    shaped: np.ndarray,
    *,
    epsilon: float = IDENTITY_EPSILON,
    residual_gain_max: float = RESIDUAL_GAIN_MAX,
) -> np.ndarray:
    residual = shaped - base
    cap = residual_gain_max * float(np.max(np.abs(base)) + 1e-12)
    residual = np.clip(residual, -cap, cap)
    return base + float(epsilon) * residual


def apply_rms_guard(
    audio: np.ndarray,
    reference: np.ndarray,
    *,
    max_db: float = RMS_GUARD_MAX_DB,
) -> Tuple[np.ndarray, Dict[str, float]]:
    ref_rms = max(_rms(reference), 1e-12)
    out_rms = _rms(audio)
    if out_rms <= 1e-15:
        return audio.copy(), {"rms_guard_gain": 1.0}
    ratio_db = 20.0 * math.log10(out_rms / ref_rms)
    gain = 1.0
    if abs(ratio_db) > max_db:
        target = ref_rms * (10.0 ** (max_db / 20.0)) if ratio_db > 0 else ref_rms / (10.0 ** (max_db / 20.0))
        gain = target / out_rms
    guarded = audio * gain
    peak = float(np.max(np.abs(guarded)))
    if peak >= 1.0:
        gain *= 0.99 / peak
        guarded = guarded * (0.99 / peak)
    return guarded, {"rms_guard_gain": round(gain, 6), "rms_delta_db": round(ratio_db, 4)}


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(audio * 32767.0, -32767, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())


def physical_distance(z_a: Mapping[str, Any], z_b: Mapping[str, Any]) -> float:
    va = np.asarray(z_a.get("vector") or [], dtype=np.float64)
    vb = np.asarray(z_b.get("vector") or [], dtype=np.float64)
    n = min(len(va), len(vb))
    if n == 0:
        return 0.0
    return float(np.linalg.norm(va[:n] - vb[:n]) / math.sqrt(n))


def audio_timbre_vector(
    audio: np.ndarray,
    *,
    sample_rate: int,
    segment_meta: Optional[Mapping[str, Any]] = None,
) -> List[float]:
    from diagnostic_synthesis import _spectral_features
    from stage48_timbre_decomposition_report import _attack_time_ms, _spectral_flux

    spec = _spectral_features(audio, sample_rate)
    meta = segment_meta or {}
    return [
        float(spec.get("centroid_hz") or 0.0) / 1000.0,
        float(spec.get("low_energy") or 0.0),
        float(spec.get("mid_energy") or 0.0),
        float(spec.get("high_energy") or 0.0),
        _spectral_flux(audio, sample_rate) * 1e-4,
        _attack_time_ms(audio, sample_rate) * 0.01,
        float(meta.get("output_decay_slope_db_per_s") or 0.0) * -0.01,
        float(meta.get("final_rms_dbfs") or 0.0) * 0.01,
    ]


def audio_distance(t_a: Sequence[float], t_b: Sequence[float]) -> float:
    va = np.asarray(t_a, dtype=np.float64)
    vb = np.asarray(t_b, dtype=np.float64)
    n = min(len(va), len(vb))
    if n == 0:
        return 0.0
    return float(np.linalg.norm(va[:n] - vb[:n]) / math.sqrt(n))


def spearman_correlation(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) < 3 or len(xs) != len(ys):
        return None

    def _rank(vals: Sequence[float]) -> List[float]:
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        ranks = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    rx, ry = _rank(list(xs)), _rank(list(ys))
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den_x = math.sqrt(sum((a - mx) ** 2 for a in rx))
    den_y = math.sqrt(sum((b - my) ** 2 for b in ry))
    if den_x <= 0 or den_y <= 0:
        return None
    return round(num / (den_x * den_y), 6)


def distance_consistency_report(
    samples: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Pairwise physical vs audio distance + Spearman correlation."""
    n = len(samples)
    phys: List[float] = []
    audio: List[float] = []
    pairs: List[Dict[str, Any]] = []
    for i in range(n):
        for j in range(i + 1, n):
            si, sj = samples[i], samples[j]
            pd = physical_distance(si.get("z_body") or {}, sj.get("z_body") or {})
            ad = audio_distance(si.get("timbre") or [], sj.get("timbre") or [])
            phys.append(pd)
            audio.append(ad)
            pairs.append(
                {
                    "sample_a": si.get("sample_id"),
                    "sample_b": sj.get("sample_id"),
                    "physical_distance": round(pd, 6),
                    "audio_distance": round(ad, 6),
                }
            )
    rho = spearman_correlation(phys, audio)
    return {
        "pair_count": len(pairs),
        "physical_distances": phys,
        "audio_distances": audio,
        "spearman_rho": rho,
        "pairs_sample": pairs[:15],
    }


def nearest_neighbor_preservation_report(
    samples: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Check whether nearest physical neighbors stay closer in audio space."""
    n = len(samples)
    if n < 3:
        return {"nn_preservation_rate": None, "pair_count": 0}
    preserved = 0
    nn_pairs: List[Dict[str, Any]] = []
    for i in range(n):
        si = samples[i]
        phys_dists: List[Tuple[int, float]] = []
        audio_dists: List[float] = []
        for j in range(n):
            if i == j:
                continue
            sj = samples[j]
            pd = physical_distance(si.get("z_body") or {}, sj.get("z_body") or {})
            ad = audio_distance(si.get("timbre") or [], sj.get("timbre") or [])
            phys_dists.append((j, pd))
            audio_dists.append(ad)
        nn_j, nn_phys = min(phys_dists, key=lambda t: t[1])
        ad_nn = audio_distance(si.get("timbre") or [], samples[nn_j].get("timbre") or [])
        med_audio = float(np.median(audio_dists)) if audio_dists else ad_nn
        ok = ad_nn <= med_audio
        if ok:
            preserved += 1
        nn_pairs.append(
            {
                "sample": si.get("sample_id"),
                "nearest_neighbor": samples[nn_j].get("sample_id"),
                "physical_distance_nn": round(nn_phys, 6),
                "audio_distance_nn": round(ad_nn, 6),
                "audio_distance_median": round(med_audio, 6),
                "preserved": ok,
            }
        )
    per_sample_rhos: List[float] = []
    for i in range(n):
        phys_row = []
        audio_row = []
        for j in range(n):
            if i == j:
                continue
            phys_row.append(
                physical_distance(samples[i].get("z_body") or {}, samples[j].get("z_body") or {})
            )
            audio_row.append(
                audio_distance(samples[i].get("timbre") or [], samples[j].get("timbre") or [])
            )
        rho = spearman_correlation(phys_row, audio_row)
        if rho is not None:
            per_sample_rhos.append(float(rho))
    return {
        "nn_preservation_rate": round(preserved / n, 4),
        "nn_preservation_count": preserved,
        "sample_count": n,
        "mean_local_spearman": round(float(np.mean(per_sample_rhos)), 6) if per_sample_rhos else None,
        "pairs_sample": nn_pairs[:10],
    }


def estimate_audible_clusters(audio_distances_flat: Sequence[float], *, threshold: float = 0.12) -> int:
    """Greedy cluster count from pairwise audio distances (diagnostic estimate)."""
    if not audio_distances_flat:
        return 0
    # approximate: count distance bins above threshold relative to median
    med = float(np.median(audio_distances_flat))
    far = sum(1 for d in audio_distances_flat if d > max(threshold, med * 0.85))
    # rough clusters ~ sqrt(far pairs) heuristic capped 2..10
    est = max(2, min(10, int(round(math.sqrt(max(far, 1)) + 1))))
    return est
