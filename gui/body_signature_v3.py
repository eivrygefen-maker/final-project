#!/usr/bin/env python3
"""
Stage 4.9 — modal_body_signature_v3 diagnostic model (starts from radiation v1, not v2).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from bridge_mobility_proxy import bridge_body_coupling_factor, compute_body_mass_proxies
from string_body_balance import _smoothstep, low_body_color_strength

V3_MOBILITY_CLAMP = (0.85, 1.15)
V3_PROXY_MISSING_FLOOR = 0.12


@dataclass(frozen=True)
class V3Ablation:
    """Which V3 components are active for the current diagnostic mode."""

    low_f0_imprint: bool = False
    mobility: bool = False
    far_color: bool = False


V3_MODE_ABLATIONS: Dict[str, V3Ablation] = {
    "modal_body_signature_v3": V3Ablation(True, True, True),
    "modal_body_signature_v3_full": V3Ablation(True, True, True),
    "modal_radiation_color_v3": V3Ablation(True, True, True),
    "modal_body_signature_v3_core": V3Ablation(False, False, False),
    "modal_body_signature_v3_low_f0_imprint_only": V3Ablation(True, False, False),
    "modal_body_signature_v3_mobility_only": V3Ablation(False, True, False),
    "modal_body_signature_v3_far_color_only": V3Ablation(False, False, True),
}


def get_v3_ablation(mode_name: Optional[str]) -> Optional[V3Ablation]:
    if not mode_name:
        return None
    return V3_MODE_ABLATIONS.get(mode_name)


def is_v3_family_mode(mode_name: Optional[str]) -> bool:
    return get_v3_ablation(mode_name) is not None


def low_f0_imprint_strength(f0: float, *, full_hz: float = 165.0, zero_hz: float = 320.0) -> float:
    """Continuous low-f0 imprint strength — strongest below ~165 Hz, fades by mid range."""
    f0 = max(40.0, float(f0))
    if f0 >= zero_hz:
        return 0.0
    if f0 <= full_hz:
        return 1.0
    return 1.0 - _smoothstep(full_hz, zero_hz, f0)


def _rank_normalize(val: float, pool: Sequence[float]) -> float:
    if not pool or val <= 0:
        return V3_PROXY_MISSING_FLOOR
    sorted_pool = sorted(float(v) for v in pool if v > 0)
    if not sorted_pool:
        return V3_PROXY_MISSING_FLOOR
    p95 = sorted_pool[min(len(sorted_pool) - 1, int(math.ceil(0.95 * len(sorted_pool))) - 1)]
    ref = max(p95, sorted_pool[-1], 1e-12)
    return max(0.08, min(1.0, float(val) / ref))


def proxy_pools_from_modes(band_modes: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    bridges: List[float] = []
    rads: List[float] = []
    mics: List[float] = []
    for mode in band_modes:
        b = mode.get("bridge_excitation_abs") or mode.get("bridge_excitation_coupling")
        if b is not None and float(b) > 0:
            bridges.append(abs(float(b)))
        r = mode.get("radiation_proxy")
        if r is not None and float(r) > 0:
            rads.append(float(r))
        m = mode.get("mic_output_proxy")
        if m is not None and float(m) > 0:
            mics.append(float(m))
    return {
        "bridge_values": bridges,
        "radiation_values": rads,
        "mic_values": mics,
    }


def bounded_mobility_factor(
    mode: Mapping[str, Any],
    damp_rec: Mapping[str, Any],
    parameters: Optional[Mapping[str, Any]],
    *,
    clamp: Tuple[float, float] = V3_MOBILITY_CLAMP,
) -> Tuple[float, Dict[str, Any]]:
    mass = compute_body_mass_proxies(parameters)
    bridge_raw = float(damp_rec.get("bridge_weight") or mode.get("bridge_excitation_abs") or 1.0)
    coupled, mrec = bridge_body_coupling_factor(mode, parameters, existing_bridge=max(bridge_raw, 1e-9))
    sample_mob = float(mass.get("bridge_mobility_proxy") or 1.0)
    top_s = float(damp_rec.get("top_share") or 0.25)
    back_s = float(damp_rec.get("back_share") or 0.25)
    air_s = float(damp_rec.get("air_share") or 0.25)
    plate_share = top_s + back_s
    blended = 0.55 * sample_mob + 0.30 * (coupled / max(bridge_raw, 1e-9)) + 0.15 * (1.0 + 0.08 * air_s)
    lo, hi = clamp
    factor = max(lo, min(hi, blended**0.55))
    return factor, {**mrec, "mobility_factor_m": round(factor, 6)}


def far_color_factor_v3(
    damp_rec: Mapping[str, Any],
    comp: Mapping[str, Any],
    *,
    radiation_factor: float,
    mobility_factor: float,
) -> float:
    """Smoothed sample-specific far-path color — envelope not pitch peaks."""
    top_s = float(damp_rec.get("top_share") or 0.0)
    back_s = float(damp_rec.get("back_share") or 0.0)
    air_s = float(damp_rec.get("air_share") or 0.0)
    tau = float(damp_rec.get("mode_tau_s") or 0.2)
    q_bw = float(damp_rec.get("mode_bandwidth_hz") or 6.0)
    smooth = 0.78 + 0.22 * min(1.0, q_bw / 16.0) * min(1.0, tau / 0.4)
    share_color = 0.40 * top_s + 0.28 * back_s + 0.32 * air_s
    rad_n = max(0.0, min(1.0, radiation_factor))
    base = 0.48 + 0.28 * rad_n + 0.24 * share_color
    base *= 0.92 + 0.08 * mobility_factor
    return base * smooth


def decompose_modal_amplitude_v3(
    mode: Mapping[str, Any],
    damp_rec: Mapping[str, Any],
    comp: Mapping[str, Any],
    f_hz: float,
    v1_amp_meta: Mapping[str, Any],
    pools: Mapping[str, Any],
    *,
    ablation: V3Ablation,
    parameters: Optional[Mapping[str, Any]],
    note_hz: float,
) -> Dict[str, Any]:
    """
    Per-mode amplitude decomposition on top of radiation v1 base.
    """
    bridge_vals = list(pools.get("bridge_values") or [])
    rad_vals = list(pools.get("radiation_values") or [])
    mic_vals = list(pools.get("mic_values") or [])

    bridge_raw = float(comp.get("bridge_weight") or 1.0)
    rad_raw = _safe_float(mode.get("radiation_proxy")) or float(comp.get("radiation_weight") or 1.0)
    mic_raw = _safe_float(mode.get("mic_output_proxy")) or float(comp.get("mic_weight") or 1.0)

    excitation = _rank_normalize(bridge_raw, bridge_vals) ** 0.82
    radiation = (
        0.45 * _rank_normalize(rad_raw, rad_vals)
        + 0.35 * _rank_normalize(mic_raw, mic_vals)
        + 0.20 * math.sqrt(max(_rank_normalize(rad_raw, rad_vals) * _rank_normalize(mic_raw, mic_vals), 1e-9))
    ) ** 0.88
    f_eff = (max(f_hz, 60.0) / 200.0) ** 0.10
    radiation *= f_eff

    category = str(damp_rec.get("mode_category") or "coupled")
    cat_map = {"top": 1.05, "back": 0.96, "air": 1.10, "coupled": 1.0}
    category_factor = cat_map.get(category, 1.0)

    mat = float(damp_rec.get("mode_material_damping") or 1.0)
    damping_factor = max(0.90, min(1.10, 1.0 + 0.06 * (mat - 1.0)))

    mobility_factor = 1.0
    mobility_meta: Dict[str, Any] = {}
    if ablation.mobility:
        mobility_factor, mobility_meta = bounded_mobility_factor(mode, damp_rec, parameters)

    v1_modal_amp = float(v1_amp_meta.get("mode_amplitude_factor") or 1.0)
    far_color = 1.0
    if ablation.far_color:
        far_color = far_color_factor_v3(
            damp_rec, comp, radiation_factor=radiation, mobility_factor=mobility_factor
        )

    low_s = low_body_color_strength(note_hz)
    low_boost = 1.0
    if ablation.low_f0_imprint and low_s > 0 and f_hz < 280.0:
        low_boost = 1.0 + low_s * 0.12 * max(0.0, excitation * radiation - 0.55)

    final_modal_amp = v1_modal_amp * mobility_factor * low_boost
    if ablation.far_color:
        prox = float(comp.get("harmonic_proximity") or 0.0)
        if prox < 0.35:
            final_modal_amp *= 0.85 + 0.30 * far_color

    return {
        "excitation_factor_m": round(excitation, 6),
        "radiation_factor_m": round(radiation, 6),
        "mobility_factor_m": round(mobility_factor, 6),
        "category_factor_m": round(category_factor, 6),
        "damping_factor_m": round(damping_factor, 6),
        "far_color_factor_m": round(far_color, 6),
        "v1_modal_amp_m": round(v1_modal_amp, 6),
        "final_modal_amp_m": round(max(final_modal_amp, 1e-9), 6),
        "low_f0_imprint_strength": round(low_f0_imprint_strength(note_hz), 6),
        "bridge_mobility_meta": mobility_meta,
    }


def _safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def build_body_signature_envelope(
    band_modes: Sequence[Mapping[str, Any]],
    note_hz: float,
    n_bins: int,
    sample_rate: int,
    *,
    per_mode_weights: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Smooth body transfer envelope H_sig(f) from modal layout and weights."""
    freqs = np.fft.rfftfreq(n_bins, d=1.0 / float(sample_rate))
    env = np.zeros_like(freqs, dtype=np.float64)
    weights = per_mode_weights or [1.0] * len(band_modes)
    for mode, w in zip(band_modes, weights):
        if w <= 0:
            continue
        f_m = float(mode.get("frequency_hz") or 0.0)
        if f_m <= 0:
            continue
        sigma = max(8.0, f_m * 0.045)
        bump = w * np.exp(-0.5 * ((freqs - f_m) / sigma) ** 2)
        env += bump
    if float(np.max(env)) > 1e-12:
        env /= float(np.max(env))
    env = 0.55 + 0.45 * env
    low_s = low_f0_imprint_strength(note_hz)
    if low_s > 0:
        low_mask = np.exp(-0.5 * ((freqs - note_hz) / max(note_hz * 0.35, 20.0)) ** 2)
        env = env * (1.0 + low_s * 0.22 * low_mask)
    kernel = np.ones(5, dtype=np.float64) / 5.0
    env = np.convolve(env, kernel, mode="same")
    return np.clip(env, 0.72, 1.28)


def apply_harmonic_body_imprint(
    signal: np.ndarray,
    sample_rate: int,
    frequency_hz: float,
    envelope: np.ndarray,
    *,
    strength: float,
) -> np.ndarray:
    """Shape string harmonics using body signature envelope — preserves fundamental."""
    if strength <= 1e-6:
        return signal
    x = np.asarray(signal, dtype=np.float64)
    n = len(x)
    if n < 16:
        return x
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sample_rate))
    f0 = max(float(frequency_hz), 40.0)
    shaped = spec.copy()
    for k in range(1, 12):
        fk = k * f0
        if fk >= freqs[-1]:
            break
        idx = int(np.argmin(np.abs(freqs - fk)))
        env_val = float(envelope[idx]) if idx < len(envelope) else 1.0
        if k == 1:
            tilt = 1.0 + strength * 0.08 * (env_val - 1.0)
        else:
            tilt = 1.0 + strength * 0.28 * (env_val - 1.0)
        tilt = max(0.82, min(1.22, tilt))
        shaped[idx] *= tilt
    y = np.fft.irfft(shaped, n=n)
    return y


def imprint_only_layer(
    string_signal: np.ndarray,
    sample_rate: int,
    frequency_hz: float,
    band_modes: Sequence[Mapping[str, Any]],
    *,
    strength: float,
) -> np.ndarray:
    env = build_body_signature_envelope(band_modes, frequency_hz, len(string_signal), sample_rate)
    return apply_harmonic_body_imprint(
        string_signal, sample_rate, frequency_hz, env, strength=strength
    ) - string_signal
