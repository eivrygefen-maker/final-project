#!/usr/bin/env python3
"""
STK V6.2.2 — single onset / anti-thump / tail continuity repair (diagnostic only).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from body_response_synth import (
    DEFAULT_SAMPLE_RATE,
    DEFAULT_VELOCITY,
    FIXED_PLUCK_POSITION,
    ModalInput,
    _rms,
    synthesize_plucked_string,
)
from sample_parameters import normalize_sample_parameters
from stk_v5_design_helpers import (
    _band_energy,
    _peak_dbfs,
    render_v5_alpha_body_radiation_path,
    rms_matched_body_dominant_mix,
)
from stk_v6_2_audit_features import (
    collect_features_for_synthesis,
    feature_value,
    get_sample_record,
)
from stk_v6_2_physical_routing import (
    DEFAULT_DURATION_S,
    STK_V6_2_MODE,
    V6_2_1_VARIANTS,
    _apply_note_hf_damping,
    _build_bridge_body_stem,
    _build_cavity_body_tail_stem,
    _build_soundhole_air_stem,
    _build_top_radiation_stem,
    _feature_used_for,
    _finalize_mix_with_strategy,
    _mean_modal_share,
    _window_rms,
    compute_balance_diagnostics,
)

V6_2_2_VERSION = "stk_v6_2_2_onset_tail_repair_v0"

V6_2_2_MODES: Dict[str, str] = {
    "stk_v6_2_2_single_onset_soft_tail_alpha": "STK V6.2.2 single onset soft tail alpha",
    "stk_v6_2_2_no_thump_body_tail_alpha": "STK V6.2.2 no thump body tail alpha",
    "stk_v6_2_2_v5_body_v6_pluck_hybrid_alpha": "STK V6.2.2 V5 body / V6 pluck hybrid alpha",
}

V6_2_2_VARIANTS: Dict[str, Dict[str, Any]] = {
    "stk_v6_2_2_single_onset_soft_tail_alpha": {
        "unified_attack_ms": 16.0,
        "unified_decay_s": 0.28,
        "unified_gain": 0.42,
        "body_delay_ms": 14.0,
        "body_ramp_ms": 42.0,
        "resonator_delay_ms": 55.0,
        "resonator_ramp_ms": 110.0,
        "thump_limit_strength": 0.38,
        "tail_floor_strength": 0.14,
        "cavity_decay_mult": 1.35,
        "stem_gains": {
            "unified_attack_stem": 0.34,
            "bridge_body_stem": 0.72,
            "top_radiation_stem": 0.92,
            "soundhole_air_stem": 0.78,
            "cavity_body_tail_stem": 1.05,
        },
        "norm_method": "sustain_window_rms",
        "sustain_target_rms": 0.046,
        "hybrid_v5_body": False,
    },
    "stk_v6_2_2_no_thump_body_tail_alpha": {
        "unified_attack_ms": 14.0,
        "unified_decay_s": 0.24,
        "unified_gain": 0.36,
        "body_delay_ms": 18.0,
        "body_ramp_ms": 55.0,
        "resonator_delay_ms": 70.0,
        "resonator_ramp_ms": 140.0,
        "thump_limit_strength": 0.58,
        "tail_floor_strength": 0.18,
        "cavity_decay_mult": 1.50,
        "stem_gains": {
            "unified_attack_stem": 0.28,
            "bridge_body_stem": 0.68,
            "top_radiation_stem": 0.88,
            "soundhole_air_stem": 0.72,
            "cavity_body_tail_stem": 1.12,
        },
        "norm_method": "sustain_window_rms",
        "sustain_target_rms": 0.044,
        "hybrid_v5_body": False,
    },
    "stk_v6_2_2_v5_body_v6_pluck_hybrid_alpha": {
        "unified_attack_ms": 15.0,
        "unified_decay_s": 0.22,
        "unified_gain": 0.38,
        "body_delay_ms": 12.0,
        "body_ramp_ms": 38.0,
        "resonator_delay_ms": 50.0,
        "resonator_ramp_ms": 95.0,
        "thump_limit_strength": 0.45,
        "tail_floor_strength": 0.16,
        "cavity_decay_mult": 1.40,
        "v5_string_weight": 0.0,
        "v5_body_weight": 1.0,
        "attack_mix_gain": 0.22,
        "stem_gains": {
            "unified_attack_stem": 0.26,
            "bridge_body_stem": 0.65,
            "top_radiation_stem": 0.85,
            "soundhole_air_stem": 0.70,
            "cavity_body_tail_stem": 1.08,
        },
        "norm_method": "sustain_window_rms",
        "sustain_target_rms": 0.045,
        "hybrid_v5_body": True,
    },
}

V621_SOFT_PLUCK = "stk_v6_2_1_soft_pluck_tail_alpha"


def _smooth_ramp_in(
    audio: np.ndarray,
    sample_rate: int,
    *,
    delay_ms: float,
    ramp_ms: float,
) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float64)
    sr = int(sample_rate)
    n = len(x)
    t = np.arange(n, dtype=np.float64) / sr
    delay = float(delay_ms) / 1000.0
    ramp = max(float(ramp_ms) / 1000.0, 1e-4)
    env = np.where(
        t < delay,
        0.0,
        np.where(t < delay + ramp, (t - delay) / ramp, 1.0),
    )
    # Hann-like ease on ramp
    ramp_mask = (t >= delay) & (t < delay + ramp)
    if np.any(ramp_mask):
        u = (t[ramp_mask] - delay) / ramp
        env[ramp_mask] = 0.5 * (1.0 - np.cos(np.pi * u))
    return x * env


def _build_unified_attack_stem(
    string_excitation: np.ndarray,
    *,
    sample_rate: int,
    frequency_hz: float,
    attack_ms: float,
    decay_s: float,
    gain: float,
    hf_absorb: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Single coherent onset: pluck + short string share one envelope (no double peak)."""
    x = np.asarray(string_excitation, dtype=np.float64)
    sr = int(sample_rate)
    n = len(x)
    t = np.arange(n, dtype=np.float64) / sr
    t_a = max(float(attack_ms), 6.0) / 1000.0
    f0 = max(40.0, float(frequency_hz))

    rise = np.sin(0.5 * math.pi * np.minimum(t, t_a) / t_a) ** 1.15
    post = np.maximum(t - t_a, 0.0)
    decay_env = np.exp(-post / max(float(decay_s), 0.05))
    master = rise * np.where(t <= t_a, 1.0, decay_env)

    unified = gain * x * master
    unified = _apply_note_hf_damping(
        unified,
        sample_rate=sr,
        frequency_hz=f0,
        hf_absorb=hf_absorb + 0.04,
    )

    pluck_n = max(8, int(sr * t_a * 1.05))
    pluck_stem = np.zeros(n, dtype=np.float64)
    pluck_stem[:pluck_n] = unified[:pluck_n]

    return unified, pluck_stem, {
        "unified_attack_ms": attack_ms,
        "unified_decay_s": decay_s,
        "unified_gain": gain,
        "onset_coherent": True,
        "deriv_weight": 0.0,
    }


def _build_continuous_cavity_tail(
    bridge_body: np.ndarray,
    *,
    sample_rate: int,
    cavity_decay_s: float,
    cavity_q: float,
    mass_loading: float,
    hf_absorb: float,
    tail_boost: float,
    decay_mult: float = 1.0,
) -> np.ndarray:
    from stk_v6_2_physical_routing import _damped_resonator_ir

    x = np.asarray(bridge_body, dtype=np.float64)
    n = len(x)
    sr = int(sample_rate)
    t = np.arange(n, dtype=np.float64) / sr
    mass_factor = max(0.85, min(1.15, 1.0 + 0.08 * (float(mass_loading) * 1000.0 - 0.47)))
    decay = max(float(cavity_decay_s) * mass_factor * float(decay_mult), 0.55)
    env = np.exp(-t / decay) * 0.82 + 0.18 * np.exp(-t / (decay * 2.2))
    q_tail = max(float(cavity_q) * 0.85, 6.0)
    ir = _damped_resonator_ir(95.0, q_tail, 0.35, n, sr)
    drv = x * env
    tail = np.convolve(drv / (float(np.sqrt(np.mean(drv**2))) + 1e-12), ir, mode="same")
    tail_rms = float(np.sqrt(np.mean(tail**2))) + 1e-12
    tail *= float(np.sqrt(np.mean(x**2))) / tail_rms * 0.72 * float(tail_boost)
    spec = np.fft.rfft(tail)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    spec[freqs > 1800] *= max(0.08, 0.35 - hf_absorb * 0.4)
    tail = np.real(np.fft.irfft(spec, n=n))
    return tail


def _apply_anti_thump(
    audio: np.ndarray,
    sample_rate: int,
    *,
    strength: float,
    window_s: float = 0.30,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    x = np.asarray(audio, dtype=np.float64).copy()
    sr = int(sample_rate)
    n_win = min(len(x), max(64, int(window_s * sr)))
    if n_win < 64:
        return x, {"thump_limit_applied": False}

    seg = x[:n_win]
    spec = np.fft.rfft(seg)
    freqs = np.fft.rfftfreq(n_win, d=1.0 / sr)
    s = float(strength)
    for lo, hi in ((60.0, 250.0), (250.0, 700.0)):
        band = (freqs >= lo) & (freqs < hi)
        spec[band] *= 1.0 - s * 0.28
    limited = np.real(np.fft.irfft(spec, n=n_win))
    t = np.arange(n_win, dtype=np.float64) / sr
    time_gain = 0.72 + 0.28 * (1.0 - np.exp(-t / 0.09))
    limited *= time_gain
    x[:n_win] = limited
    return x, {"thump_limit_applied": True, "thump_limit_strength": s, "thump_window_s": window_s}


def _apply_tail_continuity_floor(
    audio: np.ndarray,
    sample_rate: int,
    *,
    duration_s: float,
    floor_strength: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    x = np.asarray(audio, dtype=np.float64).copy()
    sr = int(sample_rate)
    n = len(x)
    t = np.arange(n, dtype=np.float64) / sr
    ref0 = int(0.35 * sr)
    ref1 = int(min(1.2 * sr, n))
    ref_rms = _window_rms(x, 0.35, 1.2, sr)
    if ref_rms < 1e-9:
        return x, {"tail_floor_applied": False}

    template = np.exp(-np.maximum(t - 0.25, 0.0) / 1.6) * ref_rms * 0.55
    blend = float(floor_strength) * np.clip((t - 1.0) / 1.2, 0.0, 1.0)
    x = x + template * blend

    tail_start = int(1.5 * sr)
    if tail_start < n - 64:
        tail_rms = float(np.sqrt(np.mean(x[tail_start:n] ** 2)))
        mid_rms = _window_rms(x, 0.8, 1.5, sr)
        if tail_rms < mid_rms * 0.22:
            lift = mid_rms * 0.28 - tail_rms
            if lift > 0:
                fade = np.linspace(1.0, 0.35, n - tail_start)
                x[tail_start:n] += lift * fade * np.sign(x[tail_start:n] + 1e-12)

    return x, {"tail_floor_applied": True, "tail_floor_strength": floor_strength}


def _find_onset_peaks(
    envelope: np.ndarray,
    sample_rate: int,
    *,
    max_time_s: float = 0.25,
    min_dist_ms: float = 12.0,
    threshold_ratio: float = 0.22,
) -> List[int]:
    sr = int(sample_rate)
    n = min(len(envelope), int(max_time_s * sr))
    env = np.asarray(envelope[:n], dtype=np.float64)
    if n < 8:
        return []
    win = max(3, int(0.002 * sr))
    kernel = np.ones(win) / win
    smooth = np.convolve(np.abs(env), kernel, mode="same")
    peak_val = float(np.max(smooth))
    if peak_val < 1e-12:
        return []
    thresh = peak_val * threshold_ratio
    min_dist = max(1, int(min_dist_ms * sr / 1000.0))
    peaks: List[int] = []
    for i in range(2, n - 2):
        if smooth[i] < thresh:
            continue
        if not (smooth[i] >= smooth[i - 1] and smooth[i] >= smooth[i + 1]):
            continue
        if smooth[i] < smooth[i - 2] * 0.92 and smooth[i] < smooth[i + 2] * 0.92:
            continue
        if peaks and (i - peaks[-1]) < min_dist:
            if smooth[i] > smooth[peaks[-1]]:
                peaks[-1] = i
        else:
            peaks.append(i)
    # Keep only peaks within 12 dB of strongest
    if len(peaks) > 1:
        vals = [smooth[p] for p in peaks]
        top = max(vals)
        peaks = [p for p, v in zip(peaks, vals) if v >= top * 0.25]
    return peaks


def compute_onset_diagnostics(
    audio: np.ndarray,
    *,
    sample_rate: int,
) -> Dict[str, Any]:
    x = np.asarray(audio, dtype=np.float64)
    sr = int(sample_rate)
    peaks = _find_onset_peaks(x, sr)
    env = np.abs(x[: int(0.25 * sr)])
    peak_vals = [float(np.abs(x[p])) for p in peaks] if peaks else []
    strongest_ms = round(float(peaks[0]) / sr * 1000.0, 2) if peaks else None
    second_ratio = round(peak_vals[1] / max(peak_vals[0], 1e-12), 4) if len(peak_vals) >= 2 else 0.0
    count = len(peaks)
    double_risk = min(
        1.0,
        max(0.0, (count - 1) * 0.35 + max(0.0, second_ratio - 0.35) * 1.2),
    )
    coherence_pass = count <= 1 or second_ratio < 0.35
    return {
        "onset_peak_count_0_250ms": count,
        "strongest_onset_time_ms": strongest_ms,
        "second_onset_ratio": second_ratio,
        "double_pluck_risk_score": round(double_risk, 4),
        "onset_coherence_pass": coherence_pass,
    }


def compute_thump_diagnostics(
    audio: np.ndarray,
    *,
    sample_rate: int,
) -> Dict[str, Any]:
    x = np.asarray(audio, dtype=np.float64)
    sr = int(sample_rate)
    seg = x[: int(0.30 * sr)] if len(x) > int(0.30 * sr) else x
    seg_rms = max(_rms(seg), 1e-12)
    low = _band_energy(seg, sr, 60.0, 250.0)
    mid = _band_energy(seg, sr, 250.0, 700.0)
    thump_index = (low + mid * 0.85) / seg_rms
    low_mid_impulse = (low + mid) / seg_rms
    rms_early = _window_rms(x, 0.05, 0.25, sr)
    rms_mid = _window_rms(x, 0.25, 0.55, sr)
    boom_disc = max(0.0, (rms_early - rms_mid * 1.35) / max(rms_early, 1e-12))
    drum_tap = min(1.0, max(0.0, (thump_index - 1.8) * 0.22 + boom_disc * 0.65))
    return {
        "thump_index_0_300ms": round(thump_index, 6),
        "low_mid_impulse_ratio": round(low_mid_impulse, 6),
        "boom_decay_discontinuity_score": round(boom_disc, 4),
        "drum_tap_risk_score": round(drum_tap, 4),
    }


def compute_tail_continuity_diagnostics(
    audio: np.ndarray,
    *,
    sample_rate: int,
    duration_s: float = DEFAULT_DURATION_S,
) -> Dict[str, Any]:
    sr = int(sample_rate)
    r300_800 = _window_rms(audio, 0.30, 0.80, sr)
    r800_1500 = _window_rms(audio, 0.80, 1.50, sr)
    r1500_2500 = _window_rms(audio, 1.50, min(duration_s, 2.5), sr)
    ratio = r1500_2500 / max(r800_1500, 1e-12)
    drop = max(0.0, 1.0 - ratio)
    sustain_pass = ratio >= 0.25
    return {
        "rms_300_800ms": round(r300_800, 8),
        "rms_800_1500ms": round(r800_1500, 8),
        "rms_1500_2500ms": round(r1500_2500, 8),
        "tail_continuity_ratio": round(ratio, 4),
        "tail_drop_score": round(drop, 4),
        "sustain_continuity_pass": sustain_pass,
    }


def compute_v622_diagnostics(
    audio: np.ndarray,
    *,
    sample_rate: int,
    frequency_hz: float,
    duration_s: float = DEFAULT_DURATION_S,
) -> Dict[str, Any]:
    onset = compute_onset_diagnostics(audio, sample_rate=sample_rate)
    thump = compute_thump_diagnostics(audio, sample_rate=sample_rate)
    tail = compute_tail_continuity_diagnostics(
        audio, sample_rate=sample_rate, duration_s=duration_s
    )
    balance = compute_balance_diagnostics(
        audio,
        sample_rate=sample_rate,
        frequency_hz=frequency_hz,
        duration_s=duration_s,
    )
    warnings: List[str] = []
    if not onset.get("onset_coherence_pass"):
        warnings.append("double onset detected — multiple peaks in first 250 ms")
    if float(thump.get("thump_index_0_300ms") or 0.0) > 2.2:
        warnings.append("thump_index high — low/mid boom in attack window")
    if float(thump.get("drum_tap_risk_score") or 0.0) > 0.45:
        warnings.append("drum_tap_risk_score elevated")
    if float(tail.get("tail_continuity_ratio") or 0.0) < 0.25:
        warnings.append("tail collapses after 1.5 s — continuity ratio low")
    out = {**onset, **thump, **tail}
    out["balance_diagnostics"] = balance
    out["v622_warnings"] = warnings
    out["v622_pass"] = len(warnings) == 0
    return out


def build_body_tail_stem(
    stems: Mapping[str, np.ndarray],
    stem_gains: Mapping[str, float],
) -> np.ndarray:
    body_keys = (
        "bridge_body_stem",
        "top_radiation_stem",
        "soundhole_air_stem",
        "cavity_body_tail_stem",
    )
    parts = []
    for k in body_keys:
        if k in stems:
            parts.append(float(stem_gains.get(k, 1.0)) * stems[k])
    if not parts:
        return np.zeros(1, dtype=np.float64)
    return sum(parts)


def synthesize_v6_2_2_onset_tail_repair(
    *,
    frequency_hz: float,
    duration_s: float = DEFAULT_DURATION_S,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    modal_data: ModalInput,
    sample_parameters: Mapping[str, Any],
    audit: Mapping[str, Any],
    sample_id: str = "sample_000",
    repo_root: Optional[Any] = None,
    velocity: float = DEFAULT_VELOCITY,
    variant: str = "stk_v6_2_2_single_onset_soft_tail_alpha",
) -> Tuple[Dict[str, np.ndarray], np.ndarray, Dict[str, Any]]:
    if variant not in V6_2_2_VARIANTS:
        raise ValueError(f"unknown V6.2.2 variant: {variant}")
    vcfg = V6_2_2_VARIANTS[variant]

    sample_rec = get_sample_record(audit, sample_id)
    feat = collect_features_for_synthesis(audit, sample_id)
    params = normalize_sample_parameters(sample_parameters)

    helm = float(feature_value(sample_rec, "helmholtz_like_frequency_proxy", audit=audit, default=104.0))
    cav_q = float(feature_value(sample_rec, "cavity_q_proxy", audit=audit, default=14.0))
    cav_decay = float(feature_value(sample_rec, "cavity_decay_proxy", audit=audit, default=1.2))
    hf_abs = float(feature_value(sample_rec, "high_frequency_absorption_proxy", audit=audit, default=0.28))
    mobility = float(feature_value(sample_rec, "bridge_mobility_proxy", audit=audit, default=1.0))
    mass_load = float(feature_value(sample_rec, "mass_loading_proxy", audit=audit, default=0.00047))
    top_damp = float(feature_value(sample_rec, "top_damping_coeff_proxy", audit=audit, default=1.0))
    top_share = _mean_modal_share(modal_data, "top_share")
    rad_gain = feature_value(sample_rec, "top_radiation_gain_proxy", audit=audit)
    hole_gain = feature_value(sample_rec, "soundhole_radiation_gain_proxy", audit=audit)

    string_excitation = synthesize_plucked_string(
        frequency_hz,
        duration_s,
        sample_rate,
        pluck_position=FIXED_PLUCK_POSITION,
        velocity=velocity,
    )

    unified, pluck_stem, attack_meta = _build_unified_attack_stem(
        string_excitation,
        sample_rate=sample_rate,
        frequency_hz=frequency_hz,
        attack_ms=float(vcfg["unified_attack_ms"]),
        decay_s=float(vcfg["unified_decay_s"]),
        gain=float(vcfg["unified_gain"]),
        hf_absorb=hf_abs,
    )

    if vcfg.get("hybrid_v5_body"):
        body_radiated, _, v5_meta = render_v5_alpha_body_radiation_path(
            frequency_hz=frequency_hz,
            duration_s=duration_s,
            sample_rate=sample_rate,
            modal_data=modal_data,
            sample_parameters=params,
            repo_root=repo_root,
            sample_id=sample_id,
            velocity=velocity,
        )
        bridge_stem = body_radiated
        bridge_meta = {"hybrid_v5_body": True, **v5_meta}
    else:
        bridge_stem, bridge_meta = _build_bridge_body_stem(
            frequency_hz=frequency_hz,
            duration_s=duration_s,
            sample_rate=sample_rate,
            modal_data=modal_data,
            sample_parameters=params,
            repo_root=repo_root,
            sample_id=sample_id,
            mobility=mobility,
        )

    bridge_stem = _smooth_ramp_in(
        bridge_stem,
        sample_rate,
        delay_ms=float(vcfg["body_delay_ms"]),
        ramp_ms=float(vcfg["body_ramp_ms"]),
    )

    top_stem = _build_top_radiation_stem(
        bridge_stem,
        sample_rate=sample_rate,
        frequency_hz=frequency_hz,
        top_share=top_share,
        top_damping=top_damp,
        hf_absorb=hf_abs,
        radiation_gain=rad_gain,
        sustain_boost=1.15,
    )
    top_stem = _smooth_ramp_in(
        top_stem,
        sample_rate,
        delay_ms=float(vcfg["body_delay_ms"]),
        ramp_ms=float(vcfg["body_ramp_ms"]),
    )

    soundhole_stem = _build_soundhole_air_stem(
        bridge_stem,
        sample_rate=sample_rate,
        helmholtz_hz=helm,
        cavity_q=cav_q,
        soundhole_gain=hole_gain,
        hf_absorb=hf_abs,
        soundhole_boost=1.05,
    )
    soundhole_stem = _smooth_ramp_in(
        soundhole_stem,
        sample_rate,
        delay_ms=float(vcfg["resonator_delay_ms"]),
        ramp_ms=float(vcfg["resonator_ramp_ms"]),
    )

    cavity_stem = _build_continuous_cavity_tail(
        bridge_stem,
        sample_rate=sample_rate,
        cavity_decay_s=cav_decay,
        cavity_q=cav_q,
        mass_loading=mass_load,
        hf_absorb=hf_abs,
        tail_boost=1.1,
        decay_mult=float(vcfg.get("cavity_decay_mult", 1.35)),
    )
    cavity_stem = _smooth_ramp_in(
        cavity_stem,
        sample_rate,
        delay_ms=float(vcfg["resonator_delay_ms"]),
        ramp_ms=float(vcfg["resonator_ramp_ms"]),
    )

    stem_gains = dict(vcfg["stem_gains"])
    attack_gain = float(stem_gains.pop("unified_attack_stem", 0.30))

    stems: Dict[str, np.ndarray] = {
        "pluck_attack_stem": pluck_stem,
        "direct_string_short_stem": unified,
        "bridge_body_stem": bridge_stem,
        "top_radiation_stem": top_stem,
        "soundhole_air_stem": soundhole_stem,
        "cavity_body_tail_stem": cavity_stem,
    }

    body_tail = build_body_tail_stem(stems, stem_gains)

    if vcfg.get("hybrid_v5_body"):
        v5_mix, _ = rms_matched_body_dominant_mix(
            body_tail,
            unified,
            body_weight=float(vcfg.get("v5_body_weight", 1.0)),
            string_weight=float(vcfg.get("v5_string_weight", 0.0)),
        )
        mixed = v5_mix + unified * float(vcfg.get("attack_mix_gain", 0.22))
    else:
        mixed = attack_gain * unified
        for k, g in stem_gains.items():
            if k in stems:
                mixed = mixed + float(g) * stems[k]

    mixed, thump_info = _apply_anti_thump(
        mixed,
        sample_rate,
        strength=float(vcfg["thump_limit_strength"]),
    )
    mixed, tail_info = _apply_tail_continuity_floor(
        mixed,
        sample_rate,
        duration_s=duration_s,
        floor_strength=float(vcfg["tail_floor_strength"]),
    )

    final, finalize_info = _finalize_mix_with_strategy(
        mixed,
        sample_rate=sample_rate,
        duration_s=duration_s,
        norm_method=str(vcfg.get("norm_method", "sustain_window_rms")),
        sustain_target_rms=float(vcfg.get("sustain_target_rms", 0.045)),
    )

    diagnostics = compute_v622_diagnostics(
        final,
        sample_rate=sample_rate,
        frequency_hz=frequency_hz,
        duration_s=duration_s,
    )

    provenance_used: Dict[str, Any] = {}
    for name, rec in feat.items():
        provenance_used[name] = {
            "value": rec.get("value"),
            "status": rec.get("status"),
            "per_sample": rec.get("per_sample"),
            "source_path": rec.get("source_path"),
            "confidence": rec.get("confidence"),
            "used_for": _feature_used_for(name),
        }

    stems["body_tail_stem"] = body_tail
    stems["final_mix"] = final

    meta = {
        "diagnostic_mode": variant,
        "v6_2_2_version": V6_2_2_VERSION,
        "user_label": V6_2_2_MODES.get(variant, variant),
        "variant": variant,
        "sample_id": sample_id,
        "frequency_hz": frequency_hz,
        "duration_s": duration_s,
        "attack_meta": attack_meta,
        "bridge_meta": bridge_meta,
        "stem_gains": {"unified_attack_stem": attack_gain, **stem_gains},
        "thump_info": thump_info,
        "tail_info": tail_info,
        "finalize_info": finalize_info,
        "v622_diagnostics": diagnostics,
        "feature_provenance_used": provenance_used,
        "peak_dbfs": finalize_info.get("peak_dbfs"),
        "clipping_avoided": finalize_info.get("clipping_avoided"),
        "limitations": [
            "V6.2.2 does not prove multi-guitar differentiation yet.",
            "sample_000 A4 review scope only in builder.",
            "Not solved — diagnostic repair iteration.",
        ],
    }
    return stems, final, meta
