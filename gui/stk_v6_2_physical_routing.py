#!/usr/bin/env python3
"""
STK V6.2 — physical routing single-guitar skeleton (diagnostic only).

Routed stems: pluck → bridge/body → top / soundhole / cavity → final mix.
Does not modify website default or production synthesis.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from body_response_first_v4_2 import build_body_transfer_function_v4_2
from body_response_synth import (
    DEFAULT_SAMPLE_RATE,
    DEFAULT_VELOCITY,
    FIXED_PLUCK_POSITION,
    ModalInput,
    _rms,
    _string_acceleration,
    apply_anti_click_taper,
    apply_loudness_finalize,
    modes_in_validated_band,
    parse_modal_modes,
    synthesize_plucked_string,
    write_wav_int16,
)
from body_signature_cache import load_body_signature_cache
from modal_damping import compute_per_mode_damping
from sample_parameters import normalize_sample_parameters
from stk_v5_design_helpers import (
    _band_energy,
    _peak_dbfs,
    _spectral_centroid,
    compute_realism_metrics,
    enrich_metrics_with_levels,
    render_v5_alpha_body_radiation_path,
)
from stk_v6_2_audit_features import (
    collect_features_for_synthesis,
    feature_value,
    get_feature,
    get_sample_record,
    load_audit_report,
)

V6_2_VERSION = "stk_v6_2_physical_routing_alpha_v0"
V6_2_1_VERSION = "stk_v6_2_1_balance_repair_v0"
STK_V6_2_MODE = "stk_v6_2_physical_routing_alpha"
STK_V6_2_GUI_LABEL = "STK V6.2 physical routing alpha"

V6_2_1_MODES: Dict[str, str] = {
    "stk_v6_2_1_balanced_tail_alpha": "STK V6.2.1 balanced tail alpha",
    "stk_v6_2_1_soft_pluck_tail_alpha": "STK V6.2.1 soft pluck tail alpha",
    "stk_v6_2_1_more_string_body_alpha": "STK V6.2.1 more string/body alpha",
}

DEFAULT_DURATION_S = 2.5
E5_METALLICITY_THRESHOLD = 0.08

# V6.2 original mix weights (preserved for stk_v6_2_physical_routing_alpha)
V6_2_ORIGINAL_STEM_GAINS = {
    "pluck_attack_stem": 1.05,
    "direct_string_short_stem": 0.90,
    "bridge_body_stem": 0.55,
    "top_radiation_stem": 0.72,
    "soundhole_air_stem": 0.58,
    "cavity_body_tail_stem": 0.68,
}

V6_2_1_VARIANTS: Dict[str, Dict[str, Any]] = {
    "stk_v6_2_1_balanced_tail_alpha": {
        "pluck_gain": 0.38,
        "pluck_brightness": 0.50,
        "pluck_deriv_weight": 0.10,
        "pluck_envelope_power": 0.88,
        "direct_gain": 0.36,
        "direct_decay_s": 0.24,
        "direct_gate_end_s": 0.30,
        "cavity_tail_boost": 1.45,
        "soundhole_boost": 1.35,
        "top_sustain_boost": 1.25,
        "bridge_boost": 1.15,
        "stem_gains": {
            "pluck_attack_stem": 0.38,
            "direct_string_short_stem": 0.72,
            "bridge_body_stem": 0.78,
            "top_radiation_stem": 1.05,
            "soundhole_air_stem": 0.95,
            "cavity_body_tail_stem": 1.15,
        },
        "norm_method": "sustain_window_rms",
        "sustain_target_rms": 0.048,
    },
    "stk_v6_2_1_soft_pluck_tail_alpha": {
        "pluck_gain": 0.24,
        "pluck_brightness": 0.38,
        "pluck_deriv_weight": 0.06,
        "pluck_envelope_power": 0.95,
        "direct_gain": 0.32,
        "direct_decay_s": 0.20,
        "direct_gate_end_s": 0.26,
        "cavity_tail_boost": 1.55,
        "soundhole_boost": 1.40,
        "top_sustain_boost": 1.30,
        "bridge_boost": 1.20,
        "stem_gains": {
            "pluck_attack_stem": 0.24,
            "direct_string_short_stem": 0.65,
            "bridge_body_stem": 0.82,
            "top_radiation_stem": 1.10,
            "soundhole_air_stem": 1.00,
            "cavity_body_tail_stem": 1.22,
        },
        "norm_method": "sustain_window_rms",
        "sustain_target_rms": 0.050,
    },
    "stk_v6_2_1_more_string_body_alpha": {
        "pluck_gain": 0.32,
        "pluck_brightness": 0.55,
        "pluck_deriv_weight": 0.08,
        "pluck_envelope_power": 0.90,
        "direct_gain": 0.48,
        "direct_decay_s": 0.30,
        "direct_gate_end_s": 0.38,
        "cavity_tail_boost": 1.35,
        "soundhole_boost": 1.25,
        "top_sustain_boost": 1.20,
        "bridge_boost": 1.10,
        "stem_gains": {
            "pluck_attack_stem": 0.32,
            "direct_string_short_stem": 0.88,
            "bridge_body_stem": 0.75,
            "top_radiation_stem": 1.00,
            "soundhole_air_stem": 0.90,
            "cavity_body_tail_stem": 1.08,
        },
        "norm_method": "sustain_window_rms",
        "sustain_target_rms": 0.046,
    },
}

STEM_NAMES = (
    "pluck_attack_stem",
    "direct_string_short_stem",
    "bridge_body_stem",
    "top_radiation_stem",
    "soundhole_air_stem",
    "cavity_body_tail_stem",
)


def load_reference_modal_from_audit(audit: Mapping[str, Any], repo_root: Path) -> Dict[str, Any]:
    rel = (audit.get("reference_modal_catalog") or {}).get("path") or "FEM/outputs/rom_stk_body.json"
    path = Path(repo_root) / str(rel)
    if not path.is_file():
        return {"predicted_modes": [], "analysis": "missing_reference_catalog"}
    return json.loads(path.read_text(encoding="utf-8"))


def _damped_resonator_ir(
    frequency_hz: float,
    q: float,
    gain: float,
    n_samples: int,
    sample_rate: int,
) -> np.ndarray:
    t = np.arange(n_samples, dtype=np.float64) / float(sample_rate)
    w = 2.0 * math.pi * max(40.0, float(frequency_hz))
    decay = w / (2.0 * max(float(q), 2.0))
    ir = gain * np.sin(w * t) * np.exp(-decay * t)
    ir_rms = float(np.sqrt(np.mean(ir**2))) + 1e-12
    return ir / ir_rms


def _apply_note_hf_damping(
    audio: np.ndarray,
    *,
    sample_rate: int,
    frequency_hz: float,
    hf_absorb: float,
) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float64)
    n = len(x)
    if n < 64:
        return x.copy()
    f0 = max(40.0, float(frequency_hz))
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sample_rate))
    spec = np.fft.rfft(x)
    if f0 >= 600:
        f_start = 1800.0
        strength = float(hf_absorb) + 0.24
    elif f0 >= 400:
        f_start = 2200.0
        strength = float(hf_absorb) + 0.12
    else:
        f_start = 2800.0
        strength = float(hf_absorb)
    shelf = np.ones_like(freqs)
    above = freqs > f_start
    shelf[above] *= 10.0 ** (
        -strength * np.minimum((freqs[above] - f_start) / 3500.0, 1.0) / 20.0 * 6.0
    )
    shelf[freqs > 4500] *= 10.0 ** (-2.0 / 20.0)
    return np.real(np.fft.irfft(spec * shelf, n=n))


def _build_pluck_attack_stem(
    string_excitation: np.ndarray,
    *,
    sample_rate: int,
    frequency_hz: float,
    attack_ms: float,
    brightness: float,
    metallic_damping: float,
    deriv_weight: float = 0.28,
    envelope_power: float = 0.62,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    x = np.asarray(string_excitation, dtype=np.float64)
    n = len(x)
    sr = int(sample_rate)
    attack_n = max(8, min(int(sr * attack_ms / 1000.0), n // 3, int(sr * 0.035)))
    t = np.arange(attack_n, dtype=np.float64) / sr
    f0 = max(40.0, float(frequency_hz))
    deriv = np.zeros(attack_n, dtype=np.float64)
    seg = x[:attack_n]
    if attack_n >= 2:
        d = np.diff(seg)
        deriv[1:attack_n] = d
        deriv[0] = d[0]
    env = np.sin(0.5 * math.pi * t / max(t[-1], 1e-6)) ** float(envelope_power)
    nail = env * (
        0.58 * seg
        + float(deriv_weight) * deriv
        + 0.18 * brightness * np.sin(2 * math.pi * f0 * 2.05 * t)
    )
    out = np.zeros(n, dtype=np.float64)
    out[:attack_n] = nail
    out = _apply_note_hf_damping(
        out,
        sample_rate=sr,
        frequency_hz=f0,
        hf_absorb=metallic_damping,
    )
    return out, {
        "pluck_attack_ms": attack_ms,
        "pluck_brightness": brightness,
        "pluck_metallic_damping": metallic_damping,
        "pluck_deriv_weight": deriv_weight,
        "pluck_envelope_power": envelope_power,
    }


def _build_direct_string_short_stem(
    string_excitation: np.ndarray,
    *,
    sample_rate: int,
    decay_s: float,
    gain: float,
    gate_end_s: float = 0.14,
    gate_fade_s: float = 0.045,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    x = np.asarray(string_excitation, dtype=np.float64)
    sr = int(sample_rate)
    t = np.arange(len(x), dtype=np.float64) / sr
    env = np.exp(-t / max(float(decay_s), 0.02))
    gate = np.ones_like(t)
    ge = float(gate_end_s)
    gf = float(gate_fade_s)
    gate[t > ge] *= np.exp(-(t[t > ge] - ge) / max(gf, 0.02))
    out = gain * x * env * gate
    return out, {
        "direct_string_gain": gain,
        "direct_string_decay_s": decay_s,
        "direct_string_gate_end_s": ge,
        "direct_string_gate_fade_s": gf,
    }


def _build_bridge_body_stem(
    *,
    frequency_hz: float,
    duration_s: float,
    sample_rate: int,
    modal_data: ModalInput,
    sample_parameters: Mapping[str, Any],
    repo_root: Optional[Path],
    sample_id: str,
    mobility: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    body, _, meta = render_v5_alpha_body_radiation_path(
        frequency_hz=frequency_hz,
        duration_s=duration_s,
        sample_rate=sample_rate,
        modal_data=modal_data,
        sample_parameters=sample_parameters,
        repo_root=repo_root,
        sample_id=sample_id,
    )
    scale = 0.88 + 0.18 * max(0.0, min(1.0, (float(mobility) - 0.85) / 0.35))
    return body * scale, {"bridge_mobility_scale": round(scale, 6), **meta}


def _mean_modal_share(modal_data: Mapping[str, Any], key: str) -> float:
    modes, _ = parse_modal_modes(modal_data)
    band = modes_in_validated_band(modes)
    if not band:
        return 0.33
    vals = [float(m.get(key) or 0.0) for m in band]
    return float(np.mean(vals)) if vals else 0.33


def _build_top_radiation_stem(
    bridge_body: np.ndarray,
    *,
    sample_rate: int,
    frequency_hz: float,
    top_share: float,
    top_damping: float,
    hf_absorb: float,
    radiation_gain: Optional[float],
    sustain_boost: float = 1.0,
) -> np.ndarray:
    x = np.asarray(bridge_body, dtype=np.float64)
    rad_g = float(radiation_gain or 0.0001) * 8000.0
    top = x * max(0.25, min(0.85, float(top_share))) * (0.75 + 0.35 * min(1.0, rad_g)) * float(sustain_boost)
    top = _apply_note_hf_damping(
        top,
        sample_rate=sample_rate,
        frequency_hz=frequency_hz,
        hf_absorb=hf_absorb * (0.85 + 0.15 * float(top_damping)),
    )
    return top


def _build_soundhole_air_stem(
    bridge_body: np.ndarray,
    *,
    sample_rate: int,
    helmholtz_hz: float,
    cavity_q: float,
    soundhole_gain: Optional[float],
    hf_absorb: float,
    soundhole_boost: float = 1.0,
) -> np.ndarray:
    x = np.asarray(bridge_body, dtype=np.float64)
    n = len(x)
    sr = int(sample_rate)
    drv = x / (float(np.sqrt(np.mean(x**2))) + 1e-12)
    g = 0.42 + 0.25 * min(1.0, float(soundhole_gain or 0.0) * 5000.0)
    ir = _damped_resonator_ir(helmholtz_hz, cavity_q, g, n, sr)
    air = np.convolve(drv, ir, mode="same")
    body_rms = float(np.sqrt(np.mean(x**2))) + 1e-12
    air *= body_rms * 0.75 * float(soundhole_boost)
    spec = np.fft.rfft(air)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    spec[freqs > 900] *= 0.35
    spec[freqs > 2200] *= 0.12
    air = np.real(np.fft.irfft(spec, n=n))
    return air * (1.0 - 0.25 * hf_absorb)


def _build_cavity_body_tail_stem(
    bridge_body: np.ndarray,
    *,
    sample_rate: int,
    cavity_decay_s: float,
    cavity_q: float,
    mass_loading: float,
    hf_absorb: float,
    tail_boost: float = 1.0,
) -> np.ndarray:
    x = np.asarray(bridge_body, dtype=np.float64)
    n = len(x)
    sr = int(sample_rate)
    t = np.arange(n, dtype=np.float64) / sr
    mass_factor = max(0.85, min(1.15, 1.0 + 0.08 * (float(mass_loading) * 1000.0 - 0.47)))
    decay = max(float(cavity_decay_s) * mass_factor, 0.35)
    env = np.exp(-t / decay)
    q_tail = max(float(cavity_q) * 0.85, 6.0)
    ir = _damped_resonator_ir(95.0, q_tail, 0.35, n, sr)
    drv = x * env
    tail = np.convolve(drv / (float(np.sqrt(np.mean(drv**2))) + 1e-12), ir, mode="same")
    tail_rms = float(np.sqrt(np.mean(tail**2))) + 1e-12
    tail *= float(np.sqrt(np.mean(x**2))) / tail_rms * 0.65 * float(tail_boost)
    spec = np.fft.rfft(tail)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    spec[freqs > 1800] *= max(0.08, 0.35 - hf_absorb * 0.4)
    tail = np.real(np.fft.irfft(spec, n=n))
    fade_out = int(min(n // 3, 0.05 * sr))
    if fade_out > 1:
        tail[-fade_out:] *= np.linspace(1.0, 0.0, fade_out)
    return tail


def _time_to_decay_db(audio: np.ndarray, sample_rate: int, db: float) -> Optional[float]:
    x = np.asarray(audio, dtype=np.float64)
    peak = float(np.max(np.abs(x)))
    if peak < 1e-12:
        return None
    target = peak * (10.0 ** (float(db) / 20.0))
    idx = np.where(np.abs(x) <= target)[0]
    if len(idx) == 0:
        return None
    return round(float(idx[0]) / float(sample_rate), 4)


def _window_rms(audio: np.ndarray, start_s: float, end_s: float, sample_rate: int) -> float:
    x = np.asarray(audio, dtype=np.float64)
    sr = int(sample_rate)
    i0 = max(0, int(start_s * sr))
    i1 = min(len(x), int(end_s * sr))
    if i1 <= i0:
        return 0.0
    return float(np.sqrt(np.mean(x[i0:i1] ** 2)))


def _soft_limit_attack(audio: np.ndarray, sample_rate: int, *, attack_ms: float = 35.0) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float64).copy()
    sr = int(sample_rate)
    n_atk = max(1, int(sr * attack_ms / 1000.0))
    seg = x[:n_atk]
    peak = float(np.max(np.abs(seg)))
    if peak > 0.55:
        x[:n_atk] = seg * (0.55 / peak)
    return x


def _finalize_mix_with_strategy(
    mixed: np.ndarray,
    *,
    sample_rate: int,
    duration_s: float,
    norm_method: str,
    sustain_target_rms: float = 0.045,
    loudness_strength: float = 0.14,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    x = np.asarray(mixed, dtype=np.float64)
    sr = int(sample_rate)
    info: Dict[str, Any] = {"norm_method": norm_method}

    mixed_taper, taper_info = apply_anti_click_taper(x, sr, duration_s=duration_s)
    info["taper_info"] = taper_info

    if norm_method == "sustain_window_rms":
        body_rms = _window_rms(mixed_taper, 0.20, 0.80, sr)
        tail_rms_pre = _window_rms(mixed_taper, 1.0, min(duration_s, 2.5), sr)
        scale = float(sustain_target_rms) / max(body_rms, 1e-12)
        scaled = mixed_taper * scale
        scaled = _soft_limit_attack(scaled, sr)
        peak = float(np.max(np.abs(scaled)))
        if peak > 0.94:
            scaled = scaled * (0.92 / peak)
        final = scaled
        info.update(
            {
                "sustain_window_rms_pre_norm": round(body_rms, 8),
                "tail_window_rms_pre_norm": round(tail_rms_pre, 8),
                "sustain_target_rms": sustain_target_rms,
                "sustain_scale_applied": round(scale, 6),
                "attack_soft_limit_ms": 35.0,
            }
        )
    else:
        final, loudness_info = apply_loudness_finalize(
            mixed_taper,
            sr,
            loudness_normalization_strength=loudness_strength,
            raw_body_variation_preserve=0.70,
        )
        info["loudness_info"] = loudness_info

    peak_db, clip_ok = _peak_dbfs(final)
    if peak_db > -0.5:
        final = final * (10.0 ** ((-0.5 - peak_db) / 20.0))
        peak_db, clip_ok = _peak_dbfs(final)
    info["peak_dbfs"] = peak_db
    info["clipping_avoided"] = clip_ok
    return final, info


def compute_balance_diagnostics(
    audio: np.ndarray,
    *,
    sample_rate: int,
    frequency_hz: float,
    duration_s: float = DEFAULT_DURATION_S,
) -> Dict[str, Any]:
    x = np.asarray(audio, dtype=np.float64)
    sr = int(sample_rate)
    attack_rms = _window_rms(x, 0.0, 0.05, sr)
    body_rms = _window_rms(x, 0.20, 0.80, sr)
    tail_rms = _window_rms(x, 1.0, min(duration_s, 2.5), sr)
    atk_body = attack_rms / max(body_rms, 1e-12)
    atk_tail = attack_rms / max(tail_rms, 1e-12)
    very_hi = _band_energy(x, sr, 3500.0, 12000.0)
    mid_lo = _band_energy(x, sr, 0.0, 3500.0)
    hi_met = very_hi / max(mid_lo, 1e-12)
    pluck_click = float(np.sqrt(np.mean(x[: max(1, int(0.012 * sr))] ** 2))) / max(attack_rms, 1e-12)
    drum_tap = min(1.0, max(0.0, (pluck_click - 0.55) * 1.8 + (atk_tail - 8.0) * 0.04))
    sustain_body = min(1.0, body_rms / max(attack_rms * 0.35, 1e-12))
    tail_aud = min(1.0, tail_rms / max(body_rms * 0.55, 1e-12))
    warnings: List[str] = []
    if atk_tail > 12.0:
        warnings.append("attack_to_tail_ratio too high — sustain likely inaudible after pluck")
    if tail_rms < 0.008:
        warnings.append("tail_rms_1_2p5s too low — body tail may disappear")
    if pluck_click > 1.35:
        warnings.append("pluck_click_index high — percussive click risk")
    if drum_tap > 0.55:
        warnings.append("drum_tap_risk_score high — attack may sound like drum tap")
    if attack_rms > 0.12 and tail_rms < attack_rms * 0.08:
        warnings.append("final_mix mostly transient — tail crushed by attack")
    f0 = float(frequency_hz)
    if f0 >= 620.0 and hi_met > E5_METALLICITY_THRESHOLD:
        warnings.append("E5 metallicity still high")
    return {
        "attack_rms_0_50ms": round(attack_rms, 8),
        "body_rms_200_800ms": round(body_rms, 8),
        "tail_rms_1_2p5s": round(tail_rms, 8),
        "attack_to_body_ratio": round(atk_body, 6),
        "attack_to_tail_ratio": round(atk_tail, 6),
        "tail_audibility_score": round(tail_aud, 4),
        "pluck_click_index": round(pluck_click, 6),
        "drum_tap_risk_score": round(drum_tap, 4),
        "sustain_body_presence_score": round(sustain_body, 4),
        "metallicity_index": round(hi_met, 6),
        "balance_warnings": warnings,
        "balance_pass": len(warnings) == 0,
    }


def compute_stem_metrics(
    audio: np.ndarray,
    *,
    sample_rate: int,
    frequency_hz: float,
    duration_s: float,
) -> Dict[str, Any]:
    x = np.asarray(audio, dtype=np.float64)
    sr = int(sample_rate)
    rms = _rms(x)
    peak = float(np.max(np.abs(x)))
    peak_db, clip_ok = _peak_dbfs(x)
    very_hi = _band_energy(x, sr, 3500.0, 12000.0)
    mid_lo = _band_energy(x, sr, 0.0, 3500.0)
    hi_met = very_hi / max(mid_lo, 1e-12)
    total_e = very_hi + mid_lo + 1e-12
    i1 = int(1.0 * sr)
    i25 = int(min(duration_s, 2.5) * sr)
    tail = x[i1:min(i25, len(x))] if i1 < len(x) else np.zeros(0)
    tail_e = float(np.sqrt(np.mean(tail**2))) / max(rms, 1e-12) if tail.size else 0.0
    atk = float(np.sqrt(np.mean(x[: max(1, int(0.05 * sr))] ** 2)))
    sus = float(np.sqrt(np.mean(x[int(0.35 * sr) : int(1.0 * sr)] ** 2))) if len(x) > int(0.35 * sr) else atk
    f0 = float(frequency_hz)
    e5_fail = f0 >= 620.0 and hi_met > E5_METALLICITY_THRESHOLD
    return {
        "duration_s": round(len(x) / sr, 4),
        "rms": round(rms, 8),
        "peak": round(peak, 8),
        "peak_dbfs": peak_db,
        "clipping_avoided": clip_ok,
        "spectral_centroid_hz": round(_spectral_centroid(x, sr), 2),
        "metallicity_index": round(hi_met, 6),
        "high_note_metallicity_index": round(hi_met, 6),
        "very_high_band_fraction": round(very_hi / total_e, 6),
        "attack_to_sustain_ratio": round(atk / max(sus, 1e-12), 6),
        "time_to_decay_20db_s": _time_to_decay_db(x, sr, -20.0),
        "time_to_decay_30db_s": _time_to_decay_db(x, sr, -30.0),
        "tail_energy_1s_to_2p5s": round(tail_e, 6),
        "sustain_naturalness_score": round(max(0.0, min(1.0, 1.0 - abs(math.log10(max(atk / max(sus, 1e-12), 1e-6)) / 2.5))), 4),
        "e5_metallicity_pass": not e5_fail,
        "e5_metallicity_warning": bool(e5_fail),
    }


def synthesize_v6_2_physical_routing(
    *,
    frequency_hz: float,
    duration_s: float = DEFAULT_DURATION_S,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    modal_data: ModalInput,
    sample_parameters: Mapping[str, Any],
    audit: Mapping[str, Any],
    sample_id: str = "sample_000",
    repo_root: Optional[Path] = None,
    velocity: float = DEFAULT_VELOCITY,
    variant: str = STK_V6_2_MODE,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, Dict[str, Any]]:
    """Build all V6.2/V6.2.1 stems and final mix for one note."""
    is_v621 = variant in V6_2_1_VARIANTS
    vcfg = V6_2_1_VARIANTS.get(variant, {})
    diagnostic_mode = variant if is_v621 else STK_V6_2_MODE
    user_label = V6_2_1_MODES.get(variant, STK_V6_2_GUI_LABEL)
    version_tag = V6_2_1_VERSION if is_v621 else V6_2_VERSION

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

    pluck_ms = 28.0 if float(frequency_hz) >= 500 else 22.0
    pluck_decay = 0.055 if float(frequency_hz) >= 500 else 0.075
    pluck_brightness = 0.85 if float(frequency_hz) < 500 else 0.65
    pluck_metallic = hf_abs + (0.12 if float(frequency_hz) >= 620 else 0.05)
    pluck_deriv = 0.28
    pluck_env_pow = 0.62
    direct_gain = 0.32 if float(frequency_hz) >= 500 else 0.38
    direct_gate_end = 0.14
    cavity_boost = 1.0
    soundhole_boost = 1.0
    top_boost = 1.0
    bridge_boost = 1.0
    norm_method = "global_loudness_finalize"
    sustain_target = 0.045

    if is_v621:
        pluck_brightness = float(vcfg.get("pluck_brightness", pluck_brightness))
        pluck_deriv = float(vcfg.get("pluck_deriv_weight", 0.10))
        pluck_env_pow = float(vcfg.get("pluck_envelope_power", 0.88))
        direct_gain = float(vcfg.get("direct_gain", direct_gain))
        pluck_decay = float(vcfg.get("direct_decay_s", 0.24))
        direct_gate_end = float(vcfg.get("direct_gate_end_s", 0.30))
        cavity_boost = float(vcfg.get("cavity_tail_boost", 1.35))
        soundhole_boost = float(vcfg.get("soundhole_boost", 1.25))
        top_boost = float(vcfg.get("top_sustain_boost", 1.20))
        bridge_boost = float(vcfg.get("bridge_boost", 1.10))
        norm_method = str(vcfg.get("norm_method", "sustain_window_rms"))
        sustain_target = float(vcfg.get("sustain_target_rms", 0.045))

    string_excitation = synthesize_plucked_string(
        frequency_hz,
        duration_s,
        sample_rate,
        pluck_position=FIXED_PLUCK_POSITION,
        velocity=velocity,
    )

    pluck_stem, pluck_meta = _build_pluck_attack_stem(
        string_excitation,
        sample_rate=sample_rate,
        frequency_hz=frequency_hz,
        attack_ms=pluck_ms,
        brightness=pluck_brightness,
        metallic_damping=pluck_metallic,
        deriv_weight=pluck_deriv,
        envelope_power=pluck_env_pow,
    )
    pluck_stem *= 0.95 if not is_v621 else 1.0

    direct_stem, direct_meta = _build_direct_string_short_stem(
        string_excitation,
        sample_rate=sample_rate,
        decay_s=pluck_decay,
        gain=direct_gain,
        gate_end_s=direct_gate_end,
        gate_fade_s=0.08 if is_v621 else 0.045,
    )
    direct_stem = _apply_note_hf_damping(
        direct_stem,
        sample_rate=sample_rate,
        frequency_hz=frequency_hz,
        hf_absorb=hf_abs + (0.06 if float(frequency_hz) >= 620 else 0.0),
    )

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
    bridge_stem = bridge_stem * float(bridge_boost)

    top_stem = _build_top_radiation_stem(
        bridge_stem,
        sample_rate=sample_rate,
        frequency_hz=frequency_hz,
        top_share=top_share,
        top_damping=top_damp,
        hf_absorb=hf_abs,
        radiation_gain=rad_gain,
        sustain_boost=top_boost,
    )

    soundhole_stem = _build_soundhole_air_stem(
        bridge_stem,
        sample_rate=sample_rate,
        helmholtz_hz=helm,
        cavity_q=cav_q,
        soundhole_gain=hole_gain,
        hf_absorb=hf_abs,
        soundhole_boost=soundhole_boost,
    )

    cavity_stem = _build_cavity_body_tail_stem(
        bridge_stem,
        sample_rate=sample_rate,
        cavity_decay_s=cav_decay,
        cavity_q=cav_q,
        mass_loading=mass_load,
        hf_absorb=hf_abs,
        tail_boost=cavity_boost,
    )

    stem_gains = dict(V6_2_ORIGINAL_STEM_GAINS if not is_v621 else vcfg.get("stem_gains", V6_2_ORIGINAL_STEM_GAINS))
    stems = {
        "pluck_attack_stem": pluck_stem,
        "direct_string_short_stem": direct_stem,
        "bridge_body_stem": bridge_stem,
        "top_radiation_stem": top_stem,
        "soundhole_air_stem": soundhole_stem,
        "cavity_body_tail_stem": cavity_stem,
    }
    mixed = sum(stem_gains[k] * stems[k] for k in stems)
    final, finalize_info = _finalize_mix_with_strategy(
        mixed,
        sample_rate=sample_rate,
        duration_s=duration_s,
        norm_method=norm_method,
        sustain_target_rms=sustain_target,
    )
    peak_db = finalize_info.get("peak_dbfs")
    clip_ok = finalize_info.get("clipping_avoided")

    stem_rms = {k: _rms(v) for k, v in stems.items()}
    total_stem_rms = sum(stem_rms.values()) + 1e-12
    stem_ratios = {k: round(v / total_stem_rms, 6) for k, v in stem_rms.items()}

    sustain_window = slice(int(0.35 * sample_rate), int(1.2 * sample_rate))
    str_sus = _rms(direct_stem[sustain_window])
    body_sus = _rms(
        (
            stem_gains["bridge_body_stem"] * bridge_stem
            + stem_gains["top_radiation_stem"] * top_stem
            + stem_gains["soundhole_air_stem"] * soundhole_stem
            + stem_gains["cavity_body_tail_stem"] * cavity_stem
        )[sustain_window]
    )
    string_dom = str_sus / max(str_sus + body_sus, 1e-12)

    balance = compute_balance_diagnostics(
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

    norm_desc = (
        "sustain-window RMS normalization (200–800 ms) with attack soft-limit — avoids pluck-dominated peak norm"
        if norm_method == "sustain_window_rms"
        else "single global timbre-preserving finalize on stem sum (stems not individually RMS-normalized)"
    )

    meta = {
        "diagnostic_mode": diagnostic_mode,
        "v6_2_version": version_tag,
        "user_label": user_label,
        "variant": variant,
        "sample_id": sample_id,
        "frequency_hz": frequency_hz,
        "duration_s": duration_s,
        "pluck_params": {**pluck_meta, **direct_meta, "pluck_gain": stem_gains["pluck_attack_stem"]},
        "bridge_meta": bridge_meta,
        "stem_gains": stem_gains,
        "stem_contribution_ratios": stem_ratios,
        "finalize_info": finalize_info,
        "normalization": norm_desc,
        "norm_method": norm_method,
        "back_side_stem": "omitted — no per-sample back radiation path in audit; back_share used only via reference routing ratios",
        "feature_provenance_used": provenance_used,
        "string_dominance_ratio_sustain_window": round(string_dom, 6),
        "body_to_string_energy_ratio_sustain": round(body_sus / max(str_sus, 1e-12), 6),
        "pluck_attack_contribution": stem_ratios.get("pluck_attack_stem"),
        "direct_string_contribution": stem_ratios.get("direct_string_short_stem"),
        "bridge_body_contribution": stem_ratios.get("bridge_body_stem"),
        "top_radiation_contribution": stem_ratios.get("top_radiation_stem"),
        "soundhole_air_contribution": stem_ratios.get("soundhole_air_stem"),
        "cavity_body_tail_contribution": stem_ratios.get("cavity_body_tail_stem"),
        "cavity_contribution_proxy": round(
            (stem_rms.get("cavity_body_tail_stem", 0) + stem_rms.get("soundhole_air_stem", 0))
            / total_stem_rms,
            6,
        ),
        "soundhole_contribution_proxy": stem_ratios.get("soundhole_air_stem"),
        "top_radiation_contribution_proxy": stem_ratios.get("top_radiation_stem"),
        "balance_diagnostics": balance,
        "peak_dbfs": peak_db,
        "clipping_avoided": clip_ok,
        "limitations": [
            "V6.2/V6.2.1 does not prove multi-guitar differentiation yet.",
            "Reference modal catalog used for routing architecture (reference_shared features).",
            "Single guitar sample_000 only.",
        ],
    }
    if "cavity_body_tail_contribution" not in meta or meta.get("cavity_body_tail_contribution") is None:
        meta["cavity_body_tail_contribution"] = stem_ratios.get("cavity_body_tail_stem")
    if is_v621:
        meta["v6_2_1_changes_from_v6_2"] = [
            "Reduced pluck_attack_stem gain and derivative click component",
            "Smoother pluck envelope; longer direct_string decay (180–350 ms gate)",
            "Raised cavity/top/soundhole sustain stem gains",
            "Sustain-window RMS normalization instead of attack-dominated peak norm",
        ]
    stems["final_mix"] = final
    return stems, final, meta


def _feature_used_for(name: str) -> str:
    mapping = {
        "body_depth": "cavity/soundhole volume proxy",
        "body_volume_proxy": "Helmholtz / cavity tail",
        "soundhole_area": "soundhole_air_stem aperture",
        "helmholtz_like_frequency_proxy": "soundhole_air_stem resonance",
        "cavity_decay_proxy": "cavity_body_tail_stem decay",
        "cavity_q_proxy": "air/cavity resonator Q",
        "bridge_mobility_proxy": "bridge_body_stem scaling",
        "high_frequency_absorption_proxy": "E5 HF damping all paths",
        "top_radiation_gain_proxy": "top_radiation_stem (reference_shared)",
        "soundhole_radiation_gain_proxy": "soundhole_air_stem (reference_shared)",
        "top_to_back_ratio": "routing reference only (reference_shared)",
        "air_to_structural_ratio": "routing reference only (reference_shared)",
    }
    return mapping.get(name, "V6.2 routing context")


def metrics_for_stems_and_final(
    stems: Mapping[str, np.ndarray],
    *,
    sample_rate: int,
    frequency_hz: float,
    duration_s: float,
    final_meta: Mapping[str, Any],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"stems": {}, "final_mix": {}}
    for name in STEM_NAMES + ("final_mix",):
        if name not in stems:
            continue
        m = compute_stem_metrics(
            stems[name],
            sample_rate=sample_rate,
            frequency_hz=frequency_hz,
            duration_s=duration_s,
        )
        if name == "final_mix":
            str_rms = float(final_meta.get("direct_string_contribution") or 0.1)
            bod_rms = 1.0 - str_rms
            rm = compute_realism_metrics(
                stems[name],
                sample_rate=sample_rate,
                frequency_hz=frequency_hz,
                string_rms=str_rms,
                body_rms=bod_rms,
            )
            m.update(enrich_metrics_with_levels(rm, stems[name]))
            m["string_dominance_ratio"] = final_meta.get("string_dominance_ratio_sustain_window")
            m["body_to_string_energy_ratio"] = final_meta.get("body_to_string_energy_ratio_sustain")
            for k in (
                "stem_contribution_ratios",
                "pluck_attack_contribution",
                "direct_string_contribution",
                "bridge_body_contribution",
                "top_radiation_contribution",
                "soundhole_air_contribution",
                "cavity_body_tail_contribution",
                "cavity_contribution_proxy",
                "soundhole_contribution_proxy",
                "top_radiation_contribution_proxy",
                "balance_diagnostics",
                "norm_method",
            ):
                if k in final_meta:
                    m[k] = final_meta[k]
            out["final_mix"] = m
        else:
            out["stems"][name] = m
    return out
