#!/usr/bin/env python3
"""
Classic guitar body-response synthesizer (modal transfer-function model).

Signal path (physically motivated):
  plucked-string harmonics -> bridge acceleration
  -> H_body(f) = sum_m W_m H_m(f)  [primary timbre / radiation]
  + small direct attack tap      [pitch anchor / pluck clarity only]

All modes in the validated body band contribute. Final RMS target + soft limiter.
"""
from __future__ import annotations

import json
import math
import wave
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

FULL_MODAL_BAND_HZ: Tuple[float, float] = (60.0, 550.0)
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_DURATION_S = 3.0
DEFAULT_VELOCITY = 1.0

# Fixed excitation (body ROM weights are the main guitar-to-guitar variable).
FIXED_PLUCK_POSITION = 0.18
BODY_REFERENCE_GAIN = 1.0
BODY_MODAL_RICHNESS_GAIN = 1.20
BODY_MODAL_GAIN = 1.0
BODY_TO_STRING_TARGET_RATIO = 4.2
STRING_PLUCK_GAIN = 0.10
STRING_PITCH_LAYER_GAIN = 0.055
STRING_PITCH_LAYER_DECAY_S = 0.14
BODY_MODAL_BANDWIDTH_WIDENING = 1.12
TOP_MODE_DOMINANCE_SOFTEN = 0.14
MODAL_MAG_SMOOTH_BINS = 3
FADE_IN_MS = 5.0
FADE_OUT_MS_MIN = 35.0
FADE_OUT_MS_MAX = 70.0
FADE_OUT_FRAC_OF_DURATION = 0.025
PREVIEW_CROSSFADE_MS = 10.0
PREVIEW_SILENCE_MS = 40.0
# Backward-compatible aliases used in older metadata/tests.
TARGET_BODY_TO_ATTACK_RMS_RATIO = BODY_TO_STRING_TARGET_RATIO
DIRECT_ATTACK_GAIN = STRING_PLUCK_GAIN
ATTACK_DECAY_S = 0.040
PLUCK_TRANSIENT_MS = 0.006
FIXED_RAD_K = 0.08
MAX_HARMONICS = 48
Q_MIN = 22.0
Q_MAX = 75.0
CONTRIBUTION_THRESHOLD_REL = 1e-5
TOP_CONTRIBUTING_MODES_N = 15
HARMONIC_ROLLOFF_POWER = 1.15
TARGET_RMS_DBFS = -18.0
FINAL_PEAK_CEILING_DBFS = -1.0
LOW_NOTE_FUNDAMENTAL_MAX_HZ = 165.0
FUNDAMENTAL_ANCHOR_GAIN = 0.14
FUNDAMENTAL_ANCHOR_DECAY_S = 1.35

# Temporal decay (note / harmonic / body radiation)
NOTE_DECAY_REF_HZ = 82.41
NOTE_DECAY_TAU_MIN_S = 0.42
NOTE_DECAY_TAU_MAX_S = 2.65
NOTE_DECAY_FREQ_POWER = 0.58
HARMONIC_DECAY_FACTOR = 0.42
PARTIAL_FREQ_DECAY_HZ = 280.0
BODY_DECAY_TAU_MIN_S = 0.32
BODY_DECAY_TAU_MAX_S = 2.35
BODY_DECAY_FREQ_POWER = 0.52
RADIATION_TAU_SHORTENING = 0.58
BODY_DECAY_LOW_NOTE_BLEND = 0.48
HIGH_NOTE_DECAY_THRESHOLD_HZ = 300.0
HIGH_NOTE_PLUCK_SOFTEN_THRESHOLD_HZ = 300.0
HIGH_NOTE_PLUCK_SOFTEN_FULL_HZ = 620.0
HIGH_NOTE_PLUCK_GAIN_FLOOR = 0.46
HIGH_NOTE_PLUCK_TRANSIENT_BOOST = 0.75
HIGH_NOTE_PLUCK_TRANSIENT_REDUCTION = 0.58
HIGH_NOTE_PITCH_LAYER_ATTACK_SOFTEN = 1.85
HIGH_NOTE_ATTACK_DECAY_SHORTEN = 0.44
HIGH_NOTE_HF_ROLLOFF_K_POWER = 0.28
LOUDNESS_RMS_WINDOW_START_S = 0.025
LOUDNESS_RMS_WINDOW_END_S = 0.42
DECAY_EARLY_END_S = 0.28
DECAY_LATE_START_S = 2.05
DECAY_SLOPE_T_START_S = 0.10
DECAY_SLOPE_T_END_S = 2.55
HARMONIC_DECAY_MODEL = (
    "tau_k = note_base_tau(f0) / (1 + harmonic_decay_factor*(k-1)); "
    "partial_freq_scale; body_env = exp(-t/tau_body(f0,radiation))"
)

ModalInput = Union[Mapping[str, Any], Sequence[Mapping[str, Any]]]


def _safe_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        out = float(val)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(math.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def _rms_window(
    x: np.ndarray,
    sample_rate: int,
    start_s: float,
    end_s: float,
) -> float:
    n = len(x)
    if n == 0:
        return 0.0
    i0 = max(0, min(n, int(start_s * sample_rate)))
    i1 = max(i0 + 1, min(n, int(end_s * sample_rate)))
    return _rms(x[i0:i1])


def _linear_from_dbfs(dbfs: float) -> float:
    return float(10.0 ** (dbfs / 20.0))


def _dbfs_from_linear(amplitude: float) -> float:
    if amplitude <= 1e-12:
        return -120.0
    return float(20.0 * math.log10(amplitude))


def _mode_radiation_damping_scale(mode: Mapping[str, Any]) -> float:
    """Air/radiation increases energy loss (lower effective Q)."""
    rad = _safe_float(mode.get("radiation_proxy"))
    air = _safe_float(mode.get("air_share"))
    mic = _safe_float(mode.get("mic_output_proxy"))
    scale = 1.0
    if rad is not None and rad > 0:
        scale += 0.65 * min(rad, 0.05) / 0.05
    if air is not None and air > 0:
        scale += 0.45 * min(air, 0.5)
    if mic is not None and mic > 0:
        scale += 0.20 * min(mic, 0.05) / 0.05
    return max(1.0, min(2.8, scale))


def parse_modal_modes(modal_data: ModalInput) -> Tuple[List[Dict[str, Any]], List[str]]:
    defaults: List[str] = []
    if isinstance(modal_data, list):
        modes = [_normalize_mode_record(m, defaults) for m in modal_data]
        return [m for m in modes if m.get("frequency_hz")], defaults

    if not isinstance(modal_data, dict):
        raise TypeError(f"modal_data must be dict or list, got {type(modal_data)}")

    if modal_data.get("predicted_modes"):
        modes = [_normalize_mode_record(m, defaults) for m in modal_data["predicted_modes"]]
        return [m for m in modes if m.get("frequency_hz")], defaults

    if modal_data.get("modes"):
        modes = [_normalize_mode_record(m, defaults) for m in modal_data["modes"]]
        return [m for m in modes if m.get("frequency_hz")], defaults

    freqs = modal_data.get("modes_hz") or modal_data.get("frequencies_hz") or []
    weights = modal_data.get("mode_weights") or []
    modes = []
    for i, raw_f in enumerate(freqs):
        f_hz = _safe_float(raw_f)
        if f_hz is None or f_hz <= 0:
            continue
        rec: Dict[str, Any] = {"frequency_hz": f_hz, "mode_index": i}
        w = _safe_float(weights[i]) if i < len(weights) else None
        if w is not None:
            rec["mode_weight_fallback"] = w
        else:
            defaults.append("mode_weights:rolloff_fallback")
        modes.append(rec)
    if modes and not any("mode_weight_fallback" in m for m in modes):
        defaults.append("mode_weights:1/(1+0.25*i)")
    return modes, defaults


def _normalize_mode_record(raw: Mapping[str, Any], defaults: List[str]) -> Dict[str, Any]:
    rec = dict(raw)
    f_hz = _safe_float(rec.get("frequency_hz") or rec.get("freq_hz") or rec.get("f_hz"))
    if f_hz is not None:
        rec["frequency_hz"] = f_hz
    return rec


def modes_in_validated_band(modes: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    lo, hi = FULL_MODAL_BAND_HZ
    out: List[Dict[str, Any]] = []
    for m in modes:
        f = _safe_float(m.get("frequency_hz"))
        if f is None or f < lo or f > hi:
            continue
        out.append(dict(m))
    out.sort(key=lambda r: float(r["frequency_hz"]))
    return out


def available_modal_stats(modes: Sequence[Mapping[str, Any]]) -> Tuple[int, Optional[float], Optional[float]]:
    freqs = [float(m["frequency_hz"]) for m in modes if _safe_float(m.get("frequency_hz"))]
    if not freqs:
        return 0, None, None
    return len(freqs), min(freqs), max(freqs)


def compute_mode_weight_components(
    mode: Mapping[str, Any],
    *,
    defaults_used: List[str],
    flags: Dict[str, bool],
) -> Dict[str, float]:
    bridge_w = 1.0
    mic_w = 1.0
    rad_w = 1.0

    combined = _safe_float(mode.get("bridge_to_mic_gain_raw"))
    if combined is not None and combined > 0:
        flags["bridge_weighting_used"] = True
        flags["mic_proxy_used"] = True
        return {
            "bridge_weight": 1.0,
            "mic_weight": 1.0,
            "radiation_weight": 1.0,
            "combined": combined,
        }

    bridge = _safe_float(mode.get("bridge_excitation_abs"))
    if bridge is None:
        coup = _safe_float(mode.get("bridge_excitation_coupling"))
        bridge = abs(coup) if coup is not None else None
    if bridge is None or bridge <= 0:
        bridge = 1.0
        defaults_used.append("bridge_excitation_abs=1.0")
    else:
        flags["bridge_weighting_used"] = True
    bridge_w = bridge

    mic = _safe_float(mode.get("mic_output_proxy"))
    if mic is not None and mic > 0:
        flags["mic_proxy_used"] = True
        mic_w = mic
    else:
        mic_w = 1.0
        defaults_used.append("mic_output_proxy=1.0")

    rad = _safe_float(mode.get("radiation_proxy"))
    if rad is not None and rad > 0:
        flags["radiation_proxy_used"] = True
        rad_w = rad
    else:
        rad_w = 1.0
        defaults_used.append("radiation_proxy=1.0")

    w = bridge_w * (0.55 * mic_w + 0.45 * rad_w)
    fallback = _safe_float(mode.get("mode_weight_fallback"))
    if fallback is not None and fallback > 0:
        w *= fallback

    top = _safe_float(mode.get("top_share"))
    back = _safe_float(mode.get("back_share"))
    air = _safe_float(mode.get("air_share"))
    if top is not None or back is not None or air is not None:
        flags["participation_used"] = True
        share_sum = (top or 0.0) + (back or 0.0) + (air or 0.0)
        w *= max(0.35, min(1.2, 0.5 + 0.5 * share_sum))

    return {
        "bridge_weight": bridge_w,
        "mic_weight": mic_w,
        "radiation_weight": rad_w,
        "combined": max(w, 1e-12),
    }


def estimate_mode_q(mode: Mapping[str, Any], f_hz: float, defaults_used: List[str]) -> float:
    damp_scale = _mode_radiation_damping_scale(mode)
    for key in ("Q", "q", "modal_q", "quality_factor"):
        q = _safe_float(mode.get(key))
        if q is not None and q > 0:
            return float(max(Q_MIN, min(Q_MAX, q / damp_scale)))

    air = _safe_float(mode.get("air_share")) or 0.22
    wood = (_safe_float(mode.get("top_share")) or 0.33) + (_safe_float(mode.get("back_share")) or 0.33)
    lo, hi = FULL_MODAL_BAND_HZ
    f_norm = min(max((f_hz - lo) / max(hi - lo, 1.0), 0.0), 1.0)
    q_est = (42.0 + 28.0 * air + 10.0 * wood) * (1.0 - 0.32 * f_norm)
    defaults_used.append("Q_estimated_from_frequency_and_participation")
    if damp_scale > 1.01:
        defaults_used.append("radiation_air_damping_applied_to_Q")
    return float(max(Q_MIN, min(Q_MAX, q_est / damp_scale)))


def _total_q_with_radiation_loss(
    q_wood: float,
    f_hz: float,
    rad_k: float,
    *,
    radiation_proxy: float = 0.0,
) -> float:
    inv_q = (1.0 / max(q_wood, 0.5)) + rad_k * (f_hz / 1000.0)
    if radiation_proxy > 0:
        inv_q += 0.035 * radiation_proxy * math.sqrt(max(f_hz, 1.0) / 200.0)
    return max(0.5, 1.0 / max(inv_q, 1e-9))


def _effective_q_with_bandwidth_widening(q: float) -> float:
    return max(0.5, float(q) / BODY_MODAL_BANDWIDTH_WIDENING)


def _complex_mode_response(f_hz: np.ndarray, f_m: float, q: float) -> np.ndarray:
    fm = max(float(f_m), 1.0)
    qv = max(_effective_q_with_bandwidth_widening(q), 0.5)
    r = np.asarray(f_hz, dtype=np.float64) / fm
    denom = (1.0 - r * r) + 1.0j * (r / qv)
    return 1.0 / denom


def _soften_mode_weights(weights: Sequence[float]) -> Tuple[List[float], float, float]:
    """Gently reduce top-mode dominance while preserving relative differences."""
    w = np.asarray(weights, dtype=np.float64)
    total = float(np.sum(w))
    if total <= 0.0 or w.size == 0:
        return list(w), 0.0, 0.0
    dom_before = float(np.max(w) / total)
    w_norm = w / total
    w_max = float(np.max(w_norm))
    if w_max <= 1e-12:
        return list(w), dom_before, dom_before
    excess = np.maximum(w_norm / w_max - 0.35, 0.0)
    soften = 1.0 - TOP_MODE_DOMINANCE_SOFTEN * (excess ** 0.65)
    w_new = w * np.maximum(soften, 0.52)
    total_after = float(np.sum(w_new))
    dom_after = float(np.max(w_new) / total_after) if total_after > 0 else dom_before
    return [float(v) for v in w_new], dom_before, dom_after


def _smooth_complex_magnitude(H: np.ndarray, bins: int) -> np.ndarray:
    if bins < 2 or H.size < bins:
        return H
    mag = np.abs(H)
    phase = np.angle(H)
    kernel = np.ones(int(bins), dtype=np.float64) / float(bins)
    mag_s = np.convolve(mag, kernel, mode="same")
    return mag_s * np.exp(1.0j * phase)


def _hf_transfer_envelope(f_hz: np.ndarray) -> np.ndarray:
    _, hi = FULL_MODAL_BAND_HZ
    f = np.asarray(f_hz, dtype=np.float64)
    env = np.ones_like(f)
    above = f > hi
    env[above] = np.maximum(0.06, (hi / f[above]) ** 1.15)
    return env


def high_note_pluck_soften_t(frequency_hz: float) -> float:
    """0 below threshold, 1 at/above full soften frequency."""
    f0 = float(frequency_hz)
    if f0 <= HIGH_NOTE_PLUCK_SOFTEN_THRESHOLD_HZ:
        return 0.0
    span = max(1.0, HIGH_NOTE_PLUCK_SOFTEN_FULL_HZ - HIGH_NOTE_PLUCK_SOFTEN_THRESHOLD_HZ)
    return float(max(0.0, min(1.0, (f0 - HIGH_NOTE_PLUCK_SOFTEN_THRESHOLD_HZ) / span)))


def high_note_pluck_softening_gain(frequency_hz: float) -> float:
    """Reduce direct string/pluck layer for high notes; low notes unchanged."""
    t = high_note_pluck_soften_t(frequency_hz)
    if t <= 0.0:
        return 1.0
    return 1.0 - t * (1.0 - HIGH_NOTE_PLUCK_GAIN_FLOOR)


def high_note_string_hf_rolloff_factor(frequency_hz: float, harmonic_index: int) -> float:
    """Extra HF harmonic attenuation for high fundamentals (reduces metallic ping)."""
    f0 = float(frequency_hz)
    t = high_note_pluck_soften_t(f0)
    if t <= 0.0:
        return 1.0
    k = max(1, int(harmonic_index))
    fk = k * f0
    if k <= 2:
        return 1.0 - 0.10 * t
    hf = max(0.0, min(1.0, (fk - 700.0) / 2200.0))
    cut = t * (HIGH_NOTE_HF_ROLLOFF_K_POWER * (k - 2) ** 0.55 + 0.22 * hf)
    return float(max(0.30, 1.0 - cut))


def harmonic_series(
    frequency_hz: float,
    sample_rate: int,
    *,
    pluck_position: float = FIXED_PLUCK_POSITION,
    max_harmonics: int = MAX_HARMONICS,
) -> Tuple[List[float], List[float]]:
    f0 = max(1.0, float(frequency_hz))
    harm_f: List[float] = []
    harm_a: List[float] = []
    max_harm = min(max_harmonics, int(sample_rate / (2.0 * f0)))
    for k in range(1, max_harm + 1):
        fk = k * f0
        if fk >= sample_rate * 0.49:
            break
        pluck_factor = abs(math.sin(math.pi * pluck_position * k))
        if pluck_factor < 1e-8:
            continue
        amp = pluck_factor / (k ** HARMONIC_ROLLOFF_POWER)
        amp *= high_note_string_hf_rolloff_factor(f0, k)
        if k == 1 and f0 <= LOW_NOTE_FUNDAMENTAL_MAX_HZ:
            amp *= 1.55
        harm_f.append(fk)
        harm_a.append(amp)
    return harm_f, harm_a


def nearest_harmonic_hz(mode_hz: float, f0: float, harmonics_hz: Sequence[float]) -> float:
    if not harmonics_hz:
        return f0
    return min(harmonics_hz, key=lambda h: abs(h - mode_hz))


def note_base_decay_tau_s(frequency_hz: float) -> float:
    """Longer sustain for low notes; shorter for high notes."""
    f0 = max(40.0, float(frequency_hz))
    tau = NOTE_DECAY_TAU_MAX_S * (NOTE_DECAY_REF_HZ / f0) ** NOTE_DECAY_FREQ_POWER
    return float(max(NOTE_DECAY_TAU_MIN_S, min(NOTE_DECAY_TAU_MAX_S, tau)))


def harmonic_decay_tau_s(frequency_hz: float, harmonic_index: int) -> float:
    """Per-partial decay: higher k and higher partial frequency → shorter tau."""
    f0 = max(40.0, float(frequency_hz))
    k = max(1, int(harmonic_index))
    base = note_base_decay_tau_s(f0)
    tau = base / (1.0 + HARMONIC_DECAY_FACTOR * (k - 1))
    fk = k * f0
    tau /= 1.0 + 0.22 * (fk / PARTIAL_FREQ_DECAY_HZ) ** 0.9
    if k == 1 and f0 <= LOW_NOTE_FUNDAMENTAL_MAX_HZ:
        tau *= 1.28
    return float(max(0.07, tau))


def summarize_body_radiation(band_modes: Sequence[Mapping[str, Any]]) -> float:
    """0..1 summary of air/radiation heaviness across evaluated modes."""
    if not band_modes:
        return 0.0
    weights: List[float] = []
    rad_vals: List[float] = []
    for mode in band_modes:
        comp = compute_mode_weight_components(mode, defaults_used=[], flags={})
        w = comp["combined"]
        rad = _safe_float(mode.get("radiation_proxy")) or 0.0
        air = _safe_float(mode.get("air_share")) or 0.0
        rad_vals.append(min(1.0, 0.55 * min(rad / 0.05, 1.0) + 0.45 * min(air, 0.5)))
        weights.append(w)
    wsum = sum(weights)
    if wsum <= 0:
        return float(np.mean(rad_vals)) if rad_vals else 0.0
    return float(sum(r * wt for r, wt in zip(rad_vals, weights)) / wsum)


def body_decay_tau_s(note_hz: float, radiation_summary: float) -> float:
    """Body/radiation envelope time constant — high notes and radiating bodies decay faster."""
    f0 = max(40.0, float(note_hz))
    tau = BODY_DECAY_TAU_MAX_S * (NOTE_DECAY_REF_HZ / f0) ** BODY_DECAY_FREQ_POWER
    rad = max(0.0, min(1.0, float(radiation_summary)))
    shorten = 1.0 - rad * RADIATION_TAU_SHORTENING
    tau *= max(0.35, shorten)
    if f0 > HIGH_NOTE_DECAY_THRESHOLD_HZ:
        tau *= (HIGH_NOTE_DECAY_THRESHOLD_HZ / f0) ** 0.35
    return float(max(BODY_DECAY_TAU_MIN_S, min(BODY_DECAY_TAU_MAX_S, tau)))


def apply_exponential_decay_envelope(
    signal: np.ndarray,
    sample_rate: int,
    tau_s: float,
    *,
    floor_mix: float = 0.0,
) -> np.ndarray:
    n = len(signal)
    if n == 0 or tau_s <= 0:
        return signal
    t = np.arange(n, dtype=np.float64) / float(sample_rate)
    # Slight curvature: faster initial loss, smooth tail (not a hard gate).
    env = np.exp(-t / tau_s) * (0.9 + 0.1 * np.exp(-t / max(tau_s * 0.12, 1e-4)))
    if floor_mix > 0:
        env = (1.0 - floor_mix) + floor_mix * env
    return np.asarray(signal, dtype=np.float64) * env


def _decay_analysis_windows(duration_s: float) -> Tuple[float, float, float, float]:
    """Early/late RMS and log-slope fit windows scaled to note length."""
    dur = max(0.1, float(duration_s))
    early_end = min(DECAY_EARLY_END_S, max(0.08, dur * 0.14))
    late_start = DECAY_LATE_START_S if dur >= 1.5 else dur * 0.58
    late_start = min(max(late_start, early_end + 0.05), max(early_end + 0.05, dur - 0.04))
    t_start = min(DECAY_SLOPE_T_START_S, max(0.03, dur * 0.06))
    t_end = max(t_start + 0.12, min(DECAY_SLOPE_T_END_S, dur * 0.94))
    return early_end, late_start, t_start, t_end


def compute_decay_diagnostics(
    samples: np.ndarray,
    sample_rate: int,
) -> Dict[str, Any]:
    x = np.asarray(samples, dtype=np.float64)
    duration_s = len(x) / float(sample_rate) if sample_rate > 0 else 0.0
    early_end, late_start, t_start, t_end = _decay_analysis_windows(duration_s)
    early_rms = _rms_window(x, sample_rate, 0.0, early_end)
    late_rms = _rms_window(x, sample_rate, late_start, duration_s)
    if early_rms > 1e-12 and late_rms > 0:
        late_to_early_db = 20.0 * math.log10(late_rms / early_rms)
    else:
        late_to_early_db = -120.0

    slope = _estimate_decay_slope_db_per_s(x, sample_rate, t_start_s=t_start, t_end_s=t_end)
    return {
        "output_decay_slope_db_per_s": round(slope, 4),
        "early_rms_dbfs": round(_dbfs_from_linear(early_rms), 4),
        "late_rms_dbfs": round(_dbfs_from_linear(late_rms), 4),
        "late_to_early_rms_db": round(late_to_early_db, 4),
    }


def _estimate_decay_slope_db_per_s(
    samples: np.ndarray,
    sample_rate: int,
    *,
    t_start_s: float = DECAY_SLOPE_T_START_S,
    t_end_s: float = DECAY_SLOPE_T_END_S,
) -> float:
    n = len(samples)
    i0 = max(0, min(n, int(t_start_s * sample_rate)))
    i1 = max(i0 + 16, min(n, int(t_end_s * sample_rate)))
    if i1 <= i0 + 16:
        return 0.0
    seg = np.abs(samples[i0:i1])
    win = max(1, int(0.018 * sample_rate))
    kernel = np.ones(win, dtype=np.float64) / float(win)
    env = np.convolve(seg, kernel, mode="same")
    env = np.maximum(env, 1e-12)
    log_env = 20.0 * np.log10(env)
    t = np.arange(len(log_env), dtype=np.float64) / float(sample_rate) + t_start_s
    slope, _ = np.polyfit(t, log_env, 1)
    return float(slope)


def _pluck_attack_envelope(n: int, sample_rate: int, frequency_hz: float) -> np.ndarray:
    """Short onset emphasis for pluck realism; softer transient for high notes."""
    t = np.arange(n, dtype=np.float64) / float(sample_rate)
    soften_t = high_note_pluck_soften_t(frequency_hz)
    boost = HIGH_NOTE_PLUCK_TRANSIENT_BOOST * (1.0 - soften_t * HIGH_NOTE_PLUCK_TRANSIENT_REDUCTION)
    transient = 1.0 + boost * np.exp(-t / max(PLUCK_TRANSIENT_MS, 1e-4))
    return transient


def _fundamental_pitch_anchor(
    frequency_hz: float,
    duration_s: float,
    sample_rate: int,
    *,
    velocity: float = DEFAULT_VELOCITY,
) -> np.ndarray:
    """Subtle low-note fundamental anchor — not a dominant pure sine."""
    f0 = float(frequency_hz)
    if f0 > LOW_NOTE_FUNDAMENTAL_MAX_HZ:
        return np.zeros(max(1, int(duration_s * sample_rate)), dtype=np.float64)
    n = max(1, int(duration_s * sample_rate))
    t = np.arange(n, dtype=np.float64) / float(sample_rate)
    attack = 1.0 - np.exp(-t / 0.010)
    decay = np.exp(-t / FUNDAMENTAL_ANCHOR_DECAY_S)
    blend = 0.65 + 0.35 * np.exp(-t / 0.25)
    return (
        FUNDAMENTAL_ANCHOR_GAIN
        * velocity
        * blend
        * np.sin(2.0 * math.pi * f0 * t)
        * attack
        * decay
    )


def _attack_decay_s_for_note(frequency_hz: float) -> float:
    """High notes: faster attack decay → less harsh metallic pick transient."""
    soften_t = high_note_pluck_soften_t(frequency_hz)
    if soften_t <= 0.0:
        return ATTACK_DECAY_S
    return ATTACK_DECAY_S * (1.0 - soften_t * HIGH_NOTE_ATTACK_DECAY_SHORTEN)


def _direct_attack_tap(dry: np.ndarray, sample_rate: int, frequency_hz: float) -> np.ndarray:
    """Direct string component: short pluck attack for pitch identity."""
    t = np.arange(len(dry), dtype=np.float64) / float(sample_rate)
    attack_env = np.exp(-t / _attack_decay_s_for_note(frequency_hz))
    return dry * attack_env


def _string_pitch_layer(dry: np.ndarray, sample_rate: int, frequency_hz: float) -> np.ndarray:
    """Controlled harmonic string layer — plucked, not a pure sustained sine."""
    t = np.arange(len(dry), dtype=np.float64) / float(sample_rate)
    soften_t = high_note_pluck_soften_t(frequency_hz)
    attack_tc = 0.008 * (1.0 + soften_t * (HIGH_NOTE_PITCH_LAYER_ATTACK_SOFTEN - 1.0))
    attack = 1.0 - np.exp(-t / attack_tc)
    decay = np.exp(-t / STRING_PITCH_LAYER_DECAY_S)
    return dry * attack * decay


def fade_out_ms_for_duration(duration_s: float) -> float:
    ms = float(duration_s) * 1000.0 * FADE_OUT_FRAC_OF_DURATION
    return float(max(FADE_OUT_MS_MIN, min(FADE_OUT_MS_MAX, ms)))


def apply_anti_click_taper(
    samples: np.ndarray,
    sample_rate: int,
    *,
    duration_s: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Short fade-in and release taper so WAV boundaries are near zero (anti-click)."""
    x = np.asarray(samples, dtype=np.float64).copy()
    n = len(x)
    if n == 0:
        return x, {
            "fade_in_ms": FADE_IN_MS,
            "fade_out_ms": 0.0,
            "anti_click_taper_applied": False,
            "end_abs_sample_before_taper": 0.0,
            "end_abs_sample_after_taper": 0.0,
        }

    dur = float(duration_s) if duration_s is not None else n / float(sample_rate)
    fade_in_n = max(1, min(n // 4, int(FADE_IN_MS * 1e-3 * sample_rate)))
    fade_out_ms = fade_out_ms_for_duration(dur)
    fade_out_n = max(1, min(n // 4, int(fade_out_ms * 1e-3 * sample_rate)))

    end_before = float(abs(x[-1]))
    if fade_in_n > 1:
        ramp = np.sin(np.linspace(0.0, 0.5 * math.pi, fade_in_n))
        x[:fade_in_n] *= ramp
    if fade_out_n > 1:
        ramp = np.sin(np.linspace(0.5 * math.pi, 0.0, fade_out_n))
        x[-fade_out_n:] *= ramp
    x[-1] = 0.0
    end_after = float(abs(x[-1]))

    return x, {
        "fade_in_ms": FADE_IN_MS,
        "fade_out_ms": round(fade_out_ms, 4),
        "anti_click_taper_applied": True,
        "end_abs_sample_before_taper": round(end_before, 8),
        "end_abs_sample_after_taper": round(end_after, 8),
    }


def read_wav_float_mono(path: Path) -> Tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        n = wf.getnframes()
        sr = wf.getframerate()
        raw = wf.readframes(n)
        width = wf.getsampwidth()
    if width != 2:
        raise ValueError(f"unsupported sample width {width} in {path}")
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
    return pcm / 32767.0, int(sr)


def concatenate_audio_with_crossfade(
    segments: Sequence[np.ndarray],
    sample_rate: int,
    *,
    crossfade_ms: float = PREVIEW_CROSSFADE_MS,
    silence_ms: float = PREVIEW_SILENCE_MS,
) -> np.ndarray:
    """Concatenate note segments with short silence and cosine crossfades."""
    if not segments:
        return np.zeros(0, dtype=np.float64)
    cross_n = max(0, int(crossfade_ms * 1e-3 * sample_rate))
    silence_n = max(0, int(silence_ms * 1e-3 * sample_rate))
    out: List[np.ndarray] = []
    for i, seg in enumerate(segments):
        s = np.asarray(seg, dtype=np.float64)
        if s.size == 0:
            continue
        if i > 0 and silence_n > 0:
            out.append(np.zeros(silence_n, dtype=np.float64))
        if out and cross_n > 1:
            prev = out[-1]
            if prev.size >= cross_n and s.size >= cross_n:
                fade_out = np.sin(np.linspace(0.5 * math.pi, 0.0, cross_n))
                fade_in = np.sin(np.linspace(0.0, 0.5 * math.pi, cross_n))
                overlap = prev[-cross_n:] * fade_out + s[:cross_n] * fade_in
                out[-1] = np.concatenate([prev[:-cross_n], overlap])
                s = s[cross_n:]
        out.append(s)
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float64)


def synthesize_plucked_string(
    frequency_hz: float,
    duration_s: float,
    sample_rate: int,
    *,
    pluck_position: float = FIXED_PLUCK_POSITION,
    velocity: float = DEFAULT_VELOCITY,
    amplitude: float = 0.38,
) -> np.ndarray:
    n = max(1, int(duration_s * sample_rate))
    t = np.arange(n, dtype=np.float64) / float(sample_rate)
    signal = np.zeros(n, dtype=np.float64)
    f0 = max(1.0, float(frequency_hz))
    harm_f, harm_a = harmonic_series(frequency_hz, sample_rate, pluck_position=pluck_position)
    for fk, ak in zip(harm_f, harm_a):
        amp_k = velocity * amplitude * ak
        k = max(1, int(round(fk / f0)))
        tau_k = harmonic_decay_tau_s(f0, k)
        signal += amp_k * np.sin(2.0 * math.pi * fk * t) * np.exp(-t / tau_k)
    signal *= _pluck_attack_envelope(n, sample_rate, f0)
    return signal


def _string_acceleration(dry: np.ndarray) -> np.ndarray:
    acc = np.zeros_like(dry)
    if len(dry) >= 3:
        acc[1:-1] = dry[:-2] - 2.0 * dry[1:-1] + dry[2:]
        acc[0] = acc[1]
        acc[-1] = acc[-2]
    return acc


def _high_note_hf_fallback(note_hz: float) -> bool:
    return float(note_hz) > FULL_MODAL_BAND_HZ[1]


def synthesize_body_via_transfer_function(
    acc: np.ndarray,
    sample_rate: int,
    band_modes: Sequence[Mapping[str, Any]],
    *,
    defaults_used: List[str],
    flags: Dict[str, bool],
    note_hz: float,
    harmonics_hz: Sequence[float],
) -> Tuple[np.ndarray, List[Dict[str, Any]], List[float], Dict[str, Any]]:
    """
    H_body(f) = sum_m W_m H_m(f) on bridge acceleration spectrum.
    Uses stable BODY_REFERENCE_GAIN — no per-guitar H normalization or 1/sqrt(N) scaling.
    """
    n = len(acc)
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sample_rate))
    acc_spec = np.fft.rfft(acc)
    hf_env = _hf_transfer_envelope(freqs)

    mode_rows: List[Dict[str, Any]] = []
    q_values: List[float] = []
    raw_weights: List[float] = []

    for mode in band_modes:
        f_m = float(mode["frequency_hz"])
        comp = compute_mode_weight_components(mode, defaults_used=defaults_used, flags=flags)
        w = comp["combined"]
        q_wood = estimate_mode_q(mode, f_m, defaults_used)
        q_total = _total_q_with_radiation_loss(
            q_wood,
            f_m,
            FIXED_RAD_K,
            radiation_proxy=float(comp.get("radiation_weight") or 0.0),
        )
        flags["q_or_damping_used"] = True
        q_values.append(q_total)
        raw_weights.append(w)
        H_m = _complex_mode_response(freqs, f_m, q_total)
        mode_rows.append(
            {
                "mode": mode,
                "f_m": f_m,
                "w": w,
                "comp": comp,
                "q": q_total,
                "H_m": H_m,
            }
        )

    softened_weights, dom_before, dom_after = _soften_mode_weights(raw_weights)
    if dom_after < dom_before - 1e-6:
        defaults_used.append("top_mode_dominance_softened")

    H_body = np.zeros_like(freqs, dtype=np.complex128)
    for row, w_eff in zip(mode_rows, softened_weights):
        H_body += w_eff * row["H_m"]
        row["w_eff"] = w_eff

    H_body *= hf_env
    if MODAL_MAG_SMOOTH_BINS >= 2:
        H_body = _smooth_complex_magnitude(H_body, MODAL_MAG_SMOOTH_BINS)
        defaults_used.append("modal_peak_smoothing_applied")
    body_spec = acc_spec * H_body * BODY_REFERENCE_GAIN
    body = np.fft.irfft(body_spec, n=n)

    contributions: List[Dict[str, Any]] = []
    f0 = max(float(note_hz), 1.0)
    for row in mode_rows:
        mode = row["mode"]
        w = float(row.get("w_eff", row["w"]))
        comp = row["comp"]
        q_total = row["q"]
        f_m = row["f_m"]
        H_m = row["H_m"] * hf_env
        mode_spec = acc_spec * w * H_m * BODY_REFERENCE_GAIN
        energy = float(np.sum(np.abs(mode_spec) ** 2))
        nearest_h = nearest_harmonic_hz(f_m, f0, harmonics_hz)
        contributions.append(
            {
                "mode_index": int(mode.get("mode_index", -1)),
                "frequency_hz": round(f_m, 4),
                "contribution_weight": energy,
                "bridge_weight": round(comp["bridge_weight"], 8),
                "mic_weight": round(comp["mic_weight"], 8),
                "radiation_weight": round(comp["radiation_weight"], 8),
                "q": round(q_total, 4),
                "nearest_harmonic_hz": round(nearest_h, 4),
            }
        )

    broaden_info = {
        "body_modal_bandwidth_widening": BODY_MODAL_BANDWIDTH_WIDENING,
        "modal_peak_smoothing_applied": MODAL_MAG_SMOOTH_BINS >= 2,
        "top_mode_dominance_before": round(dom_before, 6),
        "top_mode_dominance_after": round(dom_after, 6),
        "effective_q_scale_or_bandwidth_scale": BODY_MODAL_BANDWIDTH_WIDENING,
    }
    return body, contributions, q_values, broaden_info


def apply_loudness_finalize(
    samples: np.ndarray,
    sample_rate: int,
    *,
    target_rms_dbfs: float = TARGET_RMS_DBFS,
    peak_ceiling_dbfs: float = FINAL_PEAK_CEILING_DBFS,
    rms_window_start_s: float = LOUDNESS_RMS_WINDOW_START_S,
    rms_window_end_s: float = LOUDNESS_RMS_WINDOW_END_S,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Target-RMS gain (early/mid window), then tanh soft limiter, then peak ceiling.
    Windowed RMS avoids lifting a quiet tail when the attack/sustain is louder.
    """
    x = np.asarray(samples, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        raise ValueError("Samples contain NaN or Inf")

    target_rms = _linear_from_dbfs(target_rms_dbfs)
    ceiling = _linear_from_dbfs(peak_ceiling_dbfs)
    rms_window = _rms_window(x, sample_rate, rms_window_start_s, rms_window_end_s)
    rms_full = _rms(x)
    rms_in = math.sqrt(0.68 * rms_window**2 + 0.32 * rms_full**2)
    if rms_in < 1e-12:
        rms_in = max(rms_window, rms_full, 1e-12)
    rms_gain = target_rms / max(rms_in, 1e-12)
    y = x * rms_gain

    peak_pre_limit = float(np.max(np.abs(y))) if y.size else 0.0
    limiter_used = False
    limiter_gr_db = 0.0

    if peak_pre_limit > ceiling * 0.92:
        limiter_used = True
        drive = max(peak_pre_limit / ceiling, 1.0)
        y = ceiling * np.tanh(y / max(peak_pre_limit, 1e-12) * drive) / math.tanh(drive)
        peak_after = float(np.max(np.abs(y))) if y.size else 0.0
        if peak_after > ceiling:
            y *= ceiling / peak_after
            peak_after = ceiling
        if peak_pre_limit > 1e-12:
            limiter_gr_db = _dbfs_from_linear(peak_after) - _dbfs_from_linear(peak_pre_limit)
    else:
        peak_after = peak_pre_limit
        if peak_after > ceiling:
            limiter_used = True
            y *= ceiling / peak_after
            limiter_gr_db = _dbfs_from_linear(ceiling) - _dbfs_from_linear(peak_after)
            peak_after = ceiling

    rms_out = _rms(y)
    info = {
        "target_rms_dbfs": target_rms_dbfs,
        "final_peak_ceiling_dbfs": peak_ceiling_dbfs,
        "rms_gain_applied": rms_gain,
        "loudness_rms_window_s": [rms_window_start_s, rms_window_end_s],
        "peak_before_loudness": peak_pre_limit,
        "limiter_used": limiter_used,
        "limiter_gain_reduction_db": round(limiter_gr_db, 4),
        "output_rms_dbfs": round(_dbfs_from_linear(rms_out), 4),
        "output_peak_dbfs": round(_dbfs_from_linear(peak_after), 4),
        "peak_before_normalize": peak_pre_limit,
        "final_peak_normalization_gain": rms_gain,
    }
    info.update(compute_decay_diagnostics(y, sample_rate))
    return y, info


def write_wav_int16(
    path: Path,
    samples: np.ndarray,
    sample_rate: int,
    *,
    duration_s: Optional[float] = None,
) -> Dict[str, Any]:
    """Write mono int16 WAV after anti-click taper and loudness finalize."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tapered, taper_info = apply_anti_click_taper(
        samples,
        sample_rate,
        duration_s=duration_s,
    )
    y, loudness_info = apply_loudness_finalize(tapered, sample_rate)
    loudness_info.update(taper_info)
    pcm = np.clip(y * 32767.0, -32767, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())
    return loudness_info


def synthesize_note_with_body_response(
    frequency_hz: float,
    note_name: str,
    duration_s: float,
    sample_rate: int,
    modal_data: ModalInput,
    output_wav: Path,
    output_metadata_json: Optional[Path] = None,
    velocity: float = DEFAULT_VELOCITY,
) -> Dict[str, Any]:
    all_modes, parse_defaults = parse_modal_modes(modal_data)
    band_modes = modes_in_validated_band(all_modes)
    avail_n, avail_min, avail_max = available_modal_stats(all_modes)
    eval_n, eval_min, eval_max = available_modal_stats(band_modes)

    defaults_used: List[str] = list(parse_defaults)
    flags = {
        "bridge_weighting_used": False,
        "mic_proxy_used": False,
        "radiation_proxy_used": False,
        "participation_used": False,
        "q_or_damping_used": False,
    }

    harmonics_hz, _ = harmonic_series(frequency_hz, sample_rate)
    string_excitation = synthesize_plucked_string(
        frequency_hz,
        duration_s,
        sample_rate,
        pluck_position=FIXED_PLUCK_POSITION,
        velocity=velocity,
    )
    acc = _string_acceleration(string_excitation)

    radiation_summary = summarize_body_radiation(band_modes)
    note_decay_tau_s = note_base_decay_tau_s(frequency_hz)
    body_decay_tau_s_val = body_decay_tau_s(frequency_hz, radiation_summary)
    high_note_decay_applied = float(frequency_hz) >= HIGH_NOTE_DECAY_THRESHOLD_HZ

    broaden_info: Dict[str, Any] = {}
    if band_modes:
        body_raw, contributions, q_values, broaden_info = synthesize_body_via_transfer_function(
            acc,
            sample_rate,
            band_modes,
            defaults_used=defaults_used,
            flags=flags,
            note_hz=frequency_hz,
            harmonics_hz=harmonics_hz,
        )
        defaults_used.append(f"body_modal_bandwidth_widening={BODY_MODAL_BANDWIDTH_WIDENING}")
        body_floor = (
            BODY_DECAY_LOW_NOTE_BLEND if float(frequency_hz) <= LOW_NOTE_FUNDAMENTAL_MAX_HZ else 0.0
        )
        body_raw = apply_exponential_decay_envelope(
            body_raw,
            sample_rate,
            body_decay_tau_s_val,
            floor_mix=body_floor,
        )
        defaults_used.append("body_radiation_decay_envelope")
    else:
        body_raw = np.zeros_like(string_excitation)
        contributions = []
        q_values = []
        defaults_used.append("no_modes_in_validated_band_60_550:body_bypass")

    pluck_soften = high_note_pluck_softening_gain(frequency_hz)
    high_note_pluck_softening_applied = pluck_soften < 0.999
    hf_rolloff = high_note_string_hf_rolloff_factor(float(frequency_hz), 6)
    effective_pluck_gain = STRING_PLUCK_GAIN * pluck_soften
    effective_pitch_gain = STRING_PITCH_LAYER_GAIN * pluck_soften
    string_pluck = effective_pluck_gain * _direct_attack_tap(
        string_excitation, sample_rate, float(frequency_hz)
    )
    string_pitch_layer = effective_pitch_gain * _string_pitch_layer(
        string_excitation, sample_rate, float(frequency_hz)
    )
    string_path = string_pluck + string_pitch_layer
    string_pluck_gain = effective_pluck_gain
    string_pitch_layer_gain = effective_pitch_gain
    effective_string_pluck_gain = effective_pluck_gain + effective_pitch_gain
    body_rms_before_calibration = _rms(body_raw)
    string_rms_before_mix = _rms(string_path)

    if body_rms_before_calibration > 1e-15 and string_rms_before_mix > 1e-15:
        body_gain_applied = (
            BODY_TO_STRING_TARGET_RATIO * string_rms_before_mix / body_rms_before_calibration
        )
        defaults_used.append(
            f"body_gain_calibration_to_target_ratio={BODY_TO_STRING_TARGET_RATIO}"
        )
    elif body_rms_before_calibration > 0:
        body_gain_applied = BODY_REFERENCE_GAIN
        defaults_used.append("body_gain_calibration_fallback")
    else:
        body_gain_applied = 0.0

    body_signal = body_raw * body_gain_applied * BODY_MODAL_GAIN
    body_rms_before_richness_gain = _rms(body_signal)
    body_signal = body_signal * BODY_MODAL_RICHNESS_GAIN
    body_rms_after_richness_gain = _rms(body_signal)
    body_rms_before = body_rms_after_richness_gain
    body_modal_gain = BODY_MODAL_GAIN
    mixed = body_signal + string_path
    defaults_used.append(f"body_modal_richness_gain={BODY_MODAL_RICHNESS_GAIN}")
    body_to_string_rms_ratio_before_loudness = body_rms_before / max(string_rms_before_mix, 1e-12)
    fundamental_anchor_used = float(frequency_hz) <= LOW_NOTE_FUNDAMENTAL_MAX_HZ
    if fundamental_anchor_used:
        mixed += _fundamental_pitch_anchor(
            frequency_hz,
            duration_s,
            sample_rate,
            velocity=velocity,
        )
        defaults_used.append("low_note_fundamental_anchor")
    defaults_used.append("note_harmonic_frequency_decay_envelope")
    final_dry_to_body_rms_ratio = string_rms_before_mix / max(body_rms_before, 1e-12)

    loudness_info = write_wav_int16(
        Path(output_wav),
        mixed,
        sample_rate,
        duration_s=duration_s,
    )
    body_to_string_rms_ratio_after_loudness = body_to_string_rms_ratio_before_loudness

    hf_fallback = _high_note_hf_fallback(float(frequency_hz))
    max_contrib = max((c["contribution_weight"] for c in contributions), default=0.0)
    threshold = CONTRIBUTION_THRESHOLD_REL * max_contrib if max_contrib > 0 else 0.0
    active_n = sum(1 for c in contributions if c["contribution_weight"] >= threshold)

    top_modes = sorted(contributions, key=lambda c: c["contribution_weight"], reverse=True)[
        :TOP_CONTRIBUTING_MODES_N
    ]
    for row in top_modes:
        row["contribution_weight"] = round(row["contribution_weight"], 8)

    q_sorted = sorted(q_values)
    q_min = q_sorted[0] if q_sorted else None
    q_max = q_sorted[-1] if q_sorted else None
    q_median = q_sorted[len(q_sorted) // 2] if q_sorted else None

    metadata: Dict[str, Any] = {
        "note_name": note_name,
        "frequency_hz": float(frequency_hz),
        "duration_s": float(duration_s),
        "sample_rate": int(sample_rate),
        "pitch_preserved": True,
        "synthesis_model": "modal_transfer_function_H_body_sum_m_Wm_Hm",
        "available_modal_count": avail_n,
        "available_modal_frequency_min_hz": avail_min,
        "available_modal_frequency_max_hz": avail_max,
        "evaluated_modal_count": eval_n,
        "evaluated_modal_frequency_min_hz": eval_min,
        "evaluated_modal_frequency_max_hz": eval_max,
        "active_modal_count_after_threshold": active_n,
        "selected_or_pruned_policy": (
            "all_modes_in_validated_band_60_550_hz;"
            f"post_response_threshold_rel={CONTRIBUTION_THRESHOLD_REL};"
            "no_per_guitar_H_body_peak_normalize;"
            f"body_rms_calibration_target_ratio={BODY_TO_STRING_TARGET_RATIO}"
        ),
        "harmonics_used_hz": [round(h, 4) for h in harmonics_hz],
        "top_contributing_modes": top_modes,
        "full_modal_band_hz": list(FULL_MODAL_BAND_HZ),
        "high_frequency_fallback_used": bool(hf_fallback),
        "bridge_weighting_used": flags["bridge_weighting_used"],
        "mic_proxy_used": flags["mic_proxy_used"],
        "radiation_proxy_used": flags["radiation_proxy_used"],
        "q_or_damping_used": flags["q_or_damping_used"],
        "direct_string_role": "string_pluck_plus_pitch_layer",
        "string_pluck_gain": round(string_pluck_gain, 6),
        "string_pitch_layer_gain": round(string_pitch_layer_gain, 6),
        "high_note_pluck_softening_applied": high_note_pluck_softening_applied,
        "high_note_pluck_softening_gain": round(pluck_soften, 6),
        "string_hf_rolloff_factor": round(hf_rolloff, 6),
        "effective_string_pluck_gain": round(effective_string_pluck_gain, 6),
        "body_modal_gain": round(body_modal_gain, 6),
        "body_to_string_target_ratio": BODY_TO_STRING_TARGET_RATIO,
        "body_to_string_rms_ratio_before_loudness": round(body_to_string_rms_ratio_before_loudness, 6),
        "body_to_string_rms_ratio_after_loudness": round(body_to_string_rms_ratio_after_loudness, 6),
        "direct_string_gain": round(string_pluck_gain, 6),
        "body_filter_gain": round(BODY_REFERENCE_GAIN, 6),
        "body_rms_before_calibration": round(body_rms_before_calibration, 8),
        "body_modal_richness_gain": BODY_MODAL_RICHNESS_GAIN,
        "body_rms_before_richness_gain": round(body_rms_before_richness_gain, 8),
        "body_rms_after_richness_gain": round(body_rms_after_richness_gain, 8),
        "dry_mix": round(string_pluck_gain, 6),
        "wet_mix": round(body_gain_applied, 6),
        "dry_rms_before_mix": round(string_rms_before_mix, 8),
        "string_rms_before_mix": round(string_rms_before_mix, 8),
        "body_rms_before_mix": round(body_rms_before, 8),
        "target_body_to_attack_rms_ratio": BODY_TO_STRING_TARGET_RATIO,
        "dry_gain_applied": round(string_pluck_gain, 6),
        "body_gain_applied": round(body_gain_applied, 6),
        "final_dry_to_body_rms_ratio": round(final_dry_to_body_rms_ratio, 6),
        "fundamental_anchor_used": fundamental_anchor_used,
        "target_rms_dbfs": loudness_info["target_rms_dbfs"],
        "final_peak_ceiling_dbfs": loudness_info["final_peak_ceiling_dbfs"],
        "output_rms_dbfs": loudness_info["output_rms_dbfs"],
        "output_peak_dbfs": loudness_info["output_peak_dbfs"],
        "limiter_used": loudness_info["limiter_used"],
        "limiter_gain_reduction_db": loudness_info["limiter_gain_reduction_db"],
        "rms_gain_applied": round(loudness_info["rms_gain_applied"], 6),
        "final_peak_normalization_gain": round(loudness_info["final_peak_normalization_gain"], 6),
        "peak_before_normalize": loudness_info["peak_before_normalize"],
        "output_decay_slope_db_per_s": loudness_info["output_decay_slope_db_per_s"],
        "early_rms_dbfs": loudness_info["early_rms_dbfs"],
        "late_rms_dbfs": loudness_info["late_rms_dbfs"],
        "late_to_early_rms_db": loudness_info["late_to_early_rms_db"],
        "note_decay_tau_s": round(note_decay_tau_s, 4),
        "body_decay_tau_s": round(body_decay_tau_s_val, 4),
        "harmonic_decay_model": HARMONIC_DECAY_MODEL,
        "high_note_decay_applied": high_note_decay_applied,
        "body_radiation_summary": round(radiation_summary, 4),
        "body_modal_bandwidth_widening": broaden_info.get(
            "body_modal_bandwidth_widening", BODY_MODAL_BANDWIDTH_WIDENING
        ),
        "modal_peak_smoothing_applied": bool(
            broaden_info.get("modal_peak_smoothing_applied", MODAL_MAG_SMOOTH_BINS >= 2)
        ),
        "top_mode_dominance_before": broaden_info.get("top_mode_dominance_before"),
        "top_mode_dominance_after": broaden_info.get("top_mode_dominance_after"),
        "effective_q_scale_or_bandwidth_scale": broaden_info.get(
            "effective_q_scale_or_bandwidth_scale", BODY_MODAL_BANDWIDTH_WIDENING
        ),
        "fade_in_ms": loudness_info.get("fade_in_ms"),
        "fade_out_ms": loudness_info.get("fade_out_ms"),
        "anti_click_taper_applied": loudness_info.get("anti_click_taper_applied"),
        "end_abs_sample_before_taper": loudness_info.get("end_abs_sample_before_taper"),
        "end_abs_sample_after_taper": loudness_info.get("end_abs_sample_after_taper"),
        "q_min": q_min,
        "q_median": q_median,
        "q_max": q_max,
        "defaults_used": sorted(set(defaults_used)),
        "excitation": {
            "pluck_position": FIXED_PLUCK_POSITION,
            "velocity": float(velocity),
            "attack_decay_s": ATTACK_DECAY_S,
            "pluck_transient_ms": PLUCK_TRANSIENT_MS,
            "harmonic_rolloff_power": HARMONIC_ROLLOFF_POWER,
            "body_reference_gain": BODY_REFERENCE_GAIN,
            "string_pluck_gain": STRING_PLUCK_GAIN,
            "string_pitch_layer_gain": STRING_PITCH_LAYER_GAIN,
            "body_modal_gain": BODY_MODAL_GAIN,
            "body_to_string_target_ratio": BODY_TO_STRING_TARGET_RATIO,
            "body_modal_bandwidth_widening": BODY_MODAL_BANDWIDTH_WIDENING,
            "body_modal_richness_gain": BODY_MODAL_RICHNESS_GAIN,
            "rad_k": FIXED_RAD_K,
            "q_clamp": [Q_MIN, Q_MAX],
            "target_rms_dbfs": TARGET_RMS_DBFS,
            "peak_ceiling_dbfs": FINAL_PEAK_CEILING_DBFS,
            "fundamental_anchor_gain": FUNDAMENTAL_ANCHOR_GAIN,
            "note_decay_tau_s": round(note_decay_tau_s, 4),
            "body_decay_tau_s": round(body_decay_tau_s_val, 4),
            "harmonic_decay_factor": HARMONIC_DECAY_FACTOR,
            "loudness_rms_window_s": [
                LOUDNESS_RMS_WINDOW_START_S,
                LOUDNESS_RMS_WINDOW_END_S,
            ],
        },
        "output_wav": str(output_wav),
        "samples_finite": True,
    }
    if output_metadata_json is not None:
        output_metadata_json.parent.mkdir(parents=True, exist_ok=True)
        output_metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        metadata["output_metadata_json"] = str(output_metadata_json)
    return metadata


def synthetic_classic_body_modes(n_modes: int = 55) -> List[Dict[str, Any]]:
    lo, hi = FULL_MODAL_BAND_HZ
    modes: List[Dict[str, Any]] = []
    for i in range(n_modes):
        t = i / max(n_modes - 1, 1)
        f = lo + t * (hi - lo)
        modes.append(
            {
                "frequency_hz": round(f, 2),
                "mode_index": i,
                "bridge_excitation_abs": 0.012 + 0.008 * ((i % 3) + 1) / 3.0,
                "mic_output_proxy": 0.006 + 0.004 * ((i + 1) % 4) / 4.0,
                "radiation_proxy": 0.005 + 0.003 * (i % 2),
                "top_share": 0.38 + 0.04 * (i % 2),
                "back_share": 0.34,
                "air_share": 0.22 + 0.06 * (i % 3) / 3.0,
            }
        )
    return modes


def load_modal_data_from_path(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    if not isinstance(doc, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return doc


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Body-response note smoke test")
    parser.add_argument("--modal-json", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("audio/stage1_loudness"))
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_S)
    args = parser.parse_args()

    if args.modal_json and args.modal_json.is_file():
        modal_data = load_modal_data_from_path(args.modal_json)
    else:
        modal_data = {"predicted_modes": synthetic_classic_body_modes(), "analysis": "synthetic_fixture"}

    cases = (("E2", 82.41), ("A2", 110.0), ("A4", 440.0), ("E5", 659.25))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = args.out_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    for name, hz in cases:
        wav_path = args.out_dir / f"{name}_body.wav"
        meta_path = meta_dir / f"{name}_metadata.json"
        meta = synthesize_note_with_body_response(
            frequency_hz=hz,
            note_name=name,
            duration_s=args.duration,
            sample_rate=DEFAULT_SAMPLE_RATE,
            modal_data=modal_data,
            output_wav=wav_path,
            output_metadata_json=meta_path,
        )
        print(
            f"{name}: rms={meta['output_rms_dbfs']:.1f} dBFS peak={meta['output_peak_dbfs']:.1f} dBFS "
            f"slope={meta['output_decay_slope_db_per_s']:.1f} dB/s "
            f"late/early={meta['late_to_early_rms_db']:.1f} dB "
            f"limiter={meta['limiter_used']} hf={meta['high_frequency_fallback_used']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
