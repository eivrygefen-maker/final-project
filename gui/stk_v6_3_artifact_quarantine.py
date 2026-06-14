#!/usr/bin/env python3
"""
STK V6.3 — artifact quarantine, scanner, and clean pluck/body rebuild (diagnostic only).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from body_response_synth import (
    DEFAULT_SAMPLE_RATE,
    DEFAULT_VELOCITY,
    FIXED_PLUCK_POSITION,
    ModalInput,
    _rms,
    read_wav_float_mono,
    synthesize_plucked_string,
)
from sample_parameters import normalize_sample_parameters
from stk_v5_design_helpers import (
    _band_energy,
    _peak_dbfs,
    render_v5_alpha_body_radiation_path,
)
from stk_v6_2_audit_features import collect_features_for_synthesis, feature_value, get_sample_record
from stk_v6_2_physical_routing import DEFAULT_DURATION_S, _apply_note_hf_damping, _window_rms
from stk_v6_2_2_onset_tail_repair import compute_tail_continuity_diagnostics

V6_3_VERSION = "stk_v6_3_artifact_quarantine_v0"
V6_3_MODE = "stk_v6_3_clean_pluck_body_alpha"
V6_3_LABEL = "STK V6.3 clean pluck/body alpha"

REJECTED_V622_MODES = (
    "stk_v6_2_2_single_onset_soft_tail_alpha",
    "stk_v6_2_2_no_thump_body_tail_alpha",
    "stk_v6_2_2_v5_body_v6_pluck_hybrid_alpha",
)

REJECTED_V621_MODES = (
    "stk_v6_2_1_soft_pluck_tail_alpha",
    "stk_v6_2_1_balanced_tail_alpha",
    "stk_v6_2_1_more_string_body_alpha",
)

ARTIFACT_QUARANTINE: Dict[str, Any] = {
    "rejected_modes": list(REJECTED_V622_MODES) + list(REJECTED_V621_MODES),
    "reason": {
        "stk_v6_2_2_single_onset_soft_tail_alpha": [
            "double_pluck",
            "drum_tap",
            "thump",
            "delayed_body_pulse",
            "tail_collapse",
            "end_noise",
        ],
        "stk_v6_2_2_no_thump_body_tail_alpha": [
            "double_pluck",
            "drum_tap",
            "thump",
            "delayed_body_pulse",
            "tail_collapse",
        ],
        "stk_v6_2_2_v5_body_v6_pluck_hybrid_alpha": [
            "double_pluck",
            "drum_tap",
            "thump",
            "delayed_body_pulse",
            "tail_collapse",
            "end_noise",
        ],
        "stk_v6_2_1_soft_pluck_tail_alpha": [
            "double_pluck",
            "drum_tap",
            "thump",
            "tail_collapse",
        ],
        "stk_v6_2_1_balanced_tail_alpha": [
            "double_pluck",
            "drum_tap",
            "thump",
            "tail_collapse",
        ],
        "stk_v6_2_1_more_string_body_alpha": [
            "double_pluck",
            "drum_tap",
            "thump",
            "delayed_body_pulse",
            "tail_collapse",
        ],
    },
    "allowed_future_use": "baseline_only",
    "do_not_recommend": True,
}

# Hard acceptance thresholds for V6.3 clean candidate
ACCEPTANCE = {
    "second_onset_ratio_max": 0.35,
    "onset_peak_count_max": 1,
    "body_tail_peak_count_80_350ms_max": 0,
    "body_tail_impulse_ratio_max": 1.8,
    "tail_continuity_ratio_min": 0.12,
    "end_click_or_gate_fail": False,
    "final_discontinuity_max": 0.15,
}


def _band_rms(audio: np.ndarray, sample_rate: int, lo_hz: float, hi_hz: float) -> float:
    x = np.asarray(audio, dtype=np.float64)
    n = len(x)
    if n < 64:
        return 0.0
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sample_rate))
    band = (freqs >= lo_hz) & (freqs < hi_hz)
    spec_f = np.zeros_like(spec)
    spec_f[band] = spec[band]
    return float(np.sqrt(np.mean(np.real(np.fft.irfft(spec_f, n=n)) ** 2)))


def _delayed_onset_energy(audio: np.ndarray, sample_rate: int) -> float:
    return _window_rms(audio, 0.040, 0.250, int(sample_rate))


def _envelope_peak_times(
    audio: np.ndarray,
    sample_rate: int,
    *,
    start_s: float,
    end_s: float,
    block_ms: float = 8.0,
    min_dist_ms: float = 40.0,
    threshold_ratio: float = 0.38,
) -> Tuple[List[float], float]:
    """Peak times (ms from file start) on short-time RMS envelope — avoids pitch-cycle false peaks."""
    x = np.asarray(audio, dtype=np.float64)
    sr = int(sample_rate)
    i0 = max(0, int(start_s * sr))
    i1 = min(len(x), int(end_s * sr))
    seg = x[i0:i1]
    if len(seg) < 32:
        return [], 0.0
    block = max(4, int(block_ms * sr / 1000.0))
    times: List[float] = []
    rms_vals: List[float] = []
    for j in range(0, len(seg) - block, block):
        rms_vals.append(float(np.sqrt(np.mean(seg[j : j + block] ** 2))))
        times.append((i0 + j + block // 2) / sr * 1000.0)
    if not rms_vals:
        return [], 0.0
    env = np.asarray(rms_vals, dtype=np.float64)
    top = float(np.max(env))
    if top < 1e-12:
        return [], 0.0
    thresh = top * threshold_ratio
    min_blocks = max(1, int(min_dist_ms / block_ms))
    peak_idx: List[int] = []
    for i in range(1, len(env) - 1):
        if env[i] < thresh:
            continue
        if env[i] < env[i - 1] or env[i] < env[i + 1]:
            continue
        if peak_idx and (i - peak_idx[-1]) < min_blocks:
            if env[i] > env[peak_idx[-1]]:
                peak_idx[-1] = i
        else:
            peak_idx.append(i)
    peak_times = [round(times[i], 2) for i in peak_idx]
    tail_ref = _window_rms(x, 0.50, 1.20, sr)
    impulse_ratio = top / max(tail_ref, 1e-12)
    return peak_times, impulse_ratio


def _body_tail_peaks(audio: np.ndarray, sample_rate: int) -> Tuple[List[float], float]:
    return _envelope_peak_times(
        audio,
        sample_rate,
        start_s=0.080,
        end_s=0.350,
        block_ms=10.0,
        min_dist_ms=45.0,
        threshold_ratio=0.42,
    )


def scan_artifacts(
    audio: np.ndarray,
    *,
    sample_rate: int,
    duration_s: float = DEFAULT_DURATION_S,
    file_label: str = "",
    is_body_tail_stem: bool = False,
) -> Dict[str, Any]:
    x = np.asarray(audio, dtype=np.float64)
    sr = int(sample_rate)
    n = len(x)

    peaks_ms, _ = _envelope_peak_times(
        x, sr, start_s=0.0, end_s=0.25, block_ms=8.0, min_dist_ms=42.0, threshold_ratio=0.36
    )
    peaks = [int(t / 1000.0 * sr) for t in peaks_ms]
    peak_vals = [float(np.sqrt(np.mean(x[max(0, p - 4) : p + 4] ** 2))) for p in peaks] if peaks else []
    second_ratio = round(peak_vals[1] / max(peak_vals[0], 1e-12), 4) if len(peak_vals) >= 2 else 0.0
    strongest_ms = round(float(peaks[0]) / sr * 1000.0, 2) if peaks else None
    delayed_energy = round(_delayed_onset_energy(x, sr), 8)
    double_fail = len(peaks) > 1 and second_ratio >= ACCEPTANCE["second_onset_ratio_max"]

    low_imp = _band_rms(x[: int(0.30 * sr)], sr, 60.0, 250.0)
    mid_imp = _band_rms(x[: int(0.30 * sr)], sr, 250.0, 700.0)
    body_rms = _window_rms(x, 0.20, 0.80, sr)
    tail_rms = _window_rms(x, 1.0, min(duration_s, 2.5), sr)
    thump_to_body = (low_imp + mid_imp) / max(body_rms, 1e-12)
    thump_to_tail = (low_imp + mid_imp) / max(tail_rms, 1e-12)
    drum_tap = min(
        1.0,
        max(0.0, (thump_to_body - 1.2) * 0.18 + (second_ratio - 0.3) * 0.5 + delayed_energy * 8.0),
    )
    thump_fail = thump_to_body > 3.5 or drum_tap > 0.55

    bt_peak_times, bt_impulse = _body_tail_peaks(x, sr)
    bt_peak_count = len(bt_peak_times)
    bt_peak_ms = bt_peak_times[0] if bt_peak_times else None
    if is_body_tail_stem:
        delayed_body_fail = bt_peak_count > 1 or (
            bt_peak_count > 0 and bt_impulse > ACCEPTANCE["body_tail_impulse_ratio_max"]
        )
    else:
        delayed_body_fail = bt_peak_count > 1 and bt_impulse > 2.5

    tail_diag = compute_tail_continuity_diagnostics(x, sample_rate=sr, duration_s=duration_s)
    tail_collapse_fail = float(tail_diag.get("tail_continuity_ratio") or 0.0) < ACCEPTANCE["tail_continuity_ratio_min"]

    last_300 = x[max(0, n - int(0.30 * sr)) :]
    last_100 = x[max(0, n - int(0.10 * sr)) :]
    rms_last_300 = float(np.sqrt(np.mean(last_300**2))) if len(last_300) else 0.0
    rms_last_100 = float(np.sqrt(np.mean(last_100**2))) if len(last_100) else 0.0
    hi_last_300 = _band_energy(last_300, sr, 3500.0, 12000.0) if len(last_300) > 32 else 0.0
    last_abs = float(np.abs(x[-1])) if n else 0.0
    fade_zone = x[max(0, n - int(0.15 * sr)) :]
    fade_to_zero = bool(len(fade_zone) > 4 and float(np.max(np.abs(fade_zone[-4:]))) < 0.002)
    if n >= 2:
        disc = float(np.abs(x[-1] - x[-2]))
    else:
        disc = 0.0
    end_fail = (disc > 0.08 and rms_last_100 > 0.01) or (rms_last_100 > 0.04 and not fade_to_zero)

    return {
        "file_label": file_label,
        "is_body_tail_stem": is_body_tail_stem,
        "onset_peak_count_0_250ms": len(peaks),
        "second_onset_ratio": second_ratio,
        "delayed_onset_energy_40_250ms": delayed_energy,
        "strongest_onset_time_ms": strongest_ms,
        "double_pluck_fail": double_fail,
        "low_mid_impulse_rms_60_250hz": round(low_imp, 8),
        "low_mid_impulse_rms_250_700hz": round(mid_imp, 8),
        "thump_to_body_ratio": round(thump_to_body, 4),
        "thump_to_tail_ratio": round(thump_to_tail, 4),
        "drum_tap_risk_score": round(drum_tap, 4),
        "thump_fail": thump_fail,
        "body_tail_peak_time_ms": bt_peak_ms,
        "body_tail_peak_count_80_350ms": bt_peak_count,
        "body_tail_impulse_ratio": round(bt_impulse, 4),
        "delayed_body_event_fail": delayed_body_fail,
        **tail_diag,
        "tail_collapse_fail": tail_collapse_fail,
        "rms_last_300ms": round(rms_last_300, 8),
        "rms_last_100ms": round(rms_last_100, 8),
        "high_band_last_300ms": round(hi_last_300, 8),
        "final_100ms_fade_to_zero_check": fade_to_zero,
        "end_click_or_gate_fail": end_fail,
        "last_sample_abs": round(last_abs, 8),
        "final_discontinuity": round(disc, 6),
        "artifact_fail": any(
            [double_fail, thump_fail, delayed_body_fail, tail_collapse_fail, end_fail]
        ),
    }


def scan_wav_file(
    path: Path,
    *,
    duration_s: float = DEFAULT_DURATION_S,
    file_label: str = "",
    is_body_tail_stem: bool = False,
) -> Dict[str, Any]:
    audio, sr = read_wav_float_mono(path)
    n_target = int(duration_s * sr)
    if len(audio) > n_target:
        audio = audio[:n_target]
    return scan_artifacts(
        audio,
        sample_rate=sr,
        duration_s=duration_s,
        file_label=file_label or path.name,
        is_body_tail_stem=is_body_tail_stem,
    )


def evaluate_v63_acceptance(
    clean_diag: Mapping[str, Any],
    *,
    final_v1_diag: Mapping[str, Any],
    v622_diag: Mapping[str, Any],
) -> Dict[str, Any]:
    checks: Dict[str, bool] = {
        "no_delayed_body_tail_pulse": not bool(clean_diag.get("delayed_body_event_fail")),
        "no_double_onset": not bool(clean_diag.get("double_pluck_fail")),
        "no_body_tail_peak_80_350": int(clean_diag.get("body_tail_peak_count_80_350ms") or 0) <= 1,
        "no_abrupt_end_gating": not bool(clean_diag.get("end_click_or_gate_fail")),
        "final_150ms_fade_ok": bool(clean_diag.get("final_100ms_fade_to_zero_check")),
        "drum_tap_lower_than_final_v1": float(clean_diag.get("drum_tap_risk_score") or 1.0)
        < float(final_v1_diag.get("drum_tap_risk_score") or 1.0),
        "drum_tap_lower_than_v622": float(clean_diag.get("drum_tap_risk_score") or 1.0)
        < float(v622_diag.get("drum_tap_risk_score") or 1.0),
        "thump_to_body_lower_than_v622": float(clean_diag.get("thump_to_body_ratio") or 1e9)
        < float(v622_diag.get("thump_to_body_ratio") or 0.0),
        "tail_continuity_present": float(clean_diag.get("tail_continuity_ratio") or 0.0) >= 0.08,
    }
    passed = all(checks.values())
    if passed:
        status = "accepted_for_listening"
    elif sum(checks.values()) >= len(checks) // 2:
        status = "needs_more_work"
    else:
        status = "rejected"
    return {"checks": checks, "acceptance_pass": passed, "candidate_acceptance_status": status}


def _unified_excitation_envelope(
    n_samples: int,
    sample_rate: int,
    *,
    rise_ms: float = 11.0,
    body_decay_s: float = 0.38,
) -> np.ndarray:
    sr = int(sample_rate)
    t = np.arange(n_samples, dtype=np.float64) / sr
    t_r = max(float(rise_ms), 4.0) / 1000.0
    rise = np.sin(0.5 * math.pi * np.minimum(t, t_r) / t_r) ** 1.05
    body = np.exp(-np.maximum(t - t_r * 0.5, 0.0) / max(float(body_decay_s), 0.08))
    return rise * body


def _soften_body_mid_pulse(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Attenuate impulsive 80–350 ms burst without removing body warmth."""
    x = np.asarray(audio, dtype=np.float64).copy()
    sr = int(sample_rate)
    i0, i1 = int(0.08 * sr), int(0.35 * sr)
    if i1 <= i0:
        return x
    ref = _window_rms(x, 0.45, 1.10, sr)
    seg = x[i0:i1]
    pk = float(np.max(np.abs(seg)))
    if pk > max(ref * 1.55, 1e-12):
        x[i0:i1] = seg * (ref * 1.25 / pk)
    return x


def _smooth_body_from_excitation(
    body_radiated: np.ndarray,
    excitation_env: np.ndarray,
    *,
    sample_rate: int,
    hf_absorb: float,
    frequency_hz: float,
) -> np.ndarray:
    x = np.asarray(body_radiated, dtype=np.float64) * excitation_env
    n = len(x)
    sr = int(sample_rate)
    t = np.arange(n, dtype=np.float64) / sr
    sustain = np.exp(-t / 2.2) * 0.55 + 0.45 * np.exp(-t / 4.5)
    x = x * sustain
    x = _apply_note_hf_damping(x, sample_rate=sr, frequency_hz=frequency_hz, hf_absorb=hf_absorb)
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    spec[freqs > 2800] *= 0.25
    x = np.real(np.fft.irfft(spec, n=n))
    return _soften_body_mid_pulse(x, sr)


def _build_clean_pluck_stem(
    string_excitation: np.ndarray,
    excitation_env: np.ndarray,
    *,
    sample_rate: int,
    frequency_hz: float,
    gain: float,
    hf_absorb: float,
) -> np.ndarray:
    sr = int(sample_rate)
    f0 = max(40.0, float(frequency_hz))
    pluck = gain * string_excitation * excitation_env
    return _apply_note_hf_damping(pluck, sample_rate=sr, frequency_hz=f0, hf_absorb=hf_absorb + 0.05)


def _apply_final_fade_out(audio: np.ndarray, sample_rate: int, *, fade_ms: float = 200.0) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float64).copy()
    sr = int(sample_rate)
    fade_n = max(8, int(fade_ms * sr / 1000.0))
    if fade_n >= len(x):
        return x
    ramp = np.linspace(1.0, 0.0, fade_n) ** 1.2
    x[-fade_n:] *= ramp
    return x


def _gentle_peak_normalize(audio: np.ndarray, *, target_peak: float = 0.88) -> Tuple[np.ndarray, float]:
    x = np.asarray(audio, dtype=np.float64)
    peak = float(np.max(np.abs(x)))
    if peak < 1e-12:
        return x.copy(), 1.0
    scale = min(float(target_peak) / peak, 2.5)
    return x * scale, scale


def synthesize_v6_3_clean_pluck_body(
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
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Clean V6.3 candidate: one excitation envelope, no resonator pulse, no delayed body onset.
    Returns (stems, final, pre_finalize, meta).
    """
    sample_rec = get_sample_record(audit, sample_id)
    feat = collect_features_for_synthesis(audit, sample_id)
    params = normalize_sample_parameters(sample_parameters)
    hf_abs = float(feature_value(sample_rec, "high_frequency_absorption_proxy", audit=audit, default=0.28))

    string_excitation = synthesize_plucked_string(
        frequency_hz,
        duration_s,
        sample_rate,
        pluck_position=FIXED_PLUCK_POSITION,
        velocity=velocity,
    )
    n = len(string_excitation)
    env = _unified_excitation_envelope(n, sample_rate, rise_ms=11.0, body_decay_s=0.36)

    body_radiated, _, body_meta = render_v5_alpha_body_radiation_path(
        frequency_hz=frequency_hz,
        duration_s=duration_s,
        sample_rate=sample_rate,
        modal_data=modal_data,
        sample_parameters=params,
        repo_root=repo_root,
        sample_id=sample_id,
        velocity=velocity,
    )

    pluck_stem = _build_clean_pluck_stem(
        string_excitation,
        env,
        sample_rate=sample_rate,
        frequency_hz=frequency_hz,
        gain=0.28,
        hf_absorb=hf_abs,
    )
    body_tail_stem = _smooth_body_from_excitation(
        body_radiated,
        env,
        sample_rate=sample_rate,
        hf_absorb=hf_abs,
        frequency_hz=frequency_hz,
    )

    pluck_gain = 0.26
    body_gain = 0.72
    mixed = pluck_gain * pluck_stem + body_gain * body_tail_stem
    mixed = _soften_body_mid_pulse(mixed, sample_rate)
    pre_finalize, norm_scale = _gentle_peak_normalize(mixed, target_peak=0.82)
    final = _apply_final_fade_out(pre_finalize, sample_rate, fade_ms=200.0)
    peak_db, clip_ok = _peak_dbfs(final)
    if peak_db > -0.5:
        final = final * (10.0 ** ((-0.5 - peak_db) / 20.0))
        peak_db, clip_ok = _peak_dbfs(final)

    stems = {
        "pluck_stem": pluck_stem,
        "body_tail_stem": body_tail_stem,
        "final_mix": final,
        "pre_finalize": pre_finalize,
    }

    meta = {
        "diagnostic_mode": V6_3_MODE,
        "v6_3_version": V6_3_VERSION,
        "user_label": V6_3_LABEL,
        "sample_id": sample_id,
        "frequency_hz": frequency_hz,
        "duration_s": duration_s,
        "design": {
            "unified_excitation": "single rise 11ms + smooth decay — no derivative, no delayed ramp",
            "body_path": "v5_alpha body radiation × same excitation at t=0",
            "no_helmholtz_resonator": True,
            "no_soundhole_ir": True,
            "no_delayed_body_ramp": True,
            "final_fade_ms": 200.0,
            "normalization": "gentle peak normalize only (no sustain-window boost)",
        },
        "stem_gains": {"pluck_stem": pluck_gain, "body_tail_stem": body_gain},
        "body_meta": body_meta,
        "norm_scale": round(norm_scale, 6),
        "peak_dbfs": peak_db,
        "clipping_avoided": clip_ok,
        "limitations": [
            "V6.3 does not prove multi-guitar differentiation.",
            "Clean rebuild after V6.2.2 quarantine — not solved until listening confirms.",
        ],
    }
    return stems, final, pre_finalize, meta
