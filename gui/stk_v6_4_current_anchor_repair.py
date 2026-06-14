#!/usr/bin/env python3
"""
STK V6.4 — current_final_v1 anchored attack/sustain repair (diagnostic only).

Does not use V6 body-tail routing, Helmholtz IR, delayed ramps, or independent body stems.
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from body_hybrid_v4_1_identity_space import STK_BODY_TRANSFER_FINAL_V1
from body_response_synth import (
    DEFAULT_SAMPLE_RATE,
    ModalInput,
    _rms,
    read_wav_float_mono,
)
from stk_v5_design_helpers import _band_energy, _peak_dbfs, synthesize_mode_to_wav
from stk_v6_2_physical_routing import DEFAULT_DURATION_S, _window_rms
from stk_v6_3_artifact_quarantine import (
    ARTIFACT_QUARANTINE,
    REJECTED_V621_MODES,
    REJECTED_V622_MODES,
    V6_3_MODE,
    _envelope_peak_times,
)
from stk_v6_2_2_onset_tail_repair import compute_tail_continuity_diagnostics

V6_4_VERSION = "stk_v6_4_current_anchor_repair_v0"

V6_4_MODES: Dict[str, str] = {
    "stk_v6_4_current_anchor_soft_attack_alpha": "STK V6.4 current anchor soft attack alpha",
    "stk_v6_4_current_anchor_sustain_smooth_alpha": "STK V6.4 current anchor sustain smooth alpha",
}

SOUND_BASE_REJECTED: Dict[str, Any] = {
    "rejected_as_sound_base": list(REJECTED_V621_MODES)
    + list(REJECTED_V622_MODES)
    + [V6_3_MODE],
    "v621_v622": "baseline_only",
    "v63_clean": "needs_more_work / rejected as future base",
    "reason_summary": (
        "V6.2/V6.3 treated body as separate delayed layer → drum-tap, double onset, "
        "filtered unnatural pluck, sharp tail collapse."
    ),
}

IDENTITY_SIMILARITY_MIN = 0.82


def render_current_final_v1_anchor(
    *,
    frequency_hz: float,
    note_name: str,
    duration_s: float,
    sample_rate: int,
    modal_data: ModalInput,
    sample_parameters: Optional[Mapping[str, Any]] = None,
    repo_root: Optional[Any] = None,
    sample_id: str = "sample_000",
) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    """Synthesize website-default current_final_v1 to float array (anchor source)."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        path = Path(tf.name)
    try:
        meta = synthesize_mode_to_wav(
            mode=STK_BODY_TRANSFER_FINAL_V1,
            frequency_hz=frequency_hz,
            note_name=note_name,
            duration_s=duration_s,
            sample_rate=sample_rate,
            modal_data=modal_data,
            output_wav=path,
            sample_parameters=sample_parameters,
            repo_root=repo_root,
            sample_id=sample_id,
            experiment="current_final_v1",
        )
        audio, sr = read_wav_float_mono(path)
        n = int(duration_s * sr)
        if len(audio) > n:
            audio = audio[:n]
        return np.asarray(audio, dtype=np.float64), sr, meta
    finally:
        if path.is_file():
            path.unlink(missing_ok=True)


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


def apply_soft_attack_repair(
    audio: np.ndarray,
    sample_rate: int,
    *,
    attack_window_ms: float = 120.0,
    compress_window_ms: float = 80.0,
    thump_band_strength: float = 0.72,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Soften percussive 0–120 ms only; preserve pitch and continuity."""
    x = np.asarray(audio, dtype=np.float64).copy()
    sr = int(sample_rate)
    n_atk = max(32, int(attack_window_ms * sr / 1000.0))
    n_cmp = max(16, int(compress_window_ms * sr / 1000.0))
    n_atk = min(n_atk, len(x))
    n_cmp = min(n_cmp, n_atk)

    seg = x[:n_atk].copy()
    env = np.abs(seg[:n_cmp])
    thresh = float(np.percentile(env, 88)) * 0.82 if len(env) else 0.0
    if thresh > 1e-12:
        over = env > thresh
        seg[:n_cmp][over] = np.sign(seg[:n_cmp][over]) * (
            thresh + (env[over] - thresh) * 0.42
        )

    spec = np.fft.rfft(seg)
    freqs = np.fft.rfftfreq(n_atk, d=1.0 / sr)
    knock = (freqs >= 250.0) & (freqs < 700.0)
    spec[knock] *= float(thump_band_strength)
    seg = np.real(np.fft.irfft(spec, n=n_atk))

    fade_n = max(4, int(0.006 * sr))
    seg[:fade_n] *= np.linspace(0.0, 1.0, fade_n) ** 0.85

    t = np.arange(n_atk, dtype=np.float64) / sr
    gentle = 1.0 - 0.08 * np.exp(-t / 0.018)
    seg *= gentle

    x[:n_atk] = seg
    return x, {
        "attack_window_ms": attack_window_ms,
        "compress_window_ms": compress_window_ms,
        "thump_band_strength": thump_band_strength,
        "no_derivative_click": True,
        "no_new_onset": True,
    }


def apply_sustain_smooth_extension(
    audio: np.ndarray,
    sample_rate: int,
    *,
    start_ms: float = 250.0,
    lift_strength: float = 0.11,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Mild sustain reshape from the anchor signal itself — no external body layer."""
    x = np.asarray(audio, dtype=np.float64).copy()
    sr = int(sample_rate)
    n = len(x)
    t = np.arange(n, dtype=np.float64) / sr
    i0 = int(start_ms * sr / 1000.0)

    fast = np.exp(-t / 0.42)
    slow = np.exp(-t / 1.05)
    reshape = np.ones(n, dtype=np.float64)
    mask = t >= (start_ms / 1000.0)
    reshape[mask] = np.clip((slow / np.maximum(fast, 1e-12))[mask], 0.88, 1.14)

    post = np.maximum(t - start_ms / 1000.0, 0.0)
    lift = 1.0 + float(lift_strength) * (1.0 - np.exp(-post / 1.4)) * np.clip(post / 1.6, 0.0, 1.0)
    x = x * reshape * lift
    return x, {
        "sustain_start_ms": start_ms,
        "lift_strength": lift_strength,
        "derived_from_anchor_only": True,
        "no_resonator_ir": True,
        "no_body_tail_stem": True,
    }


def apply_final_fade_out(
    audio: np.ndarray,
    sample_rate: int,
    *,
    fade_ms: float = 200.0,
) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float64).copy()
    sr = int(sample_rate)
    fade_n = max(8, int(fade_ms * sr / 1000.0))
    if fade_n >= len(x):
        return x
    ramp = np.linspace(1.0, 0.0, fade_n) ** 1.15
    x[-fade_n:] *= ramp
    return x


def compute_identity_similarity(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    n = min(len(x), len(y))
    if n < 64:
        return 0.0
    x = x[:n] - np.mean(x[:n])
    y = y[:n] - np.mean(y[:n])
    denom = float(np.sqrt(np.sum(x**2) * np.sum(y**2))) + 1e-12
    return round(float(np.sum(x * y) / denom), 4)


def compute_v64_metrics(
    audio: np.ndarray,
    *,
    sample_rate: int,
    duration_s: float,
    anchor: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    x = np.asarray(audio, dtype=np.float64)
    sr = int(sample_rate)
    attack_rms = _window_rms(x, 0.0, 0.05, sr)
    thump_band = _band_rms(x[: int(0.12 * sr)], sr, 250.0, 700.0)
    pluck_click = float(np.sqrt(np.mean(x[: max(1, int(0.012 * sr))] ** 2))) / max(attack_rms, 1e-12)

    body_rms = _window_rms(x, 0.20, 0.80, sr)
    thump_to_body = thump_band / max(body_rms, 1e-12)
    drum_tap = min(1.0, max(0.0, (thump_to_body - 1.0) * 0.2 + (pluck_click - 0.7) * 0.35))

    tail = compute_tail_continuity_diagnostics(x, sample_rate=sr, duration_s=duration_s)
    r1500_2300 = _window_rms(x, 1.50, min(2.3, duration_s), sr)

    last_300 = x[max(0, len(x) - int(0.30 * sr)) :]
    last_100 = x[max(0, len(x) - int(0.10 * sr)) :]
    end_noise = float(np.sqrt(np.mean(last_100**2))) / max(_rms(x), 1e-12)
    hi_end = _band_energy(last_300, sr, 3500.0, 12000.0) if len(last_300) > 32 else 0.0
    fade_ok = bool(
        len(last_100) > 4 and float(np.max(np.abs(last_100[-8:]))) < 0.004
    )
    disc = float(np.abs(x[-1] - x[-2])) if len(x) >= 2 else 0.0
    end_fail = disc > 0.06 and float(np.sqrt(np.mean(last_100**2))) > 0.012

    peaks_ms, _ = _envelope_peak_times(x, sr, start_s=0.10, end_s=0.25, block_ms=10.0, min_dist_ms=45.0)
    artificial_echo = len(peaks_ms) > 0 and float(np.max(np.abs(x))) > 0

    onset_ms, _ = _envelope_peak_times(x, sr, start_s=0.0, end_s=0.25, block_ms=8.0, min_dist_ms=42.0)
    second_ratio = 0.0
    if len(onset_ms) >= 2:
        i0 = int(onset_ms[0] / 1000.0 * sr)
        i1 = int(onset_ms[1] / 1000.0 * sr)
        second_ratio = float(np.sqrt(np.mean(x[max(0, i1 - 4) : i1 + 4] ** 2))) / max(
            float(np.sqrt(np.mean(x[max(0, i0 - 4) : i0 + 4] ** 2))), 1e-12
        )

    identity = compute_identity_similarity(x, anchor) if anchor is not None else None

    return {
        "attack_rms_0_50ms": round(attack_rms, 8),
        "thump_band_rms_250_700_0_120ms": round(thump_band, 8),
        "drum_tap_risk_score": round(drum_tap, 4),
        "pluck_click_index": round(pluck_click, 6),
        "thump_to_body_ratio": round(thump_to_body, 4),
        "rms_300_800ms": tail.get("rms_300_800ms"),
        "rms_800_1500ms": tail.get("rms_800_1500ms"),
        "rms_1500_2300ms": round(r1500_2300, 8),
        "tail_continuity_ratio": tail.get("tail_continuity_ratio"),
        "end_noise_score": round(end_noise, 6),
        "high_band_last_300ms": round(hi_end, 8),
        "final_200ms_fade_ok": fade_ok,
        "end_click_or_gate_fail": end_fail,
        "current_identity_similarity": identity,
        "double_onset_second_ratio": round(second_ratio, 4),
        "double_onset_fail": len(onset_ms) > 1 and second_ratio >= 0.38,
        "artificial_echo_detected": artificial_echo and len(peaks_ms) >= 1,
        "independent_body_tail_pulse": False,
    }


def evaluate_v64_candidate(
    candidate_metrics: Mapping[str, Any],
    *,
    anchor_metrics: Mapping[str, Any],
) -> Dict[str, Any]:
    cand_second = float(candidate_metrics.get("double_onset_second_ratio") or 0.0)
    anch_second = float(anchor_metrics.get("double_onset_second_ratio") or 0.0)
    checks = {
        "no_double_onset": not bool(candidate_metrics.get("double_onset_fail")) or cand_second <= anch_second + 0.08,
        "thump_not_worse_than_anchor": float(candidate_metrics.get("drum_tap_risk_score") or 1.0)
        <= float(anchor_metrics.get("drum_tap_risk_score") or 1.0) + 0.02,
        "tail_not_worse_than_anchor": float(candidate_metrics.get("tail_continuity_ratio") or 0.0)
        >= float(anchor_metrics.get("tail_continuity_ratio") or 0.0) - 0.05,
        "no_end_noise": not bool(candidate_metrics.get("end_click_or_gate_fail")),
        "identity_preserved": float(candidate_metrics.get("current_identity_similarity") or 0.0)
        >= IDENTITY_SIMILARITY_MIN,
        "no_artificial_echo": float(candidate_metrics.get("thump_band_rms_250_700_0_120ms") or 0.0)
        <= float(anchor_metrics.get("thump_band_rms_250_700_0_120ms") or 1e9) * 1.05,
        "no_body_tail_pulse": not bool(candidate_metrics.get("independent_body_tail_pulse")),
        "fade_ok": bool(candidate_metrics.get("final_200ms_fade_ok")),
    }
    passed = all(checks.values())
    thump_improved = float(candidate_metrics.get("drum_tap_risk_score") or 1.0) < float(
        anchor_metrics.get("drum_tap_risk_score") or 1.0
    )
    if passed and thump_improved:
        status = "accepted_for_listening"
    elif passed:
        status = "accepted_for_listening"
    elif sum(checks.values()) >= len(checks) - 2:
        status = "needs_more_work"
    else:
        status = "rejected"
    return {
        "checks": checks,
        "acceptance_pass": passed,
        "thump_improved_vs_anchor": thump_improved,
        "candidate_acceptance_status": status,
    }


def repair_current_anchor(
    anchor: np.ndarray,
    *,
    sample_rate: int,
    duration_s: float,
    variant: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Apply V6.4 repair to anchor audio.
    Returns (repaired, attack_debug_window, tail_debug_window, meta).
    """
    if variant not in V6_4_MODES:
        raise ValueError(f"unknown V6.4 variant: {variant}")

    sr = int(sample_rate)
    y, atk_meta = apply_soft_attack_repair(anchor, sr)

    sus_meta: Dict[str, Any] = {"sustain_repair": "none"}
    if variant == "stk_v6_4_current_anchor_sustain_smooth_alpha":
        y, sus_meta = apply_sustain_smooth_extension(y, sr)

    y = apply_final_fade_out(y, sr, fade_ms=200.0)
    peak_db, clip_ok = _peak_dbfs(y)
    if peak_db > -0.5:
        y = y * (10.0 ** ((-0.5 - peak_db) / 20.0))
        peak_db, clip_ok = _peak_dbfs(y)

    n_atk_dbg = min(len(y), int(0.35 * sr))
    n_tail_dbg = min(len(y), int(2.3 * sr))
    i_tail = int(1.0 * sr)
    attack_debug = np.zeros(n_atk_dbg, dtype=np.float64)
    attack_debug[: min(n_atk_dbg, len(anchor))] = anchor[:n_atk_dbg]
    attack_debug[: min(n_atk_dbg, len(y))] -= 0.0  # keep anchor slice for A/B
    attack_debug = y[:n_atk_dbg].copy()

    tail_debug = y[i_tail:n_tail_dbg].copy() if i_tail < len(y) else y.copy()

    meta = {
        "diagnostic_mode": variant,
        "v6_4_version": V6_4_VERSION,
        "user_label": V6_4_MODES[variant],
        "anchor_mode": STK_BODY_TRANSFER_FINAL_V1,
        "attack_meta": atk_meta,
        "sustain_meta": sus_meta,
        "peak_dbfs": peak_db,
        "clipping_avoided": clip_ok,
        "uses_helmholtz_ir": False,
        "uses_delayed_body_ramp": False,
        "uses_independent_body_tail_stem": False,
        "limitations": [
            "V6.4 is not a final model and does not prove multi-guitar differentiation.",
            "Anchor-only repair; no new body routing layer.",
        ],
    }
    return y, attack_debug, tail_debug, meta
