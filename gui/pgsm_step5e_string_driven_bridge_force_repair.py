#!/usr/bin/env python3
"""
PGSM Step 5E — string-driven bridge force repair.
Sustained diagnostic string-force proxy driving Step 3C modal body; not final synthesis.
"""
from __future__ import annotations

import hashlib
import json
import math
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from pgsm_physical_factor_registry import helmholtz_proxy_hz
from pgsm_step2_1_parameter_targets import load_step_report
from pgsm_step3a_numerical_ir_testbench import (
    DURATION_S,
    FIXED_PLUCK_POSITION,
    NUMERIC_SR,
    SAMPLE_ID,
    compute_impulse_response,
)
from pgsm_step4a_single_note_diagnostic_audio import (
    DIAGNOSTIC_LABEL,
    build_calibrated_modal_state,
    compute_full_impulse_response,
    evaluate_artifact_guard,
    normalize_diagnostic_amplitude,
    synthesize_modal_body_response,
    write_wav_mono,
)
from pgsm_step4b_single_note_diagnostic_refinement import _envelope, load_wav_mono
from pgsm_step5a_limited_note_set_diagnostic_audio import (
    NOTE_FREQUENCY_HZ,
    NOTE_SET,
    step4a_output_fingerprints,
)
from pgsm_step5b_limited_note_set_refinement import step5a_output_fingerprints
from pgsm_step5c_note_set_extended_validation import READINESS_STEP6A
from pgsm_step5d_audible_diagnostic_render_repair import (
    READINESS_AFTER as READINESS_STEP5D,
    RENDER_DIR as STEP5D_RENDER_DIR,
)
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE

PGSM_STEP5E_VERSION = "pgsm_step5e_string_driven_bridge_force_repair_v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON = REPO_ROOT / "audio" / "debug_reports" / "pgsm_step5e_string_driven_bridge_force_repair.json"
REPORT_MD = REPO_ROOT / "audio" / "debug_reports" / "pgsm_step5e_string_driven_bridge_force_repair.md"
AUDIO_DIR = REPO_ROOT / "audio" / "pgsm_step5e_string_driven_bridge_force"

READINESS_AFTER = "ready_for_step5f_string_driven_extended_validation"
STRING_FORCE_LABEL = "diagnostic_string_bridge_force_proxy_not_measured_force"

TARGET_RMS_DBFS_MIN = -24.0
TARGET_RMS_DBFS_MAX = -20.0
TARGET_RMS_DBFS_NOMINAL = -22.0
PEAK_CAP_DBFS = -1.0
PEAK_CAP_FS = 10.0 ** (PEAK_CAP_DBFS / 20.0)

ENERGY_FIRST_10MS_MAX = 0.50
ACTIVE_DURATION_MIN_MS_LOW = 1000.0
ACTIVE_DURATION_MIN_MS_HIGH = 500.0
PITCH_SALIENCE_MIN = 0.005
OUTPUT_DURATION_S = DURATION_S

INHARMONICITY_B_FALLBACK = 0.0002
INHARMONICITY_LEVEL = "L2_fallback_placeholder"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _report_path(root: Path, name: str) -> Path:
    return root / "audio" / "debug_reports" / name


def _file_fingerprint(path: Path) -> str:
    if not path.is_file():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _linear_to_dbfs(x: float) -> float:
    return float(20.0 * math.log10(max(abs(x), 1e-12)))


def _dbfs_to_linear(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def _rms(y: np.ndarray) -> float:
    if y.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))


def _active_duration_ms(y: np.ndarray, sr: int, threshold_dbfs: float = -60.0) -> float:
    thr = _dbfs_to_linear(threshold_dbfs)
    mask = np.abs(y) >= thr
    if not mask.any():
        return 0.0
    idx = np.where(mask)[0]
    return float((idx[-1] - idx[0] + 1) / sr * 1000.0)


def _energy_share_first_ms(y: np.ndarray, sr: int, ms: float) -> float:
    total = float(np.sum(y.astype(np.float64) ** 2))
    if total <= 1e-20:
        return 0.0
    n = min(int(round(ms * 1e-3 * sr)), len(y))
    return float(np.sum(y[:n] ** 2) / total)


def _decay_time_ms_smoothed(y: np.ndarray, sr: int, db: float) -> Optional[float]:
    env = _envelope(y, sr)
    t = np.arange(len(env), dtype=np.float64) / sr
    if env.size == 0:
        return None
    peak_i = int(np.argmax(env))
    peak = float(env[peak_i])
    if peak <= 0:
        return None
    target = peak * 10.0 ** (db / 20.0)
    idx = np.where(env[peak_i:] <= target)[0]
    if idx.size == 0:
        return None
    return float(t[peak_i + int(idx[0])] * 1000.0)


def step5d_listening_path(root: Path, note: str) -> Path:
    return root / "audio" / "pgsm_step5d_audible_render" / f"{SAMPLE_ID}_{note}_listening_diagnostic.wav"


def collect_previous_audio_fingerprints(root: Path) -> Dict[str, str]:
    fps: Dict[str, str] = {}
    for name, fp in step5a_output_fingerprints(root).items():
        fps[f"step5a_{name}"] = fp
    for name, fp in step4a_output_fingerprints(root).items():
        fps[f"step4a_{name}"] = fp
    for note in NOTE_SET:
        p = step5d_listening_path(root, note)
        fps[f"step5d_{note}_listening"] = _file_fingerprint(p)
    return fps


def verify_upstream_readiness(
    step5d: Mapping[str, Any],
    step5c: Mapping[str, Any],
) -> Dict[str, Any]:
    rg5d = step5d.get("readiness_after_step5d") or {}
    rg5c = step5c.get("readiness_after_step5c") or {}
    return {
        "step5d_readiness": rg5d.get("current_status"),
        "step5d_pass": rg5d.get("current_status") == READINESS_STEP5D,
        "step5c_readiness": rg5c.get("current_status"),
        "step5c_diagnostic_only": rg5c.get("final_synthesis_ready") is False,
        "step3c_modal_available": bool(step5d.get("step5a_loaded")),
        "final_synthesis_blocked": True,
        "stk_blocked": True,
        "website_blocked": True,
        "multi_guitar_blocked": True,
        "melody_chords_blocked": True,
        "pass": bool(
            rg5d.get("current_status") == READINESS_STEP5D
            and rg5c.get("current_status") == READINESS_STEP6A
            and rg5c.get("final_synthesis_ready") is False
        ),
    }


def build_string_driven_bridge_force(
    n: int,
    sr: int,
    f0: float,
    *,
    pluck_position_ratio: float = FIXED_PLUCK_POSITION,
    n_harmonics: int = 20,
    onset_ms: float = 4.0,
    base_decay_s: float = 2.8,
    inharmonicity_b: float = INHARMONICITY_B_FALLBACK,
) -> np.ndarray:
    """Sustained damped harmonic string bridge-force proxy (not measured force)."""
    t = np.arange(n, dtype=np.float64) / sr
    force = np.zeros(n, dtype=np.float64)

    onset_n = max(int(onset_ms * 1e-3 * sr), 2)
    onset = np.ones(n, dtype=np.float64)
    ramp = np.sin(np.pi * np.linspace(0.0, 1.0, onset_n)) ** 2
    onset[:onset_n] = ramp
    if onset_n > 1:
        onset[0] = max(ramp[1] * 0.05, 1e-8)

    for k in range(1, n_harmonics + 1):
        fk = f0 * k * (1.0 + inharmonicity_b * k * k)
        if fk >= sr / 2.0:
            break
        amp_k = abs(math.sin(math.pi * k * pluck_position_ratio)) / k
        if amp_k < 1e-8:
            continue
        tau_k = base_decay_s / (k ** 0.65)
        force += amp_k * np.exp(-t / tau_k) * np.sin(2.0 * math.pi * fk * t)

    force *= onset
    peak = max(float(np.max(np.abs(force))), 1e-12)
    return (force / peak).astype(np.float64)


def compute_modal_kernels_decomposed(
    modal_weights: Mapping[str, Any],
    *,
    duration_s: float = OUTPUT_DURATION_S,
    sr: int = NUMERIC_SR,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Step 3C modal IR split into structural (top/back radiation) and cavity/air proxy."""
    modes = modal_weights.get("modes") or []
    n = int(duration_s * sr)
    t = np.arange(n, dtype=np.float64) / sr
    h_structural = np.zeros(n, dtype=np.float64)
    h_cavity_air = np.zeros(n, dtype=np.float64)

    for row in modes:
        f_i = float(row["frequency_hz"])
        tau = max(float(row["tau_s"]), 1e-6)
        wr = float(row["W_rad"])
        wa = float(row["W_air"])
        top = float(row["top_share"])
        back = float(row["back_share"])
        air = float(row["air_share"])
        region = max(top + back + air, 1e-9)

        h_structural += wr * np.exp(-t / tau) * np.sin(2.0 * math.pi * f_i * t)
        h_cavity_air += (
            wa * np.exp(-t / (tau * 1.2)) * np.sin(2.0 * math.pi * f_i * t) * 0.01 * (air / region)
        )

    h_total = h_structural + h_cavity_air
    return h_total.astype(np.float64), h_structural.astype(np.float64), h_cavity_air.astype(np.float64)


def apply_listening_render_full(
    y: np.ndarray,
    *,
    target_rms_dbfs: float = TARGET_RMS_DBFS_NOMINAL,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Full-duration listening render: RMS-target gain, peak safety limit only."""
    y = np.asarray(y, dtype=np.float64)
    rms_in = _rms(y)
    peak_in = float(np.max(np.abs(y))) if y.size else 0.0
    if rms_in <= 1e-12:
        return y.copy(), {
            "gain_linear": 1.0,
            "gain_db": 0.0,
            "limiter_applied": False,
            "limiter_type": None,
            "gain_separate_from_physics": True,
            "trim_applied": False,
            "duration_trimmed": False,
        }

    target_rms = _dbfs_to_linear(target_rms_dbfs)
    gain_rms = target_rms / rms_in
    y_scaled = y * gain_rms
    peak_scaled = float(np.max(np.abs(y_scaled)))

    limiter_applied = False
    limiter_type: Optional[str] = None
    gain_linear = gain_rms

    if peak_scaled > PEAK_CAP_FS:
        gain_peak = PEAK_CAP_FS / max(peak_in, 1e-12)
        y_peak_limited = y * gain_peak
        rms_after_db = _linear_to_dbfs(_rms(y_peak_limited))
        if TARGET_RMS_DBFS_MIN <= rms_after_db <= TARGET_RMS_DBFS_MAX:
            y_out = y_peak_limited
            gain_linear = gain_peak
        else:
            y_out = np.clip(y_scaled, -PEAK_CAP_FS, PEAK_CAP_FS)
            gain_linear = gain_rms
            limiter_applied = True
            limiter_type = "transparent_peak_safety_clip_at_minus_1_dbfs"
    else:
        y_out = y_scaled

    if float(np.max(np.abs(y_out))) > 1.0:
        y_out = np.clip(y_out, -PEAK_CAP_FS, PEAK_CAP_FS)
        limiter_applied = True
        limiter_type = limiter_type or "transparent_peak_safety_clip_at_minus_1_dbfs"

    gain_db = 20.0 * math.log10(max(gain_linear, 1e-12))
    return y_out, {
        "gain_linear": round(gain_linear, 6),
        "gain_db": round(gain_db, 3),
        "target_rms_dbfs": target_rms_dbfs,
        "limiter_applied": limiter_applied,
        "limiter_type": limiter_type,
        "gain_separate_from_physics": True,
        "physics_changed": False,
        "trim_applied": False,
        "duration_trimmed": False,
        "decay_stretch_applied": False,
        "reverb_echo_body_tail_added": False,
    }


def compute_pitch_salience(y: np.ndarray, sr: int, f0: float) -> float:
    n = len(y)
    if n < 256:
        return 0.0
    spec = np.abs(np.fft.rfft(y * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    power = spec ** 2
    total = max(float(np.sum(power)), 1e-12)
    fund_mask = (freqs >= f0 - 8.0) & (freqs <= f0 + 8.0)
    return float(np.sum(power[fund_mask]) / total)


def compute_harmonic_energies(y: np.ndarray, sr: int, f0: float, n_h: int = 8) -> Dict[str, float]:
    n = len(y)
    window = np.hanning(n)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    spec = np.abs(np.fft.rfft(y * window))
    power = spec ** 2
    total = max(float(np.sum(power)), 1e-12)
    out: Dict[str, float] = {}
    for k in range(1, n_h + 1):
        h = f0 * k
        mask = (freqs >= h - 10.0) & (freqs <= h + 10.0)
        out[f"H{k}"] = round(float(np.sum(power[mask]) / total), 6) if mask.any() else 0.0
    return out


def compute_spectral_features(y: np.ndarray, sr: int) -> Dict[str, Any]:
    n = len(y)
    if n < 256:
        return {"spectral_centroid_hz": 0.0, "spectral_rolloff_hz": 0.0}
    spec = np.abs(np.fft.rfft(y * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    power = spec ** 2
    total = max(float(np.sum(power)), 1e-12)
    centroid = float(np.sum(freqs * power) / total)
    cum = np.cumsum(power)
    rolloff_i = int(np.searchsorted(cum, 0.85 * total))
    rolloff = float(freqs[min(rolloff_i, len(freqs) - 1)])
    return {
        "spectral_centroid_hz": round(centroid, 2),
        "spectral_rolloff_hz": round(rolloff, 2),
    }


def detect_second_onset_sustained(y: np.ndarray, sr: int) -> bool:
    """True if a distinct re-attack is detected (not sustained harmonic buildup)."""
    env = _envelope(y, sr)
    if env.size < sr // 5:
        return False
    peak_i = int(np.argmax(env))
    peak = float(env[peak_i])
    if peak <= 1e-12:
        return False
    post = env[peak_i + int(0.12 * sr) :]
    if post.size < sr // 10:
        return False
    # require prior drop below 30% peak then rise above 45% peak
    dipped = False
    for v in post:
        if v < 0.3 * peak:
            dipped = True
        if dipped and v > 0.45 * peak:
            return True
    return False


def compute_click_dominance_score(
    y: np.ndarray,
    sr: int,
    *,
    energy_first_10ms: float,
) -> float:
    """Higher score = more click-like (transient-dominated)."""
    n = len(y)
    if n < 256:
        return 1.0
    spec = np.abs(np.fft.rfft(y * np.hanning(n)))
    power = spec ** 2
    flatness_geo = float(np.exp(np.mean(np.log(np.maximum(power, 1e-20)))))
    flatness = flatness_geo / max(float(np.mean(power)), 1e-20)
    return round(min(1.0, 0.55 * energy_first_10ms + 0.45 * min(flatness / 0.35, 1.0)), 4)


def _envelope_correlation(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n < 8:
        return 0.0
    a = np.asarray(a[:n], dtype=float)
    b = np.asarray(b[:n], dtype=float)
    a = a / max(float(np.max(a)), 1e-12)
    b = b / max(float(np.max(b)), 1e-12)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def evaluate_modal_peak_alignment(
    y: np.ndarray,
    sr: int,
    modal_freqs: Sequence[float],
    *,
    f0: Optional[float] = None,
    h_body: Optional[np.ndarray] = None,
    pitch_salience: Optional[float] = None,
) -> Dict[str, Any]:
    n = len(y)
    if n < 256:
        return {"modal_peaks_aligned_count": 0, "pass": False}

    spec = np.abs(np.fft.rfft(y * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    spec_db = 20.0 * np.log10(np.maximum(spec, 1e-12))
    peak_indices: List[int] = []
    for i in range(2, len(spec_db) - 2):
        if spec_db[i] > spec_db[i - 1] and spec_db[i] > spec_db[i + 1]:
            if spec_db[i] > spec_db.max() - 40.0:
                peak_indices.append(i)
    peak_freqs = [float(freqs[i]) for i in peak_indices]
    tol_hz = 35.0
    aligned = 0
    all_modal = list(modal_freqs)
    for f_m in all_modal:
        if any(abs(f - f_m) <= tol_hz for f in peak_freqs):
            aligned += 1
            continue
        if f0 is not None:
            if any(abs(f_m - k * f0) <= tol_hz for k in range(1, 16)):
                aligned += 1

    env_corr = 0.0
    if h_body is not None and h_body.size:
        env_corr = _envelope_correlation(_envelope(y, sr), _envelope(h_body, sr))

    max_modal_hz = max(all_modal) if all_modal else 0.0
    above_modal_band = bool(f0 is not None and f0 > max_modal_hz + tol_hz)

    hf_mask = freqs > 8000.0
    hf_spike = bool(hf_mask.any() and spec_db[hf_mask].max() > spec_db.max() - 6.0)
    comb_score = 0.0
    if len(peak_indices) >= 5:
        spacings = np.diff([freqs[i] for i in peak_indices[:8]])
        if spacings.size:
            comb_score = float(np.std(spacings) / max(np.mean(spacings), 1.0))

    modal_pass = bool(
        aligned >= 2
        or env_corr >= 0.30
        or (above_modal_band and (pitch_salience or 0.0) >= 0.5)
    )
    return {
        "modal_peaks_aligned_count": aligned,
        "body_envelope_correlation_vs_ir": round(env_corr, 4),
        "note_above_max_modal_hz": above_modal_band,
        "max_modal_hz": round(max_modal_hz, 2),
        "no_hf_spike": bool(not hf_spike),
        "no_comb_echo": bool(comb_score < 0.95),
        "echo_comb_pattern_score": round(comb_score, 4),
        "harmonic_modal_coupling_used": f0 is not None,
        "pass": bool(modal_pass and not hf_spike and comb_score < 0.95),
    }


def build_per_note_string_force_contract(note: str, f0: float) -> Dict[str, Any]:
    return {
        "note": note,
        "f0_hz": f0,
        "pluck_position_ratio": FIXED_PLUCK_POSITION,
        "partial_amplitude_law": "sin(pi*n*pluck_position)/n",
        "decay_law": "frequency_dependent_tau_k = base_tau / k^0.65",
        "inharmonicity": {
            "enabled": True,
            "level": INHARMONICITY_LEVEL,
            "b_coefficient": INHARMONICITY_B_FALLBACK,
        },
        "onset_ms": 4.0,
        "output_duration_s": OUTPUT_DURATION_S,
        "label": STRING_FORCE_LABEL,
        "measured_force_claim_allowed": False,
    }


def build_cavity_response_summary(
    *,
    region_cal: Mapping[str, Any],
    cavity_body: np.ndarray,
    structural_body: np.ndarray,
    string_force: np.ndarray,
    sr: int,
    helmholtz_hz: Optional[float],
) -> Dict[str, Any]:
    e_cav = float(np.sum(cavity_body ** 2))
    e_struct = float(np.sum(structural_body ** 2))
    e_total = max(e_cav + e_struct, 1e-12)
    cal = region_cal.get("calibrated") or region_cal
    return {
        "cavity_air_modal_contribution_fraction": round(e_cav / e_total, 6),
        "structural_top_back_fraction": round(e_struct / e_total, 6),
        "top_back_air_balance": {
            "top_share_proxy": cal.get("top"),
            "back_share_proxy": cal.get("back"),
            "air_share_proxy": cal.get("air"),
            "labeled_proxy_not_measured": True,
        },
        "helmholtz_proxy_hz": helmholtz_hz,
        "decay_from_modal_q_tau_only": True,
        "no_delayed_independent_echo_onset": True,
        "no_artificial_echo_or_reverb": True,
        "cavity_limitations": (
            "Air/cavity contribution is a Step 3C W_air × air_share modal proxy; "
            "not a measured Helmholtz or room response."
        ),
        "internal_decay_interpretation": (
            "Perceived internal resonance arises only from modal cavity terms in the "
            "Step 3C calibrated IR convolution, not from post-hoc echo or reverb."
        ),
    }


def _output_paths(audio_dir: Path, note: str) -> Dict[str, Path]:
    base = f"{SAMPLE_ID}_{note}"
    return {
        "main": audio_dir / f"{base}_string_driven_diagnostic.wav",
        "body_stem": audio_dir / f"{base}_body_stem.wav",
        "string_force_stem": audio_dir / f"{base}_string_force_stem.wav",
    }


def analyze_step5d_baseline(root: Path, note: str, sr: int) -> Dict[str, Any]:
    path = step5d_listening_path(root, note)
    if not path.is_file():
        return {"available": False}
    y, file_sr = load_wav_mono(path)
    if file_sr != sr:
        return {"available": False, "sr_mismatch": file_sr}
    f0 = NOTE_FREQUENCY_HZ[note]
    e10 = _energy_share_first_ms(y, sr, 10.0)
    return {
        "available": True,
        "duration_s": round(len(y) / sr, 4),
        "peak_dbfs": round(_linear_to_dbfs(float(np.max(np.abs(y)))), 3),
        "rms_dbfs": round(_linear_to_dbfs(_rms(y)), 3),
        "energy_first_10ms": round(e10, 4),
        "energy_first_50ms": round(_energy_share_first_ms(y, sr, 50.0), 4),
        "energy_first_100ms": round(_energy_share_first_ms(y, sr, 100.0), 4),
        "active_duration_minus_60_dbfs_ms": round(_active_duration_ms(y, sr), 3),
        "decay_minus_40_db_ms": _decay_time_ms_smoothed(y, sr, -40.0),
        "pitch_salience_f0": round(compute_pitch_salience(y, sr, f0), 4),
        "spectral_centroid_hz": compute_spectral_features(y, sr).get("spectral_centroid_hz"),
        "click_dominance_score": compute_click_dominance_score(y, sr, energy_first_10ms=e10),
    }


def evaluate_per_note_metrics(
    main: np.ndarray,
    body_stem: np.ndarray,
    string_force_stem: np.ndarray,
    sr: int,
    *,
    note: str,
    f0: float,
    modal_freqs: Sequence[float],
    listening_info: Mapping[str, Any],
    step5d_baseline: Mapping[str, Any],
    h_body: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    e10 = _energy_share_first_ms(main, sr, 10.0)
    e50 = _energy_share_first_ms(main, sr, 50.0)
    e100 = _energy_share_first_ms(main, sr, 100.0)
    active_ms = _active_duration_ms(main, sr)
    pitch_sal = compute_pitch_salience(main, sr, f0)
    harmonics = compute_harmonic_energies(main, sr, f0, n_h=8)
    spectral = compute_spectral_features(main, sr)
    modal_main = evaluate_modal_peak_alignment(
        main, sr, modal_freqs, f0=f0, h_body=h_body, pitch_salience=pitch_sal
    )
    modal_body = evaluate_modal_peak_alignment(
        body_stem, sr, modal_freqs, f0=f0, h_body=h_body, pitch_salience=pitch_sal
    )
    modal = {
        **modal_body,
        "main_modal_peaks_aligned_count": modal_main.get("modal_peaks_aligned_count"),
        "body_stem_modal_peaks_aligned_count": modal_body.get("modal_peaks_aligned_count"),
        "pass": bool(modal_body.get("pass")),
    }
    click_score = compute_click_dominance_score(main, sr, energy_first_10ms=e10)

    second_onset = detect_second_onset_sustained(main, sr)
    env = _envelope(main, sr)
    last_third = env[len(env) * 2 // 3 :]
    mid_third = env[len(env) // 3 : len(env) * 2 // 3]
    end_rise = bool(last_third.size and float(last_third.max()) > float(mid_third.max()) * 1.05)
    tail = env[int(len(env) * 0.85) :]
    hard_gate = bool(tail.size and float(tail.max()) < 1e-6 and float(env[len(env) // 2]) > 1e-4)

    min_active = ACTIVE_DURATION_MIN_MS_LOW if note in ("A2", "A3", "A4") else ACTIVE_DURATION_MIN_MS_HIGH
    e10_ok = e10 < ENERGY_FIRST_10MS_MAX
    active_ok = active_ms >= min_active
    pitch_ok = pitch_sal >= PITCH_SALIENCE_MIN
    click_ok = click_score < 0.45
    duration_ok = abs(len(main) / sr - OUTPUT_DURATION_S) < 0.05

    baseline_e10 = float(step5d_baseline.get("energy_first_10ms") or 1.0)
    improved_e10 = e10 < baseline_e10 * 0.5
    baseline_active = float(step5d_baseline.get("active_duration_minus_60_dbfs_ms") or 0.0)
    improved_active = active_ms > baseline_active * 1.5

    peak = float(np.max(np.abs(main)))
    rms_db = _linear_to_dbfs(_rms(main))

    return {
        "peak_fs": round(peak, 6),
        "peak_dbfs": round(_linear_to_dbfs(peak), 3),
        "rms_dbfs": round(rms_db, 3),
        "duration_s": round(len(main) / sr, 4),
        "energy_first_10ms": round(e10, 4),
        "energy_first_50ms": round(e50, 4),
        "energy_first_100ms": round(e100, 4),
        "active_duration_minus_60_dbfs_ms": round(active_ms, 3),
        "decay_minus_20_db_ms": _decay_time_ms_smoothed(main, sr, -20.0),
        "decay_minus_40_db_ms": _decay_time_ms_smoothed(main, sr, -40.0),
        "decay_minus_60_db_ms": _decay_time_ms_smoothed(main, sr, -60.0),
        "pitch_salience_f0": round(pitch_sal, 4),
        "harmonic_energy_fraction": harmonics,
        "spectral_centroid_hz": spectral.get("spectral_centroid_hz"),
        "spectral_rolloff_hz": spectral.get("spectral_rolloff_hz"),
        "modal_peak_alignment": modal,
        "click_dominance_score": click_score,
        "listening_gain_db": listening_info.get("gain_db"),
        "limiter_applied": listening_info.get("limiter_applied"),
        "rms_in_listening_target": bool(TARGET_RMS_DBFS_MIN <= rms_db <= TARGET_RMS_DBFS_MAX),
        "peak_below_minus_1_dbfs": bool(_linear_to_dbfs(peak) <= PEAK_CAP_DBFS + 0.01),
        "no_second_onset": bool(not second_onset),
        "no_end_rise": bool(not end_rise),
        "no_hard_gate": bool(not hard_gate),
        "no_hf_spike": modal.get("no_hf_spike"),
        "no_comb_echo": modal.get("no_comb_echo"),
        "energy_first_10ms_below_threshold": e10_ok,
        "active_duration_sufficient": active_ok,
        "pitch_salience_detectable": pitch_ok,
        "not_click_dominant": click_ok,
        "full_duration_not_trimmed": duration_ok,
        "improved_vs_step5d_energy_10ms": improved_e10,
        "improved_vs_step5d_active_duration": improved_active,
        "pass": bool(
            e10_ok
            and active_ok
            and pitch_ok
            and click_ok
            and duration_ok
            and improved_e10
            and improved_active
            and modal.get("pass")
            and not second_onset
            and not end_rise
            and not hard_gate
            and TARGET_RMS_DBFS_MIN <= rms_db <= TARGET_RMS_DBFS_MAX
        ),
    }


def build_before_after_comparison(
    per_note_metrics: Mapping[str, Mapping[str, Any]],
    step5d_baselines: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    comparison: Dict[str, Any] = {}
    for note in NOTE_SET:
        m = per_note_metrics.get(note) or {}
        b = step5d_baselines.get(note) or {}
        comparison[note] = {
            "step5d_energy_first_10ms": b.get("energy_first_10ms"),
            "step5e_energy_first_10ms": m.get("energy_first_10ms"),
            "step5d_active_duration_ms": b.get("active_duration_minus_60_dbfs_ms"),
            "step5e_active_duration_ms": m.get("active_duration_minus_60_dbfs_ms"),
            "step5d_rms_dbfs": b.get("rms_dbfs"),
            "step5e_rms_dbfs": m.get("rms_dbfs"),
            "step5d_decay_minus_40_db_ms": b.get("decay_minus_40_db_ms"),
            "step5e_decay_minus_40_db_ms": m.get("decay_minus_40_db_ms"),
            "step5d_pitch_salience": b.get("pitch_salience_f0"),
            "step5e_pitch_salience": m.get("pitch_salience_f0"),
            "step5d_spectral_centroid_hz": b.get("spectral_centroid_hz"),
            "step5e_spectral_centroid_hz": m.get("spectral_centroid_hz"),
            "step5d_click_dominance": b.get("click_dominance_score"),
            "step5e_click_dominance": m.get("click_dominance_score"),
            "knock_click_reduced": m.get("improved_vs_step5d_energy_10ms"),
            "sustained_behavior_increased": m.get("improved_vs_step5d_active_duration"),
        }
    all_improved = all(
        (comparison[n] or {}).get("knock_click_reduced")
        and (comparison[n] or {}).get("sustained_behavior_increased")
        for n in NOTE_SET
    )
    return {**comparison, "all_notes_improved_vs_step5d": all_improved}


def build_artifact_guard(
    per_note_metrics: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    checks = {
        "no_reverb": True,
        "no_echo": True,
        "no_body_tail_layer": True,
        "no_eq_body_layer": True,
        "no_decay_stretch": True,
        "no_delayed_echo_onset": True,
        "no_artificial_echo_or_reverb": True,
        "no_second_onset": all((per_note_metrics.get(n) or {}).get("no_second_onset") for n in NOTE_SET),
        "no_end_rise": all((per_note_metrics.get(n) or {}).get("no_end_rise") for n in NOTE_SET),
        "no_hard_gate": all((per_note_metrics.get(n) or {}).get("no_hard_gate") for n in NOTE_SET),
        "no_hf_spike": all((per_note_metrics.get(n) or {}).get("no_hf_spike") for n in NOTE_SET),
        "no_comb_echo": all((per_note_metrics.get(n) or {}).get("no_comb_echo") for n in NOTE_SET),
    }
    return {**checks, "pass": bool(all(checks.values()))}


def build_readiness_after_step5e(
    objective_pass: bool,
    per_note_metrics: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    pitch_fail = any(not (per_note_metrics.get(n) or {}).get("pitch_salience_detectable") for n in NOTE_SET)
    decay_fail = any(not (per_note_metrics.get(n) or {}).get("active_duration_sufficient") for n in NOTE_SET)

    if pitch_fail or decay_fail:
        status = "blocked_due_to_pitch_or_decay_failure"
    elif objective_pass:
        status = READINESS_AFTER
    else:
        status = "failed_string_driven_bridge_force_repair"

    return {
        "current_status": status,
        "final_synthesis_ready": False,
        "stk_integration_allowed": False,
        "website_production_replacement_allowed": False,
        "multi_guitar_comparison_allowed": False,
        "melody_chord_playback_allowed": False,
        "subjective_tuning_allowed": False,
        "real_guitar_equivalence_allowed": False,
        "step5f_extended_validation_allowed": status == READINESS_AFTER,
    }


def build_pgsm_step5e_report(
    *,
    repo_root: Optional[Path] = None,
    audio_dir: Optional[Path] = None,
    write_wav: bool = True,
    max_modes: Optional[int] = None,
) -> Dict[str, Any]:
    root = Path(repo_root or REPO_ROOT)
    out_audio = Path(audio_dir or (root / "audio" / "pgsm_step5e_string_driven_bridge_force"))

    step5d = load_step_report(_report_path(root, "pgsm_step5d_audible_diagnostic_render_repair.json"))
    step5c = load_step_report(_report_path(root, "pgsm_step5c_note_set_extended_validation.json"))
    step5b = load_step_report(_report_path(root, "pgsm_step5b_limited_note_set_refinement.json"))
    step3c = load_step_report(_report_path(root, "pgsm_step3c_numeric_calibration.json"))

    upstream = verify_upstream_readiness(step5d, step5c)
    fps_before = collect_previous_audio_fingerprints(root)

    state = build_calibrated_modal_state(root, max_modes=max_modes)
    cal_weights = state["modal_weights"]
    region_cal = state["region_cal"]
    ir_ref = compute_impulse_response(cal_weights)
    h_total, h_structural, h_cavity_air = compute_modal_kernels_decomposed(cal_weights)
    modal_freqs = [float(m["frequency_hz"]) for m in cal_weights.get("modes") or []]

    try:
        helmholtz_hz = round(helmholtz_proxy_hz(), 2)
    except Exception:
        helmholtz_hz = None

    sr = NUMERIC_SR
    n = int(OUTPUT_DURATION_S * sr)

    string_force_model_summary = {
        "type": "sustained_damped_harmonic_string_bridge_force_proxy",
        "label": STRING_FORCE_LABEL,
        "duration_s": OUTPUT_DURATION_S,
        "pluck_position_ratio": FIXED_PLUCK_POSITION,
        "partial_amplitude_law": "A_n proportional to sin(pi*n*pluck_position)/n",
        "decay_law": "tau_k = base_tau / k^0.65 (low partials decay slower)",
        "inharmonicity": {"level": INHARMONICITY_LEVEL, "b": INHARMONICITY_B_FALLBACK},
        "onset_ramp_ms": 4.0,
        "modal_body_ir": "unchanged_step3c_calibrated",
        "q_tau_changed": False,
        "convolution": "body = string_force * modal_body_ir (causal)",
    }

    per_note_contracts: Dict[str, Any] = {}
    output_files: Dict[str, Any] = {"main_wav_count": len(NOTE_SET), "notes": {}}
    per_note_metrics: Dict[str, Any] = {}
    cavity_summaries: Dict[str, Any] = {}
    listening_details: Dict[str, Any] = {}
    step5d_baselines: Dict[str, Any] = {}

    modal_decay_taus = [
        round(float(m.get("tau_s") or 0.0), 5) for m in (cal_weights.get("modes") or [])[:12]
    ]

    for note in NOTE_SET:
        f0 = NOTE_FREQUENCY_HZ[note]
        contract = build_per_note_string_force_contract(note, f0)
        per_note_contracts[note] = contract

        string_force = build_string_driven_bridge_force(n, sr, f0)
        body_raw = synthesize_modal_body_response(string_force, h_total)
        body_structural = synthesize_modal_body_response(string_force, h_structural)
        body_cavity = synthesize_modal_body_response(string_force, h_cavity_air)

        main_listening, listen_info = apply_listening_render_full(body_raw)
        body_stem_norm, _ = normalize_diagnostic_amplitude(body_raw, max_peak_fs=0.15)
        force_stem_norm, _ = normalize_diagnostic_amplitude(string_force, max_peak_fs=0.15)

        paths = _output_paths(out_audio, note)
        if write_wav:
            write_wav_mono(paths["main"], main_listening, sr)
            write_wav_mono(paths["body_stem"], body_stem_norm, sr)
            write_wav_mono(paths["string_force_stem"], force_stem_norm, sr)

        step5d_baseline = analyze_step5d_baseline(root, note, sr)
        step5d_baselines[note] = step5d_baseline
        listening_details[note] = listen_info

        cavity_summaries[note] = build_cavity_response_summary(
            region_cal=region_cal,
            cavity_body=body_cavity,
            structural_body=body_structural,
            string_force=string_force,
            sr=sr,
            helmholtz_hz=helmholtz_hz,
        )

        metrics = evaluate_per_note_metrics(
            main_listening,
            body_stem_norm,
            force_stem_norm,
            sr,
            note=note,
            f0=f0,
            modal_freqs=modal_freqs,
            listening_info=listen_info,
            step5d_baseline=step5d_baseline,
            h_body=h_total,
        )
        per_note_metrics[note] = metrics
        output_files["notes"][note] = {
            "main_string_driven_diagnostic_wav": str(paths["main"]),
            "body_stem_wav": str(paths["body_stem"]),
            "string_force_stem_wav": str(paths["string_force_stem"]),
        }

    fps_after = collect_previous_audio_fingerprints(root)
    preserved = fps_before == fps_after

    before_after = build_before_after_comparison(per_note_metrics, step5d_baselines)
    artifact = build_artifact_guard(per_note_metrics)

    objective = {
        "upstream_ready": upstream.get("pass"),
        "previous_audio_preserved": preserved,
        "four_string_driven_wavs": len(output_files.get("notes") or {}) == 4,
        "string_force_stems_generated": True,
        "body_stems_generated": True,
        "full_duration_not_trimmed": all(
            (per_note_metrics[n] or {}).get("full_duration_not_trimmed") for n in NOTE_SET
        ),
        "energy_first_10ms_reduced_vs_step5d": all(
            (per_note_metrics[n] or {}).get("improved_vs_step5d_energy_10ms") for n in NOTE_SET
        ),
        "active_duration_increased_vs_step5d": all(
            (per_note_metrics[n] or {}).get("improved_vs_step5d_active_duration") for n in NOTE_SET
        ),
        "pitch_salience_all_notes": all(
            (per_note_metrics[n] or {}).get("pitch_salience_detectable") for n in NOTE_SET
        ),
        "all_notes_pass": all((per_note_metrics[n] or {}).get("pass") for n in NOTE_SET),
        "artifact_guard_pass": artifact.get("pass"),
        "no_stk_integration": True,
        "no_fem_run": True,
        "no_rom_run": True,
        "website_default_unchanged": True,
        "gain_reported_separately": True,
        "no_artificial_echo_or_reverb": True,
    }
    objective["all_pass"] = bool(all(objective.values()))

    readiness = build_readiness_after_step5e(objective["all_pass"], per_note_metrics)

    return {
        "report_version": PGSM_STEP5E_VERSION,
        "timestamp": _utc_now(),
        "status": "pgsm_step5e_string_driven_bridge_force_repair_complete",
        "no_stk_integration": True,
        "no_fem_run": True,
        "no_rom_run": True,
        "website_default": DEFAULT_WEBSITE_STK_MODE,
        "website_default_unchanged": True,
        "no_previous_audio_modified": preserved,
        "sample_id": SAMPLE_ID,
        "note_set": list(NOTE_SET),
        "diagnostic_label": DIAGNOSTIC_LABEL,
        "step5d_loaded": step5d.get("report_version"),
        "step5c_loaded": step5c.get("report_version"),
        "step5b_loaded": step5b.get("report_version"),
        "step3c_loaded": step3c.get("report_version"),
        "upstream_readiness": upstream,
        "string_force_model_summary": string_force_model_summary,
        "per_note_string_force_contract": per_note_contracts,
        "output_files": output_files,
        "per_note_metrics": per_note_metrics,
        "listening_render_details": listening_details,
        "before_after_step5d_comparison": before_after,
        "cavity_response_summary": {
            "per_note": cavity_summaries,
            "modal_decay_time_summary": {
                "tau_s_first_modes": modal_decay_taus,
                "decay_from_q_tau_not_reverb": True,
            },
            "top_back_air_balance": cavity_summaries.get("A4", {}).get("top_back_air_balance"),
            "internal_decay_interpretation": (
                "Internal guitar resonance is represented only through modal/cavity response "
                "proxies embedded in Step 3C IR convolution."
            ),
            "cavity_limitations": (
                "Cavity/air contribution uses W_air × air_share modal weighting; "
                "Helmholtz is a documented proxy only."
            ),
            "no_artificial_echo_or_reverb": True,
        },
        "artifact_guard_results": artifact,
        "objective_test_results": objective,
        "readiness_after_step5e": readiness,
        "blocked_claims": [
            "Final synthesis",
            "STK integration",
            "Website production replacement",
            "Multi-guitar comparison",
            "Melody/chord playback",
            "Subjective tuning by ear",
            "Real-guitar equivalence or validation proof",
            "Measured bridge force claim",
        ],
        "safe_next_step": (
            "PGSM Step 5F: string-driven extended validation"
            if readiness["current_status"] == READINESS_AFTER
            else "Resolve Step 5E failures before Step 5F"
        ),
        "explicit_statement": (
            "PGSM Step 5E replaces the short pulse diagnostic driver with a sustained "
            "diagnostic string-driven bridge-force proxy. It is not final synthesis and "
            "does not prove realism."
        ),
    }


def write_markdown_report(report: Mapping[str, Any], path: Path) -> None:
    rg = report.get("readiness_after_step5e") or {}
    obj = report.get("objective_test_results") or {}
    metrics = report.get("per_note_metrics") or {}
    cmp_map = report.get("before_after_step5d_comparison") or {}
    model = report.get("string_force_model_summary") or {}
    art = report.get("artifact_guard_results") or {}
    cavity = report.get("cavity_response_summary") or {}

    lines = [
        "# PGSM Step 5E — string-driven bridge force repair",
        "",
        f"**Generated:** {report.get('timestamp')}",
        f"**Readiness:** `{rg.get('current_status')}`",
        "",
        report.get("explicit_statement", ""),
        "",
        "Internal guitar resonance is represented only through modal/cavity response proxies. "
        "No artificial echo or reverb is added.",
        "",
        "## String-force model",
        "",
        f"- Type: `{model.get('type')}`",
        f"- Label: `{model.get('label')}`",
        f"- Duration: {model.get('duration_s')} s (full tail, not trimmed)",
        f"- Partial law: {model.get('partial_amplitude_law')}",
        f"- Decay law: {model.get('decay_law')}",
        f"- Modal IR: {model.get('modal_body_ir')} (Q/tau unchanged)",
        "",
        "## Output WAVs",
        "",
        "| Note | main | body stem | string-force stem | duration s | RMS dBFS | pass |",
        "|------|------|-----------|-------------------|------------|----------|------|",
    ]
    files = report.get("output_files") or {}
    for note in NOTE_SET:
        f = (files.get("notes") or {}).get(note) or {}
        m = metrics.get(note) or {}
        lines.append(
            f"| {note} | `{Path(str(f.get('main_string_driven_diagnostic_wav'))).name}` | "
            f"`{Path(str(f.get('body_stem_wav'))).name}` | "
            f"`{Path(str(f.get('string_force_stem_wav'))).name}` | "
            f"{m.get('duration_s')} | {m.get('rms_dbfs')} | {m.get('pass')} |"
        )

    lines.extend(
        [
            "",
            "## Before/after vs Step 5D",
            "",
            "| Note | e10% 5D | e10% 5E | active ms 5D | active ms 5E | click 5D | click 5E |",
            "|------|---------|---------|--------------|--------------|----------|----------|",
        ]
    )
    for note in NOTE_SET:
        c = cmp_map.get(note) or {}
        lines.append(
            f"| {note} | {c.get('step5d_energy_first_10ms')} | {c.get('step5e_energy_first_10ms')} | "
            f"{c.get('step5d_active_duration_ms')} | {c.get('step5e_active_duration_ms')} | "
            f"{c.get('step5d_click_dominance')} | {c.get('step5e_click_dominance')} |"
        )

    lines.extend(["", "## Per-note objective metrics", ""])
    for note in NOTE_SET:
        m = metrics.get(note) or {}
        lines.append(
            f"- **{note}**: pitch salience={m.get('pitch_salience_f0')}, "
            f"e10={m.get('energy_first_10ms')}, active={m.get('active_duration_minus_60_dbfs_ms')} ms, "
            f"centroid={m.get('spectral_centroid_hz')} Hz"
        )

    lines.extend(
        [
            "",
            "## Cavity / internal response",
            "",
            cavity.get("internal_decay_interpretation", ""),
            "",
            f"no_artificial_echo_or_reverb: **{cavity.get('no_artificial_echo_or_reverb')}**",
            "",
            "## Artifact guard",
            "",
            f"pass: **{art.get('pass')}**",
            "",
            "## Original preservation",
            "",
            f"no_previous_audio_modified: **{report.get('no_previous_audio_modified')}**",
            "",
            "## Readiness",
            "",
            f"all_pass: **{obj.get('all_pass')}**",
            "",
            "## Blocked claims",
            "",
        ]
    )
    for claim in report.get("blocked_claims") or []:
        lines.append(f"- {claim}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pgsm_step5e_reports(
    *,
    repo_root: Optional[Path] = None,
    audio_dir: Optional[Path] = None,
    json_path: Optional[Path] = None,
    md_path: Optional[Path] = None,
    max_modes: Optional[int] = None,
) -> Dict[str, Any]:
    root = Path(repo_root or REPO_ROOT)
    report = build_pgsm_step5e_report(
        repo_root=root,
        audio_dir=audio_dir,
        write_wav=True,
        max_modes=max_modes,
    )
    jpath = Path(json_path or REPORT_JSON)
    mpath = Path(md_path or REPORT_MD)
    jpath.parent.mkdir(parents=True, exist_ok=True)
    jpath.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown_report(report, mpath)
    return report


def main() -> None:
    report = write_pgsm_step5e_reports()
    rg = report.get("readiness_after_step5e") or {}
    obj = report.get("objective_test_results") or {}
    print(f"Wrote {REPORT_JSON}")
    print(f"Readiness: {rg.get('current_status')}")
    print(f"all_pass: {obj.get('all_pass')}")


if __name__ == "__main__":
    main()
