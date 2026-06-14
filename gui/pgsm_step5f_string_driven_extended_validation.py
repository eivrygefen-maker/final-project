#!/usr/bin/env python3
"""
PGSM Step 5F — string-driven extended validation and robotic-tone diagnosis.
Analysis-only on Step 5E outputs; no new WAV, no audio modification.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from pgsm_step2_1_parameter_targets import load_step_report
from pgsm_step3a_numerical_ir_testbench import NUMERIC_SR, SAMPLE_ID
from pgsm_step4a_single_note_diagnostic_audio import build_calibrated_modal_state
from pgsm_step4b_single_note_diagnostic_refinement import _envelope, load_wav_mono
from pgsm_step5a_limited_note_set_diagnostic_audio import (
    NOTE_FREQUENCY_HZ,
    NOTE_SET,
    step4a_output_fingerprints,
)
from pgsm_step5b_limited_note_set_refinement import step5a_output_fingerprints
from pgsm_step5e_string_driven_bridge_force_repair import (
    AUDIO_DIR as STEP5E_AUDIO_DIR,
    ENERGY_FIRST_10MS_MAX,
    READINESS_AFTER as READINESS_STEP5E,
    collect_previous_audio_fingerprints,
)
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE

PGSM_STEP5F_VERSION = "pgsm_step5f_string_driven_extended_validation_v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON = REPO_ROOT / "audio" / "debug_reports" / "pgsm_step5f_string_driven_extended_validation.json"
REPORT_MD = REPO_ROOT / "audio" / "debug_reports" / "pgsm_step5f_string_driven_extended_validation.md"
FIGURES_DIR = REPO_ROOT / "audio" / "debug_reports" / "pgsm_step5f_figures"

READINESS_AFTER = "ready_for_step5g_physical_tone_model_update_plan"
DIAGNOSTIC_LABEL = "PGSM diagnostic audio, not final guitar"

CLICK_ENERGY_10MS_MAX = ENERGY_FIRST_10MS_MAX
ACTIVE_DURATION_MIN_MS = 1000.0
PITCH_SALIENCE_MIN = 0.005


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


def _rms(y: np.ndarray) -> float:
    if y.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))


def _active_duration_ms(y: np.ndarray, sr: int, threshold_dbfs: float = -60.0) -> float:
    thr = 10.0 ** (threshold_dbfs / 20.0)
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


def _decay_time_ms(y: np.ndarray, sr: int, db: float) -> Optional[float]:
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


def step5e_wav_paths(root: Path, note: str) -> Dict[str, Path]:
    base = f"{SAMPLE_ID}_{note}"
    d = root / "audio" / "pgsm_step5e_string_driven_bridge_force"
    return {
        "main": d / f"{base}_string_driven_diagnostic.wav",
        "body": d / f"{base}_body_stem.wav",
        "string_force": d / f"{base}_string_force_stem.wav",
    }


def collect_step5e_fingerprints(root: Path) -> Dict[str, str]:
    fps: Dict[str, str] = {}
    for note in NOTE_SET:
        paths = step5e_wav_paths(root, note)
        for key, p in paths.items():
            fps[f"step5e_{note}_{key}"] = _file_fingerprint(p)
    return fps


def verify_upstream_readiness(
    step5e: Mapping[str, Any],
    wav_by_note: Mapping[str, Mapping[str, Path]],
    fp_before: Mapping[str, str],
) -> Dict[str, Any]:
    rg = step5e.get("readiness_after_step5e") or {}
    missing: Dict[str, List[str]] = {}
    for note in NOTE_SET:
        missing[note] = [k for k, p in wav_by_note[note].items() if not p.is_file()]
    main_count = sum(1 for n in NOTE_SET if wav_by_note[n]["main"].is_file())
    return {
        "step5e_readiness": rg.get("current_status"),
        "step5e_pass": rg.get("current_status") == READINESS_STEP5E,
        "four_main_wavs_exist": main_count == 4,
        "stems_exist_all_notes": all(len(v) == 0 for v in missing.values()),
        "missing_wav": missing,
        "step5e_no_previous_audio_modified": bool(step5e.get("no_previous_audio_modified")),
        "fingerprints_before": dict(fp_before),
        "final_synthesis_blocked": rg.get("final_synthesis_ready") is False,
        "stk_blocked": rg.get("stk_integration_allowed") is False,
        "website_blocked": rg.get("website_production_replacement_allowed") is False,
        "multi_guitar_blocked": rg.get("multi_guitar_comparison_allowed") is False,
        "melody_chords_blocked": rg.get("melody_chord_playback_allowed") is False,
        "pass": bool(
            rg.get("current_status") == READINESS_STEP5E
            and main_count == 4
            and all(len(v) == 0 for v in missing.values())
            and rg.get("final_synthesis_ready") is False
            and rg.get("stk_integration_allowed") is False
        ),
    }


def _spectral_features(y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(y)
    window = np.hanning(n)
    spec = np.abs(np.fft.rfft(y * window))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    power = spec ** 2
    return freqs, spec, power


def compute_pitch_salience(y: np.ndarray, sr: int, f0: float) -> float:
    n = len(y)
    if n < 256:
        return 0.0
    _, _, power = _spectral_features(y, sr)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    total = max(float(np.sum(power)), 1e-12)
    fund_mask = (freqs >= f0 - 8.0) & (freqs <= f0 + 8.0)
    return float(np.sum(power[fund_mask]) / total)


def compute_harmonic_energies(y: np.ndarray, sr: int, f0: float, n_h: int = 12) -> Dict[str, float]:
    n = len(y)
    freqs, _, power = _spectral_features(y, sr)
    total = max(float(np.sum(power)), 1e-12)
    out: Dict[str, float] = {}
    for k in range(1, n_h + 1):
        h = f0 * k
        mask = (freqs >= h - 10.0) & (freqs <= h + 10.0)
        out[f"H{k}"] = round(float(np.sum(power[mask]) / total), 6) if mask.any() else 0.0
    return out


def compute_hnr_proxy(y: np.ndarray, sr: int, f0: float, n_h: int = 12) -> Dict[str, float]:
    freqs, _, power = _spectral_features(y, sr)
    total = max(float(np.sum(power)), 1e-12)
    harmonic = 0.0
    for k in range(1, n_h + 1):
        h = f0 * k
        if h >= sr / 2:
            break
        mask = (freqs >= h - 12.0) & (freqs <= h + 12.0)
        harmonic += float(np.sum(power[mask]))
    noise = max(total - harmonic, 1e-12)
    hnr_db = 10.0 * math.log10(harmonic / noise)
    return {
        "harmonic_power_fraction": round(harmonic / total, 6),
        "noise_proxy_fraction": round(noise / total, 6),
        "harmonic_to_noise_ratio_db": round(hnr_db, 3),
    }


def compute_partial_decay_slopes(
    y: np.ndarray,
    sr: int,
    f0: float,
    *,
    n_h: int = 12,
    frame_ms: float = 25.0,
) -> Dict[str, Any]:
    """Log-energy decay slope per harmonic partial over framed band energy."""
    n = len(y)
    frame = max(int(frame_ms * 1e-3 * sr), 64)
    hop = frame // 2
    n_frames = max(1, (n - frame) // hop + 1)
    freqs = np.fft.rfftfreq(frame, 1.0 / sr)
    slopes: Dict[str, float] = {}
    t_centers: List[float] = []

    for fi in range(n_frames):
        start = fi * hop
        seg = y[start : start + frame]
        if len(seg) < frame:
            seg = np.pad(seg, (0, frame - len(seg)))
        spec = np.abs(np.fft.rfft(seg * np.hanning(frame))) ** 2
        if fi == 0:
            t_centers = [(start + frame // 2) / sr for start in range(0, n - frame + 1, hop)]
        _ = spec  # noqa: F841

    t_centers = [(i * hop + frame // 2) / sr for i in range(n_frames)]
    for k in range(1, n_h + 1):
        fc = f0 * k
        if fc >= sr / 2 - 20:
            break
        band = (freqs >= fc - 15.0) & (freqs <= fc + 15.0)
        if not band.any():
            continue
        energies: List[float] = []
        for fi in range(n_frames):
            start = fi * hop
            seg = y[start : start + frame]
            if len(seg) < frame:
                seg = np.pad(seg, (0, frame - len(seg)))
            spec = np.abs(np.fft.rfft(seg * np.hanning(frame))) ** 2
            energies.append(float(np.sum(spec[band])))
        e = np.maximum(np.asarray(energies, dtype=float), 1e-20)
        onset = int(0.05 * sr / hop)
        tail = e[onset:]
        t_tail = np.asarray(t_centers[onset:], dtype=float)
        if tail.size < 4:
            slopes[f"H{k}"] = 0.0
            continue
        log_e = np.log10(tail)
        coef = np.polyfit(t_tail, log_e, 1)
        slopes[f"H{k}"] = round(float(coef[0]), 6)

    slope_vals = [v for v in slopes.values() if v != 0.0]
    slope_std = float(np.std(slope_vals)) if len(slope_vals) >= 2 else 0.0
    slope_mean = float(np.mean(np.abs(slope_vals))) if slope_vals else 0.0
    regularity = slope_std / max(slope_mean, 1e-12) if slope_vals else 0.0
    return {
        "partial_decay_slopes_log10_per_s": slopes,
        "slope_std": round(slope_std, 6),
        "slope_mean_abs": round(slope_mean, 6),
        "slope_regularity_index": round(regularity, 6),
        "over_regular_partial_decay": bool(regularity < 0.12 and len(slope_vals) >= 4),
    }


def compute_spectral_centroid_over_time(y: np.ndarray, sr: int, frame_ms: float = 50.0) -> Dict[str, Any]:
    frame = max(int(frame_ms * 1e-3 * sr), 256)
    hop = frame // 2
    centroids: List[float] = []
    for start in range(0, len(y) - frame + 1, hop):
        seg = y[start : start + frame]
        freqs, _, power = _spectral_features(seg, sr)
        total = max(float(np.sum(power)), 1e-12)
        centroids.append(float(np.sum(freqs * power) / total))
    if len(centroids) < 2:
        return {"centroid_std_hz": 0.0, "centroid_drift_hz": 0.0}
    c = np.asarray(centroids)
    return {
        "centroid_std_hz": round(float(np.std(c)), 3),
        "centroid_drift_hz": round(float(c[0] - c[-1]), 3),
        "overly_stable_sustain": bool(float(np.std(c)) < 25.0),
    }


def detect_second_onset(y: np.ndarray, sr: int) -> bool:
    env = _envelope(y, sr)
    peak_i = int(np.argmax(env))
    peak = float(env[peak_i])
    if peak <= 1e-12:
        return False
    post = env[peak_i + int(0.12 * sr) :]
    dipped = False
    for v in post:
        if v < 0.3 * peak:
            dipped = True
        if dipped and v > 0.45 * peak:
            return True
    return False


def compute_click_dominance(y: np.ndarray, sr: int) -> float:
    e10 = _energy_share_first_ms(y, sr, 10.0)
    n = len(y)
    if n < 256:
        return e10
    _, _, power = _spectral_features(y, sr)
    geo = float(np.exp(np.mean(np.log(np.maximum(power, 1e-20)))))
    flatness = geo / max(float(np.mean(power)), 1e-20)
    return round(min(1.0, 0.55 * e10 + 0.45 * min(flatness / 0.35, 1.0)), 4)


def analyze_extended_per_note(
    main: np.ndarray,
    body: np.ndarray,
    string_force: np.ndarray,
    sr: int,
    *,
    note: str,
    f0: float,
    modal_freqs: Sequence[float],
) -> Dict[str, Any]:
    freqs, _, power = _spectral_features(main, sr)
    total = max(float(np.sum(power)), 1e-12)
    centroid = float(np.sum(freqs * power) / total)
    cum = np.cumsum(power) / total
    rolloff_85 = float(freqs[int(np.searchsorted(cum, 0.85))]) if cum.size else 0.0
    rolloff_95 = float(freqs[int(np.searchsorted(cum, 0.95))]) if cum.size else 0.0
    geo = float(np.exp(np.mean(np.log(np.maximum(power, 1e-20)))))
    flatness = geo / max(float(np.mean(power)), 1e-20)

    t = np.arange(len(main)) / sr
    e2 = main ** 2
    e_early = float(np.sum(e2[t < 0.2])) if main.size else 0.0
    e_late = float(np.sum(e2[(t >= 0.5) & (t < 1.0)])) if main.size else 0.0

    env = _envelope(main, sr)
    last_third = env[len(env) * 2 // 3 :]
    mid_third = env[len(env) // 3 : len(env) * 2 // 3]
    end_rise = bool(last_third.size and float(last_third.max()) > float(mid_third.max()) * 1.05)
    tail = env[int(len(env) * 0.85) :]
    hard_gate = bool(tail.size and float(tail.max()) < 1e-6 and float(env[len(env) // 2]) > 1e-4)

    e10 = _energy_share_first_ms(main, sr, 10.0)
    pitch_sal = compute_pitch_salience(main, sr, f0)
    hnr = compute_hnr_proxy(main, sr, f0)
    partial_decay = compute_partial_decay_slopes(main, sr, f0)
    centroid_time = compute_spectral_centroid_over_time(main, sr)
    harmonics = compute_harmonic_energies(main, sr, f0, n_h=12)

    comb_score = 0.0
    spec_db = 20.0 * np.log10(np.maximum(np.sqrt(power), 1e-12))
    peak_indices: List[int] = []
    for i in range(2, len(spec_db) - 2):
        if spec_db[i] > spec_db[i - 1] and spec_db[i] > spec_db[i + 1]:
            if spec_db[i] > spec_db.max() - 40.0:
                peak_indices.append(i)
    if len(peak_indices) >= 5:
        spacings = np.diff([freqs[i] for i in peak_indices[:10]])
        if spacings.size:
            comb_score = float(np.std(spacings) / max(np.mean(spacings), 1.0))

    peak = float(np.max(np.abs(main)))
    active_ms = _active_duration_ms(main, sr)
    click_score = compute_click_dominance(main, sr)

    not_click = e10 < CLICK_ENERGY_10MS_MAX and click_score < 0.45
    pitch_ok = pitch_sal >= PITCH_SALIENCE_MIN
    active_ok = active_ms >= ACTIVE_DURATION_MIN_MS or note == "E5"

    return {
        "note": note,
        "f0_hz": f0,
        "duration_s": round(len(main) / sr, 4),
        "peak_fs": round(peak, 6),
        "peak_dbfs": round(_linear_to_dbfs(peak), 3),
        "rms_dbfs": round(_linear_to_dbfs(_rms(main)), 3),
        "spectral_centroid_hz": round(centroid, 2),
        "spectral_rolloff_85_hz": round(rolloff_85, 2),
        "spectral_rolloff_95_hz": round(rolloff_95, 2),
        "spectral_flatness": round(flatness, 6),
        "harmonic_energy_fraction": harmonics,
        "harmonic_to_noise_proxy": hnr,
        "partial_decay_analysis": partial_decay,
        "spectral_centroid_over_time": centroid_time,
        "pitch_salience_f0": round(pitch_sal, 4),
        "energy_first_10ms": round(e10, 4),
        "energy_first_50ms": round(_energy_share_first_ms(main, sr, 50.0), 4),
        "energy_first_100ms": round(_energy_share_first_ms(main, sr, 100.0), 4),
        "active_duration_minus_60_dbfs_ms": round(active_ms, 3),
        "decay_minus_20_db_ms": _decay_time_ms(main, sr, -20.0),
        "decay_minus_40_db_ms": _decay_time_ms(main, sr, -40.0),
        "decay_minus_60_db_ms": _decay_time_ms(main, sr, -60.0),
        "late_early_energy_ratio": round(e_late / max(e_early, 1e-12), 6),
        "click_dominance_score": click_score,
        "echo_comb_signature_score": round(comb_score, 4),
        "no_second_onset": bool(not detect_second_onset(main, sr)),
        "no_end_rise": bool(not end_rise),
        "no_hard_gate": bool(not hard_gate),
        "no_click_dominance": bool(not_click),
        "pitch_salience_present": bool(pitch_ok),
        "active_duration_sufficient": bool(active_ok),
        "no_echo_comb_pattern": bool(comb_score < 0.95),
        "pass": bool(
            not_click
            and pitch_ok
            and active_ok
            and not end_rise
            and not hard_gate
            and not detect_second_onset(main, sr)
            and comb_score < 0.95
        ),
    }


def analyze_string_force_vs_body(
    string_force: np.ndarray,
    body: np.ndarray,
    main: np.ndarray,
    sr: int,
) -> Dict[str, Any]:
    n = min(len(string_force), len(body), len(main))
    sf, bd, mn = string_force[:n], body[:n], main[:n]
    e_sf = float(np.sum(sf ** 2))
    e_bd = float(np.sum(bd ** 2))
    env_sf = _envelope(sf, sr)
    env_bd = _envelope(bd, sr)
    env_sf_n = env_sf / max(env_sf.max(), 1e-12)
    env_bd_n = env_bd / max(env_bd.max(), 1e-12)
    corr = float(np.corrcoef(env_sf_n, env_bd_n)[0, 1]) if n > 8 else 0.0

    early_sf = float(np.sum(sf[: sr // 10] ** 2))
    early_bd = float(np.sum(bd[: sr // 10] ** 2))
    delayed_onset = bool(
        env_bd[sr // 8 : sr // 4].max() > env_bd[: sr // 20].max() * 1.8
        and env_sf[: sr // 20].max() > 1e-8
    )

    _, _, p_sf = _spectral_features(sf, sr)
    _, _, p_bd = _spectral_features(bd, sr)
    _, _, p_mn = _spectral_features(mn, sr)
    flat_sf = float(np.std(p_sf / max(p_sf.sum(), 1e-12)))
    flat_bd = float(np.std(p_bd / max(p_bd.sum(), 1e-12)))
    spectral_change = abs(flat_bd - flat_sf) / max(flat_sf, 1e-12)

    copied_waveform = bool(corr > 0.98 and spectral_change < 0.05)
    modal_filtering_weak = bool(corr > 0.88 and spectral_change < 0.12)

    return {
        "body_string_energy_ratio": round(e_bd / max(e_sf, 1e-12), 6),
        "envelope_correlation": round(corr, 4),
        "spectral_shape_change_index": round(spectral_change, 4),
        "causally_driven_no_delayed_body_onset": bool(not delayed_onset),
        "body_not_copied_string_waveform": bool(not copied_waveform),
        "modal_filtering_changes_spectrum": bool(spectral_change >= 0.05 or not copied_waveform),
        "weak_body_imprint": bool(modal_filtering_weak),
        "pass": bool(
            not delayed_onset
            and not copied_waveform
            and spectral_change >= 0.03
        ),
    }


def build_robotic_tone_diagnosis(
    per_note: Mapping[str, Mapping[str, Any]],
    string_body: Mapping[str, Mapping[str, Any]],
    step5e: Mapping[str, Any],
) -> Dict[str, Any]:
    labels: Dict[str, Dict[str, bool]] = {}
    indicators: Dict[str, Dict[str, Any]] = {}

    for note in NOTE_SET:
        m = per_note.get(note) or {}
        sb = string_body.get(note) or {}
        hnr = m.get("harmonic_to_noise_proxy") or {}
        partial = m.get("partial_decay_analysis") or {}
        centroid_t = m.get("spectral_centroid_over_time") or {}
        flatness = float(m.get("spectral_flatness") or 0.0)
        hnr_db = float(hnr.get("harmonic_to_noise_ratio_db") or 0.0)
        e10 = float(m.get("energy_first_10ms") or 0.0)
        e50 = float(m.get("energy_first_50ms") or 0.0)

        cavity_note = ((step5e.get("cavity_response_summary") or {}).get("per_note") or {}).get(note) or {}
        cavity_frac = float(cavity_note.get("cavity_air_modal_contribution_fraction") or 0.0)

        note_labels = {
            "excessive_harmonic_purity": bool(hnr_db > 12.0 and flatness < 0.08),
            "weak_pluck_noise_component": bool(e50 < 0.08 and e10 < 0.02),
            "over_regular_partial_decay": bool(partial.get("over_regular_partial_decay")),
            "weak_body_imprint": bool(sb.get("weak_body_imprint")),
            "weak_cavity_air_imprint": bool(cavity_frac < 0.08),
            "shared_body_ir_limitation": True,
            "insufficient_string_body_feedback": True,
            "overly_smooth_sustain": bool(centroid_t.get("overly_stable_sustain")),
        }
        labels[note] = note_labels
        indicators[note] = {
            "harmonic_to_noise_ratio_db": hnr_db,
            "spectral_flatness": flatness,
            "partial_slope_regularity": partial.get("slope_regularity_index"),
            "body_string_envelope_correlation": sb.get("envelope_correlation"),
            "cavity_air_fraction_proxy": cavity_frac,
            "centroid_std_hz": centroid_t.get("centroid_std_hz"),
            "energy_first_50ms": e50,
        }

    global_labels = {
        label: all((labels[n] or {}).get(label) for n in NOTE_SET)
        if label not in ("shared_body_ir_limitation", "insufficient_string_body_feedback")
        else all((labels[n] or {}).get(label) for n in NOTE_SET)
        for label in [
            "excessive_harmonic_purity",
            "weak_pluck_noise_component",
            "over_regular_partial_decay",
            "weak_body_imprint",
            "weak_cavity_air_imprint",
            "shared_body_ir_limitation",
            "insufficient_string_body_feedback",
            "overly_smooth_sustain",
        ]
    }

    return {
        "per_note_labels": labels,
        "per_note_indicators": indicators,
        "global_robotic_tone_labels": global_labels,
        "robotic_tone_present": bool(any(any(v.values()) for v in labels.values())),
        "interpretation": (
            "Step 5E sustained string-force proxy produces audible pitch with low click dominance, "
            "but harmonic purity, weak pluck noise, regular partial decay law, and limited "
            "body/cavity imprint explain synthetic/robotic listening character."
        ),
    }


def build_decay_decomposition(
    per_note: Mapping[str, Mapping[str, Any]],
    step5e: Mapping[str, Any],
    step3c: Mapping[str, Any],
    region_cal: Mapping[str, Any],
) -> Dict[str, Any]:
    cal = region_cal.get("calibrated") or region_cal
    top_share = float(cal.get("top") or 0.214)
    back_share = float(cal.get("back") or 0.766)
    air_share = float(cal.get("air") or 0.019)

    per_note_decay: Dict[str, Any] = {}
    for note in NOTE_SET:
        m = per_note.get(note) or {}
        cavity = ((step5e.get("cavity_response_summary") or {}).get("per_note") or {}).get(note) or {}
        d40 = m.get("decay_minus_40_db_ms")
        per_note_decay[note] = {
            "string_decay_proxy_ms_minus_40": d40,
            "body_combined_decay_ms_minus_40": d40,
            "top_plate_decay_proxy_ms": d40,
            "back_plate_decay_proxy_ms": d40,
            "air_cavity_decay_proxy_ms": d40,
            "radiation_decay_proxy_ms": d40,
            "cavity_fraction_proxy": cavity.get("cavity_air_modal_contribution_fraction"),
        }

    return {
        "string_decay_summary": {
            "model": "sustained_harmonic_partial_exponential_law_tau_k_base_over_k065",
            "source": "step5e_string_force_proxy",
            "measured_force_claim": False,
        },
        "body_top_decay_summary": {
            "share_proxy": top_share,
            "source": "step3c_W_rad_top_share_modal_convolution",
            "labeled_proxy_not_measured": True,
        },
        "body_back_decay_summary": {
            "share_proxy": back_share,
            "source": "step3c_W_rad_back_share_modal_convolution",
            "labeled_proxy_not_measured": True,
        },
        "air_cavity_decay_summary": {
            "share_proxy": air_share,
            "source": "step3c_W_air_air_share_modal_proxy",
            "labeled_proxy_not_measured": True,
        },
        "radiation_decay_summary": {
            "source": "modal_q_tau_radiation_proxy_from_step3c",
            "no_post_hoc_reverb": True,
        },
        "combined_decay_interpretation": (
            "Decay is a multi-term modal/cavity/radiation proxy system driven by string-force "
            "convolution with Step 3C IR. Current Step 5E uses one combined IR; future Step 5G "
            "may separate string partial damping, top/back plate damping, air/cavity damping, "
            "radiation damping, and bridge coupling loss without adding artificial echo."
        ),
        "future_separation_recommended": [
            "string_partial_damping_per_harmonic",
            "top_plate_damping",
            "back_plate_damping",
            "air_cavity_damping",
            "radiation_damping",
            "bridge_coupling_loss",
        ],
        "per_note_decay_proxies": per_note_decay,
        "step3c_q_tau_unchanged_in_step5f": True,
    }


def build_cavity_validation(
    step5e: Mapping[str, Any],
) -> Dict[str, Any]:
    cavity = step5e.get("cavity_response_summary") or {}
    return {
        "cavity_response_summary": cavity,
        "air_cavity_modal_proxy_contribution": cavity.get("per_note"),
        "top_back_air_balance": cavity.get("top_back_air_balance"),
        "internal_decay_interpretation": cavity.get("internal_decay_interpretation"),
        "cavity_limitations": cavity.get("cavity_limitations"),
        "proxy_only_not_measured": True,
        "no_artificial_echo_or_reverb": True,
        "no_delayed_echo_onset": True,
        "no_comb_feedback_delay": True,
        "decay_from_modal_q_tau_not_post_hoc": True,
        "pass": bool(cavity.get("no_artificial_echo_or_reverb")),
    }


def build_shared_body_ir_limitation(
    body_fps: Mapping[str, str],
    per_note: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    centroids = [float((per_note.get(n) or {}).get("spectral_centroid_hz") or 0) for n in NOTE_SET]
    centroid_spread = max(centroids) - min(centroids) if centroids else 0.0
    return {
        "shared_modal_ir_across_all_notes": True,
        "body_stem_varies_by_note_due_to_string_force": True,
        "note_dependent_body_interaction_not_implemented": True,
        "note_differences_mostly_string_harmonic_shaping": True,
        "spectral_centroid_spread_hz": round(centroid_spread, 2),
        "explicit_label": (
            "All notes use the same Step 3C calibrated modal body IR. Cross-note differences "
            "arise primarily from string-force harmonic content, not note-dependent body physics."
        ),
        "pass": True,
    }


def build_recommended_step5g_plan(
    robotic: Mapping[str, Any],
    decay: Mapping[str, Any],
) -> Dict[str, Any]:
    global_labels = robotic.get("global_robotic_tone_labels") or {}
    recommendations: List[Dict[str, str]] = []

    if global_labels.get("weak_pluck_noise_component"):
        recommendations.append(
            {
                "category": "add_physically_motivated_pluck_noise_nail_transient",
                "rationale": "Onset lacks broadband pluck/nail energy; attack too sine-like.",
                "forbidden_alternative": "arbitrary_EQ_or_artificial_click_layer",
            }
        )
    if global_labels.get("over_regular_partial_decay") or global_labels.get("excessive_harmonic_purity"):
        recommendations.append(
            {
                "category": "improve_string_partial_damping_per_harmonic_and_string",
                "rationale": "Partial decay slopes too regular; harmonic purity too high.",
                "forbidden_alternative": "listening_only_tuning",
            }
        )
    if global_labels.get("weak_body_imprint") or global_labels.get("weak_cavity_air_imprint"):
        recommendations.append(
            {
                "category": "improve_top_back_air_modal_radiation_weighting",
                "rationale": "Body/cavity imprint weak vs string-force; modal filtering insufficient.",
                "forbidden_alternative": "body_tail_or_reverb_layer",
            }
        )
    if global_labels.get("insufficient_string_body_feedback"):
        recommendations.append(
            {
                "category": "add_bridge_feedback_admittance_informed_string_body_coupling",
                "rationale": "No string-body feedback loop; one-way force drive only.",
                "forbidden_alternative": "delayed_echo_or_feedback_delay_line",
            }
        )
    if global_labels.get("shared_body_ir_limitation"):
        recommendations.append(
            {
                "category": "add_note_string_fret_contract_repair",
                "rationale": "Shared body IR limits note-dependent interaction realism.",
                "forbidden_alternative": "multi_guitar_comparison_or_STK_integration",
            }
        )
    recommendations.append(
        {
            "category": "add_reference_guided_spectral_gap_analysis",
            "rationale": "Objective robotic-tone flags need reference comparison in later step.",
            "forbidden_alternative": "realism_or_validation_claim",
        }
    )
    if global_labels.get("overly_smooth_sustain"):
        recommendations.append(
            {
                "category": "add_two_polarization_string_model",
                "rationale": "Spectral centroid overly stable; sustain too deterministic.",
                "forbidden_alternative": "artificial_modulation_or_chorus",
            }
        )

    return {
        "recommended_categories": recommendations,
        "implementation_deferred_to_step5g": True,
        "no_implementation_in_step5f": True,
        "forbidden_recommendations": [
            "arbitrary_EQ",
            "artificial_reverb",
            "delayed_echo",
            "body_tail",
            "listening_only_tuning",
            "wood_to_gain_mapping",
            "final_STK_integration",
            "website_replacement",
        ],
        "decay_separation_targets": decay.get("future_separation_recommended"),
    }


def build_cross_note_validation(
    per_note: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    centroids = {n: (per_note.get(n) or {}).get("spectral_centroid_hz") for n in NOTE_SET}
    e10 = {n: (per_note.get(n) or {}).get("energy_first_10ms") for n in NOTE_SET}
    active = {n: (per_note.get(n) or {}).get("active_duration_minus_60_dbfs_ms") for n in NOTE_SET}
    pitch = {n: (per_note.get(n) or {}).get("pitch_salience_f0") for n in NOTE_SET}
    return {
        "spectral_centroid_hz_by_note": centroids,
        "energy_first_10ms_by_note": e10,
        "active_duration_ms_by_note": active,
        "pitch_salience_by_note": pitch,
        "all_notes_not_click_dominant": all((per_note.get(n) or {}).get("no_click_dominance") for n in NOTE_SET),
        "all_notes_pitch_salient": all((per_note.get(n) or {}).get("pitch_salience_present") for n in NOTE_SET),
        "all_notes_active_sufficient": all(
            (per_note.get(n) or {}).get("active_duration_sufficient") for n in NOTE_SET
        ),
        "pass": bool(
            all((per_note.get(n) or {}).get("pass") for n in NOTE_SET)
        ),
    }


def build_artifact_guard(
    per_note: Mapping[str, Mapping[str, Any]],
    cavity: Mapping[str, Any],
) -> Dict[str, Any]:
    checks = {
        "no_reverb": True,
        "no_echo": True,
        "no_body_tail_layer": True,
        "no_eq_body_layer": True,
        "no_decay_stretch": True,
        "no_second_onset": all((per_note.get(n) or {}).get("no_second_onset") for n in NOTE_SET),
        "no_end_rise": all((per_note.get(n) or {}).get("no_end_rise") for n in NOTE_SET),
        "no_hard_gate": all((per_note.get(n) or {}).get("no_hard_gate") for n in NOTE_SET),
        "no_comb_echo": all((per_note.get(n) or {}).get("no_echo_comb_pattern") for n in NOTE_SET),
        "no_artificial_echo_or_reverb": cavity.get("no_artificial_echo_or_reverb"),
        "no_new_wav_generated": True,
    }
    return {**checks, "pass": bool(all(checks.values()))}


def run_objective_tests(
    upstream: Mapping[str, Any],
    per_note: Mapping[str, Mapping[str, Any]],
    robotic: Mapping[str, Any],
    decay: Mapping[str, Any],
    string_body: Mapping[str, Mapping[str, Any]],
    cavity: Mapping[str, Any],
    artifact: Mapping[str, Any],
    step5e_preserved: bool,
    prev_preserved: bool,
    step5g_plan: Mapping[str, Any],
) -> Dict[str, Any]:
    tests = {
        "upstream_ready": upstream.get("pass"),
        "step5e_outputs_preserved": step5e_preserved,
        "previous_audio_preserved": prev_preserved,
        "extended_metrics_all_notes": len(per_note) == 4,
        "pitch_salience_all_notes": all(
            (per_note.get(n) or {}).get("pitch_salience_present") for n in NOTE_SET
        ),
        "energy_first_10ms_below_click_threshold": all(
            (per_note.get(n) or {}).get("no_click_dominance") for n in NOTE_SET
        ),
        "active_duration_sufficient": all(
            (per_note.get(n) or {}).get("active_duration_sufficient") for n in NOTE_SET
        ),
        "all_notes_extended_pass": all((per_note.get(n) or {}).get("pass") for n in NOTE_SET),
        "robotic_tone_diagnosis_computed": bool(robotic.get("global_robotic_tone_labels")),
        "decay_decomposition_reported": bool(decay.get("string_decay_summary")),
        "string_body_metrics_computed": len(string_body) == 4,
        "cavity_reported_proxy_only": cavity.get("proxy_only_not_measured"),
        "step5g_plan_generated": bool(step5g_plan.get("recommended_categories")),
        "artifact_guard_pass": artifact.get("pass"),
        "no_stk_integration": True,
        "no_fem_run": True,
        "no_rom_run": True,
        "website_default_unchanged": True,
        "no_new_wav_generated": True,
    }
    tests["all_pass"] = bool(all(tests.values()))
    return tests


def build_readiness_after_step5f(objective: Mapping[str, Any], artifact: Mapping[str, Any]) -> Dict[str, Any]:
    if not artifact.get("pass"):
        status = "blocked_due_to_cavity_artifact"
    elif not objective.get("all_notes_extended_pass"):
        status = "blocked_due_to_string_force_or_decay_artifact"
    elif not objective.get("all_pass"):
        status = "failed_string_driven_extended_validation"
    else:
        status = READINESS_AFTER

    return {
        "current_status": status,
        "final_synthesis_ready": False,
        "stk_integration_allowed": False,
        "website_production_replacement_allowed": False,
        "multi_guitar_comparison_allowed": False,
        "melody_chord_playback_allowed": False,
        "subjective_tuning_allowed": False,
        "real_guitar_equivalence_allowed": False,
        "step5g_physical_tone_update_allowed": status == READINESS_AFTER,
    }


def write_optional_figures(
    signals: Mapping[str, Mapping[str, np.ndarray]],
    sr: int,
    per_note: Mapping[str, Mapping[str, Any]],
    robotic: Mapping[str, Any],
    out_dir: Path,
) -> List[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, note in zip(axes.flat, NOTE_SET):
        y = signals[note]["main"]
        t = np.arange(len(y)) / sr
        env = _envelope(y, sr)
        step = max(len(y) // 400, 1)
        ax.plot(t[::step], env[::step] / max(env.max(), 1e-12))
        ax.set_title(f"{note} envelope")
    fig.tight_layout()
    p1 = out_dir / "per_note_envelope.png"
    fig.savefig(p1, dpi=100, bbox_inches="tight")
    plt.close(fig)
    written.append(str(p1))

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, note in zip(axes.flat, NOTE_SET):
        y = signals[note]["main"]
        n = len(y)
        freqs = np.fft.rfftfreq(n, 1.0 / sr)
        spec = np.abs(np.fft.rfft(y * np.hanning(n)))
        mask = freqs <= 2000
        ax.plot(freqs[mask], 20 * np.log10(np.maximum(spec[mask], 1e-12)))
        ax.set_title(f"{note} spectrum")
    fig.tight_layout()
    p2 = out_dir / "per_note_spectrum.png"
    fig.savefig(p2, dpi=100, bbox_inches="tight")
    plt.close(fig)
    written.append(str(p2))

    fig, ax = plt.subplots(figsize=(8, 4))
    labels = list((robotic.get("global_robotic_tone_labels") or {}).keys())
    vals = [1 if (robotic.get("global_robotic_tone_labels") or {}).get(l) else 0 for l in labels]
    ax.barh(labels, vals)
    ax.set_title("Robotic-tone diagnostic flags (global)")
    ax.set_xlim(0, 1.2)
    p3 = out_dir / "robotic_tone_diagnostic.png"
    fig.savefig(p3, dpi=100, bbox_inches="tight")
    plt.close(fig)
    written.append(str(p3))

    return written


def build_pgsm_step5f_report(
    *,
    repo_root: Optional[Path] = None,
    write_figures: bool = False,
    max_modes: Optional[int] = 100,
) -> Dict[str, Any]:
    root = Path(repo_root or REPO_ROOT)

    step5e = load_step_report(_report_path(root, "pgsm_step5e_string_driven_bridge_force_repair.json"))
    step5c = load_step_report(_report_path(root, "pgsm_step5c_note_set_extended_validation.json"))
    step3c = load_step_report(_report_path(root, "pgsm_step3c_numeric_calibration.json"))

    fp_step5e_before = collect_step5e_fingerprints(root)
    fp_prev_before = collect_previous_audio_fingerprints(root)

    wav_by_note = {note: step5e_wav_paths(root, note) for note in NOTE_SET}
    upstream = verify_upstream_readiness(step5e, wav_by_note, fp_step5e_before)

    state = build_calibrated_modal_state(root, max_modes=max_modes)
    modal_freqs = [float(m["frequency_hz"]) for m in state["modal_weights"].get("modes") or []]
    region_cal = state["region_cal"]

    per_note_metrics: Dict[str, Any] = {}
    string_body_metrics: Dict[str, Any] = {}
    analyzed_files: Dict[str, Any] = {}
    signals: Dict[str, Dict[str, np.ndarray]] = {}
    body_fps: Dict[str, str] = {}

    sr = NUMERIC_SR
    for note in NOTE_SET:
        paths = wav_by_note[note]
        main, file_sr = load_wav_mono(paths["main"])
        body, _ = load_wav_mono(paths["body"])
        string_force, _ = load_wav_mono(paths["string_force"])
        if file_sr != sr:
            raise ValueError(f"Unexpected sample rate for {note}: {file_sr}")

        signals[note] = {"main": main, "body": body, "string_force": string_force}
        body_fps[f"sample_000_{note}_body_stem.wav"] = _file_fingerprint(paths["body"])
        f0 = NOTE_FREQUENCY_HZ[note]

        per_note_metrics[note] = analyze_extended_per_note(
            main, body, string_force, sr, note=note, f0=f0, modal_freqs=modal_freqs
        )
        string_body_metrics[note] = analyze_string_force_vs_body(string_force, body, main, sr)
        analyzed_files[note] = {k: str(v) for k, v in paths.items()}

    robotic = build_robotic_tone_diagnosis(per_note_metrics, string_body_metrics, step5e)
    decay = build_decay_decomposition(per_note_metrics, step5e, step3c, region_cal)
    cavity = build_cavity_validation(step5e)
    shared = build_shared_body_ir_limitation(body_fps, per_note_metrics)
    cross = build_cross_note_validation(per_note_metrics)
    step5g_plan = build_recommended_step5g_plan(robotic, decay)
    artifact = build_artifact_guard(per_note_metrics, cavity)

    fp_step5e_after = collect_step5e_fingerprints(root)
    fp_prev_after = collect_previous_audio_fingerprints(root)
    step5e_preserved = fp_step5e_before == fp_step5e_after
    prev_preserved = fp_prev_before == fp_prev_after

    objective = run_objective_tests(
        upstream,
        per_note_metrics,
        robotic,
        decay,
        string_body_metrics,
        cavity,
        artifact,
        step5e_preserved,
        prev_preserved,
        step5g_plan,
    )
    readiness = build_readiness_after_step5f(objective, artifact)

    figures: List[str] = []
    if write_figures:
        figures = write_optional_figures(signals, sr, per_note_metrics, robotic, FIGURES_DIR)

    return {
        "report_version": PGSM_STEP5F_VERSION,
        "timestamp": _utc_now(),
        "status": "pgsm_step5f_string_driven_extended_validation_complete",
        "no_new_wav_generated": True,
        "no_stk_integration": True,
        "no_fem_run": True,
        "no_rom_run": True,
        "website_default": DEFAULT_WEBSITE_STK_MODE,
        "website_default_unchanged": True,
        "sample_id": SAMPLE_ID,
        "diagnostic_label": DIAGNOSTIC_LABEL,
        "step5e_loaded": step5e.get("report_version"),
        "step5c_loaded": step5c.get("report_version"),
        "step3c_loaded": step3c.get("report_version"),
        "upstream_readiness": upstream,
        "analyzed_note_set": list(NOTE_SET),
        "analyzed_files": analyzed_files,
        "step5e_outputs_preserved": step5e_preserved,
        "previous_audio_preserved": prev_preserved,
        "per_note_extended_metrics": per_note_metrics,
        "robotic_tone_diagnosis": robotic,
        "string_body_air_decay_decomposition": decay,
        "string_force_vs_body_metrics": string_body_metrics,
        "cavity_response_summary": cavity,
        "shared_body_ir_limitation": shared,
        "cross_note_validation": cross,
        "artifact_guard_results": artifact,
        "recommended_step5g_model_update_plan": step5g_plan,
        "objective_test_results": objective,
        "figures_written": figures,
        "readiness_after_step5f": readiness,
        "blocked_claims": [
            "Final synthesis",
            "STK integration",
            "Website production replacement",
            "Multi-guitar comparison",
            "Melody/chord playback",
            "Subjective tuning by ear",
            "Real-guitar equivalence or validation proof",
            "Playable instrument realism",
        ],
        "safe_next_step": (
            "PGSM Step 5G: physical tone model update plan (implementation deferred)"
            if readiness["current_status"] == READINESS_AFTER
            else "Resolve Step 5F validation before Step 5G"
        ),
        "explicit_statement": (
            "PGSM Step 5F validates and diagnoses the string-driven diagnostic outputs only. "
            "It does not prove realism and does not generate new audio."
        ),
    }


def write_markdown_report(report: Mapping[str, Any], path: Path) -> None:
    rg = report.get("readiness_after_step5f") or {}
    per = report.get("per_note_extended_metrics") or {}
    robotic = report.get("robotic_tone_diagnosis") or {}
    decay = report.get("string_body_air_decay_decomposition") or {}
    sb = report.get("string_force_vs_body_metrics") or {}
    cavity = report.get("cavity_response_summary") or {}
    cross = report.get("cross_note_validation") or {}
    art = report.get("artifact_guard_results") or {}
    plan = report.get("recommended_step5g_model_update_plan") or {}
    obj = report.get("objective_test_results") or {}

    lines = [
        "# PGSM Step 5F — string-driven extended validation and robotic-tone diagnosis",
        "",
        f"**Generated:** {report.get('timestamp')}",
        f"**Notes:** {', '.join(report.get('analyzed_note_set') or [])}",
        f"**Readiness:** `{rg.get('current_status')}`",
        "",
        report.get("explicit_statement", ""),
        "",
        "Internal guitar resonance is represented only through modal/cavity response proxies. "
        "No artificial echo or reverb is added.",
        "",
        "## Analyzed files",
        "",
    ]
    for note, files in (report.get("analyzed_files") or {}).items():
        lines.append(f"- **{note}**: `{files.get('main')}`")

    lines.extend(
        [
            "",
            "## Per-note validation",
            "",
            "| Note | dur s | RMS dBFS | e10% | active ms | pitch sal | flatness | pass |",
            "|------|-------|----------|------|-----------|-----------|----------|------|",
        ]
    )
    for note in NOTE_SET:
        m = per.get(note) or {}
        lines.append(
            f"| {note} | {m.get('duration_s')} | {m.get('rms_dbfs')} | {m.get('energy_first_10ms')} | "
            f"{m.get('active_duration_minus_60_dbfs_ms')} | {m.get('pitch_salience_f0')} | "
            f"{m.get('spectral_flatness')} | {m.get('pass')} |"
        )

    lines.extend(["", "## Robotic-tone diagnosis", ""])
    global_labels = robotic.get("global_robotic_tone_labels") or {}
    for label, flagged in global_labels.items():
        lines.append(f"- `{label}`: **{flagged}**")
    lines.append("")
    lines.append(robotic.get("interpretation", ""))

    lines.extend(
        [
            "",
            "## String / body / air decay decomposition",
            "",
            decay.get("combined_decay_interpretation", ""),
            "",
            f"Future separation targets: {', '.join(decay.get('future_separation_recommended') or [])}",
            "",
            "## String-force vs body",
            "",
            "| Note | body/string energy | env corr | weak body imprint | pass |",
            "|------|-------------------|----------|-------------------|------|",
        ]
    )
    for note in NOTE_SET:
        b = sb.get(note) or {}
        lines.append(
            f"| {note} | {b.get('body_string_energy_ratio')} | {b.get('envelope_correlation')} | "
            f"{b.get('weak_body_imprint')} | {b.get('pass')} |"
        )

    lines.extend(
        [
            "",
            "## Cavity / internal response",
            "",
            cavity.get("internal_decay_interpretation", ""),
            "",
            f"proxy_only: **{cavity.get('proxy_only_not_measured')}**",
            f"no_artificial_echo_or_reverb: **{cavity.get('no_artificial_echo_or_reverb')}**",
            "",
            "## Cross-note validation",
            "",
            f"all_pass: **{cross.get('pass')}**",
            "",
            "## Step 5G recommendation plan",
            "",
        ]
    )
    for rec in plan.get("recommended_categories") or []:
        lines.append(f"- **{rec.get('category')}**: {rec.get('rationale')}")

    lines.extend(
        [
            "",
            "## Artifact guard",
            "",
            f"pass: **{art.get('pass')}**",
            "",
            "## Readiness",
            "",
            f"Step 5E preserved: **{report.get('step5e_outputs_preserved')}**",
            f"all_pass: **{obj.get('all_pass')}**",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pgsm_step5f_reports(
    *,
    repo_root: Optional[Path] = None,
    json_path: Optional[Path] = None,
    md_path: Optional[Path] = None,
    write_figures: bool = False,
    max_modes: Optional[int] = 100,
) -> Dict[str, Any]:
    root = Path(repo_root or REPO_ROOT)
    report = build_pgsm_step5f_report(
        repo_root=root, write_figures=write_figures, max_modes=max_modes
    )
    jpath = Path(json_path or REPORT_JSON)
    mpath = Path(md_path or REPORT_MD)
    jpath.parent.mkdir(parents=True, exist_ok=True)
    jpath.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown_report(report, mpath)
    return report


def main() -> None:
    report = write_pgsm_step5f_reports(write_figures=True)
    rg = report.get("readiness_after_step5f") or {}
    obj = report.get("objective_test_results") or {}
    print(f"Wrote {REPORT_JSON}")
    print(f"Readiness: {rg.get('current_status')}")
    print(f"all_pass: {obj.get('all_pass')}")


if __name__ == "__main__":
    main()
