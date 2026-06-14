#!/usr/bin/env python3
"""
PGSM Step 4C — single-note extended diagnostic validation.
Analyzes existing Step 4A audio only; no new WAV, no STK/FEM/ROM.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from pgsm_step2_1_parameter_targets import load_step_report
from pgsm_step3a_numerical_ir_testbench import F0_HZ, MODAL_BANDS, NUMERIC_SR, SAMPLE_ID
from pgsm_step4a_single_note_diagnostic_audio import build_calibrated_modal_state
from pgsm_step4b_single_note_diagnostic_refinement import (
    READINESS_STEP4C,
    STEP4A_AUDIO_DIR,
    _envelope,
    load_wav_mono,
)
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE

PGSM_STEP4C_VERSION = "pgsm_step4c_single_note_extended_validation_v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON = REPO_ROOT / "audio" / "debug_reports" / "pgsm_step4c_single_note_extended_validation.json"
REPORT_MD = REPO_ROOT / "audio" / "debug_reports" / "pgsm_step4c_single_note_extended_validation.md"
FIGURES_DIR = REPO_ROOT / "audio" / "debug_reports" / "pgsm_step4c_figures"

READINESS_STEP5A = "ready_for_step5a_limited_note_set_diagnostic_audio"
NOTE = "A4"
DIAGNOSTIC_LABEL = "PGSM diagnostic audio, not final guitar"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _report_path(root: Path, name: str) -> Path:
    return root / "audio" / "debug_reports" / name


def _wav_paths(root: Path, step4a: Mapping[str, Any]) -> Dict[str, Path]:
    outputs = step4a.get("output_files") or {}
    keys = {
        "main": "main_diagnostic_wav",
        "body": "body_stem_wav",
        "excitation": "excitation_stem_wav",
    }
    defaults = {
        "main": STEP4A_AUDIO_DIR / "sample_000_A4_diagnostic.wav",
        "body": STEP4A_AUDIO_DIR / "sample_000_A4_body_stem.wav",
        "excitation": STEP4A_AUDIO_DIR / "sample_000_A4_excitation_stem.wav",
    }
    paths: Dict[str, Path] = {}
    for k, out_key in keys.items():
        rel = outputs.get(out_key)
        p = Path(str(rel)) if rel else defaults[k]
        paths[k] = p if p.is_file() else defaults[k]
    return paths


def verify_upstream_readiness(
    step4b: Mapping[str, Any],
    step4a: Mapping[str, Any],
    wav_paths: Mapping[str, Path],
) -> Dict[str, Any]:
    rg4b = step4b.get("readiness_after_step4b") or {}
    missing = [k for k, p in wav_paths.items() if not p.is_file()]
    return {
        "step4b_readiness": rg4b.get("current_status"),
        "step4b_pass": rg4b.get("current_status") == READINESS_STEP4C,
        "diagnostic_label": step4a.get("diagnostic_label"),
        "final_synthesis_blocked": rg4b.get("final_synthesis_ready") is False,
        "stk_blocked": rg4b.get("stk_integration_allowed") is False,
        "website_blocked": rg4b.get("website_production_replacement_allowed") is False,
        "multi_guitar_blocked": rg4b.get("multi_guitar_comparison_allowed") is False,
        "wav_files_exist": len(missing) == 0,
        "missing_wav": missing,
        "pass": bool(
            rg4b.get("current_status") == READINESS_STEP4C
            and step4a.get("diagnostic_label") == DIAGNOSTIC_LABEL
            and rg4b.get("final_synthesis_ready") is False
            and len(missing) == 0
        ),
    }


def _decay_time_ms(env: np.ndarray, t: np.ndarray, db: float) -> Optional[float]:
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


def analyze_extended_envelope(y: np.ndarray, sr: int, q_mean: float) -> Dict[str, Any]:
    t = np.arange(len(y)) / sr
    env = _envelope(y, sr)
    e2 = y ** 2
    cum_energy = np.cumsum(e2) / max(np.sum(e2), 1e-12)

    peak_i = int(np.argmax(env))
    attack_ms = float(t[int(np.argmax(env >= 0.05 * env.max()))] * 1000.0) if env.max() > 0 else 0.0

    def _band_e(t0: float, t1: float) -> float:
        m = (t >= t0) & (t < t1)
        return float(np.sum(e2[m])) if m.any() else 0.0

    e_early = _band_e(0.0, 0.2)
    e_mid = _band_e(0.5, 1.0)
    e_late = _band_e(1.0, 2.0)

    log_env = np.log10(np.maximum(env, 1e-12))
    d_log = np.diff(log_env)
    log_smoothness = float(np.std(d_log)) if d_log.size else 0.0

    late_region = env[int(len(env) * 0.6) : int(len(env) * 0.85)]
    mid_region = env[int(len(env) * 0.3) : int(len(env) * 0.55)]
    bump_score = float(late_region.max() / max(mid_region.max(), 1e-12)) if late_region.size else 0.0

    last_third = env[len(env) * 2 // 3 :]
    mid_third = env[len(env) // 3 : len(env) * 2 // 3]
    end_rise_score = float(last_third.max() / max(mid_third.max(), 1e-12)) if last_third.size else 0.0

    tail = env[int(len(env) * 0.9) :]
    hard_gate_score = float(tail.max() / max(env[int(len(env) * 0.5)], 1e-12)) if tail.size else 0.0

    expected_tau = q_mean / (math.pi * max(F0_HZ, 1.0))
    decay_40 = _decay_time_ms(env, t, -40.0)

    beating_allowed = log_smoothness > 0.02 and bump_score < 1.3

    no_late_rise = bool(end_rise_score <= 1.05)
    no_hard_cut = bool(hard_gate_score > 1e-4 or env[int(len(env) * 0.5)] < 1e-6)
    no_unexplained_bump = bool(bump_score <= 1.25)
    decay_consistent = bool(decay_40 is None or decay_40 >= expected_tau * 500)

    return {
        "attack_time_ms": round(attack_ms, 3),
        "peak_time_ms": round(float(t[peak_i] * 1000.0), 3),
        "decay_ms": {
            "minus_20_dB": _decay_time_ms(env, t, -20.0),
            "minus_40_dB": decay_40,
            "minus_60_dB": _decay_time_ms(env, t, -60.0),
        },
        "cumulative_energy_final": round(float(cum_energy[-1]), 6),
        "late_energy_ratio_0p5_1_vs_0_0p2": round(e_mid / max(e_early, 1e-12), 6),
        "late_energy_ratio_1_2_vs_0_0p2": round(e_late / max(e_early, 1e-12), 6),
        "log_envelope_smoothness_std": round(log_smoothness, 6),
        "envelope_bump_score": round(bump_score, 4),
        "end_rise_score": round(end_rise_score, 4),
        "hard_gate_score": round(hard_gate_score, 6),
        "expected_tau_s_from_Q": round(expected_tau, 6),
        "modal_beating_allowed_if_bounded": bool(beating_allowed),
        "no_late_energy_rise": no_late_rise,
        "no_hard_cut": no_hard_cut,
        "no_unexplained_late_bump": no_unexplained_bump,
        "decay_consistent_with_calibrated_Q": decay_consistent,
        "pass": bool(no_late_rise and no_hard_cut and no_unexplained_bump and decay_consistent),
    }


def _spectral_features(y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(y)
    window = np.hanning(n)
    spec = np.abs(np.fft.rfft(y * window))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    power = spec ** 2
    return freqs, spec, power


def analyze_extended_spectral(
    y: np.ndarray,
    sr: int,
    modal_freqs: Sequence[float],
) -> Dict[str, Any]:
    freqs, spec, power = _spectral_features(y, sr)
    spec_db = 20.0 * np.log10(np.maximum(spec, 1e-12))
    total_power = max(float(np.sum(power)), 1e-12)

    centroid = float(np.sum(freqs * power) / total_power)
    cum = np.cumsum(power) / total_power
    rolloff_85 = float(freqs[int(np.searchsorted(cum, 0.85))]) if cum.size else 0.0
    rolloff_95 = float(freqs[int(np.searchsorted(cum, 0.95))]) if cum.size else 0.0

    geo_mean = float(np.exp(np.mean(np.log(np.maximum(power, 1e-20)))))
    arith_mean = float(np.mean(power))
    flatness = geo_mean / max(arith_mean, 1e-20)

    hf_mask = freqs > 4000.0
    hf_ratio = float(np.sum(power[hf_mask]) / total_power) if hf_mask.any() else 0.0

    band_energy: Dict[str, float] = {}
    for name, lo, hi in MODAL_BANDS:
        mask = (freqs >= lo) & (freqs < hi)
        band_energy[name] = round(float(np.sum(power[mask]) / total_power), 6) if mask.any() else 0.0

    peak_indices: List[int] = []
    for i in range(2, len(spec_db) - 2):
        if spec_db[i] > spec_db[i - 1] and spec_db[i] > spec_db[i + 1]:
            if spec_db[i] > spec_db.max() - 40.0:
                peak_indices.append(i)

    aligned: List[Dict[str, float]] = []
    tol = 25.0
    modal_list = sorted({float(f) for f in modal_freqs})[:25]
    for f_m in modal_list:
        if not peak_indices:
            break
        best_i = min(peak_indices, key=lambda i: abs(freqs[i] - f_m))
        if abs(freqs[best_i] - f_m) <= tol:
            aligned.append({"modal_hz": f_m, "peak_hz": float(freqs[best_i]), "deviation_hz": round(float(freqs[best_i] - f_m), 2)})

    unexplained = 0
    for i in peak_indices[:20]:
        f_pk = float(freqs[i])
        near_modal = any(abs(f_pk - f_m) <= tol for f_m in modal_list)
        near_a4 = any(abs(f_pk - F0_HZ * h) <= 15.0 for h in range(1, 8))
        if not near_modal and not near_a4:
            unexplained += 1

    comb_score = 0.0
    if len(peak_indices) >= 5:
        spacings = np.diff([freqs[i] for i in peak_indices[:10]])
        if spacings.size:
            comb_score = float(np.std(spacings) / max(np.mean(spacings), 1.0))

    hf_spike = bool(hf_mask.any() and spec_db[hf_mask].max() > spec_db.max() - 6.0)
    click_dominated = flatness > 0.35 and hf_ratio > 0.45

    return {
        "spectral_centroid_hz": round(centroid, 2),
        "spectral_rolloff_85_hz": round(rolloff_85, 2),
        "spectral_rolloff_95_hz": round(rolloff_95, 2),
        "spectral_flatness": round(flatness, 6),
        "high_frequency_energy_ratio": round(hf_ratio, 6),
        "modal_band_energy_distribution": band_energy,
        "aligned_modal_peaks": aligned[:15],
        "aligned_modal_peak_count": len(aligned),
        "unexplained_peak_count": unexplained,
        "echo_comb_signature_score": round(comb_score, 4),
        "no_artificial_hf_spike": bool(not hf_spike),
        "no_echo_comb_pattern": bool(comb_score < 0.95),
        "modal_peak_alignment_strong": bool(len(aligned) >= 5),
        "not_excitation_click_dominated": bool(not click_dominated),
        "pass": bool(not hf_spike and comb_score < 0.95 and len(aligned) >= 5 and not click_dominated),
    }


def _stft_mag(y: np.ndarray, sr: int, n_fft: int = 2048, hop: int = 512) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_frames = 1 + max(0, (len(y) - n_fft) // hop)
    if n_frames <= 0:
        z = np.zeros((1, n_fft // 2 + 1))
        return np.array([0.0]), np.fft.rfftfreq(n_fft, 1.0 / sr), z
    window = np.hanning(n_fft)
    rows = []
    times = []
    for i in range(n_frames):
        start = i * hop
        frame = y[start : start + n_fft]
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))
        rows.append(np.abs(np.fft.rfft(frame * window)))
        times.append(start / sr)
    return np.array(times), np.fft.rfftfreq(n_fft, 1.0 / sr), np.array(rows)


def analyze_time_frequency(y: np.ndarray, sr: int) -> Dict[str, Any]:
    times, freqs, mag = _stft_mag(y, sr)
    if mag.size == 0:
        return {"status": "empty", "pass": False}

    def _window_mask(t0: float, t1: float) -> np.ndarray:
        return (times >= t0) & (times < t1)

    early = _window_mask(0.0, 0.3)
    mid = _window_mask(0.5, 1.0)
    late = _window_mask(1.5, 2.5)

    def _band_mean(mask: np.ndarray) -> Dict[str, float]:
        if not mask.any():
            return {}
        m = np.mean(mag[mask], axis=0)
        p = m ** 2
        s = max(float(np.sum(p)), 1e-12)
        out = {"centroid_hz": round(float(np.sum(freqs * p) / s), 2)}
        hf = freqs > 4000.0
        out["hf_ratio"] = round(float(np.sum(p[hf]) / s), 6) if hf.any() else 0.0
        return out

    early_s = _band_mean(early)
    mid_s = _band_mean(mid)
    late_s = _band_mean(late)

    hf_early = early_s.get("hf_ratio", 0.0)
    hf_late = late_s.get("hf_ratio", 0.0)
    hf_decays = bool(hf_late <= hf_early * 1.15 + 0.02)

    centroids = []
    for row in mag:
        p = row ** 2
        s = max(float(np.sum(p)), 1e-12)
        centroids.append(float(np.sum(freqs * p) / s))
    centroid_over_time = [round(c, 2) for c in centroids[:: max(1, len(centroids) // 40)]]

    band_decay: Dict[str, float] = {}
    for name, lo, hi in MODAL_BANDS:
        mask_f = (freqs >= lo) & (freqs < hi)
        if not mask_f.any():
            continue
        e_early = float(np.sum(mag[early][:, mask_f] ** 2)) if early.any() else 0.0
        e_late = float(np.sum(mag[late][:, mask_f] ** 2)) if late.any() else 0.0
        band_decay[name] = round(e_late / max(e_early, 1e-12), 6)

    late_noise_dom = late_s.get("hf_ratio", 0.0) > 0.5 and late_s.get("centroid_hz", 0) > 6000
    delayed_body = False
    if early.any() and mid.any():
        e_early_t = float(np.sum(mag[early] ** 2))
        e_mid_t = float(np.sum(mag[mid] ** 2))
        delayed_body = e_mid_t > e_early_t * 2.5

    return {
        "early_window": early_s,
        "mid_window": mid_s,
        "late_window": late_s,
        "spectral_centroid_over_time": centroid_over_time,
        "modal_band_energy_decay_late_vs_early": band_decay,
        "high_frequency_decays_over_time": hf_decays,
        "late_spectrum_not_noise_dominated": bool(not late_noise_dom),
        "no_delayed_independent_body_event": bool(not delayed_body),
        "pass": bool(hf_decays and not late_noise_dom and not delayed_body),
    }


def analyze_stem_coherence(
    main: np.ndarray,
    body: np.ndarray,
    excitation: np.ndarray,
    sr: int,
) -> Dict[str, Any]:
    n = min(len(main), len(body), len(excitation))
    main, body, excitation = main[:n], body[:n], excitation[:n]
    expected = np.convolve(excitation, body, mode="full")[:n]
    peak_main = max(float(np.max(np.abs(main))), 1e-12)
    peak_exp = max(float(np.max(np.abs(expected))), 1e-12)
    expected = expected * (peak_main / peak_exp)

    env_main = _envelope(main, sr)
    env_exp = _envelope(expected, sr)
    if np.std(env_main) > 1e-12 and np.std(env_exp) > 1e-12:
        stem_corr = float(np.corrcoef(env_main, env_exp)[0, 1])
    else:
        stem_corr = 0.0

    e_body = float(np.sum(body ** 2))
    e_exc = float(np.sum(excitation ** 2))
    ratio = e_body / max(e_exc, 1e-12)

    exc_peak = float(np.max(np.abs(excitation)))
    main_peak = float(np.max(np.abs(main)))
    exc_not_click = bool(exc_peak < main_peak * 1.2 or e_exc < e_body * 0.5)

    onset_exc = int(np.argmax(_envelope(excitation, sr)))
    onset_body = int(np.argmax(_envelope(body, sr)))
    aligned_t0 = bool(abs(onset_exc - onset_body) <= int(0.01 * sr))

    delayed_tail = False
    env_b = _envelope(body, sr)
    if len(env_b) > sr:
        delayed_tail = bool(float(env_b[sr // 2 : sr].max()) > float(env_b[: sr // 10].max()) * 0.8)

    stem_pass = bool(stem_corr >= 0.75 and exc_not_click and aligned_t0 and not delayed_tail)
    return {
        "main_vs_stem_model_correlation": round(stem_corr, 4),
        "body_excitation_energy_ratio": round(ratio, 4),
        "excitation_not_dominant_click": exc_not_click,
        "excitation_body_aligned_at_t0": aligned_t0,
        "no_hidden_delayed_tail_layer": bool(not delayed_tail),
        "body_stem_follows_modal_ir": bool(stem_corr >= 0.75),
        "pass": stem_pass,
    }


def build_contract_compliance(
    step3d: Mapping[str, Any],
    step4a: Mapping[str, Any],
    step4b: Mapping[str, Any],
) -> Dict[str, Any]:
    rg3d = step3d.get("readiness_after_step3d") or {}
    rg4a = step4a.get("readiness_after_step4a") or {}
    rg4b = step4b.get("readiness_after_step4b") or {}
    artifact = step4b.get("artifact_guard_results") or {}
    checks = {
        "no_body_tail": artifact.get("body_tail_stem_added") is False,
        "no_helmholtz_echo_ir": artifact.get("helmholtz_echo_added") is False,
        "no_reverb": artifact.get("reverb_added") is False,
        "no_EQ_body_layer": artifact.get("post_hoc_EQ_added") is False,
        "no_arbitrary_wood_gain": artifact.get("arbitrary_wood_gain") is False,
        "no_final_synthesis_claim": rg4b.get("final_synthesis_ready") is False,
        "no_multi_guitar_proof": rg4b.get("multi_guitar_comparison_allowed") is False,
        "no_stk_integration": rg4b.get("stk_integration_allowed") is False,
        "no_website_production_change": rg4b.get("website_production_replacement_allowed") is False,
        "no_listening_based_acceptance": step4b.get("listening_based_tuning_used") is False,
    }
    return {**checks, "step3d_contract_active": True, "pass": bool(all(checks.values()))}


def build_artifact_guard_extended(
    envelope: Mapping[str, Any],
    spectral: Mapping[str, Any],
    time_freq: Mapping[str, Any],
    stem: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> Dict[str, Any]:
    checks = {
        "no_end_rise": envelope.get("no_late_energy_rise"),
        "no_hard_gate": envelope.get("no_hard_cut"),
        "no_hf_spike": spectral.get("no_artificial_hf_spike"),
        "no_comb_echo": spectral.get("no_echo_comb_pattern"),
        "hf_decays": time_freq.get("high_frequency_decays_over_time"),
        "stem_coherent": stem.get("pass"),
        "contract_ok": contract.get("pass"),
    }
    return {**checks, "pass": bool(all(bool(v) for v in checks.values()))}


def run_objective_tests(
    upstream: Mapping[str, Any],
    envelope: Mapping[str, Any],
    spectral: Mapping[str, Any],
    time_freq: Mapping[str, Any],
    stem: Mapping[str, Any],
    artifact: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> Dict[str, Any]:
    tests = {
        "upstream_ready": upstream.get("pass"),
        "extended_envelope": envelope.get("pass"),
        "extended_spectral": spectral.get("pass"),
        "time_frequency": time_freq.get("pass"),
        "stem_coherence": stem.get("pass"),
        "artifact_guard": artifact.get("pass"),
        "contract_compliance": contract.get("pass"),
        "no_listening_tuning": contract.get("no_listening_based_acceptance"),
    }
    tests["all_pass"] = bool(all(tests.values()))
    return tests


def build_readiness_after_step4c(objective: Mapping[str, Any], artifact: Mapping[str, Any]) -> Dict[str, Any]:
    if artifact.get("pass") is False:
        status = "blocked_due_to_extended_validation_artifact"
    elif not objective.get("all_pass"):
        status = "failed_extended_validation"
    else:
        status = READINESS_STEP5A

    return {
        "current_status": status,
        "final_synthesis_ready": False,
        "stk_integration_allowed": False,
        "website_production_replacement_allowed": False,
        "multi_guitar_comparison_allowed": False,
        "step5a_limited_note_set_allowed": status == READINESS_STEP5A,
    }


def write_optional_figures(
    main: np.ndarray,
    body: np.ndarray,
    sr: int,
    modal_freqs: Sequence[float],
    envelope: Mapping[str, Any],
    out_dir: Path,
) -> List[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    t = np.arange(len(main)) / sr
    env = _envelope(main, sr)

    fig, ax = plt.subplots(figsize=(8, 3))
    mask = env > 1e-8
    ax.semilogy(t[mask], env[mask])
    ax.set_title("Log-scale envelope decay")
    p = out_dir / "envelope_log_decay.png"
    fig.savefig(p, dpi=100, bbox_inches="tight")
    plt.close(fig)
    written.append(str(p))

    times, freqs, mag = _stft_mag(main, sr)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(20 * np.log10(np.maximum(mag.T, 1e-12)), aspect="auto", origin="lower", extent=[times[0], times[-1], freqs[0], freqs[-1]], cmap="magma")
    ax.set_title("Spectrogram")
    p2 = out_dir / "spectrogram.png"
    fig.savefig(p2, dpi=100, bbox_inches="tight")
    plt.close(fig)
    written.append(str(p2))

    freqs_s, spec, _ = _spectral_features(main, sr)
    mask_f = freqs_s <= 1200.0
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(freqs_s[mask_f], 20 * np.log10(np.maximum(spec[mask_f], 1e-12)))
    for f in modal_freqs[:12]:
        ax.axvline(float(f), color="r", alpha=0.2)
    ax.set_title("Modal peak overlay")
    p3 = out_dir / "modal_peak_overlay.png"
    fig.savefig(p3, dpi=100, bbox_inches="tight")
    plt.close(fig)
    written.append(str(p3))

    cum = np.cumsum(main ** 2) / max(np.sum(main ** 2), 1e-12)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(t, cum)
    ax.set_title("Cumulative energy decay")
    p4 = out_dir / "cumulative_energy_decay.png"
    fig.savefig(p4, dpi=100, bbox_inches="tight")
    plt.close(fig)
    written.append(str(p4))

    return written


def build_pgsm_step4c_report(
    *,
    repo_root: Optional[Path] = None,
    write_figures: bool = False,
    max_modes: Optional[int] = None,
) -> Dict[str, Any]:
    root = Path(repo_root or REPO_ROOT)
    step4b = load_step_report(_report_path(root, "pgsm_step4b_single_note_diagnostic_refinement.json"))
    step4a = load_step_report(_report_path(root, "pgsm_step4a_single_note_diagnostic_audio.json"))
    step3c = load_step_report(_report_path(root, "pgsm_step3c_numeric_calibration.json"))
    step3d = load_step_report(_report_path(root, "pgsm_step3d_pre_synthesis_contract.json"))

    wav_paths = _wav_paths(root, step4a)
    upstream = verify_upstream_readiness(step4b, step4a, wav_paths)

    main, sr = load_wav_mono(wav_paths["main"])
    body, _ = load_wav_mono(wav_paths["body"])
    excitation, _ = load_wav_mono(wav_paths["excitation"])

    state = build_calibrated_modal_state(root, max_modes=max_modes)
    modal_freqs = [float(m["frequency_hz"]) for m in state["modal_weights"].get("modes") or []]
    q_mean = float((state["q_summary"].get("after") or {}).get("mean_Q") or 34.0)

    envelope = analyze_extended_envelope(main, sr, q_mean)
    spectral = analyze_extended_spectral(main, sr, modal_freqs)
    time_freq = analyze_time_frequency(main, sr)
    stem = analyze_stem_coherence(main, body, excitation, sr)
    contract = build_contract_compliance(step3d, step4a, step4b)
    artifact = build_artifact_guard_extended(envelope, spectral, time_freq, stem, contract)
    objective = run_objective_tests(upstream, envelope, spectral, time_freq, stem, artifact, contract)
    readiness = build_readiness_after_step4c(objective, artifact)

    figures: List[str] = []
    if write_figures:
        figures = write_optional_figures(main, body, sr, modal_freqs, envelope, FIGURES_DIR)

    return {
        "report_version": PGSM_STEP4C_VERSION,
        "timestamp": _utc_now(),
        "status": "pgsm_step4c_single_note_extended_validation_complete",
        "no_new_wav_generated": True,
        "no_stk_integration": True,
        "no_fem_run": True,
        "no_rom_run": True,
        "website_default": DEFAULT_WEBSITE_STK_MODE,
        "website_default_unchanged": True,
        "sample_id": SAMPLE_ID,
        "note": NOTE,
        "diagnostic_label": step4a.get("diagnostic_label"),
        "upstream_readiness": upstream,
        "step4b_loaded": step4b.get("report_version"),
        "step4a_loaded": step4a.get("report_version"),
        "step3c_loaded": step3c.get("report_version"),
        "step3d_loaded": step3d.get("report_version"),
        "analyzed_files": {k: str(v) for k, v in wav_paths.items()},
        "extended_envelope_metrics": envelope,
        "extended_spectral_metrics": spectral,
        "time_frequency_metrics": time_freq,
        "stem_coherence_metrics": stem,
        "contract_compliance": contract,
        "artifact_guard_results": artifact,
        "objective_test_results": objective,
        "figures_written": figures,
        "readiness_after_step4c": readiness,
        "blocked_next_steps": [
            "Final synthesis",
            "STK integration",
            "Website production replacement",
            "Multi-guitar comparison",
            "Melody/chord playback",
            "Subjective tuning by ear",
        ],
        "safe_next_step": (
            "PGSM Step 5A: limited note-set diagnostic audio (controlled, not final synthesis)"
            if readiness["current_status"] == READINESS_STEP5A
            else "Resolve extended validation failures before Step 5A"
        ),
        "explicit_statement": (
            "PGSM Step 4C performs extended validation of existing diagnostic audio only. "
            "It does not generate new audio."
        ),
    }


def write_markdown_report(report: Mapping[str, Any], path: Path) -> None:
    e = report.get("extended_envelope_metrics") or {}
    s = report.get("extended_spectral_metrics") or {}
    tf = report.get("time_frequency_metrics") or {}
    st = report.get("stem_coherence_metrics") or {}
    c = report.get("contract_compliance") or {}
    a = report.get("artifact_guard_results") or {}
    rg = report.get("readiness_after_step4c") or {}

    lines = [
        "# PGSM Step 4C — extended diagnostic validation",
        "",
        f"**Generated:** {report.get('timestamp')}",
        f"**Readiness:** `{rg.get('current_status')}`",
        "",
        report.get("explicit_statement", ""),
        "",
        "## Analyzed files",
        "",
    ]
    for k, v in (report.get("analyzed_files") or {}).items():
        lines.append(f"- {k}: `{v}`")

    lines.extend(
        [
            "",
            "## Extended envelope",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Peak time ms | {e.get('peak_time_ms')} |",
            f"| −40 dB ms | {e.get('decay_ms', {}).get('minus_40_dB')} |",
            f"| End rise score | {e.get('end_rise_score')} |",
            f"| Bump score | {e.get('envelope_bump_score')} |",
            f"| Pass | {e.get('pass')} |",
            "",
            "## Extended spectral",
            "",
            f"- Centroid Hz: {s.get('spectral_centroid_hz')}",
            f"- Aligned modal peaks: {s.get('aligned_modal_peak_count')}",
            f"- HF spike: {not s.get('no_artificial_hf_spike', True)}",
            f"- Pass: {s.get('pass')}",
            "",
            "## Time-frequency",
            "",
            f"- HF decays: {tf.get('high_frequency_decays_over_time')}",
            f"- Pass: {tf.get('pass')}",
            "",
            "## Stem coherence",
            "",
            f"- Main/stem correlation: {st.get('main_vs_stem_model_correlation')}",
            f"- Pass: {st.get('pass')}",
            "",
            "## Contract compliance",
            "",
            f"- Pass: {c.get('pass')}",
            "",
            "## Artifact guard",
            "",
            f"- Pass: {a.get('pass')}",
            "",
            f"all_pass: **{(report.get('objective_test_results') or {}).get('all_pass')}**",
            "",
            report.get("safe_next_step", ""),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pgsm_step4c_reports(
    *,
    repo_root: Optional[Path] = None,
    json_path: Optional[Path] = None,
    md_path: Optional[Path] = None,
    write_figures: bool = False,
    max_modes: Optional[int] = None,
) -> Dict[str, Any]:
    root = Path(repo_root or REPO_ROOT)
    report = build_pgsm_step4c_report(repo_root=root, write_figures=write_figures, max_modes=max_modes)
    jpath = Path(json_path or REPORT_JSON)
    mpath = Path(md_path or REPORT_MD)
    jpath.parent.mkdir(parents=True, exist_ok=True)
    jpath.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown_report(report, mpath)
    return report


def main() -> None:
    report = write_pgsm_step4c_reports(write_figures=True)
    rg = report.get("readiness_after_step4c") or {}
    print(f"Wrote {REPORT_JSON}")
    print(f"Readiness: {rg.get('current_status')}")
    print(f"all_pass: {(report.get('objective_test_results') or {}).get('all_pass')}")


if __name__ == "__main__":
    main()
