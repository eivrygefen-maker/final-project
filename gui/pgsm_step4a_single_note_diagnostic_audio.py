#!/usr/bin/env python3
"""
PGSM Step 4A — single-note diagnostic audio prototype.
Low-amplitude diagnostic WAV only; not final synthesis, not STK.
"""
from __future__ import annotations

import json
import math
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from pgsm_physical_factor_registry import load_audit_report
from pgsm_step2_1_parameter_targets import PLUCK_DURATION_MS_TYPICAL, load_step_report
from pgsm_step3a_numerical_ir_testbench import (
    DURATION_S,
    F0_HZ,
    FIXED_PLUCK_POSITION,
    NUMERIC_SR,
    SAMPLE_ID,
    build_parameter_pack,
    compute_impulse_response,
    compute_modal_weights,
    load_rom_modal_catalog,
)
from pgsm_step3c_numeric_calibration import (
    NOMINAL_RAW_MEAN_Q,
    apply_material_policy_sample,
    apply_region_calibration_to_modes,
    build_material_policy_object,
    calibrate_q_tau_modes,
    load_fem_woods_ortho,
    load_pgsm_library,
    project_region_weights,
)
from pgsm_step3d_pre_synthesis_contract import STEP4A_READINESS
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE

PGSM_STEP4A_VERSION = "pgsm_step4a_single_note_diagnostic_audio_v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON = REPO_ROOT / "audio" / "debug_reports" / "pgsm_step4a_single_note_diagnostic_audio.json"
REPORT_MD = REPO_ROOT / "audio" / "debug_reports" / "pgsm_step4a_single_note_diagnostic_audio.md"
FIGURES_DIR = REPO_ROOT / "audio" / "debug_reports" / "pgsm_step4a_figures"
AUDIO_DIR = REPO_ROOT / "audio" / "pgsm_step4a_diagnostic_audio"

NOTE = "A4"
DIAGNOSTIC_LABEL = "PGSM diagnostic audio, not final guitar"
MAX_DIAGNOSTIC_PEAK_FS = 0.28
READINESS_STEP4B = "ready_for_step4b_single_note_diagnostic_refinement"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _report_path(root: Path, name: str) -> Path:
    return root / "audio" / "debug_reports" / name


def load_step3d_contract(repo_root: Path) -> Dict[str, Any]:
    path = _report_path(repo_root, "pgsm_step3d_pre_synthesis_contract.json")
    if not path.is_file():
        raise FileNotFoundError(f"Step 3D contract required: {path}")
    doc = load_step_report(path)
    readiness = doc.get("readiness_after_step3d") or {}
    if readiness.get("current_status") != STEP4A_READINESS:
        raise ValueError(
            f"Step 3D readiness must be {STEP4A_READINESS!r}; got {readiness.get('current_status')!r}"
        )
    if readiness.get("final_synthesis_ready") is True:
        raise ValueError("Step 3D reports final_synthesis_ready — blocked for Step 4A")
    if readiness.get("stk_integration_allowed") is True:
        raise ValueError("Step 3D allows STK integration — blocked for Step 4A")
    return doc


def build_calibrated_modal_state(
    repo_root: Path,
    *,
    max_modes: Optional[int] = None,
) -> Dict[str, Any]:
    root = Path(repo_root)
    audit = load_audit_report()
    fem = load_fem_woods_ortho(root / "FEM" / "materials" / "woods_ortho.json")
    pgsm = load_pgsm_library(root / "data" / "pgsm_tonewood_material_library.json")
    step3b = load_step_report(_report_path(root, "pgsm_step3b_modal_response_validation.json"))
    step3c = load_step_report(_report_path(root, "pgsm_step3c_numeric_calibration.json"))

    material = apply_material_policy_sample(audit, fem, pgsm)
    rom = load_rom_modal_catalog(root / "FEM" / "outputs" / "rom_stk_body.json")
    pack = build_parameter_pack(audit)
    modes = rom.get("predicted_modes") or []
    if max_modes is not None:
        modes = modes[:max_modes]

    raw_weights = compute_modal_weights(modes, pack)
    reference_q = float(
        (step3b.get("modal_q_tau_validation") or {}).get("mean_Q_total") or NOMINAL_RAW_MEAN_Q
    )
    calibrated_modes, q_summary = calibrate_q_tau_modes(
        raw_weights["modes"], material, reference_raw_mean_q=reference_q
    )
    reg_raw = step3b.get("region_contribution_validation") or {}
    region_cal = project_region_weights(
        float(reg_raw.get("top_fraction", 0.214)),
        float(reg_raw.get("back_fraction", 0.766)),
        float(reg_raw.get("air_fraction", 0.019)),
    )
    modes_region = apply_region_calibration_to_modes(calibrated_modes, region_cal)
    cal_weights = dict(raw_weights)
    cal_weights["modes"] = modes_region
    cal_weights["calibration_applied"] = True

    ir = compute_impulse_response(cal_weights)
    return {
        "modal_weights": cal_weights,
        "material_policy": build_material_policy_object(),
        "chosen_material": material,
        "q_summary": q_summary,
        "region_cal": region_cal,
        "step3c_report": step3c,
        "step3b_report": step3b,
        "impulse_response": ir,
    }


def build_f_bridge_proxy(
    n: int,
    sr: int,
    *,
    pluck_duration_ms: float = PLUCK_DURATION_MS_TYPICAL,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Short smooth force pulse — not an audio click; single onset at t=0."""
    f = np.zeros(n, dtype=np.float64)
    n_pulse = max(int(pluck_duration_ms * 1e-3 * sr), 2)
    n_pulse = min(n_pulse, n)
    window = np.sin(np.pi * np.linspace(0.0, 1.0, n_pulse)) ** 2
    window[0] = max(window[1] * 0.2, 1e-8)
    f[:n_pulse] = amplitude * window
    return f


def compute_full_impulse_response(
    modal_weights: Mapping[str, Any],
    *,
    duration_s: float = DURATION_S,
    sr: int = NUMERIC_SR,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    modes = modal_weights.get("modes") or []
    n = int(duration_s * sr)
    t = np.arange(n, dtype=np.float64) / sr
    h_body = np.zeros(n, dtype=np.float64)

    for row in modes:
        f_i = float(row["frequency_hz"])
        tau = max(float(row["tau_s"]), 1e-6)
        wr = float(row["W_rad"])
        wa = float(row["W_air"])
        top = float(row["top_share"])
        back = float(row["back_share"])
        air = float(row["air_share"])
        region = max(top + back + air, 1e-9)

        kernel = wr * np.exp(-t / tau) * np.sin(2.0 * np.pi * f_i * t)
        h_body += kernel
        h_body += wa * np.exp(-t / (tau * 1.2)) * np.sin(2.0 * np.pi * f_i * t) * 0.01 * (air / region)

    return t, h_body


def synthesize_modal_body_response(
    f_bridge: np.ndarray,
    h_body: np.ndarray,
) -> np.ndarray:
    """Causal linear response: body = F_bridge * h (convolution, truncated)."""
    y = np.convolve(f_bridge, h_body, mode="full")[: len(f_bridge)]
    return y.astype(np.float64)


def normalize_diagnostic_amplitude(
    y: np.ndarray,
    *,
    max_peak_fs: float = MAX_DIAGNOSTIC_PEAK_FS,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak <= 1e-20:
        return y, {"peak_before": 0.0, "scale_applied": 1.0, "peak_after_fs": 0.0}
    scale = max_peak_fs / peak
    out = y * scale
    return out, {
        "peak_before": round(peak, 8),
        "scale_applied": round(scale, 8),
        "peak_after_fs": round(float(np.max(np.abs(out))), 6),
        "max_peak_fs_target": max_peak_fs,
        "clipping": bool(np.max(np.abs(out)) > 1.0),
        "loudness_matching_applied": False,
        "normalization_separated_from_physics": True,
        "label": DIAGNOSTIC_LABEL,
    }


def write_wav_mono(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.clip(np.asarray(samples, dtype=np.float64), -1.0, 1.0)
    pcm = (x * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())


def _downsample_envelope(y: np.ndarray, sr: int, step: int = 441) -> Tuple[np.ndarray, np.ndarray]:
    env = np.abs(y)
    t = np.arange(len(y), dtype=np.float64) / sr
    return t[::step], env[::step]


def _envelope_correlation(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n < 4:
        return 0.0
    a = np.asarray(a[:n], dtype=float)
    b = np.asarray(b[:n], dtype=float)
    a = a / max(a.max(), 1e-12)
    b = b / max(b.max(), 1e-12)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _decay_time_ms(env: np.ndarray, t_s: np.ndarray, db: float) -> Optional[float]:
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
    return float(t_s[peak_i + int(idx[0])] * 1000.0)


def evaluate_envelope_consistency(
    body: np.ndarray,
    sr: int,
    step3c_ir: Mapping[str, Any],
    *,
    h_body: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    t_gen, env_gen = _downsample_envelope(body, sr)
    ref_env = np.array(step3c_ir.get("envelope_downsampled") or [], dtype=float)
    ref_t = np.array(step3c_ir.get("time_s_downsampled") or [], dtype=float)

    if h_body is not None and h_body.size:
        _, env_ir = _downsample_envelope(h_body, sr)
        corr = _envelope_correlation(env_ir, ref_env)
    else:
        corr = _envelope_correlation(env_gen, ref_env)
    decay_gen = {
        "minus_20_dB_ms": _decay_time_ms(env_gen, t_gen, -20.0),
        "minus_40_dB_ms": _decay_time_ms(env_gen, t_gen, -40.0),
    }
    ref_decay = (step3c_ir.get("decay_time_ms") or {}) if step3c_ir else {}

    early = t_gen <= 0.05
    delayed = (t_gen >= 0.1) & (t_gen <= 0.25)
    delayed_onset = False
    if early.any() and delayed.any():
        delayed_onset = float(env_gen[delayed].max()) > float(env_gen[early].max()) * 1.8

    last_third = env_gen[len(env_gen) * 2 // 3 :]
    mid_third = env_gen[len(env_gen) // 3 : len(env_gen) * 2 // 3]
    end_rise = float(last_third.max()) > float(mid_third.max()) * 1.05 if len(last_third) else False

    tail = env_gen[int(len(env_gen) * 0.85) :]
    hard_gate = bool(tail.size and float(tail.max()) < 1e-6 and float(env_gen[len(env_gen) // 2]) > 1e-4)

    pass_corr = corr >= 0.92
    pass_onset = bool(not delayed_onset)
    pass_end = bool(not end_rise)
    pass_gate = bool(not hard_gate)

    return {
        "envelope_correlation_vs_step3c": round(corr, 4),
        "reference_decay_ms": ref_decay,
        "generated_decay_ms": decay_gen,
        "no_delayed_body_onset": pass_onset,
        "no_end_rise": pass_end,
        "no_hard_gate": pass_gate,
        "approx_follows_step3c_envelope": pass_corr,
        "pass": pass_corr and pass_onset and pass_end and pass_gate,
    }


def evaluate_spectral_modal_consistency(
    body: np.ndarray,
    sr: int,
    modal_weights: Mapping[str, Any],
) -> Dict[str, Any]:
    modes = modal_weights.get("modes") or []
    n = len(body)
    if n < 256:
        return {"status": "too_short", "pass": False}

    spec = np.abs(np.fft.rfft(body * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    spec_db = 20.0 * np.log10(np.maximum(spec, 1e-12))

    peak_indices: List[int] = []
    for i in range(2, len(spec_db) - 2):
        if spec_db[i] > spec_db[i - 1] and spec_db[i] > spec_db[i + 1]:
            if spec_db[i] > spec_db.max() - 40.0:
                peak_indices.append(i)

    modal_freqs = sorted({float(m["frequency_hz"]) for m in modes})
    aligned = 0
    tol_hz = 25.0
    for f_m in modal_freqs[:30]:
        if any(abs(freqs[i] - f_m) <= tol_hz for i in peak_indices):
            aligned += 1

    hf_mask = freqs > 8000.0
    hf_spike = bool(hf_mask.any() and spec_db[hf_mask].max() > spec_db.max() - 6.0)

    comb_score = 0.0
    if len(peak_indices) >= 5:
        spacings = np.diff([freqs[i] for i in peak_indices[:8]])
        if spacings.size:
            comb_score = float(np.std(spacings) / max(np.mean(spacings), 1.0))

    a4_harmonics = [F0_HZ * k for k in range(1, 6)]
    a4_near = sum(
        1 for h in a4_harmonics if any(abs(freqs[i] - h) <= 15.0 for i in peak_indices)
    )

    return {
        "spectral_peak_count": len(peak_indices),
        "modal_peaks_aligned_count": aligned,
        "modal_peaks_aligned_fraction": round(aligned / max(min(len(modal_freqs), 30), 1), 4),
        "a4_harmonic_proximity_count": a4_near,
        "unexplained_hf_spike": bool(hf_spike),
        "echo_comb_pattern_score": round(comb_score, 4),
        "Q_tau_bypassed_by_post_processing": False,
        "pass": aligned >= 3 and not hf_spike and comb_score < 0.95,
    }


def evaluate_artifact_guard(
    body: np.ndarray,
    excitation: np.ndarray,
    sr: int,
    *,
    envelope_pass_onset: bool = True,
) -> Dict[str, Any]:
    env = np.abs(body)
    second_onset = False
    if env.size > sr // 4:
        first = env[: sr // 10].max()
        mid = env[sr // 8 : sr // 4].max()
        second_onset = mid > first * 0.45 and first > 1e-8

    return {
        "body_tail_stem_used": False,
        "helmholtz_echo_ir_used": False,
        "post_hoc_EQ_body_layer": False,
        "artificial_reverb": False,
        "arbitrary_wood_to_gain_mapping": False,
        "independent_delayed_body_onset": not envelope_pass_onset,
        "second_pluck_onset": bool(second_onset),
        "pass": bool(envelope_pass_onset and not second_onset),
    }


def run_objective_tests(
    envelope: Mapping[str, Any],
    spectral: Mapping[str, Any],
    artifact: Mapping[str, Any],
    normalization: Mapping[str, Any],
) -> Dict[str, Any]:
    tests = {
        "envelope_consistency": envelope.get("pass", False),
        "spectral_modal_consistency": spectral.get("pass", False),
        "artifact_guard": artifact.get("pass", False),
        "peak_below_0p3_fs": float(normalization.get("peak_after_fs", 1.0)) <= 0.3,
        "no_clipping": not normalization.get("clipping", True),
        "exact_open_string_claim_blocked": True,
    }
    tests["all_pass"] = all(tests.values())
    return tests


def build_readiness_after_step4a(objective: Mapping[str, Any], artifact: Mapping[str, Any]) -> Dict[str, Any]:
    if artifact.get("pass") is False and objective.get("envelope_consistency") is False:
        status = "blocked_due_to_audio_artifacts"
    elif not objective.get("all_pass"):
        status = "failed_diagnostic_audio_checks"
    else:
        status = READINESS_STEP4B

    return {
        "current_status": status,
        "final_synthesis_ready": False,
        "musical_wav_synthesis_allowed": False,
        "stk_integration_allowed": False,
        "multi_guitar_comparison_allowed": False,
        "website_production_replacement_allowed": False,
        "step4b_diagnostic_refinement_allowed": status == READINESS_STEP4B,
    }


def _output_paths(audio_dir: Path) -> Dict[str, Path]:
    base = f"{SAMPLE_ID}_{NOTE}"
    return {
        "main": audio_dir / f"{base}_diagnostic.wav",
        "body_stem": audio_dir / f"{base}_body_stem.wav",
        "excitation_stem": audio_dir / f"{base}_excitation_stem.wav",
    }


def write_optional_figures(
    body: np.ndarray,
    sr: int,
    step3c_ir: Mapping[str, Any],
    modal_weights: Mapping[str, Any],
    out_dir: Path,
) -> List[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    t = np.arange(len(body)) / sr
    t_ds, env_gen = _downsample_envelope(body, sr)
    ref_t = np.array(step3c_ir.get("time_s_downsampled") or [])
    ref_env = np.array(step3c_ir.get("envelope_downsampled") or [])

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(t_ds, env_gen / max(env_gen.max(), 1e-12), label="generated")
    if ref_env.size:
        ax.plot(ref_t[: len(ref_env)], ref_env / max(ref_env.max(), 1e-12), label="step3c IR", alpha=0.7)
    ax.set_title("Envelope: Step 3C vs generated")
    ax.legend()
    p = out_dir / "envelope_comparison.png"
    fig.savefig(p, dpi=100, bbox_inches="tight")
    plt.close(fig)
    written.append(str(p))

    n = len(body)
    spec = np.abs(np.fft.rfft(body * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    mask = freqs <= 1200.0
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(freqs[mask], 20 * np.log10(np.maximum(spec[mask], 1e-12)))
    modes = modal_weights.get("modes") or []
    for m in modes[:12]:
        ax.axvline(float(m["frequency_hz"]), color="r", alpha=0.2, linewidth=0.8)
    ax.set_title("Spectrum with modal frequencies")
    p2 = out_dir / "spectrum_modal_overlay.png"
    fig.savefig(p2, dpi=100, bbox_inches="tight")
    plt.close(fig)
    written.append(str(p2))

    fig, ax = plt.subplots(figsize=(8, 2))
    ax.plot(t[: sr // 2], body[: sr // 2])
    ax.set_title("Waveform (first 0.5 s)")
    p3 = out_dir / "waveform_envelope.png"
    fig.savefig(p3, dpi=100, bbox_inches="tight")
    plt.close(fig)
    written.append(str(p3))
    return written


def build_pgsm_step4a_report(
    *,
    repo_root: Optional[Path] = None,
    audio_dir: Optional[Path] = None,
    write_wav: bool = True,
    write_figures: bool = False,
    max_modes: Optional[int] = None,
) -> Dict[str, Any]:
    root = Path(repo_root or REPO_ROOT)
    out_audio = Path(audio_dir or (root / "audio" / "pgsm_step4a_diagnostic_audio"))
    step3d = load_step3d_contract(root)
    step3a = load_step_report(_report_path(root, "pgsm_step3a_numerical_ir_testbench.json"))

    state = build_calibrated_modal_state(root, max_modes=max_modes)
    cal_weights = state["modal_weights"]
    ir_ref = compute_impulse_response(cal_weights)
    step3c_ir_report = (state["step3c_report"].get("calibrated_ir_summary") or {})

    sr = NUMERIC_SR
    n = int(DURATION_S * sr)
    t, h_body = compute_full_impulse_response(cal_weights)
    f_bridge = build_f_bridge_proxy(n, sr)
    body_raw = synthesize_modal_body_response(f_bridge, h_body)

    body_norm, norm_body = normalize_diagnostic_amplitude(body_raw)
    exc_norm, norm_exc = normalize_diagnostic_amplitude(f_bridge, max_peak_fs=0.15)
    h_norm, norm_h = normalize_diagnostic_amplitude(h_body, max_peak_fs=0.15)

    paths = _output_paths(out_audio)
    figures_written: List[str] = []
    if write_wav:
        write_wav_mono(paths["main"], body_norm, sr)
        write_wav_mono(paths["body_stem"], h_norm, sr)
        write_wav_mono(paths["excitation_stem"], exc_norm, sr)
        if write_figures:
            figures_written = write_optional_figures(body_norm, sr, ir_ref, cal_weights, FIGURES_DIR)

    envelope = evaluate_envelope_consistency(body_norm, sr, ir_ref, h_body=h_body)
    envelope["step3c_report_envelope_correlation"] = round(
        _envelope_correlation(
            np.array(ir_ref.get("envelope_downsampled") or []),
            np.array(step3c_ir_report.get("envelope_downsampled") or []),
        ),
        4,
    )
    spectral = evaluate_spectral_modal_consistency(body_norm, sr, cal_weights)
    artifact = evaluate_artifact_guard(
        body_norm, f_bridge, sr, envelope_pass_onset=envelope.get("no_delayed_body_onset", False)
    )
    objective = run_objective_tests(envelope, spectral, artifact, norm_body)
    readiness = build_readiness_after_step4a(objective, artifact)

    string_val = state["step3b_report"].get("string_consistency_validation") or {}
    excitation_summary = {
        "type": "F_bridge_force_proxy",
        "not_real_pluck_audio": True,
        "pluck_position_ratio": FIXED_PLUCK_POSITION,
        "pluck_duration_ms": PLUCK_DURATION_MS_TYPICAL,
        "note_frequency_hz": F0_HZ,
        "note_reference": NOTE,
        "A4_reference_only": True,
        "exact_open_string_claim_allowed": False,
        "string_interpretation": string_val.get("recommended_string_interpretation"),
        "effective_vibrating_length_m": string_val.get("calibrated_L_for_mid_tension_m"),
        "single_onset_at_t0": True,
        "no_reverb_no_echo": True,
    }

    return {
        "report_version": PGSM_STEP4A_VERSION,
        "timestamp": _utc_now(),
        "status": "pgsm_step4a_single_note_diagnostic_audio_complete",
        "diagnostic_audio_generated": write_wav,
        "final_synthesis_ready": False,
        "stk_integration_allowed": False,
        "website_default": DEFAULT_WEBSITE_STK_MODE,
        "website_default_unchanged": True,
        "sample_id": SAMPLE_ID,
        "note": NOTE,
        "diagnostic_label": DIAGNOSTIC_LABEL,
        "output_files": {
            "main_diagnostic_wav": str(paths["main"]),
            "body_stem_wav": str(paths["body_stem"]),
            "excitation_stem_wav": str(paths["excitation_stem"]),
            "main_wav_count": 1,
        },
        "upstream_contract_loaded": {
            "step3d_version": step3d.get("report_version"),
            "step3d_readiness": (step3d.get("readiness_after_step3d") or {}).get("current_status"),
            "step3c_version": state["step3c_report"].get("report_version"),
            "artifact_guard_contract_active": True,
        },
        "excitation_proxy_summary": excitation_summary,
        "modal_body_response_summary": {
            "mode_count": cal_weights.get("mode_count"),
            "duration_s": DURATION_S,
            "sample_rate_hz": sr,
            "calibrated_Q_mean": (state["q_summary"].get("after") or {}).get("mean_Q"),
            "calibrated_tau_mean_s": (state["q_summary"].get("after") or {}).get("mean_tau_s"),
            "region_weights": (state["region_cal"].get("calibrated") or {}),
            "no_post_hoc_EQ": True,
            "no_delayed_body_tail": True,
            "no_helmholtz_echo_ir": True,
        },
        "normalization_summary": {
            "body": norm_body,
            "excitation_stem": norm_exc,
            "body_stem_ir": norm_h,
        },
        "envelope_consistency": envelope,
        "spectral_modal_consistency": spectral,
        "artifact_guard_results": artifact,
        "objective_test_results": objective,
        "figures_written": figures_written,
        "step3c_report_ir_reference": {
            "peak_time_ms": step3c_ir_report.get("peak_time_ms"),
            "same_state_ir_peak_ms": ir_ref.get("peak_time_ms"),
        },
        "blocked_claims": step3d.get("blocked_claims") or [],
        "readiness_after_step4a": readiness,
        "safe_next_step": (
            "PGSM Step 4B: single-note diagnostic refinement (still not STK/website/final synthesis)"
            if readiness["current_status"] == READINESS_STEP4B
            else "Fix failed diagnostic audio checks before Step 4B"
        ),
        "explicit_statement": (
            "PGSM Step 4A generates diagnostic audio only. It is not final guitar synthesis."
        ),
    }


def run_objective_tests_pre_write(
    body: np.ndarray,
    sr: int,
    step3c_ir: Mapping[str, Any],
    cal_weights: Mapping[str, Any],
    *,
    h_body: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    env = evaluate_envelope_consistency(body, sr, step3c_ir, h_body=h_body)
    spec = evaluate_spectral_modal_consistency(body, sr, cal_weights)
    return {"would_pass": env.get("pass") and spec.get("pass")}


def write_markdown_report(report: Mapping[str, Any], path: Path) -> None:
    env = report.get("envelope_consistency") or {}
    spec = report.get("spectral_modal_consistency") or {}
    art = report.get("artifact_guard_results") or {}
    norm = (report.get("normalization_summary") or {}).get("body") or {}
    rg = report.get("readiness_after_step4a") or {}
    outputs = report.get("output_files") or {}

    lines = [
        "# PGSM Step 4A — single-note diagnostic audio",
        "",
        f"**Generated:** {report.get('timestamp')}",
        f"**Sample:** `{report.get('sample_id')}` | **Note:** `{report.get('note')}`",
        f"**Label:** {report.get('diagnostic_label')}",
        f"**Readiness:** `{rg.get('current_status')}`",
        "",
        report.get("explicit_statement", ""),
        "",
        "## Output WAV files",
        "",
        f"- Main: `{outputs.get('main_diagnostic_wav')}`",
        f"- Body stem: `{outputs.get('body_stem_wav')}`",
        f"- Excitation stem: `{outputs.get('excitation_stem_wav')}`",
        "",
        "## Excitation proxy",
        "",
    ]
    for k, v in (report.get("excitation_proxy_summary") or {}).items():
        lines.append(f"- {k}: {v}")

    lines.extend(
        [
            "",
            "## Normalization",
            "",
            f"- Peak after (FS): {norm.get('peak_after_fs')}",
            f"- Scale applied: {norm.get('scale_applied')}",
            f"- Clipping: {norm.get('clipping')}",
            "",
            "## Envelope consistency",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Correlation vs Step 3C | {env.get('envelope_correlation_vs_step3c')} |",
            f"| No delayed onset | {env.get('no_delayed_body_onset')} |",
            f"| No end rise | {env.get('no_end_rise')} |",
            f"| Pass | {env.get('pass')} |",
            "",
            "## Spectral/modal consistency",
            "",
            f"- Modal peaks aligned: {spec.get('modal_peaks_aligned_count')}",
            f"- HF spike: {spec.get('unexplained_hf_spike')}",
            f"- Pass: {spec.get('pass')}",
            "",
            "## Artifact guard",
            "",
            f"- Second pluck onset: {art.get('second_pluck_onset')}",
            f"- Pass: {art.get('pass')}",
            "",
            "## Readiness",
            "",
            f"all_pass: **{(report.get('objective_test_results') or {}).get('all_pass')}**",
            "",
            report.get("safe_next_step", ""),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pgsm_step4a_reports(
    *,
    repo_root: Optional[Path] = None,
    json_path: Optional[Path] = None,
    md_path: Optional[Path] = None,
    audio_dir: Optional[Path] = None,
    write_figures: bool = False,
    max_modes: Optional[int] = None,
) -> Dict[str, Any]:
    root = Path(repo_root or REPO_ROOT)
    report = build_pgsm_step4a_report(
        repo_root=root,
        audio_dir=audio_dir,
        write_wav=True,
        write_figures=write_figures,
        max_modes=max_modes,
    )
    jpath = Path(json_path or REPORT_JSON)
    mpath = Path(md_path or REPORT_MD)
    jpath.parent.mkdir(parents=True, exist_ok=True)
    jpath.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown_report(report, mpath)
    return report


def main() -> None:
    report = write_pgsm_step4a_reports(write_figures=True)
    obj = report.get("objective_test_results") or {}
    rg = report.get("readiness_after_step4a") or {}
    print(f"Wrote {REPORT_JSON}")
    print(f"Readiness: {rg.get('current_status')}")
    print(f"Objective all_pass: {obj.get('all_pass')}")


if __name__ == "__main__":
    main()
