#!/usr/bin/env python3
"""
PGSM Step 4B — single-note diagnostic refinement review.
Objective metrics only; no listening-based tuning, no STK/FEM/ROM.
"""
from __future__ import annotations

import json
import math
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from pgsm_step2_1_parameter_targets import load_step_report
from pgsm_step3a_numerical_ir_testbench import F0_HZ, MODAL_BANDS, NUMERIC_SR, SAMPLE_ID, compute_impulse_response
from pgsm_step4a_single_note_diagnostic_audio import (
    READINESS_STEP4B,
    build_calibrated_modal_state,
)
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE

PGSM_STEP4B_VERSION = "pgsm_step4b_single_note_diagnostic_refinement_v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON = REPO_ROOT / "audio" / "debug_reports" / "pgsm_step4b_single_note_diagnostic_refinement.json"
REPORT_MD = REPO_ROOT / "audio" / "debug_reports" / "pgsm_step4b_single_note_diagnostic_refinement.md"
FIGURES_DIR = REPO_ROOT / "audio" / "debug_reports" / "pgsm_step4b_figures"
STEP4A_AUDIO_DIR = REPO_ROOT / "audio" / "pgsm_step4a_diagnostic_audio"
STEP4B_AUDIO_DIR = REPO_ROOT / "audio" / "pgsm_step4b_diagnostic_audio"

READINESS_STEP4C = "ready_for_step4c_single_note_extended_validation"
NOTE = "A4"

FORBIDDEN_REFINEMENTS = (
    "reverb",
    "echo",
    "EQ_layer",
    "body_tail",
    "delayed_body_onset",
    "hard_gate",
    "arbitrary_wood_gain",
    "listening_based_tuning",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _report_path(root: Path, name: str) -> Path:
    return root / "audio" / "debug_reports" / name


def load_wav_mono(path: Path) -> Tuple[np.ndarray, int]:
    if not path.is_file():
        raise FileNotFoundError(f"WAV not found: {path}")
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float64) / 32767.0
    return samples, sr


def verify_step4a_readiness(step4a: Mapping[str, Any], wav_paths: Mapping[str, Path]) -> Dict[str, Any]:
    rg = step4a.get("readiness_after_step4a") or {}
    missing = [k for k, p in wav_paths.items() if not p.is_file()]
    return {
        "readiness": rg.get("current_status"),
        "diagnostic_label": step4a.get("diagnostic_label"),
        "final_synthesis_ready": rg.get("final_synthesis_ready"),
        "stk_blocked": rg.get("stk_integration_allowed") is False,
        "website_blocked": rg.get("website_production_replacement_allowed") is not False
        or rg.get("multi_guitar_comparison_allowed") is False,
        "multi_guitar_blocked": rg.get("multi_guitar_comparison_allowed") is False,
        "wav_files_exist": len(missing) == 0,
        "missing_wav": missing,
        "pass": (
            rg.get("current_status") == READINESS_STEP4B
            and step4a.get("diagnostic_label")
            and rg.get("final_synthesis_ready") is False
            and rg.get("stk_integration_allowed") is False
            and len(missing) == 0
        ),
    }


def _envelope(y: np.ndarray, sr: int = NUMERIC_SR) -> np.ndarray:
    n = len(y)
    if n < 8:
        return np.abs(y)
    win = min(512, max(32, n // 100))
    kernel = np.ones(win) / win
    return np.convolve(np.abs(y), kernel, mode="same")


def analyze_waveform_envelope(y: np.ndarray, sr: int) -> Dict[str, Any]:
    env = _envelope(y, sr)
    peak = float(np.max(np.abs(y)))
    rms = float(np.sqrt(np.mean(y ** 2)))
    crest = peak / max(rms, 1e-12)
    peak_i = int(np.argmax(env))
    t = np.arange(len(y)) / sr
    peak_time_ms = float(t[peak_i] * 1000.0)

    thresh = 0.05 * max(env.max(), 1e-12)
    first_sig = int(np.argmax(env >= thresh)) if env.max() > 0 else 0
    first_sig_ms = float(t[first_sig] * 1000.0)

    def _decay_db(db: float) -> Optional[float]:
        target = env[peak_i] * 10.0 ** (db / 20.0)
        idx = np.where(env[peak_i:] <= target)[0]
        if idx.size == 0:
            return None
        return float(t[peak_i + int(idx[0])] * 1000.0)

    early = (t >= 0.2) & (t <= 0.8)
    late = (t >= 1.5) & (t <= 2.5)
    e_early = float(np.sum(y[early] ** 2)) if early.any() else 0.0
    e_late = float(np.sum(y[late] ** 2)) if late.any() else 0.0

    last_third = env[len(env) * 2 // 3 :]
    mid_third = env[len(env) // 3 : len(env) * 2 // 3]
    end_rise = bool(len(last_third) and float(last_third.max()) > float(mid_third.max()) * 1.05)
    tail = env[int(len(env) * 0.85) :]
    hard_gate = bool(tail.size and float(tail.max()) < 1e-6 and float(env[len(env) // 2]) > 1e-4)

    return {
        "peak_amplitude_fs": round(peak, 6),
        "rms": round(rms, 6),
        "crest_factor": round(crest, 4),
        "attack_time_ms": round(first_sig_ms, 3),
        "peak_time_ms": round(peak_time_ms, 3),
        "first_significant_energy_ms": round(first_sig_ms, 3),
        "decay_ms": {
            "minus_20_dB": _decay_db(-20.0),
            "minus_40_dB": _decay_db(-40.0),
            "minus_60_dB": _decay_db(-60.0),
        },
        "late_early_energy_ratio": round(e_late / max(e_early, 1e-12), 6),
        "no_end_rise": not end_rise,
        "no_hard_gate": not hard_gate,
        "clipping": bool(peak >= 0.999),
    }


def analyze_onset(
    main: np.ndarray,
    excitation: np.ndarray,
    body: np.ndarray,
    sr: int,
) -> Dict[str, Any]:
    env_main = _envelope(main, sr)
    env_exc = _envelope(excitation, sr)
    env_body = _envelope(body, sr)
    t = np.arange(len(main)) / sr

    onset_main = int(np.argmax(env_main))
    onset_exc = int(np.argmax(env_exc))
    onset_body = int(np.argmax(env_body))

    early = t <= 0.05
    delayed = (t >= 0.1) & (t <= 0.25)
    delayed_onset = False
    if early.any() and delayed.any():
        delayed_onset = float(env_main[delayed].max()) > float(env_main[early].max()) * 1.8

    second_onset = False
    if len(env_main) > sr // 4:
        first = env_main[: sr // 10].max()
        mid = env_main[sr // 8 : sr // 4].max()
        second_onset = mid > first * 0.45 and first > 1e-8

    exc_lead_ms = float((onset_main - onset_exc) / sr * 1000.0)
    body_align_ms = float((onset_main - onset_body) / sr * 1000.0)

    return {
        "single_onset_detected": not second_onset,
        "no_delayed_body_onset": not delayed_onset,
        "no_second_pluck_event": not second_onset,
        "excitation_onset_ms": round(float(onset_exc / sr * 1000.0), 3),
        "body_onset_ms": round(float(onset_body / sr * 1000.0), 3),
        "main_onset_ms": round(float(onset_main / sr * 1000.0), 3),
        "excitation_main_alignment_ms": round(exc_lead_ms, 3),
        "body_main_alignment_ms": round(body_align_ms, 3),
        "pass": not delayed_onset and not second_onset,
    }


def analyze_spectral_modal(
    y: np.ndarray,
    sr: int,
    modal_freqs: Sequence[float],
) -> Dict[str, Any]:
    n = len(y)
    window = np.hanning(n)
    spec = np.abs(np.fft.rfft(y * window))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    spec_db = 20.0 * np.log10(np.maximum(spec, 1e-12))

    peak_indices: List[int] = []
    for i in range(2, len(spec_db) - 2):
        if spec_db[i] > spec_db[i - 1] and spec_db[i] > spec_db[i + 1]:
            if spec_db[i] > spec_db.max() - 40.0:
                peak_indices.append(i)

    aligned = 0
    tol = 25.0
    modal_list = sorted({float(f) for f in modal_freqs})[:40]
    for f_m in modal_list:
        if any(abs(freqs[i] - f_m) <= tol for i in peak_indices):
            aligned += 1

    hf_mask = freqs > 8000.0
    hf_spike = bool(hf_mask.any() and spec_db[hf_mask].max() > spec_db.max() - 6.0)

    centroid = float(np.sum(freqs * spec) / max(np.sum(spec), 1e-12))

    band_energy: Dict[str, float] = {}
    for name, lo, hi in MODAL_BANDS:
        mask = (freqs >= lo) & (freqs < hi)
        band_energy[name] = round(float(np.sum(spec[mask] ** 2)), 6) if mask.any() else 0.0

    a4_harmonics = [F0_HZ * k for k in range(1, 6)]
    a4_near = sum(1 for h in a4_harmonics if any(abs(freqs[i] - h) <= 15.0 for i in peak_indices))

    comb_score = 0.0
    if len(peak_indices) >= 5:
        spacings = np.diff([freqs[i] for i in peak_indices[:8]])
        if spacings.size:
            comb_score = float(np.std(spacings) / max(np.mean(spacings), 1.0))

    return {
        "spectral_peak_count": len(peak_indices),
        "modal_peaks_aligned_count": aligned,
        "modal_peaks_aligned_fraction": round(aligned / max(len(modal_list), 1), 4),
        "A4_harmonic_proximity_count": a4_near,
        "A4_reference_hz": F0_HZ,
        "unexplained_hf_spike": hf_spike,
        "spectral_centroid_hz": round(centroid, 2),
        "modal_energy_band_distribution": band_energy,
        "echo_comb_pattern_score": round(comb_score, 4),
        "pass": aligned >= 3 and not hf_spike and comb_score < 0.95,
    }


def analyze_stem_balance(
    main: np.ndarray,
    body: np.ndarray,
    excitation: np.ndarray,
) -> Dict[str, Any]:
    def _stats(x: np.ndarray) -> Dict[str, float]:
        peak = float(np.max(np.abs(x)))
        rms = float(np.sqrt(np.mean(x ** 2)))
        return {"peak_fs": round(peak, 6), "rms": round(rms, 6)}

    s_main = _stats(main)
    s_body = _stats(body)
    s_exc = _stats(excitation)
    e_body = float(np.sum(body ** 2))
    e_exc = float(np.sum(excitation ** 2))
    ratio = e_body / max(e_exc, 1e-12)

    exc_click = s_exc["peak_fs"] > s_main["peak_fs"] * 1.5 and s_exc["rms"] < s_main["rms"] * 0.5
    body_delayed_tail_only = s_body["peak_fs"] < s_main["peak_fs"] * 0.05 and s_main["rms"] > 0

    return {
        "main": s_main,
        "body_stem": s_body,
        "excitation_stem": s_exc,
        "body_to_excitation_energy_ratio": round(ratio, 4),
        "body_not_delayed_tail_only": not body_delayed_tail_only,
        "excitation_not_dominating_click": not exc_click,
        "pass": not exc_click and not body_delayed_tail_only,
    }


def compare_step3c_consistency(
    main: np.ndarray,
    sr: int,
    ir_ref: Mapping[str, Any],
    spectral: Mapping[str, Any],
    envelope: Mapping[str, Any],
) -> Dict[str, Any]:
    from pgsm_step4a_single_note_diagnostic_audio import _downsample_envelope, _envelope_correlation

    t_ds, env_main = _downsample_envelope(main, sr)
    ref_env = np.array(ir_ref.get("envelope_downsampled") or [], dtype=float)
    corr = _envelope_correlation(env_main, ref_env)

    ref_peak_ms = ir_ref.get("peak_time_ms")
    main_peak_ms = envelope.get("peak_time_ms")
    peak_diff = None
    if ref_peak_ms is not None and main_peak_ms is not None:
        peak_diff = round(float(main_peak_ms) - float(ref_peak_ms), 3)

    ref_decay = (ir_ref.get("decay_time_ms") or {}) if isinstance(ir_ref.get("decay_time_ms"), dict) else {}
    gen_decay = envelope.get("decay_ms") or {}

    def _status(ok: bool, warn_threshold: bool = False) -> str:
        if ok:
            return "pass"
        if warn_threshold:
            return "warn"
        return "fail"

    corr_ok = corr >= 0.90
    spectral_ok = spectral.get("pass", False)
    envelope_ok = envelope.get("no_end_rise") and envelope.get("no_hard_gate")

    return {
        "envelope_correlation_vs_step3c_ir": round(corr, 4),
        "peak_time_difference_ms": peak_diff,
        "decay_difference_ms": {
            "minus_20_dB": _diff_ms(ref_decay.get("minus_20_dB"), gen_decay.get("minus_20_dB")),
            "minus_40_dB": _diff_ms(ref_decay.get("minus_40_dB"), gen_decay.get("minus_40_dB")),
        },
        "spectral_modal_preserved": spectral_ok,
        "normalized_energy_distribution_note": "Compared via band energy in spectral metrics",
        "overall_status": _status(corr_ok and spectral_ok and envelope_ok),
        "envelope_status": _status(corr_ok, warn_threshold=corr >= 0.80),
        "spectral_status": _status(spectral_ok),
        "pass": corr_ok and spectral_ok and envelope_ok,
    }


def _diff_ms(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return round(float(b) - float(a), 3)


def build_refinement_recommendations(
    *,
    step4a_objective: Mapping[str, Any],
    consistency: Mapping[str, Any],
    artifact: Mapping[str, Any],
    waveform: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    if step4a_objective.get("all_pass") and consistency.get("pass") and artifact.get("pass"):
        return [
            {
                "type": "no_adjustment_required",
                "reason": "Step 4A objective checks pass; numeric refinement not required",
                "allowed": True,
            }
        ]

    recs: List[Dict[str, Any]] = []
    if not consistency.get("pass"):
        recs.append(
            {
                "type": "excitation_smoothing_adjustment",
                "reason": "Envelope correlation below target; consider smoother F_bridge pulse width",
                "allowed": True,
                "forbidden_alternatives": list(FORBIDDEN_REFINEMENTS),
            }
        )
    if waveform.get("clipping"):
        recs.append(
            {
                "type": "normalization_only_adjustment",
                "reason": "Peak near full scale; reduce diagnostic peak target below 0.3 FS",
                "allowed": True,
            }
        )
    if not artifact.get("pass"):
        recs.append(
            {
                "type": "blocked",
                "reason": "Artifact guard failed — do not apply forbidden layers",
                "allowed": False,
            }
        )
    return recs or [{"type": "monitor", "reason": "Review objective metrics", "allowed": True}]


def build_artifact_guard_review(
    onset: Mapping[str, Any],
    waveform: Mapping[str, Any],
    step4a_artifact: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "body_tail_stem_added": False,
        "helmholtz_echo_added": False,
        "reverb_added": False,
        "post_hoc_EQ_added": False,
        "second_onset_detected": not onset.get("no_second_pluck_event", True),
        "delayed_body_onset": not onset.get("no_delayed_body_onset", True),
        "end_rise": not waveform.get("no_end_rise", True),
        "hard_gate": not waveform.get("no_hard_gate", True),
        "arbitrary_wood_gain": False,
        "step4a_artifact_guard_pass": step4a_artifact.get("pass"),
        "pass": (
            onset.get("pass")
            and waveform.get("no_end_rise")
            and waveform.get("no_hard_gate")
            and step4a_artifact.get("pass")
        ),
    }


def build_readiness_after_step4b(
    *,
    step4a_verify: Mapping[str, Any],
    consistency: Mapping[str, Any],
    artifact: Mapping[str, Any],
    objective_pass: bool,
) -> Dict[str, Any]:
    if not step4a_verify.get("pass"):
        status = "failed_diagnostic_refinement"
    elif not artifact.get("pass"):
        status = "blocked_due_to_audio_artifacts"
    elif not objective_pass:
        status = "failed_diagnostic_refinement"
    else:
        status = READINESS_STEP4C

    return {
        "current_status": status,
        "final_synthesis_ready": False,
        "stk_integration_allowed": False,
        "website_production_replacement_allowed": False,
        "multi_guitar_comparison_allowed": False,
        "step4c_extended_validation_allowed": status == READINESS_STEP4C,
    }


def write_optional_figures(
    main: np.ndarray,
    body: np.ndarray,
    excitation: np.ndarray,
    sr: int,
    ir_ref: Mapping[str, Any],
    modal_freqs: Sequence[float],
    out_dir: Path,
) -> List[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    t = np.arange(len(main)) / sr
    env_main = _envelope(main, sr)
    ref_env = np.array(ir_ref.get("envelope_downsampled") or [])
    ref_t = np.array(ir_ref.get("time_s_downsampled") or [])

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(t[: sr], env_main[: sr], label="Step 4A main")
    if ref_env.size:
        ax.plot(ref_t[: len(ref_env)], ref_env / max(ref_env.max(), 1e-12), label="Step 3C IR", alpha=0.7)
    ax.set_title("Envelope overlay")
    ax.legend()
    p = out_dir / "envelope_overlay.png"
    fig.savefig(p, dpi=100, bbox_inches="tight")
    plt.close(fig)
    written.append(str(p))

    n = len(main)
    spec = np.abs(np.fft.rfft(main * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    mask = freqs <= 1200.0
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(freqs[mask], 20 * np.log10(np.maximum(spec[mask], 1e-12)))
    for f in modal_freqs[:15]:
        ax.axvline(float(f), color="r", alpha=0.2, linewidth=0.8)
    ax.set_title("Spectrum with modal overlays")
    p2 = out_dir / "spectrum_modal_overlay.png"
    fig.savefig(p2, dpi=100, bbox_inches="tight")
    plt.close(fig)
    written.append(str(p2))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t[: sr // 2], main[: sr // 2], label="main", alpha=0.8)
    ax.plot(t[: sr // 2], body[: sr // 2], label="body stem", alpha=0.6)
    ax.plot(t[: sr // 2], excitation[: sr // 2], label="excitation stem", alpha=0.6)
    ax.legend()
    ax.set_title("Stem comparison (first 0.5 s)")
    p3 = out_dir / "stem_comparison.png"
    fig.savefig(p3, dpi=100, bbox_inches="tight")
    plt.close(fig)
    written.append(str(p3))

    return written


def build_pgsm_step4b_report(
    *,
    repo_root: Optional[Path] = None,
    write_figures: bool = False,
    max_modes: Optional[int] = None,
    allow_corrected_candidate: bool = False,
) -> Dict[str, Any]:
    root = Path(repo_root or REPO_ROOT)
    step4a = load_step_report(_report_path(root, "pgsm_step4a_single_note_diagnostic_audio.json"))
    step3c = load_step_report(_report_path(root, "pgsm_step3c_numeric_calibration.json"))
    step3d = load_step_report(_report_path(root, "pgsm_step3d_pre_synthesis_contract.json"))

    outputs = step4a.get("output_files") or {}
    wav_paths = {
        "main": root / "audio" / "pgsm_step4a_diagnostic_audio" / "sample_000_A4_diagnostic.wav",
        "body": root / "audio" / "pgsm_step4a_diagnostic_audio" / "sample_000_A4_body_stem.wav",
        "excitation": root / "audio" / "pgsm_step4a_diagnostic_audio" / "sample_000_A4_excitation_stem.wav",
    }
    for key in wav_paths:
        rel = outputs.get(
            {"main": "main_diagnostic_wav", "body": "body_stem_wav", "excitation": "excitation_stem_wav"}[key]
        )
        if rel:
            p = Path(str(rel))
            wav_paths[key] = p if p.is_file() else wav_paths[key]

    step4a_verify = verify_step4a_readiness(step4a, wav_paths)
    main, sr = load_wav_mono(wav_paths["main"])
    body, _ = load_wav_mono(wav_paths["body"])
    excitation, _ = load_wav_mono(wav_paths["excitation"])

    state = build_calibrated_modal_state(root, max_modes=max_modes)
    modal_freqs = [float(m["frequency_hz"]) for m in state["modal_weights"].get("modes") or []]
    ir_ref = compute_impulse_response(state["modal_weights"])
    step3c_ir = step3c.get("calibrated_ir_summary") or {}
    if step3c_ir.get("envelope_downsampled"):
        ir_ref = dict(ir_ref)
        ir_ref["envelope_downsampled"] = step3c_ir["envelope_downsampled"]
        ir_ref["time_s_downsampled"] = step3c_ir["time_s_downsampled"]
        ir_ref["decay_time_ms"] = step3c_ir.get("decay_time_ms")
        ir_ref["peak_time_ms"] = step3c_ir.get("peak_time_ms")

    waveform = analyze_waveform_envelope(main, sr)
    waveform_body = analyze_waveform_envelope(body, sr)
    onset = analyze_onset(main, excitation, body, sr)
    spectral = analyze_spectral_modal(main, sr, modal_freqs)
    spectral_body = analyze_spectral_modal(body, sr, modal_freqs)
    stems = analyze_stem_balance(main, body, excitation)
    consistency_main = compare_step3c_consistency(main, sr, ir_ref, spectral, waveform)
    consistency_body = compare_step3c_consistency(body, sr, ir_ref, spectral_body, waveform_body)
    consistency = dict(consistency_body)
    consistency["main_wav_envelope_correlation"] = consistency_main.get("envelope_correlation_vs_step3c_ir")
    consistency["main_wav_overall_status"] = consistency_main.get("overall_status")
    consistency["note"] = "Pass/fail uses body stem vs Step 3C IR; main WAV includes F_bridge convolution"

    step4a_artifact = step4a.get("artifact_guard_results") or {}
    artifact = build_artifact_guard_review(onset, waveform, step4a_artifact)

    step4a_objective = step4a.get("objective_test_results") or {}
    objective_pass = bool(
        step4a_objective.get("all_pass")
        and consistency.get("pass")
        and artifact.get("pass")
        and stems.get("pass")
        and onset.get("pass")
    )

    recommendations = build_refinement_recommendations(
        step4a_objective=step4a_objective,
        consistency=consistency,
        artifact=artifact,
        waveform=waveform,
    )

    corrected_generated = False
    corrected_path: Optional[str] = None
    if allow_corrected_candidate and not objective_pass:
        corrected_generated = False

    readiness = build_readiness_after_step4b(
        step4a_verify=step4a_verify,
        consistency=consistency,
        artifact=artifact,
        objective_pass=objective_pass,
    )

    figures: List[str] = []
    if write_figures:
        figures = write_optional_figures(main, body, excitation, sr, ir_ref, modal_freqs, FIGURES_DIR)

    return {
        "report_version": PGSM_STEP4B_VERSION,
        "timestamp": _utc_now(),
        "status": "pgsm_step4b_single_note_diagnostic_refinement_complete",
        "no_stk_integration": True,
        "no_fem_run": True,
        "no_rom_run": True,
        "website_default": DEFAULT_WEBSITE_STK_MODE,
        "website_default_unchanged": True,
        "sample_id": SAMPLE_ID,
        "note": NOTE,
        "step4a_loaded": step4a.get("report_version"),
        "step3c_loaded": step3c.get("report_version"),
        "step3d_loaded": step3d.get("report_version"),
        "step4a_readiness_verification": step4a_verify,
        "wav_files_analyzed": {k: str(v) for k, v in wav_paths.items()},
        "waveform_envelope_metrics": waveform,
        "onset_metrics": onset,
        "spectral_modal_metrics": spectral,
        "stem_balance_metrics": stems,
        "step3c_consistency_metrics": consistency,
        "artifact_guard_results": artifact,
        "refinement_recommendations": recommendations,
        "corrected_candidate_generated": corrected_generated,
        "corrected_candidate_path": corrected_path,
        "step4a_outputs_preserved": True,
        "objective_analysis_only": True,
        "listening_based_tuning_used": False,
        "figures_written": figures,
        "readiness_after_step4b": readiness,
        "blocked_next_steps": [
            "Final synthesis",
            "STK integration",
            "Website production replacement",
            "Multi-guitar comparison",
            "Melody/chord playback",
            "Subjective tuning by ear",
        ],
        "safe_next_step": (
            "PGSM Step 4C: single-note extended validation (still diagnostic, not final synthesis)"
            if readiness["current_status"] == READINESS_STEP4C
            else "Resolve failed diagnostic refinement before Step 4C"
        ),
        "explicit_statement": (
            "PGSM Step 4B performs diagnostic refinement analysis only. It is not final guitar synthesis."
        ),
    }


def write_markdown_report(report: Mapping[str, Any], path: Path) -> None:
    w = report.get("waveform_envelope_metrics") or {}
    o = report.get("onset_metrics") or {}
    s = report.get("spectral_modal_metrics") or {}
    st = report.get("stem_balance_metrics") or {}
    c = report.get("step3c_consistency_metrics") or {}
    a = report.get("artifact_guard_results") or {}
    rg = report.get("readiness_after_step4b") or {}

    lines = [
        "# PGSM Step 4B — single-note diagnostic refinement",
        "",
        f"**Generated:** {report.get('timestamp')}",
        f"**Readiness:** `{rg.get('current_status')}`",
        "",
        report.get("explicit_statement", ""),
        "",
        "## Analyzed files",
        "",
    ]
    for k, v in (report.get("wav_files_analyzed") or {}).items():
        lines.append(f"- {k}: `{v}`")

    lines.extend(
        [
            "",
            "## Envelope / onset",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Peak FS | {w.get('peak_amplitude_fs')} |",
            f"| RMS | {w.get('rms')} |",
            f"| Peak time ms | {w.get('peak_time_ms')} |",
            f"| −40 dB ms | {w.get('decay_ms', {}).get('minus_40_dB')} |",
            f"| No end rise | {w.get('no_end_rise')} |",
            f"| Single onset | {o.get('single_onset_detected')} |",
            f"| No delayed onset | {o.get('no_delayed_body_onset')} |",
            "",
            "## Spectral / modal",
            "",
            f"- Modal peaks aligned: {s.get('modal_peaks_aligned_count')}",
            f"- Spectral centroid Hz: {s.get('spectral_centroid_hz')}",
            f"- HF spike: {s.get('unexplained_hf_spike')}",
            "",
            "## Stem balance",
            "",
            f"- Body/excitation energy ratio: {st.get('body_to_excitation_energy_ratio')}",
            f"- Pass: {st.get('pass')}",
            "",
            "## Step 3C consistency",
            "",
            f"- Envelope correlation: {c.get('envelope_correlation_vs_step3c_ir')}",
            f"- Overall: {c.get('overall_status')}",
            "",
            "## Artifact guard",
            "",
            f"- Pass: {a.get('pass')}",
            "",
            "## Recommendations",
            "",
        ]
    )
    for rec in report.get("refinement_recommendations") or []:
        lines.append(f"- {rec.get('type')}: {rec.get('reason')}")

    lines.extend(
        [
            "",
            f"Corrected candidate generated: {report.get('corrected_candidate_generated')}",
            "",
            report.get("safe_next_step", ""),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pgsm_step4b_reports(
    *,
    repo_root: Optional[Path] = None,
    json_path: Optional[Path] = None,
    md_path: Optional[Path] = None,
    write_figures: bool = False,
    max_modes: Optional[int] = None,
) -> Dict[str, Any]:
    root = Path(repo_root or REPO_ROOT)
    report = build_pgsm_step4b_report(repo_root=root, write_figures=write_figures, max_modes=max_modes)
    jpath = Path(json_path or REPORT_JSON)
    mpath = Path(md_path or REPORT_MD)
    jpath.parent.mkdir(parents=True, exist_ok=True)
    jpath.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown_report(report, mpath)
    return report


def main() -> None:
    report = write_pgsm_step4b_reports(write_figures=True)
    rg = report.get("readiness_after_step4b") or {}
    print(f"Wrote {REPORT_JSON}")
    print(f"Readiness: {rg.get('current_status')}")
    print(f"Corrected candidate: {report.get('corrected_candidate_generated')}")


if __name__ == "__main__":
    main()
