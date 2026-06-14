#!/usr/bin/env python3
"""
PGSM Step 5A — limited note-set diagnostic audio.
Generates A2/A3/A4/E5 diagnostic WAVs for sample_000 only; not final synthesis.
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

from pgsm_step2_1_parameter_targets import PLUCK_DURATION_MS_TYPICAL, load_step_report
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
    evaluate_envelope_consistency,
    normalize_diagnostic_amplitude,
    synthesize_modal_body_response,
    write_wav_mono,
)
from pgsm_step4b_single_note_diagnostic_refinement import STEP4A_AUDIO_DIR
from pgsm_step4c_single_note_extended_validation import READINESS_STEP5A
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE

PGSM_STEP5A_VERSION = "pgsm_step5a_limited_note_set_diagnostic_audio_v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON = REPO_ROOT / "audio" / "debug_reports" / "pgsm_step5a_limited_note_set_diagnostic_audio.json"
REPORT_MD = REPO_ROOT / "audio" / "debug_reports" / "pgsm_step5a_limited_note_set_diagnostic_audio.md"
FIGURES_DIR = REPO_ROOT / "audio" / "debug_reports" / "pgsm_step5a_figures"
AUDIO_DIR = REPO_ROOT / "audio" / "pgsm_step5a_limited_note_set"

READINESS_STEP5B = "ready_for_step5b_limited_note_set_refinement"
MAX_DIAGNOSTIC_PEAK_FS = 0.28

NOTE_SET: Tuple[str, ...] = ("A2", "A3", "A4", "E5")
NOTE_FREQUENCY_HZ: Dict[str, float] = {
    "A2": 110.0,
    "A3": 220.0,
    "A4": 440.0,
    "E5": 659.255118865,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _report_path(root: Path, name: str) -> Path:
    return root / "audio" / "debug_reports" / name


def _file_fingerprint(path: Path) -> str:
    if not path.is_file():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def step4a_output_fingerprints(root: Path) -> Dict[str, str]:
    d = root / "audio" / "pgsm_step4a_diagnostic_audio"
    if not d.is_dir():
        return {}
    return {p.name: _file_fingerprint(p) for p in sorted(d.glob("*.wav"))}


def verify_upstream_readiness(
    step4c: Mapping[str, Any],
    step4a_fp_before: Mapping[str, str],
) -> Dict[str, Any]:
    rg4c = step4c.get("readiness_after_step4c") or {}
    return {
        "step4c_readiness": rg4c.get("current_status"),
        "step4c_pass": rg4c.get("current_status") == READINESS_STEP5A,
        "final_synthesis_blocked": rg4c.get("final_synthesis_ready") is False,
        "stk_blocked": rg4c.get("stk_integration_allowed") is False,
        "website_blocked": rg4c.get("website_production_replacement_allowed") is False,
        "multi_guitar_blocked": rg4c.get("multi_guitar_comparison_allowed") is False,
        "melody_chords_blocked": True,
        "step4a_fingerprints_before": dict(step4a_fp_before),
        "pass": bool(
            rg4c.get("current_status") == READINESS_STEP5A
            and rg4c.get("final_synthesis_ready") is False
            and rg4c.get("stk_integration_allowed") is False
            and rg4c.get("website_production_replacement_allowed") is False
            and rg4c.get("multi_guitar_comparison_allowed") is False
        ),
    }


def build_note_contract(
    note: str,
    *,
    string_val: Mapping[str, Any],
    reference_hz: float = 440.0,
) -> Dict[str, Any]:
    note_hz = NOTE_FREQUENCY_HZ[note]
    l_cal = float(string_val.get("calibrated_L_for_mid_tension_m") or 0.45)
    l_eff = round(l_cal * (reference_hz / note_hz), 5)
    tension_mid = 0.5 * sum(string_val.get("plausible_tension_range_N") or [40.0, 120.0])
    return {
        "note": note,
        "note_frequency_hz": note_hz,
        "effective_vibrating_length_m": l_eff,
        "string_interpretation": (
            f"diagnostic_reference_frequency_{note}_not_exact_fingering"
        ),
        "exact_open_string_claim_allowed": False,
        "tension_interpretation": (
            f"mid_plausible_tension_proxy_{round(tension_mid, 1)}N_not_measured"
        ),
        "source_fallback_level": "L2_diagnostic_harmonic_reference",
        "harmonic_modal_excitation_only": True,
        "physical_fingering_validated": False,
    }


def build_f_bridge_note_proxy(
    n: int,
    sr: int,
    note_hz: float,
    *,
    pluck_duration_ms: float = PLUCK_DURATION_MS_TYPICAL,
    amplitude: float = 1.0,
    n_harmonics: int = 6,
) -> np.ndarray:
    """Smooth F_bridge pulse with note-frequency harmonic shaping (not audio click)."""
    f = np.zeros(n, dtype=np.float64)
    n_pulse = max(int(pluck_duration_ms * 1e-3 * sr), 2)
    n_pulse = min(n_pulse, n)
    t_pulse = np.arange(n_pulse, dtype=np.float64) / sr
    window = np.sin(np.pi * np.linspace(0.0, 1.0, n_pulse)) ** 2
    if n_pulse > 1:
        window[0] = max(window[1] * 0.2, 1e-8)

    harmonic = np.zeros(n_pulse, dtype=np.float64)
    for k in range(1, n_harmonics + 1):
        harmonic += (1.0 / k) * np.sin(2.0 * math.pi * k * note_hz * t_pulse)
    hmax = max(float(np.max(np.abs(harmonic))), 1e-12)
    harmonic /= hmax

    shaped = window * (0.82 + 0.18 * harmonic)
    f[:n_pulse] = amplitude * shaped
    return f


def _output_paths(audio_dir: Path, note: str) -> Dict[str, Path]:
    base = f"{SAMPLE_ID}_{note}"
    return {
        "main": audio_dir / f"{base}_diagnostic.wav",
        "body_stem": audio_dir / f"{base}_body_stem.wav",
        "excitation_stem": audio_dir / f"{base}_excitation_stem.wav",
    }


def _rms(y: np.ndarray) -> float:
    if y.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(y ** 2)))


def _attack_peak_ms(y: np.ndarray, sr: int) -> Tuple[float, float]:
    env = np.abs(y)
    t = np.arange(len(y)) / sr
    if env.size == 0 or env.max() <= 0:
        return 0.0, 0.0
    peak_i = int(np.argmax(env))
    attack_i = int(np.argmax(env >= 0.05 * env.max()))
    return float(t[attack_i] * 1000.0), float(t[peak_i] * 1000.0)


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


def evaluate_spectral_modal_for_note(
    body: np.ndarray,
    sr: int,
    modal_weights: Mapping[str, Any],
    note_hz: float,
) -> Dict[str, Any]:
    modes = modal_weights.get("modes") or []
    n = len(body)
    if n < 256:
        return {"status": "too_short", "pass": False}

    spec = np.abs(np.fft.rfft(body * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    spec_db = 20.0 * np.log10(np.maximum(spec, 1e-12))
    total_power = max(float(np.sum(spec ** 2)), 1e-12)
    centroid = float(np.sum(freqs * spec ** 2) / total_power)

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

    note_harmonics = [note_hz * k for k in range(1, 6)]
    harmonic_near = sum(
        1 for h in note_harmonics if any(abs(freqs[i] - h) <= 15.0 for i in peak_indices)
    )

    return {
        "spectral_centroid_hz": round(centroid, 2),
        "spectral_peak_count": len(peak_indices),
        "modal_peaks_aligned_count": aligned,
        "note_harmonic_proximity_count": harmonic_near,
        "unexplained_hf_spike": bool(hf_spike),
        "echo_comb_pattern_score": round(comb_score, 4),
        "no_artificial_hf_spike": bool(not hf_spike),
        "no_echo_comb_pattern": bool(comb_score < 0.95),
        "modal_peak_alignment_strong": bool(aligned >= 3),
        "pass": bool(aligned >= 3 and not hf_spike and comb_score < 0.95),
    }


def evaluate_stem_balance(
    main: np.ndarray,
    body: np.ndarray,
    excitation: np.ndarray,
    sr: int,
) -> Dict[str, Any]:
    n = min(len(main), len(body), len(excitation))
    main, body, excitation = main[:n], body[:n], excitation[:n]
    e_body = float(np.sum(body ** 2))
    e_exc = float(np.sum(excitation ** 2))
    ratio = e_body / max(e_exc, 1e-12)
    exc_peak = float(np.max(np.abs(excitation)))
    main_peak = float(np.max(np.abs(main)))
    exc_not_click = bool(exc_peak < main_peak * 1.2 or e_exc < e_body * 0.5)
    return {
        "body_excitation_energy_ratio": round(ratio, 4),
        "excitation_not_dominant_click": exc_not_click,
        "pass": exc_not_click,
    }


def evaluate_per_note_audio(
    main: np.ndarray,
    body_stem: np.ndarray,
    excitation_stem: np.ndarray,
    sr: int,
    *,
    step3c_ir: Mapping[str, Any],
    modal_weights: Mapping[str, Any],
    note_hz: float,
    h_body: np.ndarray,
    normalization: Mapping[str, Any],
) -> Dict[str, Any]:
    env = np.abs(main)
    t = np.arange(len(main)) / sr
    attack_ms, peak_ms = _attack_peak_ms(main, sr)
    decay = {
        "minus_20_dB_ms": _decay_time_ms(env, t, -20.0),
        "minus_40_dB_ms": _decay_time_ms(env, t, -40.0),
        "minus_60_dB_ms": _decay_time_ms(env, t, -60.0),
    }

    envelope = evaluate_envelope_consistency(main, sr, step3c_ir, h_body=h_body)
    spectral = evaluate_spectral_modal_for_note(main, sr, modal_weights, note_hz)
    artifact = evaluate_artifact_guard(
        main, excitation_stem, sr, envelope_pass_onset=envelope.get("no_delayed_body_onset", False)
    )
    stem = evaluate_stem_balance(main, body_stem, excitation_stem, sr)

    peak_fs = float(normalization.get("peak_after_fs", 0.0))
    return {
        "peak_fs": round(peak_fs, 6),
        "rms": round(_rms(main), 6),
        "attack_time_ms": round(attack_ms, 3),
        "peak_time_ms": round(peak_ms, 3),
        "decay_ms": decay,
        "no_delayed_onset": bool(envelope.get("no_delayed_body_onset")),
        "no_second_onset": bool(not artifact.get("second_pluck_onset")),
        "no_end_rise": bool(envelope.get("no_end_rise")),
        "no_hard_gate": bool(envelope.get("no_hard_gate")),
        "envelope_correlation_vs_step3c_ir": envelope.get("envelope_correlation_vs_step3c"),
        "envelope_consistent_with_modal_ir": bool(envelope.get("pass")),
        "spectral_centroid_hz": spectral.get("spectral_centroid_hz"),
        "modal_peaks_aligned_count": spectral.get("modal_peaks_aligned_count"),
        "no_artificial_hf_spike": spectral.get("no_artificial_hf_spike"),
        "no_echo_comb_pattern": spectral.get("no_echo_comb_pattern"),
        "excitation_not_dominant_click": stem.get("excitation_not_dominant_click"),
        "body_excitation_energy_ratio": stem.get("body_excitation_energy_ratio"),
        "clipping": bool(normalization.get("clipping")),
        "loudness_matching_applied": bool(normalization.get("loudness_matching_applied")),
        "pass": bool(
            peak_fs <= 0.3
            and not normalization.get("clipping")
            and envelope.get("pass")
            and spectral.get("pass")
            and artifact.get("pass")
            and stem.get("pass")
        ),
    }


def build_per_note_artifact_guard(
    metrics: Mapping[str, Any],
    artifact_base: Mapping[str, Any],
) -> Dict[str, Any]:
    checks = {
        "no_body_tail": artifact_base.get("body_tail_stem_used") is False,
        "no_helmholtz_echo_ir": artifact_base.get("helmholtz_echo_ir_used") is False,
        "no_reverb": artifact_base.get("artificial_reverb") is False,
        "no_post_hoc_EQ": artifact_base.get("post_hoc_EQ_body_layer") is False,
        "no_arbitrary_wood_gain": artifact_base.get("arbitrary_wood_to_gain_mapping") is False,
        "no_delayed_onset": metrics.get("no_delayed_onset"),
        "no_second_onset": metrics.get("no_second_onset"),
        "no_end_rise": metrics.get("no_end_rise"),
        "no_hard_gate": metrics.get("no_hard_gate"),
        "no_hf_spike": metrics.get("no_artificial_hf_spike"),
        "no_comb_echo": metrics.get("no_echo_comb_pattern"),
    }
    return {**checks, "pass": bool(all(bool(v) for v in checks.values()))}


def build_cross_note_sanity(
    per_note_metrics: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    notes = list(NOTE_SET)
    peaks = [float((per_note_metrics[n] or {}).get("peak_fs", 0.0)) for n in notes]
    centroids = [float((per_note_metrics[n] or {}).get("spectral_centroid_hz", 0.0)) for n in notes]
    decays_40 = [
        (per_note_metrics[n] or {}).get("decay_ms", {}).get("minus_40_dB_ms") for n in notes
    ]

    amp_in_range = bool(all(0.05 < p <= 0.3 for p in peaks))
    no_late_rise = bool(all((per_note_metrics[n] or {}).get("no_end_rise") for n in notes))
    no_click = bool(all((per_note_metrics[n] or {}).get("excitation_not_dominant_click") for n in notes))
    modal_strong = bool(
        all((per_note_metrics[n] or {}).get("modal_peaks_aligned_count", 0) >= 3 for n in notes)
    )
    no_loudness_match = bool(
        all((per_note_metrics[n] or {}).get("loudness_matching_applied") is False for n in notes)
    )
    centroid_spread = max(centroids) - min(centroids) if centroids else 0.0
    centroid_trend_reasonable = bool(centroid_spread < 4000.0)

    return {
        "peak_fs_by_note": {n: peaks[i] for i, n in enumerate(notes)},
        "spectral_centroid_hz_by_note": {n: centroids[i] for i, n in enumerate(notes)},
        "decay_minus_40_db_ms_by_note": {n: decays_40[i] for i, n in enumerate(notes)},
        "centroid_spread_hz": round(centroid_spread, 2),
        "amplitudes_within_diagnostic_range": amp_in_range,
        "no_note_has_late_rise": no_late_rise,
        "no_note_click_dominant_excitation": no_click,
        "spectral_centroid_trend_reasonable_not_realism_claim": centroid_trend_reasonable,
        "modal_peak_alignment_strong_all_notes": modal_strong,
        "no_hidden_loudness_matching": no_loudness_match,
        "pass": bool(
            amp_in_range
            and no_late_rise
            and no_click
            and modal_strong
            and no_loudness_match
            and centroid_trend_reasonable
        ),
    }


def run_objective_tests(
    upstream: Mapping[str, Any],
    per_note_metrics: Mapping[str, Mapping[str, Any]],
    per_note_artifacts: Mapping[str, Mapping[str, Any]],
    cross_note: Mapping[str, Any],
    step4a_preserved: bool,
) -> Dict[str, Any]:
    per_note_pass = {
        n: bool((per_note_metrics.get(n) or {}).get("pass"))
        and bool((per_note_artifacts.get(n) or {}).get("pass"))
        for n in NOTE_SET
    }
    tests = {
        "upstream_ready": upstream.get("pass"),
        "step4a_outputs_preserved": step4a_preserved,
        "all_notes_pass": all(per_note_pass.values()),
        "cross_note_sanity": cross_note.get("pass"),
        "final_synthesis_blocked": True,
        "stk_integration_blocked": True,
        "melody_chords_blocked": True,
        "exact_open_string_claims_blocked": True,
        "no_subjective_listening_acceptance": True,
    }
    tests["all_pass"] = bool(all(tests.values()))
    return tests


def build_readiness_after_step5a(
    objective: Mapping[str, Any],
    per_note_artifacts: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    any_artifact_block = any(not (per_note_artifacts.get(n) or {}).get("pass") for n in NOTE_SET)
    if any_artifact_block:
        status = "blocked_due_to_note_specific_artifacts"
    elif not objective.get("all_pass"):
        status = "failed_limited_note_set_diagnostic_audio"
    else:
        status = READINESS_STEP5B

    return {
        "current_status": status,
        "final_synthesis_ready": False,
        "stk_integration_allowed": False,
        "website_production_replacement_allowed": False,
        "multi_guitar_comparison_allowed": False,
        "melody_chord_playback_allowed": False,
        "subjective_tuning_allowed": False,
        "step5b_limited_note_set_refinement_allowed": status == READINESS_STEP5B,
    }


def write_optional_figures(
    note_signals: Mapping[str, np.ndarray],
    sr: int,
    modal_freqs: Sequence[float],
    per_note_metrics: Mapping[str, Mapping[str, Any]],
    out_dir: Path,
) -> List[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []

    fig, ax = plt.subplots(figsize=(9, 3))
    for note in NOTE_SET:
        y = note_signals.get(note)
        if y is None:
            continue
        t = np.arange(len(y)) / sr
        env = np.abs(y)
        step = max(len(y) // 500, 1)
        ax.plot(t[::step], env[::step] / max(env.max(), 1e-12), label=note)
    ax.set_title("Envelope comparison across notes")
    ax.legend()
    p = out_dir / "envelope_comparison_across_notes.png"
    fig.savefig(p, dpi=100, bbox_inches="tight")
    plt.close(fig)
    written.append(str(p))

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, note in zip(axes.flat, NOTE_SET):
        y = note_signals.get(note)
        if y is None:
            continue
        n = len(y)
        spec = np.abs(np.fft.rfft(y * np.hanning(n)))
        freqs = np.fft.rfftfreq(n, 1.0 / sr)
        mask = freqs <= 1200.0
        ax.plot(freqs[mask], 20 * np.log10(np.maximum(spec[mask], 1e-12)))
        for f in modal_freqs[:10]:
            ax.axvline(float(f), color="r", alpha=0.15, linewidth=0.6)
        ax.set_title(f"{note} spectrum + modal overlay")
    fig.tight_layout()
    p2 = out_dir / "spectrum_modal_overlay_by_note.png"
    fig.savefig(p2, dpi=100, bbox_inches="tight")
    plt.close(fig)
    written.append(str(p2))

    peaks = [(per_note_metrics[n] or {}).get("peak_fs", 0.0) for n in NOTE_SET]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(list(NOTE_SET), peaks)
    ax.axhline(0.3, color="r", linestyle="--", alpha=0.5, label="0.3 FS limit")
    ax.set_title("Peak amplitude by note")
    ax.legend()
    p3 = out_dir / "note_set_summary_peaks.png"
    fig.savefig(p3, dpi=100, bbox_inches="tight")
    plt.close(fig)
    written.append(str(p3))

    decays = [
        (per_note_metrics[n] or {}).get("decay_ms", {}).get("minus_40_dB_ms") or 0.0
        for n in NOTE_SET
    ]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(list(NOTE_SET), decays)
    ax.set_title("Decay to -40 dB by note (ms)")
    ax.set_ylabel("ms")
    p4 = out_dir / "decay_metrics_by_note.png"
    fig.savefig(p4, dpi=100, bbox_inches="tight")
    plt.close(fig)
    written.append(str(p4))

    return written


def build_pgsm_step5a_report(
    *,
    repo_root: Optional[Path] = None,
    audio_dir: Optional[Path] = None,
    write_wav: bool = True,
    write_figures: bool = False,
    max_modes: Optional[int] = None,
) -> Dict[str, Any]:
    root = Path(repo_root or REPO_ROOT)
    out_audio = Path(audio_dir or (root / "audio" / "pgsm_step5a_limited_note_set"))

    step4c = load_step_report(_report_path(root, "pgsm_step4c_single_note_extended_validation.json"))
    step4b = load_step_report(_report_path(root, "pgsm_step4b_single_note_diagnostic_refinement.json"))
    step4a = load_step_report(_report_path(root, "pgsm_step4a_single_note_diagnostic_audio.json"))
    step3c = load_step_report(_report_path(root, "pgsm_step3c_numeric_calibration.json"))
    step3d = load_step_report(_report_path(root, "pgsm_step3d_pre_synthesis_contract.json"))

    step4a_fp_before = step4a_output_fingerprints(root)
    upstream = verify_upstream_readiness(step4c, step4a_fp_before)

    state = build_calibrated_modal_state(root, max_modes=max_modes)
    cal_weights = state["modal_weights"]
    ir_ref = compute_impulse_response(cal_weights)
    string_val = state["step3b_report"].get("string_consistency_validation") or {}

    sr = NUMERIC_SR
    n = int(DURATION_S * sr)
    _, h_body = compute_full_impulse_response(cal_weights)

    per_note_contracts: Dict[str, Any] = {}
    per_note_metrics: Dict[str, Any] = {}
    per_note_artifacts: Dict[str, Any] = {}
    output_files: Dict[str, Any] = {"main_wav_count": len(NOTE_SET), "notes": {}}
    note_signals: Dict[str, np.ndarray] = {}
    modal_freqs = [float(m["frequency_hz"]) for m in cal_weights.get("modes") or []]

    body_stem_norm, norm_h = normalize_diagnostic_amplitude(h_body, max_peak_fs=0.15)

    for note in NOTE_SET:
        note_hz = NOTE_FREQUENCY_HZ[note]
        contract = build_note_contract(note, string_val=string_val)
        per_note_contracts[note] = contract

        f_bridge = build_f_bridge_note_proxy(n, sr, note_hz)
        body_raw = synthesize_modal_body_response(f_bridge, h_body)
        main_norm, norm_main = normalize_diagnostic_amplitude(body_raw, max_peak_fs=MAX_DIAGNOSTIC_PEAK_FS)
        exc_norm, _norm_exc = normalize_diagnostic_amplitude(f_bridge, max_peak_fs=0.15)

        paths = _output_paths(out_audio, note)
        if write_wav:
            write_wav_mono(paths["main"], main_norm, sr)
            write_wav_mono(paths["body_stem"], body_stem_norm, sr)
            write_wav_mono(paths["excitation_stem"], exc_norm, sr)

        artifact_base = evaluate_artifact_guard(
            main_norm, f_bridge, sr, envelope_pass_onset=True
        )
        metrics = evaluate_per_note_audio(
            main_norm,
            body_stem_norm,
            exc_norm,
            sr,
            step3c_ir=ir_ref,
            modal_weights=cal_weights,
            note_hz=note_hz,
            h_body=h_body,
            normalization=norm_main,
        )
        per_note_metrics[note] = metrics
        per_note_artifacts[note] = build_per_note_artifact_guard(metrics, artifact_base)
        note_signals[note] = main_norm
        output_files["notes"][note] = {
            "main_diagnostic_wav": str(paths["main"]),
            "body_stem_wav": str(paths["body_stem"]),
            "excitation_stem_wav": str(paths["excitation_stem"]),
        }

    step4a_fp_after = step4a_output_fingerprints(root)
    step4a_preserved = step4a_fp_before == step4a_fp_after

    cross_note = build_cross_note_sanity(per_note_metrics)
    objective = run_objective_tests(
        upstream, per_note_metrics, per_note_artifacts, cross_note, step4a_preserved
    )
    readiness = build_readiness_after_step5a(objective, per_note_artifacts)

    figures: List[str] = []
    if write_figures and write_wav:
        figures = write_optional_figures(
            note_signals, sr, modal_freqs, per_note_metrics, FIGURES_DIR
        )

    return {
        "report_version": PGSM_STEP5A_VERSION,
        "timestamp": _utc_now(),
        "status": "pgsm_step5a_limited_note_set_diagnostic_audio_complete",
        "diagnostic_audio_generated": write_wav,
        "final_synthesis_ready": False,
        "stk_integration_allowed": False,
        "website_default": DEFAULT_WEBSITE_STK_MODE,
        "website_default_unchanged": True,
        "sample_id": SAMPLE_ID,
        "note_set": list(NOTE_SET),
        "diagnostic_label": DIAGNOSTIC_LABEL,
        "output_files": output_files,
        "step4a_outputs_preserved": step4a_preserved,
        "step4a_fingerprints_after": step4a_fp_after,
        "upstream_readiness": upstream,
        "step4c_loaded": step4c.get("report_version"),
        "step4b_loaded": step4b.get("report_version"),
        "step4a_loaded": step4a.get("report_version"),
        "step3c_loaded": step3c.get("report_version"),
        "step3d_loaded": step3d.get("report_version"),
        "per_note_contracts": per_note_contracts,
        "per_note_audio_metrics": per_note_metrics,
        "per_note_artifact_guard": per_note_artifacts,
        "cross_note_sanity_checks": cross_note,
        "modal_body_shared_across_notes": True,
        "normalization_per_note_independent": True,
        "objective_test_results": objective,
        "blocked_claims": step3d.get("blocked_claims") or [
            "Final synthesis",
            "STK integration",
            "Website production replacement",
            "Multi-guitar comparison",
            "Melody/chord playback",
            "Subjective tuning by ear",
            "Exact open-string physical claims",
        ],
        "figures_written": figures,
        "readiness_after_step5a": readiness,
        "safe_next_step": (
            "PGSM Step 5B: limited note-set diagnostic refinement (still not final synthesis)"
            if readiness["current_status"] == READINESS_STEP5B
            else "Resolve Step 5A failures before Step 5B"
        ),
        "explicit_statement": (
            "PGSM Step 5A generates limited note-set diagnostic audio only. "
            "It is not final guitar synthesis."
        ),
    }


def write_markdown_report(report: Mapping[str, Any], path: Path) -> None:
    rg = report.get("readiness_after_step5a") or {}
    cross = report.get("cross_note_sanity_checks") or {}
    obj = report.get("objective_test_results") or {}
    contracts = report.get("per_note_contracts") or {}
    metrics = report.get("per_note_audio_metrics") or {}
    artifacts = report.get("per_note_artifact_guard") or {}

    lines = [
        "# PGSM Step 5A — limited note-set diagnostic audio",
        "",
        f"**Generated:** {report.get('timestamp')}",
        f"**Sample:** `{report.get('sample_id')}` | **Notes:** {', '.join(report.get('note_set') or [])}",
        f"**Readiness:** `{rg.get('current_status')}`",
        "",
        report.get("explicit_statement", ""),
        "",
        "## Generated WAV files",
        "",
    ]
    for note, files in (report.get("output_files") or {}).get("notes", {}).items():
        lines.append(f"### {note}")
        lines.append(f"- Main: `{files.get('main_diagnostic_wav')}`")
        lines.append(f"- Body stem: `{files.get('body_stem_wav')}`")
        lines.append(f"- Excitation stem: `{files.get('excitation_stem_wav')}`")
        lines.append("")

    lines.extend(["## Note contract", "", "| Note | f (Hz) | L_eff (m) | fallback | exact open |", "|------|--------|-----------|----------|------------|"])
    for note in NOTE_SET:
        c = contracts.get(note) or {}
        lines.append(
            f"| {note} | {c.get('note_frequency_hz')} | {c.get('effective_vibrating_length_m')} | "
            f"{c.get('source_fallback_level')} | {c.get('exact_open_string_claim_allowed')} |"
        )

    lines.extend(["", "## Per-note metrics", "", "| Note | peak FS | RMS | −40 dB ms | modal aligned | pass |", "|------|---------|-----|-----------|---------------|------|"])
    for note in NOTE_SET:
        m = metrics.get(note) or {}
        d40 = (m.get("decay_ms") or {}).get("minus_40_dB_ms")
        lines.append(
            f"| {note} | {m.get('peak_fs')} | {m.get('rms')} | {d40} | "
            f"{m.get('modal_peaks_aligned_count')} | {m.get('pass')} |"
        )

    lines.extend(
        [
            "",
            "## Cross-note sanity",
            "",
            f"- Amplitudes in range: {cross.get('amplitudes_within_diagnostic_range')}",
            f"- No late rise: {cross.get('no_note_has_late_rise')}",
            f"- Modal alignment all notes: {cross.get('modal_peak_alignment_strong_all_notes')}",
            f"- Pass: {cross.get('pass')}",
            "",
            "## Artifact guard (per note)",
            "",
        ]
    )
    for note in NOTE_SET:
        lines.append(f"- {note}: pass={ (artifacts.get(note) or {}).get('pass') }")

    lines.extend(
        [
            "",
            "## Readiness",
            "",
            f"Step 4A preserved: **{report.get('step4a_outputs_preserved')}**",
            f"all_pass: **{obj.get('all_pass')}**",
            "",
            report.get("safe_next_step", ""),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pgsm_step5a_reports(
    *,
    repo_root: Optional[Path] = None,
    json_path: Optional[Path] = None,
    md_path: Optional[Path] = None,
    audio_dir: Optional[Path] = None,
    write_figures: bool = False,
    max_modes: Optional[int] = None,
) -> Dict[str, Any]:
    root = Path(repo_root or REPO_ROOT)
    report = build_pgsm_step5a_report(
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
    report = write_pgsm_step5a_reports(write_figures=True)
    obj = report.get("objective_test_results") or {}
    rg = report.get("readiness_after_step5a") or {}
    print(f"Wrote {REPORT_JSON}")
    print(f"Readiness: {rg.get('current_status')}")
    print(f"Objective all_pass: {obj.get('all_pass')}")


if __name__ == "__main__":
    main()
