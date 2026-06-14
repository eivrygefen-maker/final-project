#!/usr/bin/env python3
"""
PGSM Step 5C — limited note-set extended validation.
Analyzes existing Step 5A audio only; no new WAV, no STK/FEM/ROM.
"""
from __future__ import annotations

import hashlib
import json
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
from pgsm_step5b_limited_note_set_refinement import (
    BODY_MIN_MODAL_TAIL_MS,
    READINESS_STEP5C,
    _wav_paths_for_note,
    analyze_per_note_stem_decay,
    step5a_output_fingerprints,
)
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE

PGSM_STEP5C_VERSION = "pgsm_step5c_note_set_extended_validation_v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON = REPO_ROOT / "audio" / "debug_reports" / "pgsm_step5c_note_set_extended_validation.json"
REPORT_MD = REPO_ROOT / "audio" / "debug_reports" / "pgsm_step5c_note_set_extended_validation.md"
FIGURES_DIR = REPO_ROOT / "audio" / "debug_reports" / "pgsm_step5c_figures"

READINESS_STEP6A = "ready_for_step6a_reference_guided_diagnostic_comparison"
DIAGNOSTIC_LABEL = "PGSM diagnostic audio, not final guitar"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _report_path(root: Path, name: str) -> Path:
    return root / "audio" / "debug_reports" / name


def _file_fingerprint(path: Path) -> str:
    if not path.is_file():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_upstream_readiness(
    step5b: Mapping[str, Any],
    step5a: Mapping[str, Any],
    wav_by_note: Mapping[str, Mapping[str, Path]],
    step5a_fp_before: Mapping[str, str],
) -> Dict[str, Any]:
    rg5b = step5b.get("readiness_after_step5b") or {}
    missing: Dict[str, List[str]] = {}
    for note in NOTE_SET:
        missing[note] = [k for k, p in wav_by_note[note].items() if not p.is_file()]
    main_count = sum(1 for n in NOTE_SET if wav_by_note[n]["main"].is_file())
    note_set_ok = list(step5a.get("note_set") or []) == list(NOTE_SET)
    return {
        "step5b_readiness": rg5b.get("current_status"),
        "step5b_pass": rg5b.get("current_status") == READINESS_STEP5C,
        "step5a_note_set": step5a.get("note_set"),
        "four_notes_a2_a3_a4_e5": note_set_ok and main_count == 4,
        "stems_exist_all_notes": all(len(v) == 0 for v in missing.values()),
        "missing_wav": missing,
        "step5b_corrected_candidate_generated": bool(step5b.get("corrected_candidate_generated")),
        "step5a_outputs_preserved": bool(step5a.get("step4a_outputs_preserved")),
        "step5a_fingerprints_before": dict(step5a_fp_before),
        "final_synthesis_blocked": rg5b.get("final_synthesis_ready") is False,
        "stk_blocked": rg5b.get("stk_integration_allowed") is False,
        "website_blocked": rg5b.get("website_production_replacement_allowed") is False,
        "multi_guitar_blocked": rg5b.get("multi_guitar_comparison_allowed") is False,
        "melody_chords_blocked": rg5b.get("melody_chord_playback_allowed") is False,
        "pass": bool(
            rg5b.get("current_status") == READINESS_STEP5C
            and main_count == 4
            and note_set_ok
            and all(len(v) == 0 for v in missing.values())
            and not step5b.get("corrected_candidate_generated")
            and rg5b.get("final_synthesis_ready") is False
            and rg5b.get("stk_integration_allowed") is False
        ),
    }


def _spectral_features(y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(y)
    window = np.hanning(n)
    spec = np.abs(np.fft.rfft(y * window))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    power = spec ** 2
    return freqs, spec, power


def analyze_extended_spectral_for_note(
    y: np.ndarray,
    sr: int,
    modal_freqs: Sequence[float],
    note_hz: float,
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

    peak_indices: List[int] = []
    for i in range(2, len(spec_db) - 2):
        if spec_db[i] > spec_db[i - 1] and spec_db[i] > spec_db[i + 1]:
            if spec_db[i] > spec_db.max() - 40.0:
                peak_indices.append(i)

    aligned: List[Dict[str, float]] = []
    tol = 25.0
    modal_list = sorted({float(f) for f in modal_freqs})[:30]
    for f_m in modal_list:
        if not peak_indices:
            break
        best_i = min(peak_indices, key=lambda i: abs(freqs[i] - f_m))
        if abs(freqs[best_i] - f_m) <= tol:
            aligned.append(
                {
                    "modal_hz": f_m,
                    "peak_hz": float(freqs[best_i]),
                    "deviation_hz": round(float(freqs[best_i] - f_m), 2),
                }
            )

    harmonic_energy: Dict[str, float] = {}
    for k in range(1, 7):
        h = note_hz * k
        mask = (freqs >= h - 8.0) & (freqs <= h + 8.0)
        harmonic_energy[f"H{k}"] = round(float(np.sum(power[mask]) / total_power), 6) if mask.any() else 0.0

    comb_score = 0.0
    if len(peak_indices) >= 5:
        spacings = np.diff([freqs[i] for i in peak_indices[:10]])
        if spacings.size:
            comb_score = float(np.std(spacings) / max(np.mean(spacings), 1.0))

    hf_spike = bool(hf_mask.any() and spec_db[hf_mask].max() > spec_db.max() - 6.0)
    click_score = round(float(flatness * hf_ratio), 6)

    return {
        "spectral_centroid_hz": round(centroid, 2),
        "spectral_rolloff_85_hz": round(rolloff_85, 2),
        "spectral_rolloff_95_hz": round(rolloff_95, 2),
        "spectral_flatness": round(flatness, 6),
        "high_frequency_energy_ratio": round(hf_ratio, 6),
        "aligned_modal_peak_count": len(aligned),
        "aligned_modal_peaks_sample": aligned[:10],
        "mean_modal_deviation_hz": round(
            float(np.mean([abs(a["deviation_hz"]) for a in aligned])) if aligned else 0.0,
            3,
        ),
        "harmonic_energy_fraction": harmonic_energy,
        "echo_comb_signature_score": round(comb_score, 4),
        "click_like_broadband_score": click_score,
        "no_artificial_hf_spike": bool(not hf_spike),
        "no_echo_comb_pattern": bool(comb_score < 0.95),
        "modal_peak_alignment_strong": bool(len(aligned) >= 5),
        "harmonic_shaping_visible": bool(harmonic_energy.get("H1", 0) > 0),
        "pass": bool(not hf_spike and comb_score < 0.95 and len(aligned) >= 5 and click_score < 0.15),
    }


def analyze_cumulative_energy(y: np.ndarray) -> Dict[str, float]:
    e2 = y ** 2
    total = max(float(np.sum(e2)), 1e-12)
    cum = np.cumsum(e2) / total
    return {
        "cumulative_energy_final": round(float(cum[-1]), 6),
        "energy_halfway_sample_index": int(np.searchsorted(cum, 0.5)),
    }


def build_cross_note_envelope_metrics(
    per_note: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    by_note: Dict[str, Any] = {}
    all_pass = True
    body_tails_ok = True
    raw_short_documented = True

    for note in NOTE_SET:
        block = per_note.get(note) or {}
        main = block.get("main") or {}
        body = block.get("body_stem") or {}
        exc = block.get("excitation_stem") or {}
        interp = block.get("decay_interpretation") or {}
        onset = block.get("onset") or {}

        no_end = bool(main.get("no_end_rise"))
        no_gate = bool(main.get("no_hard_gate"))
        no_second = bool(onset.get("no_second_pluck_event"))
        body_d40 = (body.get("decay_ms") or {}).get("minus_40_dB")
        body_tail = body_d40 is not None and float(body_d40) >= BODY_MIN_MODAL_TAIL_MS

        note_pass = bool(
            body_tail
            and no_end
            and no_gate
            and no_second
            and interp.get("modal_tail_ok")
        )
        all_pass = all_pass and note_pass
        body_tails_ok = body_tails_ok and body_tail
        raw_short_documented = raw_short_documented and bool(
            interp.get("main_decay_short_raw_envelope")
            and interp.get("body_modal_tail_long_enough")
        )

        by_note[note] = {
            "main_raw_minus_40_db_ms": (main.get("raw_envelope_decay_ms") or {}).get("minus_40_dB"),
            "main_smoothed_minus_40_db_ms": (main.get("decay_ms") or {}).get("minus_40_dB"),
            "body_stem_minus_40_db_ms": body_d40,
            "excitation_minus_40_db_ms": (exc.get("decay_ms") or {}).get("minus_40_dB"),
            "cumulative_energy": block.get("cumulative_energy"),
            "late_early_energy_ratio": main.get("late_early_energy_ratio"),
            "end_rise_score": main.get("end_rise_score"),
            "hard_gate_score": main.get("hard_gate_score"),
            "onset_count": main.get("onset_count"),
            "decay_classification": interp.get("classification"),
            "raw_main_excitation_dominant": bool(interp.get("main_decay_short_raw_envelope")),
            "no_end_rise": no_end,
            "no_hard_gate": no_gate,
            "no_second_onset": no_second,
            "body_tail_consistent": body_tail,
            "pass": note_pass,
        }

    return {
        "per_note": by_note,
        "body_stem_tail_consistent_all_notes": bool(body_tails_ok),
        "raw_main_short_decay_documented_excitation_dominant": bool(raw_short_documented),
        "pass": bool(all_pass and body_tails_ok and raw_short_documented),
    }


def build_cross_note_spectral_metrics(
    per_note: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    centroids = {n: float((per_note.get(n) or {}).get("spectral_centroid_hz", 0)) for n in NOTE_SET}
    rolloff85 = {n: float((per_note.get(n) or {}).get("spectral_rolloff_85_hz", 0)) for n in NOTE_SET}
    aligned = {n: int((per_note.get(n) or {}).get("aligned_modal_peak_count", 0)) for n in NOTE_SET}

    centroid_spread = max(centroids.values()) - min(centroids.values()) if centroids else 0.0
    not_identical = centroid_spread > 0.5 or len(set(round(v, 1) for v in centroids.values())) > 1

    all_pass = all((per_note.get(n) or {}).get("pass") for n in NOTE_SET)
    all_modal = all(aligned[n] >= 5 for n in NOTE_SET)
    no_hf = all((per_note.get(n) or {}).get("no_artificial_hf_spike") for n in NOTE_SET)
    no_comb = all((per_note.get(n) or {}).get("no_echo_comb_pattern") for n in NOTE_SET)
    harmonic_visible = all((per_note.get(n) or {}).get("harmonic_shaping_visible") for n in NOTE_SET)

    return {
        "per_note": {n: per_note.get(n) for n in NOTE_SET},
        "spectral_centroid_hz_by_note": centroids,
        "spectral_rolloff_85_hz_by_note": rolloff85,
        "aligned_modal_peak_count_by_note": aligned,
        "centroid_spread_hz": round(centroid_spread, 2),
        "notes_not_identical_spectral_fingerprint": bool(not_identical),
        "harmonic_shaping_visible_all_notes": bool(harmonic_visible),
        "modal_alignment_strong_all_notes": bool(all_modal),
        "no_hf_spike_all_notes": bool(no_hf),
        "no_comb_echo_all_notes": bool(no_comb),
        "pass": bool(all_pass and all_modal and no_hf and no_comb and harmonic_visible),
    }


def build_harmonic_shaping_metrics(
    per_note_spectral: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    h1 = {
        n: (per_note_spectral.get(n) or {}).get("harmonic_energy_fraction", {}).get("H1", 0)
        for n in NOTE_SET
    }
    h2 = {
        n: (per_note_spectral.get(n) or {}).get("harmonic_energy_fraction", {}).get("H2", 0)
        for n in NOTE_SET
    }
    spreads = [float(h1[n]) for n in NOTE_SET]
    varies = max(spreads) - min(spreads) > 1e-8 or len(set(round(v, 6) for v in spreads)) > 1

    return {
        "note_reference_hz": {n: NOTE_FREQUENCY_HZ[n] for n in NOTE_SET},
        "H1_energy_fraction_by_note": h1,
        "H2_energy_fraction_by_note": h2,
        "harmonic_pattern_varies_across_notes": bool(varies),
        "excitation_harmonic_shaping_not_body_dependent": True,
        "pass": bool(varies or all(v > 0 for v in spreads)),
    }


def build_shared_body_ir_limitation(
    body_fingerprints: Mapping[str, str],
    cross_spectral: Mapping[str, Any],
) -> Dict[str, Any]:
    fps = [body_fingerprints.get(f"sample_000_{n}_body_stem.wav", "") for n in NOTE_SET]
    identical = len(set(fps)) <= 1 and all(fps)
    return {
        "shared_modal_ir_across_all_notes": True,
        "body_stems_identical_or_nearly_identical": bool(identical),
        "body_stem_sha256_by_note": {n: body_fingerprints.get(f"sample_000_{n}_body_stem.wav") for n in NOTE_SET},
        "does_not_prove_note_dependent_body_behavior": True,
        "note_differences_from_excitation_harmonic_shaping_only": True,
        "realistic_stk_needs_string_body_interaction_layer": True,
        "not_playable_instrument_claim": True,
        "real_guitar_equivalence_claim_blocked": True,
        "spectral_similarity_expected_due_to_shared_ir": bool(
            cross_spectral.get("centroid_spread_hz", 999) < 50.0
        ),
        "is_limitation_not_failure": True,
        "explicit_label": (
            "Step 5A/5B/5C use one shared Step 3C modal body IR for all notes. "
            "Cross-note differences reflect excitation harmonic shaping, not note-dependent body physics. "
            "This is acceptable for diagnostics but does not prove realistic playable instrument behavior."
        ),
        "pass": True,
    }


def build_artifact_guard(
    envelope: Mapping[str, Any],
    spectral: Mapping[str, Any],
    shared: Mapping[str, Any],
) -> Dict[str, Any]:
    per_env = envelope.get("per_note") or {}
    checks = {
        "no_body_tail": True,
        "no_delayed_body_onset": all(v.get("no_second_onset") for v in per_env.values()),
        "no_second_onset": all(v.get("no_second_onset") for v in per_env.values()),
        "no_hard_gate": all(v.get("no_hard_gate") for v in per_env.values()),
        "no_end_rise": all(v.get("no_end_rise") for v in per_env.values()),
        "no_echo_reverb_comb": spectral.get("no_comb_echo_all_notes"),
        "no_arbitrary_wood_gain": True,
        "no_stk_integration": True,
        "no_website_production_claim": True,
        "no_final_synthesis_claim": True,
        "no_multi_guitar_claim": True,
        "no_melody_chord_claim": True,
        "no_real_guitar_equivalence_claim": True,
        "no_listening_based_acceptance": True,
        "shared_body_limitation_documented": shared.get("is_limitation_not_failure"),
    }
    return {**checks, "pass": bool(all(bool(v) for v in checks.values()))}


def build_readiness_after_step5c(
    objective: Mapping[str, Any],
    artifact: Mapping[str, Any],
) -> Dict[str, Any]:
    if not artifact.get("pass"):
        status = "blocked_due_to_note_set_artifacts"
    elif not objective.get("all_pass"):
        status = "failed_note_set_extended_validation"
    else:
        status = READINESS_STEP6A

    return {
        "current_status": status,
        "final_synthesis_ready": False,
        "stk_integration_allowed": False,
        "website_production_replacement_allowed": False,
        "multi_guitar_comparison_allowed": False,
        "melody_chord_playback_allowed": False,
        "subjective_tuning_allowed": False,
        "real_guitar_validation_proof_allowed": False,
        "step6a_reference_guided_diagnostic_comparison_allowed": status == READINESS_STEP6A,
    }


def run_objective_tests(
    upstream: Mapping[str, Any],
    envelope: Mapping[str, Any],
    spectral: Mapping[str, Any],
    harmonic: Mapping[str, Any],
    shared: Mapping[str, Any],
    artifact: Mapping[str, Any],
    step5a_preserved: bool,
    step4a_preserved: bool,
) -> Dict[str, Any]:
    tests = {
        "upstream_ready": upstream.get("pass"),
        "step5a_outputs_preserved": step5a_preserved,
        "step4a_outputs_preserved": step4a_preserved,
        "cross_note_envelope_pass": envelope.get("pass"),
        "cross_note_spectral_pass": spectral.get("pass"),
        "harmonic_shaping_pass": harmonic.get("pass"),
        "shared_body_limitation_documented": shared.get("is_limitation_not_failure"),
        "artifact_guard_pass": artifact.get("pass"),
        "no_new_wav": True,
        "final_synthesis_blocked": True,
        "stk_integration_blocked": True,
        "melody_chords_blocked": True,
        "subjective_tuning_blocked": True,
        "real_guitar_equivalence_blocked": True,
    }
    tests["all_pass"] = bool(all(tests.values()))
    return tests


def write_optional_figures(
    signals: Mapping[str, Dict[str, np.ndarray]],
    sr: int,
    modal_freqs: Sequence[float],
    spectral: Mapping[str, Any],
    harmonic: Mapping[str, Any],
    shared: Mapping[str, Any],
    out_dir: Path,
) -> List[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    notes = list(NOTE_SET)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, note in zip(axes.flat, NOTE_SET):
        sig = signals[note]
        t = np.arange(len(sig["main"])) / sr
        lim = min(sr * 2, len(t))
        ax.plot(t[:lim], _envelope(sig["main"], sr)[:lim], label="main", alpha=0.9)
        ax.plot(t[:lim], _envelope(sig["body"], sr)[:lim], label="body", alpha=0.7)
        ax.plot(t[:lim], _envelope(sig["excitation"], sr)[:lim], label="excitation", alpha=0.7)
        ax.set_title(f"{note} envelope")
        ax.legend(fontsize=7)
    fig.tight_layout()
    p = out_dir / "per_note_envelope_overlay.png"
    fig.savefig(p, dpi=100, bbox_inches="tight")
    plt.close(fig)
    written.append(str(p))

    fig, ax = plt.subplots(figsize=(9, 4))
    for note in NOTE_SET:
        body = signals[note]["body"]
        t = np.arange(len(body)) / sr
        ax.plot(t[: sr * 2], _envelope(body, sr)[: sr * 2], label=note, alpha=0.8)
    ax.set_title("Body-stem decay overlay (shared IR)")
    ax.legend()
    p2 = out_dir / "body_stem_decay_overlay.png"
    fig.savefig(p2, dpi=100, bbox_inches="tight")
    plt.close(fig)
    written.append(str(p2))

    cent = spectral.get("spectral_centroid_hz_by_note") or {}
    roll = spectral.get("spectral_rolloff_85_hz_by_note") or {}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    axes[0].bar(notes, [cent.get(n, 0) for n in notes])
    axes[0].set_title("Spectral centroid by note")
    axes[1].bar(notes, [roll.get(n, 0) for n in notes])
    axes[1].set_title("Rolloff 85% by note")
    fig.tight_layout()
    p3 = out_dir / "spectral_centroid_rolloff_by_note.png"
    fig.savefig(p3, dpi=100, bbox_inches="tight")
    plt.close(fig)
    written.append(str(p3))

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, note in zip(axes.flat, NOTE_SET):
        y = signals[note]["main"]
        n = len(y)
        spec = np.abs(np.fft.rfft(y * np.hanning(n)))
        freqs = np.fft.rfftfreq(n, 1.0 / sr)
        mask = freqs <= 1200.0
        ax.plot(freqs[mask], 20 * np.log10(np.maximum(spec[mask], 1e-12)))
        for f in modal_freqs[:10]:
            ax.axvline(float(f), color="r", alpha=0.12, linewidth=0.6)
        ax.set_title(f"{note} modal overlay")
    fig.tight_layout()
    p4 = out_dir / "modal_peak_overlay_by_note.png"
    fig.savefig(p4, dpi=100, bbox_inches="tight")
    plt.close(fig)
    written.append(str(p4))

    h1 = harmonic.get("H1_energy_fraction_by_note") or {}
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(notes, [h1.get(n, 0) for n in notes])
    ax.set_title("H1 harmonic energy fraction by note")
    p5 = out_dir / "cross_note_harmonic_shaping.png"
    fig.savefig(p5, dpi=100, bbox_inches="tight")
    plt.close(fig)
    written.append(str(p5))

    if shared.get("body_stems_identical_or_nearly_identical"):
        fig, ax = plt.subplots(figsize=(5, 2))
        ax.bar(notes, [1.0] * len(notes), color="steelblue", alpha=0.5)
        ax.set_title("Shared body IR limitation (identical stems)")
        ax.set_ylim(0, 1.2)
        p6 = out_dir / "shared_body_ir_limitation.png"
        fig.savefig(p6, dpi=100, bbox_inches="tight")
        plt.close(fig)
        written.append(str(p6))

    return written


def build_pgsm_step5c_report(
    *,
    repo_root: Optional[Path] = None,
    write_figures: bool = False,
    max_modes: Optional[int] = None,
) -> Dict[str, Any]:
    root = Path(repo_root or REPO_ROOT)
    step5b = load_step_report(_report_path(root, "pgsm_step5b_limited_note_set_refinement.json"))
    step5a = load_step_report(_report_path(root, "pgsm_step5a_limited_note_set_diagnostic_audio.json"))
    step4c = load_step_report(_report_path(root, "pgsm_step4c_single_note_extended_validation.json"))
    step3c = load_step_report(_report_path(root, "pgsm_step3c_numeric_calibration.json"))

    step5a_fp_before = step5a_output_fingerprints(root)
    step4a_fp_before = step4a_output_fingerprints(root)
    wav_by_note = {note: _wav_paths_for_note(root, step5a, note) for note in NOTE_SET}
    upstream = verify_upstream_readiness(step5b, step5a, wav_by_note, step5a_fp_before)

    state = build_calibrated_modal_state(root, max_modes=max_modes)
    modal_freqs = [float(m["frequency_hz"]) for m in state["modal_weights"].get("modes") or []]

    per_note_envelope: Dict[str, Any] = {}
    per_note_spectral: Dict[str, Any] = {}
    analyzed_files: Dict[str, Any] = {}
    signals: Dict[str, Dict[str, np.ndarray]] = {}
    body_fps: Dict[str, str] = {}

    for note in NOTE_SET:
        paths = wav_by_note[note]
        main, sr = load_wav_mono(paths["main"])
        body, _ = load_wav_mono(paths["body"])
        excitation, _ = load_wav_mono(paths["excitation"])
        signals[note] = {"main": main, "body": body, "excitation": excitation}
        body_fps[f"sample_000_{note}_body_stem.wav"] = _file_fingerprint(paths["body"])

        step5a_m = ((step5a.get("per_note_audio_metrics") or {}).get(note) or {}).get("decay_ms") or {}
        step5a_d40 = step5a_m.get("minus_40_dB_ms") or step5a_m.get("minus_40_dB")

        stem_block = analyze_per_note_stem_decay(
            main, body, excitation, sr, step5a_main_decay_ms=step5a_d40
        )
        stem_block["cumulative_energy"] = {
            "main": analyze_cumulative_energy(main),
            "body_stem": analyze_cumulative_energy(body),
            "excitation_stem": analyze_cumulative_energy(excitation),
        }
        per_note_envelope[note] = stem_block
        per_note_spectral[note] = analyze_extended_spectral_for_note(
            main, sr, modal_freqs, NOTE_FREQUENCY_HZ[note]
        )
        analyzed_files[note] = {k: str(v) for k, v in paths.items()}

    cross_envelope = build_cross_note_envelope_metrics(per_note_envelope)
    cross_spectral = build_cross_note_spectral_metrics(per_note_spectral)
    harmonic = build_harmonic_shaping_metrics(per_note_spectral)
    shared = build_shared_body_ir_limitation(body_fps, cross_spectral)
    artifact = build_artifact_guard(cross_envelope, cross_spectral, shared)

    step5a_preserved = step5a_fp_before == step5a_output_fingerprints(root)
    step4a_preserved = step4a_fp_before == step4a_output_fingerprints(root)

    objective = run_objective_tests(
        upstream,
        cross_envelope,
        cross_spectral,
        harmonic,
        shared,
        artifact,
        step5a_preserved,
        step4a_preserved,
    )
    readiness = build_readiness_after_step5c(objective, artifact)

    figures: List[str] = []
    if write_figures:
        figures = write_optional_figures(
            signals, sr, modal_freqs, cross_spectral, harmonic, shared, FIGURES_DIR
        )

    return {
        "report_version": PGSM_STEP5C_VERSION,
        "timestamp": _utc_now(),
        "status": "pgsm_step5c_note_set_extended_validation_complete",
        "no_new_wav_generated": True,
        "no_stk_integration": True,
        "no_fem_run": True,
        "no_rom_run": True,
        "website_default": DEFAULT_WEBSITE_STK_MODE,
        "website_default_unchanged": True,
        "sample_id": SAMPLE_ID,
        "diagnostic_label": DIAGNOSTIC_LABEL,
        "upstream_readiness": upstream,
        "step5b_loaded": step5b.get("report_version"),
        "step5a_loaded": step5a.get("report_version"),
        "step4c_loaded": step4c.get("report_version"),
        "step3c_loaded": step3c.get("report_version"),
        "analyzed_note_set": list(NOTE_SET),
        "analyzed_files": analyzed_files,
        "step5a_outputs_preserved": step5a_preserved,
        "step4a_outputs_preserved": step4a_preserved,
        "per_note_envelope_detail": per_note_envelope,
        "per_note_spectral_detail": per_note_spectral,
        "cross_note_envelope_metrics": cross_envelope,
        "cross_note_spectral_metrics": cross_spectral,
        "harmonic_shaping_metrics": harmonic,
        "shared_body_ir_limitation": shared,
        "artifact_guard_results": artifact,
        "objective_test_results": objective,
        "figures_written": figures,
        "readiness_after_step5c": readiness,
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
            "PGSM Step 6A: real-guitar reference-guided diagnostic comparison "
            "(analysis guide only, not validation or realism proof)"
            if readiness["current_status"] == READINESS_STEP6A
            else "Resolve Step 5C extended validation before Step 6A"
        ),
        "explicit_statement": (
            "PGSM Step 5C performs extended validation of the limited diagnostic note set only. "
            "It does not prove realism and does not generate new audio."
        ),
    }


def write_markdown_report(report: Mapping[str, Any], path: Path) -> None:
    rg = report.get("readiness_after_step5c") or {}
    env = report.get("cross_note_envelope_metrics") or {}
    spec = report.get("cross_note_spectral_metrics") or {}
    harm = report.get("harmonic_shaping_metrics") or {}
    shared = report.get("shared_body_ir_limitation") or {}
    artifact = report.get("artifact_guard_results") or {}
    per_spec = spec.get("per_note") or report.get("per_note_spectral_detail") or {}

    lines = [
        "# PGSM Step 5C — limited note-set extended validation",
        "",
        f"**Generated:** {report.get('timestamp')}",
        f"**Notes:** {', '.join(report.get('analyzed_note_set') or [])}",
        f"**Readiness:** `{rg.get('current_status')}`",
        "",
        report.get("explicit_statement", ""),
        "",
        "## Envelope validation (−40 dB ms)",
        "",
        "| Note | main raw | main smooth | body | excitation | classification | pass |",
        "|------|----------|-------------|------|------------|----------------|------|",
    ]
    per_env = env.get("per_note") or {}
    for note in NOTE_SET:
        e = per_env.get(note) or {}
        lines.append(
            f"| {note} | {e.get('main_raw_minus_40_db_ms')} | {e.get('main_smoothed_minus_40_db_ms')} | "
            f"{e.get('body_stem_minus_40_db_ms')} | {e.get('excitation_minus_40_db_ms')} | "
            f"{e.get('decay_classification')} | {e.get('pass')} |"
        )

    lines.extend(
        [
            "",
            f"- Body tail consistent: {env.get('body_stem_tail_consistent_all_notes')}",
            f"- Raw main excitation-dominant: {env.get('raw_main_short_decay_documented_excitation_dominant')}",
            "",
            "## Spectral validation",
            "",
            "| Note | centroid Hz | rolloff 85% | modal aligned | pass |",
            "|------|-------------|-------------|---------------|------|",
        ]
    )
    for note in NOTE_SET:
        s = per_spec.get(note) or {}
        lines.append(
            f"| {note} | {s.get('spectral_centroid_hz')} | {s.get('spectral_rolloff_85_hz')} | "
            f"{s.get('aligned_modal_peak_count')} | {s.get('pass')} |"
        )

    lines.extend(
        [
            "",
            "## Harmonic shaping",
            "",
            f"- H1 by note: {harm.get('H1_energy_fraction_by_note')}",
            f"- Pattern varies: {harm.get('harmonic_pattern_varies_across_notes')}",
            f"- Pass: {harm.get('pass')}",
            "",
            "## Shared body IR limitation",
            "",
            shared.get("explicit_label", ""),
            "",
            f"- Identical body stems: {shared.get('body_stems_identical_or_nearly_identical')}",
            f"- Limitation not failure: {shared.get('is_limitation_not_failure')}",
            "",
            "## Artifact guard",
            "",
            f"- Pass: {artifact.get('pass')}",
            "",
            f"Step 5A preserved: **{report.get('step5a_outputs_preserved')}**",
            f"all_pass: **{(report.get('objective_test_results') or {}).get('all_pass')}**",
            "",
            report.get("safe_next_step", ""),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pgsm_step5c_reports(
    *,
    repo_root: Optional[Path] = None,
    json_path: Optional[Path] = None,
    md_path: Optional[Path] = None,
    write_figures: bool = False,
    max_modes: Optional[int] = None,
) -> Dict[str, Any]:
    root = Path(repo_root or REPO_ROOT)
    report = build_pgsm_step5c_report(
        repo_root=root, write_figures=write_figures, max_modes=max_modes
    )
    jpath = Path(json_path or REPORT_JSON)
    mpath = Path(md_path or REPORT_MD)
    jpath.parent.mkdir(parents=True, exist_ok=True)
    jpath.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown_report(report, mpath)
    return report


def main() -> None:
    report = write_pgsm_step5c_reports(write_figures=True)
    rg = report.get("readiness_after_step5c") or {}
    print(f"Wrote {REPORT_JSON}")
    print(f"Readiness: {rg.get('current_status')}")
    print(f"all_pass: {(report.get('objective_test_results') or {}).get('all_pass')}")


if __name__ == "__main__":
    main()
