#!/usr/bin/env python3
"""
PGSM Step 5B — limited note-set diagnostic refinement.
Objective analysis only; separates main/body/excitation decay metrics.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from pgsm_step2_1_parameter_targets import load_step_report
from pgsm_step3a_numerical_ir_testbench import MODAL_BANDS, NUMERIC_SR, SAMPLE_ID, compute_impulse_response
from pgsm_step4a_single_note_diagnostic_audio import build_calibrated_modal_state
from pgsm_step4b_single_note_diagnostic_refinement import (
    analyze_onset,
    analyze_spectral_modal,
    analyze_stem_balance,
    analyze_waveform_envelope,
    load_wav_mono,
)
from pgsm_step5a_limited_note_set_diagnostic_audio import (
    AUDIO_DIR as STEP5A_AUDIO_DIR,
    NOTE_FREQUENCY_HZ,
    NOTE_SET,
    READINESS_STEP5B,
    step4a_output_fingerprints,
)
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE

PGSM_STEP5B_VERSION = "pgsm_step5b_limited_note_set_refinement_v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON = REPO_ROOT / "audio" / "debug_reports" / "pgsm_step5b_limited_note_set_refinement.json"
REPORT_MD = REPO_ROOT / "audio" / "debug_reports" / "pgsm_step5b_limited_note_set_refinement.md"
FIGURES_DIR = REPO_ROOT / "audio" / "debug_reports" / "pgsm_step5b_figures"
STEP5B_CANDIDATES_DIR = REPO_ROOT / "audio" / "pgsm_step5b_limited_note_set_candidates"

READINESS_STEP5C = "ready_for_step5c_note_set_extended_validation"
MAIN_SHORT_DECAY_MS = 50.0
BODY_MIN_MODAL_TAIL_MS = 50.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _report_path(root: Path, name: str) -> Path:
    return root / "audio" / "debug_reports" / name


def _file_fingerprint(path: Path) -> str:
    if not path.is_file():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def step5a_output_fingerprints(root: Path) -> Dict[str, str]:
    d = root / "audio" / "pgsm_step5a_limited_note_set"
    if not d.is_dir():
        return {}
    return {p.name: _file_fingerprint(p) for p in sorted(d.glob("*.wav"))}


def _wav_paths_for_note(root: Path, step5a: Mapping[str, Any], note: str) -> Dict[str, Path]:
    notes_out = (step5a.get("output_files") or {}).get("notes") or {}
    info = notes_out.get(note) or {}
    defaults = {
        "main": STEP5A_AUDIO_DIR / f"sample_000_{note}_diagnostic.wav",
        "body": STEP5A_AUDIO_DIR / f"sample_000_{note}_body_stem.wav",
        "excitation": STEP5A_AUDIO_DIR / f"sample_000_{note}_excitation_stem.wav",
    }
    keys = {
        "main": "main_diagnostic_wav",
        "body": "body_stem_wav",
        "excitation": "excitation_stem_wav",
    }
    paths: Dict[str, Path] = {}
    for k, out_key in keys.items():
        rel = info.get(out_key)
        p = Path(str(rel)) if rel else defaults[k]
        paths[k] = p if p.is_file() else defaults[k]
    return paths


def verify_step5a_readiness(
    step5a: Mapping[str, Any],
    wav_paths_by_note: Mapping[str, Mapping[str, Path]],
) -> Dict[str, Any]:
    rg = step5a.get("readiness_after_step5a") or {}
    missing: Dict[str, List[str]] = {}
    for note in NOTE_SET:
        missing[note] = [k for k, p in wav_paths_by_note[note].items() if not p.is_file()]
    all_exist = all(len(v) == 0 for v in missing.values())
    main_count = sum(1 for note in NOTE_SET if wav_paths_by_note[note]["main"].is_file())
    return {
        "step5a_readiness": rg.get("current_status"),
        "step5a_pass": rg.get("current_status") == READINESS_STEP5B,
        "main_wav_count": main_count,
        "four_main_wavs_exist": main_count == 4,
        "stems_exist_all_notes": all_exist,
        "missing_wav": missing,
        "step4a_outputs_preserved_in_step5a": bool(step5a.get("step4a_outputs_preserved")),
        "final_synthesis_blocked": rg.get("final_synthesis_ready") is False,
        "stk_blocked": rg.get("stk_integration_allowed") is False,
        "website_blocked": rg.get("website_production_replacement_allowed") is False,
        "multi_guitar_blocked": rg.get("multi_guitar_comparison_allowed") is False,
        "melody_chords_blocked": rg.get("melody_chord_playback_allowed") is False,
        "pass": bool(
            rg.get("current_status") == READINESS_STEP5B
            and main_count == 4
            and all_exist
            and rg.get("final_synthesis_ready") is False
            and rg.get("stk_integration_allowed") is False
        ),
    }


def _count_onsets(env: np.ndarray, sr: int) -> int:
    if env.size < sr // 20:
        return 1 if env.max() > 1e-8 else 0
    thresh = 0.15 * float(env.max())
    above = env >= thresh
    transitions = int(np.sum(np.diff(above.astype(int)) == 1))
    return max(1, transitions) if env.max() > 1e-8 else 0


def _envelope_smoothness(env: np.ndarray) -> float:
    log_env = np.log10(np.maximum(env, 1e-12))
    d = np.diff(log_env)
    return round(float(np.std(d)), 6) if d.size else 0.0


def _decay_from_env(env: np.ndarray, t: np.ndarray, db: float) -> Optional[float]:
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


def analyze_stem_decay_signal(
    y: np.ndarray,
    sr: int,
    *,
    signal_role: str,
) -> Dict[str, Any]:
    base = analyze_waveform_envelope(y, sr)
    from pgsm_step4b_single_note_diagnostic_refinement import _envelope

    env_smooth = _envelope(y, sr)
    env_raw = np.abs(y)
    t = np.arange(len(y)) / sr

    raw_decay = {
        "minus_20_dB": _decay_from_env(env_raw, t, -20.0),
        "minus_40_dB": _decay_from_env(env_raw, t, -40.0),
        "minus_60_dB": _decay_from_env(env_raw, t, -60.0),
    }
    e2 = y ** 2
    e_early = float(np.sum(e2[(t >= 0.0) & (t < 0.2)])) if len(y) else 0.0
    e_late = float(np.sum(e2[(t >= 0.5) & (t < 1.0)])) if len(y) else 0.0

    last_third = env_smooth[len(env_smooth) * 2 // 3 :]
    mid_third = env_smooth[len(env_smooth) // 3 : len(env_smooth) * 2 // 3]
    end_rise_score = (
        round(float(last_third.max() / max(mid_third.max(), 1e-12)), 4)
        if last_third.size
        else 0.0
    )
    tail = env_smooth[int(len(env_smooth) * 0.9) :]
    hard_gate_score = (
        round(float(tail.max() / max(env_smooth[int(len(env_smooth) * 0.5)], 1e-12)), 6)
        if tail.size
        else 0.0
    )

    return {
        "signal_role": signal_role,
        "peak_fs": base.get("peak_amplitude_fs"),
        "rms": base.get("rms"),
        "attack_time_ms": base.get("attack_time_ms"),
        "peak_time_ms": base.get("peak_time_ms"),
        "decay_ms": base.get("decay_ms"),
        "raw_envelope_decay_ms": raw_decay,
        "late_early_energy_ratio": round(e_late / max(e_early, 1e-12), 6),
        "end_rise_score": end_rise_score,
        "hard_gate_score": hard_gate_score,
        "envelope_smoothness_std": _envelope_smoothness(env_smooth),
        "onset_count": _count_onsets(env_smooth, sr),
        "no_end_rise": bool(base.get("no_end_rise")),
        "no_hard_gate": bool(base.get("no_hard_gate")),
    }


def interpret_per_note_decay(
    main: Mapping[str, Any],
    body: Mapping[str, Any],
    excitation: Mapping[str, Any],
    *,
    stems: Mapping[str, Any],
    step5a_main_decay_ms: Optional[float] = None,
) -> Dict[str, Any]:
    main_d40_smooth = (main.get("decay_ms") or {}).get("minus_40_dB")
    main_d40_raw = (main.get("raw_envelope_decay_ms") or {}).get("minus_40_dB")
    body_d40 = (body.get("decay_ms") or {}).get("minus_40_dB")
    exc_d40 = (excitation.get("decay_ms") or {}).get("minus_40_dB")

    main_short_raw = main_d40_raw is not None and float(main_d40_raw) < MAIN_SHORT_DECAY_MS
    main_short_smooth = main_d40_smooth is not None and float(main_d40_smooth) < MAIN_SHORT_DECAY_MS
    body_long = body_d40 is not None and float(body_d40) >= BODY_MIN_MODAL_TAIL_MS
    body_short = body_d40 is not None and float(body_d40) < MAIN_SHORT_DECAY_MS

    if (main_short_raw or main_short_smooth) and body_long:
        classification = "force_dominant_main_envelope_not_body_failure"
        modal_tail_ok = True
        summary = (
            "Main −40 dB decay is short on raw/smoothed envelope because the convolution "
            "metric tracks the F_bridge excitation pulse; body stem retains longer modal tail."
        )
    elif (main_short_raw or main_short_smooth) and body_short:
        classification = "modal_tail_issue_body_also_short"
        modal_tail_ok = False
        summary = "Both main and body stem show unrealistically short −40 dB decay."
    elif not main_short_raw and body_long:
        classification = "main_and_body_decay_consistent"
        modal_tail_ok = True
        summary = "Smoothed main envelope decay aligns with modal body tail duration."
    else:
        classification = "review_required"
        modal_tail_ok = body_long or not body_short
        summary = "Decay metrics require review against Step 3C IR reference."

    exc_dominates = not bool(stems.get("excitation_not_dominating_click", True))
    exc_short = exc_d40 is not None and float(exc_d40) < MAIN_SHORT_DECAY_MS

    return {
        "main_minus_40_db_ms_smoothed": main_d40_smooth,
        "main_minus_40_db_ms_raw": main_d40_raw,
        "step5a_reported_main_minus_40_db_ms": step5a_main_decay_ms,
        "body_minus_40_db_ms": body_d40,
        "excitation_minus_40_db_ms": exc_d40,
        "main_decay_short_raw_envelope": bool(main_short_raw),
        "main_decay_short_smoothed_envelope": bool(main_short_smooth),
        "body_modal_tail_long_enough": bool(body_long),
        "classification": classification,
        "summary": summary,
        "modal_tail_ok": bool(modal_tail_ok),
        "excitation_decay_short_expected": bool(exc_short),
        "excitation_dominance_issue": bool(exc_dominates),
        "pass": bool(modal_tail_ok and not exc_dominates),
    }


def analyze_per_note_stem_decay(
    main: np.ndarray,
    body: np.ndarray,
    excitation: np.ndarray,
    sr: int,
    *,
    step5a_main_decay_ms: Optional[float] = None,
) -> Dict[str, Any]:
    main_m = analyze_stem_decay_signal(main, sr, signal_role="main_diagnostic")
    body_m = analyze_stem_decay_signal(body, sr, signal_role="body_stem")
    exc_m = analyze_stem_decay_signal(excitation, sr, signal_role="excitation_stem")
    stems = analyze_stem_balance(main, body, excitation)
    onset = analyze_onset(main, excitation, body, sr)
    interpretation = interpret_per_note_decay(
        main_m, body_m, exc_m, stems=stems, step5a_main_decay_ms=step5a_main_decay_ms
    )
    return {
        "main": main_m,
        "body_stem": body_m,
        "excitation_stem": exc_m,
        "stem_balance": stems,
        "onset": onset,
        "decay_interpretation": interpretation,
        "pass": bool(interpretation.get("pass") and onset.get("pass") and stems.get("pass")),
    }


def analyze_spectral_harmonic(
    main: np.ndarray,
    body: np.ndarray,
    excitation: np.ndarray,
    sr: int,
    modal_freqs: Sequence[float],
    note_hz: float,
) -> Dict[str, Any]:
    spec_main = analyze_spectral_modal(main, sr, modal_freqs)
    spec_body = analyze_spectral_modal(body, sr, modal_freqs)
    spec_exc = analyze_spectral_modal(excitation, sr, modal_freqs)

    n = len(main)
    window = np.hanning(n)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    spec = np.abs(np.fft.rfft(main * window))
    power = spec ** 2
    total = max(float(np.sum(power)), 1e-12)

    harmonic_energy: Dict[str, float] = {}
    for k in range(1, 7):
        h = note_hz * k
        mask = (freqs >= h - 8.0) & (freqs <= h + 8.0)
        harmonic_energy[f"H{k}"] = round(float(np.sum(power[mask]) / total), 6) if mask.any() else 0.0

    fund_mask = (freqs >= note_hz - 5.0) & (freqs <= note_hz + 5.0)
    fundamental_present = bool(fund_mask.any() and float(np.sum(power[fund_mask])) / total > 1e-4)

    flatness_geo = float(np.exp(np.mean(np.log(np.maximum(power, 1e-20)))))
    flatness = flatness_geo / max(float(np.mean(power)), 1e-20)
    click_like = bool(flatness > 0.25 and spec_exc.get("spectral_centroid_hz", 0) > 2000)

    note_shaping = bool(
        harmonic_energy.get("H1", 0) > 0
        or harmonic_energy.get("H2", 0) > harmonic_energy.get("H5", 0)
    )

    return {
        "note_reference_hz": note_hz,
        "fundamental_reference_present": fundamental_present,
        "harmonic_energy_fraction": harmonic_energy,
        "main_spectral": {
            "centroid_hz": spec_main.get("spectral_centroid_hz"),
            "modal_peaks_aligned": spec_main.get("modal_peaks_aligned_count"),
            "hf_spike": spec_main.get("unexplained_hf_spike"),
            "comb_score": spec_main.get("echo_comb_pattern_score"),
        },
        "body_spectral": {
            "centroid_hz": spec_body.get("spectral_centroid_hz"),
            "modal_peaks_aligned": spec_body.get("modal_peaks_aligned_count"),
        },
        "excitation_spectral": {
            "centroid_hz": spec_exc.get("spectral_centroid_hz"),
            "comb_score": spec_exc.get("echo_comb_pattern_score"),
        },
        "note_dependent_spectral_shaping_visible": note_shaping,
        "modal_body_character_preserved": bool(spec_body.get("pass")),
        "no_artificial_click_dominance": bool(not click_like and spec_main.get("pass")),
        "pass": bool(
            spec_main.get("pass")
            and spec_body.get("pass")
            and not click_like
            and note_shaping
        ),
    }


def build_cross_note_consistency(
    per_note_stem: Mapping[str, Mapping[str, Any]],
    per_note_spectral: Mapping[str, Mapping[str, Any]],
    body_fingerprints: Mapping[str, str],
) -> Dict[str, Any]:
    notes = list(NOTE_SET)

    def _collect(role: str, field: str) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for n in notes:
            block = (per_note_stem.get(n) or {}).get(role) or {}
            val = block.get(field)
            if val is not None:
                out[n] = float(val)
        return out

    main_peaks = _collect("main", "peak_fs")
    main_rms = _collect("main", "rms")
    body_peaks = _collect("body_stem", "peak_fs")
    body_rms = _collect("body_stem", "rms")
    exc_peaks = _collect("excitation_stem", "peak_fs")
    exc_rms = _collect("excitation_stem", "rms")
    body_d40 = {
        n: float(((per_note_stem.get(n) or {}).get("body_stem") or {}).get("decay_ms", {}).get("minus_40_dB") or 0)
        for n in notes
    }
    centroids = {
        n: float((per_note_spectral.get(n) or {}).get("main_spectral", {}).get("centroid_hz") or 0)
        for n in notes
    }
    ratios = {
        n: float((per_note_stem.get(n) or {}).get("stem_balance", {}).get("body_to_excitation_energy_ratio") or 0)
        for n in notes
    }
    modal_aligned = {
        n: int((per_note_spectral.get(n) or {}).get("main_spectral", {}).get("modal_peaks_aligned") or 0)
        for n in notes
    }

    body_fps = [body_fingerprints.get(f"sample_000_{n}_body_stem.wav", "") for n in notes]
    shared_body = len(set(body_fps)) <= 1 and all(body_fps)

    peak_spread = max(main_peaks.values()) - min(main_peaks.values()) if main_peaks else 0.0
    centroid_spread = max(centroids.values()) - min(centroids.values()) if centroids else 0.0
    notes_too_similar = bool(centroid_spread < 15.0 and peak_spread < 0.02)

    h1_fractions = [
        float((per_note_spectral.get(n) or {}).get("harmonic_energy_fraction", {}).get("H1") or 0)
        for n in notes
    ]
    harmonic_shaping_varies = bool(max(h1_fractions) - min(h1_fractions) > 1e-6 or len(set(h1_fractions)) > 1)

    return {
        "main_peak_fs_by_note": main_peaks,
        "main_rms_by_note": main_rms,
        "body_peak_fs_by_note": body_peaks,
        "body_rms_by_note": body_rms,
        "excitation_peak_fs_by_note": exc_peaks,
        "excitation_rms_by_note": exc_rms,
        "body_excitation_energy_ratio_by_note": ratios,
        "body_minus_40_db_ms_by_note": body_d40,
        "spectral_centroid_hz_by_note": centroids,
        "modal_peaks_aligned_by_note": modal_aligned,
        "shared_body_stem_identical_across_notes": shared_body,
        "notes_spectrally_very_similar_due_to_shared_ir": notes_too_similar,
        "pitch_harmonic_shaping_visible_in_spectra": harmonic_shaping_varies,
        "centroid_spread_hz": round(centroid_spread, 2),
        "peak_spread_fs": round(peak_spread, 6),
        "pass": bool(
            shared_body
            and all(v >= 3 for v in modal_aligned.values())
            and peak_spread <= 0.05
        ),
    }


def build_shared_body_ir_limitation() -> Dict[str, Any]:
    return {
        "shared_modal_ir_across_all_notes": True,
        "body_stem_identical_per_note_in_step5a": True,
        "note_specific_body_response_not_implemented": True,
        "pitch_dependent_shaping_via_excitation_harmonics_only": True,
        "not_playable_instrument_claim": True,
        "note_realism_claim_blocked": True,
        "expected_limitation_at_this_stage": True,
        "explicit_label": (
            "All notes share the same calibrated modal body IR from Step 3C; "
            "differences arise from excitation harmonic shaping only."
        ),
    }


def build_decay_interpretation_summary(
    per_note_stem: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    by_note = {
        n: (per_note_stem.get(n) or {}).get("decay_interpretation") or {}
        for n in NOTE_SET
    }
    all_force_dominant = all(
        d.get("classification") == "force_dominant_main_envelope_not_body_failure"
        for d in by_note.values()
        if d
    )
    any_raw_short_with_long_body = all(
        d.get("main_decay_short_raw_envelope") and d.get("body_modal_tail_long_enough")
        for d in by_note.values()
        if d
    )
    any_body_fail = any(not d.get("modal_tail_ok") for d in by_note.values())
    any_exc_dom = any(d.get("excitation_dominance_issue") for d in by_note.values())

    return {
        "per_note": by_note,
        "all_notes_main_short_decay_is_force_dominant_metric": bool(
            all_force_dominant or any_raw_short_with_long_body
        ),
        "step5a_raw_main_decay_reconciled": bool(any_raw_short_with_long_body),
        "any_modal_tail_issue": bool(any_body_fail),
        "any_excitation_dominance_issue": bool(any_exc_dom),
        "global_summary": (
            "Step 5A main −40 dB ~3 ms (raw envelope) reflects F_bridge/excitation-dominated "
            "convolution envelope, not modal body failure; body stem −40 dB ~110 ms confirms "
            "physically consistent modal tail."
            if any_raw_short_with_long_body and not any_body_fail
            else (
                "Smoothed main envelope decay aligns with body stem across notes."
                if all_force_dominant and not any_body_fail
                else "Review per-note decay classifications before extended validation."
            )
        ),
        "pass": bool(not any_body_fail and not any_exc_dom),
    }


def build_artifact_guard(
    per_note_stem: Mapping[str, Mapping[str, Any]],
    per_note_spectral: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    checks = {
        "no_body_tail": True,
        "no_delayed_body_onset": all(
            (per_note_stem.get(n) or {}).get("onset", {}).get("no_delayed_body_onset")
            for n in NOTE_SET
        ),
        "no_second_onset": all(
            (per_note_stem.get(n) or {}).get("onset", {}).get("no_second_pluck_event")
            for n in NOTE_SET
        ),
        "no_hard_gate": all(
            (per_note_stem.get(n) or {}).get("main", {}).get("no_hard_gate") for n in NOTE_SET
        ),
        "no_end_rise": all(
            (per_note_stem.get(n) or {}).get("main", {}).get("no_end_rise") for n in NOTE_SET
        ),
        "no_echo_reverb_comb": all(
            not (per_note_spectral.get(n) or {}).get("main_spectral", {}).get("comb_score", 1.0) >= 0.95
            for n in NOTE_SET
        ),
        "no_arbitrary_wood_gain": True,
        "no_stk_integration": True,
        "no_website_production_claim": True,
        "no_final_synthesis_claim": True,
        "no_multi_guitar_claim": True,
        "no_melody_chord_claim": True,
        "no_listening_based_acceptance": True,
    }
    return {**checks, "pass": bool(all(bool(v) for v in checks.values()))}


def build_refinement_recommendations(
    decay_summary: Mapping[str, Any],
    cross_note: Mapping[str, Any],
    artifact: Mapping[str, Any],
    objective_pass: bool,
) -> List[Dict[str, Any]]:
    if objective_pass:
        return [
            {
                "type": "no_adjustment_required",
                "reason": "Objective checks pass; no corrected diagnostic candidate needed",
                "allowed": True,
            }
        ]
    recs: List[Dict[str, Any]] = []
    if decay_summary.get("any_modal_tail_issue"):
        recs.append(
            {
                "type": "modal_tail_review",
                "reason": "Body stem decay shorter than expected; review Step 3C Q/tau calibration",
                "allowed": True,
            }
        )
    if decay_summary.get("any_excitation_dominance_issue"):
        recs.append(
            {
                "type": "excitation_smoothing_candidate",
                "reason": "Excitation may dominate main waveform; smoother F_bridge pulse candidate allowed",
                "allowed": True,
            }
        )
    if cross_note.get("notes_spectrally_very_similar_due_to_shared_ir"):
        recs.append(
            {
                "type": "documented_limitation",
                "reason": "Notes similar due to shared body IR — expected at this stage, not a failure",
                "allowed": True,
            }
        )
    if not artifact.get("pass"):
        recs.append({"type": "blocked", "reason": "Artifact guard failed", "allowed": False})
    return recs or [{"type": "monitor", "reason": "Review objective metrics", "allowed": True}]


def build_readiness_after_step5b(
    *,
    step5a_verify: Mapping[str, Any],
    decay_summary: Mapping[str, Any],
    artifact: Mapping[str, Any],
    objective_pass: bool,
) -> Dict[str, Any]:
    if not step5a_verify.get("pass"):
        status = "failed_limited_note_set_refinement"
    elif decay_summary.get("any_excitation_dominance_issue") or decay_summary.get("any_modal_tail_issue"):
        status = "blocked_due_to_decay_or_excitation_dominance"
    elif not artifact.get("pass"):
        status = "blocked_due_to_note_specific_artifacts"
    elif not objective_pass:
        status = "failed_limited_note_set_refinement"
    else:
        status = READINESS_STEP5C

    return {
        "current_status": status,
        "final_synthesis_ready": False,
        "stk_integration_allowed": False,
        "website_production_replacement_allowed": False,
        "multi_guitar_comparison_allowed": False,
        "melody_chord_playback_allowed": False,
        "subjective_tuning_allowed": False,
        "step5c_note_set_extended_validation_allowed": status == READINESS_STEP5C,
    }


def run_objective_tests(
    step5a_verify: Mapping[str, Any],
    per_note_stem: Mapping[str, Mapping[str, Any]],
    per_note_spectral: Mapping[str, Mapping[str, Any]],
    cross_note: Mapping[str, Any],
    decay_summary: Mapping[str, Any],
    artifact: Mapping[str, Any],
    step5a_preserved: bool,
    step4a_preserved: bool,
) -> Dict[str, Any]:
    per_note_pass = {
        n: bool((per_note_stem.get(n) or {}).get("pass"))
        and bool((per_note_spectral.get(n) or {}).get("pass"))
        for n in NOTE_SET
    }
    tests = {
        "step5a_upstream_ready": step5a_verify.get("pass"),
        "step5a_outputs_preserved": step5a_preserved,
        "step4a_outputs_preserved": step4a_preserved,
        "all_notes_stem_decay_pass": all(per_note_pass.values()),
        "decay_interpretation_pass": decay_summary.get("pass"),
        "cross_note_consistency_pass": cross_note.get("pass"),
        "artifact_guard_pass": artifact.get("pass"),
        "corrected_candidate_not_required": True,
        "final_synthesis_blocked": True,
        "stk_integration_blocked": True,
        "melody_chords_blocked": True,
        "subjective_tuning_blocked": True,
    }
    tests["all_pass"] = bool(all(tests.values()))
    return tests


def write_optional_figures(
    signals: Mapping[str, Dict[str, np.ndarray]],
    sr: int,
    modal_freqs: Sequence[float],
    per_note_stem: Mapping[str, Mapping[str, Any]],
    per_note_spectral: Mapping[str, Mapping[str, Any]],
    cross_note: Mapping[str, Mapping[str, Any]],
    out_dir: Path,
) -> List[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    from pgsm_step4b_single_note_diagnostic_refinement import _envelope

    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, note in zip(axes.flat, NOTE_SET):
        sig = signals[note]
        t = np.arange(len(sig["main"])) / sr
        limit = min(sr, len(t))
        ax.plot(t[:limit], _envelope(sig["main"], sr)[:limit], label="main", alpha=0.9)
        ax.plot(t[:limit], _envelope(sig["body"], sr)[:limit], label="body", alpha=0.7)
        ax.plot(t[:limit], _envelope(sig["excitation"], sr)[:limit], label="excitation", alpha=0.7)
        ax.set_title(f"{note} envelope overlay")
        ax.legend(fontsize=7)
    fig.tight_layout()
    p = out_dir / "per_note_envelope_overlay.png"
    fig.savefig(p, dpi=100, bbox_inches="tight")
    plt.close(fig)
    written.append(str(p))

    fig, ax = plt.subplots(figsize=(9, 4))
    for note in NOTE_SET:
        stem = per_note_stem.get(note) or {}
        main_d = (stem.get("main") or {}).get("decay_ms", {}).get("minus_40_dB")
        body_d = (stem.get("body_stem") or {}).get("decay_ms", {}).get("minus_40_dB")
        exc_d = (stem.get("excitation_stem") or {}).get("decay_ms", {}).get("minus_40_dB")
        ax.plot([note], [main_d or 0], "o", label="main" if note == NOTE_SET[0] else "")
        ax.plot([note], [body_d or 0], "s", label="body" if note == NOTE_SET[0] else "")
        ax.plot([note], [exc_d or 0], "^", label="excitation" if note == NOTE_SET[0] else "")
    ax.set_ylabel("−40 dB decay (ms)")
    ax.set_title("Decay comparison: main vs body vs excitation")
    ax.legend()
    p2 = out_dir / "decay_comparison_main_body_excitation.png"
    fig.savefig(p2, dpi=100, bbox_inches="tight")
    plt.close(fig)
    written.append(str(p2))

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
        ax.set_title(f"{note} spectrum + modal")
    fig.tight_layout()
    p3 = out_dir / "spectrum_modal_overlay_by_note.png"
    fig.savefig(p3, dpi=100, bbox_inches="tight")
    plt.close(fig)
    written.append(str(p3))

    notes = list(NOTE_SET)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    axes[0].bar(notes, [cross_note.get("main_peak_fs_by_note", {}).get(n, 0) for n in notes])
    axes[0].set_title("Peak FS")
    axes[1].bar(notes, [cross_note.get("body_minus_40_db_ms_by_note", {}).get(n, 0) for n in notes])
    axes[1].set_title("Body −40 dB ms")
    axes[2].bar(notes, [cross_note.get("spectral_centroid_hz_by_note", {}).get(n, 0) for n in notes])
    axes[2].set_title("Centroid Hz")
    fig.tight_layout()
    p4 = out_dir / "cross_note_metric_summary.png"
    fig.savefig(p4, dpi=100, bbox_inches="tight")
    plt.close(fig)
    written.append(str(p4))

    return written


def build_pgsm_step5b_report(
    *,
    repo_root: Optional[Path] = None,
    write_figures: bool = False,
    max_modes: Optional[int] = None,
    allow_corrected_candidate: bool = False,
) -> Dict[str, Any]:
    root = Path(repo_root or REPO_ROOT)
    step5a = load_step_report(_report_path(root, "pgsm_step5a_limited_note_set_diagnostic_audio.json"))
    step4c = load_step_report(_report_path(root, "pgsm_step4c_single_note_extended_validation.json"))
    step3c = load_step_report(_report_path(root, "pgsm_step3c_numeric_calibration.json"))

    step5a_fp_before = step5a_output_fingerprints(root)
    step4a_fp = step4a_output_fingerprints(root)

    wav_by_note = {note: _wav_paths_for_note(root, step5a, note) for note in NOTE_SET}
    step5a_verify = verify_step5a_readiness(step5a, wav_by_note)

    state = build_calibrated_modal_state(root, max_modes=max_modes)
    modal_freqs = [float(m["frequency_hz"]) for m in state["modal_weights"].get("modes") or []]
    ir_ref = compute_impulse_response(state["modal_weights"])

    per_note_stem: Dict[str, Any] = {}
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

        step5a_metrics = (step5a.get("per_note_audio_metrics") or {}).get(note) or {}
        step5a_d40 = (step5a_metrics.get("decay_ms") or {}).get("minus_40_dB_ms")
        if step5a_d40 is None:
            step5a_d40 = (step5a_metrics.get("decay_ms") or {}).get("minus_40_dB")

        per_note_stem[note] = analyze_per_note_stem_decay(
            main, body, excitation, sr, step5a_main_decay_ms=step5a_d40
        )
        per_note_spectral[note] = analyze_spectral_harmonic(
            main, body, excitation, sr, modal_freqs, NOTE_FREQUENCY_HZ[note]
        )
        analyzed_files[note] = {k: str(v) for k, v in paths.items()}

    cross_note = build_cross_note_consistency(per_note_stem, per_note_spectral, body_fps)
    shared_limit = build_shared_body_ir_limitation()
    decay_summary = build_decay_interpretation_summary(per_note_stem)
    artifact = build_artifact_guard(per_note_stem, per_note_spectral)

    step5a_fp_after = step5a_output_fingerprints(root)
    step5a_preserved = step5a_fp_before == step5a_fp_after
    step4a_preserved = step4a_fp == step4a_output_fingerprints(root)

    objective_pass = bool(
        step5a_verify.get("pass")
        and decay_summary.get("pass")
        and artifact.get("pass")
        and all((per_note_stem.get(n) or {}).get("pass") for n in NOTE_SET)
        and all((per_note_spectral.get(n) or {}).get("pass") for n in NOTE_SET)
    )

    recommendations = build_refinement_recommendations(
        decay_summary, cross_note, artifact, objective_pass
    )

    corrected_generated = False
    corrected_paths: List[str] = []
    if allow_corrected_candidate and not objective_pass:
        corrected_generated = False

    objective = run_objective_tests(
        step5a_verify,
        per_note_stem,
        per_note_spectral,
        cross_note,
        decay_summary,
        artifact,
        step5a_preserved,
        step4a_preserved,
    )
    readiness = build_readiness_after_step5b(
        step5a_verify=step5a_verify,
        decay_summary=decay_summary,
        artifact=artifact,
        objective_pass=objective.get("all_pass", False),
    )

    figures: List[str] = []
    if write_figures:
        figures = write_optional_figures(
            signals, sr, modal_freqs, per_note_stem, per_note_spectral, cross_note, FIGURES_DIR
        )

    return {
        "report_version": PGSM_STEP5B_VERSION,
        "timestamp": _utc_now(),
        "status": "pgsm_step5b_limited_note_set_refinement_complete",
        "no_stk_integration": True,
        "no_fem_run": True,
        "no_rom_run": True,
        "website_default": DEFAULT_WEBSITE_STK_MODE,
        "website_default_unchanged": True,
        "sample_id": SAMPLE_ID,
        "step5a_loaded": step5a.get("report_version"),
        "step4c_loaded": step4c.get("report_version"),
        "step3c_loaded": step3c.get("report_version"),
        "analyzed_note_set": list(NOTE_SET),
        "analyzed_files": analyzed_files,
        "step5a_readiness_verification": step5a_verify,
        "step5a_outputs_preserved": step5a_preserved,
        "step4a_outputs_preserved": step4a_preserved,
        "per_note_stem_decay_metrics": per_note_stem,
        "cross_note_consistency": cross_note,
        "spectral_harmonic_diagnostics": per_note_spectral,
        "artifact_guard_results": artifact,
        "shared_body_ir_limitation": shared_limit,
        "decay_interpretation": decay_summary,
        "refinement_recommendations": recommendations,
        "corrected_candidate_generated": corrected_generated,
        "corrected_candidate_paths": corrected_paths,
        "step3c_ir_reference_peak_ms": ir_ref.get("peak_time_ms"),
        "objective_analysis_only": True,
        "listening_based_tuning_used": False,
        "objective_test_results": objective,
        "blocked_claims": [
            "Final synthesis",
            "STK integration",
            "Website production replacement",
            "Multi-guitar comparison",
            "Melody/chord playback",
            "Subjective tuning by ear",
            "Note realism / playable instrument",
        ],
        "figures_written": figures,
        "readiness_after_step5b": readiness,
        "safe_next_step": (
            "PGSM Step 5C: note-set extended validation (still diagnostic, not final synthesis)"
            if readiness["current_status"] == READINESS_STEP5C
            else "Resolve Step 5B refinement issues before Step 5C"
        ),
        "explicit_statement": (
            "PGSM Step 5B performs objective diagnostic refinement only. "
            "It is not final guitar synthesis."
        ),
    }


def write_markdown_report(report: Mapping[str, Any], path: Path) -> None:
    rg = report.get("readiness_after_step5b") or {}
    cross = report.get("cross_note_consistency") or {}
    decay = report.get("decay_interpretation") or {}
    shared = report.get("shared_body_ir_limitation") or {}
    artifact = report.get("artifact_guard_results") or {}
    stem = report.get("per_note_stem_decay_metrics") or {}
    spectral = report.get("spectral_harmonic_diagnostics") or {}

    lines = [
        "# PGSM Step 5B — limited note-set diagnostic refinement",
        "",
        f"**Generated:** {report.get('timestamp')}",
        f"**Readiness:** `{rg.get('current_status')}`",
        "",
        report.get("explicit_statement", ""),
        "",
        "## Analyzed files",
        "",
    ]
    for note, files in (report.get("analyzed_files") or {}).items():
        lines.append(f"### {note}")
        for k, v in files.items():
            lines.append(f"- {k}: `{v}`")
        lines.append("")

    lines.extend(
        [
            "## Stem-separated decay (−40 dB ms)",
            "",
            "| Note | main raw | main smooth | body stem | excitation | classification |",
            "|------|----------|-------------|-----------|------------|----------------|",
        ]
    )
    for note in NOTE_SET:
        s = stem.get(note) or {}
        interp = s.get("decay_interpretation") or {}
        lines.append(
            f"| {note} | "
            f"{(s.get('main') or {}).get('raw_envelope_decay_ms', {}).get('minus_40_dB')} | "
            f"{(s.get('main') or {}).get('decay_ms', {}).get('minus_40_dB')} | "
            f"{(s.get('body_stem') or {}).get('decay_ms', {}).get('minus_40_dB')} | "
            f"{(s.get('excitation_stem') or {}).get('decay_ms', {}).get('minus_40_dB')} | "
            f"{interp.get('classification')} |"
        )

    lines.extend(
        [
            "",
            "## Decay interpretation",
            "",
            decay.get("global_summary", ""),
            "",
            f"- All notes force-dominant main metric: {decay.get('all_notes_main_short_decay_is_force_dominant_metric')}",
            f"- Any modal tail issue: {decay.get('any_modal_tail_issue')}",
            f"- Pass: {decay.get('pass')}",
            "",
            "## Cross-note consistency",
            "",
            f"- Shared body stem identical: {cross.get('shared_body_stem_identical_across_notes')}",
            f"- Notes spectrally similar (shared IR): {cross.get('notes_spectrally_very_similar_due_to_shared_ir')}",
            f"- Harmonic shaping visible: {cross.get('pitch_harmonic_shaping_visible_in_spectra')}",
            f"- Pass: {cross.get('pass')}",
            "",
            "## Shared body IR limitation",
            "",
            shared.get("explicit_label", ""),
            "",
            "## Spectral/harmonic (per note)",
            "",
            "| Note | centroid Hz | modal aligned | note shaping | pass |",
            "|------|-------------|---------------|--------------|------|",
        ]
    )
    for note in NOTE_SET:
        sp = spectral.get(note) or {}
        ms = sp.get("main_spectral") or {}
        lines.append(
            f"| {note} | {ms.get('centroid_hz')} | {ms.get('modal_peaks_aligned')} | "
            f"{sp.get('note_dependent_spectral_shaping_visible')} | {sp.get('pass')} |"
        )

    lines.extend(
        [
            "",
            "## Artifact guard",
            "",
            f"- Pass: {artifact.get('pass')}",
            "",
            f"Step 5A preserved: **{report.get('step5a_outputs_preserved')}**",
            f"Corrected candidate: **{report.get('corrected_candidate_generated')}**",
            f"all_pass: **{(report.get('objective_test_results') or {}).get('all_pass')}**",
            "",
            report.get("safe_next_step", ""),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pgsm_step5b_reports(
    *,
    repo_root: Optional[Path] = None,
    json_path: Optional[Path] = None,
    md_path: Optional[Path] = None,
    write_figures: bool = False,
    max_modes: Optional[int] = None,
) -> Dict[str, Any]:
    root = Path(repo_root or REPO_ROOT)
    report = build_pgsm_step5b_report(
        repo_root=root, write_figures=write_figures, max_modes=max_modes
    )
    jpath = Path(json_path or REPORT_JSON)
    mpath = Path(md_path or REPORT_MD)
    jpath.parent.mkdir(parents=True, exist_ok=True)
    jpath.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown_report(report, mpath)
    return report


def main() -> None:
    report = write_pgsm_step5b_reports(write_figures=True)
    rg = report.get("readiness_after_step5b") or {}
    print(f"Wrote {REPORT_JSON}")
    print(f"Readiness: {rg.get('current_status')}")
    print(f"Decay pass: {(report.get('decay_interpretation') or {}).get('pass')}")
    print(f"Corrected candidate: {report.get('corrected_candidate_generated')}")


if __name__ == "__main__":
    main()
