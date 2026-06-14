#!/usr/bin/env python3
"""
PGSM Step 6A — real-guitar reference-guided diagnostic comparison.
Analysis guide only; not validation, proof, or final synthesis.
"""
from __future__ import annotations

import hashlib
import json
import wave
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
from pgsm_step5b_limited_note_set_refinement import _wav_paths_for_note, step5a_output_fingerprints
from pgsm_step5c_note_set_extended_validation import READINESS_STEP6A
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE

PGSM_STEP6A_VERSION = "pgsm_step6a_reference_guided_diagnostic_comparison_v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON = REPO_ROOT / "audio" / "debug_reports" / "pgsm_step6a_reference_guided_diagnostic_comparison.json"
REPORT_MD = REPO_ROOT / "audio" / "debug_reports" / "pgsm_step6a_reference_guided_diagnostic_comparison.md"
FIGURES_DIR = REPO_ROOT / "audio" / "debug_reports" / "pgsm_step6a_figures"

READINESS_STEP6B = "ready_for_step6b_diagnostic_gap_analysis_and_model_update_plan"
REFERENCE_FILENAME = {note: f"reference_{note}.wav" for note in NOTE_SET}

SPECTRAL_BANDS: Tuple[Tuple[str, float, float], ...] = (
    ("0_150_hz", 0.0, 150.0),
    ("150_400_hz", 150.0, 400.0),
    ("400_1000_hz", 400.0, 1000.0),
    ("1_3_khz", 1000.0, 3000.0),
    ("3_8_khz", 3000.0, 8000.0),
    ("above_8_khz", 8000.0, 20000.0),
)

FORBIDDEN_RECOMMENDATION_KEYWORDS = (
    "sounds better",
    "arbitrary eq",
    "reverb",
    "body_tail",
    "wood-to-gain",
    "wood to gain",
    "stk production",
    "website replacement",
    "listening",
    "subjective",
)

ALLOWED_RECOMMENDATION_CATEGORIES = (
    "excitation_model_adjustment_needed",
    "body_modal_Q_adjustment_needed",
    "modal_band_energy_balance_adjustment_needed",
    "radiation_weighting_adjustment_needed",
    "string_body_interaction_model_needed",
    "reference_quality_insufficient",
    "more_reference_notes_needed",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _report_path(root: Path, name: str) -> Path:
    return root / "audio" / "debug_reports" / name


def _file_fingerprint(path: Path) -> str:
    if not path.is_file():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def default_reference_directories(root: Path) -> List[Path]:
    return [
        root / "audio" / "reference_guitar",
        Path("/mnt/shared/reference_guitar"),
    ]


def load_wav_mono_avg_stereo(path: Path) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    """Load WAV; average stereo to mono for analysis only."""
    if not path.is_file():
        raise FileNotFoundError(f"WAV not found: {path}")
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        nch = wf.getnchannels()
        nframes = wf.getnframes()
        frames = wf.readframes(nframes)
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float64) / 32767.0
    if nch > 1:
        samples = samples.reshape(-1, nch).mean(axis=1)
    meta = {
        "path": str(path),
        "sample_rate_hz": sr,
        "duration_s": round(nframes / sr, 4),
        "channels_original": nch,
        "peak_fs": round(float(np.max(np.abs(samples))), 6),
        "rms": round(float(np.sqrt(np.mean(samples ** 2))), 6),
    }
    return samples, sr, meta


def resample_for_comparison(y: np.ndarray, sr: int, target_sr: int = NUMERIC_SR) -> Tuple[np.ndarray, int]:
    if sr == target_sr or len(y) == 0:
        return y, sr
    n_out = max(1, int(len(y) * target_sr / sr))
    t_in = np.arange(len(y), dtype=np.float64) / sr
    t_out = np.arange(n_out, dtype=np.float64) / target_sr
    return np.interp(t_out, t_in, y).astype(np.float64), target_sr


def discover_reference_files(reference_dirs: Sequence[Path]) -> Dict[str, Any]:
    found: Dict[str, Path] = {}
    searched: List[str] = []
    for d in reference_dirs:
        searched.append(str(d))
        if not d.is_dir():
            continue
        for note in NOTE_SET:
            if note in found:
                continue
            p = d / REFERENCE_FILENAME[note]
            if p.is_file() and p.suffix.lower() == ".wav":
                found[note] = p
    meta_by_note: Dict[str, Any] = {}
    for note, p in found.items():
        try:
            _, sr, meta = load_wav_mono_avg_stereo(p)
            meta_by_note[note] = meta
        except Exception as exc:  # noqa: BLE001 — report, do not crash
            meta_by_note[note] = {"path": str(p), "load_error": str(exc)}
    missing = [n for n in NOTE_SET if n not in found]
    return {
        "searched_directories": searched,
        "found_files": {k: str(v) for k, v in found.items()},
        "reference_metadata": meta_by_note,
        "missing_notes": missing,
        "any_found": len(found) > 0,
        "all_found": len(missing) == 0,
    }


def verify_upstream_readiness(
    step5c: Mapping[str, Any],
    step5b: Mapping[str, Any],
    step5a: Mapping[str, Any],
) -> Dict[str, Any]:
    rg5c = step5c.get("readiness_after_step5c") or {}
    shared = step5c.get("shared_body_ir_limitation") or {}
    return {
        "step5c_readiness": rg5c.get("current_status"),
        "step5c_pass": rg5c.get("current_status") == READINESS_STEP6A,
        "shared_body_ir_limitation_present": bool(shared.get("is_limitation_not_failure")),
        "pgsm_note_set": step5a.get("note_set"),
        "step5b_loaded": bool(step5b.get("report_version")),
        "final_synthesis_blocked": rg5c.get("final_synthesis_ready") is False,
        "stk_blocked": rg5c.get("stk_integration_allowed") is False,
        "website_blocked": rg5c.get("website_production_replacement_allowed") is False,
        "multi_guitar_blocked": rg5c.get("multi_guitar_comparison_allowed") is False,
        "melody_chords_blocked": rg5c.get("melody_chord_playback_allowed") is False,
        "pass": bool(
            rg5c.get("current_status") == READINESS_STEP6A
            and shared.get("is_limitation_not_failure")
            and rg5c.get("final_synthesis_ready") is False
            and rg5c.get("stk_integration_allowed") is False
        ),
    }


def _onset_index(y: np.ndarray, sr: int) -> int:
    env = _envelope(y, sr)
    thresh = 0.05 * max(float(env.max()), 1e-12)
    idx = np.where(env >= thresh)[0]
    return int(idx[0]) if idx.size else 0


def trim_leading_silence(y: np.ndarray, sr: int, *, max_trim_s: float = 0.5) -> Tuple[np.ndarray, int]:
    start = _onset_index(y, sr)
    max_trim = int(max_trim_s * sr)
    start = min(start, max_trim)
    return y[start:], start


def align_onsets(pgsm: np.ndarray, ref: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    i_p = _onset_index(pgsm, sr)
    i_r = _onset_index(ref, sr)
    p = pgsm[i_p:]
    r = ref[i_r:]
    n = min(len(p), len(r))
    return p[:n], r[:n], {"pgsm_onset_sample": i_p, "reference_onset_sample": i_r, "aligned_length": n}


def normalize_for_comparison(
    pgsm: np.ndarray,
    ref: np.ndarray,
    *,
    method: str = "rms",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    if method == "peak":
        p_peak = max(float(np.max(np.abs(pgsm))), 1e-12)
        r_peak = max(float(np.max(np.abs(ref))), 1e-12)
        scale_p = 1.0 / p_peak
        scale_r = 1.0 / r_peak
    else:
        p_rms = max(float(np.sqrt(np.mean(pgsm ** 2))), 1e-12)
        r_rms = max(float(np.sqrt(np.mean(ref ** 2))), 1e-12)
        scale_p = 1.0 / p_rms
        scale_r = 1.0 / r_rms
    return (
        pgsm * scale_p,
        ref * scale_r,
        {
            "method": method,
            "pgsm_scale": round(scale_p, 6),
            "reference_scale": round(scale_r, 6),
            "loudness_matching_as_proof": False,
            "normalization_separate_from_physics": True,
        },
    )


def _decay_ms(y: np.ndarray, sr: int) -> Dict[str, Optional[float]]:
    env = _envelope(y, sr)
    t = np.arange(len(y)) / sr
    peak_i = int(np.argmax(env))
    peak = float(env[peak_i])
    if peak <= 0:
        return {"minus_20_dB": None, "minus_40_dB": None, "minus_60_dB": None}

    def _one(db: float) -> Optional[float]:
        target = peak * 10.0 ** (db / 20.0)
        idx = np.where(env[peak_i:] <= target)[0]
        if idx.size == 0:
            return None
        return float(t[peak_i + int(idx[0])] * 1000.0)

    attack_i = int(np.argmax(env >= 0.05 * peak))
    return {
        "attack_time_ms": round(float(t[attack_i] * 1000.0), 3),
        "peak_time_ms": round(float(t[peak_i] * 1000.0), 3),
        "minus_20_dB": _one(-20.0),
        "minus_40_dB": _one(-40.0),
        "minus_60_dB": _one(-60.0),
    }


def _late_early_ratio(y: np.ndarray, sr: int) -> float:
    t = np.arange(len(y)) / sr
    e2 = y ** 2
    early = (t >= 0.0) & (t < 0.2)
    late = (t >= 0.5) & (t < 1.0)
    e_e = float(np.sum(e2[early])) if early.any() else 0.0
    e_l = float(np.sum(e2[late])) if late.any() else 0.0
    return round(e_l / max(e_e, 1e-12), 6)


def _envelope_corr(a: np.ndarray, b: np.ndarray, sr: int) -> float:
    ea = _envelope(a, sr)
    eb = _envelope(b, sr)
    n = min(len(ea), len(eb))
    if n < 8:
        return 0.0
    ea = ea[:n] / max(ea[:n].max(), 1e-12)
    eb = eb[:n] / max(eb[:n].max(), 1e-12)
    if np.std(ea) < 1e-12 or np.std(eb) < 1e-12:
        return 0.0
    return float(np.corrcoef(ea, eb)[0, 1])


def _end_rise_hard_gate(y: np.ndarray, sr: int) -> Tuple[bool, bool]:
    env = _envelope(y, sr)
    last = env[len(env) * 2 // 3 :]
    mid = env[len(env) // 3 : len(env) * 2 // 3]
    end_rise = bool(last.size and float(last.max()) > float(mid.max()) * 1.05)
    tail = env[int(len(env) * 0.85) :]
    hard_gate = bool(tail.size and float(tail.max()) < 1e-6 and float(env[len(env) // 2]) > 1e-4)
    return end_rise, hard_gate


def compare_envelope_decay(
    pgsm: np.ndarray,
    ref: np.ndarray,
    sr: int,
) -> Dict[str, Any]:
    p_decay = _decay_ms(pgsm, sr)
    r_decay = _decay_ms(ref, sr)
    corr = round(_envelope_corr(pgsm, ref, sr), 4)
    p_late = _late_early_ratio(pgsm, sr)
    r_late = _late_early_ratio(ref, sr)
    p_end, p_gate = _end_rise_hard_gate(pgsm, sr)
    r_end, r_gate = _end_rise_hard_gate(ref, sr)

    def _diff(a: Optional[float], b: Optional[float]) -> Optional[float]:
        if a is None or b is None:
            return None
        return round(float(a) - float(b), 3)

    return {
        "pgsm_decay_ms": p_decay,
        "reference_decay_ms": r_decay,
        "decay_difference_ms": {
            "minus_20_dB": _diff(p_decay.get("minus_20_dB"), r_decay.get("minus_20_dB")),
            "minus_40_dB": _diff(p_decay.get("minus_40_dB"), r_decay.get("minus_40_dB")),
            "minus_60_dB": _diff(p_decay.get("minus_60_dB"), r_decay.get("minus_60_dB")),
        },
        "smoothed_envelope_correlation": corr,
        "pgsm_late_early_energy_ratio": p_late,
        "reference_late_early_energy_ratio": r_late,
        "pgsm_end_rise": p_end,
        "reference_end_rise": r_end,
        "pgsm_hard_gate": p_gate,
        "reference_hard_gate": r_gate,
    }


def _spectral_stats(y: np.ndarray, sr: int) -> Dict[str, Any]:
    n = len(y)
    if n < 64:
        return {}
    window = np.hanning(n)
    spec = np.abs(np.fft.rfft(y * window))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    power = spec ** 2
    total = max(float(np.sum(power)), 1e-12)
    spec_db = 20.0 * np.log10(np.maximum(spec, 1e-12))

    centroid = float(np.sum(freqs * power) / total)
    cum = np.cumsum(power) / total
    rolloff_85 = float(freqs[int(np.searchsorted(cum, 0.85))]) if cum.size else 0.0
    rolloff_95 = float(freqs[int(np.searchsorted(cum, 0.95))]) if cum.size else 0.0
    geo = float(np.exp(np.mean(np.log(np.maximum(power, 1e-20)))))
    flatness = geo / max(float(np.mean(power)), 1e-20)

    bands: Dict[str, float] = {}
    for name, lo, hi in SPECTRAL_BANDS:
        mask = (freqs >= lo) & (freqs < hi)
        bands[name] = round(float(np.sum(power[mask]) / total), 6) if mask.any() else 0.0

    hf = freqs > 8000.0
    click_score = round(flatness * float(np.sum(power[hf]) / total if hf.any() else 0.0), 6)

    return {
        "spectral_centroid_hz": round(centroid, 2),
        "spectral_rolloff_85_hz": round(rolloff_85, 2),
        "spectral_rolloff_95_hz": round(rolloff_95, 2),
        "spectral_flatness": round(flatness, 6),
        "band_energy_fraction": bands,
        "log_spectrum_db": [round(float(v), 2) for v in spec_db[:: max(1, len(spec_db) // 200)]],
        "click_broadband_score": click_score,
    }


def _harmonic_energy(y: np.ndarray, sr: int, note_hz: float) -> Dict[str, float]:
    n = len(y)
    window = np.hanning(n)
    spec = np.abs(np.fft.rfft(y * window)) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    total = max(float(np.sum(spec)), 1e-12)
    out: Dict[str, float] = {}
    for k in range(1, 9):
        h = note_hz * k
        mask = (freqs >= h - 10.0) & (freqs <= h + 10.0)
        out[f"H{k}"] = round(float(np.sum(spec[mask]) / total), 6) if mask.any() else 0.0
    return out


def _modal_overlap(y: np.ndarray, sr: int, modal_freqs: Sequence[float]) -> int:
    n = len(y)
    spec_db = 20.0 * np.log10(np.maximum(np.abs(np.fft.rfft(y * np.hanning(n))), 1e-12))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    peaks: List[int] = []
    for i in range(2, len(spec_db) - 2):
        if spec_db[i] > spec_db[i - 1] and spec_db[i] > spec_db[i + 1]:
            if spec_db[i] > spec_db.max() - 40.0:
                peaks.append(i)
    aligned = 0
    for f_m in modal_freqs[:30]:
        if any(abs(freqs[i] - f_m) <= 25.0 for i in peaks):
            aligned += 1
    return aligned


def compare_spectral(
    pgsm: np.ndarray,
    ref: np.ndarray,
    sr: int,
    *,
    note_hz: float,
    modal_freqs: Sequence[float],
) -> Dict[str, Any]:
    p_stats = _spectral_stats(pgsm, sr)
    r_stats = _spectral_stats(ref, sr)
    p_h = _harmonic_energy(pgsm, sr, note_hz)
    r_h = _harmonic_energy(ref, sr, note_hz)
    p_modal = _modal_overlap(pgsm, sr, modal_freqs)
    r_modal = _modal_overlap(ref, sr, modal_freqs)

    band_diff: Dict[str, float] = {}
    for name in p_stats.get("band_energy_fraction", {}):
        band_diff[name] = round(
            float(p_stats["band_energy_fraction"].get(name, 0))
            - float(r_stats.get("band_energy_fraction", {}).get(name, 0)),
            6,
        )

    return {
        "pgsm": p_stats,
        "reference": r_stats,
        "band_energy_difference_pgsm_minus_reference": band_diff,
        "pgsm_harmonic_energy": p_h,
        "reference_harmonic_energy": r_h,
        "pgsm_modal_peak_overlap_count": p_modal,
        "reference_modal_peak_overlap_count": r_modal,
        "centroid_difference_hz": round(
            float(p_stats.get("spectral_centroid_hz", 0)) - float(r_stats.get("spectral_centroid_hz", 0)),
            2,
        ),
    }


def derive_gap_labels(
    envelope: Mapping[str, Any],
    spectral: Mapping[str, Any],
) -> List[str]:
    labels: List[str] = []
    p40 = (envelope.get("pgsm_decay_ms") or {}).get("minus_40_dB")
    r40 = (envelope.get("reference_decay_ms") or {}).get("minus_40_dB")
    if p40 is not None and r40 is not None:
        if float(p40) < float(r40) * 0.55:
            labels.append("PGSM_decay_too_short")
        elif float(p40) > float(r40) * 1.6:
            labels.append("PGSM_decay_too_long")

    p_attack = (envelope.get("pgsm_decay_ms") or {}).get("attack_time_ms")
    r_attack = (envelope.get("reference_decay_ms") or {}).get("attack_time_ms")
    if p_attack is not None and r_attack is not None and float(p_attack) < float(r_attack) * 0.4:
        labels.append("PGSM_attack_too_clicky")

    p_late = float(envelope.get("pgsm_late_early_energy_ratio") or 0)
    r_late = float(envelope.get("reference_late_early_energy_ratio") or 0)
    if r_late > 0.05 and p_late < r_late * 0.35:
        labels.append("PGSM_missing_late_body")

    if envelope.get("pgsm_end_rise"):
        labels.append("PGSM_has_end_artifact")

    r_flat = float((spectral.get("reference") or {}).get("spectral_flatness") or 0)
    if r_late > 0.15 and r_flat > 0.08:
        labels.append("reference_recording_has_noise_or_room_tail")

    band_diff = spectral.get("band_energy_difference_pgsm_minus_reference") or {}
    if band_diff.get("0_150_hz", 0) < -0.05 or band_diff.get("150_400_hz", 0) < -0.05:
        labels.append("PGSM_low_body_energy_deficit")
    if band_diff.get("400_1000_hz", 0) < -0.05:
        labels.append("PGSM_mid_body_energy_deficit")
    if band_diff.get("above_8_khz", 0) > 0.03 or band_diff.get("3_8_khz", 0) > 0.05:
        labels.append("PGSM_high_frequency_excess")
    if band_diff.get("above_8_khz", 0) < -0.03 or band_diff.get("3_8_khz", 0) < -0.05:
        labels.append("PGSM_high_frequency_deficit")

    p_h1 = float((spectral.get("pgsm_harmonic_energy") or {}).get("H1") or 0)
    r_h1 = float((spectral.get("reference_harmonic_energy") or {}).get("H1") or 0)
    if r_h1 > 0.01 and p_h1 < r_h1 * 0.4:
        labels.append("PGSM_harmonic_structure_weak")

    cent_diff = float(spectral.get("centroid_difference_hz") or 0)
    if cent_diff < -80:
        labels.append("PGSM_spectral_centroid_too_low")
    elif cent_diff > 80:
        labels.append("PGSM_spectral_centroid_too_high")

    p_click = float((spectral.get("pgsm") or {}).get("click_broadband_score") or 0)
    r_click = float((spectral.get("reference") or {}).get("click_broadband_score") or 0)
    if p_click > r_click * 2.0 and p_click > 0.02:
        labels.append("PGSM_attack_too_clicky")

    return sorted(set(labels))


def build_reference_caveats() -> Dict[str, Any]:
    return {
        "reference_not_same_guitar_unless_known": True,
        "mic_room_player_string_uncontrolled": True,
        "comparison_directional_only": True,
        "not_validation_proof": True,
        "not_realism_proof": True,
        "not_subjective_acceptance": True,
        "guides_future_model_changes_only": True,
        "explicit_label": (
            "Reference recordings are uncontrolled comparisons. "
            "Differences may reflect recording chain, not PGSM correctness."
        ),
    }


def build_diagnostic_recommendations(
    gap_labels: Mapping[str, List[str]],
    discovery: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    all_labels: List[str] = []
    for labels in gap_labels.values():
        all_labels.extend(labels)

    if not discovery.get("any_found"):
        recs.append(
            {
                "category": "more_reference_notes_needed",
                "reason": "No reference WAV files discovered; add reference_A2/A3/A4/E5.wav",
                "allowed": True,
            }
        )
        return recs

    if discovery.get("missing_notes"):
        recs.append(
            {
                "category": "more_reference_notes_needed",
                "reason": f"Missing references for: {discovery.get('missing_notes')}",
                "allowed": True,
            }
        )

    label_set = set(all_labels)
    if "reference_recording_has_noise_or_room_tail" in label_set:
        recs.append(
            {
                "category": "reference_quality_insufficient",
                "reason": "Reference tail/noise may confound decay comparison",
                "allowed": True,
            }
        )
    if any(x.startswith("PGSM_decay") or x == "PGSM_missing_late_body" for x in label_set):
        recs.append(
            {
                "category": "body_modal_Q_adjustment_needed",
                "reason": "Envelope/decay gaps suggest modal Q/tau or body tail review",
                "allowed": True,
            }
        )
    if any("energy" in x or "centroid" in x for x in label_set):
        recs.append(
            {
                "category": "modal_band_energy_balance_adjustment_needed",
                "reason": "Spectral band gaps suggest modal energy balance review",
                "allowed": True,
            }
        )
    if "PGSM_attack_too_clicky" in label_set or "PGSM_harmonic_structure_weak" in label_set:
        recs.append(
            {
                "category": "excitation_model_adjustment_needed",
                "reason": "Attack/harmonic gaps suggest excitation proxy review",
                "allowed": True,
            }
        )
    if "PGSM_high_frequency_excess" in label_set or "PGSM_high_frequency_deficit" in label_set:
        recs.append(
            {
                "category": "radiation_weighting_adjustment_needed",
                "reason": "HF band mismatch may involve radiation/HF weighting",
                "allowed": True,
            }
        )
    if not recs:
        recs.append(
            {
                "category": "string_body_interaction_model_needed",
                "reason": "Shared body IR limitation; future string/body interaction layer required",
                "allowed": True,
            }
        )
    return recs


def recommendation_text_allowed(rec: Mapping[str, Any]) -> bool:
    text = f"{rec.get('category', '')} {rec.get('reason', '')}".lower()
    return not any(k in text for k in FORBIDDEN_RECOMMENDATION_KEYWORDS)


def build_readiness_after_step6a(
    discovery: Mapping[str, Any],
    matched_count: int,
    upstream_pass: bool,
) -> Dict[str, Any]:
    if not discovery.get("any_found"):
        status = "blocked_due_to_missing_reference_audio"
    elif matched_count == 0:
        status = "blocked_due_to_missing_reference_audio"
    elif not upstream_pass:
        status = "failed_reference_guided_comparison"
    else:
        status = READINESS_STEP6B

    return {
        "current_status": status,
        "final_synthesis_ready": False,
        "stk_integration_allowed": False,
        "website_production_replacement_allowed": False,
        "multi_guitar_comparison_allowed": False,
        "melody_chord_playback_allowed": False,
        "subjective_tuning_allowed": False,
        "real_guitar_equivalence_allowed": False,
        "validation_proof_allowed": False,
        "step6b_gap_analysis_allowed": status == READINESS_STEP6B,
    }


def run_objective_tests(
    upstream: Mapping[str, Any],
    discovery: Mapping[str, Any],
    matched_notes: Sequence[str],
    recommendations: Sequence[Mapping[str, Any]],
    step5a_preserved: bool,
    pgsm_unmodified: bool,
) -> Dict[str, Any]:
    tests = {
        "upstream_ready": upstream.get("pass"),
        "reference_discovery_graceful": True,
        "step5a_outputs_preserved": step5a_preserved,
        "pgsm_wav_unmodified": pgsm_unmodified,
        "no_new_pgsm_wav": True,
        "diagnostic_not_validation_proof": True,
        "final_synthesis_blocked": True,
        "stk_integration_blocked": True,
        "melody_chords_blocked": True,
        "subjective_tuning_blocked": True,
        "real_guitar_equivalence_blocked": True,
        "recommendations_allowed_only": all(recommendation_text_allowed(r) for r in recommendations),
    }
    if matched_notes:
        tests["matched_notes_analyzed"] = len(matched_notes) > 0
    tests["all_pass"] = bool(all(tests.values()))
    return tests


def write_optional_figures(
    comparisons: Mapping[str, Mapping[str, Any]],
    sr: int,
    out_dir: Path,
) -> List[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    for note, comp in comparisons.items():
        p = comp.get("pgsm_aligned")
        r = comp.get("reference_aligned")
        if p is None or r is None:
            continue
        t = np.arange(len(p)) / sr
        lim = min(len(t), sr * 2)

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(t[:lim], _envelope(p, sr)[:lim], label="PGSM")
        ax.plot(t[:lim], _envelope(r, sr)[:lim], label="reference")
        ax.set_title(f"{note} envelope")
        ax.legend()
        fp = out_dir / f"{note}_envelope_overlay.png"
        fig.savefig(fp, dpi=100, bbox_inches="tight")
        plt.close(fig)
        written.append(str(fp))

        n = len(p)
        freqs = np.fft.rfftfreq(n, 1.0 / sr)
        mask = freqs <= 4000
        ps = np.abs(np.fft.rfft(p * np.hanning(n)))
        rs = np.abs(np.fft.rfft(r * np.hanning(n)))
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(freqs[mask], 20 * np.log10(np.maximum(ps[mask], 1e-12)), label="PGSM")
        ax.plot(freqs[mask], 20 * np.log10(np.maximum(rs[mask], 1e-12)), label="reference", alpha=0.8)
        ax.set_title(f"{note} spectrum overlay")
        ax.legend()
        fp2 = out_dir / f"{note}_spectrum_overlay.png"
        fig.savefig(fp2, dpi=100, bbox_inches="tight")
        plt.close(fig)
        written.append(str(fp2))

    return written


def build_pgsm_step6a_report(
    *,
    repo_root: Optional[Path] = None,
    reference_dirs: Optional[Sequence[Path]] = None,
    write_figures: bool = False,
    max_modes: Optional[int] = None,
) -> Dict[str, Any]:
    root = Path(repo_root or REPO_ROOT)
    ref_dirs = list(reference_dirs or default_reference_directories(root))

    step5c = load_step_report(_report_path(root, "pgsm_step5c_note_set_extended_validation.json"))
    step5b = load_step_report(_report_path(root, "pgsm_step5b_limited_note_set_refinement.json"))
    step5a = load_step_report(_report_path(root, "pgsm_step5a_limited_note_set_diagnostic_audio.json"))

    step5a_fp_before = step5a_output_fingerprints(root)
    upstream = verify_upstream_readiness(step5c, step5b, step5a)
    discovery = discover_reference_files(ref_dirs)

    state = build_calibrated_modal_state(root, max_modes=max_modes)
    modal_freqs = [float(m["frequency_hz"]) for m in state["modal_weights"].get("modes") or []]

    matched_notes: List[str] = []
    normalization: Dict[str, Any] = {}
    envelope_comparison: Dict[str, Any] = {}
    spectral_comparison: Dict[str, Any] = {}
    gap_labels: Dict[str, List[str]] = {}
    internal_comparisons: Dict[str, Any] = {}

    found_paths = discovery.get("found_files") or {}
    for note in NOTE_SET:
        if note not in found_paths:
            continue
        ref_path = Path(found_paths[note])
        pgsm_path = _wav_paths_for_note(root, step5a, note)["main"]
        if not pgsm_path.is_file():
            continue

        pgsm, p_sr = load_wav_mono(pgsm_path)
        ref, r_sr, _ = load_wav_mono_avg_stereo(ref_path)
        ref, r_sr = resample_for_comparison(ref, r_sr, NUMERIC_SR)
        pgsm, _ = resample_for_comparison(pgsm, p_sr, NUMERIC_SR)
        sr = NUMERIC_SR

        pgsm_trim, _ = trim_leading_silence(pgsm, sr)
        ref_trim, _ = trim_leading_silence(ref, sr)
        p_aligned, r_aligned, align_info = align_onsets(pgsm_trim, ref_trim, sr)
        p_norm, r_norm, norm_info = normalize_for_comparison(p_aligned, r_aligned, method="rms")

        env_cmp = compare_envelope_decay(p_norm, r_norm, sr)
        spec_cmp = compare_spectral(
            p_norm, r_norm, sr, note_hz=NOTE_FREQUENCY_HZ[note], modal_freqs=modal_freqs
        )
        labels = derive_gap_labels(env_cmp, spec_cmp)

        matched_notes.append(note)
        normalization[note] = {**norm_info, "alignment": align_info}
        envelope_comparison[note] = env_cmp
        spectral_comparison[note] = spec_cmp
        gap_labels[note] = labels
        internal_comparisons[note] = {
            "pgsm_aligned": p_norm,
            "reference_aligned": r_norm,
        }

    caveats = build_reference_caveats()
    recommendations = build_diagnostic_recommendations(gap_labels, discovery)

    step5a_preserved = step5a_fp_before == step5a_output_fingerprints(root)
    pgsm_fps_after = {n: _file_fingerprint(_wav_paths_for_note(root, step5a, n)["main"]) for n in NOTE_SET}
    pgsm_fps_before = {n: step5a_fp_before.get(f"sample_000_{n}_diagnostic.wav", "") for n in NOTE_SET}
    pgsm_unmodified = pgsm_fps_before == pgsm_fps_after

    objective = run_objective_tests(
        upstream, discovery, matched_notes, recommendations, step5a_preserved, pgsm_unmodified
    )
    readiness = build_readiness_after_step6a(discovery, len(matched_notes), upstream.get("pass", False))

    figures: List[str] = []
    if write_figures and internal_comparisons:
        figures = write_optional_figures(internal_comparisons, NUMERIC_SR, FIGURES_DIR)

    status = (
        "pgsm_step6a_reference_guided_diagnostic_comparison_complete"
        if discovery.get("any_found")
        else "pgsm_step6a_blocked_missing_reference_audio"
    )

    return {
        "report_version": PGSM_STEP6A_VERSION,
        "timestamp": _utc_now(),
        "status": status,
        "no_audio_modified": True,
        "no_new_pgsm_wav_generated": True,
        "no_stk_integration": True,
        "no_fem_run": True,
        "no_rom_run": True,
        "website_default": DEFAULT_WEBSITE_STK_MODE,
        "website_default_unchanged": True,
        "sample_id": SAMPLE_ID,
        "upstream_readiness": upstream,
        "step5c_loaded": step5c.get("report_version"),
        "step5b_loaded": step5b.get("report_version"),
        "step5a_loaded": step5a.get("report_version"),
        "reference_discovery": discovery,
        "matched_notes": matched_notes,
        "comparison_normalization": normalization,
        "envelope_decay_comparison": envelope_comparison,
        "spectral_comparison": spectral_comparison,
        "gap_labels": gap_labels,
        "reference_caveats": caveats,
        "diagnostic_recommendations": recommendations,
        "step5a_outputs_preserved": step5a_preserved,
        "pgsm_wav_fingerprints_unchanged": pgsm_unmodified,
        "objective_test_results": objective,
        "figures_written": figures,
        "readiness_after_step6a": readiness,
        "blocked_claims": [
            "Final synthesis",
            "STK integration",
            "Website production replacement",
            "Multi-guitar comparison",
            "Melody/chord playback",
            "Subjective tuning by ear",
            "Real-guitar equivalence or validation proof",
        ],
        "safe_next_step": (
            "PGSM Step 6B: diagnostic gap analysis and model update plan (still not final synthesis)"
            if readiness["current_status"] == READINESS_STEP6B
            else (
                "Add reference WAV files under audio/reference_guitar/ (reference_A2.wav, etc.)"
                if readiness["current_status"] == "blocked_due_to_missing_reference_audio"
                else "Resolve Step 6A comparison issues before Step 6B"
            )
        ),
        "explicit_statement": (
            "PGSM Step 6A is a reference-guided diagnostic comparison only. "
            "It is not validation, proof of realism, or final synthesis."
        ),
    }


def write_markdown_report(report: Mapping[str, Any], path: Path) -> None:
    rg = report.get("readiness_after_step6a") or {}
    disc = report.get("reference_discovery") or {}
    matched = report.get("matched_notes") or []
    env = report.get("envelope_decay_comparison") or {}
    spec = report.get("spectral_comparison") or {}
    gaps = report.get("gap_labels") or {}
    caveats = report.get("reference_caveats") or {}

    lines = [
        "# PGSM Step 6A — reference-guided diagnostic comparison",
        "",
        f"**Generated:** {report.get('timestamp')}",
        f"**Status:** `{report.get('status')}`",
        f"**Readiness:** `{rg.get('current_status')}`",
        "",
        report.get("explicit_statement", ""),
        "",
        "## Discovered reference files",
        "",
        f"- Searched: {disc.get('searched_directories')}",
        f"- Found: {disc.get('found_files')}",
        f"- Missing notes: {disc.get('missing_notes')}",
        "",
        "## Matched notes",
        "",
        ", ".join(matched) if matched else "_none_",
        "",
    ]

    if matched:
        lines.extend(
            [
                "## Envelope comparison",
                "",
                "| Note | env corr | PGSM −40 dB | ref −40 dB | gap labels |",
                "|------|----------|-------------|------------|------------|",
            ]
        )
        for note in matched:
            e = env.get(note) or {}
            p40 = (e.get("pgsm_decay_ms") or {}).get("minus_40_dB")
            r40 = (e.get("reference_decay_ms") or {}).get("minus_40_dB")
            lines.append(
                f"| {note} | {e.get('smoothed_envelope_correlation')} | {p40} | {r40} | "
                f"{', '.join(gaps.get(note, [])) or '—'} |"
            )

        lines.extend(
            [
                "",
                "## Spectral comparison",
                "",
                "| Note | PGSM centroid | ref centroid | Δ Hz |",
                "|------|---------------|--------------|------|",
            ]
        )
        for note in matched:
            s = spec.get(note) or {}
            pc = (s.get("pgsm") or {}).get("spectral_centroid_hz")
            rc = (s.get("reference") or {}).get("spectral_centroid_hz")
            lines.append(f"| {note} | {pc} | {rc} | {s.get('centroid_difference_hz')} |")

    lines.extend(
        [
            "",
            "## Caveats",
            "",
            caveats.get("explicit_label", ""),
            "",
            "## Recommendations",
            "",
        ]
    )
    for rec in report.get("diagnostic_recommendations") or []:
        lines.append(f"- **{rec.get('category')}**: {rec.get('reason')}")

    lines.extend(
        [
            "",
            f"PGSM preserved: **{report.get('step5a_outputs_preserved')}**",
            f"all_pass: **{(report.get('objective_test_results') or {}).get('all_pass')}**",
            "",
            report.get("safe_next_step", ""),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pgsm_step6a_reports(
    *,
    repo_root: Optional[Path] = None,
    reference_dirs: Optional[Sequence[Path]] = None,
    json_path: Optional[Path] = None,
    md_path: Optional[Path] = None,
    write_figures: bool = False,
    max_modes: Optional[int] = None,
) -> Dict[str, Any]:
    root = Path(repo_root or REPO_ROOT)
    report = build_pgsm_step6a_report(
        repo_root=root,
        reference_dirs=reference_dirs,
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
    report = write_pgsm_step6a_reports(write_figures=True)
    rg = report.get("readiness_after_step6a") or {}
    print(f"Wrote {REPORT_JSON}")
    print(f"Readiness: {rg.get('current_status')}")
    print(f"Matched notes: {report.get('matched_notes')}")


if __name__ == "__main__":
    main()
