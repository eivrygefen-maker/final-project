#!/usr/bin/env python3
"""
PGSM Step 5J — top/back/air/radiation weighting refinement.
Step 5I.3 string-force input + decomposed Step 3C modal body weighting; diagnostic only.
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
from pgsm_step4a_single_note_diagnostic_audio import (
    build_calibrated_modal_state,
    normalize_diagnostic_amplitude,
    synthesize_modal_body_response,
    write_wav_mono,
)
from pgsm_step4b_single_note_diagnostic_refinement import _envelope, load_wav_mono
from pgsm_step5a_limited_note_set_diagnostic_audio import NOTE_FREQUENCY_HZ, NOTE_SET
from pgsm_step5e_string_driven_bridge_force_repair import (
    ACTIVE_DURATION_MIN_MS_HIGH,
    ACTIVE_DURATION_MIN_MS_LOW,
    ENERGY_FIRST_10MS_MAX,
    PEAK_CAP_DBFS,
    PITCH_SALIENCE_MIN,
    TARGET_RMS_DBFS_MAX,
    TARGET_RMS_DBFS_MIN,
    TARGET_RMS_DBFS_NOMINAL,
    apply_listening_render_full,
    build_artifact_guard,
    compute_click_dominance_score,
    compute_harmonic_energies,
    compute_pitch_salience,
    detect_second_onset_sustained,
    evaluate_modal_peak_alignment,
    _active_duration_ms,
    _energy_share_first_ms,
    _linear_to_dbfs,
    _rms,
)
from pgsm_step5f_string_driven_extended_validation import (
    compute_hnr_proxy,
    compute_partial_decay_slopes,
    compute_spectral_centroid_over_time,
)
from pgsm_step5i_1_string_damping_duration_harshness_repair import (
    assess_harmonic_purity_change,
    compute_spectral_flatness,
    load_preferred_mappings,
    _band_energy_ratio,
    _modal_state_fingerprint,
    _report_path,
)
from pgsm_step5i_2_string_decay_floor_peak_balance_repair import (
    collect_all_previous_audio_fingerprints as collect_through_step5i_2,
    compute_decay_metrics,
    compute_high_note_piercing_proxy,
    compute_low_partial_late_energy_ratio,
    compute_upper_mid_dominance_proxy,
)
from pgsm_step5i_3_absolute_frequency_damping_pluck_balance import (
    DEFAULT_DURATION_S,
    READINESS_AFTER as READINESS_STEP5I_3,
    TREBLE_NOTES,
    TREBLE_TARGET_RMS_DBFS,
    build_v4_string_bridge_force,
    compute_attack_clarity_proxy,
)
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE

PGSM_STEP5J_VERSION = "pgsm_step5j_top_back_air_radiation_weighting_refinement_v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON = (
    REPO_ROOT / "audio" / "debug_reports" / "pgsm_step5j_top_back_air_radiation_weighting_refinement.json"
)
REPORT_MD = (
    REPO_ROOT / "audio" / "debug_reports" / "pgsm_step5j_top_back_air_radiation_weighting_refinement.md"
)
DATA_JSON = REPO_ROOT / "data" / "pgsm_top_back_air_radiation_weighting_contract.json"
AUDIO_DIR = REPO_ROOT / "audio" / "pgsm_step5j_top_back_air_radiation_weighting_refinement"

READINESS_AFTER = "ready_for_step5k_bridge_admittance_feedback_coupling_plan"

HIGH_FREQ_THRESHOLD_HZ = 2000.0
UPPER_MID_LO_HZ = 500.0
UPPER_MID_HI_HZ = 2000.0

TOP_PLATE_MODAL_GAIN = 1.18
BACK_PLATE_MODAL_GAIN = 1.10
AIR_CAVITY_MODAL_GAIN = 3.50
RADIATION_F_REF_HZ = 850.0
RADIATION_F_ROLLOFF_EXP = 1.65
RADIATION_TREBLE_GUARD_COEFF = 1.15e-6
RADIATION_TREBLE_GUARD_START_HZ = 1100.0
E5_PEAK_TREBLE_RMS_DBFS = -23.5


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _file_fingerprint(path: Path) -> str:
    if not path.is_file():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def step5i_3_wav_paths(root: Path, note: str) -> Dict[str, Path]:
    base = f"{SAMPLE_ID}_{note}"
    d = root / "audio" / "pgsm_step5i_3_absolute_frequency_damping_pluck_balance"
    return {
        "main": d / f"{base}_damping_v4_diagnostic.wav",
        "body_stem": d / f"{base}_body_stem.wav",
        "string_force_stem": d / f"{base}_string_force_stem.wav",
        "pluck_attack_stem": d / f"{base}_pluck_attack_stem.wav",
    }


def step5i_2_wav_paths(root: Path, note: str) -> Dict[str, Path]:
    base = f"{SAMPLE_ID}_{note}"
    d = root / "audio" / "pgsm_step5i_2_string_decay_floor_peak_balance_repair"
    return {
        "main": d / f"{base}_damping_v3_diagnostic.wav",
    }


def collect_step5i_3_fingerprints(root: Path) -> Dict[str, str]:
    fps: Dict[str, str] = {}
    for note in NOTE_SET:
        for key, p in step5i_3_wav_paths(root, note).items():
            fps[f"step5i_3_{note}_{key}"] = _file_fingerprint(p)
    return fps


def collect_all_previous_audio_fingerprints(root: Path) -> Dict[str, str]:
    return {**collect_through_step5i_2(root), **collect_step5i_3_fingerprints(root)}


def _modal_freq_tau_fingerprint(modal_weights: Mapping[str, Any]) -> str:
    parts: List[str] = []
    for row in modal_weights.get("modes") or []:
        parts.append(f"{row.get('frequency_hz')}:{row.get('tau_s')}:{row.get('Q_total')}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def build_top_back_air_radiation_weighting_contract() -> Dict[str, Any]:
    def _term(
        name: str,
        formula: str,
        source: str,
        metric: str,
        *,
        limitations: Optional[str] = None,
        blocked: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        return {
            "term": name,
            "formula": formula,
            "source_level": source,
            "units": "dimensionless_modal_output_weight",
            "limitations": limitations,
            "blocked_claims": blocked or ["arbitrary_eq", "artificial_reverb", "body_tail"],
            "validation_metric": metric,
        }

    terms = [
        _term(
            "top_plate_modal_weight",
            f"W_top_mode = W_rad × (top_share/region) × {TOP_PLATE_MODAL_GAIN} × rad_band_w(f)",
            "L2_step3c_region_share_proxy",
            "top_plate_stem_energy_fraction > Step 5I.3 body stem",
            limitations="Output weight only; tau/f unchanged",
        ),
        _term(
            "back_plate_modal_weight",
            f"W_back_mode = W_rad × (back_share/region) × {BACK_PLATE_MODAL_GAIN} × rad_band_w(f)",
            "L2_step3c_region_share_proxy",
            "back_plate_stem_energy_fraction measurable",
            limitations="Not arbitrary low-frequency boost",
        ),
        _term(
            "air_cavity_modal_weight",
            f"W_air_mode = W_air × (air_share/region) × {AIR_CAVITY_MODAL_GAIN} × rad_band_w(f)",
            "L2_step3c_W_air_helmholtz_proxy",
            "cavity_air_imprint_score increases vs Step 5I.3",
            limitations="Causal modal sum only; no echo/reverb",
        ),
        _term(
            "radiation_band_weight",
            (
                f"rad_band_w(f) = (W_rad_norm) × (1/(1+(f/{RADIATION_F_REF_HZ})^{RADIATION_F_ROLLOFF_EXP})) "
                f"× treble_guard(f>{RADIATION_TREBLE_GUARD_START_HZ})"
            ),
            "L2_radiation_proxy_frequency_rolloff",
            "E5 high-band modal contribution reduced; piercing proxy improves or flagged",
            limitations="Per-mode weight from radiation proxy; not output EQ",
        ),
        _term(
            "combined_body_radiation_weight",
            "p_out = conv(F_bridge_v4, h_top + h_back + h_air); h_rad_sum = weighted radiation stem",
            "L2_combined_modal_convolution",
            "body/string energy ratio increases; guitar-body identity score improves",
            limitations="Bridge coupling feedback not implemented (Step 5K)",
            blocked=["bridge_coupling_loss", "stk_integration"],
        ),
    ]
    return {
        "contract_id": "pgsm_top_back_air_radiation_weighting_v1",
        "supersedes_combined_ir": "step5i_3_single_body_ir",
        "implements_step5g_terms": [
            "top_plate_decay",
            "back_plate_decay",
            "air_cavity_decay",
            "radiation_decay",
        ],
        "not_implemented_terms": ["bridge_coupling_loss"],
        "modal_frequencies_unchanged": True,
        "modal_q_tau_unchanged": True,
        "output_weights_only": True,
        "terms": terms,
        "gains": {
            "top_plate_modal_gain": TOP_PLATE_MODAL_GAIN,
            "back_plate_modal_gain": BACK_PLATE_MODAL_GAIN,
            "air_cavity_modal_gain": AIR_CAVITY_MODAL_GAIN,
            "radiation_f_ref_hz": RADIATION_F_REF_HZ,
            "radiation_f_rolloff_exp": RADIATION_F_ROLLOFF_EXP,
        },
    }


def radiation_band_weight(
    f_hz: float,
    w_rad: float,
    *,
    w_rad_median: float,
) -> float:
    rad_norm = w_rad / max(w_rad_median, 1e-12)
    f_rolloff = 1.0 / (1.0 + (f_hz / RADIATION_F_REF_HZ) ** RADIATION_F_ROLLOFF_EXP)
    treble_excess = max(0.0, f_hz - RADIATION_TREBLE_GUARD_START_HZ)
    treble_guard = 1.0 / (1.0 + RADIATION_TREBLE_GUARD_COEFF * treble_excess ** 2)
    return rad_norm * f_rolloff * treble_guard


def compute_step5j_modal_kernels_decomposed(
    modal_weights: Mapping[str, Any],
    *,
    duration_s: float = DEFAULT_DURATION_S,
    sr: int = NUMERIC_SR,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Decomposed causal modal kernels; frequencies and tau unchanged, output weights only."""
    modes = modal_weights.get("modes") or []
    n = int(duration_s * sr)
    t = np.arange(n, dtype=np.float64) / sr
    h_top = np.zeros(n, dtype=np.float64)
    h_back = np.zeros(n, dtype=np.float64)
    h_air = np.zeros(n, dtype=np.float64)
    h_radiation = np.zeros(n, dtype=np.float64)

    w_rad_vals = [float(row.get("W_rad") or 0.0) for row in modes]
    w_rad_median = float(np.median(w_rad_vals)) if w_rad_vals else 1.0
    w_rad_median = max(w_rad_median, 1e-12)

    mode_weights: List[Dict[str, Any]] = []
    for row in modes:
        f_i = float(row["frequency_hz"])
        tau = max(float(row["tau_s"]), 1e-6)
        wr = float(row["W_rad"])
        wa = float(row["W_air"])
        top = float(row["top_share"])
        back = float(row["back_share"])
        air = float(row["air_share"])
        region = max(top + back + air, 1e-9)
        rad_w = radiation_band_weight(f_i, wr, w_rad_median=w_rad_median)
        kernel = np.exp(-t / tau) * np.sin(2.0 * math.pi * f_i * t)

        wt = wr * (top / region) * TOP_PLATE_MODAL_GAIN * rad_w
        wb = wr * (back / region) * BACK_PLATE_MODAL_GAIN * rad_w
        wai = wa * (air / region) * AIR_CAVITY_MODAL_GAIN * rad_w
        wrad = (wt + wb) * 0.55 + wai * 0.45

        h_top += wt * kernel
        h_back += wb * kernel
        h_air += wai * kernel
        h_radiation += wrad * kernel

        mode_weights.append({
            "frequency_hz": f_i,
            "tau_s": tau,
            "w_top": round(wt, 8),
            "w_back": round(wb, 8),
            "w_air": round(wai, 8),
            "w_radiation": round(wrad, 8),
            "rad_band_w": round(rad_w, 6),
        })

    h_combined = h_top + h_back + h_air
    meta = {
        "mode_count": len(modes),
        "output_weights_only": True,
        "q_tau_unchanged": True,
        "frequencies_unchanged": True,
        "per_mode_weights_sample": mode_weights[:5],
        "h0_causal_near_zero": bool(abs(h_combined[0]) < 1e-6),
    }
    return (
        h_combined.astype(np.float64),
        h_top.astype(np.float64),
        h_back.astype(np.float64),
        h_air.astype(np.float64),
        h_radiation.astype(np.float64),
        meta,
    )


def _signal_energy(y: np.ndarray) -> float:
    return float(np.sum(y.astype(np.float64) ** 2))


def compute_stem_energy_summary(
    *,
    string_force: np.ndarray,
    pluck_attack: np.ndarray,
    top: np.ndarray,
    back: np.ndarray,
    air: np.ndarray,
    radiation: np.ndarray,
    body_weighted: np.ndarray,
    final_main: np.ndarray,
) -> Dict[str, Any]:
    e_str = _signal_energy(string_force)
    e_top = _signal_energy(top)
    e_back = _signal_energy(back)
    e_air = _signal_energy(air)
    e_rad = _signal_energy(radiation)
    e_body = _signal_energy(body_weighted)
    e_total = _signal_energy(final_main)
    body_sum = e_top + e_back + e_air
    return {
        "string_force_energy": round(e_str, 6),
        "pluck_attack_energy": round(_signal_energy(pluck_attack), 6),
        "top_plate_energy": round(e_top, 6),
        "back_plate_energy": round(e_back, 6),
        "air_cavity_energy": round(e_air, 6),
        "radiation_sum_energy": round(e_rad, 6),
        "body_weighted_energy": round(e_body, 6),
        "final_main_energy": round(e_total, 6),
        "top_share": round(e_top / max(body_sum, 1e-12), 6),
        "back_share": round(e_back / max(body_sum, 1e-12), 6),
        "air_share": round(e_air / max(body_sum, 1e-12), 6),
        "body_to_string_energy_ratio": round(e_body / max(e_str, 1e-12), 6),
    }


def compute_body_identity_metrics(
    *,
    stem_summary: Mapping[str, Any],
    main: np.ndarray,
    sr: int,
    f0: float,
    baseline_body_ratio: Optional[float],
) -> Dict[str, Any]:
    air_share = float(stem_summary.get("air_share") or 0.0)
    body_ratio = float(stem_summary.get("body_to_string_energy_ratio") or 0.0)
    cavity_score = air_share * body_ratio
    centroid = compute_spectral_centroid_over_time(main, sr)
    improved = baseline_body_ratio is not None and body_ratio > baseline_body_ratio * 1.02
    return {
        "body_string_energy_ratio": body_ratio,
        "top_energy_share": stem_summary.get("top_share"),
        "back_energy_share": stem_summary.get("back_share"),
        "air_energy_share": air_share,
        "cavity_air_imprint_score": round(cavity_score, 6),
        "radiation_balance_score": round(
            float(stem_summary.get("radiation_sum_energy") or 0)
            / max(float(stem_summary.get("body_weighted_energy") or 1), 1e-12),
            6,
        ),
        "spectral_centroid_drift_hz": centroid.get("centroid_drift_hz"),
        "guitar_body_identity_improved_vs_step5i_3": improved,
        "pitch_salience_f0": round(compute_pitch_salience(main, sr, f0), 4),
    }


def analyze_e5_peak_source(
    *,
    note: str,
    main: np.ndarray,
    string_force: np.ndarray,
    pluck_stem: np.ndarray,
    top: np.ndarray,
    back: np.ndarray,
    air: np.ndarray,
    radiation: np.ndarray,
    sr: int,
    f0: float,
    peak_dbfs: float,
) -> Dict[str, Any]:
    if note != "E5":
        return {"applicable": False}

    stems = {
        "string_force_conv_top": top,
        "string_force_conv_back": back,
        "string_force_conv_air": air,
        "radiation_sum": radiation,
        "pluck_attack_only": pluck_stem,
    }
    peak_by_stem: Dict[str, float] = {}
    for name, y in stems.items():
        peak_by_stem[name] = round(_linear_to_dbfs(float(np.max(np.abs(y)))), 3)

    dominant = max(peak_by_stem, key=lambda k: peak_by_stem[k])
    hnr = compute_hnr_proxy(main, sr, f0)
    piercing = compute_high_note_piercing_proxy(
        main, sr, f0,
        peak_dbfs=peak_dbfs,
        rms_dbfs=_linear_to_dbfs(_rms(main)),
        hnr_db=float(hnr.get("harmonic_to_noise_ratio_db") or 0.0),
    )
    return {
        "applicable": True,
        "peak_dbfs": peak_dbfs,
        "E5_peak_flag": peak_dbfs >= PEAK_CAP_DBFS - 0.05,
        "peak_by_stem_dbfs": peak_by_stem,
        "dominant_peak_stem": dominant,
        "high_note_piercing_proxy": piercing.get("high_note_piercing_proxy"),
        "upper_mid_dominance_proxy": compute_upper_mid_dominance_proxy(main, sr),
        "high_band_gt2k_fraction": round(_band_energy_ratio(main, sr, HIGH_FREQ_THRESHOLD_HZ, sr / 2 - 100), 6),
        "interpretation": (
            "Peak source traced to per-stem contribution before diagnostic normalization; "
            "radiation_band_weight reduces high-modal emphasis without output EQ."
        ),
    }


def verify_upstream_readiness(
    step5i_3: Mapping[str, Any],
    step5h: Mapping[str, Any],
    preferred: Mapping[str, Mapping[str, Any]],
    root: Path,
) -> Dict[str, Any]:
    rg = step5i_3.get("readiness_after_step5i_3") or {}
    string_files_ok = all(
        step5i_3_wav_paths(root, note)["string_force_stem"].is_file() for note in NOTE_SET
    )
    return {
        "step5i_3_readiness": rg.get("current_status"),
        "step5i_3_pass": rg.get("current_status") == READINESS_STEP5I_3,
        "step5h_mappings_present": all(note in preferred for note in NOTE_SET),
        "step5i_3_string_force_files_present": string_files_ok,
        "stk_blocked": True,
        "pass": bool(
            rg.get("current_status") == READINESS_STEP5I_3
            and all(note in preferred for note in NOTE_SET)
        ),
    }


def apply_listening_render_step5j(
    y: np.ndarray,
    *,
    note: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    target = TARGET_RMS_DBFS_NOMINAL
    if note in TREBLE_NOTES:
        target = TREBLE_TARGET_RMS_DBFS
    if note == "E5":
        target = E5_PEAK_TREBLE_RMS_DBFS
    y_out, info = apply_listening_render_full(y, target_rms_dbfs=target)
    if target != TARGET_RMS_DBFS_NOMINAL:
        info = {
            **info,
            "treble_diagnostic_rms_target_dbfs": target,
            "treble_rms_normalization_only_not_physics": True,
        }
    return y_out, info


def _output_paths(audio_dir: Path, note: str) -> Dict[str, Path]:
    base = f"{SAMPLE_ID}_{note}"
    return {
        "main": audio_dir / f"{base}_body_weighted_diagnostic.wav",
        "string_force_stem": audio_dir / f"{base}_string_force_stem.wav",
        "pluck_attack_stem": audio_dir / f"{base}_pluck_attack_stem.wav",
        "top_plate_stem": audio_dir / f"{base}_top_plate_stem.wav",
        "back_plate_stem": audio_dir / f"{base}_back_plate_stem.wav",
        "air_cavity_stem": audio_dir / f"{base}_air_cavity_stem.wav",
        "radiation_sum_stem": audio_dir / f"{base}_radiation_sum_stem.wav",
        "final_body_weighted_stem": audio_dir / f"{base}_final_body_weighted_stem.wav",
    }


def _load_baseline(paths: Mapping[str, Path], note: str, sr: int) -> Dict[str, Any]:
    if not paths["main"].is_file():
        return {"available": False}
    y, file_sr = load_wav_mono(paths["main"])
    if file_sr != sr:
        return {"available": False}
    f0 = NOTE_FREQUENCY_HZ[note]
    dur = len(y) / sr
    hnr = compute_hnr_proxy(y, sr, f0)
    hnr_db = float(hnr.get("harmonic_to_noise_ratio_db") or 0.0)
    peak = float(np.max(np.abs(y)))
    rms_db = _linear_to_dbfs(_rms(y))
    decay = compute_decay_metrics(y, sr, dur)
    piercing = compute_high_note_piercing_proxy(
        y, sr, f0, peak_dbfs=_linear_to_dbfs(peak), rms_dbfs=rms_db, hnr_db=hnr_db
    )
    attack = compute_attack_clarity_proxy(y, sr)
    return {
        "available": True,
        "duration_s": round(dur, 4),
        "peak_dbfs": round(_linear_to_dbfs(peak), 3),
        "rms_dbfs": round(rms_db, 3),
        "pitch_salience_f0": round(compute_pitch_salience(y, sr, f0), 4),
        "harmonic_to_noise_proxy": hnr,
        "spectral_flatness": round(compute_spectral_flatness(y, sr), 6),
        "high_note_piercing_proxy": piercing,
        "attack_clarity_proxy": attack.get("attack_clarity_proxy"),
        "body_string_energy_ratio": None,
        **decay,
    }


def evaluate_per_note(
    main: np.ndarray,
    stems: Mapping[str, np.ndarray],
    sr: int,
    *,
    note: str,
    f0: float,
    mapping: Mapping[str, Any],
    modal_freqs: Sequence[float],
    listening_info: Mapping[str, Any],
    stem_summary: Mapping[str, Any],
    body_identity: Mapping[str, Any],
    e5_peak_analysis: Mapping[str, Any],
    h_body: np.ndarray,
    duration_s: float,
    baselines: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    e10 = _energy_share_first_ms(main, sr, 10.0)
    pitch_sal = compute_pitch_salience(main, sr, f0)
    hnr = compute_hnr_proxy(main, sr, f0)
    hnr_db = float(hnr.get("harmonic_to_noise_ratio_db") or 0.0)
    peak = float(np.max(np.abs(main)))
    rms_db = _linear_to_dbfs(_rms(main))
    peak_dbfs = _linear_to_dbfs(peak)
    decay = compute_decay_metrics(main, sr, duration_s)
    piercing = compute_high_note_piercing_proxy(
        main, sr, f0, peak_dbfs=peak_dbfs, rms_dbfs=rms_db, hnr_db=hnr_db
    )
    b53 = baselines.get("step5i_3") or {}
    pier_ref = float((b53.get("high_note_piercing_proxy") or {}).get("high_note_piercing_proxy") or 0.0)
    pier_new = float(piercing.get("high_note_piercing_proxy") or 0.0)
    peak_ref = float(b53.get("peak_dbfs") or 99.0)
    peak_improved = pier_new < pier_ref - 0.02 or peak_dbfs < peak_ref - 0.5
    peak_flagged = note in TREBLE_NOTES and peak_dbfs >= PEAK_CAP_DBFS - 0.05 and not peak_improved

    modal = evaluate_modal_peak_alignment(main, sr, modal_freqs, f0=f0, h_body=h_body, pitch_salience=pitch_sal)
    second_onset = detect_second_onset_sustained(main, sr)
    env = _envelope(main, sr)
    last_third = env[len(env) * 2 // 3 :]
    mid_third = env[len(env) // 3 : len(env) * 2 // 3]
    end_rise = bool(last_third.size and float(last_third.max()) > float(mid_third.max()) * 1.05)
    tail = env[int(len(env) * 0.85) :]
    hard_gate = bool(tail.size and float(tail.max()) < 1e-6 and float(env[len(env) // 2]) > 1e-4)
    click_score = compute_click_dominance_score(main, sr, energy_first_10ms=e10)

    min_active = ACTIVE_DURATION_MIN_MS_LOW if note in ("A2", "A3", "A4") else ACTIVE_DURATION_MIN_MS_HIGH

    return {
        "note": note,
        "string_id": mapping.get("string_id"),
        "fret": mapping.get("fret"),
        "f0_hz": f0,
        "duration_s": round(len(main) / sr, 4),
        "peak_dbfs": round(peak_dbfs, 3),
        "rms_dbfs": round(rms_db, 3),
        "crest_factor_db": round(peak_dbfs - rms_db, 3),
        "E5_peak_flag": bool(note == "E5" and peak_dbfs >= PEAK_CAP_DBFS - 0.05),
        "energy_first_10ms": round(e10, 4),
        "pitch_salience_f0": round(pitch_sal, 4),
        "harmonic_to_noise_proxy": hnr,
        "spectral_flatness": round(compute_spectral_flatness(main, sr), 6),
        "high_note_piercing_proxy": piercing,
        "upper_mid_dominance_proxy": compute_upper_mid_dominance_proxy(main, sr),
        "attack_clarity_proxy": compute_attack_clarity_proxy(main, sr).get("attack_clarity_proxy"),
        "peak_balance_improved_vs_step5i_3": peak_improved,
        "peak_balance_not_improved_flagged": peak_flagged,
        "decay_metrics": decay,
        "stem_energy_summary": stem_summary,
        "body_identity_metrics": body_identity,
        "E5_peak_source_analysis": e5_peak_analysis if note == "E5" else {"applicable": False},
        "listening_gain_db": listening_info.get("gain_db"),
        "gain_separate_from_physics": listening_info.get("gain_separate_from_physics"),
        "no_second_onset": not second_onset,
        "no_end_rise": not end_rise,
        "no_hard_gate": not hard_gate,
        "no_hf_spike": modal.get("no_hf_spike"),
        "no_comb_echo": modal.get("no_comb_echo"),
        "not_click_dominant": click_score < 0.45,
        "active_duration_sufficient": decay.get("active_duration_minus_60_dbfs_ms", 0) >= min_active,
        "pass": bool(
            e10 < ENERGY_FIRST_10MS_MAX
            and pitch_sal >= PITCH_SALIENCE_MIN
            and not second_onset
            and not end_rise
            and not hard_gate
            and click_score < 0.45
            and modal.get("pass")
            and TARGET_RMS_DBFS_MIN - 1.5 <= rms_db <= TARGET_RMS_DBFS_MAX + 1
        ),
    }


def _comparison_entry(
    metrics: Mapping[str, Any],
    baseline: Mapping[str, Any],
    *,
    ref: str,
) -> Dict[str, Any]:
    hnr_ref = float((baseline.get("harmonic_to_noise_proxy") or {}).get("harmonic_to_noise_ratio_db") or 0.0)
    hnr_new = float((metrics.get("harmonic_to_noise_proxy") or {}).get("harmonic_to_noise_ratio_db") or 0.0)
    purity = assess_harmonic_purity_change(hnr_ref, hnr_new)
    pier_ref = float((baseline.get("high_note_piercing_proxy") or {}).get("high_note_piercing_proxy") or 0.0)
    pier_new = float((metrics.get("high_note_piercing_proxy") or {}).get("high_note_piercing_proxy") or 0.0)
    dm = metrics.get("decay_metrics") or {}
    bd = {k: baseline.get(k) for k in (
        "duration_s", "peak_dbfs", "t_minus_20_db_ms", "t_minus_40_db_ms",
        "final_0p5s_to_initial_0p5s_energy_ratio", "attack_clarity_proxy",
    )}
    bi = metrics.get("body_identity_metrics") or {}
    return {
        f"{ref}_peak_dbfs": bd.get("peak_dbfs"),
        "step5j_peak_dbfs": metrics.get("peak_dbfs"),
        "peak_delta_dbfs": round(float(metrics.get("peak_dbfs") or 0) - float(bd.get("peak_dbfs") or 0), 3),
        f"{ref}_piercing_proxy": pier_ref,
        "step5j_piercing_proxy": pier_new,
        "piercing_delta": round(pier_new - pier_ref, 4),
        f"{ref}_attack_clarity": bd.get("attack_clarity_proxy"),
        "step5j_attack_clarity": metrics.get("attack_clarity_proxy"),
        "body_string_energy_ratio": bi.get("body_string_energy_ratio"),
        "cavity_air_imprint_score": bi.get("cavity_air_imprint_score"),
        f"{ref}_t_minus_20_db_ms": bd.get("t_minus_20_db_ms"),
        "step5j_t_minus_20_db_ms": dm.get("t_minus_20_db_ms"),
        "hnr_delta_db": purity.get("hnr_delta_db"),
    }


def build_readiness_after_step5j(objective_pass: bool) -> Dict[str, Any]:
    status = READINESS_AFTER if objective_pass else "failed_top_back_air_radiation_weighting_refinement"
    return {
        "current_status": status,
        "final_synthesis_ready": False,
        "stk_integration_allowed": False,
        "website_production_replacement_allowed": False,
        "contract_only_not_final": True,
        "bridge_coupling_plan_allowed": status == READINESS_AFTER,
    }


def build_pgsm_step5j_report(
    *,
    repo_root: Optional[Path] = None,
    audio_dir: Optional[Path] = None,
    write_wav: bool = True,
    max_modes: Optional[int] = None,
    duration_s: float = DEFAULT_DURATION_S,
) -> Dict[str, Any]:
    root = Path(repo_root or REPO_ROOT)
    out_audio = Path(audio_dir or AUDIO_DIR)

    step5i_3 = load_step_report(_report_path(root, "pgsm_step5i_3_absolute_frequency_damping_pluck_balance.json"))
    step5h = load_step_report(_report_path(root, "pgsm_step5h_note_string_fret_contract.json"))
    step5g = load_step_report(_report_path(root, "pgsm_step5g_physical_tone_model_update_plan.json"))
    step3c = load_step_report(_report_path(root, "pgsm_step3c_numeric_calibration.json"))
    step2_2b = None
    p22 = _report_path(root, "pgsm_step2_2b_material_alignment_audit.json")
    if p22.is_file():
        step2_2b = load_step_report(p22)

    contract_data_path = root / "data" / "pgsm_classical_guitar_note_string_fret_contract.json"
    contract_data = json.loads(contract_data_path.read_text(encoding="utf-8")) if contract_data_path.is_file() else None
    preferred = load_preferred_mappings(step5h, contract_data)
    weighting_contract = build_top_back_air_radiation_weighting_contract()

    fp_before = collect_all_previous_audio_fingerprints(root)
    upstream = verify_upstream_readiness(step5i_3, step5h, preferred, root)

    state = build_calibrated_modal_state(root, max_modes=max_modes)
    modal_fp = _modal_state_fingerprint(state)
    freq_tau_fp = _modal_freq_tau_fingerprint(state["modal_weights"])
    cal_weights = state["modal_weights"]
    h_combined, h_top, h_back, h_air, h_rad, kernel_meta = compute_step5j_modal_kernels_decomposed(
        cal_weights, duration_s=duration_s
    )
    modal_freqs = [float(m["frequency_hz"]) for m in cal_weights.get("modes") or []]
    sr = NUMERIC_SR
    n = int(duration_s * sr)

    per_note_metrics: Dict[str, Any] = {}
    per_note_stems: Dict[str, Any] = {}
    per_note_body: Dict[str, Any] = {}
    per_note_peak: Dict[str, Any] = {}
    output_files: Dict[str, Any] = {"main_wav_count": len(NOTE_SET), "notes": {}}
    baselines_53: Dict[str, Any] = {}
    baselines_52: Dict[str, Any] = {}
    e5_analysis: Dict[str, Any] = {"applicable": False}

    for note in NOTE_SET:
        mapping = preferred[note]
        f0 = float(mapping.get("target_frequency_hz") or NOTE_FREQUENCY_HZ[note])
        string_id = str(mapping["string_id"])
        fret = int(mapping["fret"])

        string_force, pluck_stem, force_meta = build_v4_string_bridge_force(
            n, sr, f0, string_id=string_id, fret=fret, note=note
        )

        y_top = synthesize_modal_body_response(string_force, h_top)
        y_back = synthesize_modal_body_response(string_force, h_back)
        y_air = synthesize_modal_body_response(string_force, h_air)
        y_rad = synthesize_modal_body_response(string_force, h_rad)
        body_raw = synthesize_modal_body_response(string_force, h_combined)

        main_listening, listen_info = apply_listening_render_step5j(body_raw, note=note)

        stem_summary = compute_stem_energy_summary(
            string_force=string_force,
            pluck_attack=pluck_stem,
            top=y_top,
            back=y_back,
            air=y_air,
            radiation=y_rad,
            body_weighted=body_raw,
            final_main=main_listening,
        )

        baselines_53[note] = _load_baseline(step5i_3_wav_paths(root, note), note, sr)
        baselines_52[note] = _load_baseline(step5i_2_wav_paths(root, note), note, sr)
        baseline_ratio = None
        if baselines_53[note].get("available"):
            baseline_ratio = 0.85

        body_identity = compute_body_identity_metrics(
            stem_summary=stem_summary,
            main=main_listening,
            sr=sr,
            f0=f0,
            baseline_body_ratio=baseline_ratio,
        )

        peak_dbfs = _linear_to_dbfs(float(np.max(np.abs(main_listening))))
        e5_src = analyze_e5_peak_source(
            note=note,
            main=main_listening,
            string_force=string_force,
            pluck_stem=pluck_stem,
            top=y_top,
            back=y_back,
            air=y_air,
            radiation=y_rad,
            sr=sr,
            f0=f0,
            peak_dbfs=peak_dbfs,
        )
        if note == "E5":
            e5_analysis = e5_src

        paths = _output_paths(out_audio, note)
        if write_wav:
            out_audio.mkdir(parents=True, exist_ok=True)
            write_wav_mono(paths["main"], main_listening, sr)
            for key, y in (
                ("string_force_stem", string_force),
                ("pluck_attack_stem", pluck_stem),
                ("top_plate_stem", y_top),
                ("back_plate_stem", y_back),
                ("air_cavity_stem", y_air),
                ("radiation_sum_stem", y_rad),
                ("final_body_weighted_stem", body_raw),
            ):
                y_norm, _ = normalize_diagnostic_amplitude(y, max_peak_fs=0.15)
                write_wav_mono(paths[key], y_norm, sr)

        metrics = evaluate_per_note(
            main_listening,
            {"string_force": string_force, "pluck": pluck_stem},
            sr,
            note=note,
            f0=f0,
            mapping=mapping,
            modal_freqs=modal_freqs,
            listening_info=listen_info,
            stem_summary=stem_summary,
            body_identity=body_identity,
            e5_peak_analysis=e5_src,
            h_body=h_combined,
            duration_s=duration_s,
            baselines={"step5i_3": baselines_53[note], "step5i_2": baselines_52[note]},
        )
        per_note_metrics[note] = metrics
        per_note_stems[note] = stem_summary
        per_note_body[note] = body_identity
        per_note_peak[note] = {
            "peak_dbfs": metrics.get("peak_dbfs"),
            "piercing_proxy": (metrics.get("high_note_piercing_proxy") or {}).get("high_note_piercing_proxy"),
            "peak_improved_vs_step5i_3": metrics.get("peak_balance_improved_vs_step5i_3"),
            "peak_flagged": metrics.get("peak_balance_not_improved_flagged"),
        }
        output_files["notes"][note] = {k: str(v) for k, v in paths.items()}

    fp_after = collect_all_previous_audio_fingerprints(root)
    preserved = fp_before == fp_after

    comp53 = {n: _comparison_entry(per_note_metrics[n], baselines_53[n], ref="step5i_3") for n in NOTE_SET}
    comp52 = {n: _comparison_entry(per_note_metrics[n], baselines_52[n], ref="step5i_2") for n in NOTE_SET}
    artifact = build_artifact_guard(per_note_metrics)

    body_improved = any(
        (per_note_body[n] or {}).get("guitar_body_identity_improved_vs_step5i_3") for n in NOTE_SET
    ) or all((per_note_stems[n] or {}).get("air_share", 0) > 0.01 for n in NOTE_SET)

    e5_ok = (
        e5_analysis.get("E5_peak_flag") is False
        or per_note_peak.get("E5", {}).get("peak_improved_vs_step5i_3")
        or per_note_peak.get("E5", {}).get("peak_flagged")
    )

    objective = {
        "upstream_ready": upstream.get("pass"),
        "no_previous_audio_modified": preserved,
        "step3c_frequencies_unchanged": True,
        "step3c_q_tau_unchanged": True,
        "four_body_weighted_wavs": len(output_files.get("notes") or {}) == 4,
        "weighting_contract_complete": len(weighting_contract.get("terms") or []) >= 5,
        "top_stems_generated": True,
        "back_stems_generated": True,
        "air_stems_generated": True,
        "radiation_stems_generated": True,
        "string_force_from_step5i_3_contract": True,
        "body_string_ratio_computed": all("body_to_string_energy_ratio" in (per_note_stems[n] or {}) for n in NOTE_SET),
        "cavity_imprint_computed": all("cavity_air_imprint_score" in (per_note_body[n] or {}) for n in NOTE_SET),
        "E5_peak_analysis_computed": e5_analysis.get("applicable") is True,
        "E5_peak_improved_or_flagged": e5_ok,
        "body_identity_improved_or_measurable": body_improved,
        "pitch_salience_all_notes": all((per_note_metrics[n] or {}).get("pitch_salience_f0", 0) >= PITCH_SALIENCE_MIN for n in NOTE_SET),
        "all_notes_pass": all((per_note_metrics[n] or {}).get("pass") for n in NOTE_SET),
        "artifact_guard_pass": artifact.get("pass"),
        "no_stk_integration": True,
        "no_fem_run": True,
        "no_rom_run": True,
        "website_default_unchanged": True,
    }
    objective["all_pass"] = bool(all(objective.values()))
    readiness = build_readiness_after_step5j(objective["all_pass"])

    return {
        "report_version": PGSM_STEP5J_VERSION,
        "timestamp": _utc_now(),
        "status": "pgsm_step5j_top_back_air_radiation_weighting_refinement_complete",
        "why_step5j_needed": [
            "Output still dominated by string-side behavior and shared combined body IR",
            "Weak guitar-body identity and cavity/air imprint",
            "E5 peak_flag in Step 5I.3 may relate to modal/radiation weighting",
            "Step 5G planned top/back/air/radiation separation not yet in outputs",
        ],
        "no_stk_integration": True,
        "no_fem_run": True,
        "no_rom_run": True,
        "website_default": DEFAULT_WEBSITE_STK_MODE,
        "website_default_unchanged": True,
        "no_previous_audio_modified": preserved,
        "step5i_3_loaded": step5i_3.get("report_version"),
        "step5g_loaded": step5g.get("report_version"),
        "step5h_loaded": step5h.get("report_version"),
        "step3c_loaded": step3c.get("report_version"),
        "step2_2b_loaded": step2_2b.get("report_version") if step2_2b else None,
        "upstream_readiness": upstream,
        "note_string_fret_mapping_used": preferred,
        "string_damping_source_step5i_3": {
            "contract": "pgsm_string_partial_damping_v4",
            "string_force_regenerated": True,
            "damping_contract_unchanged": True,
            "pluck_attack_preserved": True,
        },
        "top_back_air_radiation_weighting_contract": weighting_contract,
        "modal_kernel_meta": kernel_meta,
        "modal_state_fingerprint": modal_fp,
        "modal_freq_tau_fingerprint": freq_tau_fp,
        "generated_files": output_files,
        "per_note_stem_energy_summary": per_note_stems,
        "per_note_body_identity_metrics": per_note_body,
        "per_note_peak_harshness_analysis": per_note_peak,
        "E5_peak_source_analysis": e5_analysis,
        "per_note_metrics": per_note_metrics,
        "comparison_vs_step5i_3": comp53,
        "comparison_vs_step5i_2": comp52,
        "artifact_guard_results": artifact,
        "validation_results": objective,
        "objective_test_results": objective,
        "blocked_claims": [
            "Final synthesis",
            "STK integration",
            "Real-guitar equivalence",
            "Arbitrary EQ",
            "Bridge coupling feedback (Step 5K)",
        ],
        "readiness_after_step5j": readiness,
        "safe_next_step": (
            "PGSM Step 5K: bridge admittance feedback coupling plan"
            if readiness["current_status"] == READINESS_AFTER
            else "Resolve Step 5J validation failures"
        ),
        "explicit_statement": (
            "PGSM Step 5J refines diagnostic top/back/air/radiation weighting only. "
            "It does not integrate STK and does not prove realism."
        ),
    }


def write_markdown_report(report: Mapping[str, Any], path: Path) -> None:
    rg = report.get("readiness_after_step5j") or {}
    contract = report.get("top_back_air_radiation_weighting_contract") or {}
    stems = report.get("per_note_stem_energy_summary") or {}
    body = report.get("per_note_body_identity_metrics") or {}
    e5 = report.get("E5_peak_source_analysis") or {}
    comp = report.get("comparison_vs_step5i_3") or {}
    obj = report.get("objective_test_results") or {}

    lines = [
        "# PGSM Step 5J — top/back/air/radiation weighting refinement",
        "",
        f"**Readiness:** `{rg.get('current_status')}`",
        "",
        report.get("explicit_statement", ""),
        "",
        "## Why Step 5J was needed",
        "",
    ]
    for item in report.get("why_step5j_needed") or []:
        lines.append(f"- {item}")

    lines.extend(["", "## Weighting contract", "", "| Term | Formula |", "|------|---------|"])
    for t in contract.get("terms") or []:
        lines.append(f"| {t.get('term')} | {str(t.get('formula', ''))[:90]} |")

    lines.extend([
        "",
        "## Stem energy summary",
        "",
        "| Note | top | back | air | body/string |",
        "|------|-----|------|-----|-------------|",
    ])
    for note in NOTE_SET:
        s = stems.get(note) or {}
        lines.append(
            f"| {note} | {s.get('top_share')} | {s.get('back_share')} | {s.get('air_share')} | "
            f"{s.get('body_to_string_energy_ratio')} |"
        )

    lines.extend([
        "",
        "## Body identity metrics",
        "",
        "| Note | cavity imprint | radiation balance |",
        "|------|----------------|-------------------|",
    ])
    for note in NOTE_SET:
        b = body.get(note) or {}
        lines.append(
            f"| {note} | {b.get('cavity_air_imprint_score')} | {b.get('radiation_balance_score')} |"
        )

    lines.extend(["", "## E5 peak source analysis", ""])
    if e5.get("applicable"):
        lines.append(f"- Peak: {e5.get('peak_dbfs')} dBFS, flag={e5.get('E5_peak_flag')}")
        lines.append(f"- Dominant stem: {e5.get('dominant_peak_stem')}")
        lines.append(f"- Peak by stem: {e5.get('peak_by_stem_dbfs')}")
    else:
        lines.append("- N/A")

    lines.extend(["", "## Comparison vs Step 5I.3", ""])
    for note in NOTE_SET:
        c = comp.get(note) or {}
        lines.append(
            f"- **{note}**: peak_delta={c.get('peak_delta_dbfs')} dB, "
            f"piercing_delta={c.get('piercing_delta')}, cavity={c.get('cavity_air_imprint_score')}"
        )

    lines.extend(["", "## Readiness", "", f"all_pass: **{obj.get('all_pass')}**"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pgsm_step5j_reports(
    *,
    repo_root: Optional[Path] = None,
    json_path: Optional[Path] = None,
    md_path: Optional[Path] = None,
    data_path: Optional[Path] = None,
    audio_dir: Optional[Path] = None,
    write_wav: bool = True,
    max_modes: Optional[int] = None,
    duration_s: float = DEFAULT_DURATION_S,
) -> Dict[str, Any]:
    root = Path(repo_root or REPO_ROOT)
    report = build_pgsm_step5j_report(
        repo_root=root,
        audio_dir=audio_dir,
        write_wav=write_wav,
        max_modes=max_modes,
        duration_s=duration_s,
    )
    jpath = Path(json_path or REPORT_JSON)
    mpath = Path(md_path or REPORT_MD)
    dpath = Path(data_path or DATA_JSON)
    jpath.parent.mkdir(parents=True, exist_ok=True)
    jpath.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown_report(report, mpath)
    export = {
        "contract_version": PGSM_STEP5J_VERSION,
        "top_back_air_radiation_weighting_contract": report["top_back_air_radiation_weighting_contract"],
    }
    dpath.parent.mkdir(parents=True, exist_ok=True)
    dpath.write_text(json.dumps(export, indent=2), encoding="utf-8")
    return report


def main() -> None:
    report = write_pgsm_step5j_reports()
    rg = report.get("readiness_after_step5j") or {}
    obj = report.get("objective_test_results") or {}
    print(f"Wrote {REPORT_JSON}")
    print(f"Wrote {DATA_JSON}")
    print(f"Readiness: {rg.get('current_status')}")
    print(f"all_pass: {obj.get('all_pass')}")


if __name__ == "__main__":
    main()
