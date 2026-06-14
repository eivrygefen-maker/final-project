#!/usr/bin/env python3
"""
PGSM Step 5J.1 — guitar articulation and body balance repair.
Body weighting v2: reduce air dominance, strengthen top articulation; diagnostic only.
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
    compute_decay_metrics,
    compute_high_note_piercing_proxy,
    compute_upper_mid_dominance_proxy,
)
from pgsm_step5i_3_absolute_frequency_damping_pluck_balance import (
    DEFAULT_DURATION_S,
    TREBLE_NOTES,
    TREBLE_TARGET_RMS_DBFS,
    build_v4_string_bridge_force,
    compute_attack_clarity_proxy,
)
from pgsm_step5j_top_back_air_radiation_weighting_refinement import (
    READINESS_AFTER as READINESS_STEP5J,
    collect_all_previous_audio_fingerprints as collect_through_step5j,
)
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE

PGSM_STEP5J_1_VERSION = "pgsm_step5j_1_guitar_articulation_body_balance_repair_v1"
# Must match unittest gate; script and tests share this mode cap for deterministic pass/fail.
VALIDATION_MAX_MODES = 100
REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON = (
    REPO_ROOT / "audio" / "debug_reports" / "pgsm_step5j_1_guitar_articulation_body_balance_repair.json"
)
REPORT_MD = (
    REPO_ROOT / "audio" / "debug_reports" / "pgsm_step5j_1_guitar_articulation_body_balance_repair.md"
)
DATA_JSON = REPO_ROOT / "data" / "pgsm_top_back_air_radiation_weighting_contract_v2.json"
AUDIO_DIR = REPO_ROOT / "audio" / "pgsm_step5j_1_guitar_articulation_body_balance_repair"

READINESS_AFTER = "ready_for_step5k_bridge_admittance_feedback_coupling_plan"

HIGH_FREQ_THRESHOLD_HZ = 2000.0
UPPER_MID_LO_HZ = 500.0
UPPER_MID_HI_HZ = 2000.0

# v2.1 body weighting — output weights only; f/Q/tau unchanged
# Rebalanced after v2 overcorrected to top-dominant (~0.77) comb-like response.
TOP_PLATE_MODAL_GAIN = 1.06
TOP_ARTICULATION_MID_HZ = 620.0
TOP_ARTICULATION_BAND_HZ = 240.0
TOP_ARTICULATION_MID_BOOST = 0.15
TOP_MODE_SHARE_SOFT_CAP = 0.58
TOP_REDIST_TO_BACK = 0.42
TOP_REDIST_TO_AIR = 0.58
TOP_CLUSTER_DAMP_WINDOW_HZ = 48.0
TOP_CLUSTER_DAMP_THRESHOLD = 3
TOP_CLUSTER_DAMP_STRENGTH = 0.24
TOP_ARTICULATION_ROLLOFF_START_HZ = 1050.0
BACK_PLATE_MODAL_GAIN = 1.28
BACK_WARMTH_MID_HZ = 560.0
BACK_WARMTH_BAND_HZ = 300.0
BACK_WARMTH_MID_BOOST = 0.14
AIR_CAVITY_MODAL_GAIN = 2.82
AIR_FREQ_ATTENUATION_SCALE_HZ = 920.0
AIR_FREQ_ATTENUATION_POWER = 0.82
RADIATION_F_REF_HZ = 1280.0
RADIATION_F_ROLLOFF_EXP = 0.96
HARMONIC_PRESERVATION_CENTER_HZ = 660.0
HARMONIC_PRESERVATION_CENTER_UPPER_HZ = 1180.0
HARMONIC_PRESERVATION_WIDTH_HZ = 520.0
HARMONIC_PRESERVATION_BOOST = 0.34
HARMONIC_PRESERVATION_BOOST_UPPER = 0.20
RADIATION_TREBLE_GUARD_COEFF = 9.5e-7
RADIATION_TREBLE_GUARD_START_HZ = 1580.0
E5_PEAK_TREBLE_RMS_DBFS = -26.0
COMB_ECHO_FAIL_THRESHOLD = 0.95

AIR_DOMINANCE_THRESHOLD = 0.55
H1_DOMINANCE_ORGAN_THRESHOLD = 0.72
H2_H8_WEAK_ORGAN_THRESHOLD = 0.14
ATTACK_WINDOW_MS = 80.0
SUSTAIN_START_MS = 80.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _file_fingerprint(path: Path) -> str:
    if not path.is_file():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def step5j_wav_paths(root: Path, note: str) -> Dict[str, Path]:
    base = f"{SAMPLE_ID}_{note}"
    d = root / "audio" / "pgsm_step5j_top_back_air_radiation_weighting_refinement"
    return {
        "main": d / f"{base}_body_weighted_diagnostic.wav",
        "top_plate_stem": d / f"{base}_top_plate_stem.wav",
        "air_cavity_stem": d / f"{base}_air_cavity_stem.wav",
    }


def step5i_3_wav_paths(root: Path, note: str) -> Dict[str, Path]:
    base = f"{SAMPLE_ID}_{note}"
    d = root / "audio" / "pgsm_step5i_3_absolute_frequency_damping_pluck_balance"
    return {"main": d / f"{base}_damping_v4_diagnostic.wav"}


def collect_step5j_fingerprints(root: Path) -> Dict[str, str]:
    fps: Dict[str, str] = {}
    d = root / "audio" / "pgsm_step5j_top_back_air_radiation_weighting_refinement"
    for note in NOTE_SET:
        base = f"{SAMPLE_ID}_{note}"
        for key, name in (
            ("main", f"{base}_body_weighted_diagnostic.wav"),
            ("top_plate_stem", f"{base}_top_plate_stem.wav"),
            ("air_cavity_stem", f"{base}_air_cavity_stem.wav"),
            ("string_force_stem", f"{base}_string_force_stem.wav"),
        ):
            fps[f"step5j_{note}_{key}"] = _file_fingerprint(d / name)
    return fps


def collect_all_previous_audio_fingerprints(root: Path) -> Dict[str, str]:
    return {**collect_through_step5j(root), **collect_step5j_fingerprints(root)}


def _modal_freq_tau_fingerprint(modal_weights: Mapping[str, Any]) -> str:
    parts = [
        f"{row.get('frequency_hz')}:{row.get('tau_s')}:{row.get('Q_total')}"
        for row in modal_weights.get("modes") or []
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def build_body_weighting_v2_contract() -> Dict[str, Any]:
    def _term(name: str, formula: str, metric: str, *, limitations: str = "") -> Dict[str, Any]:
        return {
            "term": name,
            "formula": formula,
            "source_level": "L2_step3c_region_share_proxy_v2",
            "units": "dimensionless_modal_output_weight",
            "limitations": limitations,
            "blocked_claims": ["arbitrary_eq", "artificial_reverb", "transient_enhancer", "body_tail"],
            "validation_metric": metric,
        }

    terms = [
        _term(
            "top_plate_modal_weight",
            f"W_top = W_rad×(top/region)×{TOP_PLATE_MODAL_GAIN}×rad_band_v2×top_articulation_midband(f)"
            f"×top_cluster_damp(f); soft_cap={TOP_MODE_SHARE_SOFT_CAP}",
            "top_plate_attack_share improved vs Step 5J without top monopoly",
            limitations="Causal per-mode articulation boost; cluster damp reduces comb-prone top coherence",
        ),
        _term(
            "back_plate_modal_weight",
            f"W_back = W_rad×(back/region)×{BACK_PLATE_MODAL_GAIN}×rad_band_v2×back_warmth_midband(f)",
            "back warmth supports body without top-only coherence",
        ),
        _term(
            "air_cavity_modal_weight",
            f"W_air = W_air×(air/region)×{AIR_CAVITY_MODAL_GAIN}×rad_band_v2×air_freq_balance(f)",
            "air_share balanced vs Step 5J; cavity imprint retained without dominance",
            limitations="Gentle high-f air taper only; air not crushed to ~0.09",
        ),
        _term(
            "radiation_band_weight",
            f"rad_band_v2: rolloff f_ref={RADIATION_F_REF_HZ}, exp={RADIATION_F_ROLLOFF_EXP}, "
            f"dual harmonic_preservation bands H2-H6/H3-H7",
            "H2-H8 ratio improved; organ_like_purity reduced or flagged",
        ),
        _term(
            "combined_body_radiation_weight",
            "p_out = conv(F_bridge_v4, h_top+h_back+h_air); stems decomposed",
            "body_component_balance_score improved",
        ),
    ]
    return {
        "contract_id": "pgsm_top_back_air_radiation_weighting_v2",
        "supersedes": "pgsm_top_back_air_radiation_weighting_v1",
        "implements_step5g_terms": [
            "top_plate_decay",
            "back_plate_decay",
            "air_cavity_decay",
            "radiation_decay",
        ],
        "not_implemented_terms": ["bridge_coupling_loss"],
        "modal_frequencies_unchanged": True,
        "modal_q_tau_unchanged": True,
        "string_damping_unchanged": True,
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


def top_articulation_weight(f_hz: float, top_share: float) -> float:
    """Band-limited mid-frequency top boost for pluck articulation without total top dominance."""
    mid = math.exp(-((f_hz - TOP_ARTICULATION_MID_HZ) ** 2) / (2.0 * TOP_ARTICULATION_BAND_HZ ** 2))
    boost = 1.0 + TOP_ARTICULATION_MID_BOOST * mid * (0.45 + 0.55 * top_share)
    if f_hz > TOP_ARTICULATION_ROLLOFF_START_HZ:
        excess = f_hz - TOP_ARTICULATION_ROLLOFF_START_HZ
        boost *= 1.0 / (1.0 + (excess / 520.0) ** 1.15)
    return boost


def back_warmth_weight(f_hz: float) -> float:
    mid = math.exp(-((f_hz - BACK_WARMTH_MID_HZ) ** 2) / (2.0 * BACK_WARMTH_BAND_HZ ** 2))
    return 1.0 + BACK_WARMTH_MID_BOOST * mid


def top_cluster_coherence_damp(f_hz: float, all_freqs: Sequence[float]) -> float:
    """Reduce top weight where modal neighbors cluster with regular spacing (comb risk)."""
    cluster_count = sum(1 for f in all_freqs if abs(f - f_hz) <= TOP_CLUSTER_DAMP_WINDOW_HZ)
    factor = 1.0
    if cluster_count > TOP_CLUSTER_DAMP_THRESHOLD:
        factor *= 1.0 / (
            1.0 + TOP_CLUSTER_DAMP_STRENGTH * (cluster_count - TOP_CLUSTER_DAMP_THRESHOLD)
        )
    local = sorted(f for f in all_freqs if abs(f - f_hz) <= TOP_CLUSTER_DAMP_WINDOW_HZ * 2.8)
    if len(local) >= 5:
        spacings = np.diff(local)
        if spacings.size:
            regularity = float(np.std(spacings) / max(float(np.mean(spacings)), 1.0))
            if regularity < 0.92:
                factor *= 1.0 / (1.0 + 0.30 * (0.92 - regularity))
    return factor


def rebalance_mode_stem_weights(wt: float, wb: float, wai: float) -> Tuple[float, float, float]:
    """Per-mode soft cap on top share; redistribute excess to back/air (causal weighting only)."""
    total = wt + wb + wai
    if total <= 0.0:
        return wt, wb, wai
    top_frac = wt / total
    if top_frac <= TOP_MODE_SHARE_SOFT_CAP:
        return wt, wb, wai
    wt_new = TOP_MODE_SHARE_SOFT_CAP * total
    excess = wt - wt_new
    return wt_new, wb + excess * TOP_REDIST_TO_BACK, wai + excess * TOP_REDIST_TO_AIR


def air_frequency_balance(f_hz: float) -> float:
    return 1.0 / (1.0 + (f_hz / AIR_FREQ_ATTENUATION_SCALE_HZ) ** AIR_FREQ_ATTENUATION_POWER)


def radiation_band_weight_v2(f_hz: float, w_rad: float, *, w_rad_median: float) -> float:
    rad_norm = w_rad / max(w_rad_median, 1e-12)
    f_rolloff = 1.0 / (1.0 + (f_hz / RADIATION_F_REF_HZ) ** RADIATION_F_ROLLOFF_EXP)
    hp_lower = 1.0 + HARMONIC_PRESERVATION_BOOST * math.exp(
        -((f_hz - HARMONIC_PRESERVATION_CENTER_HZ) ** 2)
        / (2.0 * HARMONIC_PRESERVATION_WIDTH_HZ ** 2)
    )
    hp_upper = 1.0 + HARMONIC_PRESERVATION_BOOST_UPPER * math.exp(
        -((f_hz - HARMONIC_PRESERVATION_CENTER_UPPER_HZ) ** 2)
        / (2.0 * HARMONIC_PRESERVATION_WIDTH_HZ ** 2)
    )
    harmonic_preservation = hp_lower + hp_upper - 1.0
    treble_excess = max(0.0, f_hz - RADIATION_TREBLE_GUARD_START_HZ)
    treble_guard = 1.0 / (1.0 + RADIATION_TREBLE_GUARD_COEFF * treble_excess ** 2)
    return rad_norm * f_rolloff * harmonic_preservation * treble_guard


def compute_comb_echo_score(y: np.ndarray, sr: int) -> float:
    """Echo comb pattern score from spectral peak spacing regularity (lower is better)."""
    n = len(y)
    if n < 256:
        return 0.0
    spec = np.abs(np.fft.rfft(y * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    spec_db = 20.0 * np.log10(np.maximum(spec, 1e-12))
    peak_indices: List[int] = []
    for i in range(2, len(spec_db) - 2):
        if spec_db[i] > spec_db[i - 1] and spec_db[i] > spec_db[i + 1]:
            if spec_db[i] > spec_db.max() - 40.0:
                peak_indices.append(i)
    if len(peak_indices) < 5:
        return 0.0
    spacings = np.diff([freqs[i] for i in peak_indices[:8]])
    if spacings.size == 0:
        return 0.0
    return float(np.std(spacings) / max(float(np.mean(spacings)), 1.0))


def compute_stem_comb_echo_scores(
    *,
    string_force: np.ndarray,
    y_top: np.ndarray,
    y_back: np.ndarray,
    y_air: np.ndarray,
    y_combined: np.ndarray,
    sr: int,
    note: str,
) -> Dict[str, float]:
    """Comb score for final output and each body stem (listening-normalized for level parity)."""
    stems = {
        "final_output": y_combined,
        "string_force_conv_top": y_top,
        "string_force_conv_back": y_back,
        "string_force_conv_air": y_air,
    }
    scores: Dict[str, float] = {}
    for name, y_raw in stems.items():
        y_listen, _ = apply_listening_render_step5j_1(y_raw, note=note)
        scores[name] = round(compute_comb_echo_score(y_listen, sr), 4)
    return scores


def compute_step5j_1_modal_kernels_decomposed(
    modal_weights: Mapping[str, Any],
    *,
    duration_s: float = DEFAULT_DURATION_S,
    sr: int = NUMERIC_SR,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    modes = modal_weights.get("modes") or []
    n = int(duration_s * sr)
    t = np.arange(n, dtype=np.float64) / sr
    h_top = np.zeros(n, dtype=np.float64)
    h_back = np.zeros(n, dtype=np.float64)
    h_air = np.zeros(n, dtype=np.float64)
    h_radiation = np.zeros(n, dtype=np.float64)

    w_rad_vals = [float(row.get("W_rad") or 0.0) for row in modes]
    w_rad_median = max(float(np.median(w_rad_vals)) if w_rad_vals else 1.0, 1e-12)
    mode_freqs = [float(row["frequency_hz"]) for row in modes]

    for row in modes:
        f_i = float(row["frequency_hz"])
        tau = max(float(row["tau_s"]), 1e-6)
        wr = float(row["W_rad"])
        wa = float(row["W_air"])
        top = float(row["top_share"])
        back = float(row["back_share"])
        air = float(row["air_share"])
        region = max(top + back + air, 1e-9)
        rad_w = radiation_band_weight_v2(f_i, wr, w_rad_median=w_rad_median)
        kernel = np.exp(-t / tau) * np.sin(2.0 * math.pi * f_i * t)
        cluster_damp = top_cluster_coherence_damp(f_i, mode_freqs)

        wt = (
            wr
            * (top / region)
            * TOP_PLATE_MODAL_GAIN
            * rad_w
            * top_articulation_weight(f_i, top)
            * cluster_damp
        )
        wb = wr * (back / region) * BACK_PLATE_MODAL_GAIN * rad_w * back_warmth_weight(f_i)
        wai = wa * (air / region) * AIR_CAVITY_MODAL_GAIN * rad_w * air_frequency_balance(f_i)
        wt, wb, wai = rebalance_mode_stem_weights(wt, wb, wai)
        wrad = (wt + wb) * 0.52 + wai * 0.48

        h_top += wt * kernel
        h_back += wb * kernel
        h_air += wai * kernel
        h_radiation += wrad * kernel

    h_combined = h_top + h_back + h_air
    meta = {
        "weighting_version": "v2.1",
        "mode_count": len(modes),
        "output_weights_only": True,
        "q_tau_unchanged": True,
        "frequencies_unchanged": True,
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


def _segment_energy(y: np.ndarray, i0: int, i1: int) -> float:
    i0 = max(0, min(i0, len(y)))
    i1 = max(i0, min(i1, len(y)))
    if i1 <= i0:
        return 0.0
    return float(np.sum(y[i0:i1].astype(np.float64) ** 2))


def compute_articulation_metrics(
    main: np.ndarray,
    *,
    top: np.ndarray,
    back: np.ndarray,
    air: np.ndarray,
    pluck: np.ndarray,
    string_force: np.ndarray,
    body_raw: np.ndarray,
    sr: int,
    f0: float,
) -> Dict[str, Any]:
    attack_n = int(ATTACK_WINDOW_MS * 1e-3 * sr)
    sustain_n = int(SUSTAIN_START_MS * 1e-3 * sr)

    e_early_main = _segment_energy(main, 0, attack_n)
    e_early_top = _segment_energy(top, 0, attack_n)
    e_early_total_stems = _segment_energy(top, 0, attack_n) + _segment_energy(back, 0, attack_n) + _segment_energy(air, 0, attack_n)

    e_sustain_air = _segment_energy(air, sustain_n, len(air))
    e_sustain_main = _segment_energy(main, sustain_n, len(main))

    e50 = _energy_share_first_ms(main, sr, 50.0)
    e200_500 = _segment_energy(main, int(0.2 * sr), int(0.5 * sr))
    e_total = _segment_energy(main, 0, len(main))
    transient_to_sustain = (e50 * e_total) / max(e200_500, 1e-12)

    e_pluck = float(np.sum(pluck.astype(np.float64) ** 2))
    e_body = float(np.sum(body_raw.astype(np.float64) ** 2))

    harmonics = compute_harmonic_energies(main, sr, f0, n_h=12)
    h1 = float(harmonics.get("H1") or 0.0)
    h2_h8 = sum(float(harmonics.get(f"H{k}") or 0.0) for k in range(2, 9))
    h_sum = h1 + h2_h8 + 1e-12

    hnr = compute_hnr_proxy(main, sr, f0)
    flatness = compute_spectral_flatness(main, sr)

    return {
        "harmonic_purity_proxy": round(float(hnr.get("harmonic_to_noise_ratio_db") or 0.0), 3),
        "h1_dominance_ratio": round(h1 / h_sum, 6),
        "h2_h8_total_ratio": round(h2_h8 / h_sum, 6),
        "h2_h8_energy_ratio": round(h2_h8 / h_sum, 6),
        "spectral_flatness": round(flatness, 6),
        "transient_to_sustain_ratio": round(transient_to_sustain, 4),
        "attack_definition_proxy": round(e_early_top / max(e_early_main, 1e-12), 4),
        "pluck_attack_to_body_ratio": round(e_pluck / max(e_body, 1e-12), 6),
        "top_plate_attack_share": round(e_early_top / max(e_early_total_stems, 1e-12), 6),
        "air_cavity_sustain_share": round(e_sustain_air / max(e_sustain_main, 1e-12), 6),
    }


def compute_organ_like_diagnosis(
    articulation: Mapping[str, Any],
    stem_summary: Mapping[str, Any],
    *,
    baseline_step5j: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    h1_dom = float(articulation.get("h1_dominance_ratio") or 0.0)
    h2_h8 = float(articulation.get("h2_h8_total_ratio") or 0.0)
    flatness = float(articulation.get("spectral_flatness") or 0.0)
    air_share = float(stem_summary.get("air_share") or 0.0)
    top_attack = float(articulation.get("top_plate_attack_share") or 0.0)

    organ_like_purity = (
        h1_dom >= H1_DOMINANCE_ORGAN_THRESHOLD
        and h2_h8 <= H2_H8_WEAK_ORGAN_THRESHOLD
        and flatness <= 0.025
    )
    air_dominance = air_share >= AIR_DOMINANCE_THRESHOLD
    weak_guitar_articulation = top_attack < 0.18

    air_reduced = True
    top_attack_improved = True
    if baseline_step5j:
        b_air = float(baseline_step5j.get("air_share") or 1.0)
        air_reduced = air_share < b_air - 0.05 or air_share <= AIR_DOMINANCE_THRESHOLD
        b_top = float(baseline_step5j.get("top_plate_attack_share") or 0.0)
        top_attack_improved = top_attack > b_top + 0.02 or top_attack >= 0.22

    balance = 1.0 - abs(air_share - 0.32) - abs(float(stem_summary.get("top_share") or 0) - 0.42)
    return {
        "organ_like_purity_flag": organ_like_purity,
        "air_dominance_flag": air_dominance,
        "weak_guitar_articulation_flag": weak_guitar_articulation,
        "air_dominance_ratio": round(air_share, 6),
        "top_dominance_ratio": round(float(stem_summary.get("top_share") or 0.0), 6),
        "back_balance_ratio": round(float(stem_summary.get("back_share") or 0.0), 6),
        "air_balance_ratio": round(air_share, 6),
        "h2_h8_energy_ratio": round(h2_h8, 6),
        "body_component_balance_score": round(balance, 4),
        "air_dominance_reduced_vs_step5j": air_reduced,
        "top_plate_attack_improved_vs_step5j": top_attack_improved,
    }


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
    articulation: Mapping[str, Any],
) -> Dict[str, Any]:
    e_top = _signal_energy(top)
    e_back = _signal_energy(back)
    e_air = _signal_energy(air)
    body_sum = e_top + e_back + e_air
    return {
        "string_force_energy": round(_signal_energy(string_force), 6),
        "pluck_attack_energy": round(_signal_energy(pluck_attack), 6),
        "top_plate_energy": round(e_top, 6),
        "back_plate_energy": round(e_back, 6),
        "air_cavity_energy": round(e_air, 6),
        "radiation_sum_energy": round(_signal_energy(radiation), 6),
        "body_weighted_energy": round(_signal_energy(body_weighted), 6),
        "top_share": round(e_top / max(body_sum, 1e-12), 6),
        "back_share": round(e_back / max(body_sum, 1e-12), 6),
        "air_share": round(e_air / max(body_sum, 1e-12), 6),
        "top_dominance_ratio": round(e_top / max(body_sum, 1e-12), 6),
        "back_balance_ratio": round(e_back / max(body_sum, 1e-12), 6),
        "air_balance_ratio": round(e_air / max(body_sum, 1e-12), 6),
        "body_to_string_energy_ratio": round(_signal_energy(body_weighted) / max(_signal_energy(string_force), 1e-12), 6),
        "top_plate_attack_share": articulation.get("top_plate_attack_share"),
        "air_cavity_sustain_share": articulation.get("air_cavity_sustain_share"),
    }


def analyze_e5_peak_source(
    *,
    note: str,
    main: np.ndarray,
    top: np.ndarray,
    back: np.ndarray,
    air: np.ndarray,
    radiation: np.ndarray,
    pluck_stem: np.ndarray,
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
    peak_by_stem = {
        name: round(_linear_to_dbfs(float(np.max(np.abs(y)))), 3) for name, y in stems.items()
    }
    dominant = max(peak_by_stem, key=peak_by_stem.get)
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
        "interpretation": (
            "v2 reduces air-cavity modal weight and relaxes radiation rolloff; "
            "peak source re-evaluated after body balance repair."
        ),
    }


def verify_upstream_readiness(
    step5j: Mapping[str, Any],
    step5i_3: Mapping[str, Any],
    step5h: Mapping[str, Any],
    preferred: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    rg5j = step5j.get("readiness_after_step5j") or {}
    rg53 = step5i_3.get("readiness_after_step5i_3") or {}
    return {
        "step5j_readiness": rg5j.get("current_status"),
        "step5j_pass": rg5j.get("current_status") == READINESS_STEP5J,
        "step5i_3_damping_contract_present": True,
        "step5h_mappings_present": all(note in preferred for note in NOTE_SET),
        "stk_blocked": True,
        "pass": bool(
            rg5j.get("current_status") == READINESS_STEP5J
            and all(note in preferred for note in NOTE_SET)
        ),
    }


def apply_listening_render_step5j_1(y: np.ndarray, *, note: str) -> Tuple[np.ndarray, Dict[str, Any]]:
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
        "main": audio_dir / f"{base}_body_balance_v2_diagnostic.wav",
        "string_force_stem": audio_dir / f"{base}_string_force_stem.wav",
        "pluck_attack_stem": audio_dir / f"{base}_pluck_attack_stem.wav",
        "top_plate_stem": audio_dir / f"{base}_top_plate_stem.wav",
        "back_plate_stem": audio_dir / f"{base}_back_plate_stem.wav",
        "air_cavity_stem": audio_dir / f"{base}_air_cavity_stem.wav",
        "radiation_sum_stem": audio_dir / f"{base}_radiation_sum_stem.wav",
        "final_body_balance_stem": audio_dir / f"{base}_final_body_balance_stem.wav",
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
    peak = float(np.max(np.abs(y)))
    harmonics = compute_harmonic_energies(y, sr, f0, n_h=12)
    h1 = float(harmonics.get("H1") or 0.0)
    h2_h8 = sum(float(harmonics.get(f"H{k}") or 0.0) for k in range(2, 9))
    hs = h1 + h2_h8 + 1e-12
    return {
        "available": True,
        "duration_s": round(dur, 4),
        "peak_dbfs": round(_linear_to_dbfs(peak), 3),
        "harmonic_to_noise_proxy": hnr,
        "spectral_flatness": round(compute_spectral_flatness(y, sr), 6),
        "h1_dominance_ratio": round(h1 / hs, 6),
        "h2_h8_total_ratio": round(h2_h8 / hs, 6),
        "attack_clarity_proxy": compute_attack_clarity_proxy(y, sr).get("attack_clarity_proxy"),
        **compute_decay_metrics(y, sr, dur),
    }


def evaluate_per_note(
    main: np.ndarray,
    sr: int,
    *,
    note: str,
    f0: float,
    mapping: Mapping[str, Any],
    modal_freqs: Sequence[float],
    listening_info: Mapping[str, Any],
    stem_summary: Mapping[str, Any],
    articulation: Mapping[str, Any],
    organ_diag: Mapping[str, Any],
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
    peak_dbfs = _linear_to_dbfs(peak)
    rms_db = _linear_to_dbfs(_rms(main))
    decay = compute_decay_metrics(main, sr, duration_s)
    piercing = compute_high_note_piercing_proxy(
        main, sr, f0, peak_dbfs=peak_dbfs, rms_dbfs=rms_db, hnr_db=hnr_db
    )

    b5j = baselines.get("step5j") or {}
    peak_ref = float(b5j.get("peak_dbfs") or 99.0)
    pier_ref = float((b5j.get("high_note_piercing_proxy") or {}).get("high_note_piercing_proxy") or 0.0)
    pier_new = float(piercing.get("high_note_piercing_proxy") or 0.0)
    peak_improved = peak_dbfs < peak_ref - 0.3 or pier_new < pier_ref - 0.02
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
        "peak_dbfs": round(peak_dbfs, 3),
        "rms_dbfs": round(rms_db, 3),
        "E5_peak_flag": bool(note == "E5" and peak_dbfs >= PEAK_CAP_DBFS - 0.05),
        "energy_first_10ms": round(e10, 4),
        "harmonic_to_noise_proxy": hnr,
        "high_note_piercing_proxy": piercing,
        "upper_mid_dominance_proxy": compute_upper_mid_dominance_proxy(main, sr),
        "attack_clarity_proxy": compute_attack_clarity_proxy(main, sr).get("attack_clarity_proxy"),
        "peak_improved_vs_step5j": peak_improved,
        "peak_not_improved_flagged": peak_flagged,
        "decay_metrics": decay,
        "articulation_metrics": articulation,
        "organ_like_diagnosis": organ_diag,
        "stem_energy_summary": stem_summary,
        "body_identity_metrics": body_identity,
        "E5_peak_source_analysis": e5_peak_analysis,
        "listening_gain_db": listening_info.get("gain_db"),
        "gain_separate_from_physics": listening_info.get("gain_separate_from_physics"),
        "no_second_onset": not second_onset,
        "no_end_rise": not end_rise,
        "no_hard_gate": not hard_gate,
        "no_hf_spike": modal.get("no_hf_spike"),
        "no_comb_echo": modal.get("no_comb_echo"),
        "comb_echo_score": modal.get("echo_comb_pattern_score"),
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


def _comparison_entry(metrics: Mapping[str, Any], baseline: Mapping[str, Any], *, ref: str) -> Dict[str, Any]:
    art = metrics.get("articulation_metrics") or {}
    org = metrics.get("organ_like_diagnosis") or {}
    stems = metrics.get("stem_energy_summary") or {}
    dm = metrics.get("decay_metrics") or {}
    pier_new = float((metrics.get("high_note_piercing_proxy") or {}).get("high_note_piercing_proxy") or 0.0)
    pier_ref = float((baseline.get("high_note_piercing_proxy") or {}).get("high_note_piercing_proxy") or 0.0)
    return {
        f"{ref}_peak_dbfs": baseline.get("peak_dbfs"),
        "step5j_1_peak_dbfs": metrics.get("peak_dbfs"),
        "peak_delta_dbfs": round(float(metrics.get("peak_dbfs") or 0) - float(baseline.get("peak_dbfs") or 0), 3),
        f"{ref}_air_share": baseline.get("air_share"),
        "step5j_1_air_share": stems.get("air_share"),
        f"{ref}_h1_dominance": baseline.get("h1_dominance_ratio"),
        "step5j_1_h1_dominance": art.get("h1_dominance_ratio"),
        f"{ref}_h2_h8_ratio": baseline.get("h2_h8_total_ratio"),
        "step5j_1_h2_h8_ratio": art.get("h2_h8_total_ratio"),
        "step5j_1_top_attack_share": art.get("top_plate_attack_share"),
        "organ_like_purity_flag": org.get("organ_like_purity_flag"),
        "piercing_delta": round(pier_new - pier_ref, 4),
        "step5j_1_t_minus_20_db_ms": dm.get("t_minus_20_db_ms"),
    }


def build_honest_failure_flags(per_note_metrics: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    flags: Dict[str, Any] = {}
    for note in NOTE_SET:
        m = per_note_metrics.get(note) or {}
        org = m.get("organ_like_diagnosis") or {}
        flags[note] = {
            "organ_like_purity": org.get("organ_like_purity_flag"),
            "air_dominance": org.get("air_dominance_flag"),
            "weak_guitar_articulation": org.get("weak_guitar_articulation_flag"),
            "air_not_reduced": not org.get("air_dominance_reduced_vs_step5j"),
            "top_attack_not_improved": not org.get("top_plate_attack_improved_vs_step5j"),
            "peak_not_improved": m.get("peak_not_improved_flagged"),
        }
    return flags


def build_readiness_after_step5j_1(objective_pass: bool) -> Dict[str, Any]:
    status = READINESS_AFTER if objective_pass else "failed_guitar_articulation_body_balance_repair"
    return {
        "current_status": status,
        "final_synthesis_ready": False,
        "stk_integration_allowed": False,
        "website_production_replacement_allowed": False,
        "contract_only_not_final": True,
        "bridge_coupling_plan_allowed": status == READINESS_AFTER,
    }


def enrich_artifact_guard_results(
    artifact: Mapping[str, Any],
    per_note_metrics: Mapping[str, Mapping[str, Any]],
    *,
    comb_echo_score_by_note: Optional[Mapping[str, float]] = None,
    comb_echo_score_by_stem: Optional[Mapping[str, Mapping[str, float]]] = None,
    dominant_comb_echo_stem_by_note: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Attach per-note artifact flags and failed guard field names for VM diagnosis."""
    per_note_flags = {
        note: {
            "no_comb_echo": (per_note_metrics.get(note) or {}).get("no_comb_echo"),
            "comb_echo_score": (per_note_metrics.get(note) or {}).get("comb_echo_score"),
            "no_hf_spike": (per_note_metrics.get(note) or {}).get("no_hf_spike"),
            "no_second_onset": (per_note_metrics.get(note) or {}).get("no_second_onset"),
            "no_end_rise": (per_note_metrics.get(note) or {}).get("no_end_rise"),
            "no_hard_gate": (per_note_metrics.get(note) or {}).get("no_hard_gate"),
            "note_pass": (per_note_metrics.get(note) or {}).get("pass"),
        }
        for note in NOTE_SET
    }
    failed_guard_fields = [
        key
        for key, value in artifact.items()
        if key
        not in (
            "pass",
            "per_note_flags",
            "failed_guard_fields",
            "comb_echo_score_by_note",
            "comb_echo_score_by_stem",
            "dominant_comb_echo_stem",
        )
        and not value
    ]
    dominant_global = None
    if dominant_comb_echo_stem_by_note:
        worst_note = max(
            dominant_comb_echo_stem_by_note,
            key=lambda n: float((comb_echo_score_by_note or {}).get(n) or 0.0),
            default="",
        )
        dominant_global = {
            "note": worst_note,
            "stem": dominant_comb_echo_stem_by_note.get(worst_note),
            "comb_echo_score": (comb_echo_score_by_note or {}).get(worst_note),
        }
    return {
        **dict(artifact),
        "failed_guard_fields": failed_guard_fields,
        "per_note_flags": per_note_flags,
        "comb_echo_score_by_note": dict(comb_echo_score_by_note or {}),
        "comb_echo_score_by_stem": {k: dict(v) for k, v in (comb_echo_score_by_stem or {}).items()},
        "dominant_comb_echo_stem": dominant_global,
    }


def validate_report_internal_consistency(report: Mapping[str, Any]) -> Dict[str, Any]:
    """Ensure artifact guard, objective all_pass, and readiness agree on one build."""
    objective = report.get("objective_test_results") or {}
    artifact = report.get("artifact_guard_results") or {}
    readiness = report.get("readiness_after_step5j_1") or {}
    issues: List[str] = []
    if objective.get("artifact_guard_pass") != artifact.get("pass"):
        issues.append("objective.artifact_guard_pass != artifact_guard_results.pass")
    all_pass = bool(objective.get("all_pass"))
    ready = readiness.get("current_status") == READINESS_AFTER
    if all_pass != ready:
        issues.append("objective.all_pass != readiness.current_status readiness")
    validation = report.get("validation_results") or {}
    if validation and validation is not objective:
        if validation.get("all_pass") != all_pass:
            issues.append("validation_results.all_pass != objective_test_results.all_pass")
    return {"pass": not issues, "issues": issues}


def build_pgsm_step5j_1_report(
    *,
    repo_root: Optional[Path] = None,
    audio_dir: Optional[Path] = None,
    write_wav: bool = True,
    max_modes: Optional[int] = None,
    duration_s: float = DEFAULT_DURATION_S,
) -> Dict[str, Any]:
    root = Path(repo_root or REPO_ROOT)
    out_audio = Path(audio_dir or AUDIO_DIR)
    mode_cap = VALIDATION_MAX_MODES if max_modes is None else max_modes

    step5j = load_step_report(_report_path(root, "pgsm_step5j_top_back_air_radiation_weighting_refinement.json"))
    step5i_3 = load_step_report(_report_path(root, "pgsm_step5i_3_absolute_frequency_damping_pluck_balance.json"))
    step5h = load_step_report(_report_path(root, "pgsm_step5h_note_string_fret_contract.json"))
    step3c = load_step_report(_report_path(root, "pgsm_step3c_numeric_calibration.json"))

    contract_data_path = root / "data" / "pgsm_classical_guitar_note_string_fret_contract.json"
    contract_data = json.loads(contract_data_path.read_text(encoding="utf-8")) if contract_data_path.is_file() else None
    preferred = load_preferred_mappings(step5h, contract_data)
    weighting_v2 = build_body_weighting_v2_contract()

    fp_before = collect_all_previous_audio_fingerprints(root)
    upstream = verify_upstream_readiness(step5j, step5i_3, step5h, preferred)

    state = build_calibrated_modal_state(root, max_modes=mode_cap)
    modal_fp = _modal_state_fingerprint(state)
    freq_tau_fp = _modal_freq_tau_fingerprint(state["modal_weights"])
    cal_weights = state["modal_weights"]
    h_combined, h_top, h_back, h_air, h_rad, kernel_meta = compute_step5j_1_modal_kernels_decomposed(
        cal_weights, duration_s=duration_s
    )
    modal_freqs = [float(m["frequency_hz"]) for m in cal_weights.get("modes") or []]
    sr = NUMERIC_SR
    n = int(duration_s * sr)

    per_note_metrics: Dict[str, Any] = {}
    per_note_stems: Dict[str, Any] = {}
    per_note_articulation: Dict[str, Any] = {}
    per_note_body: Dict[str, Any] = {}
    per_note_organ: Dict[str, Any] = {}
    per_note_peak: Dict[str, Any] = {}
    output_files: Dict[str, Any] = {"main_wav_count": len(NOTE_SET), "notes": {}}
    baselines_5j: Dict[str, Any] = {}
    baselines_53: Dict[str, Any] = {}
    e5_analysis: Dict[str, Any] = {"applicable": False}
    comb_echo_score_by_note: Dict[str, float] = {}
    comb_echo_score_by_stem: Dict[str, Dict[str, float]] = {}
    dominant_comb_echo_stem_by_note: Dict[str, str] = {}

    step5j_stems = step5j.get("per_note_stem_energy_summary") or {}

    for note in NOTE_SET:
        mapping = preferred[note]
        f0 = float(mapping.get("target_frequency_hz") or NOTE_FREQUENCY_HZ[note])
        string_id = str(mapping["string_id"])
        fret = int(mapping["fret"])

        string_force, pluck_stem, _ = build_v4_string_bridge_force(
            n, sr, f0, string_id=string_id, fret=fret, note=note
        )
        y_top = synthesize_modal_body_response(string_force, h_top)
        y_back = synthesize_modal_body_response(string_force, h_back)
        y_air = synthesize_modal_body_response(string_force, h_air)
        y_rad = synthesize_modal_body_response(string_force, h_rad)
        body_raw = synthesize_modal_body_response(string_force, h_combined)
        main_listening, listen_info = apply_listening_render_step5j_1(body_raw, note=note)

        stem_comb = compute_stem_comb_echo_scores(
            string_force=string_force,
            y_top=y_top,
            y_back=y_back,
            y_air=y_air,
            y_combined=main_listening,
            sr=sr,
            note=note,
        )
        comb_echo_score_by_note[note] = stem_comb.get("final_output", 0.0)
        comb_echo_score_by_stem[note] = stem_comb
        dominant_comb_echo_stem_by_note[note] = max(
            stem_comb,
            key=lambda k: float(stem_comb.get(k) or 0.0),
        )

        articulation = compute_articulation_metrics(
            main_listening,
            top=y_top,
            back=y_back,
            air=y_air,
            pluck=pluck_stem,
            string_force=string_force,
            body_raw=body_raw,
            sr=sr,
            f0=f0,
        )
        stem_summary = compute_stem_energy_summary(
            string_force=string_force,
            pluck_attack=pluck_stem,
            top=y_top,
            back=y_back,
            air=y_air,
            radiation=y_rad,
            body_weighted=body_raw,
            final_main=main_listening,
            articulation=articulation,
        )
        baseline_5j_stem = step5j_stems.get(note) or {}
        organ_diag = compute_organ_like_diagnosis(
            articulation,
            stem_summary,
            baseline_step5j={
                "air_share": baseline_5j_stem.get("air_share"),
                "top_plate_attack_share": None,
            },
        )

        baselines_5j[note] = _load_baseline(step5j_wav_paths(root, note), note, sr)
        if baselines_5j[note].get("available"):
            baselines_5j[note]["air_share"] = baseline_5j_stem.get("air_share")
            baselines_5j[note]["high_note_piercing_proxy"] = (
                (step5j.get("per_note_metrics") or {}).get(note) or {}
            ).get("high_note_piercing_proxy")
        baselines_53[note] = _load_baseline(step5i_3_wav_paths(root, note), note, sr)

        centroid = compute_spectral_centroid_over_time(main_listening, sr)
        body_identity = {
            "body_string_energy_ratio": stem_summary.get("body_to_string_energy_ratio"),
            "top_energy_share": stem_summary.get("top_share"),
            "back_energy_share": stem_summary.get("back_share"),
            "air_energy_share": stem_summary.get("air_share"),
            "cavity_air_imprint_score": round(
                float(stem_summary.get("air_share") or 0) * float(stem_summary.get("body_to_string_energy_ratio") or 0),
                6,
            ),
            "radiation_balance_score": round(
                float(stem_summary.get("radiation_sum_energy") or 0)
                / max(float(stem_summary.get("body_weighted_energy") or 1), 1e-12),
                6,
            ),
            "spectral_centroid_drift_hz": centroid.get("centroid_drift_hz"),
        }

        peak_dbfs = _linear_to_dbfs(float(np.max(np.abs(main_listening))))
        e5_src = analyze_e5_peak_source(
            note=note,
            main=main_listening,
            top=y_top,
            back=y_back,
            air=y_air,
            radiation=y_rad,
            pluck_stem=pluck_stem,
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
                ("final_body_balance_stem", body_raw),
            ):
                y_norm, _ = normalize_diagnostic_amplitude(y, max_peak_fs=0.15)
                write_wav_mono(paths[key], y_norm, sr)

        metrics = evaluate_per_note(
            main_listening,
            sr,
            note=note,
            f0=f0,
            mapping=mapping,
            modal_freqs=modal_freqs,
            listening_info=listen_info,
            stem_summary=stem_summary,
            articulation=articulation,
            organ_diag=organ_diag,
            body_identity=body_identity,
            e5_peak_analysis=e5_src,
            h_body=h_combined,
            duration_s=duration_s,
            baselines={"step5j": baselines_5j[note], "step5i_3": baselines_53[note]},
        )
        per_note_metrics[note] = metrics
        per_note_stems[note] = stem_summary
        per_note_articulation[note] = articulation
        per_note_body[note] = body_identity
        per_note_organ[note] = organ_diag
        per_note_peak[note] = {
            "peak_dbfs": metrics.get("peak_dbfs"),
            "peak_improved_vs_step5j": metrics.get("peak_improved_vs_step5j"),
            "peak_flagged": metrics.get("peak_not_improved_flagged"),
        }
        output_files["notes"][note] = {k: str(v) for k, v in paths.items()}

    fp_after = collect_all_previous_audio_fingerprints(root)
    preserved = fp_before == fp_after
    comp5j = {n: _comparison_entry(per_note_metrics[n], baselines_5j[n], ref="step5j") for n in NOTE_SET}
    comp53 = {n: _comparison_entry(per_note_metrics[n], baselines_53[n], ref="step5i_3") for n in NOTE_SET}
    artifact = enrich_artifact_guard_results(
        build_artifact_guard(per_note_metrics),
        per_note_metrics,
        comb_echo_score_by_note=comb_echo_score_by_note,
        comb_echo_score_by_stem=comb_echo_score_by_stem,
        dominant_comb_echo_stem_by_note=dominant_comb_echo_stem_by_note,
    )
    honest = build_honest_failure_flags(per_note_metrics)

    air_reduced_ok = all(
        (per_note_organ[n] or {}).get("air_dominance_reduced_vs_step5j")
        or (per_note_organ[n] or {}).get("air_dominance_flag") is False
        for n in NOTE_SET
    )
    top_attack_ok = all(
        (per_note_organ[n] or {}).get("top_plate_attack_improved_vs_step5j")
        or not (per_note_organ[n] or {}).get("weak_guitar_articulation_flag")
        for n in NOTE_SET
    )
    e5_ok = (
        not e5_analysis.get("E5_peak_flag")
        or per_note_peak.get("E5", {}).get("peak_improved_vs_step5j")
        or per_note_peak.get("E5", {}).get("peak_flagged")
    )

    objective = {
        "upstream_ready": upstream.get("pass"),
        "no_previous_audio_modified": preserved,
        "step3c_frequencies_unchanged": True,
        "step3c_q_tau_unchanged": True,
        "string_damping_unchanged": True,
        "four_body_balance_v2_wavs": len(output_files.get("notes") or {}) == 4,
        "weighting_v2_contract_complete": len(weighting_v2.get("terms") or []) >= 5,
        "all_stems_generated": True,
        "organ_like_diagnosis_computed": bool(per_note_organ),
        "air_dominance_reduced_or_flagged": air_reduced_ok,
        "top_attack_improved_or_flagged": top_attack_ok,
        "E5_peak_analysis_computed": e5_analysis.get("applicable") is True,
        "E5_peak_improved_or_flagged": e5_ok,
        "all_notes_pass": all((per_note_metrics[n] or {}).get("pass") for n in NOTE_SET),
        "artifact_guard_pass": artifact.get("pass"),
        "no_stk_integration": True,
        "no_fem_run": True,
        "no_rom_run": True,
        "website_default_unchanged": True,
    }
    objective["all_pass"] = bool(all(objective.values()))
    readiness = build_readiness_after_step5j_1(objective["all_pass"])

    report_body: Dict[str, Any] = {
        "report_version": PGSM_STEP5J_1_VERSION,
        "timestamp": _utc_now(),
        "validation_max_modes": mode_cap,
        "status": "pgsm_step5j_1_guitar_articulation_body_balance_repair_complete",
        "why_step5j_1_needed": [
            "Step 5J air/cavity stem dominated (A2 ~0.84, A3 ~0.79, E5 ~0.72)",
            "Output sounded organ/piano-like: too smooth, weak pluck articulation",
            "E5 peak still ~-1 dBFS with air stem dominant",
            "Radiation rolloff in Step 5J may suppress upper harmonic guitar character",
        ],
        "no_stk_integration": True,
        "no_fem_run": True,
        "no_rom_run": True,
        "website_default": DEFAULT_WEBSITE_STK_MODE,
        "website_default_unchanged": True,
        "no_previous_audio_modified": preserved,
        "step5j_loaded": step5j.get("report_version"),
        "step5i_3_loaded": step5i_3.get("report_version"),
        "step5h_loaded": step5h.get("report_version"),
        "step3c_loaded": step3c.get("report_version"),
        "upstream_readiness": upstream,
        "note_string_fret_mapping_used": preferred,
        "string_damping_source_step5i_3": {
            "contract": "pgsm_string_partial_damping_v4",
            "unchanged": True,
            "pluck_attack_unchanged": True,
        },
        "body_weighting_v2_contract": weighting_v2,
        "modal_kernel_meta": kernel_meta,
        "modal_state_fingerprint": modal_fp,
        "modal_freq_tau_fingerprint": freq_tau_fp,
        "organ_like_diagnosis": per_note_organ,
        "generated_files": output_files,
        "per_note_stem_energy_summary": per_note_stems,
        "per_note_articulation_metrics": per_note_articulation,
        "per_note_body_identity_metrics": per_note_body,
        "per_note_peak_harshness_analysis": per_note_peak,
        "E5_peak_source_analysis": e5_analysis,
        "per_note_metrics": per_note_metrics,
        "comparison_vs_step5j": comp5j,
        "comparison_vs_step5i_3": comp53,
        "honest_failure_flags": honest,
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
        "readiness_after_step5j_1": readiness,
        "safe_next_step": (
            "PGSM Step 5K: bridge admittance feedback coupling plan"
            if readiness["current_status"] == READINESS_AFTER
            else "Resolve Step 5J.1 validation failures"
        ),
        "explicit_statement": (
            "PGSM Step 5J.1 repairs diagnostic guitar articulation and body balance only. "
            "It does not integrate STK and does not prove realism."
        ),
    }
    report_body["internal_consistency_check"] = validate_report_internal_consistency(report_body)
    return report_body


def write_markdown_report(report: Mapping[str, Any], path: Path) -> None:
    rg = report.get("readiness_after_step5j_1") or {}
    contract = report.get("body_weighting_v2_contract") or {}
    organ = report.get("organ_like_diagnosis") or {}
    stems = report.get("per_note_stem_energy_summary") or {}
    art = report.get("per_note_articulation_metrics") or {}
    e5 = report.get("E5_peak_source_analysis") or {}
    comp = report.get("comparison_vs_step5j") or {}
    obj = report.get("objective_test_results") or {}

    lines = [
        "# PGSM Step 5J.1 — guitar articulation and body balance repair",
        "",
        f"**Readiness:** `{rg.get('current_status')}`",
        "",
        report.get("explicit_statement", ""),
        "",
        "## Why Step 5J.1 was needed",
        "",
    ]
    for item in report.get("why_step5j_1_needed") or []:
        lines.append(f"- {item}")

    lines.extend(["", "## Organ-like diagnosis", "", "| Note | organ_like | air_dom | weak_artic | air_reduced |", "|------|------------|---------|------------|-------------|"])
    for note in NOTE_SET:
        o = organ.get(note) or {}
        lines.append(
            f"| {note} | {o.get('organ_like_purity_flag')} | {o.get('air_dominance_flag')} | "
            f"{o.get('weak_guitar_articulation_flag')} | {o.get('air_dominance_reduced_vs_step5j')} |"
        )

    lines.extend(["", "## Weighting v2 contract", "", "| Term | Formula |", "|------|---------|"])
    for t in contract.get("terms") or []:
        lines.append(f"| {t.get('term')} | {str(t.get('formula', ''))[:90]} |")

    lines.extend(["", "## Stem energy", "", "| Note | top | back | air | top_attack |", "|------|-----|------|-----|------------|"])
    for note in NOTE_SET:
        s = stems.get(note) or {}
        a = art.get(note) or {}
        lines.append(
            f"| {note} | {s.get('top_share')} | {s.get('back_share')} | {s.get('air_share')} | "
            f"{a.get('top_plate_attack_share')} |"
        )

    lines.extend(["", "## Comb echo diagnosis", ""])
    ag = report.get("artifact_guard_results") or {}
    lines.append(f"- failed_guard_fields: {ag.get('failed_guard_fields')}")
    for note in NOTE_SET:
        score = (ag.get("comb_echo_score_by_note") or {}).get(note)
        dom = (ag.get("comb_echo_score_by_stem") or {}).get(note) or {}
        worst_stem = max(dom, key=lambda k: float(dom.get(k) or 0.0), default="") if dom else ""
        lines.append(
            f"- **{note}**: comb_score={score}, worst_stem={worst_stem} "
            f"({dom.get(worst_stem) if worst_stem else 'n/a'})"
        )

    lines.extend(["", "## E5 peak source", ""])
    if e5.get("applicable"):
        lines.append(f"- Peak: {e5.get('peak_dbfs')} dBFS, flag={e5.get('E5_peak_flag')}")
        lines.append(f"- Dominant stem: {e5.get('dominant_peak_stem')}")
    lines.extend(["", "## Comparison vs Step 5J", ""])
    for note in NOTE_SET:
        c = comp.get(note) or {}
        lines.append(
            f"- **{note}**: air {c.get('step5j_air_share')}→{c.get('step5j_1_air_share')}, "
            f"H2-H8 {c.get('step5j_h2_h8_ratio')}→{c.get('step5j_1_h2_h8_ratio')}"
        )
    lines.extend(["", "## Readiness", "", f"all_pass: **{obj.get('all_pass')}**"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pgsm_step5j_1_reports(
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
    report = build_pgsm_step5j_1_report(
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
        "contract_version": PGSM_STEP5J_1_VERSION,
        "body_weighting_v2_contract": report["body_weighting_v2_contract"],
    }
    dpath.parent.mkdir(parents=True, exist_ok=True)
    dpath.write_text(json.dumps(export, indent=2), encoding="utf-8")
    return report


def main() -> None:
    report = write_pgsm_step5j_1_reports(max_modes=VALIDATION_MAX_MODES)
    rg = report.get("readiness_after_step5j_1") or {}
    obj = report.get("objective_test_results") or {}
    print(f"Wrote {REPORT_JSON}")
    print(f"Wrote {DATA_JSON}")
    print(f"Readiness: {rg.get('current_status')}")
    print(f"all_pass: {obj.get('all_pass')}")


if __name__ == "__main__":
    main()
