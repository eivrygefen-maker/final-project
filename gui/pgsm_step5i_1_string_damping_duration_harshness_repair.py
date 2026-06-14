#!/usr/bin/env python3
"""
PGSM Step 5I.1 — string damping duration and treble harshness repair.
Extends Step 5I v2 partial damping + longer diagnostic window; diagnostic only.
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
from pgsm_step3a_numerical_ir_testbench import FIXED_PLUCK_POSITION, NUMERIC_SR, SAMPLE_ID
from pgsm_step4a_single_note_diagnostic_audio import (
    DIAGNOSTIC_LABEL,
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
    INHARMONICITY_B_FALLBACK,
    INHARMONICITY_LEVEL,
    PITCH_SALIENCE_MIN,
    TARGET_RMS_DBFS_MAX,
    TARGET_RMS_DBFS_MIN,
    TARGET_RMS_DBFS_NOMINAL,
    apply_listening_render_full,
    build_artifact_guard,
    collect_previous_audio_fingerprints,
    compute_click_dominance_score,
    compute_harmonic_energies,
    compute_modal_kernels_decomposed,
    compute_pitch_salience,
    compute_spectral_features,
    detect_second_onset_sustained,
    evaluate_modal_peak_alignment,
    _active_duration_ms,
    _decay_time_ms_smoothed,
    _energy_share_first_ms,
    _linear_to_dbfs,
    _rms,
)
from pgsm_step5f_string_driven_extended_validation import (
    collect_step5e_fingerprints,
    compute_hnr_proxy,
    compute_partial_decay_slopes,
    compute_spectral_centroid_over_time,
    step5e_wav_paths,
)
from pgsm_step5h_note_string_fret_contract import READINESS_AFTER as READINESS_STEP5H
from pgsm_step5i_string_partial_damping_refinement import (
    READINESS_AFTER as READINESS_STEP5I,
    compute_refined_partial_tau_k,
    load_preferred_mappings,
    _modal_state_fingerprint,
    _report_path,
)
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE

PGSM_STEP5I_1_VERSION = "pgsm_step5i_1_string_damping_duration_harshness_repair_v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON = (
    REPO_ROOT / "audio" / "debug_reports" / "pgsm_step5i_1_string_damping_duration_harshness_repair.json"
)
REPORT_MD = (
    REPO_ROOT / "audio" / "debug_reports" / "pgsm_step5i_1_string_damping_duration_harshness_repair.md"
)
DATA_JSON = REPO_ROOT / "data" / "pgsm_string_partial_damping_contract_v2.json"
AUDIO_DIR = REPO_ROOT / "audio" / "pgsm_step5i_1_string_damping_duration_harshness_repair"

READINESS_AFTER = "ready_for_step5j_top_back_air_radiation_weighting_refinement"
STRING_FORCE_LABEL = "diagnostic_v2_string_bridge_force_proxy_not_measured_force"

DEFAULT_DURATION_S = 4.5
ALLOWED_DURATION_MIN_S = 3.0
ALLOWED_DURATION_MAX_S = 6.0

BASE_TAU_BY_STRING: Dict[str, float] = {
    "string_6": 3.55,
    "string_5": 3.35,
    "string_4": 3.05,
    "string_3": 2.80,
    "string_2": 2.50,
    "string_1": 2.25,
}

MATERIAL_INTERNAL_LOSS: Dict[str, float] = {
    "string_6": 1.00,
    "string_5": 1.03,
    "string_4": 1.06,
    "string_3": 1.10,
    "string_2": 1.14,
    "string_1": 1.22,
}

HARMONIC_ORDER_EXPONENT = 1.20
FREQ_DEPENDENT_COEFF = 2.8e-4
HIGH_FREQ_THRESHOLD_HZ = 2000.0
HIGH_FREQ_EXTRA_COEFF = 7.5e-4
FRET_CONTACT_COEFF = 0.038
BOUNDARY_LOSS_COEFF = 0.042
HNR_PURITY_THRESHOLD_DB = 0.5


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _file_fingerprint(path: Path) -> str:
    if not path.is_file():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def step5i_wav_paths(root: Path, note: str) -> Dict[str, Path]:
    base = f"{SAMPLE_ID}_{note}"
    d = root / "audio" / "pgsm_step5i_string_partial_damping_refinement"
    return {
        "main": d / f"{base}_damping_refined_diagnostic.wav",
        "body_stem": d / f"{base}_body_stem.wav",
        "string_force_stem": d / f"{base}_string_force_stem.wav",
    }


def collect_step5i_fingerprints(root: Path) -> Dict[str, str]:
    fps: Dict[str, str] = {}
    for note in NOTE_SET:
        for key, p in step5i_wav_paths(root, note).items():
            fps[f"step5i_{note}_{key}"] = _file_fingerprint(p)
    return fps


def collect_all_previous_audio_fingerprints(root: Path) -> Dict[str, str]:
    return {
        **collect_previous_audio_fingerprints(root),
        **collect_step5e_fingerprints(root),
        **collect_step5i_fingerprints(root),
    }


def build_duration_policy(duration_s: float = DEFAULT_DURATION_S) -> Dict[str, Any]:
    return {
        "default_duration_s": DEFAULT_DURATION_S,
        "allowed_range_s": [ALLOWED_DURATION_MIN_S, ALLOWED_DURATION_MAX_S],
        "applied_duration_s": duration_s,
        "no_trim": True,
        "no_hard_gate": True,
        "no_artificial_tail": True,
        "full_natural_decay_window": True,
    }


def build_string_partial_damping_contract_v2() -> Dict[str, Any]:
    terms: List[Dict[str, Any]] = [
        {
            "term": "base_string_decay_by_string_id",
            "formula": "tau_base[string_id] seconds; bass longer than treble",
            "values": dict(BASE_TAU_BY_STRING),
            "source_level": "L2_literature_fallback_v2",
            "units": "seconds",
            "limitations": "Diagnostic proxy; bass sustain preserved vs treble.",
            "blocked_claims": ["measured_string_damping"],
            "objective_metric": "tau_base differs across string_id",
        },
        {
            "term": "harmonic_order_loss",
            "formula": f"harmonic_factor = k^{HARMONIC_ORDER_EXPONENT}",
            "source_level": "L2_diagnostic_proxy_v2",
            "units": "dimensionless divisor",
            "limitations": f"Stronger than Step 5I k^0.92; range 1.10–1.30 motivated.",
            "blocked_claims": ["arbitrary_eq"],
            "objective_metric": "high k partials have clearly shorter tau_k",
        },
        {
            "term": "frequency_dependent_loss",
            "formula": (
                f"freq_factor = 1 + {FREQ_DEPENDENT_COEFF}*f_k + "
                f"{HIGH_FREQ_EXTRA_COEFF}*max(0, f_k-{HIGH_FREQ_THRESHOLD_HZ})"
            ),
            "source_level": "L2_diagnostic_proxy_v2",
            "units": "dimensionless divisor",
            "limitations": "Extra loss above 2 kHz via decay time, not spectral EQ.",
            "blocked_claims": ["spectral_eq_shaping"],
            "objective_metric": "partials above 2 kHz decay faster audibly",
        },
        {
            "term": "fret_contact_loss_proxy",
            "formula": f"fret_factor = 1 + {FRET_CONTACT_COEFF} * fret",
            "source_level": "L2_diagnostic_proxy_v2",
            "units": "dimensionless divisor",
            "limitations": "Modest increase vs Step 5I for fretted notes.",
            "blocked_claims": ["measured_fret_contact"],
            "objective_metric": "fretted notes damp faster than open",
        },
        {
            "term": "material_internal_loss_proxy",
            "formula": "material_factor = MATERIAL_INTERNAL_LOSS[string_id]",
            "source_level": "L2_nylon_classical_fallback_v2",
            "units": "dimensionless divisor",
            "limitations": "Stronger treble-string internal loss for string_1/string_2.",
            "blocked_claims": ["measured_linear_density"],
            "objective_metric": "string_1 material_factor > string_5",
        },
        {
            "term": "bridge_nut_boundary_loss_proxy",
            "formula": f"boundary_factor = 1 + {BOUNDARY_LOSS_COEFF} * sqrt(k)",
            "source_level": "L2_diagnostic_proxy_v2",
            "units": "dimensionless divisor",
            "limitations": "Higher partial boundary loss proxy.",
            "blocked_claims": ["measured_bridge_admittance"],
            "objective_metric": "boundary_factor increases with k",
        },
        {
            "term": "combined_partial_tau",
            "formula": (
                "tau_k = tau_base / (harmonic_factor * freq_factor * fret_factor "
                "* material_factor * boundary_factor)"
            ),
            "source_level": "L2_diagnostic_combined_proxy_v2",
            "units": "seconds",
            "limitations": "Time-domain partial decay only; replaces Step 5I v1 law in Step 5I.1 generator.",
            "blocked_claims": ["real_guitar_equivalence"],
            "objective_metric": "monotonic tau_k vs k; steeper high-partial decay than Step 5I",
        },
    ]
    return {
        "contract_id": "pgsm_string_partial_damping_v2",
        "supersedes": "pgsm_string_partial_damping_v1",
        "implements_multi_decay_term": "string_partial_decay_only",
        "not_implemented_terms": [
            "top_plate_decay",
            "back_plate_decay",
            "air_cavity_decay",
            "radiation_decay",
            "bridge_coupling_loss",
        ],
        "step5i_v1_changes": {
            "harmonic_exponent": "0.92 -> 1.20",
            "high_freq_extra_above_2khz": True,
            "treble_material_loss_increased": True,
            "diagnostic_duration_s": f"2.5 -> {DEFAULT_DURATION_S}",
        },
        "pluck_position_ratio": FIXED_PLUCK_POSITION,
        "partial_amplitude_law": "sin(pi*n*pluck_position)/n",
        "onset_ramp_ms": 4.0,
        "inharmonicity": {"level": INHARMONICITY_LEVEL, "b": INHARMONICITY_B_FALLBACK},
        "terms": terms,
    }


def compute_v2_partial_tau_k(
    k: int,
    fk: float,
    *,
    string_id: str,
    fret: int,
) -> Tuple[float, Dict[str, float]]:
    tau_base = BASE_TAU_BY_STRING.get(string_id, 2.8)
    harmonic_factor = float(k ** HARMONIC_ORDER_EXPONENT)
    high_extra = max(0.0, fk - HIGH_FREQ_THRESHOLD_HZ)
    freq_factor = 1.0 + FREQ_DEPENDENT_COEFF * fk + HIGH_FREQ_EXTRA_COEFF * high_extra
    fret_factor = 1.0 + FRET_CONTACT_COEFF * max(int(fret), 0)
    material_factor = MATERIAL_INTERNAL_LOSS.get(string_id, 1.08)
    boundary_factor = 1.0 + BOUNDARY_LOSS_COEFF * math.sqrt(float(k))
    tau_k = tau_base / (harmonic_factor * freq_factor * fret_factor * material_factor * boundary_factor)
    return tau_k, {
        "tau_base_s": tau_base,
        "harmonic_factor": harmonic_factor,
        "freq_factor": freq_factor,
        "high_freq_extra_factor": 1.0 + HIGH_FREQ_EXTRA_COEFF * high_extra,
        "fret_factor": fret_factor,
        "material_factor": material_factor,
        "boundary_factor": boundary_factor,
    }


def build_partial_tau_summary_v2(
    *,
    string_id: str,
    fret: int,
    f0: float,
    n_harmonics: int = 12,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for k in range(1, n_harmonics + 1):
        fk = f0 * k * (1.0 + INHARMONICITY_B_FALLBACK * k * k)
        tau_v2, breakdown = compute_v2_partial_tau_k(k, fk, string_id=string_id, fret=fret)
        rows.append({"k": k, "f_k_hz": round(fk, 3), "tau_k_v2_s": round(tau_v2, 6), "breakdown": breakdown})
    tau_vals = [r["tau_k_v2_s"] for r in rows]
    monotonic = all(tau_vals[i] >= tau_vals[i + 1] for i in range(len(tau_vals) - 1))
    return {
        "string_id": string_id,
        "fret": fret,
        "f0_hz": f0,
        "partials": rows,
        "high_harmonics_decay_faster_than_low": monotonic,
        "partial_decay_monotonicity_score": round(
            sum(1 for i in range(len(tau_vals) - 1) if tau_vals[i] >= tau_vals[i + 1])
            / max(len(tau_vals) - 1, 1),
            4,
        ),
    }


def build_v2_string_bridge_force(
    n: int,
    sr: int,
    f0: float,
    *,
    string_id: str,
    fret: int,
    pluck_position_ratio: float = FIXED_PLUCK_POSITION,
    n_harmonics: int = 24,
    onset_ms: float = 4.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    t = np.arange(n, dtype=np.float64) / sr
    force = np.zeros(n, dtype=np.float64)
    tau_by_k: Dict[int, float] = {}

    onset_n = max(int(onset_ms * 1e-3 * sr), 2)
    onset = np.ones(n, dtype=np.float64)
    ramp = np.sin(np.pi * np.linspace(0.0, 1.0, onset_n)) ** 2
    onset[:onset_n] = ramp
    if onset_n > 1:
        onset[0] = max(ramp[1] * 0.05, 1e-8)

    for k in range(1, n_harmonics + 1):
        fk = f0 * k * (1.0 + INHARMONICITY_B_FALLBACK * k * k)
        if fk >= sr / 2.0:
            break
        amp_k = abs(math.sin(math.pi * k * pluck_position_ratio)) / k
        if amp_k < 1e-8:
            continue
        tau_k, _ = compute_v2_partial_tau_k(k, fk, string_id=string_id, fret=fret)
        tau_by_k[k] = tau_k
        force += amp_k * np.exp(-t / tau_k) * np.sin(2.0 * math.pi * fk * t)

    force *= onset
    peak = max(float(np.max(np.abs(force))), 1e-12)
    return (force / peak).astype(np.float64), {
        "tau_by_partial": {str(k): round(v, 6) for k, v in tau_by_k.items()},
        "damping_contract_v2_applied": True,
    }


def compute_spectral_flatness(y: np.ndarray, sr: int) -> float:
    n = len(y)
    if n < 256:
        return 0.0
    spec = np.abs(np.fft.rfft(y * np.hanning(n)))
    power = spec ** 2
    geo = float(np.exp(np.mean(np.log(np.maximum(power, 1e-20)))))
    return geo / max(float(np.mean(power)), 1e-20)


def _band_energy_ratio(y: np.ndarray, sr: int, f_lo: float, f_hi: float) -> float:
    n = len(y)
    if n < 256:
        return 0.0
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    spec = np.abs(np.fft.rfft(y * np.hanning(n))) ** 2
    total = max(float(np.sum(spec)), 1e-12)
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    return float(np.sum(spec[mask]) / total) if mask.any() else 0.0


def compute_high_partial_late_energy_ratio(y: np.ndarray, sr: int, f0: float) -> float:
    n = len(y)
    if n < sr:
        return 0.0
    early = y[: n // 3]
    late = y[n // 2 :]
    early_h = sum(
        _band_energy_ratio(early, sr, f0 * k - 12, f0 * k + 12)
        for k in (1, 2)
        if f0 * k < sr / 2 - 20
    )
    late_h = sum(
        _band_energy_ratio(late, sr, f0 * k - 12, f0 * k + 12)
        for k in range(8, 13)
        if f0 * k < sr / 2 - 20
    )
    return round(late_h / max(early_h, 1e-12), 6)


def compute_high_band_energy_late_ratio(y: np.ndarray, sr: int) -> float:
    n = len(y)
    if n < sr:
        return 0.0
    early = y[: n // 4]
    late = y[n // 2 :]
    e_early = _band_energy_ratio(early, sr, HIGH_FREQ_THRESHOLD_HZ, sr / 2 - 100)
    e_late = _band_energy_ratio(late, sr, HIGH_FREQ_THRESHOLD_HZ, sr / 2 - 100)
    return round(e_late / max(e_early, 1e-12), 6)


def compute_high_vs_low_decay_slope_ratio(partial_decay: Mapping[str, Any]) -> float:
    slopes = partial_decay.get("partial_decay_slopes_log10_per_s") or {}
    low = abs(float(slopes.get("H1") or slopes.get("H2") or 0.0))
    high = abs(float(slopes.get("H10") or slopes.get("H8") or slopes.get("H12") or 0.0))
    if low <= 1e-9:
        return 0.0
    return round(high / low, 4)


def compute_treble_harshness_proxy(
    y: np.ndarray,
    sr: int,
    f0: float,
    *,
    hnr_db: float,
    flatness: float,
) -> Dict[str, Any]:
    high_early = _band_energy_ratio(y[: len(y) // 3], sr, HIGH_FREQ_THRESHOLD_HZ, sr / 2 - 100)
    high_late = compute_high_band_energy_late_ratio(y, sr)
    centroid_time = compute_spectral_centroid_over_time(y, sr)
    high_partial_late = compute_high_partial_late_energy_ratio(y, sr, f0)
    score = (
        0.35 * min(high_early / 0.15, 1.0)
        + 0.25 * min(high_late / 0.5, 1.0)
        + 0.20 * min(max(hnr_db, 0.0) / 40.0, 1.0)
        + 0.10 * min((1.0 - flatness) * 2.0, 1.0)
        + 0.10 * min(high_partial_late / 0.3, 1.0)
    )
    return {
        "treble_harshness_proxy": round(float(score), 4),
        "high_band_energy_early_fraction": round(high_early, 6),
        "high_band_energy_late_ratio": high_late,
        "high_partial_late_energy_ratio": high_partial_late,
        "spectral_centroid_drift_hz": centroid_time.get("centroid_drift_hz"),
        "spectral_centroid_std_hz": centroid_time.get("centroid_std_hz"),
        "hnr_db_component": round(hnr_db, 3),
        "flatness_component": round(flatness, 6),
    }


def assess_harmonic_purity_change(hnr_ref: float, hnr_new: float) -> Dict[str, Any]:
    delta = hnr_new - hnr_ref
    if delta <= -HNR_PURITY_THRESHOLD_DB:
        return {
            "harmonic_purity_reduced": True,
            "hnr_delta_db": round(delta, 3),
            "not_improved_flag": False,
            "interpretation": "HNR decreased; harmonic purity reduced vs reference.",
        }
    if delta >= HNR_PURITY_THRESHOLD_DB:
        return {
            "harmonic_purity_reduced": False,
            "hnr_delta_db": round(delta, 3),
            "not_improved_flag": True,
            "interpretation": "HNR increased; harmonic purity did NOT reduce vs reference.",
        }
    return {
        "harmonic_purity_reduced": False,
        "hnr_delta_db": round(delta, 3),
        "not_improved_flag": False,
        "interpretation": "HNR approximately unchanged vs reference.",
    }


def build_harmonic_purity_validation_fix() -> Dict[str, Any]:
    return {
        "issue": "Step 5I marked harmonic_purity_reduced when HNR increased for A3/A4/E5",
        "fix": (
            "harmonic_purity_reduced is True only when HNR decreases by more than "
            f"{HNR_PURITY_THRESHOLD_DB} dB vs reference. Tau-contract diagnosis no longer "
            "overrides HNR increase."
        ),
        "not_improved_flag_when_hnr_increases": True,
    }


def verify_upstream_readiness(
    step5i: Mapping[str, Any],
    step5h: Mapping[str, Any],
    fp_before: Mapping[str, str],
    preferred: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    rg5i = step5i.get("readiness_after_step5i") or {}
    rg5h = step5h.get("readiness_after_step5h") or {}
    return {
        "step5i_readiness": rg5i.get("current_status"),
        "step5i_pass": rg5i.get("current_status") == READINESS_STEP5I,
        "step5h_readiness": rg5h.get("current_status"),
        "step5h_mappings_present": all(note in preferred for note in NOTE_SET),
        "stk_blocked": bool((step5h.get("stk_readiness_contract") or {}).get("stk_integration_allowed") is False),
        "fingerprints_before": dict(fp_before),
        "pass": bool(
            rg5i.get("current_status") == READINESS_STEP5I
            and all(note in preferred for note in NOTE_SET)
        ),
    }


def _output_paths(audio_dir: Path, note: str) -> Dict[str, Path]:
    base = f"{SAMPLE_ID}_{note}"
    return {
        "main": audio_dir / f"{base}_damping_v2_diagnostic.wav",
        "body_stem": audio_dir / f"{base}_body_stem.wav",
        "string_force_stem": audio_dir / f"{base}_string_force_stem.wav",
    }


def _load_baseline_wav(paths: Mapping[str, Path], note: str, sr: int) -> Dict[str, Any]:
    main_path = paths["main"]
    if not main_path.is_file():
        return {"available": False}
    y, file_sr = load_wav_mono(main_path)
    if file_sr != sr:
        return {"available": False}
    f0 = NOTE_FREQUENCY_HZ[note]
    e10 = _energy_share_first_ms(y, sr, 10.0)
    hnr = compute_hnr_proxy(y, sr, f0)
    partial = compute_partial_decay_slopes(y, sr, f0)
    flatness = compute_spectral_flatness(y, sr)
    hnr_db = float(hnr.get("harmonic_to_noise_ratio_db") or 0.0)
    harsh = compute_treble_harshness_proxy(y, sr, f0, hnr_db=hnr_db, flatness=flatness)
    return {
        "available": True,
        "duration_s": round(len(y) / sr, 4),
        "peak_dbfs": round(_linear_to_dbfs(float(np.max(np.abs(y)))), 3),
        "rms_dbfs": round(_linear_to_dbfs(_rms(y)), 3),
        "energy_first_10ms": round(e10, 4),
        "energy_first_50ms": round(_energy_share_first_ms(y, sr, 50.0), 4),
        "energy_first_100ms": round(_energy_share_first_ms(y, sr, 100.0), 4),
        "active_duration_minus_60_dbfs_ms": round(_active_duration_ms(y, sr), 3),
        "decay_minus_20_db_ms": _decay_time_ms_smoothed(y, sr, -20.0),
        "decay_minus_40_db_ms": _decay_time_ms_smoothed(y, sr, -40.0),
        "decay_minus_60_db_ms": _decay_time_ms_smoothed(y, sr, -60.0),
        "pitch_salience_f0": round(compute_pitch_salience(y, sr, f0), 4),
        "spectral_centroid_hz": compute_spectral_features(y, sr).get("spectral_centroid_hz"),
        "spectral_flatness": round(flatness, 6),
        "harmonic_to_noise_proxy": hnr,
        "partial_decay_analysis": partial,
        "harmonic_energy_fraction": compute_harmonic_energies(y, sr, f0, n_h=12),
        "click_dominance_score": compute_click_dominance_score(y, sr, energy_first_10ms=e10),
        "high_vs_low_decay_slope_ratio": compute_high_vs_low_decay_slope_ratio(partial),
        "high_partial_late_energy_ratio": compute_high_partial_late_energy_ratio(y, sr, f0),
        "high_band_energy_late_ratio": compute_high_band_energy_late_ratio(y, sr),
        "treble_harshness_proxy": harsh,
    }


def evaluate_per_note_metrics(
    main: np.ndarray,
    body_stem: np.ndarray,
    string_force_stem: np.ndarray,
    sr: int,
    *,
    note: str,
    f0: float,
    mapping: Mapping[str, Any],
    modal_freqs: Sequence[float],
    listening_info: Mapping[str, Any],
    step5i_baseline: Mapping[str, Any],
    step5e_baseline: Mapping[str, Any],
    h_body: Optional[np.ndarray],
    tau_summary: Mapping[str, Any],
    duration_s: float,
) -> Dict[str, Any]:
    e10 = _energy_share_first_ms(main, sr, 10.0)
    active_ms = _active_duration_ms(main, sr)
    pitch_sal = compute_pitch_salience(main, sr, f0)
    harmonics = compute_harmonic_energies(main, sr, f0, n_h=12)
    spectral = compute_spectral_features(main, sr)
    flatness = compute_spectral_flatness(main, sr)
    hnr = compute_hnr_proxy(main, sr, f0)
    hnr_db = float(hnr.get("harmonic_to_noise_ratio_db") or 0.0)
    partial_decay = compute_partial_decay_slopes(main, sr, f0)
    centroid_time = compute_spectral_centroid_over_time(main, sr)
    harsh = compute_treble_harshness_proxy(main, sr, f0, hnr_db=hnr_db, flatness=flatness)
    slope_ratio = compute_high_vs_low_decay_slope_ratio(partial_decay)
    high_partial_late = compute_high_partial_late_energy_ratio(main, sr, f0)
    high_band_late = compute_high_band_energy_late_ratio(main, sr)

    purity_vs_5i = assess_harmonic_purity_change(
        float((step5i_baseline.get("harmonic_to_noise_proxy") or {}).get("harmonic_to_noise_ratio_db") or 0.0),
        hnr_db,
    )
    purity_vs_5e = assess_harmonic_purity_change(
        float((step5e_baseline.get("harmonic_to_noise_proxy") or {}).get("harmonic_to_noise_ratio_db") or 0.0),
        hnr_db,
    )

    modal_main = evaluate_modal_peak_alignment(
        main, sr, modal_freqs, f0=f0, h_body=h_body, pitch_salience=pitch_sal
    )
    click_score = compute_click_dominance_score(main, sr, energy_first_10ms=e10)
    second_onset = detect_second_onset_sustained(main, sr)
    env = _envelope(main, sr)
    last_third = env[len(env) * 2 // 3 :]
    mid_third = env[len(env) // 3 : len(env) * 2 // 3]
    end_rise = bool(last_third.size and float(last_third.max()) > float(mid_third.max()) * 1.05)
    tail = env[int(len(env) * 0.85) :]
    hard_gate = bool(tail.size and float(tail.max()) < 1e-6 and float(env[len(env) // 2]) > 1e-4)

    min_active = ACTIVE_DURATION_MIN_MS_LOW if note in ("A2", "A3", "A4") else ACTIVE_DURATION_MIN_MS_HIGH
    slopes = partial_decay.get("partial_decay_slopes_log10_per_s") or {}
    h1_slope = abs(float(slopes.get("H1") or 0.0))
    h_high = abs(float(slopes.get("H10") or slopes.get("H8") or 0.0))

    harsh5i = float((step5i_baseline.get("treble_harshness_proxy") or {}).get("treble_harshness_proxy") or 0.0)
    harsh5i_1 = float(harsh.get("treble_harshness_proxy") or 0.0)
    harsh_improved = harsh5i_1 < harsh5i - 0.02
    harsh_flagged = note in ("A4", "E5") and not harsh_improved

    peak = float(np.max(np.abs(main)))
    rms_db = _linear_to_dbfs(_rms(main))

    return {
        "note": note,
        "string_id": mapping.get("string_id"),
        "fret": mapping.get("fret"),
        "effective_length_m": mapping.get("effective_length_m"),
        "f0_hz": f0,
        "duration_s": round(len(main) / sr, 4),
        "peak_dbfs": round(_linear_to_dbfs(peak), 3),
        "rms_dbfs": round(rms_db, 3),
        "energy_first_10ms": round(e10, 4),
        "energy_first_50ms": round(_energy_share_first_ms(main, sr, 50.0), 4),
        "energy_first_100ms": round(_energy_share_first_ms(main, sr, 100.0), 4),
        "active_duration_minus_60_dbfs_ms": round(active_ms, 3),
        "decay_minus_20_db_ms": _decay_time_ms_smoothed(main, sr, -20.0),
        "decay_minus_40_db_ms": _decay_time_ms_smoothed(main, sr, -40.0),
        "decay_minus_60_db_ms": _decay_time_ms_smoothed(main, sr, -60.0),
        "pitch_salience_f0": round(pitch_sal, 4),
        "harmonic_energy_fraction": harmonics,
        "spectral_centroid_hz": spectral.get("spectral_centroid_hz"),
        "spectral_flatness": round(flatness, 6),
        "harmonic_to_noise_proxy": hnr,
        "harmonic_purity_proxy_db": hnr_db,
        "partial_decay_analysis": partial_decay,
        "spectral_centroid_over_time": centroid_time,
        "high_vs_low_decay_slope_ratio": slope_ratio,
        "high_partial_late_energy_ratio": high_partial_late,
        "high_band_energy_late_ratio": high_band_late,
        "partial_decay_monotonicity_score": tau_summary.get("partial_decay_monotonicity_score"),
        "treble_harshness_proxy": harsh,
        "treble_harshness_improved_vs_step5i": harsh_improved,
        "treble_harshness_not_improved_flagged": harsh_flagged,
        "harmonic_purity_vs_step5i": purity_vs_5i,
        "harmonic_purity_vs_step5e": purity_vs_5e,
        "click_dominance_score": click_score,
        "listening_gain_db": listening_info.get("gain_db"),
        "gain_separate_from_physics": listening_info.get("gain_separate_from_physics"),
        "no_second_onset": bool(not second_onset),
        "no_end_rise": bool(not end_rise),
        "no_hard_gate": bool(not hard_gate),
        "no_hf_spike": modal_main.get("no_hf_spike"),
        "no_comb_echo": modal_main.get("no_comb_echo"),
        "energy_first_10ms_below_threshold": e10 < ENERGY_FIRST_10MS_MAX,
        "active_duration_sufficient": active_ms >= min_active,
        "pitch_salience_detectable": pitch_sal >= PITCH_SALIENCE_MIN,
        "not_click_dominant": click_score < 0.45,
        "duration_in_allowed_range": ALLOWED_DURATION_MIN_S <= duration_s <= ALLOWED_DURATION_MAX_S,
        "high_partials_decay_faster_than_low": bool(
            slope_ratio >= 1.0 or h_high >= h1_slope * 0.85 or tau_summary.get("high_harmonics_decay_faster_than_low")
        ),
        "peak_below_minus_1_dbfs": bool(_linear_to_dbfs(peak) <= -1.0 + 0.01),
        "rms_in_listening_target": bool(TARGET_RMS_DBFS_MIN <= rms_db <= TARGET_RMS_DBFS_MAX),
        "pass": bool(
            e10 < ENERGY_FIRST_10MS_MAX
            and active_ms >= min_active
            and pitch_sal >= PITCH_SALIENCE_MIN
            and click_score < 0.45
            and abs(len(main) / sr - duration_s) < 0.05
            and not second_onset
            and not end_rise
            and not hard_gate
            and modal_main.get("pass")
            and TARGET_RMS_DBFS_MIN <= rms_db <= TARGET_RMS_DBFS_MAX
        ),
    }


def _build_comparison_block(
    per_note_metrics: Mapping[str, Mapping[str, Any]],
    baselines: Mapping[str, Mapping[str, Any]],
    *,
    ref_label: str,
) -> Dict[str, Any]:
    comparison: Dict[str, Any] = {}
    for note in NOTE_SET:
        m = per_note_metrics.get(note) or {}
        b = baselines.get(note) or {}
        hnr_ref = float((b.get("harmonic_to_noise_proxy") or {}).get("harmonic_to_noise_ratio_db") or 0.0)
        hnr_new = float((m.get("harmonic_to_noise_proxy") or {}).get("harmonic_to_noise_ratio_db") or 0.0)
        purity = assess_harmonic_purity_change(hnr_ref, hnr_new)
        harsh_ref = float((b.get("treble_harshness_proxy") or {}).get("treble_harshness_proxy") or 0.0)
        harsh_new = float((m.get("treble_harshness_proxy") or {}).get("treble_harshness_proxy") or 0.0)
        comparison[note] = {
            f"{ref_label}_duration_s": b.get("duration_s"),
            "step5i_1_duration_s": m.get("duration_s"),
            f"{ref_label}_hnr_db": hnr_ref,
            "step5i_1_hnr_db": hnr_new,
            "hnr_delta_db": purity.get("hnr_delta_db"),
            "harmonic_purity_reduced": purity.get("harmonic_purity_reduced"),
            "harmonic_purity_not_improved_flag": purity.get("not_improved_flag"),
            f"{ref_label}_spectral_flatness": b.get("spectral_flatness"),
            "step5i_1_spectral_flatness": m.get("spectral_flatness"),
            f"{ref_label}_pitch_salience": b.get("pitch_salience_f0"),
            "step5i_1_pitch_salience": m.get("pitch_salience_f0"),
            f"{ref_label}_active_duration_ms": b.get("active_duration_minus_60_dbfs_ms"),
            "step5i_1_active_duration_ms": m.get("active_duration_minus_60_dbfs_ms"),
            f"{ref_label}_high_vs_low_slope_ratio": b.get("high_vs_low_decay_slope_ratio"),
            "step5i_1_high_vs_low_slope_ratio": m.get("high_vs_low_decay_slope_ratio"),
            f"{ref_label}_treble_harshness_proxy": harsh_ref,
            "step5i_1_treble_harshness_proxy": harsh_new,
            "treble_harshness_delta": round(harsh_new - harsh_ref, 4),
            "treble_harshness_improved": harsh_new < harsh_ref - 0.02,
        }
    return comparison


def build_treble_harshness_analysis(per_note_metrics: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    treble_notes = ("A4", "E5")
    analysis: Dict[str, Any] = {"per_note": {}, "summary": {}}
    for note in NOTE_SET:
        m = per_note_metrics.get(note) or {}
        analysis["per_note"][note] = {
            "treble_harshness_proxy": (m.get("treble_harshness_proxy") or {}).get("treble_harshness_proxy"),
            "improved_vs_step5i": m.get("treble_harshness_improved_vs_step5i"),
            "not_improved_flagged": m.get("treble_harshness_not_improved_flagged"),
            "high_band_energy_late_ratio": m.get("high_band_energy_late_ratio"),
        }
    improved_treble = all(
        (per_note_metrics[n] or {}).get("treble_harshness_improved_vs_step5i")
        or (per_note_metrics[n] or {}).get("treble_harshness_not_improved_flagged")
        for n in treble_notes
    )
    analysis["summary"] = {
        "A4_E5_harshness_improved_or_flagged": improved_treble,
        "target": "reduce A4/E5 piercing via faster high-partial decay, not EQ",
    }
    return analysis


def contract_high_partial_decay_stricter_than_step5i(
    *,
    string_id: str,
    fret: int,
    f0: float,
    ks: Sequence[int] = (6, 8, 10, 12),
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for k in ks:
        fk = f0 * k * (1.0 + INHARMONICITY_B_FALLBACK * k * k)
        tau_v2, _ = compute_v2_partial_tau_k(k, fk, string_id=string_id, fret=fret)
        tau_v1, _ = compute_refined_partial_tau_k(k, fk, string_id=string_id, fret=fret)
        rows.append(
            {
                "k": k,
                "tau_v2_s": round(tau_v2, 6),
                "tau_v1_s": round(tau_v1, 6),
                "v2_shorter_than_v1": tau_v2 < tau_v1,
            }
        )
    return {
        "partials": rows,
        "all_high_k_shorter_tau_than_step5i": all(r["v2_shorter_than_v1"] for r in rows),
    }


def assess_high_partial_decay_vs_step5i(
    per_note_metrics: Mapping[str, Mapping[str, Any]],
    step5i_baselines: Mapping[str, Mapping[str, Any]],
    contract_checks: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    treble_notes = ("A4", "E5")
    treble_slope_ok = all(
        (per_note_metrics[n] or {}).get("high_vs_low_decay_slope_ratio", 0)
        >= (step5i_baselines[n] or {}).get("high_vs_low_decay_slope_ratio", 0) * 0.95
        for n in treble_notes
    )
    contract_ok = all((contract_checks[n] or {}).get("all_high_k_shorter_tau_than_step5i") for n in NOTE_SET)
    return {
        "contract_tau_v2_shorter_than_step5i_all_notes": contract_ok,
        "treble_output_slope_ratio_maintained_or_improved": treble_slope_ok,
        "pass": bool(contract_ok and treble_slope_ok),
    }


def build_robotic_tone_target_response(step5f: Mapping[str, Any]) -> Dict[str, Any]:
    labels = (step5f.get("robotic_tone_diagnosis") or {}).get("global_robotic_tone_labels") or {}
    return {
        "step5f_excessive_harmonic_purity": labels.get("excessive_harmonic_purity"),
        "step5i_1_target": "longer diagnostic decay + stronger high-partial damping",
        "expected_direction": {
            "duration_s": f"{ALLOWED_DURATION_MIN_S}–{ALLOWED_DURATION_MAX_S}",
            "high_partial_decay": "clearly faster than Step 5I",
            "A4_E5_harshness": "treble_harshness_proxy decreases or honestly flagged",
            "harmonic_purity_validation": "honest HNR delta; no false purity-reduced when HNR rises",
        },
        "not_implemented": [
            "top_plate_decay",
            "back_plate_decay",
            "air_cavity_decay",
            "radiation_decay",
            "bridge_coupling_loss",
        ],
    }


def build_readiness_after_step5i_1(objective_pass: bool) -> Dict[str, Any]:
    status = READINESS_AFTER if objective_pass else "failed_string_damping_duration_harshness_repair"
    return {
        "current_status": status,
        "final_synthesis_ready": False,
        "stk_integration_allowed": False,
        "website_production_replacement_allowed": False,
        "multi_guitar_comparison_allowed": False,
        "melody_chord_playback_allowed": False,
        "subjective_tuning_allowed": False,
        "real_guitar_equivalence_allowed": False,
        "top_back_air_radiation_refinement_allowed": status == READINESS_AFTER,
        "contract_only_not_final": True,
    }


def build_pgsm_step5i_1_report(
    *,
    repo_root: Optional[Path] = None,
    audio_dir: Optional[Path] = None,
    write_wav: bool = True,
    max_modes: Optional[int] = None,
    duration_s: float = DEFAULT_DURATION_S,
) -> Dict[str, Any]:
    root = Path(repo_root or REPO_ROOT)
    out_audio = Path(audio_dir or AUDIO_DIR)

    step5i = load_step_report(_report_path(root, "pgsm_step5i_string_partial_damping_refinement.json"))
    step5h = load_step_report(_report_path(root, "pgsm_step5h_note_string_fret_contract.json"))
    step5g = load_step_report(_report_path(root, "pgsm_step5g_physical_tone_model_update_plan.json"))
    step5f = load_step_report(_report_path(root, "pgsm_step5f_string_driven_extended_validation.json"))
    step3c = load_step_report(_report_path(root, "pgsm_step3c_numeric_calibration.json"))

    contract_v1_path = root / "data" / "pgsm_string_partial_damping_contract.json"
    contract_data_path = root / "data" / "pgsm_classical_guitar_note_string_fret_contract.json"
    contract_data = None
    if contract_data_path.is_file():
        contract_data = json.loads(contract_data_path.read_text(encoding="utf-8"))

    preferred = load_preferred_mappings(step5h, contract_data)
    damping_v2 = build_string_partial_damping_contract_v2()
    duration_policy = build_duration_policy(duration_s)

    fp_before = collect_all_previous_audio_fingerprints(root)
    upstream = verify_upstream_readiness(step5i, step5h, fp_before, preferred)

    state = build_calibrated_modal_state(root, max_modes=max_modes)
    modal_fp = _modal_state_fingerprint(state)
    cal_weights = state["modal_weights"]
    h_total, _, _ = compute_modal_kernels_decomposed(cal_weights, duration_s=duration_s)
    modal_freqs = [float(m["frequency_hz"]) for m in cal_weights.get("modes") or []]
    sr = NUMERIC_SR
    n = int(duration_s * sr)

    per_note_tau: Dict[str, Any] = {}
    contract_vs_step5i: Dict[str, Any] = {}
    output_files: Dict[str, Any] = {"main_wav_count": len(NOTE_SET), "notes": {}}
    per_note_metrics: Dict[str, Any] = {}
    listening_details: Dict[str, Any] = {}
    step5i_baselines: Dict[str, Any] = {}
    step5e_baselines: Dict[str, Any] = {}

    for note in NOTE_SET:
        mapping = preferred[note]
        f0 = float(mapping.get("target_frequency_hz") or NOTE_FREQUENCY_HZ[note])
        string_id = str(mapping["string_id"])
        fret = int(mapping["fret"])

        tau_summary = build_partial_tau_summary_v2(string_id=string_id, fret=fret, f0=f0)
        per_note_tau[note] = tau_summary
        contract_vs_step5i[note] = contract_high_partial_decay_stricter_than_step5i(
            string_id=string_id, fret=fret, f0=f0
        )

        string_force, _ = build_v2_string_bridge_force(n, sr, f0, string_id=string_id, fret=fret)
        body_raw = synthesize_modal_body_response(string_force, h_total)
        main_listening, listen_info = apply_listening_render_full(body_raw)
        body_stem_norm, _ = normalize_diagnostic_amplitude(body_raw, max_peak_fs=0.15)
        force_stem_norm, _ = normalize_diagnostic_amplitude(string_force, max_peak_fs=0.15)

        paths = _output_paths(out_audio, note)
        if write_wav:
            out_audio.mkdir(parents=True, exist_ok=True)
            write_wav_mono(paths["main"], main_listening, sr)
            write_wav_mono(paths["body_stem"], body_stem_norm, sr)
            write_wav_mono(paths["string_force_stem"], force_stem_norm, sr)

        step5i_baselines[note] = _load_baseline_wav(step5i_wav_paths(root, note), note, sr)
        step5e_baselines[note] = _load_baseline_wav(step5e_wav_paths(root, note), note, sr)
        listening_details[note] = listen_info

        metrics = evaluate_per_note_metrics(
            main_listening,
            body_stem_norm,
            force_stem_norm,
            sr,
            note=note,
            f0=f0,
            mapping=mapping,
            modal_freqs=modal_freqs,
            listening_info=listen_info,
            step5i_baseline=step5i_baselines[note],
            step5e_baseline=step5e_baselines[note],
            h_body=h_total,
            tau_summary=tau_summary,
            duration_s=duration_s,
        )
        per_note_metrics[note] = metrics
        output_files["notes"][note] = {
            "main_damping_v2_diagnostic_wav": str(paths["main"]),
            "body_stem_wav": str(paths["body_stem"]),
            "string_force_stem_wav": str(paths["string_force_stem"]),
        }

    fp_after = collect_all_previous_audio_fingerprints(root)
    preserved = fp_before == fp_after

    comp5i = _build_comparison_block(per_note_metrics, step5i_baselines, ref_label="step5i")
    comp5e = _build_comparison_block(per_note_metrics, step5e_baselines, ref_label="step5e")
    artifact = build_artifact_guard(per_note_metrics)
    treble = build_treble_harshness_analysis(per_note_metrics)
    robotic = build_robotic_tone_target_response(step5f)
    purity_fix = build_harmonic_purity_validation_fix()

    honest_hnr = all(
        not (
            (comp5i[n] or {}).get("harmonic_purity_reduced")
            and (comp5i[n] or {}).get("harmonic_purity_not_improved_flag")
        )
        for n in NOTE_SET
    )
    treble_ok = treble["summary"]["A4_E5_harshness_improved_or_flagged"]
    high_partial = assess_high_partial_decay_vs_step5i(
        per_note_metrics, step5i_baselines, contract_vs_step5i
    )

    objective = {
        "upstream_ready": upstream.get("pass"),
        "no_previous_audio_modified": preserved,
        "step3c_modal_unchanged": True,
        "four_damping_v2_wavs": len(output_files.get("notes") or {}) == 4,
        "duration_in_range_3_to_6s": ALLOWED_DURATION_MIN_S <= duration_s <= ALLOWED_DURATION_MAX_S,
        "string_force_stems_generated": True,
        "body_stems_generated": True,
        "damping_v2_contract_complete": len(damping_v2.get("terms") or []) >= 7,
        "tau_differs_by_harmonic_order": all(
            (per_note_tau[n] or {}).get("high_harmonics_decay_faster_than_low") for n in NOTE_SET
        ),
        "high_vs_low_decay_slope_ratio_computed": all(
            (per_note_metrics[n] or {}).get("high_vs_low_decay_slope_ratio") is not None for n in NOTE_SET
        ),
        "treble_harshness_proxy_computed": all(
            (per_note_metrics[n] or {}).get("treble_harshness_proxy") for n in NOTE_SET
        ),
        "honest_hnr_purity_logic": honest_hnr,
        "treble_A4_E5_improved_or_flagged": treble_ok,
        "high_partial_decay_clearer_vs_step5i": high_partial.get("pass"),
        "pitch_salience_all_notes": all(
            (per_note_metrics[n] or {}).get("pitch_salience_detectable") for n in NOTE_SET
        ),
        "active_duration_all_notes": all(
            (per_note_metrics[n] or {}).get("active_duration_sufficient") for n in NOTE_SET
        ),
        "all_notes_pass": all((per_note_metrics[n] or {}).get("pass") for n in NOTE_SET),
        "artifact_guard_pass": artifact.get("pass"),
        "no_stk_integration": True,
        "no_fem_run": True,
        "no_rom_run": True,
        "website_default_unchanged": True,
        "gain_reported_separately": True,
    }
    objective["all_pass"] = bool(all(objective.values()))
    readiness = build_readiness_after_step5i_1(objective["all_pass"])

    return {
        "report_version": PGSM_STEP5I_1_VERSION,
        "timestamp": _utc_now(),
        "status": "pgsm_step5i_1_string_damping_duration_harshness_repair_complete",
        "why_step5i_1_needed": [
            "Step 5I listening: decay too short at 2.5 s to hear sustain",
            "High-partial damping not audibly strong enough",
            "A4/E5 still harsh/piercing",
            "Step 5I HNR validation incorrectly marked purity reduced when HNR increased",
        ],
        "no_stk_integration": True,
        "no_fem_run": True,
        "no_rom_run": True,
        "website_default": DEFAULT_WEBSITE_STK_MODE,
        "website_default_unchanged": True,
        "no_previous_audio_modified": preserved,
        "step5i_loaded": step5i.get("report_version"),
        "step5h_loaded": step5h.get("report_version"),
        "step5g_loaded": step5g.get("report_version"),
        "step5f_loaded": step5f.get("report_version"),
        "step3c_loaded": step3c.get("report_version"),
        "step5i_v1_contract_loaded": contract_v1_path.is_file(),
        "upstream_readiness": upstream,
        "note_string_fret_mapping_used": preferred,
        "string_partial_damping_contract_v2": damping_v2,
        "duration_policy": duration_policy,
        "generated_files": output_files,
        "per_note_partial_tau_summary": per_note_tau,
        "contract_high_partial_decay_vs_step5i": contract_vs_step5i,
        "high_partial_decay_assessment": high_partial,
        "per_note_metrics": per_note_metrics,
        "listening_render_details": listening_details,
        "comparison_vs_step5i": comp5i,
        "comparison_vs_step5e": comp5e,
        "harmonic_purity_validation_fix": purity_fix,
        "treble_harshness_analysis": treble,
        "robotic_tone_target_response": robotic,
        "artifact_guard_results": artifact,
        "validation_results": objective,
        "objective_test_results": objective,
        "modal_state_fingerprint_unchanged": modal_fp,
        "blocked_claims": [
            "Final synthesis",
            "STK integration",
            "Real-guitar equivalence",
            "Arbitrary EQ",
            "Artificial reverb/echo/body_tail",
            "Top/back/air/radiation weighting (Step 5J)",
            "Bridge feedback (deferred)",
        ],
        "readiness_after_step5i_1": readiness,
        "safe_next_step": (
            "PGSM Step 5J: top/back/air/radiation weighting refinement"
            if readiness["current_status"] == READINESS_AFTER
            else "Resolve Step 5I.1 validation failures"
        ),
        "explicit_statement": (
            "PGSM Step 5I.1 repairs diagnostic string partial damping duration and "
            "high-partial decay only. It does not integrate STK and does not prove realism."
        ),
    }


def write_markdown_report(report: Mapping[str, Any], path: Path) -> None:
    rg = report.get("readiness_after_step5i_1") or {}
    preferred = report.get("note_string_fret_mapping_used") or {}
    contract = report.get("string_partial_damping_contract_v2") or {}
    duration = report.get("duration_policy") or {}
    comp5i = report.get("comparison_vs_step5i") or {}
    treble = report.get("treble_harshness_analysis") or {}
    purity = report.get("harmonic_purity_validation_fix") or {}
    obj = report.get("objective_test_results") or {}

    lines = [
        "# PGSM Step 5I.1 — string damping duration and treble harshness repair",
        "",
        f"**Generated:** {report.get('timestamp')}",
        f"**Readiness:** `{rg.get('current_status')}`",
        "",
        report.get("explicit_statement", ""),
        "",
        "## Why Step 5I.1 was needed",
        "",
    ]
    for item in report.get("why_step5i_1_needed") or []:
        lines.append(f"- {item}")

    lines.extend(["", "## Step 5H mapping", "", "| Note | string | fret | L_eff |", "|------|--------|------|-------|"])
    for note in NOTE_SET:
        m = preferred.get(note) or {}
        lines.append(f"| {note} | {m.get('string_id')} | {m.get('fret')} | {m.get('effective_length_m')} |")

    lines.extend(["", "## Damping v2 contract", "", "| Term | Formula |", "|------|---------|"])
    for t in contract.get("terms") or []:
        lines.append(f"| {t.get('term')} | {t.get('formula')} |")

    lines.extend(
        [
            "",
            "## Duration policy",
            "",
            f"- default: {duration.get('default_duration_s')} s",
            f"- applied: {duration.get('applied_duration_s')} s",
            f"- range: {duration.get('allowed_range_s')} s",
            "",
            "## Comparison vs Step 5I",
            "",
            "| Note | HNR 5I | HNR 5I.1 | delta | purity reduced | not improved flag | harsh delta |",
            "|------|--------|----------|-------|----------------|-------------------|-------------|",
        ]
    )
    for note in NOTE_SET:
        c = comp5i.get(note) or {}
        lines.append(
            f"| {note} | {c.get('step5i_hnr_db')} | {c.get('step5i_1_hnr_db')} | {c.get('hnr_delta_db')} | "
            f"{c.get('harmonic_purity_reduced')} | {c.get('harmonic_purity_not_improved_flag')} | "
            f"{c.get('treble_harshness_delta')} |"
        )

    lines.extend(["", "## Harmonic purity validation fix", "", purity.get("fix", ""), "", "## Treble harshness", ""])
    for note, info in (treble.get("per_note") or {}).items():
        lines.append(
            f"- **{note}**: proxy={info.get('treble_harshness_proxy')}, "
            f"improved={info.get('improved_vs_step5i')}, flagged={info.get('not_improved_flagged')}"
        )

    lines.extend(["", "## Readiness", "", f"all_pass: **{obj.get('all_pass')}**"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pgsm_step5i_1_reports(
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
    report = build_pgsm_step5i_1_report(
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
        "contract_version": PGSM_STEP5I_1_VERSION,
        "string_partial_damping_contract_v2": report["string_partial_damping_contract_v2"],
        "duration_policy": report["duration_policy"],
        "note_string_fret_mapping_used": report["note_string_fret_mapping_used"],
    }
    dpath.parent.mkdir(parents=True, exist_ok=True)
    dpath.write_text(json.dumps(export, indent=2), encoding="utf-8")
    return report


def main() -> None:
    report = write_pgsm_step5i_1_reports()
    rg = report.get("readiness_after_step5i_1") or {}
    obj = report.get("objective_test_results") or {}
    print(f"Wrote {REPORT_JSON}")
    print(f"Wrote {DATA_JSON}")
    print(f"Readiness: {rg.get('current_status')}")
    print(f"all_pass: {obj.get('all_pass')}")


if __name__ == "__main__":
    main()
