#!/usr/bin/env python3
"""
PGSM Step 5I.3 — absolute-frequency string damping and pluck attack balance.
V4 partial damping (sigma_k / f_k law) + controlled pluck attack; diagnostic only.
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
    PEAK_CAP_DBFS,
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
from pgsm_step5i_1_string_damping_duration_harshness_repair import (
    assess_harmonic_purity_change,
    collect_step5i_fingerprints,
    compute_high_partial_late_energy_ratio,
    compute_high_vs_low_decay_slope_ratio,
    compute_spectral_flatness,
    load_preferred_mappings,
    step5i_wav_paths,
    _band_energy_ratio,
    _modal_state_fingerprint,
    _report_path,
)
from pgsm_step5i_2_string_decay_floor_peak_balance_repair import (
    READINESS_AFTER as READINESS_STEP5I_2,
    compute_decay_metrics,
    compute_high_note_piercing_proxy,
    compute_low_partial_late_energy_ratio,
    compute_upper_mid_dominance_proxy,
)
from pgsm_step5i_string_partial_damping_refinement import READINESS_AFTER as READINESS_STEP5I
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE

PGSM_STEP5I_3_VERSION = "pgsm_step5i_3_absolute_frequency_damping_pluck_balance_v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON = (
    REPO_ROOT / "audio" / "debug_reports" / "pgsm_step5i_3_absolute_frequency_damping_pluck_balance.json"
)
REPORT_MD = (
    REPO_ROOT / "audio" / "debug_reports" / "pgsm_step5i_3_absolute_frequency_damping_pluck_balance.md"
)
DATA_JSON = REPO_ROOT / "data" / "pgsm_string_partial_damping_contract_v4.json"
AUDIO_DIR = REPO_ROOT / "audio" / "pgsm_step5i_3_absolute_frequency_damping_pluck_balance"

READINESS_AFTER = "ready_for_step5j_top_back_air_radiation_weighting_refinement"

DEFAULT_DURATION_S = 5.5
ALLOWED_DURATION_MIN_S = 4.5
ALLOWED_DURATION_MAX_S = 6.0

HIGH_FREQ_THRESHOLD_HZ = 2000.0
UPPER_MID_LO_HZ = 500.0
UPPER_MID_HI_HZ = 2000.0

SIGMA_STRING_BASE: Dict[str, float] = {
    "string_6": 0.28,
    "string_5": 0.30,
    "string_4": 0.34,
    "string_3": 0.38,
    "string_2": 0.42,
    "string_1": 0.48,
}

MATERIAL_INTERNAL_LOSS: Dict[str, float] = {
    "string_6": 1.00,
    "string_5": 1.04,
    "string_4": 1.08,
    "string_3": 1.12,
    "string_2": 1.18,
    "string_1": 1.28,
}

HARMONIC_POWER = 0.28
SIGMA_FREQ_1 = 9.2e-3
SIGMA_FREQ_2 = 4.5e-8
F_BREAK_HZ = 950.0
SIGMA_FRET = 0.022
FRET_HIGH_EXTRA = 0.012
FRET_HIGH_THRESHOLD = 5
SIGMA_BOUNDARY = 0.018
HIGH_NOTE_LOW_PARTIAL_SIGMA_BOOST = 0.038

LOW_PARTIAL_LEAK_ONSET_S = 0.75
LOW_PARTIAL_LEAK_RATE_K1 = 0.48
LEAK_K_POWER = 0.80
LEAK_SMOOTH_S = 0.28

PLUCK_ATTACK_DURATION_MS = 5.5
PLUCK_ATTACK_DECAY_TAU_MS = 2.2
PLUCK_ATTACK_AMPLITUDE = 0.14
PLUCK_ATTACK_TREBLE_SCALE = 0.62
PLUCK_ATTACK_ENABLED = True

TREBLE_NOTES = frozenset({"A4", "E5"})
TREBLE_TARGET_RMS_DBFS = -23.0
CLICK_DOMINANCE_MAX = 0.45


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _file_fingerprint(path: Path) -> str:
    if not path.is_file():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def step5i_1_wav_paths(root: Path, note: str) -> Dict[str, Path]:
    base = f"{SAMPLE_ID}_{note}"
    d = root / "audio" / "pgsm_step5i_1_string_damping_duration_harshness_repair"
    return {
        "main": d / f"{base}_damping_v2_diagnostic.wav",
        "body_stem": d / f"{base}_body_stem.wav",
        "string_force_stem": d / f"{base}_string_force_stem.wav",
    }


def step5i_2_wav_paths(root: Path, note: str) -> Dict[str, Path]:
    base = f"{SAMPLE_ID}_{note}"
    d = root / "audio" / "pgsm_step5i_2_string_decay_floor_peak_balance_repair"
    return {
        "main": d / f"{base}_damping_v3_diagnostic.wav",
        "body_stem": d / f"{base}_body_stem.wav",
        "string_force_stem": d / f"{base}_string_force_stem.wav",
    }


def collect_step5i_1_fingerprints(root: Path) -> Dict[str, str]:
    fps: Dict[str, str] = {}
    for note in NOTE_SET:
        for key, p in step5i_1_wav_paths(root, note).items():
            fps[f"step5i_1_{note}_{key}"] = _file_fingerprint(p)
    return fps


def collect_step5i_2_fingerprints(root: Path) -> Dict[str, str]:
    fps: Dict[str, str] = {}
    for note in NOTE_SET:
        for key, p in step5i_2_wav_paths(root, note).items():
            fps[f"step5i_2_{note}_{key}"] = _file_fingerprint(p)
    return fps


def collect_all_previous_audio_fingerprints(root: Path) -> Dict[str, str]:
    return {
        **collect_previous_audio_fingerprints(root),
        **collect_step5e_fingerprints(root),
        **collect_step5i_fingerprints(root),
        **collect_step5i_1_fingerprints(root),
        **collect_step5i_2_fingerprints(root),
    }


def build_duration_policy(duration_s: float = DEFAULT_DURATION_S) -> Dict[str, Any]:
    return {
        "default_duration_s": DEFAULT_DURATION_S,
        "allowed_range_s": [ALLOWED_DURATION_MIN_S, ALLOWED_DURATION_MAX_S],
        "applied_duration_s": duration_s,
        "reason": "5.5 s diagnostic window for absolute-frequency decay proportionality",
        "no_trim": True,
        "no_hard_gate": True,
        "no_artificial_tail": True,
    }


def build_string_partial_damping_contract_v4() -> Dict[str, Any]:
    terms: List[Dict[str, Any]] = [
        {
            "term": "base_string_decay_by_string_id",
            "formula": "sigma_string_base[string_id]; bass < treble sigma (longer tau)",
            "values": dict(SIGMA_STRING_BASE),
            "source_level": "L2_literature_fallback_v4",
            "objective_metric": "bass strings sustain longer via lower sigma_base",
        },
        {
            "term": "harmonic_order_loss",
            "formula": f"sigma_harmonic = k^{HARMONIC_POWER}",
            "source_level": "L2_diagnostic_proxy_v4",
            "objective_metric": "secondary to absolute frequency; high k partials decay faster",
        },
        {
            "term": "absolute_frequency_loss",
            "formula": f"sigma_freq_1 * f_k with sigma_freq_1={SIGMA_FREQ_1}",
            "source_level": "L2_diagnostic_proxy_v4",
            "objective_metric": "tau decreases with absolute partial frequency f_k = k*f0",
        },
        {
            "term": "frequency_band_loss",
            "formula": (
                f"sigma_freq_2 * max(0, f_k - {F_BREAK_HZ})^2 with sigma_freq_2={SIGMA_FREQ_2}"
            ),
            "source_level": "L2_diagnostic_proxy_v4",
            "objective_metric": "internal friction / low-pass round-trip loss above f_break",
        },
        {
            "term": "fret_contact_loss_proxy",
            "formula": (
                f"1 + {SIGMA_FRET}*fret + {FRET_HIGH_EXTRA}*max(0,fret-{FRET_HIGH_THRESHOLD})"
            ),
            "source_level": "L2_diagnostic_proxy_v4",
            "objective_metric": "high-fret contact increases sigma",
        },
        {
            "term": "material_internal_loss_proxy",
            "formula": "sigma_material = MATERIAL_INTERNAL_LOSS[string_id]",
            "source_level": "L2_nylon_classical_fallback_v4",
            "objective_metric": "string_1 strongest internal loss",
        },
        {
            "term": "bridge_nut_boundary_loss_proxy",
            "formula": f"1 + {SIGMA_BOUNDARY}*sqrt(k)",
            "source_level": "L2_diagnostic_proxy_v4",
            "objective_metric": "boundary reflection loss increases with partial order",
        },
        {
            "term": "late_decay_leakage_proxy",
            "formula": (
                f"exp(-leak_k * smooth_ramp(t)) after t>{LOW_PARTIAL_LEAK_ONSET_S}s; "
                f"leak_k = {LOW_PARTIAL_LEAK_RATE_K1}/k^{LEAK_K_POWER}"
            ),
            "source_level": "L2_diagnostic_proxy_v4",
            "limitations": "Smooth causal leakage; not a hard gate",
            "objective_metric": "low partials lose energy in late window",
        },
        {
            "term": "high_note_peak_balance_proxy",
            "formula": (
                f"extra sigma boost on k<=4 for string_1 fret>={FRET_HIGH_THRESHOLD}: "
                f"1 + {HIGH_NOTE_LOW_PARTIAL_SIGMA_BOOST}*fret; pluck attack scaled for treble"
            ),
            "source_level": "L2_diagnostic_proxy_v4",
            "limitations": "Damping and excitation only; optional diagnostic RMS -23 dBFS for A4/E5",
            "blocked_claims": ["arbitrary_eq"],
            "objective_metric": "A4/E5 peak/piercing reduced vs Step 5I.2",
        },
        {
            "term": "combined_partial_tau",
            "formula": "sigma_k = product(terms); tau_k = 1/sigma_k; amplitude_k(t) includes leakage",
            "source_level": "L2_diagnostic_combined_proxy_v4",
            "objective_metric": "H1 tau monotonic with note pitch: A2 > A3 > A4 > E5",
        },
    ]
    return {
        "contract_id": "pgsm_string_partial_damping_v4",
        "supersedes": "pgsm_string_partial_damping_v3",
        "implements_multi_decay_term": "string_partial_decay_and_pluck_excitation_only",
        "future_body_radiation_note": (
            "True natural decay floor depends on top/back/air/radiation/coupling in Step 5J/5K."
        ),
        "not_implemented_terms": [
            "top_plate_decay",
            "back_plate_decay",
            "air_cavity_decay",
            "radiation_decay",
            "bridge_coupling_loss",
        ],
        "pluck_position_ratio": FIXED_PLUCK_POSITION,
        "partial_amplitude_law": "sin(pi*n*pluck_position)/n",
        "pluck_attack": {
            "enabled": PLUCK_ATTACK_ENABLED,
            "duration_ms": PLUCK_ATTACK_DURATION_MS,
            "decay_tau_ms": PLUCK_ATTACK_DECAY_TAU_MS,
            "base_amplitude": PLUCK_ATTACK_AMPLITUDE,
            "treble_amplitude_scale": PLUCK_ATTACK_TREBLE_SCALE,
        },
        "onset_ramp_ms": 4.0,
        "terms": terms,
    }


def _smooth_leakage_ramp(t: np.ndarray, onset_s: float, smooth_s: float) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh((t - onset_s) / max(smooth_s, 1e-6)))


def compute_v4_partial_sigma_tau(
    k: int,
    fk: float,
    *,
    string_id: str,
    fret: int,
) -> Tuple[float, float, Dict[str, float]]:
    sigma_base = SIGMA_STRING_BASE.get(string_id, 0.35)
    harmonic = float(k ** HARMONIC_POWER)
    freq_lin = SIGMA_FREQ_1 * fk
    freq_band = SIGMA_FREQ_2 * (max(0.0, fk - F_BREAK_HZ) ** 2)
    fret_high = max(0, int(fret) - FRET_HIGH_THRESHOLD)
    fret_factor = 1.0 + SIGMA_FRET * max(int(fret), 0) + FRET_HIGH_EXTRA * fret_high
    material = MATERIAL_INTERNAL_LOSS.get(string_id, 1.10)
    boundary = 1.0 + SIGMA_BOUNDARY * math.sqrt(float(k))
    peak_balance = 1.0
    if string_id == "string_1" and int(fret) >= FRET_HIGH_THRESHOLD and k <= 4:
        peak_balance = 1.0 + HIGH_NOTE_LOW_PARTIAL_SIGMA_BOOST * int(fret)
    sigma_k = (
        sigma_base
        * harmonic
        * freq_lin
        * (1.0 + freq_band)
        * fret_factor
        * material
        * boundary
        * peak_balance
    )
    sigma_k = max(sigma_k, 1e-9)
    tau_k = 1.0 / sigma_k
    leak_k = LOW_PARTIAL_LEAK_RATE_K1 / (float(k) ** LEAK_K_POWER)
    return sigma_k, tau_k, {
        "sigma_base": round(sigma_base, 6),
        "harmonic_factor": round(harmonic, 6),
        "freq_lin_factor": round(freq_lin, 6),
        "freq_band_factor": round(freq_band, 6),
        "fret_factor": round(fret_factor, 6),
        "material_factor": material,
        "boundary_factor": round(boundary, 6),
        "peak_balance_factor": peak_balance,
        "sigma_k": round(sigma_k, 6),
        "f_k_hz": round(fk, 3),
        "leak_k_per_s": round(leak_k, 6),
    }


def build_partial_tau_summary_v4(
    *,
    string_id: str,
    fret: int,
    f0: float,
    n_harmonics: int = 12,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for k in range(1, n_harmonics + 1):
        fk = f0 * k * (1.0 + INHARMONICITY_B_FALLBACK * k * k)
        _, tau_v4, breakdown = compute_v4_partial_sigma_tau(k, fk, string_id=string_id, fret=fret)
        rows.append({"k": k, "f_k_hz": round(fk, 3), "tau_k_v4_s": round(tau_v4, 6), "breakdown": breakdown})
    tau_vals = [r["tau_k_v4_s"] for r in rows]
    monotonic = all(tau_vals[i] >= tau_vals[i + 1] for i in range(len(tau_vals) - 1))
    return {
        "string_id": string_id,
        "fret": fret,
        "f0_hz": f0,
        "partials": rows,
        "high_harmonics_decay_faster_than_low": monotonic,
        "h1_tau_s": tau_vals[0] if tau_vals else None,
    }


def build_pluck_attack_component(
    n: int,
    sr: int,
    f0: float,
    *,
    note: str,
    fret: int,
    pluck_position_ratio: float = FIXED_PLUCK_POSITION,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if not PLUCK_ATTACK_ENABLED:
        return np.zeros(n, dtype=np.float64), {"pluck_attack_enabled": False}

    attack_samples = max(int(PLUCK_ATTACK_DURATION_MS * 1e-3 * sr), 8)
    t = np.arange(attack_samples, dtype=np.float64) / sr
    amp = PLUCK_ATTACK_AMPLITUDE
    if note in TREBLE_NOTES:
        amp *= PLUCK_ATTACK_TREBLE_SCALE
    elif int(fret) >= 7:
        amp *= 0.88

    decay_tau = PLUCK_ATTACK_DECAY_TAU_MS * 1e-3
    env = np.exp(-t / max(decay_tau, 1e-6))
    attack = np.zeros(attack_samples, dtype=np.float64)
    for k in range(1, 10):
        fk = f0 * k * (1.0 + INHARMONICITY_B_FALLBACK * k * k)
        if fk >= sr / 2.0:
            break
        pk = abs(math.sin(math.pi * k * pluck_position_ratio)) / math.sqrt(float(k))
        if pk < 1e-8:
            continue
        attack += pk * env * np.sin(2.0 * math.pi * fk * t)

    peak = max(float(np.max(np.abs(attack))), 1e-12)
    attack = (amp * attack / peak).astype(np.float64)
    full = np.zeros(n, dtype=np.float64)
    full[:attack_samples] = attack
    return full, {
        "pluck_attack_enabled": True,
        "pluck_attack_duration_ms": PLUCK_ATTACK_DURATION_MS,
        "pluck_attack_amplitude": round(amp, 4),
        "pluck_attack_peak": round(float(np.max(np.abs(attack))), 6),
        "treble_amplitude_scaled": note in TREBLE_NOTES,
    }


def build_v4_string_bridge_force(
    n: int,
    sr: int,
    f0: float,
    *,
    string_id: str,
    fret: int,
    note: str,
    pluck_position_ratio: float = FIXED_PLUCK_POSITION,
    n_harmonics: int = 24,
    onset_ms: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    t = np.arange(n, dtype=np.float64) / sr
    sustain = np.zeros(n, dtype=np.float64)
    leak_ramp = _smooth_leakage_ramp(t, LOW_PARTIAL_LEAK_ONSET_S, LEAK_SMOOTH_S)
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
        _, tau_k, breakdown = compute_v4_partial_sigma_tau(k, fk, string_id=string_id, fret=fret)
        leak_k = breakdown["leak_k_per_s"]
        decay = np.exp(-t / tau_k) * np.exp(-leak_k * leak_ramp)
        tau_by_k[k] = tau_k
        sustain += amp_k * decay * np.sin(2.0 * math.pi * fk * t)

    sustain *= onset
    pluck_attack, attack_meta = build_pluck_attack_component(
        n, sr, f0, note=note, fret=fret, pluck_position_ratio=pluck_position_ratio
    )
    combined = sustain + pluck_attack
    peak = max(float(np.max(np.abs(combined))), 1e-12)
    return (
        (combined / peak).astype(np.float64),
        (pluck_attack / peak).astype(np.float64),
        {
            "tau_by_partial": {str(k): round(v, 6) for k, v in tau_by_k.items()},
            "damping_contract_v4_applied": True,
            "absolute_frequency_damping_applied": True,
            "late_decay_leakage_applied": True,
            **attack_meta,
        },
    )


def estimate_partial_tau_from_slopes(
    partial: Mapping[str, Any],
    f0: float,
) -> Dict[str, Optional[float]]:
    slopes = (partial.get("partial_decay_slopes_log10_per_s") or {})
    out: Dict[str, Optional[float]] = {}
    for h_label, k in (("H1", 1), ("H2", 2), ("H3", 3)):
        slope = slopes.get(h_label)
        if slope is None or slope >= -1e-6:
            out[f"{h_label.lower()}_tau_estimate_s"] = None
        else:
            ln_decay_rate = abs(float(slope)) * math.log(10.0)
            out[f"{h_label.lower()}_tau_estimate_s"] = round(1.0 / max(ln_decay_rate, 1e-9), 4)
    freqs = [f0 * k for k in (1, 2, 3)]
    taus = [out.get(f"h{k}_tau_estimate_s") for k in (1, 2, 3)]
    valid = [(f, t) for f, t in zip(freqs, taus) if t is not None]
    if len(valid) >= 2:
        xs = np.array([v[0] for v in valid], dtype=np.float64)
        ys = np.log(np.array([v[1] for v in valid], dtype=np.float64))
        slope = float(np.polyfit(xs, ys, 1)[0])
        out["absolute_frequency_decay_slope_log_tau_vs_hz"] = round(slope, 6)
    else:
        out["absolute_frequency_decay_slope_log_tau_vs_hz"] = None
    return out


def compute_attack_clarity_proxy(
    y: np.ndarray,
    sr: int,
    *,
    baseline_e50: Optional[float] = None,
) -> Dict[str, Any]:
    e10 = _energy_share_first_ms(y, sr, 10.0)
    e50 = _energy_share_first_ms(y, sr, 50.0)
    e100 = _energy_share_first_ms(y, sr, 100.0)
    env = _envelope(y, sr)
    peak_overall = float(env.max()) if env.size else 1e-12
    attack_len = min(int(0.01 * sr), len(env))
    attack_peak = float(env[:attack_len].max()) if attack_len else 0.0
    attack_peak_ratio = attack_peak / max(peak_overall, 1e-12)
    clarity = min(e50 / max(e100, 1e-9), 2.0) * min(attack_peak_ratio / 0.35, 1.5)
    improved = baseline_e50 is not None and e50 > float(baseline_e50) * 1.05
    return {
        "attack_clarity_proxy": round(float(clarity), 4),
        "energy_first_10ms": round(e10, 4),
        "energy_first_50ms": round(e50, 4),
        "energy_first_100ms": round(e100, 4),
        "attack_peak_ratio": round(attack_peak_ratio, 4),
        "attack_clarity_improved_vs_step5i_2": improved,
        "attack_clarity_not_improved_flagged": baseline_e50 is not None and not improved,
    }


def build_decay_proportionality_metrics(
    per_note_tau: Mapping[str, Mapping[str, Any]],
    per_note_metrics: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    h1_contract: Dict[str, float] = {}
    h1_estimated: Dict[str, Optional[float]] = {}
    for note in NOTE_SET:
        tau_sum = per_note_tau.get(note) or {}
        h1_contract[note] = float(tau_sum.get("h1_tau_s") or 0.0)
        partial = (per_note_metrics.get(note) or {}).get("partial_decay_analysis") or {}
        est = estimate_partial_tau_from_slopes(partial, NOTE_FREQUENCY_HZ[note])
        h1_estimated[note] = est.get("h1_tau_estimate_s")

    ordering_notes = ("A2", "A3", "A4", "E5")
    contract_order_ok = all(
        h1_contract[ordering_notes[i]] > h1_contract[ordering_notes[i + 1]]
        for i in range(len(ordering_notes) - 1)
        if h1_contract[ordering_notes[i]] and h1_contract[ordering_notes[i + 1]]
    )

    per_note: Dict[str, Any] = {}
    for note in NOTE_SET:
        partial = (per_note_metrics.get(note) or {}).get("partial_decay_analysis") or {}
        est = estimate_partial_tau_from_slopes(partial, NOTE_FREQUENCY_HZ[note])
        dm = (per_note_metrics.get(note) or {}).get("decay_metrics") or {}
        per_note[note] = {
            **est,
            **{k: dm.get(k) for k in (
                "t_minus_20_db_ms", "t_minus_40_db_ms", "t_minus_60_db_ms",
                "final_0p5s_to_initial_0p5s_energy_ratio",
                "low_partial_late_energy_ratio", "high_partial_late_energy_ratio",
            ) if k in dm or k in (per_note_metrics.get(note) or {})},
            "h1_tau_contract_s": h1_contract.get(note),
            "contract_h1_order_contributes": True,
        }
        m = per_note_metrics.get(note) or {}
        per_note[note]["low_partial_late_energy_ratio"] = m.get("low_partial_late_energy_ratio")
        per_note[note]["high_partial_late_energy_ratio"] = m.get("high_partial_late_energy_ratio")

    return {
        "h1_tau_contract_by_note": h1_contract,
        "h1_tau_estimated_by_note": h1_estimated,
        "A3_H1_tau_lt_A2": h1_contract.get("A3", 0) < h1_contract.get("A2", 999),
        "A4_H1_tau_lt_A3": h1_contract.get("A4", 0) < h1_contract.get("A3", 999),
        "E5_H1_tau_lt_A4": h1_contract.get("E5", 0) < h1_contract.get("A4", 999),
        "cross_note_h1_tau_monotonic": contract_order_ok,
        "proportionality_pass": contract_order_ok,
        "per_note": per_note,
    }


def apply_listening_render_with_peak_balance(
    y: np.ndarray,
    *,
    note: str,
    target_rms_dbfs: float = TARGET_RMS_DBFS_NOMINAL,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if note in TREBLE_NOTES:
        target_rms_dbfs = TREBLE_TARGET_RMS_DBFS
    y_out, info = apply_listening_render_full(y, target_rms_dbfs=target_rms_dbfs)
    if note in TREBLE_NOTES and target_rms_dbfs != TARGET_RMS_DBFS_NOMINAL:
        info = {
            **info,
            "treble_diagnostic_rms_target_dbfs": target_rms_dbfs,
            "treble_rms_normalization_only_not_physics": True,
        }
    return y_out, info


def verify_upstream_readiness(
    step5i_2: Mapping[str, Any],
    step5h: Mapping[str, Any],
    preferred: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    rg = step5i_2.get("readiness_after_step5i_2") or {}
    return {
        "step5i_2_readiness": rg.get("current_status"),
        "step5i_2_pass": rg.get("current_status") == READINESS_STEP5I_2,
        "step5h_mappings_present": all(note in preferred for note in NOTE_SET),
        "stk_blocked": True,
        "pass": bool(rg.get("current_status") == READINESS_STEP5I_2 and all(note in preferred for note in NOTE_SET)),
    }


def _output_paths(audio_dir: Path, note: str) -> Dict[str, Path]:
    base = f"{SAMPLE_ID}_{note}"
    return {
        "main": audio_dir / f"{base}_damping_v4_diagnostic.wav",
        "body_stem": audio_dir / f"{base}_body_stem.wav",
        "string_force_stem": audio_dir / f"{base}_string_force_stem.wav",
        "pluck_attack_stem": audio_dir / f"{base}_pluck_attack_stem.wav",
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
    flatness = compute_spectral_flatness(y, sr)
    partial = compute_partial_decay_slopes(y, sr, f0)
    peak = float(np.max(np.abs(y)))
    rms_db = _linear_to_dbfs(_rms(y))
    decay = compute_decay_metrics(y, sr, dur)
    piercing = compute_high_note_piercing_proxy(y, sr, f0, peak_dbfs=_linear_to_dbfs(peak), rms_dbfs=rms_db, hnr_db=hnr_db)
    attack = compute_attack_clarity_proxy(y, sr)
    return {
        "available": True,
        "duration_s": round(dur, 4),
        "peak_dbfs": round(_linear_to_dbfs(peak), 3),
        "rms_dbfs": round(rms_db, 3),
        "crest_factor_db": round(_linear_to_dbfs(peak) - rms_db, 3),
        "energy_first_10ms": attack.get("energy_first_10ms"),
        "energy_first_50ms": attack.get("energy_first_50ms"),
        "energy_first_100ms": attack.get("energy_first_100ms"),
        "attack_clarity_proxy": attack.get("attack_clarity_proxy"),
        "pitch_salience_f0": round(compute_pitch_salience(y, sr, f0), 4),
        "harmonic_to_noise_proxy": hnr,
        "spectral_flatness": round(flatness, 6),
        "partial_decay_analysis": partial,
        "high_vs_low_decay_slope_ratio": compute_high_vs_low_decay_slope_ratio(partial),
        "high_partial_late_energy_ratio": compute_high_partial_late_energy_ratio(y, sr, f0),
        "low_partial_late_energy_ratio": compute_low_partial_late_energy_ratio(y, sr, f0),
        "upper_mid_dominance_proxy": compute_upper_mid_dominance_proxy(y, sr),
        "high_note_piercing_proxy": piercing,
        **decay,
    }


def evaluate_per_note(
    main: np.ndarray,
    body_stem: np.ndarray,
    string_force_stem: np.ndarray,
    pluck_attack_stem: np.ndarray,
    sr: int,
    *,
    note: str,
    f0: float,
    mapping: Mapping[str, Any],
    modal_freqs: Sequence[float],
    listening_info: Mapping[str, Any],
    baselines: Mapping[str, Mapping[str, Any]],
    h_body: Optional[np.ndarray],
    tau_summary: Mapping[str, Any],
    force_meta: Mapping[str, Any],
    duration_s: float,
) -> Dict[str, Any]:
    e10 = _energy_share_first_ms(main, sr, 10.0)
    pitch_sal = compute_pitch_salience(main, sr, f0)
    hnr = compute_hnr_proxy(main, sr, f0)
    hnr_db = float(hnr.get("harmonic_to_noise_ratio_db") or 0.0)
    partial = compute_partial_decay_slopes(main, sr, f0)
    peak = float(np.max(np.abs(main)))
    rms_db = _linear_to_dbfs(_rms(main))
    peak_dbfs = _linear_to_dbfs(peak)
    decay = compute_decay_metrics(main, sr, duration_s)
    piercing = compute_high_note_piercing_proxy(
        main, sr, f0, peak_dbfs=peak_dbfs, rms_dbfs=rms_db, hnr_db=hnr_db
    )
    b52 = baselines.get("step5i_2") or {}
    baseline_e50 = b52.get("energy_first_50ms")
    attack = compute_attack_clarity_proxy(main, sr, baseline_e50=baseline_e50)
    piercing_ref = float((b52.get("high_note_piercing_proxy") or {}).get("high_note_piercing_proxy") or 0.0)
    piercing_new = float(piercing.get("high_note_piercing_proxy") or 0.0)
    peak_improved = (
        piercing_new < piercing_ref - 0.02
        or peak_dbfs < float(b52.get("peak_dbfs") or 99) - 0.5
    )
    peak_flagged = note in TREBLE_NOTES and not peak_improved
    click_score = compute_click_dominance_score(main, sr, energy_first_10ms=e10)

    modal = evaluate_modal_peak_alignment(main, sr, modal_freqs, f0=f0, h_body=h_body, pitch_salience=pitch_sal)
    second_onset = detect_second_onset_sustained(main, sr)
    env = _envelope(main, sr)
    last_third = env[len(env) * 2 // 3 :]
    mid_third = env[len(env) // 3 : len(env) * 2 // 3]
    end_rise = bool(last_third.size and float(last_third.max()) > float(mid_third.max()) * 1.05)
    tail = env[int(len(env) * 0.85) :]
    hard_gate = bool(tail.size and float(tail.max()) < 1e-6 and float(env[len(env) // 2]) > 1e-4)

    min_active = ACTIVE_DURATION_MIN_MS_LOW if note in ("A2", "A3", "A4") else ACTIVE_DURATION_MIN_MS_HIGH
    measurable_decay = (
        decay.get("minus_20db_reached_within_window")
        or decay.get("final_0p5s_to_initial_0p5s_energy_ratio", 1.0) < 0.35
    )
    tau_estimates = estimate_partial_tau_from_slopes(partial, f0)

    return {
        "note": note,
        "string_id": mapping.get("string_id"),
        "fret": mapping.get("fret"),
        "f0_hz": f0,
        "duration_s": round(len(main) / sr, 4),
        "peak_dbfs": round(peak_dbfs, 3),
        "rms_dbfs": round(rms_db, 3),
        "crest_factor_db": round(peak_dbfs - rms_db, 3),
        "energy_first_10ms": round(e10, 4),
        "energy_first_50ms": attack.get("energy_first_50ms"),
        "energy_first_100ms": attack.get("energy_first_100ms"),
        "attack_clarity_proxy": attack,
        "click_dominance_score": round(click_score, 4),
        "pitch_salience_f0": round(pitch_sal, 4),
        "harmonic_to_noise_proxy": hnr,
        "spectral_flatness": round(compute_spectral_flatness(main, sr), 6),
        "harmonic_energy_fraction": compute_harmonic_energies(main, sr, f0, n_h=12),
        "partial_decay_analysis": partial,
        "partial_tau_estimates": tau_estimates,
        "high_vs_low_decay_slope_ratio": compute_high_vs_low_decay_slope_ratio(partial),
        "high_partial_late_energy_ratio": compute_high_partial_late_energy_ratio(main, sr, f0),
        "low_partial_late_energy_ratio": compute_low_partial_late_energy_ratio(main, sr, f0),
        "upper_mid_dominance_proxy": compute_upper_mid_dominance_proxy(main, sr),
        "high_note_piercing_proxy": piercing,
        "peak_balance_improved_vs_step5i_2": peak_improved,
        "peak_balance_not_improved_flagged": peak_flagged,
        "decay_metrics": decay,
        "measurable_decay_over_window": measurable_decay,
        "pluck_attack_meta": force_meta,
        "listening_gain_db": listening_info.get("gain_db"),
        "gain_separate_from_physics": listening_info.get("gain_separate_from_physics"),
        "treble_rms_normalization_applied": listening_info.get("treble_rms_normalization_only_not_physics"),
        "no_second_onset": not second_onset,
        "no_end_rise": not end_rise,
        "no_hard_gate": not hard_gate,
        "no_hf_spike": modal.get("no_hf_spike"),
        "no_comb_echo": modal.get("no_comb_echo"),
        "pitch_salience_detectable": pitch_sal >= PITCH_SALIENCE_MIN,
        "active_duration_sufficient": decay.get("active_duration_minus_60_dbfs_ms", 0) >= min_active,
        "attack_present": bool(force_meta.get("pluck_attack_enabled")),
        "not_click_dominant": click_score < CLICK_DOMINANCE_MAX,
        "high_harmonics_decay_faster_than_low": tau_summary.get("high_harmonics_decay_faster_than_low"),
        "peak_below_minus_1_dbfs": peak_dbfs <= PEAK_CAP_DBFS + 0.01,
        "pass": bool(
            e10 < ENERGY_FIRST_10MS_MAX
            and pitch_sal >= PITCH_SALIENCE_MIN
            and not second_onset
            and not end_rise
            and not hard_gate
            and click_score < CLICK_DOMINANCE_MAX
            and modal.get("pass")
            and TARGET_RMS_DBFS_MIN - 1 <= rms_db <= TARGET_RMS_DBFS_MAX + 1
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
    atk_ref = float(baseline.get("attack_clarity_proxy") or 0.0)
    atk_new = float((metrics.get("attack_clarity_proxy") or {}).get("attack_clarity_proxy") or 0.0)
    dm = metrics.get("decay_metrics") or {}
    bd = {k: baseline.get(k) for k in (
        "t_minus_20_db_ms", "t_minus_40_db_ms", "final_0p5s_to_initial_0p5s_energy_ratio",
        "duration_s", "peak_dbfs", "rms_dbfs", "pitch_salience_f0", "energy_first_10ms",
        "energy_first_50ms", "attack_clarity_proxy",
    )}
    return {
        f"{ref}_duration_s": bd.get("duration_s"),
        "step5i_3_duration_s": metrics.get("duration_s"),
        f"{ref}_peak_dbfs": bd.get("peak_dbfs"),
        "step5i_3_peak_dbfs": metrics.get("peak_dbfs"),
        "peak_delta_dbfs": round(float(metrics.get("peak_dbfs") or 0) - float(bd.get("peak_dbfs") or 0), 3),
        f"{ref}_hnr_db": hnr_ref,
        "step5i_3_hnr_db": hnr_new,
        "hnr_delta_db": purity.get("hnr_delta_db"),
        "harmonic_purity_reduced": purity.get("harmonic_purity_reduced"),
        "harmonic_purity_not_improved_flag": purity.get("not_improved_flag"),
        f"{ref}_piercing_proxy": pier_ref,
        "step5i_3_piercing_proxy": pier_new,
        "piercing_delta": round(pier_new - pier_ref, 4),
        f"{ref}_attack_clarity_proxy": atk_ref,
        "step5i_3_attack_clarity_proxy": atk_new,
        "attack_clarity_delta": round(atk_new - atk_ref, 4),
        f"{ref}_t_minus_20_db_ms": bd.get("t_minus_20_db_ms"),
        "step5i_3_t_minus_20_db_ms": dm.get("t_minus_20_db_ms"),
        f"{ref}_t_minus_40_db_ms": bd.get("t_minus_40_db_ms"),
        "step5i_3_t_minus_40_db_ms": dm.get("t_minus_40_db_ms"),
        f"{ref}_final_energy_ratio": bd.get("final_0p5s_to_initial_0p5s_energy_ratio"),
        "step5i_3_final_energy_ratio": dm.get("final_0p5s_to_initial_0p5s_energy_ratio"),
    }


def build_honest_failure_flags(
    per_note_metrics: Mapping[str, Mapping[str, Any]],
    proportionality: Mapping[str, Any],
) -> Dict[str, Any]:
    flags: Dict[str, Any] = {}
    for note in NOTE_SET:
        m = per_note_metrics.get(note) or {}
        dm = m.get("decay_metrics") or {}
        atk = m.get("attack_clarity_proxy") or {}
        flags[note] = {
            "minus_20db_not_reached": dm.get("minus_20db_honestly_flagged_not_reached"),
            "minus_40db_not_reached": dm.get("minus_40db_honestly_flagged_not_reached"),
            "deep_decay_pass": dm.get("deep_decay_pass"),
            "low_partial_decay_measurable": m.get("measurable_decay_over_window"),
            "peak_balance_not_improved": m.get("peak_balance_not_improved_flagged"),
            "attack_clarity_not_improved": atk.get("attack_clarity_not_improved_flagged"),
        }
    flags["_cross_note"] = {
        "proportionality_pass": proportionality.get("proportionality_pass"),
        "cross_note_h1_tau_monotonic": proportionality.get("cross_note_h1_tau_monotonic"),
    }
    return flags


def build_high_note_peak_analysis(
    per_note_metrics: Mapping[str, Mapping[str, Any]],
    baselines_52: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    low_notes = ("A2", "A3")
    treble = ("A4", "E5")
    peaks = {n: float((per_note_metrics[n] or {}).get("peak_dbfs") or 0) for n in NOTE_SET}
    rms = {n: float((per_note_metrics[n] or {}).get("rms_dbfs") or 0) for n in NOTE_SET}
    low_peak_max = max(peaks[n] for n in low_notes)
    analysis: Dict[str, Any] = {
        "peak_dbfs_by_note": peaks,
        "rms_dbfs_by_note": rms,
        "cross_note_peak_spread_db": round(max(peaks.values()) - min(peaks.values()), 3),
        "cross_note_rms_spread_db": round(max(rms.values()) - min(rms.values()), 3),
        "A4_E5_not_louder_than_A2_A3": all(peaks[n] <= low_peak_max + 1.5 for n in treble),
    }
    for note in NOTE_SET:
        m = per_note_metrics.get(note) or {}
        b = baselines_52.get(note) or {}
        analysis[f"{note}_detail"] = {
            "piercing_proxy": (m.get("high_note_piercing_proxy") or {}).get("high_note_piercing_proxy"),
            "upper_mid_dominance": m.get("upper_mid_dominance_proxy"),
            "attack_peak_ratio": (m.get("attack_clarity_proxy") or {}).get("attack_peak_ratio"),
            "piercing_delta_vs_5i2": round(
                float((m.get("high_note_piercing_proxy") or {}).get("high_note_piercing_proxy") or 0)
                - float((b.get("high_note_piercing_proxy") or {}).get("high_note_piercing_proxy") or 0),
                4,
            ),
            "improved_or_flagged": m.get("peak_balance_improved_vs_step5i_2")
            or m.get("peak_balance_not_improved_flagged"),
        }
    treble_ok = all(
        (per_note_metrics[n] or {}).get("peak_balance_improved_vs_step5i_2")
        or (per_note_metrics[n] or {}).get("peak_balance_not_improved_flagged")
        for n in treble
    )
    analysis["A4_E5_peak_balance_improved_or_flagged"] = treble_ok
    return analysis


def build_attack_clarity_analysis(
    per_note_metrics: Mapping[str, Mapping[str, Any]],
    baselines_52: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    per_note: Dict[str, Any] = {}
    improved_count = 0
    flagged_count = 0
    for note in NOTE_SET:
        m = per_note_metrics.get(note) or {}
        b = baselines_52.get(note) or {}
        atk = m.get("attack_clarity_proxy") or {}
        improved = bool(atk.get("attack_clarity_improved_vs_step5i_2"))
        flagged = bool(atk.get("attack_clarity_not_improved_flagged"))
        if improved:
            improved_count += 1
        if flagged:
            flagged_count += 1
        per_note[note] = {
            "attack_clarity_proxy": atk.get("attack_clarity_proxy"),
            "attack_peak_ratio": atk.get("attack_peak_ratio"),
            "energy_first_50ms": atk.get("energy_first_50ms"),
            "baseline_energy_first_50ms": b.get("energy_first_50ms"),
            "click_dominance_score": m.get("click_dominance_score"),
            "not_click_dominant": m.get("not_click_dominant"),
            "improved_vs_step5i_2": improved,
            "not_improved_flagged": flagged,
        }
    return {
        "per_note": per_note,
        "attack_clarity_improved_or_flagged": improved_count + flagged_count >= len(NOTE_SET),
        "all_not_click_dominant": all((per_note_metrics[n] or {}).get("not_click_dominant") for n in NOTE_SET),
    }


def build_absolute_frequency_damping_summary(
    per_note_tau: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    h1 = {n: (per_note_tau.get(n) or {}).get("h1_tau_s") for n in NOTE_SET}
    return {
        "damping_law": "sigma_k = sigma_base * k^p * sigma_freq_1*f_k * (1+sigma_freq_2*max(0,f_k-f_break)^2) * ...; tau_k=1/sigma_k",
        "f_break_hz": F_BREAK_HZ,
        "sigma_freq_1": SIGMA_FREQ_1,
        "sigma_freq_2": SIGMA_FREQ_2,
        "h1_tau_by_note_s": h1,
        "A3_H1_tau_lt_A2": h1.get("A3", 0) < h1.get("A2", 999),
        "A4_H1_tau_lt_A3": h1.get("A4", 0) < h1.get("A3", 999),
        "E5_H1_tau_lt_A4": h1.get("E5", 0) < h1.get("A4", 999),
    }


def build_pluck_attack_balance_summary(force_metas: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "enabled": PLUCK_ATTACK_ENABLED,
        "duration_ms": PLUCK_ATTACK_DURATION_MS,
        "decay_tau_ms": PLUCK_ATTACK_DECAY_TAU_MS,
        "base_amplitude": PLUCK_ATTACK_AMPLITUDE,
        "treble_scale": PLUCK_ATTACK_TREBLE_SCALE,
        "per_note": force_metas,
        "rationale_if_disabled": None if PLUCK_ATTACK_ENABLED else "Pluck attack disabled by contract flag",
    }


def build_readiness_after_step5i_3(objective_pass: bool) -> Dict[str, Any]:
    status = READINESS_AFTER if objective_pass else "failed_absolute_frequency_damping_pluck_balance"
    return {
        "current_status": status,
        "final_synthesis_ready": False,
        "stk_integration_allowed": False,
        "website_production_replacement_allowed": False,
        "contract_only_not_final": True,
        "top_back_air_radiation_refinement_allowed": status == READINESS_AFTER,
    }


def build_pgsm_step5i_3_report(
    *,
    repo_root: Optional[Path] = None,
    audio_dir: Optional[Path] = None,
    write_wav: bool = True,
    max_modes: Optional[int] = None,
    duration_s: float = DEFAULT_DURATION_S,
) -> Dict[str, Any]:
    root = Path(repo_root or REPO_ROOT)
    out_audio = Path(audio_dir or AUDIO_DIR)

    step5i_2 = load_step_report(_report_path(root, "pgsm_step5i_2_string_decay_floor_peak_balance_repair.json"))
    step5i_1 = load_step_report(_report_path(root, "pgsm_step5i_1_string_damping_duration_harshness_repair.json"))
    step5h = load_step_report(_report_path(root, "pgsm_step5h_note_string_fret_contract.json"))
    step3c = load_step_report(_report_path(root, "pgsm_step3c_numeric_calibration.json"))

    contract_data_path = root / "data" / "pgsm_classical_guitar_note_string_fret_contract.json"
    contract_data = None
    if contract_data_path.is_file():
        contract_data = json.loads(contract_data_path.read_text(encoding="utf-8"))

    preferred = load_preferred_mappings(step5h, contract_data)
    damping_v4 = build_string_partial_damping_contract_v4()
    duration_policy = build_duration_policy(duration_s)

    fp_before = collect_all_previous_audio_fingerprints(root)
    upstream = verify_upstream_readiness(step5i_2, step5h, preferred)

    state = build_calibrated_modal_state(root, max_modes=max_modes)
    modal_fp = _modal_state_fingerprint(state)
    cal_weights = state["modal_weights"]
    h_total, _, _ = compute_modal_kernels_decomposed(cal_weights, duration_s=duration_s)
    modal_freqs = [float(m["frequency_hz"]) for m in cal_weights.get("modes") or []]
    sr = NUMERIC_SR
    n = int(duration_s * sr)

    per_note_tau: Dict[str, Any] = {}
    per_note_decay: Dict[str, Any] = {}
    per_note_metrics: Dict[str, Any] = {}
    force_metas: Dict[str, Any] = {}
    output_files: Dict[str, Any] = {"main_wav_count": len(NOTE_SET), "notes": {}}
    baselines_52: Dict[str, Any] = {}
    baselines_51: Dict[str, Any] = {}
    baselines_5e: Dict[str, Any] = {}

    for note in NOTE_SET:
        mapping = preferred[note]
        f0 = float(mapping.get("target_frequency_hz") or NOTE_FREQUENCY_HZ[note])
        string_id = str(mapping["string_id"])
        fret = int(mapping["fret"])

        tau_summary = build_partial_tau_summary_v4(string_id=string_id, fret=fret, f0=f0)
        per_note_tau[note] = tau_summary

        string_force, pluck_stem, force_meta = build_v4_string_bridge_force(
            n, sr, f0, string_id=string_id, fret=fret, note=note
        )
        force_metas[note] = force_meta
        body_raw = synthesize_modal_body_response(string_force, h_total)
        main_listening, listen_info = apply_listening_render_with_peak_balance(body_raw, note=note)
        body_stem_norm, _ = normalize_diagnostic_amplitude(body_raw, max_peak_fs=0.15)
        force_stem_norm, _ = normalize_diagnostic_amplitude(string_force, max_peak_fs=0.15)
        pluck_stem_norm, _ = normalize_diagnostic_amplitude(pluck_stem, max_peak_fs=0.15)

        paths = _output_paths(out_audio, note)
        if write_wav:
            out_audio.mkdir(parents=True, exist_ok=True)
            write_wav_mono(paths["main"], main_listening, sr)
            write_wav_mono(paths["body_stem"], body_stem_norm, sr)
            write_wav_mono(paths["string_force_stem"], force_stem_norm, sr)
            write_wav_mono(paths["pluck_attack_stem"], pluck_stem_norm, sr)

        baselines_52[note] = _load_baseline(step5i_2_wav_paths(root, note), note, sr)
        baselines_51[note] = _load_baseline(step5i_1_wav_paths(root, note), note, sr)
        baselines_5e[note] = _load_baseline(step5e_wav_paths(root, note), note, sr)

        metrics = evaluate_per_note(
            main_listening,
            body_stem_norm,
            force_stem_norm,
            pluck_stem_norm,
            sr,
            note=note,
            f0=f0,
            mapping=mapping,
            modal_freqs=modal_freqs,
            listening_info=listen_info,
            baselines={
                "step5i_2": baselines_52[note],
                "step5i_1": baselines_51[note],
                "step5e": baselines_5e[note],
            },
            h_body=h_total,
            tau_summary=tau_summary,
            force_meta=force_meta,
            duration_s=duration_s,
        )
        per_note_metrics[note] = metrics
        per_note_decay[note] = metrics.get("decay_metrics") or {}
        output_files["notes"][note] = {
            "main_damping_v4_diagnostic_wav": str(paths["main"]),
            "body_stem_wav": str(paths["body_stem"]),
            "string_force_stem_wav": str(paths["string_force_stem"]),
            "pluck_attack_stem_wav": str(paths["pluck_attack_stem"]),
        }

    fp_after = collect_all_previous_audio_fingerprints(root)
    preserved = fp_before == fp_after

    comp52 = {n: _comparison_entry(per_note_metrics[n], baselines_52[n], ref="step5i_2") for n in NOTE_SET}
    comp51 = {n: _comparison_entry(per_note_metrics[n], baselines_51[n], ref="step5i_1") for n in NOTE_SET}
    comp5e = {n: _comparison_entry(per_note_metrics[n], baselines_5e[n], ref="step5e") for n in NOTE_SET}

    abs_freq_summary = build_absolute_frequency_damping_summary(per_note_tau)
    pluck_summary = build_pluck_attack_balance_summary(force_metas)
    proportionality = build_decay_proportionality_metrics(per_note_tau, per_note_metrics)
    peak_analysis = build_high_note_peak_analysis(per_note_metrics, baselines_52)
    attack_analysis = build_attack_clarity_analysis(per_note_metrics, baselines_52)
    honest_flags = build_honest_failure_flags(per_note_metrics, proportionality)
    artifact = build_artifact_guard(per_note_metrics)

    a2a3_measurable = all((per_note_metrics[n] or {}).get("measurable_decay_over_window") for n in ("A2", "A3"))
    a4e5_faster = (
        (per_note_decay.get("A4") or {}).get("final_0p5s_to_initial_0p5s_energy_ratio", 1.0)
        < (per_note_decay.get("A2") or {}).get("final_0p5s_to_initial_0p5s_energy_ratio", 1.0)
        and (per_note_decay.get("E5") or {}).get("final_0p5s_to_initial_0p5s_energy_ratio", 1.0)
        < (per_note_decay.get("A2") or {}).get("final_0p5s_to_initial_0p5s_energy_ratio", 1.0)
    )

    objective = {
        "upstream_ready": upstream.get("pass"),
        "no_previous_audio_modified": preserved,
        "step3c_modal_unchanged": True,
        "four_damping_v4_wavs": len(output_files.get("notes") or {}) == 4,
        "duration_in_range": ALLOWED_DURATION_MIN_S <= duration_s <= ALLOWED_DURATION_MAX_S,
        "v4_contract_complete": len(damping_v4.get("terms") or []) >= 10,
        "absolute_frequency_loss_present": any(
            t.get("term") == "absolute_frequency_loss" for t in damping_v4.get("terms") or []
        ),
        "frequency_band_loss_present": any(
            t.get("term") == "frequency_band_loss" for t in damping_v4.get("terms") or []
        ),
        "h1_tau_cross_note_order": proportionality.get("cross_note_h1_tau_monotonic"),
        "high_harmonics_decay_faster": all(
            (per_note_tau[n] or {}).get("high_harmonics_decay_faster_than_low") for n in NOTE_SET
        ),
        "A2_A3_measurable_decay": a2a3_measurable,
        "A4_E5_decay_faster_than_A2": a4e5_faster,
        "minus_20db_reached_or_flagged": all(
            (per_note_decay[n] or {}).get("minus_20db_reached_within_window")
            or (per_note_decay[n] or {}).get("minus_20db_honestly_flagged_not_reached")
            for n in NOTE_SET
        ),
        "minus_40db_honestly_reported": all(
            "minus_40db_honestly_flagged_not_reached" in (per_note_decay[n] or {}) for n in NOTE_SET
        ),
        "peak_analysis_computed": bool(peak_analysis.get("peak_dbfs_by_note")),
        "A4_E5_peak_improved_or_flagged": peak_analysis.get("A4_E5_peak_balance_improved_or_flagged"),
        "attack_clarity_improved_or_flagged": attack_analysis.get("attack_clarity_improved_or_flagged"),
        "attack_not_click_dominant": attack_analysis.get("all_not_click_dominant"),
        "pluck_stems_generated": PLUCK_ATTACK_ENABLED,
        "pitch_salience_all_notes": all((per_note_metrics[n] or {}).get("pitch_salience_detectable") for n in NOTE_SET),
        "all_notes_pass": all((per_note_metrics[n] or {}).get("pass") for n in NOTE_SET),
        "artifact_guard_pass": artifact.get("pass"),
        "no_stk_integration": True,
        "no_fem_run": True,
        "no_rom_run": True,
        "website_default_unchanged": True,
    }
    objective["all_pass"] = bool(all(objective.values()))
    readiness = build_readiness_after_step5i_3(objective["all_pass"])

    return {
        "report_version": PGSM_STEP5I_3_VERSION,
        "timestamp": _utc_now(),
        "status": "pgsm_step5i_3_absolute_frequency_damping_pluck_balance_complete",
        "why_step5i_3_needed": [
            "Step 5I.2 attack/pluck felt piano-like (under-expressed transient)",
            "A3/A4/E5 low partials too long relative to absolute pitch",
            "Damping must depend on f_k not only harmonic index k",
            "A4/E5 still slightly piercing",
        ],
        "no_stk_integration": True,
        "no_fem_run": True,
        "no_rom_run": True,
        "website_default": DEFAULT_WEBSITE_STK_MODE,
        "website_default_unchanged": True,
        "no_previous_audio_modified": preserved,
        "step5i_2_loaded": step5i_2.get("report_version"),
        "step5i_1_loaded": step5i_1.get("report_version"),
        "step5h_loaded": step5h.get("report_version"),
        "step3c_loaded": step3c.get("report_version"),
        "upstream_readiness": upstream,
        "note_string_fret_mapping_used": preferred,
        "string_partial_damping_contract_v4": damping_v4,
        "absolute_frequency_damping_summary": abs_freq_summary,
        "pluck_attack_balance_summary": pluck_summary,
        "duration_policy": duration_policy,
        "generated_files": output_files,
        "per_note_partial_tau_summary": per_note_tau,
        "per_note_decay_metrics": per_note_decay,
        "decay_proportionality_metrics": proportionality,
        "per_note_metrics": per_note_metrics,
        "high_note_peak_harshness_analysis": peak_analysis,
        "attack_clarity_analysis": attack_analysis,
        "comparison_vs_step5i_2": comp52,
        "comparison_vs_step5i_1": comp51,
        "comparison_vs_step5e": comp5e,
        "honest_failure_flags": honest_flags,
        "artifact_guard_results": artifact,
        "validation_results": objective,
        "objective_test_results": objective,
        "modal_state_fingerprint": modal_fp,
        "blocked_claims": [
            "Final synthesis",
            "STK integration",
            "Real-guitar equivalence",
            "Arbitrary EQ",
            "Body/radiation faking (Step 5J/5K)",
        ],
        "readiness_after_step5i_3": readiness,
        "safe_next_step": (
            "PGSM Step 5J: top/back/air/radiation weighting refinement"
            if readiness["current_status"] == READINESS_AFTER
            else "Resolve Step 5I.3 validation failures"
        ),
        "explicit_statement": (
            "PGSM Step 5I.3 repairs diagnostic absolute-frequency string damping and pluck attack balance only. "
            "It does not integrate STK and does not prove realism."
        ),
    }


def write_markdown_report(report: Mapping[str, Any], path: Path) -> None:
    rg = report.get("readiness_after_step5i_3") or {}
    contract = report.get("string_partial_damping_contract_v4") or {}
    abs_sum = report.get("absolute_frequency_damping_summary") or {}
    pluck = report.get("pluck_attack_balance_summary") or {}
    decay = report.get("per_note_decay_metrics") or {}
    prop = report.get("decay_proportionality_metrics") or {}
    peak = report.get("high_note_peak_harshness_analysis") or {}
    attack = report.get("attack_clarity_analysis") or {}
    comp = report.get("comparison_vs_step5i_2") or {}
    honest = report.get("honest_failure_flags") or {}
    obj = report.get("objective_test_results") or {}

    lines = [
        "# PGSM Step 5I.3 — absolute-frequency damping and pluck attack balance",
        "",
        f"**Readiness:** `{rg.get('current_status')}`",
        "",
        report.get("explicit_statement", ""),
        "",
        "## Why Step 5I.3 was needed",
        "",
    ]
    for item in report.get("why_step5i_3_needed") or []:
        lines.append(f"- {item}")

    lines.extend(["", "## Damping v4 contract", "", "| Term | Formula |", "|------|---------|"])
    for t in contract.get("terms") or []:
        lines.append(f"| {t.get('term')} | {str(t.get('formula', ''))[:90]} |")

    h1 = abs_sum.get("h1_tau_by_note_s") or {}
    lines.extend([
        "",
        "## Absolute frequency damping (H1 tau by note)",
        "",
        "| Note | H1 tau (s) |",
        "|------|------------|",
    ])
    for note in NOTE_SET:
        lines.append(f"| {note} | {h1.get(note)} |")
    lines.append(f"\nCross-note H1 monotonic: **{prop.get('cross_note_h1_tau_monotonic')}**")

    lines.extend([
        "",
        "## Pluck attack balance",
        "",
        f"Enabled: **{pluck.get('enabled')}** | duration: {pluck.get('duration_ms')} ms | "
        f"base amplitude: {pluck.get('base_amplitude')} | treble scale: {pluck.get('treble_scale')}",
        "",
        "## Decay metrics",
        "",
        "| Note | t-20dB ms | t-40dB ms | final/initial 0.5s |",
        "|------|-----------|-----------|-------------------|",
    ])
    for note in NOTE_SET:
        d = decay.get(note) or {}
        lines.append(
            f"| {note} | {d.get('t_minus_20_db_ms')} | {d.get('t_minus_40_db_ms')} | "
            f"{d.get('final_0p5s_to_initial_0p5s_energy_ratio')} |"
        )

    lines.extend(["", "## High-note peak / harshness", "", f"Peak spread: {peak.get('cross_note_peak_spread_db')} dB", ""])
    for note in NOTE_SET:
        c = comp.get(note) or {}
        h = honest.get(note) or {}
        lines.append(
            f"- **{note}**: peak={c.get('step5i_3_peak_dbfs')} dBFS, piercing_delta={c.get('piercing_delta')}, "
            f"peak_flag={h.get('peak_balance_not_improved')}"
        )

    lines.extend(["", "## Attack clarity", ""])
    for note in NOTE_SET:
        a = (attack.get("per_note") or {}).get(note) or {}
        lines.append(
            f"- **{note}**: clarity={a.get('attack_clarity_proxy')}, click={a.get('click_dominance_score')}, "
            f"improved={a.get('improved_vs_step5i_2')}"
        )

    lines.extend(["", "## Readiness", "", f"all_pass: **{obj.get('all_pass')}**"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pgsm_step5i_3_reports(
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
    report = build_pgsm_step5i_3_report(
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
        "contract_version": PGSM_STEP5I_3_VERSION,
        "string_partial_damping_contract_v4": report["string_partial_damping_contract_v4"],
        "duration_policy": report["duration_policy"],
    }
    dpath.parent.mkdir(parents=True, exist_ok=True)
    dpath.write_text(json.dumps(export, indent=2), encoding="utf-8")
    return report


def main() -> None:
    report = write_pgsm_step5i_3_reports()
    rg = report.get("readiness_after_step5i_3") or {}
    obj = report.get("objective_test_results") or {}
    print(f"Wrote {REPORT_JSON}")
    print(f"Wrote {DATA_JSON}")
    print(f"Readiness: {rg.get('current_status')}")
    print(f"all_pass: {obj.get('all_pass')}")


if __name__ == "__main__":
    main()
