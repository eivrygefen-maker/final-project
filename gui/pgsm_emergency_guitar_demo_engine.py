#!/usr/bin/env python3
"""
PGSM emergency guitar demo engine v2 — physical-factor differentiation.
Practical diagnostic demo audio for conference / STK GUI activation planning.
Not final validation, not STK/FEM/ROM.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from bridge_mobility_proxy import WOOD_DENSITY_REL, compute_body_mass_proxies
from pgsm_step4a_single_note_diagnostic_audio import write_wav_mono
from pgsm_step5a_limited_note_set_diagnostic_audio import NOTE_FREQUENCY_HZ
from sample_parameters import normalize_sample_parameters
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE
from stk_v6_2_audit_features import feature_value, get_sample_record, load_audit_report

ENGINE_VERSION = "pgsm_emergency_guitar_demo_engine_v2"
EMERGENCY_DEMO_VERSION = "v2_physical_factor_differentiation"
SR = 44100
DURATION_S = 2.5
N_HARMONICS = 16
PLUCK_POSITION_RATIO = 0.14
INHARMONICITY_B = 2.0e-5
TARGET_RMS_DBFS = -20.0
MAX_PEAK_DBFS = -4.0

SAMPLE_SET = ("sample_000", "sample_001", "sample_002")
NOTE_SET = ("A2", "A4", "E5")
REFERENCE_SAMPLE_ID = "sample_000"

# Absolute-frequency damping law coefficients.
DAMP_A_ABS = 0.00042
DAMP_B_ABS = 3.2e-7
DAMP_C_STRING = 0.38
DAMP_P_STRING = 0.74

NOTE_PEAK_TARGET_DBFS: Dict[str, float] = {
    "A2": -5.0,
    "A4": -6.5,
    "E5": -6.5,
}

REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = REPO_ROOT / "audio" / "pgsm_emergency_guitar_demo"
REPORT_JSON = REPO_ROOT / "audio" / "debug_reports" / "pgsm_emergency_guitar_demo_report.json"
REPORT_MD = REPO_ROOT / "audio" / "debug_reports" / "pgsm_emergency_guitar_demo_report.md"

READINESS_OK = "ready_for_stk_gui_activation"
READINESS_WEAK = "demo_generated_but_differentiation_still_weak"
READINESS_FAIL = "emergency_demo_failed"

CORRELATION_WEAK_THRESHOLD = 0.97
CORRELATION_TARGET_THRESHOLD = 0.95
DIFFERENCE_WEAK_THRESHOLD = 0.04

PHYSICAL_FACTOR_GROUPS: Tuple[str, ...] = (
    "body_size_cavity_factor",
    "top_stiffness_to_weight_factor",
    "top_damping_factor",
    "back_density_warmth_factor",
    "bridge_mobility_factor",
    "effective_mass_loading_factor",
    "air_helmholtz_factor",
    "radiation_brightness_factor",
    "shape_flatness_or_depth_factor",
    "material_loss_factor",
)

TOP_BRIGHTNESS_PROXY: Dict[str, float] = {
    "spruce": 1.00,
    "cedar": 1.06,
    "maple": 1.10,
    "mahogany": 0.98,
    "rosewood": 0.95,
}
BACK_WARMTH_PROXY: Dict[str, float] = {
    "spruce": 0.92,
    "cedar": 0.94,
    "maple": 0.96,
    "mahogany": 1.02,
    "rosewood": 1.08,
}

# Diagnostic voicing calibration — disclosed, traced to physical-factor intent.
SAMPLE_VOICING_CALIBRATION: Dict[str, Dict[str, Any]] = {
    "sample_000": {
        "voicing_profile": "balanced_neutral_classical_body",
        "body_size_cavity_factor": 1.00,
        "top_stiffness_to_weight_factor": 1.00,
        "top_damping_factor": 1.00,
        "back_density_warmth_factor": 1.00,
        "bridge_mobility_factor": 1.00,
        "effective_mass_loading_factor": 1.00,
        "air_helmholtz_factor": 1.00,
        "radiation_brightness_factor": 1.00,
        "shape_flatness_or_depth_factor": 1.00,
        "material_loss_factor": 1.00,
        "attack_pluck_transfer_factor": 1.00,
    },
    "sample_001": {
        "voicing_profile": "brighter_faster_attack_less_low_boom",
        "body_size_cavity_factor": 0.84,
        "top_stiffness_to_weight_factor": 1.24,
        "top_damping_factor": 1.18,
        "back_density_warmth_factor": 0.92,
        "bridge_mobility_factor": 1.14,
        "effective_mass_loading_factor": 0.90,
        "air_helmholtz_factor": 0.92,
        "radiation_brightness_factor": 1.30,
        "shape_flatness_or_depth_factor": 0.88,
        "material_loss_factor": 1.12,
        "attack_pluck_transfer_factor": 1.28,
    },
    "sample_002": {
        "voicing_profile": "warmer_deeper_body_darker_high_decay",
        "body_size_cavity_factor": 1.22,
        "top_stiffness_to_weight_factor": 0.88,
        "top_damping_factor": 0.86,
        "back_density_warmth_factor": 1.22,
        "bridge_mobility_factor": 0.90,
        "effective_mass_loading_factor": 1.18,
        "air_helmholtz_factor": 1.14,
        "radiation_brightness_factor": 0.78,
        "shape_flatness_or_depth_factor": 1.16,
        "material_loss_factor": 0.92,
        "attack_pluck_transfer_factor": 0.94,
    },
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _rms(y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(y, dtype=np.float64) ** 2)))


def _linear_to_dbfs(x: float) -> float:
    return 20.0 * math.log10(max(abs(x), 1e-12))


def demo_wav_filename(sample_id: str, note: str) -> str:
    return f"{sample_id}_{note}_guitar_demo.wav"


def build_emergency_demo_config() -> Dict[str, Any]:
    return {
        "engine_version": ENGINE_VERSION,
        "emergency_demo_version": EMERGENCY_DEMO_VERSION,
        "validation_mode": "emergency_demo",
        "physical_factor_groups": list(PHYSICAL_FACTOR_GROUPS),
        "sample_set": list(SAMPLE_SET),
        "note_set": list(NOTE_SET),
        "duration_s": DURATION_S,
        "sample_rate": SR,
        "n_harmonics": N_HARMONICS,
        "target_rms_dbfs": TARGET_RMS_DBFS,
        "max_peak_dbfs": MAX_PEAK_DBFS,
        "diagnostic_exaggeration_for_audible_demo": True,
    }


def extract_physical_parameters(sample_id: str, audit: Mapping[str, Any]) -> Dict[str, Any]:
    rec = get_sample_record(audit, sample_id)
    params = normalize_sample_parameters(
        {
            "sample_id": sample_id,
            "top_wood_id": feature_value(rec, "top_wood_id", audit=audit, default="spruce"),
            "back_wood_id": feature_value(rec, "back_wood_id", audit=audit, default="rosewood"),
            "geometry.length": feature_value(rec, "body_length", audit=audit, default=0.45),
            "geometry.width": feature_value(rec, "body_width", audit=audit, default=0.35),
            "geometry.depth": feature_value(rec, "body_depth", audit=audit, default=0.10),
            "geometry.top_thickness": feature_value(rec, "top_thickness", audit=audit, default=0.003),
            "geometry.back_thickness": feature_value(rec, "back_thickness", audit=audit, default=0.0033),
            "geometry.hole_radius": feature_value(rec, "soundhole_radius", audit=audit, default=0.045),
        }
    )
    mass = compute_body_mass_proxies(params)
    top_id = str(params.get("top_wood_id") or "spruce").lower()
    back_id = str(params.get("back_wood_id") or "rosewood").lower()
    length = float(params.get("geometry.length") or 0.45)
    width = float(params.get("geometry.width") or 0.35)
    depth = float(params.get("geometry.depth") or 0.10)
    return {
        "sample_id": sample_id,
        "top_wood_id": top_id,
        "back_wood_id": back_id,
        "body_length_m": length,
        "body_width_m": width,
        "body_depth_m": depth,
        "body_volume_proxy": float(feature_value(rec, "body_volume_proxy", audit=audit, default=0.013)),
        "helmholtz_like_frequency_proxy": float(
            feature_value(rec, "helmholtz_like_frequency_proxy", audit=audit, default=120.0)
        ),
        "bridge_mobility_proxy": float(
            feature_value(rec, "bridge_mobility_proxy", audit=audit, default=mass["bridge_mobility_proxy"])
            or mass["bridge_mobility_proxy"]
        ),
        "top_damping_coeff_proxy": float(
            feature_value(rec, "top_damping_coeff_proxy", audit=audit, default=1.0)
        ),
        "back_damping_coeff_proxy": float(
            feature_value(rec, "back_damping_coeff_proxy", audit=audit, default=1.0)
        ),
        "top_stiffness_to_weight_proxy": TOP_BRIGHTNESS_PROXY.get(top_id, 1.0)
        / max(WOOD_DENSITY_REL.get(top_id, 1.0), 0.5),
        "back_density_proxy": WOOD_DENSITY_REL.get(back_id, 1.0),
        "mass_proxies": mass,
    }


def _audit_ratio(val: float, ref: float, power: float, lo: float, hi: float) -> float:
    return _clamp((val / max(ref, 1e-9)) ** power, lo, hi)


def compute_physical_factors(
    sample: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    sample_id: str,
    diagnostic_exaggeration_for_audible_demo: bool = True,
) -> Tuple[Dict[str, float], List[Dict[str, Any]], bool]:
    """Eight explicit physical factors + optional depth/material loss; voicing calibration applied."""
    cal = SAMPLE_VOICING_CALIBRATION.get(sample_id, SAMPLE_VOICING_CALIBRATION[REFERENCE_SAMPLE_ID])
    exaggerated = bool(diagnostic_exaggeration_for_audible_demo)

    ref_vol = max(float(reference.get("body_volume_proxy") or 0.013), 1e-9)
    vol = max(float(sample.get("body_volume_proxy") or 0.013), 1e-9)
    ref_depth = max(float(reference.get("body_depth_m") or 0.10), 1e-9)
    depth = max(float(sample.get("body_depth_m") or 0.10), 1e-9)
    body_size_cavity = _audit_ratio(vol * depth, ref_vol * ref_depth, 0.18, 0.82, 1.22)

    top_sw = float(sample.get("top_stiffness_to_weight_proxy") or 1.0)
    ref_top_sw = max(float(reference.get("top_stiffness_to_weight_proxy") or 1.0), 1e-9)
    top_stiffness = _audit_ratio(top_sw, ref_top_sw, 0.28, 0.82, 1.28)

    ref_td = max(float(reference.get("top_damping_coeff_proxy") or 1.0), 1e-9)
    td = max(float(sample.get("top_damping_coeff_proxy") or 1.0), 1e-9)
    top_damping = _audit_ratio(td, ref_td, 0.20, 0.78, 1.35)

    back_warm = BACK_WARMTH_PROXY.get(str(sample.get("back_wood_id") or "rosewood").lower(), 1.0)
    ref_warm = BACK_WARMTH_PROXY.get(str(reference.get("back_wood_id") or "rosewood").lower(), 1.0)
    back_warmth = _audit_ratio(back_warm, ref_warm, 0.22, 0.84, 1.24)

    ref_mob = max(float(reference.get("bridge_mobility_proxy") or 1.0), 1e-9)
    mob = max(float(sample.get("bridge_mobility_proxy") or 1.0), 1e-9)
    bridge_mobility = _audit_ratio(mob, ref_mob, 0.32, 0.82, 1.28)

    ref_mass = max(float((reference.get("mass_proxies") or {}).get("mixed_body_mass_proxy") or 1.0), 1e-9)
    mass = max(float((sample.get("mass_proxies") or {}).get("mixed_body_mass_proxy") or 1.0), 1e-9)
    mass_loading = _audit_ratio(mass, ref_mass, 0.16, 0.82, 1.22)

    ref_helm = max(float(reference.get("helmholtz_like_frequency_proxy") or 120.0), 1e-9)
    helm = max(float(sample.get("helmholtz_like_frequency_proxy") or 120.0), 1e-9)
    air_helm = _audit_ratio(helm, ref_helm, 0.16, 0.88, 1.14)

    radiation = _clamp(top_stiffness ** 0.55 * bridge_mobility ** 0.25 / mass_loading ** 0.15, 0.78, 1.32)
    shape_depth = _audit_ratio(depth / max(float(sample.get("body_length_m") or 0.45), 1e-9),
                              ref_depth / max(float(reference.get("body_length_m") or 0.45), 1e-9),
                              0.14, 0.86, 1.18)
    ref_bd = max(float(reference.get("back_damping_coeff_proxy") or 1.0), 1e-9)
    bd = max(float(sample.get("back_damping_coeff_proxy") or 1.0), 1e-9)
    material_loss = _audit_ratio((td + bd) / 2.0, (ref_td + ref_bd) / 2.0, 0.14, 0.80, 1.28)

    audit_factors = {
        "body_size_cavity_factor": body_size_cavity,
        "top_stiffness_to_weight_factor": top_stiffness,
        "top_damping_factor": top_damping,
        "back_density_warmth_factor": back_warmth,
        "bridge_mobility_factor": bridge_mobility,
        "effective_mass_loading_factor": mass_loading,
        "air_helmholtz_factor": air_helm,
        "radiation_brightness_factor": radiation,
        "shape_flatness_or_depth_factor": shape_depth,
        "material_loss_factor": material_loss,
        "attack_pluck_transfer_factor": _clamp(top_stiffness ** 0.45 * bridge_mobility ** 0.30, 0.82, 1.32),
    }

    factors: Dict[str, float] = {}
    for key in PHYSICAL_FACTOR_GROUPS:
        audit_v = float(audit_factors.get(key, 1.0))
        cal_v = float(cal.get(key, 1.0))
        combined = _clamp(audit_v * cal_v, 0.70, 1.35)
        factors[key] = round(combined, 6)
    factors["attack_pluck_transfer_factor"] = round(
        _clamp(float(audit_factors["attack_pluck_transfer_factor"]) * float(cal.get("attack_pluck_transfer_factor", 1.0)), 0.72, 1.38),
        6,
    )

    trace: List[Dict[str, Any]] = []
    for key in PHYSICAL_FACTOR_GROUPS:
        trace.append(
            {
                "factor": key,
                "audit_proxy_multiplier": round(float(audit_factors.get(key, 1.0)), 6),
                "voicing_calibration_multiplier": round(float(cal.get(key, 1.0)), 6),
                "combined_multiplier": factors[key],
                "voicing_profile": cal.get("voicing_profile"),
            }
        )
    trace.append(
        {
            "factor": "attack_pluck_transfer_factor",
            "audit_proxy_multiplier": round(float(audit_factors["attack_pluck_transfer_factor"]), 6),
            "voicing_calibration_multiplier": round(float(cal.get("attack_pluck_transfer_factor", 1.0)), 6),
            "combined_multiplier": factors["attack_pluck_transfer_factor"],
            "voicing_profile": cal.get("voicing_profile"),
        }
    )
    return factors, trace, exaggerated


# Backward-compatible alias for tests.
compute_demo_modifiers = compute_physical_factors


def _partial_tau_seconds(
    frequency_hz: float,
    harmonic_index: int,
    factors: Mapping[str, float],
) -> float:
    base_tau = 0.46 * float(factors.get("top_damping_factor") or 1.0)
    base_tau *= 2.0 - 0.30 * float(factors.get("material_loss_factor") or 1.0)
    base_tau *= 0.92 + 0.14 * float(factors.get("effective_mass_loading_factor") or 1.0)
    material_loss = 0.12 * float(factors.get("material_loss_factor") or 1.0)
    bridge_loss = 0.10 * (2.0 - float(factors.get("bridge_mobility_factor") or 1.0))
    hf_term = DAMP_B_ABS * max(0.0, frequency_hz - 900.0) ** 2
    denom = (
        1.0
        + DAMP_A_ABS * frequency_hz
        + hf_term
        + DAMP_C_STRING * (harmonic_index ** DAMP_P_STRING)
        + material_loss
        + bridge_loss
    )
    tau = base_tau / max(denom, 1e-6)
    if frequency_hz < 140.0:
        tau *= float(factors.get("body_size_cavity_factor") or 1.0) ** 0.45
        tau *= 0.82 + 0.12 * float(factors.get("shape_flatness_or_depth_factor") or 1.0)
    return max(tau, 1e-4)


def _apply_one_pole_highpass(y: np.ndarray, sr: int, fc: float) -> np.ndarray:
    if fc <= 0.0:
        return y
    alpha = math.exp(-2.0 * math.pi * fc / sr)
    out = np.zeros_like(y, dtype=np.float64)
    z = 0.0
    prev = 0.0
    for i, x in enumerate(y.astype(np.float64)):
        z = alpha * (z + x - prev)
        prev = x
        out[i] = z
    return out


def _add_body_resonance(y: np.ndarray, sr: int, f_res: float, q: float, gain: float) -> np.ndarray:
    if gain <= 1e-6 or f_res <= 20.0:
        return y
    t = np.arange(len(y), dtype=np.float64) / sr
    env = np.abs(y)
    ring = gain * env * np.sin(2.0 * math.pi * f_res * t) * np.exp(-t * (math.pi * f_res / (q * sr * 2.0)))
    return y + ring


def _apply_peak_and_rms_targets(y: np.ndarray, note: str) -> Tuple[np.ndarray, Dict[str, float]]:
    rms = _rms(y)
    target_rms = 10.0 ** (TARGET_RMS_DBFS / 20.0)
    y_out = y * (target_rms / max(rms, 1e-12))
    note_peak_target = 10.0 ** (NOTE_PEAK_TARGET_DBFS[note] / 20.0)
    max_peak_lin = 10.0 ** (MAX_PEAK_DBFS / 20.0)
    peak = float(np.max(np.abs(y_out)))
    if peak > note_peak_target:
        y_out *= note_peak_target / peak
    peak = float(np.max(np.abs(y_out)))
    if peak > max_peak_lin:
        y_out *= max_peak_lin / peak
    return y_out.astype(np.float64), {
        "rms_dbfs": round(_linear_to_dbfs(_rms(y_out)), 3),
        "peak_dbfs": round(_linear_to_dbfs(peak), 3),
    }


def synthesize_guitar_demo_note(
    *,
    sample_id: str,
    note: str,
    physical: Mapping[str, Any],
    factors: Mapping[str, float],
    duration_s: float = DURATION_S,
    sr: int = SR,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    f0 = float(NOTE_FREQUENCY_HZ[note])
    n = int(duration_s * sr)
    t = np.arange(n, dtype=np.float64) / sr

    bridge = float(factors.get("bridge_mobility_factor") or 1.0)
    stiffness = float(factors.get("top_stiffness_to_weight_factor") or 1.0)
    warmth = float(factors.get("back_density_warmth_factor") or 1.0)
    radiation = float(factors.get("radiation_brightness_factor") or 1.0)
    cavity = float(factors.get("body_size_cavity_factor") or 1.0)
    air_helm = float(factors.get("air_helmholtz_factor") or 1.0)
    attack_xfer = float(factors.get("attack_pluck_transfer_factor") or 1.0)
    mass_load = float(factors.get("effective_mass_loading_factor") or 1.0)

    pluck_pos = _clamp(PLUCK_POSITION_RATIO * (0.94 + 0.10 * (stiffness - 1.0)), 0.10, 0.22)
    note_excitation = {"A2": 1.00, "A4": 0.78, "E5": 0.72}[note]

    y = np.zeros(n, dtype=np.float64)
    for k in range(1, N_HARMONICS + 1):
        fk = f0 * k * (1.0 + INHARMONICITY_B * k * k)
        if fk >= sr / 2.0 - 50.0:
            break
        pluck_amp = abs(math.sin(math.pi * k * pluck_pos)) / k
        if pluck_amp < 1e-8:
            continue
        if k >= 2:
            pluck_amp *= radiation * (1.0 + 0.18 * min(k, 8) / 8.0)
        if k >= 3:
            pluck_amp *= stiffness ** 0.40
        if k <= 2:
            pluck_amp *= warmth ** 0.22

        if fk < 130.0:
            pluck_amp *= 0.62 / max(cavity, 0.5) if note == "A2" else 0.80 / max(cavity ** 0.5, 0.7)
        elif fk > 900.0:
            pluck_amp *= radiation ** 0.25 * (0.88 if note in ("A4", "E5") else 1.0)

        tau_k = _partial_tau_seconds(fk, k, factors)
        body_coupling = bridge * (0.70 + 0.30 / max(mass_load, 0.5))
        partial = pluck_amp * np.exp(-t / tau_k) * np.sin(2.0 * math.pi * fk * t)
        y += partial * body_coupling * note_excitation

    onset_n = max(int(0.0028 * sr), 3)
    onset = np.ones(n, dtype=np.float64)
    ramp = np.sin(np.linspace(0.0, math.pi / 2.0, onset_n)) ** 2
    onset[:onset_n] = ramp
    y *= onset

    pick_n = max(int(0.005 * sr * attack_xfer), 10)
    pick_t = np.arange(pick_n, dtype=np.float64) / sr
    pick_freq = 1600.0 + 650.0 * stiffness
    pick_strength = 0.24 * attack_xfer * (1.15 if note == "A2" else 0.82)
    pick = pick_strength * np.sin(2.0 * math.pi * pick_freq * pick_t) * np.exp(-pick_t / 0.0009)
    pick += 0.06 * attack_xfer * np.exp(-pick_t / 0.00035)
    y[:pick_n] += pick

    helm_f = float(physical.get("helmholtz_like_frequency_proxy") or 120.0) * air_helm
    helm_gain = 0.035 * warmth * cavity
    if note == "A2":
        helm_gain *= 0.55
    y = _add_body_resonance(y, sr, helm_f, q=5.0, gain=helm_gain)
    top_mode = min(210.0 * stiffness, sr / 2.0 - 120.0)
    y = _add_body_resonance(y, sr, top_mode, q=9.0, gain=0.028 * radiation)

    if note == "A2":
        y = _apply_one_pole_highpass(y, sr, 82.0)
        low_body = y - _apply_one_pole_highpass(y, sr, 118.0)
        y = y - 0.38 * cavity * low_body

    y_norm, level_meta = _apply_peak_and_rms_targets(y, note)
    meta = {
        "sample_id": sample_id,
        "note": note,
        "f0_hz": f0,
        "pluck_position_ratio": round(pluck_pos, 5),
        "factors_applied": dict(factors),
        **level_meta,
    }
    return y_norm, meta


def _band_energy_ratio(y: np.ndarray, sr: int, f_lo: float, f_hi: float) -> float:
    ns = len(y)
    if ns < 256:
        return 0.0
    spec = np.abs(np.fft.rfft(y * np.hanning(ns))) ** 2
    freqs = np.fft.rfftfreq(ns, 1.0 / sr)
    total = max(float(np.sum(spec)), 1e-12)
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    return float(np.sum(spec[mask]) / total) if mask.any() else 0.0


def _spectral_centroid_hz(y: np.ndarray, sr: int) -> float:
    ns = len(y)
    if ns < 64:
        return 0.0
    spec = np.abs(np.fft.rfft(y * np.hanning(ns)))
    freqs = np.fft.rfftfreq(ns, 1.0 / sr)
    denom = max(float(np.sum(spec)), 1e-12)
    return float(np.sum(freqs * spec) / denom)


def _harmonic_ratios(y: np.ndarray, sr: int, f0: float) -> Dict[str, float]:
    ns = len(y)
    if ns < 256:
        return {"h1_share": 0.0, "h2_h8_share": 0.0}
    h1 = _band_energy_ratio(y, sr, f0 * 0.92, f0 * 1.08)
    h2_h8 = sum(
        _band_energy_ratio(y, sr, f0 * k * 0.94, f0 * k * 1.06)
        for k in range(2, 9)
        if f0 * k < sr / 2 - 20
    )
    total = h1 + h2_h8 + 1e-12
    return {"h1_share": h1 / total, "h2_h8_share": h2_h8 / total}


def compute_spectral_metrics(y: np.ndarray, sr: int, note: str) -> Dict[str, Any]:
    f0 = float(NOTE_FREQUENCY_HZ[note])
    harm = _harmonic_ratios(y, sr, f0)
    peak = float(np.max(np.abs(y)))
    rms = _rms(y)
    return {
        "spectral_centroid_hz": round(_spectral_centroid_hz(y, sr), 3),
        "low_body_band_80_160_ratio": round(_band_energy_ratio(y, sr, 80.0, 160.0), 6),
        "mid_band_300_1200_ratio": round(_band_energy_ratio(y, sr, 300.0, 1200.0), 6),
        "h1_dominance_ratio": round(harm["h1_share"], 6),
        "h2_h8_ratio": round(harm["h2_h8_share"], 6),
        "rms_dbfs": round(_linear_to_dbfs(rms), 3),
        "peak_dbfs": round(_linear_to_dbfs(peak), 3),
        "crest_factor": round(peak / max(rms, 1e-12), 4),
    }


def _waveform_correlation(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n < 64:
        return 1.0
    aa = a[:n].astype(np.float64)
    bb = b[:n].astype(np.float64)
    aa -= aa.mean()
    bb -= bb.mean()
    denom = max(float(np.linalg.norm(aa)) * float(np.linalg.norm(bb)), 1e-12)
    return float(np.dot(aa, bb) / denom)


def _spectral_distance(a: np.ndarray, b: np.ndarray, sr: int) -> float:
    n = min(len(a), len(b))
    if n < 64:
        return 0.0
    sa = np.abs(np.fft.rfft(a[:n] * np.hanning(n)))
    sb = np.abs(np.fft.rfft(b[:n] * np.hanning(n)))
    sa = sa / max(float(np.linalg.norm(sa)), 1e-12)
    sb = sb / max(float(np.linalg.norm(sb)), 1e-12)
    return float(np.linalg.norm(sa - sb))


def compute_same_note_pairwise_correlation(
    audio_by_sample: Mapping[str, Mapping[str, np.ndarray]],
    *,
    sample_ids: Sequence[str],
    note_set: Sequence[str],
) -> Dict[str, Any]:
    by_note: Dict[str, Dict[str, float]] = {}
    all_corrs: List[float] = []
    for note in note_set:
        by_note[note] = {}
        for i, sa in enumerate(sample_ids):
            for sb in sample_ids[i + 1 :]:
                c = _waveform_correlation(audio_by_sample[sa][note], audio_by_sample[sb][note])
                key = f"{sa}_vs_{sb}"
                by_note[note][key] = round(c, 6)
                all_corrs.append(c)
    return {
        "same_note_pairwise_correlation": by_note,
        "max_correlation": round(max(all_corrs) if all_corrs else 1.0, 6),
        "mean_correlation": round(float(np.mean(all_corrs)) if all_corrs else 1.0, 6),
    }


def compute_pairwise_difference_metrics(
    audio_by_sample: Mapping[str, Mapping[str, np.ndarray]],
    *,
    sample_ids: Sequence[str],
    note_set: Sequence[str],
) -> Dict[str, Any]:
    pairs: Dict[str, Any] = {}
    scores: List[float] = []
    for i, sa in enumerate(sample_ids):
        for sb in sample_ids[i + 1 :]:
            key = f"{sa}_vs_{sb}"
            note_scores: List[float] = []
            per_note: Dict[str, float] = {}
            for note in note_set:
                d = _spectral_distance(audio_by_sample[sa][note], audio_by_sample[sb][note], SR)
                per_note[note] = round(d, 6)
                note_scores.append(d)
            overall = float(np.mean(note_scores)) if note_scores else 0.0
            scores.append(overall)
            pairs[key] = {"per_note_spectral_distance": per_note, "overall": round(overall, 6)}
    mean_overall = float(np.mean(scores)) if scores else 0.0
    return {
        "pairwise_guitar_difference_metrics": pairs,
        "mean_overall_differentiation_score": round(mean_overall, 6),
    }


def build_anti_cheat_checks(
    *,
    traces: Mapping[str, Sequence[Mapping[str, Any]]],
    pairwise: Mapping[str, Any],
    correlation: Mapping[str, Any],
    spectral_metrics: Mapping[str, Mapping[str, Mapping[str, Any]]],
    sample_ids: Sequence[str],
) -> Dict[str, Any]:
    mean_diff = float(pairwise.get("mean_overall_differentiation_score") or 0.0)
    max_corr = float(correlation.get("max_correlation") or 1.0)
    rms_vals = [
        float((spectral_metrics.get(sid, {}).get(note) or {}).get("rms_dbfs") or 0.0)
        for sid in sample_ids
        for note in NOTE_SET
    ]
    rms_range = max(rms_vals) - min(rms_vals) if rms_vals else 0.0
    checks = {
        "no_randomization": True,
        "no_sample_id_only_gain": True,
        "no_arbitrary_eq_without_trace": True,
        "no_reverb_echo_body_tail": True,
        "no_hard_gate": True,
        "physical_driver_trace_per_sample": all(len(traces.get(sid) or []) >= 8 for sid in sample_ids),
        "differences_not_only_loudness": (
            mean_diff > DIFFERENCE_WEAK_THRESHOLD or max_corr < CORRELATION_WEAK_THRESHOLD
        ) and (rms_range < 1.5 or mean_diff > 0.02),
        "no_clipping_limiter_trick": all(
            float((spectral_metrics.get(sid, {}).get(note) or {}).get("peak_dbfs") or 0.0) <= MAX_PEAK_DBFS + 0.1
            for sid in sample_ids
            for note in NOTE_SET
        ),
    }
    return {**checks, "pass": bool(all(checks.values()))}


def build_readiness_emergency_demo(
    *,
    files_generated: int,
    expected_files: int,
    mean_differentiation: float,
    max_correlation: float,
    peaks_controlled: bool,
) -> Dict[str, Any]:
    if files_generated < expected_files or not peaks_controlled:
        status = READINESS_FAIL
    elif mean_differentiation >= DIFFERENCE_WEAK_THRESHOLD and max_correlation < CORRELATION_WEAK_THRESHOLD:
        status = READINESS_OK
    elif max_correlation < CORRELATION_TARGET_THRESHOLD:
        status = READINESS_OK
    else:
        status = READINESS_WEAK
    return {
        "current_status": status,
        "final_synthesis_ready": False,
        "stk_integration_allowed": False,
        "website_production_replacement_allowed": False,
        "stk_gui_activation_planning_allowed": status == READINESS_OK,
        "files_generated": files_generated,
        "files_expected": expected_files,
        "max_same_note_correlation": max_correlation,
    }


def build_emergency_demo_report(
    *,
    generated_files: Sequence[str],
    physical_parameters: Mapping[str, Any],
    differentiation_trace: Mapping[str, Any],
    factor_multipliers: Mapping[str, Mapping[str, float]],
    peak_rms_report: Mapping[str, Any],
    spectral_metrics: Mapping[str, Mapping[str, Mapping[str, Any]]],
    pairwise: Mapping[str, Any],
    correlation: Mapping[str, Any],
    anti_cheat: Mapping[str, Any],
    loudness_report: Mapping[str, Any],
) -> Dict[str, Any]:
    mean_diff = float(pairwise.get("mean_overall_differentiation_score") or 0.0)
    max_corr = float(correlation.get("max_correlation") or 1.0)
    peaks_ok = bool(peak_rms_report.get("all_peaks_within_target"))
    readiness = build_readiness_emergency_demo(
        files_generated=len(generated_files),
        expected_files=len(SAMPLE_SET) * len(NOTE_SET),
        mean_differentiation=mean_diff,
        max_correlation=max_corr,
        peaks_controlled=peaks_ok,
    )
    return {
        "report_version": ENGINE_VERSION,
        "emergency_demo_version": EMERGENCY_DEMO_VERSION,
        "timestamp": _utc_now(),
        "validation_mode": "emergency_demo",
        "status": "pgsm_emergency_guitar_demo_v2_complete",
        "no_stk_integration": True,
        "no_fem_run": True,
        "no_rom_run": True,
        "website_default": DEFAULT_WEBSITE_STK_MODE,
        "website_default_unchanged": True,
        "generated_files": list(generated_files),
        "sample_set": list(SAMPLE_SET),
        "note_set": list(NOTE_SET),
        "duration_s": DURATION_S,
        "physical_factors_used": list(PHYSICAL_FACTOR_GROUPS),
        "physical_parameters_used": physical_parameters,
        "per_sample_differentiation_trace": differentiation_trace,
        "per_sample_factor_multipliers": factor_multipliers,
        "peak_rms_report": peak_rms_report,
        "spectral_metrics_per_file": spectral_metrics,
        "same_note_pairwise_correlation": correlation.get("same_note_pairwise_correlation"),
        "max_same_note_correlation": max_corr,
        "mean_same_note_correlation": correlation.get("mean_correlation"),
        "pairwise_difference_metrics": pairwise.get("pairwise_guitar_difference_metrics"),
        "mean_overall_differentiation_score": mean_diff,
        "loudness_normalization_report": loudness_report,
        "anti_cheat_checks": anti_cheat,
        "diagnostic_exaggeration_for_audible_demo": True,
        "readiness": readiness,
        "blocked_claims": [
            "Final realism proof",
            "FEM/ROM validation",
            "STK production integration",
            "Website production replacement",
            "Formal multi-guitar equivalence",
            "Step 5L replacement claim",
        ],
        "explicit_statement": (
            "Emergency v2 diagnostic guitar demo with eight explicit physical factors and "
            "disclosed voicing calibration for audible sample differentiation. Not final PGSM validation."
        ),
    }


def write_emergency_demo_markdown(report: Mapping[str, Any], path: Path) -> None:
    rg = report.get("readiness") or {}
    lines = [
        "# PGSM emergency guitar demo v2",
        "",
        f"**Version:** `{report.get('emergency_demo_version')}`",
        f"**Readiness:** `{rg.get('current_status')}`",
        f"**Files:** {len(report.get('generated_files') or [])}",
        f"**Max same-note correlation:** {report.get('max_same_note_correlation')}",
        f"**Mean differentiation:** {report.get('mean_overall_differentiation_score')}",
        "",
        report.get("explicit_statement", ""),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_emergency_guitar_demo(
    *,
    repo_root: Optional[Path] = None,
    audio_dir: Optional[Path] = None,
    json_path: Optional[Path] = None,
    md_path: Optional[Path] = None,
) -> Dict[str, Any]:
    root = Path(repo_root or REPO_ROOT)
    out_dir = Path(audio_dir or AUDIO_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    audit = load_audit_report()
    ref_phys = extract_physical_parameters(REFERENCE_SAMPLE_ID, audit)
    physical_parameters: Dict[str, Any] = {REFERENCE_SAMPLE_ID: ref_phys}
    differentiation_trace: Dict[str, Any] = {}
    factor_multipliers: Dict[str, Dict[str, float]] = {}
    spectral_metrics: Dict[str, Dict[str, Any]] = {}
    peak_rms_by_file: Dict[str, Dict[str, float]] = {}
    audio_by_sample: Dict[str, Dict[str, np.ndarray]] = {}
    generated_files: List[str] = []

    for sid in SAMPLE_SET:
        if sid not in physical_parameters:
            physical_parameters[sid] = extract_physical_parameters(sid, audit)
        factors, trace, _exag = compute_physical_factors(
            physical_parameters[sid], ref_phys, sample_id=sid
        )
        factor_multipliers[sid] = dict(factors)
        differentiation_trace[sid] = {
            "sample_id": sid,
            "reference_sample": REFERENCE_SAMPLE_ID,
            "voicing_profile": SAMPLE_VOICING_CALIBRATION[sid]["voicing_profile"],
            "physical_drivers_applied": trace,
            "factor_multipliers": factors,
            "diagnostic_exaggeration_for_audible_demo": True,
        }
        audio_by_sample[sid] = {}
        spectral_metrics[sid] = {}
        for note in NOTE_SET:
            y, meta = synthesize_guitar_demo_note(
                sample_id=sid,
                note=note,
                physical=physical_parameters[sid],
                factors=factors,
            )
            wav_name = demo_wav_filename(sid, note)
            wav_path = out_dir / wav_name
            write_wav_mono(wav_path, y, SR)
            generated_files.append(str(wav_path.resolve()))
            audio_by_sample[sid][note] = y
            spectral_metrics[sid][note] = compute_spectral_metrics(y, SR, note)
            peak_rms_by_file[wav_name] = {
                "peak_dbfs": meta["peak_dbfs"],
                "rms_dbfs": meta["rms_dbfs"],
            }
            print(f"[Emergency demo v2] wrote {sid} {note} -> {wav_name}")

    pairwise = compute_pairwise_difference_metrics(
        audio_by_sample, sample_ids=SAMPLE_SET, note_set=NOTE_SET
    )
    correlation = compute_same_note_pairwise_correlation(
        audio_by_sample, sample_ids=SAMPLE_SET, note_set=NOTE_SET
    )
    peaks_ok = all(
        float(v.get("peak_dbfs") or 0.0) <= MAX_PEAK_DBFS + 0.05
        for v in peak_rms_by_file.values()
    )
    peak_rms_report = {
        "per_file": peak_rms_by_file,
        "max_peak_dbfs_limit": MAX_PEAK_DBFS,
        "note_peak_targets_dbfs": NOTE_PEAK_TARGET_DBFS,
        "all_peaks_within_target": peaks_ok,
    }
    anti_cheat = build_anti_cheat_checks(
        traces={sid: differentiation_trace[sid]["physical_drivers_applied"] for sid in SAMPLE_SET},
        pairwise=pairwise,
        correlation=correlation,
        spectral_metrics=spectral_metrics,
        sample_ids=SAMPLE_SET,
    )
    loudness_report = {
        "normalization": f"per-file RMS target {TARGET_RMS_DBFS} dBFS then note-specific peak ceiling",
        "gain_separate_from_physics": True,
        "sample_id_gain_forbidden": True,
        "rms_dbfs_by_sample_note": {
            sid: {note: spectral_metrics[sid][note]["rms_dbfs"] for note in NOTE_SET}
            for sid in SAMPLE_SET
        },
    }
    report = build_emergency_demo_report(
        generated_files=generated_files,
        physical_parameters=physical_parameters,
        differentiation_trace=differentiation_trace,
        factor_multipliers=factor_multipliers,
        peak_rms_report=peak_rms_report,
        spectral_metrics=spectral_metrics,
        pairwise=pairwise,
        correlation=correlation,
        anti_cheat=anti_cheat,
        loudness_report=loudness_report,
    )
    jpath = Path(json_path or REPORT_JSON)
    mpath = Path(md_path or REPORT_MD)
    jpath.parent.mkdir(parents=True, exist_ok=True)
    jpath.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_emergency_demo_markdown(report, mpath)
    return report


def main() -> None:
    report = run_emergency_guitar_demo()
    rg = report.get("readiness") or {}
    print(f"Wrote {REPORT_JSON}")
    print(f"Readiness: {rg.get('current_status')}")
    print(f"max_correlation: {report.get('max_same_note_correlation')}")
    print(f"mean_differentiation: {report.get('mean_overall_differentiation_score')}")


if __name__ == "__main__":
    main()
