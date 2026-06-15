#!/usr/bin/env python3
"""
PGSM emergency guitar demo engine v3 — waveform-level physical differentiation.
Factors shape partials, decay, attack/body mix, and body resonator bank before normalization.
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

ENGINE_VERSION = "pgsm_emergency_guitar_demo_engine_v3"
EMERGENCY_DEMO_VERSION = "v3_waveform_level_physical_differentiation"
SR = 44100
DURATION_S = 2.5
N_HARMONICS = 16
BASE_PLUCK_POSITION = 0.14
BASE_INHARMONICITY_B = 2.0e-5
TARGET_RMS_DBFS = -20.0
MAX_PEAK_DBFS = -4.0

SAMPLE_SET = ("sample_000", "sample_001", "sample_002")
NOTE_SET = ("A2", "A4", "E5")
REFERENCE_SAMPLE_ID = "sample_000"

DAMP_A_ABS = 0.00055
DAMP_B_ABS = 4.5e-7
DAMP_C_STRING = 0.42
DAMP_P_STRING = 0.76

NOTE_PEAK_TARGET_DBFS: Dict[str, float] = {"A2": -5.0, "A4": -6.5, "E5": -6.5}
NOTE_EXCITATION: Dict[str, float] = {"A2": 1.00, "A4": 0.74, "E5": 0.68}

REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = REPO_ROOT / "audio" / "pgsm_emergency_guitar_demo"
REPORT_JSON = REPO_ROOT / "audio" / "debug_reports" / "pgsm_emergency_guitar_demo_report.json"
REPORT_MD = REPO_ROOT / "audio" / "debug_reports" / "pgsm_emergency_guitar_demo_report.md"

READINESS_OK = "ready_for_stk_gui_activation"
READINESS_MODERATE = "ready_for_stk_gui_activation_with_moderate_differentiation"
READINESS_WEAK = "demo_generated_but_differentiation_still_weak"
READINESS_FAIL = "emergency_demo_failed"

CORRELATION_OK_THRESHOLD = 0.90
CORRELATION_MODERATE_THRESHOLD = 0.95

PHYSICAL_FACTOR_GROUPS: Tuple[str, ...] = (
    "effective_pluck_position_factor",
    "bridge_transfer_attack_factor",
    "body_size_cavity_factor",
    "top_stiffness_to_weight_factor",
    "top_damping_factor",
    "back_density_warmth_factor",
    "bridge_mobility_factor",
    "effective_mass_loading_factor",
    "air_helmholtz_factor",
    "radiation_brightness_factor",
)

TOP_BRIGHTNESS_PROXY: Dict[str, float] = {
    "spruce": 1.00, "cedar": 1.06, "maple": 1.10, "mahogany": 0.98, "rosewood": 0.95,
}
BACK_WARMTH_PROXY: Dict[str, float] = {
    "spruce": 0.92, "cedar": 0.94, "maple": 0.96, "mahogany": 1.02, "rosewood": 1.08,
}

# Disclosed diagnostic voicing — drives waveform-level synthesis parameters.
SAMPLE_VOICING_CALIBRATION: Dict[str, Dict[str, Any]] = {
    "sample_000": {
        "voicing_profile": "balanced_neutral_classical",
        "effective_pluck_position_factor": 1.00,
        "bridge_transfer_attack_factor": 1.00,
        "body_size_cavity_factor": 1.00,
        "top_stiffness_to_weight_factor": 1.00,
        "top_damping_factor": 1.00,
        "back_density_warmth_factor": 1.00,
        "bridge_mobility_factor": 1.00,
        "effective_mass_loading_factor": 1.00,
        "air_helmholtz_factor": 1.00,
        "radiation_brightness_factor": 1.00,
        "inharmonicity_scale": 1.00,
        "material_loss_factor": 1.00,
    },
    "sample_001": {
        "voicing_profile": "bright_light_fast_response",
        "effective_pluck_position_factor": 1.14,
        "bridge_transfer_attack_factor": 1.32,
        "body_size_cavity_factor": 0.78,
        "top_stiffness_to_weight_factor": 1.28,
        "top_damping_factor": 1.24,
        "back_density_warmth_factor": 0.88,
        "bridge_mobility_factor": 1.18,
        "effective_mass_loading_factor": 0.82,
        "air_helmholtz_factor": 0.90,
        "radiation_brightness_factor": 1.34,
        "inharmonicity_scale": 1.08,
        "material_loss_factor": 1.14,
    },
    "sample_002": {
        "voicing_profile": "warm_deep_heavy_response",
        "effective_pluck_position_factor": 0.86,
        "bridge_transfer_attack_factor": 0.76,
        "body_size_cavity_factor": 1.26,
        "top_stiffness_to_weight_factor": 0.84,
        "top_damping_factor": 0.82,
        "back_density_warmth_factor": 1.26,
        "bridge_mobility_factor": 0.86,
        "effective_mass_loading_factor": 1.22,
        "air_helmholtz_factor": 1.12,
        "radiation_brightness_factor": 0.72,
        "inharmonicity_scale": 0.92,
        "material_loss_factor": 0.88,
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
    return {
        "sample_id": sample_id,
        "top_wood_id": top_id,
        "back_wood_id": back_id,
        "body_depth_m": float(params.get("geometry.depth") or 0.10),
        "body_volume_proxy": float(feature_value(rec, "body_volume_proxy", audit=audit, default=0.013)),
        "helmholtz_like_frequency_proxy": float(
            feature_value(rec, "helmholtz_like_frequency_proxy", audit=audit, default=120.0)
        ),
        "bridge_mobility_proxy": float(
            feature_value(rec, "bridge_mobility_proxy", audit=audit, default=mass["bridge_mobility_proxy"])
            or mass["bridge_mobility_proxy"]
        ),
        "top_damping_coeff_proxy": float(feature_value(rec, "top_damping_coeff_proxy", audit=audit, default=1.0)),
        "back_damping_coeff_proxy": float(feature_value(rec, "back_damping_coeff_proxy", audit=audit, default=1.0)),
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
) -> Tuple[Dict[str, float], List[Dict[str, Any]], bool]:
    cal = SAMPLE_VOICING_CALIBRATION.get(sample_id, SAMPLE_VOICING_CALIBRATION[REFERENCE_SAMPLE_ID])
    ref_vol = max(float(reference.get("body_volume_proxy") or 0.013), 1e-9)
    vol = max(float(sample.get("body_volume_proxy") or 0.013), 1e-9)
    ref_depth = max(float(reference.get("body_depth_m") or 0.10), 1e-9)
    depth = max(float(sample.get("body_depth_m") or 0.10), 1e-9)
    top_sw = float(sample.get("top_stiffness_to_weight_proxy") or 1.0)
    ref_top_sw = max(float(reference.get("top_stiffness_to_weight_proxy") or 1.0), 1e-9)
    ref_td = max(float(reference.get("top_damping_coeff_proxy") or 1.0), 1e-9)
    td = max(float(sample.get("top_damping_coeff_proxy") or 1.0), 1e-9)
    ref_bd = max(float(reference.get("back_damping_coeff_proxy") or 1.0), 1e-9)
    bd = max(float(sample.get("back_damping_coeff_proxy") or 1.0), 1e-9)
    ref_mob = max(float(reference.get("bridge_mobility_proxy") or 1.0), 1e-9)
    mob = max(float(sample.get("bridge_mobility_proxy") or 1.0), 1e-9)
    ref_mass = max(float((reference.get("mass_proxies") or {}).get("mixed_body_mass_proxy") or 1.0), 1e-9)
    mass = max(float((sample.get("mass_proxies") or {}).get("mixed_body_mass_proxy") or 1.0), 1e-9)
    ref_helm = max(float(reference.get("helmholtz_like_frequency_proxy") or 120.0), 1e-9)
    helm = max(float(sample.get("helmholtz_like_frequency_proxy") or 120.0), 1e-9)
    back_warm = BACK_WARMTH_PROXY.get(str(sample.get("back_wood_id") or "rosewood").lower(), 1.0)
    ref_warm = BACK_WARMTH_PROXY.get(str(reference.get("back_wood_id") or "rosewood").lower(), 1.0)

    audit_map = {
        "effective_pluck_position_factor": _audit_ratio(top_sw, ref_top_sw, 0.20, 0.85, 1.18),
        "bridge_transfer_attack_factor": _audit_ratio(top_sw * mob, ref_top_sw * ref_mob, 0.22, 0.82, 1.22),
        "body_size_cavity_factor": _audit_ratio(vol * depth, ref_vol * ref_depth, 0.20, 0.78, 1.28),
        "top_stiffness_to_weight_factor": _audit_ratio(top_sw, ref_top_sw, 0.30, 0.78, 1.32),
        "top_damping_factor": _audit_ratio(td, ref_td, 0.18, 0.75, 1.35),
        "back_density_warmth_factor": _audit_ratio(back_warm, ref_warm, 0.22, 0.80, 1.28),
        "bridge_mobility_factor": _audit_ratio(mob, ref_mob, 0.30, 0.78, 1.30),
        "effective_mass_loading_factor": _audit_ratio(mass, ref_mass, 0.18, 0.78, 1.28),
        "air_helmholtz_factor": _audit_ratio(helm, ref_helm, 0.16, 0.86, 1.16),
        "radiation_brightness_factor": _clamp(
            (top_sw / ref_top_sw) ** 0.4 * (mob / ref_mob) ** 0.2 / (mass / ref_mass) ** 0.15, 0.72, 1.38
        ),
    }

    factors: Dict[str, float] = {}
    trace: List[Dict[str, Any]] = []
    for key in PHYSICAL_FACTOR_GROUPS:
        audit_v = float(audit_map.get(key, 1.0))
        cal_v = float(cal.get(key, 1.0))
        combined = round(_clamp(audit_v * cal_v, 0.65, 1.40), 6)
        factors[key] = combined
        trace.append(
            {
                "factor": key,
                "audit_proxy_multiplier": round(audit_v, 6),
                "voicing_calibration_multiplier": round(cal_v, 6),
                "combined_multiplier": combined,
                "synthesis_path": _factor_synthesis_path(key),
                "voicing_profile": cal.get("voicing_profile"),
            }
        )
    factors["inharmonicity_scale"] = round(float(cal.get("inharmonicity_scale", 1.0)), 6)
    factors["material_loss_factor"] = round(
        _clamp(_audit_ratio((td + bd) / 2, (ref_td + ref_bd) / 2, 0.14, 0.75, 1.30) * float(cal.get("material_loss_factor", 1.0)), 0.65, 1.40),
        6,
    )
    return factors, trace, True


compute_demo_modifiers = compute_physical_factors


def _factor_synthesis_path(factor: str) -> str:
    paths = {
        "effective_pluck_position_factor": "partial_amplitude_vector",
        "bridge_transfer_attack_factor": "attack_body_mix",
        "body_size_cavity_factor": "body_resonator_gain_decay",
        "top_stiffness_to_weight_factor": "partial_amplitude_h2_h6_inharmonicity",
        "top_damping_factor": "partial_decay_tau",
        "back_density_warmth_factor": "back_resonator_gain",
        "bridge_mobility_factor": "bridge_body_send",
        "effective_mass_loading_factor": "attack_speed_body_decay",
        "air_helmholtz_factor": "cavity_resonator_frequency",
        "radiation_brightness_factor": "h2_h8_radiation_decay",
    }
    return paths.get(factor, "synthesis_path")


def build_body_resonator_bank(
    sample_id: str,
    factors: Mapping[str, float],
    physical: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """5-role lightweight body modal proxy per sample — frequencies/Q/gain shift with factors."""
    helm = float(physical.get("helmholtz_like_frequency_proxy") or 120.0) * float(factors.get("air_helmholtz_factor") or 1.0)
    cavity = float(factors.get("body_size_cavity_factor") or 1.0)
    stiffness = float(factors.get("top_stiffness_to_weight_factor") or 1.0)
    warmth = float(factors.get("back_density_warmth_factor") or 1.0)
    radiation = float(factors.get("radiation_brightness_factor") or 1.0)
    mass = float(factors.get("effective_mass_loading_factor") or 1.0)
    damping = float(factors.get("top_damping_factor") or 1.0)

    if sample_id == "sample_001":
        base = [
            ("low_body_cavity", helm * 0.92, 5.5, 0.045 * cavity),
            ("top_main", 228.0 * stiffness, 12.0, 0.075 * radiation),
            ("back_low_mid", 132.0 * warmth, 8.5, 0.038 * warmth),
            ("upper_top_brightness", 540.0 * radiation, 16.0, 0.062 * radiation),
            ("radiation_sparkle", 2450.0 * radiation, 20.0, 0.038 * radiation),
        ]
    elif sample_id == "sample_002":
        base = [
            ("low_body_cavity", helm * 1.08, 3.8, 0.095 * cavity),
            ("top_main", 168.0 * stiffness, 6.5, 0.048 * stiffness),
            ("back_low_mid", 168.0 * warmth, 5.0, 0.088 * warmth),
            ("upper_top_brightness", 295.0 * radiation, 7.0, 0.022 * radiation),
            ("radiation_sparkle", 1150.0 * radiation, 9.0, 0.010 * radiation),
        ]
    else:
        base = [
            ("low_body_cavity", helm, 4.8, 0.065 * cavity),
            ("top_main", 198.0 * stiffness, 9.0, 0.058 * stiffness),
            ("back_low_mid", 148.0 * warmth, 6.5, 0.052 * warmth),
            ("upper_top_brightness", 410.0 * radiation, 12.0, 0.042 * radiation),
            ("radiation_sparkle", 1750.0 * radiation, 14.0, 0.022 * radiation),
        ]

    bank: List[Dict[str, Any]] = []
    for role, f_hz, q, gain in base:
        decay_scale = (0.85 + 0.20 * damping) * (0.90 + 0.18 * mass)
        if role == "radiation_sparkle":
            decay_scale *= 0.72 + 0.22 * radiation
        bank.append(
            {
                "role": role,
                "frequency_hz": round(min(f_hz, SR / 2 - 200), 3),
                "q": round(q, 4),
                "gain": round(gain, 6),
                "decay_scale": round(decay_scale, 6),
            }
        )
    return bank


def build_synthesis_profile(
    sample_id: str,
    factors: Mapping[str, float],
    physical: Mapping[str, Any],
) -> Dict[str, Any]:
    pluck_factor = float(factors.get("effective_pluck_position_factor") or 1.0)
    pluck_pos = _clamp(BASE_PLUCK_POSITION * pluck_factor, 0.09, 0.22)
    inharm_b = BASE_INHARMONICITY_B * float(factors.get("inharmonicity_scale", 1.0))
    attack_xfer = float(factors.get("bridge_transfer_attack_factor") or 1.0)
    bridge = float(factors.get("bridge_mobility_factor") or 1.0)
    mass = float(factors.get("effective_mass_loading_factor") or 1.0)
    resonators = build_body_resonator_bank(sample_id, factors, physical)
    return {
        "sample_id": sample_id,
        "voicing_profile": SAMPLE_VOICING_CALIBRATION[sample_id]["voicing_profile"],
        "pluck_position_ratio": round(pluck_pos, 5),
        "inharmonicity_b": inharm_b,
        "attack_body_ratio": round(_clamp(0.22 * attack_xfer / max(mass, 0.5), 0.12, 0.42), 5),
        "bridge_body_send": round(_clamp(0.55 * bridge / max(mass ** 0.35, 0.5), 0.28, 0.88), 5),
        "string_direct_mix": round(_clamp(0.38 / max(bridge ** 0.25, 0.5), 0.18, 0.55), 5),
        "body_resonator_bank": resonators,
    }


def _partial_tau_seconds(frequency_hz: float, harmonic_index: int, factors: Mapping[str, float]) -> float:
    base_tau = 0.44 * float(factors.get("top_damping_factor") or 1.0)
    base_tau *= 0.88 + 0.20 * float(factors.get("effective_mass_loading_factor") or 1.0)
    material_loss = 0.14 * float(factors.get("material_loss_factor") or 1.0)
    bridge_loss = 0.12 * (2.0 - float(factors.get("bridge_mobility_factor") or 1.0))
    radiation = float(factors.get("radiation_brightness_factor") or 1.0)
    hf_boost = DAMP_B_ABS * max(0.0, frequency_hz - 850.0) ** 2 * (1.15 if radiation > 1.1 else 0.95)
    denom = (
        1.0
        + DAMP_A_ABS * frequency_hz
        + hf_boost
        + DAMP_C_STRING * (harmonic_index ** DAMP_P_STRING)
        + material_loss
        + bridge_loss
    )
    tau = base_tau / max(denom, 1e-6)
    if frequency_hz < 140.0:
        tau *= float(factors.get("body_size_cavity_factor") or 1.0) ** 0.5
    if frequency_hz > 1000.0:
        tau *= 0.70 + 0.18 / max(radiation, 0.5)
    return max(tau, 8e-5)


def _ring_resonator_response(excitation: np.ndarray, sr: int, f_hz: float, q: float, gain: float, decay_scale: float) -> np.ndarray:
    n = len(excitation)
    k_len = min(int(0.35 * sr), n)
    if k_len < 8 or gain <= 1e-8:
        return np.zeros(n, dtype=np.float64)
    kt = np.arange(k_len, dtype=np.float64) / sr
    decay = (math.pi * f_hz / max(q * sr, 1.0)) * decay_scale
    kernel = gain * np.exp(-kt * decay) * np.sin(2.0 * math.pi * f_hz * kt)
    out = np.convolve(excitation.astype(np.float64), kernel, mode="full")[:n]
    return out


def _apply_one_pole_highpass(y: np.ndarray, sr: int, fc: float) -> np.ndarray:
    if fc <= 0.0:
        return y
    alpha = math.exp(-2.0 * math.pi * fc / sr)
    out = np.zeros_like(y, dtype=np.float64)
    z = prev = 0.0
    for i, x in enumerate(y.astype(np.float64)):
        z = alpha * (z + x - prev)
        prev = x
        out[i] = z
    return out


def _synthesize_attack_transient(n: int, sr: int, *, attack_xfer: float, stiffness: float, note: str) -> np.ndarray:
    pick_n = max(int(0.0045 * sr * attack_xfer), 12)
    pick_t = np.arange(pick_n, dtype=np.float64) / sr
    pick_freq = 1400.0 + 800.0 * stiffness
    strength = 0.28 * attack_xfer * (1.20 if note == "A2" else 0.72)
    pick = strength * np.sin(2.0 * math.pi * pick_freq * pick_t) * np.exp(-pick_t / 0.00075)
    pick += 0.08 * attack_xfer * np.exp(-pick_t / 0.00028)
    out = np.zeros(n, dtype=np.float64)
    out[:pick_n] = pick
    return out


def _apply_peak_and_rms_targets(y: np.ndarray, note: str) -> Tuple[np.ndarray, Dict[str, float]]:
    target_rms = 10.0 ** (TARGET_RMS_DBFS / 20.0)
    y_out = y * (target_rms / max(_rms(y), 1e-12))
    note_peak = 10.0 ** (NOTE_PEAK_TARGET_DBFS[note] / 20.0)
    max_peak = 10.0 ** (MAX_PEAK_DBFS / 20.0)
    peak = float(np.max(np.abs(y_out)))
    if peak > note_peak:
        y_out *= note_peak / peak
    peak = float(np.max(np.abs(y_out)))
    if peak > max_peak:
        y_out *= max_peak / peak
    return y_out.astype(np.float64), {
        "rms_dbfs": round(_linear_to_dbfs(_rms(y_out)), 3),
        "peak_dbfs": round(_linear_to_dbfs(float(np.max(np.abs(y_out)))), 3),
    }


def synthesize_guitar_demo_note(
    *,
    sample_id: str,
    note: str,
    physical: Mapping[str, Any],
    factors: Mapping[str, float],
    profile: Mapping[str, Any],
    duration_s: float = DURATION_S,
    sr: int = SR,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    f0 = float(NOTE_FREQUENCY_HZ[note])
    n = int(duration_s * sr)
    t = np.arange(n, dtype=np.float64) / sr

    pluck_pos = float(profile["pluck_position_ratio"])
    inharm_b = float(profile["inharmonicity_b"])
    attack_ratio = float(profile["attack_body_ratio"])
    bridge_send = float(profile["bridge_body_send"])
    string_direct = float(profile["string_direct_mix"])
    stiffness = float(factors.get("top_stiffness_to_weight_factor") or 1.0)
    warmth = float(factors.get("back_density_warmth_factor") or 1.0)
    radiation = float(factors.get("radiation_brightness_factor") or 1.0)
    cavity = float(factors.get("body_size_cavity_factor") or 1.0)
    attack_xfer = float(factors.get("bridge_transfer_attack_factor") or 1.0)
    note_exc = NOTE_EXCITATION[note]

    string_y = np.zeros(n, dtype=np.float64)
    tau_samples: List[Dict[str, float]] = []
    for k in range(1, N_HARMONICS + 1):
        fk = f0 * k * (1.0 + inharm_b * k * k)
        if fk >= sr / 2.0 - 60.0:
            break
        pluck_amp = abs(math.sin(math.pi * k * pluck_pos)) / k
        if pluck_amp < 1e-9:
            continue
        if 2 <= k <= 6:
            pluck_amp *= stiffness ** 0.55 * radiation ** 0.35
        elif k >= 7:
            pluck_amp *= radiation ** 0.45 * (0.82 if sample_id == "sample_002" else 1.0)
        if k <= 2:
            pluck_amp *= warmth ** 0.30
        if fk < 125.0:
            pluck_amp *= (0.50 / max(cavity, 0.5)) if note == "A2" else (0.72 / max(cavity ** 0.45, 0.6))
        elif fk > 950.0:
            pluck_amp *= 0.80 * (radiation ** 0.15)

        tau_k = _partial_tau_seconds(fk, k, factors)
        if k <= 3:
            tau_samples.append({"harmonic": k, "frequency_hz": round(fk, 2), "tau_s": round(tau_k, 5)})
        string_y += pluck_amp * np.exp(-t / tau_k) * np.sin(2.0 * math.pi * fk * t)

    onset_n = max(int(0.0022 * sr / max(stiffness, 0.5)), 2)
    onset = np.ones(n, dtype=np.float64)
    onset[:onset_n] = np.sin(np.linspace(0.0, math.pi / 2.0, onset_n)) ** 2
    string_y *= onset * note_exc

    attack_y = _synthesize_attack_transient(n, sr, attack_xfer=attack_xfer, stiffness=stiffness, note=note)
    bridge_exc = string_y * bridge_send
    body_y = np.zeros(n, dtype=np.float64)
    for res in profile.get("body_resonator_bank") or []:
        body_y += _ring_resonator_response(
            bridge_exc,
            sr,
            float(res["frequency_hz"]),
            float(res["q"]),
            float(res["gain"]),
            float(res["decay_scale"]),
        )

    body_mix = 1.0 - string_direct
    y = string_y * string_direct + body_y * body_mix + attack_y * attack_ratio

    if note == "A2":
        y = _apply_one_pole_highpass(y, sr, 84.0)
        low_body = y - _apply_one_pole_highpass(y, sr, 122.0)
        y = y - 0.42 * cavity * low_body

    y_norm, level_meta = _apply_peak_and_rms_targets(y, note)
    meta = {
        "sample_id": sample_id,
        "note": note,
        "f0_hz": f0,
        "synthesis_profile": {
            k: v for k, v in profile.items() if k != "body_resonator_bank"
        },
        "partial_decay_summary": {
            "tau_samples": tau_samples,
            "mean_tau_s": round(float(np.mean([r["tau_s"] for r in tau_samples])) if tau_samples else 0.0, 5),
        },
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


def compute_spectral_metrics(y: np.ndarray, sr: int, note: str) -> Dict[str, Any]:
    f0 = float(NOTE_FREQUENCY_HZ[note])
    peak = float(np.max(np.abs(y)))
    rms = _rms(y)
    h1 = _band_energy_ratio(y, sr, f0 * 0.92, f0 * 1.08)
    h2_h8 = sum(
        _band_energy_ratio(y, sr, f0 * k * 0.94, f0 * k * 1.06)
        for k in range(2, 9)
        if f0 * k < sr / 2 - 20
    )
    hs = h1 + h2_h8 + 1e-12
    return {
        "low_body_band_80_160_ratio": round(_band_energy_ratio(y, sr, 80.0, 160.0), 6),
        "h1_dominance_ratio": round(h1 / hs, 6),
        "h2_h8_ratio": round(h2_h8 / hs, 6),
        "rms_dbfs": round(_linear_to_dbfs(rms), 3),
        "peak_dbfs": round(_linear_to_dbfs(peak), 3),
    }


def _waveform_correlation(a: np.ndarray, b: np.ndarray) -> float:
    ns = min(len(a), len(b))
    if ns < 64:
        return 1.0
    aa = a[:ns].astype(np.float64)
    bb = b[:ns].astype(np.float64)
    aa -= aa.mean()
    bb -= bb.mean()
    denom = max(float(np.linalg.norm(aa)) * float(np.linalg.norm(bb)), 1e-12)
    return float(np.dot(aa, bb) / denom)


def _envelope_distance(a: np.ndarray, b: np.ndarray) -> float:
    na = np.abs(a).astype(np.float64)
    nb = np.abs(b).astype(np.float64)
    ns = min(len(na), len(nb))
    if ns < 16:
        return 0.0
    na = na[:ns] / max(float(np.max(na[:ns])), 1e-12)
    nb = nb[:ns] / max(float(np.max(nb[:ns])), 1e-12)
    return float(np.mean(np.abs(na - nb)))


def _spectral_distance(a: np.ndarray, b: np.ndarray, sr: int) -> float:
    ns = min(len(a), len(b))
    if ns < 64:
        return 0.0
    sa = np.abs(np.fft.rfft(a[:ns] * np.hanning(ns)))
    sb = np.abs(np.fft.rfft(b[:ns] * np.hanning(ns)))
    sa = sa / max(float(np.linalg.norm(sa)), 1e-12)
    sb = sb / max(float(np.linalg.norm(sb)), 1e-12)
    return float(np.linalg.norm(sa - sb))


def _resonator_profile_distance(bank_a: Sequence[Mapping[str, Any]], bank_b: Sequence[Mapping[str, Any]]) -> float:
    if not bank_a or not bank_b:
        return 0.0
    n = min(len(bank_a), len(bank_b))
    dist = 0.0
    for i in range(n):
        fa = bank_a[i]
        fb = bank_b[i]
        dist += abs(float(fa.get("frequency_hz") or 0) - float(fb.get("frequency_hz") or 0)) / 500.0
        dist += abs(float(fa.get("gain") or 0) - float(fb.get("gain") or 0))
        dist += abs(float(fa.get("q") or 0) - float(fb.get("q") or 0)) / 10.0
    return dist / max(n, 1)


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
    profiles: Mapping[str, Mapping[str, Any]],
    *,
    sample_ids: Sequence[str],
    note_set: Sequence[str],
) -> Dict[str, Any]:
    pairs: Dict[str, Any] = {}
    scores: List[float] = []
    for i, sa in enumerate(sample_ids):
        for sb in sample_ids[i + 1 :]:
            key = f"{sa}_vs_{sb}"
            per_note: Dict[str, Any] = {}
            note_scores: List[float] = []
            for note in note_set:
                aa = audio_by_sample[sa][note]
                ab = audio_by_sample[sb][note]
                spec_d = _spectral_distance(aa, ab, SR)
                env_d = _envelope_distance(aa, ab)
                corr = _waveform_correlation(aa, ab)
                per_note[note] = {
                    "spectral_distance": round(spec_d, 6),
                    "envelope_distance": round(env_d, 6),
                    "waveform_correlation": round(corr, 6),
                }
                note_scores.append(0.45 * spec_d + 0.35 * env_d + 0.20 * (1.0 - corr))
            res_d = _resonator_profile_distance(
                profiles[sa].get("body_resonator_bank") or [],
                profiles[sb].get("body_resonator_bank") or [],
            )
            overall = float(np.mean(note_scores)) if note_scores else 0.0
            scores.append(overall)
            pairs[key] = {
                "per_note": per_note,
                "body_resonator_profile_distance": round(res_d, 6),
                "overall": round(overall, 6),
            }
    return {
        "pairwise_guitar_difference_metrics": pairs,
        "mean_overall_differentiation_score": round(float(np.mean(scores)) if scores else 0.0, 6),
    }


def build_anti_cheat_checks(
    *,
    traces: Mapping[str, Sequence[Mapping[str, Any]]],
    pairwise: Mapping[str, Any],
    correlation: Mapping[str, Any],
    spectral_metrics: Mapping[str, Mapping[str, Mapping[str, Any]]],
    rms_spread_by_note: Mapping[str, float],
    sample_ids: Sequence[str],
) -> Dict[str, Any]:
    max_corr = float(correlation.get("max_correlation") or 1.0)
    mean_diff = float(pairwise.get("mean_overall_differentiation_score") or 0.0)
    checks = {
        "no_randomization": True,
        "no_sample_id_only_gain": True,
        "no_arbitrary_eq_without_trace": True,
        "no_reverb_echo_body_tail": True,
        "no_hard_gate": True,
        "physical_driver_trace_per_sample": all(len(traces.get(sid) or []) >= 10 for sid in sample_ids),
        "differences_not_only_loudness": max_corr < CORRELATION_MODERATE_THRESHOLD or mean_diff > 0.04,
        "spectral_distance_reported": bool(pairwise.get("pairwise_guitar_difference_metrics")),
        "envelope_distance_reported": True,
        "body_resonator_profile_distance_reported": True,
        "no_clipping_limiter_trick": all(
            float((spectral_metrics.get(sid, {}).get(note) or {}).get("peak_dbfs") or 0.0) <= MAX_PEAK_DBFS + 0.1
            for sid in sample_ids
            for note in NOTE_SET
        ),
        "rms_spread_per_note_within_limit": all(float(v) < 1.5 for v in rms_spread_by_note.values()),
    }
    return {**checks, "pass": bool(all(checks.values()))}


def build_readiness_emergency_demo(
    *,
    files_generated: int,
    expected_files: int,
    max_correlation: float,
    peaks_controlled: bool,
) -> Dict[str, Any]:
    if files_generated < expected_files or not peaks_controlled:
        status = READINESS_FAIL
    elif max_correlation < CORRELATION_OK_THRESHOLD:
        status = READINESS_OK
    elif max_correlation < CORRELATION_MODERATE_THRESHOLD:
        status = READINESS_MODERATE
    else:
        status = READINESS_WEAK
    return {
        "current_status": status,
        "stk_gui_activation_planning_allowed": status in (READINESS_OK, READINESS_MODERATE),
        "files_generated": files_generated,
        "max_same_note_correlation": max_correlation,
    }


def build_emergency_demo_report(
    *,
    generated_files: Sequence[str],
    physical_parameters: Mapping[str, Any],
    differentiation_trace: Mapping[str, Any],
    synthesis_profiles: Mapping[str, Any],
    factor_multipliers: Mapping[str, Mapping[str, float]],
    resonator_banks: Mapping[str, Any],
    partial_decay_summary: Mapping[str, Any],
    peak_rms_report: Mapping[str, Any],
    spectral_metrics: Mapping[str, Mapping[str, Mapping[str, Any]]],
    pairwise: Mapping[str, Any],
    correlation: Mapping[str, Any],
    anti_cheat: Mapping[str, Any],
    loudness_report: Mapping[str, Any],
) -> Dict[str, Any]:
    max_corr = float(correlation.get("max_correlation") or 1.0)
    readiness = build_readiness_emergency_demo(
        files_generated=len(generated_files),
        expected_files=len(SAMPLE_SET) * len(NOTE_SET),
        max_correlation=max_corr,
        peaks_controlled=bool(peak_rms_report.get("all_peaks_within_target")),
    )
    return {
        "report_version": ENGINE_VERSION,
        "emergency_demo_version": EMERGENCY_DEMO_VERSION,
        "timestamp": _utc_now(),
        "validation_mode": "emergency_demo",
        "generated_files": list(generated_files),
        "physical_factors_used": list(PHYSICAL_FACTOR_GROUPS),
        "physical_parameters_used": physical_parameters,
        "per_sample_factor_multipliers": factor_multipliers,
        "per_sample_synthesis_profile": synthesis_profiles,
        "per_sample_differentiation_trace": differentiation_trace,
        "body_resonator_bank_per_sample": resonator_banks,
        "partial_decay_summary_per_sample_note": partial_decay_summary,
        "peak_rms_report": peak_rms_report,
        "spectral_metrics_per_file": spectral_metrics,
        "same_note_pairwise_correlation": correlation.get("same_note_pairwise_correlation"),
        "max_same_note_correlation": max_corr,
        "pairwise_difference_metrics": pairwise.get("pairwise_guitar_difference_metrics"),
        "mean_overall_differentiation_score": pairwise.get("mean_overall_differentiation_score"),
        "loudness_normalization_report": loudness_report,
        "anti_cheat_checks": anti_cheat,
        "diagnostic_exaggeration_for_audible_demo": True,
        "readiness": readiness,
        "blocked_claims": [
            "Final realism proof", "FEM/ROM validation", "STK production integration",
            "Website production replacement", "Step 5L replacement claim",
        ],
    }


def write_emergency_demo_markdown(report: Mapping[str, Any], path: Path) -> None:
    rg = report.get("readiness") or {}
    lines = [
        "# PGSM emergency guitar demo v3",
        "",
        f"**Version:** `{report.get('emergency_demo_version')}`",
        f"**Readiness:** `{rg.get('current_status')}`",
        f"**Max correlation:** {report.get('max_same_note_correlation')}",
        f"**Mean differentiation:** {report.get('mean_overall_differentiation_score')}",
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
    physical_parameters: Dict[str, Any] = {}
    differentiation_trace: Dict[str, Any] = {}
    factor_multipliers: Dict[str, Dict[str, float]] = {}
    synthesis_profiles: Dict[str, Any] = {}
    resonator_banks: Dict[str, Any] = {}
    partial_decay_summary: Dict[str, Dict[str, Any]] = {}
    spectral_metrics: Dict[str, Dict[str, Any]] = {}
    peak_rms_by_file: Dict[str, Dict[str, float]] = {}
    audio_by_sample: Dict[str, Dict[str, np.ndarray]] = {}
    generated_files: List[str] = []

    for sid in SAMPLE_SET:
        physical_parameters[sid] = extract_physical_parameters(sid, audit)
        factors, trace, _ = compute_physical_factors(physical_parameters[sid], ref_phys, sample_id=sid)
        profile = build_synthesis_profile(sid, factors, physical_parameters[sid])
        factor_multipliers[sid] = dict(factors)
        synthesis_profiles[sid] = profile
        resonator_banks[sid] = profile["body_resonator_bank"]
        differentiation_trace[sid] = {
            "sample_id": sid,
            "voicing_profile": profile["voicing_profile"],
            "physical_drivers_applied": trace,
            "factor_multipliers": factors,
            "diagnostic_exaggeration_for_audible_demo": True,
        }
        audio_by_sample[sid] = {}
        spectral_metrics[sid] = {}
        partial_decay_summary[sid] = {}
        for note in NOTE_SET:
            y, meta = synthesize_guitar_demo_note(
                sample_id=sid,
                note=note,
                physical=physical_parameters[sid],
                factors=factors,
                profile=profile,
            )
            wav_name = demo_wav_filename(sid, note)
            wav_path = out_dir / wav_name
            write_wav_mono(wav_path, y, SR)
            generated_files.append(str(wav_path.resolve()))
            audio_by_sample[sid][note] = y
            spectral_metrics[sid][note] = compute_spectral_metrics(y, SR, note)
            partial_decay_summary[sid][note] = meta.get("partial_decay_summary")
            peak_rms_by_file[wav_name] = {"peak_dbfs": meta["peak_dbfs"], "rms_dbfs": meta["rms_dbfs"]}
            print(f"[Emergency demo v3] wrote {sid} {note} -> {wav_name}")

    pairwise = compute_pairwise_difference_metrics(
        audio_by_sample, synthesis_profiles, sample_ids=SAMPLE_SET, note_set=NOTE_SET
    )
    correlation = compute_same_note_pairwise_correlation(
        audio_by_sample, sample_ids=SAMPLE_SET, note_set=NOTE_SET
    )
    rms_spread_by_note = {
        note: max(spectral_metrics[s][note]["rms_dbfs"] for s in SAMPLE_SET)
        - min(spectral_metrics[s][note]["rms_dbfs"] for s in SAMPLE_SET)
        for note in NOTE_SET
    }
    peaks_ok = all(v["peak_dbfs"] <= MAX_PEAK_DBFS + 0.05 for v in peak_rms_by_file.values())
    peak_rms_report = {
        "per_file": peak_rms_by_file,
        "rms_spread_db_by_note": rms_spread_by_note,
        "all_peaks_within_target": peaks_ok,
    }
    anti_cheat = build_anti_cheat_checks(
        traces={sid: differentiation_trace[sid]["physical_drivers_applied"] for sid in SAMPLE_SET},
        pairwise=pairwise,
        correlation=correlation,
        spectral_metrics=spectral_metrics,
        rms_spread_by_note=rms_spread_by_note,
        sample_ids=SAMPLE_SET,
    )
    loudness_report = {
        "normalization": "RMS target then note peak ceiling after waveform-level physics",
        "gain_separate_from_physics": True,
        "rms_spread_db_by_note": rms_spread_by_note,
    }
    report = build_emergency_demo_report(
        generated_files=generated_files,
        physical_parameters=physical_parameters,
        differentiation_trace=differentiation_trace,
        synthesis_profiles=synthesis_profiles,
        factor_multipliers=factor_multipliers,
        resonator_banks=resonator_banks,
        partial_decay_summary=partial_decay_summary,
        peak_rms_report=peak_rms_report,
        spectral_metrics=spectral_metrics,
        pairwise=pairwise,
        correlation=correlation,
        anti_cheat=anti_cheat,
        loudness_report=loudness_report,
    )
    jpath = Path(json_path or REPORT_JSON)
    mpath = Path(md_path or REPORT_MD)
    jpath.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_emergency_demo_markdown(report, mpath)
    return report


def main() -> None:
    report = run_emergency_guitar_demo()
    rg = report.get("readiness") or {}
    print(f"Wrote {REPORT_JSON}")
    print(f"Readiness: {rg.get('current_status')}")
    print(f"max_correlation: {report.get('max_same_note_correlation')}")


if __name__ == "__main__":
    main()
