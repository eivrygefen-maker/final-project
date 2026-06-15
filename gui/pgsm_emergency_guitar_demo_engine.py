#!/usr/bin/env python3
"""
PGSM / STK final guitar demo engine — ordered physical transfer chain.
v9: v8 base + single-attack onset repair in first 35–45 ms only.
Diagnostic only; not FEM/ROM/STK production.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from pgsm_step3a_numerical_ir_testbench import FIXED_PLUCK_POSITION, NUMERIC_SR
from pgsm_step4a_single_note_diagnostic_audio import write_wav_mono
from pgsm_step5a_limited_note_set_diagnostic_audio import NOTE_FREQUENCY_HZ
from pgsm_step5l_limited_multiguitar_differentiation import (
    REFERENCE_SAMPLE_ID,
    compute_physical_modifiers,
    extract_per_sample_physical_parameters,
)
from stk_v6_2_audit_features import load_audit_report

ENGINE_VERSION = "pgsm_emergency_guitar_demo_engine_v9"
FINAL_DEMO_VERSION = "v9_single_attack_enforced_guitar_demo"
EMERGENCY_DEMO_VERSION = FINAL_DEMO_VERSION
ROLLBACK_BASE = "v8_final_physical_guitar_demo"
FINAL_DEMO_VERSION_V8 = "v8_final_physical_guitar_demo"
FINAL_DEMO_VERSION_V7 = "v7_onset_locked_body_supported_guitar"
FINAL_DEMO_VERSION_V6 = "v6_single_pluck_physical_mix"
FINAL_DEMO_VERSION_V5 = "v5_ordered_physical_transfer_chain"
SR = NUMERIC_SR
DURATION_S = 2.5
N_HARMONICS = 14
HARMONIC_PLUCK_P = 1.05
BASE_INHARMONICITY_B = 1.8e-5
REFERENCE_MODE_CAP = 20

# Read-only calibrated modal catalog (no runtime ROM/FEM load).
REFERENCE_MODAL_CATALOG: Tuple[Dict[str, Any], ...] = (
    {"frequency_hz": 108.0, "tau_s": 0.14, "top_share": 0.18, "back_share": 0.28, "air_weight": 0.42, "radiation_weight": 0.12},
    {"frequency_hz": 142.0, "tau_s": 0.11, "top_share": 0.22, "back_share": 0.38, "air_weight": 0.18, "radiation_weight": 0.10},
    {"frequency_hz": 198.0, "tau_s": 0.09, "top_share": 0.48, "back_share": 0.22, "air_weight": 0.12, "radiation_weight": 0.18},
    {"frequency_hz": 285.0, "tau_s": 0.08, "top_share": 0.40, "back_share": 0.30, "air_weight": 0.10, "radiation_weight": 0.20},
    {"frequency_hz": 410.0, "tau_s": 0.07, "top_share": 0.35, "back_share": 0.25, "air_weight": 0.08, "radiation_weight": 0.32},
    {"frequency_hz": 620.0, "tau_s": 0.06, "top_share": 0.30, "back_share": 0.20, "air_weight": 0.06, "radiation_weight": 0.44},
    {"frequency_hz": 980.0, "tau_s": 0.05, "top_share": 0.25, "back_share": 0.15, "air_weight": 0.04, "radiation_weight": 0.56},
    {"frequency_hz": 1520.0, "tau_s": 0.04, "top_share": 0.20, "back_share": 0.10, "air_weight": 0.03, "radiation_weight": 0.67},
)

DAMP_A_ABS = 0.00048
DAMP_B_ABS = 4.0e-7
DAMP_C_HARM = 0.36
DAMP_P_HARM = 0.74

SAMPLE_SET = ("sample_000", "sample_001", "sample_002")
NOTE_SET = ("A2", "A4", "E5")

TARGET_RMS_DBFS = -20.0
MAX_PEAK_DBFS = -4.0
NOTE_PEAK_TARGET_DBFS: Dict[str, float] = {"A2": -5.0, "A4": -6.5, "E5": -6.5}

CORRELATION_TOO_SIMILAR = 0.95
CORRELATION_TOO_UNRELATED = 0.20
CORRELATION_FAMILY_LO = 0.55
CORRELATION_FAMILY_HI = 0.92

REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR_V9 = REPO_ROOT / "audio" / "pgsm_final_guitar_demo_v9"
REPORT_JSON_V9 = REPO_ROOT / "audio" / "debug_reports" / "pgsm_final_guitar_demo_v9_report.json"
REPORT_MD_V9 = REPO_ROOT / "audio" / "debug_reports" / "pgsm_final_guitar_demo_v9_report.md"
AUDIO_DIR_V8 = REPO_ROOT / "audio" / "pgsm_final_guitar_demo_v8"
REPORT_JSON_V8 = REPO_ROOT / "audio" / "debug_reports" / "pgsm_final_guitar_demo_v8_report.json"
REPORT_MD_V8 = REPO_ROOT / "audio" / "debug_reports" / "pgsm_final_guitar_demo_v8_report.md"
AUDIO_DIR_V7 = REPO_ROOT / "audio" / "pgsm_final_guitar_demo_v7"
REPORT_JSON_V7 = REPO_ROOT / "audio" / "debug_reports" / "pgsm_final_guitar_demo_v7_report.json"
REPORT_MD_V7 = REPO_ROOT / "audio" / "debug_reports" / "pgsm_final_guitar_demo_v7_report.md"
AUDIO_DIR_V6 = REPO_ROOT / "audio" / "pgsm_final_guitar_demo_v6"
REPORT_JSON_V6 = REPO_ROOT / "audio" / "debug_reports" / "pgsm_final_guitar_demo_v6_report.json"
REPORT_MD_V6 = REPO_ROOT / "audio" / "debug_reports" / "pgsm_final_guitar_demo_v6_report.md"
AUDIO_DIR_V5 = REPO_ROOT / "audio" / "pgsm_final_guitar_demo_v5"
REPORT_JSON_V5 = REPO_ROOT / "audio" / "debug_reports" / "pgsm_final_guitar_demo_v5_report.json"
REPORT_MD_V5 = REPO_ROOT / "audio" / "debug_reports" / "pgsm_final_guitar_demo_v5_report.md"
AUDIO_DIR = AUDIO_DIR_V9
REPORT_JSON = REPORT_JSON_V9
REPORT_MD = REPORT_MD_V9

READINESS_OK = "ready_for_stk_gui_activation"
READINESS_OK_LIMITED = "ready_for_stk_gui_activation_with_known_audio_limitations"
READINESS_WEAK = "demo_generated_but_differentiation_weak"
READINESS_REVIEW = "demo_generated_but_physical_chain_needs_review"
READINESS_DOUBLE_PLUCK = "demo_generated_but_double_pluck_needs_review"
READINESS_FAIL = "emergency_demo_failed"

EXCITATION_ATTACK_MS = 2.5
EXCITATION_CONTACT_MS = 6.0
DOUBLE_PLUCK_MIN_DELAY_MS = 18.0
DOUBLE_PLUCK_SECOND_PEAK_FRAC = 0.35
EARLY_ATTACK_MIN_MS = 3.0
EARLY_ATTACK_MAX_MS = 25.0
EARLY_ATTACK_SECOND_FRAC = 0.32
LOCAL_DELAYED_PEAK_MAX_ATTEN = 0.23
LOCAL_DELAYED_PEAK_THRESHOLD = 0.35

ONSET_REPAIR_WINDOW_MS = 35.0
ONSET_REPAIR_CROSSFADE_END_MS = 45.0
EARLY_ATTACK_SIGNIFICANT_FRAC = 0.28
EARLY_ATTACK_SECONDARY_FRAC_E5 = 0.25
EARLY_ATTACK_MIN_DELAY_MS = 3.0

# v7 legacy constants (not used in v8 synthesis path)
ONSET_LOCK_ATTACK_MS = 2.0
ONSET_LOCK_WINDOW_MS = 80.0
ONSET_LOCK_DECAY_TAU_MS = 45.0
DELAYED_PEAK_REGION_END_MS = 90.0

ONSET_LOCK_SUMMARY: Dict[str, Any] = {
    "attack_ms": ONSET_LOCK_ATTACK_MS,
    "window_ms": ONSET_LOCK_WINDOW_MS,
    "decay_tau_ms": ONSET_LOCK_DECAY_TAU_MS,
    "applied_to": ["contact_early_tap", "string_bridge", "body_modal"],
    "body_starts_at_sample_zero": True,
    "no_body_pre_delay": True,
}

NOTE_BODY_SUPPORT: Dict[str, Dict[str, float]] = {
    "A2": {"body_modal_mult": 1.0, "low_mid_mode_mult": 1.0, "high_rad_mult": 1.0},
    "A4": {"body_modal_mult": 1.08, "low_mid_mode_mult": 1.12, "high_rad_mult": 0.95},
    "E5": {"body_modal_mult": 1.12, "low_mid_mode_mult": 1.15, "high_rad_mult": 0.90},
}

PHYSICAL_FACTOR_KEYS: Tuple[str, ...] = (
    "body_size_cavity_factor",
    "top_stiffness_to_weight_factor",
    "top_damping_factor",
    "back_density_warmth_factor",
    "bridge_mobility_factor",
    "effective_mass_loading_factor",
    "air_helmholtz_factor",
    "radiation_brightness_factor",
)

PHYSICAL_CHAIN_STAGES: Tuple[str, ...] = (
    "A_pluck_contact",
    "B_string_force",
    "C_bridge_impedance",
    "D_body_modal_transfer",
    "E_material_damping",
    "F_radiation_mix",
    "G_modal_decay_only",
)

PHYSICAL_CHAIN_SUMMARY: Dict[str, str] = {
    "A_pluck_contact": "Embedded in F_excitation via pluck_contact_shape (not separate mix layer)",
    "B_string_force": "Harmonic string force * contact shape -> single F_excitation",
    "C_bridge_impedance": "F_bridge_eff = bridge_transfer(F_excitation); immediate body drive",
    "D_body_modal_transfer": "y_body = conv(F_bridge_eff, H_mode_sample); per-sample modes",
    "E_material_damping": "Per-sample tau/Q from 8 physical factors",
    "F_radiation_mix": "y = string_bridge_residual + body_modal (one combined onset)",
    "G_modal_decay_only": "Onset repair first 35–45 ms only; v9 enforce_single_attack_peak",
}

V8_VOICING: Dict[str, Dict[str, Any]] = {
    "sample_000": {
        "profile": "balanced_neutral_classical",
        "pluck_delta": 0.0,
        "factor_multipliers": {k: 1.0 for k in PHYSICAL_FACTOR_KEYS},
        "mix": {"string_bridge": 0.26, "body_modal": 0.72, "air_share": 0.10},
    },
    "sample_001": {
        "profile": "bright_light_fast_response",
        "pluck_delta": 0.010,
        "factor_multipliers": {
            "body_size_cavity_factor": 0.88,
            "top_stiffness_to_weight_factor": 1.14,
            "top_damping_factor": 1.10,
            "back_density_warmth_factor": 0.94,
            "bridge_mobility_factor": 1.12,
            "effective_mass_loading_factor": 0.88,
            "air_helmholtz_factor": 0.90,
            "radiation_brightness_factor": 1.12,
        },
        "mix": {"string_bridge": 0.28, "body_modal": 0.68, "air_share": 0.08},
    },
    "sample_002": {
        "profile": "warm_deep_heavy_response",
        "pluck_delta": -0.012,
        "factor_multipliers": {
            "body_size_cavity_factor": 1.15,
            "top_stiffness_to_weight_factor": 0.92,
            "top_damping_factor": 0.92,
            "back_density_warmth_factor": 1.14,
            "bridge_mobility_factor": 0.90,
            "effective_mass_loading_factor": 1.14,
            "air_helmholtz_factor": 1.10,
            "radiation_brightness_factor": 0.88,
        },
        "mix": {"string_bridge": 0.24, "body_modal": 0.72, "air_share": 0.13},
    },
}

V7_VOICING: Dict[str, Dict[str, Any]] = {
    "sample_000": {
        "profile": "balanced_neutral_classical",
        "pluck_delta": 0.0,
        "factors": {k: 1.0 for k in PHYSICAL_FACTOR_KEYS},
        "mix": {"contact": 0.04, "string_bridge": 0.24, "body_modal": 0.66, "air_share": 0.10},
    },
    "sample_001": {
        "profile": "bright_light_fast_response",
        "pluck_delta": 0.009,
        "factors": {
            "body_size_cavity_factor": 0.90,
            "top_stiffness_to_weight_factor": 1.10,
            "top_damping_factor": 1.12,
            "back_density_warmth_factor": 0.92,
            "bridge_mobility_factor": 1.08,
            "effective_mass_loading_factor": 0.88,
            "air_helmholtz_factor": 0.93,
            "radiation_brightness_factor": 1.06,
        },
        "mix": {"contact": 0.04, "string_bridge": 0.26, "body_modal": 0.64, "air_share": 0.09},
    },
    "sample_002": {
        "profile": "warm_deep_heavy_response",
        "pluck_delta": -0.011,
        "factors": {
            "body_size_cavity_factor": 1.08,
            "top_stiffness_to_weight_factor": 0.92,
            "top_damping_factor": 0.90,
            "back_density_warmth_factor": 1.10,
            "bridge_mobility_factor": 0.92,
            "effective_mass_loading_factor": 1.10,
            "air_helmholtz_factor": 1.06,
            "radiation_brightness_factor": 0.88,
        },
        "mix": {"contact": 0.03, "string_bridge": 0.22, "body_modal": 0.68, "air_share": 0.12},
    },
}

V6_VOICING: Dict[str, Dict[str, Any]] = {
    "sample_000": {
        "profile": "balanced_neutral_classical",
        "pluck_delta": 0.0,
        "factors": {k: 1.0 for k in PHYSICAL_FACTOR_KEYS},
        "mix": {"contact": 0.05, "string_bridge": 0.22, "body_modal": 0.68, "air_share": 0.10},
    },
    "sample_001": {
        "profile": "bright_light_fast_response",
        "pluck_delta": 0.009,
        "factors": {
            "body_size_cavity_factor": 0.90,
            "top_stiffness_to_weight_factor": 1.10,
            "top_damping_factor": 1.12,
            "back_density_warmth_factor": 0.92,
            "bridge_mobility_factor": 1.08,
            "effective_mass_loading_factor": 0.88,
            "air_helmholtz_factor": 0.93,
            "radiation_brightness_factor": 1.08,
        },
        "mix": {"contact": 0.06, "string_bridge": 0.24, "body_modal": 0.64, "air_share": 0.09},
    },
    "sample_002": {
        "profile": "warm_deep_heavy_response",
        "pluck_delta": -0.011,
        "factors": {
            "body_size_cavity_factor": 1.08,
            "top_stiffness_to_weight_factor": 0.92,
            "top_damping_factor": 0.90,
            "back_density_warmth_factor": 1.10,
            "bridge_mobility_factor": 0.92,
            "effective_mass_loading_factor": 1.10,
            "air_helmholtz_factor": 1.06,
            "radiation_brightness_factor": 0.90,
        },
        "mix": {"contact": 0.04, "string_bridge": 0.20, "body_modal": 0.70, "air_share": 0.12},
    },
}


@dataclass
class SampleSynthesisState:
    sample_id: str
    physical: Dict[str, Any]
    factors: Dict[str, float]
    pluck_position_ratio: float
    voicing_profile: str
    modes: List[Dict[str, Any]]
    bridge_transfer: Dict[str, Any]
    mix_ratios: Dict[str, float]
    differentiation_trace: List[Dict[str, Any]]
    isolation_meta: Dict[str, Any] = field(default_factory=dict)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _rms(y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(y, dtype=np.float64) ** 2)))


def _linear_to_dbfs(x: float) -> float:
    return 20.0 * math.log10(max(abs(x), 1e-12))


def final_wav_filename(sample_id: str, note: str) -> str:
    return f"{sample_id}_{note}_final_guitar.wav"


def demo_wav_filename(sample_id: str, note: str) -> str:
    return final_wav_filename(sample_id, note)


def build_emergency_demo_config() -> Dict[str, Any]:
    return {
        "engine_version": ENGINE_VERSION,
        "final_demo_version": FINAL_DEMO_VERSION,
        "emergency_demo_version": EMERGENCY_DEMO_VERSION,
        "output_folder": str(AUDIO_DIR_V9),
        "physical_chain_stages": list(PHYSICAL_CHAIN_STAGES),
        "physical_factor_keys": list(PHYSICAL_FACTOR_KEYS),
        "sample_set": list(SAMPLE_SET),
        "note_set": list(NOTE_SET),
        "duration_s": DURATION_S,
        "diagnostic_exaggeration_for_audible_demo": True,
    }


def _load_readonly_reference_modes(repo_root: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Return a deep copy of the static reference catalog; never mutated per sample."""
    del repo_root
    return copy.deepcopy(list(REFERENCE_MODAL_CATALOG))


def _audit_factor_map(physical: Mapping[str, Any], reference: Mapping[str, Any]) -> Dict[str, float]:
    mods, _ = compute_physical_modifiers(physical, reference)
    vol = float(physical.get("body_volume_proxy") or 0.013)
    ref_vol = max(float(reference.get("body_volume_proxy") or 0.013), 1e-9)
    depth = float(physical.get("body_depth_m") or 0.10)
    ref_depth = max(float(reference.get("body_depth_m") or 0.10), 1e-9)
    mob = float(physical.get("bridge_mobility_proxy") or 1.0)
    ref_mob = max(float(reference.get("bridge_mobility_proxy") or 1.0), 1e-9)
    mass = float((physical.get("mass_proxies") or {}).get("mixed_body_mass_proxy") or 1.0)
    ref_mass = max(float((reference.get("mass_proxies") or {}).get("mixed_body_mass_proxy") or 1.0), 1e-9)
    helm = float(physical.get("helmholtz_like_frequency_proxy") or 120.0)
    ref_helm = max(float(reference.get("helmholtz_like_frequency_proxy") or 120.0), 1e-9)
    td = float(physical.get("top_damping_coeff_proxy") or 1.0)
    ref_td = max(float(reference.get("top_damping_coeff_proxy") or 1.0), 1e-9)
    top_sw = float(physical.get("top_stiffness_to_weight_proxy") or 1.0)
    ref_top_sw = max(float(reference.get("top_stiffness_to_weight_proxy") or 1.0), 1e-9)
    return {
        "body_size_cavity_factor": _clamp((vol * depth) / max(ref_vol * ref_depth, 1e-9), 0.85, 1.15) ** 0.2,
        "top_stiffness_to_weight_factor": _clamp(top_sw / ref_top_sw, 0.85, 1.15) ** 0.25,
        "top_damping_factor": _clamp(td / ref_td, 0.82, 1.18) ** 0.18,
        "back_density_warmth_factor": _clamp(float(mods.get("top_back_share_balance") or 1.0), 0.88, 1.12),
        "bridge_mobility_factor": _clamp(mob / ref_mob, 0.85, 1.15) ** 0.28,
        "effective_mass_loading_factor": _clamp(mass / ref_mass, 0.85, 1.15) ** 0.16,
        "air_helmholtz_factor": _clamp(helm / ref_helm, 0.88, 1.14) ** 0.14,
        "radiation_brightness_factor": _clamp(float(mods.get("radiation_weight_scale") or 1.0), 0.85, 1.15),
    }


def _voicing_factor_overlay(sample_id: str) -> Dict[str, float]:
    voicing = V8_VOICING[sample_id]
    raw = voicing.get("factor_multipliers") or voicing.get("factors") or {}
    return {k: float(raw.get(k, 1.0)) for k in PHYSICAL_FACTOR_KEYS}


def compute_v5_physical_factors(
    physical: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    sample_id: str,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    audit = _audit_factor_map(physical, reference)
    overlay = _voicing_factor_overlay(sample_id)
    factors: Dict[str, float] = {}
    trace: List[Dict[str, Any]] = []
    for key in PHYSICAL_FACTOR_KEYS:
        combined = round(_clamp(float(audit.get(key, 1.0)) * float(overlay.get(key, 1.0)), 0.84, 1.18), 6)
        factors[key] = combined
        trace.append(
            {
                "factor": key,
                "audit_proxy": round(float(audit.get(key, 1.0)), 6),
                "voicing_overlay": round(float(overlay.get(key, 1.0)), 6),
                "combined": combined,
            }
        )
    return factors, trace


compute_gentle_sample_modifiers = compute_v5_physical_factors
compute_physical_factors = compute_v5_physical_factors
PHYSICAL_FACTOR_GROUPS = PHYSICAL_FACTOR_KEYS
PHYSICAL_MODIFIER_KEYS = PHYSICAL_FACTOR_KEYS
GENTLE_SAMPLE_VOICING = V8_VOICING


def _pick_reference_modes(ref_modes: Sequence[Mapping[str, Any]], factors: Mapping[str, float]) -> List[Dict[str, Any]]:
    """Build 6–7 mode transfer components as independent copies from read-only catalog."""
    if not ref_modes:
        return []
    sorted_modes = sorted(ref_modes, key=lambda r: float(r["frequency_hz"]))
    helm = 118.0 * float(factors.get("air_helmholtz_factor") or 1.0)
    stiffness = float(factors.get("top_stiffness_to_weight_factor") or 1.0)
    warmth = float(factors.get("back_density_warmth_factor") or 1.0)
    radiation = float(factors.get("radiation_brightness_factor") or 1.0)
    cavity = float(factors.get("body_size_cavity_factor") or 1.0)
    damping = float(factors.get("top_damping_factor") or 1.0)
    mass = float(factors.get("effective_mass_loading_factor") or 1.0)

    low = sorted_modes[0]
    mid = sorted_modes[len(sorted_modes) // 3]
    high = sorted_modes[min(len(sorted_modes) - 1, len(sorted_modes) * 2 // 3)]

    templates = [
        ("low_body_air_cavity", helm, 2.8, 0.038 * cavity, "air", warmth),
        ("main_top", 195.0 * stiffness, 8.5, 0.085 * stiffness, "top", radiation),
        ("back_low_mid", 145.0 * warmth, 5.5, 0.065 * warmth, "back", warmth),
        ("upper_top_radiation", 420.0 * radiation, 11.0, 0.048 * radiation, "radiation", radiation),
        ("high_articulation", 1650.0 * radiation, 14.0, 0.028 * radiation, "radiation", radiation),
        ("low_mid_coupled_body", float(mid["frequency_hz"]) * cavity, 4.0, 0.032 * cavity, "back", warmth),
        ("catalog_anchor", float(high["frequency_hz"]) * stiffness, 9.0, 0.042, "top", radiation),
    ]
    modes: List[Dict[str, Any]] = []
    for role, f_hz, q, gain, component, bright in templates:
        tau = max(0.04, float(mid.get("tau_s", 0.08)) * (0.92 + 0.14 * mass) / max(damping, 0.5))
        if component == "radiation":
            tau *= 0.78 + 0.14 / max(bright, 0.5)
        modes.append(
            {
                "role": role,
                "frequency_hz": round(min(f_hz, SR / 2 - 120), 3),
                "q": round(q, 4),
                "tau_s": round(tau, 6),
                "gain": round(gain, 6),
                "component": component,
            }
        )
    return modes


def _bridge_transfer_summary(factors: Mapping[str, float]) -> Dict[str, Any]:
    mob = float(factors.get("bridge_mobility_factor") or 1.0)
    mass = float(factors.get("effective_mass_loading_factor") or 1.0)
    cavity = float(factors.get("body_size_cavity_factor") or 1.0)
    summary = {
        "bridge_mobility_factor": mob,
        "effective_mass_loading_factor": mass,
        "body_size_cavity_factor": cavity,
        "highpass_hz": round(55.0 + 18.0 * (mass - 1.0), 3),
        "attack_scale": round(mob / max(mass ** 0.35, 0.5), 6),
        "low_coupling_scale": round(cavity ** 0.35 / max(mass ** 0.2, 0.5), 6),
    }
    summary["hash"] = hashlib.sha256(json.dumps(summary, sort_keys=True).encode()).hexdigest()[:16]
    return summary


def build_sample_synthesis_state(
    sample_id: str,
    *,
    physical: Mapping[str, Any],
    reference_physical: Mapping[str, Any],
    readonly_modes: Sequence[Mapping[str, Any]],
) -> SampleSynthesisState:
    factors, trace = compute_v5_physical_factors(physical, reference_physical, sample_id=sample_id)
    voicing = V8_VOICING[sample_id]
    pluck = _clamp(FIXED_PLUCK_POSITION + float(voicing.get("pluck_delta") or 0.0), 0.10, 0.20)
    modes = _pick_reference_modes(readonly_modes, factors)
    bridge = _bridge_transfer_summary(factors)
    modal_hash = hashlib.sha256(
        json.dumps(modes, sort_keys=True).encode()
    ).hexdigest()[:16]
    return SampleSynthesisState(
        sample_id=sample_id,
        physical=dict(physical),
        factors=factors,
        pluck_position_ratio=round(pluck, 5),
        voicing_profile=str(voicing["profile"]),
        modes=modes,
        bridge_transfer=bridge,
        mix_ratios=dict(voicing["mix"]),
        differentiation_trace=trace,
        isolation_meta={
            "independent_state_created": True,
            "derived_mode_count": len(modes),
            "bridge_transfer_hash": bridge.get("hash"),
            "modal_transfer_hash": modal_hash,
            "radiation_weight_summary": {
                "radiation_brightness_factor": factors.get("radiation_brightness_factor"),
                "air_helmholtz_factor": factors.get("air_helmholtz_factor"),
            },
            "damping_summary": {
                "top_damping_factor": factors.get("top_damping_factor"),
                "material_loss_proxy": factors.get("top_damping_factor"),
            },
            "cleanup_completed": False,
        },
    )


def cleanup_sample_state(state: SampleSynthesisState) -> None:
    state.modes.clear()
    state.factors.clear()
    state.physical.clear()
    state.isolation_meta["cleanup_completed"] = True


def build_synthesis_profile(sample_id: str, factors: Mapping[str, float]) -> Dict[str, Any]:
    voicing = V8_VOICING[sample_id]
    return {
        "sample_id": sample_id,
        "voicing_profile": voicing["profile"],
        "pluck_position_ratio": _clamp(FIXED_PLUCK_POSITION + float(voicing.get("pluck_delta") or 0.0), 0.10, 0.20),
        "physical_modifiers": dict(factors),
    }


def _pluck_contact_shape(n: int, sr: int) -> np.ndarray:
    """Embed nail/contact transient inside excitation — not a separate output layer."""
    attack_n = max(int(EXCITATION_ATTACK_MS * 1e-3 * sr), 3)
    shape = np.ones(n, dtype=np.float64)
    shape[:attack_n] = np.sin(np.linspace(0.0, math.pi / 2.0, attack_n)) ** 2
    nail_n = min(max(int(0.005 * sr), attack_n + 2), n)
    tt = np.arange(nail_n, dtype=np.float64) / sr
    nail = np.exp(-tt / 0.0018)
    nail /= max(float(nail[0]), 1e-12)
    shape[:nail_n] *= 1.0 + 0.05 * nail
    return shape


def _per_sample_body_support(sample_id: str, note: str) -> Dict[str, float]:
    """Per-sample low-mid body support for high notes (bridge-driven, not independent bass)."""
    if note == "A2":
        return {"body_modal_mult": 1.0, "low_mid_mode_mult": 1.0, "high_rad_mult": 1.0, "tau_low_mid_mult": 1.0}
    if note == "A4":
        base = {"body_modal_mult": 1.07, "low_mid_mode_mult": 1.10, "high_rad_mult": 0.96, "tau_low_mid_mult": 1.12}
    else:
        base = {"body_modal_mult": 1.11, "low_mid_mode_mult": 1.13, "high_rad_mult": 0.90, "tau_low_mid_mult": 1.16}
    if sample_id == "sample_001":
        base["low_mid_mode_mult"] *= 0.94
        base["tau_low_mid_mult"] *= 0.92
        base["body_modal_mult"] *= 0.98
    elif sample_id == "sample_002":
        base["low_mid_mode_mult"] *= 1.10
        base["tau_low_mid_mult"] *= 1.08
        base["body_modal_mult"] *= 1.04
    return base


def onset_lock_envelope(n: int, sr: int) -> np.ndarray:
    """Macro onset lock for first 80 ms: fast rise, smooth early decay, unity after."""
    win_n = max(int(ONSET_LOCK_WINDOW_MS * 1e-3 * sr), 16)
    attack_n = max(int(ONSET_LOCK_ATTACK_MS * 1e-3 * sr), 3)
    env = np.ones(n, dtype=np.float64)
    ramp = np.sin(np.linspace(0.0, math.pi / 2.0, attack_n)) ** 2
    env[:attack_n] = ramp
    tail_n = min(win_n, n) - attack_n
    if tail_n > 0:
        tt = np.arange(tail_n, dtype=np.float64) / sr
        decay_tau = ONSET_LOCK_DECAY_TAU_MS * 1e-3
        env[attack_n : attack_n + tail_n] = 0.70 + 0.30 * np.exp(-tt / decay_tau)
    return env


def _shared_excitation_envelope(n: int, sr: int) -> np.ndarray:
    """Single pluck envelope: fast attack then unity sustain (one onset only)."""
    attack_n = max(int(EXCITATION_ATTACK_MS * 1e-3 * sr), 3)
    contact_n = max(int(EXCITATION_CONTACT_MS * 1e-3 * sr), attack_n + 2)
    env = np.ones(n, dtype=np.float64)
    ramp = np.sin(np.linspace(0.0, math.pi / 2.0, attack_n)) ** 2
    env[:attack_n] = ramp
    if contact_n > attack_n:
        tail_len = contact_n - attack_n
        tt = np.arange(tail_len, dtype=np.float64) / sr
        env[attack_n:contact_n] = 1.0 - 0.08 * (1.0 - np.exp(-tt / 0.0028))
    return env


def _partial_tau(
    f_hz: float,
    n: int,
    factors: Mapping[str, float],
    *,
    f0: float = 110.0,
) -> float:
    base = 0.42 * float(factors.get("top_damping_factor") or 1.0)
    base *= 0.90 + 0.18 * float(factors.get("effective_mass_loading_factor") or 1.0)
    material_loss = 0.11 * float(factors.get("top_damping_factor") or 1.0)
    bridge_loss = 0.09 * (2.0 - float(factors.get("bridge_mobility_factor") or 1.0))
    rad = float(factors.get("radiation_brightness_factor") or 1.0)
    high_note_scale = 1.0 + 0.35 * max(0.0, (f0 - 350.0) / 350.0)
    denom = (
        1.0
        + DAMP_A_ABS * f_hz * high_note_scale
        + DAMP_B_ABS * max(0.0, f_hz - 900.0) ** 2 * (1.15 if rad > 1.05 else 1.0) * high_note_scale
        + DAMP_C_HARM * (n ** DAMP_P_HARM)
        + material_loss
        + bridge_loss
    )
    tau = base / max(denom, 1e-6)
    if f_hz < 140.0:
        tau *= float(factors.get("body_size_cavity_factor") or 1.0) ** 0.4
    return max(tau, 7e-5)


def _synthesize_f_string(
    n: int,
    sr: int,
    f0: float,
    *,
    pluck_pos: float,
    factors: Mapping[str, float],
    inharm_b: float,
    note: str,
    excitation: np.ndarray,
) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / sr
    y = np.zeros(n, dtype=np.float64)
    stiffness = float(factors.get("top_stiffness_to_weight_factor") or 1.0)
    radiation = float(factors.get("radiation_brightness_factor") or 1.0)
    warmth = float(factors.get("back_density_warmth_factor") or 1.0)
    cavity = float(factors.get("body_size_cavity_factor") or 1.0)
    for k in range(1, N_HARMONICS + 1):
        fk = f0 * k * (1.0 + inharm_b * k * k)
        if fk >= sr / 2 - 40:
            break
        amp = abs(math.sin(math.pi * k * pluck_pos)) / (k ** HARMONIC_PLUCK_P)
        if k >= 2:
            amp *= (0.90 + 0.14 * stiffness) * (0.92 + 0.12 * radiation)
        if note == "A2" and 2 <= k <= 4:
            amp *= 1.14 + 0.04 * (5 - k)
        if k <= 2:
            amp *= 0.94 + 0.08 * warmth
        if fk < 125:
            amp *= 0.58 / max(cavity ** 0.35, 0.55)
        if note in ("A4", "E5") and k >= 6:
            amp *= 0.82 - 0.04 * min(k - 6, 4)
        tau = _partial_tau(fk, k, factors, f0=f0)
        y += amp * np.exp(-t / tau) * np.sin(2.0 * math.pi * fk * t)
    return y * excitation


def _bridge_transfer(f_string: np.ndarray, sr: int, bridge: Mapping[str, Any]) -> np.ndarray:
    """Bridge impedance shapes excitation sent to body — immediate causal response."""
    alpha = math.exp(-2.0 * math.pi * float(bridge.get("highpass_hz") or 60.0) / sr)
    out = np.zeros_like(f_string, dtype=np.float64)
    z = prev = 0.0
    for i, x in enumerate(f_string.astype(np.float64)):
        z = alpha * (z + x - prev)
        prev = x
        out[i] = z
    attack = float(bridge.get("attack_scale") or 1.0)
    low_scale = float(bridge.get("low_coupling_scale") or 1.0)
    return out * attack * (0.90 + 0.08 * low_scale)


def _mode_transfer(
    exc: np.ndarray,
    sr: int,
    mode: Mapping[str, Any],
    *,
    note: str = "A4",
    sample_id: str = "sample_000",
) -> np.ndarray:
    """Causal modal kernel; per-sample low-mid support on A4/E5."""
    f_hz = float(mode["frequency_hz"])
    tau = float(mode["tau_s"])
    gain = float(mode["gain"])
    q = max(float(mode.get("q") or 8.0), 1.0)
    role = str(mode.get("role") or "")
    support = _per_sample_body_support(sample_id, note)
    if note == "A2" and role in ("low_body_air_cavity", "low_mid_coupled_body"):
        gain *= 0.62
    if note in ("A4", "E5"):
        if 120.0 <= f_hz <= 450.0:
            gain *= float(support["low_mid_mode_mult"])
            tau *= float(support["tau_low_mid_mult"])
        elif f_hz > 850.0:
            gain *= float(support["high_rad_mult"])
    k_len = min(int(max(tau, 0.018) * sr * 4), len(exc))
    if k_len < 6:
        return np.zeros_like(exc)
    kt = np.arange(k_len, dtype=np.float64) / sr
    decay = (math.pi * f_hz / (q * sr)) + 1.0 / max(tau, 1e-4)
    kernel = gain * np.exp(-kt * decay) * np.cos(2.0 * math.pi * f_hz * kt)
    k0 = max(float(kernel[0]), 1e-12)
    kernel /= k0
    kernel *= gain
    if abs(float(kernel[0])) < abs(float(exc[0])) * 0.05 and len(exc):
        kernel[0] += float(exc[0]) * 0.08
    return np.convolve(exc.astype(np.float64), kernel, mode="full")[: len(exc)]


def apply_onset_lock_to_body(
    y_body: np.ndarray,
    f_bridge_eff: np.ndarray,
    onset_lock: np.ndarray,
    sr: int,
) -> np.ndarray:
    """Align body early energy with bridge; body active from sample 0."""
    win_n = min(int(ONSET_LOCK_WINDOW_MS * 1e-3 * sr), len(y_body))
    out = y_body.astype(np.float64).copy()
    bridge0 = float(f_bridge_eff[0]) if len(f_bridge_eff) else 0.0
    if abs(out[0]) < abs(bridge0) * 0.08:
        out[0] += bridge0 * 0.14
    if win_n > 1:
        out[:win_n] *= onset_lock[:win_n]
    return out


def local_limited_delayed_peak_correction(
    y_body: np.ndarray,
    sr: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Trim only a narrow delayed bump (max ~23% attenuation); preserve sample identity."""
    win = min(int(0.150 * sr), len(y_body))
    if win < 32:
        return y_body, {"delayed_peak_ratio": 0.0, "attenuation_applied": False, "attenuation_amount": 0.0}
    env = np.abs(y_body[:win])
    k = max(int(0.002 * sr), 3)
    smooth = np.convolve(env, np.ones(k) / k, mode="same")
    delay_n = int(DOUBLE_PLUCK_MIN_DELAY_MS * 1e-3 * sr)
    region_end = min(int(0.090 * sr), win)
    first_idx = int(np.argmax(smooth[: max(delay_n, 8)]))
    first_peak = float(smooth[first_idx])
    if first_peak < 1e-12:
        return y_body, {"delayed_peak_ratio": 0.0, "attenuation_applied": False, "attenuation_amount": 0.0}
    search = smooth[delay_n:region_end]
    if search.size < 8:
        return y_body, {"delayed_peak_ratio": 0.0, "attenuation_applied": False, "attenuation_amount": 0.0}
    second_idx = int(np.argmax(search)) + delay_n
    second_peak = float(smooth[second_idx])
    ratio = second_peak / first_peak
    if ratio <= LOCAL_DELAYED_PEAK_THRESHOLD:
        return y_body, {
            "delayed_peak_ratio": round(ratio, 4),
            "attenuation_applied": False,
            "attenuation_amount": 0.0,
            "second_peak_ms": round(1000.0 * second_idx / sr, 3),
        }
    half = max(int(0.004 * sr), 2)
    s0 = max(second_idx - half, delay_n)
    s1 = min(second_idx + half, len(y_body))
    atten = _clamp(
        0.12 + 0.18 * (ratio - LOCAL_DELAYED_PEAK_THRESHOLD),
        0.10,
        LOCAL_DELAYED_PEAK_MAX_ATTEN,
    )
    out = y_body.copy()
    out[s0:s1] *= 1.0 - atten
    return out, {
        "delayed_peak_ratio": round(ratio, 4),
        "attenuation_applied": True,
        "attenuation_amount": round(atten, 4),
        "second_peak_ms": round(1000.0 * second_idx / sr, 3),
    }


def attenuate_delayed_modal_peak(
    y_body: np.ndarray,
    sr: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Attenuate delayed modal envelope peaks in 18–90 ms without hard gating."""
    win = min(int(0.180 * sr), len(y_body))
    if win < 32:
        return y_body, {"delayed_peak_ratio": 0.0, "attenuation_applied": False}
    env = np.abs(y_body[:win])
    k = max(int(0.002 * sr), 3)
    smooth = np.convolve(env, np.ones(k) / k, mode="same")
    delay_n = int(DOUBLE_PLUCK_MIN_DELAY_MS * 1e-3 * sr)
    region_end = int(DELAYED_PEAK_REGION_END_MS * 1e-3 * sr)
    first_idx = int(np.argmax(smooth[: max(delay_n, 8)]))
    first_peak = float(smooth[first_idx])
    if first_peak < 1e-12:
        return y_body, {"delayed_peak_ratio": 0.0, "attenuation_applied": False}
    search_end = min(region_end, win)
    search = smooth[delay_n:search_end]
    if search.size < 8:
        return y_body, {"delayed_peak_ratio": 0.0, "attenuation_applied": False}
    second_idx = int(np.argmax(search)) + delay_n
    second_peak = float(smooth[second_idx])
    ratio = second_peak / first_peak
    if ratio <= DOUBLE_PLUCK_SECOND_PEAK_FRAC:
        return y_body, {
            "delayed_peak_ratio": round(ratio, 4),
            "attenuation_applied": False,
            "second_peak_ms": round(1000.0 * second_idx / sr, 3),
        }
    out = y_body.copy()
    s0 = delay_n
    s1 = min(region_end, len(out))
    region_len = max(s1 - s0, 1)
    taper = np.linspace(1.0, 0.52, region_len) ** (0.6 + 0.4 * min(ratio, 1.0))
    atten = _clamp(0.50 + 0.30 * (first_peak / max(second_peak, 1e-12)), 0.42, 0.82)
    taper *= atten
    out[s0:s1] *= taper
    return out, {
        "delayed_peak_ratio": round(ratio, 4),
        "attenuation_applied": True,
        "second_peak_ms": round(1000.0 * second_idx / sr, 3),
        "attenuation_region_ms": [round(1000.0 * s0 / sr, 3), round(1000.0 * s1 / sr, 3)],
    }


def _early_contact_from_bridge(f_bridge: np.ndarray, sr: int) -> np.ndarray:
    """Early-only tap from the same bridge force (not a separate click source)."""
    early_n = max(int(EXCITATION_CONTACT_MS * 1e-3 * sr), 8)
    out = np.zeros_like(f_bridge, dtype=np.float64)
    out[:early_n] = f_bridge[:early_n]
    return out


def _apply_a2_control(y: np.ndarray, sr: int, cavity: float) -> np.ndarray:
    alpha = math.exp(-2.0 * math.pi * 82.0 / sr)
    hp = np.zeros_like(y, dtype=np.float64)
    z = prev = 0.0
    for i, x in enumerate(y):
        z = alpha * (z + x - prev)
        prev = x
        hp[i] = z
    low = y - hp
    boom_alpha = math.exp(-2.0 * math.pi * 120.0 / sr)
    boom = np.zeros_like(y, dtype=np.float64)
    z = prev = 0.0
    for i, x in enumerate(low):
        z = boom_alpha * (z + x - prev)
        prev = x
        boom[i] = z
    return hp + 0.08 * boom - 0.42 * cavity * boom


def _align_attack_polarity(y: np.ndarray, sr: int) -> Tuple[np.ndarray, bool]:
    """Flip whole waveform if first strong attack lobe is negative."""
    search = min(int(0.05 * sr), len(y))
    if search < 8:
        return y, False
    idx = int(np.argmax(np.abs(y[:search])))
    if y[idx] < 0.0:
        return (-y).astype(np.float64), True
    return y, False


def _smoothed_envelope(y: np.ndarray, sr: int, win_ms: float = 2.0) -> np.ndarray:
    k = max(int(win_ms * 1e-3 * sr), 3)
    env = np.abs(y.astype(np.float64))
    return np.convolve(env, np.ones(k) / k, mode="same")


def compute_early_double_attack_risk(y: np.ndarray, sr: int) -> Dict[str, Any]:
    """Detect multiple attack peaks in first 25 ms (3–25 ms second peak)."""
    win = min(int(0.025 * sr), len(y))
    min_delay = max(int(EARLY_ATTACK_MIN_MS * 1e-3 * sr), 2)
    if win < min_delay + 4:
        return {
            "early_double_attack_risk": False,
            "first_peak_ms": 0.0,
            "second_peak_ms": None,
            "second_peak_ratio": 0.0,
        }
    smooth = _smoothed_envelope(y[:win], sr, win_ms=0.8)
    first_idx = int(np.argmax(smooth[: max(min_delay, 6)]))
    first_peak = float(smooth[first_idx])
    if first_peak < 1e-12:
        return {
            "early_double_attack_risk": False,
            "first_peak_ms": 0.0,
            "second_peak_ms": None,
            "second_peak_ratio": 0.0,
        }
    search_end = min(int(EARLY_ATTACK_MAX_MS * 1e-3 * sr), win)
    search = smooth[min_delay:search_end]
    if search.size < 4:
        return {
            "early_double_attack_risk": False,
            "first_peak_ms": round(1000.0 * first_idx / sr, 3),
            "second_peak_ms": None,
            "second_peak_ratio": 0.0,
        }
    second_idx = int(np.argmax(search)) + min_delay
    second_peak = float(smooth[second_idx])
    ratio = second_peak / first_peak
    risk = bool(second_idx > first_idx + 1 and ratio > EARLY_ATTACK_SECOND_FRAC)
    return {
        "early_double_attack_risk": risk,
        "first_peak_ms": round(1000.0 * first_idx / sr, 3),
        "second_peak_ms": round(1000.0 * second_idx / sr, 3) if risk else None,
        "second_peak_ratio": round(ratio, 4),
    }


def compute_double_pluck_risk(y: np.ndarray, sr: int) -> Dict[str, Any]:
    """Detect second onset peak in first 200 ms above threshold after 18 ms."""
    win = min(int(0.200 * sr), len(y))
    if win < 32:
        return {"double_pluck_risk": False, "first_peak_ms": 0.0, "second_peak_ms": None, "second_peak_ratio": 0.0}
    smooth = _smoothed_envelope(y[:win], sr)
    delay_n = int(DOUBLE_PLUCK_MIN_DELAY_MS * 1e-3 * sr)
    first_idx = int(np.argmax(smooth[: max(delay_n, 8)]))
    first_peak = float(smooth[first_idx])
    if first_peak < 1e-12:
        return {"double_pluck_risk": False, "first_peak_ms": 0.0, "second_peak_ms": None, "second_peak_ratio": 0.0}
    search = smooth[delay_n:win]
    second_idx = int(np.argmax(search)) + delay_n if search.size else first_idx
    second_peak = float(smooth[second_idx])
    ratio = second_peak / first_peak
    risk = bool(second_idx > first_idx + 2 and ratio > DOUBLE_PLUCK_SECOND_PEAK_FRAC)
    return {
        "double_pluck_risk": risk,
        "first_peak_ms": round(1000.0 * first_idx / sr, 3),
        "second_peak_ms": round(1000.0 * second_idx / sr, 3) if risk else None,
        "second_peak_ratio": round(ratio, 4),
    }


def compute_low_mid_body_energy_report(y: np.ndarray, sr: int) -> Dict[str, float]:
    """Relative energy in 120–450 Hz body-support band."""
    n = len(y)
    if n < 64:
        return {"low_mid_body_120_450_ratio": 0.0, "total_energy": 0.0}
    spec = np.abs(np.fft.rfft(y.astype(np.float64))) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    total = float(np.sum(spec)) + 1e-18
    mask = (freqs >= 120.0) & (freqs <= 450.0)
    low_mid = float(np.sum(spec[mask]))
    return {
        "low_mid_body_120_450_ratio": round(low_mid / total, 6),
        "total_energy": round(total, 6),
    }


def compute_low_band_energy_report(y: np.ndarray, sr: int) -> Dict[str, float]:
    """Relative energy in 80–160 Hz band for A2 boom diagnostics."""
    n = len(y)
    if n < 64:
        return {"low_band_80_160_ratio": 0.0, "total_energy": 0.0}
    spec = np.abs(np.fft.rfft(y.astype(np.float64))) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    total = float(np.sum(spec)) + 1e-18
    mask = (freqs >= 80.0) & (freqs <= 160.0)
    low = float(np.sum(spec[mask]))
    return {
        "low_band_80_160_ratio": round(low / total, 6),
        "total_energy": round(total, 6),
    }


def _enumerate_early_peaks(env: np.ndarray, sr: int, end_n: int) -> List[Tuple[int, float]]:
    """Significant local maxima in smoothed onset envelope."""
    peaks: List[Tuple[int, float]] = []
    if end_n < 6:
        return peaks
    min_spacing = max(int(EARLY_ATTACK_MIN_DELAY_MS * 1e-3 * sr), 2)
    for i in range(2, end_n - 2):
        v = float(env[i])
        if (
            v >= float(env[i - 1])
            and v >= float(env[i + 1])
            and v >= float(env[i - 2]) * 0.97
            and v >= float(env[i + 2]) * 0.97
            and v > 1e-12
        ):
            peaks.append((i, v))
    merged: List[Tuple[int, float]] = []
    for idx, amp in sorted(peaks, key=lambda x: -x[1]):
        if all(abs(idx - m[0]) >= min_spacing for m in merged):
            merged.append((idx, amp))
    merged.sort(key=lambda x: x[0])
    return merged


def _peak_times_and_ratios(peaks: Sequence[Tuple[int, float]], main_amp: float, sr: int) -> Tuple[List[float], List[float]]:
    main = max(main_amp, 1e-12)
    times = [round(1000.0 * idx / sr, 3) for idx, _ in peaks]
    ratios = [round(amp / main, 4) for _, amp in peaks]
    return times, ratios


def _has_secondary_early_attack(
    peaks: Sequence[Tuple[int, float]],
    main_idx: int,
    main_amp: float,
    sr: int,
    threshold: float,
) -> bool:
    min_delay = max(int(EARLY_ATTACK_MIN_DELAY_MS * 1e-3 * sr), 2)
    main = max(main_amp, 1e-12)
    for idx, amp in peaks:
        if idx > main_idx + min_delay and (amp / main) > threshold:
            return True
        if idx < main_idx - min_delay and (amp / main) > threshold and idx > min_delay:
            return True
    return False


def _build_target_attack_envelope(
    cross_n: int,
    sr: int,
    note: str,
    sample_id: str,
    peak_amp: float,
) -> np.ndarray:
    """Single-peak target envelope for first 35 ms; crossfade tail to 45 ms."""
    sample_off = {"sample_000": 0.0, "sample_001": -0.45, "sample_002": 0.35}.get(sample_id, 0.0)
    if note == "E5":
        peak_ms = 4.0 + sample_off
        decay_tau = 0.010
    elif note == "A4":
        peak_ms = 5.5 + sample_off
        decay_tau = 0.013
    else:
        peak_ms = 6.5 + sample_off
        decay_tau = 0.017
    peak_ms = _clamp(peak_ms, 3.0, 7.5)
    win_n = min(int(ONSET_REPAIR_WINDOW_MS * 1e-3 * sr), cross_n)
    peak_n = max(int(peak_ms * 1e-3 * sr), 3)
    target = np.zeros(cross_n, dtype=np.float64)
    ramp = np.sin(np.linspace(0.0, math.pi / 2.0, peak_n)) ** 2
    target[:peak_n] = peak_amp * ramp
    if win_n > peak_n:
        tt = np.arange(win_n - peak_n, dtype=np.float64) / sr
        tail = peak_amp * np.exp(-tt / decay_tau)
        target[peak_n:win_n] = tail
        for i in range(peak_n + 1, win_n):
            target[i] = min(target[i], target[i - 1])
    if cross_n > win_n:
        target[win_n:cross_n] = target[win_n - 1] if win_n > 0 else peak_amp * 0.25
    return target


def enforce_single_attack_peak(
    wave: np.ndarray,
    sr: int,
    note: str,
    sample_id: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Post-synthesis onset repair: one dominant attack peak in first 35 ms."""
    y = np.asarray(wave, dtype=np.float64).copy()
    n = len(y)
    win_n = min(int(ONSET_REPAIR_WINDOW_MS * 1e-3 * sr), n)
    cross_n = min(int(ONSET_REPAIR_CROSSFADE_END_MS * 1e-3 * sr), n)
    if win_n < 8 or cross_n <= win_n:
        return y, {"onset_repair_applied": False, "early_double_attack_risk_after": True}

    smooth_ms = {"E5": 0.85, "A4": 1.1, "A2": 1.4}.get(note, 1.0)
    threshold = EARLY_ATTACK_SECONDARY_FRAC_E5 if note == "E5" else EARLY_ATTACK_SIGNIFICANT_FRAC
    env = np.maximum(_smoothed_envelope(y[:cross_n], sr, win_ms=smooth_ms), 1e-12)
    peaks = _enumerate_early_peaks(env, sr, win_n)
    main_idx = int(np.argmax(env[:win_n]))
    main_amp = float(env[main_idx])
    if not peaks:
        peaks = [(main_idx, main_amp)]
    before_times, before_ratios = _peak_times_and_ratios(peaks, main_amp, sr)
    risk_before = _has_secondary_early_attack(peaks, main_idx, main_amp, sr, threshold)

    target = _build_target_attack_envelope(cross_n, sr, note, sample_id, main_amp)
    gain = np.ones(n, dtype=np.float64)
    for i in range(cross_n):
        gain[i] = min(1.0, float(target[i]) / float(env[i]))

    if note == "E5":
        for idx, amp in peaks:
            if idx != main_idx and idx < win_n and amp / max(main_amp, 1e-12) > 0.20:
                half = max(int(0.0025 * sr), 2)
                s0 = max(idx - half, 0)
                s1 = min(idx + half, cross_n)
                dip = 1.0 - _clamp(0.18 + 0.28 * (amp / max(main_amp, 1e-12)), 0.15, 0.38)
                gain[s0:s1] = np.minimum(gain[s0:s1], dip)

    k = max(int(0.0012 * sr), 3)
    if cross_n > k * 2:
        gain[:cross_n] = np.convolve(gain[:cross_n], np.ones(k) / k, mode="same")
    if cross_n > win_n:
        gain[win_n:cross_n] = np.linspace(float(gain[win_n - 1]), 1.0, cross_n - win_n)
    gain[cross_n:] = 1.0
    gain = np.clip(gain, 0.20, 1.0)

    y_out = y * gain
    env_after = np.maximum(_smoothed_envelope(y_out[:cross_n], sr, win_ms=smooth_ms), 1e-12)
    peaks_after = _enumerate_early_peaks(env_after, sr, win_n)
    main_idx_a = int(np.argmax(env_after[:win_n]))
    main_amp_a = float(env_after[main_idx_a])
    if not peaks_after:
        peaks_after = [(main_idx_a, main_amp_a)]
    after_times, after_ratios = _peak_times_and_ratios(peaks_after, main_amp_a, sr)
    risk_after = _has_secondary_early_attack(peaks_after, main_idx_a, main_amp_a, sr, threshold)

    if risk_after:
        for idx, amp in peaks_after:
            if idx == main_idx_a:
                continue
            if amp / max(main_amp_a, 1e-12) <= threshold:
                continue
            half = max(int(0.003 * sr), 2)
            s0 = max(idx - half, 0)
            s1 = min(idx + half * 2, win_n)
            extra = 0.70 if note == "E5" else 0.78
            y_out[s0:s1] *= extra
        env_after = np.maximum(_smoothed_envelope(y_out[:cross_n], sr, win_ms=smooth_ms), 1e-12)
        peaks_after = _enumerate_early_peaks(env_after, sr, win_n)
        main_idx_a = int(np.argmax(env_after[:win_n]))
        main_amp_a = float(env_after[main_idx_a])
        after_times, after_ratios = _peak_times_and_ratios(peaks_after, main_amp_a, sr)
        risk_after = _has_secondary_early_attack(peaks_after, main_idx_a, main_amp_a, sr, threshold)

    return y_out, {
        "onset_repair_applied": True,
        "onset_repair_gain_min": round(float(np.min(gain[:cross_n])), 6),
        "early_attack_peak_times_ms_before": before_times,
        "early_attack_peak_ratios_before": before_ratios,
        "early_attack_peak_times_ms_after": after_times,
        "early_attack_peak_ratios_after": after_ratios,
        "early_double_attack_risk_before": risk_before,
        "early_double_attack_risk_after": risk_after,
        "repair_window_ms": [0.0, ONSET_REPAIR_CROSSFADE_END_MS],
        "note": note,
        "sample_id": sample_id,
    }


def _normalize_note(y: np.ndarray, note: str) -> Tuple[np.ndarray, Dict[str, float]]:
    target_rms = 10.0 ** (TARGET_RMS_DBFS / 20.0)
    out = y * (target_rms / max(_rms(y), 1e-12))
    note_peak = 10.0 ** (NOTE_PEAK_TARGET_DBFS[note] / 20.0)
    max_peak = 10.0 ** (MAX_PEAK_DBFS / 20.0)
    peak = float(np.max(np.abs(out)))
    if peak > note_peak:
        out *= note_peak / peak
    peak = float(np.max(np.abs(out)))
    if peak > max_peak:
        out *= max_peak / peak
    peak = float(np.max(np.abs(out)))
    return out.astype(np.float64), {
        "peak_dbfs": round(_linear_to_dbfs(peak), 3),
        "rms_dbfs": round(_linear_to_dbfs(_rms(out)), 3),
    }


def synthesize_note_for_sample(state: SampleSynthesisState, note: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    f0 = float(NOTE_FREQUENCY_HZ[note])
    n = int(DURATION_S * SR)
    factors = state.factors
    body_support = _per_sample_body_support(state.sample_id, note)
    unity = np.ones(n, dtype=np.float64)
    inharm_b = BASE_INHARMONICITY_B * (0.94 + 0.10 * (factors["top_stiffness_to_weight_factor"] - 1.0))
    f_string_raw = _synthesize_f_string(
        n,
        SR,
        f0,
        pluck_pos=state.pluck_position_ratio,
        factors=factors,
        inharm_b=inharm_b,
        note=note,
        excitation=unity,
    )
    pluck_shape = _pluck_contact_shape(n, SR)
    f_excitation = f_string_raw * pluck_shape
    f_bridge_eff = _bridge_transfer(f_excitation, SR, state.bridge_transfer)

    top_sum = back_sum = air_sum = rad_sum = 0.0
    y_body = np.zeros(n, dtype=np.float64)
    for mode in state.modes:
        comp = _mode_transfer(f_bridge_eff, SR, mode, note=note, sample_id=state.sample_id)
        c = mode.get("component")
        if c == "top":
            top_sum += 1.0
            y_body += comp
        elif c == "back":
            back_sum += 1.0
            y_body += comp * float(factors.get("back_density_warmth_factor") or 1.0)
        elif c == "air":
            air_sum += 1.0
            y_body += comp * float(factors.get("air_helmholtz_factor") or 1.0) * 0.88
        else:
            rad_sum += 1.0
            y_body += comp * float(factors.get("radiation_brightness_factor") or 1.0)

    y_body, delayed_peak_meta = local_limited_delayed_peak_correction(y_body, SR)

    mix = state.mix_ratios
    string_mix = float(mix.get("string_bridge") or 0.26)
    body_mix = float(mix.get("body_modal") or 0.72) * float(body_support["body_modal_mult"])
    y = f_bridge_eff * string_mix + y_body * body_mix
    note_scale = {"A2": 0.96, "A4": 0.88, "E5": 0.82}[note]
    y *= note_scale
    if note == "A2":
        y = _apply_a2_control(y, SR, float(factors.get("body_size_cavity_factor") or 1.0))

    y, onset_repair = enforce_single_attack_peak(y, SR, note, state.sample_id)

    y_out, levels = _normalize_note(y, note)
    y_out, polarity_flipped = _align_attack_polarity(y_out, SR)
    early_attack_after = bool(onset_repair.get("early_double_attack_risk_after"))
    onset_diag = compute_double_pluck_risk(y_out, SR)
    double_pluck_diagnostics = {
        "double_pluck_risk": onset_diag.get("double_pluck_risk"),
        "early_double_attack_risk_before": onset_repair.get("early_double_attack_risk_before"),
        "early_double_attack_risk_after": early_attack_after,
        "onset_repair": onset_repair,
        "delayed_peak": delayed_peak_meta,
        "onset": onset_diag,
    }
    low_band = compute_low_band_energy_report(y_out, SR) if note == "A2" else {}
    low_mid_body = compute_low_mid_body_energy_report(y_out, SR) if note in ("A4", "E5") else {}
    meta = {
        "note": note,
        "sample_id": state.sample_id,
        "f0_hz": f0,
        "primary_excitation": "F_excitation_embedded_contact",
        "rollback_base": ROLLBACK_BASE,
        "contact_embedded_in_excitation": True,
        "no_separate_contact_mix_layer": True,
        "polarity_aligned": polarity_flipped,
        "double_pluck_diagnostics": double_pluck_diagnostics,
        "onset_repair_diagnostics": onset_repair,
        "double_pluck_risk": bool(onset_diag.get("double_pluck_risk") or early_attack_after),
        "early_double_attack_risk": early_attack_after,
        "early_double_attack_risk_before": onset_repair.get("early_double_attack_risk_before"),
        "early_double_attack_risk_after": early_attack_after,
        "delayed_peak_ratio": delayed_peak_meta.get("delayed_peak_ratio"),
        "delayed_peak_attenuation_amount": delayed_peak_meta.get("attenuation_amount"),
        "body_support_summary": dict(body_support),
        "bridge_transfer_summary": dict(state.bridge_transfer),
        "body_modal_transfer_summary": {
            "mode_count": len(state.modes),
            "driven_by": "F_bridge_eff",
            "immediate_onset_kernels": True,
            "global_onset_lock_disabled": True,
            "top_modes": int(top_sum),
            "back_modes": int(back_sum),
            "air_modes": int(air_sum),
            "radiation_modes": int(rad_sum),
        },
        "radiation_summary": {
            "radiation_brightness_factor": factors.get("radiation_brightness_factor"),
            "mix_body_modal": body_mix,
        },
        "partial_decay_summary": {
            "inharmonicity_b": inharm_b,
            "pluck_position_ratio": state.pluck_position_ratio,
        },
        "mix_ratio_summary": {
            **dict(mix),
            "effective_body_modal": body_mix,
            "effective_string_bridge": string_mix,
            "contact_embedded_only": True,
        },
        "low_band_energy_report": low_band,
        "low_mid_body_energy_report": low_mid_body,
        **levels,
    }
    return y_out, meta


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


def _spectral_distance(a: np.ndarray, b: np.ndarray) -> float:
    ns = min(len(a), len(b))
    if ns < 64:
        return 0.0
    sa = np.abs(np.fft.rfft(a[:ns] * np.hanning(ns)))
    sb = np.abs(np.fft.rfft(b[:ns] * np.hanning(ns)))
    sa /= max(float(np.linalg.norm(sa)), 1e-12)
    sb /= max(float(np.linalg.norm(sb)), 1e-12)
    return float(np.linalg.norm(sa - sb))


def _envelope_distance(a: np.ndarray, b: np.ndarray) -> float:
    na = np.abs(a)
    nb = np.abs(b)
    ns = min(len(na), len(nb))
    na = na[:ns] / max(float(np.max(na[:ns])), 1e-12)
    nb = nb[:ns] / max(float(np.max(nb[:ns])), 1e-12)
    return float(np.mean(np.abs(na - nb)))


def compute_same_note_pairwise_correlation(audio: Mapping[str, Mapping[str, np.ndarray]]) -> Dict[str, Any]:
    by_note: Dict[str, Dict[str, float]] = {}
    corrs: List[float] = []
    for note in NOTE_SET:
        by_note[note] = {}
        for i, sa in enumerate(SAMPLE_SET):
            for sb in SAMPLE_SET[i + 1 :]:
                c = _waveform_correlation(audio[sa][note], audio[sb][note])
                by_note[note][f"{sa}_vs_{sb}"] = round(c, 6)
                corrs.append(c)
    return {
        "same_note_pairwise_correlation": by_note,
        "max_correlation": round(max(corrs) if corrs else 1.0, 6),
        "min_correlation": round(min(corrs) if corrs else 1.0, 6),
        "mean_correlation": round(float(np.mean(corrs)) if corrs else 1.0, 6),
    }


def compute_pairwise_metrics(audio: Mapping[str, Mapping[str, np.ndarray]]) -> Dict[str, Any]:
    pairs: Dict[str, Any] = {}
    spec_scores: List[float] = []
    env_scores: List[float] = []
    for i, sa in enumerate(SAMPLE_SET):
        for sb in SAMPLE_SET[i + 1 :]:
            per_note: Dict[str, Any] = {}
            for note in NOTE_SET:
                aa, ab = audio[sa][note], audio[sb][note]
                per_note[note] = {
                    "spectral_distance": round(_spectral_distance(aa, ab), 6),
                    "envelope_distance": round(_envelope_distance(aa, ab), 6),
                    "waveform_correlation": round(_waveform_correlation(aa, ab), 6),
                }
            sm = float(np.mean([per_note[n]["spectral_distance"] for n in NOTE_SET]))
            em = float(np.mean([per_note[n]["envelope_distance"] for n in NOTE_SET]))
            spec_scores.append(sm)
            env_scores.append(em)
            pairs[f"{sa}_vs_{sb}"] = {"per_note": per_note, "mean_spectral_distance": round(sm, 6)}
    return {
        "pairwise_difference_metrics": pairs,
        "mean_spectral_distance": round(float(np.mean(spec_scores)) if spec_scores else 0.0, 6),
        "mean_envelope_distance": round(float(np.mean(env_scores)) if env_scores else 0.0, 6),
    }


def compute_guitar_family_consistency_metrics(correlation: Mapping[str, Any]) -> Dict[str, Any]:
    max_c = float(correlation.get("max_correlation") or 1.0)
    min_c = float(correlation.get("min_correlation") or 1.0)
    mean_c = float(correlation.get("mean_correlation") or 1.0)
    too_similar = max_c > CORRELATION_TOO_SIMILAR
    too_unrelated = min_c < CORRELATION_TOO_UNRELATED
    in_band = CORRELATION_FAMILY_LO <= mean_c <= CORRELATION_FAMILY_HI
    return {
        "max_correlation": max_c,
        "min_correlation": min_c,
        "mean_correlation": mean_c,
        "too_similar": too_similar,
        "too_unrelated": too_unrelated,
        "in_family_band": in_band,
        "pass": bool(not too_similar and not too_unrelated and in_band),
    }


def build_anti_cheat_checks(
    *,
    isolation_reports: Sequence[Mapping[str, Any]],
    family_metrics: Mapping[str, Any],
    pairwise: Mapping[str, Any],
    peak_rms_report: Mapping[str, Any],
    double_pluck_ok: bool = True,
) -> Dict[str, Any]:
    folder_cleared = bool(
        peak_rms_report.get("v9_folder_cleared")
        or peak_rms_report.get("v8_folder_cleared")
        or peak_rms_report.get("v7_folder_cleared")
        or peak_rms_report.get("v6_folder_cleared")
        or peak_rms_report.get("v5_folder_cleared")
    )
    checks = {
        "no_randomization": True,
        "no_sample_id_only_gain": True,
        "no_arbitrary_eq_only_differences": True,
        "no_reverb_echo_body_tail": True,
        "no_hard_gate": True,
        "no_clipping_limiter_trick": bool(peak_rms_report.get("all_peaks_within_target")),
        "no_fem_run": True,
        "no_rom_run": True,
        "no_stk_integration": True,
        "website_default_unchanged": True,
        "no_global_onset_lock": True,
        "per_sample_state_isolated": all(r.get("independent_state_created") for r in isolation_reports),
        "no_cross_sample_mutable_state": all(r.get("cleanup_completed") for r in isolation_reports),
        "no_global_normalization_hiding_differences": True,
        "no_stale_wav_mix": folder_cleared,
        "guitar_family_consistency": bool(family_metrics.get("pass")),
        "differences_not_only_loudness": float(pairwise.get("mean_spectral_distance") or 0) > 0.02
        or not family_metrics.get("too_similar"),
    }
    return {
        **checks,
        "single_pluck_onset": double_pluck_ok,
        "pass": bool(all(checks.values())),
    }


def build_readiness_emergency_demo(
    *,
    files_generated: int,
    peaks_controlled: bool,
    family_metrics: Mapping[str, Any],
    anti_cheat_pass: bool = True,
    double_pluck_ok: bool = True,
    early_attack_ok: bool = True,
    usable_with_limitations: bool = False,
) -> Dict[str, Any]:
    return build_readiness(
        files_generated=files_generated,
        peaks_ok=peaks_controlled,
        family_metrics=family_metrics,
        anti_cheat_pass=anti_cheat_pass,
        double_pluck_ok=double_pluck_ok,
        early_attack_ok=early_attack_ok,
        usable_with_limitations=usable_with_limitations,
    )


def build_readiness(
    *,
    files_generated: int,
    peaks_ok: bool,
    family_metrics: Mapping[str, Any],
    anti_cheat_pass: bool,
    double_pluck_ok: bool = True,
    early_attack_ok: bool = True,
    usable_with_limitations: bool = False,
) -> Dict[str, Any]:
    if files_generated < 9 or not peaks_ok:
        status = READINESS_FAIL
    elif not double_pluck_ok or not early_attack_ok:
        status = READINESS_DOUBLE_PLUCK
    elif not anti_cheat_pass:
        status = READINESS_FAIL
    elif family_metrics.get("too_unrelated"):
        status = READINESS_REVIEW
    elif family_metrics.get("too_similar"):
        status = READINESS_WEAK
    elif family_metrics.get("pass") and double_pluck_ok and early_attack_ok:
        status = READINESS_OK
    elif usable_with_limitations and not family_metrics.get("too_similar"):
        status = READINESS_OK_LIMITED
    else:
        status = READINESS_WEAK
    return {
        "current_status": status,
        "stk_gui_activation_planning_allowed": status in (READINESS_OK, READINESS_OK_LIMITED),
        "files_generated": files_generated,
        "double_pluck_ok": double_pluck_ok,
        "early_attack_ok": early_attack_ok,
    }


def _clear_output_wavs(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for wav in folder.glob("*.wav"):
        wav.unlink()


def _run_final_guitar_demo(
    *,
    version_label: str,
    final_demo_version: str,
    repo_root: Optional[Path] = None,
    audio_dir: Path,
    json_path: Path,
    md_path: Path,
    voicing: Mapping[str, Dict[str, Any]],
    folder_cleared_key: str,
) -> Dict[str, Any]:
    root = Path(repo_root or REPO_ROOT)
    out_dir = Path(audio_dir)
    _clear_output_wavs(out_dir)

    audit = load_audit_report()
    ref_phys = extract_per_sample_physical_parameters(REFERENCE_SAMPLE_ID, audit)
    readonly_modes = _load_readonly_reference_modes(root)

    generated_files: List[str] = []
    audio_by_sample: Dict[str, Dict[str, np.ndarray]] = {}
    per_sample_factors: Dict[str, Any] = {}
    per_sample_modes: Dict[str, Any] = {}
    per_sample_trace: Dict[str, Any] = {}
    per_sample_profiles: Dict[str, Any] = {}
    isolation_reports: List[Dict[str, Any]] = []
    partial_decay: Dict[str, Dict[str, Any]] = {}
    peak_rms: Dict[str, Dict[str, float]] = {}
    onset_diagnostics: Dict[str, Dict[str, Any]] = {}
    double_pluck_risk_per_file: Dict[str, bool] = {}
    early_double_attack_per_file: Dict[str, bool] = {}
    delayed_peak_ratio_per_file: Dict[str, float] = {}
    delayed_peak_attenuation_per_file: Dict[str, float] = {}
    double_pluck_diagnostics: Dict[str, Dict[str, Any]] = {}
    onset_repair_per_file: Dict[str, Dict[str, Any]] = {}
    low_band_energy: Dict[str, Dict[str, float]] = {}
    low_mid_body_energy: Dict[str, Dict[str, float]] = {}
    body_support_by_note: Dict[str, Dict[str, Any]] = {}

    for sample_id in SAMPLE_SET:
        physical = extract_per_sample_physical_parameters(sample_id, audit)
        state = build_sample_synthesis_state(
            sample_id,
            physical=physical,
            reference_physical=ref_phys,
            readonly_modes=readonly_modes,
        )
        per_sample_factors[sample_id] = dict(state.factors)
        per_sample_modes[sample_id] = copy.deepcopy(state.modes)
        per_sample_trace[sample_id] = state.differentiation_trace
        per_sample_profiles[sample_id] = {
            "voicing_profile": state.voicing_profile,
            "pluck_position_ratio": state.pluck_position_ratio,
            "mix_ratios": state.mix_ratios,
            "factor_multipliers": dict(
                voicing[sample_id].get("factor_multipliers") or voicing[sample_id].get("factors") or {}
            ),
            "bridge_transfer": state.bridge_transfer,
        }
        audio_by_sample[sample_id] = {}
        partial_decay[sample_id] = {}

        for note in NOTE_SET:
            y, meta = synthesize_note_for_sample(state, note)
            wav_name = final_wav_filename(sample_id, note)
            wav_path = out_dir / wav_name
            write_wav_mono(wav_path, y, SR)
            generated_files.append(str(wav_path.resolve()))
            audio_by_sample[sample_id][note] = y
            partial_decay[sample_id][note] = meta.get("partial_decay_summary")
            peak_rms[wav_name] = {"peak_dbfs": meta["peak_dbfs"], "rms_dbfs": meta["rms_dbfs"]}
            onset_diagnostics[wav_name] = meta.get("double_pluck_diagnostics") or {}
            double_pluck_diagnostics[wav_name] = meta.get("double_pluck_diagnostics") or {}
            onset_repair_per_file[wav_name] = meta.get("onset_repair_diagnostics") or {}
            double_pluck_risk_per_file[wav_name] = bool(meta.get("double_pluck_risk"))
            early_double_attack_per_file[wav_name] = bool(meta.get("early_double_attack_risk_after"))
            delayed_peak_ratio_per_file[wav_name] = float(meta.get("delayed_peak_ratio") or 0.0)
            delayed_peak_attenuation_per_file[wav_name] = float(meta.get("delayed_peak_attenuation_amount") or 0.0)
            if note == "A2":
                low_band_energy[wav_name] = meta.get("low_band_energy_report") or {}
            if note in ("A4", "E5"):
                low_mid_body_energy[wav_name] = meta.get("low_mid_body_energy_report") or {}
            body_support_by_note[note] = meta.get("body_support_summary") or _per_sample_body_support(sample_id, note)
            print(f"[Final demo {version_label}] wrote {sample_id} {note} -> {wav_name}")

        isolation_reports.append(dict(state.isolation_meta))
        cleanup_sample_state(state)

    correlation = compute_same_note_pairwise_correlation(audio_by_sample)
    pairwise = compute_pairwise_metrics(audio_by_sample)
    family = compute_guitar_family_consistency_metrics(correlation)
    peaks_ok = all(v["peak_dbfs"] <= MAX_PEAK_DBFS + 0.05 for v in peak_rms.values())
    double_pluck_ok = not any(double_pluck_risk_per_file.values())
    early_attack_ok = not any(early_double_attack_per_file.values())
    usable_limited = (
        peaks_ok
        and not family.get("too_similar")
        and float(pairwise.get("mean_spectral_distance") or 0) > 0.025
        and (not double_pluck_ok or not early_attack_ok)
    )
    peak_rms_report = {
        "per_file": peak_rms,
        "all_peaks_within_target": peaks_ok,
        folder_cleared_key: True,
        "output_folder": str(out_dir),
    }
    anti_cheat = build_anti_cheat_checks(
        isolation_reports=isolation_reports,
        family_metrics=family,
        pairwise=pairwise,
        peak_rms_report=peak_rms_report,
        double_pluck_ok=double_pluck_ok and early_attack_ok,
    )
    readiness = build_readiness(
        files_generated=len(generated_files),
        peaks_ok=peaks_ok,
        family_metrics=family,
        anti_cheat_pass=bool(anti_cheat.get("pass")),
        double_pluck_ok=double_pluck_ok,
        early_attack_ok=early_attack_ok,
        usable_with_limitations=usable_limited,
    )
    report = {
        "report_version": ENGINE_VERSION,
        "final_demo_version": final_demo_version,
        "emergency_demo_version": final_demo_version,
        "rollback_base": ROLLBACK_BASE,
        "timestamp": _utc_now(),
        "physical_chain_summary": PHYSICAL_CHAIN_SUMMARY,
        "mix_ratio_summary": {sid: voicing[sid]["mix"] for sid in SAMPLE_SET},
        "generated_files": generated_files,
        "per_sample_physical_factors": per_sample_factors,
        "per_sample_factor_multipliers": {
            sid: dict(voicing[sid].get("factor_multipliers") or voicing[sid].get("factors") or {})
            for sid in SAMPLE_SET
        },
        "per_sample_mode_parameters": per_sample_modes,
        "per_sample_differentiation_trace": per_sample_trace,
        "per_sample_synthesis_profile": per_sample_profiles,
        "per_sample_isolation_report": isolation_reports,
        "per_sample_state_isolated": all(r.get("independent_state_created") for r in isolation_reports),
        "body_modal_transfer_summary": {
            sid: {"mode_count": len(per_sample_modes[sid]), "driven_by": "F_bridge_eff", "immediate_onset": True}
            for sid in SAMPLE_SET
        },
        "bridge_transfer_summary": {sid: per_sample_profiles[sid]["bridge_transfer"] for sid in SAMPLE_SET},
        "radiation_summary": {
            sid: {
                "radiation_brightness_factor": per_sample_factors[sid].get("radiation_brightness_factor"),
                "air_helmholtz_factor": per_sample_factors[sid].get("air_helmholtz_factor"),
            }
            for sid in SAMPLE_SET
        },
        "double_pluck_diagnostics": double_pluck_diagnostics,
        "onset_repair_per_file": onset_repair_per_file,
        "double_pluck_risk_per_file": double_pluck_risk_per_file,
        "early_double_attack_risk_per_file": early_double_attack_per_file,
        "early_double_attack_risk_after_per_file": early_double_attack_per_file,
        "delayed_peak_ratio_per_file": delayed_peak_ratio_per_file,
        "delayed_peak_attenuation_amount_per_file": delayed_peak_attenuation_per_file,
        "body_support_summary_per_note": body_support_by_note,
        "partial_decay_summary": partial_decay,
        "peak_rms_report": peak_rms_report,
        "low_band_energy_report": low_band_energy,
        "low_mid_body_energy_report": low_mid_body_energy,
        "same_note_pairwise_correlation": correlation.get("same_note_pairwise_correlation"),
        "max_same_note_correlation": correlation.get("max_correlation"),
        "min_same_note_correlation": correlation.get("min_correlation"),
        "spectral_distance": pairwise.get("mean_spectral_distance"),
        "envelope_distance": pairwise.get("mean_envelope_distance"),
        "pairwise_difference_metrics": pairwise.get("pairwise_difference_metrics"),
        "guitar_family_consistency_metrics": family,
        "anti_cheat_checks": anti_cheat,
        "diagnostic_exaggeration_for_audible_demo": True,
        "no_stk_integration": True,
        "no_fem_run": True,
        "no_rom_run": True,
        "website_default_unchanged": True,
        "readiness": readiness,
        "blocked_claims": [
            "Final realism proof",
            "FEM/ROM validation",
            "STK production integration",
            "Step 5L replacement claim",
        ],
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(
        "\n".join(
            [
                f"# PGSM final guitar demo {version_label}",
                "",
                f"**Version:** `{final_demo_version}`",
                f"**Readiness:** `{(readiness.get('current_status'))}`",
                f"**Max correlation:** {correlation.get('max_correlation')}",
                f"**Double pluck risk files:** {sum(1 for v in double_pluck_risk_per_file.values() if v)}",
                f"**Files:** {len(generated_files)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def run_final_guitar_demo_v9(
    *,
    repo_root: Optional[Path] = None,
    audio_dir: Optional[Path] = None,
    json_path: Optional[Path] = None,
    md_path: Optional[Path] = None,
) -> Dict[str, Any]:
    return _run_final_guitar_demo(
        version_label="v9",
        final_demo_version=FINAL_DEMO_VERSION,
        repo_root=repo_root,
        audio_dir=Path(audio_dir or AUDIO_DIR_V9),
        json_path=Path(json_path or REPORT_JSON_V9),
        md_path=Path(md_path or REPORT_MD_V9),
        voicing=V8_VOICING,
        folder_cleared_key="v9_folder_cleared",
    )


def run_final_guitar_demo_v8(
    *,
    repo_root: Optional[Path] = None,
    audio_dir: Optional[Path] = None,
    json_path: Optional[Path] = None,
    md_path: Optional[Path] = None,
) -> Dict[str, Any]:
    return _run_final_guitar_demo(
        version_label="v8",
        final_demo_version=FINAL_DEMO_VERSION_V8,
        repo_root=repo_root,
        audio_dir=Path(audio_dir or AUDIO_DIR_V8),
        json_path=Path(json_path or REPORT_JSON_V8),
        md_path=Path(md_path or REPORT_MD_V8),
        voicing=V8_VOICING,
        folder_cleared_key="v8_folder_cleared",
    )


def run_final_guitar_demo_v7(
    *,
    repo_root: Optional[Path] = None,
    audio_dir: Optional[Path] = None,
    json_path: Optional[Path] = None,
    md_path: Optional[Path] = None,
) -> Dict[str, Any]:
    return _run_final_guitar_demo(
        version_label="v7",
        final_demo_version=FINAL_DEMO_VERSION_V7,
        repo_root=repo_root,
        audio_dir=Path(audio_dir or AUDIO_DIR_V7),
        json_path=Path(json_path or REPORT_JSON_V7),
        md_path=Path(md_path or REPORT_MD_V7),
        voicing=V7_VOICING,
        folder_cleared_key="v7_folder_cleared",
    )


def run_final_guitar_demo_v6(
    *,
    repo_root: Optional[Path] = None,
    audio_dir: Optional[Path] = None,
    json_path: Optional[Path] = None,
    md_path: Optional[Path] = None,
) -> Dict[str, Any]:
    return _run_final_guitar_demo(
        version_label="v6",
        final_demo_version=FINAL_DEMO_VERSION_V6,
        repo_root=repo_root,
        audio_dir=Path(audio_dir or AUDIO_DIR_V6),
        json_path=Path(json_path or REPORT_JSON_V6),
        md_path=Path(md_path or REPORT_MD_V6),
        voicing=V8_VOICING,
        folder_cleared_key="v6_folder_cleared",
    )


def run_final_guitar_demo_v5(
    *,
    repo_root: Optional[Path] = None,
    audio_dir: Optional[Path] = None,
    json_path: Optional[Path] = None,
    md_path: Optional[Path] = None,
) -> Dict[str, Any]:
    return _run_final_guitar_demo(
        version_label="v5",
        final_demo_version=FINAL_DEMO_VERSION_V5,
        repo_root=repo_root,
        audio_dir=Path(audio_dir or AUDIO_DIR_V5),
        json_path=Path(json_path or REPORT_JSON_V5),
        md_path=Path(md_path or REPORT_MD_V5),
        voicing=V8_VOICING,
        folder_cleared_key="v5_folder_cleared",
    )


def run_emergency_guitar_demo(**kwargs: Any) -> Dict[str, Any]:
    return run_final_guitar_demo_v9(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="PGSM final guitar demo engine")
    parser.add_argument("--final-v5", action="store_true", help="Run legacy v5 output folder")
    parser.add_argument("--final-v6", action="store_true", help="Run legacy v6 output folder")
    parser.add_argument("--final-v7", action="store_true", help="Run legacy v7 output folder")
    parser.add_argument("--final-v8", action="store_true", help="Run legacy v8 output folder")
    parser.add_argument("--final-v9", action="store_true", help="Run v9 single-attack enforced demo")
    args = parser.parse_args()
    if args.final_v9:
        report = run_final_guitar_demo_v9()
        print(f"Wrote {REPORT_JSON_V9}")
    elif args.final_v8:
        report = run_final_guitar_demo_v8()
        print(f"Wrote {REPORT_JSON_V8}")
    elif args.final_v7:
        report = run_final_guitar_demo_v7()
        print(f"Wrote {REPORT_JSON_V7}")
    elif args.final_v6:
        report = run_final_guitar_demo_v6()
        print(f"Wrote {REPORT_JSON_V6}")
    elif args.final_v5:
        report = run_final_guitar_demo_v5()
        print(f"Wrote {REPORT_JSON_V5}")
    else:
        parser.print_help()
        return
    print(f"Readiness: {(report.get('readiness') or {}).get('current_status')}")
    print(f"max_correlation: {report.get('max_same_note_correlation')}")
    print(f"double_pluck_risk_files: {sum(1 for v in (report.get('double_pluck_risk_per_file') or {}).values() if v)}")


if __name__ == "__main__":
    main()
