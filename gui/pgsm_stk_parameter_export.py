#!/usr/bin/env python3
"""
PGSM STK parameter export — Python orchestration only.

Exports physical synthesis parameters for the C++/STK renderer.
No audio generation, no FEM/ROM heavy calls, no WAV synthesis.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from pgsm_emergency_guitar_demo_engine import (
    NOTE_PEAK_TARGET_DBFS,
    PHYSICAL_FACTOR_KEYS,
    TARGET_RMS_DBFS,
    V11_VOICING,
    _bridge_transfer_summary,
    _load_readonly_reference_modes,
    _pick_reference_modes,
    compute_v5_physical_factors,
)
from pgsm_step3a_numerical_ir_testbench import FIXED_PLUCK_POSITION, NUMERIC_SR
from pgsm_step5a_limited_note_set_diagnostic_audio import NOTE_FREQUENCY_HZ
from pgsm_step5l_limited_multiguitar_differentiation import (
    REFERENCE_SAMPLE_ID,
    extract_per_sample_physical_parameters,
)
from stk_v6_2_audit_features import load_audit_report

EXPORT_VERSION = "pgsm_stk_parameter_export_v1"
EXPORT_VERSION_V2 = "pgsm_stk_parameter_export_v2"
EXPORT_VERSION_V3 = "pgsm_stk_parameter_export_v3"
RENDERER_TARGET = "stk_cpp"
PYTHON_ROLE = "parameter_export_only"
DURATION_S = 2.5
SAMPLE_SET: Tuple[str, ...] = ("sample_000", "sample_001", "sample_002")
NOTE_SET: Tuple[str, ...] = ("A2", "A4", "E5")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_JSON = REPO_ROOT / "audio" / "debug_reports" / "pgsm_stk_demo_parameters.json"
DEFAULT_OUTPUT_JSON_V2 = REPO_ROOT / "audio" / "debug_reports" / "pgsm_stk_demo_parameters_v2.json"

DEMO_VERSIONS: Dict[str, Dict[str, str]] = {
    "v1": {
        "demo_id": "pgsm_stk_guitar_demo",
        "export_version": EXPORT_VERSION,
        "audio_subdir": "audio/pgsm_stk_guitar_demo",
        "params_json": "audio/debug_reports/pgsm_stk_demo_parameters.json",
        "report_json": "audio/debug_reports/pgsm_stk_guitar_demo_report.json",
        "report_md": "audio/debug_reports/pgsm_stk_guitar_demo_report.md",
    },
    "v2": {
        "demo_id": "pgsm_stk_guitar_demo_v2",
        "export_version": EXPORT_VERSION_V2,
        "audio_subdir": "audio/pgsm_stk_guitar_demo_v2",
        "params_json": "audio/debug_reports/pgsm_stk_demo_parameters_v2.json",
        "report_json": "audio/debug_reports/pgsm_stk_guitar_demo_v2_report.json",
        "report_md": "audio/debug_reports/pgsm_stk_guitar_demo_v2_report.md",
    },
    "v3": {
        "demo_id": "pgsm_stk_guitar_demo_v3",
        "export_version": EXPORT_VERSION_V3,
        "audio_subdir": "audio/pgsm_stk_guitar_demo_v3",
        "params_json": "audio/debug_reports/pgsm_stk_demo_parameters_v3.json",
        "report_json": "audio/debug_reports/pgsm_stk_guitar_demo_v3_report.json",
        "report_md": "audio/debug_reports/pgsm_stk_guitar_demo_v3_report.md",
    },
}

REQUIRED_RENDER_GROUPS: Tuple[str, ...] = (
    "string_model",
    "bridge_model",
    "body_model",
    "material_model",
    "radiation_model",
    "output_model",
)

AUDIT_SCALAR_KEYS: Tuple[str, ...] = (
    "body_size_cavity_factor",
    "body_depth_m",
    "body_volume_proxy",
    "soundhole_area_proxy",
    "soundhole_radiation_factor",
    "bridge_mobility_factor",
    "effective_mass_loading_factor",
    "top_stiffness_to_weight_factor",
    "top_damping_factor",
    "material_loss_factor",
    "back_density_warmth_factor",
    "air_helmholtz_factor",
    "radiation_brightness_factor",
    "top_weight",
    "back_weight",
    "air_weight",
    "string_body_mix",
)

MEANINGFUL_SPREAD_THRESHOLD = 0.035

SAMPLE_PROFILES: Dict[str, str] = {
    "sample_000": "balanced_neutral",
    "sample_001": "bright_light_fast",
    "sample_002": "warm_deep_heavy",
}

# Applied only when source/LHS spread is too small for audible demo (disclosed in report).
DIAGNOSTIC_MULTIPLIERS: Dict[str, Dict[str, float]] = {
    "sample_000": {},
    "sample_001": {
        "bridge_mobility_factor": 1.20,
        "effective_mass_loading_factor": 0.80,
        "radiation_brightness_factor": 1.20,
        "top_stiffness_to_weight_factor": 1.15,
        "air_helmholtz_factor": 0.85,
        "top_damping_factor": 1.12,
        "body_size_cavity_factor": 0.90,
    },
    "sample_002": {
        "effective_mass_loading_factor": 1.24,
        "body_size_cavity_factor": 1.20,
        "soundhole_radiation_factor": 1.20,
        "back_density_warmth_factor": 1.28,
        "radiation_brightness_factor": 0.85,
        "air_helmholtz_factor": 1.15,
        "top_damping_factor": 0.90,
    },
}

# v3 perceptual calibration — always applied on v3 export (disclosed in report).
V3_FACTOR_MULTIPLIERS: Dict[str, Dict[str, float]] = {
    "sample_000": {},
    "sample_001": {
        "bridge_mobility_factor": 1.28,
        "effective_mass_loading_factor": 0.75,
        "radiation_brightness_factor": 1.28,
        "top_stiffness_to_weight_factor": 1.22,
        "air_helmholtz_factor": 0.80,
        "top_damping_factor": 1.28,
        "body_size_cavity_factor": 0.82,
        "soundhole_radiation_factor": 0.85,
        "back_density_warmth_factor": 0.92,
    },
    "sample_002": {
        "effective_mass_loading_factor": 1.28,
        "body_size_cavity_factor": 1.22,
        "soundhole_radiation_factor": 1.28,
        "back_density_warmth_factor": 1.35,
        "radiation_brightness_factor": 0.78,
        "air_helmholtz_factor": 1.22,
        "top_damping_factor": 0.82,
        "bridge_mobility_factor": 0.88,
        "top_stiffness_to_weight_factor": 0.94,
    },
}

V3_MIX_SCALES: Dict[str, Dict[str, float]] = {
    "sample_000": {
        "direct_string_gain": 1.00,
        "body_modal_gain": 1.00,
        "string_to_body_send_scale": 1.00,
    },
    "sample_001": {
        "direct_string_gain": 1.28,
        "body_modal_gain": 0.85,
        "string_to_body_send_scale": 1.25,
    },
    "sample_002": {
        "direct_string_gain": 0.80,
        "body_modal_gain": 1.28,
        "string_to_body_send_scale": 1.12,
    },
}

V3_NOTE_MODIFIERS: Dict[Tuple[str, str], Dict[str, float]] = {
    ("sample_001", "A2"): {"low_mid_gain_scale": 0.82, "tau_low_scale": 0.88, "tau_mid_high_scale": 0.72, "air_scale": 0.78},
    ("sample_002", "A2"): {"low_mid_gain_scale": 1.32, "tau_low_scale": 1.38, "tau_mid_high_scale": 1.12, "air_scale": 1.18},
    ("sample_001", "A4"): {"low_mid_gain_scale": 0.92, "tau_low_scale": 0.90, "tau_mid_high_scale": 0.75, "top_brightness_scale": 1.15},
    ("sample_002", "A4"): {"low_mid_gain_scale": 1.28, "tau_low_scale": 1.32, "tau_mid_high_scale": 1.22, "top_brightness_scale": 0.88},
    ("sample_001", "E5"): {"low_mid_gain_scale": 0.88, "tau_low_scale": 0.92, "tau_mid_high_scale": 0.70, "top_brightness_scale": 1.12},
    ("sample_002", "E5"): {"low_mid_gain_scale": 1.22, "tau_low_scale": 1.28, "tau_mid_high_scale": 1.18, "top_brightness_scale": 0.85},
}

FALLBACK_PHYSICAL: Dict[str, Dict[str, Any]] = {
    "sample_000": {
        "sample_id": "sample_000",
        "body_depth_m": 0.100,
        "body_volume_proxy": 0.0130,
        "helmholtz_like_frequency_proxy": 118.0,
        "bridge_mobility_proxy": 1.00,
        "top_damping_coeff_proxy": 1.00,
        "back_damping_coeff_proxy": 1.00,
        "soundhole_area": 0.00636,
        "mass_proxies": {"mixed_body_mass_proxy": 1.00},
    },
    "sample_001": {
        "sample_id": "sample_001",
        "body_depth_m": 0.092,
        "body_volume_proxy": 0.0116,
        "helmholtz_like_frequency_proxy": 126.0,
        "bridge_mobility_proxy": 1.14,
        "top_damping_coeff_proxy": 1.12,
        "back_damping_coeff_proxy": 0.96,
        "soundhole_area": 0.00585,
        "mass_proxies": {"mixed_body_mass_proxy": 0.88},
    },
    "sample_002": {
        "sample_id": "sample_002",
        "body_depth_m": 0.112,
        "body_volume_proxy": 0.0148,
        "helmholtz_like_frequency_proxy": 108.0,
        "bridge_mobility_proxy": 0.88,
        "top_damping_coeff_proxy": 0.90,
        "back_damping_coeff_proxy": 1.08,
        "soundhole_area": 0.00700,
        "mass_proxies": {"mixed_body_mass_proxy": 1.16},
    },
}

NOTE_BODY_SUPPORT: Dict[str, Dict[str, float]] = {
    "A2": {"body_modal_mult": 1.00, "low_mid_mode_mult": 1.00, "high_rad_mult": 1.00, "tau_mult": 1.00},
    "A4": {"body_modal_mult": 1.08, "low_mid_mode_mult": 1.12, "high_rad_mult": 0.95, "tau_mult": 1.10},
    "E5": {"body_modal_mult": 1.12, "low_mid_mode_mult": 1.15, "high_rad_mult": 0.90, "tau_mult": 1.14},
}

PER_SAMPLE_NOTE_SUPPORT: Dict[str, Dict[str, Dict[str, float]]] = {
    "A4": {
        "sample_000": {"low_mid_mode_mult": 1.10},
        "sample_001": {"low_mid_mode_mult": 1.06},
        "sample_002": {"low_mid_mode_mult": 1.16},
    },
    "E5": {
        "sample_000": {"low_mid_mode_mult": 1.13},
        "sample_001": {"low_mid_mode_mult": 1.08},
        "sample_002": {"low_mid_mode_mult": 1.20},
    },
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def demo_config(demo_version: str = "v1") -> Dict[str, str]:
    if demo_version not in DEMO_VERSIONS:
        raise ValueError(f"unknown demo_version {demo_version!r}; use v1, v2, or v3")
    return dict(DEMO_VERSIONS[demo_version])


def expected_wav_filename(sample_id: str, note_name: str) -> str:
    return f"{sample_id}_{note_name}_stk_guitar.wav"


def audio_output_dir(repo_root: Path, demo_version: str = "v1") -> Path:
    cfg = demo_config(demo_version)
    return repo_root / cfg["audio_subdir"]


def expected_wav_paths(repo_root: Optional[Path] = None, demo_version: str = "v1") -> List[Path]:
    root = Path(repo_root or REPO_ROOT)
    out_dir = audio_output_dir(root, demo_version)
    return [out_dir / expected_wav_filename(sample_id, note) for sample_id in SAMPLE_SET for note in NOTE_SET]


def load_physical_parameters(
    sample_id: str,
    *,
    audit: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    if audit is not None:
        return extract_per_sample_physical_parameters(sample_id, audit)
    try:
        audit_doc = load_audit_report()
        return extract_per_sample_physical_parameters(sample_id, audit_doc)
    except (FileNotFoundError, KeyError, ValueError):
        fb = FALLBACK_PHYSICAL.get(sample_id)
        if fb is None:
            raise KeyError(f"no physical fallback for {sample_id!r}")
        return dict(fb)


def _soundhole_area_proxy(physical: Mapping[str, Any]) -> float:
    return float(physical.get("soundhole_area") or physical.get("soundhole_area_proxy") or 0.00636)


def _normalized_spread(values: Mapping[str, float]) -> float:
    nums = [float(v) for v in values.values()]
    if not nums:
        return 0.0
    lo, hi = min(nums), max(nums)
    mid = sum(nums) / len(nums)
    if abs(mid) < 1e-9:
        return hi - lo
    return (hi - lo) / abs(mid)


def _apply_diagnostic_multipliers(factors: Dict[str, float], sample_id: str) -> Dict[str, float]:
    mults = DIAGNOSTIC_MULTIPLIERS.get(sample_id) or {}
    out = dict(factors)
    for key, mult in mults.items():
        if key in out:
            out[key] = round(_clamp(float(out[key]) * float(mult), 0.70, 1.35), 6)
    return out


def _note_support(sample_id: str, note_name: str) -> Dict[str, float]:
    base = dict(NOTE_BODY_SUPPORT.get(note_name, NOTE_BODY_SUPPORT["A2"]))
    extra = (PER_SAMPLE_NOTE_SUPPORT.get(note_name) or {}).get(sample_id) or {}
    base.update(extra)
    return base


def _string_decay(factors: Mapping[str, float], note_name: str) -> float:
    top_damp = float(factors.get("top_damping_factor") or 1.0)
    brightness = float(factors.get("radiation_brightness_factor") or 1.0)
    note_scale = {"A2": 0.92, "A4": 1.00, "E5": 1.08}.get(note_name, 1.0)
    sustain = 0.68 / (top_damp ** 0.22 * brightness ** 0.08 * note_scale)
    return round(_clamp(sustain, 0.42, 0.88), 6)


def _harmonic_brightness(factors: Mapping[str, float], mix: Mapping[str, Any]) -> float:
    rad = float(factors.get("radiation_brightness_factor") or 1.0)
    stiff = float(factors.get("top_stiffness_to_weight_factor") or 1.0)
    string_share = float(mix.get("string_bridge") or 0.25)
    return round(_clamp(0.55 + 0.22 * rad + 0.10 * stiff + 0.18 * string_share, 0.35, 1.25), 6)


def _modes_for_stk(
    modes: Sequence[Mapping[str, Any]],
    *,
    note_support: Mapping[str, float],
    factors: Mapping[str, float],
) -> List[Dict[str, Any]]:
    body_mult = float(note_support.get("body_modal_mult") or 1.0)
    low_mid_mult = float(note_support.get("low_mid_mode_mult") or 1.0)
    tau_mult = float(note_support.get("tau_mult") or 1.0)
    soundhole = float(factors.get("soundhole_radiation_factor") or 1.0)
    out: List[Dict[str, Any]] = []
    for row in modes:
        f_hz = float(row["frequency_hz"])
        gain = float(row.get("gain") or 0.0)
        component = str(row.get("component") or "top")
        tau = float(row.get("tau_s") or 0.08) * tau_mult
        q = float(row.get("q") or max(8.0, math.pi * f_hz * tau))
        if component in ("back", "air") and f_hz < 260.0:
            gain *= low_mid_mult
        if component == "air":
            gain *= soundhole
        gain *= body_mult
        out.append(
            {
                "frequency_hz": round(f_hz, 4),
                "gain": round(gain, 6),
                "tau_or_q": round(tau, 6),
                "q": round(q, 4),
                "component": component,
                "role": str(row.get("role") or component),
            }
        )
    return out


def _radiation_weights(mix: Mapping[str, Any], factors: Mapping[str, float]) -> Dict[str, float]:
    air_share = float(mix.get("air_share") or 0.10)
    body_share = float(mix.get("body_modal") or 0.65)
    string_share = float(mix.get("string_bridge") or 0.25)
    warmth = float(factors.get("back_density_warmth_factor") or 1.0)
    brightness = float(factors.get("radiation_brightness_factor") or 1.0)
    top_w = round(_clamp(0.42 * body_share * brightness, 0.15, 0.75), 6)
    back_w = round(_clamp(0.34 * body_share * warmth, 0.12, 0.65), 6)
    air_w = round(_clamp(air_share * float(factors.get("air_helmholtz_factor") or 1.0), 0.04, 0.35), 6)
    total = top_w + back_w + air_w + string_share
    if total > 1e-9:
        scale = (1.0 - string_share) / (top_w + back_w + air_w)
        top_w *= scale
        back_w *= scale
        air_w *= scale
    return {
        "radiation_brightness": round(brightness, 6),
        "top_weight": round(top_w, 6),
        "back_weight": round(back_w, 6),
        "air_weight": round(air_w, 6),
        "string_direct_weight": round(string_share, 6),
    }


def _sample_audit_row(
    sample_id: str,
    *,
    physical: Mapping[str, Any],
    factors: Mapping[str, float],
    mix: Mapping[str, Any],
    radiation: Mapping[str, float],
    modes: Sequence[Mapping[str, Any]],
    string_to_body: float,
) -> Dict[str, Any]:
    return {
        "sample_id": sample_id,
        "profile": SAMPLE_PROFILES.get(sample_id),
        "source_physical_raw": {
            "body_depth_m": physical.get("body_depth_m"),
            "body_volume_proxy": physical.get("body_volume_proxy"),
            "soundhole_area_proxy": _soundhole_area_proxy(physical),
            "bridge_mobility_proxy": physical.get("bridge_mobility_proxy"),
            "helmholtz_like_frequency_proxy": physical.get("helmholtz_like_frequency_proxy"),
        },
        "body_size_cavity_factor": factors.get("body_size_cavity_factor"),
        "body_depth_m": physical.get("body_depth_m"),
        "body_volume_proxy": physical.get("body_volume_proxy"),
        "soundhole_area_proxy": _soundhole_area_proxy(physical),
        "soundhole_radiation_factor": factors.get("soundhole_radiation_factor"),
        "bridge_mobility_factor": factors.get("bridge_mobility_factor"),
        "effective_mass_loading_factor": factors.get("effective_mass_loading_factor"),
        "top_stiffness_to_weight_factor": factors.get("top_stiffness_to_weight_factor"),
        "top_damping_factor": factors.get("top_damping_factor"),
        "material_loss_factor": round(float(factors.get("top_damping_factor") or 1.0) * 0.92, 6),
        "back_density_warmth_factor": factors.get("back_density_warmth_factor"),
        "air_helmholtz_factor": factors.get("air_helmholtz_factor"),
        "radiation_brightness_factor": factors.get("radiation_brightness_factor"),
        "top_weight": radiation.get("top_weight"),
        "back_weight": radiation.get("back_weight"),
        "air_weight": radiation.get("air_weight"),
        "string_body_mix": {
            "string_direct": radiation.get("string_direct_weight"),
            "body_modal": mix.get("body_modal"),
            "string_to_body_send": string_to_body,
        },
        "modal_frequency_hz": [m.get("frequency_hz") for m in modes],
        "modal_gain": [m.get("gain") for m in modes],
        "modal_tau_or_q": [m.get("tau_or_q") for m in modes],
    }


def build_physical_difference_audit(
  renders_a2: Sequence[Mapping[str, Any]],
    *,
    per_sample_physical: Mapping[str, Mapping[str, Any]],
    diagnostic_exaggeration: bool,
) -> Dict[str, Any]:
    per_sample: Dict[str, Any] = {}
    for row in renders_a2:
        sid = str(row["sample_id"])
        physical = per_sample_physical[sid]
        factors = row.get("physical_factors") or {}
        mix = (V11_VOICING.get(sid) or {}).get("mix") or {}
        radiation = row.get("radiation_model") or {}
        modes = (row.get("body_model") or {}).get("modes") or []
        bridge = row.get("bridge_model") or {}
        per_sample[sid] = _sample_audit_row(
            sid,
            physical=physical,
            factors=factors,
            mix=mix,
            radiation=radiation,
            modes=modes,
            string_to_body=float(bridge.get("string_to_body_send") or 0.0),
        )

    factor_spread: Dict[str, Any] = {}
    too_small: List[str] = []
    for key in AUDIT_SCALAR_KEYS:
        if key == "string_body_mix":
            vals = {
                sid: float((per_sample[sid].get("string_body_mix") or {}).get("string_to_body_send") or 0.0)
                for sid in SAMPLE_SET
            }
        else:
            vals = {sid: float(per_sample[sid].get(key) or 0.0) for sid in SAMPLE_SET}
        spread = _normalized_spread(vals)
        status = "meaningful" if spread >= MEANINGFUL_SPREAD_THRESHOLD else "too_small_from_source"
        if status == "too_small_from_source":
            too_small.append(key)
        factor_spread[key] = {
            "values_by_sample": vals,
            "min": min(vals.values()),
            "max": max(vals.values()),
            "range": max(vals.values()) - min(vals.values()),
            "normalized_spread": round(spread, 6),
            "factor_spread_status": status,
        }

    modal_spread = _normalized_spread(
        {sid: float((per_sample[sid].get("modal_frequency_hz") or [0.0])[0] or 0.0) for sid in SAMPLE_SET}
    )

    return {
        "anchor_note": "A2",
        "per_sample": per_sample,
        "factor_spread": factor_spread,
        "too_small_factors": too_small,
        "first_modal_frequency_normalized_spread": round(modal_spread, 6),
        "diagnostic_exaggeration_for_audible_demo": diagnostic_exaggeration,
        "lhs_note": "Samples originate from LHS database; near-zero spread indicates mapping/export weakening, not identical LHS draws.",
    }


def _needs_diagnostic_exaggeration(audit_preview: Mapping[str, Mapping[str, float]]) -> bool:
    spreads = [_normalized_spread(vals) for vals in audit_preview.values()]
    if not spreads:
        return True
    meaningful = sum(1 for s in spreads if s >= MEANINGFUL_SPREAD_THRESHOLD)
    return meaningful < max(4, len(spreads) // 3)


def _apply_v3_mode_modifiers(
    modes: List[Dict[str, Any]],
    *,
    sample_id: str,
    note_name: str,
    body_modal_gain: float,
) -> List[Dict[str, Any]]:
    note_mod = V3_NOTE_MODIFIERS.get((sample_id, note_name), {})
    low_mid_scale = float(note_mod.get("low_mid_gain_scale") or 1.0)
    tau_low = float(note_mod.get("tau_low_scale") or 1.0)
    tau_mid_high = float(note_mod.get("tau_mid_high_scale") or 1.0)
    air_scale = float(note_mod.get("air_scale") or 1.0)
    top_brightness = float(note_mod.get("top_brightness_scale") or 1.0)
    out: List[Dict[str, Any]] = []
    for row in modes:
        m = dict(row)
        f_hz = float(m["frequency_hz"])
        gain = float(m.get("gain") or 0.0) * body_modal_gain
        tau = float(m.get("tau_or_q") or 0.08)
        component = str(m.get("component") or "top")
        if 120.0 <= f_hz <= 450.0:
            gain *= low_mid_scale
            tau *= tau_low if f_hz < 260.0 else tau_mid_high
        elif f_hz < 120.0:
            tau *= tau_low
        else:
            tau *= tau_mid_high
        if component == "air":
            gain *= air_scale
        if component in ("top", "radiation"):
            gain *= top_brightness
        m["gain"] = round(gain, 6)
        m["tau_or_q"] = round(max(tau, 0.015), 6)
        if m.get("q"):
            m["q"] = round(max(4.0, math.pi * f_hz * m["tau_or_q"]), 4)
        out.append(m)
    return out


def build_render_entry(
    sample_id: str,
    note_name: str,
    *,
    physical: Mapping[str, Any],
    reference_physical: Mapping[str, Any],
    sample_rate: int = NUMERIC_SR,
    duration_s: float = DURATION_S,
    repo_root: Optional[Path] = None,
    demo_version: str = "v1",
    factor_multipliers: Optional[Mapping[str, float]] = None,
    perceptual_mix: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    root = Path(repo_root or REPO_ROOT)
    cfg = demo_config(demo_version)
    readonly_modes = _load_readonly_reference_modes(root)

    factors, _ = compute_v5_physical_factors(
        physical, reference_physical, sample_id=sample_id, voicing=V11_VOICING
    )
    if factor_multipliers:
        lo, hi = (0.65, 1.40) if demo_version == "v3" else (0.70, 1.35)
        for key, mult in factor_multipliers.items():
            if key in factors:
                factors[key] = round(_clamp(float(factors[key]) * float(mult), lo, hi), 6)

    voicing = V11_VOICING[sample_id]
    pluck = _clamp(FIXED_PLUCK_POSITION + float(voicing.get("pluck_delta") or 0.0), 0.10, 0.20)
    modes_raw = _pick_reference_modes(readonly_modes, factors)
    bridge = _bridge_transfer_summary(factors)
    mix = dict(voicing.get("mix") or {})
    pluck_position_ratio = round(pluck, 5)
    note_support = _note_support(sample_id, note_name)
    modes = _modes_for_stk(modes_raw, note_support=note_support, factors=factors)
    wav_name = expected_wav_filename(sample_id, note_name)
    wav_path = root / cfg["audio_subdir"] / wav_name

    string_to_body = round(
        _clamp(
            1.0 - float(mix.get("string_bridge") or 0.25)
            + 0.12 * float(factors.get("bridge_mobility_factor") or 1.0)
            - 0.12,
            0.35,
            0.92,
        ),
        6,
    )
    radiation = _radiation_weights(mix, factors)
    mix_scales = dict(perceptual_mix or {})
    direct_gain = float(mix_scales.get("direct_string_gain") or 1.0)
    body_modal_gain = float(mix_scales.get("body_modal_gain") or 1.0)
    send_scale = float(mix_scales.get("string_to_body_send_scale") or 1.0)
    if demo_version == "v3":
        radiation["string_direct_weight"] = round(
            _clamp(radiation["string_direct_weight"] * direct_gain, 0.12, 0.62), 6
        )
        string_to_body = round(_clamp(string_to_body * send_scale, 0.30, 0.98), 6)
        modes = _apply_v3_mode_modifiers(
            modes, sample_id=sample_id, note_name=note_name, body_modal_gain=body_modal_gain
        )
        body_share = float(mix.get("body_modal") or 0.65) * body_modal_gain
        radiation["top_weight"] = round(radiation["top_weight"] * (0.92 if sample_id == "sample_002" else 1.0), 6)
        radiation["back_weight"] = round(
            radiation["back_weight"] * (1.18 if sample_id == "sample_002" else (0.92 if sample_id == "sample_001" else 1.0)),
            6,
        )
        radiation["air_weight"] = round(
            radiation["air_weight"] * (1.15 if sample_id == "sample_002" else (0.82 if sample_id == "sample_001" else 1.0)),
            6,
        )
    else:
        body_share = float(mix.get("body_modal") or 0.65)

    material_loss = round(float(factors.get("top_damping_factor") or 1.0) * 0.92, 6)
    perceptual_calibration: Optional[Dict[str, Any]] = None
    if demo_version == "v3":
        perceptual_calibration = {
            "diagnostic_exaggeration_for_audible_demo": True,
            "profile": SAMPLE_PROFILES.get(sample_id),
            "direct_string_gain": direct_gain,
            "body_modal_gain": body_modal_gain,
            "string_to_body_send_scale": send_scale,
            "note_modifiers": V3_NOTE_MODIFIERS.get((sample_id, note_name), {}),
            "mapping_strength": 1.45,
        }

    entry: Dict[str, Any] = {
        "sample_id": sample_id,
        "note_name": note_name,
        "frequency_hz": float(NOTE_FREQUENCY_HZ[note_name]),
        "duration_s": float(duration_s),
        "sample_rate": int(sample_rate),
        "profile": SAMPLE_PROFILES.get(sample_id, str(voicing.get("profile"))),
        "physical_factors": {k: factors[k] for k in PHYSICAL_FACTOR_KEYS if k in factors},
        "source_physical_raw": {
            "body_depth_m": physical.get("body_depth_m"),
            "body_volume_proxy": physical.get("body_volume_proxy"),
            "soundhole_area_proxy": _soundhole_area_proxy(physical),
            "bridge_mobility_proxy": physical.get("bridge_mobility_proxy"),
            "helmholtz_like_frequency_proxy": physical.get("helmholtz_like_frequency_proxy"),
        },
        "string_model": {
            "pluck_position": pluck_position_ratio,
            "string_decay": _string_decay(factors, note_name),
            "harmonic_brightness": _harmonic_brightness(factors, mix),
            "excitation_strength": round(
                _clamp(0.82 + 0.10 * float(factors.get("bridge_mobility_factor") or 1.0), 0.65, 1.15), 6
            ),
        },
        "bridge_model": {
            "bridge_mobility": round(float(factors.get("bridge_mobility_factor") or 1.0), 6),
            "bridge_damping": round(0.045 / max(float(bridge.get("attack_scale") or 1.0), 0.5), 6),
            "string_to_body_send": string_to_body,
            "highpass_hz": float(bridge.get("highpass_hz") or 55.0),
            "low_coupling_scale": float(bridge.get("low_coupling_scale") or 1.0),
        },
        "body_model": {
            "effective_mass_loading": round(float(factors.get("effective_mass_loading_factor") or 1.0), 6),
            "body_size_cavity_factor": round(float(factors.get("body_size_cavity_factor") or 1.0), 6),
            "depth_factor": round(float(physical.get("body_depth_m") or 0.10) / 0.10, 6),
            "body_depth_m": float(physical.get("body_depth_m") or 0.10),
            "body_volume_proxy": float(physical.get("body_volume_proxy") or 0.013),
            "soundhole_area_proxy": _soundhole_area_proxy(physical),
            "soundhole_radiation_factor": round(float(factors.get("soundhole_radiation_factor") or 1.0), 6),
            "low_mid_body_support": round(float(note_support.get("low_mid_mode_mult") or 1.0), 6),
            "body_modal_gain": round(body_modal_gain, 6),
            "modes": modes,
        },
        "material_model": {
            "top_damping": round(float(factors.get("top_damping_factor") or 1.0), 6),
            "back_warmth": round(float(factors.get("back_density_warmth_factor") or 1.0), 6),
            "material_loss": material_loss,
            "material_loss_factor": material_loss,
            "stiffness_to_weight": round(float(factors.get("top_stiffness_to_weight_factor") or 1.0), 6),
        },
        "radiation_model": radiation,
        "string_body_mix": {
            "string_direct": radiation["string_direct_weight"],
            "direct_string_gain": direct_gain,
            "body_modal": body_share,
            "body_modal_gain": body_modal_gain,
            "string_to_body_send": string_to_body,
        },
        "output_model": {
            "peak_ceiling_dbfs": float(NOTE_PEAK_TARGET_DBFS.get(note_name, -6.0)),
            "peak_target_dbfs": float(NOTE_PEAK_TARGET_DBFS.get(note_name, -6.0)),
            "loudness_reference_dbfs": float(TARGET_RMS_DBFS),
            "normalize_rms": False,
            "output_wav_path": str(wav_path.relative_to(root)).replace("\\", "/"),
        },
    }
    if perceptual_calibration is not None:
        entry["perceptual_calibration"] = perceptual_calibration
    return entry


def build_parameter_export(
    *,
    repo_root: Optional[Path] = None,
    audit: Optional[Mapping[str, Any]] = None,
    sample_rate: int = NUMERIC_SR,
    duration_s: float = DURATION_S,
    demo_version: str = "v1",
) -> Dict[str, Any]:
    root = Path(repo_root or REPO_ROOT)
    cfg = demo_config(demo_version)
    reference_physical = load_physical_parameters(REFERENCE_SAMPLE_ID, audit=audit)
    per_sample_physical: Dict[str, Dict[str, Any]] = {
        sid: load_physical_parameters(sid, audit=audit) for sid in SAMPLE_SET
    }

    preview_factors: Dict[str, Dict[str, float]] = {}
    for sid in SAMPLE_SET:
        fac, _ = compute_v5_physical_factors(
            per_sample_physical[sid], reference_physical, sample_id=sid, voicing=V11_VOICING
        )
        preview_factors[sid] = {k: float(fac.get(k) or 1.0) for k in PHYSICAL_FACTOR_KEYS}

    preview_spread = {
        k: {sid: preview_factors[sid].get(k, 1.0) for sid in SAMPLE_SET} for k in PHYSICAL_FACTOR_KEYS
    }
    diagnostic = demo_version == "v3" or (_needs_diagnostic_exaggeration(preview_spread) and demo_version == "v2")
    factor_table = V3_FACTOR_MULTIPLIERS if demo_version == "v3" else DIAGNOSTIC_MULTIPLIERS

    renders: List[Dict[str, Any]] = []
    per_sample_summary: Dict[str, Any] = {}
    for sample_id in SAMPLE_SET:
        physical = per_sample_physical[sample_id]
        mults = factor_table.get(sample_id) if diagnostic else None
        mix_scales = V3_MIX_SCALES.get(sample_id) if demo_version == "v3" else None
        per_sample_summary[sample_id] = {
            "profile": SAMPLE_PROFILES.get(sample_id),
            "physical_source": "audit_json" if audit is not None else "audit_or_fallback",
            "body_depth_m": physical.get("body_depth_m"),
            "bridge_mobility_proxy": physical.get("bridge_mobility_proxy"),
            "diagnostic_multipliers_applied": bool(mults),
            "v3_perceptual_calibration": demo_version == "v3",
        }
        for note_name in NOTE_SET:
            factor_mults = None
            if mults:
                base_fac, _ = compute_v5_physical_factors(
                    physical, reference_physical, sample_id=sample_id, voicing=V11_VOICING
                )
                adjusted = dict(base_fac)
                table = V3_FACTOR_MULTIPLIERS if demo_version == "v3" else DIAGNOSTIC_MULTIPLIERS
                lo, hi = (0.65, 1.40) if demo_version == "v3" else (0.70, 1.35)
                for key, mult in (table.get(sample_id) or {}).items():
                    if key in adjusted:
                        adjusted[key] = round(_clamp(float(adjusted[key]) * float(mult), lo, hi), 6)
                factor_mults = {
                    k: adjusted[k] / max(float(base_fac.get(k) or 1.0), 1e-9) for k in adjusted
                }
            renders.append(
                build_render_entry(
                    sample_id,
                    note_name,
                    physical=physical,
                    reference_physical=reference_physical,
                    sample_rate=sample_rate,
                    duration_s=duration_s,
                    repo_root=root,
                    demo_version=demo_version,
                    factor_multipliers=factor_mults,
                    perceptual_mix=mix_scales,
                )
            )

    renders_a2 = [r for r in renders if r.get("note_name") == "A2"]
    physical_difference_audit = build_physical_difference_audit(
        renders_a2,
        per_sample_physical=per_sample_physical,
        diagnostic_exaggeration=diagnostic,
    )

    doc: Dict[str, Any] = {
        "export_version": cfg["export_version"],
        "demo_version": cfg["demo_id"],
        "generated_at": _utc_now(),
        "renderer": RENDERER_TARGET,
        "python_role": PYTHON_ROLE,
        "repo_root": str(root),
        "audio_output_subdir": cfg["audio_subdir"],
        "report_json_path": cfg["report_json"],
        "report_md_path": cfg["report_md"],
        "sample_set": list(SAMPLE_SET),
        "note_set": list(NOTE_SET),
        "sample_rate": int(sample_rate),
        "duration_s": float(duration_s),
        "physical_factor_keys": list(PHYSICAL_FACTOR_KEYS),
        "physical_difference_audit": physical_difference_audit,
        "per_sample_summary": per_sample_summary,
        "per_sample_differences": _per_sample_difference_summary(renders),
        "renders": renders,
        "expected_render_count": len(SAMPLE_SET) * len(NOTE_SET),
        "expected_wav_files": [expected_wav_filename(s, n) for s in SAMPLE_SET for n in NOTE_SET],
        "limitations": [
            "Python exports parameters only; WAV synthesis is C++/STK on VM.",
            "Modal catalog is read-only PGSM reference — not live FEM/ROM at export time.",
            "Body response in STK must be driven by bridge force, not an independent pluck.",
            "v2: peak ceiling only in C++; RMS not forced equal across samples.",
            "v3: perceptual calibration for clearer sample differentiation; diagnostic_exaggeration_for_audible_demo.",
        ],
    }
    if demo_version == "v3":
        doc["perceptual_calibration_policy"] = {
            "diagnostic_exaggeration_for_audible_demo": True,
            "base": "v2_physical_audit",
            "sample_000": "balanced_neutral_reference",
            "sample_001": "bright_light_fast_attack_forward",
            "sample_002": "warm_deep_heavy_body_resonant",
        }
    return doc


def _per_sample_difference_summary(renders: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_sample: Dict[str, Dict[str, Any]] = {}
    for row in renders:
        if row.get("note_name") != "A2":
            continue
        sid = str(row["sample_id"])
        by_sample[sid] = {
            "profile": row.get("profile"),
            "string_to_body_send": (row.get("bridge_model") or {}).get("string_to_body_send"),
            "bridge_mobility": (row.get("bridge_model") or {}).get("bridge_mobility"),
            "soundhole_radiation_factor": (row.get("body_model") or {}).get("soundhole_radiation_factor"),
            "effective_mass_loading": (row.get("body_model") or {}).get("effective_mass_loading"),
            "radiation_brightness": (row.get("radiation_model") or {}).get("radiation_brightness"),
            "top_damping": (row.get("material_model") or {}).get("top_damping"),
            "first_mode_hz": ((row.get("body_model") or {}).get("modes") or [{}])[0].get("frequency_hz"),
        }
    ref = by_sample.get("sample_000") or {}
    deltas: Dict[str, Dict[str, float]] = {}
    for sid, row in by_sample.items():
        if sid == "sample_000":
            continue
        deltas[sid] = {
            k: round(float(row.get(k) or 0.0) - float(ref.get(k) or 0.0), 6)
            for k in row
            if k != "profile" and isinstance(row.get(k), (int, float))
        }
    return {"A2_anchor": by_sample, "delta_vs_sample_000": deltas}


def write_parameter_export(
    output_path: Optional[Path] = None,
    *,
    repo_root: Optional[Path] = None,
    audit: Optional[Mapping[str, Any]] = None,
    demo_version: str = "v1",
) -> Path:
    root = Path(repo_root or REPO_ROOT)
    cfg = demo_config(demo_version)
    out = Path(output_path or (root / cfg["params_json"]))
    doc = build_parameter_export(repo_root=root, audit=audit, demo_version=demo_version)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Export PGSM physical parameters for STK/C++ renderer.")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--demo-version",
        choices=sorted(DEMO_VERSIONS.keys()),
        default="v1",
        help="Demo pack version (v2 audit; v3 stronger perceptual differentiation)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    cfg = demo_config(args.demo_version)
    default_out = args.repo_root / cfg["params_json"]
    path = write_parameter_export(
        args.output or default_out,
        repo_root=args.repo_root,
        demo_version=args.demo_version,
    )
    print(f"Wrote STK parameter export ({cfg['demo_id']}): {path}")
    print(f"Renders: {len(SAMPLE_SET) * len(NOTE_SET)} (no audio generated)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
