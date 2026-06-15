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
EXPORT_VERSION_V4 = "pgsm_stk_parameter_export_v4_10_samples"
RENDERER_TARGET = "stk_cpp"
PYTHON_ROLE = "parameter_export_only"
DURATION_S = 2.5
SAMPLE_SET_V1_V3: Tuple[str, ...] = ("sample_000", "sample_001", "sample_002")
SAMPLE_SET_V4: Tuple[str, ...] = tuple(f"sample_{i:03d}" for i in range(10))
SAMPLE_SET: Tuple[str, ...] = SAMPLE_SET_V1_V3
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
    "v4_10_samples": {
        "demo_id": "pgsm_stk_guitar_demo_v4_10_samples",
        "export_version": EXPORT_VERSION_V4,
        "audio_subdir": "audio/pgsm_stk_guitar_demo_v4_10_samples",
        "params_json": "audio/debug_reports/pgsm_stk_demo_parameters_v4_10_samples.json",
        "report_json": "audio/debug_reports/pgsm_stk_guitar_demo_v4_10_samples_report.json",
        "report_md": "audio/debug_reports/pgsm_stk_guitar_demo_v4_10_samples_report.md",
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

LHS_POOL_JSON = REPO_ROOT / "ROM" / "classic" / "lhs_pool.json"


def sample_set_for_demo(demo_version: str) -> Tuple[str, ...]:
    if demo_version == "v4_10_samples":
        return SAMPLE_SET_V4
    return SAMPLE_SET_V1_V3


def _physical_from_lhs_entry(params: Mapping[str, Any], sample_id: str) -> Dict[str, Any]:
    depth = float(params.get("geometry.depth") or 0.10)
    length = float(params.get("geometry.length") or 0.45)
    width = float(params.get("geometry.width") or 0.35)
    hole_r = float(params.get("geometry.hole_radius") or 0.045)
    top_t = float(params.get("geometry.top_thickness") or 0.003)
    back_t = float(params.get("geometry.back_thickness") or 0.0033)
    vol = length * width * depth
    hole_area = math.pi * hole_r * hole_r
    helm = 118.0 * (0.013 / max(vol, 1e-6)) ** 0.18
    return {
        "sample_id": sample_id,
        "top_wood_id": str(params.get("top_wood_id") or "spruce"),
        "back_wood_id": str(params.get("back_wood_id") or "rosewood"),
        "body_depth_m": depth,
        "body_volume_proxy": round(vol, 6),
        "helmholtz_like_frequency_proxy": round(helm, 4),
        "bridge_mobility_proxy": round(0.92 + 0.16 * (top_t / 0.003), 4),
        "top_damping_coeff_proxy": round(0.88 + 0.24 * (top_t / 0.003), 4),
        "back_damping_coeff_proxy": round(0.90 + 0.20 * (back_t / 0.0033), 4),
        "soundhole_area": round(hole_area, 6),
        "mass_proxies": {"mixed_body_mass_proxy": round(0.85 + 0.30 * (vol / 0.013), 4)},
    }


def _load_lhs_fallback_physical() -> Dict[str, Dict[str, Any]]:
    if not LHS_POOL_JSON.is_file():
        return {}
    try:
        pool = json.loads(LHS_POOL_JSON.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for entry in pool.get("entries") or []:
        sid = str(entry.get("id") or "")
        if not sid.startswith("sample_"):
            continue
        params = entry.get("parameters") or {}
        if not params:
            continue
        out[sid] = _physical_from_lhs_entry(params, sid)
    return out


def _merge_fallback_physical() -> Dict[str, Dict[str, Any]]:
    merged = dict(_load_lhs_fallback_physical())
    manual = {
        "sample_000": FALLBACK_PHYSICAL_MANUAL.get("sample_000"),
        "sample_001": FALLBACK_PHYSICAL_MANUAL.get("sample_001"),
        "sample_002": FALLBACK_PHYSICAL_MANUAL.get("sample_002"),
    }
    for sid, row in manual.items():
        if row:
            merged[sid] = dict(row)
    return merged


FALLBACK_PHYSICAL_MANUAL: Dict[str, Dict[str, Any]] = {
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

FALLBACK_PHYSICAL: Dict[str, Dict[str, Any]] = _merge_fallback_physical()

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


def _extended_voicing(sample_ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    ext = dict(V11_VOICING)
    neutral = dict(V11_VOICING["sample_000"])
    for sid in sample_ids:
        if sid not in ext:
            row = dict(neutral)
            row["profile"] = f"lhs_neutral_{sid}"
            ext[sid] = row
    return ext


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def demo_config(demo_version: str = "v1") -> Dict[str, str]:
    if demo_version not in DEMO_VERSIONS:
        raise ValueError(f"unknown demo_version {demo_version!r}; use v1, v2, v3, or v4_10_samples")
    return dict(DEMO_VERSIONS[demo_version])


def expected_wav_filename(sample_id: str, note_name: str) -> str:
    return f"{sample_id}_{note_name}_stk_guitar.wav"


def audio_output_dir(repo_root: Path, demo_version: str = "v1") -> Path:
    cfg = demo_config(demo_version)
    return repo_root / cfg["audio_subdir"]


def expected_wav_paths(repo_root: Optional[Path] = None, demo_version: str = "v1") -> List[Path]:
    root = Path(repo_root or REPO_ROOT)
    out_dir = audio_output_dir(root, demo_version)
    sample_set = sample_set_for_demo(demo_version)
    return [out_dir / expected_wav_filename(sample_id, note) for sample_id in sample_set for note in NOTE_SET]


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


STK_FACTOR_SPECS: Tuple[Dict[str, Any], ...] = (
    {"id": "pluck_position", "category": "A", "renderer_target": "stk_plucked_pluck_position", "note_dependency": False},
    {"id": "excitation_strength", "category": "A", "renderer_target": "stk_plucked_excitation", "note_dependency": False},
    {"id": "harmonic_brightness", "category": "A", "renderer_target": "string_harmonic_envelope", "note_dependency": True},
    {"id": "string_decay", "category": "A", "renderer_target": "stk_plucked_decay", "note_dependency": True},
    {"id": "note_frequency", "category": "A", "renderer_target": "stk_plucked_frequency", "note_dependency": True},
    {"id": "note_specific_excitation_scale", "category": "A", "renderer_target": "stk_plucked_excitation_scale", "note_dependency": True},
    {"id": "bridge_mobility_factor", "category": "B", "renderer_target": "bridge_coupling_gain", "note_dependency": False},
    {"id": "bridge_damping", "category": "B", "renderer_target": "bridge_smoothing", "note_dependency": False},
    {"id": "string_to_body_send", "category": "B", "renderer_target": "bridge_to_modal_drive", "note_dependency": False},
    {"id": "string_body_mix", "category": "B", "renderer_target": "string_direct_vs_body_modal_mix", "note_dependency": False},
    {"id": "body_depth_m", "category": "C", "renderer_target": "low_mode_frequency_tau", "note_dependency": False},
    {"id": "body_volume_proxy", "category": "C", "renderer_target": "cavity_mode_gain", "note_dependency": False},
    {"id": "body_size_cavity_factor", "category": "C", "renderer_target": "low_mid_modal_gain", "note_dependency": False},
    {"id": "effective_mass_loading_factor", "category": "C", "renderer_target": "bridge_attack_smoothing", "note_dependency": False},
    {"id": "shape_flatness_or_depth_factor", "category": "C", "renderer_target": "depth_factor_modal_shift", "note_dependency": False},
    {"id": "soundhole_area_proxy", "category": "D", "renderer_target": "air_radiation_area_scaling", "note_dependency": False},
    {"id": "soundhole_radiation_factor", "category": "D", "renderer_target": "air_mode_gain", "note_dependency": False},
    {"id": "air_helmholtz_factor", "category": "D", "renderer_target": "air_mode_frequency", "note_dependency": False},
    {"id": "air_weight", "category": "D", "renderer_target": "final_radiation_mix_air", "note_dependency": False},
    {"id": "top_stiffness_to_weight_factor", "category": "E", "renderer_target": "top_mode_frequency_brightness", "note_dependency": False},
    {"id": "top_damping_factor", "category": "E", "renderer_target": "modal_tau_top", "note_dependency": False},
    {"id": "material_loss_factor", "category": "E", "renderer_target": "modal_Q_decay", "note_dependency": False},
    {"id": "back_density_warmth_factor", "category": "E", "renderer_target": "back_mode_gain_tau", "note_dependency": False},
    {"id": "per_mode_tau_q_modifiers", "category": "E", "renderer_target": "modal_bank_tau_per_mode", "note_dependency": True},
    {"id": "modal_frequencies", "category": "F", "renderer_target": "modal_bank_frequency", "note_dependency": True},
    {"id": "modal_gains", "category": "F", "renderer_target": "modal_bank_gain", "note_dependency": True},
    {"id": "modal_tau_q", "category": "F", "renderer_target": "modal_bank_tau", "note_dependency": True},
    {"id": "top_back_air_component_labels", "category": "F", "renderer_target": "modal_component_routing", "note_dependency": False},
    {"id": "low_mid_body_support_120_450_hz", "category": "F", "renderer_target": "body_modal_gain_120_450", "note_dependency": True},
    {"id": "mode_frequency_shifts_per_sample", "category": "F", "renderer_target": "per_sample_modal_frequency_shift", "note_dependency": False},
    {"id": "radiation_brightness_factor", "category": "G", "renderer_target": "radiation_mode_gain", "note_dependency": False},
    {"id": "top_weight", "category": "G", "renderer_target": "final_radiation_mix_top", "note_dependency": False},
    {"id": "back_weight", "category": "G", "renderer_target": "final_radiation_mix_back", "note_dependency": False},
    {"id": "air_weight_radiation", "category": "G", "renderer_target": "final_radiation_mix_air", "note_dependency": False},
    {"id": "high_frequency_radiation_rolloff", "category": "G", "renderer_target": "hf_radiation_rolloff", "note_dependency": True},
)


def _render_factor_scalar(row: Mapping[str, Any], factor_id: str) -> Optional[float]:
    sm = row.get("string_model") or {}
    bm = row.get("bridge_model") or {}
    body = row.get("body_model") or {}
    mat = row.get("material_model") or {}
    rad = row.get("radiation_model") or {}
    mix = row.get("string_body_mix") or {}
    pf = row.get("physical_factors") or {}
    modes = body.get("modes") or []
    if factor_id == "pluck_position":
        return float(sm.get("pluck_position") or 0.0)
    if factor_id == "excitation_strength":
        return float(sm.get("excitation_strength") or 0.0)
    if factor_id == "harmonic_brightness":
        return float(sm.get("harmonic_brightness") or 0.0)
    if factor_id == "string_decay":
        return float(sm.get("string_decay") or 0.0)
    if factor_id == "note_frequency":
        return float(row.get("frequency_hz") or 0.0)
    if factor_id == "note_specific_excitation_scale":
        return float(sm.get("note_excitation_scale") or sm.get("excitation_strength") or 0.0)
    if factor_id == "bridge_mobility_factor":
        return float(bm.get("bridge_mobility") or pf.get("bridge_mobility_factor") or 0.0)
    if factor_id == "bridge_damping":
        return float(bm.get("bridge_damping") or 0.0)
    if factor_id == "string_to_body_send":
        return float(bm.get("string_to_body_send") or 0.0)
    if factor_id == "string_body_mix":
        return float(mix.get("string_direct") or rad.get("string_direct_weight") or 0.0)
    if factor_id == "body_depth_m":
        return float(body.get("body_depth_m") or pf.get("body_depth_m") or 0.0)
    if factor_id == "body_volume_proxy":
        return float(body.get("body_volume_proxy") or 0.0)
    if factor_id == "body_size_cavity_factor":
        return float(body.get("body_size_cavity_factor") or pf.get("body_size_cavity_factor") or 0.0)
    if factor_id == "effective_mass_loading_factor":
        return float(body.get("effective_mass_loading") or pf.get("effective_mass_loading_factor") or 0.0)
    if factor_id == "shape_flatness_or_depth_factor":
        return float(body.get("depth_factor") or 0.0)
    if factor_id == "soundhole_area_proxy":
        return float(body.get("soundhole_area_proxy") or 0.0)
    if factor_id == "soundhole_radiation_factor":
        return float(body.get("soundhole_radiation_factor") or pf.get("soundhole_radiation_factor") or 0.0)
    if factor_id == "air_helmholtz_factor":
        return float(pf.get("air_helmholtz_factor") or 0.0)
    if factor_id == "air_weight":
        return float(rad.get("air_weight") or 0.0)
    if factor_id == "top_stiffness_to_weight_factor":
        return float(mat.get("stiffness_to_weight") or pf.get("top_stiffness_to_weight_factor") or 0.0)
    if factor_id == "top_damping_factor":
        return float(mat.get("top_damping") or pf.get("top_damping_factor") or 0.0)
    if factor_id == "material_loss_factor":
        return float(mat.get("material_loss_factor") or mat.get("material_loss") or 0.0)
    if factor_id == "back_density_warmth_factor":
        return float(mat.get("back_warmth") or pf.get("back_density_warmth_factor") or 0.0)
    if factor_id == "per_mode_tau_q_modifiers":
        taus = [float(m.get("tau_or_q") or 0.0) for m in modes]
        return sum(taus) / len(taus) if taus else None
    if factor_id == "modal_frequencies":
        freqs = [float(m.get("frequency_hz") or 0.0) for m in modes]
        return freqs[0] if freqs else None
    if factor_id == "modal_gains":
        gains = [float(m.get("gain") or 0.0) for m in modes]
        return sum(gains) if gains else None
    if factor_id == "modal_tau_q":
        taus = [float(m.get("tau_or_q") or 0.0) for m in modes]
        return taus[0] if taus else None
    if factor_id == "top_back_air_component_labels":
        return float(len({str(m.get("component")) for m in modes}))
    if factor_id == "low_mid_body_support_120_450_hz":
        return float(body.get("low_mid_body_support") or 0.0)
    if factor_id == "mode_frequency_shifts_per_sample":
        freqs = [float(m.get("frequency_hz") or 0.0) for m in modes]
        return freqs[0] if freqs else None
    if factor_id == "radiation_brightness_factor":
        return float(rad.get("radiation_brightness") or pf.get("radiation_brightness_factor") or 0.0)
    if factor_id == "top_weight":
        return float(rad.get("top_weight") or 0.0)
    if factor_id == "back_weight":
        return float(rad.get("back_weight") or 0.0)
    if factor_id == "air_weight_radiation":
        return float(rad.get("air_weight") or 0.0)
    if factor_id == "high_frequency_radiation_rolloff":
        return float(rad.get("high_frequency_radiation_rolloff") or 0.0)
    return None


def _factor_exists_in_pgsm(factor_id: str) -> bool:
    known = {
        "pluck_position", "excitation_strength", "harmonic_brightness", "string_decay",
        "note_frequency", "bridge_mobility_factor", "bridge_damping", "string_to_body_send",
        "body_depth_m", "body_volume_proxy", "body_size_cavity_factor", "effective_mass_loading_factor",
        "soundhole_area_proxy", "soundhole_radiation_factor", "air_helmholtz_factor",
        "top_stiffness_to_weight_factor", "top_damping_factor", "material_loss_factor",
        "back_density_warmth_factor", "radiation_brightness_factor", "top_weight", "back_weight",
        "air_weight", "modal_frequencies", "modal_gains", "modal_tau_q",
    }
    return factor_id in known or factor_id in AUDIT_SCALAR_KEYS


def build_stk_factor_activation_matrix(
    renders_a2: Sequence[Mapping[str, Any]],
    *,
    sample_set: Sequence[str],
    factor_spread_table: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    matrix: List[Dict[str, Any]] = []
    for spec in STK_FACTOR_SPECS:
        fid = str(spec["id"])
        vals: Dict[str, float] = {}
        for row in renders_a2:
            sid = str(row["sample_id"])
            if sid not in sample_set:
                continue
            v = _render_factor_scalar(row, fid)
            if v is not None:
                vals[sid] = float(v)
        spread = _normalized_spread(vals) if vals else 0.0
        spread_row = factor_spread_table.get(fid) or factor_spread_table.get(
            fid.replace("_radiation", "").replace("low_mid_body_support_120_450_hz", "low_mid_body_support")
        )
        if spread_row and isinstance(spread_row, dict):
            spread = float(spread_row.get("normalized_spread") or spread)
        exported = bool(vals)
        parsed = exported
        applied = exported and fid not in {"top_back_air_component_labels"}
        strength = 1.45 if fid in {"direct_string_gain", "body_modal_gain"} else 1.0
        if fid in ("bridge_mobility_factor", "soundhole_radiation_factor", "body_depth_m", "material_loss_factor"):
            strength = 1.35
        status = "active"
        if not exported:
            status = "missing_from_export"
        elif spread < MEANINGFUL_SPREAD_THRESHOLD and len(sample_set) > 3:
            status = "active_but_too_subtle"
        matrix.append(
            {
                "factor_id": fid,
                "category": spec["category"],
                "exists_in_pgsm_or_audit": _factor_exists_in_pgsm(fid),
                "exported_to_json": exported,
                "parsed_by_cpp": parsed,
                "applied_in_audio": applied,
                "renderer_target": spec["renderer_target"],
                "strength_used": strength,
                "per_sample_spread": round(spread, 6),
                "note_dependency": bool(spec.get("note_dependency")),
                "values_by_sample_A2": vals,
                "status": status,
            }
        )
    return matrix


def build_missing_or_weak_factor_summary(
    matrix: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in matrix:
        fid = str(row["factor_id"])
        exported = bool(row.get("exported_to_json"))
        applied = bool(row.get("applied_in_audio"))
        spread = float(row.get("per_sample_spread") or 0.0)
        if not exported:
            status = "missing_from_export"
        elif exported and not applied:
            status = "exported_but_not_applied"
        elif spread < MEANINGFUL_SPREAD_THRESHOLD:
            status = "active_but_too_subtle"
        else:
            status = "active"
        out.append({"factor_id": fid, "status": status, "per_sample_spread": spread})
    return out


def _compute_v4_continuous_mix(
    physical: Mapping[str, Any],
    reference: Mapping[str, Any],
    factors: Mapping[str, float],
) -> Dict[str, float]:
    ref_depth = float(reference.get("body_depth_m") or 0.10)
    ref_vol = float(reference.get("body_volume_proxy") or 0.013)
    ref_hole = _soundhole_area_proxy(reference)
    ref_mass = float((reference.get("mass_proxies") or {}).get("mixed_body_mass_proxy") or 1.0)
    depth = float(physical.get("body_depth_m") or ref_depth)
    vol = float(physical.get("body_volume_proxy") or ref_vol)
    hole = _soundhole_area_proxy(physical)
    mass = float((physical.get("mass_proxies") or {}).get("mixed_body_mass_proxy") or 1.0)
    mob = float(factors.get("bridge_mobility_factor") or 1.0)
    bright = float(factors.get("radiation_brightness_factor") or 1.0)
    stiff = float(factors.get("top_stiffness_to_weight_factor") or 1.0)
    depth_ratio = depth / max(ref_depth, 1e-6)
    vol_ratio = vol / max(ref_vol, 1e-6)
    hole_ratio = hole / max(ref_hole, 1e-6)
    mass_ratio = mass / max(ref_mass, 1e-6)
    direct = _clamp(1.0 + 0.22 * (bright - 1.0) - 0.18 * (mass_ratio - 1.0) + 0.12 * (mob - 1.0) + 0.08 * (stiff - 1.0), 0.72, 1.35)
    body_modal = _clamp(1.0 + 0.25 * (vol_ratio - 1.0) + 0.18 * (depth_ratio - 1.0) + 0.15 * (mass_ratio - 1.0) - 0.12 * (bright - 1.0), 0.70, 1.40)
    send = _clamp(1.0 + 0.20 * (mob - 1.0) + 0.10 * (hole_ratio - 1.0), 0.75, 1.30)
    return {
        "direct_string_gain": round(direct, 6),
        "body_modal_gain": round(body_modal, 6),
        "string_to_body_send_scale": round(send, 6),
    }


def _stretch_factors_preserving_rank(
    factors: Dict[str, float],
    all_by_sample: Mapping[str, Mapping[str, float]],
    sample_id: str,
    *,
    strength: float = 1.28,
) -> Dict[str, float]:
    keys = (
        "bridge_mobility_factor", "body_size_cavity_factor", "soundhole_radiation_factor",
        "effective_mass_loading_factor", "top_stiffness_to_weight_factor", "top_damping_factor",
        "back_density_warmth_factor", "air_helmholtz_factor", "radiation_brightness_factor",
    )
    out = dict(factors)
    for key in keys:
        vals = {sid: float((all_by_sample.get(sid) or {}).get(key) or 1.0) for sid in all_by_sample}
        if not vals:
            continue
        vmin, vmax = min(vals.values()), max(vals.values())
        if vmax - vmin < 1e-6:
            continue
        v = float(out.get(key) or 1.0)
        rank = (v - vmin) / (vmax - vmin)
        half = 0.22 * strength
        out[key] = round(_clamp(1.0 + (rank - 0.5) * 2.0 * half, 0.65, 1.40), 6)
    return out


def _v4_note_support(
    sample_id: str,
    note_name: str,
    factors: Mapping[str, float],
    physical: Mapping[str, Any],
) -> Dict[str, float]:
    base = dict(NOTE_BODY_SUPPORT.get(note_name, NOTE_BODY_SUPPORT["A2"]))
    extra = (PER_SAMPLE_NOTE_SUPPORT.get(note_name) or {}).get(sample_id) or {}
    base.update(extra)
    ref_vol = 0.013
    ref_depth = 0.10
    vol = float(physical.get("body_volume_proxy") or ref_vol)
    depth = float(physical.get("body_depth_m") or ref_depth)
    body_boost = _clamp(0.92 + 0.22 * (vol / ref_vol) + 0.12 * (depth / ref_depth), 0.85, 1.28)
    base["low_mid_mode_mult"] = float(base.get("low_mid_mode_mult") or 1.0) * body_boost
    warmth = float(factors.get("back_density_warmth_factor") or 1.0)
    if note_name in ("A4", "E5"):
        base["low_mid_mode_mult"] *= _clamp(1.02 + 0.08 * warmth, 1.0, 1.18)
    return base


def _apply_v4_continuous_mode_modifiers(
    modes: List[Dict[str, Any]],
    *,
    note_name: str,
    factors: Mapping[str, float],
    physical: Mapping[str, Any],
    body_modal_gain: float,
) -> List[Dict[str, Any]]:
    depth_ratio = float(physical.get("body_depth_m") or 0.10) / 0.10
    vol_ratio = float(physical.get("body_volume_proxy") or 0.013) / 0.013
    damp = float(factors.get("top_damping_factor") or 1.0)
    warmth = float(factors.get("back_density_warmth_factor") or 1.0)
    hole = float(factors.get("soundhole_radiation_factor") or 1.0)
    bright = float(factors.get("radiation_brightness_factor") or 1.0)
    note_tau = {"A2": 1.05, "A4": 1.0, "E5": 0.92}.get(note_name, 1.0)
    out: List[Dict[str, Any]] = []
    for row in modes:
        m = dict(row)
        f_hz = float(m["frequency_hz"])
        gain = float(m.get("gain") or 0.0) * body_modal_gain
        tau = float(m.get("tau_or_q") or 0.08)
        component = str(m.get("component") or "top")
        if 120.0 <= f_hz <= 450.0:
            gain *= _clamp(0.92 + 0.18 * vol_ratio + 0.12 * depth_ratio, 0.85, 1.35)
            tau *= _clamp(note_tau * (0.95 + 0.10 * warmth) / max(damp, 0.5), 0.75, 1.45)
        elif f_hz < 120.0:
            tau *= _clamp((0.98 + 0.14 * depth_ratio) / max(damp, 0.5), 0.80, 1.50)
        else:
            tau *= _clamp(note_tau / max(damp, 0.5), 0.70, 1.25)
        if component == "air":
            gain *= _clamp(0.90 + 0.22 * hole, 0.80, 1.35)
        if component in ("top", "radiation"):
            gain *= _clamp(0.88 + 0.20 * bright, 0.78, 1.30)
        m["gain"] = round(gain, 6)
        m["tau_or_q"] = round(max(tau, 0.015), 6)
        if m.get("q"):
            m["q"] = round(max(4.0, math.pi * f_hz * m["tau_or_q"]), 4)
        out.append(m)
    return out


def _note_excitation_scale(note_name: str, factors: Mapping[str, float]) -> float:
    base = {"A2": 1.00, "A4": 0.98, "E5": 0.94}.get(note_name, 1.0)
    mob = float(factors.get("bridge_mobility_factor") or 1.0)
    return round(_clamp(base * (0.92 + 0.14 * mob), 0.75, 1.15), 6)


def build_physical_factor_spread_table(
    renders_a2: Sequence[Mapping[str, Any]],
    *,
    sample_set: Sequence[str],
) -> Dict[str, Any]:
    table: Dict[str, Any] = {}
    for spec in STK_FACTOR_SPECS:
        fid = str(spec["id"])
        vals: Dict[str, float] = {}
        for row in renders_a2:
            sid = str(row["sample_id"])
            if sid not in sample_set:
                continue
            v = _render_factor_scalar(row, fid)
            if v is not None:
                vals[sid] = float(v)
        if not vals:
            continue
        spread = _normalized_spread(vals)
        table[fid] = {
            "values_by_sample": vals,
            "min": min(vals.values()),
            "max": max(vals.values()),
            "range": max(vals.values()) - min(vals.values()),
            "normalized_spread": round(spread, 6),
            "factor_spread_status": "meaningful" if spread >= MEANINGFUL_SPREAD_THRESHOLD else "too_small_from_source",
        }
    return table


def _summary_modal_bank_per_sample(renders_a2: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for row in renders_a2:
        sid = str(row["sample_id"])
        modes = (row.get("body_model") or {}).get("modes") or []
        out[sid] = {
            "frequencies_hz": [m.get("frequency_hz") for m in modes],
            "gains": [m.get("gain") for m in modes],
            "tau_or_q": [m.get("tau_or_q") for m in modes],
            "components": [m.get("component") for m in modes],
        }
    return out


def _summary_soundhole_radiation(renders_a2: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        str(row["sample_id"]): {
            "soundhole_area_proxy": (row.get("body_model") or {}).get("soundhole_area_proxy"),
            "soundhole_radiation_factor": (row.get("body_model") or {}).get("soundhole_radiation_factor"),
            "air_weight": (row.get("radiation_model") or {}).get("air_weight"),
        }
        for row in renders_a2
    }


def _summary_material_damping(renders_a2: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        str(row["sample_id"]): {
            "top_damping": (row.get("material_model") or {}).get("top_damping"),
            "material_loss_factor": (row.get("material_model") or {}).get("material_loss_factor"),
            "back_warmth": (row.get("material_model") or {}).get("back_warmth"),
        }
        for row in renders_a2
    }


def _summary_body_depth_volume(renders_a2: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        str(row["sample_id"]): {
            "body_depth_m": (row.get("body_model") or {}).get("body_depth_m"),
            "depth_factor": (row.get("body_model") or {}).get("depth_factor"),
            "body_volume_proxy": (row.get("body_model") or {}).get("body_volume_proxy"),
            "body_size_cavity_factor": (row.get("body_model") or {}).get("body_size_cavity_factor"),
        }
        for row in renders_a2
    }


def _per_sample_applied_mix_summary(renders: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for row in renders:
        if row.get("note_name") != "A2":
            continue
        sid = str(row["sample_id"])
        mix = row.get("string_body_mix") or {}
        out[sid] = {
            "direct_string_gain": mix.get("direct_string_gain"),
            "body_modal_gain": mix.get("body_modal_gain"),
            "string_to_body_send": (row.get("bridge_model") or {}).get("string_to_body_send"),
            "string_direct_weight": mix.get("string_direct"),
            "body_modal_share": mix.get("body_modal"),
        }
    return out


def build_physical_difference_audit(
    renders_a2: Sequence[Mapping[str, Any]],
    *,
    per_sample_physical: Mapping[str, Mapping[str, Any]],
    diagnostic_exaggeration: bool,
    sample_set: Sequence[str],
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
                for sid in sample_set
            }
        else:
            vals = {sid: float(per_sample[sid].get(key) or 0.0) for sid in sample_set}
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
        {sid: float((per_sample[sid].get("modal_frequency_hz") or [0.0])[0] or 0.0) for sid in sample_set}
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
    voicing_table = _extended_voicing((sample_id,))

    factors, _ = compute_v5_physical_factors(
        physical, reference_physical, sample_id=sample_id, voicing=voicing_table
    )
    if factor_multipliers:
        lo, hi = (0.65, 1.40) if demo_version in ("v3", "v4_10_samples") else (0.70, 1.35)
        for key, mult in factor_multipliers.items():
            if key in factors:
                factors[key] = round(_clamp(float(factors[key]) * float(mult), lo, hi), 6)

    voicing = V11_VOICING.get(sample_id) or V11_VOICING["sample_000"]
    pluck = _clamp(FIXED_PLUCK_POSITION + float(voicing.get("pluck_delta") or 0.0), 0.10, 0.20)
    modes_raw = _pick_reference_modes(readonly_modes, factors)
    bridge = _bridge_transfer_summary(factors)
    mix = dict(voicing.get("mix") or {})
    pluck_position_ratio = round(pluck, 5)
    if demo_version == "v4_10_samples":
        note_support = _v4_note_support(sample_id, note_name, factors, physical)
    else:
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
    elif demo_version == "v4_10_samples":
        radiation["string_direct_weight"] = round(
            _clamp(radiation["string_direct_weight"] * direct_gain, 0.12, 0.62), 6
        )
        string_to_body = round(_clamp(string_to_body * send_scale, 0.30, 0.98), 6)
        modes = _apply_v4_continuous_mode_modifiers(
            modes,
            note_name=note_name,
            factors=factors,
            physical=physical,
            body_modal_gain=body_modal_gain,
        )
        body_share = float(mix.get("body_modal") or 0.65) * body_modal_gain
        bright = float(factors.get("radiation_brightness_factor") or 1.0)
        warmth = float(factors.get("back_density_warmth_factor") or 1.0)
        hole = float(factors.get("soundhole_radiation_factor") or 1.0)
        radiation["top_weight"] = round(radiation["top_weight"] * _clamp(0.90 + 0.18 * bright, 0.78, 1.28), 6)
        radiation["back_weight"] = round(radiation["back_weight"] * _clamp(0.88 + 0.20 * warmth, 0.78, 1.32), 6)
        radiation["air_weight"] = round(radiation["air_weight"] * _clamp(0.85 + 0.22 * hole, 0.72, 1.35), 6)
        radiation["high_frequency_radiation_rolloff"] = round(
            _clamp(1.0 / max(bright, 0.55), 0.62, 1.38), 6
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
    elif demo_version == "v4_10_samples":
        perceptual_calibration = {
            "diagnostic_exaggeration_for_audible_demo": bool(factor_multipliers),
            "profile": f"lhs_continuous_{sample_id}",
            "direct_string_gain": direct_gain,
            "body_modal_gain": body_modal_gain,
            "string_to_body_send_scale": send_scale,
            "mapping_strength": 1.35,
            "continuous_physical_mix": True,
        }

    excitation = round(
        _clamp(0.82 + 0.10 * float(factors.get("bridge_mobility_factor") or 1.0), 0.65, 1.15), 6
    )
    note_excitation = _note_excitation_scale(note_name, factors)

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
            "excitation_strength": excitation,
            "note_excitation_scale": note_excitation,
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
    sample_set = sample_set_for_demo(demo_version)
    reference_physical = load_physical_parameters(REFERENCE_SAMPLE_ID, audit=audit)
    voicing_table = _extended_voicing(sample_set)
    per_sample_physical: Dict[str, Dict[str, Any]] = {
        sid: load_physical_parameters(sid, audit=audit) for sid in sample_set
    }

    preview_factors: Dict[str, Dict[str, float]] = {}
    for sid in sample_set:
        fac, _ = compute_v5_physical_factors(
            per_sample_physical[sid], reference_physical, sample_id=sid, voicing=voicing_table
        )
        preview_factors[sid] = {k: float(fac.get(k) or 1.0) for k in PHYSICAL_FACTOR_KEYS}

    preview_spread = {
        k: {sid: preview_factors[sid].get(k, 1.0) for sid in sample_set} for k in PHYSICAL_FACTOR_KEYS
    }
    diagnostic = demo_version == "v3" or (
        demo_version == "v4_10_samples" and _needs_diagnostic_exaggeration(preview_spread)
    ) or (_needs_diagnostic_exaggeration(preview_spread) and demo_version == "v2")
    factor_table = V3_FACTOR_MULTIPLIERS if demo_version == "v3" else DIAGNOSTIC_MULTIPLIERS

    renders: List[Dict[str, Any]] = []
    per_sample_summary: Dict[str, Any] = {}
    for sample_id in sample_set:
        physical = per_sample_physical[sample_id]
        mults = factor_table.get(sample_id) if diagnostic and demo_version in ("v2", "v3") else None
        mix_scales = V3_MIX_SCALES.get(sample_id) if demo_version == "v3" else None
        if demo_version == "v4_10_samples":
            fac, _ = compute_v5_physical_factors(
                physical, reference_physical, sample_id=sample_id, voicing=voicing_table
            )
            if diagnostic:
                fac = _stretch_factors_preserving_rank(
                    {k: float(fac.get(k) or 1.0) for k in PHYSICAL_FACTOR_KEYS if k in fac},
                    preview_factors,
                    sample_id,
                )
            mix_scales = _compute_v4_continuous_mix(physical, reference_physical, fac)
        per_sample_summary[sample_id] = {
            "profile": SAMPLE_PROFILES.get(sample_id, f"lhs_{sample_id}"),
            "physical_source": "audit_json" if audit is not None else "audit_or_lhs_fallback",
            "body_depth_m": physical.get("body_depth_m"),
            "body_volume_proxy": physical.get("body_volume_proxy"),
            "soundhole_area_proxy": _soundhole_area_proxy(physical),
            "bridge_mobility_proxy": physical.get("bridge_mobility_proxy"),
            "diagnostic_multipliers_applied": bool(mults) or (diagnostic and demo_version == "v4_10_samples"),
            "v3_perceptual_calibration": demo_version == "v3",
            "v4_continuous_mix": demo_version == "v4_10_samples",
        }
        for note_name in NOTE_SET:
            factor_mults = None
            if mults:
                base_fac, _ = compute_v5_physical_factors(
                    physical, reference_physical, sample_id=sample_id, voicing=voicing_table
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
            elif demo_version == "v4_10_samples" and diagnostic:
                base_fac, _ = compute_v5_physical_factors(
                    physical, reference_physical, sample_id=sample_id, voicing=voicing_table
                )
                stretched = _stretch_factors_preserving_rank(
                    {k: float(base_fac.get(k) or 1.0) for k in PHYSICAL_FACTOR_KEYS if k in base_fac},
                    preview_factors,
                    sample_id,
                )
                factor_mults = {
                    k: stretched[k] / max(float(base_fac.get(k) or 1.0), 1e-9)
                    for k in stretched
                    if k in base_fac
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
        sample_set=sample_set,
    )
    physical_factor_spread_table = build_physical_factor_spread_table(renders_a2, sample_set=sample_set)
    stk_factor_activation_matrix = build_stk_factor_activation_matrix(
        renders_a2,
        sample_set=sample_set,
        factor_spread_table=physical_factor_spread_table,
    )
    missing_or_weak_factor_summary = build_missing_or_weak_factor_summary(stk_factor_activation_matrix)

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
        "sample_set": list(sample_set),
        "note_set": list(NOTE_SET),
        "sample_rate": int(sample_rate),
        "duration_s": float(duration_s),
        "physical_factor_keys": list(PHYSICAL_FACTOR_KEYS),
        "physical_difference_audit": physical_difference_audit,
        "physical_factor_spread_table": physical_factor_spread_table,
        "stk_factor_activation_matrix": stk_factor_activation_matrix,
        "missing_or_weak_factor_summary": missing_or_weak_factor_summary,
        "per_sample_physical_summary": {
            sid: per_sample_summary[sid] for sid in sample_set
        },
        "per_sample_applied_mix_summary": _per_sample_applied_mix_summary(renders),
        "modal_bank_summary_per_sample": _summary_modal_bank_per_sample(renders_a2),
        "soundhole_radiation_summary": _summary_soundhole_radiation(renders_a2),
        "material_damping_summary": _summary_material_damping(renders_a2),
        "body_depth_volume_summary": _summary_body_depth_volume(renders_a2),
        "per_sample_summary": per_sample_summary,
        "per_sample_differences": _per_sample_difference_summary(renders),
        "renders": renders,
        "expected_render_count": len(sample_set) * len(NOTE_SET),
        "expected_wav_files": [expected_wav_filename(s, n) for s in sample_set for n in NOTE_SET],
        "limitations": [
            "Python exports parameters only; WAV synthesis is C++/STK on VM.",
            "Modal catalog is read-only PGSM reference — not live FEM/ROM at export time.",
            "Body response in STK must be driven by bridge force, not an independent pluck.",
            "v2: peak ceiling only in C++; RMS not forced equal across samples.",
            "v3: perceptual calibration for clearer sample differentiation; diagnostic_exaggeration_for_audible_demo.",
            "v4_10_samples: 10 LHS samples with continuous physical mix modifiers; diagnostic scaling only if spread too small.",
        ],
        "known_limitations": [
            "Modal bank uses embedded PGSM reference templates with per-sample physical shifts.",
            "STK Plucked string is a simplified excitation model vs full PGSM string FEM.",
            "Peak ceiling normalization may mask level differences between samples.",
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
    if demo_version == "v4_10_samples":
        doc["perceptual_calibration_policy"] = {
            "diagnostic_exaggeration_for_audible_demo": diagnostic,
            "base": "lhs_continuous_physical_ranking",
            "sample_count": len(sample_set),
            "continuous_mix_from_physical_proxies": True,
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
        help="Demo pack version (v2 audit; v3 stronger perceptual differentiation; v4_10_samples 10 LHS samples)",
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
    sample_set = sample_set_for_demo(args.demo_version)
    print(f"Renders: {len(sample_set) * len(NOTE_SET)} (no audio generated)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
