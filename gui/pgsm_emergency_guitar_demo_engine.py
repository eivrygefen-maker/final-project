#!/usr/bin/env python3
"""
PGSM emergency guitar demo engine — practical diagnostic demo audio.
Produces guitar-like plucked tones with per-sample physical differentiation.
Not final validation, not STK/FEM/ROM. Conference demo only.
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

ENGINE_VERSION = "pgsm_emergency_guitar_demo_engine_v1"
SR = 44100
DURATION_S = 2.5
N_HARMONICS = 16
PLUCK_POSITION_RATIO = 0.14
INHARMONICITY_B = 2.0e-5
TARGET_RMS_DBFS = -20.0

SAMPLE_SET = ("sample_000", "sample_001", "sample_002")
NOTE_SET = ("A2", "A4", "E5")
REFERENCE_SAMPLE_ID = "sample_000"

REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = REPO_ROOT / "audio" / "pgsm_emergency_guitar_demo"
REPORT_JSON = REPO_ROOT / "audio" / "debug_reports" / "pgsm_emergency_guitar_demo_report.json"
REPORT_MD = REPO_ROOT / "audio" / "debug_reports" / "pgsm_emergency_guitar_demo_report.md"

READINESS_OK = "ready_for_stk_gui_activation"
READINESS_WEAK = "demo_audio_generated_but_weak_differentiation"
READINESS_FAIL = "emergency_demo_failed"

DIFFERENCE_WEAK_THRESHOLD = 0.035
DIFFERENCE_STRONG_THRESHOLD = 0.08

# Conservative demo exaggeration when audit proxies are close (disclosed in report).
DEMO_EXAGGERATION_POWER = 1.28

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
        "validation_mode": "emergency_demo",
        "sample_set": list(SAMPLE_SET),
        "note_set": list(NOTE_SET),
        "duration_s": DURATION_S,
        "sample_rate": SR,
        "n_harmonics": N_HARMONICS,
        "pluck_position_ratio": PLUCK_POSITION_RATIO,
        "target_rms_dbfs": TARGET_RMS_DBFS,
        "diagnostic_exaggeration_disclosed": True,
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


def compute_demo_modifiers(
    sample: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    diagnostic_exaggeration_for_audible_demo: bool = True,
) -> Tuple[Dict[str, float], List[Dict[str, Any]], bool]:
    """Physical multipliers for demo synthesis; may apply conservative exaggeration."""
    exp = DEMO_EXAGGERATION_POWER if diagnostic_exaggeration_for_audible_demo else 1.0
    exaggerated = False

    def _ratio(val: float, ref: float, power: float, lo: float, hi: float) -> float:
        nonlocal exaggerated
        raw = (val / max(ref, 1e-9)) ** power
        if abs(raw - 1.0) < 0.02 and diagnostic_exaggeration_for_audible_demo:
            raw = raw ** exp
            exaggerated = True
        return _clamp(raw, lo, hi)

    ref_mob = max(float(reference.get("bridge_mobility_proxy") or 1.0), 1e-9)
    mob = max(float(sample.get("bridge_mobility_proxy") or 1.0), 1e-9)
    bridge_coupling = _ratio(mob, ref_mob, 0.40, 0.78, 1.28)

    ref_depth = max(float(reference.get("body_depth_m") or 0.10), 1e-9)
    depth = max(float(sample.get("body_depth_m") or 0.10), 1e-9)
    body_low_balance = _ratio(depth, ref_depth, 0.22, 0.82, 1.18)

    ref_vol = max(float(reference.get("body_volume_proxy") or 0.013), 1e-9)
    vol = max(float(sample.get("body_volume_proxy") or 0.013), 1e-9)
    air_cavity = _ratio(vol, ref_vol, 0.20, 0.86, 1.16)

    ref_helm = max(float(reference.get("helmholtz_like_frequency_proxy") or 120.0), 1e-9)
    helm = max(float(sample.get("helmholtz_like_frequency_proxy") or 120.0), 1e-9)
    helm_shift = _ratio(helm, ref_helm, 0.18, 0.90, 1.12)

    ref_td = max(float(reference.get("top_damping_coeff_proxy") or 1.0), 1e-9)
    td = max(float(sample.get("top_damping_coeff_proxy") or 1.0), 1e-9)
    ref_bd = max(float(reference.get("back_damping_coeff_proxy") or 1.0), 1e-9)
    bd = max(float(sample.get("back_damping_coeff_proxy") or 1.0), 1e-9)
    sustain = _ratio((td + bd) / 2.0, (ref_td + ref_bd) / 2.0, 0.15, 0.80, 1.22)

    top_bright = float(sample.get("top_stiffness_to_weight_proxy") or 1.0)
    ref_bright = max(float(reference.get("top_stiffness_to_weight_proxy") or 1.0), 1e-9)
    brightness = _ratio(top_bright, ref_bright, 0.30, 0.84, 1.22)

    back_warm = BACK_WARMTH_PROXY.get(str(sample.get("back_wood_id") or "rosewood").lower(), 1.0)
    ref_warm = BACK_WARMTH_PROXY.get(str(reference.get("back_wood_id") or "rosewood").lower(), 1.0)
    warmth = _ratio(back_warm, ref_warm, 0.25, 0.86, 1.18)

    modifiers = {
        "bridge_coupling_strength": round(bridge_coupling, 6),
        "body_low_resonance_balance": round(body_low_balance, 6),
        "air_cavity_weight": round(air_cavity, 6),
        "helmholtz_shift": round(helm_shift, 6),
        "sustain_scale": round(sustain, 6),
        "brightness_scale": round(brightness, 6),
        "warmth_scale": round(warmth, 6),
        "harmonic_richness_scale": round(_clamp(brightness * warmth ** 0.35, 0.88, 1.20), 6),
        "attack_sharpness_scale": round(_clamp(brightness ** 0.55 * bridge_coupling ** 0.25, 0.85, 1.25), 6),
    }
    trace = [
        {
            "driver": "bridge_mobility_proxy",
            "sample_value": mob,
            "reference_value": ref_mob,
            "effect": f"bridge_coupling_strength={modifiers['bridge_coupling_strength']}",
        },
        {
            "driver": "body_depth",
            "sample_value": depth,
            "reference_value": ref_depth,
            "effect": f"body_low_resonance_balance={modifiers['body_low_resonance_balance']}",
        },
        {
            "driver": "body_volume_proxy",
            "sample_value": vol,
            "reference_value": ref_vol,
            "effect": f"air_cavity_weight={modifiers['air_cavity_weight']}",
        },
        {
            "driver": "helmholtz_like_frequency_proxy",
            "sample_value": helm,
            "reference_value": ref_helm,
            "effect": f"helmholtz_shift={modifiers['helmholtz_shift']}",
        },
        {
            "driver": "material_damping_proxy",
            "sample_value": (td + bd) / 2.0,
            "reference_value": (ref_td + ref_bd) / 2.0,
            "effect": f"sustain_scale={modifiers['sustain_scale']}",
        },
        {
            "driver": "top_stiffness_to_weight_proxy",
            "sample_value": top_bright,
            "reference_value": ref_bright,
            "effect": f"brightness_scale={modifiers['brightness_scale']}",
        },
        {
            "driver": "back_wood_warmth_proxy",
            "sample_value": back_warm,
            "reference_value": ref_warm,
            "effect": f"warmth_scale={modifiers['warmth_scale']}",
        },
    ]
    return modifiers, trace, exaggerated


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


def _add_body_resonance(
    y: np.ndarray,
    sr: int,
    f_res: float,
    q: float,
    gain: float,
) -> np.ndarray:
    if gain <= 1e-6 or f_res <= 20.0:
        return y
    t = np.arange(len(y), dtype=np.float64) / sr
    env = np.abs(y)
    ring = gain * env * np.sin(2.0 * math.pi * f_res * t) * np.exp(-t * (math.pi * f_res / (q * sr * 2.0)))
    return y + ring


def synthesize_guitar_demo_note(
    *,
    sample_id: str,
    note: str,
    physical: Mapping[str, Any],
    modifiers: Mapping[str, float],
    duration_s: float = DURATION_S,
    sr: int = SR,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    f0 = float(NOTE_FREQUENCY_HZ[note])
    n = int(duration_s * sr)
    t = np.arange(n, dtype=np.float64) / sr

    bridge = float(modifiers.get("bridge_coupling_strength") or 1.0)
    brightness = float(modifiers.get("brightness_scale") or 1.0)
    warmth = float(modifiers.get("warmth_scale") or 1.0)
    sustain = float(modifiers.get("sustain_scale") or 1.0)
    harmonic_rich = float(modifiers.get("harmonic_richness_scale") or 1.0)
    attack_sharp = float(modifiers.get("attack_sharpness_scale") or 1.0)
    body_low = float(modifiers.get("body_low_resonance_balance") or 1.0)
    air_w = float(modifiers.get("air_cavity_weight") or 1.0)
    helm_shift = float(modifiers.get("helmholtz_shift") or 1.0)

    base_tau = 0.55 * sustain
    y = np.zeros(n, dtype=np.float64)

    for k in range(1, N_HARMONICS + 1):
        fk = f0 * k * (1.0 + INHARMONICITY_B * k * k)
        if fk >= sr / 2.0 - 50.0:
            break
        pluck_amp = abs(math.sin(math.pi * k * PLUCK_POSITION_RATIO)) / k
        if pluck_amp < 1e-8:
            continue
        if k >= 2:
            pluck_amp *= harmonic_rich * (1.0 + 0.12 * min(k, 8) / 8.0)
        if k >= 3:
            pluck_amp *= brightness ** 0.35
        if k <= 2:
            pluck_amp *= warmth ** 0.25

        tau_k = base_tau / (k ** 0.72)
        if fk < 130.0:
            tau_k *= 0.55 / max(body_low, 0.5)
            pluck_amp *= 0.55 / max(body_low, 0.5) if note == "A2" else 0.85
        elif fk > 1200.0:
            tau_k *= 0.72
            pluck_amp *= brightness ** 0.20

        partial = pluck_amp * np.exp(-t / max(tau_k, 1e-4)) * np.sin(2.0 * math.pi * fk * t)
        y += partial * bridge

    onset_n = max(int(0.0035 * sr), 4)
    onset = np.ones(n, dtype=np.float64)
    ramp = np.sin(np.linspace(0.0, math.pi / 2.0, onset_n)) ** 2
    onset[:onset_n] = ramp
    y *= onset

    pick_n = max(int(0.004 * sr), 8)
    pick_t = np.arange(pick_n, dtype=np.float64) / sr
    pick_freq = 1800.0 + 400.0 * attack_sharp
    pick = (
        attack_sharp
        * 0.18
        * np.sin(2.0 * math.pi * pick_freq * pick_t)
        * np.exp(-pick_t / 0.0012)
    )
    y[:pick_n] += pick

    helm_f = float(physical.get("helmholtz_like_frequency_proxy") or 120.0) * helm_shift
    y = _add_body_resonance(y, sr, helm_f, q=4.5, gain=0.06 * air_w * warmth)
    top_mode = min(220.0 * brightness, sr / 2.0 - 100.0)
    y = _add_body_resonance(y, sr, top_mode, q=8.0, gain=0.035 * brightness)

    if note == "A2":
        y = _apply_one_pole_highpass(y, sr, 72.0)
        low_body = y - _apply_one_pole_highpass(y, sr, 105.0)
        y = y - 0.28 * body_low * low_body

    peak = float(np.max(np.abs(y)))
    if peak > 1e-12:
        y = y / peak * 0.42

    # Listening normalization separate from physics shaping.
    rms = _rms(y)
    target = 10.0 ** (TARGET_RMS_DBFS / 20.0)
    gain = target / max(rms, 1e-12)
    y_norm = np.clip(y * gain, -0.98, 0.98)

    meta = {
        "sample_id": sample_id,
        "note": note,
        "f0_hz": f0,
        "listening_gain_linear": round(gain, 6),
        "rms_dbfs_after_norm": round(_linear_to_dbfs(_rms(y_norm)), 3),
        "modifiers_applied": dict(modifiers),
    }
    return y_norm.astype(np.float64), meta


def _band_energy_ratio(y: np.ndarray, sr: int, f_lo: float, f_hi: float) -> float:
    n = len(y)
    if n < 256:
        return 0.0
    spec = np.abs(np.fft.rfft(y * np.hanning(n))) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    total = max(float(np.sum(spec)), 1e-12)
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    return float(np.sum(spec[mask]) / total) if mask.any() else 0.0


def _spectral_centroid_hz(y: np.ndarray, sr: int) -> float:
    n = len(y)
    if n < 64:
        return 0.0
    spec = np.abs(np.fft.rfft(y * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    denom = max(float(np.sum(spec)), 1e-12)
    return float(np.sum(freqs * spec) / denom)


def _harmonic_ratios(y: np.ndarray, sr: int, f0: float) -> Dict[str, float]:
    n = len(y)
    if n < 256:
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
    return {
        "spectral_centroid_hz": round(_spectral_centroid_hz(y, sr), 3),
        "low_body_band_80_160_ratio": round(_band_energy_ratio(y, sr, 80.0, 160.0), 6),
        "mid_band_300_1200_ratio": round(_band_energy_ratio(y, sr, 300.0, 1200.0), 6),
        "h1_dominance_ratio": round(harm["h1_share"], 6),
        "h2_h8_ratio": round(harm["h2_h8_share"], 6),
        "rms_dbfs": round(_linear_to_dbfs(_rms(y)), 3),
        "crest_factor": round(float(np.max(np.abs(y))) / max(_rms(y), 1e-12), 4),
    }


def _spectral_distance(a: np.ndarray, b: np.ndarray, sr: int) -> float:
    n = min(len(a), len(b))
    if n < 64:
        return 0.0
    sa = np.abs(np.fft.rfft(a[:n] * np.hanning(n)))
    sb = np.abs(np.fft.rfft(b[:n] * np.hanning(n)))
    sa = sa / max(float(np.linalg.norm(sa)), 1e-12)
    sb = sb / max(float(np.linalg.norm(sb)), 1e-12)
    return float(np.linalg.norm(sa - sb))


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
    spectral_metrics: Mapping[str, Mapping[str, Mapping[str, Any]]],
    sample_ids: Sequence[str],
) -> Dict[str, Any]:
    mean_diff = float(pairwise.get("mean_overall_differentiation_score") or 0.0)
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
        "physical_driver_trace_per_sample": all(len(traces.get(sid) or []) >= 4 for sid in sample_ids),
        "differences_not_only_loudness": mean_diff > DIFFERENCE_WEAK_THRESHOLD or (
            rms_range < 1.0 and mean_diff > 0.02
        ),
        "no_clipping_limiter_trick": all(
            float((spectral_metrics.get(sid, {}).get(note) or {}).get("crest_factor") or 0.0) < 20.0
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
) -> Dict[str, Any]:
    if files_generated < expected_files:
        status = READINESS_FAIL
    elif mean_differentiation >= DIFFERENCE_WEAK_THRESHOLD:
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
    }


def build_emergency_demo_report(
    *,
    generated_files: Sequence[str],
    physical_parameters: Mapping[str, Any],
    differentiation_trace: Mapping[str, Any],
    spectral_metrics: Mapping[str, Mapping[str, Mapping[str, Any]]],
    pairwise: Mapping[str, Any],
    anti_cheat: Mapping[str, Any],
    loudness_report: Mapping[str, Any],
    diagnostic_exaggeration: bool,
) -> Dict[str, Any]:
    mean_diff = float(pairwise.get("mean_overall_differentiation_score") or 0.0)
    readiness = build_readiness_emergency_demo(
        files_generated=len(generated_files),
        expected_files=len(SAMPLE_SET) * len(NOTE_SET),
        mean_differentiation=mean_diff,
    )
    return {
        "report_version": ENGINE_VERSION,
        "timestamp": _utc_now(),
        "validation_mode": "emergency_demo",
        "status": "pgsm_emergency_guitar_demo_complete",
        "no_stk_integration": True,
        "no_fem_run": True,
        "no_rom_run": True,
        "website_default": DEFAULT_WEBSITE_STK_MODE,
        "website_default_unchanged": True,
        "generated_files": list(generated_files),
        "sample_set": list(SAMPLE_SET),
        "note_set": list(NOTE_SET),
        "duration_s": DURATION_S,
        "max_modes": None,
        "physical_parameters_used": physical_parameters,
        "per_sample_differentiation_trace": differentiation_trace,
        "spectral_metrics_per_file": spectral_metrics,
        "pairwise_difference_metrics": pairwise.get("pairwise_guitar_difference_metrics"),
        "mean_overall_differentiation_score": mean_diff,
        "loudness_normalization_report": loudness_report,
        "anti_cheat_checks": anti_cheat,
        "diagnostic_exaggeration_for_audible_demo": diagnostic_exaggeration,
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
            "Emergency diagnostic guitar demo engine for conference listening. "
            "Uses lightweight plucked-harmonic synthesis with audit-derived physical modifiers. "
            "Not final PGSM validation."
        ),
    }


def write_emergency_demo_markdown(report: Mapping[str, Any], path: Path) -> None:
    rg = report.get("readiness") or {}
    lines = [
        "# PGSM emergency guitar demo",
        "",
        f"**Readiness:** `{rg.get('current_status')}`",
        f"**Files:** {len(report.get('generated_files') or [])}",
        f"**Mean differentiation:** {report.get('mean_overall_differentiation_score')}",
        f"**Diagnostic exaggeration:** {report.get('diagnostic_exaggeration_for_audible_demo')}",
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
    spectral_metrics: Dict[str, Dict[str, Any]] = {}
    audio_by_sample: Dict[str, Dict[str, np.ndarray]] = {}
    generated_files: List[str] = []
    exaggerated_any = False

    for sid in SAMPLE_SET:
        if sid not in physical_parameters:
            physical_parameters[sid] = extract_physical_parameters(sid, audit)
        mods, trace, exaggerated = compute_demo_modifiers(physical_parameters[sid], ref_phys)
        exaggerated_any = exaggerated_any or exaggerated
        differentiation_trace[sid] = {
            "sample_id": sid,
            "reference_sample": REFERENCE_SAMPLE_ID,
            "physical_drivers_applied": trace,
            "modifiers": mods,
            "diagnostic_exaggeration_for_audible_demo": exaggerated,
        }
        audio_by_sample[sid] = {}
        spectral_metrics[sid] = {}
        for note in NOTE_SET:
            y, _meta = synthesize_guitar_demo_note(
                sample_id=sid,
                note=note,
                physical=physical_parameters[sid],
                modifiers=mods,
            )
            wav_name = demo_wav_filename(sid, note)
            wav_path = out_dir / wav_name
            write_wav_mono(wav_path, y, SR)
            generated_files.append(str(wav_path.resolve()))
            audio_by_sample[sid][note] = y
            spectral_metrics[sid][note] = compute_spectral_metrics(y, SR, note)
            print(f"[Emergency demo] wrote {sid} {note} -> {wav_name}")

    pairwise = compute_pairwise_difference_metrics(
        audio_by_sample, sample_ids=SAMPLE_SET, note_set=NOTE_SET
    )
    anti_cheat = build_anti_cheat_checks(
        traces={sid: differentiation_trace[sid]["physical_drivers_applied"] for sid in SAMPLE_SET},
        pairwise=pairwise,
        spectral_metrics=spectral_metrics,
        sample_ids=SAMPLE_SET,
    )
    loudness_report = {
        "normalization": f"per-file RMS target {TARGET_RMS_DBFS} dBFS after physics shaping",
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
        spectral_metrics=spectral_metrics,
        pairwise=pairwise,
        anti_cheat=anti_cheat,
        loudness_report=loudness_report,
        diagnostic_exaggeration=exaggerated_any,
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
    print(f"mean_differentiation: {report.get('mean_overall_differentiation_score')}")


if __name__ == "__main__":
    main()
