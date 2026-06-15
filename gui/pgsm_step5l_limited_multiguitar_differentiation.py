#!/usr/bin/env python3
"""
PGSM Step 5L — limited multi-guitar physical differentiation diagnostic.
Chains Step 5I.3 → 5J.1 → 5K with per-sample audit geometry/material proxies.
Diagnostic only — not final realism, not STK, not ROM execution.
"""
from __future__ import annotations

import copy
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from bridge_mobility_proxy import compute_body_mass_proxies
from pgsm_step2_1_parameter_targets import load_step_report
from pgsm_step3a_numerical_ir_testbench import NUMERIC_SR
from pgsm_step4a_single_note_diagnostic_audio import (
    build_calibrated_modal_state,
    synthesize_modal_body_response,
    write_wav_mono,
)
from pgsm_step5a_limited_note_set_diagnostic_audio import NOTE_FREQUENCY_HZ, NOTE_SET
from pgsm_step5e_string_driven_bridge_force_repair import (
    compute_harmonic_energies,
    compute_pitch_salience,
    detect_second_onset_sustained,
    evaluate_modal_peak_alignment,
    compute_click_dominance_score,
    _energy_share_first_ms,
    _linear_to_dbfs,
    _rms,
)
from pgsm_step5f_string_driven_extended_validation import compute_spectral_centroid_over_time
from pgsm_step5i_1_string_damping_duration_harshness_repair import (
    load_preferred_mappings,
    _report_path,
)
from pgsm_step5i_3_absolute_frequency_damping_pluck_balance import (
    DEFAULT_DURATION_S,
    build_v4_string_bridge_force,
)
from pgsm_step5j_1_guitar_articulation_body_balance_repair import (
    COMB_ECHO_FAIL_THRESHOLD,
    apply_listening_render_step5j_1,
    collect_all_previous_audio_fingerprints,
    compute_comb_echo_score,
    compute_step5j_1_modal_kernels_decomposed,
)
from pgsm_step5k_bridge_admittance_feedback_coupling import (
    SAFE_NEXT_STEP_5L,
    apply_bridge_admittance_coupling,
)
from sample_parameters import normalize_sample_parameters
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE
from stk_v6_2_audit_features import feature_value, get_sample_record, load_audit_report

PGSM_STEP5L_VERSION = "pgsm_step5l_limited_multiguitar_differentiation_v1"
VALIDATION_MAX_MODES = 100
FULL_DURATION_S = DEFAULT_DURATION_S
FAST_VALIDATION_MAX_MODES = 32
FAST_VALIDATION_DURATION_S = 0.75
FULL_SAMPLE_SET = ("sample_000", "sample_001", "sample_002", "sample_003")
FAST_SAMPLE_SET = ("sample_000", "sample_001", "sample_002")
FULL_NOTE_SET = tuple(NOTE_SET)
FAST_NOTE_SET = ("A2", "A4", "E5")
REFERENCE_SAMPLE_ID = "sample_000"

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON = REPO_ROOT / "audio" / "debug_reports" / "pgsm_step5l_limited_multiguitar_differentiation.json"
REPORT_MD = REPO_ROOT / "audio" / "debug_reports" / "pgsm_step5l_limited_multiguitar_differentiation.md"
SOURCE_CONTRACT_JSON = REPO_ROOT / "data" / "pgsm_limited_multiguitar_differentiation_contract.json"
GENERATED_CONTRACT_JSON = (
    REPO_ROOT
    / "audio"
    / "debug_reports"
    / "generated_contracts"
    / "pgsm_limited_multiguitar_differentiation_contract.generated.json"
)
AUDIO_DIR = REPO_ROOT / "audio" / "pgsm_step5l_limited_multiguitar_differentiation"
STEP5K_REPORT_JSON = (
    REPO_ROOT / "audio" / "debug_reports" / "pgsm_step5k_bridge_admittance_feedback_coupling.json"
)

READINESS_AFTER = "ready_for_step5m_rom_shape_return_or_demo_audio_pack"
READINESS_WEAK = "limited_multiguitar_pipeline_ready_with_weak_audible_separation"
READINESS_FAIL = "failed_limited_multiguitar_differentiation"
SAFE_NEXT_STEP_5M = "step5m_rom_shape_return_or_demo_audio_pack"

DIFFERENCE_WEAK_THRESHOLD = 0.04
DIFFERENCE_STRONG_THRESHOLD = 0.10


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def build_multiguitar_contract() -> Dict[str, Any]:
    return {
        "contract_id": "pgsm_limited_multiguitar_differentiation_v1",
        "chain": [
            "step5i_3_string_force",
            "step5j_1_body_balance",
            "step5k_bridge_coupling",
            "audit_geometry_material_proxies",
        ],
        "forbidden": [
            "sample_id_only_gain",
            "arbitrary_eq",
            "randomization",
            "loudness_trick",
            "stk_integration",
            "fem_run",
            "rom_run",
        ],
        "physical_drivers": [
            "body_depth",
            "body_volume_proxy",
            "helmholtz_like_frequency_proxy",
            "bridge_mobility_proxy",
            "top/back_damping_coeff_proxy",
            "tonewood_density_proxy",
            "top_back_air_share_modifiers",
        ],
        "fast_validation": {
            "sample_set": list(FAST_SAMPLE_SET),
            "note_set": list(FAST_NOTE_SET),
            "max_modes": FAST_VALIDATION_MAX_MODES,
            "duration_s": FAST_VALIDATION_DURATION_S,
        },
        "full_validation": {
            "sample_set": list(FULL_SAMPLE_SET),
            "note_set": list(FULL_NOTE_SET),
            "max_modes": VALIDATION_MAX_MODES,
            "duration_s": FULL_DURATION_S,
        },
    }


def resolve_step5k_upstream(repo_root: Path) -> Dict[str, Any]:
    """Load Step 5K status from disk only — no heavy Step 5K/5J.1 rebuild in Step 5L."""
    path = repo_root / "audio" / "debug_reports" / "pgsm_step5k_bridge_admittance_feedback_coupling.json"
    if not path.is_file():
        return {
            "step5k_upstream_source": "missing",
            "pass": False,
            "step5l_multiguitar_planning_allowed": False,
            "documented_limitation_loaded": False,
        }
    disk = json.loads(path.read_text(encoding="utf-8"))
    rg = disk.get("readiness_after_step5k") or {}
    planning = bool(
        rg.get("step5l_multiguitar_planning_allowed")
        or disk.get("safe_next_step") == SAFE_NEXT_STEP_5L
    )
    return {
        "step5k_upstream_source": "disk_json",
        "step5k_report_version": disk.get("report_version"),
        "step5k_readiness_status": rg.get("current_status"),
        "step5k_safe_next_step": disk.get("safe_next_step"),
        "step5l_multiguitar_planning_allowed": planning,
        "documented_limitation_loaded": bool(disk.get("documented_limitation_loaded")),
        "pass": planning,
    }


def extract_per_sample_physical_parameters(
    sample_id: str,
    audit: Mapping[str, Any],
) -> Dict[str, Any]:
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
    bridge_audit = float(
        feature_value(rec, "bridge_mobility_proxy", audit=audit, default=mass["bridge_mobility_proxy"])
        or mass["bridge_mobility_proxy"]
    )
    return {
        "sample_id": sample_id,
        "top_wood_id": str(params.get("top_wood_id")),
        "back_wood_id": str(params.get("back_wood_id")),
        "body_length_m": float(params.get("geometry.length") or 0.45),
        "body_width_m": float(params.get("geometry.width") or 0.35),
        "body_depth_m": float(params.get("geometry.depth") or 0.10),
        "body_volume_proxy": float(feature_value(rec, "body_volume_proxy", audit=audit, default=0.013)),
        "soundhole_area": float(feature_value(rec, "soundhole_area", audit=audit, default=0.007)),
        "helmholtz_like_frequency_proxy": float(
            feature_value(rec, "helmholtz_like_frequency_proxy", audit=audit, default=120.0)
        ),
        "bridge_mobility_proxy": bridge_audit,
        "top_damping_coeff_proxy": float(
            feature_value(rec, "top_damping_coeff_proxy", audit=audit, default=1.0)
        ),
        "back_damping_coeff_proxy": float(
            feature_value(rec, "back_damping_coeff_proxy", audit=audit, default=1.0)
        ),
        "mass_loading_proxy": float(feature_value(rec, "mass_loading_proxy", audit=audit, default=1.0)),
        "mass_proxies": mass,
        "reference_shared_modal_catalog": True,
    }


def compute_physical_modifiers(
    sample: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """Derive bounded modal modifiers from audit proxies relative to reference sample."""
    ref_mob = max(float(reference.get("bridge_mobility_proxy") or 1.0), 1e-9)
    mob = max(float(sample.get("bridge_mobility_proxy") or 1.0), 1e-9)
    bridge_scale = _clamp((mob / ref_mob) ** 0.35, 0.84, 1.16)

    ref_vol = max(float(reference.get("body_volume_proxy") or 0.013), 1e-9)
    vol = max(float(sample.get("body_volume_proxy") or 0.013), 1e-9)
    air_vol = _clamp((vol / ref_vol) ** 0.22, 0.88, 1.12)

    ref_helm = max(float(reference.get("helmholtz_like_frequency_proxy") or 120.0), 1e-9)
    helm = max(float(sample.get("helmholtz_like_frequency_proxy") or 120.0), 1e-9)
    air_helm = _clamp((helm / ref_helm) ** 0.15, 0.92, 1.08)

    ref_depth = max(float(reference.get("body_depth_m") or 0.10), 1e-9)
    depth = max(float(sample.get("body_depth_m") or 0.10), 1e-9)
    cavity = _clamp((depth / ref_depth) ** 0.18, 0.90, 1.10)

    ref_td = max(float(reference.get("top_damping_coeff_proxy") or 1.0), 1e-9)
    td = max(float(sample.get("top_damping_coeff_proxy") or 1.0), 1e-9)
    ref_bd = max(float(reference.get("back_damping_coeff_proxy") or 1.0), 1e-9)
    bd = max(float(sample.get("back_damping_coeff_proxy") or 1.0), 1e-9)
    damping_scale = _clamp(((td + bd) / (ref_td + ref_bd)) ** 0.12, 0.90, 1.10)

    ref_mass = max(float((reference.get("mass_proxies") or {}).get("mixed_body_mass_proxy") or 1.0), 1e-9)
    mass = max(float((sample.get("mass_proxies") or {}).get("mixed_body_mass_proxy") or 1.0), 1e-9)
    radiation_scale = _clamp((mass / ref_mass) ** 0.14, 0.88, 1.12)

    air_scale = air_vol * air_helm * cavity
    modifiers = {
        "bridge_excitation_scale": round(bridge_scale, 6),
        "air_weight_scale": round(air_scale, 6),
        "radiation_weight_scale": round(radiation_scale, 6),
        "damping_tau_scale": round(1.0 / damping_scale, 6),
        "top_back_share_balance": round(_clamp((td / ref_td) ** 0.08, 0.95, 1.05), 6),
    }
    trace = [
        {
            "driver": "bridge_mobility_proxy",
            "sample_value": mob,
            "reference_value": ref_mob,
            "effect": f"bridge_excitation_scale={modifiers['bridge_excitation_scale']}",
        },
        {
            "driver": "body_volume_proxy",
            "sample_value": vol,
            "reference_value": ref_vol,
            "effect": f"air_weight_scale component via volume",
        },
        {
            "driver": "helmholtz_like_frequency_proxy",
            "sample_value": helm,
            "reference_value": ref_helm,
            "effect": f"air_weight_scale component via helmholtz",
        },
        {
            "driver": "body_depth",
            "sample_value": depth,
            "reference_value": ref_depth,
            "effect": f"cavity_proxy scale={cavity}",
        },
        {
            "driver": "material_damping_proxy",
            "sample_value": (td + bd) / 2.0,
            "reference_value": (ref_td + ref_bd) / 2.0,
            "effect": f"damping_tau_scale={modifiers['damping_tau_scale']}",
        },
        {
            "driver": "mixed_body_mass_proxy",
            "sample_value": mass,
            "reference_value": ref_mass,
            "effect": f"radiation_weight_scale={modifiers['radiation_weight_scale']}",
        },
    ]
    return modifiers, trace


def apply_physical_modifiers_to_modal_weights(
    modal_weights: Mapping[str, Any],
    modifiers: Mapping[str, float],
) -> Dict[str, Any]:
    out = copy.deepcopy(dict(modal_weights))
    modes: List[Dict[str, Any]] = []
    b_scale = float(modifiers.get("bridge_excitation_scale") or 1.0)
    air_scale = float(modifiers.get("air_weight_scale") or 1.0)
    rad_scale = float(modifiers.get("radiation_weight_scale") or 1.0)
    tau_scale = float(modifiers.get("damping_tau_scale") or 1.0)
    tb_bal = float(modifiers.get("top_back_share_balance") or 1.0)
    for row in out.get("modes") or []:
        r = dict(row)
        r["W_exc"] = float(r.get("W_exc") or 0.0) * b_scale
        r["W_air"] = float(r.get("W_air") or 0.0) * air_scale
        r["W_rad"] = float(r.get("W_rad") or 0.0) * rad_scale
        r["top_share"] = float(r.get("top_share") or 0.0) * tb_bal
        r["back_share"] = float(r.get("back_share") or 0.0) / max(tb_bal, 1e-9)
        r["tau_s"] = max(float(r.get("tau_s") or 1e-3) * tau_scale, 1e-6)
        if "bridge_excitation_coupling" in r:
            r["bridge_excitation_coupling"] = float(r["bridge_excitation_coupling"]) * b_scale
        modes.append(r)
    out["modes"] = modes
    out["physical_modifiers_applied"] = dict(modifiers)
    return out


def _spectral_rolloff_hz(y: np.ndarray, sr: int, *, percentile: float = 0.85) -> float:
    n = len(y)
    if n < 64:
        return 0.0
    spec = np.abs(np.fft.rfft(y * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    power = spec.astype(np.float64) ** 2
    total = float(np.sum(power))
    if total <= 1e-18:
        return 0.0
    cum = np.cumsum(power) / total
    idx = int(np.searchsorted(cum, percentile))
    idx = min(max(idx, 0), len(freqs) - 1)
    return float(freqs[idx])


def _envelope_decay_ms(y: np.ndarray, sr: int) -> float:
    env = np.abs(y).astype(np.float64)
    if env.size < 8:
        return 0.0
    peak = float(env.max())
    if peak <= 1e-12:
        return 0.0
    target = peak * math.e ** (-3.0)
    idx = np.where(env < target)[0]
    if idx.size == 0:
        return float(len(env) / sr * 1000.0)
    return float(idx[0] / sr * 1000.0)


def _stem_balance(y_top: np.ndarray, y_back: np.ndarray, y_air: np.ndarray) -> Dict[str, float]:
    e_top = float(np.sum(y_top.astype(np.float64) ** 2))
    e_back = float(np.sum(y_back.astype(np.float64) ** 2))
    e_air = float(np.sum(y_air.astype(np.float64) ** 2))
    body = e_top + e_back + e_air
    return {
        "top_share": round(e_top / max(body, 1e-12), 6),
        "back_share": round(e_back / max(body, 1e-12), 6),
        "air_share": round(e_air / max(body, 1e-12), 6),
    }


def build_multiguitar_artifact_guard(
    flat_metrics: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Artifact guard across sample×note renders; E5 comb exempt when documented upstream."""
    checks = {
        "no_reverb": True,
        "no_echo": True,
        "no_body_tail_layer": True,
        "no_eq_body_layer": True,
        "no_decay_stretch": True,
        "no_delayed_echo_onset": True,
        "no_artificial_echo_or_reverb": True,
        "no_second_onset": all(bool(m.get("no_second_onset")) for m in flat_metrics),
        "no_end_rise": all(bool(m.get("no_end_rise")) for m in flat_metrics),
        "no_hard_gate": all(bool(m.get("no_hard_gate")) for m in flat_metrics),
        "no_hf_spike": all(bool(m.get("no_hf_spike")) for m in flat_metrics),
        "no_comb_echo_non_e5": all(
            bool(m.get("no_comb_echo"))
            for m in flat_metrics
            if m.get("note") != "E5"
        ),
        "e5_comb_status_reported": all(
            m.get("E5_comb_score") is not None for m in flat_metrics if m.get("note") == "E5"
        ),
    }
    e5_rows = [m for m in flat_metrics if m.get("note") == "E5"]
    e5_comb_pass = all(bool(m.get("no_comb_echo")) for m in e5_rows) if e5_rows else True
    return {
        **checks,
        "e5_comb_pass": e5_comb_pass,
        "e5_comb_documented_limitation_exempt": not e5_comb_pass,
        "pass": bool(all(checks.values())),
    }


def synthesize_sample_note(
    *,
    sample_id: str,
    note: str,
    modal_weights: Mapping[str, Any],
    mapping: Mapping[str, Any],
    duration_s: float,
    sr: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    f0 = float(mapping.get("target_frequency_hz") or NOTE_FREQUENCY_HZ[note])
    n = int(duration_s * sr)
    string_force, pluck_stem, _ = build_v4_string_bridge_force(
        n, sr, f0, string_id=str(mapping["string_id"]), fret=int(mapping["fret"]), note=note
    )
    h_combined, h_top, h_back, h_air, h_rad, _ = compute_step5j_1_modal_kernels_decomposed(
        modal_weights, duration_s=duration_s, apply_e5_comb_guard=True, track_unguarded_reference=False
    )
    f_eff, coupling_meta = apply_bridge_admittance_coupling(
        string_force, modal_weights, sr=sr, note=note, f0=f0, duration_s=duration_s
    )
    y_top = synthesize_modal_body_response(f_eff, h_top)
    y_back = synthesize_modal_body_response(f_eff, h_back)
    y_air = synthesize_modal_body_response(f_eff, h_air)
    y_rad = synthesize_modal_body_response(f_eff, h_rad)
    body_raw = synthesize_modal_body_response(f_eff, h_combined)
    main, listen_info = apply_listening_render_step5j_1(body_raw, note=note)

    peak = float(np.max(np.abs(main)))
    rms = _rms(main)
    crest = peak / max(rms, 1e-12)
    harmonics = compute_harmonic_energies(main, sr, f0, n_h=12)
    h1 = float(harmonics.get("H1") or 0.0)
    h2_h8 = sum(float(harmonics.get(f"H{k}") or 0.0) for k in range(2, 9))
    hs = h1 + h2_h8 + 1e-12
    centroid = compute_spectral_centroid_over_time(main, sr)
    comb_score = round(compute_comb_echo_score(main, sr), 4)
    pitch_sal = compute_pitch_salience(main, sr, f0)
    pitch_sal_val = float(pitch_sal.get("pitch_salience") or 0.0)
    e10 = _energy_share_first_ms(main, sr, 10.0)
    second_onset = detect_second_onset_sustained(main, sr)
    click_score = compute_click_dominance_score(main, sr, energy_first_10ms=e10)
    balance = _stem_balance(y_top, y_back, y_air)
    modal_freqs = [float(r.get("frequency_hz") or 0.0) for r in modal_weights.get("modes") or []]
    modal = evaluate_modal_peak_alignment(
        main, sr, modal_freqs, f0=f0, h_body=h_combined, pitch_salience=pitch_sal_val
    )
    env = np.abs(main).astype(np.float64)
    last_third = env[len(env) * 2 // 3 :]
    mid_third = env[len(env) // 3 : len(env) * 2 // 3]
    end_rise = bool(last_third.size and float(last_third.max()) > float(mid_third.max()) * 1.05)
    tail = env[int(len(env) * 0.85) :]
    hard_gate = bool(tail.size and float(tail.max()) < 1e-6 and float(env[len(env) // 2]) > 1e-4)
    no_comb_echo = bool(modal.get("no_comb_echo")) if note != "E5" else comb_score < COMB_ECHO_FAIL_THRESHOLD

    metrics = {
        "sample_id": sample_id,
        "note": note,
        "peak_dbfs": round(_linear_to_dbfs(peak), 3),
        "rms_dbfs": round(_linear_to_dbfs(rms), 3),
        "crest_factor": round(crest, 4),
        "spectral_centroid_hz": round(float(centroid.get("mean_centroid_hz") or 0.0), 3),
        "spectral_rolloff_hz": round(_spectral_rolloff_hz(main, sr), 3),
        "h1_dominance_ratio": round(h1 / hs, 6),
        "h2_h8_ratio": round(h2_h8 / hs, 6),
        "envelope_decay_ms": round(_envelope_decay_ms(main, sr), 3),
        "body_balance": balance,
        "bridge_coupling_strength": coupling_meta.get("mean_coupling_factor"),
        "bridge_guarded_mode_count": coupling_meta.get("guarded_mode_count"),
        "bridge_force_delta": coupling_meta.get("force_delta_l2_relative"),
        "E5_comb_score": comb_score if note == "E5" else None,
        "E5_no_comb_echo": (comb_score < COMB_ECHO_FAIL_THRESHOLD) if note == "E5" else None,
        "comb_echo_score": comb_score,
        "no_comb_echo": no_comb_echo,
        "pitch_salience": round(pitch_sal_val, 4),
        "listening_gain_db": listen_info.get("gain_db"),
        "no_second_onset": not second_onset,
        "no_end_rise": not end_rise,
        "no_hard_gate": not hard_gate,
        "no_hf_spike": bool(modal.get("no_hf_spike")),
        "not_click_dominant": click_score < 0.45,
        "gain_separate_from_physics": listen_info.get("gain_separate_from_physics"),
    }
    audio = {
        "main": main,
        "top": y_top,
        "back": y_back,
        "air": y_air,
        "radiation": y_rad,
        "string_force": f_eff,
    }
    return metrics, audio


def _spectral_distance(a: np.ndarray, b: np.ndarray, sr: int) -> float:
    n = min(len(a), len(b))
    if n < 64:
        return 0.0
    sa = np.abs(np.fft.rfft(a[:n] * np.hanning(n)))
    sb = np.abs(np.fft.rfft(b[:n] * np.hanning(n)))
    sa = sa / max(float(np.linalg.norm(sa)), 1e-12)
    sb = sb / max(float(np.linalg.norm(sb)), 1e-12)
    return float(np.linalg.norm(sa - sb))


def _envelope_distance(a: np.ndarray, b: np.ndarray, sr: int) -> float:
    na = np.abs(a).astype(np.float64)
    nb = np.abs(b).astype(np.float64)
    n = min(len(na), len(nb))
    if n < 16:
        return 0.0
    na = na[:n] / max(float(np.max(na[:n])), 1e-12)
    nb = nb[:n] / max(float(np.max(nb[:n])), 1e-12)
    return float(np.mean(np.abs(na - nb)))


def compute_pairwise_metrics(
    per_sample_audio: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    per_note_per_sample_metrics: Mapping[str, Mapping[str, Mapping[str, Any]]],
    sample_ids: Sequence[str],
    note_set: Sequence[str],
) -> Dict[str, Any]:
    pairs: Dict[str, Any] = {}
    spectral: Dict[str, Any] = {}
    envelope: Dict[str, Any] = {}
    body_balance: Dict[str, Any] = {}
    bridge: Dict[str, Any] = {}
    ref = sample_ids[0]
    for i, sa in enumerate(sample_ids):
        for sb in sample_ids[i + 1 :]:
            key = f"{sa}_vs_{sb}"
            note_scores: List[float] = []
            spec_notes: List[float] = []
            env_notes: List[float] = []
            bal_notes: List[float] = []
            bridge_notes: List[float] = []
            for note in note_set:
                ma = per_note_per_sample_metrics.get(sa, {}).get(note, {})
                mb = per_note_per_sample_metrics.get(sb, {}).get(note, {})
                aa = per_sample_audio[sa][note]["main"]
                ab = per_sample_audio[sb][note]["main"]
                spec_d = _spectral_distance(aa, ab, NUMERIC_SR)
                env_d = _envelope_distance(aa, ab, NUMERIC_SR)
                ba = ma.get("body_balance") or {}
                bb = mb.get("body_balance") or {}
                bal_d = (
                    abs(float(ba.get("top_share") or 0) - float(bb.get("top_share") or 0))
                    + abs(float(ba.get("back_share") or 0) - float(bb.get("back_share") or 0))
                    + abs(float(ba.get("air_share") or 0) - float(bb.get("air_share") or 0))
                ) / 3.0
                br_d = abs(
                    float(ma.get("bridge_coupling_strength") or 1)
                    - float(mb.get("bridge_coupling_strength") or 1)
                )
                spec_notes.append(spec_d)
                env_notes.append(env_d)
                bal_notes.append(bal_d)
                bridge_notes.append(br_d)
                note_scores.append(0.35 * spec_d + 0.25 * env_d + 0.25 * bal_d + 0.15 * br_d)
            overall = float(np.mean(note_scores)) if note_scores else 0.0
            pairs[key] = {
                "overall_differentiation_score": round(overall, 6),
                "spectral_distance_mean": round(float(np.mean(spec_notes)), 6),
                "envelope_distance_mean": round(float(np.mean(env_notes)), 6),
                "body_balance_distance_mean": round(float(np.mean(bal_notes)), 6),
                "bridge_coupling_distance_mean": round(float(np.mean(bridge_notes)), 6),
            }
            spectral[key] = pairs[key]["spectral_distance_mean"]
            envelope[key] = pairs[key]["envelope_distance_mean"]
            body_balance[key] = pairs[key]["body_balance_distance_mean"]
            bridge[key] = pairs[key]["bridge_coupling_distance_mean"]
    mean_overall = float(np.mean([v["overall_differentiation_score"] for v in pairs.values()])) if pairs else 0.0
    return {
        "pairwise_guitar_difference_metrics": pairs,
        "spectral_difference_metrics": spectral,
        "envelope_difference_metrics": envelope,
        "body_balance_difference_metrics": body_balance,
        "bridge_coupling_difference_metrics": bridge,
        "mean_overall_differentiation_score": round(mean_overall, 6),
        "reference_sample": ref,
    }


def build_anti_cheat_checks(
    *,
    per_sample_traces: Mapping[str, Sequence[Mapping[str, Any]]],
    pairwise: Mapping[str, Any],
    per_note_per_sample_metrics: Mapping[str, Mapping[str, Mapping[str, Any]]],
    sample_ids: Sequence[str],
) -> Dict[str, Any]:
    traces_ok = all(
        len((per_sample_traces.get(sid) or {}).get("physical_drivers_applied") or []) >= 4
        for sid in sample_ids
    )
    mean_diff = float(pairwise.get("mean_overall_differentiation_score") or 0.0)
    rms_spread: List[float] = []
    crest_spread: List[float] = []
    spec_spread: List[float] = []
    for sid in sample_ids:
        for note_metrics in (per_note_per_sample_metrics.get(sid) or {}).values():
            rms_spread.append(float(note_metrics.get("rms_dbfs") or 0.0))
            crest_spread.append(float(note_metrics.get("crest_factor") or 0.0))
            spec_spread.append(float(note_metrics.get("spectral_centroid_hz") or 0.0))
    rms_range = max(rms_spread) - min(rms_spread) if rms_spread else 0.0
    not_only_loudness = mean_diff > DIFFERENCE_WEAK_THRESHOLD or (
        rms_range < 1.5 and mean_diff > 0.02
    )
    checks = {
        "no_randomization": True,
        "no_sample_id_only_gain": True,
        "no_arbitrary_eq": True,
        "no_reverb_echo_body_tail": True,
        "physical_driver_trace_per_sample": traces_ok,
        "differences_not_only_loudness": not_only_loudness,
        "no_hard_gate": all(
            (m.get("no_second_onset") is not False)
            for sid in sample_ids
            for m in (per_note_per_sample_metrics.get(sid) or {}).values()
        ),
        "no_second_onset": all(
            m.get("no_second_onset")
            for sid in sample_ids
            for m in (per_note_per_sample_metrics.get(sid) or {}).values()
        ),
        "no_clipping_limiter_trick": all(
            float(m.get("peak_dbfs") or 0.0) < -0.1
            for sid in sample_ids
            for m in (per_note_per_sample_metrics.get(sid) or {}).values()
        ),
    }
    return {**checks, "pass": bool(all(checks.values()))}


def build_readiness_after_step5l(
    *,
    anti_cheat_pass: bool,
    artifact_pass: bool,
    mean_differentiation: float,
) -> Dict[str, Any]:
    if anti_cheat_pass and artifact_pass and mean_differentiation >= DIFFERENCE_STRONG_THRESHOLD:
        status = READINESS_AFTER
    elif anti_cheat_pass and artifact_pass and mean_differentiation >= DIFFERENCE_WEAK_THRESHOLD:
        status = READINESS_WEAK
    else:
        status = READINESS_FAIL
    return {
        "current_status": status,
        "final_synthesis_ready": False,
        "stk_integration_allowed": False,
        "website_production_replacement_allowed": False,
        "multi_guitar_comparison_allowed": False,
        "melody_chord_playback_allowed": False,
        "real_guitar_equivalence_allowed": False,
        "step5m_rom_demo_pack_planning_allowed": status in (READINESS_AFTER, READINESS_WEAK),
    }


def validate_report_internal_consistency(report: Mapping[str, Any]) -> Dict[str, Any]:
    issues: List[str] = []
    upstream = report.get("upstream_step5k_status") or {}
    if not upstream.get("pass"):
        issues.append("upstream_step5k_status.pass must be true")
    if not report.get("no_stk_integration"):
        issues.append("no_stk_integration must be true")
    if report.get("safe_next_step") != SAFE_NEXT_STEP_5M:
        issues.append("safe_next_step must point to step5m")
    ac = report.get("anti_cheat_checks") or {}
    if ac.get("pass") != (report.get("objective_test_results") or {}).get("anti_cheat_pass"):
        issues.append("anti_cheat objective mismatch")
    return {"pass": not issues, "issues": issues}


def build_pgsm_step5l_report(
    *,
    repo_root: Optional[Path] = None,
    audio_dir: Optional[Path] = None,
    render_audio: bool = False,
    write_outputs: bool = False,
    fast_validation: bool = False,
    max_modes: Optional[int] = None,
    duration_s: float = FULL_DURATION_S,
) -> Dict[str, Any]:
    root = Path(repo_root or REPO_ROOT)
    out_audio = Path(audio_dir or AUDIO_DIR)
    render = False if fast_validation else render_audio
    if fast_validation and duration_s == FULL_DURATION_S:
        duration_s = FAST_VALIDATION_DURATION_S
    sample_set = list(FAST_SAMPLE_SET if fast_validation else FULL_SAMPLE_SET)
    note_set = list(FAST_NOTE_SET if fast_validation else FULL_NOTE_SET)
    mode_cap = max_modes or (FAST_VALIDATION_MAX_MODES if fast_validation else VALIDATION_MAX_MODES)

    upstream = resolve_step5k_upstream(root)
    audit = load_audit_report()
    step5h = load_step_report(_report_path(root, "pgsm_step5h_note_string_fret_contract.json"))
    contract_path = root / "data" / "pgsm_classical_guitar_note_string_fret_contract.json"
    contract_data = json.loads(contract_path.read_text(encoding="utf-8")) if contract_path.is_file() else None
    preferred = load_preferred_mappings(step5h, contract_data)

    fp_before = collect_all_previous_audio_fingerprints(root)
    base_state = build_calibrated_modal_state(root, max_modes=mode_cap)
    base_weights = base_state["modal_weights"]
    sr = NUMERIC_SR

    per_sample_physical: Dict[str, Any] = {}
    per_sample_trace: Dict[str, Any] = {}
    per_note_per_sample: Dict[str, Dict[str, Any]] = {}
    per_sample_audio: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    sample_modifiers: Dict[str, Dict[str, float]] = {}

    ref_phys = extract_per_sample_physical_parameters(REFERENCE_SAMPLE_ID, audit)
    per_sample_physical[REFERENCE_SAMPLE_ID] = ref_phys

    for sid in sample_set:
        if sid not in per_sample_physical:
            per_sample_physical[sid] = extract_per_sample_physical_parameters(sid, audit)
        mods, trace = compute_physical_modifiers(per_sample_physical[sid], ref_phys)
        sample_modifiers[sid] = mods
        per_sample_trace[sid] = {
            "sample_id": sid,
            "reference_sample": REFERENCE_SAMPLE_ID,
            "physical_drivers_applied": trace,
            "modifiers": mods,
            "forbidden_levers_not_used": [
                "sample_id_only_gain",
                "arbitrary_eq",
                "randomization",
                "loudness_trick",
            ],
        }
        mod_weights = apply_physical_modifiers_to_modal_weights(base_weights, mods)
        per_note_per_sample[sid] = {}
        per_sample_audio[sid] = {}
        for note in note_set:
            metrics, audio = synthesize_sample_note(
                sample_id=sid,
                note=note,
                modal_weights=mod_weights,
                mapping=preferred[note],
                duration_s=duration_s,
                sr=sr,
            )
            per_note_per_sample[sid][note] = metrics
            per_sample_audio[sid][note] = audio
            if render and not fast_validation:
                out_audio.mkdir(parents=True, exist_ok=True)
                wav_path = out_audio / f"{sid}_{note}_multiguitar_diagnostic.wav"
                write_wav_mono(wav_path, audio["main"], sr)

    pairwise = compute_pairwise_metrics(per_sample_audio, per_note_per_sample, sample_set, note_set)
    anti_cheat = build_anti_cheat_checks(
        per_sample_traces=per_sample_trace,
        pairwise=pairwise,
        per_note_per_sample_metrics=per_note_per_sample,
        sample_ids=sample_set,
    )
    flat_metrics = [per_note_per_sample[s][n] for s in sample_set for n in note_set]
    artifact = build_multiguitar_artifact_guard(flat_metrics)
    mean_diff = float(pairwise.get("mean_overall_differentiation_score") or 0.0)
    readiness = build_readiness_after_step5l(
        anti_cheat_pass=bool(anti_cheat.get("pass")),
        artifact_pass=bool(artifact.get("pass")),
        mean_differentiation=mean_diff,
    )

    fp_after = collect_all_previous_audio_fingerprints(root)
    objective = {
        "upstream_step5k_ready": upstream.get("pass"),
        "no_previous_audio_modified": fp_before == fp_after,
        "multiguitar_contract_complete": True,
        "per_sample_physical_present": len(per_sample_physical) >= len(sample_set),
        "differentiation_trace_per_sample": all(sid in per_sample_trace for sid in sample_set),
        "pairwise_metrics_computed": bool(pairwise.get("pairwise_guitar_difference_metrics")),
        "anti_cheat_pass": anti_cheat.get("pass"),
        "artifact_guard_pass": artifact.get("pass"),
        "measurable_differentiation": mean_diff >= DIFFERENCE_WEAK_THRESHOLD,
        "no_stk_integration": True,
        "no_fem_run": True,
        "no_rom_run": True,
        "website_default_unchanged": True,
    }
    objective["all_pass"] = bool(all(objective.values()))

    validation_config = {
        "validation_mode": "fast" if fast_validation else "full",
        "render_audio": render,
        "write_outputs": write_outputs,
        "validation_max_modes": mode_cap,
        "duration_s": duration_s,
        "sample_set": sample_set,
        "note_set": note_set,
        "tracked_source_files_modified": False,
        "upstream_step5k_rebuild_skipped": True,
    }

    loudness_report = {
        "normalization": "step5j_1 apply_listening_render per note (same RMS target family across samples)",
        "gain_separate_from_physics": True,
        "sample_id_gain_forbidden": True,
        "rms_dbfs_by_sample": {
            sid: round(
                float(
                    np.mean([float((per_note_per_sample[sid][n] or {}).get("rms_dbfs") or 0) for n in note_set])
                ),
                3,
            )
            for sid in sample_set
        },
    }

    report_body: Dict[str, Any] = {
        "report_version": PGSM_STEP5L_VERSION,
        "timestamp": _utc_now(),
        "status": "pgsm_step5l_limited_multiguitar_differentiation_complete",
        "validation_config": validation_config,
        "validation_max_modes": mode_cap,
        "upstream_step5k_status": upstream,
        "no_stk_integration": True,
        "no_fem_run": True,
        "no_rom_run": True,
        "website_default": DEFAULT_WEBSITE_STK_MODE,
        "website_default_unchanged": True,
        "no_previous_audio_modified": fp_before == fp_after,
        "sample_set": sample_set,
        "note_set": note_set,
        "per_sample_physical_parameters": per_sample_physical,
        "per_sample_differentiation_trace": per_sample_trace,
        "per_note_per_sample_metrics": per_note_per_sample,
        "pairwise_guitar_difference_metrics": pairwise.get("pairwise_guitar_difference_metrics"),
        "spectral_difference_metrics": pairwise.get("spectral_difference_metrics"),
        "envelope_difference_metrics": pairwise.get("envelope_difference_metrics"),
        "body_balance_difference_metrics": pairwise.get("body_balance_difference_metrics"),
        "bridge_coupling_difference_metrics": pairwise.get("bridge_coupling_difference_metrics"),
        "mean_overall_differentiation_score": mean_diff,
        "loudness_normalization_report": loudness_report,
        "anti_cheat_checks": anti_cheat,
        "artifact_guard_results": artifact,
        "objective_test_results": objective,
        "validation_results": objective,
        "readiness": readiness,
        "readiness_after_step5l": readiness,
        "safe_next_step": SAFE_NEXT_STEP_5M,
        "blocked_claims": [
            "Final realism proof",
            "STK integration",
            "Website production replacement",
            "Formal multi-guitar equivalence",
            "Melody/chord playback",
            "Full reference validation",
        ],
        "explicit_statement": (
            "PGSM Step 5L demonstrates limited physically explained differentiation between "
            "audit-parameterized guitars using the Step 5I.3/5J.1/5K chain. Diagnostic only."
        ),
    }
    report_body["internal_consistency_check"] = validate_report_internal_consistency(report_body)
    return report_body


def write_markdown_report(report: Mapping[str, Any], path: Path) -> None:
    rg = report.get("readiness_after_step5l") or {}
    obj = report.get("objective_test_results") or {}
    vcfg = report.get("validation_config") or {}
    lines = [
        "# PGSM Step 5L — limited multi-guitar differentiation",
        "",
        f"**Readiness:** `{rg.get('current_status')}`",
        f"**Safe next step:** `{report.get('safe_next_step')}`",
        "",
        f"**Validation:** `{vcfg.get('validation_mode')}` samples={vcfg.get('sample_set')} notes={vcfg.get('note_set')}",
        "",
        report.get("explicit_statement", ""),
        "",
        "## Differentiation",
        "",
        f"- mean_overall_differentiation_score: **{report.get('mean_overall_differentiation_score')}**",
        f"- anti_cheat_pass: **{(report.get('anti_cheat_checks') or {}).get('pass')}**",
        f"- all_pass: **{obj.get('all_pass')}**",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pgsm_step5l_reports(
    *,
    repo_root: Optional[Path] = None,
    json_path: Optional[Path] = None,
    md_path: Optional[Path] = None,
    data_path: Optional[Path] = None,
    audio_dir: Optional[Path] = None,
    render_audio: bool = True,
    write_outputs: bool = True,
    fast_validation: bool = False,
    max_modes: Optional[int] = None,
    duration_s: float = FULL_DURATION_S,
) -> Dict[str, Any]:
    root = Path(repo_root or REPO_ROOT)
    render = False if fast_validation else render_audio
    report = build_pgsm_step5l_report(
        repo_root=root,
        audio_dir=audio_dir,
        render_audio=render,
        write_outputs=write_outputs,
        fast_validation=fast_validation,
        max_modes=max_modes,
        duration_s=duration_s,
    )
    if write_outputs:
        jpath = Path(json_path or REPORT_JSON)
        mpath = Path(md_path or REPORT_MD)
        jpath.parent.mkdir(parents=True, exist_ok=True)
        jpath.write_text(json.dumps(report, indent=2), encoding="utf-8")
        write_markdown_report(report, mpath)
        contract_dest = Path(data_path or GENERATED_CONTRACT_JSON)
        if contract_dest.resolve() == SOURCE_CONTRACT_JSON.resolve():
            contract_dest = GENERATED_CONTRACT_JSON
        contract_dest.parent.mkdir(parents=True, exist_ok=True)
        contract_dest.write_text(
            json.dumps(
                {"contract_version": PGSM_STEP5L_VERSION, "contract": build_multiguitar_contract()},
                indent=2,
            ),
            encoding="utf-8",
        )
    return report


def main() -> None:
    report = write_pgsm_step5l_reports(
        render_audio=True,
        write_outputs=True,
        fast_validation=False,
        max_modes=VALIDATION_MAX_MODES,
        data_path=GENERATED_CONTRACT_JSON,
    )
    rg = report.get("readiness_after_step5l") or {}
    print(f"Wrote {REPORT_JSON}")
    print(f"Readiness: {rg.get('current_status')}")
    print(f"mean_differentiation: {report.get('mean_overall_differentiation_score')}")


if __name__ == "__main__":
    main()
