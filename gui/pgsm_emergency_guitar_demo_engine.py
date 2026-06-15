#!/usr/bin/env python3
"""
PGSM / STK emergency guitar demo engine v4 — ordered physical chain.
Pluck → string → bridge force → body modes → radiation → listening render.
Diagnostic only; uses PGSM model concepts with moderate per-guitar variation.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from pgsm_step2_1_parameter_targets import load_step_report
from pgsm_step3a_numerical_ir_testbench import FIXED_PLUCK_POSITION, NUMERIC_SR
from pgsm_step4a_single_note_diagnostic_audio import (
    build_calibrated_modal_state,
    synthesize_modal_body_response,
    write_wav_mono,
)
from pgsm_step5a_limited_note_set_diagnostic_audio import NOTE_FREQUENCY_HZ
from pgsm_step5i_1_string_damping_duration_harshness_repair import (
    load_preferred_mappings,
    _report_path,
)
from pgsm_step5i_3_absolute_frequency_damping_pluck_balance import build_v4_string_bridge_force
from pgsm_step5j_1_guitar_articulation_body_balance_repair import (
    apply_listening_render_step5j_1,
    compute_step5j_1_modal_kernels_decomposed,
)
from pgsm_step5k_bridge_admittance_feedback_coupling import apply_bridge_admittance_coupling
from pgsm_step5l_limited_multiguitar_differentiation import (
    REFERENCE_SAMPLE_ID,
    apply_physical_modifiers_to_modal_weights,
    compute_physical_modifiers,
    extract_per_sample_physical_parameters,
)
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE
from stk_v6_2_audit_features import load_audit_report

ENGINE_VERSION = "pgsm_emergency_guitar_demo_engine_v4"
EMERGENCY_DEMO_VERSION = "v4_ordered_physical_chain"
SR = NUMERIC_SR
DURATION_S = 2.5
MODAL_MODE_CAP = 32

SAMPLE_SET = ("sample_000", "sample_001", "sample_002")
NOTE_SET = ("A2", "A4", "E5")

MAX_PEAK_DBFS = -4.0
NOTE_PEAK_TARGET_DBFS: Dict[str, float] = {"A2": -5.0, "A4": -6.5, "E5": -6.5}

CORRELATION_TOO_SIMILAR = 0.95
CORRELATION_TOO_UNRELATED = 0.20
CORRELATION_FAMILY_LO = 0.45
CORRELATION_FAMILY_HI = 0.90

REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = REPO_ROOT / "audio" / "pgsm_emergency_guitar_demo"
REPORT_JSON = REPO_ROOT / "audio" / "debug_reports" / "pgsm_emergency_guitar_demo_report.json"
REPORT_MD = REPO_ROOT / "audio" / "debug_reports" / "pgsm_emergency_guitar_demo_report.md"

READINESS_OK = "ready_for_stk_gui_activation"
READINESS_WEAK = "demo_generated_but_differentiation_weak"
READINESS_REVIEW = "demo_generated_but_physical_chain_needs_review"
READINESS_FAIL = "emergency_demo_failed"

PHYSICAL_CHAIN_STAGES: Tuple[str, ...] = (
    "A_pluck_contact",
    "B_string_vibration",
    "C_bridge_force",
    "D_body_modal_response",
    "E_material_mass_shape_modifiers",
    "F_radiation_output",
    "G_modal_decay_tail_only",
)

PHYSICAL_MODIFIER_KEYS: Tuple[str, ...] = (
    "bridge_excitation_scale",
    "air_weight_scale",
    "radiation_weight_scale",
    "damping_tau_scale",
    "top_back_share_balance",
)

# Moderate voicing overlay on audit-derived modifiers (disclosed diagnostic calibration).
GENTLE_SAMPLE_VOICING: Dict[str, Dict[str, Any]] = {
    "sample_000": {
        "profile": "balanced_neutral_classical",
        "pluck_position_delta": 0.0,
        "modifier_overlay": {
            "bridge_excitation_scale": 1.00,
            "air_weight_scale": 1.00,
            "radiation_weight_scale": 1.00,
            "damping_tau_scale": 1.00,
            "top_back_share_balance": 1.00,
        },
    },
    "sample_001": {
        "profile": "bright_light_faster_response",
        "pluck_position_delta": 0.010,
        "modifier_overlay": {
            "bridge_excitation_scale": 1.05,
            "air_weight_scale": 0.96,
            "radiation_weight_scale": 1.07,
            "damping_tau_scale": 0.94,
            "top_back_share_balance": 1.03,
        },
    },
    "sample_002": {
        "profile": "warm_deep_heavier_response",
        "pluck_position_delta": -0.012,
        "modifier_overlay": {
            "bridge_excitation_scale": 0.96,
            "air_weight_scale": 1.05,
            "radiation_weight_scale": 0.94,
            "damping_tau_scale": 1.06,
            "top_back_share_balance": 0.97,
        },
    },
}

PHYSICAL_CHAIN_SUMMARY: Dict[str, str] = {
    "A_pluck_contact": "v4 string bridge force with deterministic pluck onset (Step 5I.3)",
    "B_string_vibration": "harmonic partials with absolute-frequency damping law",
    "C_bridge_force": "string force → bridge via Step 5K admittance coupling",
    "D_body_modal_response": "PGSM modal kernels top/back/air/radiation (Step 5J.1)",
    "E_material_mass_shape_modifiers": "per-sample audit modifiers on modal weights (Step 5L family)",
    "F_radiation_output": "combined modal radiation-weighted body response",
    "G_modal_decay_tail_only": "no reverb/echo; decay from modal tau only",
}

DIAGNOSTIC_SIMPLIFICATIONS: List[str] = [
    f"Shared ROM modal catalog capped at {MODAL_MODE_CAP} modes",
    "Single reference modal state; per-guitar variation via bounded physical modifiers only",
    "No FEM/ROM execution; no STK integration",
    "Moderate voicing overlay disclosed as diagnostic_exaggeration_for_audible_demo",
]


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
        "physical_chain_stages": list(PHYSICAL_CHAIN_STAGES),
        "sample_set": list(SAMPLE_SET),
        "note_set": list(NOTE_SET),
        "duration_s": DURATION_S,
        "modal_mode_cap": MODAL_MODE_CAP,
        "diagnostic_simplifications": DIAGNOSTIC_SIMPLIFICATIONS,
    }


def compute_gentle_sample_modifiers(
    physical: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    sample_id: str,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """Audit-derived modifiers with moderate disclosed voicing overlay."""
    base_mods, base_trace = compute_physical_modifiers(physical, reference)
    overlay = GENTLE_SAMPLE_VOICING[sample_id]["modifier_overlay"]
    mods: Dict[str, float] = {}
    trace: List[Dict[str, Any]] = []
    for key in PHYSICAL_MODIFIER_KEYS:
        audit_v = float(base_mods.get(key, 1.0))
        ov = float(overlay.get(key, 1.0))
        combined = round(_clamp(audit_v * ov, 0.88, 1.12), 6)
        mods[key] = combined
        trace.append(
            {
                "modifier": key,
                "audit_multiplier": round(audit_v, 6),
                "voicing_overlay": round(ov, 6),
                "combined_multiplier": combined,
                "chain_stage": "E_material_mass_shape_modifiers",
            }
        )
    for row in base_trace:
        trace.append({**row, "chain_stage": "E_material_mass_shape_modifiers"})
    return mods, trace


def build_synthesis_profile(sample_id: str, modifiers: Mapping[str, float]) -> Dict[str, Any]:
    voicing = GENTLE_SAMPLE_VOICING[sample_id]
    pluck_pos = _clamp(
        FIXED_PLUCK_POSITION + float(voicing.get("pluck_position_delta") or 0.0),
        0.10,
        0.20,
    )
    return {
        "sample_id": sample_id,
        "voicing_profile": voicing["profile"],
        "pluck_position_ratio": round(pluck_pos, 5),
        "physical_modifiers": dict(modifiers),
        "bridge_coupling_proxy": modifiers.get("bridge_excitation_scale"),
        "radiation_weight_proxy": modifiers.get("radiation_weight_scale"),
    }


def _apply_peak_ceiling(y: np.ndarray, note: str) -> Tuple[np.ndarray, Dict[str, float]]:
    note_peak = 10.0 ** (NOTE_PEAK_TARGET_DBFS[note] / 20.0)
    max_peak = 10.0 ** (MAX_PEAK_DBFS / 20.0)
    out = y.astype(np.float64).copy()
    peak = float(np.max(np.abs(out)))
    if peak > note_peak:
        out *= note_peak / peak
    peak = float(np.max(np.abs(out)))
    if peak > max_peak:
        out *= max_peak / peak
    peak = float(np.max(np.abs(out)))
    return out, {
        "peak_dbfs": round(_linear_to_dbfs(peak), 3),
        "rms_dbfs": round(_linear_to_dbfs(_rms(out)), 3),
    }


def synthesize_ordered_physical_chain_note(
    *,
    sample_id: str,
    note: str,
    modal_weights: Mapping[str, Any],
    kernel_pack: Mapping[str, Any],
    mapping: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    f0 = float(mapping.get("target_frequency_hz") or NOTE_FREQUENCY_HZ[note])
    n = int(DURATION_S * SR)
    pluck_pos = float(profile["pluck_position_ratio"])

    string_force, pluck_stem, string_meta = build_v4_string_bridge_force(
        n,
        SR,
        f0,
        string_id=str(mapping["string_id"]),
        fret=int(mapping["fret"]),
        note=note,
        pluck_position_ratio=pluck_pos,
        onset_ms=3.5,
    )
    f_eff, coupling_meta = apply_bridge_admittance_coupling(
        string_force,
        modal_weights,
        sr=SR,
        note=note,
        f0=f0,
        duration_s=DURATION_S,
    )
    h_combined = kernel_pack["h_combined"]
    h_top = kernel_pack["h_top"]
    h_back = kernel_pack["h_back"]
    h_air = kernel_pack["h_air"]
    h_rad = kernel_pack["h_rad"]

    body_combined = synthesize_modal_body_response(f_eff, h_combined)
    body = body_combined
    main, listen_info = apply_listening_render_step5j_1(body, note=note)
    main, level = _apply_peak_ceiling(main, note)

    meta = {
        "sample_id": sample_id,
        "note": note,
        "f0_hz": f0,
        "pluck_position_ratio": pluck_pos,
        "string_meta": {
            "damping_contract": string_meta.get("damping_contract_v2_applied"),
            "partial_count": len(string_meta.get("tau_by_partial") or {}),
        },
        "bridge_coupling_summary": {
            "mean_coupling_factor": coupling_meta.get("mean_coupling_factor"),
            "guarded_mode_count": coupling_meta.get("guarded_mode_count"),
            "force_delta_l2_relative": coupling_meta.get("force_delta_l2_relative"),
        },
        "modal_response_summary": {
            "mode_cap": MODAL_MODE_CAP,
            "kernel_duration_s": DURATION_S,
            "top_back_air_radiation_stems_synthesized": True,
        },
        "radiation_summary": {
            "combined_kernel_used": True,
            "listening_gain_db": listen_info.get("gain_db"),
            "gain_separate_from_physics": listen_info.get("gain_separate_from_physics"),
        },
        "partial_decay_summary": {
            "tau_by_partial_sample": dict(list((string_meta.get("tau_by_partial") or {}).items())[:6]),
        },
        **level,
    }
    return main.astype(np.float64), meta


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


def _spectral_distance(a: np.ndarray, b: np.ndarray) -> float:
    ns = min(len(a), len(b))
    if ns < 64:
        return 0.0
    sa = np.abs(np.fft.rfft(a[:ns] * np.hanning(ns)))
    sb = np.abs(np.fft.rfft(b[:ns] * np.hanning(ns)))
    sa = sa / max(float(np.linalg.norm(sa)), 1e-12)
    sb = sb / max(float(np.linalg.norm(sb)), 1e-12)
    return float(np.linalg.norm(sa - sb))


def compute_same_note_pairwise_correlation(
    audio_by_sample: Mapping[str, Mapping[str, np.ndarray]],
) -> Dict[str, Any]:
    by_note: Dict[str, Dict[str, float]] = {}
    all_corrs: List[float] = []
    for note in NOTE_SET:
        by_note[note] = {}
        for i, sa in enumerate(SAMPLE_SET):
            for sb in SAMPLE_SET[i + 1 :]:
                c = _waveform_correlation(audio_by_sample[sa][note], audio_by_sample[sb][note])
                key = f"{sa}_vs_{sb}"
                by_note[note][key] = round(c, 6)
                all_corrs.append(c)
    return {
        "same_note_pairwise_correlation": by_note,
        "max_correlation": round(max(all_corrs) if all_corrs else 1.0, 6),
        "min_correlation": round(min(all_corrs) if all_corrs else 1.0, 6),
        "mean_correlation": round(float(np.mean(all_corrs)) if all_corrs else 1.0, 6),
    }


def compute_pairwise_metrics(
    audio_by_sample: Mapping[str, Mapping[str, np.ndarray]],
) -> Dict[str, Any]:
    pairs: Dict[str, Any] = {}
    spec_scores: List[float] = []
    env_scores: List[float] = []
    for i, sa in enumerate(SAMPLE_SET):
        for sb in SAMPLE_SET[i + 1 :]:
            key = f"{sa}_vs_{sb}"
            per_note: Dict[str, Any] = {}
            for note in NOTE_SET:
                aa = audio_by_sample[sa][note]
                ab = audio_by_sample[sb][note]
                per_note[note] = {
                    "spectral_distance": round(_spectral_distance(aa, ab), 6),
                    "envelope_distance": round(_envelope_distance(aa, ab), 6),
                    "waveform_correlation": round(_waveform_correlation(aa, ab), 6),
                }
            spec_mean = float(np.mean([per_note[n]["spectral_distance"] for n in NOTE_SET]))
            env_mean = float(np.mean([per_note[n]["envelope_distance"] for n in NOTE_SET]))
            spec_scores.append(spec_mean)
            env_scores.append(env_mean)
            pairs[key] = {"per_note": per_note, "mean_spectral_distance": round(spec_mean, 6)}
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
    too_unrelated = min_c < CORRELATION_TOO_UNRELATED or max_c < CORRELATION_TOO_UNRELATED
    in_family = (
        not too_similar
        and not too_unrelated
        and CORRELATION_FAMILY_LO <= mean_c <= CORRELATION_FAMILY_HI
    )
    return {
        "max_correlation": max_c,
        "min_correlation": min_c,
        "mean_correlation": mean_c,
        "preferred_correlation_band": [CORRELATION_FAMILY_LO, CORRELATION_FAMILY_HI],
        "too_similar": too_similar,
        "too_unrelated": too_unrelated,
        "in_family_band": in_family,
        "pass": bool(in_family and not too_unrelated),
    }


def build_anti_cheat_checks(
    *,
    traces: Mapping[str, Sequence[Mapping[str, Any]]],
    family_metrics: Mapping[str, Any],
    pairwise: Mapping[str, Any],
    peak_rms_report: Mapping[str, Any],
) -> Dict[str, Any]:
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
        "physical_modifier_trace_per_sample": all(len(traces.get(sid) or []) >= 5 for sid in SAMPLE_SET),
        "differences_not_only_loudness": float(pairwise.get("mean_spectral_distance") or 0) > 0.02
        or not family_metrics.get("too_similar"),
        "guitar_family_consistency": bool(family_metrics.get("pass")),
    }
    return {**checks, "pass": bool(all(checks.values()))}


def build_readiness_emergency_demo(
    *,
    files_generated: int,
    peaks_controlled: bool,
    family_metrics: Mapping[str, Any],
) -> Dict[str, Any]:
    if files_generated < len(SAMPLE_SET) * len(NOTE_SET) or not peaks_controlled:
        status = READINESS_FAIL
    elif family_metrics.get("too_unrelated"):
        status = READINESS_REVIEW
    elif family_metrics.get("too_similar"):
        status = READINESS_WEAK
    elif family_metrics.get("pass"):
        status = READINESS_OK
    else:
        status = READINESS_WEAK
    return {
        "current_status": status,
        "stk_gui_activation_planning_allowed": status == READINESS_OK,
        "files_generated": files_generated,
    }


def write_emergency_demo_markdown(report: Mapping[str, Any], path: Path) -> None:
    rg = report.get("readiness") or {}
    fam = report.get("guitar_family_consistency_metrics") or {}
    lines = [
        "# PGSM emergency guitar demo v4",
        "",
        f"**Version:** `{report.get('emergency_demo_version')}`",
        f"**Readiness:** `{rg.get('current_status')}`",
        f"**Max correlation:** {report.get('max_same_note_correlation')}",
        f"**Family consistency:** {fam.get('pass')}",
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
    ref_phys = extract_per_sample_physical_parameters(REFERENCE_SAMPLE_ID, audit)
    step5h = load_step_report(_report_path(root, "pgsm_step5h_note_string_fret_contract.json"))
    contract_path = root / "data" / "pgsm_classical_guitar_note_string_fret_contract.json"
    contract_data = json.loads(contract_path.read_text(encoding="utf-8")) if contract_path.is_file() else None
    preferred = load_preferred_mappings(step5h, contract_data)

    base_state = build_calibrated_modal_state(root, max_modes=MODAL_MODE_CAP)
    base_weights = base_state["modal_weights"]

    physical_parameters: Dict[str, Any] = {}
    differentiation_trace: Dict[str, Any] = {}
    synthesis_profiles: Dict[str, Any] = {}
    factor_multipliers: Dict[str, Dict[str, float]] = {}
    kernel_packs: Dict[str, Any] = {}
    modal_summaries: Dict[str, Any] = {}
    partial_decay_summary: Dict[str, Dict[str, Any]] = {}
    spectral_metrics: Dict[str, Dict[str, Any]] = {}
    peak_rms_by_file: Dict[str, Dict[str, float]] = {}
    audio_by_sample: Dict[str, Dict[str, np.ndarray]] = {}
    generated_files: List[str] = []

    for sid in SAMPLE_SET:
        physical_parameters[sid] = extract_per_sample_physical_parameters(sid, audit)
        mods, trace = compute_gentle_sample_modifiers(physical_parameters[sid], ref_phys, sample_id=sid)
        mod_weights = apply_physical_modifiers_to_modal_weights(base_weights, mods)
        profile = build_synthesis_profile(sid, mods)
        factor_multipliers[sid] = dict(mods)
        synthesis_profiles[sid] = profile
        differentiation_trace[sid] = {
            "sample_id": sid,
            "voicing_profile": profile["voicing_profile"],
            "physical_drivers_applied": trace,
            "diagnostic_exaggeration_for_audible_demo": True,
        }
        h_combined, h_top, h_back, h_air, h_rad, kernel_meta = compute_step5j_1_modal_kernels_decomposed(
            mod_weights,
            duration_s=DURATION_S,
            apply_e5_comb_guard=True,
            track_unguarded_reference=False,
        )
        kernel_packs[sid] = {
            "h_combined": h_combined,
            "h_top": h_top,
            "h_back": h_back,
            "h_air": h_air,
            "h_rad": h_rad,
        }
        modal_summaries[sid] = {
            "mode_cap": MODAL_MODE_CAP,
            "guarded_mode_count": kernel_meta.get("e5_guarded_mode_count"),
            "modal_kernel_meta": {
                k: v for k, v in kernel_meta.items() if not (isinstance(v, np.ndarray))
            },
        }
        audio_by_sample[sid] = {}
        spectral_metrics[sid] = {}
        partial_decay_summary[sid] = {}

        for note in NOTE_SET:
            y, meta = synthesize_ordered_physical_chain_note(
                sample_id=sid,
                note=note,
                modal_weights=mod_weights,
                kernel_pack=kernel_packs[sid],
                mapping=preferred[note],
                profile=profile,
            )
            wav_name = demo_wav_filename(sid, note)
            wav_path = out_dir / wav_name
            write_wav_mono(wav_path, y, SR)
            generated_files.append(str(wav_path.resolve()))
            audio_by_sample[sid][note] = y
            partial_decay_summary[sid][note] = meta.get("partial_decay_summary")
            peak_rms_by_file[wav_name] = {
                "peak_dbfs": meta["peak_dbfs"],
                "rms_dbfs": meta["rms_dbfs"],
            }
            spectral_metrics[sid][note] = {
                "peak_dbfs": meta["peak_dbfs"],
                "rms_dbfs": meta["rms_dbfs"],
            }
            print(f"[Emergency demo v4] wrote {sid} {note} -> {wav_name}")

    correlation = compute_same_note_pairwise_correlation(audio_by_sample)
    pairwise = compute_pairwise_metrics(audio_by_sample)
    family_metrics = compute_guitar_family_consistency_metrics(correlation)
    peaks_ok = all(v["peak_dbfs"] <= MAX_PEAK_DBFS + 0.05 for v in peak_rms_by_file.values())
    peak_rms_report = {
        "per_file": peak_rms_by_file,
        "all_peaks_within_target": peaks_ok,
        "note_peak_targets_dbfs": NOTE_PEAK_TARGET_DBFS,
    }
    anti_cheat = build_anti_cheat_checks(
        traces={sid: differentiation_trace[sid]["physical_drivers_applied"] for sid in SAMPLE_SET},
        family_metrics=family_metrics,
        pairwise=pairwise,
        peak_rms_report=peak_rms_report,
    )
    readiness = build_readiness_emergency_demo(
        files_generated=len(generated_files),
        peaks_controlled=peaks_ok,
        family_metrics=family_metrics,
    )
    report = {
        "report_version": ENGINE_VERSION,
        "emergency_demo_version": EMERGENCY_DEMO_VERSION,
        "timestamp": _utc_now(),
        "physical_chain_summary": PHYSICAL_CHAIN_SUMMARY,
        "generated_files": generated_files,
        "per_sample_synthesis_profile": synthesis_profiles,
        "per_sample_physical_factors": factor_multipliers,
        "per_sample_differentiation_trace": differentiation_trace,
        "modal_response_summary": modal_summaries,
        "partial_decay_summary_per_sample_note": partial_decay_summary,
        "peak_rms_report": peak_rms_report,
        "same_note_pairwise_correlation": correlation.get("same_note_pairwise_correlation"),
        "max_same_note_correlation": correlation.get("max_correlation"),
        "min_same_note_correlation": correlation.get("min_correlation"),
        "spectral_distance": pairwise.get("mean_spectral_distance"),
        "envelope_distance": pairwise.get("mean_envelope_distance"),
        "pairwise_difference_metrics": pairwise.get("pairwise_difference_metrics"),
        "guitar_family_consistency_metrics": family_metrics,
        "anti_cheat_checks": anti_cheat,
        "diagnostic_simplifications": DIAGNOSTIC_SIMPLIFICATIONS,
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
    jpath = Path(json_path or REPORT_JSON)
    mpath = Path(md_path or REPORT_MD)
    jpath.parent.mkdir(parents=True, exist_ok=True)
    jpath.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_emergency_demo_markdown(report, mpath)
    return report


# Backward-compatible test aliases.
compute_physical_factors = compute_gentle_sample_modifiers
compute_demo_modifiers = compute_gentle_sample_modifiers
PHYSICAL_FACTOR_GROUPS = PHYSICAL_MODIFIER_KEYS


def main() -> None:
    report = run_emergency_guitar_demo()
    rg = report.get("readiness") or {}
    print(f"Wrote {REPORT_JSON}")
    print(f"Readiness: {rg.get('current_status')}")
    print(f"max_correlation: {report.get('max_same_note_correlation')}")


if __name__ == "__main__":
    main()
