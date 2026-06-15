#!/usr/bin/env python3
"""
PGSM Step 5K — minimal bridge/admittance feedback coupling diagnostic.
Couples Step 5I.3 string force to Step 5J.1 body kernels via admittance-derived C_bridge(ω).
Diagnostic only — not final synthesis, not STK, not multi-guitar proof.
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
from pgsm_step3a_numerical_ir_testbench import NUMERIC_SR, SAMPLE_ID, compute_admittance_curve
from pgsm_step4a_single_note_diagnostic_audio import (
    build_calibrated_modal_state,
    synthesize_modal_body_response,
    write_wav_mono,
)
from pgsm_step5a_limited_note_set_diagnostic_audio import NOTE_FREQUENCY_HZ, NOTE_SET
from pgsm_step5e_string_driven_bridge_force_repair import (
    ENERGY_FIRST_10MS_MAX,
    build_artifact_guard,
    evaluate_modal_peak_alignment,
    compute_pitch_salience,
    detect_second_onset_sustained,
    compute_click_dominance_score,
    _energy_share_first_ms,
    _linear_to_dbfs,
    _rms,
)
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
    READINESS_DOCUMENTED_E5_COMB_LIMITATION,
    apply_listening_render_step5j_1,
    collect_all_previous_audio_fingerprints,
    compute_comb_echo_score,
    compute_stem_comb_echo_scores,
    compute_step5j_1_modal_kernels_decomposed,
)
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE

PGSM_STEP5K_VERSION = "pgsm_step5k_bridge_admittance_feedback_coupling_v1"
VALIDATION_MAX_MODES = 100
FAST_VALIDATION_MAX_MODES = 40
FAST_VALIDATION_DURATION_S = 1.0

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON = REPO_ROOT / "audio" / "debug_reports" / "pgsm_step5k_bridge_admittance_feedback_coupling.json"
REPORT_MD = REPO_ROOT / "audio" / "debug_reports" / "pgsm_step5k_bridge_admittance_feedback_coupling.md"
SOURCE_CONTRACT_JSON = REPO_ROOT / "data" / "pgsm_bridge_admittance_coupling_contract.json"
GENERATED_CONTRACT_JSON = (
    REPO_ROOT
    / "audio"
    / "debug_reports"
    / "generated_contracts"
    / "pgsm_bridge_admittance_coupling_contract.generated.json"
)
AUDIO_DIR = REPO_ROOT / "audio" / "pgsm_step5k_bridge_admittance_feedback_coupling"

READINESS_AFTER = "ready_for_step5l_limited_multiguitar_differentiation"
READINESS_PARTIAL = (
    "step5k_diagnostic_partial_improvement_ready_for_multiguitar_with_known_e5_limitation"
)
READINESS_INSUFFICIENT = "bridge_coupling_minimal_proxy_insufficient"
SAFE_NEXT_STEP_5L = "step5l_limited_multiguitar_differentiation"

E5_LOW_BODY_LO_HZ = 55.0
E5_LOW_BODY_HI_HZ = 220.0
COUPLING_ALPHA = 0.42
MIN_MODE_COUPLING_FACTOR = 0.72
COUPLING_IR_DECAY_S = 0.12


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_bridge_admittance_coupling_contract() -> Dict[str, Any]:
    return {
        "contract_id": "pgsm_bridge_admittance_coupling_v1",
        "model": "F_bridge_eff(ω) = F_string(ω) × C_bridge(ω)",
        "source_inputs": [
            "pgsm_step5i_3_absolute_frequency_damping_pluck_balance string/pluck force",
            "pgsm_step5j_1 top/back/air/radiation body kernels",
            "step3c modal W_exc and bridge_excitation_coupling proxies",
        ],
        "forbidden": [
            "stk_integration",
            "fem_run",
            "rom_run",
            "artificial_reverb",
            "body_tail",
            "arbitrary_eq",
            "limiter_clipping",
            "phase_tricks",
        ],
        "terms": [
            {
                "term": "bridge_admittance_proxy",
                "formula": "Y_bridge(ω) = Σ_i W_exc,i × mobility / ((ω_i²−ω²)+j2ζω_iω)",
                "source_level": "L2_step3a_admittance_curve",
            },
            {
                "term": "mode_coupling_factor",
                "formula": "c_i = 1/(1+α·Y_i·low_body_comb_risk(f_i,f0)) for E5 low-body band",
                "source_level": "L2_modal_bridge_excitation",
            },
            {
                "term": "causal_coupling_ir",
                "formula": "h_c(t) causal IFFT(C_bridge(ω)) with exponential decay window",
                "source_level": "diagnostic_causal_fir",
            },
            {
                "term": "e5_low_body_mode_comb_target",
                "formula": f"target band {E5_LOW_BODY_LO_HZ}–{E5_LOW_BODY_HI_HZ} Hz per Step 5J.1 documented limitation",
                "validation_metric": "E5 radiation_sum comb before/after",
            },
        ],
        "parameters": {
            "e5_low_body_lo_hz": E5_LOW_BODY_LO_HZ,
            "e5_low_body_hi_hz": E5_LOW_BODY_HI_HZ,
            "coupling_alpha": COUPLING_ALPHA,
            "min_mode_coupling_factor": MIN_MODE_COUPLING_FACTOR,
            "comb_echo_fail_threshold": COMB_ECHO_FAIL_THRESHOLD,
        },
    }


def load_step5j_1_report(repo_root: Path) -> Dict[str, Any]:
    path = repo_root / "audio" / "debug_reports" / "pgsm_step5j_1_guitar_articulation_body_balance_repair.json"
    if not path.is_file():
        return {"loaded": False, "error": f"missing {path}"}
    return json.loads(path.read_text(encoding="utf-8"))


def verify_upstream_step5j_1(step5j_1: Mapping[str, Any]) -> Dict[str, Any]:
    rg = step5j_1.get("readiness_after_step5j_1") or {}
    doc = step5j_1.get("step5j_1_documented_limitation") or {}
    art = step5j_1.get("artifact_guard_results") or {}
    failed = list(art.get("failed_guard_fields") or [])
    e5_flags = (art.get("per_note_flags") or {}).get("E5") or {}
    documented_explicit = bool(step5j_1.get("documented_limitation"))
    documented_inferred = bool(
        rg.get("current_status") == READINESS_DOCUMENTED_E5_COMB_LIMITATION
        or (
            bool(rg.get("bridge_coupling_plan_allowed"))
            and failed == ["no_comb_echo"]
            and e5_flags.get("no_comb_echo") is False
        )
    )
    documented_loaded = documented_explicit or documented_inferred
    return {
        "step5j_1_report_version": step5j_1.get("report_version"),
        "step5j_1_status": step5j_1.get("status"),
        "step5j_1_readiness_status": rg.get("current_status"),
        "documented_limitation_loaded": documented_loaded,
        "documented_limitation_explicit": documented_explicit,
        "documented_limitation_inferred": documented_inferred and not documented_explicit,
        "documented_limitation_type": step5j_1.get("limitation_type")
        or (doc.get("limitation_type") if documented_loaded else None),
        "bridge_coupling_plan_allowed": bool(rg.get("bridge_coupling_plan_allowed")),
        "e5_comb_baseline_score": (art.get("comb_echo_score_by_note") or {}).get("E5")
        or e5_flags.get("comb_echo_score"),
        "e5_guard_applied_at_step5j_1": (step5j_1.get("E5_radiation_guard_analysis") or {}).get(
            "e5_radiation_guard_applied"
        ),
        "pass": bool(
            documented_loaded
            and rg.get("bridge_coupling_plan_allowed")
        ),
        "documented_limitation_detail": doc if doc else None,
    }


def _e5_low_body_comb_risk(f_hz: float, f0: float, y_proxy: float, w_exc: float) -> float:
    harm_overlap = sum(
        math.exp(-((f_hz - k * f0) ** 2) / (2.0 * (f0 * 0.08) ** 2)) for k in range(1, 4)
    )
    y_norm = y_proxy / max(w_exc, 1e-12)
    band_risk = (f_hz / max(E5_LOW_BODY_HI_HZ, 1.0)) ** 0.5
    return band_risk * y_norm * (0.35 + harm_overlap)


def compute_mode_bridge_coupling_factors(
    modal_weights: Mapping[str, Any],
    *,
    note: str,
    f0: float,
) -> List[Dict[str, Any]]:
    modes = modal_weights.get("modes") or []
    mob_amp = float(modal_weights.get("mobility_amplitude") or 1.0)
    records: List[Dict[str, Any]] = []
    for idx, row in enumerate(modes):
        f_i = float(row["frequency_hz"])
        w_exc = float(row.get("W_exc") or 0.0) * mob_amp
        bridge_c = float(
            row.get("bridge_excitation_coupling") or row.get("bridge_excitation_abs") or w_exc
        )
        y_proxy = w_exc * max(bridge_c, 1e-12)
        factor = 1.0
        if note == "E5" and E5_LOW_BODY_LO_HZ <= f_i <= E5_LOW_BODY_HI_HZ:
            risk = _e5_low_body_comb_risk(f_i, f0, y_proxy, w_exc)
            factor = 1.0 / (1.0 + COUPLING_ALPHA * risk)
            factor = max(MIN_MODE_COUPLING_FACTOR, min(1.0, factor))
        elif f_i < 250.0:
            factor = max(0.85, min(1.0, 1.0 - 0.08 * (y_proxy / max(w_exc, 1e-12))))
        records.append(
            {
                "mode_index": idx,
                "frequency_hz": round(f_i, 3),
                "W_exc": round(w_exc, 6),
                "bridge_excitation_proxy": round(bridge_c, 6),
                "Y_i_proxy": round(y_proxy, 6),
                "coupling_factor": round(factor, 6),
            }
        )
    return records


def build_causal_bridge_coupling_ir(
    mode_factors: Sequence[Mapping[str, Any]],
    *,
    duration_s: float,
    sr: int,
) -> np.ndarray:
    n = int(duration_s * sr)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    c_bridge = np.ones(len(freqs), dtype=np.float64)
    for rec in mode_factors:
        f_i = float(rec["frequency_hz"])
        c_i = float(rec["coupling_factor"])
        for j, f in enumerate(freqs):
            if f < 1.0:
                continue
            spread = math.exp(-((f - f_i) ** 2) / (2.0 * 14.0 ** 2))
            c_bridge[j] *= 1.0 + (c_i - 1.0) * spread
    h = np.fft.irfft(c_bridge, n=n).astype(np.float64)
    win = np.exp(-np.arange(n, dtype=np.float64) / max(sr * COUPLING_IR_DECAY_S, 1e-6))
    h *= win
    dc = float(np.sum(h))
    if abs(dc) > 1e-12:
        h /= dc
    return h


def apply_bridge_admittance_coupling(
    string_force: np.ndarray,
    modal_weights: Mapping[str, Any],
    *,
    sr: int,
    note: str,
    f0: float,
    duration_s: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    mode_factors = compute_mode_bridge_coupling_factors(modal_weights, note=note, f0=f0)
    h_c = build_causal_bridge_coupling_ir(mode_factors, duration_s=duration_s, sr=sr)
    f_eff = np.convolve(string_force.astype(np.float64), h_c, mode="same")
    force_delta = float(np.linalg.norm(f_eff - string_force) / max(np.linalg.norm(string_force), 1e-12))
    guarded = [r for r in mode_factors if float(r["coupling_factor"]) < 0.999]
    return f_eff, {
        "mode_coupling_factors": mode_factors,
        "guarded_mode_count": len(guarded),
        "mean_coupling_factor": round(
            float(np.mean([float(r["coupling_factor"]) for r in mode_factors])) if mode_factors else 1.0,
            6,
        ),
        "min_coupling_factor": round(
            min((float(r["coupling_factor"]) for r in mode_factors), default=1.0),
            6,
        ),
        "force_delta_l2_relative": round(force_delta, 6),
        "coupling_applied": force_delta > 1e-6,
        "h_coupling_peak": round(float(np.max(np.abs(h_c))), 6),
        "causal_ir_length_samples": len(h_c),
    }


def synthesize_note_paths(
    string_force: np.ndarray,
    *,
    h_top: np.ndarray,
    h_back: np.ndarray,
    h_air: np.ndarray,
    h_rad: np.ndarray,
    h_combined: np.ndarray,
    sr: int,
    note: str,
) -> Dict[str, np.ndarray]:
    y_top = synthesize_modal_body_response(string_force, h_top)
    y_back = synthesize_modal_body_response(string_force, h_back)
    y_air = synthesize_modal_body_response(string_force, h_air)
    y_rad = synthesize_modal_body_response(string_force, h_rad)
    body_raw = synthesize_modal_body_response(string_force, h_combined)
    main_listening, _ = apply_listening_render_step5j_1(body_raw, note=note)
    return {
        "top": y_top,
        "back": y_back,
        "air": y_air,
        "radiation": y_rad,
        "body_raw": body_raw,
        "main": main_listening,
    }


def _stem_balance_snapshot(stems: Mapping[str, np.ndarray], string_force: np.ndarray) -> Dict[str, float]:
    e_top = float(np.sum(stems["top"].astype(np.float64) ** 2))
    e_back = float(np.sum(stems["back"].astype(np.float64) ** 2))
    e_air = float(np.sum(stems["air"].astype(np.float64) ** 2))
    body_sum = e_top + e_back + e_air
    e_rad = float(np.sum(stems["radiation"].astype(np.float64) ** 2))
    return {
        "top_share": round(e_top / max(body_sum, 1e-12), 6),
        "back_share": round(e_back / max(body_sum, 1e-12), 6),
        "air_share": round(e_air / max(body_sum, 1e-12), 6),
        "radiation_sum_energy": round(e_rad, 6),
    }


def evaluate_note_metrics(
    main: np.ndarray,
    *,
    sr: int,
    note: str,
    f0: float,
    modal_freqs: Sequence[float],
    h_body: np.ndarray,
) -> Dict[str, Any]:
    e10 = _energy_share_first_ms(main, sr, 10.0)
    pitch_sal = compute_pitch_salience(main, sr, f0)
    peak = float(np.max(np.abs(main)))
    rms_db = _linear_to_dbfs(_rms(main))
    second_onset = detect_second_onset_sustained(main, sr)
    click_score = compute_click_dominance_score(main, sr, energy_first_10ms=e10)
    modal = evaluate_modal_peak_alignment(main, sr, modal_freqs, f0=f0, h_body=h_body, pitch_salience=pitch_sal)
    return {
        "note": note,
        "peak_dbfs": round(_linear_to_dbfs(peak), 3),
        "rms_dbfs": round(rms_db, 3),
        "energy_first_10ms": round(e10, 4),
        "no_comb_echo": modal.get("no_comb_echo"),
        "comb_echo_score": modal.get("echo_comb_pattern_score"),
        "no_second_onset": not second_onset,
        "not_click_dominant": click_score < 0.45,
        "pass": bool(
            e10 < ENERGY_FIRST_10MS_MAX
            and not second_onset
            and click_score < 0.45
            and modal.get("pass")
        ),
    }


def build_bridge_admittance_proxy_summary(
    modal_weights: Mapping[str, Any],
    mode_factors: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    adm = compute_admittance_curve(modal_weights)
    low_body = [
        r for r in mode_factors if E5_LOW_BODY_LO_HZ <= float(r["frequency_hz"]) <= E5_LOW_BODY_HI_HZ
    ]
    return {
        "admittance_status": adm.get("status"),
        "max_abs_Y_bridge": adm.get("max_abs_Y"),
        "peak_count": adm.get("peak_count"),
        "band_summary": adm.get("band_summary"),
        "low_body_mode_count": len(low_body),
        "low_body_mean_coupling_factor": round(
            float(np.mean([float(r["coupling_factor"]) for r in low_body])) if low_body else 1.0,
            6,
        ),
        "low_body_min_coupling_factor": round(
            min((float(r["coupling_factor"]) for r in low_body), default=1.0),
            6,
        ),
        "coupling_alpha": COUPLING_ALPHA,
        "model": "F_bridge_eff(ω) = F_string(ω) × C_bridge(ω)",
    }


def build_readiness_after_step5k(
    *,
    objective_pass: bool,
    e5_comb_improved: bool,
    e5_comb_pass: bool,
) -> Dict[str, Any]:
    if objective_pass and e5_comb_pass:
        status = READINESS_AFTER
    elif e5_comb_improved:
        status = READINESS_PARTIAL
    else:
        status = READINESS_INSUFFICIENT
    return {
        "current_status": status,
        "final_synthesis_ready": False,
        "stk_integration_allowed": False,
        "website_production_replacement_allowed": False,
        "multi_guitar_comparison_allowed": False,
        "melody_chord_playback_allowed": False,
        "real_guitar_equivalence_allowed": False,
        "bridge_coupling_diagnostic_completed": True,
        "step5l_multiguitar_planning_allowed": True,
    }


def validate_report_internal_consistency(report: Mapping[str, Any]) -> Dict[str, Any]:
    objective = report.get("objective_test_results") or {}
    artifact = report.get("artifact_guard_results") or {}
    readiness = report.get("readiness_after_step5k") or {}
    issues: List[str] = []
    if objective.get("artifact_guard_pass") != artifact.get("pass"):
        issues.append("objective.artifact_guard_pass != artifact_guard_results.pass")
    if not report.get("documented_limitation_loaded"):
        issues.append("documented_limitation_loaded must be true")
    if not report.get("bridge_coupling_plan_allowed"):
        issues.append("bridge_coupling_plan_allowed must be true")
    if readiness.get("final_synthesis_ready"):
        issues.append("final_synthesis_ready must remain false")
    if readiness.get("stk_integration_allowed"):
        issues.append("stk_integration_allowed must remain false")
    if readiness.get("multi_guitar_comparison_allowed"):
        issues.append("multi_guitar_comparison_allowed must remain false until Step 5L")
    if report.get("safe_next_step") != SAFE_NEXT_STEP_5L:
        issues.append("safe_next_step must point to Step 5L")
    return {"pass": not issues, "issues": issues}


def build_pgsm_step5k_report(
    *,
    repo_root: Optional[Path] = None,
    audio_dir: Optional[Path] = None,
    render_audio: bool = True,
    write_outputs: bool = False,
    fast_validation: bool = False,
    max_modes: Optional[int] = None,
    duration_s: float = DEFAULT_DURATION_S,
) -> Dict[str, Any]:
    root = Path(repo_root or REPO_ROOT)
    out_audio = Path(audio_dir or AUDIO_DIR)
    if fast_validation:
        render = False
        if duration_s == DEFAULT_DURATION_S:
            duration_s = FAST_VALIDATION_DURATION_S
    else:
        render = render_audio
    mode_cap = (
        max_modes
        if max_modes is not None
        else (FAST_VALIDATION_MAX_MODES if fast_validation else VALIDATION_MAX_MODES)
    )
    validation_mode = "fast" if fast_validation else "full"

    step5j_1 = load_step5j_1_report(root)
    upstream = verify_upstream_step5j_1(step5j_1)
    step5h = load_step_report(_report_path(root, "pgsm_step5h_note_string_fret_contract.json"))
    contract_data_path = root / "data" / "pgsm_classical_guitar_note_string_fret_contract.json"
    contract_data = json.loads(contract_data_path.read_text(encoding="utf-8")) if contract_data_path.is_file() else None
    preferred = load_preferred_mappings(step5h, contract_data)
    coupling_contract = build_bridge_admittance_coupling_contract()

    fp_before = collect_all_previous_audio_fingerprints(root)
    state = build_calibrated_modal_state(root, max_modes=mode_cap)
    cal_weights = state["modal_weights"]
    modal_freqs = [float(m["frequency_hz"]) for m in cal_weights.get("modes") or []]
    sr = NUMERIC_SR

    h_combined, h_top, h_back, h_air, h_rad, kernel_meta = compute_step5j_1_modal_kernels_decomposed(
        cal_weights,
        duration_s=duration_s,
        apply_e5_comb_guard=True,
        track_unguarded_reference=False,
    )

    per_note_coupling: Dict[str, Any] = {}
    per_note_metrics: Dict[str, Any] = {}
    e5_before_after: Dict[str, Any] = {"applicable": False}
    e5_rad_before_after: Dict[str, Any] = {"applicable": False}
    balance_before_after: Dict[str, Any] = {}
    output_files: Dict[str, Any] = {"notes": {}}
    e5_mode_factors: List[Dict[str, Any]] = []
    e5_coupling_meta: Dict[str, Any] = {}

    for note in NOTE_SET:
        mapping = preferred[note]
        f0 = float(mapping.get("target_frequency_hz") or NOTE_FREQUENCY_HZ[note])
        string_id = str(mapping["string_id"])
        fret = int(mapping["fret"])
        n = int(duration_s * sr)
        string_force, pluck_stem, _ = build_v4_string_bridge_force(
            n, sr, f0, string_id=string_id, fret=fret, note=note
        )

        before_stems = synthesize_note_paths(
            string_force,
            h_top=h_top,
            h_back=h_back,
            h_air=h_air,
            h_rad=h_rad,
            h_combined=h_combined,
            sr=sr,
            note=note,
        )
        f_eff, coupling_meta = apply_bridge_admittance_coupling(
            string_force,
            cal_weights,
            sr=sr,
            note=note,
            f0=f0,
            duration_s=duration_s,
        )
        after_stems = synthesize_note_paths(
            f_eff,
            h_top=h_top,
            h_back=h_back,
            h_air=h_air,
            h_rad=h_rad,
            h_combined=h_combined,
            sr=sr,
            note=note,
        )

        before_balance = _stem_balance_snapshot(before_stems, string_force)
        after_balance = _stem_balance_snapshot(after_stems, f_eff)
        balance_before_after[note] = {
            "before": before_balance,
            "after": after_balance,
            "top_share_delta": round(after_balance["top_share"] - before_balance["top_share"], 6),
            "back_share_delta": round(after_balance["back_share"] - before_balance["back_share"], 6),
            "air_share_delta": round(after_balance["air_share"] - before_balance["air_share"], 6),
        }

        before_comb_stems = (
            {"final_output": round(compute_comb_echo_score(before_stems["main"], sr), 4)}
            if fast_validation and note != "E5"
            else compute_stem_comb_echo_scores(
                string_force=string_force,
                y_top=before_stems["top"],
                y_back=before_stems["back"],
                y_air=before_stems["air"],
                y_rad=before_stems["radiation"],
                y_combined=before_stems["main"],
                sr=sr,
                note=note,
            )
        )
        after_comb_stems = (
            compute_stem_comb_echo_scores(
                string_force=f_eff,
                y_top=after_stems["top"],
                y_back=after_stems["back"],
                y_air=after_stems["air"],
                y_rad=after_stems["radiation"],
                y_combined=after_stems["main"],
                sr=sr,
                note=note,
            )
            if note == "E5"
            else {"final_output": round(compute_comb_echo_score(after_stems["main"], sr), 4)}
        )

        metrics = evaluate_note_metrics(
            after_stems["main"],
            sr=sr,
            note=note,
            f0=f0,
            modal_freqs=modal_freqs,
            h_body=h_combined,
        )
        per_note_metrics[note] = metrics
        per_note_coupling[note] = {
            **coupling_meta,
            "comb_before": before_comb_stems,
            "comb_after": after_comb_stems,
            "comb_final_output_delta": round(
                float(after_comb_stems.get("final_output") or 0)
                - float(before_comb_stems.get("final_output") or 0),
                4,
            ),
        }

        if note == "E5":
            e5_mode_factors = list(coupling_meta.get("mode_coupling_factors") or [])
            e5_coupling_meta = dict(coupling_meta)
            e5_before_after = {
                "applicable": True,
                "comb_score_before": before_comb_stems.get("final_output"),
                "comb_score_after": after_comb_stems.get("final_output"),
                "comb_improved": float(after_comb_stems.get("final_output") or 1)
                < float(before_comb_stems.get("final_output") or 0) - 1e-4,
                "comb_pass": bool(metrics.get("no_comb_echo")),
                "radiation_sum_comb_before": before_comb_stems.get("radiation_sum"),
                "radiation_sum_comb_after": after_comb_stems.get("radiation_sum"),
                "radiation_sum_improved": float(after_comb_stems.get("radiation_sum") or 1)
                < float(before_comb_stems.get("radiation_sum") or 0) - 1e-4,
            }
            e5_rad_before_after = {
                "applicable": True,
                "before": before_comb_stems.get("radiation_sum"),
                "after": after_comb_stems.get("radiation_sum"),
                "delta": round(
                    float(after_comb_stems.get("radiation_sum") or 0)
                    - float(before_comb_stems.get("radiation_sum") or 0),
                    4,
                ),
            }

        if render and not fast_validation:
            out_audio.mkdir(parents=True, exist_ok=True)
            base = f"{SAMPLE_ID}_{note}"
            main_path = out_audio / f"{base}_bridge_coupling_diagnostic.wav"
            write_wav_mono(main_path, after_stems["main"], sr)
            output_files["notes"][note] = {"main": str(main_path), "rendered": True}
        else:
            output_files["notes"][note] = {"rendered": False}

    fp_after = collect_all_previous_audio_fingerprints(root)
    artifact = build_artifact_guard(per_note_metrics)
    adm_summary = build_bridge_admittance_proxy_summary(cal_weights, e5_mode_factors)

    e5_improved = bool(e5_before_after.get("comb_improved"))
    e5_comb_pass = bool(e5_before_after.get("comb_pass"))
    objective = {
        "upstream_step5j_1_ready": upstream.get("pass"),
        "no_previous_audio_modified": fp_before == fp_after,
        "coupling_contract_complete": len(coupling_contract.get("terms") or []) >= 4,
        "coupling_applied_on_e5": bool(e5_coupling_meta.get("coupling_applied")),
        "e5_force_path_modified": float(e5_coupling_meta.get("force_delta_l2_relative") or 0) > 1e-6,
        "e5_comb_before_after_computed": e5_before_after.get("applicable") is True,
        "e5_comb_improved": e5_improved,
        "e5_radiation_sum_before_after_computed": e5_rad_before_after.get("applicable") is True,
        "balance_before_after_computed": bool(balance_before_after),
        "artifact_guard_pass": artifact.get("pass"),
        "no_stk_integration": True,
        "no_fem_run": True,
        "no_rom_run": True,
        "website_default_unchanged": True,
    }
    objective["all_pass"] = bool(all(objective.values()))
    readiness = build_readiness_after_step5k(
        objective_pass=objective["all_pass"],
        e5_comb_improved=e5_improved,
        e5_comb_pass=e5_comb_pass,
    )

    validation_config = {
        "validation_mode": validation_mode,
        "render_audio": render,
        "write_outputs": write_outputs,
        "validation_max_modes": mode_cap,
        "fast_validation_max_modes": FAST_VALIDATION_MAX_MODES,
        "full_validation_max_modes": VALIDATION_MAX_MODES,
        "duration_s": duration_s,
        "fast_duration_s": FAST_VALIDATION_DURATION_S if fast_validation else None,
        "full_duration_s": DEFAULT_DURATION_S,
        "tracked_source_files_modified": False,
        "source_contract_path": str(SOURCE_CONTRACT_JSON),
        "generated_contract_path": str(GENERATED_CONTRACT_JSON),
        "audio_render_skipped": not render,
    }

    report_body: Dict[str, Any] = {
        "report_version": PGSM_STEP5K_VERSION,
        "timestamp": _utc_now(),
        "validation_max_modes": mode_cap,
        "validation_config": validation_config,
        "status": "pgsm_step5k_bridge_admittance_feedback_coupling_diagnostic_complete",
        "upstream_step5j_1_status": upstream,
        "documented_limitation_loaded": upstream.get("documented_limitation_loaded"),
        "bridge_coupling_plan_allowed": upstream.get("bridge_coupling_plan_allowed"),
        "no_stk_integration": True,
        "no_fem_run": True,
        "no_rom_run": True,
        "website_default": DEFAULT_WEBSITE_STK_MODE,
        "website_default_unchanged": True,
        "no_previous_audio_modified": fp_before == fp_after,
        "coupling_contract": coupling_contract,
        "bridge_admittance_proxy_summary": adm_summary,
        "modal_kernel_meta": {
            k: v for k, v in kernel_meta.items() if not (k.startswith("h_") and hasattr(v, "shape"))
        },
        "per_note_coupling_metrics": per_note_coupling,
        "E5_comb_before_after": e5_before_after,
        "E5_radiation_sum_before_after": e5_rad_before_after,
        "stem_balance_before_after": balance_before_after,
        "per_note_metrics": per_note_metrics,
        "artifact_guard_results": artifact,
        "objective_test_results": objective,
        "validation_results": objective,
        "readiness": readiness,
        "readiness_after_step5k": readiness,
        "safe_next_step": SAFE_NEXT_STEP_5L,
        "explicit_statement": (
            "PGSM Step 5K applies minimal causal bridge/admittance coupling between Step 5I.3 "
            "string force and Step 5J.1 body kernels. Diagnostic only — not final synthesis, "
            "not STK integration, not multi-guitar proof."
        ),
        "step5k_limitation_note": (
            "If E5 comb remains after coupling, output weighting plus minimal admittance proxy "
            "is insufficient; proceed to Step 5L limited multi-guitar differentiation with limitation documented."
        ),
        "generated_files": output_files,
        "blocked_claims": [
            "Final synthesis",
            "STK integration",
            "Website production replacement",
            "Multi-guitar equivalence proof",
            "Real-guitar equivalence",
        ],
    }
    report_body["internal_consistency_check"] = validate_report_internal_consistency(report_body)
    return report_body


def write_markdown_report(report: Mapping[str, Any], path: Path) -> None:
    rg = report.get("readiness_after_step5k") or {}
    obj = report.get("objective_test_results") or {}
    e5 = report.get("E5_comb_before_after") or {}
    adm = report.get("bridge_admittance_proxy_summary") or {}
    vcfg = report.get("validation_config") or {}
    lines = [
        "# PGSM Step 5K — bridge/admittance feedback coupling",
        "",
        f"**Readiness:** `{rg.get('current_status')}`",
        f"**Safe next step:** `{report.get('safe_next_step')}`",
        "",
        f"**Validation:** `{vcfg.get('validation_mode')}` "
        f"(render_audio={vcfg.get('render_audio')}, max_modes={vcfg.get('validation_max_modes')})",
        "",
        report.get("explicit_statement", ""),
        "",
        "## Upstream Step 5J.1",
        "",
        f"- documented_limitation_loaded: **{report.get('documented_limitation_loaded')}**",
        f"- bridge_coupling_plan_allowed: **{report.get('bridge_coupling_plan_allowed')}**",
        "",
        "## E5 comb before/after",
        "",
        f"- comb before: **{e5.get('comb_score_before')}**",
        f"- comb after: **{e5.get('comb_score_after')}**",
        f"- comb improved: **{e5.get('comb_improved')}**",
        f"- radiation_sum before/after: **{e5.get('radiation_sum_comb_before')}** / "
        f"**{e5.get('radiation_sum_comb_after')}**",
        "",
        "## Bridge admittance proxy",
        "",
        f"- model: {adm.get('model')}",
        f"- low_body_mode_count: {adm.get('low_body_mode_count')}",
        f"- low_body_mean_coupling_factor: {adm.get('low_body_mean_coupling_factor')}",
        "",
        "## Objective",
        "",
        f"- all_pass: **{obj.get('all_pass')}**",
        f"- artifact_guard_pass: **{obj.get('artifact_guard_pass')}**",
        f"- e5_comb_improved: **{obj.get('e5_comb_improved')}**",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pgsm_step5k_reports(
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
    duration_s: float = DEFAULT_DURATION_S,
) -> Dict[str, Any]:
    root = Path(repo_root or REPO_ROOT)
    render = False if fast_validation else render_audio
    report = build_pgsm_step5k_report(
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
        export = {
            "contract_version": PGSM_STEP5K_VERSION,
            "coupling_contract": report["coupling_contract"],
        }
        contract_dest.write_text(json.dumps(export, indent=2), encoding="utf-8")
    return report


def main() -> None:
    report = write_pgsm_step5k_reports(
        render_audio=True,
        write_outputs=True,
        fast_validation=False,
        max_modes=VALIDATION_MAX_MODES,
        data_path=GENERATED_CONTRACT_JSON,
    )
    rg = report.get("readiness_after_step5k") or {}
    obj = report.get("objective_test_results") or {}
    print(f"Wrote {REPORT_JSON}")
    print(f"Readiness: {rg.get('current_status')}")
    print(f"all_pass: {obj.get('all_pass')}")
    print(f"safe_next_step: {report.get('safe_next_step')}")


if __name__ == "__main__":
    main()
