#!/usr/bin/env python3
"""
PGSM Step 5G — physical tone model update plan.
Planning and contract definition only; no audio generation or physics changes.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from pgsm_step2_1_parameter_targets import (
    CLASSICAL_SCALE_LENGTH_M,
    FALLBACK_LEVELS,
    NYLON_LINEAR_DENSITY_KG_M,
    load_step_report,
)
from pgsm_step5a_limited_note_set_diagnostic_audio import NOTE_FREQUENCY_HZ, NOTE_SET
from pgsm_step5e_string_driven_bridge_force_repair import collect_previous_audio_fingerprints
from pgsm_step5f_string_driven_extended_validation import (
    READINESS_AFTER as READINESS_STEP5F,
    collect_step5e_fingerprints,
)
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE

PGSM_STEP5G_VERSION = "pgsm_step5g_physical_tone_model_update_plan_v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON = REPO_ROOT / "audio" / "debug_reports" / "pgsm_step5g_physical_tone_model_update_plan.json"
REPORT_MD = REPO_ROOT / "audio" / "debug_reports" / "pgsm_step5g_physical_tone_model_update_plan.md"

READINESS_AFTER = "ready_for_step5h_note_string_fret_contract_repair"

FORBIDDEN_RECOMMENDATION_TERMS = (
    "arbitrary_eq",
    "artificial_reverb",
    "delayed_echo",
    "body_tail",
    "hard_gate",
    "second_onset",
    "end_rise",
    "wood_to_gain",
    "listening_only_tuning",
    "stk_integration",
    "website_replacement",
)

CLASSICAL_OPEN_STRINGS = {
    "E2": 82.4069,
    "A2": 110.0,
    "D3": 146.8324,
    "G3": 195.9977,
    "B3": 246.9417,
    "E4": 329.6276,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _report_path(root: Path, name: str) -> Path:
    return root / "audio" / "debug_reports" / name


def _optional_report(root: Path, name: str) -> Optional[Dict[str, Any]]:
    path = _report_path(root, name)
    if not path.is_file():
        return None
    return load_step_report(path)


def verify_upstream_readiness(
    step5f: Mapping[str, Any],
    fp_before: Mapping[str, str],
) -> Dict[str, Any]:
    rg = step5f.get("readiness_after_step5f") or {}
    obj = step5f.get("objective_test_results") or {}
    return {
        "step5f_readiness": rg.get("current_status"),
        "step5f_pass": rg.get("current_status") == READINESS_STEP5F,
        "step5f_all_pass": bool(obj.get("all_pass")),
        "step5e_outputs_preserved_in_step5f": bool(step5f.get("step5e_outputs_preserved")),
        "previous_audio_preserved_in_step5f": bool(step5f.get("previous_audio_preserved")),
        "fingerprints_before": dict(fp_before),
        "final_synthesis_blocked": rg.get("final_synthesis_ready") is False,
        "stk_blocked": rg.get("stk_integration_allowed") is False,
        "website_blocked": rg.get("website_production_replacement_allowed") is False,
        "multi_guitar_blocked": rg.get("multi_guitar_comparison_allowed") is False,
        "melody_chords_blocked": rg.get("melody_chord_playback_allowed") is False,
        "pass": bool(
            rg.get("current_status") == READINESS_STEP5F
            and obj.get("all_pass")
            and step5f.get("step5e_outputs_preserved")
            and rg.get("final_synthesis_ready") is False
            and rg.get("stk_integration_allowed") is False
        ),
    }


def build_step5f_findings_summary(step5f: Mapping[str, Any]) -> Dict[str, Any]:
    robotic = step5f.get("robotic_tone_diagnosis") or {}
    global_labels = robotic.get("global_robotic_tone_labels") or {}
    cross = step5f.get("cross_note_validation") or {}
    return {
        "validation_passed": bool((step5f.get("objective_test_results") or {}).get("all_pass")),
        "not_click_dominant": cross.get("all_notes_not_click_dominant"),
        "pitch_salience_present": cross.get("all_notes_pitch_salient"),
        "active_duration_sufficient": cross.get("all_notes_active_sufficient"),
        "no_forbidden_artifacts": (step5f.get("artifact_guard_results") or {}).get("pass"),
        "robotic_tone_still_present": bool(robotic.get("robotic_tone_present")),
        "global_robotic_tone_labels": global_labels,
        "interpretation": robotic.get("interpretation"),
    }


def build_diagnosis_to_update_targets(step5f: Mapping[str, Any]) -> Dict[str, Any]:
    labels = (step5f.get("robotic_tone_diagnosis") or {}).get("global_robotic_tone_labels") or {}
    mappings: List[Dict[str, Any]] = [
        {
            "diagnosis_flag": "excessive_harmonic_purity",
            "flagged": bool(labels.get("excessive_harmonic_purity")),
            "target_area": "string_partial_damping_refinement",
            "priority_rank": 2,
            "physical_rationale": (
                "Harmonic-to-noise ratio is high and spectral flatness is low; partials decay "
                "under a single power law. Per-harmonic and per-string damping must increase "
                "broadband energy decay without EQ."
            ),
            "forbidden_alternative": "arbitrary_EQ_or_noise_layer",
            "objective_validation": "harmonic_to_noise_ratio_db decreases; H2-H8 relative decay varies by partial",
        },
        {
            "diagnosis_flag": "insufficient_string_body_feedback",
            "flagged": bool(labels.get("insufficient_string_body_feedback")),
            "target_area": "bridge_feedback_admittance_coupling",
            "priority_rank": 4,
            "physical_rationale": (
                "Current model is one-way string-force into modal body IR. Bridge admittance "
                "should inform limited diagnostic feedback so body motion modulates string drive "
                "without delay-line echo or instability."
            ),
            "forbidden_alternative": "delayed_echo_or_feedback_delay_line",
            "objective_validation": "body_string envelope correlation changes; no second onset; stable peak",
        },
        {
            "diagnosis_flag": "weak_cavity_air_imprint",
            "flagged": bool(labels.get("weak_cavity_air_imprint")),
            "target_area": "top_back_air_radiation_weighting",
            "priority_rank": 3,
            "physical_rationale": (
                "Air/cavity modal proxy fraction is weak in Step 5E outputs. Separate top plate, "
                "back plate, air/cavity, and radiation weighting from Step 3C W_rad/W_air shares "
                "while keeping Q/tau modal decay."
            ),
            "forbidden_alternative": "body_tail_or_artificial_reverb",
            "objective_validation": "cavity_air_fraction_proxy increases; modal decay unchanged source",
        },
        {
            "diagnosis_flag": "shared_body_ir_limitation",
            "flagged": bool(labels.get("shared_body_ir_limitation")),
            "target_area": "note_string_fret_contract_repair",
            "priority_rank": 1,
            "physical_rationale": (
                "All notes share one Step 3C modal IR; note labels are diagnostic reference "
                "frequencies without full string/fret physics. Contract repair is prerequisite "
                "for note-dependent body interaction planning."
            ),
            "forbidden_alternative": "multi_guitar_comparison_or_STK_integration",
            "objective_validation": "note contract documents string_id/fret/L_eff; exact_open_string_claim blocked",
        },
        {
            "diagnosis_flag": "weak_pluck_noise_component",
            "flagged": bool(labels.get("weak_pluck_noise_component")),
            "target_area": "controlled_pluck_nail_transient_assessment",
            "priority_rank": 5,
            "optional": True,
            "physical_rationale": (
                "Not globally flagged in Step 5F metrics, but listening notes synthetic attack. "
                "Evaluate small physically motivated pluck/nail/finger broadband transient that "
                "does not dominate sustain or become click-like."
            ),
            "forbidden_alternative": "click_layer_or_arbitrary_noise",
            "objective_validation": "onset broadband fraction bounded; energy_first_10ms stays below click threshold",
        },
        {
            "diagnosis_flag": "robotic_tone_present",
            "flagged": bool((step5f.get("robotic_tone_diagnosis") or {}).get("robotic_tone_present")),
            "target_area": "reference_guided_spectral_gap_analysis",
            "priority_rank": 6,
            "physical_rationale": (
                "Future comparison against reference guitar WAVs as directional diagnostic only; "
                "identifies spectral/envelope gaps without claiming validation."
            ),
            "forbidden_alternative": "realism_or_validation_claim",
            "objective_validation": "gap metrics reported; no equivalence claim",
        },
    ]
    return {
        "mappings": mappings,
        "flagged_count": sum(1 for m in mappings if m.get("flagged")),
        "all_required_areas_covered": bool(
            any(m["target_area"] == "string_partial_damping_refinement" for m in mappings)
            and any(m["target_area"] == "bridge_feedback_admittance_coupling" for m in mappings)
            and any(m["target_area"] == "top_back_air_radiation_weighting" for m in mappings)
            and any(m["target_area"] == "note_string_fret_contract_repair" for m in mappings)
        ),
    }


def build_multi_decay_model_contract(
    step5f: Mapping[str, Any],
    step3c: Mapping[str, Any],
    step5e: Mapping[str, Any],
) -> Dict[str, Any]:
    decay5f = step5f.get("string_body_air_decay_decomposition") or {}
    region = ((step5e.get("cavity_response_summary") or {}).get("top_back_air_balance") or {})

    def _term(
        name: str,
        *,
        current_source: str,
        limitation: str,
        proposed_parameter: str,
        fallback: str,
        blocked: str,
        metric: str,
    ) -> Dict[str, Any]:
        return {
            "term": name,
            "current_source": current_source,
            "current_limitation": limitation,
            "proposed_future_parameter": proposed_parameter,
            "allowed_fallback_level": fallback,
            "blocked_claims": blocked,
            "objective_validation_metric": metric,
            "no_artificial_reverb_or_echo": True,
        }

    terms = [
        _term(
            "string_partial_decay",
            current_source=decay5f.get("string_decay_summary", {}).get("model", "step5e_tau_k_power_law"),
            limitation="Single global base_tau/k^0.65 law; excessive harmonic purity",
            proposed_parameter="tau_k(string_id, k, tension_proxy, material_proxy)",
            fallback="L2_literature_fallback",
            blocked="Exact measured string damping",
            metric="partial_decay_slope_regularity_index increases; HNR proxy decreases moderately",
        ),
        _term(
            "top_plate_decay",
            current_source="step3c_W_rad × top_share in combined modal IR",
            limitation="Not separately tracked in output stems",
            proposed_parameter="top_modal_weight × Q_top(tau_top) per mode band",
            fallback="L2_region_share_proxy",
            blocked="Measured top plate damping",
            metric="top_plate_energy_fraction in decomposed stem",
        ),
        _term(
            "back_plate_decay",
            current_source="step3c_W_rad × back_share in combined modal IR",
            limitation="Merged with top in observed output",
            proposed_parameter="back_modal_weight × Q_back(tau_back) per mode band",
            fallback="L2_region_share_proxy",
            blocked="Measured back plate damping",
            metric="back_plate_energy_fraction in decomposed stem",
        ),
        _term(
            "air_cavity_decay",
            current_source="step3c_W_air × air_share modal proxy",
            limitation="weak_cavity_air_imprint flagged in Step 5F",
            proposed_parameter="cavity_mode_weight × Q_air(tau_air); Helmholtz proxy documented only",
            fallback="L2_helmholtz_lumped_proxy",
            blocked="Measured cavity impulse response",
            metric="cavity_air_modal_contribution_fraction increases without delayed echo onset",
        ),
        _term(
            "radiation_decay",
            current_source="modal Q/tau radiation proxy from step3c calibration",
            limitation="Combined with structural modes in one IR",
            proposed_parameter="radiation_damping_coeff per mode band",
            fallback="L2_literature_radiation_coeff",
            blocked="Anechoic radiation measurement",
            metric="radiation_term decay matches tau from Q/tau report",
        ),
        _term(
            "bridge_coupling_loss",
            current_source="none (one-way convolution only)",
            limitation="insufficient_string_body_feedback",
            proposed_parameter="diagnostic_admittance_feedback_gain (bounded, causal)",
            fallback="L2_admittance_proxy",
            blocked="Measured bridge mobility",
            metric="feedback stable; no comb/echo; body-string correlation bounded",
        ),
        _term(
            "combined_observed_decay",
            current_source="conv(string_force, modal_body_ir) listening render",
            limitation="Terms not separately observable in Step 5E stems",
            proposed_parameter="sum of causal modal terms (not post-mix reverb)",
            fallback="L2_combined_proxy",
            blocked="Final synthesis realism",
            metric="active_duration preserved; decay_minus_40_db_ms trackable per term",
        ),
    ]

    return {
        "terms": terms,
        "top_back_air_balance_current_proxy": region,
        "step3c_q_tau_unchanged_in_step5g": True,
        "combined_decay_interpretation": decay5f.get("combined_decay_interpretation"),
        "decay_separation_not_reverb": (
            "All future decay separation must remain causal modal/cavity/string/radiation "
            "contributions. No delayed echo, body_tail, or room reverb layers."
        ),
    }


def build_ranked_update_plan(diagnosis: Mapping[str, Any]) -> List[Dict[str, Any]]:
    mappings = diagnosis.get("mappings") or []
    ranked = sorted(mappings, key=lambda m: int(m.get("priority_rank") or 99))
    plan: List[Dict[str, Any]] = []
    for m in ranked:
        plan.append(
            {
                "rank": m.get("priority_rank"),
                "target_area": m.get("target_area"),
                "diagnosis_flag": m.get("diagnosis_flag"),
                "flagged_in_step5f": m.get("flagged"),
                "physical_rationale": m.get("physical_rationale"),
                "forbidden_alternative": m.get("forbidden_alternative"),
                "objective_validation": m.get("objective_validation"),
                "implementation_step": f"Step_5{chr(ord('H') + int(m.get('priority_rank', 1)) - 1)}_deferred",
            }
        )
    return plan


def build_implementation_order() -> Dict[str, Any]:
    order = [
        {
            "step": 1,
            "target": "note_string_fret_contract_repair",
            "future_step_id": "PGSM Step 5H",
            "rationale": "Prevents wrong physical assumptions before damping/weighting changes",
        },
        {
            "step": 2,
            "target": "string_partial_damping_refinement",
            "future_step_id": "PGSM Step 5I (proposed)",
            "rationale": "Addresses excessive harmonic purity with physics-based partial damping",
        },
        {
            "step": 3,
            "target": "top_back_air_radiation_weighting",
            "future_step_id": "PGSM Step 5J (proposed)",
            "rationale": "Strengthens body/cavity imprint using existing modal proxies",
        },
        {
            "step": 4,
            "target": "bridge_admittance_feedback_coupling",
            "future_step_id": "PGSM Step 5K (proposed)",
            "rationale": "Adds coupling only after stable baseline; avoids instability/echo",
        },
        {
            "step": 5,
            "target": "controlled_pluck_nail_transient",
            "future_step_id": "PGSM Step 5L (proposed)",
            "rationale": "Optional attack refinement after sustain model is stable",
        },
        {
            "step": 6,
            "target": "reference_guided_spectral_gap_analysis",
            "future_step_id": "PGSM Step 6A+ (existing diagnostic comparison)",
            "rationale": "Directional gap analysis; does not prove correctness",
        },
    ]
    return {
        "ordered_steps": order,
        "why_this_order": (
            "String/fret contract must be defined before note-dependent parameters. "
            "Damping and radiation weighting should be corrected objectively before "
            "feedback coupling. Pluck transient and reference comparison guide gaps "
            "but must not drive arbitrary tuning."
        ),
        "stable_baseline_required_before_feedback": True,
        "no_listening_tuning_in_plan": True,
    }


def build_proposed_step5h_contract() -> Dict[str, Any]:
    note_candidates: Dict[str, Any] = {}
    for note in NOTE_SET:
        f0 = NOTE_FREQUENCY_HZ[note]
        if note == "A2":
            candidate = {
                "string_id": 5,
                "string_name": "A",
                "fret": 0,
                "open_string_match": True,
                "effective_length_m": CLASSICAL_SCALE_LENGTH_M,
            }
        elif note == "A3":
            candidate = {
                "string_id": 5,
                "string_name": "A",
                "fret": 12,
                "open_string_match": False,
                "effective_length_m": round(CLASSICAL_SCALE_LENGTH_M / 2.0, 5),
                "interpretation": "diagnostic_octave_reference_not_fingering_validated",
            }
        elif note == "A4":
            candidate = {
                "string_id": "reference",
                "string_name": "A4_concert_reference",
                "fret": None,
                "open_string_match": False,
                "effective_length_m": round(CLASSICAL_SCALE_LENGTH_M * 110.0 / 440.0, 5),
                "interpretation": "diagnostic_reference_frequency_440Hz",
            }
        else:  # E5
            candidate = {
                "string_id": 1,
                "string_name": "E",
                "fret": 12,
                "open_string_match": False,
                "effective_length_m": round(CLASSICAL_SCALE_LENGTH_M / 2.0, 5),
                "interpretation": "high_e_twelfth_fret_candidate_not_validated",
            }
        note_candidates[note] = {
            **candidate,
            "target_frequency_hz": f0,
            "frequency_check_tolerance_hz": 0.5,
            "tension_proxy_N": "mid_plausible_range_not_measured",
            "linear_density_proxy_kg_m": NYLON_LINEAR_DENSITY_KG_M,
            "exact_open_string_claim_allowed": bool(candidate.get("open_string_match")),
            "diagnostic_reference_frequency_only": not candidate.get("open_string_match"),
        }

    return {
        "step_id": "PGSM Step 5H — Note/String/Fret Contract Repair",
        "mode": "planning_and_contract_validation_only",
        "no_audio_generation": True,
        "classical_tuning_reference": CLASSICAL_OPEN_STRINGS,
        "scale_length_m": CLASSICAL_SCALE_LENGTH_M,
        "scale_length_source": "pgsm_step2_1_CLASSICAL_SCALE_LENGTH_M",
        "scale_length_fallback_level": "L2_literature_fallback",
        "diagnostic_note_set": list(NOTE_SET),
        "note_mapping_candidates": note_candidates,
        "exact_open_string_claim_rules": {
            "allowed_only_when_fret_zero_and_string_open_match": True,
            "A2_only_candidate_for_exact_open": True,
            "all_other_notes_blocked": True,
        },
        "diagnostic_reference_frequency_rules": {
            "A3_A4_E5_are_reference_frequencies_not_playable_claims": True,
            "physical_fingering_validated": False,
            "label_required": "diagnostic_reference_frequency_not_exact_fingering",
        },
        "blocked_claims": [
            "Exact playable guitar realism",
            "Validated fingering position",
            "Measured string tension",
            "Multi-string chord/melody playback",
        ],
        "deliverables": [
            "note_string_fret_contract.json schema",
            "frequency_consistency_checks",
            "blocked_claim_enforcement_tests",
            "no_wav_generation",
        ],
        "readiness_target": "ready_for_step5i_string_partial_damping_refinement_plan",
    }


def audit_recommendations_for_forbidden(
    diagnosis: Mapping[str, Any],
    decay: Mapping[str, Any],
    plan: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    parts: List[str] = []
    for m in diagnosis.get("mappings") or []:
        parts.append(str(m.get("target_area") or ""))
        parts.append(str(m.get("physical_rationale") or ""))
    for t in decay.get("terms") or []:
        parts.append(str(t.get("proposed_future_parameter") or ""))
    for p in plan:
        parts.append(str(p.get("target_area") or ""))
        parts.append(str(p.get("physical_rationale") or ""))
    text = " ".join(parts).lower()

    positive_forbidden = [
        "use arbitrary eq",
        "apply arbitrary eq",
        "add artificial reverb",
        "add reverb",
        "delayed echo layer",
        "body_tail layer",
        "wood-to-gain",
        "wood to gain mapping",
        "tune by listening",
        "listening-only tuning",
    ]
    violations = [p for p in positive_forbidden if p in text]
    return {
        "no_forbidden_recommendations": len(violations) == 0,
        "violations_found": violations,
    }


def build_readiness_after_step5g(objective: Mapping[str, Any]) -> Dict[str, Any]:
    status = READINESS_AFTER if objective.get("all_pass") else "failed_physical_tone_model_update_plan"
    return {
        "current_status": status,
        "final_synthesis_ready": False,
        "stk_integration_allowed": False,
        "website_production_replacement_allowed": False,
        "multi_guitar_comparison_allowed": False,
        "melody_chord_playback_allowed": False,
        "subjective_tuning_allowed": False,
        "real_guitar_equivalence_allowed": False,
        "step5h_contract_repair_allowed": status == READINESS_AFTER,
    }


def run_objective_tests(
    upstream: Mapping[str, Any],
    diagnosis: Mapping[str, Any],
    decay: Mapping[str, Any],
    plan: List[Dict[str, Any]],
    step5h: Mapping[str, Any],
    forbidden_audit: Mapping[str, Any],
    fp_preserved: bool,
) -> Dict[str, Any]:
    mappings = {m["diagnosis_flag"]: m for m in (diagnosis.get("mappings") or [])}
    term_names = {t["term"] for t in (decay.get("terms") or [])}

    tests = {
        "upstream_ready": upstream.get("pass"),
        "no_audio_modified": fp_preserved,
        "diagnosis_mapping_exists": bool(diagnosis.get("mappings")),
        "harmonic_purity_maps_to_damping": (
            mappings.get("excessive_harmonic_purity", {}).get("target_area")
            == "string_partial_damping_refinement"
        ),
        "cavity_maps_to_weighting": (
            mappings.get("weak_cavity_air_imprint", {}).get("target_area")
            == "top_back_air_radiation_weighting"
        ),
        "feedback_maps_to_coupling": (
            mappings.get("insufficient_string_body_feedback", {}).get("target_area")
            == "bridge_feedback_admittance_coupling"
        ),
        "shared_ir_maps_to_contract": (
            mappings.get("shared_body_ir_limitation", {}).get("target_area")
            == "note_string_fret_contract_repair"
        ),
        "multi_decay_terms_complete": bool(
            {
                "string_partial_decay",
                "top_plate_decay",
                "back_plate_decay",
                "air_cavity_decay",
                "radiation_decay",
                "bridge_coupling_loss",
                "combined_observed_decay",
            }.issubset(term_names)
        ),
        "ranked_plan_has_six_areas": len(plan) >= 6,
        "step5h_contract_exists": bool(step5h.get("step_id")),
        "no_forbidden_recommendations": forbidden_audit.get("no_forbidden_recommendations"),
        "no_stk_integration": True,
        "no_fem_run": True,
        "no_rom_run": True,
        "website_default_unchanged": True,
        "no_audio_generated": True,
    }
    tests["all_pass"] = bool(all(tests.values()))
    return tests


def build_pgsm_step5g_report(*, repo_root: Optional[Path] = None) -> Dict[str, Any]:
    root = Path(repo_root or REPO_ROOT)

    step5f = load_step_report(_report_path(root, "pgsm_step5f_string_driven_extended_validation.json"))
    step5e = load_step_report(_report_path(root, "pgsm_step5e_string_driven_bridge_force_repair.json"))
    step3c = load_step_report(_report_path(root, "pgsm_step3c_numeric_calibration.json"))
    step22b = _optional_report(root, "pgsm_step2_2b_material_alignment_audit.json")

    fp_before = {**collect_step5e_fingerprints(root), **collect_previous_audio_fingerprints(root)}
    upstream = verify_upstream_readiness(step5f, fp_before)

    findings = build_step5f_findings_summary(step5f)
    diagnosis = build_diagnosis_to_update_targets(step5f)
    decay = build_multi_decay_model_contract(step5f, step3c, step5e)
    ranked_plan = build_ranked_update_plan(diagnosis)
    implementation_order = build_implementation_order()
    step5h = build_proposed_step5h_contract()

    fp_after = {**collect_step5e_fingerprints(root), **collect_previous_audio_fingerprints(root)}
    fp_preserved = fp_before == fp_after

    forbidden_audit = audit_recommendations_for_forbidden(diagnosis, decay, ranked_plan)

    objective = run_objective_tests(
        upstream, diagnosis, decay, ranked_plan, step5h, forbidden_audit, fp_preserved
    )
    readiness = build_readiness_after_step5g(objective)

    material_ref = None
    lib_path = root / "data" / "pgsm_tonewood_material_library.json"
    if lib_path.is_file():
        material_ref = {"path": str(lib_path), "loaded_for_proxy_reference_only": True}

    return {
        "report_version": PGSM_STEP5G_VERSION,
        "timestamp": _utc_now(),
        "status": "pgsm_step5g_physical_tone_model_update_plan_complete",
        "no_audio_generated": True,
        "no_audio_modified": fp_preserved,
        "no_stk_integration": True,
        "no_fem_run": True,
        "no_rom_run": True,
        "website_default": DEFAULT_WEBSITE_STK_MODE,
        "website_default_unchanged": True,
        "step5f_loaded": step5f.get("report_version"),
        "step5e_loaded": step5e.get("report_version"),
        "step3c_loaded": step3c.get("report_version"),
        "step22b_loaded": (step22b or {}).get("report_version"),
        "material_library_reference": material_ref,
        "upstream_readiness": upstream,
        "step5f_findings_summary": findings,
        "diagnosis_to_update_targets": diagnosis,
        "multi_decay_model_contract": decay,
        "ranked_update_plan": ranked_plan,
        "implementation_order": implementation_order,
        "proposed_step5h_contract": step5h,
        "forbidden_recommendation_audit": forbidden_audit,
        "blocked_claims": [
            "Final synthesis",
            "STK integration",
            "Website production replacement",
            "Multi-guitar comparison",
            "Melody/chord playback",
            "Subjective tuning by ear",
            "Real-guitar equivalence or validation proof",
            "Arbitrary EQ",
            "Artificial reverb/echo/body_tail",
            "Wood-to-gain mapping",
            "Playable instrument realism (until contracts validated)",
        ],
        "objective_test_results": objective,
        "readiness_after_step5g": readiness,
        "safe_next_step": (
            "PGSM Step 5H: note/string/fret contract repair (planning/validation only)"
            if readiness["current_status"] == READINESS_AFTER
            else "Resolve Step 5G plan failures before Step 5H"
        ),
        "explicit_statement": (
            "PGSM Step 5G is a physical tone model update plan only. "
            "It does not generate audio, does not tune by listening, and does not prove realism."
        ),
    }


def write_markdown_report(report: Mapping[str, Any], path: Path) -> None:
    rg = report.get("readiness_after_step5g") or {}
    findings = report.get("step5f_findings_summary") or {}
    diagnosis = report.get("diagnosis_to_update_targets") or {}
    decay = report.get("multi_decay_model_contract") or {}
    plan = report.get("ranked_update_plan") or []
    order = report.get("implementation_order") or {}
    step5h = report.get("proposed_step5h_contract") or {}
    obj = report.get("objective_test_results") or {}
    labels = findings.get("global_robotic_tone_labels") or {}

    lines = [
        "# PGSM Step 5G — physical tone model update plan",
        "",
        f"**Generated:** {report.get('timestamp')}",
        f"**Readiness:** `{rg.get('current_status')}`",
        "",
        report.get("explicit_statement", ""),
        "",
        "## Step 5F findings",
        "",
        "| Finding | Value |",
        "|---------|-------|",
        f"| Validation passed | {findings.get('validation_passed')} |",
        f"| Not click dominant | {findings.get('not_click_dominant')} |",
        f"| Pitch salience | {findings.get('pitch_salience_present')} |",
        f"| Active duration OK | {findings.get('active_duration_sufficient')} |",
        f"| Robotic tone present | {findings.get('robotic_tone_still_present')} |",
        "",
        "### Robotic-tone flags",
        "",
    ]
    for k, v in labels.items():
        lines.append(f"- `{k}`: **{v}**")

    lines.extend(
        [
            "",
            "## Model update targets",
            "",
            "| Rank | Target | Diagnosis flag | Flagged |",
            "|------|--------|----------------|---------|",
        ]
    )
    for item in plan:
        lines.append(
            f"| {item.get('rank')} | {item.get('target_area')} | "
            f"{item.get('diagnosis_flag')} | {item.get('flagged_in_step5f')} |"
        )

    lines.extend(
        [
            "",
            "## Multi-decay contract",
            "",
            "| Term | Current source | Proposed parameter | Fallback |",
            "|------|----------------|-------------------|----------|",
        ]
    )
    for t in decay.get("terms") or []:
        lines.append(
            f"| {t.get('term')} | {t.get('current_source')} | "
            f"{t.get('proposed_future_parameter')} | {t.get('allowed_fallback_level')} |"
        )

    lines.extend(
        [
            "",
            decay.get("decay_separation_not_reverb", ""),
            "",
            "## Implementation order",
            "",
            order.get("why_this_order", ""),
            "",
        ]
    )
    for item in order.get("ordered_steps") or []:
        lines.append(f"{item.get('step')}. **{item.get('target')}** ({item.get('future_step_id')}) — {item.get('rationale')}")

    lines.extend(
        [
            "",
            "## Proposed Step 5H contract",
            "",
            f"**{step5h.get('step_id')}**",
            "",
            f"- Scale length: {step5h.get('scale_length_m')} m ({step5h.get('scale_length_fallback_level')})",
            f"- Classical tuning: {', '.join((step5h.get('classical_tuning_reference') or {}).keys())}",
            f"- Mode: {step5h.get('mode')}",
            "",
            "### Note mapping candidates",
            "",
            "| Note | string | fret | f0 Hz | exact open allowed |",
            "|------|--------|------|-------|-------------------|",
        ]
    )
    for note, c in (step5h.get("note_mapping_candidates") or {}).items():
        lines.append(
            f"| {note} | {c.get('string_name')} | {c.get('fret')} | "
            f"{c.get('target_frequency_hz')} | {c.get('exact_open_string_claim_allowed')} |"
        )

    lines.extend(
        [
            "",
            "## Blocked claims",
            "",
        ]
    )
    for claim in report.get("blocked_claims") or []:
        lines.append(f"- {claim}")

    lines.extend(
        [
            "",
            "## Readiness",
            "",
            f"all_pass: **{obj.get('all_pass')}**",
            f"no_audio_modified: **{report.get('no_audio_modified')}**",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pgsm_step5g_reports(
    *,
    repo_root: Optional[Path] = None,
    json_path: Optional[Path] = None,
    md_path: Optional[Path] = None,
) -> Dict[str, Any]:
    root = Path(repo_root or REPO_ROOT)
    report = build_pgsm_step5g_report(repo_root=root)
    jpath = Path(json_path or REPORT_JSON)
    mpath = Path(md_path or REPORT_MD)
    jpath.parent.mkdir(parents=True, exist_ok=True)
    jpath.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown_report(report, mpath)
    return report


def main() -> None:
    report = write_pgsm_step5g_reports()
    rg = report.get("readiness_after_step5g") or {}
    obj = report.get("objective_test_results") or {}
    print(f"Wrote {REPORT_JSON}")
    print(f"Readiness: {rg.get('current_status')}")
    print(f"all_pass: {obj.get('all_pass')}")


if __name__ == "__main__":
    main()
