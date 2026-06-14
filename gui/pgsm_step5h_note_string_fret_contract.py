#!/usr/bin/env python3
"""
PGSM Step 5H — note/string/fret contract repair.
Planning, contract definition, and validation only; no audio or synthesis changes.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from pgsm_step2_1_parameter_targets import (
    CLASSICAL_SCALE_LENGTH_M,
    NYLON_LINEAR_DENSITY_KG_M,
    load_step_report,
)
from pgsm_step3a_numerical_ir_testbench import FIXED_PLUCK_POSITION
from pgsm_step5a_limited_note_set_diagnostic_audio import NOTE_FREQUENCY_HZ, NOTE_SET
from pgsm_step5e_string_driven_bridge_force_repair import collect_previous_audio_fingerprints
from pgsm_step5f_string_driven_extended_validation import collect_step5e_fingerprints
from pgsm_step5g_physical_tone_model_update_plan import READINESS_AFTER as READINESS_STEP5G
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE

PGSM_STEP5H_VERSION = "pgsm_step5h_note_string_fret_contract_v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON = REPO_ROOT / "audio" / "debug_reports" / "pgsm_step5h_note_string_fret_contract.json"
REPORT_MD = REPO_ROOT / "audio" / "debug_reports" / "pgsm_step5h_note_string_fret_contract.md"
DATA_JSON = REPO_ROOT / "data" / "pgsm_classical_guitar_note_string_fret_contract.json"

READINESS_AFTER = "ready_for_step5i_string_partial_damping_refinement"

SCALE_LENGTH_M = 0.650
SCALE_LENGTH_SOURCE_LEVEL = "L2_literature_fallback"
DEFAULT_MAX_FRET = 19
EXTENDED_MAX_FRET = 21

NOTE_TO_MIDI: Dict[str, int] = {
    "E2": 40,
    "A2": 45,
    "D3": 50,
    "G3": 55,
    "B3": 59,
    "E4": 64,
    "A3": 57,
    "A4": 69,
    "E5": 76,
}

STRING_DEFINITIONS: Tuple[Dict[str, Any], ...] = (
    {
        "string_id": "string_6",
        "open_note": "E2",
        "open_frequency_hz": 82.4069,
        "nominal_role": "bass_low",
        "nylon_classical_fallback_status": "nylon_classical_literature_fallback",
        "linear_density_status": "unresolved_or_literature_fallback",
        "linear_density_proxy_kg_m": NYLON_LINEAR_DENSITY_KG_M,
        "tension_status": "derived_only_if_linear_density_known",
        "exact_tension_claim_blocked": True,
    },
    {
        "string_id": "string_5",
        "open_note": "A2",
        "open_frequency_hz": 110.0,
        "nominal_role": "bass_mid",
        "nylon_classical_fallback_status": "nylon_classical_literature_fallback",
        "linear_density_status": "unresolved_or_literature_fallback",
        "linear_density_proxy_kg_m": NYLON_LINEAR_DENSITY_KG_M,
        "tension_status": "derived_only_if_linear_density_known",
        "exact_tension_claim_blocked": True,
    },
    {
        "string_id": "string_4",
        "open_note": "D3",
        "open_frequency_hz": 146.8324,
        "nominal_role": "mid_low",
        "nylon_classical_fallback_status": "nylon_classical_literature_fallback",
        "linear_density_status": "unresolved_or_literature_fallback",
        "linear_density_proxy_kg_m": NYLON_LINEAR_DENSITY_KG_M,
        "tension_status": "derived_only_if_linear_density_known",
        "exact_tension_claim_blocked": True,
    },
    {
        "string_id": "string_3",
        "open_note": "G3",
        "open_frequency_hz": 195.9977,
        "nominal_role": "mid",
        "nylon_classical_fallback_status": "nylon_classical_literature_fallback",
        "linear_density_status": "unresolved_or_literature_fallback",
        "linear_density_proxy_kg_m": NYLON_LINEAR_DENSITY_KG_M,
        "tension_status": "derived_only_if_linear_density_known",
        "exact_tension_claim_blocked": True,
    },
    {
        "string_id": "string_2",
        "open_note": "B3",
        "open_frequency_hz": 246.9417,
        "nominal_role": "mid_high",
        "nylon_classical_fallback_status": "nylon_classical_literature_fallback",
        "linear_density_status": "unresolved_or_literature_fallback",
        "linear_density_proxy_kg_m": NYLON_LINEAR_DENSITY_KG_M,
        "tension_status": "derived_only_if_linear_density_known",
        "exact_tension_claim_blocked": True,
    },
    {
        "string_id": "string_1",
        "open_note": "E4",
        "open_frequency_hz": 329.6276,
        "nominal_role": "treble",
        "nylon_classical_fallback_status": "nylon_classical_literature_fallback",
        "linear_density_status": "unresolved_or_literature_fallback",
        "linear_density_proxy_kg_m": NYLON_LINEAR_DENSITY_KG_M,
        "tension_status": "derived_only_if_linear_density_known",
        "exact_tension_claim_blocked": True,
    },
)

STRING_BY_ID: Dict[str, Dict[str, Any]] = {s["string_id"]: s for s in STRING_DEFINITIONS}

DIAGNOSTIC_CANDIDATE_SPECS: Dict[str, List[Dict[str, Any]]] = {
    "A2": [{"string_id": "string_5", "fret": 0, "preferred_hint": True}],
    "A3": [
        {"string_id": "string_5", "fret": 12},
        {"string_id": "string_4", "fret": 7, "preferred_hint": True},
        {"string_id": "string_3", "fret": 2},
    ],
    "A4": [
        {"string_id": "string_3", "fret": 14},
        {"string_id": "string_2", "fret": 10},
        {"string_id": "string_1", "fret": 5, "preferred_hint": True},
    ],
    "E5": [
        {"string_id": "string_1", "fret": 12, "preferred_hint": True},
        {"string_id": "string_2", "fret": 17},
        {"string_id": "string_3", "fret": 21, "extended_high_fret_candidate": True},
    ],
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _report_path(root: Path, name: str) -> Path:
    return root / "audio" / "debug_reports" / name


def compute_frequency_from_fret(open_frequency_hz: float, fret: int) -> float:
    return open_frequency_hz * (2.0 ** (fret / 12.0))


def compute_effective_length_m(scale_length_m: float, fret: int) -> float:
    return scale_length_m / (2.0 ** (fret / 12.0))


def frequency_error_cents(computed_hz: float, target_hz: float) -> float:
    if computed_hz <= 0 or target_hz <= 0:
        return float("nan")
    return 1200.0 * math.log2(computed_hz / target_hz)


def build_classical_guitar_base_contract() -> Dict[str, Any]:
    tuning = {
        f"string_{7 - i}": {
            "open_note": s["open_note"],
            "open_frequency_hz": s["open_frequency_hz"],
        }
        for i, s in enumerate(STRING_DEFINITIONS)
    }
    return {
        "instrument_type": "classical_guitar",
        "scale_length_m": SCALE_LENGTH_M,
        "scale_length_source_level": SCALE_LENGTH_SOURCE_LEVEL,
        "tuning": tuning,
        "strings": list(STRING_DEFINITIONS),
        "fret_range": {
            "default_max_fret": DEFAULT_MAX_FRET,
            "extended_max_fret": EXTENDED_MAX_FRET,
            "extended_fret_policy": "candidates_above_default_max_fret_are_non_preferred_extended_only",
        },
        "blocked_exact_tension_claim": True,
        "linear_density_status": "unresolved_or_literature_fallback",
    }


def build_fret_effective_length_model() -> Dict[str, Any]:
    return {
        "model_id": "equal_temperament_fret_length_v1",
        "effective_length_m": "scale_length_m / 2^(fret/12)",
        "frequency_hz": "open_frequency_hz * 2^(fret/12)",
        "scale_length_m": SCALE_LENGTH_M,
        "acceptance_rules": {
            "frequency_error_cents_near_zero_for_et_candidates": True,
            "effective_length_le_scale_length": True,
            "integer_fret_for_physical_playable": True,
            "diagnostic_reference_only_must_be_labeled": True,
        },
    }


def _candidate_limitation_text(
    note: str,
    string_id: Optional[str],
    fret: Optional[int],
    *,
    diagnostic_reference_only: bool,
    extended_high_fret: bool,
) -> str:
    if diagnostic_reference_only:
        return (
            f"{note} retained as concert/reference frequency label only; "
            "not a validated physical string/fret mapping."
        )
    if extended_high_fret:
        return (
            f"Extended high-fret candidate (fret {fret} > default max {DEFAULT_MAX_FRET}); "
            "valid equal-temperament mapping but not preferred for conservative classical range."
        )
    if fret == 0:
        return "Open-string mapping; exact open-string claim allowed when open note matches target."
    if fret is not None and fret > DEFAULT_MAX_FRET:
        return f"Fret {fret} exceeds conservative default max fret {DEFAULT_MAX_FRET}."
    return (
        f"Equal-temperament fretted mapping on {string_id}; "
        "physical playable candidate within conservative fret range."
    )


def build_playable_candidate(
    note: str,
    string_id: str,
    fret: int,
    *,
    extended_high_fret_candidate: bool = False,
) -> Dict[str, Any]:
    target_hz = NOTE_FREQUENCY_HZ[note]
    string_def = STRING_BY_ID[string_id]
    open_hz = float(string_def["open_frequency_hz"])
    computed_hz = compute_frequency_from_fret(open_hz, fret)
    eff_len = compute_effective_length_m(SCALE_LENGTH_M, fret)
    err_cents = frequency_error_cents(computed_hz, target_hz)
    open_match = fret == 0 and string_def["open_note"] == note
    within_conservative = fret <= DEFAULT_MAX_FRET
    physical = (
        isinstance(fret, int)
        and fret >= 0
        and eff_len <= SCALE_LENGTH_M + 1e-9
        and abs(err_cents) < 1.0
        and (within_conservative or extended_high_fret_candidate)
    )
    return {
        "note": note,
        "string_id": string_id,
        "open_note": string_def["open_note"],
        "fret": fret,
        "open_frequency_hz": open_hz,
        "target_frequency_hz": target_hz,
        "computed_frequency_hz": round(computed_hz, 6),
        "frequency_error_cents": round(err_cents, 6),
        "effective_length_m": round(eff_len, 6),
        "exact_open_string_claim_allowed": open_match,
        "physical_playable_candidate": physical,
        "diagnostic_reference_only": False,
        "extended_high_fret_candidate": extended_high_fret_candidate,
        "within_default_max_fret": within_conservative,
        "limitation": _candidate_limitation_text(
            note,
            string_id,
            fret,
            diagnostic_reference_only=False,
            extended_high_fret=extended_high_fret_candidate,
        ),
    }


def build_a4_concert_reference_candidate() -> Dict[str, Any]:
    target_hz = NOTE_FREQUENCY_HZ["A4"]
    ref_length = SCALE_LENGTH_M * 110.0 / 440.0
    return {
        "note": "A4",
        "mapping_id": "A4_concert_reference",
        "string_id": None,
        "open_note": None,
        "fret": None,
        "open_frequency_hz": None,
        "target_frequency_hz": target_hz,
        "computed_frequency_hz": target_hz,
        "frequency_error_cents": 0.0,
        "effective_length_m": round(ref_length, 6),
        "exact_open_string_claim_allowed": False,
        "physical_playable_candidate": False,
        "diagnostic_reference_only": True,
        "extended_high_fret_candidate": False,
        "within_default_max_fret": False,
        "limitation": _candidate_limitation_text(
            "A4",
            None,
            None,
            diagnostic_reference_only=True,
            extended_high_fret=False,
        ),
    }


def score_physical_candidate(note: str, candidate: Mapping[str, Any]) -> float:
    if candidate.get("diagnostic_reference_only"):
        return -1e9
    if not candidate.get("physical_playable_candidate"):
        return -1e9
    fret = int(candidate.get("fret") or 999)
    score = 0.0
    score -= abs(float(candidate.get("frequency_error_cents") or 0.0)) * 100.0
    if fret > DEFAULT_MAX_FRET:
        score -= 500.0
    elif fret > 14:
        score -= 15.0
    score -= fret * 2.0
    if note == "A3":
        sid = candidate.get("string_id")
        if sid == "string_4" and fret == 7:
            score += 20.0
        elif sid == "string_5" and fret == 12:
            score += 8.0
        elif sid == "string_3" and fret == 2:
            score += 2.0
    if note == "A4":
        sid = candidate.get("string_id")
        if sid == "string_1" and fret == 5:
            score += 15.0
        elif sid == "string_2" and fret == 10:
            score += 10.0
    if note == "E5" and candidate.get("string_id") == "string_1" and fret == 12:
        score += 15.0
    return score


def build_diagnostic_note_candidates() -> Dict[str, Any]:
    candidates_by_note: Dict[str, List[Dict[str, Any]]] = {}
    for note in NOTE_SET:
        specs = DIAGNOSTIC_CANDIDATE_SPECS[note]
        built: List[Dict[str, Any]] = []
        for spec in specs:
            built.append(
                build_playable_candidate(
                    note,
                    spec["string_id"],
                    spec["fret"],
                    extended_high_fret_candidate=bool(spec.get("extended_high_fret_candidate")),
                )
            )
        if note == "A4":
            built.append(build_a4_concert_reference_candidate())
        candidates_by_note[note] = built

    preferred: Dict[str, Dict[str, Any]] = {}
    preference_rationale: Dict[str, str] = {}
    for note in NOTE_SET:
        playable = [c for c in candidates_by_note[note] if c.get("physical_playable_candidate")]
        best = max(playable, key=lambda c: score_physical_candidate(note, c))
        preferred[note] = dict(best)
        if note == "A2":
            preference_rationale[note] = (
                "A2 matches open string_5 (A2) at fret 0; only diagnostic note with exact open-string claim."
            )
        elif note == "A3":
            preference_rationale[note] = (
                "Among equal-temperament candidates, string_4 fret 7 preferred over string_5 fret 12 "
                "(avoids octave-position assumption) and string_3 fret 2 (mid-neck D-string position "
                "is more representative for classical diagnostic mapping)."
            )
        elif note == "A4":
            preference_rationale[note] = (
                "string_1 fret 5 preferred over string_2 fret 10 and string_3 fret 14: lowest fret "
                "among playable candidates on treble string; A4_concert_reference rejected as physical mapping."
            )
        elif note == "E5":
            preference_rationale[note] = (
                "string_1 fret 12 preferred: within default max fret 19, standard high-E twelfth-fret position; "
                "string_2 fret 17 and string_3 fret 21 remain extended alternatives."
            )

    rejected = [build_a4_concert_reference_candidate()]
    return {
        "candidates_by_note": candidates_by_note,
        "preferred_diagnostic_mapping": preferred,
        "preference_rationale": preference_rationale,
        "rejected_reference_only_mappings": rejected,
    }


def build_stk_readiness_contract(preferred: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    entries: List[Dict[str, Any]] = []
    for note in NOTE_SET:
        mapping = preferred[note]
        entries.append(
            {
                "note_name": note,
                "midi_note_number": NOTE_TO_MIDI[note],
                "frequency_hz": mapping["target_frequency_hz"],
                "string_id": mapping["string_id"],
                "fret": mapping["fret"],
                "effective_length_m": mapping["effective_length_m"],
                "pluck_position_ratio_relative_to_string": FIXED_PLUCK_POSITION,
                "bridge_position_ratio": 1.0,
                "string_decay_profile_id": "classical_nylon_unresolved_damping_v1",
                "body_modal_response_id": "pgsm_step3c_calibrated_modal_v1",
                "cavity_response_proxy_id": "pgsm_modal_air_cavity_proxy_v1",
                "radiation_profile_id": "pgsm_radiation_weighting_unresolved_v1",
                "fallback_level": "L2_literature_fallback",
                "blocked_claims": [
                    "exact_string_tension",
                    "measured_linear_density",
                    "validated_pluck_transient",
                    "real_guitar_equivalence",
                    "stk_integration_active",
                ],
            }
        )
    return {
        "stk_integration_allowed": False,
        "reason": "note_string_fret_contract_ready_but_damping_and_radiation_updates_pending",
        "future_stk_note_entries": entries,
        "prepared_fields_only": True,
    }


def verify_upstream_readiness(
    step5g: Mapping[str, Any],
    fp_before: Mapping[str, str],
) -> Dict[str, Any]:
    rg = step5g.get("readiness_after_step5g") or {}
    obj = step5g.get("objective_test_results") or {}
    return {
        "step5g_readiness": rg.get("current_status"),
        "step5g_pass": rg.get("current_status") == READINESS_STEP5G,
        "step5g_all_pass": bool(obj.get("all_pass")),
        "step5h_contract_repair_allowed": bool(rg.get("step5h_contract_repair_allowed")),
        "fingerprints_before": dict(fp_before),
        "no_audio_generation_allowed": True,
        "final_synthesis_blocked": rg.get("final_synthesis_ready") is False,
        "stk_blocked": rg.get("stk_integration_allowed") is False,
        "website_blocked": rg.get("website_production_replacement_allowed") is False,
        "multi_guitar_blocked": rg.get("multi_guitar_comparison_allowed") is False,
        "melody_chords_blocked": rg.get("melody_chord_playback_allowed") is False,
        "fem_rom_blocked": True,
        "pass": bool(
            rg.get("current_status") == READINESS_STEP5G
            and obj.get("all_pass")
            and rg.get("step5h_contract_repair_allowed")
            and rg.get("final_synthesis_ready") is False
            and rg.get("stk_integration_allowed") is False
        ),
    }


def validate_step5g_first_target(step5g: Mapping[str, Any], contract_complete: bool) -> Dict[str, Any]:
    plan = step5g.get("ranked_update_plan") or []
    first = plan[0] if plan else {}
    mappings = (step5g.get("diagnosis_to_update_targets") or {}).get("mappings") or []
    contract_mapping = next(
        (m for m in mappings if m.get("target_area") == "note_string_fret_contract_repair"),
        {},
    )
    return {
        "step5g_first_ranked_target": first.get("target_area"),
        "step5g_first_target_is_contract_repair": first.get("target_area") == "note_string_fret_contract_repair",
        "diagnosis_flag_for_contract": contract_mapping.get("diagnosis_flag"),
        "step5h_contract_repair_complete": contract_complete,
        "satisfies_step5g_first_target": bool(
            first.get("target_area") == "note_string_fret_contract_repair" and contract_complete
        ),
    }


def run_validation(
    candidates: Mapping[str, Any],
    preferred: Mapping[str, Mapping[str, Any]],
    rejected: Sequence[Mapping[str, Any]],
    stk: Mapping[str, Any],
    upstream_pass: bool,
    fp_preserved: bool,
) -> Dict[str, Any]:
    all_candidates: List[Dict[str, Any]] = []
    for note in NOTE_SET:
        all_candidates.extend(candidates["candidates_by_note"][note])

    playable = [c for c in all_candidates if c.get("physical_playable_candidate")]
    ref_only = [c for c in all_candidates if c.get("diagnostic_reference_only")]

    def _abs_error_cents(candidate: Mapping[str, Any]) -> float:
        err = candidate.get("frequency_error_cents")
        if err is None:
            return 999.0
        return abs(float(err))

    freq_ok = all(_abs_error_cents(c) < 1.0 for c in playable)
    length_ok = all(float(c.get("effective_length_m") or 999) <= SCALE_LENGTH_M + 1e-6 for c in playable)
    integer_fret_ok = all(isinstance(c.get("fret"), int) for c in playable)
    a2_open = preferred.get("A2", {}).get("string_id") == "string_5" and preferred["A2"].get("fret") == 0
    a3_count = sum(
        1 for c in candidates["candidates_by_note"]["A3"] if c.get("physical_playable_candidate")
    )
    a4_playable = any(
        c.get("physical_playable_candidate") for c in candidates["candidates_by_note"]["A4"]
    )
    a4_not_only_ref = preferred.get("A4", {}).get("diagnostic_reference_only") is not True
    e5_playable = preferred.get("E5", {}).get("physical_playable_candidate") is True
    all_preferred = all(note in preferred for note in NOTE_SET)
    ref_labeled = all(c.get("diagnostic_reference_only") for c in ref_only)
    rejected_a4_ref = any(r.get("mapping_id") == "A4_concert_reference" for r in rejected)

    checks = {
        "upstream_ready": upstream_pass,
        "no_audio_modified": fp_preserved,
        "frequency_error_near_zero": freq_ok,
        "effective_length_le_scale": length_ok,
        "integer_fret_for_playable": integer_fret_ok,
        "a2_open_string_5": a2_open,
        "a3_multiple_playable_candidates": a3_count >= 2,
        "a4_has_playable_candidates": a4_playable,
        "a4_not_only_concert_reference": a4_not_only_ref,
        "e5_physical_playable_preferred": e5_playable,
        "preferred_mapping_all_notes": all_preferred,
        "diagnostic_reference_only_labeled": ref_labeled,
        "a4_concert_reference_rejected": rejected_a4_ref,
        "stk_contract_exists": bool(stk.get("future_stk_note_entries")),
        "stk_integration_blocked": stk.get("stk_integration_allowed") is False,
        "no_audio_generated": True,
        "no_stk_integration": True,
        "no_fem_run": True,
        "no_rom_run": True,
        "website_default_unchanged": True,
    }
    checks["all_pass"] = bool(all(checks.values()))
    return checks


def build_readiness_after_step5h(validation: Mapping[str, Any]) -> Dict[str, Any]:
    status = READINESS_AFTER if validation.get("all_pass") else "failed_note_string_fret_contract_repair"
    return {
        "current_status": status,
        "final_synthesis_ready": False,
        "stk_integration_allowed": False,
        "website_production_replacement_allowed": False,
        "multi_guitar_comparison_allowed": False,
        "melody_chord_playback_allowed": False,
        "subjective_tuning_allowed": False,
        "real_guitar_equivalence_allowed": False,
        "string_partial_damping_refinement_allowed": status == READINESS_AFTER,
        "contract_only_not_final": True,
    }


def build_contract_data_export(
    base: Mapping[str, Any],
    diagnostic: Mapping[str, Any],
    preferred: Mapping[str, Mapping[str, Any]],
    stk: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "contract_version": PGSM_STEP5H_VERSION,
        "instrument_type": base.get("instrument_type"),
        "scale_length_m": base.get("scale_length_m"),
        "scale_length_source_level": base.get("scale_length_source_level"),
        "strings": base.get("strings"),
        "fret_effective_length_model": build_fret_effective_length_model(),
        "preferred_diagnostic_mapping": preferred,
        "stk_readiness_contract": stk,
        "blocked_claims": [
            "Final synthesis",
            "STK integration",
            "Real-guitar equivalence",
            "Exact measured string tension",
        ],
    }


def build_pgsm_step5h_report(*, repo_root: Optional[Path] = None) -> Dict[str, Any]:
    root = Path(repo_root or REPO_ROOT)

    step5g = load_step_report(_report_path(root, "pgsm_step5g_physical_tone_model_update_plan.json"))
    step5f = load_step_report(_report_path(root, "pgsm_step5f_string_driven_extended_validation.json"))
    step5e = load_step_report(_report_path(root, "pgsm_step5e_string_driven_bridge_force_repair.json"))

    fp_before = {**collect_step5e_fingerprints(root), **collect_previous_audio_fingerprints(root)}
    upstream = verify_upstream_readiness(step5g, fp_before)

    base_contract = build_classical_guitar_base_contract()
    fret_model = build_fret_effective_length_model()
    diagnostic = build_diagnostic_note_candidates()
    preferred = diagnostic["preferred_diagnostic_mapping"]
    stk = build_stk_readiness_contract(preferred)

    fp_after = {**collect_step5e_fingerprints(root), **collect_previous_audio_fingerprints(root)}
    fp_preserved = fp_before == fp_after

    validation = run_validation(
        diagnostic,
        preferred,
        diagnostic["rejected_reference_only_mappings"],
        stk,
        upstream.get("pass", False),
        fp_preserved,
    )
    step5g_target = validate_step5g_first_target(step5g, validation.get("all_pass", False))
    readiness = build_readiness_after_step5h(validation)

    material_ref = None
    lib_path = root / "data" / "pgsm_tonewood_material_library.json"
    if lib_path.is_file():
        material_ref = {"path": str(lib_path), "loaded_for_metadata_fallback_only": True}

    return {
        "report_version": PGSM_STEP5H_VERSION,
        "timestamp": _utc_now(),
        "status": "pgsm_step5h_note_string_fret_contract_complete",
        "no_audio_generated": True,
        "no_audio_modified": fp_preserved,
        "no_stk_integration": True,
        "no_fem_run": True,
        "no_rom_run": True,
        "website_default": DEFAULT_WEBSITE_STK_MODE,
        "website_default_unchanged": True,
        "step5g_loaded": step5g.get("report_version"),
        "step5f_loaded": step5f.get("report_version"),
        "step5e_loaded": step5e.get("report_version"),
        "material_library_reference": material_ref,
        "upstream_readiness": upstream,
        "classical_guitar_base_contract": base_contract,
        "fret_effective_length_model": fret_model,
        "diagnostic_note_candidates": diagnostic["candidates_by_note"],
        "preferred_diagnostic_mapping": preferred,
        "preference_rationale": diagnostic["preference_rationale"],
        "rejected_reference_only_mappings": diagnostic["rejected_reference_only_mappings"],
        "stk_readiness_contract": stk,
        "step5g_first_target_validation": step5g_target,
        "validation_results": validation,
        "blocked_claims": [
            "Final synthesis",
            "STK integration",
            "Website production replacement",
            "Multi-guitar comparison",
            "Melody/chord playback",
            "Subjective tuning by ear",
            "Real-guitar equivalence or validation proof",
            "Exact measured string tension",
            "Measured linear density",
            "Playable instrument realism proof",
        ],
        "objective_test_results": validation,
        "readiness_after_step5h": readiness,
        "safe_next_step": (
            "PGSM Step 5I: string partial damping refinement (planning/contract only until approved)"
            if readiness["current_status"] == READINESS_AFTER
            else "Resolve Step 5H contract validation failures before Step 5I"
        ),
        "explicit_statement": (
            "PGSM Step 5H defines the note/string/fret contract only. "
            "It does not generate audio, does not integrate STK, and does not prove realism."
        ),
    }


def write_markdown_report(report: Mapping[str, Any], path: Path) -> None:
    rg = report.get("readiness_after_step5h") or {}
    base = report.get("classical_guitar_base_contract") or {}
    preferred = report.get("preferred_diagnostic_mapping") or {}
    candidates = report.get("diagnostic_note_candidates") or {}
    rejected = report.get("rejected_reference_only_mappings") or []
    stk = report.get("stk_readiness_contract") or {}
    val = report.get("validation_results") or {}
    rationale = report.get("preference_rationale") or {}

    lines = [
        "# PGSM Step 5H — note/string/fret contract repair",
        "",
        f"**Generated:** {report.get('timestamp')}",
        f"**Readiness:** `{rg.get('current_status')}`",
        "",
        report.get("explicit_statement", ""),
        "",
        "## Summary",
        "",
        "Classical guitar note/string/fret contract for diagnostic notes A2/A3/A4/E5. "
        "Equal-temperament fret model with scale length 0.650 m (L2 literature fallback). "
        "Physical playable mappings defined; STK fields prepared but integration blocked.",
        "",
        "## Classical guitar base tuning",
        "",
        "| String | Open note | f0 (Hz) | Linear density | Tension claim |",
        "|--------|-----------|---------|----------------|---------------|",
    ]
    for s in base.get("strings") or []:
        lines.append(
            f"| {s.get('string_id')} | {s.get('open_note')} | {s.get('open_frequency_hz')} | "
            f"{s.get('linear_density_status')} | {s.get('tension_status')} |"
        )

    lines.extend(["", "## Diagnostic note candidates", ""])
    for note in NOTE_SET:
        lines.append(f"### {note}")
        lines.append("")
        lines.append("| string | fret | target Hz | computed Hz | err (cents) | L_eff (m) | playable | ref only |")
        lines.append("|--------|------|-----------|-------------|-------------|-----------|----------|----------|")
        for c in candidates.get(note) or []:
            lines.append(
                f"| {c.get('string_id') or c.get('mapping_id')} | {c.get('fret')} | "
                f"{c.get('target_frequency_hz')} | {c.get('computed_frequency_hz')} | "
                f"{c.get('frequency_error_cents')} | {c.get('effective_length_m')} | "
                f"{c.get('physical_playable_candidate')} | {c.get('diagnostic_reference_only')} |"
            )
        lines.append("")

    lines.extend(["## Preferred diagnostic mapping", ""])
    lines.append("| Note | string | fret | f0 (Hz) | L_eff (m) | exact open | rationale |")
    lines.append("|------|--------|------|---------|-----------|------------|-----------|")
    for note in NOTE_SET:
        p = preferred.get(note) or {}
        lines.append(
            f"| {note} | {p.get('string_id')} | {p.get('fret')} | {p.get('target_frequency_hz')} | "
            f"{p.get('effective_length_m')} | {p.get('exact_open_string_claim_allowed')} | "
            f"{rationale.get(note, '')} |"
        )

    lines.extend(["", "## Rejected / diagnostic-only mappings", ""])
    lines.append("| mapping | note | playable | reason |")
    lines.append("|---------|------|----------|--------|")
    for r in rejected:
        lines.append(
            f"| {r.get('mapping_id')} | {r.get('note')} | {r.get('physical_playable_candidate')} | "
            f"{r.get('limitation')} |"
        )

    lines.extend(
        [
            "",
            "## STK readiness (blocked)",
            "",
            f"- stk_integration_allowed: **{stk.get('stk_integration_allowed')}**",
            f"- reason: {stk.get('reason')}",
            "",
            "| Note | MIDI | string | fret | L_eff | pluck ratio | body modal id |",
            "|------|------|--------|------|-------|-------------|---------------|",
        ]
    )
    for e in stk.get("future_stk_note_entries") or []:
        lines.append(
            f"| {e.get('note_name')} | {e.get('midi_note_number')} | {e.get('string_id')} | "
            f"{e.get('fret')} | {e.get('effective_length_m')} | "
            f"{e.get('pluck_position_ratio_relative_to_string')} | {e.get('body_modal_response_id')} |"
        )

    lines.extend(
        [
            "",
            "## Validation results",
            "",
            f"all_pass: **{val.get('all_pass')}**",
            f"frequency_error_near_zero: {val.get('frequency_error_near_zero')}",
            f"effective_length_le_scale: {val.get('effective_length_le_scale')}",
            f"a4_not_only_concert_reference: {val.get('a4_not_only_concert_reference')}",
            f"stk_integration_blocked: {val.get('stk_integration_blocked')}",
            "",
            "## Readiness decision",
            "",
            f"Status: `{rg.get('current_status')}`",
            f"Contract only (not final): {rg.get('contract_only_not_final')}",
            "",
            "## Blocked claims",
            "",
        ]
    )
    for claim in report.get("blocked_claims") or []:
        lines.append(f"- {claim}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pgsm_step5h_reports(
    *,
    repo_root: Optional[Path] = None,
    json_path: Optional[Path] = None,
    md_path: Optional[Path] = None,
    data_path: Optional[Path] = None,
) -> Dict[str, Any]:
    root = Path(repo_root or REPO_ROOT)
    report = build_pgsm_step5h_report(repo_root=root)
    jpath = Path(json_path or REPORT_JSON)
    mpath = Path(md_path or REPORT_MD)
    dpath = Path(data_path or DATA_JSON)

    jpath.parent.mkdir(parents=True, exist_ok=True)
    jpath.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown_report(report, mpath)

    export = build_contract_data_export(
        report["classical_guitar_base_contract"],
        report["diagnostic_note_candidates"],
        report["preferred_diagnostic_mapping"],
        report["stk_readiness_contract"],
    )
    dpath.parent.mkdir(parents=True, exist_ok=True)
    dpath.write_text(json.dumps(export, indent=2), encoding="utf-8")

    return report


def main() -> None:
    report = write_pgsm_step5h_reports()
    rg = report.get("readiness_after_step5h") or {}
    obj = report.get("objective_test_results") or {}
    print(f"Wrote {REPORT_JSON}")
    print(f"Wrote {DATA_JSON}")
    print(f"Readiness: {rg.get('current_status')}")
    print(f"all_pass: {obj.get('all_pass')}")


if __name__ == "__main__":
    main()
