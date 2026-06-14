#!/usr/bin/env python3
"""
PGSM Step 3D — numeric pre-synthesis contract review.
Defines what is safe/blocked before any future audio work. No WAV, no STK, no FEM/ROM.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from pgsm_step2_1_parameter_targets import load_step_report
from pgsm_step3c_numeric_calibration import STEP22B_POLICY_PRIMARY, resolve_step22b_policy
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE

PGSM_STEP3D_VERSION = "pgsm_step3d_pre_synthesis_contract_v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON = REPO_ROOT / "audio" / "debug_reports" / "pgsm_step3d_pre_synthesis_contract.json"
REPORT_MD = REPO_ROOT / "audio" / "debug_reports" / "pgsm_step3d_pre_synthesis_contract.md"

STEP3C_READINESS_REQUIRED = "ready_for_step3d_numeric_pre_synthesis_contract"
STEP4A_READINESS = "ready_for_step4a_single_note_diagnostic_audio_only"

UPSTREAM_REPORTS: Tuple[Tuple[str, str], ...] = (
    ("step1", "pgsm_step1_physical_factor_registry.json"),
    ("step2", "pgsm_step2_physical_interaction_map.json"),
    ("step2_1", "pgsm_step2_1_parameter_targets.json"),
    ("step2_2", "pgsm_step2_2_tonewood_material_library.json"),
    ("step2_2b", "pgsm_step2_2b_material_alignment_audit.json"),
    ("step3a", "pgsm_step3a_numerical_ir_testbench.json"),
    ("step3b", "pgsm_step3b_modal_response_validation.json"),
    ("step3c", "pgsm_step3c_numeric_calibration.json"),
)

MUSICAL_AUDIO_KEYWORDS = (
    "ready_for_musical",
    "musical_wav_synthesis_allowed",
    "musical_audio_synthesis_allowed",
    "ready_for_final_synthesis",
    "ready_for_production",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _report_path(root: Path, filename: str) -> Path:
    return root / "audio" / "debug_reports" / filename


def load_upstream_reports(repo_root: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    root = Path(repo_root or REPO_ROOT)
    loaded: Dict[str, Dict[str, Any]] = {}
    missing: List[str] = []
    for key, filename in UPSTREAM_REPORTS:
        path = _report_path(root, filename)
        if not path.is_file():
            missing.append(str(path))
            continue
        loaded[key] = load_step_report(path)
    if missing:
        raise FileNotFoundError(
            "PGSM Step 3D requires upstream reports:\n  " + "\n  ".join(missing)
        )
    return loaded


def _walk_bool_flags(obj: Any, prefix: str = "") -> List[Tuple[str, bool]]:
    found: List[Tuple[str, bool]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, bool) and any(m in k.lower() for m in ("musical", "wav", "audio", "stk")):
                found.append((key, v))
            found.extend(_walk_bool_flags(v, key))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            found.extend(_walk_bool_flags(item, f"{prefix}[{i}]"))
    return found


def _claims_musical_audio_readiness(report: Mapping[str, Any]) -> bool:
    text = json.dumps(report).lower()
    for kw in MUSICAL_AUDIO_KEYWORDS:
        if kw in text and "false" not in text.split(kw)[0][-20:]:
            pass
    for key, val in _walk_bool_flags(report):
        if val is True and any(x in key.lower() for x in ("musical_wav", "musical_audio")):
            return True
    for block_key in (
        "readiness_after_step3c",
        "readiness_after_step3b",
        "readiness_after_step3a",
        "readiness_gate",
        "readiness_after_step3d",
    ):
        block = report.get(block_key) or {}
        if block.get("musical_wav_synthesis_allowed") is True:
            return True
        if block.get("musical_audio_synthesis_allowed") is True:
            return True
        status = str(block.get("current_status", "")).lower()
        if "musical" in status and "not" not in status:
            return True
    status = str(report.get("status", "")).lower()
    if "musical" in status and "no_" not in status:
        return True
    return False


def _contract_field(
    name: str,
    *,
    required: bool,
    source_level: str,
    source_report: str,
    value_summary: Any,
    notes: str = "",
) -> Dict[str, Any]:
    return {
        "field": name,
        "required_for_step4a": required,
        "source_fallback_level": source_level,
        "source_report": source_report,
        "value_summary": value_summary,
        "notes": notes,
    }


def verify_upstream_readiness(
    reports: Mapping[str, Mapping[str, Any]],
    *,
    repo_root: Path,
) -> Dict[str, Any]:
    checks: Dict[str, Dict[str, Any]] = {}

    step1 = reports["step1"]
    checks["step1"] = {
        "report_version": step1.get("report_version"),
        "status": step1.get("status"),
        "complete": "complete" in str(step1.get("status", "")),
        "pass": "complete" in str(step1.get("status", "")),
    }

    step2 = reports["step2"]
    checks["step2"] = {
        "report_version": step2.get("report_version"),
        "status": step2.get("status"),
        "complete": "complete" in str(step2.get("status", "")),
        "pass": "complete" in str(step2.get("status", "")),
    }

    step21 = reports["step2_1"]
    gate = step21.get("readiness_gate") or {}
    checks["step2_1"] = {
        "readiness": gate.get("current_status"),
        "numerical_ir_allowed": gate.get("numerical_impulse_response_allowed") is True,
        "musical_blocked": gate.get("musical_audio_synthesis_allowed") is False,
        "pass": gate.get("current_status") == "ready_for_numerical_impulse_response_only",
    }

    step22 = reports["step2_2"]
    checks["step2_2"] = {
        "report_version": step22.get("report_version"),
        "wood_entries_present": bool(step22.get("wood_entries") or step22.get("library_summary")),
        "validation_all_pass": (step22.get("validation_results") or {}).get("all_pass"),
        "pass": bool(step22.get("report_version")),
    }

    step22b_path = _report_path(repo_root, "pgsm_step2_2b_material_alignment_audit.json")
    policy_id, step22b = resolve_step22b_policy(step22b_path)
    checks["step2_2b"] = {
        "report_version": step22b.get("report_version"),
        "primary_policy": policy_id,
        "fem_primary": policy_id == STEP22B_POLICY_PRIMARY,
        "pass": policy_id == STEP22B_POLICY_PRIMARY,
    }

    step3a = reports["step3a"]
    r3a = step3a.get("readiness_after_step3a") or {}
    checks["step3a"] = {
        "status": step3a.get("status"),
        "readiness": r3a.get("current_status"),
        "objective_all_pass": (step3a.get("objective_test_results") or {}).get("all_pass"),
        "musical_blocked": r3a.get("musical_wav_synthesis_allowed") is False,
        "pass": "complete" in str(step3a.get("status", "")),
    }

    step3b = reports["step3b"]
    r3b = step3b.get("readiness_after_step3b") or {}
    checks["step3b"] = {
        "status": step3b.get("status"),
        "readiness": r3b.get("current_status"),
        "musical_blocked": r3b.get("musical_wav_synthesis_allowed") is False,
        "pass": "complete" in str(step3b.get("status", "")),
    }

    step3c = reports["step3c"]
    r3c = step3c.get("readiness_after_step3c") or {}
    checks["step3c"] = {
        "status": step3c.get("status"),
        "readiness": r3c.get("current_status"),
        "objective_all_pass": (step3c.get("objective_test_results") or {}).get("all_pass"),
        "step22b_policy": step3c.get("step22b_policy_loaded"),
        "pass": r3c.get("current_status") == STEP3C_READINESS_REQUIRED
        and (step3c.get("objective_test_results") or {}).get("all_pass") is True,
    }

    musical_claims: List[str] = []
    for key, doc in reports.items():
        if _claims_musical_audio_readiness(doc):
            musical_claims.append(key)

    all_pass = all(c.get("pass") for c in checks.values()) and not musical_claims

    return {
        "checks": checks,
        "musical_audio_readiness_claims_found": musical_claims,
        "no_musical_audio_readiness": len(musical_claims) == 0,
        "all_pass": all_pass,
    }


def build_synthesis_input_contract(reports: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    step3c = reports["step3c"]
    step3b = reports["step3b"]
    step3a = reports["step3a"]
    step21 = reports["step2_1"]
    string_val = step3b.get("string_consistency_validation") or {}
    ir3c = step3c.get("calibrated_ir_summary") or {}
    region = step3c.get("region_weight_calibration") or {}
    adm = step3c.get("admittance_normalization") or {}
    material = step3c.get("chosen_material_values") or {}
    policy = step3c.get("material_policy_applied") or {}

    q_after = (step3c.get("q_tau_calibration") or {}).get("after") or {}
    pack = step3a.get("parameter_pack_summary") or {}

    guitar_body: List[Dict[str, Any]] = [
        _contract_field(
            "sample_id",
            required=True,
            source_level="L0_measured_audit",
            source_report="step3c",
            value_summary=step3c.get("sample_id"),
        ),
        _contract_field(
            "modal_frequencies_hz",
            required=True,
            source_level="L1_reference_shared_ROM",
            source_report="step3a",
            value_summary=f"{step3a.get('modal_catalog', {}).get('mode_count', '643')} modes reference_shared",
            notes="Per-sample predicted_modes from ROM catalog; not multi-guitar proof",
        ),
        _contract_field(
            "calibrated_Q",
            required=True,
            source_level="L2_numeric_calibration_target",
            source_report="step3c",
            value_summary={"mean_Q": q_after.get("mean_Q"), "Q_range": [20, 80]},
            notes="numeric_target_not_measurement",
        ),
        _contract_field(
            "calibrated_tau_s",
            required=True,
            source_level="L2_numeric_calibration_target",
            source_report="step3c",
            value_summary={"mean_tau_s": q_after.get("mean_tau_s")},
        ),
        _contract_field(
            "calibrated_modal_weights",
            required=True,
            source_level="L1_reference_shared_ROM",
            source_report="step3a/3c",
            value_summary="W_rad, W_air, W_exc from Step 3A with Step 3C Q/tau and region calibration",
        ),
        _contract_field(
            "normalized_bridge_admittance",
            required=True,
            source_level="L2_normalized_proxy",
            source_report="step3c",
            value_summary={
                "Y_abs_normalized_peak1_max": adm.get("Y_abs_normalized_peak1_max"),
                "absolute_claim_blocked": adm.get("absolute_bridge_mobility_claim_blocked"),
            },
        ),
        _contract_field(
            "calibrated_region_weights",
            required=True,
            source_level="L2_numeric_calibration_target",
            source_report="step3c",
            value_summary=region.get("calibrated"),
            notes="not_measured_radiation; reference_shared_limitation",
        ),
        _contract_field(
            "top_back_air_contribution_proxies",
            required=True,
            source_level="L2_radiation_proxy",
            source_report="step3c",
            value_summary=region.get("calibrated"),
        ),
        _contract_field(
            "material_policy_used",
            required=True,
            source_level="L0_FEM_primary_policy",
            source_report="step2_2b/3c",
            value_summary=policy,
        ),
        _contract_field(
            "fem_primary_material_values",
            required=True,
            source_level="L0_FEM_primary",
            source_report="step3c",
            value_summary={
                "top_wood_id": material.get("top_wood_id"),
                "back_wood_id": material.get("back_wood_id"),
                "pgsm_override_blocked": material.get("pgsm_override_blocked"),
            },
        ),
    ]

    string_excitation: List[Dict[str, Any]] = [
        _contract_field(
            "note_frequency_hz",
            required=True,
            source_level="L1_harmonic_reference",
            source_report="step3a/3b",
            value_summary=step3a.get("note_reference_hz", 440.0),
            notes="A4 harmonic reference only",
        ),
        _contract_field(
            "effective_vibrating_length_m",
            required=True,
            source_level="L2_fallback_or_documented",
            source_report="step3b",
            value_summary=string_val.get("calibrated_L_for_mid_tension_m"),
            notes="Open L=0.65 m rejected; fretted interpretation documented",
        ),
        _contract_field(
            "string_interpretation",
            required=True,
            source_level="L2_documented",
            source_report="step3b",
            value_summary=string_val.get("recommended_string_interpretation"),
        ),
        _contract_field(
            "string_tension_N",
            required=True,
            source_level="L2_proxy_blocked_exact",
            source_report="step3b",
            value_summary={
                "inferred_N": string_val.get("step3a_inferred_tension_N"),
                "realistic": string_val.get("step3a_tension_realistic"),
            },
        ),
        _contract_field(
            "linear_density_kg_m",
            required=True,
            source_level="L2_literature_fallback",
            source_report="step2_1/3b",
            value_summary=string_val.get("fallback_mu_kg_m", 0.0018),
        ),
        _contract_field(
            "pluck_position_ratio",
            required=True,
            source_level="L2_fixed_testbench",
            source_report="step3a",
            value_summary=step3a.get("fixed_pluck_position", 0.18),
        ),
        _contract_field(
            "pluck_duration_ms",
            required=True,
            source_level="L2_proxy",
            source_report="step2_1",
            value_summary="From parameter pack / pluck contact proxy",
        ),
        _contract_field(
            "F_bridge_proxy",
            required=True,
            source_level="L2_proxy",
            source_report="step2",
            value_summary="Harmonic excitation × bridge mobility × modal coupling",
        ),
        _contract_field(
            "exact_open_string_claim_allowed",
            required=True,
            source_level="L0_blocked",
            source_report="step3b",
            value_summary=False,
            notes="Blocked unless measured/validated; 588 N case rejected",
        ),
    ]

    output_restrictions = {
        "no_calibrated_SPL_claim": True,
        "no_measured_radiation_claim": True,
        "no_multi_guitar_proof": True,
        "no_exact_material_claim_from_L2_fallback": True,
        "no_final_physical_accuracy_claim": True,
        "reference_shared_modal_limitation": True,
    }

    all_fields = guitar_body + string_excitation
    every_field_has_level = all(f.get("source_fallback_level") for f in all_fields)

    return {
        "guitar_body_numeric_fields": guitar_body,
        "string_excitation_fields": string_excitation,
        "output_restrictions": output_restrictions,
        "calibrated_ir_reference": {
            "peak_time_ms": ir3c.get("peak_time_ms"),
            "decay_time_ms": ir3c.get("decay_time_ms"),
            "no_delayed_body_event": ir3c.get("no_delayed_body_event"),
            "no_artificial_end_rise": ir3c.get("no_artificial_end_rise"),
        },
        "every_field_has_source_fallback_level": every_field_has_level,
        "implementation_status": "contract_defined_not_implemented",
    }


def build_allowed_step4a_scope() -> Dict[str, Any]:
    return {
        "label": "PGSM diagnostic audio, not final guitar",
        "allowed": [
            "single_guitar (sample_000 only initially)",
            "single_note diagnostic WAV",
            "low_amplitude diagnostic WAV only",
            "clearly labeled non-final output",
            "save stems/diagnostics separately",
            "compare numeric envelope against Step 3C before accepting WAV",
            "objective decay/envelope metrics only",
        ],
        "required_controls": [
            "no_website_default_change",
            "no_listening_based_parameter_tuning",
            "no_hidden_EQ_body_tail_echo_layer",
            "no_multi_guitar_comparison",
        ],
        "not_final_synthesis": True,
    }


def build_blocked_steps() -> List[str]:
    return [
        "Final STK integration",
        "Website production replacement",
        "Multi-guitar timbre proof",
        "Melody/chord playback",
        "Subjective tuning by ear",
        "Claims of real guitar equivalence",
        "Absolute SPL/radiation claims",
        "FEM/ROM/M4 surrogate inference",
        "Claim model is solved",
    ]


def build_artifact_guard_contract() -> Dict[str, Any]:
    forbidden = [
        {"artifact": "delayed_body_tail_stem", "forbidden": True, "reason": "V6 failure: independent delayed body layer"},
        {"artifact": "helmholtz_echo_ir", "forbidden": True, "reason": "V6 failure: late convolve echo"},
        {"artifact": "post_hoc_EQ_body_layer", "forbidden": True, "reason": "Not physical radiation path"},
        {"artifact": "independent_delayed_body_onset", "forbidden": True, "reason": "Body must start at t=0 with F_bridge"},
        {"artifact": "second_pluck_onset", "forbidden": True, "reason": "Double onset artifact"},
        {"artifact": "hard_gate_tail_cut", "forbidden": True, "reason": "Artificial tail collapse"},
        {"artifact": "end_rise_noise", "forbidden": True, "reason": "Unphysical end rise"},
        {"artifact": "artificial_reverb", "forbidden": True, "reason": "Not in PGSM causal chain"},
        {"artifact": "arbitrary_wood_to_gain_mapping", "forbidden": True, "reason": "Wood ID must not map to arbitrary EQ/gain"},
        {
            "artifact": "reference_shared_modal_as_multi_guitar_proof",
            "forbidden": True,
            "reason": "Shared ROM catalog cannot prove per-guitar timbre differentiation",
        },
    ]
    return {
        "forbidden_artifacts": forbidden,
        "causal_chain_required": "F_bridge(t) at t=0 → modal oscillators → radiation sum",
        "pass": True,
    }


def build_numeric_to_audio_consistency_checks() -> Dict[str, Any]:
    checks = [
        {
            "check": "envelope_follows_step3c_calibrated_IR",
            "required": True,
            "tolerance": "approximate; objective metric not listening",
        },
        {
            "check": "no_delayed_independent_body_onset",
            "required": True,
        },
        {
            "check": "no_end_rise",
            "required": True,
        },
        {
            "check": "no_hard_gate",
            "required": True,
        },
        {
            "check": "decay_metrics_reported",
            "required": True,
            "metrics": ["peak_time_ms", "minus_20_dB", "minus_40_dB", "minus_60_dB", "late_early_energy_ratio"],
        },
        {
            "check": "modal_peak_locations_preserved_in_spectrum",
            "required": True,
        },
        {
            "check": "Q_tau_not_bypassed_by_post_processing",
            "required": True,
        },
        {
            "check": "region_weights_radiation_proxy_only",
            "required": True,
        },
        {
            "check": "bridge_admittance_normalization_applied",
            "required": True,
        },
        {
            "check": "F_bridge_proxy_documented",
            "required": True,
        },
        {
            "check": "string_tension_interpretation_documented",
            "required": True,
        },
        {
            "check": "gain_normalization_separated_from_physics",
            "required": True,
        },
    ]
    return {
        "checks": checks,
        "must_pass_before_WAV_accepted": True,
        "tuning_by_listening_forbidden": True,
    }


def build_required_future_outputs() -> List[str]:
    return [
        "diagnostic_WAV_labeled_non_final",
        "numeric_envelope_comparison_vs_step3c.json",
        "decay_metrics_report.json",
        "spectral_peak_alignment_report.json",
        "stems_diagnostics_separate_from_master",
        "artifact_guard_pass_fail.json",
        "explicit_blocked_claims_acknowledgment",
    ]


def build_readiness_after_step3d(
    upstream: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> Dict[str, Any]:
    if not upstream.get("all_pass"):
        status = "failed_pre_synthesis_contract"
    elif not contract.get("every_field_has_source_fallback_level"):
        status = "blocked_due_to_missing_contract_fields"
    else:
        status = STEP4A_READINESS

    return {
        "current_status": status,
        "final_synthesis_ready": False,
        "musical_wav_synthesis_allowed": False,
        "stk_integration_allowed": False,
        "multi_guitar_comparison_allowed": False,
        "step4a_diagnostic_single_note_allowed": status == STEP4A_READINESS,
        "website_production_replacement_allowed": False,
    }


def build_pgsm_step3d_report(*, repo_root: Optional[Path] = None) -> Dict[str, Any]:
    root = Path(repo_root or REPO_ROOT)
    reports = load_upstream_reports(root)
    upstream = verify_upstream_readiness(reports, repo_root=root)
    contract = build_synthesis_input_contract(reports)
    allowed_4a = build_allowed_step4a_scope()
    blocked = build_blocked_steps()
    artifact_guard = build_artifact_guard_contract()
    consistency = build_numeric_to_audio_consistency_checks()
    future_outputs = build_required_future_outputs()
    readiness = build_readiness_after_step3d(upstream, contract)

    if readiness["current_status"] == STEP4A_READINESS:
        safe_next = (
            "PGSM Step 4A: single-note diagnostic audio only "
            "(low amplitude, labeled non-final, envelope check vs Step 3C; no STK/website/multi-guitar)"
        )
    elif readiness["current_status"] == "blocked_due_to_missing_contract_fields":
        safe_next = "Complete missing synthesis input contract fields before Step 4A"
    else:
        safe_next = "Fix failed upstream readiness before any audio attempt"

    return {
        "report_version": PGSM_STEP3D_VERSION,
        "timestamp": _utc_now(),
        "status": "pgsm_step3d_pre_synthesis_contract_complete",
        "no_audio_generated": True,
        "no_wav_generated": True,
        "no_stk_integration": True,
        "no_fem_run": True,
        "no_rom_run": True,
        "website_default": DEFAULT_WEBSITE_STK_MODE,
        "website_default_unchanged": True,
        "upstream_readiness_summary": upstream,
        "synthesis_input_contract": contract,
        "allowed_step4a_scope": allowed_4a,
        "blocked_steps": blocked,
        "artifact_guard_contract": artifact_guard,
        "numeric_to_audio_consistency_checks": consistency,
        "required_future_outputs": future_outputs,
        "readiness_after_step3d": readiness,
        "blocked_claims": blocked + [
            "Calibrated SPL without measurement",
            "Measured radiation from calibrated region weights",
            "Multi-guitar proof from reference_shared catalog",
            "Exact open-string physics without validation",
            "Model solved / production-ready guitar sound",
        ],
        "safe_next_step": safe_next,
        "explicit_statement": (
            "PGSM Step 3D defines a pre-synthesis contract only. It does not synthesize sound."
        ),
    }


def write_markdown_report(report: Mapping[str, Any], path: Path) -> None:
    upstream = report.get("upstream_readiness_summary") or {}
    checks = upstream.get("checks") or {}
    contract = report.get("synthesis_input_contract") or {}
    allowed = report.get("allowed_step4a_scope") or {}
    artifact = report.get("artifact_guard_contract") or {}
    consistency = report.get("numeric_to_audio_consistency_checks") or {}
    readiness = report.get("readiness_after_step3d") or {}

    lines = [
        "# PGSM Step 3D — numeric pre-synthesis contract",
        "",
        f"**Generated:** {report.get('timestamp')}",
        f"**Readiness:** `{readiness.get('current_status')}`",
        "",
        report.get("explicit_statement", ""),
        "",
        "## Upstream readiness",
        "",
        "| Step | Pass | Detail |",
        "|------|------|--------|",
    ]
    for step, c in checks.items():
        detail = c.get("readiness") or c.get("status") or c.get("primary_policy") or c.get("report_version")
        lines.append(f"| {step} | {c.get('pass')} | {detail} |")

    lines.extend(
        [
            "",
            f"Upstream all_pass: **{upstream.get('all_pass')}**",
            "",
            "## Future synthesis input contract",
            "",
            "Guitar/body and string/excitation fields require `source_fallback_level` on every entry.",
            f"Every field has level: **{contract.get('every_field_has_source_fallback_level')}**",
            "",
            "### Output restrictions",
            "",
        ]
    )
    for k, v in (contract.get("output_restrictions") or {}).items():
        lines.append(f"- {k}: {v}")

    lines.extend(["", "## Allowed Step 4A scope (not executed)", ""])
    lines.append(f"Label: {allowed.get('label')}")
    for item in allowed.get("allowed") or []:
        lines.append(f"- {item}")

    lines.extend(["", "## Blocked steps", ""])
    for item in report.get("blocked_steps") or []:
        lines.append(f"- {item}")

    lines.extend(["", "## Artifact guard", "", "| Artifact | Forbidden |", "|----------|-----------|"])
    for row in artifact.get("forbidden_artifacts") or []:
        lines.append(f"| {row.get('artifact')} | {row.get('forbidden')} |")

    lines.extend(["", "## Future numeric-to-audio consistency checks", ""])
    for chk in consistency.get("checks") or []:
        lines.append(f"- {chk.get('check')} (required={chk.get('required')})")

    lines.extend(
        [
            "",
            "## Readiness decision",
            "",
            f"- Status: `{readiness.get('current_status')}`",
            f"- Step 4A diagnostic allowed: {readiness.get('step4a_diagnostic_single_note_allowed')}",
            f"- Final synthesis ready: {readiness.get('final_synthesis_ready')}",
            "",
            "## Safe next step",
            "",
            report.get("safe_next_step", ""),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pgsm_step3d_reports(
    *,
    repo_root: Optional[Path] = None,
    json_path: Optional[Path] = None,
    md_path: Optional[Path] = None,
) -> Dict[str, Any]:
    root = Path(repo_root or REPO_ROOT)
    report = build_pgsm_step3d_report(repo_root=root)
    jpath = Path(json_path or REPORT_JSON)
    mpath = Path(md_path or REPORT_MD)
    jpath.parent.mkdir(parents=True, exist_ok=True)
    jpath.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown_report(report, mpath)
    return report


def main() -> None:
    report = write_pgsm_step3d_reports()
    rg = report.get("readiness_after_step3d") or {}
    print(f"Wrote {REPORT_JSON}")
    print(f"Readiness: {rg.get('current_status')}")
    print(f"Upstream all_pass: {(report.get('upstream_readiness_summary') or {}).get('all_pass')}")


if __name__ == "__main__":
    main()
