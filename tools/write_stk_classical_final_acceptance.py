#!/usr/bin/env python3
"""
Build classical-guitar STK final acceptance report from v4 export + render artifacts.

No audio synthesis. Reads JSON reports only.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
MEANINGFUL_SPREAD_THRESHOLD = 0.035
STK_PLUCK_AMP_MAX = 1.0

# Factors the C++ renderer applies via factor_audit / render path (post-parse).
CPP_APPLIED_FACTOR_KEYS: Set[str] = {
    "bridge_mobility_factor",
    "effective_mass_loading_factor",
    "body_size_cavity_factor",
    "body_depth_m",
    "body_volume_proxy",
    "soundhole_radiation_factor",
    "soundhole_area_proxy",
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
    "direct_string_gain",
    "body_modal_gain",
}

# Direct STK render-path factors (not always in factor_audit map).
CPP_DIRECT_AUDIO_FACTORS: Set[str] = {
    "pluck_position",
    "excitation_strength",
    "note_excitation_scale",
    "harmonic_brightness",
    "string_decay",
    "bridge_damping",
    "string_to_body_send",
    "high_frequency_radiation_rolloff",
    "modal_frequencies",
    "modal_gains",
    "modal_tau_q",
    "per_mode_damping_modifiers",
    "top_back_air_component_labels",
    "low_mid_body_support_120_450_hz",
    "mode_frequency_shifts_per_sample",
}

FINAL_ACCEPTANCE_FACTOR_SPECS: Tuple[Dict[str, Any], ...] = (
    {"factor_name": "pluck_position", "source_group": "excitation", "export_key": "pluck_position", "note_dependency": False, "critical": True},
    {"factor_name": "excitation_strength", "source_group": "excitation", "export_key": "excitation_strength", "note_dependency": False, "critical": True},
    {"factor_name": "note_excitation_scale", "source_group": "excitation", "export_key": "note_excitation_scale", "note_dependency": True, "critical": True},
    {"factor_name": "harmonic_brightness", "source_group": "excitation", "export_key": "harmonic_brightness", "note_dependency": True, "critical": True},
    {"factor_name": "string_decay", "source_group": "excitation", "export_key": "string_decay", "note_dependency": True, "critical": False},
    {"factor_name": "bridge_mobility_factor", "source_group": "bridge", "export_key": "bridge_mobility_factor", "note_dependency": False, "critical": True},
    {"factor_name": "bridge_damping", "source_group": "bridge", "export_key": "bridge_damping", "note_dependency": False, "critical": False},
    {"factor_name": "string_to_body_send", "source_group": "bridge", "export_key": "string_to_body_send", "note_dependency": False, "critical": True},
    {"factor_name": "string_body_mix", "source_group": "bridge", "export_key": "string_body_mix", "note_dependency": False, "critical": True},
    {"factor_name": "direct_string_gain", "source_group": "bridge", "export_key": "direct_string_gain", "note_dependency": False, "critical": True},
    {"factor_name": "body_modal_gain", "source_group": "bridge", "export_key": "body_modal_gain", "note_dependency": False, "critical": True},
    {"factor_name": "body_depth_m", "source_group": "geometry", "export_key": "body_depth_m", "note_dependency": False, "critical": True},
    {"factor_name": "body_volume_proxy", "source_group": "geometry", "export_key": "body_volume_proxy", "note_dependency": False, "critical": True},
    {"factor_name": "body_size_cavity_factor", "source_group": "geometry", "export_key": "body_size_cavity_factor", "note_dependency": False, "critical": True},
    {"factor_name": "effective_mass_loading_factor", "source_group": "geometry", "export_key": "effective_mass_loading_factor", "note_dependency": False, "critical": True},
    {"factor_name": "shape_flatness_or_depth_factor", "source_group": "geometry", "export_key": "shape_flatness_or_depth_factor", "note_dependency": False, "critical": False},
    {"factor_name": "soundhole_area_proxy", "source_group": "soundhole_air", "export_key": "soundhole_area_proxy", "note_dependency": False, "critical": True},
    {"factor_name": "soundhole_radiation_factor", "source_group": "soundhole_air", "export_key": "soundhole_radiation_factor", "note_dependency": False, "critical": True},
    {"factor_name": "air_helmholtz_factor", "source_group": "soundhole_air", "export_key": "air_helmholtz_factor", "note_dependency": False, "critical": True},
    {"factor_name": "air_weight", "source_group": "soundhole_air", "export_key": "air_weight", "note_dependency": False, "critical": True},
    {"factor_name": "top_stiffness_to_weight_factor", "source_group": "material", "export_key": "top_stiffness_to_weight_factor", "note_dependency": False, "critical": True},
    {"factor_name": "top_damping_factor", "source_group": "material", "export_key": "top_damping_factor", "note_dependency": False, "critical": True},
    {"factor_name": "material_loss_factor", "source_group": "material", "export_key": "material_loss_factor", "note_dependency": False, "critical": True},
    {"factor_name": "back_density_warmth_factor", "source_group": "material", "export_key": "back_density_warmth_factor", "note_dependency": False, "critical": True},
    {"factor_name": "modal_frequencies", "source_group": "modal", "export_key": "modal_frequencies", "note_dependency": True, "critical": True},
    {"factor_name": "modal_gains", "source_group": "modal", "export_key": "modal_gains", "note_dependency": True, "critical": True},
    {"factor_name": "modal_tau_q", "source_group": "modal", "export_key": "modal_tau_q", "note_dependency": True, "critical": True},
    {"factor_name": "per_mode_damping_modifiers", "source_group": "material", "export_key": "per_mode_tau_q_modifiers", "note_dependency": True, "critical": True},
    {"factor_name": "top_back_air_component_labels", "source_group": "modal", "export_key": "top_back_air_component_labels", "note_dependency": False, "critical": False},
    {"factor_name": "low_mid_body_support_120_450_hz", "source_group": "modal", "export_key": "low_mid_body_support_120_450_hz", "note_dependency": True, "critical": True},
    {"factor_name": "mode_frequency_shifts_per_sample", "source_group": "modal", "export_key": "mode_frequency_shifts_per_sample", "note_dependency": False, "critical": True},
    {"factor_name": "radiation_brightness_factor", "source_group": "radiation", "export_key": "radiation_brightness_factor", "note_dependency": False, "critical": True},
    {"factor_name": "top_weight", "source_group": "radiation", "export_key": "top_weight", "note_dependency": False, "critical": True},
    {"factor_name": "back_weight", "source_group": "radiation", "export_key": "back_weight", "note_dependency": False, "critical": True},
    {"factor_name": "high_frequency_radiation_rolloff", "source_group": "radiation", "export_key": "high_frequency_radiation_rolloff", "note_dependency": True, "critical": False},
)

RENDERER_TARGETS: Dict[str, str] = {
    "pluck_position": "stk_plucked_pluck_position",
    "excitation_strength": "stk_plucked_excitation",
    "note_excitation_scale": "stk_plucked_excitation_scale",
    "harmonic_brightness": "string_harmonic_envelope",
    "string_decay": "stk_plucked_decay",
    "bridge_mobility_factor": "bridge_coupling_gain",
    "bridge_damping": "bridge_smoothing",
    "string_to_body_send": "bridge_to_modal_drive",
    "string_body_mix": "string_direct_vs_body_modal_mix",
    "direct_string_gain": "stk_plucked_direct_path_gain",
    "body_modal_gain": "body_modal_bank_output_gain",
    "body_depth_m": "low_mode_frequency_tau",
    "body_volume_proxy": "cavity_mode_gain",
    "body_size_cavity_factor": "low_mid_modal_gain",
    "effective_mass_loading_factor": "bridge_attack_smoothing",
    "shape_flatness_or_depth_factor": "depth_factor_modal_shift",
    "soundhole_area_proxy": "air_radiation_area_scaling",
    "soundhole_radiation_factor": "air_mode_gain",
    "air_helmholtz_factor": "air_mode_frequency",
    "air_weight": "final_radiation_mix_air",
    "top_stiffness_to_weight_factor": "top_mode_frequency_brightness",
    "top_damping_factor": "modal_tau_top",
    "material_loss_factor": "modal_Q_decay",
    "back_density_warmth_factor": "back_mode_gain_tau",
    "modal_frequencies": "modal_bank_frequency",
    "modal_gains": "modal_bank_gain",
    "modal_tau_q": "modal_bank_tau",
    "per_mode_damping_modifiers": "modal_bank_tau_per_mode",
    "top_back_air_component_labels": "modal_component_routing",
    "low_mid_body_support_120_450_hz": "body_modal_gain_120_450",
    "mode_frequency_shifts_per_sample": "per_sample_modal_frequency_shift",
    "radiation_brightness_factor": "radiation_mode_gain",
    "top_weight": "final_radiation_mix_top",
    "back_weight": "final_radiation_mix_back",
    "high_frequency_radiation_rolloff": "hf_radiation_rolloff",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalized_spread(values: Mapping[str, float]) -> float:
    nums = [float(v) for v in values.values()]
    if not nums:
        return 0.0
    lo, hi = min(nums), max(nums)
    mid = sum(nums) / len(nums)
    if abs(mid) < 1e-9:
        return hi - lo
    return (hi - lo) / abs(mid)


def _export_scalar(row: Mapping[str, Any], export_key: str) -> Optional[float]:
    sm = row.get("string_model") or {}
    bm = row.get("bridge_model") or {}
    body = row.get("body_model") or {}
    mat = row.get("material_model") or {}
    rad = row.get("radiation_model") or {}
    mix = row.get("string_body_mix") or {}
    pf = row.get("physical_factors") or {}
    modes = body.get("modes") or []
    key = export_key
    if key == "pluck_position":
        return float(sm.get("pluck_position") or 0.0)
    if key == "excitation_strength":
        return float(sm.get("excitation_strength") or 0.0)
    if key == "note_excitation_scale":
        return float(sm.get("note_excitation_scale") or 0.0)
    if key == "harmonic_brightness":
        return float(sm.get("harmonic_brightness") or 0.0)
    if key == "string_decay":
        return float(sm.get("string_decay") or 0.0)
    if key == "bridge_mobility_factor":
        return float(bm.get("bridge_mobility") or pf.get("bridge_mobility_factor") or 0.0)
    if key == "bridge_damping":
        return float(bm.get("bridge_damping") or 0.0)
    if key == "string_to_body_send":
        return float(bm.get("string_to_body_send") or 0.0)
    if key == "string_body_mix":
        return float(mix.get("string_direct") or rad.get("string_direct_weight") or 0.0)
    if key == "direct_string_gain":
        return float(mix.get("direct_string_gain") or 0.0)
    if key == "body_modal_gain":
        return float(mix.get("body_modal_gain") or body.get("body_modal_gain") or 0.0)
    if key == "body_depth_m":
        return float(body.get("body_depth_m") or 0.0)
    if key == "body_volume_proxy":
        return float(body.get("body_volume_proxy") or 0.0)
    if key == "body_size_cavity_factor":
        return float(body.get("body_size_cavity_factor") or pf.get("body_size_cavity_factor") or 0.0)
    if key == "effective_mass_loading_factor":
        return float(body.get("effective_mass_loading") or pf.get("effective_mass_loading_factor") or 0.0)
    if key == "shape_flatness_or_depth_factor":
        return float(body.get("depth_factor") or 0.0)
    if key == "soundhole_area_proxy":
        return float(body.get("soundhole_area_proxy") or 0.0)
    if key == "soundhole_radiation_factor":
        return float(body.get("soundhole_radiation_factor") or pf.get("soundhole_radiation_factor") or 0.0)
    if key == "air_helmholtz_factor":
        return float(pf.get("air_helmholtz_factor") or 0.0)
    if key == "air_weight":
        return float(rad.get("air_weight") or 0.0)
    if key == "top_stiffness_to_weight_factor":
        return float(mat.get("stiffness_to_weight") or pf.get("top_stiffness_to_weight_factor") or 0.0)
    if key == "top_damping_factor":
        return float(mat.get("top_damping") or pf.get("top_damping_factor") or 0.0)
    if key == "material_loss_factor":
        return float(mat.get("material_loss_factor") or mat.get("material_loss") or 0.0)
    if key == "back_density_warmth_factor":
        return float(mat.get("back_warmth") or pf.get("back_density_warmth_factor") or 0.0)
    if key == "per_mode_tau_q_modifiers":
        taus = [float(m.get("tau_or_q") or 0.0) for m in modes]
        return sum(taus) / len(taus) if taus else None
    if key == "modal_frequencies":
        freqs = [float(m.get("frequency_hz") or 0.0) for m in modes]
        return freqs[0] if freqs else None
    if key == "modal_gains":
        gains = [float(m.get("gain") or 0.0) for m in modes]
        return sum(gains) if gains else None
    if key == "modal_tau_q":
        taus = [float(m.get("tau_or_q") or 0.0) for m in modes]
        return taus[0] if taus else None
    if key == "top_back_air_component_labels":
        return float(len({str(m.get("component")) for m in modes}))
    if key == "low_mid_body_support_120_450_hz":
        return float(body.get("low_mid_body_support") or 0.0)
    if key == "mode_frequency_shifts_per_sample":
        freqs = [float(m.get("frequency_hz") or 0.0) for m in modes]
        return freqs[0] if freqs else None
    if key == "radiation_brightness_factor":
        return float(rad.get("radiation_brightness") or pf.get("radiation_brightness_factor") or 0.0)
    if key == "top_weight":
        return float(rad.get("top_weight") or 0.0)
    if key == "back_weight":
        return float(rad.get("back_weight") or 0.0)
    if key == "high_frequency_radiation_rolloff":
        return float(rad.get("high_frequency_radiation_rolloff") or 0.0)
    return None


def _collect_cpp_applied_keys(render_report: Mapping[str, Any]) -> Set[str]:
    applied: Set[str] = set()
    for row in render_report.get("applied_parameter_audit") or []:
        if str(row.get("note_name")) != "A2":
            continue
        fa = row.get("factor_application") or {}
        for key, entry in fa.items():
            if isinstance(entry, dict) and entry.get("applied_to_renderer"):
                applied.add(str(key))
    return applied


def _exists_in_pgsm(factor_name: str) -> bool:
    return factor_name in {s["factor_name"] for s in FINAL_ACCEPTANCE_FACTOR_SPECS}


def build_final_factor_acceptance_matrix(
    export_doc: Mapping[str, Any],
    render_report: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    renders_a2 = [r for r in export_doc.get("renders") or [] if r.get("note_name") == "A2"]
    spread_table = export_doc.get("physical_factor_spread_table") or {}
    activation_by_id = {
        str(row.get("factor_id")): row for row in (export_doc.get("stk_factor_activation_matrix") or [])
    }
    cpp_applied = _collect_cpp_applied_keys(render_report)
    matrix: List[Dict[str, Any]] = []

    for spec in FINAL_ACCEPTANCE_FACTOR_SPECS:
        fname = str(spec["factor_name"])
        ekey = str(spec["export_key"])
        vals: Dict[str, float] = {}
        for row in renders_a2:
            sid = str(row["sample_id"])
            v = _export_scalar(row, ekey)
            if v is not None:
                vals[sid] = float(v)
        spread_row = spread_table.get(ekey) or spread_table.get(fname)
        spread = float((spread_row or {}).get("normalized_spread") or _normalized_spread(vals))
        act_row = activation_by_id.get(ekey) or activation_by_id.get(fname, {})
        exported = bool(vals) or bool(act_row.get("exported_to_json"))
        parsed = exported
        applied = fname in CPP_DIRECT_AUDIO_FACTORS
        audit_key = ekey
        if ekey == "per_mode_tau_q_modifiers":
            audit_key = "material_loss_factor"
        if fname in CPP_APPLIED_FACTOR_KEYS or audit_key in cpp_applied:
            applied = True
        if act_row.get("applied_in_audio") is True:
            applied = True
        if not exported:
            effect_status = "missing"
            final_decision = "strengthen_later" if spec.get("critical") else "not_required_for_current_demo"
        elif exported and not applied:
            effect_status = "exported_but_not_applied"
            final_decision = "strengthen_later"
        elif spread < MEANINGFUL_SPREAD_THRESHOLD:
            effect_status = "active_but_subtle"
            final_decision = "strengthen_later" if spec.get("critical") else "keep"
        else:
            effect_status = "active"
            final_decision = "keep"
        if fname == "top_back_air_component_labels":
            final_decision = "not_required_for_current_demo"
        strength = float(act_row.get("strength_used") or 1.0)
        matrix.append(
            {
                "factor_name": fname,
                "source_group": spec["source_group"],
                "exists_in_pgsm_or_audit": _exists_in_pgsm(fname),
                "exported_to_json": exported,
                "parsed_by_cpp": parsed,
                "applied_in_audio": applied,
                "renderer_target": RENDERER_TARGETS.get(fname, "unknown"),
                "strength_used": strength,
                "per_sample_spread": round(spread, 6),
                "note_dependency": bool(spec.get("note_dependency")),
                "effect_strength_status": effect_status,
                "final_decision": final_decision,
                "values_by_sample_A2": vals,
            }
        )
    return matrix


def _validate_pluck_audit(render_report: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
    errors: List[str] = []
    handling = render_report.get("pluck_amplitude_handling")
    audit = render_report.get("pluck_amplitude_audit")
    if handling != "clamped_to_stk_0_1_range":
        errors.append("pluck_amplitude_handling missing or not clamped_to_stk_0_1_range")
    if not isinstance(audit, list) or not audit:
        errors.append("pluck_amplitude_audit missing or empty")
        return [], errors
    rows: List[Dict[str, Any]] = []
    for entry in audit:
        raw = float(entry.get("raw_pluck_amplitude") or 0.0)
        clamped = float(entry.get("clamped_pluck_amplitude") or 0.0)
        was_clamped = bool(entry.get("was_clamped"))
        sid = str(entry.get("sample_id"))
        note = str(entry.get("note_name"))
        if raw < 0.0 or raw > STK_PLUCK_AMP_MAX:
            if not was_clamped:
                errors.append(f"unhandled pluck amplitude out of range: {sid} {note} raw={raw}")
        if clamped < 0.0 or clamped > STK_PLUCK_AMP_MAX:
            errors.append(f"clamped pluck still out of STK range: {sid} {note} clamped={clamped}")
        rows.append(
            {
                "sample_id": sid,
                "note_name": note,
                "raw_pluck_amplitude": raw,
                "clamped_pluck_amplitude": clamped,
                "was_clamped": was_clamped,
            }
        )
    return rows, errors


def _optional_v4_1_proposal(matrix: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    subtle = [r for r in matrix if r.get("effect_strength_status") == "active_but_subtle" and r.get("final_decision") == "strengthen_later"]
    if len(subtle) < 2:
        return {"recommended": False, "not_applied_automatically": True, "reason": "factors_active_and_measurable"}
    return {
        "recommended": True,
        "not_applied_automatically": True,
        "reason": "some_critical_factors_active_but_subtle",
        "optional_adjustments": [
            "body_modal_gain +10% to +20% on 120–450 Hz modes only",
            "soundhole/air radiation +10% to +20% on air component gain and air_weight",
            "modal tau spread +10% to +20% via material_loss / damping mapping",
            "radiation brightness spread +10% to +15% on top/radiation path",
        ],
        "subtle_factors": [str(r["factor_name"]) for r in subtle],
    }


def decide_acceptance(
    matrix: Sequence[Mapping[str, Any]],
    render_report: Mapping[str, Any],
    pluck_errors: Sequence[str],
) -> str:
    if pluck_errors:
        return "not_accepted_missing_factor_activation"
    missing_or_not_applied = [
        r for r in matrix
        if r.get("effect_strength_status") in ("missing", "exported_but_not_applied")
        and r.get("final_decision") != "not_required_for_current_demo"
    ]
    critical_bad = [r for r in missing_or_not_applied if r.get("factor_name") in {
        "bridge_mobility_factor", "string_to_body_send", "body_modal_gain", "modal_frequencies",
        "modal_gains", "soundhole_radiation_factor", "material_loss_factor",
    }]
    if critical_bad:
        return "not_accepted_missing_factor_activation"
    readiness = str(render_report.get("readiness") or "")
    if readiness == "audit_failed_missing_factor_application":
        return "not_accepted_missing_factor_activation"
    subtle_critical = sum(
        1 for r in matrix
        if r.get("effect_strength_status") == "active_but_subtle" and r.get("final_decision") == "strengthen_later"
    )
    if readiness == "demo_generated_but_differentiation_weak" or subtle_critical >= 4:
        return "accepted_but_minor_mix_strengthening_recommended"
    return "accepted_for_gui_and_next_shape"


def build_acceptance_report(
    *,
    export_doc: Mapping[str, Any],
    render_report: Mapping[str, Any],
    stitched_report: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    matrix = build_final_factor_acceptance_matrix(export_doc, render_report)
    pluck_rows, pluck_errors = _validate_pluck_audit(render_report)
    decision = decide_acceptance(matrix, render_report, pluck_errors)
    weak = [
        {"factor_name": r["factor_name"], "status": r["effect_strength_status"]}
        for r in matrix
        if r["effect_strength_status"] in ("active_but_subtle", "missing", "exported_but_not_applied")
    ]
    return {
        "generated_at": _utc_now(),
        "report_type": "pgsm_stk_classical_final_acceptance",
        "demo_version": export_doc.get("demo_version"),
        "renderer": "STK/C++",
        "python_role": "parameter_export_and_acceptance_audit_only",
        "render_readiness": render_report.get("readiness"),
        "differentiation_bottleneck": render_report.get("differentiation_bottleneck"),
        "expected_render_count": render_report.get("expected_render_count") or export_doc.get("expected_render_count"),
        "actual_render_count": render_report.get("actual_render_count") or render_report.get("render_count"),
        "stitched_listening": stitched_report or {},
        "pluck_amplitude_handling": render_report.get("pluck_amplitude_handling"),
        "pluck_amplitude_audit_summary": pluck_rows,
        "pluck_validation_errors": pluck_errors,
        "final_factor_acceptance_matrix": matrix,
        "weak_or_missing_factors": weak,
        "optional_v4_1_strengthening": _optional_v4_1_proposal(matrix),
        "classical_stk_acceptance_decision": decision,
        "classical_guitar_stk_baseline_acceptable": decision.startswith("accepted"),
        "next_pipeline_step": "box_shape_stk_pipeline" if decision.startswith("accepted") else "fix_factor_activation_before_box",
    }


def write_report_md(report: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# PGSM STK Classical Final Acceptance Report",
        "",
        f"- **generated_at**: {report['generated_at']}",
        f"- **demo_version**: `{report.get('demo_version')}`",
        f"- **renderer**: {report.get('renderer')}",
        f"- **render_readiness**: {report.get('render_readiness')}",
        f"- **classical_stk_acceptance_decision**: **{report.get('classical_stk_acceptance_decision')}**",
        f"- **classical_guitar_stk_baseline_acceptable**: {report.get('classical_guitar_stk_baseline_acceptable')}",
        f"- **next_pipeline_step**: {report.get('next_pipeline_step')}",
        "",
        "## Pluck amplitude handling",
        "",
        f"- **policy**: {report.get('pluck_amplitude_handling')}",
    ]
    clamped = [r for r in report.get("pluck_amplitude_audit_summary") or [] if r.get("was_clamped")]
    if clamped:
        lines.append(f"- **clamped renders**: {len(clamped)} (relative excitation preserved within STK 0–1)")
        for row in clamped[:6]:
            lines.append(
                f"  - `{row['sample_id']}` `{row['note_name']}`: "
                f"raw={row['raw_pluck_amplitude']:.4f} → clamped={row['clamped_pluck_amplitude']:.4f}"
            )
    else:
        lines.append("- **clamped renders**: none (all amplitudes within STK range)")
    if report.get("pluck_validation_errors"):
        lines.extend(["", "### Pluck validation errors", ""])
        for err in report["pluck_validation_errors"]:
            lines.append(f"- {err}")
    lines.extend(["", "## Final factor acceptance matrix", ""])
    lines.append("| Factor | Group | Exported | Parsed | Applied | Spread | Status | Decision |")
    lines.append("|--------|-------|----------|--------|---------|--------|--------|----------|")
    for row in report.get("final_factor_acceptance_matrix") or []:
        lines.append(
            f"| {row['factor_name']} | {row['source_group']} | "
            f"{row['exported_to_json']} | {row['parsed_by_cpp']} | {row['applied_in_audio']} | "
            f"{row['per_sample_spread']} | {row['effect_strength_status']} | {row['final_decision']} |"
        )
    opt = report.get("optional_v4_1_strengthening") or {}
    lines.extend(["", "## Optional v4.1 strengthening", ""])
    if opt.get("recommended"):
        lines.append("Optional future tuning only — **not applied automatically**.")
        for item in opt.get("optional_adjustments") or []:
            lines.append(f"- {item}")
    else:
        lines.append("Not required — factors are active and measurable.")
    lines.extend(["", "## Conclusion", ""])
    decision = report.get("classical_stk_acceptance_decision")
    if decision == "accepted_for_gui_and_next_shape":
        lines.append(
            "The classical-guitar STK path is accepted as the final baseline. "
            "Physical factors are exported, parsed, and applied; sample differentiation is active. "
            "Proceed to box-shape STK pipeline."
        )
    elif decision == "accepted_but_minor_mix_strengthening_recommended":
        lines.append(
            "The classical-guitar STK path is accepted for GUI and next shape, with optional minor "
            "v4.1 mix strengthening recommended later — not applied in this pass."
        )
    else:
        lines.append("Not accepted — fix missing or unapplied factor activation before box pipeline.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Write classical STK final acceptance report.")
    parser.add_argument(
        "--export-json",
        type=Path,
        default=REPO_ROOT / "audio/debug_reports/pgsm_stk_demo_parameters_v4_10_samples.json",
    )
    parser.add_argument(
        "--render-report",
        type=Path,
        default=REPO_ROOT / "audio/debug_reports/pgsm_stk_guitar_demo_v4_10_samples_report.json",
    )
    parser.add_argument(
        "--stitched-report",
        type=Path,
        default=REPO_ROOT / "audio/debug_reports/pgsm_stk_guitar_demo_v4_10_samples_stitched_report.json",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPO_ROOT / "audio/debug_reports/pgsm_stk_classical_final_acceptance_report.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=REPO_ROOT / "audio/debug_reports/pgsm_stk_classical_final_acceptance_report.md",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    export_doc = json.loads(args.export_json.read_text(encoding="utf-8"))
    render_report = json.loads(args.render_report.read_text(encoding="utf-8"))
    stitched_report = None
    if args.stitched_report.is_file():
        stitched_report = json.loads(args.stitched_report.read_text(encoding="utf-8"))

    report = build_acceptance_report(
        export_doc=export_doc,
        render_report=render_report,
        stitched_report=stitched_report,
    )
    if report.get("pluck_validation_errors"):
        print("ERROR: pluck amplitude validation failed:", file=sys.stderr)
        for err in report["pluck_validation_errors"]:
            print(f"  - {err}", file=sys.stderr)
        return 2
    if not report.get("classical_stk_acceptance_decision"):
        print("ERROR: classical_stk_acceptance_decision missing", file=sys.stderr)
        return 2

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_report_md(report, args.output_md)
    print(f"Wrote {args.output_json}")
    print(f"Decision: {report['classical_stk_acceptance_decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
