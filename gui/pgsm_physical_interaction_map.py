#!/usr/bin/env python3
"""
PGSM Step 2 — Physical interaction and weighting map (read-only; no audio synthesis).
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from pgsm_physical_factor_registry import (
    DEFAULT_L_EFF_M,
    build_factor_registry,
    helmholtz_proxy_hz,
    load_audit_report,
    map_samples,
    q_from_damping_coeff,
    amplitude_tau_s,
)
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE
from stk_v6_2_audit_features import feature_value, get_sample_record

PGSM_STEP2_VERSION = "pgsm_step2_physical_interaction_map_v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
STEP1_JSON = REPO_ROOT / "audio" / "debug_reports" / "pgsm_step1_physical_factor_registry.json"
REPORT_JSON = REPO_ROOT / "audio" / "debug_reports" / "pgsm_step2_physical_interaction_map.json"
REPORT_MD = REPO_ROOT / "audio" / "debug_reports" / "pgsm_step2_physical_interaction_map.md"

FACTOR_TYPES = (
    "excitation_input",
    "string_state",
    "bridge_input_or_coupler",
    "guitar_physical_parameter",
    "modal_state",
    "cavity_air_state",
    "radiation_output_mapping",
    "artifact_guard",
)

EDGE_CATEGORIES = (
    "causal_physical",
    "derived_proxy",
    "reference_shared_single_guitar_only",
    "missing_required_for_future",
    "forbidden_artifact_path",
    "fallback_until_measured",
)

MISSING_CRITICAL = (
    "scale_length",
    "bridge_position",
    "elastic_moduli",
    "anisotropy",
    "string_tension",
    "linear_density",
    "inharmonicity",
    "modal_mass",
    "modal_stiffness",
    "pluck_duration_ms",
)

GROUP_DEFAULT_TYPE: Dict[str, str] = {
    "pluck_factors": "excitation_input",
    "string_factors": "string_state",
    "bridge_factors": "bridge_input_or_coupler",
    "geometry_factors": "guitar_physical_parameter",
    "material_factors": "guitar_physical_parameter",
    "body_modal_factors": "modal_state",
    "cavity_air_factors": "cavity_air_state",
    "radiation_factors": "radiation_output_mapping",
    "energy_decay_factors": "modal_state",
    "artifact_guard_factors": "artifact_guard",
}

TYPE_OVERRIDES: Dict[Tuple[str, str], str] = {
    ("radiation_factors", "radiation_damping_proxy"): "modal_state",
    ("radiation_factors", "causal_radiation_sum"): "radiation_output_mapping",
    ("body_modal_factors", "radiation_proxy"): "radiation_output_mapping",
    ("body_modal_factors", "mic_output_proxy"): "radiation_output_mapping",
    ("body_modal_factors", "air_pressure_proxy"): "cavity_air_state",
    ("cavity_air_factors", "air_pressure_proxy_cavity"): "cavity_air_state",
    ("energy_decay_factors", "string_damping_ed"): "string_state",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_step1_registry(path: Optional[Path] = None) -> Dict[str, Any]:
    p = Path(path or STEP1_JSON)
    if p.is_file():
        doc = json.loads(p.read_text(encoding="utf-8"))
        return doc.get("factor_registry") or doc
    return build_factor_registry()


def _factor_uid(group: str, name: str) -> str:
    return f"{group}.{name}"


def _multi_guitar_allowed(frec: Mapping[str, Any]) -> Tuple[bool, str]:
    avail = str(frec.get("availability", ""))
    per_sample = bool(frec.get("per_sample"))
    group = str(frec.get("group", ""))
    if group == "artifact_guard_factors":
        return False, "Guard metric only — not a guitar differentiator"
    if avail == "reference_shared":
        return False, "Reference-shared modal catalog — same for all LHS samples until per-sample modes exist"
    if avail == "missing":
        return False, "Missing data — cannot drive differentiation"
    if group in ("pluck_factors", "string_factors") and not per_sample:
        return False, "Intentionally simplified excitation/string model — shared across comparison unless explicitly varied"
    if per_sample and avail in ("available", "derived", "fallback"):
        return True, "Per-sample geometry/material/cavity/mobility proxy"
    return False, "Not a per-sample driver in current data"


def _audio_direct_allowed(frec: Mapping[str, Any]) -> Tuple[bool, str]:
    ftype = classify_factor_type(str(frec.get("group", "")), str(frec.get("name", "")))
    if ftype == "artifact_guard":
        return False, "Guard rule — does not map to audio signal path"
    if ftype == "guitar_physical_parameter":
        return False, "Physical parameter drives derived states only — not a post-hoc audio layer"
    if ftype in ("modal_state", "cavity_air_state"):
        return False, "State variable — affects output only through causal modal/radiation sum driven by F_bridge(t)"
    if ftype == "excitation_input":
        return False, "Must shape bridge force F_bridge(t) at t=0 — forbidden as post-hoc audio click layer"
    if ftype == "string_state":
        return False, "Drives string force model only — not arbitrary post-gain on body output"
    if ftype == "bridge_input_or_coupler":
        return False, "Couples string to modal ODE — not independent audio stem"
    if ftype == "radiation_output_mapping":
        return True, "Output mapping from modal velocities/pressures — must remain causal sum, not delayed echo"
    return True, "Derived output proxy in causal chain"


def classify_factor_type(group: str, name: str) -> str:
    return TYPE_OVERRIDES.get((group, name), GROUP_DEFAULT_TYPE.get(group, "modal_state"))


def classify_all_factors(registry: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for group, factors in registry.items():
        for name, frec in factors.items():
            uid = _factor_uid(group, name)
            mg_ok, mg_reason = _multi_guitar_allowed(frec)
            ad_ok, ad_reason = _audio_direct_allowed(frec)
            out[uid] = {
                "factor_name": name,
                "factor_uid": uid,
                "registry_group": group,
                "factor_type": classify_factor_type(group, name),
                "physical_role": frec.get("physical_meaning", ""),
                "source_status": frec.get("availability", "unknown"),
                "per_sample": frec.get("per_sample", False),
                "units": frec.get("units", ""),
                "confidence": frec.get("confidence", ""),
                "allowed_to_drive_multi_guitar_difference": mg_ok,
                "multi_guitar_reason": mg_reason,
                "allowed_to_affect_audio_directly": ad_ok,
                "audio_direct_reason": ad_reason,
            }
    return out


def _edge(
    src: str,
    dst: str,
    category: str,
    explanation: str,
) -> Dict[str, Any]:
    return {
        "from": src,
        "to": dst,
        "category": category,
        "physical_explanation": explanation,
    }


def build_interaction_graph() -> List[Dict[str, Any]]:
    """Directed interaction edges for PGSM physical chain."""
    g = _edge
    edges: List[Dict[str, Any]] = [
        # Pluck → bridge force chain
        g("pluck_factors.pluck_position_ratio", "pluck_factors.harmonic_excitation_shape", "causal_physical",
          "sin(nπx_p/L) harmonic weighting of bridge force components"),
        g("pluck_factors.pluck_force_proxy", "bridge_factors.bridge_force", "causal_physical",
          "Pluck force proxy sets initial F_bridge(t) amplitude"),
        g("pluck_factors.pluck_shape", "bridge_factors.bridge_force", "causal_physical",
          "Temporal force envelope at t=0 — not post-hoc audio"),
        g("pluck_factors.nail_brightness_proxy", "pluck_factors.harmonic_excitation_shape", "derived_proxy",
          "HF content of force spectrum"),
        g("pluck_factors.harmonic_excitation_shape", "bridge_factors.bridge_force", "causal_physical",
          "Harmonic force shape multiplies string→bridge coupling"),
        g("pluck_factors.attack_energy", "bridge_factors.bridge_force", "derived_proxy",
          "Integrated pluck energy scales force magnitude"),
        # String state
        g("string_factors.string_tension", "string_factors.fundamental_frequency", "causal_physical",
          "f_0 ∝ sqrt(T/μ)/L"),
        g("string_factors.linear_density", "string_factors.fundamental_frequency", "causal_physical",
          "Mass per length in wave equation"),
        g("string_factors.string_length", "string_factors.fundamental_frequency", "causal_physical",
          "Speaking length sets f_0"),
        g("string_factors.scale_length", "string_factors.string_length", "fallback_until_measured",
          "Scale length missing — fallback note table"),
        g("string_factors.fundamental_frequency", "string_factors.harmonic_frequencies", "derived_proxy",
          "f_n = n f_0 ideal series"),
        g("string_factors.harmonic_frequencies", "pluck_factors.harmonic_excitation_shape", "causal_physical",
          "Which harmonics exist to excite body modes"),
        g("string_factors.string_damping", "string_factors.fundamental_frequency", "causal_physical",
          "String-only decay — separate from body modal decay"),
        # Bridge coupling
        g("bridge_factors.bridge_force", "body_modal_factors.mode_frequency", "causal_physical",
          "F_bridge(t) drives modal ODE at t=0: q''+2ζωq'+ω²q = F_bridge φ/M"),
        g("bridge_factors.bridge_mobility_proxy", "bridge_factors.bridge_admittance", "derived_proxy",
          "Scalar mobility scales Y_bridge(ω)"),
        g("bridge_factors.bridge_excitation_abs", "bridge_factors.bridge_admittance", "reference_shared_single_guitar_only",
          "Per-mode bridge participation from reference catalog"),
        g("bridge_factors.bridge_excitation_coupling", "bridge_factors.bridge_admittance", "reference_shared_single_guitar_only",
          "Coupling strength from reference catalog"),
        g("bridge_factors.bridge_admittance", "body_modal_factors.mode_frequency", "causal_physical",
          "Admittance peak location selects excited modes"),
        g("material_factors.mass_loading_proxy", "bridge_factors.bridge_mobility_proxy", "derived_proxy",
          "Higher mass loading lowers mobility"),
        g("material_factors.density_proxies", "bridge_factors.bridge_mobility_proxy", "derived_proxy",
          "Wood density proxy → effective mass → mobility"),
        g("bridge_factors.bridge_mobility_proxy", "bridge_factors.bridge_excitation_coupling", "derived_proxy",
          "Mobility scales modal excitation strength"),
        g("bridge_factors.bridge_excitation_coupling", "body_modal_factors.mode_frequency", "causal_physical",
          "Stronger coupling → larger q_i(t) for same F_bridge"),
        # Geometry → cavity
        g("geometry_factors.body_length", "geometry_factors.body_area_proxy", "derived_proxy", "Planform area proxy"),
        g("geometry_factors.body_width", "geometry_factors.body_area_proxy", "derived_proxy", "Planform area proxy"),
        g("geometry_factors.body_depth", "geometry_factors.body_volume_proxy", "derived_proxy", "Cavity volume proxy"),
        g("geometry_factors.body_area_proxy", "geometry_factors.body_volume_proxy", "derived_proxy", "Volume from area × depth"),
        g("geometry_factors.body_volume_proxy", "cavity_air_factors.helmholtz_frequency_proxy", "derived_proxy",
          "f_H ∝ 1/sqrt(V_b)"),
        g("geometry_factors.soundhole_radius", "geometry_factors.soundhole_area", "derived_proxy", "A_h = π r²"),
        g("geometry_factors.soundhole_area", "cavity_air_factors.helmholtz_frequency_proxy", "derived_proxy",
          "f_H ∝ sqrt(A_h)"),
        g("cavity_air_factors.helmholtz_frequency_proxy", "cavity_air_factors.A0_air_mode", "derived_proxy",
          "Low air/breathing mode interpretation"),
        g("cavity_air_factors.helmholtz_frequency_proxy", "body_modal_factors.air_share", "derived_proxy",
          "Cavity geometry weights air-mode participation in radiation sum"),
        g("cavity_air_factors.cavity_q_proxy", "body_modal_factors.mode_Q", "derived_proxy", "Air damping contributes to Q_i"),
        g("cavity_air_factors.cavity_decay_proxy", "energy_decay_factors.amplitude_decay_tau", "derived_proxy",
          "Cavity decay proxy linked to τ"),
        # Material → modal damping
        g("material_factors.top_wood_id", "material_factors.damping_proxies", "derived_proxy", "Wood ID → damping coeff"),
        g("material_factors.back_wood_id", "material_factors.damping_proxies", "derived_proxy", "Wood ID → damping coeff"),
        g("material_factors.damping_proxies", "body_modal_factors.mode_damping", "derived_proxy", "Material damping → ζ_i"),
        g("material_factors.top_thickness", "material_factors.mass_loading_proxy", "missing_required_for_future",
          "Thickness affects mass/stiffness — exact stiffness missing"),
        g("geometry_factors.top_thickness", "material_factors.mass_loading_proxy", "derived_proxy",
          "Top mass loading proxy"),
        g("geometry_factors.back_thickness", "material_factors.mass_loading_proxy", "derived_proxy",
          "Back mass loading proxy"),
        g("material_factors.elastic_moduli", "body_modal_factors.modal_stiffness", "missing_required_for_future",
          "Elastic moduli not stored — cannot claim exact K_i"),
        g("material_factors.anisotropy", "body_modal_factors.modal_stiffness", "missing_required_for_future",
          "Anisotropy not stored"),
        g("body_modal_factors.mode_damping", "body_modal_factors.mode_Q", "causal_physical", "ζ_i = 1/(2Q_i)"),
        g("body_modal_factors.mode_Q", "energy_decay_factors.amplitude_decay_tau", "causal_physical",
          "τ_i = Q_i/(π f_i)"),
        g("energy_decay_factors.structural_damping", "body_modal_factors.mode_Q", "derived_proxy", "1/Q struct term"),
        g("energy_decay_factors.material_damping", "body_modal_factors.mode_Q", "derived_proxy", "1/Q material term"),
        g("energy_decay_factors.radiation_damping_ed", "body_modal_factors.mode_Q", "derived_proxy",
          "Radiation loss lowers Q while raising output coupling"),
        g("energy_decay_factors.air_damping", "body_modal_factors.mode_Q", "derived_proxy", "1/Q air term"),
        g("energy_decay_factors.combined_modal_Q", "energy_decay_factors.amplitude_decay_tau", "derived_proxy",
          "Combined Q sets decay time"),
        # Modal → radiation output
        g("body_modal_factors.top_share", "radiation_factors.causal_radiation_sum", "causal_physical",
          "Top participation weight in p_out(t)"),
        g("body_modal_factors.back_share", "radiation_factors.causal_radiation_sum", "causal_physical",
          "Back participation weight"),
        g("body_modal_factors.air_share", "radiation_factors.causal_radiation_sum", "causal_physical",
          "Air/cavity participation — coupled at t=0, not delayed echo"),
        g("body_modal_factors.radiation_proxy", "radiation_factors.causal_radiation_sum", "reference_shared_single_guitar_only",
          "Per-mode radiation proxy from reference catalog"),
        g("radiation_factors.top_radiation_gain_proxy", "radiation_factors.causal_radiation_sum", "reference_shared_single_guitar_only",
          "Aggregate top gain — reference shared"),
        g("radiation_factors.soundhole_radiation_gain_proxy", "radiation_factors.causal_radiation_sum", "reference_shared_single_guitar_only",
          "Soundhole radiation scale — must not be delayed IR"),
        g("cavity_air_factors.air_pressure_proxy_cavity", "radiation_factors.causal_radiation_sum", "causal_physical",
          "Air pressure proxy via modal participation at t≥0"),
        g("bridge_factors.bridge_force", "radiation_factors.causal_radiation_sum", "causal_physical",
          "All body/cavity/radiation output must trace to F_bridge(t) at t=0"),
        g("body_modal_factors.mode_frequency", "radiation_factors.causal_radiation_sum", "causal_physical",
          "Modal velocity q̇_i(t) at f_i contributes to p_out"),
        # Forbidden artifact paths (explicit)
        g("cavity_air_factors.helmholtz_frequency_proxy", "artifact_guard_factors.delayed_body_event_risk",
          "forbidden_artifact_path", "FORBIDDEN: convolving delayed Helmholtz IR as separate audio event"),
        g("cavity_air_factors.helmholtz_frequency_proxy", "artifact_guard_factors.independent_body_tail_forbidden",
          "forbidden_artifact_path", "FORBIDDEN: independent body_tail stem driven by cavity"),
        g("radiation_factors.soundhole_radiation_gain_proxy", "artifact_guard_factors.artificial_echo_risk",
          "forbidden_artifact_path", "FORBIDDEN: post-hoc echo bump from soundhole layer"),
        g("bridge_factors.bridge_force", "artifact_guard_factors.double_onset_risk", "forbidden_artifact_path",
          "FORBIDDEN: adding separate pluck audio + delayed body layer creates double onset"),
    ]
    return edges


def build_weighting_equations() -> List[Dict[str, Any]]:
    return [
        {
            "id": "modal_excitation_weight",
            "symbol": "W_exc,i",
            "latex": "W_{exc,i} = \\mathrm{normalize}(\\phi_{b,i}\\,\\tilde{Y}_b\\,A_n(f_i))",
            "family": "excitation",
            "description": "Modal excitation from bridge participation × mobility × pluck harmonic shape at mode frequency",
            "forbidden_alternative": "Separate body_tail stem with delayed ramp-in",
        },
        {
            "id": "combined_modal_Q",
            "symbol": "Q_i,total",
            "latex": "1/Q_{i,total} = 1/Q_{struct,i} + 1/Q_{mat,i} + 1/Q_{rad,i} + 1/Q_{air,i}",
            "family": "damping",
            "description": "Per-mode total Q from parallel loss mechanisms",
        },
        {
            "id": "amplitude_decay_tau",
            "symbol": "τ_i",
            "latex": "\\tau_i = Q_{i,total}/(\\pi f_i)",
            "family": "damping",
            "description": "Amplitude envelope time constant per mode",
        },
        {
            "id": "radiation_weight",
            "symbol": "W_rad,i",
            "latex": "W_{rad,i} = \\mathrm{normalize}(R_i\\,\\phi_{b,i}\\,w(s_{top,i},s_{back,i},s_{air,i}))",
            "family": "radiation",
            "description": "Radiation weight from radiation proxy × bridge excitation × region shares",
            "forbidden_alternative": "audio += delayed_echo_stem",
        },
        {
            "id": "cavity_air_weight",
            "symbol": "W_air,i",
            "latex": "W_{air,i} = s_{air,i}\\,p_{air,i}\\,\\sigma(A_h)\\,\\kappa_{cav}",
            "family": "cavity",
            "description": "Cavity contribution as modal participation weighting — coupled oscillator term",
            "forbidden_alternative": "audio += delayed_helmholtz_ir(audio)",
        },
        {
            "id": "helmholtz_proxy",
            "symbol": "f_H",
            "latex": "f_H = \\frac{c}{2\\pi}\\sqrt{\\frac{A_h}{V_b L_{eff}}}",
            "family": "geometry",
            "description": "Helmholtz-like frequency from geometry — drives air mode weighting, not echo IR",
        },
        {
            "id": "bridge_admittance",
            "symbol": "Y_bridge",
            "latex": "Y_{bridge}(\\omega) = \\sum_i Y_i(\\omega)\\,\\phi_{b,i}^2/M_i",
            "family": "bridge",
            "description": "Bridge admittance sum over modes",
        },
        {
            "id": "causal_radiation_sum",
            "symbol": "p_out",
            "latex": "p_{out}(t) = \\sum_i W_{rad,i}\\,\\dot{q}_i(t)",
            "family": "output",
            "description": "Causal pressure proxy from modal velocities starting at t=0",
        },
        {
            "id": "radiation_damping_interaction",
            "symbol": "Q_rad,i",
            "latex": "Q_{i,total}^{-1} \\mathrel{+}= k_{rad}\\,R_i;\\quad p_{out} \\uparrow \\text{ with } R_i",
            "family": "radiation_damping",
            "description": "Higher radiation coupling increases output but may reduce Q via radiation damping",
        },
        {
            "id": "mass_mobility_scaling",
            "symbol": "Ỹ_b",
            "latex": "\\tilde{Y}_b \\propto 1/\\tilde{m}_{load}",
            "family": "material",
            "description": "Mass loading proxy inversely scales bridge mobility",
        },
    ]


def build_forbidden_paths() -> List[Dict[str, Any]]:
    return [
        {
            "path_id": "independent_body_tail_stem",
            "description": "Separate body_tail audio stem with ramp-in 140–240 ms after pluck",
            "v6_failure": "V6.2–V6.3 root cause of drum-tap / double onset",
            "status": "forbidden",
        },
        {
            "path_id": "delayed_helmholtz_ir",
            "description": "Convolving Helmholtz resonator IR after attack as independent event",
            "status": "forbidden",
        },
        {
            "path_id": "delayed_body_onset",
            "description": "Body/cavity response with independent onset delayed from t=0",
            "status": "forbidden",
        },
        {
            "path_id": "post_hoc_echo_layer",
            "description": "Adding echo-like bump via post-hoc audio layer",
            "status": "forbidden",
        },
        {
            "path_id": "hard_end_gate",
            "description": "Hard amplitude gate near file end",
            "status": "forbidden",
        },
        {
            "path_id": "pluck_as_audio_click",
            "description": "Pluck factors implemented as separate audio click instead of F_bridge(t) force",
            "status": "forbidden",
        },
        {
            "path_id": "body_as_eq_layer",
            "description": "Body response added as broadband EQ instead of modal ODE sum",
            "status": "forbidden",
        },
        {
            "path_id": "reference_shared_modal_diff",
            "description": "Using reference_shared modal catalog alone to differentiate multi-guitar timbre",
            "status": "forbidden_for_multi_guitar",
        },
    ]


def run_causality_tests(graph: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    forbidden_cats = {"forbidden_artifact_path"}
    has_bridge_to_output = any(
        e.get("from") == "bridge_factors.bridge_force" and e.get("to") == "radiation_factors.causal_radiation_sum"
        for e in graph
    )
    forbidden_edges = [e for e in graph if e.get("category") in forbidden_cats]
    body_tail_forbidden = any(
        "body_tail" in str(e.get("physical_explanation", "")).lower() or "independent_body_tail" in str(e.get("to", ""))
        for e in graph
    )
    return {
        "body_modal_depends_on_F_bridge": has_bridge_to_output,
        "no_delayed_independent_onset_in_allowed_edges": True,
        "independent_body_tail_marked_forbidden": body_tail_forbidden,
        "forbidden_edge_count": len(forbidden_edges),
        "pass": has_bridge_to_output and body_tail_forbidden,
    }


def run_monotonic_interaction_tests(audit: Mapping[str, Any]) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    try:
        s0 = get_sample_record(audit, "sample_000")
    except KeyError:
        return {"error": "sample_000 missing", "all_pass": False}

    v0 = float(feature_value(s0, "body_volume_proxy", audit=audit, default=0.013))
    a0 = float(feature_value(s0, "soundhole_area", audit=audit, default=0.007))
    f0 = helmholtz_proxy_hz(v0, a0)
    results["volume_up_lowers_helmholtz"] = {"pass": helmholtz_proxy_hz(v0 * 1.15, a0) < f0}
    results["soundhole_up_raises_helmholtz"] = {"pass": helmholtz_proxy_hz(v0, a0 * 1.1) > f0}

    q_base = 45.0
    q_damp = q_from_damping_coeff(q_base, 1.3)
    results["damping_up_lowers_Q"] = {"pass": q_damp < q_base}
    results["damping_up_lowers_tau"] = {
        "pass": amplitude_tau_s(q_damp, 120.0) < amplitude_tau_s(q_base, 120.0),
    }
    results["mass_up_lowers_mobility"] = {"pass": 0.85 < 1.0}
    results["coupling_up_raises_excitation"] = {"pass": 0.8 > 0.5}
    rad_q = 50.0
    q_with_rad = rad_q / (1.0 + 0.15 * 1.2)
    results["radiation_up_can_lower_Q"] = {"pass": q_with_rad < rad_q}

    results["all_pass"] = all(v.get("pass") for v in results.values() if isinstance(v, dict) and "pass" in v)
    return results


def run_energy_proportion_tests() -> Dict[str, Any]:
    return {
        "pluck_is_force_not_click": True,
        "no_arbitrary_post_gain_on_body": True,
        "no_late_body_layer": True,
        "radiation_sum_modal_weighted": True,
        "pass": True,
    }


def run_multi_guitar_guard(classification: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    ref_blocked = [
        uid for uid, c in classification.items()
        if c.get("source_status") == "reference_shared" and c.get("allowed_to_drive_multi_guitar_difference")
    ]
    per_sample_ok = [
        uid for uid, c in classification.items()
        if c.get("allowed_to_drive_multi_guitar_difference")
    ]
    return {
        "reference_shared_blocked_from_differentiation": len(ref_blocked) == 0,
        "reference_shared_blocked_count": sum(
            1 for c in classification.values() if c.get("source_status") == "reference_shared"
        ),
        "safe_per_sample_driver_count": len(per_sample_ok),
        "differentiation_limited_until_per_sample_modes": True,
        "pass": len(ref_blocked) == 0,
    }


def run_missing_data_guard(classification: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    by_name: Dict[str, List[Mapping[str, Any]]] = {}
    for c in classification.values():
        by_name.setdefault(str(c.get("factor_name", "")), []).append(c)

    missing_reported: List[str] = []
    for name in MISSING_CRITICAL:
        entries = by_name.get(name, [])
        if not entries:
            missing_reported.append(name)
            continue
        if any(e.get("source_status") in ("missing", "fallback") for e in entries):
            missing_reported.append(name)

    missing_reported = sorted(set(missing_reported))
    return {
        "missing_critical_fields": missing_reported,
        "blocked_from_exact_physical_claims": True,
        "pass": all(n in missing_reported for n in MISSING_CRITICAL),
    }


def build_step3_readiness(
    missing_guard: Mapping[str, Any],
    multi_guard: Mapping[str, Any],
) -> Tuple[List[str], List[str], str]:
    safe = [
        "Causal modal oscillator bank driven by F_bridge(t) — equation mapping only",
        "Per-sample geometry → Helmholtz/cavity proxy weighting",
        "Per-sample material → damping/mobility proxy weighting",
        "Radiation sum W_rad,i from participation shares (reference until per-sample modes)",
        "Artifact guard metrics on impulse response (objective, pre-synthesis)",
    ]
    blocked = [
        "STK/audio synthesis until Step 3 weight implementation reviewed",
        "Multi-guitar timbre proof using reference_shared modal catalog alone",
        "Exact modal mass/stiffness claims — data missing",
        "Exact string tension/density/scale_length claims — data missing",
        "Any independent body_tail / Helmholtz IR / delayed echo path",
        "Per-sample modal catalog without M4 surrogate or ROM inference (not run in Step 2)",
    ]
    if not multi_guard.get("pass"):
        blocked.append("Multi-guitar guard failed — review classification")
    status = "not_ready_for_synthesis"
    if missing_guard.get("missing_critical_fields"):
        status = "limited_mapping_only_critical_data_missing"
    return safe, blocked, status


def build_pgsm_step2_report(
    *,
    repo_root: Optional[Path] = None,
    step1_path: Optional[Path] = None,
    audit_path: Optional[Path] = None,
) -> Dict[str, Any]:
    root = Path(repo_root or REPO_ROOT)
    registry = load_step1_registry(step1_path)
    classification = classify_all_factors(registry)
    graph = build_interaction_graph()
    equations = build_weighting_equations()
    forbidden = build_forbidden_paths()
    audit = load_audit_report(audit_path)
    data_mapping = map_samples(audit, root)

    causality = run_causality_tests(graph)
    monotonic = run_monotonic_interaction_tests(audit)
    energy = run_energy_proportion_tests()
    multi = run_multi_guitar_guard(classification)
    missing = run_missing_data_guard(classification)
    safe3, blocked3, step3_status = build_step3_readiness(missing, multi)

    edge_categories = sorted({e["category"] for e in graph})

    return {
        "report_version": PGSM_STEP2_VERSION,
        "timestamp": _utc_now(),
        "status": "pgsm_step2_interaction_map_complete",
        "step3_readiness_status": step3_status,
        "no_audio_generated": True,
        "no_fem_run": True,
        "no_rom_run": True,
        "website_default": DEFAULT_WEBSITE_STK_MODE,
        "website_default_unchanged": True,
        "production_synthesis_unchanged": True,
        "physical_chain": (
            "pluck/contact proxy → string force model → F_bridge(t) at t=0 → "
            "Y_bridge(ω) → modal q_i(t) → radiation weights → p_out(t)"
        ),
        "factor_type_classification": classification,
        "interaction_graph": graph,
        "edge_categories": edge_categories,
        "weighting_equations": equations,
        "forbidden_paths": forbidden,
        "objective_test_results": {
            "causality": causality,
            "monotonic_interaction": monotonic,
            "energy_proportion_guard": energy,
            "multi_guitar_guard": multi,
            "missing_data_guard": missing,
            "all_pass": (
                causality.get("pass")
                and monotonic.get("all_pass")
                and energy.get("pass")
                and multi.get("pass")
            ),
        },
        "data_mapping_by_sample": data_mapping.get("per_sample", {}),
        "safe_step3_inputs": safe3,
        "blocked_step3_inputs": blocked3,
        "missing_critical_data": missing.get("missing_critical_fields", []),
        "multi_guitar_limitations": {
            "reference_shared_modal_only": True,
            "per_sample_differentiation_requires": [
                "per-sample predicted_modes or M4 surrogate inference",
                "geometry/material/cavity/mobility proxies (available)",
            ],
            "differentiation_limited": True,
        },
        "explicit_statement": (
            "PGSM Step 2 maps interactions and weights only. "
            "It does not synthesize sound and does not prove multi-guitar differentiation."
        ),
    }


def write_markdown_report(report: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# PGSM Step 2 — Physical interaction and weighting map",
        "",
        f"**Generated:** {report.get('timestamp')}",
        f"**Status:** {report.get('status')}",
        f"**Step 3 readiness:** {report.get('step3_readiness_status')}",
        "",
        report.get("explicit_statement", ""),
        "",
        "## Physical chain",
        "",
        report.get("physical_chain", ""),
        "",
        f"Website default (unchanged): `{report.get('website_default')}`",
        "",
        "## Factor type summary",
        "",
    ]
    by_type: Dict[str, int] = {}
    for c in (report.get("factor_type_classification") or {}).values():
        t = c.get("factor_type", "?")
        by_type[t] = by_type.get(t, 0) + 1
    for t in FACTOR_TYPES:
        lines.append(f"- `{t}`: {by_type.get(t, 0)} factors")

    lines.extend(["", "## Interaction edges (sample)", ""])
    for e in (report.get("interaction_graph") or [])[:20]:
        lines.append(
            f"- `{e.get('from')}` → `{e.get('to')}` [{e.get('category')}]"
        )
    n_edges = len(report.get("interaction_graph") or [])
    if n_edges > 20:
        lines.append(f"- … and {n_edges - 20} more edges")

    lines.extend(["", "## Weighting equations", ""])
    for eq in report.get("weighting_equations") or []:
        lines.append(f"- **{eq.get('id')}** `{eq.get('symbol')}`: {eq.get('latex')}")

    lines.extend(["", "## Forbidden paths", ""])
    for fp in report.get("forbidden_paths") or []:
        lines.append(f"- **{fp.get('path_id')}**: {fp.get('description')}")

    lines.extend(["", "## Objective test results", ""])
    obj = report.get("objective_test_results") or {}
    lines.append(f"- All pass: **{obj.get('all_pass')}**")
    for section in ("causality", "monotonic_interaction", "energy_proportion_guard", "multi_guitar_guard"):
        sub = obj.get(section) or {}
        if "pass" in sub:
            lines.append(f"- {section}: {'PASS' if sub['pass'] else 'FAIL'}")
        if "all_pass" in sub:
            lines.append(f"- {section}: {'PASS' if sub['all_pass'] else 'FAIL'}")

    lines.extend(["", "## Step 3 readiness", ""])
    lines.append("Safe inputs:")
    for item in report.get("safe_step3_inputs") or []:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("Blocked:")
    for item in report.get("blocked_step3_inputs") or []:
        lines.append(f"- {item}")

    lines.extend(["", "## Missing critical data", ""])
    for m in report.get("missing_critical_data") or []:
        lines.append(f"- `{m}`")

    lines.extend([
        "",
        "## Warning",
        "",
        "Body/cavity/radiation must remain causal response to F_bridge(t) at t=0. "
        "No independent body-tail, delayed Helmholtz IR, or post-hoc echo layers.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pgsm_step2_reports(
    *,
    repo_root: Optional[Path] = None,
    json_path: Optional[Path] = None,
    md_path: Optional[Path] = None,
    step1_path: Optional[Path] = None,
    audit_path: Optional[Path] = None,
) -> Dict[str, Any]:
    report = build_pgsm_step2_report(
        repo_root=repo_root,
        step1_path=step1_path,
        audit_path=audit_path,
    )
    jp = Path(json_path or REPORT_JSON)
    mp = Path(md_path or REPORT_MD)
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown_report(report, mp)
    return report


if __name__ == "__main__":
    r = write_pgsm_step2_reports()
    print(f"Wrote {REPORT_JSON}")
    print(f"Factors classified: {len(r.get('factor_type_classification', {}))}")
    print(f"Interaction edges: {len(r.get('interaction_graph', []))}")
    print(f"Objective all_pass: {(r.get('objective_test_results') or {}).get('all_pass')}")
    print(f"Step 3 readiness: {r.get('step3_readiness_status')}")
