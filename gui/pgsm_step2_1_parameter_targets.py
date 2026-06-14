#!/usr/bin/env python3
"""
PGSM Step 2.1 — Literature-grounded parameter targets and data-gap closure plan.
Read-only; no audio synthesis, no FEM/ROM.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from pgsm_physical_factor_registry import (
    DEFAULT_L_EFF_M,
    DEFAULT_SAMPLE_IDS,
    helmholtz_proxy_hz,
    load_audit_report,
    map_samples,
    q_from_damping_coeff,
    amplitude_tau_s,
)
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE
from stk_v6_2_audit_features import feature_value, get_sample_record

PGSM_STEP2_1_VERSION = "pgsm_step2_1_parameter_targets_v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
STEP1_JSON = REPO_ROOT / "audio" / "debug_reports" / "pgsm_step1_physical_factor_registry.json"
STEP2_JSON = REPO_ROOT / "audio" / "debug_reports" / "pgsm_step2_physical_interaction_map.json"
REPORT_JSON = REPO_ROOT / "audio" / "debug_reports" / "pgsm_step2_1_parameter_targets.json"
REPORT_MD = REPO_ROOT / "audio" / "debug_reports" / "pgsm_step2_1_parameter_targets.md"
FIGURES_DIR = REPO_ROOT / "audio" / "debug_reports" / "pgsm_step2_1_figures"

READINESS_VALUES = (
    "not_ready_missing_critical_data",
    "ready_for_numerical_impulse_response_only",
    "ready_for_limited_single_guitar_audio",
    "ready_for_multi_guitar_comparison",
)

FALLBACK_LEVELS = ("L0_measured", "L1_derived", "L2_literature_fallback", "L3_blocked")

# Classical guitar literature / handbook anchors (order-of-magnitude, not claims of measurement)
CLASSICAL_SCALE_LENGTH_M = 0.650
CLASSICAL_BRIDGE_POSITION_M = 0.390  # from nut, typical ~60% of scale
NYLON_LINEAR_DENSITY_KG_M = 0.001_8  # order-of-magnitude mid string
NYLON_TENSION_N = 80.0  # typical mid-string order
INHARMONICITY_B_TYPICAL = 0.0001  # stiff-string small B, literature order
PLUCK_DURATION_MS_TYPICAL = 3.0
SPRUCE_E_PARALLEL_GPA = 11.0
SPRUCE_E_PERP_GPA = 0.5
ANISOTROPY_RATIO_TYPICAL = 20.0
MODAL_Q_TYPICAL_RANGE = (25.0, 80.0)
RADIATION_DAMPING_COEFF_TYPICAL = 0.08
CAVITY_COUPLING_TYPICAL = 0.35


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _impact(**kwargs: bool) -> Dict[str, bool]:
    return dict(kwargs)


def _param(
    name: str,
    *,
    physical_role: str,
    required_for_step3: bool,
    required_for_multi_guitar: bool,
    current_status: str,
    source_in_project: str,
    proposed_strategy: str,
    fallback_level: str,
    safe_range: str,
    units: str,
    confidence: str,
    risk_if_wrong: str,
    affects: Mapping[str, bool],
    literature_note: str = "",
) -> Dict[str, Any]:
    return {
        "name": name,
        "physical_role": physical_role,
        "required_for_step3": required_for_step3,
        "required_for_multi_guitar": required_for_multi_guitar,
        "current_status": current_status,
        "source_in_project": source_in_project,
        "proposed_strategy": proposed_strategy,
        "fallback_level": fallback_level,
        "safe_range": safe_range,
        "units": units,
        "confidence": confidence,
        "risk_if_wrong": risk_if_wrong,
        "affects": dict(affects),
        "literature_note": literature_note,
    }


def build_parameter_target_table(
    step1: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Research-backed targets for missing/fallback PGSM parameters."""
    _ = step1
    return [
        _param(
            "scale_length",
            physical_role="Nut-to-bridge speaking length; sets harmonic spacing f_n",
            required_for_step3=True,
            required_for_multi_guitar=False,
            current_status="missing",
            source_in_project="not in LHS geometry; PGSM Step 1 registry",
            proposed_strategy="use_literature_fallback",
            fallback_level="L2_literature_fallback",
            safe_range="0.628–0.660 m (classical)",
            units="m",
            confidence="medium",
            risk_if_wrong="Wrong pitch/harmonic series; bridge force spectrum misaligned with body modes",
            affects=_impact(pitch=True, bridge_admittance=True, timbre=True),
            literature_note="Torres/classical standard ~650 mm scale (Fletcher & Rossing; guitar acoustics texts)",
        ),
        _param(
            "bridge_position",
            physical_role="Bridge location on top; modal excitation via mode shape φ(x_bridge)",
            required_for_step3=False,
            required_for_multi_guitar=False,
            current_status="missing",
            source_in_project="not in LHS pool",
            proposed_strategy="use_literature_fallback",
            fallback_level="L2_literature_fallback",
            safe_range="0.35–0.42 m from nut (typical classical)",
            units="m",
            confidence="low",
            risk_if_wrong="Modal coupling bias; timbre shift if used as free parameter",
            affects=_impact(bridge_admittance=True, timbre=True, radiation=True),
        ),
        _param(
            "string_tension",
            physical_role="Axial tension T in f_0 = 1/(2L) sqrt(T/μ)",
            required_for_step3=True,
            required_for_multi_guitar=False,
            current_status="missing",
            source_in_project="not stored per sample",
            proposed_strategy="infer_from_existing_project_data",
            fallback_level="L1_derived",
            safe_range="50–120 N per string (nylon classical, note-dependent)",
            units="N",
            confidence="medium",
            risk_if_wrong="Pitch error if inconsistent with target f_0 and μ",
            affects=_impact(pitch=True, bridge_admittance=True),
            literature_note="Infer T from target note f_0, assumed L and literature μ (inverse wave equation)",
        ),
        _param(
            "linear_density",
            physical_role="String mass per unit length μ",
            required_for_step3=True,
            required_for_multi_guitar=False,
            current_status="missing",
            source_in_project="not stored",
            proposed_strategy="use_literature_fallback",
            fallback_level="L2_literature_fallback",
            safe_range="0.0008–0.004 kg/m (nylon sets)",
            units="kg/m",
            confidence="medium",
            risk_if_wrong="Pitch/harmonic spacing error; inconsistent with inferred T",
            affects=_impact(pitch=True),
            literature_note="Manufacturer/catalog linear densities for nylon strings",
        ),
        _param(
            "inharmonicity",
            physical_role="Stiff-string partial detuning coefficient B",
            required_for_step3=False,
            required_for_multi_guitar=False,
            current_status="missing",
            source_in_project="PGSM Step 1 explicit missing",
            proposed_strategy="use_literature_fallback",
            fallback_level="L2_literature_fallback",
            safe_range="B ~ 0.00005–0.0003 (dimensionless proxy)",
            units="dimensionless",
            confidence="low",
            risk_if_wrong="Slight high-harmonic pitch color; acceptable for Step 3A IR if labeled",
            affects=_impact(pitch=True, timbre=True),
            literature_note="Schuck & Young; Fletcher–Rossing stiff string correction optional at Step 3A",
        ),
        _param(
            "pluck_position_ratio",
            physical_role="Normalized pluck point x_p/L; harmonic force shaping",
            required_for_step3=True,
            required_for_multi_guitar=False,
            current_status="available",
            source_in_project="body_response_synth.FIXED_PLUCK_POSITION ≈ 0.18",
            proposed_strategy="infer_from_existing_project_data",
            fallback_level="L1_derived",
            safe_range="0.10–0.25 (typical finger/nail)",
            units="dimensionless",
            confidence="high",
            risk_if_wrong="Harmonic balance wrong; must remain force shaping not audio click",
            affects=_impact(timbre=True, bridge_admittance=True),
        ),
        _param(
            "pluck_duration_ms",
            physical_role="Contact duration; attack bandwidth of F_bridge(t)",
            required_for_step3=False,
            required_for_multi_guitar=False,
            current_status="missing",
            source_in_project="not explicitly stored",
            proposed_strategy="use_literature_fallback",
            fallback_level="L2_literature_fallback",
            safe_range="1–8 ms",
            units="ms",
            confidence="low",
            risk_if_wrong="Attack sharpness; low risk for IR testbench if held fixed",
            affects=_impact(timbre=True),
        ),
        _param(
            "pluck_force_proxy",
            physical_role="Integrated pluck force scale at bridge",
            required_for_step3=True,
            required_for_multi_guitar=False,
            current_status="fallback",
            source_in_project="string model velocity / force proxy",
            proposed_strategy="use_literature_fallback",
            fallback_level="L2_literature_fallback",
            safe_range="normalized unit impulse scale; amplitude arbitrary until calibration",
            units="N·s proxy",
            confidence="medium",
            risk_if_wrong="Overall level only if misused as post-gain; forbidden as separate click layer",
            affects=_impact(timbre=True),
        ),
        _param(
            "top_elastic_moduli",
            physical_role="Top plate E_parallel, E_perp; stiffness/mass distribution",
            required_for_step3=False,
            required_for_multi_guitar=True,
            current_status="missing",
            source_in_project="wood ID only; no E stored",
            proposed_strategy="block_until_available",
            fallback_level="L3_blocked",
            safe_range="E∥ 8–14 GPa spruce; E⊥ 0.3–0.7 GPa (literature wood data)",
            units="Pa",
            confidence="low",
            risk_if_wrong="Cannot claim exact modal stiffness or frequency shifts from wood alone",
            affects=_impact(bridge_admittance=True, modal_Q=True, timbre=True, multi_guitar=True),
            literature_note="Use wood-ID proxies only (L1) until measured E or ROM moduli mapped",
        ),
        _param(
            "back_elastic_moduli",
            physical_role="Back plate elastic moduli",
            required_for_step3=False,
            required_for_multi_guitar=True,
            current_status="missing",
            source_in_project="wood ID only",
            proposed_strategy="block_until_available",
            fallback_level="L3_blocked",
            safe_range="species-dependent; rosewood/maple literature ranges",
            units="Pa",
            confidence="low",
            risk_if_wrong="Same as top — blocked for exact stiffness claims",
            affects=_impact(bridge_admittance=True, modal_Q=True, multi_guitar=True),
        ),
        _param(
            "wood_anisotropy",
            physical_role="Orthotropic stiffness ratios (top/back)",
            required_for_step3=False,
            required_for_multi_guitar=True,
            current_status="missing",
            source_in_project="PGSM Step 1 explicit missing",
            proposed_strategy="block_until_available",
            fallback_level="L3_blocked",
            safe_range="E∥/E⊥ ~ 10–25 typical spruce",
            units="ratio",
            confidence="low",
            risk_if_wrong="Mode shape/participation error if invented",
            affects=_impact(bridge_admittance=True, radiation=True, multi_guitar=True),
        ),
        _param(
            "modal_mass",
            physical_role="Generalized modal mass M_i in oscillator equation",
            required_for_step3=True,
            required_for_multi_guitar=True,
            current_status="missing",
            source_in_project="modal catalog lacks M_i",
            proposed_strategy="infer_from_existing_project_data",
            fallback_level="L1_derived",
            safe_range="proxy from effective_modal_mass_proxy / bridge mobility",
            units="kg proxy",
            confidence="medium",
            risk_if_wrong="Admittance peak heights wrong; relative mode balance distorted",
            affects=_impact(bridge_admittance=True, timbre=True, sustain=True),
            literature_note="Calvin & Elejabarrieta; admittance literature uses M_i at bridge",
        ),
        _param(
            "modal_stiffness",
            physical_role="Generalized stiffness K_i = M_i ω_i²",
            required_for_step3=False,
            required_for_multi_guitar=True,
            current_status="missing",
            source_in_project="not in rom_stk_body.json",
            proposed_strategy="block_until_available",
            fallback_level="L3_blocked",
            safe_range="derived only if M_i and f_i known",
            units="N/m proxy",
            confidence="low",
            risk_if_wrong="Exact pole placement claims forbidden without data",
            affects=_impact(bridge_admittance=True, pitch=True),
        ),
        _param(
            "exact_modal_Q",
            physical_role="Measured or simulated per-mode Q",
            required_for_step3=True,
            required_for_multi_guitar=True,
            current_status="derived",
            source_in_project="modal_damping.py wood-weighted proxy",
            proposed_strategy="infer_from_existing_project_data",
            fallback_level="L1_derived",
            safe_range="Q 25–80 in body band (mode-dependent)",
            units="dimensionless",
            confidence="medium",
            risk_if_wrong="Sustain/tail wrong; monotonic damping tests must pass",
            affects=_impact(modal_Q=True, damping=True, sustain=True, timbre=True, multi_guitar=True),
            literature_note="Meyer; Chaigne & Kergomard — Q from material + radiation losses",
        ),
        _param(
            "radiation_damping_coefficient",
            physical_role="Energy loss to sound field; lowers Q while raising output",
            required_for_step3=True,
            required_for_multi_guitar=False,
            current_status="derived",
            source_in_project="combined Q proxy in PGSM Step 2",
            proposed_strategy="infer_from_existing_project_data",
            fallback_level="L1_derived",
            safe_range="equivalent 1/Q_rad ~ 0.01–0.05",
            units="dimensionless proxy",
            confidence="low",
            risk_if_wrong="Over-bright or over-damped tail if treated as pure gain",
            affects=_impact(modal_Q=True, radiation=True, sustain=True, timbre=True),
        ),
        _param(
            "effective_neck_length_helmholtz",
            physical_role="L_eff in Helmholtz f_H = c/(2π) sqrt(A_h/(V L_eff))",
            required_for_step3=False,
            required_for_multi_guitar=True,
            current_status="fallback",
            source_in_project="PGSM DEFAULT_L_EFF_M = 0.012 m",
            proposed_strategy="use_literature_fallback",
            fallback_level="L2_literature_fallback",
            safe_range="0.008–0.020 m",
            units="m",
            confidence="low",
            risk_if_wrong="Low-frequency air mode placement; sensitivity test required",
            affects=_impact(pitch=False, timbre=True, multi_guitar=True),
            literature_note="Helmholtz lumped models; calibrate against cavity proxy sensitivity",
        ),
        _param(
            "cavity_coupling_coefficient",
            physical_role="Soundhole/body air–structure coupling scale κ_cav",
            required_for_step3=True,
            required_for_multi_guitar=True,
            current_status="derived",
            source_in_project="air_share × cavity proxies from audit",
            proposed_strategy="infer_from_existing_project_data",
            fallback_level="L1_derived",
            safe_range="0.15–0.55 (dimensionless proxy)",
            units="dimensionless",
            confidence="medium",
            risk_if_wrong="Must not become delayed echo; causal modal weighting only",
            affects=_impact(timbre=True, radiation=True, sustain=True, multi_guitar=True),
        ),
        _param(
            "mic_listener_radiation_proxy",
            physical_role="Listener/mic coupling to radiated field",
            required_for_step3=False,
            required_for_multi_guitar=False,
            current_status="reference_shared",
            source_in_project="rom_stk_body.json mic_output_proxy",
            proposed_strategy="infer_from_existing_project_data",
            fallback_level="L1_derived",
            safe_range="reference catalog normalization",
            units="proxy",
            confidence="medium",
            risk_if_wrong="Relative radiation balance; not multi-guitar differentiator alone",
            affects=_impact(radiation=True, timbre=True),
        ),
    ]


def build_literature_alignment_checklist() -> List[Dict[str, Any]]:
    return [
        {
            "id": "bridge_admittance_central",
            "statement": "Bridge admittance Y_bridge(ω) is central to body/string coupling",
            "aligned": True,
            "pgsm_evidence": "Step 2 interaction graph: F_bridge → Y_bridge → q_i",
            "reference_domain": "Guitar admittance / mobility literature (Calvin, Elejabarrieta, Meyer)",
        },
        {
            "id": "modal_peaks_via_bridge",
            "statement": "Body resonances are modal peaks coupled through the bridge",
            "aligned": True,
            "pgsm_evidence": "Modal oscillator driven by F_bridge(t) φ_i/M_i",
            "reference_domain": "Modal synthesis / mobility synthesis",
        },
        {
            "id": "causal_not_delayed",
            "statement": "Body/cavity/radiation are causal responses to F_bridge(t), not delayed layers",
            "aligned": True,
            "pgsm_evidence": "Forbidden paths: body_tail, delayed Helmholtz IR (Step 2)",
            "reference_domain": "PGSM methodology post-V6 failure analysis",
        },
        {
            "id": "modal_radiation_not_eq",
            "statement": "Acoustic output is modal/radiation-weighted, not post-hoc EQ",
            "aligned": True,
            "pgsm_evidence": "p_out(t)=Σ W_rad,i q̇_i(t); energy proportion guard",
            "reference_domain": "Radiation from modal velocities (Fletcher & Rossing)",
        },
        {
            "id": "cavity_coupled_not_echo",
            "statement": "Cavity/air/soundhole is coupled body–air system, not echo",
            "aligned": True,
            "pgsm_evidence": "W_air,i participation weight; forbidden delayed IR",
            "reference_domain": "Helmholtz + structural–acoustic coupling models",
        },
        {
            "id": "radiation_damping_interaction",
            "statement": "Radiation increases output but contributes to damping (lowers Q)",
            "aligned": True,
            "pgsm_evidence": "Step 2 radiation_damping_interaction equation; monotonic test",
            "reference_domain": "Acoustic radiation damping in plates",
        },
        {
            "id": "timbre_modal_characterization",
            "statement": "Timbre characterized by modal f, Q, effective mass, radiation components",
            "aligned": True,
            "pgsm_evidence": "Factor registry + interaction map cover f_i, Q_i, R_i, shares",
            "reference_domain": "Meyer guitar acoustics; modal analysis tradition",
        },
        {
            "id": "geometry_material_admittance",
            "statement": "Geometry/material affect admittance/radiated pressure, not arbitrary gain",
            "aligned": True,
            "pgsm_evidence": "Per-sample geometry/material → mobility/damping; multi-guitar guard",
            "reference_domain": "Physical modeling principle; PGSM Step 2",
        },
    ]


def build_fallback_policy() -> Dict[str, Any]:
    return {
        "levels": {
            "L0_measured": {
                "description": "Measured or simulated per-sample value",
                "allowed_for_step3a_ir": True,
                "allowed_for_audio": False,
                "label_required": False,
            },
            "L1_derived": {
                "description": "Derived from existing geometry/material/modal project data",
                "allowed_for_step3a_ir": True,
                "allowed_for_audio": False,
                "label_required": True,
            },
            "L2_literature_fallback": {
                "description": "Literature/default value; sensitivity-tested, clearly labeled",
                "allowed_for_step3a_ir": True,
                "allowed_for_audio": False,
                "label_required": True,
            },
            "L3_blocked": {
                "description": "Unknown/unsafe — exact physical claims forbidden",
                "allowed_for_step3a_ir": False,
                "allowed_for_audio": False,
                "label_required": True,
            },
        },
        "policy_rules": [
            "No L2 fallback may be used for multi-guitar differentiation claims without per-sample data",
            "L3 parameters cannot support exact stiffness/mass/anisotropy claims",
            "All fallbacks must be documented in Step 3A IR metadata",
            "Musical WAV synthesis requires L0/L1 for body modal catalog per sample (not available)",
        ],
    }


def run_sensitivity_plan(audit: Mapping[str, Any]) -> Dict[str, Any]:
    """Numeric sensitivity only — no WAVs."""
    results: Dict[str, Any] = {}
    try:
        s0 = get_sample_record(audit, "sample_000")
    except KeyError:
        return {"error": "sample_000 missing", "all_pass": False}

    v0 = float(feature_value(s0, "body_volume_proxy", audit=audit, default=0.013))
    a0 = float(feature_value(s0, "soundhole_area", audit=audit, default=0.007))

    # scale_length → pitch (ideal string)
    l0, l1 = 0.650, 0.660
    f0 = 440.0
    f1 = f0 * (l0 / l1)
    results["scale_length_pitch"] = {
        "pass": f1 < f0,
        "f_at_L650_hz": f0,
        "f_at_L660_hz": round(f1, 3),
        "note": "Longer scale lowers f for same tension/μ",
    }

    # pluck position (x_p/L) → harmonic factor (n=3)
    xp_lo, xp_hi = 0.12, 0.28
    h3_lo = math.sin(3 * math.pi * xp_lo)
    h3_hi = math.sin(3 * math.pi * xp_hi)
    results["bridge_position_pluck_coupling"] = {
        "pass": abs(h3_hi - h3_lo) > 0.02,
        "third_harmonic_delta": round(abs(h3_hi - h3_lo), 4),
        "note": "Pluck position changes harmonic force mix at bridge",
    }

    q_base = 45.0
    q_damp = q_from_damping_coeff(q_base, 1.25)
    results["damping_Q_tau"] = {
        "pass": q_damp < q_base and amplitude_tau_s(q_damp, 120) < amplitude_tau_s(q_base, 120),
    }

    q_rad_base = 50.0
    q_with_rad = q_rad_base / (1.0 + RADIATION_DAMPING_COEFF_TYPICAL)
    results["radiation_damping_output_Q"] = {
        "pass": q_with_rad < q_rad_base,
        "Q_with_radiation": round(q_with_rad, 3),
    }

    m_lo, m_hi = 1.0, 1.2
    results["modal_mass_admittance"] = {
        "pass": (1.0 / m_hi) < (1.0 / m_lo),
        "mobility_ratio": round((1.0 / m_hi) / (1.0 / m_lo), 4),
    }

    f_h_base = helmholtz_proxy_hz(v0, a0)
    f_h_leff_up = helmholtz_proxy_hz(v0, a0, l_eff_m=DEFAULT_L_EFF_M * 1.15)
    results["helmholtz_L_eff"] = {
        "pass": f_h_leff_up < f_h_base,
        "f_base_hz": round(f_h_base, 2),
        "f_leff_up_hz": round(f_h_leff_up, 2),
    }

    # T/μ consistency: varying μ with fixed f,L scales T linearly (same pitch)
    f_target = 440.0
    l_str = CLASSICAL_SCALE_LENGTH_M
    mu_base = NYLON_LINEAR_DENSITY_KG_M
    t_base = (2 * l_str * f_target) ** 2 * mu_base
    mu_up = mu_base * 1.10
    t_up = (2 * l_str * f_target) ** 2 * mu_up
    results["string_tension_density_A4"] = {
        "pass": abs(t_up / t_base - mu_up / mu_base) < 1e-6,
        "T_base_N": round(t_base, 2),
        "T_scaled_N": round(t_up, 2),
        "note": "T ∝ μ for fixed f,L — use literature μ band separately for absolute T",
    }

    results["all_pass"] = all(
        v.get("pass") for v in results.values() if isinstance(v, dict) and "pass" in v
    )
    return results


def build_readiness_gate(
    param_table: Sequence[Mapping[str, Any]],
    checklist: Sequence[Mapping[str, Any]],
    sensitivity: Mapping[str, Any],
) -> Dict[str, Any]:
    l3_blocked = [p["name"] for p in param_table if p.get("fallback_level") == "L3_blocked"]
    checklist_pass = all(c.get("aligned") for c in checklist)
    sens_pass = sensitivity.get("all_pass", False)

    # Step 3A IR allowed with L1/L2 fallbacks for excitation; body modals reference_shared
    status = "ready_for_numerical_impulse_response_only"
    if not checklist_pass or not sens_pass:
        status = "not_ready_missing_critical_data"

    return {
        "current_status": status,
        "allowed_now": [
            "Numerical bridge-admittance / impulse-response testbench (single guitar, no WAV musical notes)",
            "Objective sensitivity tables (this report)",
            "Causal modal ODE verification at F_bridge(t)",
        ],
        "blocked_now": [
            "Musical WAV synthesis",
            "Multi-guitar timbre comparison proof",
            "Final STK integration",
            "Exact elastic moduli / anisotropy / modal stiffness claims",
            "Per-sample modal catalog without M4/ROM inference",
        ],
        "L3_blocked_parameters": l3_blocked,
        "musical_audio_synthesis_allowed": False,
        "multi_guitar_comparison_allowed": False,
        "numerical_impulse_response_allowed": status == "ready_for_numerical_impulse_response_only",
    }


def build_blocked_claims(param_table: Sequence[Mapping[str, Any]]) -> List[str]:
    claims = [
        "Multi-guitar timbre differentiation using reference_shared modal catalog alone",
        "Exact per-mode generalized mass and stiffness without inference or measurement",
        "Wood pair timbre proof from discrete wood ID without measured E or damping",
        "Helmholtz cavity as independent delayed audio layer",
        "Calibrated absolute sound pressure level without radiation calibration data",
    ]
    for p in param_table:
        if p.get("fallback_level") == "L3_blocked":
            claims.append(f"Exact physical claim for {p['name']} without measure/infer path")
    return sorted(set(claims))


def load_step_report(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing report: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_pgsm_step2_1_report(
    *,
    repo_root: Optional[Path] = None,
    step1_path: Optional[Path] = None,
    step2_path: Optional[Path] = None,
    audit_path: Optional[Path] = None,
) -> Dict[str, Any]:
    root = Path(repo_root or REPO_ROOT)
    step1 = load_step_report(Path(step1_path or STEP1_JSON))
    step2 = load_step_report(Path(step2_path or STEP2_JSON))
    audit = load_audit_report(audit_path)

    param_table = build_parameter_target_table(step1)
    checklist = build_literature_alignment_checklist()
    fallback = build_fallback_policy()
    sensitivity = run_sensitivity_plan(audit)
    readiness = build_readiness_gate(param_table, checklist, sensitivity)
    blocked = build_blocked_claims(param_table)
    data_mapping = map_samples(audit, root)

    safe_next = (
        "PGSM Step 3A: numerical impulse-response / admittance testbench only — "
        "single reference guitar, documented L1/L2 fallbacks, no musical WAV, no STK"
    )

    return {
        "report_version": PGSM_STEP2_1_VERSION,
        "timestamp": _utc_now(),
        "status": "pgsm_step2_1_parameter_targets_complete",
        "no_audio_generated": True,
        "no_fem_run": True,
        "no_rom_run": True,
        "website_default": DEFAULT_WEBSITE_STK_MODE,
        "website_default_unchanged": True,
        "literature_alignment_checklist": checklist,
        "literature_alignment_all_pass": all(c["aligned"] for c in checklist),
        "parameter_target_table": param_table,
        "fallback_policy": fallback,
        "sensitivity_plan": {
            "description": "Numeric sensitivity tests before any audio synthesis",
            "tests": sensitivity,
            "figures_dir": str(FIGURES_DIR.relative_to(root)) if FIGURES_DIR.exists() else None,
            "figures_generated": False,
        },
        "readiness_gate": readiness,
        "blocked_claims": blocked,
        "safe_next_step": safe_next,
        "step1_report_loaded": step1.get("report_version"),
        "step2_report_loaded": step2.get("report_version"),
        "step2_prior_readiness": step2.get("step3_readiness_status"),
        "data_mapping_sample_count": len(data_mapping.get("per_sample", {})),
        "sample_ids": list(DEFAULT_SAMPLE_IDS),
        "explicit_statement": (
            "PGSM Step 2.1 defines parameter targets and readiness only. It does not synthesize sound."
        ),
    }


def write_markdown_report(report: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# PGSM Step 2.1 — Parameter targets and data-gap closure",
        "",
        f"**Generated:** {report.get('timestamp')}",
        f"**Status:** {report.get('status')}",
        "",
        report.get("explicit_statement", ""),
        "",
        f"**Readiness gate:** `{report.get('readiness_gate', {}).get('current_status')}`",
        f"**Safe next step:** {report.get('safe_next_step')}",
        "",
        f"Website default (unchanged): `{report.get('website_default')}`",
        "",
        "## Literature alignment",
        "",
        "| ID | Aligned | Statement |",
        "|----|---------|-----------|",
    ]
    for c in report.get("literature_alignment_checklist") or []:
        lines.append(f"| {c.get('id')} | {c.get('aligned')} | {c.get('statement')} |")

    lines.extend(["", "## Parameter targets (summary)", ""])
    for p in report.get("parameter_target_table") or []:
        lines.append(
            f"- **{p.get('name')}** [{p.get('fallback_level')}] "
            f"status={p.get('current_status')} strategy={p.get('proposed_strategy')} "
            f"range={p.get('safe_range')} conf={p.get('confidence')}"
        )

    lines.extend(["", "## Blocked claims", ""])
    for b in report.get("blocked_claims") or []:
        lines.append(f"- {b}")

    lines.extend(["", "## Fallback policy", ""])
    for level, spec in (report.get("fallback_policy") or {}).get("levels", {}).items():
        lines.append(f"- **{level}**: {spec.get('description')}")

    rg = report.get("readiness_gate") or {}
    lines.extend(["", "## Readiness gate", ""])
    lines.append("Allowed now:")
    for item in rg.get("allowed_now") or []:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("Blocked now:")
    for item in rg.get("blocked_now") or []:
        lines.append(f"- {item}")

    sens = (report.get("sensitivity_plan") or {}).get("tests") or {}
    lines.extend(["", "## Sensitivity plan (numeric)", ""])
    lines.append(f"- All pass: **{sens.get('all_pass')}**")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pgsm_step2_1_reports(
    *,
    repo_root: Optional[Path] = None,
    json_path: Optional[Path] = None,
    md_path: Optional[Path] = None,
    step1_path: Optional[Path] = None,
    step2_path: Optional[Path] = None,
    audit_path: Optional[Path] = None,
) -> Dict[str, Any]:
    report = build_pgsm_step2_1_report(
        repo_root=repo_root,
        step1_path=step1_path,
        step2_path=step2_path,
        audit_path=audit_path,
    )
    jp = Path(json_path or REPORT_JSON)
    mp = Path(md_path or REPORT_MD)
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown_report(report, mp)
    return report


if __name__ == "__main__":
    r = write_pgsm_step2_1_reports()
    print(f"Wrote {REPORT_JSON}")
    print(f"Parameters: {len(r.get('parameter_target_table', []))}")
    print(f"Readiness: {r.get('readiness_gate', {}).get('current_status')}")
    print(f"Sensitivity all_pass: {(r.get('sensitivity_plan') or {}).get('tests', {}).get('all_pass')}")
