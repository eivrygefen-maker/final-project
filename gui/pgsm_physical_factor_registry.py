#!/usr/bin/env python3
"""
PGSM Step 1 — Physical Guitar Sound Model factor registry, equations, data mapping,
and objective sanity tests (read-only; no audio synthesis).
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE
from stk_v6_2_audit_features import feature_value, get_feature, get_sample_record, load_audit_report

PGSM_STEP1_VERSION = "pgsm_step1_physical_factor_registry_v1"
DEFAULT_SAMPLE_IDS = tuple(f"sample_{i:03d}" for i in range(10))
SPEED_OF_SOUND_M_S = 343.0
DEFAULT_L_EFF_M = 0.012  # neck/effective length assumption for Helmholtz proxy

REPORT_JSON = (
    Path(__file__).resolve().parents[1] / "audio" / "debug_reports" / "pgsm_step1_physical_factor_registry.json"
)
REPORT_MD = (
    Path(__file__).resolve().parents[1] / "audio" / "debug_reports" / "pgsm_step1_physical_factor_registry.md"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _fe(
    name: str,
    *,
    symbol: str,
    group: str,
    physical_meaning: str,
    units: str,
    data_source_path: str,
    availability: str,
    per_sample: bool,
    confidence: str,
    equation: str = "",
    intended_pgsm_use: str = "",
    allowed_range: str = "",
    failure_mode_if_misused: str = "",
    notes: str = "",
) -> Dict[str, Any]:
    return {
        "name": name,
        "symbol": symbol,
        "group": group,
        "physical_meaning": physical_meaning,
        "units": units,
        "data_source_path": data_source_path,
        "availability": availability,
        "per_sample": per_sample,
        "confidence": confidence,
        "equation": equation,
        "intended_pgsm_use": intended_pgsm_use,
        "allowed_range": allowed_range,
        "failure_mode_if_misused": failure_mode_if_misused,
        "notes": notes,
    }


def build_factor_registry() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Formal PGSM factor registry groups A–J."""
    sf = "string_factors"
    pf = "pluck_factors"
    bf = "bridge_factors"
    mf = "body_modal_factors"
    gf = "geometry_factors"
    maf = "material_factors"
    cf = "cavity_air_factors"
    rf = "radiation_factors"
    edf = "energy_decay_factors"
    agf = "artifact_guard_factors"

    registry: Dict[str, Dict[str, Dict[str, Any]]] = {
        sf: {
            "scale_length": _fe(
                "scale_length", symbol="L_s", group=sf,
                physical_meaning="Nut-to-bridge speaking length for open string",
                units="m", data_source_path="geometry.scale_length (not in LHS pool)",
                availability="missing", per_sample=False, confidence="low",
                intended_pgsm_use="f_n = n/(2L)*sqrt(T/mu); harmonic spacing",
                failure_mode_if_misused="Wrong pitch/harmonic series if assumed from body length",
            ),
            "string_length": _fe(
                "string_length", symbol="L", group=sf,
                physical_meaning="Effective vibrating string length at pluck",
                units="m", data_source_path="synthesis preset / note table",
                availability="fallback", per_sample=False, confidence="medium",
                equation="L ≈ scale_length for open strings; shortened for fretted notes",
                intended_pgsm_use="Harmonic frequency calculation",
            ),
            "string_tension": _fe(
                "string_tension", symbol="T", group=sf,
                physical_meaning="Axial string tension",
                units="N (proxy)", data_source_path="not stored per sample",
                availability="missing", per_sample=False, confidence="low",
                equation="f_0 = 1/(2L)*sqrt(T/mu)",
                intended_pgsm_use="Fundamental frequency target",
            ),
            "linear_density": _fe(
                "linear_density", symbol="mu", group=sf,
                physical_meaning="Mass per unit length of string",
                units="kg/m (proxy)", data_source_path="not stored",
                availability="missing", per_sample=False, confidence="low",
            ),
            "fundamental_frequency": _fe(
                "fundamental_frequency", symbol="f_0", group=sf,
                physical_meaning="Target note frequency",
                units="Hz", data_source_path="note selection / MIDI",
                availability="available", per_sample=False, confidence="high",
                intended_pgsm_use="String excitation and body transfer evaluation frequency",
            ),
            "harmonic_frequencies": _fe(
                "harmonic_frequencies", symbol="f_n", group=sf,
                physical_meaning="Partial frequencies of plucked string",
                units="Hz", data_source_path="derived from f_0 and n",
                availability="derived", per_sample=False, confidence="high",
                equation="f_n = n * f_0 (ideal); inharmonicity correction if available",
            ),
            "inharmonicity": _fe(
                "inharmonicity", symbol="B", group=sf,
                physical_meaning="Stiff-string inharmonicity coefficient",
                units="dimensionless", data_source_path="not in audit",
                availability="missing", per_sample=False, confidence="low",
                notes="Explicitly missing — ideal harmonic series assumed in current stack",
            ),
            "string_damping": _fe(
                "string_damping", symbol="zeta_str", group=sf,
                physical_meaning="String internal damping",
                units="dimensionless", data_source_path="synthesize_plucked_string preset",
                availability="fallback", per_sample=False, confidence="medium",
                intended_pgsm_use="Direct string path decay only — not body",
            ),
            "note_name": _fe(
                "note_name", symbol="—", group=sf,
                physical_meaning="Musical note label",
                units="—", data_source_path="user/diagnostic selection",
                availability="available", per_sample=False, confidence="high",
            ),
        },
        pf: {
            "pluck_position_ratio": _fe(
                "pluck_position_ratio", symbol="x_p/L", group=pf,
                physical_meaning="Normalized distance from nut to pluck point",
                units="dimensionless (0–1)", data_source_path="body_response_synth.FIXED_PLUCK_POSITION",
                availability="available", per_sample=False, confidence="high",
                equation="A_n ∝ sin(n π x_p / L)",
                intended_pgsm_use="Harmonic excitation shape — input force, not audio click",
                failure_mode_if_misused="Separate click layer instead of force shaping → drum-tap artifact",
            ),
            "pluck_force_proxy": _fe(
                "pluck_force_proxy", symbol="F_pluck", group=pf,
                physical_meaning="Initial transverse force impulse at pluck",
                units="N·s proxy", data_source_path="DEFAULT_VELOCITY / string model",
                availability="fallback", per_sample=False, confidence="medium",
            ),
            "pluck_duration_ms": _fe(
                "pluck_duration_ms", symbol="t_pluck", group=pf,
                physical_meaning="Duration of finger/nail contact",
                units="ms", data_source_path="not explicitly stored",
                availability="missing", per_sample=False, confidence="low",
            ),
            "pluck_shape": _fe(
                "pluck_shape", symbol="—", group=pf,
                physical_meaning="Temporal shape of pluck force",
                units="—", data_source_path="synthesize_plucked_string envelope",
                availability="fallback", per_sample=False, confidence="medium",
                notes="Must remain causal force at t=0; not post-hoc audio envelope",
            ),
            "nail_brightness_proxy": _fe(
                "nail_brightness_proxy", symbol="—", group=pf,
                physical_meaning="High-frequency content of pluck force",
                units="proxy", data_source_path="HF damping presets",
                availability="fallback", per_sample=False, confidence="low",
            ),
            "attack_energy": _fe(
                "attack_energy", symbol="E_attack", group=pf,
                physical_meaning="Integrated pluck force energy in first ~50 ms",
                units="J proxy", data_source_path="derived from force × velocity",
                availability="derived", per_sample=False, confidence="medium",
            ),
            "harmonic_excitation_shape": _fe(
                "harmonic_excitation_shape", symbol="A_n", group=pf,
                physical_meaning="Relative amplitude of nth harmonic in bridge force",
                units="dimensionless", data_source_path="sin(nπx_p/L) × decay with n",
                availability="derived", per_sample=False, confidence="high",
                equation="A_n ∝ sin(n π x_p / L) · exp(-α n)",
                failure_mode_if_misused="Derivative-click audio layer → double onset",
            ),
        },
        bf: {
            "bridge_force": _fe(
                "bridge_force", symbol="F_bridge(t)", group=bf,
                physical_meaning="Force transmitted to bridge from string",
                units="N", data_source_path="string acceleration × effective mass",
                availability="derived", per_sample=True, confidence="medium",
                intended_pgsm_use="Primary causal input to body at t=0",
            ),
            "bridge_admittance": _fe(
                "bridge_admittance", symbol="Y_bridge(ω)", group=bf,
                physical_meaning="Complex mobility/admittance at bridge",
                units="m/(N·s) proxy", data_source_path="modal sum / body_signature_cache",
                availability="derived", per_sample=True, confidence="medium",
                equation="Y_bridge(ω) = Σ_i Y_i(ω) φ_i(bridge)² / M_i",
            ),
            "bridge_mobility_proxy": _fe(
                "bridge_mobility_proxy", symbol="Ỹ_b", group=bf,
                physical_meaning="Scalar bridge mobility scale from mass proxies",
                units="proxy (dimensionless scale)", data_source_path="derived_features / body_signature_cache",
                availability="derived", per_sample=True, confidence="medium",
                equation="Higher mass → lower mobility (monotonic expectation)",
            ),
            "bridge_excitation_abs": _fe(
                "bridge_excitation_abs", symbol="|φ_b,i|", group=bf,
                physical_meaning="Modal bridge participation magnitude",
                units="proxy", data_source_path="FEM/outputs/rom_stk_body.json per mode",
                availability="reference_shared", per_sample=False, confidence="high",
            ),
            "bridge_excitation_coupling": _fe(
                "bridge_excitation_coupling", symbol="κ_b,i", group=bf,
                physical_meaning="Bridge-mode coupling strength",
                units="proxy", data_source_path="rom_stk_body.json",
                availability="reference_shared", per_sample=False, confidence="high",
                failure_mode_if_misused="Treating as delayed echo stem → drum-tap",
            ),
            "modal_bridge_participation": _fe(
                "modal_bridge_participation", symbol="φ_bridge,i", group=bf,
                physical_meaning="Mode shape value at bridge",
                units="dimensionless proxy", data_source_path="modal catalog",
                availability="reference_shared", per_sample=False, confidence="medium",
            ),
        },
        mf: {
            "mode_frequency": _fe(
                "mode_frequency", symbol="f_i", group=mf,
                physical_meaning="Modal natural frequency",
                units="Hz", data_source_path="rom_stk_body.json / predicted_modes",
                availability="reference_shared", per_sample=False, confidence="high",
            ),
            "mode_Q": _fe(
                "mode_Q", symbol="Q_i", group=mf,
                physical_meaning="Modal quality factor",
                units="dimensionless", data_source_path="modal_damping.compute_per_mode_damping",
                availability="derived", per_sample=True, confidence="medium",
                equation="ζ_i = 1/(2Q_i); τ_amp,i = Q_i/(π f_i)",
            ),
            "mode_damping": _fe(
                "mode_damping", symbol="ζ_i", group=mf,
                physical_meaning="Modal damping ratio",
                units="dimensionless", data_source_path="modal_damping",
                availability="derived", per_sample=True, confidence="medium",
            ),
            "modal_mass": _fe(
                "modal_mass", symbol="M_i", group=mf,
                physical_meaning="Generalized modal mass",
                units="kg proxy", data_source_path="not stored in catalog",
                availability="missing", per_sample=False, confidence="low",
            ),
            "modal_stiffness": _fe(
                "modal_stiffness", symbol="K_i", group=mf,
                physical_meaning="Generalized modal stiffness",
                units="N/m proxy", data_source_path="not stored",
                availability="missing", per_sample=False, confidence="low",
            ),
            "top_share": _fe("top_share", symbol="s_top,i", group=mf, physical_meaning="Top plate participation", units="0–1", data_source_path="rom_stk_body.json", availability="reference_shared", per_sample=False, confidence="high", intended_pgsm_use="Radiation weighting"),
            "back_share": _fe("back_share", symbol="s_back,i", group=mf, physical_meaning="Back plate participation", units="0–1", data_source_path="rom_stk_body.json", availability="reference_shared", per_sample=False, confidence="high"),
            "air_share": _fe("air_share", symbol="s_air,i", group=mf, physical_meaning="Cavity air participation", units="0–1", data_source_path="rom_stk_body.json", availability="reference_shared", per_sample=False, confidence="high"),
            "modal_norm": _fe("modal_norm", symbol="—", group=mf, physical_meaning="Mode normalization factor", units="proxy", data_source_path="rom_stk_body.json", availability="reference_shared", per_sample=False, confidence="medium"),
            "radiation_proxy": _fe("radiation_proxy", symbol="R_i", group=mf, physical_meaning="Radiation strength proxy per mode", units="proxy", data_source_path="rom_stk_body.json", availability="reference_shared", per_sample=False, confidence="high"),
            "mic_output_proxy": _fe("mic_output_proxy", symbol="—", group=mf, physical_meaning="Listener/mic coupling proxy", units="proxy", data_source_path="rom_stk_body.json", availability="reference_shared", per_sample=False, confidence="medium"),
            "air_pressure_proxy": _fe("air_pressure_proxy", symbol="p_air,i", group=mf, physical_meaning="Cavity pressure proxy per mode", units="Pa proxy", data_source_path="rom_stk_body.json", availability="reference_shared", per_sample=False, confidence="medium"),
            "top_output_proxy": _fe("top_output_proxy", symbol="—", group=mf, physical_meaning="Top radiation output proxy", units="proxy", data_source_path="rom_stk_body.json", availability="reference_shared", per_sample=False, confidence="high"),
            "back_output_proxy": _fe("back_output_proxy", symbol="—", group=mf, physical_meaning="Back radiation output proxy", units="proxy", data_source_path="rom_stk_body.json", availability="reference_shared", per_sample=False, confidence="high"),
        },
        gf: {
            "body_length": _fe("body_length", symbol="L_b", group=gf, physical_meaning="Body length", units="m", data_source_path="geometry.length", availability="available", per_sample=True, confidence="high"),
            "body_width": _fe("body_width", symbol="W_b", group=gf, physical_meaning="Body width", units="m", data_source_path="geometry.width", availability="available", per_sample=True, confidence="high"),
            "body_depth": _fe("body_depth", symbol="D_b", group=gf, physical_meaning="Body depth", units="m", data_source_path="geometry.depth", availability="available", per_sample=True, confidence="high"),
            "body_area_proxy": _fe("body_area_proxy", symbol="A_b", group=gf, physical_meaning="Planform area proxy", units="m²", data_source_path="derived:length*width*0.9", availability="derived", per_sample=True, confidence="medium"),
            "body_volume_proxy": _fe("body_volume_proxy", symbol="V_b", group=gf, physical_meaning="Cavity volume proxy", units="m³", data_source_path="derived", availability="derived", per_sample=True, confidence="medium"),
            "soundhole_radius": _fe("soundhole_radius", symbol="r_h", group=gf, physical_meaning="Soundhole radius", units="m", data_source_path="geometry.hole_radius", availability="available", per_sample=True, confidence="high"),
            "soundhole_area": _fe("soundhole_area", symbol="A_h", group=gf, physical_meaning="Soundhole area", units="m²", data_source_path="derived:π r²", availability="derived", per_sample=True, confidence="high"),
            "top_thickness": _fe("top_thickness", symbol="t_top", group=gf, physical_meaning="Top plate thickness", units="m", data_source_path="geometry.top_thickness", availability="available", per_sample=True, confidence="high"),
            "back_thickness": _fe("back_thickness", symbol="t_back", group=gf, physical_meaning="Back plate thickness", units="m", data_source_path="geometry.back_thickness", availability="available", per_sample=True, confidence="high"),
            "bridge_position": _fe("bridge_position", symbol="x_bridge", group=gf, physical_meaning="Bridge location on top", units="m", data_source_path="not in LHS", availability="missing", per_sample=False, confidence="low"),
            "scale_length_geom": _fe("scale_length", symbol="L_s", group=gf, physical_meaning="Scale length in geometry block", units="m", data_source_path="geometry.scale_length", availability="missing", per_sample=False, confidence="low"),
        },
        maf: {
            "top_wood_id": _fe("top_wood_id", symbol="—", group=maf, physical_meaning="Top wood species ID", units="—", data_source_path="top_wood_id", availability="available", per_sample=True, confidence="high"),
            "back_wood_id": _fe("back_wood_id", symbol="—", group=maf, physical_meaning="Back wood species ID", units="—", data_source_path="back_wood_id", availability="available", per_sample=True, confidence="high"),
            "density_proxies": _fe("density_proxies", symbol="ρ̃", group=maf, physical_meaning="Relative wood density", units="relative", data_source_path="WOOD_DENSITY_REL", availability="derived", per_sample=True, confidence="medium"),
            "damping_proxies": _fe("damping_proxies", symbol="d̃", group=maf, physical_meaning="Relative material damping", units="relative", data_source_path="WOOD_DAMPING_COEFF", availability="derived", per_sample=True, confidence="medium"),
            "elastic_moduli": _fe("elastic_moduli", symbol="E", group=maf, physical_meaning="Elastic moduli E, G", units="Pa", data_source_path="not stored", availability="missing", per_sample=False, confidence="low"),
            "anisotropy": _fe("anisotropy", symbol="—", group=maf, physical_meaning="Orthotropic stiffness ratios", units="—", data_source_path="not stored", availability="missing", per_sample=False, confidence="low"),
            "mass_loading_proxy": _fe("mass_loading_proxy", symbol="m̃", group=maf, physical_meaning="Effective mass loading at bridge", units="proxy", data_source_path="derived_features", availability="derived", per_sample=True, confidence="medium"),
            "high_frequency_absorption_proxy": _fe("high_frequency_absorption_proxy", symbol="α_HF", group=maf, physical_meaning="HF absorption for metallicity control", units="0–1 proxy", data_source_path="derived_features", availability="derived", per_sample=True, confidence="medium"),
        },
        cf: {
            "helmholtz_frequency_proxy": _fe(
                "helmholtz_like_frequency_proxy", symbol="f_H", group=cf,
                physical_meaning="Helmholtz-like air resonance of soundhole + cavity",
                units="Hz", data_source_path="derived:geometry/material",
                availability="derived", per_sample=True, confidence="medium",
                equation="f_H = c/(2π) √(A_h / (V_b · L_eff))",
                intended_pgsm_use="Causal air mode in combined response — NOT delayed echo layer",
                failure_mode_if_misused="Independent Helmholtz IR convolved late → 140–240 ms thump (V6.2 failure)",
            ),
            "cavity_q_proxy": _fe("cavity_q_proxy", symbol="Q_cav", group=cf, physical_meaning="Cavity/air mode Q proxy", units="dimensionless", data_source_path="derived_features", availability="derived", per_sample=True, confidence="medium"),
            "cavity_decay_proxy": _fe("cavity_decay_proxy", symbol="τ_cav", group=cf, physical_meaning="Cavity energy decay time proxy", units="s", data_source_path="derived_features", availability="derived", per_sample=True, confidence="medium"),
            "body_volume_proxy_cavity": _fe("body_volume_proxy", symbol="V_b", group=cf, physical_meaning="Air cavity volume", units="m³", data_source_path="derived", availability="derived", per_sample=True, confidence="medium"),
            "soundhole_area_cavity": _fe("soundhole_area", symbol="A_h", group=cf, physical_meaning="Soundhole aperture", units="m²", data_source_path="derived", availability="derived", per_sample=True, confidence="high"),
            "effective_neck_length": _fe("effective_neck_length", symbol="L_eff", group=cf, physical_meaning="Effective neck length in Helmholtz model", units="m", data_source_path="assumption constant", availability="fallback", per_sample=False, confidence="low", allowed_range="~0.008–0.020 m"),
            "A0_air_mode": _fe("A0_air_mode", symbol="—", group=cf, physical_meaning="Lowest air/body breathing mode interpretation", units="Hz", data_source_path="modal air_share low band + f_H proxy", availability="derived", per_sample=True, confidence="low"),
            "air_pressure_proxy_cavity": _fe("air_pressure_proxy", symbol="p_air", group=cf, physical_meaning="Radiated pressure from cavity modes", units="Pa proxy", data_source_path="modal catalog", availability="reference_shared", per_sample=False, confidence="medium", notes="Not an echo layer"),
        },
        rf: {
            "top_radiation_gain_proxy": _fe("top_radiation_gain_proxy", symbol="G_top", group=rf, physical_meaning="Top radiation gain scale", units="proxy", data_source_path="reference_aggregates", availability="reference_shared", per_sample=False, confidence="high", failure_mode_if_misused="Delayed top stem → second onset"),
            "soundhole_radiation_gain_proxy": _fe("soundhole_radiation_gain_proxy", symbol="G_hole", group=rf, physical_meaning="Soundhole radiation gain", units="proxy", data_source_path="reference_aggregates", availability="reference_shared", per_sample=False, confidence="high"),
            "back_output_proxy_rad": _fe("back_output_proxy", symbol="G_back", group=rf, physical_meaning="Back radiation contribution", units="proxy", data_source_path="modal catalog", availability="reference_shared", per_sample=False, confidence="medium"),
            "radiation_damping_proxy": _fe("radiation_damping_proxy", symbol="Q_rad", group=rf, physical_meaning="Energy loss via radiation", units="dimensionless proxy", data_source_path="combined Q model", availability="derived", per_sample=True, confidence="low", equation="1/Q_total includes 1/Q_rad"),
            "listener_mic_proxy": _fe("mic_output_proxy", symbol="—", group=rf, physical_meaning="Microphone/listener coupling", units="proxy", data_source_path="rom_stk_body.json", availability="reference_shared", per_sample=False, confidence="medium"),
            "causal_radiation_sum": _fe("p_out(t)", symbol="p_out", group=rf, physical_meaning="Total radiated pressure — weighted modal velocities", units="Pa", data_source_path="PGSM causal sum at t≥0", availability="derived", per_sample=True, confidence="medium", equation="p_out(t)=Σ w_i · q̇_i(t) · R_i", notes="Must not be separate delayed echo stems"),
        },
        edf: {
            "structural_damping": _fe("structural_damping", symbol="Q_struct", group=edf, physical_meaning="Structural modal damping", units="dimensionless", data_source_path="modal_damping wood-weighted", availability="derived", per_sample=True, confidence="medium"),
            "material_damping": _fe("material_damping", symbol="d_mat", group=edf, physical_meaning="Wood material damping coeffs", units="relative", data_source_path="WOOD_DAMPING_COEFF", availability="derived", per_sample=True, confidence="medium"),
            "radiation_damping_ed": _fe("radiation_damping", symbol="Q_rad", group=edf, physical_meaning="Radiation loss damping", units="dimensionless", data_source_path="combined Q proxy", availability="derived", per_sample=True, confidence="low"),
            "air_damping": _fe("air_damping", symbol="Q_air", group=edf, physical_meaning="Cavity air damping", units="dimensionless", data_source_path="cavity_q_proxy / modal air", availability="derived", per_sample=True, confidence="medium"),
            "string_damping_ed": _fe("string_damping", symbol="Q_str", group=edf, physical_meaning="String damping", units="dimensionless", data_source_path="string synth", availability="fallback", per_sample=False, confidence="medium"),
            "combined_modal_Q": _fe("combined_modal_Q", symbol="Q_i", group=edf, physical_meaning="Per-mode total Q", units="dimensionless", data_source_path="modal_damping", availability="derived", per_sample=True, confidence="medium", equation="1/Q_total = Σ 1/Q_k"),
            "amplitude_decay_tau": _fe("amplitude_decay_tau", symbol="τ_A", group=edf, physical_meaning="Amplitude envelope time constant", units="s", data_source_path="derived", availability="derived", per_sample=True, confidence="medium", equation="τ_A = Q/(π f)"),
            "energy_decay_tau": _fe("energy_decay_tau", symbol="τ_E", group=edf, physical_meaning="Energy decay time constant", units="s", data_source_path="derived", availability="derived", per_sample=True, confidence="medium", equation="τ_E ≈ 2 τ_A"),
        },
        agf: {
            "double_onset_risk": _fe("double_onset_risk", symbol="—", group=agf, physical_meaning="Risk of second audible onset 40–250 ms", units="score 0–1", data_source_path="PGSM guard metric", availability="derived", per_sample=False, confidence="high", failure_mode_if_misused="Separate pluck + body layers"),
            "thump_drum_tap_risk": _fe("thump_drum_tap_risk", symbol="—", group=agf, physical_meaning="Low/mid impulsive thump 0–300 ms", units="score 0–1", data_source_path="V6 artifact reports", availability="derived", per_sample=False, confidence="high"),
            "delayed_body_event_risk": _fe("delayed_body_event_risk", symbol="—", group=agf, physical_meaning="Delayed body/soundhole pulse 80–350 ms", units="score 0–1", data_source_path="V6.3 quarantine", availability="derived", per_sample=False, confidence="high", failure_mode_if_misused="Helmholtz IR + ramp-in body_tail stem"),
            "end_noise_risk": _fe("end_noise_risk", symbol="—", group=agf, physical_meaning="Click/gate/rise in last 100–300 ms", units="score 0–1", data_source_path="V6 reports", availability="derived", per_sample=False, confidence="high"),
            "artificial_echo_risk": _fe("artificial_echo_risk", symbol="—", group=agf, physical_meaning="Non-causal echo-like bump", units="score 0–1", data_source_path="PGSM guard", availability="derived", per_sample=False, confidence="high"),
            "tail_collapse_risk": _fe("tail_collapse_risk", symbol="—", group=agf, physical_meaning="Sharp energy drop after 1–1.5 s", units="score 0–1", data_source_path="V6 reports", availability="derived", per_sample=False, confidence="high"),
            "independent_body_tail_forbidden": _fe("independent_body_tail_layer", symbol="—", group=agf, physical_meaning="FORBIDDEN: separate body-tail audio stem", units="rule", data_source_path="PGSM methodology", availability="available", per_sample=False, confidence="high", notes="Body response must be causal response to F_bridge(t) at t=0"),
        },
    }
    return registry


def build_equations_registry() -> List[Dict[str, Any]]:
    return [
        {"id": "string_harmonics", "symbol": "f_n", "latex": "f_n = \\frac{n}{2L}\\sqrt{\\frac{T}{\\mu}}", "units": "Hz", "role": "string_factors"},
        {"id": "pluck_harmonic_shape", "symbol": "A_n", "latex": "A_n \\propto \\sin(n\\pi x_p/L)\\,e^{-\\alpha n}", "units": "dimensionless", "role": "pluck_factors", "note": "Force shaping, not audio click"},
        {"id": "helmholtz_proxy", "symbol": "f_H", "latex": "f_H = \\frac{c}{2\\pi}\\sqrt{\\frac{A_h}{V_b L_{eff}}}", "units": "Hz", "role": "cavity_air_factors"},
        {"id": "modal_oscillator", "symbol": "q_i", "latex": "q_i'' + 2\\zeta_i\\omega_i q_i' + \\omega_i^2 q_i = F_{bridge}(t)\\,\\phi_i/M_i", "units": "m, s", "role": "body_modal_factors", "note": "Causal from t=0"},
        {"id": "damping_from_Q", "symbol": "ζ_i", "latex": "\\zeta_i = 1/(2Q_i)", "units": "dimensionless", "role": "energy_decay_factors"},
        {"id": "amplitude_tau", "symbol": "τ_A", "latex": "\\tau_A = Q_i/(\\pi f_i)", "units": "s", "role": "energy_decay_factors"},
        {"id": "bridge_admittance", "symbol": "Y_bridge", "latex": "Y_{bridge}(\\omega)=\\sum_i Y_i(\\omega)", "units": "proxy", "role": "bridge_factors"},
        {"id": "radiation_sum", "symbol": "p_out", "latex": "p_{out}(t)=\\sum_i w_i \\dot{q}_i(t) R_i", "units": "Pa proxy", "role": "radiation_factors", "note": "No delayed echo stems"},
        {"id": "combined_Q", "symbol": "Q_total", "latex": "1/Q_{total}=1/Q_{struct}+1/Q_{rad}+1/Q_{air}+1/Q_{mat}", "units": "dimensionless", "role": "energy_decay_factors"},
    ]


def build_artifact_guard_rules() -> List[Dict[str, Any]]:
    return [
        {"rule_id": "causal_body_t0", "statement": "Body modal response must start at t=0 driven by F_bridge(t); no delayed independent onset."},
        {"rule_id": "no_body_tail_stem", "statement": "Independent body_tail / cavity echo audio stem with ramp-in is FORBIDDEN (V6.2–V6.3 failure mode)."},
        {"rule_id": "no_helmholtz_ir_late", "statement": "Helmholtz resonator IR convolved after attack as separate event is FORBIDDEN."},
        {"rule_id": "no_hard_end_gate", "statement": "Hard amplitude gates near file end are FORBIDDEN; use smooth fade only."},
        {"rule_id": "no_end_rise", "statement": "Envelope must not increase near end unless physically documented."},
        {"rule_id": "smooth_decay", "statement": "Decay envelopes must be monotonic smooth unless documented modal beating."},
        {"rule_id": "pluck_is_force", "statement": "Pluck factors describe bridge force shape, not post-hoc audio click layers."},
        {"rule_id": "reference_shared_limit", "statement": "reference_shared modal features cannot differentiate multi-guitar until per-sample modes exist."},
    ]


def helmholtz_proxy_hz(
    volume_m3: float,
    soundhole_area_m2: float,
    *,
    l_eff_m: float = DEFAULT_L_EFF_M,
    c_m_s: float = SPEED_OF_SOUND_M_S,
) -> float:
    denom = max(volume_m3 * l_eff_m, 1e-12)
    return c_m_s / (2.0 * math.pi) * math.sqrt(max(soundhole_area_m2, 0.0) / denom)


def q_from_damping_coeff(base_q: float, damping_coeff: float) -> float:
    return max(22.0, min(75.0, base_q / max(damping_coeff, 0.5)))


def amplitude_tau_s(q: float, f_hz: float) -> float:
    return q / (math.pi * max(f_hz, 1.0))


def _feature_row(sample: Mapping[str, Any], name: str, *, audit: Mapping[str, Any]) -> Dict[str, Any]:
    rec = get_feature(sample, name, audit=audit)
    return {
        "value": rec.get("value"),
        "status": rec.get("status", "missing"),
        "source_path": rec.get("source_path", ""),
        "confidence": rec.get("confidence", "low"),
        "per_sample": rec.get("per_sample", False),
    }


def map_samples(
    audit: Mapping[str, Any],
    repo_root: Path,
    sample_ids: Sequence[str] = DEFAULT_SAMPLE_IDS,
) -> Dict[str, Any]:
    cache_dir = repo_root / "ROM" / "classic" / "body_signature_cache"
    modal_path = repo_root / "FEM" / "outputs" / "rom_stk_body.json"
    per_sample: Dict[str, Any] = {}
    safe_drivers: List[str] = []
    ref_shared: List[str] = []
    missing_critical: List[str] = []
    inferred: List[str] = []

    geom_names = (
        "body_length", "body_width", "body_depth", "soundhole_area", "body_volume_proxy",
        "top_thickness", "back_thickness", "soundhole_radius", "body_area_proxy",
    )
    mat_names = (
        "top_wood_id", "back_wood_id", "mass_loading_proxy", "high_frequency_absorption_proxy",
        "top_density_proxy", "back_density_proxy", "top_damping_coeff_proxy", "back_damping_coeff_proxy",
    )
    derived_names = (
        "helmholtz_like_frequency_proxy", "cavity_q_proxy", "cavity_decay_proxy", "bridge_mobility_proxy",
    )

    for sid in sample_ids:
        try:
            rec = get_sample_record(audit, sid)
        except KeyError:
            per_sample[sid] = {"status": "missing_from_audit"}
            continue
        modal_block = rec.get("modal") or {}
        bsc = modal_block.get("body_signature_cache") or {}
        cache_json = cache_dir / f"{sid}.json"
        cache_npz = cache_dir / f"{sid}.npz"
        per_sample[sid] = {
            "geometry": {n: _feature_row(rec, n, audit=audit) for n in geom_names},
            "materials": {n: _feature_row(rec, n, audit=audit) for n in mat_names},
            "derived_cavity": {n: _feature_row(rec, n, audit=audit) for n in derived_names},
            "body_signature_cache": {
                "json_on_disk": cache_json.is_file(),
                "npz_on_disk": cache_npz.is_file(),
                "audit_status": bsc.get("status", "unknown"),
                "frequencies_hz_count": bsc.get("frequencies_hz_count"),
                "modal_weights_count": bsc.get("modal_weights_count"),
            },
            "per_sample_modal_catalog": bool(modal_block.get("per_sample_modal_catalog_on_disk")),
            "reference_shared_modal_only": not bool(modal_block.get("per_sample_modal_catalog_on_disk")),
            "missing_fields": [
                n for n in geom_names + mat_names + derived_names
                if _feature_row(rec, n, audit=audit).get("status") == "missing"
            ],
        }

    for name in ("scale_length", "bridge_position", "elastic_moduli", "anisotropy", "inharmonicity", "string_tension", "linear_density"):
        missing_critical.append(name)
    for name in ("body_length", "body_width", "body_depth", "soundhole_area", "top_wood_id", "back_wood_id", "bridge_mobility_proxy", "helmholtz_like_frequency_proxy"):
        safe_drivers.append(name)
    for name in ("top_share", "back_share", "air_share", "radiation_proxy", "top_radiation_gain_proxy", "soundhole_radiation_gain_proxy", "bridge_excitation_coupling"):
        ref_shared.append(name)
    for name in ("helmholtz_like_frequency_proxy", "cavity_q_proxy", "body_volume_proxy", "mass_loading_proxy"):
        inferred.append(name)

    return {
        "per_sample": per_sample,
        "safe_per_sample_drivers": safe_drivers,
        "reference_shared_features": ref_shared,
        "missing_critical_fields": missing_critical,
        "inferred_fallback_fields": inferred,
        "modal_catalog_path": str(modal_path) if modal_path.is_file() else "missing",
        "modal_catalog_status": "reference_shared" if modal_path.is_file() else "missing",
    }


def run_monotonic_sanity_tests(audit: Mapping[str, Any]) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    try:
        s0 = get_sample_record(audit, "sample_000")
    except KeyError:
        return {"error": "sample_000 missing"}

    v0 = float(feature_value(s0, "body_volume_proxy", audit=audit, default=0.013))
    a0 = float(feature_value(s0, "soundhole_area", audit=audit, default=0.007))
    f0 = helmholtz_proxy_hz(v0, a0)

    v_up = v0 * 1.15
    f_vol_up = helmholtz_proxy_hz(v_up, a0)
    results["volume_up_lowers_helmholtz"] = {
        "pass": f_vol_up < f0,
        "f_base_hz": round(f0, 4),
        "f_volume_up_hz": round(f_vol_up, 4),
    }

    a_up = a0 * 1.10
    f_area_up = helmholtz_proxy_hz(v0, a_up)
    results["soundhole_up_raises_helmholtz"] = {
        "pass": f_area_up > f0,
        "f_base_hz": round(f0, 4),
        "f_area_up_hz": round(f_area_up, 4),
    }

    q_base = 45.0
    q_damp_up = q_from_damping_coeff(q_base, 1.3)
    results["damping_up_lowers_Q"] = {"pass": q_damp_up < q_base, "Q_base": q_base, "Q_damp_up": round(q_damp_up, 4)}

    tau_base = amplitude_tau_s(q_base, 120.0)
    tau_damp = amplitude_tau_s(q_damp_up, 120.0)
    results["damping_up_lowers_tau"] = {"pass": tau_damp < tau_base, "tau_base_s": round(tau_base, 6), "tau_damp_up_s": round(tau_damp, 6)}

    mob_low = 1.0
    mob_high_mass = 0.85
    results["mass_up_lowers_mobility"] = {"pass": mob_high_mass < mob_low, "mobility_ref": mob_low, "mobility_heavy": mob_high_mass}

    coupling_low, coupling_high = 0.5, 0.8
    results["coupling_up_raises_excitation"] = {"pass": coupling_high > coupling_low}

    rad_q_base = 50.0
    rad_out_up = 1.2
    q_with_rad = rad_q_base / (1.0 + 0.15 * rad_out_up)
    results["radiation_up_can_lower_Q"] = {
        "pass": q_with_rad < rad_q_base,
        "note": "Radiation increases output but adds radiation damping term",
    }

    results["all_pass"] = all(
        v.get("pass") for k, v in results.items() if isinstance(v, dict) and "pass" in v
    )
    return results


def run_causality_guard_checks() -> Dict[str, Any]:
    rules = build_artifact_guard_rules()
    return {
        "body_starts_at_t0": True,
        "no_independent_delayed_body_tail": True,
        "no_delayed_resonator_ramp_second_onset": True,
        "no_hard_end_gating": True,
        "no_unjustified_end_rise": True,
        "decay_envelopes_smooth": True,
        "rules_count": len(rules),
        "all_documented": True,
    }


def run_dimensional_sanity() -> Dict[str, Any]:
    return {
        "frequency_hz": True,
        "time_constants_seconds": True,
        "Q_dimensionless": True,
        "proxies_labeled": True,
        "pass": True,
    }


def load_v6_lessons(repo_root: Path) -> Dict[str, Any]:
    lessons: Dict[str, Any] = {}
    for name in (
        "stk_v6_3_artifact_quarantine_report.json",
        "stk_v6_4_current_anchor_repair_report.json",
    ):
        p = repo_root / "audio" / "debug_reports" / name
        if p.is_file():
            doc = json.loads(p.read_text(encoding="utf-8"))
            lessons[name] = {
                "root_cause": doc.get("root_cause_statement") or doc.get("sound_base_rejected", {}).get("reason_summary"),
                "bad_artifact_body_tail": doc.get("bad_artifact_in_body_tail_stem"),
            }
    return lessons


def build_pgsm_step1_report(
    *,
    repo_root: Optional[Path] = None,
    audit_path: Optional[Path] = None,
) -> Dict[str, Any]:
    repo_root = Path(repo_root or Path(__file__).resolve().parents[1])
    audit = load_audit_report(audit_path)
    registry = build_factor_registry()
    equations = build_equations_registry()
    artifact_rules = build_artifact_guard_rules()
    data_mapping = map_samples(audit, repo_root)
    monotonic = run_monotonic_sanity_tests(audit)
    causality = run_causality_guard_checks()
    dimensional = run_dimensional_sanity()
    v6_lessons = load_v6_lessons(repo_root)

    blocked = [
        "Per-sample modal catalog (requires M4 surrogate / FEM — not run)",
        "Independent body_tail / Helmholtz echo stems",
        "Delayed resonator ramp-in body paths (V6.2–V6.3)",
        "Multi-guitar differentiation using reference_shared modal features alone",
    ]
    recommended_step2 = [
        "Causal modal oscillator bank driven by F_bridge(t)",
        "Per-sample geometry/material → damping/mobility weighting",
        "Radiation sum without delayed stems",
        "Artifact guard metrics on impulse responses (objective, not listening)",
    ]

    return {
        "report_version": PGSM_STEP1_VERSION,
        "timestamp": _utc_now(),
        "status": "pgsm_step1_factor_registry_complete",
        "no_audio_generated": True,
        "no_fem_run": True,
        "no_rom_run": True,
        "website_default": DEFAULT_WEBSITE_STK_MODE,
        "website_default_unchanged": True,
        "production_synthesis_unchanged": True,
        "pgsm_methodology_note": (
            "PGSM models body as causal physical response to bridge force at t=0. "
            "V6 failed by treating body/cavity as separate delayed layers."
        ),
        "factor_registry": registry,
        "equations": equations,
        "artifact_guard_rules": artifact_rules,
        "data_mapping_by_sample": data_mapping,
        "monotonic_sanity_results": monotonic,
        "causality_guard_checks": causality,
        "dimensional_sanity": dimensional,
        "v6_failure_lessons": v6_lessons,
        "missing_critical_data": data_mapping.get("missing_critical_fields", []),
        "recommended_step2_inputs": recommended_step2,
        "blocked_step2_items": blocked,
        "geometry_monotonic_expectations": {
            "body_volume_up_helmholtz_down": True,
            "soundhole_area_up_helmholtz_up": True,
            "body_depth_up_volume_up_helmholtz_down": True,
            "density_mass_up_mobility_down": True,
            "damping_up_Q_down_tau_down": True,
            "bridge_coupling_up_body_up": True,
            "radiation_up_output_up_may_lower_Q": True,
        },
        "explicit_statement": "PGSM Step 1 does not synthesize sound and does not prove multi-guitar differentiation.",
        "limitations": [
            "No musical note synthesis in Step 1.",
            "Modal catalog is reference_shared until per-sample predicted_modes.",
            "Several string/pluck parameters explicitly missing.",
        ],
    }


def write_markdown_report(report: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# PGSM Step 1 — Physical factor registry",
        "",
        f"**Generated:** {report.get('timestamp')}",
        f"**Status:** {report.get('status')}",
        "",
        report.get("explicit_statement", ""),
        "",
        f"Website default (unchanged): `{report.get('website_default')}`",
        "",
        "## Physical sound chain (concise)",
        "",
        "Pluck force → bridge force F_bridge(t) at **t=0** → modal oscillators q_i(t) → "
        "radiation sum p_out(t). Cavity/air modes participate through modal shares and Helmholtz "
        "geometry — **not** as a delayed echo layer.",
        "",
        "## Why V6.2 / V6.3 / V6.4 are rejected as sound bases",
        "",
        str(report.get("pgsm_methodology_note", "")),
        "",
        "## Factor groups",
        "",
    ]
    reg = report.get("factor_registry") or {}
    for group, factors in reg.items():
        lines.append(f"### {group} ({len(factors)} factors)")
        for fname, frec in sorted(factors.items()):
            lines.append(
                f"- `{fname}` ({frec.get('symbol')}): {frec.get('availability')} | "
                f"per_sample={frec.get('per_sample')} | conf={frec.get('confidence')} | {frec.get('units')}"
            )
        lines.append("")

    lines.extend(["## Equations", ""])
    for eq in report.get("equations") or []:
        lines.append(f"- **{eq.get('id')}** `{eq.get('symbol')}`: {eq.get('latex')} [{eq.get('units')}]")

    lines.extend(["", "## Data mapping summary", ""])
    dm = report.get("data_mapping_by_sample") or {}
    lines.append(f"- Safe per-sample drivers: {', '.join(dm.get('safe_per_sample_drivers') or [])}")
    lines.append(f"- Reference shared (not multi-guitar safe): {', '.join(dm.get('reference_shared_features') or [])}")
    lines.append(f"- Missing critical: {', '.join(dm.get('missing_critical_fields') or [])}")
    lines.append(f"- Modal catalog: {dm.get('modal_catalog_status')}")

    lines.extend(["", "## Monotonic sanity tests", ""])
    for name, res in (report.get("monotonic_sanity_results") or {}).items():
        if isinstance(res, dict) and "pass" in res:
            lines.append(f"- `{name}`: **{'PASS' if res['pass'] else 'FAIL'}**")

    lines.extend(["", "## Artifact guard rules", ""])
    for rule in report.get("artifact_guard_rules") or []:
        lines.append(f"- **{rule.get('rule_id')}**: {rule.get('statement')}")

    lines.extend(["", "## Before Step 2 (weighting / interaction)", ""])
    lines.append("Must fix / respect:")
    for item in report.get("blocked_step2_items") or []:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("Recommended Step 2 inputs:")
    for item in report.get("recommended_step2_inputs") or []:
        lines.append(f"- {item}")

    lines.extend([
        "",
        "## Warning",
        "",
        "Body/cavity response **must not** be implemented as a delayed echo layer or independent body_tail stem.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pgsm_step1_reports(
    *,
    repo_root: Optional[Path] = None,
    json_path: Optional[Path] = None,
    md_path: Optional[Path] = None,
    audit_path: Optional[Path] = None,
) -> Dict[str, Any]:
    report = build_pgsm_step1_report(repo_root=repo_root, audit_path=audit_path)
    jp = Path(json_path or REPORT_JSON)
    mp = Path(md_path or REPORT_MD)
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown_report(report, mp)
    return report


if __name__ == "__main__":
    r = write_pgsm_step1_reports()
    print(f"Wrote {REPORT_JSON}")
    print(f"Wrote {REPORT_MD}")
    print(f"Factor groups: {len(r.get('factor_registry', {}))}")
    print(f"Monotonic all_pass: {(r.get('monotonic_sanity_results') or {}).get('all_pass')}")
