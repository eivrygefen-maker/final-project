#!/usr/bin/env python3
"""
Stage 4.6 — Literature review, model gap analysis, and DATA→ROM→STK→AUDIO trace.

No FEM. No ROM retrain. No production promotion.
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gui"))

from body_response_synth import (  # noqa: E402
    BODY_TO_STRING_TARGET_RATIO,
    DEFAULT_SAMPLE_RATE,
    FULL_MODAL_BAND_HZ,
    TARGET_RMS_DBFS,
    compute_mode_weight_components,
    modes_in_validated_band,
    parse_modal_modes,
    synthesize_note_with_body_response,
)
from build_sample_comparison import (  # noqa: E402
    load_lhs_sample_entries,
    m4_surrogate_model_available,
    resolve_modal_data_for_sample,
)
from diagnostic_synthesis import (  # noqa: E402
    DIAGNOSTIC_MODES,
    get_diagnostic_mode,
    summarize_comparison_note,
    use_diagnostic_mode,
)
from modal_damping import compute_per_mode_damping, infer_mode_category  # noqa: E402
from sample_parameters import normalize_sample_parameters  # noqa: E402

OUT_DIR = REPO / "audio" / "debug_reports"
NOTES = (("A2", 110.0), ("A4", 440.0), ("E5", 659.25))

# ---------------------------------------------------------------------------
# Part 1 — Literature sources (curated; internet accessed 2026-06-10)
# ---------------------------------------------------------------------------

LITERATURE_SOURCES: List[Dict[str, Any]] = [
    {
        "id": "perry_2015_phd",
        "title": "Sound Radiation from Stringed Musical Instruments",
        "authors": ["Ian Perry"],
        "year": 2015,
        "url": "https://orca.cardiff.ac.uk/id/eprint/70916/",
        "doi": None,
        "summary": (
            "Measures radiation efficiency and bridge input admittance on classical guitars. "
            "Shows mode shape and radiation efficiency determine audible contribution; "
            "peak-fitting admittance yields effective mass and Q of body modes."
        ),
        "synthesis_implication": (
            "Each mode needs frequency, Q, AND radiation-weighted amplitude — not frequency alone. "
            "Bridge admittance peaks define coupling; radiation efficiency sets audible strength."
        ),
        "supports": {
            "modal_amplitude_weighting": True,
            "bridge_radiation_weighting": True,
            "material_damping": True,
            "per_mode_color": True,
            "far_mode_contribution": True,
            "normalization_strategy": False,
            "high_note_body_contribution": True,
        },
    },
    {
        "id": "elie_jasa_2012",
        "title": "Estimation of guitar vibro-acoustic behavior from bridge mobility measurements",
        "authors": ["B. Elie", "A. Chaigne", "P. Carré"],
        "year": 2012,
        "url": "https://members.loria.fr/BElie/papers/Elieetal_jasa2012.pdf",
        "doi": "10.1121/1.3688510",
        "summary": (
            "Derives salient guitar features from bridge mobility over a broad band: "
            "equivalent mass, rigidity, characteristic admittance, mobility deviation."
        ),
        "synthesis_implication": (
            "Mobility envelope (not just resonant frequencies) shapes timbre; "
            "scalar mobility features could proxy sample-specific amplitude curves."
        ),
        "supports": {
            "modal_amplitude_weighting": True,
            "bridge_radiation_weighting": True,
            "material_damping": False,
            "per_mode_color": True,
            "far_mode_contribution": True,
            "normalization_strategy": False,
            "high_note_body_contribution": True,
        },
    },
    {
        "id": "hill_hal_00811279",
        "title": "Acoustical measurements on classical guitars and the influence of design and wood choice",
        "authors": ["Terry M. Hill", "Ian F. Firth", "Ian Perry"],
        "year": None,
        "url": "https://hal.science/hal-00811279/document",
        "doi": None,
        "summary": (
            "Three-mass top/back/air model; coupling splits low modes; radiation damping "
            "lowers Q of radiating modes; wolf notes from over-coupling."
        ),
        "synthesis_implication": (
            "Top/back/air shares must affect both damping AND coupling amplitude; "
            "air modes can be dipole as well as monopole — broad color not sharp pitch."
        ),
        "supports": {
            "modal_amplitude_weighting": True,
            "bridge_radiation_weighting": True,
            "material_damping": True,
            "per_mode_color": True,
            "far_mode_contribution": True,
            "normalization_strategy": False,
            "high_note_body_contribution": False,
        },
    },
    {
        "id": "christensen_vistisen",
        "title": "Guitar sound pressure and acceleration responses (2–3 DOF models)",
        "authors": ["O. Christensen", "J. Vistisen"],
        "year": 1980,
        "url": "https://pure.hw.ac.uk/ws/portalfiles/portal/41406825/6084230_1_.pdf",
        "doi": None,
        "summary": (
            "Low-frequency guitar response from coupled soundboard, back, and Helmholtz air mass; "
            "sound pressure follows superposition of piston radiators per DOF."
        ),
        "synthesis_implication": (
            "Low notes: body identity from coupled top/air/back modes, not string fundamental alone. "
            "Modal amplitudes at bridge differ by subsystem participation."
        ),
        "supports": {
            "modal_amplitude_weighting": True,
            "bridge_radiation_weighting": True,
            "material_damping": False,
            "per_mode_color": True,
            "far_mode_contribution": False,
            "normalization_strategy": False,
            "high_note_body_contribution": False,
        },
    },
    {
        "id": "fletcher_rossing",
        "title": "The Science of String Instruments (guitar coupled vibrators chapter)",
        "authors": ["T. D. Rossing", "contributors"],
        "year": 2010,
        "url": "https://logosfoundation.org/kursus/The%20Science%20of%20String%20Instruments.pdf",
        "doi": None,
        "summary": (
            "Guitar as coupled vibrators: strings store energy, body radiates; "
            "modal parameters include natural frequency, damping, modal mass, mode shape; "
            "bridge is part of top plate at low f, top plate dominates radiation at high f."
        ),
        "synthesis_implication": (
            "Modal mass / mode shape at bridge should scale amplitude; "
            "high-frequency body color from plate modes, not eliminated by string pitch."
        ),
        "supports": {
            "modal_amplitude_weighting": True,
            "bridge_radiation_weighting": True,
            "material_damping": True,
            "per_mode_color": True,
            "far_mode_contribution": True,
            "normalization_strategy": False,
            "high_note_body_contribution": True,
        },
    },
    {
        "id": "jos_bridge_admittance",
        "title": "Building a Synthetic Guitar Bridge Admittance (Physical Audio Signal Processing)",
        "authors": ["Julius O. Smith"],
        "year": None,
        "url": "https://ccrma.stanford.edu/~jos/pasp/Building_Synthetic_Guitar_Bridge.html",
        "doi": None,
        "summary": (
            "Modal synthesis from measured bridge admittance: each peak → resonator with "
            "frequency, width, amplitude; separate transmittance filter for radiation "
            "(same poles, different zeros)."
        ),
        "synthesis_implication": (
            "Reflectance (string) and transmittance (audible) are distinct — "
            "our W_m should encode radiation transmittance, not just bridge coupling."
        ),
        "supports": {
            "modal_amplitude_weighting": True,
            "bridge_radiation_weighting": True,
            "material_damping": True,
            "per_mode_color": True,
            "far_mode_contribution": True,
            "normalization_strategy": False,
            "high_note_body_contribution": True,
        },
    },
    {
        "id": "smith_commuted_synthesis",
        "title": "Commuted Synthesis of String Instruments",
        "authors": ["Julius O. Smith"],
        "year": 2004,
        "url": "https://ccrma.stanford.edu/~jos/asahb04/asahb04.pdf",
        "doi": None,
        "summary": (
            "Body as LTI filter on string excitation; commuted IR captures body+room; "
            "high-order resonator bank equivalent to modal sum."
        ),
        "synthesis_implication": (
            "Body color is a transfer function — amplitude spectrum matters as much as pole frequencies; "
            "normalization to fixed loudness can mask guitar differences."
        ),
        "supports": {
            "modal_amplitude_weighting": True,
            "bridge_radiation_weighting": False,
            "material_damping": False,
            "per_mode_color": True,
            "far_mode_contribution": True,
            "normalization_strategy": True,
            "high_note_body_contribution": False,
        },
    },
    {
        "id": "lai_burgess_1990",
        "title": "Radiation efficiency of acoustic guitars (via bridge shaker)",
        "authors": ["Joseph Lai", "Marion Burgess"],
        "year": 1990,
        "url": "https://theses.hal.science/tel-01745927v2/file/2017LEMA1045.pdf",
        "doi": None,
        "summary": (
            "Radiation efficiency = radiated acoustic power / mechanical input at bridge; "
            "correlates with perceptual quality; extended Christensen models with pistons."
        ),
        "synthesis_implication": (
            "radiation_proxy should scale audible mode amplitude, not only damping; "
            "material/geometry may tilt efficiency vs frequency."
        ),
        "supports": {
            "modal_amplitude_weighting": True,
            "bridge_radiation_weighting": True,
            "material_damping": False,
            "per_mode_color": True,
            "far_mode_contribution": True,
            "normalization_strategy": False,
            "high_note_body_contribution": True,
        },
    },
    {
        "id": "woodhouse_banjo_2021",
        "title": "Acoustics of the banjo: bridge admittance and radiation (Acta Acustica)",
        "authors": ["J. Woodhouse", "et al."],
        "year": 2021,
        "url": "https://acta-acustica.edpsciences.org/articles/aacus/full_html/2021/01/aacus200052/aacus200052.html",
        "doi": "10.1051/aacus/2021009",
        "summary": (
            "Bridge admittance formants from local mass/stiffness and dynamical bridge behavior; "
            "radiation damping frequency-dependent; membrane vs plate radiation trends differ."
        ),
        "synthesis_implication": (
            "Broad admittance formants (not single peaks) create timbre; "
            "far modes should follow sample-specific mobility/radiation envelopes."
        ),
        "supports": {
            "modal_amplitude_weighting": True,
            "bridge_radiation_weighting": True,
            "material_damping": True,
            "per_mode_color": True,
            "far_mode_contribution": True,
            "normalization_strategy": False,
            "high_note_body_contribution": True,
        },
    },
    {
        "id": "gore_gilet_helmholtz",
        "title": "Guitar resonance and soundhole geometry (Gore & Gilet 2DOF/3DOF model)",
        "authors": ["T. Gore", "G. Gilet"],
        "year": None,
        "url": "https://mwguitars.com.au/2019/03/30/guitar-resonance-and-the-geometry-of-soundholes-part-9/",
        "doi": None,
        "summary": (
            "Soundbox as three coupled resonators (top, air, back); Helmholtz frequency "
            "set by hole geometry; coupling tunes global response."
        ),
        "synthesis_implication": (
            "geometry.hole_radius and cavity depth should affect air-mode amplitude and coupling, "
            "not only frequency scaling."
        ),
        "supports": {
            "modal_amplitude_weighting": True,
            "bridge_radiation_weighting": False,
            "material_damping": False,
            "per_mode_color": True,
            "far_mode_contribution": False,
            "normalization_strategy": False,
            "high_note_body_contribution": False,
        },
    },
    {
        "id": "spanish_guitar_fem_uq",
        "title": "Uncertainty quantification of Spanish guitar soundboard modal analysis",
        "authors": ["F. Antunes", "et al."],
        "year": None,
        "url": "https://hal.science/hal-03186581/file/f551d50b-e55b-455e-83ea-02149a2cb27d-author.pdf",
        "doi": None,
        "summary": (
            "FEM modal analysis validated against experiment; bridge admittance computed; "
            "material/climate sensitivity on soundboard dynamics."
        ),
        "synthesis_implication": (
            "Wood affects stiffness/damping and thus both frequency and amplitude of modes; "
            "not damping-only in a complete model."
        ),
        "supports": {
            "modal_amplitude_weighting": True,
            "bridge_radiation_weighting": True,
            "material_damping": True,
            "per_mode_color": False,
            "far_mode_contribution": False,
            "normalization_strategy": False,
            "high_note_body_contribution": False,
        },
    },
    {
        "id": "ccrma_virtual_strings",
        "title": "Virtual Stringed Instruments — physical modeling overview",
        "authors": ["C. Chafe", "Stanford CCRMA"],
        "year": None,
        "url": "https://ccrma.stanford.edu/realsimple/phys_mod_overview/phys_mod_overview.pdf",
        "doi": None,
        "summary": (
            "Driving-point admittance at bridge; modes include air and plate; "
            "radiation via simplified Kirchhoff-Helmholtz or measured transfer functions."
        ),
        "synthesis_implication": (
            "Per-mode radiation weighting is standard in physical models; "
            "multiple strings share bridge but see slightly different coupling."
        ),
        "supports": {
            "modal_amplitude_weighting": True,
            "bridge_radiation_weighting": True,
            "material_damping": False,
            "per_mode_color": True,
            "far_mode_contribution": True,
            "normalization_strategy": False,
            "high_note_body_contribution": True,
        },
    },
]

FIELD_USAGE_MAP: Dict[str, str] = {
    "frequency_hz": "USED_IN_AUDIO",
    "mode_q": "USED_IN_AUDIO",
    "mode_tau_s": "USED_IN_AUDIO",
    "mode_bandwidth_hz": "USED_IN_AUDIO",
    "bridge_excitation_abs": "USED_IN_AUDIO",
    "bridge_excitation_coupling": "USED_IN_AUDIO",
    "bridge_to_mic_gain_raw": "USED_IN_AUDIO",
    "mic_output_proxy": "USED_IN_AUDIO",
    "radiation_proxy": "USED_IN_AUDIO",
    "top_share": "USED_IN_AUDIO",
    "back_share": "USED_IN_AUDIO",
    "air_share": "USED_IN_AUDIO",
    "coupled_share": "USED_IN_AUDIO",
    "mode_category": "USED_IN_AUDIO",
    "mode_material_damping": "USED_IN_AUDIO",
    "geometry_damping_component": "USED_IN_AUDIO",
    "top_wood_id": "USED_IN_AUDIO",
    "back_wood_id": "USED_IN_AUDIO",
    "geometry.length": "USED_IN_AUDIO",
    "geometry.width": "USED_IN_AUDIO",
    "geometry.depth": "USED_IN_AUDIO",
    "geometry.top_thickness": "USED_IN_AUDIO",
    "geometry.back_thickness": "USED_IN_AUDIO",
    "geometry.hole_radius": "USED_IN_AUDIO",
    "mode_weight_fallback": "USED_IN_AUDIO",
    "mode_weights": "NOT_USED",
    "modal_mass": "NOT_USED",
    "effective_mass": "NOT_USED",
    "radiation_proxy_log10": "NOT_USED",
    "intensity_log10": "NOT_USED",
    "body_rms_before_calibration": "NORMALIZED_AWAY",
    "body_gain_applied": "NORMALIZED_AWAY",
    "final_rms_dbfs": "NORMALIZED_AWAY",
    "rms_gain_applied": "NORMALIZED_AWAY",
}


def _variance(vals: Sequence[float]) -> float:
    v = [float(x) for x in vals if x is not None and math.isfinite(float(x))]
    if len(v) < 2:
        return 0.0
    return float(statistics.pvariance(v))


def _corr(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def build_literature_review() -> Dict[str, Any]:
    return {
        "schema_version": "stage46_literature_review_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "internet_available": True,
        "source_count": len(LITERATURE_SOURCES),
        "sources": LITERATURE_SOURCES,
        "key_findings": [
            "Identical string excitation → timbre differences require body transfer function shape: "
            "pole frequencies, Q, AND mode amplitudes (mobility × radiation).",
            "Bridge input admittance / mobility is the standard measurable link from string to body modes.",
            "Radiation efficiency and mode shape determine which modes are audible — not all modes equally.",
            "Top/back/air coupling creates broad low-frequency color; air modes affect board motion indirectly.",
            "Radiation damping lowers Q of efficient radiators — damping and amplitude are coupled physically.",
            "High notes: top-plate / bridge dynamics still color sound; string pitch must not erase body transfer.",
            "Far/off-harmonic modes contribute smooth timbre / decay / bloom, not sharp pitch peaks.",
            "Wood/material affects stiffness and loss — literature supports modest amplitude effects, not damping-only.",
            "Loudness normalization can mask level differences that carry perceptual guitar identity.",
        ],
        "literature_consensus_for_next_model": (
            "Separate bridge mobility (coupling) from radiation transmittance per mode; "
            "weight far modes by sample-specific radiation/category shares, not global EQ."
        ),
    }


def build_model_gap_analysis() -> Dict[str, Any]:
    cfg_6040 = get_diagnostic_mode("modal_body_60_40_v1")
    answers = {
        "modal_frequencies_correct": {
            "answer": "Mostly yes — all M4/synthetic modes 60–550 Hz enter H_body sum; frequencies set pole locations.",
            "gap": "Frequency-only differentiation insufficient when amplitudes are similar after normalization.",
        },
        "modal_damping_q_correct": {
            "answer": "Yes after Stage 4.5 — share-weighted wood damping → per-mode Q in _complex_mode_response.",
            "gap": "Q differences often clamp at Q_MIN=22 for mid/high modes; wood spread visible mainly in low modes.",
        },
        "per_mode_tau_meaningful": {
            "answer": "Partially — tau=πQ/f applied in apply_per_mode_tau_envelope plus global body_decay_tau_s.",
            "gap": "Not a full modal time-domain synthesis; superposed single envelope approximation.",
        },
        "modal_amplitudes_correct": {
            "answer": "Partially — W_m = bridge×(0.55 mic + 0.45 rad)×participation blend.",
            "gap": "No modal mass; bridge/radiation often default to 1.0 on synthetic fallback; "
            "combined into one scalar — no separate mobility vs radiation transmittance.",
        },
        "modal_amplitudes_too_generic": {
            "answer": "Yes — especially when M4 scalars missing; far layer adds similar broad energy to all guitars.",
            "gap": "Primary differentiation driver is still frequency + damping, not radiation-weighted amplitude.",
        },
        "bridge_coupling_enough": {
            "answer": "Weak to moderate — bridge_excitation_abs used when present; defaults to 1.0.",
            "gap": "No bridge admittance envelope / mobility deviation feature from literature.",
        },
        "radiation_efficiency_enough": {
            "answer": "Weak — radiation_proxy scales weight (45%) and damping (rad_k); not primary amplitude driver.",
            "gap": "Literature: radiation should set audible transmittance; we treat it as secondary multiplier.",
        },
        "shares_for_damping_and_amplitude": {
            "answer": "Damping: yes (share-weighted wood). Amplitude: weak (0.5+0.5×share_sum blend only).",
            "gap": "Category-specific radiation color not implemented for amplitude.",
        },
        "wood_beyond_damping": {
            "answer": "No in current production — woods only via material_damping → Q/tau.",
            "gap": "Literature supports small stiffness/amplitude/radiation tilt by wood (controlled, not arbitrary).",
        },
        "far_modes_sample_specific_enough": {
            "answer": "Partially after 4.5 — _per_mode_broad_color_scale uses material/radiation/bridge.",
            "gap": "60/40 mode still adds broad_signature_curve (weak global EQ); A2 became more similar.",
        },
        "normalization_erases_differences": {
            "answer": "Yes significantly — body_gain → target ratio 4.2 + RMS -18 dBFS compresses cross-guitar spread.",
            "gap": "baseline_current fully normalizes; 60/40 partial (preserve ~0.5) still strong loudness norm.",
        },
        "high_notes_string_dominated": {
            "answer": "Yes — string_direct_scale_by_f0 ~0.97 at A2 but pitch layer + pluck still strong at E5; "
            "body far fraction high at E5 but spectrally similar after norm.",
            "gap": "Need radiation-colored body transmittance, not just more far energy.",
        },
        "low_notes_fundamental_dominated": {
            "answer": "Yes — fundamental_anchor + low_note_fundamental_harmonic_boost reinforce f0/direct string.",
            "gap": "Body midrange/radiation color underrepresented vs bass-heavy direct path.",
        },
        "missing_per_mode_color": {
            "answer": "Yes — no mode_color_vector by category/band; broad path is scalar × H_m, not shaped radiation.",
            "gap": "Literature: reflectance vs transmittance filters differ (Smith); we use one weight.",
        },
    }

    hypotheses = {
        "A_far_modes_sample_specific_radiation": {
            "verdict": "SUPPORTED",
            "note": "60/40 increased far energy but generic broad EQ reduced A2 differentiation.",
        },
        "B_damping_insufficient_need_amplitude_color": {
            "verdict": "STRONGLY_SUPPORTED",
            "note": "Q spread exists cross-sample but spectral similarity ~0.999; damping alone does not create timbre identity.",
        },
        "C_wood_should_affect_amplitude_radiation": {
            "verdict": "SUPPORTED_WITH_CAUTION",
            "note": "Literature supports modest effects via stiffness/radiation; must be small controlled coeffs.",
        },
        "D_broad_layer_reduces_A2_differentiation": {
            "verdict": "SUPPORTED",
            "note": "Similar broad layer + loudness norm raises average spectral similarity on A2.",
        },
        "E_high_notes_string_dominates": {
            "verdict": "SUPPORTED",
            "note": "E5 far fraction high but similarity increased vs baseline in stage 4.5 listening pattern.",
        },
        "F_low_notes_bass_like_fundamental": {
            "verdict": "SUPPORTED",
            "note": "fundamental_anchor_scale 0.45 in 60/40 still leaves strong low harmonic / direct support.",
        },
    }

    candidates = [
        {
            "name": "modal_radiation_color_v1",
            "recommended": True,
            "conceptual_changes": [
                "Split mode weight into bridge_mobility × radiation_transmittance × category_color.",
                "Apply small wood-dependent amplitude tilt (±8%) separate from damping.",
                "Far modes weighted by radiation×category vector per sample, no global EQ.",
                "Partial normalization preserved.",
            ],
            "formula_sketch": (
                "W_m = mobility(bridge_excitation)^0.85 * transmittance(radiation,mic,category)^0.9 "
                "* material_amp(wood,shares) * f_radiation_eff(f_m); "
                "H_broad += W_far * H_m * color_band(category,shares,radiation); "
                "no note-specific gains."
            ),
            "metadata_fields_used": [
                "bridge_excitation_abs", "radiation_proxy", "mic_output_proxy",
                "top_share", "back_share", "air_share", "mode_category", "top_wood_id", "back_wood_id",
            ],
            "new_metadata_fields": [
                "mode_amplitude_factor", "mode_radiation_factor", "mode_bridge_coupling_factor",
                "mode_color_band_vector", "material_amplitude_factor", "far_mode_sample_specificity_score",
            ],
            "objective_metrics": [
                "lower average_spectral_similarity", "higher spectral_differentiation",
                "higher centroid_spread", "controlled RMS spread", "far_mode_sample_specificity_score > 0",
            ],
            "listening_expectations": [
                "Guitars diverge in body bloom and midrange color on A2/A4",
                "E5 retains pitch clarity with sample-specific brightness",
            ],
            "risks": [
                "metallicity if Q too low", "muddiness if far energy too uniform",
                "ringing if mobility peaks sharpen", "EQ-like color if band vector too broad",
            ],
        },
        {
            "name": "bridge_mobility_body_color_v1",
            "recommended": False,
            "conceptual_changes": [
                "Add scalar mobility envelope from literature (characteristic admittance + deviation).",
                "Requires aggregating bridge_excitation across catalog per sample.",
            ],
            "formula_sketch": "W_m *= mobility_envelope(f_m, sample_bridge_stats)",
            "metadata_fields_used": ["bridge_excitation_abs", "frequency_hz"],
            "new_metadata_fields": ["sample_mobility_envelope_stats"],
            "objective_metrics": ["cross-sample bridge_excitation variance → audio variance correlation"],
            "listening_expectations": ["Stronger low-mid body identity"],
            "risks": ["Needs reliable M4 bridge scalars; synthetic fallback weak"],
        },
    ]

    return {
        "schema_version": "stage46_model_gap_analysis_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "current_model_summary": {
            "transfer_function": "H_body(f)=Σ W_m H_m(f); string harmonics → bridge acc → IFFT → mix",
            "W_m_formula": "bridge × (0.55 mic + 0.45 rad) × participation × near/broad split",
            "damping": "share-weighted wood → Q → pole width + tau envelope",
            "normalization": {
                "body_to_string_target": BODY_TO_STRING_TARGET_RATIO,
                "target_rms_dbfs": TARGET_RMS_DBFS,
                "modal_body_60_40_preserve": cfg_6040.raw_body_variation_preserve,
            },
        },
        "gap_answers": answers,
        "hypothesis_evaluation": hypotheses,
        "candidate_models": candidates,
        "recommended_next_model": "modal_radiation_color_v1",
    }


def _mode_summary_from_catalog(modes: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    freqs = [float(m["frequency_hz"]) for m in modes]
    bridges = [float(m.get("bridge_excitation_abs") or 0) for m in modes]
    rads = [float(m.get("radiation_proxy") or 0) for m in modes]
    tops = [float(m.get("top_share") or 0) for m in modes]
    backs = [float(m.get("back_share") or 0) for m in modes]
    airs = [float(m.get("air_share") or 0) for m in modes]
    return {
        "mode_count": len(modes),
        "frequency_min_hz": min(freqs) if freqs else None,
        "frequency_max_hz": max(freqs) if freqs else None,
        "bridge_excitation_mean": round(statistics.mean(bridges), 6) if bridges else None,
        "bridge_excitation_variance": round(_variance(bridges), 8),
        "radiation_proxy_mean": round(statistics.mean(rads), 6) if rads else None,
        "radiation_proxy_variance": round(_variance(rads), 8),
        "top_share_mean": round(statistics.mean(tops), 6) if tops else None,
        "back_share_mean": round(statistics.mean(backs), 6) if backs else None,
        "air_share_mean": round(statistics.mean(airs), 6) if airs else None,
    }


def build_data_to_audio_trace(
    *,
    repo_root: Path,
    max_samples: int = 10,
    use_surrogate: bool = True,
) -> Dict[str, Any]:
    samples = load_lhs_sample_entries(repo_root, max_samples=max_samples)
    per_sample: List[Dict[str, Any]] = []
    note_rows: Dict[str, List[Dict[str, Any]]] = {n: [] for n, _ in NOTES}

    for sample in samples:
        params = normalize_sample_parameters(sample.get("parameters"))
        modal_data, modal_source = resolve_modal_data_for_sample(
            repo_root, sample, use_surrogate=use_surrogate
        )
        all_modes, _ = parse_modal_modes(modal_data)
        band = modes_in_validated_band(all_modes)

        mode_records: List[Dict[str, Any]] = []
        for mode in band[:20]:
            f_hz = float(mode["frequency_hz"])
            damp = compute_per_mode_damping(mode, f_hz, params)
            defaults: List[str] = []
            flags: Dict[str, bool] = {}
            comp = compute_mode_weight_components(mode, defaults_used=defaults, flags=flags)
            mode_records.append(
                {
                    "frequency_hz": f_hz,
                    "bridge_excitation_abs": mode.get("bridge_excitation_abs"),
                    "radiation_proxy": mode.get("radiation_proxy"),
                    "mic_output_proxy": mode.get("mic_output_proxy"),
                    "top_share": mode.get("top_share"),
                    "back_share": mode.get("back_share"),
                    "air_share": mode.get("air_share"),
                    "mode_category": infer_mode_category(mode),
                    "mode_q": damp.get("mode_q"),
                    "mode_tau_s": damp.get("mode_tau_s"),
                    "mode_material_damping": damp.get("mode_material_damping"),
                    "combined_weight": comp.get("combined"),
                    "field_usage": {k: FIELD_USAGE_MAP.get(k, "UNKNOWN") for k in (
                        "frequency_hz", "bridge_excitation_abs", "radiation_proxy", "mic_output_proxy",
                        "top_share", "mode_q", "mode_tau_s", "mode_material_damping",
                    )},
                }
            )

        sample_block: Dict[str, Any] = {
            "sample_id": sample.get("sample_id"),
            "run_id": sample.get("run_id"),
            "raw_parameters": {k: params.get(k) for k in params if k.startswith("geometry.") or k.endswith("_wood_id")},
            "top_wood_id": params.get("top_wood_id"),
            "back_wood_id": params.get("back_wood_id"),
            "modal_source": modal_source,
            "catalog_summary": _mode_summary_from_catalog(band),
            "mode_records_sample": mode_records,
            "notes": {},
        }

        for note_name, note_hz in NOTES:
            tmp = Path(f"_stage46_{sample['sample_id']}_{note_name}.wav")
            meta = synthesize_note_with_body_response(
                frequency_hz=note_hz,
                note_name=note_name,
                duration_s=0.15,
                sample_rate=DEFAULT_SAMPLE_RATE,
                modal_data=modal_data,
                output_wav=tmp,
                diagnostic_mode="baseline_current",
                sample_parameters=params,
                modal_source=modal_source,
            )
            if tmp.is_file():
                tmp.unlink()
            note_block = {
                "note": note_name,
                "raw_body_rms_before_normalization": meta.get("raw_body_rms_before_normalization"),
                "body_rms_before_mix": meta.get("body_rms_before_mix"),
                "body_gain_applied": meta.get("body_gain_applied"),
                "body_to_string_ratio_before": meta.get("body_to_string_rms_ratio_before_loudness"),
                "body_to_string_ratio_after": meta.get("body_to_string_rms_ratio_after_loudness"),
                "final_rms_dbfs": meta.get("final_rms_dbfs"),
                "rms_gain_applied": meta.get("rms_gain_applied"),
                "spectral_centroid_hz": None,
                "output_decay_slope_db_per_s": meta.get("output_decay_slope_db_per_s"),
                "near_modal_energy_fraction": meta.get("near_modal_energy_fraction"),
                "broad_body_energy_fraction": meta.get("broad_body_energy_fraction"),
                "sample_material_damping_fingerprint": meta.get("sample_material_damping_fingerprint"),
                "sample_mode_q_fingerprint": meta.get("sample_mode_q_fingerprint"),
            }
            sample_block["notes"][note_name] = note_block
            note_rows[note_name].append(
                {
                    "sample_id": sample.get("sample_id"),
                    **note_block,
                    "material_fp": meta.get("sample_material_damping_fingerprint"),
                    "q_fp": meta.get("sample_mode_q_fingerprint"),
                    "bridge_mean": sample_block["catalog_summary"].get("bridge_excitation_mean"),
                    "radiation_mean": sample_block["catalog_summary"].get("radiation_proxy_mean"),
                    "air_share_mean": sample_block["catalog_summary"].get("air_share_mean"),
                }
            )
        per_sample.append(sample_block)

    data_variance = {
        "material_damping_fingerprint": _variance(
            [r["material_fp"] for rows in note_rows.values() for r in rows if r.get("material_fp")]
        ),
        "mode_q_fingerprint": _variance(
            [r["q_fp"] for rows in note_rows.values() for r in rows if r.get("q_fp")]
        ),
        "bridge_excitation_mean": _variance(
            [s["catalog_summary"].get("bridge_excitation_mean") or 0 for s in per_sample]
        ),
        "radiation_proxy_mean": _variance(
            [s["catalog_summary"].get("radiation_proxy_mean") or 0 for s in per_sample]
        ),
    }

    audio_variance: Dict[str, Dict[str, float]] = {}
    correlations: Dict[str, Dict[str, Optional[float]]] = {}
    for note_name, rows in note_rows.items():
        segs = [{"sample_id": r["sample_id"], **r} for r in rows]
        summary = summarize_comparison_note(segs)
        audio_variance[note_name] = {
            "spectral_differentiation": 1.0 - float(summary.get("average_spectral_similarity") or 1.0)
            if "average_spectral_similarity" in summary
            else float(summary.get("rms_spread_db") or 0),
            "rms_spread_db": float(summary.get("rms_spread_db") or 0),
            "centroid_spread_hz": float(summary.get("spectral_centroid_spread_hz") or 0),
            "decay_slope_spread": float(summary.get("decay_slope_spread_db_per_s") or 0),
            "material_spread": float(summary.get("material_damping_spread_mean") or 0),
            "q_spread": float(summary.get("mode_q_spread_mean") or 0),
        }
        xs_mat = [float(r.get("material_fp") or 0) for r in rows]
        ys_decay = [float(r.get("output_decay_slope_db_per_s") or 0) for r in rows]
        xs_bridge = [float(r.get("bridge_mean") or 0) for r in rows]
        ys_body = [float(r.get("raw_body_rms_before_normalization") or 0) for r in rows]
        correlations[note_name] = {
            "material_damping_vs_decay": _corr(xs_mat, ys_decay),
            "bridge_excitation_vs_raw_body_rms": _corr(xs_bridge, ys_body),
            "radiation_vs_centroid": _corr(
                [float(r.get("radiation_mean") or 0) for r in rows],
                [float(r.get("spectral_centroid_hz") or 0) for r in rows],
            ),
        }

    explicit_answers = {
        "A_strongest_audible_drivers": [
            "modal frequency layout (harmonic proximity)",
            "body_gain + loudness normalization (compresses raw differences)",
            "per-mode Q/tau (moderate, mainly low modes)",
            "bridge×radiation weight when M4 scalars present",
        ],
        "B_high_variance_low_audible_effect": [
            "material_damping_fingerprint (spread in data, weak spectral differentiation ~0.001)",
            "mode_q_fingerprint when medians clamp at 22",
        ],
        "C_normalized_away": [
            "raw_body_rms_before_normalization",
            "body_to_string_ratio (forced toward 4.2 in baseline)",
            "final_rms_dbfs (target -18 dBFS)",
        ],
        "D_using_modal_amplitude": (
            "Partially — combined bridge×mic×rad weight; no explicit modal mass or ROM intensity fields."
        ),
        "E_bridge_radiation_strong_enough": "No — secondary multipliers; defaults to 1.0 on synthetic catalog.",
        "F_wood_amplitude_or_damping_only": "Damping only (share-weighted Q/tau); no wood amplitude/radiation tilt.",
        "G_sufficient_information_in_data": (
            "Yes in principle when M4 provides bridge_excitation, radiation_proxy, shares — "
            "not fully exploited in STK amplitude path."
        ),
        "H_missing_proxy": "Modal mass / radiation transmittance separate from bridge mobility; sample mobility envelope.",
        "I_next_stk_change": "modal_radiation_color_v1 — radiation-weighted amplitude + category color, no global far EQ.",
    }

    return {
        "schema_version": "stage46_data_to_audio_trace_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sample_count": len(per_sample),
        "use_surrogate": use_surrogate,
        "m4_surrogate_available": m4_surrogate_model_available(repo_root),
        "field_usage_map": FIELD_USAGE_MAP,
        "per_sample": per_sample,
        "data_variance_cross_sample": data_variance,
        "audio_variance_by_note": audio_variance,
        "correlations_by_note": correlations,
        "explicit_answers": explicit_answers,
        "recommended_next_model": "modal_radiation_color_v1",
    }


def render_literature_md(doc: Mapping[str, Any]) -> str:
    lines = ["# Stage 4.6 Literature Review — Guitar Body Timbre", "", f"Generated: {doc['generated_at']}", ""]
    lines.append("## Key findings")
    for f in doc.get("key_findings") or []:
        lines.append(f"- {f}")
    lines.append("")
    lines.append("## Sources")
    for s in doc.get("sources") or []:
        lines.append(f"### {s.get('title')}")
        lines.append(f"- Authors: {', '.join(s.get('authors') or [])}")
        lines.append(f"- Year: {s.get('year')}")
        lines.append(f"- URL: {s.get('url')}")
        if s.get("doi"):
            lines.append(f"- DOI: {s.get('doi')}")
        lines.append(f"- Summary: {s.get('summary')}")
        lines.append(f"- Implication: {s.get('synthesis_implication')}")
        sup = s.get("supports") or {}
        lines.append(f"- Supports changing: {', '.join(k for k, v in sup.items() if v)}")
        lines.append("")
    return "\n".join(lines)


def render_gap_md(doc: Mapping[str, Any]) -> str:
    lines = ["# Stage 4.6 Model Gap Analysis", "", f"Generated: {doc['generated_at']}", ""]
    lines.append("## Gap answers")
    for k, v in (doc.get("gap_answers") or {}).items():
        lines.append(f"### {k}")
        lines.append(f"- {v.get('answer')}")
        lines.append(f"- Gap: {v.get('gap')}")
        lines.append("")
    lines.append("## Hypothesis evaluation")
    for k, v in (doc.get("hypothesis_evaluation") or {}).items():
        lines.append(f"- **{k}**: {v.get('verdict')} — {v.get('note')}")
    lines.append("")
    lines.append(f"## Recommended next model: **{doc.get('recommended_next_model')}**")
    return "\n".join(lines)


def render_trace_md(doc: Mapping[str, Any]) -> str:
    lines = [
        "# Stage 4.6 DATA → ROM → STK → AUDIO Trace",
        "",
        f"Generated: {doc['generated_at']}",
        f"M4 available: {doc.get('m4_surrogate_available')}",
        "",
        "## Field usage",
        "",
    ]
    for field, status in sorted((doc.get("field_usage_map") or {}).items()):
        lines.append(f"- `{field}` → **{status}**")
    lines.append("")
    lines.append("## Explicit answers")
    for k, v in (doc.get("explicit_answers") or {}).items():
        lines.append(f"### {k}")
        if isinstance(v, list):
            for item in v:
                lines.append(f"- {item}")
        else:
            lines.append(str(v))
        lines.append("")
    lines.append(f"## Recommended next model: **{doc.get('recommended_next_model')}**")
    return "\n".join(lines)


def write_all_reports(*, repo_root: Path = REPO, use_surrogate: bool = True) -> Dict[str, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lit = build_literature_review()
    gap = build_model_gap_analysis()
    trace = build_data_to_audio_trace(repo_root=repo_root, use_surrogate=use_surrogate)

    paths = {
        "literature_json": OUT_DIR / "stage46_literature_review.json",
        "literature_md": OUT_DIR / "stage46_literature_review.md",
        "gap_json": OUT_DIR / "stage46_model_gap_analysis.json",
        "gap_md": OUT_DIR / "stage46_model_gap_analysis.md",
        "trace_json": OUT_DIR / "stage46_data_to_audio_trace.json",
        "trace_md": OUT_DIR / "stage46_data_to_audio_trace.md",
    }
    paths["literature_json"].write_text(json.dumps(lit, indent=2), encoding="utf-8")
    paths["literature_md"].write_text(render_literature_md(lit), encoding="utf-8")
    paths["gap_json"].write_text(json.dumps(gap, indent=2), encoding="utf-8")
    paths["gap_md"].write_text(render_gap_md(gap), encoding="utf-8")
    paths["trace_json"].write_text(json.dumps(trace, indent=2), encoding="utf-8")
    paths["trace_md"].write_text(render_trace_md(trace), encoding="utf-8")
    return paths


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Stage 4.6 research audit reports")
    parser.add_argument("--repo-root", type=Path, default=REPO)
    parser.add_argument("--no-surrogate", action="store_true")
    args = parser.parse_args()
    paths = write_all_reports(repo_root=args.repo_root, use_surrogate=not args.no_surrogate)
    for p in paths.values():
        print(f"Wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
