#!/usr/bin/env python3
"""
Stage 4.4 synthesis implementation audit — documents what enters body-response STK.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gui"))

from body_response_synth import (  # noqa: E402
    BODY_TO_STRING_TARGET_RATIO,
    BROAD_MODE_MIN_FRACTION,
    FULL_MODAL_BAND_HZ,
    LOW_NOTE_FUNDAMENTAL_HARMONIC_BOOST,
    NEAR_HARMONIC_WINDOW_REL,
    PRODUCTION_BROAD_ALL_MODE_STRENGTH,
    TARGET_RMS_DBFS,
    _combine_near_and_broad_weights,
    _harmonic_proximity,
    harmonic_series,
    modes_in_validated_band,
    synthetic_classic_body_modes,
)
from diagnostic_synthesis import get_diagnostic_mode, use_diagnostic_mode  # noqa: E402
from modal_damping import (  # noqa: E402
    WOOD_DAMPING_COEFF,
    AIR_DAMPING_COEFF,
    COUPLED_DAMPING_COEFF,
    compute_per_mode_damping,
    list_wood_damping_constants,
)
from string_body_balance import (  # noqa: E402
    body_color_gain_by_f0,
    pitch_layer_scale_by_f0,
    string_direct_scale_by_f0,
)


def measure_modal_contribution_split(
    band_modes: Sequence[Mapping[str, Any]],
    note_hz: float,
    *,
    diagnostic_mode: Optional[str] = None,
) -> Dict[str, Any]:
    harmonics_hz, _ = harmonic_series(note_hz, 44100)
    near_e = mid_e = far_e = 0.0
    near_count = mid_count = far_count = 0
    with use_diagnostic_mode(diagnostic_mode):
        for mode in band_modes:
            f_m = float(mode["frequency_hz"])
            prox = _harmonic_proximity(f_m, harmonics_hz)
            w = float(mode.get("bridge_excitation_abs") or 0.01)
            combined, near_frac, broad_frac = _combine_near_and_broad_weights(w, f_m, harmonics_hz)
            near_part = combined * near_frac
            broad_part = combined * broad_frac
            if prox >= 0.55:
                near_e += near_part + broad_part * 0.25
                near_count += 1
            elif prox >= 0.15:
                mid_e += near_part + broad_part
                mid_count += 1
            else:
                far_e += near_part * 0.35 + broad_part
                far_count += 1
    total = near_e + mid_e + far_e
    return {
        "note_hz": note_hz,
        "diagnostic_mode": diagnostic_mode or "production",
        "mode_count": len(band_modes),
        "near_energy_fraction": round(near_e / max(total, 1e-12), 4),
        "mid_energy_fraction": round(mid_e / max(total, 1e-12), 4),
        "far_energy_fraction": round(far_e / max(total, 1e-12), 4),
        "near_mode_count": near_count,
        "mid_mode_count": mid_count,
        "far_mode_count": far_count,
    }


def compare_wood_damping_same_geometry() -> Dict[str, Any]:
    mode = synthetic_classic_body_modes(1)[0]
    mode["top_share"] = 0.50
    mode["back_share"] = 0.50
    mode["air_share"] = 0.0
    geom = {"geometry": {"length": 0.48, "width": 0.37, "depth": 0.10, "top_thickness": 0.003}}
    pairs = []
    for top, back in (("spruce", "mahogany"), ("cedar", "rosewood"), ("maple", "rosewood")):
        params = {**geom, "top_wood_id": top, "back_wood_id": back}
        rec = compute_per_mode_damping(mode, float(mode["frequency_hz"]), params)
        pairs.append(
            {
                "top_wood": top,
                "back_wood": back,
                "mode_q": rec["mode_q"],
                "mode_tau_s": rec["mode_tau_s"],
                "mode_material_damping": rec["mode_material_damping"],
            }
        )
    return {"mixed_50_50_top_back_modes": pairs}


def build_audit_report() -> Dict[str, Any]:
    modes = synthetic_classic_body_modes(55)
    band = modes_in_validated_band(modes)
    notes = (("A2", 110.0), ("A4", 440.0), ("E5", 659.25))

    contribution_by_mode: Dict[str, Any] = {}
    for label, diag in (
        ("production", None),
        ("baseline_current", "baseline_current"),
        ("modal_damping_body_signature_v1", "modal_damping_body_signature_v1"),
        ("modal_body_60_40_v1", "modal_body_60_40_v1"),
    ):
        contribution_by_mode[label] = {
            note: measure_modal_contribution_split(band, hz, diagnostic_mode=diag)
            for note, hz in notes
        }

    report: Dict[str, Any] = {
        "schema_version": "stage44_synthesis_audit_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "modal_usage": {
            "validated_band_hz": list(FULL_MODAL_BAND_HZ),
            "modes_evaluated_per_note": len(band),
            "all_modes_in_band_used": True,
            "subset_pruning": "none — all modes 60–550 Hz enter H_body sum",
            "constants": {
                "near_modal_boost_production": 1.0,
                "broad_all_mode_strength_production": PRODUCTION_BROAD_ALL_MODE_STRENGTH,
                "broad_mode_min_fraction": BROAD_MODE_MIN_FRACTION,
                "near_harmonic_window_rel": NEAR_HARMONIC_WINDOW_REL,
                "far_fraction_formula": "1 - harmonic_proximity",
                "near_weight_formula": "w * (1 + (near_boost-1) * proximity)",
                "broad_weight_formula": "w * (min_fraction + broad_strength * (0.22 + 0.78*far)) * far_color_gain",
            },
            "contribution_split_by_note": contribution_by_mode,
        },
        "damping_q_usage": {
            "per_mode_damping_in_production": True,
            "per_mode_damping_in_generate_sound": True,
            "per_mode_damping_diagnostics_only": False,
            "mode_q_affects_bandwidth": True,
            "mode_q_mechanism": "H_m(f) complex pole width via _complex_mode_response(Q)",
            "mode_tau_affects_temporal_decay": False,
            "mode_tau_note": (
                "mode_tau_s is derived from Q and frequency (tau=pi*Q/f) and stored in metadata; "
                "temporal decay uses a global body_decay_tau_s envelope after IFFT, not per-mode tau."
            ),
            "metadata_per_note": [
                "per_mode_damping[] with mode_q, mode_tau_s, mode_bandwidth_hz",
                "damping_q_summary aggregates",
                "material damping components per mode (stage 4.4+)",
            ],
        },
        "material_wood_usage": {
            "stage43_issue": (
                "Pre-4.4: wood affected damping via global category scale on entire guitar, "
                "not share-weighted per mode — identical geometry + different woods had weak Q spread."
            ),
            "stage44_fix": (
                "Share-weighted: top_share*damping(top_wood) + back_share*damping(back_wood) "
                "+ air_share*AIR + coupled_share*COUPLED per mode."
            ),
            "wood_damping_coefficients": list_wood_damping_constants(),
            "air_damping_coefficient": AIR_DAMPING_COEFF,
            "coupled_damping_coefficient": COUPLED_DAMPING_COEFF,
            "identical_geometry_different_woods": compare_wood_damping_same_geometry(),
        },
        "modal_participation": {
            "fields_used": [
                "top_share", "back_share", "air_share",
                "bridge_excitation_abs", "mic_output_proxy", "radiation_proxy",
            ],
            "weight_effect": "bridge * (0.55*mic + 0.45*rad) * participation blend",
            "damping_effect": "share-weighted material damping scales inv_q per mode",
            "tau_effect": "derived from final Q — metadata only for temporal envelope",
            "radiation_color": "radiation_proxy in damping + broad signature EQ bands",
        },
        "normalization": {
            "body_gain_calibration": f"body_gain = target_ratio * string_rms/body_rms (target={BODY_TO_STRING_TARGET_RATIO})",
            "body_string_target_ratio": BODY_TO_STRING_TARGET_RATIO,
            "final_loudness_rms_target_dbfs": TARGET_RMS_DBFS,
            "limiter_ceiling": "tanh soft limiter + peak ceiling -1 dBFS",
            "masks_differences": (
                "Full calibration + loudness normalize compress guitar-to-guitar level differences; "
                "diagnostic modes reduce via body_gain_normalization_strength and "
                "final_loudness_normalization_strength."
            ),
        },
        "string_body_balance_f0": {
            "string_direct_scale_by_f0_A2": round(string_direct_scale_by_f0(110.0), 4),
            "string_direct_scale_by_f0_A4": round(string_direct_scale_by_f0(440.0), 4),
            "string_direct_scale_by_f0_E5": round(string_direct_scale_by_f0(659.25), 4),
            "pitch_layer_scale_by_f0_E5": round(pitch_layer_scale_by_f0(659.25), 4),
            "body_color_gain_by_f0_E5": round(body_color_gain_by_f0(659.25), 4),
            "low_note_fundamental_harmonic_boost": LOW_NOTE_FUNDAMENTAL_HARMONIC_BOOST,
        },
        "diagnostic_modes_snapshot": {
            name: get_diagnostic_mode(name).to_metadata_dict()
            for name in (
                "baseline_current",
                "modal_damping_body_signature_v1",
                "modal_body_60_40_v1",
            )
        },
    }
    return report


def render_audit_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Stage 4.4 Synthesis Implementation Audit",
        "",
        f"Generated: {report.get('generated_at', '')}",
        "",
        "## 1. Modal usage",
        "",
        f"- Modes evaluated per note: **{report['modal_usage']['modes_evaluated_per_note']}** (all in {report['modal_usage']['validated_band_hz']} Hz)",
        f"- Near harmonic window (rel): `{report['modal_usage']['constants']['near_harmonic_window_rel']}`",
        f"- Production broad strength: `{report['modal_usage']['constants']['broad_all_mode_strength_production']}`",
        f"- Broad min fraction: `{report['modal_usage']['constants']['broad_mode_min_fraction']}`",
        "",
        "### Contribution split (production)",
        "",
    ]
    prod = report["modal_usage"]["contribution_split_by_note"].get("production", {})
    for note, row in prod.items():
        lines.append(
            f"- **{note}**: near {row['near_energy_fraction']:.0%}, "
            f"mid {row['mid_energy_fraction']:.0%}, far {row['far_energy_fraction']:.0%}"
        )
    lines.extend(
        [
            "",
            "## 2. Damping / Q",
            "",
            f"- Per-mode damping in production: **{report['damping_q_usage']['per_mode_damping_in_production']}**",
            f"- mode_q affects resonance width: **{report['damping_q_usage']['mode_q_affects_bandwidth']}**",
            f"- mode_tau affects temporal decay: **{report['damping_q_usage']['mode_tau_affects_temporal_decay']}**",
            f"- Note: {report['damping_q_usage']['mode_tau_note']}",
            "",
            "## 3. Material / wood",
            "",
            f"- Stage 4.3 issue: {report['material_wood_usage']['stage43_issue']}",
            f"- Stage 4.4 fix: {report['material_wood_usage']['stage44_fix']}",
            "",
            "### Wood damping coefficients (relative, spruce=1.0)",
            "",
        ]
    )
    for wood, coeff in report["material_wood_usage"]["wood_damping_coefficients"].items():
        lines.append(f"- {wood}: `{coeff}`")
    lines.append("")
    lines.append("### Same geometry, different woods (50% top / 50% back mode)")
    for row in report["material_wood_usage"]["identical_geometry_different_woods"]["mixed_50_50_top_back_modes"]:
        lines.append(
            f"- {row['top_wood']}/{row['back_wood']}: Q={row['mode_q']}, "
            f"material_damping={row['mode_material_damping']}"
        )
    lines.extend(
        [
            "",
            "## 4. Normalization",
            "",
            f"- {report['normalization']['body_gain_calibration']}",
            f"- Final RMS target: {report['normalization']['final_loudness_rms_target_dbfs']} dBFS",
            f"- Masking risk: {report['normalization']['masks_differences']}",
            "",
        ]
    )
    return "\n".join(lines)


def write_audit_reports(out_dir: Path) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    report = build_audit_report()
    json_path = out_dir / "stage44_current_synthesis_audit.json"
    md_path = out_dir / "stage44_current_synthesis_audit.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_audit_markdown(report), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}


def main() -> int:
    out = REPO / "audio" / "debug_reports"
    paths = write_audit_reports(out)
    print(f"Wrote {paths['json']}")
    print(f"Wrote {paths['markdown']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
