#!/usr/bin/env python3
"""Stage 5.0 STK V4 hybrid body-transfer report."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


def _diff(mode_summaries: Mapping[str, Any], mode: str, note: str) -> float:
    notes = (mode_summaries.get(mode) or {}).get("notes") or {}
    return float((notes.get(note) or {}).get("spectral_differentiation") or 0.0)


def _rms_spread(mode_summaries: Mapping[str, Any], mode: str, note: str) -> float:
    notes = (mode_summaries.get(mode) or {}).get("notes") or {}
    return float((notes.get(note) or {}).get("rms_spread_db") or 0.0)


V4_ABLATIONS = (
    "modal_body_hybrid_v4_core",
    "modal_body_hybrid_v4_contrast_imprint_only",
    "modal_body_hybrid_v4_contrast_body_layer_only",
    "modal_body_hybrid_v4_mobility_light_only",
    "modal_body_hybrid_v4_full",
)


def build_stage50_report(
    *,
    mode_summaries: Mapping[str, Any],
    notes: Sequence[str],
    modes: Sequence[str],
    build_manifest: Mapping[str, Any],
    out_json: Path,
    out_md: Path,
) -> Dict[str, Any]:
    base = "baseline_current"
    v1 = "modal_radiation_color_v1"
    v3 = "modal_body_signature_v3_full"
    v4 = "modal_body_hybrid_v4_full"

    comparison: Dict[str, Dict[str, float]] = {}
    for note in notes:
        comparison[note] = {
            "baseline": _diff(mode_summaries, base, note),
            "radiation_v1": _diff(mode_summaries, v1, note),
            "v3_full": _diff(mode_summaries, v3, note) if v3 in mode_summaries else 0.0,
            "v4_full": _diff(mode_summaries, v4, note),
        }
        for ab in V4_ABLATIONS:
            if ab in mode_summaries:
                comparison[note][ab] = _diff(mode_summaries, ab, note)

    ablation_table = {
        ab: {note: _diff(mode_summaries, ab, note) for note in notes}
        for ab in V4_ABLATIONS
        if ab in mode_summaries
    }

    gates: Dict[str, Any] = {}
    for note in notes:
        v4d = _diff(mode_summaries, v4, note)
        v1d = _diff(mode_summaries, v1, note)
        based = _diff(mode_summaries, base, note)
        rms = _rms_spread(mode_summaries, v4, note)
        if note == "A2":
            passed = v4d >= based - 1e-6 and v4d >= v1d - 1e-6
            rule = "v4 >= baseline and v4 >= radiation_v1"
        else:
            passed = v4d >= 0.95 * v1d and v4d >= based - 1e-6
            rule = "v4 >= 0.95*v1 and v4 >= baseline"
        gates[note] = {
            "v4_spectral_differentiation": v4d,
            "v1_spectral_differentiation": v1d,
            "baseline_spectral_differentiation": based,
            "rms_spread_db": rms,
            "passed": passed,
            "rms_ok": rms <= 4.0,
            "rule": rule,
        }

    imprint_a2 = _diff(mode_summaries, "modal_body_hybrid_v4_contrast_imprint_only", "A2") if "A2" in notes else 0.0
    core_a2 = _diff(mode_summaries, "modal_body_hybrid_v4_core", "A2") if "A2" in notes else 0.0
    mob_a2 = _diff(mode_summaries, "modal_body_hybrid_v4_mobility_light_only", "A2") if "A2" in notes else 0.0
    far_a2 = _diff(mode_summaries, "modal_body_hybrid_v4_contrast_body_layer_only", "A2") if "A2" in notes else 0.0

    report: Dict[str, Any] = {
        "stage": "5.0",
        "title": "STK V4 hybrid body-transfer model",
        "formula": {
            "hybrid": "y_core = (1-w_rad)*baseline_current + w_rad*modal_radiation_color_v1",
            "w_rad": "smoothstep(160 Hz, 320 Hz, f0)",
            "contrast": "D_sample = clip(logG_sample - logG_ref, +/-Dmax)",
            "harmonic_gain_k": "exp(alpha(f0)*beta_k*D(f_k))",
            "body_layer": "~-21 dB contrast-filtered string residual, fades at high f0",
        },
        "starts_from_baseline_and_v1_not_v3_v2": True,
        "f0_continuous_no_note_names": True,
        "modes_run": list(modes),
        "notes": list(notes),
        "build_manifest": dict(build_manifest),
        "ablation_table": ablation_table,
        "per_note_comparison": comparison,
        "acceptance_gates": gates,
        "analysis": {
            "contrast_imprint_helps_a2": imprint_a2 > core_a2 if "A2" in notes else None,
            "mobility_helps_a2": mob_a2 > core_a2 if "A2" in notes else None,
            "body_layer_helps_a2": far_a2 > core_a2 if "A2" in notes else None,
            "a4_preserves_v1": gates.get("A4", {}).get("passed"),
            "e5_preserves_v1": gates.get("E5", {}).get("passed"),
            "v4_beats_v1_all_notes": all(
                _diff(mode_summaries, v4, n) >= _diff(mode_summaries, v1, n) for n in notes
            ),
            "v4_beats_baseline_all_notes": all(
                _diff(mode_summaries, v4, n) >= _diff(mode_summaries, base, n) for n in notes
            ),
            "promotion_recommendation": "DIAGNOSTIC_ONLY",
        },
        "fem_launched": False,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(_render_md(report), encoding="utf-8")
    return report


def _render_md(report: Mapping[str, Any]) -> str:
    lines = [
        "# Stage 5.0 — STK V4 hybrid body-transfer report",
        "",
        "## Formula",
        f"- Hybrid: `{report.get('formula', {}).get('hybrid')}`",
        f"- w_rad: `{report.get('formula', {}).get('w_rad')}`",
        "",
        "## Acceptance gates",
    ]
    for note, gate in (report.get("acceptance_gates") or {}).items():
        lines.append(f"- **{note}**: passed={gate.get('passed')}, v4={gate.get('v4_spectral_differentiation')}")
    lines.append("")
    lines.append(f"Promotion: **{(report.get('analysis') or {}).get('promotion_recommendation')}**")
    lines.append("FEM launched: **no**")
    return "\n".join(lines)
