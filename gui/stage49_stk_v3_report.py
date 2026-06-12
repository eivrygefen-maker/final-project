#!/usr/bin/env python3
"""Stage 4.9 STK V3 body-signature diagnostic report."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

V3_MODES = (
    "modal_body_signature_v3_core",
    "modal_body_signature_v3_low_f0_imprint_only",
    "modal_body_signature_v3_mobility_only",
    "modal_body_signature_v3_far_color_only",
    "modal_body_signature_v3_full",
)

REFERENCE_MODES = ("baseline_current", "modal_radiation_color_v1")


def _note_diff(mode_summaries: Mapping[str, Any], mode: str, note: str) -> float:
    notes = (mode_summaries.get(mode) or {}).get("notes") or {}
    return float((notes.get(note) or {}).get("spectral_differentiation") or 0.0)


def _note_rms_spread(mode_summaries: Mapping[str, Any], mode: str, note: str) -> float:
    notes = (mode_summaries.get(mode) or {}).get("notes") or {}
    return float((notes.get(note) or {}).get("rms_spread_db") or 0.0)


def evaluate_acceptance(
    mode_summaries: Mapping[str, Any],
    *,
    notes: Sequence[str],
) -> Dict[str, Any]:
    v3 = "modal_body_signature_v3_full"
    v1 = "modal_radiation_color_v1"
    base = "baseline_current"
    gates: Dict[str, Any] = {}
    for note in notes:
        v3_d = _note_diff(mode_summaries, v3, note)
        v1_d = _note_diff(mode_summaries, v1, note)
        base_d = _note_diff(mode_summaries, base, note)
        rms = _note_rms_spread(mode_summaries, v3, note)
        if note == "A2":
            passed = v3_d >= v1_d - 1e-6 or v3_d >= base_d
            gates[note] = {
                "v3_spectral_differentiation": v3_d,
                "v1_spectral_differentiation": v1_d,
                "baseline_spectral_differentiation": base_d,
                "rms_spread_db": rms,
                "passed": passed,
                "rule": "v3 >= v1 or v3 >= baseline",
            }
        else:
            passed = v3_d >= 0.90 * v1_d and v3_d >= base_d - 1e-6
            gates[note] = {
                "v3_spectral_differentiation": v3_d,
                "v1_spectral_differentiation": v1_d,
                "baseline_spectral_differentiation": base_d,
                "rms_spread_db": rms,
                "passed": passed,
                "rule": "v3 >= 0.90*v1 and v3 >= baseline",
            }
        gates[note]["rms_ok"] = rms <= 4.0
    return gates


def build_stage49_report(
    *,
    mode_summaries: Mapping[str, Any],
    notes: Sequence[str],
    modes: Sequence[str],
    build_manifest: Mapping[str, Any],
    out_json: Path,
    out_md: Path,
) -> Dict[str, Any]:
    v3_full = "modal_body_signature_v3_full"
    v1 = "modal_radiation_color_v1"
    base = "baseline_current"

    ablation_table: Dict[str, Dict[str, float]] = {}
    for mode in modes:
        if mode not in V3_MODES and mode not in ("modal_body_signature_v3", "modal_radiation_color_v3"):
            continue
        ablation_table[mode] = {note: _note_diff(mode_summaries, mode, note) for note in notes}

    comparison: Dict[str, Dict[str, float]] = {}
    for note in notes:
        comparison[note] = {
            "baseline": _note_diff(mode_summaries, base, note),
            "radiation_v1": _note_diff(mode_summaries, v1, note),
            "v3_full": _note_diff(mode_summaries, v3_full, note),
        }
        for mode in V3_MODES:
            comparison[note][mode] = _note_diff(mode_summaries, mode, note)

    gates = evaluate_acceptance(mode_summaries, notes=notes)
    a2_imprint = _note_diff(mode_summaries, "modal_body_signature_v3_low_f0_imprint_only", "A2") if "A2" in notes else 0.0
    a2_mob = _note_diff(mode_summaries, "modal_body_signature_v3_mobility_only", "A2") if "A2" in notes else 0.0
    a2_far = _note_diff(mode_summaries, "modal_body_signature_v3_far_color_only", "A2") if "A2" in notes else 0.0
    v3_better_than_v1 = all(
        _note_diff(mode_summaries, v3_full, n) >= _note_diff(mode_summaries, v1, n) for n in notes
    )
    a2_v1 = comparison.get("A2", {}).get("radiation_v1", 0.0)

    report: Dict[str, Any] = {
        "stage": "4.9",
        "title": "STK V3 body-signature model",
        "implemented": {
            "base": "modal_radiation_color_v1 (not v2)",
            "components": [
                "continuous low-f0 harmonic/body imprint",
                "bounded geometry bridge mobility",
                "smoothed far/background body color",
                "per-mode amplitude decomposition metadata",
            ],
        },
        "starts_from_v1_not_v2": True,
        "modes_run": list(modes),
        "notes": list(notes),
        "build_manifest": dict(build_manifest),
        "ablation_table": ablation_table,
        "per_note_comparison": comparison,
        "acceptance_gates": gates,
        "analysis": {
            "a2_low_f0_imprint_helped": a2_imprint > a2_v1 if "A2" in notes else None,
            "a4_preserved_v1": (
                _note_diff(mode_summaries, v3_full, "A4") >= 0.90 * _note_diff(mode_summaries, v1, "A4")
                if "A4" in notes
                else None
            ),
            "e5_preserved_v1": (
                _note_diff(mode_summaries, v3_full, "E5") >= 0.90 * _note_diff(mode_summaries, v1, "E5")
                if "E5" in notes
                else None
            ),
            "mobility_contribution_a2": round(a2_mob - a2_v1, 6) if "A2" in notes else None,
            "far_color_contribution_a2": round(a2_far - a2_v1, 6) if "A2" in notes else None,
            "v3_better_than_v1_all_notes": v3_better_than_v1,
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
        "# Stage 4.9 — STK V3 body-signature report",
        "",
        "## Implementation",
        f"- Base: **radiation v1** (not v2): `{report.get('starts_from_v1_not_v2')}`",
        "- Components: low-f0 imprint, bounded mobility, far/background color",
        "",
        "## Ablation table (spectral_differentiation)",
    ]
    for mode, per_note in (report.get("ablation_table") or {}).items():
        lines.append(f"### `{mode}`")
        for note, val in per_note.items():
            lines.append(f"- {note}: {val}")
        lines.append("")

    lines.append("## Per-note comparison")
    for note, row in (report.get("per_note_comparison") or {}).items():
        lines.append(f"### {note}")
        for k, v in row.items():
            lines.append(f"- {k}: {v}")
        lines.append("")

    lines.append("## Acceptance gates")
    for note, gate in (report.get("acceptance_gates") or {}).items():
        lines.append(f"- **{note}**: passed={gate.get('passed')}, rms_ok={gate.get('rms_ok')}")

    lines.append("")
    lines.append("## Verdict")
    analysis = report.get("analysis") or {}
    lines.append(f"- V3 better than v1 (all notes): {analysis.get('v3_better_than_v1_all_notes')}")
    lines.append(f"- Promotion: **{analysis.get('promotion_recommendation')}**")
    lines.append(f"- FEM launched: **no**")
    return "\n".join(lines)
