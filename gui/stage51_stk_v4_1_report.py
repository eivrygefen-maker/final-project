#!/usr/bin/env python3
"""Stage 5.1 STK V4.1 strict hybrid report."""
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


def build_stage51_v4_1_report(
    *,
    mode_summaries: Mapping[str, Any],
    endpoint_rows: Sequence[Mapping[str, Any]],
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
    v41 = "modal_body_hybrid_v4_1_full"

    comparison: Dict[str, Dict[str, float]] = {}
    for note in notes:
        comparison[note] = {
            "baseline": _diff(mode_summaries, base, note),
            "radiation_v1": _diff(mode_summaries, v1, note),
            "v3_full": _diff(mode_summaries, v3, note) if v3 in mode_summaries else 0.0,
            "v4_full": _diff(mode_summaries, v4, note) if v4 in mode_summaries else 0.0,
            "v4_1_core": _diff(mode_summaries, "modal_body_hybrid_v4_1_core", note),
            "v4_1_full": _diff(mode_summaries, v41, note),
        }

    gates: Dict[str, Any] = {}
    for note in notes:
        v41d = _diff(mode_summaries, v41, note)
        v1d = _diff(mode_summaries, v1, note)
        based = _diff(mode_summaries, base, note)
        rms = _rms_spread(mode_summaries, v41, note)
        if note == "A2":
            passed = v41d >= 0.98 * based and v41d >= v1d - 1e-9
            rule = "v4.1 >= 0.98*baseline and v4.1 >= v1"
        else:
            passed = v41d >= 0.98 * v1d and v41d >= based - 1e-9
            rule = "v4.1 >= 0.98*v1 and v4.1 >= baseline"
        gates[note] = {
            "v4_1_spectral_differentiation": v41d,
            "v1_spectral_differentiation": v1d,
            "baseline_spectral_differentiation": based,
            "ratio_to_baseline": round(v41d / max(based, 1e-12), 6),
            "ratio_to_v1": round(v41d / max(v1d, 1e-12), 6),
            "rms_spread_db": rms,
            "passed": passed,
            "rms_ok": rms <= 4.0,
            "rule": rule,
        }

    endpoint_summary: Dict[str, Any] = {}
    for note in notes:
        rows = [r for r in endpoint_rows if r.get("note") == note]
        if not rows:
            continue
        endpoint_summary[note] = {
            "mean_w_rad": round(sum(float(r.get("w_rad") or 0) for r in rows) / len(rows), 6),
            "mean_error_to_baseline": round(
                sum(float(r.get("endpoint_equivalence_error_to_baseline") or 0) for r in rows) / len(rows),
                8,
            ),
            "mean_error_to_v1": round(
                sum(float(r.get("endpoint_equivalence_error_to_v1") or 0) for r in rows if r.get("endpoint_equivalence_error_to_v1") is not None)
                / max(1, sum(1 for r in rows if r.get("endpoint_equivalence_error_to_v1") is not None)),
                8,
            ),
            "endpoints": [r.get("v4_1_endpoint") for r in rows[:3]],
        }

    report: Dict[str, Any] = {
        "stage": "5.1",
        "title": "STK V4.1 strict hybrid core",
        "formula": {
            "w_rad": "smoothstep(160 Hz, 320 Hz, f0)",
            "low_endpoint": "f0 <= 160 Hz → delegate baseline_current exactly",
            "high_endpoint": "f0 >= 320 Hz → delegate modal_radiation_color_v1 exactly",
            "transition": "equal_loudness_crossfade on dry mixes, unified finalize",
        },
        "f0_continuous_no_note_names": True,
        "uses_v3_v2_as_base": False,
        "modes_run": list(modes),
        "notes": list(notes),
        "build_manifest": dict(build_manifest),
        "per_note_comparison": comparison,
        "acceptance_gates": gates,
        "endpoint_equivalence_summary": endpoint_summary,
        "endpoint_rows_sample": list(endpoint_rows[:12]),
        "analysis": {
            "v4_1_preserves_baseline_a2": gates.get("A2", {}).get("ratio_to_baseline", 0) >= 0.98,
            "v4_1_preserves_v1_a4": gates.get("A4", {}).get("ratio_to_v1", 0) >= 0.98,
            "v4_1_preserves_v1_e5": gates.get("E5", {}).get("ratio_to_v1", 0) >= 0.98,
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
        "# Stage 5.1 — STK V4.1 strict hybrid report",
        "",
        "## Formula",
        f"- Low endpoint: {report.get('formula', {}).get('low_endpoint')}",
        f"- High endpoint: {report.get('formula', {}).get('high_endpoint')}",
        "",
        "## Acceptance gates",
    ]
    for note, gate in (report.get("acceptance_gates") or {}).items():
        lines.append(
            f"- **{note}**: passed={gate.get('passed')}, "
            f"v4.1={gate.get('v4_1_spectral_differentiation')}, "
            f"ratio_baseline={gate.get('ratio_to_baseline')}, ratio_v1={gate.get('ratio_to_v1')}"
        )
    lines.append("")
    lines.append("## Endpoint equivalence")
    for note, eq in (report.get("endpoint_equivalence_summary") or {}).items():
        lines.append(f"- **{note}**: mean_w_rad={eq.get('mean_w_rad')}, error_baseline={eq.get('mean_error_to_baseline')}")
    lines.append("")
    lines.append(f"Promotion: **{(report.get('analysis') or {}).get('promotion_recommendation')}**")
    lines.append("FEM launched: **no**")
    return "\n".join(lines)
