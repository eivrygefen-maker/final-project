#!/usr/bin/env python3
"""Stage 5.1C STK V4.1 identity-space validation report."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from body_hybrid_v4_1_identity_space import estimate_audible_clusters

RMS_SPREAD_MAX_DB = 4.0
DIFF_IMPROVE_MIN = 1.05
SPREAD_IMPROVE_MIN = 1.08
RHO_IMPROVE_MIN = 0.05


def _note_summary(mode_summaries: Mapping[str, Any], mode: str, note: str) -> Dict[str, Any]:
    return dict((mode_summaries.get(mode) or {}).get("notes", {}).get(note) or {})


def _spread_metrics(note_summary: Mapping[str, Any], segments: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    flux = [float(s.get("spectral_flux") or 0) for s in segments]
    attacks = [float(s.get("attack_time_ms") or 0) for s in segments]
    return {
        "spectral_differentiation": float(note_summary.get("spectral_differentiation") or 0.0),
        "centroid_spread_hz": float(note_summary.get("centroid_spread_hz") or 0.0),
        "low_energy_spread": float(note_summary.get("low_energy_spread") or 0.0),
        "mid_energy_spread": float(note_summary.get("mid_energy_spread") or 0.0),
        "high_energy_spread": float(note_summary.get("high_energy_spread") or 0.0),
        "decay_slope_spread_db_per_s": float(note_summary.get("decay_slope_spread_db_per_s") or 0.0),
        "rms_spread_db": float(note_summary.get("rms_spread_db") or 0.0),
        "spectral_flux_spread": round(max(flux) - min(flux), 6) if len(flux) >= 2 else 0.0,
        "attack_time_spread_ms": round(max(attacks) - min(attacks), 3) if len(attacks) >= 2 else 0.0,
        "clipping_any": any(bool(s.get("clipping_detected")) for s in segments),
    }


def build_stage51c_identity_space_report(
    *,
    mode_summaries: Mapping[str, Any],
    distance_by_mode: Mapping[str, Mapping[str, Any]],
    notes: Sequence[str],
    modes: Sequence[str],
    build_manifest: Mapping[str, Any],
    out_json: Path,
    out_md: Path,
) -> Dict[str, Any]:
    v41 = "modal_body_hybrid_v4_1_full"
    ident = "modal_body_hybrid_v4_1_identity_space"
    base = "baseline_current"
    v1 = "modal_radiation_color_v1"

    per_note: Dict[str, Any] = {}
    gates: Dict[str, Any] = {}

    for note in notes:
        rows: Dict[str, Any] = {}
        for mode in (base, v1, v41, ident):
            ns = _note_summary(mode_summaries, mode, note)
            segs = list(ns.get("segments") or [])
            rows[mode] = _spread_metrics(ns, segs)

        v41m, idm = rows[v41], rows[ident]
        diff_ratio = idm["spectral_differentiation"] / max(v41m["spectral_differentiation"], 1e-9)
        spread_keys = (
            "centroid_spread_hz",
            "low_energy_spread",
            "mid_energy_spread",
            "high_energy_spread",
            "decay_slope_spread_db_per_s",
            "attack_time_spread_ms",
            "spectral_flux_spread",
        )
        spread_improved = sum(
            1 for k in spread_keys if idm.get(k, 0) >= v41m.get(k, 0) * SPREAD_IMPROVE_MIN
        )

        rho_v41 = (distance_by_mode.get(v41) or {}).get("spearman_rho")
        rho_id = (distance_by_mode.get(ident) or {}).get("spearman_rho")
        rho_delta = None
        if rho_v41 is not None and rho_id is not None:
            rho_delta = round(float(rho_id) - float(rho_v41), 6)

        audio_dists = (distance_by_mode.get(ident) or {}).get("audio_distances") or []
        est_clusters = estimate_audible_clusters(audio_dists)

        passed = (
            diff_ratio >= DIFF_IMPROVE_MIN
            and spread_improved >= 3
            and not idm["clipping_any"]
            and idm["rms_spread_db"] <= RMS_SPREAD_MAX_DB
            and (rho_delta is None or rho_delta >= RHO_IMPROVE_MIN)
        )

        per_note[note] = {
            "modes": rows,
            "v4_1_to_identity_diff_ratio": round(diff_ratio, 4),
            "spread_axes_improved_count": spread_improved,
            "distance_consistency": {
                "v4_1_spearman_rho": rho_v41,
                "identity_spearman_rho": rho_id,
                "rho_improvement": rho_delta,
            },
            "estimated_audible_clusters": est_clusters,
            "passed": passed,
        }
        gates[note] = per_note[note]

    note = notes[0] if notes else "A3"
    gate = gates.get(note) or {}
    useful = bool(gate.get("passed"))
    diff_ratio = float(gate.get("v4_1_to_identity_diff_ratio") or 1.0)
    if useful:
        character = "natural"
        improvement = "useful"
        test_d4 = True
        default_rec = "Continue — identity_space improves multi-axis spread on A3; test D4 next"
    elif diff_ratio > 1.15:
        character = "possibly_aggressive"
        improvement = "marginal"
        test_d4 = False
        default_rec = "Tune epsilon/harmonic gains — differentiation up but gates not fully met"
    elif diff_ratio < 1.02:
        character = "too_subtle"
        improvement = "not_useful"
        test_d4 = False
        default_rec = "Increase bounded identity gains slightly or enrich z_body features"
    else:
        character = "mixed"
        improvement = "not_useful"
        test_d4 = bool(gate.get("spread_axes_improved_count", 0) >= 2)
        default_rec = "Review listening — partial spread improvement without full gate pass"

    reveals_more_groups = (gate.get("estimated_audible_clusters") or 0) >= 5

    report: Dict[str, Any] = {
        "stage": "5.1C",
        "title": "STK V4.1 continuous body-identity space",
        "formula": {
            "base": "y_base = modal_body_hybrid_v4_1_full (endpoints unchanged)",
            "residual": "y = y_base + epsilon * bounded_residual, epsilon=0.18",
            "harmonic": "harmonic_gain_k = exp(gain * z_body), k=2..8 dominant; fundamental ~preserved",
            "bounds": {
                "epsilon": 0.18,
                "harmonic_gain_max": 0.12,
                "fundamental_gain_max": 0.025,
                "residual_gain_max": 0.08,
                "rms_guard_db": 1.5,
            },
        },
        "identity_vector_field_groups": [
            "geometry (length, width, depth, thicknesses, hole, areas, aspect, cavity)",
            "materials (wood embed, damping)",
            "modal density / near-mode counts by band",
            "Q/tau fingerprint, participation shares",
            "bridge mobility, effective modal mass proxy",
            "rank-normalized radiation/mic/bridge (no raw gain)",
            "low/mid/high body color bands",
        ],
        "f0_continuous_no_note_names": True,
        "uses_v3_v2_as_base": False,
        "v4_1_base_preserved": True,
        "modes_run": list(modes),
        "notes": list(notes),
        "build_manifest": dict(build_manifest),
        "per_note_metrics": per_note,
        "acceptance_gates": gates,
        "improvement_verdict": improvement,
        "character": character,
        "reveals_more_than_3_4_groups": reveals_more_groups,
        "test_d4_next": test_d4,
        "listening_recommendation": default_rec,
        "default_stk_note": "V4.1 remains production base; identity_space is diagnostic-only overlay",
        "distance_consistency_by_mode": dict(distance_by_mode),
        "fem_launched": False,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    out_md.write_text(_render_md(report), encoding="utf-8")
    return report


def _render_md(report: Mapping[str, Any]) -> str:
    lines = [
        "# Stage 5.1C — STK V4.1 body-identity space",
        "",
        f"Improvement: **{report.get('improvement_verdict')}** | Character: **{report.get('character')}**",
        f"Test D4 next: **{report.get('test_d4_next')}**",
        f"Reveals >3–4 groups: **{report.get('reveals_more_than_3_4_groups')}**",
        "",
        "## Formula",
        f"- Base: {report.get('formula', {}).get('base')}",
        f"- Residual: {report.get('formula', {}).get('residual')}",
        "",
        "## Metrics by mode",
        "",
        "| note/mode | spec diff | centroid spread | rms spread | clipping |",
        "|-----------|-----------|-----------------|------------|----------|",
    ]
    for note, block in (report.get("per_note_metrics") or {}).items():
        for mode, m in (block.get("modes") or {}).items():
            lines.append(
                f"| {note}/{mode} | {m.get('spectral_differentiation')} | "
                f"{m.get('centroid_spread_hz')} | {m.get('rms_spread_db')} | {m.get('clipping_any')} |"
            )
    dc = report.get("distance_consistency_by_mode") or {}
    lines.extend(
        [
            "",
            "## Distance consistency (Spearman rho)",
            f"- V4.1: {(dc.get('modal_body_hybrid_v4_1_full') or {}).get('spearman_rho')}",
            f"- identity_space: {(dc.get('modal_body_hybrid_v4_1_identity_space') or {}).get('spearman_rho')}",
            "",
            f"**Recommendation:** {report.get('listening_recommendation')}",
            "FEM launched: **no**",
        ]
    )
    return "\n".join(lines)
