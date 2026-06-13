#!/usr/bin/env python3
"""Stage 5.2A STK body-response-first V4.2 validation report."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from body_hybrid_v4_1_identity_space import estimate_audible_clusters

RMS_SPREAD_MAX_DB = 4.0
DIFF_BEAT_V41_MIN = 1.001
DIFF_BEAT_CONTRAST_MIN = 1.001
RHO_IMPROVE_MIN = 0.03
SPREAD_IMPROVE_MIN = 1.05

V41_MODE = "modal_body_hybrid_v4_1_full"
CONTRAST_STRONG = "modal_body_hybrid_v4_1_identity_contrast_strong"
V42_MODE = "modal_body_response_first_v4_2"


def _note_summary(mode_summaries: Mapping[str, Any], mode: str, note: str) -> Dict[str, Any]:
    return dict((mode_summaries.get(mode) or {}).get("notes", {}).get(note) or {})


def _spectral_slope(seg: Mapping[str, Any]) -> float:
    low = float(seg.get("spectral_low_energy") or 0.0)
    high = float(seg.get("spectral_high_energy") or 0.0)
    return high / max(low, 1e-9)


def _spread_metrics(note_summary: Mapping[str, Any], segments: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    flux = [float(s.get("spectral_flux") or 0) for s in segments]
    attacks = [float(s.get("attack_time_ms") or 0) for s in segments]
    decays = [float(s.get("output_decay_slope_db_per_s") or 0) for s in segments]
    slopes = [_spectral_slope(s) for s in segments]
    f0_ratios = [
        float((s.get("fundamental_stability") or {}).get("f0_to_h2_ratio") or 0)
        for s in segments
        if s.get("fundamental_stability")
    ]
    return {
        "spectral_differentiation": float(note_summary.get("spectral_differentiation") or 0.0),
        "centroid_spread_hz": float(note_summary.get("centroid_spread_hz") or 0.0),
        "spectral_slope_spread": round(max(slopes) - min(slopes), 6) if len(slopes) >= 2 else 0.0,
        "low_energy_spread": float(note_summary.get("low_energy_spread") or 0.0),
        "mid_energy_spread": float(note_summary.get("mid_energy_spread") or 0.0),
        "high_energy_spread": float(note_summary.get("high_energy_spread") or 0.0),
        "decay_slope_spread_db_per_s": round(max(decays) - min(decays), 4) if len(decays) >= 2 else 0.0,
        "rms_spread_db": float(note_summary.get("rms_spread_db") or 0.0),
        "spectral_flux_spread": round(max(flux) - min(flux), 6) if len(flux) >= 2 else 0.0,
        "attack_time_spread_ms": round(max(attacks) - min(attacks), 3) if len(attacks) >= 2 else 0.0,
        "clipping_any": any(bool(s.get("clipping_detected")) for s in segments),
        "fundamental_stability_median": round(float(sorted(f0_ratios)[len(f0_ratios) // 2]), 4) if f0_ratios else None,
    }


def _compare_v42(
    *,
    v41m: Mapping[str, Any],
    contrastm: Mapping[str, Any],
    v42m: Mapping[str, Any],
    rho_v41: Optional[float],
    rho_contrast: Optional[float],
    rho_v42: Optional[float],
    audio_dists: Sequence[float],
    nn_report: Mapping[str, Any],
) -> Dict[str, Any]:
    diff_vs_v41 = v42m["spectral_differentiation"] / max(v41m["spectral_differentiation"], 1e-9)
    diff_vs_contrast = v42m["spectral_differentiation"] / max(contrastm["spectral_differentiation"], 1e-9)
    spread_keys = (
        "centroid_spread_hz",
        "spectral_slope_spread",
        "low_energy_spread",
        "mid_energy_spread",
        "high_energy_spread",
        "decay_slope_spread_db_per_s",
        "attack_time_spread_ms",
        "spectral_flux_spread",
    )
    spread_vs_v41 = sum(1 for k in spread_keys if v42m.get(k, 0) >= v41m.get(k, 0) * SPREAD_IMPROVE_MIN)
    rho_delta = None
    if rho_v41 is not None and rho_v42 is not None:
        rho_delta = round(float(rho_v42) - float(rho_v41), 6)
    passed = (
        diff_vs_v41 >= DIFF_BEAT_V41_MIN
        and diff_vs_contrast >= DIFF_BEAT_CONTRAST_MIN
        and spread_vs_v41 >= 2
        and not v42m["clipping_any"]
        and v42m["rms_spread_db"] <= RMS_SPREAD_MAX_DB
        and (rho_delta is None or rho_delta >= RHO_IMPROVE_MIN)
        and (nn_report.get("nn_preservation_rate") or 0) >= 0.5
    )
    return {
        "spectral_diff_vs_v41_ratio": round(diff_vs_v41, 4),
        "spectral_diff_vs_contrast_strong_ratio": round(diff_vs_contrast, 4),
        "beats_v41": diff_vs_v41 >= DIFF_BEAT_V41_MIN,
        "beats_contrast_strong": diff_vs_contrast >= DIFF_BEAT_CONTRAST_MIN,
        "spread_axes_vs_v41": spread_vs_v41,
        "rho_vs_v41": rho_delta,
        "estimated_audible_clusters": estimate_audible_clusters(audio_dists),
        "nn_preservation_rate": nn_report.get("nn_preservation_rate"),
        "passed": passed,
    }


def build_stage52a_body_response_first_report(
    *,
    mode_summaries: Mapping[str, Any],
    distance_by_mode: Mapping[str, Mapping[str, Any]],
    nn_by_mode: Mapping[str, Mapping[str, Any]],
    notes: Sequence[str],
    modes: Sequence[str],
    build_manifest: Mapping[str, Any],
    out_json: Path,
    out_md: Path,
) -> Dict[str, Any]:
    rho_v41 = (distance_by_mode.get(V41_MODE) or {}).get("spearman_rho")
    rho_contrast = (distance_by_mode.get(CONTRAST_STRONG) or {}).get("spearman_rho")

    per_note: Dict[str, Any] = {}
    for note in notes:
        v41_ns = _note_summary(mode_summaries, V41_MODE, note)
        contrast_ns = _note_summary(mode_summaries, CONTRAST_STRONG, note)
        v42_ns = _note_summary(mode_summaries, V42_MODE, note)
        v41m = _spread_metrics(v41_ns, list(v41_ns.get("segments") or []))
        contrastm = _spread_metrics(contrast_ns, list(contrast_ns.get("segments") or []))
        v42m = _spread_metrics(v42_ns, list(v42_ns.get("segments") or []))
        rho_v42 = (distance_by_mode.get(V42_MODE) or {}).get("spearman_rho")
        audio_dists = (distance_by_mode.get(V42_MODE) or {}).get("audio_distances") or []
        comparison = _compare_v42(
            v41m=v41m,
            contrastm=contrastm,
            v42m=v42m,
            rho_v41=rho_v41,
            rho_contrast=rho_contrast,
            rho_v42=rho_v42,
            audio_dists=audio_dists,
            nn_report=nn_by_mode.get(V42_MODE) or {},
        )
        per_note[note] = {
            "modes": {
                V41_MODE: v41m,
                CONTRAST_STRONG: contrastm,
                V42_MODE: v42m,
            },
            "v42_comparison": comparison,
        }

    note = notes[0] if notes else "A3"
    comp = (per_note.get(note) or {}).get("v42_comparison") or {}
    beats_v41 = bool(comp.get("beats_v41"))
    beats_contrast = bool(comp.get("beats_contrast_strong"))
    passed = bool(comp.get("passed"))

    if passed:
        verdict = "useful"
        character = "body_response_separates"
        continue_v42 = True
        listen = "Listen to modal_body_response_first_v4_2 A3 stitch — gates passed"
    elif beats_contrast and beats_v41:
        verdict = "marginal"
        character = "beats_both_partial_gates"
        continue_v42 = True
        listen = "V4.2 beats V4.1 and contrast_strong on spec diff — VM listen before 5.1F"
    elif beats_v41:
        verdict = "marginal"
        character = "beats_v41_only"
        continue_v42 = True
        listen = "V4.2 improves over V4.1 but not contrast_strong — tune H_guitar coupling"
    else:
        verdict = "not_useful"
        character = "no_separation_gain"
        continue_v42 = False
        listen = "Return to 5.1F overlay tuning or revise H_guitar amplitude law"

    report: Dict[str, Any] = {
        "stage": "5.2A",
        "title": "STK body-response-first V4.2 diagnostic",
        "formula": {
            "path": "y = IFFT(FFT(string_acc) * H_guitar,note) + bounded_direct_tap",
            "H_guitar_note": "Σ_m A_m,note · Lorentzian(f, f_m, Q_m)",
            "A_m_note": "bridge_inj * rad_rank * mic_rank * inv_mass_rank * participation * harmonic_coupling",
            "not_used": [
                "modal_radiation_color_v1 base",
                "V4.1 hybrid endpoint",
                "V1 high-note overlay",
                "identity residual overlay",
                "sample_id",
                "raw radiation gain",
            ],
        },
        "transfer_variables": [
            "bridge_excitation / bridge_mobility_proxy",
            "modal frequency f_m, Q_m from material/geometry damping",
            "rank-normalized radiation_proxy, mic_output_proxy",
            "inverse effective_modal_mass (top/back weighted)",
            "top/back/air participation shares",
            "harmonic proximity (strong h2–h8, weak h1, low far-mode texture)",
        ],
        "f0_continuous_no_note_names": True,
        "v4_1_unchanged": True,
        "modes_run": list(modes),
        "notes": list(notes),
        "build_manifest": dict(build_manifest),
        "per_note_metrics": per_note,
        "improvement_verdict": verdict,
        "character": character,
        "continue_v42_path": continue_v42,
        "recommended_listen_mode": V42_MODE,
        "listening_recommendation": listen,
        "distance_consistency_by_mode": dict(distance_by_mode),
        "nearest_neighbor_by_mode": dict(nn_by_mode),
        "fem_launched": False,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    out_md.write_text(_render_md(report), encoding="utf-8")
    return report


def _render_md(report: Mapping[str, Any]) -> str:
    lines = [
        "# Stage 5.2A — STK body-response-first V4.2",
        "",
        f"Verdict: **{report.get('improvement_verdict')}** | Continue V4.2: **{report.get('continue_v42_path')}**",
        "",
        "## A3 comparison",
        "",
        "| mode | spec diff | ρ | rms spread | NN preserve | f0 stability | clipping |",
        "|------|-----------|---|------------|-------------|--------------|----------|",
    ]
    note = (report.get("notes") or ["A3"])[0]
    block = (report.get("per_note_metrics") or {}).get(note) or {}
    dc = report.get("distance_consistency_by_mode") or {}
    nn = report.get("nearest_neighbor_by_mode") or {}
    for mode in [V41_MODE, CONTRAST_STRONG, V42_MODE]:
        m = (block.get("modes") or {}).get(mode) or {}
        rho = (dc.get(mode) or {}).get("spearman_rho")
        lines.append(
            f"| {mode} | {m.get('spectral_differentiation')} | {rho} | "
            f"{m.get('rms_spread_db')} | {(nn.get(mode) or {}).get('nn_preservation_rate')} | "
            f"{m.get('fundamental_stability_median')} | {m.get('clipping_any')} |"
        )
    comp = block.get("v42_comparison") or {}
    lines.extend(
        [
            "",
            f"V4.2 vs V4.1 spec ratio: **{comp.get('spectral_diff_vs_v41_ratio')}**",
            f"V4.2 vs contrast_strong ratio: **{comp.get('spectral_diff_vs_contrast_strong_ratio')}**",
            f"Clusters: **{comp.get('estimated_audible_clusters')}**",
            "",
            f"**Recommendation:** {report.get('listening_recommendation')}",
            "FEM launched: **no**",
        ]
    )
    return "\n".join(lines)
