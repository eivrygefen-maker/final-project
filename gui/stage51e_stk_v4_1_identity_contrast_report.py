#!/usr/bin/env python3
"""Stage 5.1E STK V4.1 sample-relative identity contrast validation report."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from body_hybrid_v4_1_identity_space import (
    DZ_BODY_CLIP,
    IQR_FLOOR,
    estimate_audible_clusters,
)

RMS_SPREAD_MAX_DB = 4.0
DIFF_BEAT_STRONG_MIN = 1.001
RHO_IMPROVE_MIN = 0.03
RHO_IMPROVE_TARGET = 0.05
SPREAD_IMPROVE_MIN = 1.05
RMS_AUDIBLE_LO = -30.0
RMS_AUDIBLE_HI = -20.0

V41_MODE = "modal_body_hybrid_v4_1_full"
STRONG_MODE = "modal_body_hybrid_v4_1_identity_strong"
CONTRAST_MODES = (
    "modal_body_hybrid_v4_1_identity_contrast_medium",
    "modal_body_hybrid_v4_1_identity_contrast_strong",
)


def _note_summary(mode_summaries: Mapping[str, Any], mode: str, note: str) -> Dict[str, Any]:
    return dict((mode_summaries.get(mode) or {}).get("notes", {}).get(note) or {})


def _spectral_slope(seg: Mapping[str, Any]) -> float:
    low = float(seg.get("spectral_low_energy") or 0.0)
    high = float(seg.get("spectral_high_energy") or 0.0)
    return high / max(low, 1e-9)


def _spread_metrics(note_summary: Mapping[str, Any], segments: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    flux = [float(s.get("spectral_flux") or 0) for s in segments]
    attacks = [float(s.get("attack_time_ms") or 0) for s in segments]
    slopes = [_spectral_slope(s) for s in segments]
    vs_vals = [
        float((s.get("vs_v41_reference") or {}).get("rms_diff_db_vs_reference"))
        for s in segments
        if (s.get("vs_v41_reference") or {}).get("rms_diff_db_vs_reference") is not None
    ]
    max_abs_vals = [
        float((s.get("vs_v41_reference") or {}).get("max_abs_diff"))
        for s in segments
        if (s.get("vs_v41_reference") or {}).get("max_abs_diff") is not None
    ]
    return {
        "spectral_differentiation": float(note_summary.get("spectral_differentiation") or 0.0),
        "centroid_spread_hz": float(note_summary.get("centroid_spread_hz") or 0.0),
        "spectral_slope_spread": round(max(slopes) - min(slopes), 6) if len(slopes) >= 2 else 0.0,
        "low_energy_spread": float(note_summary.get("low_energy_spread") or 0.0),
        "mid_energy_spread": float(note_summary.get("mid_energy_spread") or 0.0),
        "high_energy_spread": float(note_summary.get("high_energy_spread") or 0.0),
        "decay_slope_spread_db_per_s": float(note_summary.get("decay_slope_spread_db_per_s") or 0.0),
        "rms_spread_db": float(note_summary.get("rms_spread_db") or 0.0),
        "spectral_flux_spread": round(max(flux) - min(flux), 6) if len(flux) >= 2 else 0.0,
        "attack_time_spread_ms": round(max(attacks) - min(attacks), 3) if len(attacks) >= 2 else 0.0,
        "clipping_any": any(bool(s.get("clipping_detected")) for s in segments),
        "rms_diff_db_vs_v41_median": round(float(sorted(vs_vals)[len(vs_vals) // 2]), 4) if vs_vals else None,
        "rms_diff_db_vs_v41_mean": round(sum(vs_vals) / len(vs_vals), 4) if vs_vals else None,
        "max_abs_diff_vs_v41_median": round(float(sorted(max_abs_vals)[len(max_abs_vals) // 2]), 8) if max_abs_vals else None,
        "likely_audible_vs_v41": any(
            bool((s.get("vs_v41_reference") or {}).get("likely_audible")) for s in segments
        ),
    }


def _compare_contrast(
    *,
    v41m: Mapping[str, Any],
    strongm: Mapping[str, Any],
    contrastm: Mapping[str, Any],
    rho_v41: Optional[float],
    rho_strong: Optional[float],
    rho_contrast: Optional[float],
    audio_dists: Sequence[float],
    nn_report: Mapping[str, Any],
) -> Dict[str, Any]:
    diff_vs_strong = contrastm["spectral_differentiation"] / max(strongm["spectral_differentiation"], 1e-9)
    diff_vs_v41 = contrastm["spectral_differentiation"] / max(v41m["spectral_differentiation"], 1e-9)
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
    spread_vs_strong = sum(
        1 for k in spread_keys if contrastm.get(k, 0) >= strongm.get(k, 0) * SPREAD_IMPROVE_MIN
    )
    rho_delta_v41 = None
    if rho_v41 is not None and rho_contrast is not None:
        rho_delta_v41 = round(float(rho_contrast) - float(rho_v41), 6)
    rho_delta_strong = None
    if rho_strong is not None and rho_contrast is not None:
        rho_delta_strong = round(float(rho_contrast) - float(rho_strong), 6)
    rms_med = contrastm.get("rms_diff_db_vs_v41_median")
    rms_in_target = (
        rms_med is not None and RMS_AUDIBLE_HI <= float(rms_med) <= RMS_AUDIBLE_LO
    ) or (rms_med is not None and float(rms_med) > RMS_AUDIBLE_HI)
    beats_strong = diff_vs_strong >= DIFF_BEAT_STRONG_MIN
    passed = (
        beats_strong
        and spread_vs_strong >= 2
        and not contrastm["clipping_any"]
        and contrastm["rms_spread_db"] <= RMS_SPREAD_MAX_DB
        and (rho_delta_v41 is None or rho_delta_v41 >= RHO_IMPROVE_MIN)
        and (nn_report.get("nn_preservation_rate") or 0) >= 0.5
    )
    return {
        "spectral_diff_vs_v41_ratio": round(diff_vs_v41, 4),
        "spectral_diff_vs_strong_ratio": round(diff_vs_strong, 4),
        "beats_identity_strong": beats_strong,
        "spread_axes_vs_strong": spread_vs_strong,
        "rho_vs_v41": rho_delta_v41,
        "rho_vs_strong": rho_delta_strong,
        "estimated_audible_clusters": estimate_audible_clusters(audio_dists),
        "nn_preservation_rate": nn_report.get("nn_preservation_rate"),
        "rms_in_audible_target": rms_in_target,
        "passed": passed,
    }


def build_stage51e_identity_contrast_report(
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
    rho_strong = (distance_by_mode.get(STRONG_MODE) or {}).get("spearman_rho")

    per_note: Dict[str, Any] = {}
    contrast_results: Dict[str, Any] = {}

    for note in notes:
        v41_ns = _note_summary(mode_summaries, V41_MODE, note)
        strong_ns = _note_summary(mode_summaries, STRONG_MODE, note)
        v41m = _spread_metrics(v41_ns, list(v41_ns.get("segments") or []))
        strongm = _spread_metrics(strong_ns, list(strong_ns.get("segments") or []))
        rows: Dict[str, Any] = {V41_MODE: v41m, STRONG_MODE: strongm}
        comparisons: Dict[str, Any] = {}

        for cmode in CONTRAST_MODES:
            if cmode not in modes:
                continue
            ns = _note_summary(mode_summaries, cmode, note)
            segs = list(ns.get("segments") or [])
            cm = _spread_metrics(ns, segs)
            rows[cmode] = cm
            rho_c = (distance_by_mode.get(cmode) or {}).get("spearman_rho")
            audio_dists = (distance_by_mode.get(cmode) or {}).get("audio_distances") or []
            nn_rep = nn_by_mode.get(cmode) or {}
            comparisons[cmode] = _compare_contrast(
                v41m=v41m,
                strongm=strongm,
                contrastm=cm,
                rho_v41=rho_v41,
                rho_strong=rho_strong,
                rho_contrast=rho_c,
                audio_dists=audio_dists,
                nn_report=nn_rep,
            )
            contrast_results[cmode] = {"metrics": cm, "comparison": comparisons[cmode]}

        per_note[note] = {"modes": rows, "comparisons": comparisons}

    note = notes[0] if notes else "A3"
    note_block = per_note.get(note) or {}
    best_mode: Optional[str] = None
    best_diff = -1.0
    for cmode in CONTRAST_MODES:
        comp = (note_block.get("comparisons") or {}).get(cmode) or {}
        if comp.get("passed"):
            ratio = float(comp.get("spectral_diff_vs_strong_ratio") or 0)
            if ratio > best_diff:
                best_diff = ratio
                best_mode = cmode

    c_med = ((note_block.get("comparisons") or {}).get("modal_body_hybrid_v4_1_identity_contrast_medium") or {})
    c_str = ((note_block.get("comparisons") or {}).get("modal_body_hybrid_v4_1_identity_contrast_strong") or {})
    beats_strong = bool(c_med.get("beats_identity_strong") or c_str.get("beats_identity_strong"))

    if best_mode:
        improvement = "useful"
        character = "contrast_separates"
        test_d4 = True
        listen_rec = f"Listen to {best_mode} — contrast beats identity_strong on A3"
    elif beats_strong:
        improvement = "marginal"
        character = "beats_strong_partial_gates"
        test_d4 = False
        listen_rec = "Contrast improves differentiation vs identity_strong; verify gates on VM"
    else:
        improvement = "not_useful"
        character = "no_beat_vs_strong"
        test_d4 = False
        listen_rec = "Contrast did not beat identity_strong — tune dz projection or batch reference"

    report: Dict[str, Any] = {
        "stage": "5.1E",
        "title": "STK V4.1 sample-relative identity contrast",
        "formula": {
            "base": "y_base = modal_body_hybrid_v4_1_full",
            "reference": "z_ref = per-feature median(z_body) over batch; IQR floor = "
            + str(IQR_FLOOR),
            "contrast": f"dz_body = clip((z_body - z_ref) / IQR, ±{DZ_BODY_CLIP})",
            "projection": "axes_contrast = bounded_projection(dz_body / clip_limit)",
            "output": "y = y_base + epsilon * bounded_residual(band+h harmonic shape)",
            "contrast_profiles": {
                "contrast_medium": {
                    "epsilon": 0.45,
                    "harmonic_gain_max": 0.25,
                    "residual_gain_max": 0.18,
                    "band_eq_max_db": 1.2,
                    "rms_guard_max_db": 2.5,
                },
                "contrast_strong": {
                    "epsilon": 0.65,
                    "harmonic_gain_max": 0.35,
                    "residual_gain_max": 0.25,
                    "band_eq_max_db": 1.8,
                    "rms_guard_max_db": 3.0,
                },
            },
        },
        "f0_continuous_no_note_names": True,
        "uses_v3_v2_as_base": False,
        "v4_1_base_preserved": True,
        "modes_run": list(modes),
        "notes": list(notes),
        "build_manifest": dict(build_manifest),
        "per_note_metrics": per_note,
        "contrast_results": contrast_results,
        "nearest_neighbor_by_mode": dict(nn_by_mode),
        "improvement_verdict": improvement,
        "character": character,
        "beats_identity_strong": beats_strong,
        "recommended_listen_mode": best_mode or "modal_body_hybrid_v4_1_identity_contrast_strong",
        "test_d4_next": test_d4,
        "listening_recommendation": listen_rec,
        "distance_consistency_by_mode": dict(distance_by_mode),
        "fem_launched": False,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    out_md.write_text(_render_md(report), encoding="utf-8")
    return report


def _render_md(report: Mapping[str, Any]) -> str:
    lines = [
        "# Stage 5.1E — STK V4.1 sample-relative identity contrast",
        "",
        f"Verdict: **{report.get('improvement_verdict')}** | Beats strong: **{report.get('beats_identity_strong')}**",
        f"Recommended listen: **{report.get('recommended_listen_mode')}**",
        "",
        "## Formula",
        f"- {report.get('formula', {}).get('reference')}",
        f"- {report.get('formula', {}).get('contrast')}",
        "",
        "## A3 comparison",
        "",
        "| mode | spec diff | ρ | rms vs V4.1 (dB) | NN preserve | beats strong | pass |",
        "|------|-----------|---|------------------|-------------|--------------|------|",
    ]
    note = (report.get("notes") or ["A3"])[0]
    block = (report.get("per_note_metrics") or {}).get(note) or {}
    dc = report.get("distance_consistency_by_mode") or {}
    nn = report.get("nearest_neighbor_by_mode") or {}
    for mode in [V41_MODE, STRONG_MODE, *CONTRAST_MODES]:
        m = (block.get("modes") or {}).get(mode) or {}
        if not m:
            continue
        comp = (block.get("comparisons") or {}).get(mode) or {}
        rho = (dc.get(mode) or {}).get("spearman_rho")
        lines.append(
            f"| {mode} | {m.get('spectral_differentiation')} | {rho} | "
            f"{m.get('rms_diff_db_vs_v41_median')} | "
            f"{(nn.get(mode) or {}).get('nn_preservation_rate')} | "
            f"{comp.get('beats_identity_strong')} | {comp.get('passed')} |"
        )
    lines.extend(
        [
            "",
            f"**Recommendation:** {report.get('listening_recommendation')}",
            "FEM launched: **no**",
        ]
    )
    return "\n".join(lines)
