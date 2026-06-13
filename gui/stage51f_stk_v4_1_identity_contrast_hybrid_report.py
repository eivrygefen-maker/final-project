#!/usr/bin/env python3
"""Stage 5.1F STK V4.1 identity+contrast hybrid validation report."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from body_hybrid_v4_1_identity_space import estimate_audible_clusters

RMS_SPREAD_MAX_DB = 4.0
DIFF_BEAT_V41_MIN = 1.03
DIFF_BEAT_CONTRAST_MIN = 1.001
RHO_IMPROVE_MIN = 0.0
SPREAD_IMPROVE_MIN = 1.05
RMS_AUDIBLE_LO = -30.0
RMS_AUDIBLE_HI = -22.0
RMS_AUDIBLE_FLOOR = -40.0

V41_MODE = "modal_body_hybrid_v4_1_full"
STRONG_MODE = "modal_body_hybrid_v4_1_identity_strong"
CONTRAST_STRONG = "modal_body_hybrid_v4_1_identity_contrast_strong"
HYBRID_MODES = (
    "modal_body_hybrid_v4_1_identity_contrast_hybrid_25_75",
    "modal_body_hybrid_v4_1_identity_contrast_hybrid_40_60",
    "modal_body_hybrid_v4_1_identity_contrast_hybrid_50_50",
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
    decays = [float(s.get("output_decay_slope_db_per_s") or 0) for s in segments]
    slopes = [_spectral_slope(s) for s in segments]
    vs_vals = [
        float((s.get("vs_v41_reference") or {}).get("rms_diff_db_vs_reference"))
        for s in segments
        if (s.get("vs_v41_reference") or {}).get("rms_diff_db_vs_reference") is not None
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
        "rms_diff_db_vs_v41_median": round(float(sorted(vs_vals)[len(vs_vals) // 2]), 4) if vs_vals else None,
        "rms_diff_db_vs_v41_mean": round(sum(vs_vals) / len(vs_vals), 4) if vs_vals else None,
        "likely_audible_vs_v41": any(
            bool((s.get("vs_v41_reference") or {}).get("likely_audible")) for s in segments
        ),
        "rms_in_audible_band": (
            bool(vs_vals)
            and RMS_AUDIBLE_LO <= float(sorted(vs_vals)[len(vs_vals) // 2]) <= RMS_AUDIBLE_HI
            and float(sorted(vs_vals)[len(vs_vals) // 2]) > RMS_AUDIBLE_FLOOR
        ),
    }


def _compare_hybrid(
    *,
    v41m: Mapping[str, Any],
    contrastm: Mapping[str, Any],
    hybridm: Mapping[str, Any],
    rho_v41: Optional[float],
    rho_contrast: Optional[float],
    rho_hybrid: Optional[float],
    audio_dists: Sequence[float],
    nn_report: Mapping[str, Any],
) -> Dict[str, Any]:
    diff_vs_v41 = hybridm["spectral_differentiation"] / max(v41m["spectral_differentiation"], 1e-9)
    diff_vs_contrast = hybridm["spectral_differentiation"] / max(contrastm["spectral_differentiation"], 1e-9)
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
    spread_vs_v41 = sum(1 for k in spread_keys if hybridm.get(k, 0) >= v41m.get(k, 0) * SPREAD_IMPROVE_MIN)
    rho_delta_v41 = None
    if rho_v41 is not None and rho_hybrid is not None:
        rho_delta_v41 = round(float(rho_hybrid) - float(rho_v41), 6)
    rho_delta_contrast = None
    if rho_contrast is not None and rho_hybrid is not None:
        rho_delta_contrast = round(float(rho_hybrid) - float(rho_contrast), 6)
    rms_med = hybridm.get("rms_diff_db_vs_v41_median")
    rms_natural = (
        rms_med is not None
        and float(rms_med) > RMS_AUDIBLE_FLOOR
        and RMS_AUDIBLE_LO <= float(rms_med) <= RMS_AUDIBLE_HI
    )
    passed = (
        diff_vs_contrast >= DIFF_BEAT_CONTRAST_MIN
        and diff_vs_v41 >= DIFF_BEAT_V41_MIN
        and spread_vs_v41 >= 2
        and not hybridm["clipping_any"]
        and hybridm["rms_spread_db"] <= RMS_SPREAD_MAX_DB
        and (rho_delta_contrast is None or rho_delta_contrast >= RHO_IMPROVE_MIN)
        and (nn_report.get("nn_preservation_rate") or 0) >= 0.5
        and rms_natural
    )
    return {
        "spectral_diff_vs_v41_ratio": round(diff_vs_v41, 4),
        "spectral_diff_vs_contrast_strong_ratio": round(diff_vs_contrast, 4),
        "pct_vs_v41": round((diff_vs_v41 - 1.0) * 100.0, 3),
        "beats_contrast_strong": diff_vs_contrast >= DIFF_BEAT_CONTRAST_MIN,
        "beats_v41_3pct": diff_vs_v41 >= DIFF_BEAT_V41_MIN,
        "spread_axes_vs_v41": spread_vs_v41,
        "rho_vs_v41": rho_delta_v41,
        "rho_vs_contrast_strong": rho_delta_contrast,
        "estimated_audible_clusters": estimate_audible_clusters(audio_dists),
        "nn_preservation_rate": nn_report.get("nn_preservation_rate"),
        "rms_in_natural_band": rms_natural,
        "passed": passed,
    }


def _architecture_awareness(
    *,
    v41m: Mapping[str, Any],
    contrastm: Mapping[str, Any],
    best_hybrid: Optional[Mapping[str, Any]],
    best_comp: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    hybrid_beats_v41 = bool((best_comp or {}).get("beats_v41_3pct"))
    hybrid_beats_contrast = bool((best_comp or {}).get("beats_contrast_strong"))
    v41_diff = float(v41m.get("spectral_differentiation") or 0)
    contrast_diff = float(contrastm.get("spectral_differentiation") or 0)
    hybrid_diff = float((best_hybrid or {}).get("spectral_differentiation") or 0)
    overlay_headroom = hybrid_diff - v41_diff
    separation_from_overlay = hybrid_diff - contrast_diff
    base_dominance_ratio = v41_diff / max(hybrid_diff, 1e-9)
    return {
        "hybrid_improves_over_v41_without_base_block": hybrid_beats_v41,
        "most_separation_from_overlay_not_base": separation_from_overlay > v41_diff * 0.15,
        "base_still_carries_majority_of_identity": base_dominance_ratio > 0.85,
        "overlay_headroom_vs_v41": round(overlay_headroom, 6),
        "future_path_recommendation": (
            "continue_hybrid_overlay_tuning"
            if hybrid_beats_contrast and hybrid_beats_v41
            else (
                "revisit_body_response_first_or_base_architecture"
                if not hybrid_beats_contrast and contrast_diff > v41_diff * 1.01
                else "continue_overlay_tuning_with_audibility_focus"
            )
        ),
        "v41_v1_compression_risk_note": (
            "V4.1 endpoint blend may compress guitar families early; "
            "if hybrid gains plateau below +5% vs V4.1, prioritize body-response-first (5.2A) "
            "or reduce base loudness normalization — do not change V4.1 in 5.1F."
        ),
    }


def build_stage51f_identity_contrast_hybrid_report(
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
    rho_contrast = (distance_by_mode.get(CONTRAST_STRONG) or {}).get("spearman_rho")

    per_note: Dict[str, Any] = {}
    hybrid_results: Dict[str, Any] = {}

    for note in notes:
        v41_ns = _note_summary(mode_summaries, V41_MODE, note)
        strong_ns = _note_summary(mode_summaries, STRONG_MODE, note)
        contrast_ns = _note_summary(mode_summaries, CONTRAST_STRONG, note)
        v41m = _spread_metrics(v41_ns, list(v41_ns.get("segments") or []))
        strongm = _spread_metrics(strong_ns, list(strong_ns.get("segments") or []))
        contrastm = _spread_metrics(contrast_ns, list(contrast_ns.get("segments") or []))
        rows: Dict[str, Any] = {
            V41_MODE: v41m,
            STRONG_MODE: strongm,
            CONTRAST_STRONG: contrastm,
        }
        comparisons: Dict[str, Any] = {}

        for hmode in HYBRID_MODES:
            if hmode not in modes:
                continue
            ns = _note_summary(mode_summaries, hmode, note)
            segs = list(ns.get("segments") or [])
            hm = _spread_metrics(ns, segs)
            rows[hmode] = hm
            rho_h = (distance_by_mode.get(hmode) or {}).get("spearman_rho")
            audio_dists = (distance_by_mode.get(hmode) or {}).get("audio_distances") or []
            comparisons[hmode] = _compare_hybrid(
                v41m=v41m,
                contrastm=contrastm,
                hybridm=hm,
                rho_v41=rho_v41,
                rho_contrast=rho_contrast,
                rho_hybrid=rho_h,
                audio_dists=audio_dists,
                nn_report=nn_by_mode.get(hmode) or {},
            )
            hybrid_results[hmode] = {"metrics": hm, "comparison": comparisons[hmode]}

        per_note[note] = {"modes": rows, "comparisons": comparisons}

    note = notes[0] if notes else "A3"
    note_block = per_note.get(note) or {}
    best_mode: Optional[str] = None
    best_diff = -1.0
    best_comp: Optional[Dict[str, Any]] = None
    for hmode in HYBRID_MODES:
        comp = (note_block.get("comparisons") or {}).get(hmode) or {}
        ratio = float(comp.get("spectral_diff_vs_contrast_strong_ratio") or 0)
        if ratio > best_diff:
            best_diff = ratio
            best_mode = hmode
            best_comp = comp

    best_hybrid_m = ((note_block.get("modes") or {}).get(best_mode or "") or {})
    v41m = ((note_block.get("modes") or {}).get(V41_MODE) or {})
    contrastm = ((note_block.get("modes") or {}).get(CONTRAST_STRONG) or {})
    arch = _architecture_awareness(
        v41m=v41m,
        contrastm=contrastm,
        best_hybrid=best_hybrid_m,
        best_comp=best_comp,
    )

    if best_comp and best_comp.get("passed"):
        verdict = "useful"
        listen = f"Listen to {best_mode} — hybrid gates passed on A3"
        test_d4 = True
    elif best_comp and best_comp.get("beats_contrast_strong"):
        verdict = "marginal"
        listen = f"Listen to {best_mode} — beats contrast_strong; verify audibility on VM stitch"
        test_d4 = False
    else:
        verdict = "not_useful"
        listen = "Hybrid did not beat contrast_strong — try 25/75 on VM or revisit 5.2A base path"
        test_d4 = False

    report: Dict[str, Any] = {
        "stage": "5.1F",
        "title": "STK V4.1 identity + contrast hybrid",
        "formula": {
            "base": "y_base = modal_body_hybrid_v4_1_full",
            "absolute": "res_abs = identity_strong_residual(y_base, z_body)",
            "contrast": "res_contrast = contrast_strong_residual(y_base, dz_body)",
            "blend": "y = y_base + a*res_abs + b*res_contrast → audibility floor → RMS guard",
            "variants": {
                "25_75": "a=0.25, b=0.75",
                "40_60": "a=0.40, b=0.60",
                "50_50": "a=0.50, b=0.50",
            },
            "audibility_target_db_vs_v41": f"({RMS_AUDIBLE_HI}, {RMS_AUDIBLE_LO}] not below {RMS_AUDIBLE_FLOOR}",
        },
        "vm_reference_a3": {
            "v41_spec_diff": 0.072396,
            "identity_strong_spec_diff": 0.072578,
            "contrast_strong_spec_diff": 0.073817,
            "contrast_strong_pct_vs_v41": 1.96,
        },
        "f0_continuous_no_note_names": True,
        "v4_1_base_preserved": True,
        "v4_1_unchanged": True,
        "modes_run": list(modes),
        "notes": list(notes),
        "build_manifest": dict(build_manifest),
        "per_note_metrics": per_note,
        "hybrid_results": hybrid_results,
        "best_hybrid_mode": best_mode,
        "improvement_verdict": verdict,
        "architecture_awareness": arch,
        "recommended_listen_mode": best_mode or "modal_body_hybrid_v4_1_identity_contrast_hybrid_40_60",
        "listening_recommendation": listen,
        "test_d4_next": test_d4,
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
        "# Stage 5.1F — STK V4.1 identity + contrast hybrid",
        "",
        f"Verdict: **{report.get('improvement_verdict')}** | Best hybrid: **{report.get('best_hybrid_mode')}**",
        "",
        "## A3 comparison",
        "",
        "| mode | spec diff | ρ | rms vs V4.1 (dB) | NN | clipping |",
        "|------|-----------|---|------------------|----|---------|",
    ]
    note = (report.get("notes") or ["A3"])[0]
    block = (report.get("per_note_metrics") or {}).get(note) or {}
    dc = report.get("distance_consistency_by_mode") or {}
    nn = report.get("nearest_neighbor_by_mode") or {}
    for mode in [
        V41_MODE,
        STRONG_MODE,
        CONTRAST_STRONG,
        *HYBRID_MODES,
    ]:
        m = (block.get("modes") or {}).get(mode) or {}
        if not m:
            continue
        rho = (dc.get(mode) or {}).get("spearman_rho")
        lines.append(
            f"| {mode} | {m.get('spectral_differentiation')} | {rho} | "
            f"{m.get('rms_diff_db_vs_v41_median')} | "
            f"{(nn.get(mode) or {}).get('nn_preservation_rate')} | {m.get('clipping_any')} |"
        )
    arch = report.get("architecture_awareness") or {}
    lines.extend(
        [
            "",
            "## Architecture awareness",
            f"- Hybrid improves over V4.1: **{arch.get('hybrid_improves_over_v41_without_base_block')}**",
            f"- Separation mostly from overlay: **{arch.get('most_separation_from_overlay_not_base')}**",
            f"- Base still carries majority: **{arch.get('base_still_carries_majority_of_identity')}**",
            f"- Future path: **{arch.get('future_path_recommendation')}**",
            "",
            f"**Recommendation:** {report.get('listening_recommendation')}",
            "FEM launched: **no**",
        ]
    )
    return "\n".join(lines)
