#!/usr/bin/env python3
"""Stage 5.1D STK V4.1 identity strength sweep validation report."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from body_hybrid_v4_1_identity_space import (
    MODE_TO_STRENGTH,
    PERCEPTUAL_AXIS_NAMES,
    STRENGTH_PROFILES,
    estimate_audible_clusters,
)

RMS_SPREAD_MAX_DB = 4.0
DIFF_IMPROVE_MIN = 1.05
SPREAD_IMPROVE_MIN = 1.08
RHO_IMPROVE_MIN = 0.05
LAYER_ACTIVE_GATE_DB = -45.0

V41_MODE = "modal_body_hybrid_v4_1_full"
IDENTITY_MODES = (
    "modal_body_hybrid_v4_1_identity_light",
    "modal_body_hybrid_v4_1_identity_medium",
    "modal_body_hybrid_v4_1_identity_strong",
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
        "layer_active_gate_pass": (
            round(sum(vs_vals) / len(vs_vals), 4) > LAYER_ACTIVE_GATE_DB if vs_vals else False
        ),
    }


def _compare_to_v41(
    v41m: Mapping[str, Any],
    idm: Mapping[str, Any],
    *,
    rho_v41: Optional[float],
    rho_id: Optional[float],
    audio_dists: Sequence[float],
) -> Dict[str, Any]:
    diff_ratio = idm["spectral_differentiation"] / max(v41m["spectral_differentiation"], 1e-9)
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
    spread_improved = sum(
        1 for k in spread_keys if idm.get(k, 0) >= v41m.get(k, 0) * SPREAD_IMPROVE_MIN
    )
    rho_delta = None
    if rho_v41 is not None and rho_id is not None:
        rho_delta = round(float(rho_id) - float(rho_v41), 6)
    est_clusters = estimate_audible_clusters(audio_dists)
    passed = (
        diff_ratio >= DIFF_IMPROVE_MIN
        and spread_improved >= 3
        and not idm["clipping_any"]
        and idm["rms_spread_db"] <= RMS_SPREAD_MAX_DB
        and (rho_delta is None or rho_delta >= RHO_IMPROVE_MIN)
        and bool(idm.get("layer_active_gate_pass"))
    )
    return {
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


def build_stage51d_identity_sweep_report(
    *,
    mode_summaries: Mapping[str, Any],
    distance_by_mode: Mapping[str, Mapping[str, Any]],
    notes: Sequence[str],
    modes: Sequence[str],
    build_manifest: Mapping[str, Any],
    out_json: Path,
    out_md: Path,
) -> Dict[str, Any]:
    rho_v41 = (distance_by_mode.get(V41_MODE) or {}).get("spearman_rho")

    per_note: Dict[str, Any] = {}
    strength_results: Dict[str, Any] = {}

    for note in notes:
        v41_ns = _note_summary(mode_summaries, V41_MODE, note)
        v41_segs = list(v41_ns.get("segments") or [])
        v41m = _spread_metrics(v41_ns, v41_segs)
        rows: Dict[str, Any] = {V41_MODE: v41m}
        comparisons: Dict[str, Any] = {}

        for ident in IDENTITY_MODES:
            if ident not in modes:
                continue
            ns = _note_summary(mode_summaries, ident, note)
            segs = list(ns.get("segments") or [])
            idm = _spread_metrics(ns, segs)
            rows[ident] = idm
            rho_id = (distance_by_mode.get(ident) or {}).get("spearman_rho")
            audio_dists = (distance_by_mode.get(ident) or {}).get("audio_distances") or []
            comparisons[ident] = _compare_to_v41(v41m, idm, rho_v41=rho_v41, rho_id=rho_id, audio_dists=audio_dists)
            strength_results[ident] = {
                "metrics": idm,
                "comparison": comparisons[ident],
                "strength_profile": (MODE_TO_STRENGTH.get(ident) or "light"),
            }

        per_note[note] = {"modes": rows, "comparisons": comparisons}

    note = notes[0] if notes else "A3"
    note_block = per_note.get(note) or {}
    best_mode: Optional[str] = None
    best_score = -1.0
    for ident in IDENTITY_MODES:
        comp = (note_block.get("comparisons") or {}).get(ident) or {}
        if comp.get("passed"):
            score = float(comp.get("v4_1_to_identity_diff_ratio") or 0)
            if score > best_score:
                best_score = score
                best_mode = ident

    medium_m = ((note_block.get("modes") or {}).get("modal_body_hybrid_v4_1_identity_medium") or {})
    strong_m = ((note_block.get("modes") or {}).get("modal_body_hybrid_v4_1_identity_strong") or {})
    medium_active = bool(medium_m.get("layer_active_gate_pass"))
    strong_active = bool(strong_m.get("layer_active_gate_pass"))

    if best_mode:
        improvement = "useful"
        character = "natural"
        test_d4 = True
        listen_rec = f"Listen to {best_mode} stitched WAV — gates passed on A3"
    elif medium_active or strong_active:
        improvement = "marginal"
        character = "audible_but_gates_incomplete"
        test_d4 = False
        listen_rec = "Listen to identity_strong then identity_medium — layer active but acceptance gates not fully met"
    elif (medium_m.get("rms_diff_db_vs_v41_median") or -120) > LAYER_ACTIVE_GATE_DB:
        improvement = "marginal"
        character = "possibly_audible"
        test_d4 = False
        listen_rec = "Layer exceeds -45 dB vs V4.1 but timbre gates weak — tune axis projection"
    else:
        improvement = "not_useful"
        character = "too_subtle"
        test_d4 = False
        listen_rec = "Increase medium/strong epsilon or band_eq — still below -45 dB vs V4.1"

    profiles_doc = {
        name: {
            "identity_epsilon": p.identity_epsilon,
            "harmonic_gain_max": p.harmonic_gain_max,
            "fundamental_gain_max": p.fundamental_gain_max,
            "residual_gain_max": p.residual_gain_max,
            "rms_guard_max_db": p.rms_guard_max_db,
            "axis_gain_scale": p.axis_gain_scale,
            "band_eq_max_db": p.band_eq_max_db,
        }
        for name, p in STRENGTH_PROFILES.items()
    }

    report: Dict[str, Any] = {
        "stage": "5.1D",
        "title": "STK V4.1 identity strength sweep",
        "formula": {
            "base": "y_base = modal_body_hybrid_v4_1_full (endpoints unchanged)",
            "projection": "perceptual_axes = bounded blend of z_body physical/modal features (6 axes)",
            "shaping": "band EQ + harmonic gains from axes, then y = y_base + epsilon * bounded_residual",
            "perceptual_axes": list(PERCEPTUAL_AXIS_NAMES),
            "strength_profiles": profiles_doc,
        },
        "feature_projection": {
            "brightness_centroid": "high_body_color, share_air, hole/area, -top_damping",
            "low_mid_warmth": "low/mid body color, back share, depth, air volume",
            "high_freq_rolloff": "-Q fingerprint, high/mid color, top thickness",
            "attack_bloom": "bridge mobility/rank, -eff_mass, -mass_mixed",
            "decay_sustain": "Q spread, -Q fingerprint, top/back damping",
            "body_resonance_density": "modal density/near counts by band, rad_rank_median",
        },
        "critical_gate_db_vs_v41": LAYER_ACTIVE_GATE_DB,
        "medium_strong_layer_active": {"medium": medium_active, "strong": strong_active},
        "f0_continuous_no_note_names": True,
        "uses_v3_v2_as_base": False,
        "v4_1_base_preserved": True,
        "modes_run": list(modes),
        "notes": list(notes),
        "build_manifest": dict(build_manifest),
        "per_note_metrics": per_note,
        "strength_results": strength_results,
        "improvement_verdict": improvement,
        "character": character,
        "recommended_listen_mode": best_mode or "modal_body_hybrid_v4_1_identity_strong",
        "test_d4_next": test_d4,
        "listening_recommendation": listen_rec,
        "default_stk_note": "V4.1 remains production base; identity sweep is diagnostic-only",
        "distance_consistency_by_mode": dict(distance_by_mode),
        "fem_launched": False,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    out_md.write_text(_render_md(report), encoding="utf-8")
    return report


def _render_md(report: Mapping[str, Any]) -> str:
    lines = [
        "# Stage 5.1D — STK V4.1 identity strength sweep",
        "",
        f"Verdict: **{report.get('improvement_verdict')}** | Character: **{report.get('character')}**",
        f"Test D4 next: **{report.get('test_d4_next')}**",
        f"Recommended listen: **{report.get('recommended_listen_mode')}**",
        "",
        "## Strength profiles",
        "",
    ]
    for name, bounds in (report.get("formula", {}).get("strength_profiles") or {}).items():
        lines.append(f"- **{name}**: ε={bounds.get('identity_epsilon')}, "
                     f"harm_max={bounds.get('harmonic_gain_max')}, "
                     f"res_max={bounds.get('residual_gain_max')}, "
                     f"rms_guard={bounds.get('rms_guard_max_db')} dB")
    lines.extend(["", "## A3 metrics vs V4.1", ""])
    lines.append(
        "| mode | spec diff | rho | rms vs V4.1 (dB) | max abs diff | audible | clusters | pass |"
    )
    lines.append("|------|-----------|-----|------------------|--------------|---------|----------|------|")
    note = (report.get("notes") or ["A3"])[0]
    block = (report.get("per_note_metrics") or {}).get(note) or {}
    dc = report.get("distance_consistency_by_mode") or {}
    for mode in [V41_MODE, *IDENTITY_MODES]:
        m = (block.get("modes") or {}).get(mode) or {}
        if not m and mode != V41_MODE:
            continue
        comp = (block.get("comparisons") or {}).get(mode) or {}
        rho = (dc.get(mode) or {}).get("spearman_rho")
        lines.append(
            f"| {mode} | {m.get('spectral_differentiation')} | {rho} | "
            f"{m.get('rms_diff_db_vs_v41_median')} | {m.get('max_abs_diff_vs_v41_median')} | "
            f"{m.get('likely_audible_vs_v41')} | {comp.get('estimated_audible_clusters')} | "
            f"{comp.get('passed')} |"
        )
    lines.extend(
        [
            "",
            f"**Recommendation:** {report.get('listening_recommendation')}",
            "FEM launched: **no**",
        ]
    )
    return "\n".join(lines)
