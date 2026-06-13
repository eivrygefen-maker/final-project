#!/usr/bin/env python3
"""Stage 5.1G STK V4.1 maximal physical differentiation validation report."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from body_hybrid_v4_1_identity_space import estimate_audible_clusters

RMS_SPREAD_SOFT_DB = 2.0
RMS_SPREAD_MAX_DB = 4.0
DIFF_BEAT_V41_MIN = 1.05
DIFF_BEAT_F25_MIN = 1.001
RMS_AUDIBLE_LO = -30.0
RMS_AUDIBLE_HI = -22.0
RMS_AUDIBLE_FLOOR = -40.0

V41_MODE = "modal_body_hybrid_v4_1_full"
F25_MODE = "modal_body_hybrid_v4_1_identity_contrast_hybrid_25_75"
G_MODES = (
    "modal_body_hybrid_v4_1_identity_contrast_g_20_80",
    "modal_body_hybrid_v4_1_identity_contrast_g_25_75",
    "modal_body_hybrid_v4_1_identity_contrast_g_30_70",
    "modal_body_hybrid_v4_1_identity_contrast_g_25_75_decay",
    "modal_body_hybrid_v4_1_identity_contrast_g_25_75_bridge",
    "modal_body_hybrid_v4_1_identity_contrast_g_25_75_full",
)

VM_REFERENCE_A3 = {
    "v41_spec_diff": 0.072396,
    "f25_spec_diff": 0.075582,
    "f25_pct_vs_v41": 4.4,
    "contrast_strong_spec_diff": 0.073817,
}


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
    centroids = [float(s.get("spectral_centroid_hz") or 0) for s in segments]
    vs_vals = [
        float((s.get("vs_v41_reference") or {}).get("rms_diff_db_vs_reference"))
        for s in segments
        if (s.get("vs_v41_reference") or {}).get("rms_diff_db_vs_reference") is not None
    ]
    f0_vals = [float(s.get("fundamental_hz") or s.get("frequency_hz") or 0) for s in segments]
    f0_spread = round(max(f0_vals) - min(f0_vals), 4) if len(f0_vals) >= 2 else 0.0
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
        "fundamental_spread_hz": f0_spread,
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


def _artifact_likelihood(metrics: Mapping[str, Any]) -> str:
    """Heuristic: physical vs loudness/EQ artifact."""
    rms_spread = float(metrics.get("rms_spread_db") or 0)
    spec = float(metrics.get("spectral_differentiation") or 0)
    decay_spread = float(metrics.get("decay_slope_spread_db_per_s") or 0)
    attack_spread = float(metrics.get("attack_time_spread_ms") or 0)
    if rms_spread > RMS_SPREAD_SOFT_DB and spec > 0 and decay_spread < 1e-6 and attack_spread < 0.05:
        return "likely_loudness_dominant"
    if decay_spread > 0 or attack_spread > 0.1:
        return "likely_physical_envelope"
    if spec > 0 and rms_spread <= RMS_SPREAD_SOFT_DB:
        return "likely_spectral_physical"
    return "mixed_or_inconclusive"


def _compare_g_mode(
    *,
    v41m: Mapping[str, Any],
    f25m: Mapping[str, Any],
    gm: Mapping[str, Any],
    rho_v41: Optional[float],
    rho_f25: Optional[float],
    rho_g: Optional[float],
    audio_dists: Sequence[float],
    nn_report: Mapping[str, Any],
) -> Dict[str, Any]:
    diff_vs_v41 = gm["spectral_differentiation"] / max(v41m["spectral_differentiation"], 1e-9)
    diff_vs_f25 = gm["spectral_differentiation"] / max(f25m["spectral_differentiation"], 1e-9)
    rho_delta_v41 = None
    if rho_v41 is not None and rho_g is not None:
        rho_delta_v41 = round(float(rho_g) - float(rho_v41), 6)
    rho_delta_f25 = None
    if rho_f25 is not None and rho_g is not None:
        rho_delta_f25 = round(float(rho_g) - float(rho_f25), 6)
    rms_med = gm.get("rms_diff_db_vs_v41_median")
    rms_natural = (
        rms_med is not None
        and float(rms_med) > RMS_AUDIBLE_FLOOR
        and RMS_AUDIBLE_LO <= float(rms_med) <= RMS_AUDIBLE_HI
    )
    passed = (
        diff_vs_v41 >= DIFF_BEAT_V41_MIN
        and diff_vs_f25 >= DIFF_BEAT_F25_MIN
        and not gm["clipping_any"]
        and float(gm["rms_spread_db"]) <= RMS_SPREAD_MAX_DB
        and (nn_report.get("nn_preservation_rate") or 0) >= 0.5
        and float(gm.get("fundamental_spread_hz") or 0) < 0.5
    )
    return {
        "spectral_diff_vs_v41_ratio": round(diff_vs_v41, 4),
        "spectral_diff_vs_f25_ratio": round(diff_vs_f25, 4),
        "pct_vs_v41": round((diff_vs_v41 - 1.0) * 100.0, 3),
        "pct_vs_f25": round((diff_vs_f25 - 1.0) * 100.0, 3),
        "beats_f25_hybrid": diff_vs_f25 >= DIFF_BEAT_F25_MIN,
        "beats_v41_5pct": diff_vs_v41 >= DIFF_BEAT_V41_MIN,
        "rho_vs_v41": rho_delta_v41,
        "rho_vs_f25": rho_delta_f25,
        "estimated_audible_clusters": estimate_audible_clusters(audio_dists),
        "nn_preservation_rate": nn_report.get("nn_preservation_rate"),
        "rms_in_natural_band": rms_natural,
        "artifact_likelihood": _artifact_likelihood(gm),
        "passed": passed,
    }


def _architecture_awareness(
    *,
    v41m: Mapping[str, Any],
    f25m: Mapping[str, Any],
    best_g: Optional[Mapping[str, Any]],
    best_comp: Optional[Mapping[str, Any]],
    g_results: Mapping[str, Any],
) -> Dict[str, Any]:
    g_beats_f25 = bool((best_comp or {}).get("beats_f25_hybrid"))
    g_beats_v41 = bool((best_comp or {}).get("beats_v41_5pct"))
    v41_diff = float(v41m.get("spectral_differentiation") or 0)
    f25_diff = float(f25m.get("spectral_differentiation") or 0)
    g_diff = float((best_g or {}).get("spectral_differentiation") or 0)

    decay_modes = [m for m in G_MODES if "decay" in m or "full" in m]
    bridge_modes = [m for m in G_MODES if "bridge" in m or "full" in m]
    decay_help = False
    bridge_help = False
    for m in decay_modes:
        comp = (g_results.get(m) or {}).get("comparison") or {}
        if comp.get("beats_f25_hybrid"):
            decay_help = True
    for m in bridge_modes:
        comp = (g_results.get(m) or {}).get("comparison") or {}
        if comp.get("beats_f25_hybrid"):
            bridge_help = True

    ratio_only = [m for m in G_MODES if m.endswith(("20_80", "25_75", "30_70"))]
    ratio_best = max(
        (float((g_results.get(m) or {}).get("metrics", {}).get("spectral_differentiation") or 0) for m in ratio_only),
        default=0.0,
    )

    return {
        "overlay_headroom_still_exists": g_diff > f25_diff or ratio_best > f25_diff,
        "v41_still_carries_majority_of_identity": v41_diff / max(g_diff, 1e-9) > 0.85,
        "physical_components_create_new_differentiation": g_beats_f25,
        "residuals_loudness_only_risk": (best_g or {}).get("artifact_likelihood") == "likely_loudness_dominant",
        "string_body_coupling_increases_differentiation": bridge_help,
        "decay_bloom_shaping_helps": decay_help,
        "future_path_recommendation": (
            "continue_g_overlay_physical_tuning"
            if g_beats_f25 and g_beats_v41
            else (
                "revisit_body_response_first_with_revised_amplitude_law"
                if not g_beats_f25 and f25_diff >= v41_diff * 1.04
                else "continue_overlay_with_decay_bridge_focus"
            )
        ),
        "v41_compression_note": (
            "If G modes plateau below +7% vs V4.1 despite physical decay/bridge axes, "
            "V4.1 endpoint normalization may be compressing families — evaluate 5.2A body-response-first."
        ),
    }


def build_stage51g_identity_contrast_g_report(
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
    rho_f25 = (distance_by_mode.get(F25_MODE) or {}).get("spearman_rho")

    per_note: Dict[str, Any] = {}
    g_results: Dict[str, Any] = {}
    best_mode: Optional[str] = None
    best_spec = -1.0
    best_comp: Optional[Dict[str, Any]] = None

    for note in notes:
        v41_ns = _note_summary(mode_summaries, V41_MODE, note)
        f25_ns = _note_summary(mode_summaries, F25_MODE, note)
        v41m = _spread_metrics(v41_ns, list(v41_ns.get("segments") or []))
        f25m = _spread_metrics(f25_ns, list(f25_ns.get("segments") or []))
        rows: Dict[str, Any] = {V41_MODE: v41m, F25_MODE: f25m}
        comparisons: Dict[str, Any] = {}

        for mode in modes:
            if mode in (V41_MODE, F25_MODE):
                continue
            ns = _note_summary(mode_summaries, mode, note)
            gm = _spread_metrics(ns, list(ns.get("segments") or []))
            gm["artifact_likelihood"] = _artifact_likelihood(gm)
            rows[mode] = gm
            if mode in G_MODES:
                dist = (distance_by_mode.get(mode) or {}).get("audio_distances") or []
                comp = _compare_g_mode(
                    v41m=v41m,
                    f25m=f25m,
                    gm=gm,
                    rho_v41=rho_v41,
                    rho_f25=rho_f25,
                    rho_g=(distance_by_mode.get(mode) or {}).get("spearman_rho"),
                    audio_dists=dist,
                    nn_report=nn_by_mode.get(mode) or {},
                )
                comparisons[mode] = comp
                g_results[mode] = {"metrics": gm, "comparison": comp}
                spec = float(gm["spectral_differentiation"])
                if spec > best_spec:
                    best_spec = spec
                    best_mode = mode
                    best_comp = comp

        per_note[note] = {"modes": rows, "comparisons": comparisons}

    verdict = "not_useful"
    if best_comp and best_comp.get("beats_v41_5pct") and best_comp.get("beats_f25_hybrid"):
        verdict = "strong"
    elif best_comp and (best_comp.get("beats_f25_hybrid") or best_comp.get("beats_v41_5pct")):
        verdict = "marginal"

    arch = _architecture_awareness(
        v41m=per_note.get(notes[0], {}).get("modes", {}).get(V41_MODE, {}),
        f25m=per_note.get(notes[0], {}).get("modes", {}).get(F25_MODE, {}),
        best_g=(g_results.get(best_mode or "") or {}).get("metrics"),
        best_comp=best_comp,
        g_results=g_results,
    )

    listen = best_mode or "modal_body_hybrid_v4_1_identity_contrast_g_25_75_full"
    if verdict == "not_useful":
        listen_rec = f"G modes did not beat 5.1F hybrid_25_75 — listen {listen} on VM; compare decay vs bridge vs full"
    else:
        listen_rec = f"Listen {listen} — beats F25 on spec diff; verify rho/NN and natural decay/bloom on VM stitch"

    report: Dict[str, Any] = {
        "stage": "5.1G",
        "title": "STK V4.1 maximal physical guitar differentiation",
        "formula": {
            "base": "y_base = modal_body_hybrid_v4_1_full",
            "core": "res = a·identity_strong(y_base,z) + b·contrast_strong(y_base,dz)",
            "decay": "res(t) *= envelope(decay_axis(Q,damping,bridge,modal_density,participation))",
            "bridge": "res = blend(res, harmonic_shape(res, bridge_axis(mobility,rank,mass,density)))",
            "full": "25/75 + decay + bridge → audibility floor → RMS guard 2.75 dB",
            "ratio_modes": {"20_80": "a=0.20,b=0.80", "25_75": "a=0.25,b=0.75", "30_70": "a=0.30,b=0.70"},
        },
        "vm_reference_a3": VM_REFERENCE_A3,
        "f0_continuous_no_note_names": True,
        "v4_1_base_preserved": True,
        "v4_1_unchanged": True,
        "modes_run": list(modes),
        "notes": list(notes),
        "build_manifest": build_manifest,
        "per_note_metrics": per_note,
        "g_results": g_results,
        "best_g_mode": best_mode,
        "improvement_verdict": verdict,
        "architecture_awareness": arch,
        "recommended_listen_mode": listen,
        "listening_recommendation": listen_rec,
        "test_d4_next": False,
        "distance_consistency_by_mode": dict(distance_by_mode),
        "nearest_neighbor_by_mode": dict(nn_by_mode),
        "fem_launched": False,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    lines = [
        "# Stage 5.1G — STK V4.1 maximal physical differentiation",
        "",
        f"Verdict: **{verdict}** | Best G mode: **{best_mode or 'n/a'}**",
        "",
        "## A3 comparison",
        "",
        "| mode | spec diff | ρ | rms vs V4.1 (dB) | NN | decay spread | attack spread | clipping |",
        "|------|-----------|---|------------------|-----|--------------|---------------|---------|",
    ]
    for note in notes:
        for mode in modes:
            m = per_note.get(note, {}).get("modes", {}).get(mode, {})
            rho = (distance_by_mode.get(mode) or {}).get("spearman_rho")
            nn = (nn_by_mode.get(mode) or {}).get("nn_preservation_rate")
            lines.append(
                f"| {mode} | {m.get('spectral_differentiation')} | {rho} | "
                f"{m.get('rms_diff_db_vs_v41_median')} | {nn} | "
                f"{m.get('decay_slope_spread_db_per_s')} | {m.get('attack_time_spread_ms')} | "
                f"{m.get('clipping_any')} |"
            )
    if best_comp:
        lines.extend(
            [
                "",
                f"Best G vs V4.1: **{best_comp.get('pct_vs_v41')}%** | vs F25: **{best_comp.get('pct_vs_f25')}%**",
            ]
        )
    lines.extend(
        [
            "",
            "## Architecture awareness",
            f"- Overlay headroom still exists: **{arch.get('overlay_headroom_still_exists')}**",
            f"- V4.1 carries majority: **{arch.get('v41_still_carries_majority_of_identity')}**",
            f"- Physical components beat F25: **{arch.get('physical_components_create_new_differentiation')}**",
            f"- Decay/bloom helps: **{arch.get('decay_bloom_shaping_helps')}**",
            f"- Bridge coupling helps: **{arch.get('string_body_coupling_increases_differentiation')}**",
            f"- Future path: **{arch.get('future_path_recommendation')}**",
            "",
            f"**Recommendation:** {listen_rec}",
            f"FEM launched: **no**",
        ]
    )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
