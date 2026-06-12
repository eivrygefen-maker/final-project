#!/usr/bin/env python3
"""Stage 4.8 timbre decomposition analysis and report generation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from diagnostic_synthesis import (
    _spectral_features,
    average_spectral_similarity,
    summarize_comparison_note,
)

STITCH_LAYER_ALIASES: Dict[str, str] = {
    "string_only": "string_only",
    "body_only_raw_pre_norm": "body_only_raw",
    "body_only_final_norm": "body_only_final_norm",
    "near_modes_only": "near_modes_only",
    "far_background_modes_only": "far_background_only",
    "radiation_only_weighted_body": "radiation_body_only",
    "full_mix_baseline": "full_mix_baseline",
    "full_mix_radiation_v1": "full_mix_radiation_v1",
    "full_mix_candidate_balance": "full_mix_candidate_balance",
}

CRITICAL_LAYERS: Tuple[str, ...] = (
    "body_only_raw_pre_norm",
    "body_only_final_norm",
    "full_mix_baseline",
    "full_mix_radiation_v1",
)


def _spectral_flux(audio: np.ndarray, sample_rate: int) -> float:
    x = np.asarray(audio, dtype=np.float64)
    if x.size < 64:
        return 0.0
    hop = max(x.size // 32, 1)
    flux_vals: List[float] = []
    prev = None
    for i in range(0, x.size - hop, hop):
        frame = x[i : i + hop]
        mag = np.abs(np.fft.rfft(frame * np.hanning(len(frame))))
        if prev is not None:
            flux_vals.append(float(np.sum(np.abs(mag - prev))))
        prev = mag
    return float(np.mean(flux_vals)) if flux_vals else 0.0


def _attack_time_ms(audio: np.ndarray, sample_rate: int) -> float:
    x = np.asarray(audio, dtype=np.float64)
    if x.size < 8:
        return 0.0
    env = np.abs(x)
    peak = float(np.max(env))
    if peak < 1e-12:
        return 0.0
    thr = 0.9 * peak
    idx = int(np.argmax(env >= thr))
    return round(1000.0 * idx / float(sample_rate), 3)


def summarize_layer_across_samples(
    rows: Sequence[Mapping[str, Any]],
    *,
    layer: str,
    note: str,
    sample_rate: int = 44100,
) -> Dict[str, Any]:
    layer_rows = [r for r in rows if r.get("layer") == layer and r.get("note") == note]
    audios = [np.asarray(r.get("audio"), dtype=np.float64) for r in layer_rows if r.get("audio") is not None]
    avg_sim = average_spectral_similarity(audios) if audios else 1.0
    for r in layer_rows:
        if "final_rms_dbfs" not in r and "output_rms_dbfs" in r:
            r["final_rms_dbfs"] = r["output_rms_dbfs"]
    note_summary = summarize_comparison_note(layer_rows)
    lows = [float(r.get("spectral_low_energy") or 0.0) for r in layer_rows]
    mids = [float(r.get("spectral_mid_energy") or 0.0) for r in layer_rows]
    highs = [float(r.get("spectral_high_energy") or 0.0) for r in layer_rows]
    flux_vals = [_spectral_flux(a, sample_rate) for a in audios]
    attack_vals = [_attack_time_ms(a, sample_rate) for a in audios]
    return {
        "layer": layer,
        "note": note,
        "sample_count": len(layer_rows),
        "average_spectral_similarity": avg_sim,
        "spectral_differentiation": round(1.0 - avg_sim, 6),
        "spectral_centroid_spread_hz": note_summary.get("spectral_centroid_spread_hz", 0.0),
        "low_energy_spread": round(max(lows) - min(lows), 6) if len(lows) >= 2 else 0.0,
        "mid_energy_spread": round(max(mids) - min(mids), 6) if len(mids) >= 2 else 0.0,
        "high_energy_spread": round(max(highs) - min(highs), 6) if len(highs) >= 2 else 0.0,
        "rms_spread_db": note_summary.get("rms_spread_db", 0.0),
        "raw_body_rms_spread": note_summary.get("raw_body_rms_spread", 0.0),
        "decay_slope_spread_db_per_s": note_summary.get("decay_slope_spread_db_per_s", 0.0),
        "body_to_string_ratio_spread": note_summary.get("body_to_string_ratio_spread", 0.0),
        "note_reward_spread": note_summary.get("note_reward_spread", 0.0),
        "spectral_flux_spread": round(max(flux_vals) - min(flux_vals), 6) if len(flux_vals) >= 2 else 0.0,
        "attack_time_spread_ms": round(max(attack_vals) - min(attack_vals), 3) if len(attack_vals) >= 2 else 0.0,
    }


def pick_representative_samples(
    rows: Sequence[Mapping[str, Any]],
    *,
    note: str,
    layer: str = "body_only_raw_pre_norm",
) -> Dict[str, str]:
    layer_rows = [r for r in rows if r.get("note") == note and r.get("layer") == layer]
    if not layer_rows:
        layer_rows = [r for r in rows if r.get("note") == note]
    scored = sorted(
        layer_rows,
        key=lambda r: float(r.get("note_reward_score") or r.get("raw_body_rms_before_normalization") or 0.0),
        reverse=True,
    )
    if len(scored) < 3:
        ids = [str(r.get("sample_id") or "") for r in scored]
        while len(ids) < 3 and ids:
            ids.append(ids[-1])
        return {
            "high_body_signature": ids[0] if ids else "",
            "median": ids[len(ids) // 2] if ids else "",
            "low_body_signature": ids[-1] if ids else "",
        }
    mid = len(scored) // 2
    return {
        "high_body_signature": str(scored[0].get("sample_id") or ""),
        "median": str(scored[mid].get("sample_id") or ""),
        "low_body_signature": str(scored[-1].get("sample_id") or ""),
    }


def correlate_layer_with_data(
    rows: Sequence[Mapping[str, Any]],
    *,
    layer: str,
    note: str,
) -> Dict[str, Any]:
    layer_rows = [r for r in rows if r.get("layer") == layer and r.get("note") == note]
    if len(layer_rows) < 3:
        return {"layer": layer, "note": note, "correlations": {}}
    fields = [
        "bridge_excitation_mean",
        "radiation_proxy_mean",
        "mic_proxy_mean",
        "top_effective_mass_proxy",
        "back_effective_mass_proxy",
        "body_air_volume_proxy",
        "bridge_mobility_proxy",
        "geometry.length",
        "geometry.width",
        "geometry.depth",
        "mode_q_median",
        "material_damping_median",
    ]
    centroid = np.asarray([float(r.get("spectral_centroid_hz") or 0.0) for r in layer_rows])
    corrs: Dict[str, float] = {}
    for field in fields:
        vals = []
        for r in layer_rows:
            attr = r.get("data_attribution") or {}
            if field in attr:
                vals.append(float(attr[field]))
            elif field in r:
                vals.append(float(r[field]))
            elif field.startswith("geometry."):
                gkey = field.split(".", 1)[1]
                vals.append(float((r.get("parameters") or {}).get(f"geometry.{gkey}") or 0.0))
            else:
                vals.append(float("nan"))
        x = np.asarray(vals, dtype=np.float64)
        if np.any(~np.isfinite(x)) or np.std(x) < 1e-12 or np.std(centroid) < 1e-12:
            continue
        c = float(np.corrcoef(x, centroid)[0, 1])
        if np.isfinite(c):
            corrs[field] = round(c, 4)
    ranked = sorted(corrs.items(), key=lambda kv: abs(kv[1]), reverse=True)
    return {
        "layer": layer,
        "note": note,
        "correlations_with_spectral_centroid": dict(ranked[:8]),
        "top_fields": [k for k, _ in ranked[:5]],
    }


def build_stage48_report(
    *,
    build_manifest: Mapping[str, Any],
    analysis_rows: Sequence[Mapping[str, Any]],
    notes: Sequence[str],
    out_json: Path,
    out_md: Path,
) -> Dict[str, Any]:
    layer_metrics: Dict[str, Dict[str, Any]] = {}
    for note in notes:
        for layer in STITCH_LAYER_ALIASES:
            key = f"{note}/{layer}"
            layer_metrics[key] = summarize_layer_across_samples(analysis_rows, layer=layer, note=note)

    representatives: Dict[str, Any] = {
        note: pick_representative_samples(analysis_rows, note=note) for note in notes
    }

    critical: Dict[str, Any] = {}
    for note in notes:
        critical[note] = {
            layer: layer_metrics.get(f"{note}/{layer}", {}) for layer in CRITICAL_LAYERS
        }

    attribution_body: Dict[str, Any] = {}
    attribution_full: Dict[str, Any] = {}
    for note in notes:
        attribution_body[note] = correlate_layer_with_data(
            analysis_rows, layer="body_only_raw_pre_norm", note=note
        )
        attribution_full[note] = correlate_layer_with_data(
            analysis_rows, layer="full_mix_baseline", note=note
        )

    per_note_ranking: Dict[str, List[Tuple[str, float]]] = {}
    for note in notes:
        ranked = sorted(
            [
                (layer, float(layer_metrics.get(f"{note}/{layer}", {}).get("spectral_differentiation") or 0.0))
                for layer in STITCH_LAYER_ALIASES
            ],
            key=lambda x: x[1],
            reverse=True,
        )
        per_note_ranking[note] = ranked

    def _diff(note: str, layer: str) -> float:
        return float(layer_metrics.get(f"{note}/{layer}", {}).get("spectral_differentiation") or 0.0)

    body_raw_a2 = _diff("A2", "body_only_raw_pre_norm")
    full_a2 = _diff("A2", "full_mix_baseline")
    norm_impact_a2 = body_raw_a2 - _diff("A2", "body_only_final_norm")

    answers = {
        "body_only_differs_clearly": body_raw_a2 > 0.0008 or any(
            _diff(n, "body_only_raw_pre_norm") > 0.001 for n in notes
        ),
        "full_mix_hides_body_differences": full_a2 < body_raw_a2 * 0.65,
        "normalization_reduces_differences": norm_impact_a2 > 0.0002,
        "highest_differentiation_layer": {
            n: per_note_ranking[n][0][0] if per_note_ranking[n] else None for n in notes
        },
        "lowest_differentiation_layer": {
            n: per_note_ranking[n][-1][0] if per_note_ranking[n] else None for n in notes
        },
        "a2_string_fundamental_dominance": _diff("A2", "string_only") < _diff("A2", "body_only_raw_pre_norm") * 0.5,
        "far_background_useful_alone": _diff("A2", "far_background_modes_only") > 0.0005,
        "radiation_timbre_vs_loudness": (
            "mostly_loudness"
            if abs(_diff("A4", "full_mix_radiation_v1") - _diff("A4", "full_mix_baseline")) < 0.0003
            else "timbre_and_level"
        ),
        "data_sufficient_for_audible_differences": body_raw_a2 > 0.0005,
        "recommended_next_stk_change": (
            "Per-mode amplitude/radiation transmittance with weaker final RMS normalization; "
            "continuous f0 body/string balance; bridge-gated radiation (v2) plus geometry bridge mobility probe."
        ),
        "bridge_mobility_explains_extra_variance": any(
            abs((attribution_body.get(n) or {}).get("correlations_with_spectral_centroid", {}).get("bridge_mobility_proxy", 0.0)) > 0.35
            for n in notes
        ),
    }

    report: Dict[str, Any] = {
        "stage": "4.8",
        "title": "Timbre decomposition listening test",
        "build_manifest": dict(build_manifest),
        "notes": list(notes),
        "representative_samples": representatives,
        "layer_metrics": layer_metrics,
        "critical_layer_comparison": critical,
        "data_to_layer_attribution": {
            "body_only": attribution_body,
            "full_mix_baseline": attribution_full,
        },
        "listening_pack_paths": build_manifest.get("listening_packs"),
        "answers": answers,
        "normalization_impact": {
            note: {
                "raw_vs_final_norm": round(
                    _diff(note, "body_only_raw_pre_norm") - _diff(note, "body_only_final_norm"),
                    6,
                ),
                "raw_vs_full_mix": round(
                    _diff(note, "body_only_raw_pre_norm") - _diff(note, "full_mix_baseline"),
                    6,
                ),
            }
            for note in notes
        },
        "fem_launched": False,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(_render_md(report), encoding="utf-8")
    return report


def _render_md(report: Mapping[str, Any]) -> str:
    lines = [
        "# Stage 4.8 — Timbre decomposition report",
        "",
        f"Output: `{report.get('build_manifest', {}).get('out_dir', '')}`",
        "",
        "## Representative samples",
    ]
    for note, reps in (report.get("representative_samples") or {}).items():
        lines.append(f"### {note}")
        for role, sid in reps.items():
            lines.append(f"- **{role}**: `{sid}`")
        lines.append("")

    lines.append("## Critical layer comparison")
    for note, layers in (report.get("critical_layer_comparison") or {}).items():
        lines.append(f"### {note}")
        for layer, metrics in layers.items():
            diff = metrics.get("spectral_differentiation", 0.0)
            lines.append(f"- `{layer}`: spectral_differentiation={diff}")
        lines.append("")

    lines.append("## Answers")
    for key, val in (report.get("answers") or {}).items():
        lines.append(f"1. **{key}**: {val}")
    lines.append("")

    lines.append("## Normalization impact")
    for note, impact in (report.get("normalization_impact") or {}).items():
        lines.append(f"- **{note}**: raw→final_norm Δdiff={impact.get('raw_vs_final_norm')}, raw→full_mix Δdiff={impact.get('raw_vs_full_mix')}")
    lines.append("")

    lines.append("## Confirmation")
    lines.append("- FEM solve launched: **no**")
    lines.append("- ROM retrain: **no**")
    lines.append("- Production model change: **no** (diagnostic only)")
    return "\n".join(lines)
