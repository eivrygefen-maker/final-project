#!/usr/bin/env python3
"""Stage 5.1B STK V4.1 transition-band validation report."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from body_hybrid_v4_1 import V4_1_F_HIGH, V4_1_F_LOW, radiation_blend_weight_f0

TRANSITION_F_LOW = V4_1_F_LOW
TRANSITION_F_HIGH = V4_1_F_HIGH
NEAR_BASELINE_W_MAX = 0.25
NEAR_V1_W_MIN = 0.75
RMS_SPREAD_MAX_DB = 4.0
LOUDNESS_JUMP_MAX_DB = 6.0
DIFF_FRACTION_OF_BEST = 0.90
CLIP_PEAK_DBFS = -0.5


def _diff(mode_summaries: Mapping[str, Any], mode: str, note: str) -> float:
    notes = (mode_summaries.get(mode) or {}).get("notes") or {}
    return float((notes.get(note) or {}).get("spectral_differentiation") or 0.0)


def _rms_spread(mode_summaries: Mapping[str, Any], mode: str, note: str) -> float:
    notes = (mode_summaries.get(mode) or {}).get("notes") or {}
    return float((notes.get(note) or {}).get("rms_spread_db") or 0.0)


def classify_crossfade_mode(w_rad: float) -> str:
    w = float(w_rad)
    if w <= NEAR_BASELINE_W_MAX:
        return "near_baseline"
    if w >= NEAR_V1_W_MIN:
        return "near_v1"
    return "transition"


def _note_w_rad_stats(
    transition_rows: Sequence[Mapping[str, Any]],
    note: str,
    *,
    frequency_hz: Optional[float] = None,
) -> Dict[str, Any]:
    rows = [r for r in transition_rows if r.get("note") == note]
    freq = frequency_hz
    if freq is None and rows:
        freq = float(rows[0].get("frequency_hz") or 0.0)
    if not rows:
        w = radiation_blend_weight_f0(freq) if freq else 0.0
        return {
            "frequency_hz": freq,
            "mean_w_rad": round(w, 6),
            "min_w_rad": round(w, 6),
            "max_w_rad": round(w, 6),
            "crossfade_mode": classify_crossfade_mode(w),
            "sample_count": 0,
        }
    wvals = [float(r.get("w_rad") or 0) for r in rows]
    mean_w = sum(wvals) / len(wvals)
    return {
        "frequency_hz": freq or rows[0].get("frequency_hz"),
        "mean_w_rad": round(mean_w, 6),
        "min_w_rad": round(min(wvals), 6),
        "max_w_rad": round(max(wvals), 6),
        "crossfade_mode": classify_crossfade_mode(mean_w),
        "sample_count": len(rows),
    }


def _clipping_from_segments(segments: Sequence[Mapping[str, Any]]) -> bool:
    for seg in segments:
        peak = seg.get("output_peak_dbfs")
        if peak is not None and float(peak) > CLIP_PEAK_DBFS:
            return True
        if seg.get("clipping_detected"):
            return True
    return False


def _recommendation_for_note(gate: Mapping[str, Any], w_stats: Mapping[str, Any]) -> str:
    if not gate.get("passed"):
        if gate.get("clipping"):
            return "stop"
        if not gate.get("differentiation_ok"):
            mode = w_stats.get("crossfade_mode")
            if mode == "transition":
                return "add_transition_only_color_correction"
            return "move_f_low_f_high"
        if not gate.get("rms_ok"):
            return "add_transition_only_color_correction"
    return "keep_thresholds"


def build_stage51b_transition_report(
    *,
    mode_summaries: Mapping[str, Any],
    transition_rows: Sequence[Mapping[str, Any]],
    core_full_identity: Mapping[str, Any],
    notes: Sequence[str],
    modes: Sequence[str],
    build_manifest: Mapping[str, Any],
    out_json: Path,
    out_md: Path,
) -> Dict[str, Any]:
    base = "baseline_current"
    v1 = "modal_radiation_color_v1"
    v41_core = "modal_body_hybrid_v4_1_core"
    v41_full = "modal_body_hybrid_v4_1_full"

    per_note: Dict[str, Any] = {}
    gates: Dict[str, Any] = {}
    recommendations: Dict[str, str] = {}

    note_freqs = dict((build_manifest.get("note_frequencies_hz") or {}))

    for note in notes:
        freq_hz = float(note_freqs.get(note) or 0.0)
        w_stats = _note_w_rad_stats(transition_rows, note, frequency_hz=freq_hz or None)
        baseline_d = _diff(mode_summaries, base, note)
        v1_d = _diff(mode_summaries, v1, note)
        v41d = _diff(mode_summaries, v41_full, note)
        best_endpoint = max(baseline_d, v1_d)
        ratio_best = round(v41d / max(best_endpoint, 1e-12), 6)
        rms = _rms_spread(mode_summaries, v41_full, note)

        core_note = ((mode_summaries.get(v41_core) or {}).get("notes") or {}).get(note) or {}
        full_note = ((mode_summaries.get(v41_full) or {}).get("notes") or {}).get(note) or {}
        segs = list(full_note.get("segments") or [])
        clipping = _clipping_from_segments(segs)

        diff_ok = v41d >= DIFF_FRACTION_OF_BEST * best_endpoint - 1e-9
        rms_ok = rms <= RMS_SPREAD_MAX_DB
        passed = diff_ok and rms_ok and not clipping

        gate = {
            "frequency_hz": freq_hz or w_stats.get("frequency_hz"),
            "w_rad": w_stats,
            "baseline_spectral_differentiation": baseline_d,
            "radiation_v1_spectral_differentiation": v1_d,
            "v4_1_spectral_differentiation": v41d,
            "v4_1_core_spectral_differentiation": _diff(mode_summaries, v41_core, note),
            "ratio_to_best_endpoint": ratio_best,
            "best_endpoint": "baseline_current" if baseline_d >= v1_d else "modal_radiation_color_v1",
            "rms_spread_db": rms,
            "clipping": clipping,
            "differentiation_ok": diff_ok,
            "rms_ok": rms_ok,
            "passed": passed,
            "crossfade_mode": w_stats.get("crossfade_mode"),
            "rule": f"v4.1 >= {DIFF_FRACTION_OF_BEST}*max(baseline,v1), no clipping, rms_spread<={RMS_SPREAD_MAX_DB}dB",
        }
        gates[note] = gate
        recommendations[note] = _recommendation_for_note(gate, w_stats)

        per_note[note] = {
            **gate,
            "v4_full_optional": _diff(mode_summaries, "modal_body_hybrid_v4_full", note)
            if "modal_body_hybrid_v4_full" in mode_summaries
            else None,
            "v3_full_optional": _diff(mode_summaries, "modal_body_signature_v3_full", note)
            if "modal_body_signature_v3_full" in mode_summaries
            else None,
        }

    v41_means = []
    for note in notes:
        full_note = ((mode_summaries.get(v41_full) or {}).get("notes") or {}).get(note) or {}
        for seg in full_note.get("segments") or []:
            rms_db = seg.get("final_rms_dbfs")
            if rms_db is not None:
                v41_means.append(float(rms_db))
    loudness_jump_db = round(max(v41_means) - min(v41_means), 4) if len(v41_means) >= 2 else 0.0
    loudness_ok = loudness_jump_db <= LOUDNESS_JUMP_MAX_DB

    all_pass = all(g.get("passed") for g in gates.values()) and loudness_ok
    core_full_ok = bool(core_full_identity.get("core_full_identical"))
    any_stop = any(r == "stop" for r in recommendations.values())
    any_move = any(r == "move_f_low_f_high" for r in recommendations.values())
    any_color = any(r == "add_transition_only_color_correction" for r in recommendations.values())

    if any_stop:
        overall_rec = "stop"
        default_stk = "NO — fix clipping or critical failure first"
    elif all_pass and core_full_ok:
        overall_rec = "keep_thresholds"
        default_stk = "YES — V4.1 ready as default STK candidate (pending listening on VM)"
    elif any_move:
        overall_rec = "move_f_low_f_high"
        default_stk = "NOT YET — tune transition band first"
    elif any_color:
        overall_rec = "add_transition_only_color_correction"
        default_stk = "NOT YET — endpoints OK; transition needs work"
    else:
        overall_rec = "keep_thresholds"
        default_stk = "DIAGNOSTIC_ONLY — review per-note gates"

    transition_smooth = all_pass and not any_stop and loudness_ok

    report: Dict[str, Any] = {
        "stage": "5.1B",
        "title": "STK V4.1 transition-band validation",
        "transition_band_hz": [TRANSITION_F_LOW, TRANSITION_F_HIGH],
        "formula": {
            "w_rad": f"smoothstep({TRANSITION_F_LOW} Hz, {TRANSITION_F_HIGH} Hz, f0)",
            "crossfade_modes": {
                "near_baseline": f"w_rad <= {NEAR_BASELINE_W_MAX}",
                "transition": f"{NEAR_BASELINE_W_MAX} < w_rad < {NEAR_V1_W_MIN}",
                "near_v1": f"w_rad >= {NEAR_V1_W_MIN}",
            },
        },
        "f0_continuous_no_note_names": True,
        "uses_v3_v2_as_base": False,
        "modes_run": list(modes),
        "notes": list(notes),
        "build_manifest": dict(build_manifest),
        "per_note_comparison": per_note,
        "acceptance_gates": gates,
        "per_note_recommendations": recommendations,
        "core_full_identity": core_full_identity,
        "loudness": {
            "v4_1_rms_jump_across_notes_db": loudness_jump_db,
            "loudness_ok": loudness_ok,
            "max_jump_allowed_db": LOUDNESS_JUMP_MAX_DB,
        },
        "transition_smooth": transition_smooth,
        "overall_recommendation": overall_rec,
        "default_stk_model_recommendation": default_stk,
        "thresholds_160_320_recommendation": (
            "keep"
            if overall_rec == "keep_thresholds" and all_pass
            else ("adjust" if any_move else "review")
        ),
        "transition_rows_sample": list(transition_rows[:15]),
        "fem_launched": False,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(_render_md(report), encoding="utf-8")
    return report


def _render_md(report: Mapping[str, Any]) -> str:
    lines = [
        "# Stage 5.1B — STK V4.1 transition-band validation",
        "",
        f"Transition band: **{report.get('transition_band_hz')}** Hz",
        f"Transition smooth: **{report.get('transition_smooth')}**",
        f"Thresholds 160/320: **{report.get('thresholds_160_320_recommendation')}**",
        f"Default STK: **{report.get('default_stk_model_recommendation')}**",
        "",
        "## Per-note comparison",
        "",
        "| note | w_rad mean | crossfade | baseline diff | v1 diff | v4.1 diff | ratio best | rms spread | clipping | pass |",
        "|------|------------|-----------|---------------|---------|-----------|------------|------------|----------|------|",
    ]
    for note, row in (report.get("per_note_comparison") or {}).items():
        w = row.get("w_rad") or {}
        lines.append(
            f"| {note} | {w.get('mean_w_rad')} | {row.get('crossfade_mode')} | "
            f"{row.get('baseline_spectral_differentiation')} | {row.get('radiation_v1_spectral_differentiation')} | "
            f"{row.get('v4_1_spectral_differentiation')} | {row.get('ratio_to_best_endpoint')} | "
            f"{row.get('rms_spread_db')} | {row.get('clipping')} | {row.get('passed')} |"
        )
    lines.extend(
        [
            "",
            "## Recommendations",
        ]
    )
    for note, rec in (report.get("per_note_recommendations") or {}).items():
        lines.append(f"- **{note}**: {rec}")
    lines.append("")
    lines.append(f"Overall: **{report.get('overall_recommendation')}**")
    lines.append("FEM launched: **no**")
    return "\n".join(lines)
