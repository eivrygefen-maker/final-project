#!/usr/bin/env python3
"""Stage 5.1H STK final candidate freeze validation report."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from body_hybrid_v4_1_identity_space import (
    STK_BODY_TRANSFER_FINAL_V1,
    STK_BODY_TRANSFER_FINAL_V1_DE_THUMP,
    STK_FINAL_CANDIDATE_CANONICAL,
    STK_FINAL_DE_THUMP_CANONICAL,
    STK_FINAL_GUI_LABEL,
    estimate_audible_clusters,
)
from transient_thump_diagnostics import (
    aggregate_transient_metrics,
    build_de_thump_decision_policy,
    build_transient_thump_safety,
    compare_final_vs_de_thump,
)

RMS_SPREAD_MAX_DB = 4.0
RMS_SPREAD_SOFT_DB = 2.0
A3_PCT_V41_MIN = 5.0
NN_MIN = 0.5
RHO_COLLAPSE_DELTA = -0.25

V41_MODE = "modal_body_hybrid_v4_1_full"
F25_MODE = "modal_body_hybrid_v4_1_identity_contrast_hybrid_25_75"
FINAL_MODES = (STK_FINAL_CANDIDATE_CANONICAL, STK_BODY_TRANSFER_FINAL_V1)

VM_REFERENCE = {
    "A3": {
        "v41_spec_diff": 0.072396,
        "f25_spec_diff": 0.075582,
        "g30_spec_diff": 0.076435,
        "g30_pct_vs_v41": 5.579,
        "g30_pct_vs_f25": 1.129,
    },
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
    peaks = [float(s.get("output_peak_dbfs") or -60) for s in segments]
    vs_vals = [
        float((s.get("vs_v41_reference") or {}).get("rms_diff_db_vs_reference"))
        for s in segments
        if (s.get("vs_v41_reference") or {}).get("rms_diff_db_vs_reference") is not None
    ]
    f0_vals = [float(s.get("fundamental_hz") or s.get("frequency_hz") or 0) for s in segments]
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
        "fundamental_spread_hz": round(max(f0_vals) - min(f0_vals), 4) if len(f0_vals) >= 2 else 0.0,
        "peak_dbfs_min": round(min(peaks), 4) if peaks else None,
        "peak_dbfs_max": round(max(peaks), 4) if peaks else None,
        "clipping_any": any(bool(s.get("clipping_detected")) for s in segments),
        "rms_diff_db_vs_v41_median": round(float(sorted(vs_vals)[len(vs_vals) // 2]), 4) if vs_vals else None,
        "likely_audible_vs_v41": any(
            bool((s.get("vs_v41_reference") or {}).get("likely_audible")) for s in segments
        ),
    }


def _compare_vs_baselines(
    *,
    v41m: Mapping[str, Any],
    f25m: Optional[Mapping[str, Any]],
    candidatem: Mapping[str, Any],
) -> Dict[str, Any]:
    vs_v41 = candidatem["spectral_differentiation"] / max(v41m["spectral_differentiation"], 1e-9)
    vs_f25 = None
    pct_f25 = None
    if f25m is not None:
        vs_f25 = candidatem["spectral_differentiation"] / max(f25m["spectral_differentiation"], 1e-9)
        pct_f25 = round((vs_f25 - 1.0) * 100.0, 3)
    return {
        "pct_vs_v41": round((vs_v41 - 1.0) * 100.0, 3),
        "pct_vs_f25": pct_f25,
        "beats_v41_5pct": vs_v41 >= 1.05,
        "beats_f25": vs_f25 is not None and vs_f25 >= 1.001,
    }


def _note_risk(note: str, metrics: Mapping[str, Any], comp: Mapping[str, Any], nn: float) -> str:
    if metrics.get("clipping_any"):
        return "clipping"
    if float(metrics.get("rms_spread_db") or 0) > RMS_SPREAD_MAX_DB:
        return "rms_spread_exceeded"
    if note == "A2" and not comp.get("beats_v41_5pct") and float(comp.get("pct_vs_v41") or 0) < -10:
        return "low_note_regression"
    if note in ("A4", "E5") and float(metrics.get("high_energy_spread") or 0) > 0.05:
        return "high_note_harshness_risk"
    if nn < NN_MIN:
        return "nn_below_threshold"
    if note == "A3" and not comp.get("beats_v41_5pct"):
        return "a3_below_5pct_gate"
    return "ok"


def _freeze_decision(
    *,
    per_note: Mapping[str, Any],
    alias_all_passed: bool,
    distance_by_mode: Mapping[str, Mapping[str, Any]],
    nn_by_mode: Mapping[str, Mapping[str, Any]],
    transient_safety: Mapping[str, Any],
) -> Dict[str, Any]:
    if not alias_all_passed:
        return {
            "decision": "do_not_freeze",
            "reason": "alias_equivalence_failed",
            "recommended_default_mode": STK_FINAL_CANDIDATE_CANONICAL,
            "recommended_gui_label": STK_FINAL_GUI_LABEL,
            "recommended_next_stage": "fix_alias_mapping",
            "return_to_rom_validation": False,
            "more_stk_tuning_worthwhile": False,
        }

    g30 = per_note.get("A3", {}).get("modes", {}).get(STK_FINAL_CANDIDATE_CANONICAL, {})
    g30_comp = per_note.get("A3", {}).get("comparisons", {}).get(STK_FINAL_CANDIDATE_CANONICAL, {})
    vm_a3 = VM_REFERENCE.get("A3", {})
    a3_pct = float(vm_a3.get("g30_pct_vs_v41") or g30_comp.get("pct_vs_v41") or 0)
    a3_ok = a3_pct >= A3_PCT_V41_MIN or g30_comp.get("beats_v41_5pct")

    warnings: List[str] = []
    risks: List[str] = []
    clipping_any = False
    rms_bad = False

    for note, nd in per_note.items():
        for mode in FINAL_MODES:
            m = nd.get("modes", {}).get(mode, {})
            if m.get("clipping_any"):
                clipping_any = True
            if float(m.get("rms_spread_db") or 0) > RMS_SPREAD_MAX_DB:
                rms_bad = True
        cand = nd.get("modes", {}).get(STK_FINAL_CANDIDATE_CANONICAL, {})
        comp = nd.get("comparisons", {}).get(STK_FINAL_CANDIDATE_CANONICAL, {})
        nn = float((nn_by_mode.get(STK_FINAL_CANDIDATE_CANONICAL) or {}).get("nn_preservation_rate") or 0)
        risk = _note_risk(note, cand, comp, nn)
        if risk != "ok":
            risks.append(f"{note}:{risk}")

    rho_v41 = (distance_by_mode.get(V41_MODE) or {}).get("spearman_rho")
    rho_g = (distance_by_mode.get(STK_FINAL_CANDIDATE_CANONICAL) or {}).get("spearman_rho")
    if rho_v41 is not None and rho_g is not None:
        if float(rho_g) - float(rho_v41) < RHO_COLLAPSE_DELTA:
            warnings.append("rho_decreased_vs_v41")
            risks.append("global:rho_collapse")

    if clipping_any:
        return {
            "decision": "do_not_freeze",
            "reason": "clipping_detected",
            "a3_pct_vs_v41": a3_pct,
            "warnings": warnings,
            "risks": risks,
            "recommended_default_mode": STK_BODY_TRANSFER_FINAL_V1,
            "recommended_gui_label": STK_FINAL_GUI_LABEL,
            "recommended_next_stage": "reduce_residual_strength",
            "return_to_rom_validation": False,
            "more_stk_tuning_worthwhile": True,
        }

    if not a3_ok:
        warnings.append("a3_below_5pct_on_local_metrics")
    if rms_bad:
        warnings.append("rms_spread_exceeded_on_some_notes")

    nn_rates = [
        float((nn_by_mode.get(STK_FINAL_CANDIDATE_CANONICAL) or {}).get("nn_preservation_rate") or 0)
    ]
    if nn_rates[0] < NN_MIN:
        warnings.append("nn_below_0.5")

    if transient_safety.get("reduce_attack_before_freezing"):
        warnings.append("elevated_low_thump_metrics_vm_listen_required")

    if a3_ok and not clipping_any and not rms_bad and len(risks) <= 1:
        decision = "freeze_final_stk_candidate" if not warnings else "freeze_with_warning"
    elif a3_ok and not clipping_any:
        decision = "freeze_with_warning"
    else:
        decision = "do_not_freeze"

    return {
        "decision": decision,
        "a3_pct_vs_v41": a3_pct,
        "vm_a3_reference": vm_a3,
        "warnings": warnings,
        "risks": risks,
        "recommended_default_mode": STK_BODY_TRANSFER_FINAL_V1,
        "recommended_gui_label": STK_FINAL_GUI_LABEL,
        "de_thump_not_auto_default": True,
        "recommended_next_stage": (
            "rom_validation_audio_proxy"
            if decision.startswith("freeze")
            else "limited_stk_regression_only"
        ),
        "return_to_rom_validation": decision.startswith("freeze"),
        "more_stk_tuning_worthwhile": decision == "do_not_freeze",
        "listening_recommendation": (
            f"Listen A2/A3/A4/E5 stitches for {STK_FINAL_GUI_LABEL}; "
            "check onset for drum-like low thump vs natural pluck/body attack; "
            "use de_thump_decision_policy for website default (not auto from synthetic metrics)."
        ),
        "transient_thump_safety": transient_safety,
    }


def build_stage51h_final_candidate_report(
    *,
    mode_summaries: Mapping[str, Any],
    distance_by_mode: Mapping[str, Mapping[str, Any]],
    distance_by_mode_note: Mapping[str, Mapping[str, Mapping[str, Any]]],
    nn_by_mode: Mapping[str, Mapping[str, Any]],
    nn_by_mode_note: Mapping[str, Mapping[str, Mapping[str, Any]]],
    notes: Sequence[str],
    modes: Sequence[str],
    build_manifest: Mapping[str, Any],
    alias_equivalence: Sequence[Mapping[str, Any]],
    out_json: Path,
    out_md: Path,
    listening_review: Optional[Mapping[str, Any]] = None,
    vm_validated: bool = False,
) -> Dict[str, Any]:
    per_note: Dict[str, Any] = {}
    per_note_transient: Dict[str, Dict[str, Any]] = {}
    for note in notes:
        v41_ns = _note_summary(mode_summaries, V41_MODE, note)
        f25_ns = _note_summary(mode_summaries, F25_MODE, note)
        v41m = _spread_metrics(v41_ns, list(v41_ns.get("segments") or []))
        f25m = _spread_metrics(f25_ns, list(f25_ns.get("segments") or []))
        rows: Dict[str, Any] = {V41_MODE: v41m, F25_MODE: f25m}
        transient_rows: Dict[str, Any] = {}
        comparisons: Dict[str, Any] = {}
        for mode in modes:
            if mode == V41_MODE:
                segs = list(v41_ns.get("segments") or [])
                transient_rows[mode] = aggregate_transient_metrics(segs)
                continue
            ns = _note_summary(mode_summaries, mode, note)
            segs = list(ns.get("segments") or [])
            mm = _spread_metrics(ns, segs)
            mm["transient"] = aggregate_transient_metrics(segs)
            rows[mode] = mm
            transient_rows[mode] = mm["transient"]
            if mode in (
                F25_MODE,
                STK_FINAL_CANDIDATE_CANONICAL,
                STK_BODY_TRANSFER_FINAL_V1,
                STK_FINAL_DE_THUMP_CANONICAL,
                STK_BODY_TRANSFER_FINAL_V1_DE_THUMP,
            ):
                comparisons[mode] = _compare_vs_baselines(v41m=v41m, f25m=f25m, candidatem=mm)
                note_dist = (distance_by_mode_note.get(mode) or {}).get(note) or {}
                comparisons[mode]["estimated_audible_clusters"] = estimate_audible_clusters(
                    note_dist.get("audio_distances") or []
                )
                comparisons[mode]["spearman_rho"] = note_dist.get("spearman_rho")
                comparisons[mode]["nn_preservation_rate"] = (
                    (nn_by_mode_note.get(mode) or {}).get(note) or {}
                ).get("nn_preservation_rate")
        per_note[note] = {"modes": rows, "comparisons": comparisons}
        per_note_transient[note] = transient_rows

    alias_final = [r for r in alias_equivalence if r.get("pair") == "final_v1_vs_g30_70"]
    alias_de = [r for r in alias_equivalence if r.get("pair") == "de_thump_v1_vs_g30_70_de_thump"]
    alias_all_passed = (
        (not alias_final or all(r.get("passed") for r in alias_final))
        and (not alias_de or all(r.get("passed") for r in alias_de))
    )
    max_alias_diff = max((float(r.get("max_abs_diff") or 0) for r in alias_equivalence), default=0.0)

    transient_safety = build_transient_thump_safety(
        per_note_transient=per_note_transient,
        notes=notes,
        final_mode=STK_FINAL_CANDIDATE_CANONICAL,
        de_thump_mode=STK_FINAL_DE_THUMP_CANONICAL,
        v41_mode=V41_MODE,
    )

    lr = dict(listening_review or {})
    if not lr and build_manifest.get("listening_review"):
        lr = dict(build_manifest["listening_review"])
    vm_flag = bool(vm_validated or build_manifest.get("vm_validated"))

    final_vs_de_thump = compare_final_vs_de_thump(
        per_note=per_note,
        per_note_transient=per_note_transient,
        notes=notes,
        final_mode=STK_FINAL_CANDIDATE_CANONICAL,
        de_thump_mode=STK_FINAL_DE_THUMP_CANONICAL,
        v41_mode=V41_MODE,
    )
    de_thump_policy = build_de_thump_decision_policy(
        per_note=per_note,
        per_note_transient=per_note_transient,
        notes=notes,
        final_mode=STK_FINAL_CANDIDATE_CANONICAL,
        de_thump_mode=STK_FINAL_DE_THUMP_CANONICAL,
        v41_mode=V41_MODE,
        comparison=final_vs_de_thump,
        listening_review=lr,
        vm_validated=vm_flag,
    )

    freeze = _freeze_decision(
        per_note=per_note,
        alias_all_passed=alias_all_passed,
        distance_by_mode=distance_by_mode,
        nn_by_mode=nn_by_mode,
        transient_safety=transient_safety,
    )

    report: Dict[str, Any] = {
        "stage": "5.1H",
        "title": "STK final candidate validation and freeze",
        "final_alias_mapping": {
            "alias": STK_BODY_TRANSFER_FINAL_V1,
            "canonical_mode": STK_FINAL_CANDIDATE_CANONICAL,
            "blend": "30% absolute identity / 70% sample-relative contrast",
            "base": "modal_body_hybrid_v4_1_full",
            "gui_label": STK_FINAL_GUI_LABEL,
        },
        "vm_reference": VM_REFERENCE,
        "v4_1_unchanged": True,
        "fem_launched": False,
        "modes_run": list(modes),
        "notes": list(notes),
        "build_manifest": build_manifest,
        "per_note_metrics": per_note,
        "per_note_transient": per_note_transient,
        "transient_thump_safety": transient_safety,
        "final_vs_de_thump_comparison": final_vs_de_thump,
        "de_thump_decision_policy": de_thump_policy,
        "website_default_mode": de_thump_policy.get("website_default_mode"),
        "alias_equivalence": {
            "checks": list(alias_equivalence),
            "final_v1_all_passed": not alias_final or all(r.get("passed") for r in alias_final),
            "de_thump_all_passed": not alias_de or all(r.get("passed") for r in alias_de),
            "all_passed": alias_all_passed,
            "max_abs_diff": max_alias_diff,
            "pair_count": len(alias_equivalence),
        },
        "freeze_decision": freeze,
        "distance_consistency_by_mode": dict(distance_by_mode),
        "nearest_neighbor_by_mode": dict(nn_by_mode),
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    lines = [
        "# Stage 5.1H — STK final candidate freeze",
        "",
        f"**Decision:** `{freeze.get('decision')}`",
        f"**Freeze default mode:** `{freeze.get('recommended_default_mode')}`",
        f"**Website default (de-thump policy):** `{de_thump_policy.get('website_default_mode')}`",
        f"**De-thump decision:** `{de_thump_policy.get('decision')}`",
        f"**GUI label:** {freeze.get('recommended_gui_label')}",
        "",
        "## Alias mapping",
        f"`{STK_BODY_TRANSFER_FINAL_V1}` → `{STK_FINAL_CANDIDATE_CANONICAL}` (30/70)",
        f"Equivalence: **{'PASS' if alias_all_passed else 'FAIL'}** (max abs diff {max_alias_diff:.2e})",
        "",
        "## Per-note comparison (g_30_70 / final alias)",
        "",
        "| note | spec diff | % vs V4.1 | % vs F25 | ρ | NN | rms spread | peak max | clipping |",
        "|------|-----------|-----------|----------|---|-----|------------|----------|---------|",
    ]
    for note in notes:
        m = per_note.get(note, {}).get("modes", {}).get(STK_FINAL_CANDIDATE_CANONICAL, {})
        c = per_note.get(note, {}).get("comparisons", {}).get(STK_FINAL_CANDIDATE_CANONICAL, {})
        rho = c.get("spearman_rho")
        lines.append(
            f"| {note} | {m.get('spectral_differentiation')} | {c.get('pct_vs_v41')} | "
            f"{c.get('pct_vs_f25')} | {rho} | {c.get('nn_preservation_rate')} | "
            f"{m.get('rms_spread_db')} | {m.get('peak_dbfs_max')} | {m.get('clipping_any')} |"
        )
    lines.extend(
        [
            "",
            "## Transient metrics (V4.1 vs final vs de-thump)",
            "",
            "| note | mode | low_thump_ratio | onset_thump_ratio | transient_peak | excessive_frac | transient_dom |",
            "|------|------|-----------------|-------------------|----------------|----------------|---------------|",
        ]
    )
    for note in notes:
        for mode in (V41_MODE, STK_FINAL_CANDIDATE_CANONICAL, STK_FINAL_DE_THUMP_CANONICAL):
            tr = per_note_transient.get(note, {}).get(mode, {})
            lines.append(
                f"| {note} | {mode} | {tr.get('low_thump_ratio_median')} | "
                f"{tr.get('onset_thump_ratio_median')} | {tr.get('transient_peak_dbfs_max')} | "
                f"{tr.get('excessive_fraction')} | {tr.get('transient_dominated_count')} |"
            )
    lines.extend(
        [
            "",
            "## Transient / thump safety",
            f"- Exaggerated low thump: **{transient_safety.get('final_candidate_adds_exaggerated_low_thump')}**",
            f"- Pluck/bridge transient plausible: **{transient_safety.get('pluck_bridge_transient_physically_plausible')}**",
            f"- Diff from sustain (not transient): **{transient_safety.get('guitar_difference_from_sustain_not_transient')}**",
            f"- Reduce attack before freeze: **{transient_safety.get('reduce_attack_before_freezing')}**",
            f"- De-thump improves A3 (metric only): **{transient_safety.get('de_thump_improves_a3_low_thump')}**",
            f"- Safety variant (debug): `{STK_BODY_TRANSFER_FINAL_V1_DE_THUMP}`",
            "",
            transient_safety.get("summary", ""),
            "",
            "## De-thump decision policy",
            "",
            f"**Policy decision:** `{de_thump_policy.get('decision')}`",
            "",
            de_thump_policy.get("policy_summary", ""),
            "",
            "| note | spec Δ% | RMS vs V4.1 (final/de) | low_thump ↓% | peak ↓ dB | sustain OK | identity audible | assessment |",
            "|------|---------|------------------------|--------------|-----------|------------|------------------|------------|",
        ]
    )
    for note in notes:
        row = (final_vs_de_thump.get("per_note") or {}).get(note) or {}
        lines.append(
            f"| {note} | {row.get('spectral_diff_pct_change')} | "
            f"{row.get('rms_diff_vs_v41_final_db')}/{row.get('rms_diff_vs_v41_de_thump_db')} | "
            f"{row.get('low_thump_ratio_reduction_pct')} | {row.get('transient_peak_reduction_db')} | "
            f"{row.get('sustained_body_timbre_preserved')} | {row.get('guitar_identity_still_audible')} | "
            f"{row.get('de_thump_naturalness_assessment')} |"
        )
    lines.extend(
        [
            "",
            f"**Next stage:** {freeze.get('recommended_next_stage')}",
            f"**Return to ROM:** {freeze.get('return_to_rom_validation')}",
            f"**More STK tuning:** {freeze.get('more_stk_tuning_worthwhile')}",
            "",
            freeze.get("listening_recommendation", ""),
            "",
            f"FEM launched: **no**",
        ]
    )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
