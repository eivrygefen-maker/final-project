#!/usr/bin/env python3
"""Transient / low-thump diagnostics for STK final candidate validation."""
from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

THUMP_RATIO_BASELINE_FACTOR = 1.35
TRANSIENT_CLIP_DBFS = -1.0
ONSET_MS = 100.0
SUSTAIN_START_MS = 100.0
SUSTAIN_END_MS = 500.0


def _rms(seg: np.ndarray) -> float:
    s = np.asarray(seg, dtype=np.float64)
    if len(s) < 1:
        return 0.0
    return float(np.sqrt(np.mean(s**2)))


def _slice_ms(audio: np.ndarray, sample_rate: int, start_ms: float, end_ms: float) -> np.ndarray:
    sr = float(sample_rate)
    i0 = max(0, int(start_ms * 1e-3 * sr))
    i1 = min(len(audio), int(end_ms * 1e-3 * sr))
    if i1 <= i0:
        return np.asarray([], dtype=np.float64)
    return np.asarray(audio[i0:i1], dtype=np.float64)


def _band_energy(seg: np.ndarray, sample_rate: int, f_lo: float, f_hi: float) -> float:
    if len(seg) < 8:
        return 0.0
    spec = np.fft.rfft(seg)
    freqs = np.fft.rfftfreq(len(seg), d=1.0 / float(sample_rate))
    power = np.abs(spec) ** 2
    mask = (freqs >= float(f_lo)) & (freqs < float(f_hi))
    return float(np.sum(power[mask]))


def analyze_transient_thump(
    audio: np.ndarray,
    *,
    sample_rate: int,
    frequency_hz: float,
    baseline: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Onset transient and sub-120 Hz thump metrics for one segment."""
    x = np.asarray(audio, dtype=np.float64)
    n = len(x)
    sr = int(sample_rate)
    if n < 16:
        return {"valid": False}

    onset = _slice_ms(x, sr, 0.0, ONSET_MS)
    sustain = _slice_ms(x, sr, SUSTAIN_START_MS, SUSTAIN_END_MS)
    total_rms = _rms(x)
    onset_rms = _rms(onset)
    sustain_rms = _rms(sustain)

    e_onset_120 = _band_energy(onset, sr, 0.0, 120.0)
    e_onset_180 = _band_energy(onset, sr, 0.0, 180.0)
    e_sustain_120 = _band_energy(sustain, sr, 0.0, 120.0)
    e_sustain_total = _band_energy(sustain, sr, 0.0, sr * 0.5)
    e_onset_total = _band_energy(onset, sr, 0.0, sr * 0.5)

    f0 = max(40.0, float(frequency_hz))
    e_harm_onset = sum(
        _band_energy(onset, sr, f0 * k * 0.88, f0 * k * 1.12) for k in (1.0, 2.0, 3.0)
    )

    onset_thump_ratio = e_onset_120 / max(e_sustain_120, e_harm_onset * 0.05, 1e-18)
    low_thump_ratio = e_onset_120 / max(
        e_sustain_120 + e_harm_onset * 0.25,
        total_rms**2 * max(len(onset), 1) * 0.01,
        1e-18,
    )
    sub120_vs_total_onset = e_onset_120 / max(e_onset_total, 1e-18)
    sub120_vs_sustain = e_onset_120 / max(e_sustain_120, 1e-18)

    onset_end = min(n, int(ONSET_MS * 1e-3 * sr))
    transient_peak = float(np.max(np.abs(x[:onset_end]))) if onset_end > 0 else 0.0
    transient_peak_dbfs = round(20.0 * math.log10(max(transient_peak, 1e-12)), 4)

    likely_excessive = False
    reasons: list[str] = []
    if transient_peak_dbfs > TRANSIENT_CLIP_DBFS:
        likely_excessive = True
        reasons.append("transient_near_clipping")
    if sub120_vs_sustain > 2.5 and e_onset_120 > e_harm_onset * 0.8:
        likely_excessive = True
        reasons.append("sub120_dominates_onset_vs_sustain")
    if baseline is not None:
        base_ltr = float(baseline.get("low_thump_ratio") or 0.0)
        if base_ltr > 0 and low_thump_ratio > base_ltr * THUMP_RATIO_BASELINE_FACTOR:
            likely_excessive = True
            reasons.append("low_thump_vs_v41_baseline")
        base_otr = float(baseline.get("onset_thump_ratio") or 0.0)
        if base_otr > 0 and onset_thump_ratio > base_otr * THUMP_RATIO_BASELINE_FACTOR:
            likely_excessive = True
            reasons.append("onset_thump_vs_v41_baseline")

    transient_vs_sustain = onset_rms / max(sustain_rms, 1e-12)
    differentiation_from_transient = transient_vs_sustain > 2.0 and sustain_rms < onset_rms * 0.35

    if differentiation_from_transient:
        reasons.append("difference_may_be_transient_dominated")

    natural = not likely_excessive
    assessment = "likely_natural" if natural else "likely_excessive"

    return {
        "valid": True,
        "transient_rms_20ms": round(_rms(_slice_ms(x, sr, 0.0, 20.0)), 8),
        "transient_rms_50ms": round(_rms(_slice_ms(x, sr, 0.0, 50.0)), 8),
        "transient_rms_100ms": round(onset_rms, 8),
        "sustain_rms_100_500ms": round(sustain_rms, 8),
        "onset_energy_below_120hz": round(e_onset_120, 8),
        "onset_energy_below_180hz": round(e_onset_180, 8),
        "sustain_energy_below_120hz": round(e_sustain_120, 8),
        "harmonic_onset_energy_f0_h3": round(e_harm_onset, 8),
        "onset_thump_ratio": round(onset_thump_ratio, 6),
        "low_thump_ratio": round(low_thump_ratio, 6),
        "sub120_onset_fraction": round(sub120_vs_total_onset, 6),
        "sub120_onset_vs_sustain": round(sub120_vs_sustain, 6),
        "transient_peak_dbfs": transient_peak_dbfs,
        "transient_vs_sustain_rms_ratio": round(transient_vs_sustain, 6),
        "likely_excessive": likely_excessive,
        "transient_assessment": assessment,
        "suspicious_reasons": reasons,
        "differentiation_transient_dominated": differentiation_from_transient,
    }


def aggregate_transient_metrics(segments: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Summarize transient metrics across guitar samples for one note/mode."""
    vals = [s.get("transient_thump") for s in segments if (s.get("transient_thump") or {}).get("valid")]
    if not vals:
        return {"segment_count": 0}
    excessive = sum(1 for v in vals if v.get("likely_excessive"))
    return {
        "segment_count": len(vals),
        "low_thump_ratio_median": round(float(np.median([v["low_thump_ratio"] for v in vals])), 6),
        "onset_thump_ratio_median": round(float(np.median([v["onset_thump_ratio"] for v in vals])), 6),
        "transient_peak_dbfs_max": round(max(v["transient_peak_dbfs"] for v in vals), 4),
        "transient_rms_100ms_median": round(float(np.median([v["transient_rms_100ms"] for v in vals])), 8),
        "sustain_rms_median": round(float(np.median([v["sustain_rms_100_500ms"] for v in vals])), 8),
        "excessive_count": excessive,
        "excessive_fraction": round(excessive / len(vals), 4),
        "any_excessive": excessive > 0,
        "transient_dominated_count": sum(1 for v in vals if v.get("differentiation_transient_dominated")),
    }


def build_transient_thump_safety(
    *,
    per_note_transient: Mapping[str, Mapping[str, Mapping[str, Any]]],
    notes: Sequence[str],
    final_mode: str,
    de_thump_mode: str,
    v41_mode: str,
) -> Dict[str, Any]:
    """Report section: Transient / thump safety."""
    final_flags: list[str] = []
    for note in notes:
        fm = (per_note_transient.get(note) or {}).get(final_mode) or {}
        vm = (per_note_transient.get(note) or {}).get(v41_mode) or {}
        if fm.get("any_excessive"):
            final_flags.append(note)
        ltr_final = float(fm.get("low_thump_ratio_median") or 0)
        ltr_v41 = float(vm.get("low_thump_ratio_median") or 0)
        if ltr_v41 > 0 and ltr_final > ltr_v41 * THUMP_RATIO_BASELINE_FACTOR:
            final_flags.append(f"{note}:low_thump_vs_v41")

    exaggerated = len(final_flags) > 0
    de_thump = (per_note_transient.get("A3") or {}).get(de_thump_mode) or {}
    final_a3 = (per_note_transient.get("A3") or {}).get(final_mode) or {}

    return {
        "final_candidate_adds_exaggerated_low_thump": exaggerated,
        "pluck_bridge_transient_physically_plausible": not exaggerated,
        "guitar_difference_from_sustain_not_transient": (
            int(final_a3.get("transient_dominated_count") or 0) == 0
        ),
        "reduce_attack_before_freezing": exaggerated,
        "recommended_safety_variant": de_thump_mode if exaggerated else None,
        "flagged_notes": sorted(set(final_flags)),
        "de_thump_improves_a3_low_thump": (
            float(de_thump.get("low_thump_ratio_median") or 999)
            < float(final_a3.get("low_thump_ratio_median") or 0)
        )
        if de_thump and final_a3
        else None,
        "de_thump_is_safety_only_not_auto_default": True,
        "summary": (
            "Metrics suggest elevated low-frequency onset vs V4.1 on some notes — "
            "VM/listening review required; de-thump is available as safety/debug variant only."
            if exaggerated
            else "Onset transients appear within plausible guitar/body attack range vs V4.1 baseline."
        ),
    }


SPEC_DIFF_PRESERVE_MAX_LOSS_PCT = 5.0
SPEC_DIFF_TOO_SOFTENED_PCT = 8.0
LOW_THUMP_SIGNIFICANT_FACTOR = 1.35
TRANSIENT_DOMINATED_FRACTION = 0.5


def compare_final_vs_de_thump(
    *,
    per_note: Mapping[str, Mapping[str, Any]],
    per_note_transient: Mapping[str, Mapping[str, Mapping[str, Any]]],
    notes: Sequence[str],
    final_mode: str,
    de_thump_mode: str,
    v41_mode: str,
) -> Dict[str, Any]:
    """Per-note and aggregate comparison: final v1 vs de-thump safety variant."""
    per_note_out: Dict[str, Any] = {}
    spec_changes: list[float] = []
    ltr_reductions: list[float] = []
    peak_reductions: list[float] = []

    for note in notes:
        nd = per_note.get(note) or {}
        fm = (nd.get("modes") or {}).get(final_mode) or {}
        dm = (nd.get("modes") or {}).get(de_thump_mode) or {}
        ft = (per_note_transient.get(note) or {}).get(final_mode) or {}
        dt = (per_note_transient.get(note) or {}).get(de_thump_mode) or {}
        vt = (per_note_transient.get(note) or {}).get(v41_mode) or {}

        spec_f = float(fm.get("spectral_differentiation") or 0)
        spec_d = float(dm.get("spectral_differentiation") or 0)
        spec_pct = round((spec_d - spec_f) / max(spec_f, 1e-9) * 100.0, 3)
        spec_changes.append(spec_pct)

        ltr_f = float(ft.get("low_thump_ratio_median") or 0)
        ltr_d = float(dt.get("low_thump_ratio_median") or 0)
        ltr_v = float(vt.get("low_thump_ratio_median") or 0)
        ltr_red = round((ltr_f - ltr_d) / max(ltr_f, 1e-9) * 100.0, 3) if ltr_f > 0 else 0.0
        ltr_reductions.append(ltr_red)

        peak_f = float(ft.get("transient_peak_dbfs_max") or -60)
        peak_d = float(dt.get("transient_peak_dbfs_max") or -60)
        peak_red = round(peak_d - peak_f, 4)
        peak_reductions.append(peak_red)

        seg_n = int(ft.get("segment_count") or 0)
        dom_n = int(ft.get("transient_dominated_count") or 0)
        onset_dominates = seg_n > 0 and (dom_n / seg_n) >= TRANSIENT_DOMINATED_FRACTION

        too_soft = spec_pct < -SPEC_DIFF_TOO_SOFTENED_PCT
        sustained_ok = spec_pct >= -SPEC_DIFF_PRESERVE_MAX_LOSS_PCT
        if too_soft:
            natural_assessment = "too_softened"
        elif ltr_red > 0 and sustained_ok:
            natural_assessment = "more_natural_likely"
        elif sustained_ok:
            natural_assessment = "similar_identity"
        else:
            natural_assessment = "identity_reduced"

        per_note_out[note] = {
            "spectral_diff_final": spec_f,
            "spectral_diff_de_thump": spec_d,
            "spectral_diff_pct_change": spec_pct,
            "rms_diff_vs_v41_final_db": fm.get("rms_diff_db_vs_v41_median"),
            "rms_diff_vs_v41_de_thump_db": dm.get("rms_diff_db_vs_v41_median"),
            "low_thump_ratio_final": ltr_f,
            "low_thump_ratio_de_thump": ltr_d,
            "low_thump_ratio_v41": ltr_v,
            "low_thump_ratio_reduction_pct": ltr_red,
            "transient_peak_final_dbfs": peak_f,
            "transient_peak_de_thump_dbfs": peak_d,
            "transient_peak_reduction_db": peak_red,
            "onset_dominates_difference": onset_dominates,
            "sustained_body_timbre_preserved": sustained_ok,
            "guitar_identity_still_audible": bool(dm.get("likely_audible_vs_v41")),
            "de_thump_naturalness_assessment": natural_assessment,
        }

    agg_spec = round(float(np.median(spec_changes)), 3) if spec_changes else 0.0
    agg_ltr = round(float(np.median(ltr_reductions)), 3) if ltr_reductions else 0.0
    return {
        "per_note": per_note_out,
        "aggregate": {
            "spectral_diff_pct_change_median": agg_spec,
            "low_thump_ratio_reduction_pct_median": agg_ltr,
            "transient_peak_reduction_db_median": round(float(np.median(peak_reductions)), 4)
            if peak_reductions
            else 0.0,
            "de_thump_preserves_differentiation": agg_spec >= -SPEC_DIFF_PRESERVE_MAX_LOSS_PCT,
            "de_thump_reduces_thump": agg_ltr > 0,
        },
    }


def build_de_thump_decision_policy(
    *,
    per_note: Mapping[str, Mapping[str, Any]],
    per_note_transient: Mapping[str, Mapping[str, Mapping[str, Any]]],
    notes: Sequence[str],
    final_mode: str,
    de_thump_mode: str,
    v41_mode: str,
    comparison: Mapping[str, Any],
    listening_review: Optional[Mapping[str, Any]] = None,
    vm_validated: bool = False,
) -> Dict[str, Any]:
    """
    De-thump is safety-only. Default stays final v1 unless VM/listening criteria met.
    Never switch default from local synthetic low_thump improvement alone.
    """
    lr = dict(listening_review or {})
    heard_thump = bool(lr.get("heard_drum_thump"))
    de_more_natural = lr.get("de_thump_sounds_more_natural")
    agg = comparison.get("aggregate") or {}

    notes_high_thump = 0
    notes_onset_dominates = 0
    for note in notes:
        ft = (per_note_transient.get(note) or {}).get(final_mode) or {}
        vt = (per_note_transient.get(note) or {}).get(v41_mode) or {}
        ltr_f = float(ft.get("low_thump_ratio_median") or 0)
        ltr_v = float(vt.get("low_thump_ratio_median") or 0)
        if ltr_v > 0 and ltr_f > ltr_v * LOW_THUMP_SIGNIFICANT_FACTOR:
            notes_high_thump += 1
        seg_n = int(ft.get("segment_count") or 0)
        dom_n = int(ft.get("transient_dominated_count") or 0)
        if seg_n > 0 and (dom_n / seg_n) >= TRANSIENT_DOMINATED_FRACTION:
            notes_onset_dominates += 1

    de_reduces_thump = bool(agg.get("de_thump_reduces_thump"))
    de_preserves_spec = bool(agg.get("de_thump_preserves_differentiation"))

    criteria = {
        "c1_listening_heard_drum_thump": heard_thump,
        "c2_low_thump_above_v41_multiple_notes": notes_high_thump >= 2,
        "c3_onset_dominates_multiple_notes": notes_onset_dominates >= 2,
        "c4_de_thump_reduces_thump_preserves_spec": de_reduces_thump and de_preserves_spec,
        "notes_high_thump_count": notes_high_thump,
        "notes_onset_dominates_count": notes_onset_dominates,
    }

    website_default = "stk_body_transfer_final_v1"
    debug_variant = "stk_body_transfer_final_v1_de_thump_candidate"

    switch_allowed = vm_validated or heard_thump

    if switch_allowed and heard_thump and de_reduces_thump and de_preserves_spec:
        if de_more_natural is not False:
            decision = "switch_default_to_de_thump"
            website_default = debug_variant
        else:
            decision = "keep_final_v1_but_offer_de_thump_debug"
    elif (
        switch_allowed
        and vm_validated
        and criteria["c2_low_thump_above_v41_multiple_notes"]
        and criteria["c4_de_thump_reduces_thump_preserves_spec"]
        and (criteria["c3_onset_dominates_multiple_notes"] or heard_thump)
    ):
        decision = "switch_default_to_de_thump"
        website_default = debug_variant
    elif heard_thump or notes_high_thump >= 1 or notes_onset_dominates >= 1:
        if de_reduces_thump and not switch_allowed:
            decision = "needs_more_listening"
        elif de_reduces_thump and de_preserves_spec:
            decision = "keep_final_v1_but_offer_de_thump_debug"
        else:
            decision = "needs_more_listening"
    else:
        decision = "keep_final_v1_default"

    return {
        "decision": decision,
        "website_default_mode": website_default,
        "debug_safety_variant": debug_variant,
        "default_remains_final_v1_unless_vm_or_listening": True,
        "never_auto_switch_from_local_synthetic_low_thump_alone": True,
        "criteria_met": criteria,
        "listening_review": lr if lr else None,
        "vm_validated": vm_validated,
        "comparison_aggregate": agg,
        "policy_summary": (
            "Default remains stk_body_transfer_final_v1. "
            "De-thump is safety/debug only until VM or listening confirms drum-like thump."
            if decision in ("keep_final_v1_default", "needs_more_listening", "keep_final_v1_but_offer_de_thump_debug")
            else "Switch website default to de-thump safety variant per VM/listening validation."
        ),
    }
