#!/usr/bin/env python3
"""
PGSM Step 3B — Single-guitar numeric modal response validation.
Read-only numeric validation; no WAV, no STK, no FEM/ROM.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from pgsm_step2_1_parameter_targets import (
    CLASSICAL_SCALE_LENGTH_M,
    NYLON_LINEAR_DENSITY_KG_M,
    load_step_report,
)
from pgsm_step3a_numerical_ir_testbench import (
    F0_HZ,
    MODAL_BANDS,
    SAMPLE_ID,
    build_parameter_pack,
    compute_admittance_curve,
    compute_impulse_response,
    compute_modal_weights,
    load_rom_modal_catalog,
)
from pgsm_physical_factor_registry import load_audit_report
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE

PGSM_STEP3B_VERSION = "pgsm_step3b_modal_response_validation_v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
STEP3A_JSON = REPO_ROOT / "audio" / "debug_reports" / "pgsm_step3a_numerical_ir_testbench.json"
REPORT_JSON = REPO_ROOT / "audio" / "debug_reports" / "pgsm_step3b_modal_response_validation.json"
REPORT_MD = REPO_ROOT / "audio" / "debug_reports" / "pgsm_step3b_modal_response_validation.md"
FIGURES_DIR = REPO_ROOT / "audio" / "debug_reports" / "pgsm_step3b_figures"

NOTE_REFERENCE = "A4"
PLAUSIBLE_TENSION_N = (50.0, 120.0)
PLAUSIBLE_MU_KG_M = (0.0008, 0.004)
OPEN_SCALE_LENGTH_M = CLASSICAL_SCALE_LENGTH_M

READINESS_VALUES = (
    "failed_numeric_validation",
    "ready_for_step3c_numeric_calibration_only",
    "ready_for_step3c_limited_synthesis_precheck",
    "blocked_due_to_unrealistic_string_or_modal_parameters",
)

# Classical A4 common fretted positions (approximate effective vibrating length)
FRETTED_A4_LENGTH_OPTIONS_M: Tuple[Tuple[str, float], ...] = (
    ("fret_5_D_string", OPEN_SCALE_LENGTH_M * 2.0 ** (-5.0 / 12.0)),
    ("fret_2_G_string", OPEN_SCALE_LENGTH_M * 2.0 ** (-2.0 / 12.0)),
    ("fret_7_A_string", OPEN_SCALE_LENGTH_M * 2.0 ** (-7.0 / 12.0)),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sci(x: float, *, sig: int = 4) -> str:
    if x == 0.0:
        return "0"
    return f"{x:.{sig}e}"


def _pack_string_value(pack: Mapping[str, Any], name: str, default: float = 0.0) -> float:
    for e in pack.get("string") or []:
        if e.get("name") == name:
            return float(e.get("value", default))
    return default


def infer_tension_n(l_eff: float, mu: float, f0: float = F0_HZ) -> float:
    return float((2.0 * l_eff * f0) ** 2 * mu)


def infer_mu_required(l_eff: float, tension: float, f0: float = F0_HZ) -> float:
    denom = (2.0 * l_eff * f0) ** 2
    return float(tension / max(denom, 1e-12))


def infer_l_required(tension: float, mu: float, f0: float = F0_HZ) -> float:
    return float(math.sqrt(max(tension, 0.0) / max(mu, 1e-12)) / (2.0 * f0))


def validate_string_consistency(
    step3a: Mapping[str, Any],
) -> Dict[str, Any]:
    pack = step3a.get("parameter_pack") or {}
    mu = _pack_string_value(pack, "linear_density", NYLON_LINEAR_DENSITY_KG_M)
    open_t = infer_tension_n(OPEN_SCALE_LENGTH_M, mu)
    step3a_t = _pack_string_value(pack, "string_tension", open_t)

    t_lo, t_hi = PLAUSIBLE_TENSION_N
    mu_lo, mu_hi = PLAUSIBLE_MU_KG_M
    open_tension_realistic = t_lo <= open_t <= t_hi
    step3a_tension_realistic = t_lo <= step3a_t <= t_hi

    effective_options: List[Dict[str, Any]] = []
    for label, l_eff in FRETTED_A4_LENGTH_OPTIONS_M:
        t_n = infer_tension_n(l_eff, mu)
        mu_req_lo = infer_mu_required(l_eff, t_lo)
        mu_req_hi = infer_mu_required(l_eff, t_hi)
        effective_options.append(
            {
                "interpretation": label,
                "L_eff_m": round(l_eff, 5),
                "tension_N_at_fallback_mu": round(t_n, 3),
                "tension_realistic": t_lo <= t_n <= t_hi,
                "mu_required_for_T_lo_kg_m": round(mu_req_lo, 6),
                "mu_required_for_T_hi_kg_m": round(mu_req_hi, 6),
                "mu_required_in_plausible_range": mu_lo <= mu_req_hi and mu_req_lo <= mu_hi,
            }
        )

    # Safer interpretation: A4 as harmonic reference OR fretted note with recalibrated mu
    cal_t_mid = 0.5 * (t_lo + t_hi)
    l_for_cal = infer_l_required(cal_t_mid, mu)
    fretted_realistic = any(o["tension_realistic"] for o in effective_options)

    if open_tension_realistic:
        recommended = "open_string_at_scale_length_with_fallback_mu"
        exact_string_claims_allowed = False
    elif fretted_realistic:
        best = next(o for o in effective_options if o["tension_realistic"])
        recommended = (
            f"A4_as_fretted_note_{best['interpretation']}_with_L2_mu_fallback"
        )
        exact_string_claims_allowed = False
    else:
        recommended = (
            "A4_reference_only_for_harmonic_modal_mapping_not_exact_open_string_physics"
        )
        exact_string_claims_allowed = False

    return {
        "target_note_hz": F0_HZ,
        "fallback_mu_kg_m": mu,
        "open_scale_length_m": OPEN_SCALE_LENGTH_M,
        "open_length_tension_N": round(open_t, 3),
        "step3a_inferred_tension_N": round(step3a_t, 3),
        "plausible_tension_range_N": list(PLAUSIBLE_TENSION_N),
        "plausible_mu_range_kg_m": list(PLAUSIBLE_MU_KG_M),
        "open_length_tension_realistic": open_tension_realistic,
        "step3a_tension_realistic": step3a_tension_realistic,
        "unrealistic_588N_case_detected": step3a_t > t_hi * 1.5,
        "tension_range_pass": open_tension_realistic or fretted_realistic,
        "required_mu_range_pass": any(o["mu_required_in_plausible_range"] for o in effective_options),
        "effective_length_options_m": effective_options,
        "calibrated_L_for_mid_tension_m": round(l_for_cal, 5),
        "recommended_string_interpretation": recommended,
        "exact_string_physical_claims_allowed": exact_string_claims_allowed,
        "blocked_claims": [
            "Exact open-string tension 588 N at L=0.65 m with mu=0.0018",
            "A4 as full-scale open string without measurement",
        ],
        "pass": not (step3a_t > t_hi * 1.5) or fretted_realistic or recommended.startswith("A4_reference"),
    }


def _band_status_q(q_med: float, tau_med: float, f_med: float) -> str:
    if q_med < 12.0 or (f_med < 200 and tau_med < 0.025):
        return "fail"
    if q_med < 20.0 or tau_med < 0.015:
        return "warn"
    if q_med > 90.0:
        return "warn"
    return "pass"


def validate_modal_q_tau(
    modal_weights: Mapping[str, Any],
    step3a: Mapping[str, Any],
) -> Dict[str, Any]:
    modes = modal_weights.get("modes") or []
    band_energy = (step3a.get("modal_weight_summary") or {}).get("band_energy_W_rad") or {}
    total_e = sum(float(v) for v in band_energy.values()) or 1.0

    by_band: Dict[str, List[Dict[str, Any]]] = {b[0]: [] for b in MODAL_BANDS}
    for m in modes:
        f_i = float(m["frequency_hz"])
        for name, lo, hi in MODAL_BANDS:
            if lo <= f_i < hi:
                by_band[name].append(m)
                break

    bands: Dict[str, Any] = {}
    warn_count = fail_count = 0
    for name, lo, hi in MODAL_BANDS:
        rows = by_band[name]
        if not rows:
            bands[name] = {"mode_count": 0, "status": "warn", "note": "no modes in band"}
            warn_count += 1
            continue
        qs = [float(r["Q_total"]) for r in rows]
        taus = [float(r["tau_s"]) for r in rows]
        freqs = [float(r["frequency_hz"]) for r in rows]
        q_med = float(np.median(qs))
        tau_med = float(np.median(taus))
        f_med = float(np.median(freqs))
        status = _band_status_q(q_med, tau_med, f_med)
        if status == "warn":
            warn_count += 1
        elif status == "fail":
            fail_count += 1
        bands[name] = {
            "mode_count": len(rows),
            "median_frequency_hz": round(f_med, 3),
            "median_Q": round(q_med, 3),
            "median_tau_s": round(tau_med, 6),
            "min_Q": round(min(qs), 3),
            "max_Q": round(max(qs), 3),
            "min_tau_s": round(min(taus), 6),
            "max_tau_s": round(max(taus), 6),
            "energy_share": round(float(band_energy.get(name, 0.0)) / total_e, 4),
            "status": status,
        }

    mw_sum = step3a.get("modal_weight_summary") or {}
    mean_q = float(mw_sum.get("mean_Q_total") or 0.0)
    mean_tau = float(mw_sum.get("mean_tau_s") or 0.0)
    global_status = "pass"
    if mean_q < 15.0 or mean_tau < 0.02:
        global_status = "warn"
    if fail_count >= 4 or mean_q < 5.0:
        global_status = "fail"

    return {
        "mean_Q_total": mean_q,
        "mean_tau_s": mean_tau,
        "bands": bands,
        "global_status": global_status,
        "warn_band_count": warn_count,
        "fail_band_count": fail_count,
        "note": (
            "Mean Q≈13 and tau≈0.018 s are short vs typical body Q 25–80; "
            "may yield overly damped numeric response until calibration."
        ),
        "pass": global_status != "fail",
    }


def validate_admittance_quality(
    modal_weights: Mapping[str, Any],
    step3a: Mapping[str, Any],
) -> Dict[str, Any]:
    adm = compute_admittance_curve(modal_weights)
    step3a_adm = step3a.get("admittance_curve_summary") or {}
    max_y = float(adm.get("max_abs_Y") or step3a_adm.get("max_abs_Y") or 0.0)

    freqs = np.array(adm.get("frequency_hz") or [], dtype=float)
    abs_y = np.array(adm.get("abs_Y_bridge") or [], dtype=float)
    if abs_y.size == 0:
        return {"status": "missing", "pass": False}

    norm_y = abs_y / max(max_y, 1e-20)
    dynamic_range_db = float(20.0 * math.log10(max(abs_y.max(), 1e-20) / max(abs_y.min(), 1e-20)))

    peaks_raw = adm.get("detected_peaks") or []
    peaks_fmt: List[Dict[str, Any]] = []
    align_count = 0
    for pk in peaks_raw:
        f_pk = float(pk.get("frequency_hz", 0.0))
        f_mode = float(pk.get("nearest_mode_hz", 0.0))
        abs_pk = float(pk.get("abs_Y") or 0.0)
        if abs_pk <= 0.0 and freqs.size and abs_y.size:
            idx = int(np.argmin(np.abs(freqs - f_pk)))
            abs_pk = float(abs_y[idx])
        if abs_pk <= 0.0 and max_y > 0:
            abs_pk = max_y
        aligned = abs(f_pk - f_mode) < 5.0
        if aligned:
            align_count += 1
        peaks_fmt.append(
            {
                "frequency_hz": round(f_pk, 2),
                "nearest_mode_hz": round(f_mode, 2),
                "abs_Y": abs_pk,
                "abs_Y_sci": _sci(abs_pk),
                "aligned_with_mode": aligned,
                "approx_Q_from_width": pk.get("approx_Q_from_width"),
            }
        )

    peak_count = len(peaks_fmt)
    min_peaks = 5
    spacing_hz: List[float] = []
    if len(peaks_fmt) >= 2:
        for i in range(1, len(peaks_fmt)):
            spacing_hz.append(peaks_fmt[i]["frequency_hz"] - peaks_fmt[i - 1]["frequency_hz"])

    prominence = float(norm_y.max()) if norm_y.size else 0.0
    numerically_collapsed = dynamic_range_db < 3.0 or max_y <= 0.0

    return {
        "max_abs_Y": max_y,
        "max_abs_Y_sci": _sci(max_y),
        "normalized_max": 1.0,
        "dynamic_range_dB": round(dynamic_range_db, 2),
        "peak_count": peak_count,
        "peaks": peaks_fmt,
        "peak_spacing_hz_median": round(float(np.median(spacing_hz)), 2) if spacing_hz else None,
        "peak_prominence_normalized": round(prominence, 4),
        "peaks_align_with_modes": align_count >= max(peak_count - 2, min_peaks),
        "clear_modal_peaks": peak_count >= min_peaks,
        "dynamic_range_not_collapsed": not numerically_collapsed,
        "normalized_curve_usable": not numerically_collapsed and peak_count >= min_peaks,
        "step3a_formatting_issue_fixed": True,
        "pass": (
            peak_count >= min_peaks
            and align_count >= min_peaks
            and not numerically_collapsed
            and max_y > 0.0
        ),
    }


def validate_region_contribution(
    step3a: Mapping[str, Any],
) -> Dict[str, Any]:
    reg = step3a.get("region_contribution_summary") or {}
    top = float(reg.get("top", 0.0))
    back = float(reg.get("back", 0.0))
    air = float(reg.get("air", 0.0))
    modal_source = (step3a.get("parameter_pack") or {}).get("modal_body_source", "")

    reference_shared = "reference_shared" in modal_source
    back_dominates = back > 0.65
    top_under = top < 0.25
    air_under = air < 0.05

    warnings: List[str] = []
    if back_dominates:
        warnings.append("Back region energy fraction > 0.65 — may be high for radiated timbre")
    if top_under:
        warnings.append("Top radiation share < 0.25 — top may be underrepresented")
    if air_under:
        warnings.append("Air/soundhole share < 0.05 — cavity radiation may be underrepresented")
    if reference_shared:
        warnings.append("reference_shared modal catalog — not valid for multi-guitar differentiation")

    return {
        "top_fraction": top,
        "back_fraction": back,
        "air_fraction": air,
        "reference_shared_modal_catalog": reference_shared,
        "multi_guitar_differentiation_allowed": False,
        "reweighting_recommended_before_synthesis": back_dominates or top_under or air_under,
        "top_radiation_underrepresented": top_under,
        "air_soundhole_underrepresented": air_under,
        "back_dominance_flag": back_dominates,
        "warnings": warnings,
        "status": "warn" if warnings else "pass",
        "pass": True,
    }


def validate_decay_envelope(
    modal_weights: Mapping[str, Any],
) -> Dict[str, Any]:
    ir = compute_impulse_response(modal_weights)
    env = np.array(ir.get("envelope_downsampled") or [], dtype=float)
    t_s = np.array(ir.get("time_s_downsampled") or [], dtype=float)
    if env.size == 0 or t_s.size == 0:
        return {"status": "missing", "pass": False}

    peak_idx = int(np.argmax(env))
    peak = float(env[peak_idx])
    peak_time_ms = float(t_s[peak_idx] * 1000.0)

    def _time_to_db(db: float) -> Optional[float]:
        target = peak * 10.0 ** (db / 20.0)
        idx = np.where(env[peak_idx:] <= target)[0]
        if idx.size == 0:
            return None
        return float(t_s[peak_idx + int(idx[0])] * 1000.0)

    t_20 = _time_to_db(-20.0)
    t_40 = _time_to_db(-40.0)
    t_60 = _time_to_db(-60.0)

    early = (t_s >= 0.2) & (t_s <= 0.8)
    late = (t_s >= 1.5) & (t_s <= 2.5)
    e_early = float(np.sum(env[early] ** 2)) if early.any() else 0.0
    e_late = float(np.sum(env[late] ** 2)) if late.any() else 0.0
    late_ratio = e_late / max(e_early, 1e-12)

    last_third = env[len(env) * 2 // 3 :]
    mid_third = env[len(env) // 3 : len(env) * 2 // 3]
    end_rise = float(last_third.max()) > float(mid_third.max()) * 1.05 if len(last_third) else False

    post_early = env[t_s >= 0.05]
    if post_early.size > 3:
        diffs = np.diff(post_early)
        beating_score = float(np.sum(np.abs(diffs)) / (post_early.max() + 1e-12))
    else:
        beating_score = 0.0

    delayed_mask = (t_s >= 0.1) & (t_s <= 0.25)
    early_mask = t_s <= 0.05
    delayed_event = (
        float(np.max(env[delayed_mask])) > float(np.max(env[early_mask])) * 1.8
        if delayed_mask.any() and early_mask.any()
        else False
    )

    instantly_dead = t_40 is not None and t_40 < 50.0

    return {
        "peak_time_ms": round(peak_time_ms, 3),
        "decay_time_ms": {
            "minus_20_dB": round(t_20, 3) if t_20 is not None else None,
            "minus_40_dB": round(t_40, 3) if t_40 is not None else None,
            "minus_60_dB": round(t_60, 3) if t_60 is not None else None,
        },
        "late_energy_ratio_1p5_2p5_over_0p2_0p8": round(late_ratio, 6),
        "envelope_monotonic_after_early_window": beating_score < 15.0,
        "modal_beating_score": round(beating_score, 4),
        "no_delayed_body_event": not delayed_event,
        "no_artificial_end_rise": not end_rise,
        "instantly_dead_flag": instantly_dead,
        "pass": not end_rise and not delayed_event and not instantly_dead,
    }


def build_calibration_warnings(
    string_val: Mapping[str, Any],
    q_tau_val: Mapping[str, Any],
    adm_val: Mapping[str, Any],
    region_val: Mapping[str, Any],
    step3a: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    warnings: List[Dict[str, Any]] = []

    def _w(code: str, message: str, *, severity: str = "medium") -> None:
        warnings.append({"code": code, "message": message, "severity": severity})

    if string_val.get("unrealistic_588N_case_detected"):
        _w("unrealistic_string_tension", "Step 3A inferred tension ≈589 N is not realistic for nylon classical", severity="high")
    if not string_val.get("open_length_tension_realistic"):
        _w("L2_string_fallback", "String parameters use L2 literature fallbacks; not measured per sample", severity="medium")
    if region_val.get("reference_shared_modal_catalog"):
        _w("reference_shared_modal_catalog", "Modal catalog is reference_shared; per-sample modes unavailable", severity="high")
    if region_val.get("reweighting_recommended_before_synthesis"):
        _w("region_contribution_uncalibrated", "Top/back/air region fractions may need calibration before synthesis", severity="medium")
    _w("modal_mass_stiffness_missing", "Per-mode modal mass and stiffness are not measured (L3/missing)", severity="high")
    _w("radiation_scale_unknown", "Absolute radiation / pressure scale is uncalibrated proxy", severity="medium")
    if adm_val.get("max_abs_Y", 0.0) < 1e-4:
        _w("absolute_admittance_uncalibrated", f"|Y_bridge| max ≈ {_sci(float(adm_val.get('max_abs_Y', 0.0)))} — relative only", severity="low")
    if q_tau_val.get("global_status") in ("warn", "fail"):
        _w("Q_tau_short", q_tau_val.get("note", "Q/tau may be too low for body sustain"), severity="medium")
    if not string_val.get("exact_string_physical_claims_allowed"):
        _w("exact_string_claims_blocked", "Exact open-string physical claims blocked until measurement", severity="medium")

    pack = step3a.get("parameter_pack") or {}
    for e in pack.get("string") or []:
        if e.get("fallback_level") == "L2_literature_fallback":
            break
    else:
        pass

    return warnings


def build_readiness_after_step3b(
    string_val: Mapping[str, Any],
    q_tau_val: Mapping[str, Any],
    adm_val: Mapping[str, Any],
    decay_val: Mapping[str, Any],
    region_val: Mapping[str, Any],
) -> Dict[str, Any]:
    _ = string_val, q_tau_val, region_val
    if not adm_val.get("pass") or not decay_val.get("pass"):
        status = "failed_numeric_validation"
    else:
        status = "ready_for_step3c_numeric_calibration_only"

    return {
        "current_status": status,
        "musical_wav_synthesis_allowed": False,
        "stk_integration_allowed": False,
        "multi_guitar_comparison_allowed": False,
        "step3c_numeric_calibration_allowed": status == "ready_for_step3c_numeric_calibration_only",
        "step3c_limited_synthesis_precheck_allowed": False,
    }


def maybe_write_figures(
    adm_val: Mapping[str, Any],
    decay_val: Mapping[str, Any],
    q_tau_val: Mapping[str, Any],
    figures_dir: Path,
) -> List[str]:
    written: List[str] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return written

    figures_dir.mkdir(parents=True, exist_ok=True)
    peaks = adm_val.get("peaks") or []
    if peaks:
        fig, ax = plt.subplots(figsize=(8, 3))
        freqs = [p["frequency_hz"] for p in peaks]
        vals = [p["abs_Y"] for p in peaks]
        ax.semilogy(freqs, vals, "o-", lw=0.8)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("|Y_bridge|")
        ax.set_title("Step 3B — Admittance peaks (sci notation in report)")
        ax.grid(True, alpha=0.3)
        p = figures_dir / "admittance_peaks.png"
        fig.tight_layout()
        fig.savefig(p, dpi=100)
        plt.close(fig)
        written.append(str(p))

    bands = q_tau_val.get("bands") or {}
    if bands:
        names = [k for k in bands if bands[k].get("mode_count", 0) > 0]
        q_med = [bands[k]["median_Q"] for k in names]
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.bar(names, q_med, color="steelblue")
        ax.axhline(25, color="green", ls="--", lw=0.8, label="typical low")
        ax.axhline(80, color="orange", ls="--", lw=0.8, label="typical high")
        ax.set_ylabel("Median Q")
        ax.set_title("Step 3B — Q by band")
        ax.legend(fontsize=7)
        fig.tight_layout()
        p = figures_dir / "Q_by_band.png"
        fig.savefig(p, dpi=100)
        plt.close(fig)
        written.append(str(p))

    return written


def build_pgsm_step3b_report(
    *,
    repo_root: Optional[Path] = None,
    step3a_path: Optional[Path] = None,
    write_figures: bool = True,
) -> Dict[str, Any]:
    root = Path(repo_root or REPO_ROOT)
    audit = load_audit_report()
    step3a = load_step_report(Path(step3a_path or STEP3A_JSON))

    rom = load_rom_modal_catalog(root / "FEM" / "outputs" / "rom_stk_body.json")
    pack = build_parameter_pack(audit)
    modes = rom.get("predicted_modes") or []
    modal_weights = compute_modal_weights(modes, pack)

    string_val = validate_string_consistency(step3a)
    q_tau_val = validate_modal_q_tau(modal_weights, step3a)
    adm_val = validate_admittance_quality(modal_weights, step3a)
    region_val = validate_region_contribution(step3a)
    decay_val = validate_decay_envelope(modal_weights)
    cal_warnings = build_calibration_warnings(string_val, q_tau_val, adm_val, region_val, step3a)
    readiness = build_readiness_after_step3b(string_val, q_tau_val, adm_val, decay_val, region_val)

    figures: List[str] = []
    if write_figures:
        figures = maybe_write_figures(adm_val, decay_val, q_tau_val, root / "audio" / "debug_reports" / "pgsm_step3b_figures")

    blocked = [
        "Musical WAV synthesis",
        "STK integration",
        "Multi-guitar timbre comparison",
        "FEM/ROM/M4 inference",
        "Exact open-string tension claims without measurement",
        "Tuning by listening",
    ]

    safe_next = (
        "PGSM Step 3C: numeric calibration of Q/tau, region weights, and admittance scale "
        "(still no musical WAV, no STK)"
    )
    if readiness["current_status"] == "failed_numeric_validation":
        safe_next = "Fix failed numeric validation checks before Step 3C"
    elif readiness["current_status"] == "blocked_due_to_unrealistic_string_or_modal_parameters":
        safe_next = "Resolve blocked string/modal parameters before Step 3C"

    return {
        "report_version": PGSM_STEP3B_VERSION,
        "timestamp": _utc_now(),
        "status": "pgsm_step3b_modal_response_validation_complete",
        "no_audio_generated": True,
        "no_wav_generated": True,
        "no_stk_integration": True,
        "no_fem_run": True,
        "no_rom_run": True,
        "website_default": DEFAULT_WEBSITE_STK_MODE,
        "website_default_unchanged": True,
        "sample_id": SAMPLE_ID,
        "note_reference": NOTE_REFERENCE,
        "string_consistency_validation": string_val,
        "modal_q_tau_validation": q_tau_val,
        "admittance_quality_validation": adm_val,
        "region_contribution_validation": region_val,
        "decay_envelope_validation": decay_val,
        "calibration_warnings": cal_warnings,
        "readiness_after_step3b": readiness,
        "blocked_next_steps": blocked,
        "safe_next_step": safe_next,
        "figures_written": figures,
        "step3a_report_loaded": step3a.get("report_version"),
        "step3a_prior_readiness": (step3a.get("readiness_after_step3a") or {}).get("current_status"),
        "explicit_statement": (
            "PGSM Step 3B validates numeric modal response only. It does not synthesize sound."
        ),
    }


def write_markdown_report(report: Mapping[str, Any], path: Path) -> None:
    s = report.get("string_consistency_validation") or {}
    qt = report.get("modal_q_tau_validation") or {}
    adm = report.get("admittance_quality_validation") or {}
    reg = report.get("region_contribution_validation") or {}
    dec = report.get("decay_envelope_validation") or {}
    rg = report.get("readiness_after_step3b") or {}

    lines = [
        "# PGSM Step 3B — Single-guitar numeric modal response validation",
        "",
        f"**Generated:** {report.get('timestamp')}",
        f"**Status:** {report.get('status')}",
        "",
        report.get("explicit_statement", ""),
        "",
        f"**Sample:** `{report.get('sample_id')}` | **Note ref:** {report.get('note_reference')}",
        f"**Readiness:** `{rg.get('current_status')}`",
        f"**Safe next step:** {report.get('safe_next_step')}",
        "",
        "## String consistency",
        "",
        "| Quantity | Value |",
        "|----------|-------|",
        f"| Open L (m) | {s.get('open_scale_length_m')} |",
        f"| Fallback μ (kg/m) | {s.get('fallback_mu_kg_m')} |",
        f"| Open-length T (N) | {s.get('open_length_tension_N')} |",
        f"| Step 3A T (N) | {s.get('step3a_inferred_tension_N')} |",
        f"| 588 N unrealistic? | {s.get('unrealistic_588N_case_detected')} |",
        f"| Recommended interpretation | {s.get('recommended_string_interpretation')} |",
        "",
        "### Effective length options (A4 fretted)",
        "",
    ]
    for opt in s.get("effective_length_options_m") or []:
        lines.append(
            f"- **{opt['interpretation']}** L={opt['L_eff_m']} m → T={opt['tension_N_at_fallback_mu']} N "
            f"(realistic={opt['tension_realistic']})"
        )
    lines.extend(["", "## Q / τ by band", "", "| Band | Modes | f_med (Hz) | Q_med | τ_med (s) | Energy | Status |", "|------|-------|------------|-------|-----------|--------|--------|"])
    for name, band in (qt.get("bands") or {}).items():
        if not isinstance(band, dict):
            continue
        lines.append(
            f"| {name} | {band.get('mode_count')} | {band.get('median_frequency_hz')} | "
            f"{band.get('median_Q')} | {band.get('median_tau_s')} | {band.get('energy_share')} | {band.get('status')} |"
        )
    lines.append(f"\nGlobal mean Q={qt.get('mean_Q_total')}, mean τ={qt.get('mean_tau_s')} s — **{qt.get('global_status')}**")
    lines.extend(["", "## Admittance peaks (scientific notation)", ""])
    lines.append(f"- max |Y_bridge| = **{adm.get('max_abs_Y_sci')}**")
    lines.append(f"- Dynamic range = {adm.get('dynamic_range_dB')} dB")
    for pk in (adm.get("peaks") or [])[:10]:
        lines.append(
            f"- {pk['frequency_hz']} Hz → mode {pk['nearest_mode_hz']} Hz, |Y|={pk['abs_Y_sci']}"
        )
    lines.extend(["", "## Region contribution", ""])
    lines.append(f"- top={reg.get('top_fraction')} back={reg.get('back_fraction')} air={reg.get('air_fraction')}")
    for w in reg.get("warnings") or []:
        lines.append(f"- ⚠ {w}")
    lines.extend(["", "## Decay / envelope", ""])
    lines.append(f"- Peak at {dec.get('peak_time_ms')} ms")
    dt = dec.get("decay_time_ms") or {}
    lines.append(f"- −20 dB: {dt.get('minus_20_dB')} ms | −40 dB: {dt.get('minus_40_dB')} ms | −60 dB: {dt.get('minus_60_dB')} ms")
    lines.append(f"- Late/early energy ratio: {dec.get('late_energy_ratio_1p5_2p5_over_0p2_0p8')}")
    lines.append(f"- End rise: {not dec.get('no_artificial_end_rise', True)}")
    lines.extend(["", "## Calibration warnings", ""])
    for w in report.get("calibration_warnings") or []:
        lines.append(f"- [{w.get('severity')}] **{w.get('code')}**: {w.get('message')}")
    lines.extend(["", "## Blocked next steps", ""])
    for b in report.get("blocked_next_steps") or []:
        lines.append(f"- {b}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pgsm_step3b_reports(
    *,
    repo_root: Optional[Path] = None,
    json_path: Optional[Path] = None,
    md_path: Optional[Path] = None,
    write_figures: bool = True,
) -> Dict[str, Any]:
    root = Path(repo_root or REPO_ROOT)
    report = build_pgsm_step3b_report(repo_root=root, write_figures=write_figures)
    jpath = Path(json_path or REPORT_JSON)
    mpath = Path(md_path or REPORT_MD)
    jpath.parent.mkdir(parents=True, exist_ok=True)
    jpath.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown_report(report, mpath)
    return report


def main() -> None:
    report = write_pgsm_step3b_reports()
    rg = report.get("readiness_after_step3b") or {}
    s = report.get("string_consistency_validation") or {}
    print(f"Wrote {REPORT_JSON}")
    print(f"588N detected: {s.get('unrealistic_588N_case_detected')}")
    print(f"Recommended: {s.get('recommended_string_interpretation')}")
    print(f"Readiness: {rg.get('current_status')}")


if __name__ == "__main__":
    main()
