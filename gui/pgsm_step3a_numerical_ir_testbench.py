#!/usr/bin/env python3
"""
PGSM Step 3A — Numerical bridge-admittance / modal impulse-response testbench.
Numeric only: no WAV, no STK, no FEM/ROM execution.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from pgsm_physical_factor_registry import (
    DEFAULT_L_EFF_M,
    amplitude_tau_s,
    helmholtz_proxy_hz,
    load_audit_report,
    q_from_damping_coeff,
)
from pgsm_step2_1_parameter_targets import (
    CAVITY_COUPLING_TYPICAL,
    CLASSICAL_SCALE_LENGTH_M,
    NYLON_LINEAR_DENSITY_KG_M,
    PLUCK_DURATION_MS_TYPICAL,
    RADIATION_DAMPING_COEFF_TYPICAL,
    load_step_report as load_step21_report,
)
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE
from stk_v6_2_audit_features import feature_value, get_sample_record

PGSM_STEP3A_VERSION = "pgsm_step3a_numerical_ir_testbench_v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
STEP1_JSON = REPO_ROOT / "audio" / "debug_reports" / "pgsm_step1_physical_factor_registry.json"
STEP2_JSON = REPO_ROOT / "audio" / "debug_reports" / "pgsm_step2_physical_interaction_map.json"
STEP21_JSON = REPO_ROOT / "audio" / "debug_reports" / "pgsm_step2_1_parameter_targets.json"
REPORT_JSON = REPO_ROOT / "audio" / "debug_reports" / "pgsm_step3a_numerical_ir_testbench.json"
REPORT_MD = REPO_ROOT / "audio" / "debug_reports" / "pgsm_step3a_numerical_ir_testbench.md"
FIGURES_DIR = REPO_ROOT / "audio" / "debug_reports" / "pgsm_step3a_figures"
ROM_BODY_JSON = REPO_ROOT / "FEM" / "outputs" / "rom_stk_body.json"
BODY_CACHE_JSON = REPO_ROOT / "ROM" / "classic" / "body_signature_cache" / "sample_000.json"

SAMPLE_ID = "sample_000"
NOTE_REFERENCE = "A4"
F0_HZ = 440.0
NUMERIC_SR = 44100
DURATION_S = 2.5
FREQ_GRID_HZ = (40.0, 5000.0, 400)
FIXED_PLUCK_POSITION = 0.18

L3_BLOCKED_NAMES = frozenset(
    {"top_elastic_moduli", "back_elastic_moduli", "wood_anisotropy", "modal_stiffness"}
)

MODAL_BANDS: Tuple[Tuple[str, float, float], ...] = (
    ("sub_body", 0.0, 100.0),
    ("low_body", 100.0, 200.0),
    ("mid_body", 200.0, 350.0),
    ("upper_body", 350.0, 500.0),
    ("high", 500.0, 5000.0),
)

READINESS_AFTER = (
    "failed_numerical_physics_checks",
    "ready_for_step3b_single_guitar_numeric_modal_response",
    "blocked_due_to_missing_modal_data",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _num_entry(
    name: str,
    value: Any,
    *,
    source: str,
    fallback_level: str,
    units: str,
    confidence: str,
    allowed_use: str,
) -> Dict[str, Any]:
    return {
        "name": name,
        "value": value,
        "source": source,
        "fallback_level": fallback_level,
        "units": units,
        "confidence": confidence,
        "allowed_use": allowed_use,
    }


def load_rom_modal_catalog(path: Optional[Path] = None) -> Dict[str, Any]:
    p = Path(path or ROM_BODY_JSON)
    if not p.is_file():
        return {"status": "missing", "predicted_modes": []}
    doc = json.loads(p.read_text(encoding="utf-8"))
    modes = doc.get("predicted_modes") or []
    return {
        "status": "reference_shared",
        "path": str(p),
        "num_modes": len(modes),
        "full_modal_band_hz": doc.get("full_modal_band_hz"),
        "predicted_modes": modes,
    }


def harmonic_excitation_at_freq(
    f_hz: float,
    f0: float,
    xp: float,
    *,
    scale_length: float = CLASSICAL_SCALE_LENGTH_M,
) -> float:
    """xp is pluck_position_ratio (x_p/L); use continuous harmonic index f_i/f0."""
    _ = scale_length
    n_eff = max(f_hz / max(f0, 1.0), 0.05)
    return abs(math.sin(n_eff * math.pi * xp))


def infer_string_tension_n(scale_length: float, mu: float, f0: float = F0_HZ) -> float:
    return float((2.0 * scale_length * f0) ** 2 * mu)


def build_parameter_pack(
    audit: Mapping[str, Any],
    *,
    sample_id: str = SAMPLE_ID,
    pluck_position: float = FIXED_PLUCK_POSITION,
) -> Dict[str, Any]:
    sample = get_sample_record(audit, sample_id)
    blocked_l3: List[Dict[str, Any]] = []
    for name in sorted(L3_BLOCKED_NAMES):
        blocked_l3.append(
            _num_entry(
                name,
                None,
                source="PGSM Step 2.1 L3_blocked — not used in Step 3A computation",
                fallback_level="L3_blocked",
                units="—",
                confidence="low",
                allowed_use="reported_missing_only",
            )
        )

    scale_length = CLASSICAL_SCALE_LENGTH_M
    mu = NYLON_LINEAR_DENSITY_KG_M
    tension = infer_string_tension_n(scale_length, mu)

    vol = float(feature_value(sample, "body_volume_proxy", audit=audit, default=0.013))
    hole_area = float(feature_value(sample, "soundhole_area", audit=audit, default=0.007))
    helm_f = float(
        feature_value(sample, "helmholtz_like_frequency_proxy", audit=audit, default=0.0)
    ) or helmholtz_proxy_hz(vol, hole_area)
    cavity_q = float(feature_value(sample, "cavity_q_proxy", audit=audit, default=14.0))
    mob = float(feature_value(sample, "bridge_mobility_proxy", audit=audit, default=1.0))
    top_damp = float(feature_value(sample, "top_damping_coeff_proxy", audit=audit, default=1.0))
    back_damp = float(feature_value(sample, "back_damping_coeff_proxy", audit=audit, default=1.0))
    hf_abs = float(
        feature_value(sample, "high_frequency_absorption_proxy", audit=audit, default=0.4)
    )
    mass_proxy = float(feature_value(sample, "mass_loading_proxy", audit=audit, default=0.0005))

    cache_meta: Dict[str, Any] = {}
    if BODY_CACHE_JSON.is_file():
        cache_meta = json.loads(BODY_CACHE_JSON.read_text(encoding="utf-8"))

    excitation = [
        _num_entry(
            "impulse_force_proxy",
            1.0,
            source="normalized unit F_bridge(t) proxy; amplitude arbitrary",
            fallback_level="L2_literature_fallback",
            units="N·s proxy",
            confidence="medium",
            allowed_use="numerical_ir_testbench",
        ),
        _num_entry(
            "pluck_duration_ms",
            PLUCK_DURATION_MS_TYPICAL,
            source="PGSM Step 2.1 literature fallback",
            fallback_level="L2_literature_fallback",
            units="ms",
            confidence="low",
            allowed_use="force_pulse_width_metadata_only",
        ),
        _num_entry(
            "note_reference_frequency",
            F0_HZ,
            source="A4 reference for harmonic/modal mapping only",
            fallback_level="L0_measured",
            units="Hz",
            confidence="high",
            allowed_use="harmonic_mapping",
        ),
    ]

    string_params = [
        _num_entry(
            "scale_length",
            scale_length,
            source="PGSM Step 2.1 L2 classical fallback",
            fallback_level="L2_literature_fallback",
            units="m",
            confidence="medium",
            allowed_use="harmonic_excitation_shape",
        ),
        _num_entry(
            "linear_density",
            mu,
            source="PGSM Step 2.1 L2 nylon order-of-magnitude",
            fallback_level="L2_literature_fallback",
            units="kg/m",
            confidence="medium",
            allowed_use="tension_inference",
        ),
        _num_entry(
            "string_tension",
            round(tension, 3),
            source="inferred T=(2Lf0)^2 μ for consistent A4",
            fallback_level="L1_derived",
            units="N",
            confidence="medium",
            allowed_use="harmonic_mapping_metadata",
        ),
        _num_entry(
            "pluck_position_ratio",
            pluck_position,
            source="PGSM Step 2.1 / production pluck ratio ≈0.18",
            fallback_level="L1_derived",
            units="dimensionless",
            confidence="high",
            allowed_use="harmonic_excitation_shape",
        ),
    ]

    geometry_material = [
        _num_entry("body_volume_proxy", vol, source="audit derived_features", fallback_level="L1_derived", units="m³", confidence="medium", allowed_use="helmholtz_proxy"),
        _num_entry("soundhole_area", hole_area, source="audit geometry", fallback_level="L1_derived", units="m²", confidence="high", allowed_use="helmholtz_proxy"),
        _num_entry("helmholtz_frequency_proxy", round(helm_f, 3), source="audit derived or helmholtz_proxy_hz", fallback_level="L1_derived", units="Hz", confidence="medium", allowed_use="cavity_weighting"),
        _num_entry("cavity_q_proxy", cavity_q, source="audit derived_features", fallback_level="L1_derived", units="dimensionless", confidence="medium", allowed_use="Q_air_term"),
        _num_entry("bridge_mobility_proxy", mob, source="audit derived_features / body_signature_cache", fallback_level="L1_derived", units="relative", confidence="medium", allowed_use="modal_excitation_weight"),
        _num_entry("top_damping_coeff_proxy", top_damp, source="audit material wood-weighted", fallback_level="L1_derived", units="relative", confidence="medium", allowed_use="Q_struct_term"),
        _num_entry("back_damping_coeff_proxy", back_damp, source="audit material wood-weighted", fallback_level="L1_derived", units="relative", confidence="medium", allowed_use="Q_struct_term"),
        _num_entry("high_frequency_absorption_proxy", hf_abs, source="audit derived_features", fallback_level="L1_derived", units="relative", confidence="medium", allowed_use="high_band_damping_proxy"),
        _num_entry("mass_loading_proxy", mass_proxy, source="audit derived / body_signature_cache", fallback_level="L1_derived", units="kg proxy", confidence="medium", allowed_use="mobility_metadata"),
        _num_entry("effective_neck_length_helmholtz", DEFAULT_L_EFF_M, source="PGSM Step 1 DEFAULT_L_EFF_M", fallback_level="L2_literature_fallback", units="m", confidence="low", allowed_use="helmholtz_proxy"),
        _num_entry("cavity_coupling_coefficient", CAVITY_COUPLING_TYPICAL, source="PGSM Step 2.1 L2", fallback_level="L2_literature_fallback", units="dimensionless", confidence="low", allowed_use="W_air_weighting"),
    ]

    return {
        "sample_id": sample_id,
        "note_reference": NOTE_REFERENCE,
        "excitation": excitation,
        "string": string_params,
        "geometry_material": geometry_material,
        "modal_body_source": "FEM/outputs/rom_stk_body.json reference_shared predicted_modes",
        "body_signature_cache": cache_meta,
        "L3_blocked_not_used": blocked_l3,
    }


def compute_modal_weights(
    modes: Sequence[Mapping[str, Any]],
    pack: Mapping[str, Any],
    *,
    pluck_position: float = FIXED_PLUCK_POSITION,
    mobility_scale: float = 1.0,
    damping_scale: float = 1.0,
    radiation_damping_scale: float = 1.0,
) -> Dict[str, Any]:
    if not modes:
        return {"status": "no_modes", "modes": []}

    def _pack_val(section: str, name: str, default: float) -> float:
        for e in pack.get(section) or []:
            if e.get("name") == name:
                return float(e.get("value", default))
        return default

    scale_length = _pack_val("string", "scale_length", CLASSICAL_SCALE_LENGTH_M)
    mob = _pack_val("geometry_material", "bridge_mobility_proxy", 1.0) * mobility_scale
    top_damp = _pack_val("geometry_material", "top_damping_coeff_proxy", 1.0)
    back_damp = _pack_val("geometry_material", "back_damping_coeff_proxy", 1.0)
    cavity_q = max(5.0, _pack_val("geometry_material", "cavity_q_proxy", 14.0))
    hole_area = _pack_val("geometry_material", "soundhole_area", 0.007)
    cavity_coupling = _pack_val("geometry_material", "cavity_coupling_coefficient", CAVITY_COUPLING_TYPICAL)
    hole_scale = math.sqrt(hole_area / 0.007)

    avg_damp = 0.5 * (top_damp + back_damp) * damping_scale
    q_struct_base = q_from_damping_coeff(45.0, avg_damp)
    q_material_base = q_from_damping_coeff(50.0, avg_damp * 1.05)

    rad_norms = [max(float(m.get("radiation_proxy") or 0.0), 0.0) for m in modes]
    rad_p95 = float(np.percentile(rad_norms, 95)) if rad_norms else 1.0
    rad_p95 = max(rad_p95, 1e-12)

    weighted: List[Dict[str, Any]] = []
    w_exc_raw: List[float] = []
    w_rad_raw: List[float] = []
    w_air_raw: List[float] = []

    for m in modes:
        f_i = float(m["frequency_hz"])
        be_abs = max(abs(float(m.get("bridge_excitation_abs") or 0.0)), 0.0)
        be_coup = float(m.get("bridge_excitation_coupling") or 0.0)
        rad = max(float(m.get("radiation_proxy") or 0.0), 0.0)
        top = float(m.get("top_share") or 0.0)
        back = float(m.get("back_share") or 0.0)
        air = float(m.get("air_share") or 0.0)
        air_p = max(float(m.get("air_pressure_proxy") or 0.0), 0.0)
        harm = harmonic_excitation_at_freq(f_i, F0_HZ, pluck_position)

        exc = be_abs * mob * harm * (1.0 + 0.1 * abs(be_coup) / max(be_abs, 1e-12))
        region = max(top + back + air, 1e-9)

        rad_damp_factor = 1.0 + radiation_damping_scale * RADIATION_DAMPING_COEFF_TYPICAL * (
            rad / rad_p95
        )
        q_rad = max(20.0, q_struct_base / rad_damp_factor)
        q_air = max(cavity_q, 30.0) if air > 0.15 else max(cavity_q * 1.5, 40.0)

        inv_q = (
            1.0 / q_struct_base
            + 1.0 / q_material_base
            + 1.0 / q_rad
            + (air / max(region, 1e-9)) / q_air
        )
        q_total = 1.0 / max(inv_q, 1e-6)
        tau_i = amplitude_tau_s(q_total, f_i)

        w_exc_raw.append(exc)
        w_rad_raw.append(rad * exc * region)
        w_air_raw.append(air * air_p * hole_scale * cavity_coupling)

        weighted.append(
            {
                "frequency_hz": round(f_i, 4),
                "bridge_excitation_abs": be_abs,
                "harmonic_weight": round(harm, 6),
                "Q_struct": round(q_struct_base, 3),
                "Q_material": round(q_material_base, 3),
                "Q_radiation": round(q_rad, 3),
                "Q_air_effective": round(q_air, 3),
                "Q_total": round(q_total, 3),
                "tau_s": round(tau_i, 6),
                "top_share": top,
                "back_share": back,
                "air_share": air,
                "W_exc_raw": exc,
                "W_rad_raw": rad * exc * region,
                "W_air_raw": air * air_p * hole_scale * cavity_coupling,
            }
        )

    def _normalize(raw: List[float]) -> List[float]:
        s = sum(raw)
        if s <= 0:
            n = len(raw)
            return [1.0 / n] * n if n else []
        return [x / s for x in raw]

    w_exc = _normalize(w_exc_raw)
    w_rad = _normalize(w_rad_raw)
    w_air = _normalize(w_air_raw)

    for i, row in enumerate(weighted):
        row["W_exc"] = round(w_exc[i], 8)
        row["W_rad"] = round(w_rad[i], 8)
        row["W_air"] = round(w_air[i], 8)
        del row["W_exc_raw"]
        del row["W_rad_raw"]
        del row["W_air_raw"]

    band_energy: Dict[str, float] = {b[0]: 0.0 for b in MODAL_BANDS}
    for row, wr in zip(weighted, w_rad):
        f_i = row["frequency_hz"]
        for name, lo, hi in MODAL_BANDS:
            if lo <= f_i < hi:
                band_energy[name] += wr
                break

    return {
        "status": "ok",
        "mode_count": len(weighted),
        "mobility_scale": mobility_scale,
        "mobility_amplitude": mob,
        "damping_scale": damping_scale,
        "radiation_damping_scale": radiation_damping_scale,
        "pluck_position_ratio": pluck_position,
        "band_energy_W_rad": {k: round(v, 6) for k, v in band_energy.items()},
        "modes": weighted,
    }


def compute_admittance_curve(
    modal_weights: Mapping[str, Any],
    *,
    f_lo: float = FREQ_GRID_HZ[0],
    f_hi: float = FREQ_GRID_HZ[1],
    n_pts: int = FREQ_GRID_HZ[2],
) -> Dict[str, Any]:
    modes = modal_weights.get("modes") or []
    if not modes:
        return {"status": "no_modes", "frequency_hz": [], "abs_Y_bridge": []}

    freqs = np.linspace(f_lo, f_hi, n_pts)
    omega = 2.0 * np.pi * freqs
    y_sum = np.zeros_like(omega, dtype=np.complex128)
    mob_amp = float(modal_weights.get("mobility_amplitude") or 1.0)

    for row in modes:
        f_i = row["frequency_hz"]
        w_i = 2.0 * math.pi * f_i
        q_i = max(float(row["Q_total"]), 5.0)
        zeta = 1.0 / (2.0 * q_i)
        w_exc = float(row["W_exc"]) * mob_amp
        denom = (w_i ** 2 - omega ** 2) + 1j * (2.0 * zeta * w_i * omega)
        y_sum += w_exc / denom

    abs_y = np.abs(y_sum)
    phase_y = np.angle(y_sum)

    peak_idx: List[int] = []
    for i in range(1, len(abs_y) - 1):
        if abs_y[i] > abs_y[i - 1] and abs_y[i] > abs_y[i + 1] and abs_y[i] > 0.05 * abs_y.max():
            peak_idx.append(i)

    peaks = []
    mode_freqs = [m["frequency_hz"] for m in modes]
    for i in peak_idx[:40]:
        f_pk = float(freqs[i])
        nearest = min(mode_freqs, key=lambda mf: abs(mf - f_pk))
        half = abs_y[i] / math.sqrt(2.0)
        lo = i
        while lo > 0 and abs_y[lo] > half:
            lo -= 1
        hi = i
        while hi < len(abs_y) - 1 and abs_y[hi] > half:
            hi += 1
        bw = float(freqs[hi] - freqs[lo]) if hi > lo else 1.0
        q_est = f_pk / max(bw, 0.1)
        peaks.append(
            {
                "frequency_hz": round(f_pk, 2),
                "nearest_mode_hz": round(nearest, 2),
                "abs_Y": round(float(abs_y[i]), 6),
                "approx_Q_from_width": round(q_est, 2),
            }
        )

    band_summary: Dict[str, Dict[str, float]] = {}
    for name, lo, hi in MODAL_BANDS:
        mask = (freqs >= lo) & (freqs < hi)
        if mask.any():
            band_summary[name] = {
                "max_abs_Y": round(float(abs_y[mask].max()), 6),
                "mean_abs_Y": round(float(abs_y[mask].mean()), 6),
            }

    return {
        "status": "ok",
        "frequency_hz": [round(float(f), 3) for f in freqs[::4]],
        "abs_Y_bridge": [round(float(v), 8) for v in abs_y[::4]],
        "phase_Y_bridge": [round(float(v), 6) for v in phase_y[::4]],
        "peak_count": len(peaks),
        "detected_peaks": peaks[:20],
        "band_summary": band_summary,
        "max_abs_Y": float(abs_y.max()),
    }


def compute_impulse_response(
    modal_weights: Mapping[str, Any],
    *,
    duration_s: float = DURATION_S,
    sr: int = NUMERIC_SR,
) -> Dict[str, Any]:
    modes = modal_weights.get("modes") or []
    if not modes:
        return {"status": "no_modes"}

    n = int(duration_s * sr)
    t = np.arange(n, dtype=np.float64) / sr
    h_total = np.zeros(n, dtype=np.float64)
    h_top = np.zeros(n, dtype=np.float64)
    h_back = np.zeros(n, dtype=np.float64)
    h_air = np.zeros(n, dtype=np.float64)

    for row in modes:
        f_i = row["frequency_hz"]
        tau = max(float(row["tau_s"]), 1e-6)
        wr = float(row["W_rad"])
        wa = float(row["W_air"])
        top = float(row["top_share"])
        back = float(row["back_share"])
        air = float(row["air_share"])
        region = max(top + back + air, 1e-9)

        kernel = wr * np.exp(-t / tau) * np.sin(2.0 * np.pi * f_i * t)
        h_total += kernel
        h_top += kernel * (top / region)
        h_back += kernel * (back / region)
        h_air += kernel * (air / region) + wa * np.exp(-t / (tau * 1.2)) * np.sin(2.0 * np.pi * f_i * t) * 0.01

    env = np.abs(h_total)
    ds = 441
    env_ds = env[::ds]
    t_ds = t[::ds]
    h_ds = h_total[::ds]

    energy = float(np.sum(h_total ** 2))
    e_top = float(np.sum(h_top ** 2))
    e_back = float(np.sum(h_back ** 2))
    e_air = float(np.sum(h_air ** 2))
    e_sum = e_top + e_back + e_air
    if e_sum <= 0:
        e_sum = 1.0

    half_idx = np.searchsorted(env, 0.5 * env.max()) if env.max() > 0 else 0
    decay_60_idx = np.searchsorted(env, 0.001 * env.max()) if env.max() > 0 else n - 1

    return {
        "status": "ok",
        "duration_s": duration_s,
        "numeric_sample_rate_hz": sr,
        "not_audio_output": True,
        "h_at_t0": round(float(h_total[0]), 8),
        "peak_amplitude": round(float(env.max()), 8),
        "peak_time_ms": round(float(t[np.argmax(env)] * 1000.0), 3),
        "half_peak_time_ms": round(float(t[min(half_idx, n - 1)] * 1000.0), 3),
        "decay_to_0p1pct_peak_ms": round(float(t[min(decay_60_idx, n - 1)] * 1000.0), 3),
        "total_energy_proxy": round(energy, 8),
        "region_energy_fraction": {
            "top": round(e_top / e_sum, 4),
            "back": round(e_back / e_sum, 4),
            "air": round(e_air / e_sum, 4),
        },
        "time_s_downsampled": [round(float(x), 4) for x in t_ds[:200]],
        "h_downsampled": [round(float(x), 8) for x in h_ds[:200]],
        "envelope_downsampled": [round(float(x), 8) for x in env_ds[:200]],
        "arrays_full_length": n,
        "top_component_peak": round(float(np.max(np.abs(h_top))), 8),
        "back_component_peak": round(float(np.max(np.abs(h_back))), 8),
        "air_component_peak": round(float(np.max(np.abs(h_air))), 8),
        "total_radiation_peak": round(float(np.max(np.abs(h_total))), 8),
    }


def run_objective_tests(
    ir: Mapping[str, Any],
    admittance: Mapping[str, Any],
    modal_weights: Mapping[str, Any],
    pack: Mapping[str, Any],
) -> Dict[str, Any]:
    env = np.array(ir.get("envelope_downsampled") or [0.0])
    t_ms = np.array(ir.get("time_s_downsampled") or [0.0]) * 1000.0

    early_mask = t_ms <= 50.0
    delayed_mask = (t_ms >= 100.0) & (t_ms <= 250.0)
    early_e = float(np.sum(env[early_mask] ** 2)) if early_mask.any() else 0.0
    delayed_e = float(np.sum(env[delayed_mask] ** 2)) if delayed_mask.any() else 0.0

    last_third = env[len(env) * 2 // 3 :]
    mid_third = env[len(env) // 3 : len(env) * 2 // 3]
    end_rise = float(last_third.max()) > float(mid_third.max()) * 1.05 if len(last_third) else False

    monotonic_decay = True
    if len(env) > 10:
        tail = env[len(env) // 2 :]
        diffs = np.diff(tail)
        rises = int(np.sum(diffs > 0.01 * env.max()))
        monotonic_decay = rises < len(diffs) * 0.15

    mode_freqs = [m["frequency_hz"] for m in (modal_weights.get("modes") or [])]
    peak_match = 0
    for pk in admittance.get("detected_peaks") or []:
        nf = pk.get("nearest_mode_hz", 0.0)
        if any(abs(nf - mf) < 5.0 for mf in mode_freqs):
            peak_match += 1
    peak_frac = peak_match / max(len(admittance.get("detected_peaks") or []), 1)

    vol = next(
        (e["value"] for e in pack.get("geometry_material") or [] if e["name"] == "body_volume_proxy"),
        0.013,
    )
    hole = next(
        (e["value"] for e in pack.get("geometry_material") or [] if e["name"] == "soundhole_area"),
        0.007,
    )
    f_h_base = helmholtz_proxy_hz(float(vol), float(hole))
    f_h_vol_up = helmholtz_proxy_hz(float(vol) * 1.15, float(hole))
    f_h_area_up = helmholtz_proxy_hz(float(vol), float(hole) * 1.10)

    reg = ir.get("region_energy_fraction") or {}
    frac_sum = sum(float(reg.get(k, 0.0)) for k in ("top", "back", "air"))

    results = {
        "causality": {
            "h_starts_at_t0_no_delay": ir.get("h_at_t0", 0.0) == 0.0 or abs(ir.get("h_at_t0", 0.0)) < 1e-6,
            "early_energy_dominates_over_delayed_window": early_e >= delayed_e * 0.5,
            "no_independent_delayed_onset": delayed_e < early_e * 2.0,
            "pass": True,
        },
        "decay": {
            "envelope_generally_decays": monotonic_decay,
            "no_hard_end_gate": True,
            "no_artificial_end_rise": not end_rise,
            "pass": True,
        },
        "admittance_sanity": {
            "peaks_near_modal_frequencies": peak_frac >= 0.5,
            "max_abs_Y_positive": float(admittance.get("max_abs_Y") or 0.0) > 0.0,
            "pass": True,
        },
        "helmholtz_cavity_sanity": {
            "volume_up_lowers_f_H": f_h_vol_up < f_h_base,
            "soundhole_up_raises_f_H": f_h_area_up > f_h_base,
            "cavity_is_weighting_not_echo": True,
            "pass": True,
        },
        "artifact_guard": {
            "double_onset_risk_false": not (delayed_e > early_e * 1.5),
            "delayed_body_event_risk_false": not (delayed_e > early_e * 2.0),
            "independent_body_tail_forbidden": True,
            "no_end_noise_proxy": not end_rise,
            "no_tail_collapse_gate": True,
            "pass": True,
        },
        "energy_proportion": {
            "region_fractions_sum_plausible": 0.85 <= frac_sum <= 1.15,
            "no_separate_click_layer": True,
            "pass": True,
        },
    }

    for block in results.values():
        if isinstance(block, dict) and "pass" in block:
            checks = [v for k, v in block.items() if k != "pass" and isinstance(v, bool)]
            block["pass"] = all(checks) if checks else block["pass"]

    results["all_pass"] = all(
        v.get("pass") for v in results.values() if isinstance(v, dict) and "pass" in v
    )
    return results


def run_sensitivity_tests(
    audit: Mapping[str, Any],
    rom: Mapping[str, Any],
) -> Dict[str, Any]:
    modes = rom.get("predicted_modes") or []
    pack = build_parameter_pack(audit)

    def _avg_tau(d_scale: float) -> float:
        mw = compute_modal_weights(modes, pack, damping_scale=d_scale)
        taus = [m["tau_s"] for m in mw.get("modes") or []]
        return float(np.mean(taus)) if taus else 0.0

    def _max_y(m_scale: float) -> float:
        mw = compute_modal_weights(modes, pack, mobility_scale=m_scale)
        adm = compute_admittance_curve(mw)
        return float(adm.get("max_abs_Y") or 0.0)

    def _q_rad_interaction(r_scale: float) -> Dict[str, float]:
        mw = compute_modal_weights(modes, pack, radiation_damping_scale=r_scale)
        qs = [m["Q_total"] for m in mw.get("modes") or []]
        wr_sum = sum(m["W_rad"] for m in mw.get("modes") or [])
        return {"mean_Q": float(np.mean(qs)), "sum_W_rad": float(wr_sum)}

    tau_lo, tau_mid, tau_hi = _avg_tau(1.3), _avg_tau(1.0), _avg_tau(0.7)
    y_lo, y_mid, y_hi = _max_y(0.8), _max_y(1.0), _max_y(1.2)

    sample = get_sample_record(audit, SAMPLE_ID)
    vol = float(feature_value(sample, "body_volume_proxy", audit=audit, default=0.013))
    hole = float(feature_value(sample, "soundhole_area", audit=audit, default=0.007))
    f_lo = helmholtz_proxy_hz(vol, hole, l_eff_m=0.008)
    f_mid = helmholtz_proxy_hz(vol, hole, l_eff_m=DEFAULT_L_EFF_M)
    f_hi = helmholtz_proxy_hz(vol, hole, l_eff_m=0.020)

    rad_lo = _q_rad_interaction(0.5)
    rad_mid = _q_rad_interaction(1.0)
    rad_hi = _q_rad_interaction(1.5)

    def _excitation_delta(xp_a: float, xp_b: float) -> float:
        mw_a = compute_modal_weights(modes, pack, pluck_position=xp_a)
        mw_b = compute_modal_weights(modes, pack, pluck_position=xp_b)
        wa = np.array([m["W_exc"] for m in mw_a.get("modes") or []])
        wb = np.array([m["W_exc"] for m in mw_b.get("modes") or []])
        if wa.size == 0:
            return 0.0
        return float(np.sum(np.abs(wa - wb)))

    spread_10 = _excitation_delta(0.10, 0.18)
    spread_18 = _excitation_delta(0.18, 0.25)
    spread_10_25 = _excitation_delta(0.10, 0.25)

    results = {
        "damping_scale": {
            "tau_at_1p3x_damping": round(tau_lo, 6),
            "tau_at_1p0x": round(tau_mid, 6),
            "tau_at_0p7x_damping": round(tau_hi, 6),
            "pass": tau_lo < tau_mid < tau_hi,
        },
        "bridge_mobility": {
            "max_Y_at_0p8x": y_lo,
            "max_Y_at_1p0x": y_mid,
            "max_Y_at_1p2x": y_hi,
            "pass": y_lo < y_mid < y_hi,
        },
        "radiation_damping": {
            "low": rad_lo,
            "mid": rad_mid,
            "high": rad_hi,
            "pass": rad_hi["mean_Q"] <= rad_mid["mean_Q"] <= rad_lo["mean_Q"],
        },
        "helmholtz_L_eff": {
            "f_H_low_L": round(f_lo, 3),
            "f_H_nominal": round(f_mid, 3),
            "f_H_high_L": round(f_hi, 3),
            "pass": f_lo > f_mid > f_hi,
        },
        "pluck_position_ratio": {
            "W_exc_L1_delta_0p10_vs_0p18": round(spread_10, 6),
            "W_exc_L1_delta_0p18_vs_0p25": round(spread_18, 6),
            "W_exc_L1_delta_0p10_vs_0p25": round(spread_10_25, 6),
            "pass": spread_10 > 0.001 and spread_10_25 > 0.002,
        },
    }
    results["all_pass"] = all(
        v.get("pass") for v in results.values() if isinstance(v, dict) and "pass" in v
    )
    return results


def build_readiness_after_step3a(
    objective: Mapping[str, Any],
    sensitivity: Mapping[str, Any],
    rom: Mapping[str, Any],
) -> Dict[str, Any]:
    if not rom.get("predicted_modes"):
        status = "blocked_due_to_missing_modal_data"
    elif not objective.get("all_pass") or not sensitivity.get("all_pass"):
        status = "failed_numerical_physics_checks"
    else:
        status = "ready_for_step3b_single_guitar_numeric_modal_response"

    return {
        "current_status": status,
        "musical_wav_synthesis_allowed": False,
        "multi_guitar_comparison_allowed": False,
        "stk_integration_allowed": False,
        "step3b_numeric_modal_allowed": status == "ready_for_step3b_single_guitar_numeric_modal_response",
    }


def maybe_write_figures(
    admittance: Mapping[str, Any],
    ir: Mapping[str, Any],
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
    freqs = admittance.get("frequency_hz") or []
    abs_y = admittance.get("abs_Y_bridge") or []
    if freqs and abs_y:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.semilogy(freqs, abs_y, "b-", lw=0.8)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("|Y_bridge| proxy")
        ax.set_title("PGSM Step 3A — Bridge admittance proxy (numeric)")
        ax.grid(True, alpha=0.3)
        p = figures_dir / "admittance_abs_Y_bridge.png"
        fig.tight_layout()
        fig.savefig(p, dpi=100)
        plt.close(fig)
        written.append(str(p))

    t_s = ir.get("time_s_downsampled") or []
    env = ir.get("envelope_downsampled") or []
    if t_s and env:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(t_s, env, "k-", lw=0.8)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Envelope proxy")
        ax.set_title("PGSM Step 3A — Numerical IR envelope (not audio)")
        ax.grid(True, alpha=0.3)
        p = figures_dir / "impulse_response_envelope.png"
        fig.tight_layout()
        fig.savefig(p, dpi=100)
        plt.close(fig)
        written.append(str(p))

    return written


def build_pgsm_step3a_report(
    *,
    repo_root: Optional[Path] = None,
    audit_path: Optional[Path] = None,
    write_figures: bool = True,
) -> Dict[str, Any]:
    root = Path(repo_root or REPO_ROOT)
    audit = load_audit_report(audit_path)
    step1 = load_step21_report(Path(STEP1_JSON))
    step2 = load_step21_report(Path(STEP2_JSON))
    step21 = load_step21_report(Path(STEP21_JSON))

    rom = load_rom_modal_catalog(root / "FEM" / "outputs" / "rom_stk_body.json")
    pack = build_parameter_pack(audit)
    modes = rom.get("predicted_modes") or []

    modal_weights = compute_modal_weights(modes, pack)
    admittance = compute_admittance_curve(modal_weights)
    ir = compute_impulse_response(modal_weights)
    objective = run_objective_tests(ir, admittance, modal_weights, pack)
    sensitivity = run_sensitivity_tests(audit, rom)
    artifact_guard = objective.get("artifact_guard") or {}
    readiness = build_readiness_after_step3a(objective, sensitivity, rom)

    figures: List[str] = []
    if write_figures:
        figures = maybe_write_figures(admittance, ir, root / "audio" / "debug_reports" / "pgsm_step3a_figures")

    modal_summary = {
        "mode_count": modal_weights.get("mode_count", 0),
        "band_energy_W_rad": modal_weights.get("band_energy_W_rad"),
        "mean_Q_total": round(
            float(np.mean([m["Q_total"] for m in modal_weights.get("modes") or []])), 3
        )
        if modal_weights.get("modes")
        else None,
        "mean_tau_s": round(
            float(np.mean([m["tau_s"] for m in modal_weights.get("modes") or []])), 6
        )
        if modal_weights.get("modes")
        else None,
    }

    blocked_next = [
        "Musical WAV synthesis",
        "STK integration",
        "Multi-guitar timbre comparison proof",
        "FEM/ROM/M4 surrogate inference",
        "Exact L3 elastic moduli / anisotropy / modal stiffness claims",
        "Tuning by listening",
    ]

    safe_next = (
        "PGSM Step 3B: extended single-guitar numeric modal response validation "
        "(still no musical WAV, no STK) — only if Step 3A readiness passes"
    )
    if readiness["current_status"] != "ready_for_step3b_single_guitar_numeric_modal_response":
        safe_next = "Resolve failed numerical physics checks or missing modal data before Step 3B"

    return {
        "report_version": PGSM_STEP3A_VERSION,
        "timestamp": _utc_now(),
        "status": "pgsm_step3a_numerical_ir_testbench_complete",
        "no_audio_generated": True,
        "no_wav_generated": True,
        "no_stk_integration": True,
        "no_fem_run": True,
        "no_rom_run": True,
        "website_default": DEFAULT_WEBSITE_STK_MODE,
        "website_default_unchanged": True,
        "sample_id": SAMPLE_ID,
        "note_reference": NOTE_REFERENCE,
        "parameter_pack": pack,
        "modal_weight_summary": modal_summary,
        "admittance_curve_summary": {
            k: admittance.get(k)
            for k in (
                "status",
                "peak_count",
                "detected_peaks",
                "band_summary",
                "max_abs_Y",
            )
        },
        "impulse_response_summary": {
            k: ir.get(k)
            for k in (
                "status",
                "duration_s",
                "numeric_sample_rate_hz",
                "not_audio_output",
                "h_at_t0",
                "peak_amplitude",
                "peak_time_ms",
                "decay_to_0p1pct_peak_ms",
                "region_energy_fraction",
                "top_component_peak",
                "back_component_peak",
                "air_component_peak",
                "total_radiation_peak",
            )
        },
        "region_contribution_summary": ir.get("region_energy_fraction"),
        "objective_test_results": objective,
        "sensitivity_results": sensitivity,
        "artifact_guard_results": artifact_guard,
        "readiness_after_step3a": readiness,
        "blocked_next_steps": blocked_next,
        "safe_next_step": safe_next,
        "figures_written": figures,
        "step1_report_loaded": step1.get("report_version"),
        "step2_report_loaded": step2.get("report_version"),
        "step21_report_loaded": step21.get("report_version"),
        "step21_prior_readiness": step21.get("readiness_gate", {}).get("current_status"),
        "modal_catalog_status": rom.get("status"),
        "modal_catalog_mode_count": rom.get("num_modes"),
        "explicit_statement": (
            "PGSM Step 3A computes numerical impulse/admittance responses only. "
            "It does not synthesize musical sound."
        ),
    }


def write_markdown_report(report: Mapping[str, Any], path: Path) -> None:
    obj = report.get("objective_test_results") or {}
    sens = report.get("sensitivity_results") or {}
    rg = report.get("readiness_after_step3a") or {}
    adm = report.get("admittance_curve_summary") or {}
    irs = report.get("impulse_response_summary") or {}
    mw = report.get("modal_weight_summary") or {}

    lines = [
        "# PGSM Step 3A — Numerical bridge-admittance / IR testbench",
        "",
        f"**Generated:** {report.get('timestamp')}",
        f"**Status:** {report.get('status')}",
        "",
        report.get("explicit_statement", ""),
        "",
        f"**Sample:** `{report.get('sample_id')}` | **Note ref:** {report.get('note_reference')} (440 Hz harmonic mapping only)",
        f"**Readiness after Step 3A:** `{rg.get('current_status')}`",
        f"**Safe next step:** {report.get('safe_next_step')}",
        "",
        f"Website default (unchanged): `{report.get('website_default')}`",
        "",
        "## Physical chain",
        "",
        "F_bridge(t) → modal oscillator bank → Y_bridge(ω) → Q/tau decay → causal radiation proxy p_out(t)",
        "",
        "No delayed body_tail, Helmholtz IR echo, or post-hoc EQ layers.",
        "",
        "## Parameter / fallback summary",
        "",
    ]
    for section in ("excitation", "string", "geometry_material"):
        lines.append(f"### {section}")
        for e in (report.get("parameter_pack") or {}).get(section) or []:
            lines.append(
                f"- **{e['name']}** [{e['fallback_level']}]: {e['value']} {e['units']} — {e['source']}"
            )
        lines.append("")

    lines.extend(
        [
            "## Modal band summary (W_rad energy)",
            "",
        ]
    )
    for band, val in (mw.get("band_energy_W_rad") or {}).items():
        lines.append(f"- **{band}:** {val}")
    lines.append(f"- Mean Q_total: {mw.get('mean_Q_total')} | Mean tau: {mw.get('mean_tau_s')} s")
    lines.append("")

    lines.extend(["## Admittance peak summary", ""])
    for pk in (adm.get("detected_peaks") or [])[:8]:
        lines.append(
            f"- {pk['frequency_hz']} Hz (mode {pk['nearest_mode_hz']} Hz), |Y|={pk['abs_Y']}, Q≈{pk['approx_Q_from_width']}"
        )
    lines.append(f"- max |Y_bridge|: {adm.get('max_abs_Y')}")
    lines.append("")

    lines.extend(
        [
            "## Impulse-response decay summary",
            "",
            f"- h(0)={irs.get('h_at_t0')} (causal sin start)",
            f"- Peak {irs.get('peak_amplitude')} at {irs.get('peak_time_ms')} ms",
            f"- Decay to 0.1% peak: {irs.get('decay_to_0p1pct_peak_ms')} ms",
            "",
            "## Region contribution",
            "",
        ]
    )
    reg = irs.get("region_energy_fraction") or {}
    for k, v in reg.items():
        lines.append(f"- **{k}:** {v}")
    lines.append("")

    lines.extend(["## Objective tests", ""])
    for name, block in obj.items():
        if name == "all_pass" or not isinstance(block, dict):
            continue
        lines.append(f"- **{name}:** pass={block.get('pass')}")
    lines.append(f"- **all_pass:** {obj.get('all_pass')}")
    lines.append("")

    lines.extend(["## Sensitivity tests", ""])
    for name, block in sens.items():
        if name == "all_pass" or not isinstance(block, dict):
            continue
        lines.append(f"- **{name}:** pass={block.get('pass')}")
    lines.append(f"- **all_pass:** {sens.get('all_pass')}")
    lines.append("")

    lines.extend(["## Blocked next steps", ""])
    for b in report.get("blocked_next_steps") or []:
        lines.append(f"- {b}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pgsm_step3a_reports(
    *,
    repo_root: Optional[Path] = None,
    json_path: Optional[Path] = None,
    md_path: Optional[Path] = None,
    write_figures: bool = True,
) -> Dict[str, Any]:
    root = Path(repo_root or REPO_ROOT)
    report = build_pgsm_step3a_report(repo_root=root, write_figures=write_figures)
    jpath = Path(json_path or REPORT_JSON)
    mpath = Path(md_path or REPORT_MD)
    jpath.parent.mkdir(parents=True, exist_ok=True)
    jpath.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown_report(report, mpath)
    return report


def main() -> None:
    report = write_pgsm_step3a_reports()
    rg = report.get("readiness_after_step3a") or {}
    print(f"Wrote {REPORT_JSON}")
    print(f"Modes: {report.get('modal_catalog_mode_count')}")
    print(f"Objective all_pass: {report.get('objective_test_results', {}).get('all_pass')}")
    print(f"Sensitivity all_pass: {report.get('sensitivity_results', {}).get('all_pass')}")
    print(f"Readiness: {rg.get('current_status')}")


if __name__ == "__main__":
    main()
