#!/usr/bin/env python3
"""
Classic guitar body-response synthesizer (modal transfer-function model).

Signal path (physically motivated):
  plucked-string harmonics -> bridge acceleration
  -> H_body(f) = sum_m W_m H_m(f)  [primary timbre / radiation]
  + small direct attack tap      [pitch anchor / pluck clarity only]

All modes in the validated body band contribute. Final RMS target + soft limiter.
"""
from __future__ import annotations

import json
import math
import wave
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from diagnostic_synthesis import (  # noqa: E402
    active_diagnostic,
    active_sample_parameters,
    blend_toward_unity,
    broad_signature_curve,
    compute_broad_body_band_gains,
    compute_note_reward_score,
    use_diagnostic_mode,
)
from modal_damping import (  # noqa: E402
    compute_per_mode_damping,
    infer_mode_category,
    normalize_participation_shares,
    summarize_mode_damping_records,
)
from string_body_balance import (  # noqa: E402
    body_color_gain_by_f0,
    fundamental_anchor_scale_by_body_strength,
    low_body_color_strength,
    pitch_layer_scale_by_f0,
    string_direct_scale_by_f0,
)
from synthesis_presets import (  # noqa: E402
    BODY_LOW_FREQ_TILT_HZ,
    DEFAULT_SYNTHESIS_PRESET,
    active_tuning,
    use_synthesis_preset,
)

FULL_MODAL_BAND_HZ: Tuple[float, float] = (60.0, 550.0)
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_DURATION_S = 3.0
DEFAULT_VELOCITY = 1.0

# Fixed excitation (body ROM weights are the main guitar-to-guitar variable).
FIXED_PLUCK_POSITION = 0.18
BODY_REFERENCE_GAIN = 1.0
BODY_MODAL_RICHNESS_GAIN = 1.20
BODY_MODAL_GAIN = 1.0
BODY_TO_STRING_TARGET_RATIO = 4.2
STRING_PLUCK_GAIN = 0.10
STRING_PITCH_LAYER_GAIN = 0.055
STRING_PITCH_LAYER_DECAY_S = 0.14
BODY_MODAL_BANDWIDTH_WIDENING = 1.12
TOP_MODE_DOMINANCE_SOFTEN = 0.14
MODAL_MAG_SMOOTH_BINS = 3
FADE_IN_MS = 5.0
FADE_OUT_MS_MIN = 35.0
FADE_OUT_MS_MAX = 70.0
FADE_OUT_FRAC_OF_DURATION = 0.025
PREVIEW_CROSSFADE_MS = 10.0
PREVIEW_SILENCE_MS = 40.0
# Backward-compatible aliases used in older metadata/tests.
TARGET_BODY_TO_ATTACK_RMS_RATIO = BODY_TO_STRING_TARGET_RATIO
DIRECT_ATTACK_GAIN = STRING_PLUCK_GAIN
ATTACK_DECAY_S = 0.040
PLUCK_TRANSIENT_MS = 0.006
FIXED_RAD_K = 0.08
MAX_HARMONICS = 48
Q_MIN = 22.0
Q_MAX = 75.0
CONTRIBUTION_THRESHOLD_REL = 1e-5
TOP_CONTRIBUTING_MODES_N = 15
HARMONIC_ROLLOFF_POWER = 1.15
TARGET_RMS_DBFS = -18.0
FINAL_PEAK_CEILING_DBFS = -1.0
LOW_NOTE_FUNDAMENTAL_MAX_HZ = 165.0
FUNDAMENTAL_ANCHOR_GAIN = 0.14
FUNDAMENTAL_ANCHOR_DECAY_S = 1.35
LOW_NOTE_FUNDAMENTAL_HARMONIC_BOOST = 1.42
PRODUCTION_BROAD_ALL_MODE_STRENGTH = 0.18
BROAD_MODE_MIN_FRACTION = 0.10
NEAR_HARMONIC_WINDOW_REL = 0.10

# Temporal decay (note / harmonic / body radiation)
NOTE_DECAY_REF_HZ = 82.41
NOTE_DECAY_TAU_MIN_S = 0.42
NOTE_DECAY_TAU_MAX_S = 2.65
NOTE_DECAY_FREQ_POWER = 0.58
HARMONIC_DECAY_FACTOR = 0.42
PARTIAL_FREQ_DECAY_HZ = 280.0
BODY_DECAY_TAU_MIN_S = 0.32
BODY_DECAY_TAU_MAX_S = 2.35
BODY_DECAY_FREQ_POWER = 0.52
RADIATION_TAU_SHORTENING = 0.58
BODY_DECAY_LOW_NOTE_BLEND = 0.48
HIGH_NOTE_DECAY_THRESHOLD_HZ = 300.0
HIGH_NOTE_PLUCK_SOFTEN_THRESHOLD_HZ = 300.0
HIGH_NOTE_PLUCK_SOFTEN_FULL_HZ = 620.0
HIGH_NOTE_PLUCK_TRANSIENT_BOOST = 0.75
# Preset-driven (see synthesis_presets.py): gain floor, transient reduction, HF rolloff, etc.
LOUDNESS_RMS_WINDOW_START_S = 0.025
LOUDNESS_RMS_WINDOW_END_S = 0.42
DECAY_EARLY_END_S = 0.28
DECAY_LATE_START_S = 2.05
DECAY_SLOPE_T_START_S = 0.10
DECAY_SLOPE_T_END_S = 2.55
HARMONIC_DECAY_MODEL = (
    "tau_k = note_base_tau(f0) / (1 + harmonic_decay_factor*(k-1)); "
    "partial_freq_scale; body_env = exp(-t/tau_body(f0,radiation))"
)

ModalInput = Union[Mapping[str, Any], Sequence[Mapping[str, Any]]]


def _safe_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        out = float(val)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(math.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def _rms_window(
    x: np.ndarray,
    sample_rate: int,
    start_s: float,
    end_s: float,
) -> float:
    n = len(x)
    if n == 0:
        return 0.0
    i0 = max(0, min(n, int(start_s * sample_rate)))
    i1 = max(i0 + 1, min(n, int(end_s * sample_rate)))
    return _rms(x[i0:i1])


def _linear_from_dbfs(dbfs: float) -> float:
    return float(10.0 ** (dbfs / 20.0))


def _dbfs_from_linear(amplitude: float) -> float:
    if amplitude <= 1e-12:
        return -120.0
    return float(20.0 * math.log10(amplitude))


def _mode_radiation_damping_scale(mode: Mapping[str, Any]) -> float:
    """Air/radiation increases energy loss (lower effective Q)."""
    rad = _safe_float(mode.get("radiation_proxy"))
    air = _safe_float(mode.get("air_share"))
    mic = _safe_float(mode.get("mic_output_proxy"))
    scale = 1.0
    if rad is not None and rad > 0:
        scale += 0.65 * min(rad, 0.05) / 0.05
    if air is not None and air > 0:
        scale += 0.45 * min(air, 0.5)
    if mic is not None and mic > 0:
        scale += 0.20 * min(mic, 0.05) / 0.05
    return max(1.0, min(2.8, scale))


def parse_modal_modes(modal_data: ModalInput) -> Tuple[List[Dict[str, Any]], List[str]]:
    defaults: List[str] = []
    if isinstance(modal_data, list):
        modes = [_normalize_mode_record(m, defaults) for m in modal_data]
        return [m for m in modes if m.get("frequency_hz")], defaults

    if not isinstance(modal_data, dict):
        raise TypeError(f"modal_data must be dict or list, got {type(modal_data)}")

    if modal_data.get("predicted_modes"):
        modes = [_normalize_mode_record(m, defaults) for m in modal_data["predicted_modes"]]
        return [m for m in modes if m.get("frequency_hz")], defaults

    if modal_data.get("modes"):
        modes = [_normalize_mode_record(m, defaults) for m in modal_data["modes"]]
        return [m for m in modes if m.get("frequency_hz")], defaults

    freqs = modal_data.get("modes_hz") or modal_data.get("frequencies_hz") or []
    weights = modal_data.get("mode_weights") or []
    modes = []
    for i, raw_f in enumerate(freqs):
        f_hz = _safe_float(raw_f)
        if f_hz is None or f_hz <= 0:
            continue
        rec: Dict[str, Any] = {"frequency_hz": f_hz, "mode_index": i}
        w = _safe_float(weights[i]) if i < len(weights) else None
        if w is not None:
            rec["mode_weight_fallback"] = w
        else:
            defaults.append("mode_weights:rolloff_fallback")
        modes.append(rec)
    if modes and not any("mode_weight_fallback" in m for m in modes):
        defaults.append("mode_weights:1/(1+0.25*i)")
    return modes, defaults


def _normalize_mode_record(raw: Mapping[str, Any], defaults: List[str]) -> Dict[str, Any]:
    rec = dict(raw)
    f_hz = _safe_float(rec.get("frequency_hz") or rec.get("freq_hz") or rec.get("f_hz"))
    if f_hz is not None:
        rec["frequency_hz"] = f_hz
    return rec


def modes_in_validated_band(modes: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    lo, hi = FULL_MODAL_BAND_HZ
    out: List[Dict[str, Any]] = []
    for m in modes:
        f = _safe_float(m.get("frequency_hz"))
        if f is None or f < lo or f > hi:
            continue
        out.append(dict(m))
    out.sort(key=lambda r: float(r["frequency_hz"]))
    return out


def available_modal_stats(modes: Sequence[Mapping[str, Any]]) -> Tuple[int, Optional[float], Optional[float]]:
    freqs = [float(m["frequency_hz"]) for m in modes if _safe_float(m.get("frequency_hz"))]
    if not freqs:
        return 0, None, None
    return len(freqs), min(freqs), max(freqs)


def compute_mode_weight_components(
    mode: Mapping[str, Any],
    *,
    defaults_used: List[str],
    flags: Dict[str, bool],
) -> Dict[str, float]:
    bridge_w = 1.0
    mic_w = 1.0
    rad_w = 1.0

    combined = _safe_float(mode.get("bridge_to_mic_gain_raw"))
    if combined is not None and combined > 0:
        flags["bridge_weighting_used"] = True
        flags["mic_proxy_used"] = True
        return {
            "bridge_weight": 1.0,
            "mic_weight": 1.0,
            "radiation_weight": 1.0,
            "combined": combined,
        }

    bridge = _safe_float(mode.get("bridge_excitation_abs"))
    if bridge is None:
        coup = _safe_float(mode.get("bridge_excitation_coupling"))
        bridge = abs(coup) if coup is not None else None
    bridge_missing = bridge is None or bridge <= 0
    if bridge_missing:
        bridge = 1.0
        if not _radiation_color_v2_active():
            defaults_used.append("bridge_excitation_abs=1.0")
    else:
        flags["bridge_weighting_used"] = True
    bridge_w = bridge

    mic = _safe_float(mode.get("mic_output_proxy"))
    mic_missing = mic is None or mic <= 0
    if not mic_missing:
        flags["mic_proxy_used"] = True
        mic_w = mic
    else:
        mic_w = 1.0
        if not _radiation_color_v2_active():
            defaults_used.append("mic_output_proxy=1.0")

    rad = _safe_float(mode.get("radiation_proxy"))
    rad_missing = rad is None or rad <= 0
    if not rad_missing:
        flags["radiation_proxy_used"] = True
        rad_w = rad
    else:
        rad_w = 1.0
        if not _radiation_color_v2_active():
            defaults_used.append("radiation_proxy=1.0")

    w = bridge_w * (0.55 * mic_w + 0.45 * rad_w)
    fallback = _safe_float(mode.get("mode_weight_fallback"))
    if fallback is not None and fallback > 0:
        w *= fallback

    top = _safe_float(mode.get("top_share"))
    back = _safe_float(mode.get("back_share"))
    air = _safe_float(mode.get("air_share"))
    if top is not None or back is not None or air is not None:
        flags["participation_used"] = True
        share_sum = (top or 0.0) + (back or 0.0) + (air or 0.0)
        w *= max(0.35, min(1.2, 0.5 + 0.5 * share_sum))

    return {
        "bridge_weight": bridge_w,
        "mic_weight": mic_w,
        "radiation_weight": rad_w,
        "combined": max(w, 1e-12),
        "bridge_proxy_missing": bridge_missing,
        "radiation_proxy_missing": rad_missing,
        "mic_proxy_missing": mic_missing,
    }


def _per_mode_damping_strength() -> float:
    diag = active_diagnostic()
    if diag is None:
        return 1.0
    return max(float(diag.per_mode_damping_strength), float(diag.damping_strength))


def compute_mode_damping_record(
    mode: Mapping[str, Any],
    f_hz: float,
    defaults_used: List[str],
) -> Dict[str, Any]:
    """Per-mode Q/tau/bandwidth — always used in body synthesis."""
    record = compute_per_mode_damping(
        mode,
        f_hz,
        active_sample_parameters(),
        strength=_per_mode_damping_strength(),
        rad_k=FIXED_RAD_K,
    )
    defaults_used.append("per_mode_damping_computed")
    return record


def estimate_mode_q(mode: Mapping[str, Any], f_hz: float, defaults_used: List[str]) -> float:
    return float(compute_mode_damping_record(mode, f_hz, defaults_used)["mode_q"])


def _harmonic_proximity(f_m: float, harmonics_hz: Sequence[float]) -> float:
    if not harmonics_hz:
        return 0.0
    best = 1.0
    for h in harmonics_hz:
        hh = max(float(h), 1.0)
        rel = abs(f_m - hh) / hh
        best = min(best, rel)
    if best >= NEAR_HARMONIC_WINDOW_REL * 2.5:
        return 0.0
    return float(max(0.0, 1.0 - best / max(NEAR_HARMONIC_WINDOW_REL * 2.5, 1e-9)))


def _broad_all_mode_strength() -> float:
    diag = active_diagnostic()
    if diag and diag.all_mode_broad_contribution:
        return float(diag.broad_all_mode_strength)
    return PRODUCTION_BROAD_ALL_MODE_STRENGTH


def _near_modal_boost() -> float:
    diag = active_diagnostic()
    if diag and diag.all_mode_broad_contribution:
        return float(diag.near_modal_boost)
    return 1.0


def _far_mode_color_gain() -> float:
    diag = active_diagnostic()
    if diag and diag.far_mode_color_gain > 0:
        return float(diag.far_mode_color_gain)
    return 1.0


def _low_note_fundamental_harmonic_boost() -> float:
    diag = active_diagnostic()
    if diag:
        return float(diag.low_note_fundamental_harmonic_boost)
    return LOW_NOTE_FUNDAMENTAL_HARMONIC_BOOST


def _radiation_color_v1_active() -> bool:
    diag = active_diagnostic()
    if diag is None:
        return False
    if diag.name == "modal_radiation_color_v1":
        return True
    from body_signature_v3 import is_v3_family_mode

    return is_v3_family_mode(diag.name)


def _body_signature_v3_ablation():
    diag = active_diagnostic()
    if diag is None:
        return None
    from body_signature_v3 import get_v3_ablation

    return get_v3_ablation(diag.name)


def _radiation_color_v2_active() -> bool:
    diag = active_diagnostic()
    return diag is not None and diag.name == "modal_radiation_color_v2"


def _radiation_color_diagnostic_active() -> bool:
    return _radiation_color_v1_active() or _radiation_color_v2_active()


V2_TOP_AMP = 1.05
V2_BACK_AMP = 0.95
V2_AIR_AMP = 1.12
V2_COUPLED_AMP = 1.0
V2_PROXY_MISSING_FLOOR = 0.12


def _proxy_pool_from_modes(band_modes: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    bridges: List[float] = []
    rads: List[float] = []
    mics: List[float] = []
    bridge_missing = rad_missing = mic_missing = 0
    for mode in band_modes:
        b = _safe_float(mode.get("bridge_excitation_abs"))
        if b is None:
            c = _safe_float(mode.get("bridge_excitation_coupling"))
            b = abs(c) if c is not None else None
        if b is None or b <= 0:
            bridge_missing += 1
        else:
            bridges.append(float(b))
        r = _safe_float(mode.get("radiation_proxy"))
        if r is None or r <= 0:
            rad_missing += 1
        else:
            rads.append(float(r))
        m = _safe_float(mode.get("mic_output_proxy"))
        if m is None or m <= 0:
            mic_missing += 1
        else:
            mics.append(float(m))
    return {
        "bridge_max": max(bridges) if bridges else 0.0,
        "radiation_max": max(rads) if rads else 0.0,
        "mic_max": max(mics) if mics else 0.0,
        "bridge_missing_count": bridge_missing,
        "radiation_missing_count": rad_missing,
        "mic_missing_count": mic_missing,
    }


def _normalized_proxy(
    val: Optional[float],
    pool_max: float,
    *,
    missing: bool,
) -> Tuple[float, bool]:
    if missing or val is None or val <= 0 or pool_max <= 1e-12:
        return V2_PROXY_MISSING_FLOOR, True
    return max(0.08, min(1.0, float(val) / pool_max)), False


def _mode_radiation_v2_factors(
    mode: Mapping[str, Any],
    damp_rec: Mapping[str, Any],
    comp: Mapping[str, Any],
    f_hz: float,
    pools: Mapping[str, Any],
    *,
    note_hz: float,
) -> Dict[str, Any]:
    bridge_abs = _safe_float(mode.get("bridge_excitation_abs"))
    if bridge_abs is None:
        coup = _safe_float(mode.get("bridge_excitation_coupling"))
        bridge_abs = abs(coup) if coup is not None else None
    rad_raw = comp.get("radiation_weight")
    mic_raw = comp.get("mic_weight")
    bridge_miss = bool(comp.get("bridge_proxy_missing")) or bridge_abs is None or bridge_abs <= 0
    bridge, _ = _normalized_proxy(
        float(bridge_abs) if bridge_abs is not None else None,
        float(pools.get("bridge_max") or 0),
        missing=bridge_miss,
    )
    rad, rad_miss = _normalized_proxy(
        float(rad_raw) if rad_raw not in (None, 1.0) else _safe_float(mode.get("radiation_proxy")),
        float(pools.get("radiation_max") or 0),
        missing=bool(comp.get("radiation_proxy_missing")),
    )
    mic, mic_miss = _normalized_proxy(
        float(mic_raw) if mic_raw not in (None, 1.0) else _safe_float(mode.get("mic_output_proxy")),
        float(pools.get("mic_max") or 0),
        missing=bool(comp.get("mic_proxy_missing")),
    )

    bridge_gate = bridge**0.75
    output_transmittance = (0.50 * rad + 0.35 * mic + 0.15 * math.sqrt(max(rad * mic, 1e-9))) ** 0.85

    top_s = float(damp_rec.get("top_share") or 0.0)
    back_s = float(damp_rec.get("back_share") or 0.0)
    air_s = float(damp_rec.get("air_share") or 0.0)
    coupled_s = float(damp_rec.get("coupled_share") or 0.0)
    if top_s + back_s + air_s < 0.05:
        top_s, back_s, air_s, coupled_s = normalize_participation_shares(mode)  # type: ignore[misc]
    mode_category_amp = (
        top_s * V2_TOP_AMP + back_s * V2_BACK_AMP + air_s * V2_AIR_AMP + coupled_s * V2_COUPLED_AMP
    )
    mode_category_amp = max(0.35, min(1.35, mode_category_amp))

    mat = float(damp_rec.get("mode_material_damping") or 1.0)
    material_amp = max(0.88, min(1.12, 1.0 + 0.10 * (mat - 1.0)))
    f_balance = (max(f_hz, 60.0) / 200.0) ** 0.08

    # Audible output = bridge excitation gate × radiation transmittance (not independent sources).
    w_mode = bridge_gate * output_transmittance * mode_category_amp * material_amp * f_balance

    low_s = low_body_color_strength(note_hz)
    if low_s > 0 and f_hz < 300.0:
        low_color = mode_category_amp * output_transmittance
        w_mode *= 1.0 + low_s * 0.42 * max(0.0, low_color - 0.70)

    return {
        "mode_bridge_gate_factor": round(bridge_gate, 6),
        "mode_output_transmittance_factor": round(output_transmittance, 6),
        "mode_category_amplitude_factor": round(mode_category_amp, 6),
        "mode_material_amplitude_factor": round(material_amp, 6),
        "mode_final_amplitude_factor": round(max(w_mode, 1e-9), 6),
        "bridge_proxy_missing": bridge_miss,
        "radiation_proxy_missing": rad_miss,
        "mic_proxy_missing": mic_miss,
        "low_body_color_strength": round(low_s, 6),
    }


def _material_amplitude_factor(damp_rec: Mapping[str, Any]) -> float:
    """Small wood/material tilt on amplitude (±8%), separate from Q damping."""
    mat = float(damp_rec.get("mode_material_damping") or 1.0)
    return max(0.88, min(1.12, 1.0 + 0.08 * (mat - 1.0)))


def _mode_color_band_vector(damp_rec: Mapping[str, Any], comp: Mapping[str, Any]) -> List[float]:
    top = float(damp_rec.get("top_share") or comp.get("top_share") or 0.0)
    back = float(damp_rec.get("back_share") or 0.0)
    air = float(damp_rec.get("air_share") or 0.0)
    coupled = float(damp_rec.get("coupled_share") or 0.0)
    rad = float(comp.get("radiation_weight") or 0.0)
    vec = [
        max(0.0, top * (0.85 + 0.15 * rad)),
        max(0.0, back * (0.85 + 0.10 * rad)),
        max(0.0, air * (0.90 + 0.20 * rad)),
        max(0.0, coupled * 0.80),
    ]
    s = sum(vec) or 1.0
    return [round(v / s, 6) for v in vec]


def _mode_radiation_amplitude_factors(
    damp_rec: Mapping[str, Any],
    comp: Mapping[str, Any],
    mode: Mapping[str, Any],
    f_hz: float,
) -> Dict[str, Any]:
    bridge = max(float(comp.get("bridge_weight") or 1.0), 1e-6)
    rad = max(float(comp.get("radiation_weight") or 1.0), 1e-6)
    mic = max(float(comp.get("mic_weight") or 1.0), 1e-6)
    category = infer_mode_category(mode)
    cat_amp = {"top": 1.06, "back": 0.94, "air": 1.12, "coupled": 1.0}.get(category, 1.0)
    mobility = bridge**0.82
    radiation = (0.42 * rad + 0.33 * mic + 0.25 * math.sqrt(rad * mic)) ** 0.88
    f_eff = (max(f_hz, 60.0) / 200.0) ** 0.10
    mat_amp = _material_amplitude_factor(damp_rec)
    color_vec = _mode_color_band_vector(damp_rec, comp)
    category_color = (
        0.35 * color_vec[0]
        + 0.28 * color_vec[1]
        + 0.27 * color_vec[2]
        + 0.10 * color_vec[3]
    )
    amp = mobility * radiation * cat_amp * mat_amp * f_eff * (0.88 + 0.24 * category_color)
    return {
        "mode_amplitude_factor": round(amp, 6),
        "mode_bridge_coupling_factor": round(mobility, 6),
        "mode_radiation_factor": round(radiation * f_eff, 6),
        "material_amplitude_factor": round(mat_amp, 6),
        "mode_color_band_vector": color_vec,
        "mode_category": category,
    }


def _per_mode_broad_color_scale(damp_rec: Mapping[str, Any], comp: Mapping[str, Any]) -> float:
    """Sample/mode-specific broad-path color — not a uniform EQ curve."""
    if _radiation_color_v2_active():
        amp_info = {
            "mode_output_transmittance_factor": float(damp_rec.get("mode_output_transmittance_factor") or 1.0),
            "mode_category_amplitude_factor": float(damp_rec.get("mode_category_amplitude_factor") or 1.0),
        }
        out_t = float(amp_info["mode_output_transmittance_factor"])
        cat_a = float(amp_info["mode_category_amplitude_factor"])
        q_bw = float(damp_rec.get("mode_bandwidth_hz") or 6.0)
        tau = float(damp_rec.get("mode_tau_s") or 0.2)
        smooth = 0.82 + 0.18 * min(1.0, q_bw / 14.0) * min(1.0, tau / 0.35)
        low_s = float(damp_rec.get("low_body_color_strength") or 0.0)
        f_m = float(damp_rec.get("frequency_hz") or 200.0)
        base = 0.50 + 0.30 * out_t + 0.20 * cat_a
        if low_s > 0 and f_m < 300.0:
            base *= 1.0 + low_s * 0.45 * max(0.0, out_t * cat_a - 0.65)
        return base * smooth
    if _radiation_color_v1_active():
        pseudo_mode = {
            "top_share": damp_rec.get("top_share"),
            "back_share": damp_rec.get("back_share"),
            "air_share": damp_rec.get("air_share"),
        }
        amp_info = _mode_radiation_amplitude_factors(
            damp_rec, comp, pseudo_mode, float(damp_rec.get("frequency_hz") or 200.0)
        )
        vec = amp_info.get("mode_color_band_vector") or [0.25, 0.25, 0.25, 0.25]
        v3 = _body_signature_v3_ablation()
        if v3 and v3.far_color:
            from body_signature_v3 import far_color_factor_v3

            mob = float(damp_rec.get("mobility_factor_m") or 1.0)
            rad_f = float(damp_rec.get("radiation_factor_m") or amp_info.get("mode_radiation_factor") or 1.0)
            return far_color_factor_v3(damp_rec, comp, radiation_factor=rad_f, mobility_factor=mob)
        rad = float(comp.get("radiation_weight") or 0.0)
        return 0.55 + 0.25 * float(amp_info.get("mode_radiation_factor") or 1.0) + 0.20 * sum(vec[:3])
    mat = float(damp_rec.get("mode_material_damping") or 1.0)
    top_c = float(damp_rec.get("top_wood_damping_component") or 0.0)
    back_c = float(damp_rec.get("back_wood_damping_component") or 0.0)
    air_c = float(damp_rec.get("air_damping_component") or 0.0)
    rad = float(comp.get("radiation_weight") or 0.0)
    bridge = float(comp.get("bridge_weight") or 0.0)
    return 0.62 + 0.22 * mat + 0.10 * (top_c + back_c) + 0.06 * air_c + 0.12 * rad + 0.04 * bridge


def apply_per_mode_tau_envelope(
    body: np.ndarray,
    sample_rate: int,
    mode_rows: Sequence[Mapping[str, Any]],
    weights: Sequence[float],
) -> np.ndarray:
    """Apply per-mode tau_s decay in time domain (not metadata-only)."""
    n = len(body)
    if n <= 0 or not mode_rows:
        return body
    t = np.arange(n, dtype=np.float64) / float(sample_rate)
    env = np.zeros(n, dtype=np.float64)
    total_w = 0.0
    for row, w in zip(mode_rows, weights):
        if w <= 0:
            continue
        damp = row.get("damping") or {}
        tau = max(float(damp.get("mode_tau_s") or 0.05), 0.02)
        env += float(w) * np.exp(-t / tau)
        total_w += float(w)
    if total_w <= 0:
        return body
    env /= total_w
    env /= max(float(np.max(env)), 1e-9)
    blended = 0.38 + 0.62 * env
    return body * blended


def _combine_near_and_broad_weights(
    w: float,
    f_m: float,
    harmonics_hz: Sequence[float],
) -> Tuple[float, float, float]:
    """Returns (combined_weight, near_fraction, broad_fraction)."""
    prox = _harmonic_proximity(f_m, harmonics_hz)
    near_boost = _near_modal_boost()
    far_color = _far_mode_color_gain()
    w_near = w * (1.0 + (near_boost - 1.0) * prox)
    broad_s = _broad_all_mode_strength()
    far = 1.0 - prox
    w_broad = w * (BROAD_MODE_MIN_FRACTION + broad_s * (0.22 + 0.78 * far)) * far_color

    diag = active_diagnostic()
    if diag and diag.near_modal_energy_target > 0 and diag.far_broad_energy_target > 0:
        nt = float(diag.near_modal_energy_target)
        ft = float(diag.far_broad_energy_target)
        w_near *= nt / 0.55
        w_broad *= ft / 0.30

    total = w_near + w_broad
    if total <= 0:
        return 0.0, 0.0, 0.0
    return total, w_near / total, w_broad / total


def _total_q_with_radiation_loss(
    q_wood: float,
    f_hz: float,
    rad_k: float,
    *,
    radiation_proxy: float = 0.0,
) -> float:
    inv_q = (1.0 / max(q_wood, 0.5)) + rad_k * (f_hz / 1000.0)
    if radiation_proxy > 0:
        inv_q += 0.035 * radiation_proxy * math.sqrt(max(f_hz, 1.0) / 200.0)
    return max(0.5, 1.0 / max(inv_q, 1e-9))


def _effective_q_with_bandwidth_widening(q: float) -> float:
    return max(0.5, float(q) / active_tuning().body_modal_bandwidth_widening)


def _complex_mode_response(f_hz: np.ndarray, f_m: float, q: float) -> np.ndarray:
    fm = max(float(f_m), 1.0)
    qv = max(_effective_q_with_bandwidth_widening(q), 0.5)
    r = np.asarray(f_hz, dtype=np.float64) / fm
    denom = (1.0 - r * r) + 1.0j * (r / qv)
    return 1.0 / denom


def _soften_mode_weights(weights: Sequence[float]) -> Tuple[List[float], float, float]:
    """Gently reduce top-mode dominance while preserving relative differences."""
    w = np.asarray(weights, dtype=np.float64)
    total = float(np.sum(w))
    if total <= 0.0 or w.size == 0:
        return list(w), 0.0, 0.0
    dom_before = float(np.max(w) / total)
    w_norm = w / total
    w_max = float(np.max(w_norm))
    if w_max <= 1e-12:
        return list(w), dom_before, dom_before
    excess = np.maximum(w_norm / w_max - 0.35, 0.0)
    soften = 1.0 - TOP_MODE_DOMINANCE_SOFTEN * (excess ** 0.65)
    w_new = w * np.maximum(soften, 0.52)
    total_after = float(np.sum(w_new))
    dom_after = float(np.max(w_new) / total_after) if total_after > 0 else dom_before
    return [float(v) for v in w_new], dom_before, dom_after


def _smooth_complex_magnitude(H: np.ndarray, bins: int) -> np.ndarray:
    if bins < 2 or H.size < bins:
        return H
    mag = np.abs(H)
    phase = np.angle(H)
    kernel = np.ones(int(bins), dtype=np.float64) / float(bins)
    mag_s = np.convolve(mag, kernel, mode="same")
    return mag_s * np.exp(1.0j * phase)


def _hf_transfer_envelope(f_hz: np.ndarray) -> np.ndarray:
    _, hi = FULL_MODAL_BAND_HZ
    f = np.asarray(f_hz, dtype=np.float64)
    env = np.ones_like(f)
    above = f > hi
    env[above] = np.maximum(0.06, (hi / f[above]) ** 1.15)
    return env


def high_note_pluck_soften_t(frequency_hz: float) -> float:
    """0 below threshold, 1 at/above full soften frequency."""
    f0 = float(frequency_hz)
    if f0 <= HIGH_NOTE_PLUCK_SOFTEN_THRESHOLD_HZ:
        return 0.0
    span = max(1.0, HIGH_NOTE_PLUCK_SOFTEN_FULL_HZ - HIGH_NOTE_PLUCK_SOFTEN_THRESHOLD_HZ)
    return float(max(0.0, min(1.0, (f0 - HIGH_NOTE_PLUCK_SOFTEN_THRESHOLD_HZ) / span)))


def high_note_pluck_softening_gain(frequency_hz: float) -> float:
    """Reduce direct string/pluck layer for high notes; low notes unchanged."""
    t = high_note_pluck_soften_t(frequency_hz)
    if t <= 0.0:
        return 1.0
    floor = active_tuning().high_note_pluck_gain_floor
    return 1.0 - t * (1.0 - floor)


def high_note_string_hf_rolloff_factor(frequency_hz: float, harmonic_index: int) -> float:
    """Extra HF harmonic attenuation for high fundamentals (reduces metallic ping)."""
    f0 = float(frequency_hz)
    t = high_note_pluck_soften_t(f0)
    if t <= 0.0:
        return 1.0
    k = max(1, int(harmonic_index))
    fk = k * f0
    if k <= 2:
        return 1.0 - 0.10 * t
    hf = max(0.0, min(1.0, (fk - 700.0) / 2200.0))
    rolloff = active_tuning().high_note_hf_rolloff_k_power
    cut = t * (rolloff * (k - 2) ** 0.55 + 0.22 * hf)
    return float(max(0.30, 1.0 - cut))


def harmonic_series(
    frequency_hz: float,
    sample_rate: int,
    *,
    pluck_position: float = FIXED_PLUCK_POSITION,
    max_harmonics: int = MAX_HARMONICS,
) -> Tuple[List[float], List[float]]:
    f0 = max(1.0, float(frequency_hz))
    harm_f: List[float] = []
    harm_a: List[float] = []
    max_harm = min(max_harmonics, int(sample_rate / (2.0 * f0)))
    for k in range(1, max_harm + 1):
        fk = k * f0
        if fk >= sample_rate * 0.49:
            break
        pluck_factor = abs(math.sin(math.pi * pluck_position * k))
        if pluck_factor < 1e-8:
            continue
        amp = pluck_factor / (k ** HARMONIC_ROLLOFF_POWER)
        amp *= high_note_string_hf_rolloff_factor(f0, k)
        if k == 1 and f0 <= LOW_NOTE_FUNDAMENTAL_MAX_HZ:
            amp *= _low_note_fundamental_harmonic_boost()
        harm_f.append(fk)
        harm_a.append(amp)
    return harm_f, harm_a


def nearest_harmonic_hz(mode_hz: float, f0: float, harmonics_hz: Sequence[float]) -> float:
    if not harmonics_hz:
        return f0
    return min(harmonics_hz, key=lambda h: abs(h - mode_hz))


def note_base_decay_tau_s(frequency_hz: float) -> float:
    """Longer sustain for low notes; shorter for high notes."""
    f0 = max(40.0, float(frequency_hz))
    tau = NOTE_DECAY_TAU_MAX_S * (NOTE_DECAY_REF_HZ / f0) ** NOTE_DECAY_FREQ_POWER
    return float(max(NOTE_DECAY_TAU_MIN_S, min(NOTE_DECAY_TAU_MAX_S, tau)))


def harmonic_decay_tau_s(frequency_hz: float, harmonic_index: int) -> float:
    """Per-partial decay: higher k and higher partial frequency → shorter tau."""
    f0 = max(40.0, float(frequency_hz))
    k = max(1, int(harmonic_index))
    base = note_base_decay_tau_s(f0)
    tau = base / (1.0 + HARMONIC_DECAY_FACTOR * (k - 1))
    fk = k * f0
    tau /= 1.0 + 0.22 * (fk / PARTIAL_FREQ_DECAY_HZ) ** 0.9
    if k == 1 and f0 <= LOW_NOTE_FUNDAMENTAL_MAX_HZ:
        tau *= 1.28
    return float(max(0.07, tau))


def summarize_body_radiation(band_modes: Sequence[Mapping[str, Any]]) -> float:
    """0..1 summary of air/radiation heaviness across evaluated modes."""
    if not band_modes:
        return 0.0
    weights: List[float] = []
    rad_vals: List[float] = []
    for mode in band_modes:
        comp = compute_mode_weight_components(mode, defaults_used=[], flags={})
        w = comp["combined"]
        rad = _safe_float(mode.get("radiation_proxy")) or 0.0
        air = _safe_float(mode.get("air_share")) or 0.0
        rad_vals.append(min(1.0, 0.55 * min(rad / 0.05, 1.0) + 0.45 * min(air, 0.5)))
        weights.append(w)
    wsum = sum(weights)
    if wsum <= 0:
        return float(np.mean(rad_vals)) if rad_vals else 0.0
    return float(sum(r * wt for r, wt in zip(rad_vals, weights)) / wsum)


def body_decay_tau_s(note_hz: float, radiation_summary: float) -> float:
    """Body/radiation envelope time constant — high notes and radiating bodies decay faster."""
    f0 = max(40.0, float(note_hz))
    tau = BODY_DECAY_TAU_MAX_S * (NOTE_DECAY_REF_HZ / f0) ** BODY_DECAY_FREQ_POWER
    rad = max(0.0, min(1.0, float(radiation_summary)))
    shorten = 1.0 - rad * RADIATION_TAU_SHORTENING
    tau *= max(0.35, shorten)
    if f0 > HIGH_NOTE_DECAY_THRESHOLD_HZ:
        tau *= (HIGH_NOTE_DECAY_THRESHOLD_HZ / f0) ** 0.35
    return float(max(BODY_DECAY_TAU_MIN_S, min(BODY_DECAY_TAU_MAX_S, tau)))


def apply_exponential_decay_envelope(
    signal: np.ndarray,
    sample_rate: int,
    tau_s: float,
    *,
    floor_mix: float = 0.0,
) -> np.ndarray:
    n = len(signal)
    if n == 0 or tau_s <= 0:
        return signal
    t = np.arange(n, dtype=np.float64) / float(sample_rate)
    # Slight curvature: faster initial loss, smooth tail (not a hard gate).
    env = np.exp(-t / tau_s) * (0.9 + 0.1 * np.exp(-t / max(tau_s * 0.12, 1e-4)))
    if floor_mix > 0:
        env = (1.0 - floor_mix) + floor_mix * env
    return np.asarray(signal, dtype=np.float64) * env


def _decay_analysis_windows(duration_s: float) -> Tuple[float, float, float, float]:
    """Early/late RMS and log-slope fit windows scaled to note length."""
    dur = max(0.1, float(duration_s))
    early_end = min(DECAY_EARLY_END_S, max(0.08, dur * 0.14))
    late_start = DECAY_LATE_START_S if dur >= 1.5 else dur * 0.58
    late_start = min(max(late_start, early_end + 0.05), max(early_end + 0.05, dur - 0.04))
    t_start = min(DECAY_SLOPE_T_START_S, max(0.03, dur * 0.06))
    t_end = max(t_start + 0.12, min(DECAY_SLOPE_T_END_S, dur * 0.94))
    return early_end, late_start, t_start, t_end


def compute_decay_diagnostics(
    samples: np.ndarray,
    sample_rate: int,
) -> Dict[str, Any]:
    x = np.asarray(samples, dtype=np.float64)
    duration_s = len(x) / float(sample_rate) if sample_rate > 0 else 0.0
    early_end, late_start, t_start, t_end = _decay_analysis_windows(duration_s)
    early_rms = _rms_window(x, sample_rate, 0.0, early_end)
    late_rms = _rms_window(x, sample_rate, late_start, duration_s)
    if early_rms > 1e-12 and late_rms > 0:
        late_to_early_db = 20.0 * math.log10(late_rms / early_rms)
    else:
        late_to_early_db = -120.0

    slope = _estimate_decay_slope_db_per_s(x, sample_rate, t_start_s=t_start, t_end_s=t_end)
    return {
        "output_decay_slope_db_per_s": round(slope, 4),
        "early_rms_dbfs": round(_dbfs_from_linear(early_rms), 4),
        "late_rms_dbfs": round(_dbfs_from_linear(late_rms), 4),
        "late_to_early_rms_db": round(late_to_early_db, 4),
    }


def _estimate_decay_slope_db_per_s(
    samples: np.ndarray,
    sample_rate: int,
    *,
    t_start_s: float = DECAY_SLOPE_T_START_S,
    t_end_s: float = DECAY_SLOPE_T_END_S,
) -> float:
    n = len(samples)
    i0 = max(0, min(n, int(t_start_s * sample_rate)))
    i1 = max(i0 + 16, min(n, int(t_end_s * sample_rate)))
    if i1 <= i0 + 16:
        return 0.0
    seg = np.abs(samples[i0:i1])
    win = max(1, int(0.018 * sample_rate))
    kernel = np.ones(win, dtype=np.float64) / float(win)
    env = np.convolve(seg, kernel, mode="same")
    env = np.maximum(env, 1e-12)
    log_env = 20.0 * np.log10(env)
    t = np.arange(len(log_env), dtype=np.float64) / float(sample_rate) + t_start_s
    slope, _ = np.polyfit(t, log_env, 1)
    return float(slope)


def _pluck_attack_envelope(n: int, sample_rate: int, frequency_hz: float) -> np.ndarray:
    """Short onset emphasis for pluck realism; softer transient for high notes."""
    t = np.arange(n, dtype=np.float64) / float(sample_rate)
    soften_t = high_note_pluck_soften_t(frequency_hz)
    reduction = active_tuning().high_note_pluck_transient_reduction
    boost = HIGH_NOTE_PLUCK_TRANSIENT_BOOST * (1.0 - soften_t * reduction)
    transient = 1.0 + boost * np.exp(-t / max(PLUCK_TRANSIENT_MS, 1e-4))
    return transient


def _fundamental_pitch_anchor(
    frequency_hz: float,
    duration_s: float,
    sample_rate: int,
    *,
    velocity: float = DEFAULT_VELOCITY,
) -> np.ndarray:
    """Subtle low-note fundamental anchor — not a dominant pure sine."""
    f0 = float(frequency_hz)
    if f0 > LOW_NOTE_FUNDAMENTAL_MAX_HZ:
        return np.zeros(max(1, int(duration_s * sample_rate)), dtype=np.float64)
    n = max(1, int(duration_s * sample_rate))
    t = np.arange(n, dtype=np.float64) / float(sample_rate)
    attack = 1.0 - np.exp(-t / 0.010)
    decay = np.exp(-t / FUNDAMENTAL_ANCHOR_DECAY_S)
    blend = 0.65 + 0.35 * np.exp(-t / 0.25)
    return (
        FUNDAMENTAL_ANCHOR_GAIN
        * velocity
        * blend
        * np.sin(2.0 * math.pi * f0 * t)
        * attack
        * decay
    )


def _attack_decay_s_for_note(frequency_hz: float) -> float:
    """High notes: faster attack decay → less harsh metallic pick transient."""
    soften_t = high_note_pluck_soften_t(frequency_hz)
    if soften_t <= 0.0:
        return ATTACK_DECAY_S
    shorten = active_tuning().high_note_attack_decay_shorten
    return ATTACK_DECAY_S * (1.0 - soften_t * shorten)


def _direct_attack_tap(dry: np.ndarray, sample_rate: int, frequency_hz: float) -> np.ndarray:
    """Direct string component: short pluck attack for pitch identity."""
    t = np.arange(len(dry), dtype=np.float64) / float(sample_rate)
    attack_env = np.exp(-t / _attack_decay_s_for_note(frequency_hz))
    return dry * attack_env


def _string_pitch_layer(dry: np.ndarray, sample_rate: int, frequency_hz: float) -> np.ndarray:
    """Controlled harmonic string layer — plucked, not a pure sustained sine."""
    t = np.arange(len(dry), dtype=np.float64) / float(sample_rate)
    soften_t = high_note_pluck_soften_t(frequency_hz)
    soften = active_tuning().high_note_pitch_layer_attack_soften
    attack_tc = 0.008 * (1.0 + soften_t * (soften - 1.0))
    attack = 1.0 - np.exp(-t / attack_tc)
    decay = np.exp(-t / STRING_PITCH_LAYER_DECAY_S)
    return dry * attack * decay


def fade_out_ms_for_duration(duration_s: float) -> float:
    ms = float(duration_s) * 1000.0 * FADE_OUT_FRAC_OF_DURATION
    return float(max(FADE_OUT_MS_MIN, min(FADE_OUT_MS_MAX, ms)))


def apply_anti_click_taper(
    samples: np.ndarray,
    sample_rate: int,
    *,
    duration_s: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Short fade-in and release taper so WAV boundaries are near zero (anti-click)."""
    x = np.asarray(samples, dtype=np.float64).copy()
    n = len(x)
    if n == 0:
        return x, {
            "fade_in_ms": FADE_IN_MS,
            "fade_out_ms": 0.0,
            "anti_click_taper_applied": False,
            "end_abs_sample_before_taper": 0.0,
            "end_abs_sample_after_taper": 0.0,
        }

    dur = float(duration_s) if duration_s is not None else n / float(sample_rate)
    fade_in_n = max(1, min(n // 4, int(FADE_IN_MS * 1e-3 * sample_rate)))
    fade_out_ms = fade_out_ms_for_duration(dur)
    fade_out_n = max(1, min(n // 4, int(fade_out_ms * 1e-3 * sample_rate)))

    end_before = float(abs(x[-1]))
    if fade_in_n > 1:
        ramp = np.sin(np.linspace(0.0, 0.5 * math.pi, fade_in_n))
        x[:fade_in_n] *= ramp
    if fade_out_n > 1:
        ramp = np.sin(np.linspace(0.5 * math.pi, 0.0, fade_out_n))
        x[-fade_out_n:] *= ramp
    x[-1] = 0.0
    end_after = float(abs(x[-1]))

    return x, {
        "fade_in_ms": FADE_IN_MS,
        "fade_out_ms": round(fade_out_ms, 4),
        "anti_click_taper_applied": True,
        "end_abs_sample_before_taper": round(end_before, 8),
        "end_abs_sample_after_taper": round(end_after, 8),
    }


def read_wav_float_mono(path: Path) -> Tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        n = wf.getnframes()
        sr = wf.getframerate()
        raw = wf.readframes(n)
        width = wf.getsampwidth()
    if width != 2:
        raise ValueError(f"unsupported sample width {width} in {path}")
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
    return pcm / 32767.0, int(sr)


def concatenate_audio_with_crossfade(
    segments: Sequence[np.ndarray],
    sample_rate: int,
    *,
    crossfade_ms: float = PREVIEW_CROSSFADE_MS,
    silence_ms: float = PREVIEW_SILENCE_MS,
) -> np.ndarray:
    """Concatenate note segments with short silence and cosine crossfades."""
    if not segments:
        return np.zeros(0, dtype=np.float64)
    cross_n = max(0, int(crossfade_ms * 1e-3 * sample_rate))
    silence_n = max(0, int(silence_ms * 1e-3 * sample_rate))
    out: List[np.ndarray] = []
    for i, seg in enumerate(segments):
        s = np.asarray(seg, dtype=np.float64)
        if s.size == 0:
            continue
        if i > 0 and silence_n > 0:
            out.append(np.zeros(silence_n, dtype=np.float64))
        if out and cross_n > 1:
            prev = out[-1]
            if prev.size >= cross_n and s.size >= cross_n:
                fade_out = np.sin(np.linspace(0.5 * math.pi, 0.0, cross_n))
                fade_in = np.sin(np.linspace(0.0, 0.5 * math.pi, cross_n))
                overlap = prev[-cross_n:] * fade_out + s[:cross_n] * fade_in
                out[-1] = np.concatenate([prev[:-cross_n], overlap])
                s = s[cross_n:]
        out.append(s)
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float64)


def synthesize_plucked_string(
    frequency_hz: float,
    duration_s: float,
    sample_rate: int,
    *,
    pluck_position: float = FIXED_PLUCK_POSITION,
    velocity: float = DEFAULT_VELOCITY,
    amplitude: float = 0.38,
) -> np.ndarray:
    n = max(1, int(duration_s * sample_rate))
    t = np.arange(n, dtype=np.float64) / float(sample_rate)
    signal = np.zeros(n, dtype=np.float64)
    f0 = max(1.0, float(frequency_hz))
    harm_f, harm_a = harmonic_series(frequency_hz, sample_rate, pluck_position=pluck_position)
    for fk, ak in zip(harm_f, harm_a):
        amp_k = velocity * amplitude * ak
        k = max(1, int(round(fk / f0)))
        tau_k = harmonic_decay_tau_s(f0, k)
        signal += amp_k * np.sin(2.0 * math.pi * fk * t) * np.exp(-t / tau_k)
    signal *= _pluck_attack_envelope(n, sample_rate, f0)
    return signal


def _string_acceleration(dry: np.ndarray) -> np.ndarray:
    acc = np.zeros_like(dry)
    if len(dry) >= 3:
        acc[1:-1] = dry[:-2] - 2.0 * dry[1:-1] + dry[2:]
        acc[0] = acc[1]
        acc[-1] = acc[-2]
    return acc


def _high_note_hf_fallback(note_hz: float) -> bool:
    return float(note_hz) > FULL_MODAL_BAND_HZ[1]


def synthesize_body_via_transfer_function(
    acc: np.ndarray,
    sample_rate: int,
    band_modes: Sequence[Mapping[str, Any]],
    *,
    defaults_used: List[str],
    flags: Dict[str, bool],
    note_hz: float,
    harmonics_hz: Sequence[float],
    layer_filter: Optional[str] = None,
    use_bridge_mobility_proxy: bool = False,
) -> Tuple[np.ndarray, List[Dict[str, Any]], List[float], Dict[str, Any]]:
    """
    H_body(f) = sum_m W_m H_m(f) on bridge acceleration spectrum.
    Uses stable BODY_REFERENCE_GAIN — no per-guitar H normalization or 1/sqrt(N) scaling.
    """
    n = len(acc)
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sample_rate))
    acc_spec = np.fft.rfft(acc)
    hf_env = _hf_transfer_envelope(freqs)

    mode_rows: List[Dict[str, Any]] = []
    q_values: List[float] = []
    raw_weights: List[float] = []
    per_mode_damping_records: List[Dict[str, Any]] = []
    near_energy = 0.0
    mid_energy = 0.0
    broad_energy = 0.0

    tune = active_tuning()
    v3_ablation = _body_signature_v3_ablation()
    from body_signature_v3 import proxy_pools_from_modes as v3_proxy_pools

    v3_pools = v3_proxy_pools(band_modes) if v3_ablation else {}
    proxy_pools = _proxy_pool_from_modes(band_modes) if _radiation_color_v2_active() else v3_pools
    mobility_meta: Dict[str, Any] = {}
    if use_bridge_mobility_proxy:
        from bridge_mobility_proxy import bridge_body_coupling_factor, compute_body_mass_proxies

        mobility_meta = compute_body_mass_proxies(active_sample_parameters())
    for mode in band_modes:
        f_m = float(mode["frequency_hz"])
        comp = compute_mode_weight_components(mode, defaults_used=defaults_used, flags=flags)
        w = comp["combined"]
        if use_bridge_mobility_proxy:
            bridge_raw = float(comp.get("bridge_weight") or 1.0)
            coupled, mrec = bridge_body_coupling_factor(
                mode, active_sample_parameters(), existing_bridge=bridge_raw
            )
            scale = coupled / max(bridge_raw, 1e-9)
            w *= scale
            comp = dict(comp)
            comp["bridge_weight"] = coupled
            comp["bridge_mobility_scale"] = scale
            mobility_meta = {**mobility_meta, **mrec}
        if tune.body_low_mode_weight < 1.0 and f_m < BODY_LOW_FREQ_TILT_HZ:
            blend = max(0.0, min(1.0, f_m / BODY_LOW_FREQ_TILT_HZ))
            w *= tune.body_low_mode_weight + (1.0 - tune.body_low_mode_weight) * blend
        damp_rec = compute_mode_damping_record(mode, f_m, defaults_used)
        damp_rec["frequency_hz"] = f_m
        per_mode_damping_records.append(damp_rec)
        amp_meta: Dict[str, Any] = {}
        if _radiation_color_v2_active():
            amp_meta = _mode_radiation_v2_factors(
                mode, damp_rec, comp, f_m, proxy_pools, note_hz=note_hz
            )
            w = float(amp_meta["mode_final_amplitude_factor"])
            damp_rec.update(amp_meta)
        elif _radiation_color_v1_active():
            amp_meta = _mode_radiation_amplitude_factors(damp_rec, comp, mode, f_m)
            v3_has_modifiers = v3_ablation is not None and (
                v3_ablation.low_f0_imprint or v3_ablation.mobility or v3_ablation.far_color
            )
            if v3_has_modifiers:
                from body_signature_v3 import decompose_modal_amplitude_v3

                decomp = decompose_modal_amplitude_v3(
                    mode,
                    damp_rec,
                    comp,
                    f_m,
                    amp_meta,
                    v3_pools,
                    ablation=v3_ablation,
                    parameters=active_sample_parameters(),
                    note_hz=note_hz,
                )
                w *= float(decomp["final_modal_amp_m"]) / max(float(decomp["v1_modal_amp_m"]), 1e-9)
                amp_meta = {**amp_meta, **decomp}
            else:
                w *= float(amp_meta["mode_amplitude_factor"])
            damp_rec.update(amp_meta)
        q_total = float(damp_rec["mode_q"])
        flags["q_or_damping_used"] = True
        q_values.append(q_total)
        w_combined, near_frac, broad_frac = _combine_near_and_broad_weights(w, f_m, harmonics_hz)
        raw_weights.append(w_combined)
        H_m = _complex_mode_response(freqs, f_m, q_total)
        mode_rows.append(
            {
                "mode": mode,
                "f_m": f_m,
                "w": w,
                "w_combined": w_combined,
                "near_frac": near_frac,
                "broad_frac": broad_frac,
                "comp": comp,
                "q": q_total,
                "damping": damp_rec,
                "H_m": H_m,
            }
        )

    softened_weights, dom_before, dom_after = _soften_mode_weights(raw_weights)
    if dom_after < dom_before - 1e-6:
        defaults_used.append("top_mode_dominance_softened")

    H_body = np.zeros_like(freqs, dtype=np.complex128)
    H_near = np.zeros_like(freqs, dtype=np.complex128)
    H_broad = np.zeros_like(freqs, dtype=np.complex128)
    for row, w_eff in zip(mode_rows, softened_weights):
        row["w_eff"] = w_eff
        Hm = row["H_m"]
        nf = float(row.get("near_frac") or 0.0)
        bf = float(row.get("broad_frac") or 0.0)
        damp_rec = row.get("damping") or {}
        comp = row.get("comp") or {}
        broad_color = _per_mode_broad_color_scale(damp_rec, comp)
        rad_w = float(comp.get("radiation_weight") or 1.0)
        mic_w = float(comp.get("mic_weight") or 1.0)
        rad_only = (0.55 * rad_w + 0.45 * mic_w) ** 0.9
        if layer_filter == "near":
            H_body += w_eff * nf * Hm
        elif layer_filter == "far":
            H_body += w_eff * bf * Hm * broad_color
        elif layer_filter == "radiation":
            H_body += w_eff * rad_only * (nf * Hm + bf * Hm * broad_color)
        else:
            H_near += w_eff * nf * Hm
            H_broad += w_eff * bf * Hm * broad_color
            H_body += w_eff * nf * Hm + w_eff * bf * Hm * broad_color
        prox = _harmonic_proximity(float(row["f_m"]), harmonics_hz)
        near_part = w_eff * nf
        broad_part = w_eff * bf * broad_color
        if prox >= 0.55:
            near_energy += near_part + broad_part * 0.25
        elif prox >= 0.15:
            mid_energy += near_part + broad_part
        else:
            broad_energy += near_part * 0.35 + broad_part

    H_body *= hf_env
    H_near *= hf_env
    H_broad *= hf_env
    if MODAL_MAG_SMOOTH_BINS >= 2:
        H_body = _smooth_complex_magnitude(H_body, MODAL_MAG_SMOOTH_BINS)
        defaults_used.append("modal_peak_smoothing_applied")

    broad_band_gains: Dict[str, float] = {}
    diag = active_diagnostic()
    broad_sig_strength = 0.0
    if diag and diag.wide_body_signature and not _radiation_color_diagnostic_active():
        broad_sig_strength = float(diag.wide_body_signature_strength) * 0.35
    if broad_sig_strength > 0:
        broad_band_gains = compute_broad_body_band_gains(band_modes)
        H_broad *= broad_signature_curve(freqs, broad_band_gains, strength=broad_sig_strength)
        H_body = H_near + H_broad
        defaults_used.append("broad_path_per_mode_color_plus_weak_band_eq")

    defaults_used.append("far_mode_weights_sample_specific")
    defaults_used.append("per_mode_q_in_complex_pole_response")
    body_spec = acc_spec * H_body * BODY_REFERENCE_GAIN
    body = np.fft.irfft(body_spec, n=n)
    body = apply_per_mode_tau_envelope(body, sample_rate, mode_rows, softened_weights)
    defaults_used.append("per_mode_tau_time_decay_envelope")

    contributions: List[Dict[str, Any]] = []
    f0 = max(float(note_hz), 1.0)
    for row in mode_rows:
        mode = row["mode"]
        w = float(row.get("w_eff", row["w"]))
        comp = row["comp"]
        q_total = row["q"]
        f_m = row["f_m"]
        H_m = row["H_m"] * hf_env
        mode_spec = acc_spec * w * H_m * BODY_REFERENCE_GAIN
        energy = float(np.sum(np.abs(mode_spec) ** 2))
        nearest_h = nearest_harmonic_hz(f_m, f0, harmonics_hz)
        damp_rec = row.get("damping") or {}
        contributions.append(
            {
                "mode_index": int(mode.get("mode_index", -1)),
                "frequency_hz": round(f_m, 4),
                "contribution_weight": energy,
                "bridge_weight": round(comp["bridge_weight"], 8),
                "mic_weight": round(comp["mic_weight"], 8),
                "radiation_weight": round(comp["radiation_weight"], 8),
                "q": round(q_total, 4),
                "mode_q": damp_rec.get("mode_q"),
                "mode_damping": damp_rec.get("mode_damping"),
                "mode_tau_s": damp_rec.get("mode_tau_s"),
                "mode_bandwidth_hz": damp_rec.get("mode_bandwidth_hz"),
                "mode_category": damp_rec.get("mode_category"),
                "nearest_harmonic_hz": round(nearest_h, 4),
            }
        )

    energy_total = near_energy + mid_energy + broad_energy
    broad_colors = [
        _per_mode_broad_color_scale(row.get("damping") or {}, row.get("comp") or {})
        for row in mode_rows
    ]
    far_specificity = float(np.std(broad_colors) / max(float(np.mean(broad_colors)), 1e-9)) if broad_colors else 0.0
    bw = active_tuning().body_modal_bandwidth_widening
    broaden_info = {
        "body_modal_bandwidth_widening": bw,
        "modal_peak_smoothing_applied": MODAL_MAG_SMOOTH_BINS >= 2,
        "top_mode_dominance_before": round(dom_before, 6),
        "top_mode_dominance_after": round(dom_after, 6),
        "effective_q_scale_or_bandwidth_scale": bw,
        "broad_signature_band_gains": broad_band_gains,
        "per_mode_damping": per_mode_damping_records,
        "damping_q_summary": summarize_mode_damping_records(per_mode_damping_records),
        "near_modal_energy_fraction": round(near_energy / max(energy_total, 1e-12), 6),
        "mid_modal_energy_fraction": round(mid_energy / max(energy_total, 1e-12), 6),
        "broad_body_energy_fraction": round(broad_energy / max(energy_total, 1e-12), 6),
        "far_modal_energy_fraction": round(broad_energy / max(energy_total, 1e-12), 6),
        "broad_all_mode_strength": _broad_all_mode_strength(),
        "near_modal_boost": _near_modal_boost(),
        "per_mode_q_used_in_frequency_response": True,
        "per_mode_tau_used_in_time_decay": True,
        "far_mode_weights_sample_specific": True,
        "far_mode_sample_specificity_score": round(far_specificity, 6),
        "radiation_color_v1_active": _radiation_color_v1_active(),
        "modal_radiation_color_v2_active": _radiation_color_v2_active(),
        "body_signature_v3_active": v3_ablation is not None,
        "body_signature_v3_ablation": (
            {
                "low_f0_imprint": v3_ablation.low_f0_imprint,
                "mobility": v3_ablation.mobility,
                "far_color": v3_ablation.far_color,
            }
            if v3_ablation
            else {}
        ),
        "bridge_proxy_missing_count": int(proxy_pools.get("bridge_missing_count") or 0),
        "radiation_proxy_missing_count": int(proxy_pools.get("radiation_missing_count") or 0),
        "mic_proxy_missing_count": int(proxy_pools.get("mic_missing_count") or 0),
        "low_body_color_strength": round(low_body_color_strength(note_hz), 6),
        "layer_filter": layer_filter or "full",
        "bridge_mobility_proxy_active": use_bridge_mobility_proxy,
        "bridge_mobility_summary": mobility_meta if use_bridge_mobility_proxy else {},
    }
    return body, contributions, q_values, broaden_info


def apply_loudness_finalize(
    samples: np.ndarray,
    sample_rate: int,
    *,
    target_rms_dbfs: float = TARGET_RMS_DBFS,
    peak_ceiling_dbfs: float = FINAL_PEAK_CEILING_DBFS,
    rms_window_start_s: float = LOUDNESS_RMS_WINDOW_START_S,
    rms_window_end_s: float = LOUDNESS_RMS_WINDOW_END_S,
    raw_body_variation_preserve: float = 0.0,
    loudness_normalization_strength: float = 1.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Target-RMS gain (early/mid window), then tanh soft limiter, then peak ceiling.
    Windowed RMS avoids lifting a quiet tail when the attack/sustain is louder.
    """
    x = np.asarray(samples, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        raise ValueError("Samples contain NaN or Inf")

    target_rms = _linear_from_dbfs(target_rms_dbfs)
    ceiling = _linear_from_dbfs(peak_ceiling_dbfs)
    rms_window = _rms_window(x, sample_rate, rms_window_start_s, rms_window_end_s)
    rms_full = _rms(x)
    rms_in = math.sqrt(0.68 * rms_window**2 + 0.32 * rms_full**2)
    if rms_in < 1e-12:
        rms_in = max(rms_window, rms_full, 1e-12)
    rms_gain_full = target_rms / max(rms_in, 1e-12)
    preserve = max(raw_body_variation_preserve, 1.0 - max(0.0, min(1.0, loudness_normalization_strength)))
    rms_gain = blend_toward_unity(rms_gain_full, preserve)
    y = x * rms_gain

    peak_pre_limit = float(np.max(np.abs(y))) if y.size else 0.0
    limiter_used = False
    limiter_gr_db = 0.0

    if peak_pre_limit > ceiling * 0.92:
        limiter_used = True
        drive = max(peak_pre_limit / ceiling, 1.0)
        y = ceiling * np.tanh(y / max(peak_pre_limit, 1e-12) * drive) / math.tanh(drive)
        peak_after = float(np.max(np.abs(y))) if y.size else 0.0
        if peak_after > ceiling:
            y *= ceiling / peak_after
            peak_after = ceiling
        if peak_pre_limit > 1e-12:
            limiter_gr_db = _dbfs_from_linear(peak_after) - _dbfs_from_linear(peak_pre_limit)
    else:
        peak_after = peak_pre_limit
        if peak_after > ceiling:
            limiter_used = True
            y *= ceiling / peak_after
            limiter_gr_db = _dbfs_from_linear(ceiling) - _dbfs_from_linear(peak_after)
            peak_after = ceiling

    rms_out = _rms(y)
    info = {
        "target_rms_dbfs": target_rms_dbfs,
        "final_peak_ceiling_dbfs": peak_ceiling_dbfs,
        "rms_gain_applied": rms_gain,
        "raw_body_variation_preserve": raw_body_variation_preserve,
        "loudness_normalization_strength": loudness_normalization_strength,
        "rms_gain_before_preserve": rms_gain_full,
        "loudness_rms_window_s": [rms_window_start_s, rms_window_end_s],
        "peak_before_loudness": peak_pre_limit,
        "limiter_used": limiter_used,
        "limiter_gain_reduction_db": round(limiter_gr_db, 4),
        "output_rms_dbfs": round(_dbfs_from_linear(rms_out), 4),
        "output_peak_dbfs": round(_dbfs_from_linear(peak_after), 4),
        "peak_before_normalize": peak_pre_limit,
        "final_peak_normalization_gain": rms_gain,
    }
    info.update(compute_decay_diagnostics(y, sample_rate))
    return y, info


def write_wav_int16(
    path: Path,
    samples: np.ndarray,
    sample_rate: int,
    *,
    duration_s: Optional[float] = None,
    raw_body_variation_preserve: Optional[float] = None,
) -> Dict[str, Any]:
    """Write mono int16 WAV after anti-click taper and loudness finalize."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tapered, taper_info = apply_anti_click_taper(
        samples,
        sample_rate,
        duration_s=duration_s,
    )
    preserve = 0.0 if raw_body_variation_preserve is None else float(raw_body_variation_preserve)
    loudness_strength = 1.0
    diag = active_diagnostic()
    if diag is not None:
        if raw_body_variation_preserve is None:
            preserve = diag.effective_loudness_preserve()
        loudness_strength = float(diag.final_loudness_normalization_strength)
    y, loudness_info = apply_loudness_finalize(
        tapered,
        sample_rate,
        raw_body_variation_preserve=preserve,
        loudness_normalization_strength=loudness_strength,
    )
    loudness_info.update(taper_info)
    pcm = np.clip(y * 32767.0, -32767, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())
    return loudness_info


def synthesize_note_with_body_response(
    frequency_hz: float,
    note_name: str,
    duration_s: float,
    sample_rate: int,
    modal_data: ModalInput,
    output_wav: Path,
    output_metadata_json: Optional[Path] = None,
    velocity: float = DEFAULT_VELOCITY,
    synthesis_preset: Optional[str] = None,
    diagnostic_mode: Optional[str] = None,
    sample_parameters: Optional[Mapping[str, Any]] = None,
    modal_source: Optional[str] = None,
    repo_root: Optional[Path] = None,
    sample_id: Optional[str] = None,
) -> Dict[str, Any]:
    from body_hybrid_v4 import get_v4_ablation, synthesize_hybrid_v4_note
    from body_hybrid_v4_1 import get_v4_1_ablation, synthesize_hybrid_v4_1_note

    root = repo_root or Path(__file__).resolve().parents[1]
    sid = sample_id or str((sample_parameters or {}).get("sample_id") or "sample_000")

    if get_v4_1_ablation(diagnostic_mode):
        return synthesize_hybrid_v4_1_note(
            frequency_hz=frequency_hz,
            note_name=note_name,
            duration_s=duration_s,
            sample_rate=sample_rate,
            modal_data=modal_data,
            output_wav=output_wav,
            output_metadata_json=output_metadata_json,
            velocity=velocity,
            sample_parameters=sample_parameters,
            modal_source=modal_source,
            diagnostic_mode=str(diagnostic_mode),
            synthesis_preset=synthesis_preset,
            repo_root=root,
            sample_id=sid,
        )

    if get_v4_ablation(diagnostic_mode):
        return synthesize_hybrid_v4_note(
            frequency_hz=frequency_hz,
            note_name=note_name,
            duration_s=duration_s,
            sample_rate=sample_rate,
            modal_data=modal_data,
            output_wav=output_wav,
            output_metadata_json=output_metadata_json,
            velocity=velocity,
            sample_parameters=sample_parameters,
            modal_source=modal_source,
            diagnostic_mode=str(diagnostic_mode),
            synthesis_preset=synthesis_preset,
            repo_root=root,
            sample_id=sid,
        )
    with use_synthesis_preset(synthesis_preset):
        with use_diagnostic_mode(diagnostic_mode, sample_parameters=sample_parameters):
            return _synthesize_note_with_body_response_core(
                frequency_hz=frequency_hz,
                note_name=note_name,
                duration_s=duration_s,
                sample_rate=sample_rate,
                modal_data=modal_data,
                output_wav=output_wav,
                output_metadata_json=output_metadata_json,
                velocity=velocity,
                modal_source=modal_source,
            )


def _synthesize_note_with_body_response_core(
    frequency_hz: float,
    note_name: str,
    duration_s: float,
    sample_rate: int,
    modal_data: ModalInput,
    output_wav: Path,
    output_metadata_json: Optional[Path] = None,
    velocity: float = DEFAULT_VELOCITY,
    modal_source: Optional[str] = None,
    dry_mix_only: bool = False,
) -> Dict[str, Any]:
    tune = active_tuning()
    all_modes, parse_defaults = parse_modal_modes(modal_data)
    band_modes = modes_in_validated_band(all_modes)
    avail_n, avail_min, avail_max = available_modal_stats(all_modes)
    eval_n, eval_min, eval_max = available_modal_stats(band_modes)

    defaults_used: List[str] = list(parse_defaults)
    flags = {
        "bridge_weighting_used": False,
        "mic_proxy_used": False,
        "radiation_proxy_used": False,
        "participation_used": False,
        "q_or_damping_used": False,
    }

    harmonics_hz, _ = harmonic_series(frequency_hz, sample_rate)
    string_excitation = synthesize_plucked_string(
        frequency_hz,
        duration_s,
        sample_rate,
        pluck_position=FIXED_PLUCK_POSITION,
        velocity=velocity,
    )
    acc = _string_acceleration(string_excitation)

    radiation_summary = summarize_body_radiation(band_modes)
    note_decay_tau_s = note_base_decay_tau_s(frequency_hz)
    body_decay_tau_s_val = body_decay_tau_s(frequency_hz, radiation_summary)
    high_note_decay_applied = float(frequency_hz) >= HIGH_NOTE_DECAY_THRESHOLD_HZ

    broaden_info: Dict[str, Any] = {}
    if band_modes:
        body_raw, contributions, q_values, broaden_info = synthesize_body_via_transfer_function(
            acc,
            sample_rate,
            band_modes,
            defaults_used=defaults_used,
            flags=flags,
            note_hz=frequency_hz,
            harmonics_hz=harmonics_hz,
        )
        defaults_used.append(f"body_modal_bandwidth_widening={tune.body_modal_bandwidth_widening}")
        if tune.body_low_mode_weight < 0.999:
            defaults_used.append(f"body_low_mode_weight={tune.body_low_mode_weight}")
        body_floor = (
            BODY_DECAY_LOW_NOTE_BLEND if float(frequency_hz) <= LOW_NOTE_FUNDAMENTAL_MAX_HZ else 0.0
        )
        body_raw = apply_exponential_decay_envelope(
            body_raw,
            sample_rate,
            body_decay_tau_s_val,
            floor_mix=body_floor,
        )
        defaults_used.append("body_radiation_decay_envelope")
    else:
        body_raw = np.zeros_like(string_excitation)
        contributions = []
        q_values = []
        defaults_used.append("no_modes_in_validated_band_60_550:body_bypass")

    pluck_soften = high_note_pluck_softening_gain(frequency_hz)
    high_note_pluck_softening_applied = pluck_soften < 0.999
    hf_rolloff = high_note_string_hf_rolloff_factor(float(frequency_hz), 6)
    soften_t = high_note_pluck_soften_t(frequency_hz)
    diag = active_diagnostic()
    f0 = float(frequency_hz)
    string_direct_scale = string_direct_scale_by_f0(f0)
    pitch_layer_scale = pitch_layer_scale_by_f0(f0)
    if diag and diag.high_note_string_direct_scale < 0.999:
        string_direct_scale *= 1.0 - soften_t * (1.0 - float(diag.high_note_string_direct_scale))
    if diag and diag.high_note_pitch_layer_scale < 0.999:
        pitch_layer_scale *= 1.0 - soften_t * (1.0 - float(diag.high_note_pitch_layer_scale))
    pitch_high_scale = 1.0 - soften_t * (1.0 - tune.high_note_pitch_layer_high_scale)
    pitch_high_scale *= pitch_layer_scale
    effective_pluck_gain = tune.string_pluck_gain * pluck_soften * string_direct_scale
    effective_pitch_gain = tune.string_pitch_layer_gain * pluck_soften * pitch_high_scale
    string_pluck = effective_pluck_gain * _direct_attack_tap(
        string_excitation, sample_rate, float(frequency_hz)
    )
    string_pitch_layer = effective_pitch_gain * _string_pitch_layer(
        string_excitation, sample_rate, float(frequency_hz)
    )
    string_path = string_pluck + string_pitch_layer
    v3_ablation = _body_signature_v3_ablation()
    body_signature_envelope_meta: Dict[str, Any] = {}
    if v3_ablation and v3_ablation.low_f0_imprint and band_modes:
        from body_signature_v3 import (
            apply_harmonic_body_imprint,
            build_body_signature_envelope,
            low_f0_imprint_strength,
        )

        imprint_strength = low_f0_imprint_strength(float(frequency_hz))
        if imprint_strength > 1e-6:
            env = build_body_signature_envelope(
                band_modes, float(frequency_hz), len(string_excitation), sample_rate
            )
            string_path = apply_harmonic_body_imprint(
                string_path,
                sample_rate,
                float(frequency_hz),
                env,
                strength=imprint_strength,
            )
            body_signature_envelope_meta = {
                "low_f0_imprint_strength": round(imprint_strength, 6),
                "body_signature_envelope_applied": True,
            }
            defaults_used.append("v3_low_f0_harmonic_body_imprint")
    string_pluck_gain = effective_pluck_gain
    string_pitch_layer_gain = effective_pitch_gain
    effective_string_pluck_gain = effective_pluck_gain + effective_pitch_gain
    body_rms_before_calibration = _rms(body_raw)
    string_rms_before_mix = _rms(string_path)

    body_target_ratio = tune.body_to_string_target_ratio
    if diag and diag.high_note_body_to_string_target_ratio is not None and soften_t > 0:
        body_target_ratio = (
            body_target_ratio * (1.0 - soften_t)
            + float(diag.high_note_body_to_string_target_ratio) * soften_t
        )
    body_gain_norm_strength = 1.0
    variation_preserve = 0.0
    if diag:
        body_gain_norm_strength = float(diag.body_gain_normalization_strength)
        variation_preserve = diag.effective_body_gain_preserve()
    if body_rms_before_calibration > 1e-15 and string_rms_before_mix > 1e-15:
        calibrated_gain = body_target_ratio * string_rms_before_mix / body_rms_before_calibration
        blend_preserve = max(variation_preserve, 1.0 - body_gain_norm_strength)
        body_gain_applied = calibrated_gain * (1.0 - blend_preserve) + BODY_REFERENCE_GAIN * blend_preserve
        defaults_used.append(f"body_gain_calibration_to_target_ratio={body_target_ratio}")
        if blend_preserve > 0:
            defaults_used.append(f"body_gain_variation_preserve={blend_preserve}")
    elif body_rms_before_calibration > 0:
        body_gain_applied = BODY_REFERENCE_GAIN
        defaults_used.append("body_gain_calibration_fallback")
    else:
        body_gain_applied = 0.0

    body_color_boost = body_color_gain_by_f0(f0)
    if diag and diag.high_note_body_color_boost > 1.0:
        body_color_boost *= 1.0 + soften_t * (float(diag.high_note_body_color_boost) - 1.0)
    body_signal = body_raw * body_gain_applied * tune.body_modal_gain * body_color_boost
    body_rms_before_richness_gain = _rms(body_signal)
    body_signal = body_signal * tune.body_modal_richness_gain
    body_rms_after_richness_gain = _rms(body_signal)
    body_rms_before = body_rms_after_richness_gain
    body_modal_gain = tune.body_modal_gain
    mixed = body_signal + string_path
    defaults_used.append(f"body_modal_richness_gain={tune.body_modal_richness_gain}")
    body_to_string_rms_ratio_before_loudness = body_rms_before / max(string_rms_before_mix, 1e-12)
    fundamental_anchor_used = float(frequency_hz) <= LOW_NOTE_FUNDAMENTAL_MAX_HZ
    anchor_base = float(diag.fundamental_anchor_scale) if diag else 1.0
    fundamental_anchor_scale = fundamental_anchor_scale_by_body_strength(
        f0,
        body_rms=body_rms_before,
        string_rms=string_rms_before_mix,
        base_scale=anchor_base,
    )
    if fundamental_anchor_used and fundamental_anchor_scale > 0.02:
        anchor = _fundamental_pitch_anchor(
            frequency_hz,
            duration_s,
            sample_rate,
            velocity=velocity,
        )
        mixed += fundamental_anchor_scale * anchor
        defaults_used.append("low_note_fundamental_anchor")
    defaults_used.append("note_harmonic_frequency_decay_envelope")
    final_dry_to_body_rms_ratio = string_rms_before_mix / max(body_rms_before, 1e-12)

    if dry_mix_only:
        return {
            "dry_mix": np.asarray(mixed, dtype=np.float64),
            "branch_baseline_rms": round(_rms(mixed), 8),
            "string_rms_before_mix": round(string_rms_before_mix, 8),
            "body_rms_before_mix": round(body_rms_before, 8),
            "body_gain_applied": round(body_gain_applied, 6),
            "defaults_used": defaults_used,
            "note_name": note_name,
            "frequency_hz": float(frequency_hz),
        }

    loudness_info = write_wav_int16(
        Path(output_wav),
        mixed,
        sample_rate,
        duration_s=duration_s,
    )
    body_to_string_rms_ratio_after_loudness = body_to_string_rms_ratio_before_loudness

    hf_fallback = _high_note_hf_fallback(float(frequency_hz))
    max_contrib = max((c["contribution_weight"] for c in contributions), default=0.0)
    threshold = CONTRIBUTION_THRESHOLD_REL * max_contrib if max_contrib > 0 else 0.0
    active_n = sum(1 for c in contributions if c["contribution_weight"] >= threshold)

    top_modes = sorted(contributions, key=lambda c: c["contribution_weight"], reverse=True)[
        :TOP_CONTRIBUTING_MODES_N
    ]
    for row in top_modes:
        row["contribution_weight"] = round(row["contribution_weight"], 8)

    q_sorted = sorted(q_values)
    q_min = q_sorted[0] if q_sorted else None
    q_max = q_sorted[-1] if q_sorted else None
    q_median = q_sorted[len(q_sorted) // 2] if q_sorted else None

    metadata: Dict[str, Any] = {
        "note_name": note_name,
        "frequency_hz": float(frequency_hz),
        "duration_s": float(duration_s),
        "sample_rate": int(sample_rate),
        "pitch_preserved": True,
        "synthesis_model": "modal_transfer_function_H_body_sum_m_Wm_Hm",
        "available_modal_count": avail_n,
        "available_modal_frequency_min_hz": avail_min,
        "available_modal_frequency_max_hz": avail_max,
        "evaluated_modal_count": eval_n,
        "evaluated_modal_frequency_min_hz": eval_min,
        "evaluated_modal_frequency_max_hz": eval_max,
        "active_modal_count_after_threshold": active_n,
        "selected_or_pruned_policy": (
            "all_modes_in_validated_band_60_550_hz;"
            f"post_response_threshold_rel={CONTRIBUTION_THRESHOLD_REL};"
            "no_per_guitar_H_body_peak_normalize;"
            f"body_rms_calibration_target_ratio={tune.body_to_string_target_ratio}"
        ),
        "harmonics_used_hz": [round(h, 4) for h in harmonics_hz],
        "top_contributing_modes": top_modes,
        "full_modal_band_hz": list(FULL_MODAL_BAND_HZ),
        "high_frequency_fallback_used": bool(hf_fallback),
        "bridge_weighting_used": flags["bridge_weighting_used"],
        "mic_proxy_used": flags["mic_proxy_used"],
        "radiation_proxy_used": flags["radiation_proxy_used"],
        "q_or_damping_used": flags["q_or_damping_used"],
        "direct_string_role": "string_pluck_plus_pitch_layer",
        "string_pluck_gain": round(string_pluck_gain, 6),
        "string_pitch_layer_gain": round(string_pitch_layer_gain, 6),
        "high_note_pluck_softening_applied": high_note_pluck_softening_applied,
        "high_note_pluck_softening_gain": round(pluck_soften, 6),
        "string_hf_rolloff_factor": round(hf_rolloff, 6),
        "effective_string_pluck_gain": round(effective_string_pluck_gain, 6),
        "synthesis_preset": tune.name,
        "synthesis_tuning": tune.to_metadata_dict(),
        "body_low_mode_weight": tune.body_low_mode_weight,
        "high_note_pitch_layer_high_scale": round(pitch_high_scale, 6),
        "body_modal_gain": round(body_modal_gain, 6),
        "body_to_string_target_ratio": tune.body_to_string_target_ratio,
        "body_to_string_rms_ratio_before_loudness": round(body_to_string_rms_ratio_before_loudness, 6),
        "body_to_string_rms_ratio_after_loudness": round(body_to_string_rms_ratio_after_loudness, 6),
        "direct_string_gain": round(string_pluck_gain, 6),
        "body_filter_gain": round(BODY_REFERENCE_GAIN, 6),
        "body_rms_before_calibration": round(body_rms_before_calibration, 8),
        "body_modal_richness_gain": tune.body_modal_richness_gain,
        "body_rms_before_richness_gain": round(body_rms_before_richness_gain, 8),
        "body_rms_after_richness_gain": round(body_rms_after_richness_gain, 8),
        "dry_mix": round(string_pluck_gain, 6),
        "wet_mix": round(body_gain_applied, 6),
        "dry_rms_before_mix": round(string_rms_before_mix, 8),
        "string_rms_before_mix": round(string_rms_before_mix, 8),
        "body_rms_before_mix": round(body_rms_before, 8),
        "target_body_to_attack_rms_ratio": tune.body_to_string_target_ratio,
        "dry_gain_applied": round(string_pluck_gain, 6),
        "body_gain_applied": round(body_gain_applied, 6),
        "final_dry_to_body_rms_ratio": round(final_dry_to_body_rms_ratio, 6),
        "fundamental_anchor_used": fundamental_anchor_used,
        "target_rms_dbfs": loudness_info["target_rms_dbfs"],
        "final_peak_ceiling_dbfs": loudness_info["final_peak_ceiling_dbfs"],
        "output_rms_dbfs": loudness_info["output_rms_dbfs"],
        "output_peak_dbfs": loudness_info["output_peak_dbfs"],
        "limiter_used": loudness_info["limiter_used"],
        "limiter_gain_reduction_db": loudness_info["limiter_gain_reduction_db"],
        "rms_gain_applied": round(loudness_info["rms_gain_applied"], 6),
        "final_peak_normalization_gain": round(loudness_info["final_peak_normalization_gain"], 6),
        "peak_before_normalize": loudness_info["peak_before_normalize"],
        "output_decay_slope_db_per_s": loudness_info["output_decay_slope_db_per_s"],
        "early_rms_dbfs": loudness_info["early_rms_dbfs"],
        "late_rms_dbfs": loudness_info["late_rms_dbfs"],
        "late_to_early_rms_db": loudness_info["late_to_early_rms_db"],
        "note_decay_tau_s": round(note_decay_tau_s, 4),
        "body_decay_tau_s": round(body_decay_tau_s_val, 4),
        "harmonic_decay_model": HARMONIC_DECAY_MODEL,
        "high_note_decay_applied": high_note_decay_applied,
        "body_radiation_summary": round(radiation_summary, 4),
        "body_modal_bandwidth_widening": broaden_info.get(
            "body_modal_bandwidth_widening", tune.body_modal_bandwidth_widening
        ),
        "modal_peak_smoothing_applied": bool(
            broaden_info.get("modal_peak_smoothing_applied", MODAL_MAG_SMOOTH_BINS >= 2)
        ),
        "top_mode_dominance_before": broaden_info.get("top_mode_dominance_before"),
        "top_mode_dominance_after": broaden_info.get("top_mode_dominance_after"),
        "effective_q_scale_or_bandwidth_scale": broaden_info.get(
            "effective_q_scale_or_bandwidth_scale", tune.body_modal_bandwidth_widening
        ),
        "fade_in_ms": loudness_info.get("fade_in_ms"),
        "fade_out_ms": loudness_info.get("fade_out_ms"),
        "anti_click_taper_applied": loudness_info.get("anti_click_taper_applied"),
        "end_abs_sample_before_taper": loudness_info.get("end_abs_sample_before_taper"),
        "end_abs_sample_after_taper": loudness_info.get("end_abs_sample_after_taper"),
        "q_min": q_min,
        "q_median": q_median,
        "q_max": q_max,
        "defaults_used": sorted(set(defaults_used)),
        "excitation": {
            "pluck_position": FIXED_PLUCK_POSITION,
            "velocity": float(velocity),
            "attack_decay_s": ATTACK_DECAY_S,
            "pluck_transient_ms": PLUCK_TRANSIENT_MS,
            "harmonic_rolloff_power": HARMONIC_ROLLOFF_POWER,
            "body_reference_gain": BODY_REFERENCE_GAIN,
            "string_pluck_gain": tune.string_pluck_gain,
            "string_pitch_layer_gain": tune.string_pitch_layer_gain,
            "body_modal_gain": tune.body_modal_gain,
            "body_to_string_target_ratio": tune.body_to_string_target_ratio,
            "body_modal_bandwidth_widening": tune.body_modal_bandwidth_widening,
            "body_modal_richness_gain": tune.body_modal_richness_gain,
            "body_low_mode_weight": tune.body_low_mode_weight,
            "synthesis_preset": tune.name,
            "high_note_pluck_gain_floor": tune.high_note_pluck_gain_floor,
            "high_note_pluck_transient_reduction": tune.high_note_pluck_transient_reduction,
            "high_note_hf_rolloff_k_power": tune.high_note_hf_rolloff_k_power,
            "rad_k": FIXED_RAD_K,
            "q_clamp": [Q_MIN, Q_MAX],
            "target_rms_dbfs": TARGET_RMS_DBFS,
            "peak_ceiling_dbfs": FINAL_PEAK_CEILING_DBFS,
            "fundamental_anchor_gain": FUNDAMENTAL_ANCHOR_GAIN,
            "note_decay_tau_s": round(note_decay_tau_s, 4),
            "body_decay_tau_s": round(body_decay_tau_s_val, 4),
            "harmonic_decay_factor": HARMONIC_DECAY_FACTOR,
            "loudness_rms_window_s": [
                LOUDNESS_RMS_WINDOW_START_S,
                LOUDNESS_RMS_WINDOW_END_S,
            ],
        },
        "output_wav": str(output_wav),
        "samples_finite": True,
        "raw_body_rms_before_normalization": round(body_rms_before_calibration, 8),
        "final_rms_dbfs": loudness_info["output_rms_dbfs"],
        "string_gain_applied": round(effective_string_pluck_gain, 6),
        "broad_signature_band_gains": broaden_info.get("broad_signature_band_gains") or {},
        "per_mode_damping": broaden_info.get("per_mode_damping") or [],
        "damping_q_summary": dict(broaden_info.get("damping_q_summary") or {}),
        "sample_material_damping_fingerprint": round(
            float(
                np.mean(
                    [float(r.get("mode_material_damping") or 1.0) for r in (broaden_info.get("per_mode_damping") or [])]
                )
            )
            if broaden_info.get("per_mode_damping")
            else 0.0,
            6,
        ),
        "sample_mode_q_fingerprint": round(
            float(
                np.mean([float(r.get("mode_q") or r.get("final_mode_q") or 22.0) for r in (broaden_info.get("per_mode_damping") or [])])
            )
            if broaden_info.get("per_mode_damping")
            else 0.0,
            6,
        ),
        "modal_source": modal_source or "unknown",
        "per_mode_q_used_in_frequency_response": broaden_info.get("per_mode_q_used_in_frequency_response", True),
        "per_mode_tau_used_in_time_decay": broaden_info.get("per_mode_tau_used_in_time_decay", True),
        "far_mode_weights_sample_specific": broaden_info.get("far_mode_weights_sample_specific", True),
        "far_mode_sample_specificity_score": broaden_info.get("far_mode_sample_specificity_score"),
        "radiation_color_v1_active": broaden_info.get("radiation_color_v1_active", False),
        "modal_radiation_color_v2_active": broaden_info.get("modal_radiation_color_v2_active", False),
        "body_signature_v3_active": broaden_info.get("body_signature_v3_active", False),
        "body_signature_v3_ablation": broaden_info.get("body_signature_v3_ablation", {}),
        "body_signature_envelope_meta": body_signature_envelope_meta,
        "low_body_color_strength": broaden_info.get("low_body_color_strength"),
        "bridge_proxy_missing_count": broaden_info.get("bridge_proxy_missing_count"),
        "radiation_proxy_missing_count": broaden_info.get("radiation_proxy_missing_count"),
        "mic_proxy_missing_count": broaden_info.get("mic_proxy_missing_count"),
        "raw_body_rms_before_any_gain": round(body_rms_before_calibration, 8),
        "raw_body_rms_after_modal_weighting": round(body_rms_before_calibration, 8),
        "body_rms_after_body_gain": round(body_rms_before, 8),
        "body_to_string_ratio_before_normalization": round(body_to_string_rms_ratio_before_loudness, 6),
        "body_to_string_ratio_after_normalization": round(
            body_rms_before / max(string_rms_before_mix, 1e-12), 6
        ),
        "applied_body_gain": round(body_gain_applied, 6),
        "applied_loudness_gain": round(float(loudness_info.get("rms_gain_applied") or 1.0), 6),
    }
    if diag:
        metadata["diagnostic_mode"] = diag.name
        metadata["diagnostic_config"] = diag.to_metadata_dict()
        metadata["raw_body_variation_preserve"] = variation_preserve
        metadata["body_gain_normalization_strength"] = body_gain_norm_strength
        metadata["final_loudness_normalization_strength"] = float(
            diag.final_loudness_normalization_strength
        )
    else:
        metadata["diagnostic_mode"] = "baseline_current"
        metadata["body_gain_normalization_strength"] = 1.0
        metadata["final_loudness_normalization_strength"] = 1.0
    metadata["string_direct_scale_by_f0"] = round(string_direct_scale, 6)
    metadata["pitch_layer_scale_by_f0"] = round(pitch_layer_scale, 6)
    metadata["body_color_gain_by_f0"] = round(body_color_boost, 6)
    metadata["high_note_string_direct_scale_applied"] = round(string_direct_scale, 6)
    metadata["high_note_body_color_boost_applied"] = round(body_color_boost, 6)
    metadata["fundamental_anchor_scale_applied"] = round(fundamental_anchor_scale, 6)
    metadata["low_note_fundamental_harmonic_boost"] = round(_low_note_fundamental_harmonic_boost(), 6)
    metadata["near_modal_energy_fraction"] = broaden_info.get("near_modal_energy_fraction")
    metadata["mid_modal_energy_fraction"] = broaden_info.get("mid_modal_energy_fraction")
    metadata["broad_body_energy_fraction"] = broaden_info.get("broad_body_energy_fraction")
    metadata["far_modal_energy_fraction"] = broaden_info.get("far_modal_energy_fraction")
    metadata["per_mode_damping_count"] = len(broaden_info.get("per_mode_damping") or [])
    metadata.update(
        compute_note_reward_score(
            frequency_hz=float(frequency_hz),
            body_rms_before_mix=body_rms_before,
            string_rms_before_mix=string_rms_before_mix,
            body_to_string_ratio_before_loudness=body_to_string_rms_ratio_before_loudness,
            top_contributing_modes=top_modes,
            late_to_early_rms_db=float(loudness_info.get("late_to_early_rms_db") or 0.0),
            output_decay_slope_db_per_s=float(loudness_info.get("output_decay_slope_db_per_s") or 0.0),
            broad_body_energy_fraction=float(broaden_info.get("broad_body_energy_fraction") or 0.0),
            near_modal_energy_fraction=float(broaden_info.get("near_modal_energy_fraction") or 0.0),
            final_rms_dbfs=float(loudness_info.get("output_rms_dbfs") or 0.0),
        )
    )
    if output_metadata_json is not None:
        output_metadata_json.parent.mkdir(parents=True, exist_ok=True)
        output_metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        metadata["output_metadata_json"] = str(output_metadata_json)
    return metadata


def synthetic_classic_body_modes(n_modes: int = 55) -> List[Dict[str, Any]]:
    lo, hi = FULL_MODAL_BAND_HZ
    modes: List[Dict[str, Any]] = []
    for i in range(n_modes):
        t = i / max(n_modes - 1, 1)
        f = lo + t * (hi - lo)
        modes.append(
            {
                "frequency_hz": round(f, 2),
                "mode_index": i,
                "bridge_excitation_abs": 0.012 + 0.008 * ((i % 3) + 1) / 3.0,
                "mic_output_proxy": 0.006 + 0.004 * ((i + 1) % 4) / 4.0,
                "radiation_proxy": 0.005 + 0.003 * (i % 2),
                "top_share": 0.38 + 0.04 * (i % 2),
                "back_share": 0.34,
                "air_share": 0.22 + 0.06 * (i % 3) / 3.0,
            }
        )
    return modes


def load_modal_data_from_path(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    if not isinstance(doc, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return doc


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Body-response note smoke test")
    parser.add_argument("--modal-json", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("audio/stage1_loudness"))
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_S)
    args = parser.parse_args()

    if args.modal_json and args.modal_json.is_file():
        modal_data = load_modal_data_from_path(args.modal_json)
    else:
        modal_data = {"predicted_modes": synthetic_classic_body_modes(), "analysis": "synthetic_fixture"}

    cases = (("E2", 82.41), ("A2", 110.0), ("A4", 440.0), ("E5", 659.25))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = args.out_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    for name, hz in cases:
        wav_path = args.out_dir / f"{name}_body.wav"
        meta_path = meta_dir / f"{name}_metadata.json"
        meta = synthesize_note_with_body_response(
            frequency_hz=hz,
            note_name=name,
            duration_s=args.duration,
            sample_rate=DEFAULT_SAMPLE_RATE,
            modal_data=modal_data,
            output_wav=wav_path,
            output_metadata_json=meta_path,
        )
        print(
            f"{name}: rms={meta['output_rms_dbfs']:.1f} dBFS peak={meta['output_peak_dbfs']:.1f} dBFS "
            f"slope={meta['output_decay_slope_db_per_s']:.1f} dB/s "
            f"late/early={meta['late_to_early_rms_db']:.1f} dB "
            f"limiter={meta['limiter_used']} hf={meta['high_frequency_fallback_used']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
