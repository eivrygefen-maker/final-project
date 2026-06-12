#!/usr/bin/env python3
"""
Stage 4.8 — layer-separated timbre decomposition for listening diagnostics.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from body_response_synth import (
    BODY_DECAY_LOW_NOTE_BLEND,
    BODY_REFERENCE_GAIN,
    DEFAULT_VELOCITY,
    FIXED_PLUCK_POSITION,
    HIGH_NOTE_DECAY_THRESHOLD_HZ,
    LOW_NOTE_FUNDAMENTAL_MAX_HZ,
    ModalInput,
    _direct_attack_tap,
    _fundamental_pitch_anchor,
    _rms,
    _string_acceleration,
    _string_pitch_layer,
    active_diagnostic,
    active_tuning,
    apply_anti_click_taper,
    apply_exponential_decay_envelope,
    apply_loudness_finalize,
    body_color_gain_by_f0,
    body_decay_tau_s,
    fundamental_anchor_scale_by_body_strength,
    harmonic_series,
    high_note_pluck_soften_t,
    high_note_pluck_softening_gain,
    high_note_string_hf_rolloff_factor,
    note_base_decay_tau_s,
    parse_modal_modes,
    modes_in_validated_band,
    pitch_layer_scale_by_f0,
    string_direct_scale_by_f0,
    summarize_body_radiation,
    synthesize_body_via_transfer_function,
    synthesize_plucked_string,
    use_diagnostic_mode,
)
from bridge_mobility_proxy import compute_body_mass_proxies
from diagnostic_synthesis import (
    _spectral_features,
    compute_note_reward_score,
    flatten_geometry_parameters,
)
from sample_parameters import normalize_sample_parameters

LAYER_NAMES: Tuple[str, ...] = (
    "string_only",
    "body_only_raw_pre_norm",
    "body_only_post_body_gain",
    "body_only_final_norm",
    "near_modes_only",
    "far_background_modes_only",
    "radiation_only_weighted_body",
    "full_mix_baseline",
    "full_mix_radiation_v1",
    "full_mix_candidate_balance",
)


def _synthesize_body_layer(
    acc: np.ndarray,
    sample_rate: int,
    band_modes: Sequence[Mapping[str, Any]],
    *,
    defaults_used: List[str],
    flags: Dict[str, bool],
    note_hz: float,
    harmonics_hz: Sequence[float],
    layer_filter: Optional[str],
    use_bridge_mobility_proxy: bool,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if not band_modes:
        return np.zeros_like(acc), {}
    body, _, _, broaden_info = synthesize_body_via_transfer_function(
        acc,
        sample_rate,
        band_modes,
        defaults_used=defaults_used,
        flags=flags,
        note_hz=note_hz,
        harmonics_hz=harmonics_hz,
        layer_filter=layer_filter,
        use_bridge_mobility_proxy=use_bridge_mobility_proxy,
    )
    radiation_summary = summarize_body_radiation(band_modes)
    body_decay_tau_s_val = body_decay_tau_s(note_hz, radiation_summary)
    body_floor = BODY_DECAY_LOW_NOTE_BLEND if float(note_hz) <= LOW_NOTE_FUNDAMENTAL_MAX_HZ else 0.0
    body = apply_exponential_decay_envelope(
        body,
        sample_rate,
        body_decay_tau_s_val,
        floor_mix=body_floor,
    )
    return body, broaden_info


def _string_path_components(
    string_excitation: np.ndarray,
    sample_rate: int,
    frequency_hz: float,
    duration_s: float,
    velocity: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    tune = active_tuning()
    diag = active_diagnostic()
    pluck_soften = high_note_pluck_softening_gain(frequency_hz)
    soften_t = high_note_pluck_soften_t(frequency_hz)
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
    return string_path, {
        "string_pluck_gain": effective_pluck_gain,
        "string_pitch_layer_gain": effective_pitch_gain,
        "string_rms": _rms(string_path),
    }


def _body_gain_calibration(
    body_rms_before_calibration: float,
    string_rms_before_mix: float,
    frequency_hz: float,
) -> Tuple[float, float]:
    tune = active_tuning()
    diag = active_diagnostic()
    soften_t = high_note_pluck_soften_t(frequency_hz)
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
    elif body_rms_before_calibration > 0:
        body_gain_applied = BODY_REFERENCE_GAIN
    else:
        body_gain_applied = 0.0
    return body_gain_applied, body_target_ratio


def _apply_body_color_and_richness(body_raw: np.ndarray, frequency_hz: float, body_gain_applied: float) -> np.ndarray:
    tune = active_tuning()
    diag = active_diagnostic()
    soften_t = high_note_pluck_soften_t(frequency_hz)
    body_color_boost = body_color_gain_by_f0(float(frequency_hz))
    if diag and diag.high_note_body_color_boost > 1.0:
        body_color_boost *= 1.0 + soften_t * (float(diag.high_note_body_color_boost) - 1.0)
    body_signal = body_raw * body_gain_applied * tune.body_modal_gain * body_color_boost
    return body_signal * tune.body_modal_richness_gain


def _finalize_layer(
    samples: np.ndarray,
    sample_rate: int,
    duration_s: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    tapered, taper_info = apply_anti_click_taper(samples, sample_rate, duration_s=duration_s)
    diag = active_diagnostic()
    preserve = 0.0
    loudness_strength = 1.0
    if diag is not None:
        preserve = diag.effective_loudness_preserve()
        loudness_strength = float(diag.final_loudness_normalization_strength)
    y, loudness_info = apply_loudness_finalize(
        tapered,
        sample_rate,
        raw_body_variation_preserve=preserve,
        loudness_normalization_strength=loudness_strength,
    )
    loudness_info.update(taper_info)
    return y, loudness_info


def _mix_with_anchor(
    mixed: np.ndarray,
    *,
    frequency_hz: float,
    duration_s: float,
    sample_rate: int,
    velocity: float,
    body_rms_before: float,
    string_rms_before_mix: float,
) -> np.ndarray:
    diag = active_diagnostic()
    fundamental_anchor_used = float(frequency_hz) <= LOW_NOTE_FUNDAMENTAL_MAX_HZ
    anchor_base = float(diag.fundamental_anchor_scale) if diag else 1.0
    fundamental_anchor_scale = fundamental_anchor_scale_by_body_strength(
        float(frequency_hz),
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
        mixed = mixed + fundamental_anchor_scale * anchor
    return mixed


def compute_note_layers(
    frequency_hz: float,
    note_name: str,
    duration_s: float,
    sample_rate: int,
    modal_data: ModalInput,
    *,
    velocity: float = DEFAULT_VELOCITY,
    sample_parameters: Optional[Mapping[str, Any]] = None,
    use_bridge_mobility_proxy: bool = True,
    modal_source: Optional[str] = None,
) -> Dict[str, Any]:
    """Return dict layer_name -> float64 audio plus metadata."""
    return _compute_note_layers_core(
        frequency_hz=frequency_hz,
        note_name=note_name,
        duration_s=duration_s,
        sample_rate=sample_rate,
        modal_data=modal_data,
        velocity=velocity,
        use_bridge_mobility_proxy=use_bridge_mobility_proxy,
        modal_source=modal_source,
        sample_parameters=sample_parameters,
    )


def _compute_full_mix_for_mode(
    *,
    acc: np.ndarray,
    band_modes: Sequence[Mapping[str, Any]],
    harmonics_hz: Sequence[float],
    string_excitation: np.ndarray,
    sample_rate: int,
    frequency_hz: float,
    duration_s: float,
    velocity: float,
    diagnostic_mode: str,
    sample_parameters: Optional[Mapping[str, Any]],
    use_bridge_mobility_proxy: bool,
    defaults_used: List[str],
    flags: Dict[str, bool],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    with use_diagnostic_mode(diagnostic_mode, sample_parameters=sample_parameters):
        body_raw, _ = _synthesize_body_layer(
            acc,
            sample_rate,
            band_modes,
            defaults_used=list(defaults_used),
            flags=dict(flags),
            note_hz=frequency_hz,
            harmonics_hz=harmonics_hz,
            layer_filter=None,
            use_bridge_mobility_proxy=use_bridge_mobility_proxy,
        )
        string_path, string_info = _string_path_components(
            string_excitation, sample_rate, frequency_hz, duration_s, velocity
        )
        body_gain_applied, _ = _body_gain_calibration(
            _rms(body_raw), string_info["string_rms"], frequency_hz
        )
        body_sig = _apply_body_color_and_richness(body_raw, frequency_hz, body_gain_applied)
        mixed = body_sig + string_path
        mixed = _mix_with_anchor(
            mixed,
            frequency_hz=frequency_hz,
            duration_s=duration_s,
            sample_rate=sample_rate,
            velocity=velocity,
            body_rms_before=_rms(body_sig),
            string_rms_before_mix=string_info["string_rms"],
        )
        return _finalize_layer(mixed, sample_rate, duration_s)


def _compute_note_layers_core(
    frequency_hz: float,
    note_name: str,
    duration_s: float,
    sample_rate: int,
    modal_data: ModalInput,
    *,
    velocity: float,
    use_bridge_mobility_proxy: bool,
    modal_source: Optional[str],
    sample_parameters: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    with use_diagnostic_mode("baseline_current", sample_parameters=sample_parameters):
        return _compute_note_layers_baseline(
            frequency_hz=frequency_hz,
            note_name=note_name,
            duration_s=duration_s,
            sample_rate=sample_rate,
            modal_data=modal_data,
            velocity=velocity,
            use_bridge_mobility_proxy=use_bridge_mobility_proxy,
            modal_source=modal_source,
            sample_parameters=sample_parameters,
        )


def _compute_note_layers_baseline(
    frequency_hz: float,
    note_name: str,
    duration_s: float,
    sample_rate: int,
    modal_data: ModalInput,
    *,
    velocity: float,
    use_bridge_mobility_proxy: bool,
    modal_source: Optional[str],
    sample_parameters: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    all_modes, parse_defaults = parse_modal_modes(modal_data)
    band_modes = modes_in_validated_band(all_modes)
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
    string_path, string_info = _string_path_components(
        string_excitation, sample_rate, frequency_hz, duration_s, velocity
    )
    string_only = string_path.copy()

    mobility_meta = compute_body_mass_proxies(sample_parameters) if use_bridge_mobility_proxy else {}

    def body_for(filter_name: Optional[str]) -> Tuple[np.ndarray, Dict[str, Any]]:
        return _synthesize_body_layer(
            acc,
            sample_rate,
            band_modes,
            defaults_used=defaults_used,
            flags=flags,
            note_hz=frequency_hz,
            harmonics_hz=harmonics_hz,
            layer_filter=filter_name,
            use_bridge_mobility_proxy=use_bridge_mobility_proxy,
        )

    body_raw, broaden_full = body_for(None)
    body_near, _ = body_for("near")
    body_far, _ = body_for("far")
    body_rad, broaden_rad = body_for("radiation")

    body_rms_raw = _rms(body_raw)
    body_gain_applied, body_target_ratio = _body_gain_calibration(
        body_rms_raw, string_info["string_rms"], frequency_hz
    )
    body_post_gain = _apply_body_color_and_richness(body_raw, frequency_hz, body_gain_applied)
    body_rms_post = _rms(body_post_gain)
    body_final_norm, body_norm_info = _finalize_layer(body_post_gain, sample_rate, duration_s)

    layers: Dict[str, np.ndarray] = {
        "string_only": string_only,
        "body_only_raw_pre_norm": body_raw,
        "body_only_post_body_gain": body_post_gain,
        "body_only_final_norm": body_final_norm,
        "near_modes_only": body_near,
        "far_background_modes_only": body_far,
        "radiation_only_weighted_body": body_rad,
    }

    full_mix_modes = {
        "full_mix_baseline": "baseline_current",
        "full_mix_radiation_v1": "modal_radiation_color_v1",
        "full_mix_candidate_balance": "body_audibility_balance_probe_v1",
    }
    full_mix_meta: Dict[str, Any] = {}
    for layer_key, mode_name in full_mix_modes.items():
        mix_audio, norm_info = _compute_full_mix_for_mode(
            acc=acc,
            band_modes=band_modes,
            harmonics_hz=harmonics_hz,
            string_excitation=string_excitation,
            sample_rate=sample_rate,
            frequency_hz=frequency_hz,
            duration_s=duration_s,
            velocity=velocity,
            diagnostic_mode=mode_name,
            sample_parameters=sample_parameters,
            use_bridge_mobility_proxy=use_bridge_mobility_proxy,
            defaults_used=defaults_used,
            flags=flags,
        )
        layers[layer_key] = mix_audio
        full_mix_meta[layer_key] = norm_info

    reward = compute_note_reward_score(
        frequency_hz=frequency_hz,
        body_rms_before_mix=body_rms_post,
        string_rms_before_mix=string_info["string_rms"],
        body_to_string_ratio_before_loudness=body_rms_post / max(string_info["string_rms"], 1e-12),
        top_contributing_modes=broaden_full.get("top_contributing_modes") or [],
        late_to_early_rms_db=float(body_norm_info.get("late_to_early_rms_db") or -10.0),
        output_decay_slope_db_per_s=float(body_norm_info.get("output_decay_slope_db_per_s") or -8.0),
        broad_body_energy_fraction=float(broaden_full.get("broad_body_energy_fraction") or 0.0),
        near_modal_energy_fraction=float(broaden_full.get("near_modal_energy_fraction") or 0.0),
        final_rms_dbfs=float(body_norm_info.get("output_rms_dbfs") or -20.0),
    )

    params = normalize_sample_parameters(sample_parameters)
    metadata: Dict[str, Any] = {
        "note_name": note_name,
        "frequency_hz": float(frequency_hz),
        "sample_rate": int(sample_rate),
        "duration_s": float(duration_s),
        "diagnostic_mode": "baseline_current",
        "modal_source": modal_source,
        "evaluated_modal_count": len(band_modes),
        "raw_body_rms_before_normalization": round(body_rms_raw, 8),
        "body_rms_post_body_gain": round(body_rms_post, 8),
        "body_gain_applied": round(body_gain_applied, 6),
        "body_to_string_target_ratio": body_target_ratio,
        "body_to_string_rms_ratio_before_loudness": round(body_rms_post / max(string_info["string_rms"], 1e-12), 6),
        "note_reward_score": reward["note_reward_score"],
        "note_reward_detail": reward,
        "bridge_mobility_proxy": mobility_meta,
        "broaden_info": broaden_full,
        "radiation_layer_info": broaden_rad,
        "normalization_audit": {
            "body_only_final_norm": body_norm_info,
            **{k: v for k, v in full_mix_meta.items()},
        },
        "data_attribution": _data_attribution_fields(params, band_modes, mobility_meta),
        "peak_abs_max": {k: float(np.max(np.abs(v))) if v.size else 0.0 for k, v in layers.items()},
    }
    metadata.update(flatten_geometry_parameters(params))
    return {"layers": layers, "metadata": metadata}


def _data_attribution_fields(
    params: Mapping[str, Any],
    band_modes: Sequence[Mapping[str, Any]],
    mobility_meta: Mapping[str, Any],
) -> Dict[str, Any]:
    bridge_vals = [
        float(m.get("bridge_excitation") or m.get("bridge_weight") or 0.0) for m in band_modes
    ]
    rad_vals = [float(m.get("radiation_proxy") or m.get("radiation_weight") or 0.0) for m in band_modes]
    mic_vals = [float(m.get("mic_proxy") or m.get("mic_weight") or 0.0) for m in band_modes]
    freqs = [float(m.get("frequency_hz") or 0.0) for m in band_modes]
    return {
        "mode_frequency_min_hz": min(freqs) if freqs else None,
        "mode_frequency_max_hz": max(freqs) if freqs else None,
        "bridge_excitation_mean": round(sum(bridge_vals) / max(len(bridge_vals), 1), 6),
        "radiation_proxy_mean": round(sum(rad_vals) / max(len(rad_vals), 1), 6),
        "mic_proxy_mean": round(sum(mic_vals) / max(len(mic_vals), 1), 6),
        "top_effective_mass_proxy": mobility_meta.get("top_effective_mass_proxy"),
        "back_effective_mass_proxy": mobility_meta.get("back_effective_mass_proxy"),
        "body_air_volume_proxy": mobility_meta.get("body_air_volume_proxy"),
        "bridge_mobility_proxy": mobility_meta.get("bridge_mobility_proxy"),
    }


def layer_segment_row(
    *,
    sample_id: str,
    note_name: str,
    frequency_hz: float,
    layer_name: str,
    audio: np.ndarray,
    sample_rate: int,
    metadata: Mapping[str, Any],
    sample_parameters: Mapping[str, Any],
) -> Dict[str, Any]:
    spec = _spectral_features(audio, sample_rate)
    row = {
        "sample_id": sample_id,
        "note": note_name,
        "frequency_hz": frequency_hz,
        "layer": layer_name,
        "raw_body_rms_before_normalization": metadata.get("raw_body_rms_before_normalization"),
        "note_reward_score": metadata.get("note_reward_score"),
        "body_to_string_ratio": metadata.get("body_to_string_rms_ratio_before_loudness"),
        "spectral_centroid_hz": round(spec["centroid_hz"], 4),
        "spectral_low_energy": round(spec["low_energy"], 6),
        "spectral_mid_energy": round(spec["mid_energy"], 6),
        "spectral_high_energy": round(spec["high_energy"], 6),
        "output_rms_dbfs": round(20.0 * math.log10(max(_rms(audio), 1e-12)), 4),
        "data_attribution": metadata.get("data_attribution"),
        "bridge_mobility_proxy": metadata.get("bridge_mobility_proxy"),
        "parameters": normalize_sample_parameters(sample_parameters),
    }
    row.update(flatten_geometry_parameters(sample_parameters))
    return row
