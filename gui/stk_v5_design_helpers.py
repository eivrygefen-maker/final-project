#!/usr/bin/env python3
"""
STK V5 design helpers — audit metrics, component decomposition, V5 skeleton prototype.

Diagnostic-only. Does not modify website default or production synthesis path.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from body_response_synth import (
    DEFAULT_DURATION_S,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_VELOCITY,
    FIXED_PLUCK_POSITION,
    ModalInput,
    _direct_attack_tap,
    _rms,
    _string_acceleration,
    _string_pitch_layer,
    apply_anti_click_taper,
    apply_loudness_finalize,
    harmonic_series,
    modes_in_validated_band,
    parse_modal_modes,
    synthesize_body_via_transfer_function,
    synthesize_plucked_string,
    synthesize_note_with_body_response,
    write_wav_int16,
)
from sample_parameters import normalize_sample_parameters

V5_SKELETON_VERSION = "stk_v5_skeleton_v0"
V5_ALPHA_VERSION = "stk_v5_alpha_v0"

STK_V5_ALPHA_BODY_DOMINANT = "stk_v5_alpha_body_dominant"
STK_V5_ALPHA_GUI_LABEL = "STK V5 alpha — body dominant"

# (direct_string_gain, body_gain) — RMS-matched perceptual weights
V5_ALPHA_VARIANTS: Dict[str, Tuple[float, float]] = {
    STK_V5_ALPHA_BODY_DOMINANT: (0.20, 0.80),
    "v5_alpha_s10_b90": (0.10, 0.90),
    "v5_alpha_s20_b80": (0.20, 0.80),
    "v5_alpha_s35_b65": (0.35, 0.65),
}

V5_ALPHA_LOUDNESS_STRENGTH = 0.18
V5_ALPHA_VARIATION_PRESERVE = 0.68
V5_ALPHA_PEAK_CEILING_DBFS = -0.5


def list_v5_alpha_mode_names() -> List[str]:
    return list(V5_ALPHA_VARIANTS.keys())


def resolve_v5_alpha_variant(experiment_or_mode: str) -> Tuple[str, float, float]:
    """Return (canonical_name, direct_string_gain, body_gain)."""
    key = str(experiment_or_mode or STK_V5_ALPHA_BODY_DOMINANT).strip()
    if key not in V5_ALPHA_VARIANTS:
        raise ValueError(f"unknown V5 alpha variant: {key!r}")
    s_gain, b_gain = V5_ALPHA_VARIANTS[key]
    return key, float(s_gain), float(b_gain)


def is_v5_alpha_experiment(name: str) -> bool:
    n = str(name or "")
    return n in V5_ALPHA_VARIANTS or n.startswith("v5_alpha_")


def _peak_dbfs(audio: np.ndarray) -> Tuple[float, bool]:
    peak = float(np.max(np.abs(np.asarray(audio, dtype=np.float64))))
    peak_db = 20.0 * math.log10(max(peak, 1e-12))
    return round(peak_db, 4), peak < 1.0


def _band_energy(audio: np.ndarray, sample_rate: int, lo_hz: float, hi_hz: float) -> float:
    x = np.asarray(audio, dtype=np.float64)
    n = len(x)
    if n < 64:
        return 0.0
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sample_rate))
    mask = (freqs >= lo_hz) & (freqs < hi_hz)
    if not np.any(mask):
        return 0.0
    return float(np.sum(spec[mask] ** 2))


def _spectral_centroid(audio: np.ndarray, sample_rate: int) -> float:
    x = np.asarray(audio, dtype=np.float64)
    n = len(x)
    if n < 64:
        return 0.0
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sample_rate))
    denom = float(np.sum(spec)) + 1e-12
    return float(np.sum(freqs * spec) / denom)


def _onset_rms(audio: np.ndarray, sample_rate: int, ms: float = 50.0) -> float:
    n = max(1, int(sample_rate * ms / 1000.0))
    return _rms(np.asarray(audio, dtype=np.float64)[:n])


def compute_realism_metrics(
    audio: np.ndarray,
    *,
    sample_rate: int,
    frequency_hz: float,
    string_rms: Optional[float] = None,
    body_rms: Optional[float] = None,
) -> Dict[str, Any]:
    """Gates-oriented metrics beyond spectral difference."""
    x = np.asarray(audio, dtype=np.float64)
    lo_e = _band_energy(x, sample_rate, 0.0, 180.0)
    mid_e = _band_energy(x, sample_rate, 180.0, 800.0)
    hi_e = _band_energy(x, sample_rate, 800.0, 8000.0)
    very_hi_e = _band_energy(x, sample_rate, 3000.0, 12000.0)
    spec_total = lo_e + mid_e + hi_e + 1e-18

    metallicity = very_hi_e / max(mid_e + lo_e, 1e-12)
    centroid = _spectral_centroid(x, sample_rate)
    f0 = max(40.0, float(frequency_hz))

    str_rms = float(string_rms if string_rms is not None else 0.0)
    bod_rms = float(body_rms if body_rms is not None else 0.0)
    if str_rms <= 0 and bod_rms <= 0:
        str_rms = _rms(x) * 0.5
        bod_rms = _rms(x) * 0.5

    body_to_string = bod_rms / max(str_rms, 1e-12)
    string_dominance = str_rms / max(str_rms + bod_rms, 1e-12)
    body_audibility = bod_rms / max(_rms(x), 1e-12)

    attack_rms = _onset_rms(x, sample_rate, 50.0)
    sustain_rms = _rms(x[int(sample_rate * 0.35) : int(sample_rate * 1.2)])
    attack_to_sustain = attack_rms / max(sustain_rms, 1e-12)

    # Heuristic guitar realism sanity: penalize metallic + string-dominated + bright centroid
    realism_penalty = (
        0.35 * min(1.0, metallicity / 0.45)
        + 0.35 * min(1.0, string_dominance / 0.72)
        + 0.30 * min(1.0, max(0.0, (centroid - 2.8 * f0) / max(f0, 1.0)))
    )
    guitar_realism_score = round(max(0.0, 1.0 - realism_penalty), 4)

    return {
        "metallicity_index": round(metallicity, 6),
        "spectral_centroid_hz": round(centroid, 2),
        "body_audibility_index": round(body_audibility, 6),
        "string_dominance_ratio": round(string_dominance, 6),
        "body_to_string_energy_ratio": round(body_to_string, 6),
        "low_band_fraction": round(lo_e / spec_total, 6),
        "mid_band_fraction": round(mid_e / spec_total, 6),
        "high_band_fraction": round(hi_e / spec_total, 6),
        "very_high_band_fraction": round(very_hi_e / spec_total, 6),
        "attack_to_sustain_ratio": round(attack_to_sustain, 6),
        "attack_rms_50ms": round(attack_rms, 8),
        "guitar_realism_sanity_score": guitar_realism_score,
        "radiation_contribution_proxy": round(hi_e / max(lo_e + mid_e, 1e-12), 6),
    }


def enrich_metrics_with_levels(metrics: Dict[str, Any], audio: np.ndarray) -> Dict[str, Any]:
    peak_db, clipping_ok = _peak_dbfs(audio)
    out = dict(metrics)
    out["peak_dbfs"] = peak_db
    out["clipping_avoided"] = clipping_ok
    return out


def rms_matched_body_dominant_mix(
    body: np.ndarray,
    dry_string: np.ndarray,
    *,
    body_weight: float,
    string_weight: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Mix body and direct string with unit-RMS branches so weights ≈ perceptual contribution.

    output = body_weight * (body/rms_body) + string_weight * (string/rms_string)
    """
    body = np.asarray(body, dtype=np.float64)
    dry_string = np.asarray(dry_string, dtype=np.float64)
    b_rms = _rms(body)
    s_rms = _rms(dry_string)
    b_unit = body / max(b_rms, 1e-12)
    s_unit = dry_string / max(s_rms, 1e-12)
    body_branch = float(body_weight) * b_unit
    string_branch = float(string_weight) * s_unit
    mixed = body_branch + string_branch
    b_in_mix = _rms(body_branch)
    s_in_mix = _rms(string_branch)
    return mixed, {
        "body_weight": round(float(body_weight), 4),
        "string_weight": round(float(string_weight), 4),
        "body_rms_raw": round(b_rms, 8),
        "string_rms_raw": round(s_rms, 8),
        "body_rms_in_mix": round(b_in_mix, 8),
        "string_rms_in_mix": round(s_in_mix, 8),
        "body_to_string_in_mix_ratio": round(b_in_mix / max(s_in_mix, 1e-12), 6),
        "string_dominance_in_mix": round(s_in_mix / max(b_in_mix + s_in_mix, 1e-12), 6),
    }


def render_v5_alpha_body_radiation_path(
    *,
    frequency_hz: float,
    duration_s: float,
    sample_rate: int,
    modal_data: ModalInput,
    sample_parameters: Optional[Mapping[str, Any]] = None,
    repo_root: Optional[Any] = None,
    sample_id: str = "sample_000",
    velocity: float = DEFAULT_VELOCITY,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    V5 alpha body path: pluck → bridge accel → admittance bank → radiation scale.
    Returns (body_radiated, dry_string_path, meta).
    """
    from body_response_first_v4_2 import build_body_transfer_function_v4_2

    f0 = float(frequency_hz)
    layers = decompose_baseline_layers(
        frequency_hz=f0,
        duration_s=duration_s,
        sample_rate=sample_rate,
        modal_data=modal_data,
        velocity=velocity,
    )
    dry_string = layers["string_path"]
    string_excitation = layers["string_excitation"]
    bridge_force = _string_acceleration(string_excitation)
    n = len(string_excitation)

    all_modes, _ = parse_modal_modes(modal_data)
    band_modes = modes_in_validated_band(all_modes)
    params = normalize_sample_parameters(sample_parameters)

    if band_modes:
        H_adm, mode_rows, h_summary = build_body_transfer_function_v4_2(
            sample_rate=sample_rate,
            n_samples=n,
            band_modes=band_modes,
            frequency_hz=f0,
            parameters=params,
            repo_root=repo_root,
            sample_id=sample_id,
        )
        bridge_spec = np.fft.rfft(bridge_force)
        body_audio = np.real(np.fft.irfft(bridge_spec * H_adm, n=n))
    else:
        mode_rows = []
        h_summary = {"mode_count": 0}
        body_audio = layers["body_raw"].copy()

    rad_pool = [float(m.get("radiation_proxy") or 0.0) for m in band_modes]
    rad_mean = float(np.mean(rad_pool)) if rad_pool else 0.0
    rad_scale = 0.85 + 0.35 * min(1.0, rad_mean)
    body_radiated = body_audio * rad_scale

    meta = {
        "body_path": "bridge_accel→H_admittance_v4_2→radiation_scale",
        "mode_count": len(mode_rows),
        "H_summary": h_summary,
        "radiation_scale": round(rad_scale, 6),
        "body_rms_radiated": round(_rms(body_radiated), 8),
        "string_rms_dry": round(_rms(dry_string), 8),
    }
    return body_radiated, dry_string, meta


def synthesize_v5_alpha_body_dominant(
    *,
    frequency_hz: float,
    duration_s: float,
    sample_rate: int,
    modal_data: ModalInput,
    variant: str = STK_V5_ALPHA_BODY_DOMINANT,
    sample_parameters: Optional[Mapping[str, Any]] = None,
    repo_root: Optional[Any] = None,
    sample_id: str = "sample_000",
    velocity: float = DEFAULT_VELOCITY,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """STK V5.0-alpha: body-dominant RMS-matched mix (diagnostic only)."""
    variant_name, string_gain, body_gain = resolve_v5_alpha_variant(variant)
    body_radiated, dry_string, path_meta = render_v5_alpha_body_radiation_path(
        frequency_hz=frequency_hz,
        duration_s=duration_s,
        sample_rate=sample_rate,
        modal_data=modal_data,
        sample_parameters=sample_parameters,
        repo_root=repo_root,
        sample_id=sample_id,
        velocity=velocity,
    )
    mixed, mix_meta = rms_matched_body_dominant_mix(
        body_radiated,
        dry_string,
        body_weight=body_gain,
        string_weight=string_gain,
    )
    tapered, taper_info = apply_anti_click_taper(mixed, sample_rate, duration_s=duration_s)
    final, loudness_info = apply_loudness_finalize(
        tapered,
        sample_rate,
        loudness_normalization_strength=V5_ALPHA_LOUDNESS_STRENGTH,
        raw_body_variation_preserve=V5_ALPHA_VARIATION_PRESERVE,
    )
    peak_db, clipping_ok = _peak_dbfs(final)
    if peak_db > V5_ALPHA_PEAK_CEILING_DBFS:
        scale = 10.0 ** ((V5_ALPHA_PEAK_CEILING_DBFS - peak_db) / 20.0)
        final = final * scale
        peak_db, clipping_ok = _peak_dbfs(final)
        loudness_info = {**loudness_info, "peak_limit_scale": round(scale, 6)}

    meta = {
        "diagnostic_mode": variant_name,
        "v5_alpha_version": V5_ALPHA_VERSION,
        "user_label": STK_V5_ALPHA_GUI_LABEL,
        "variant": variant_name,
        "direct_string_gain": string_gain,
        "body_gain": body_gain,
        "mix_formula": "body_gain*body/rms(body) + direct_string_gain*string/rms(string)",
        "path_meta": path_meta,
        "mix_metrics": mix_meta,
        "string_rms": mix_meta["string_rms_in_mix"],
        "body_rms": mix_meta["body_rms_in_mix"],
        "taper_info": taper_info,
        "loudness_info": loudness_info,
        "peak_dbfs": peak_db,
        "clipping_avoided": clipping_ok,
        "output_rms_dbfs": loudness_info.get("output_rms_dbfs"),
    }
    return final, meta


def decompose_baseline_layers(
    *,
    frequency_hz: float,
    duration_s: float,
    sample_rate: int,
    modal_data: ModalInput,
    velocity: float = DEFAULT_VELOCITY,
) -> Dict[str, Any]:
    """
    Decompose baseline_current-style path into string vs body layers (no identity overlay).
    """
    from diagnostic_synthesis import use_diagnostic_mode

    with use_diagnostic_mode("baseline_current"):
        all_modes, _ = parse_modal_modes(modal_data)
        band_modes = modes_in_validated_band(all_modes)
        string_excitation = synthesize_plucked_string(
            frequency_hz,
            duration_s,
            sample_rate,
            pluck_position=FIXED_PLUCK_POSITION,
            velocity=velocity,
        )
        acc = _string_acceleration(string_excitation)
        harmonics_hz, _ = harmonic_series(frequency_hz, sample_rate)

        defaults_used: List[str] = []
        flags: Dict[str, bool] = {}
        if band_modes:
            body_raw, _, _, _ = synthesize_body_via_transfer_function(
                acc,
                sample_rate,
                band_modes,
                defaults_used=defaults_used,
                flags=flags,
                note_hz=frequency_hz,
                harmonics_hz=harmonics_hz,
            )
        else:
            body_raw = np.zeros_like(string_excitation)

        string_pluck = 0.10 * _direct_attack_tap(string_excitation, sample_rate, float(frequency_hz))
        string_pitch = 0.055 * _string_pitch_layer(string_excitation, sample_rate, float(frequency_hz))
        string_path = string_pluck + string_pitch

        body_rms_raw = _rms(body_raw)
        string_rms = _rms(string_path)
        mixed = body_raw + string_path

    return {
        "string_excitation": string_excitation,
        "string_path": string_path,
        "body_raw": body_raw,
        "mixed_baseline": mixed,
        "string_rms": string_rms,
        "body_rms_raw": body_rms_raw,
        "body_to_string_raw_ratio": body_rms_raw / max(string_rms, 1e-12),
    }


def render_radiation_emphasized_body(
    *,
    frequency_hz: float,
    duration_s: float,
    sample_rate: int,
    modal_data: ModalInput,
    velocity: float = DEFAULT_VELOCITY,
) -> np.ndarray:
    """Body layer with radiation filter path only (diagnostic)."""
    from diagnostic_synthesis import use_diagnostic_mode

    with use_diagnostic_mode("baseline_current"):
        all_modes, _ = parse_modal_modes(modal_data)
        band_modes = modes_in_validated_band(all_modes)
        string_excitation = synthesize_plucked_string(
            frequency_hz,
            duration_s,
            sample_rate,
            pluck_position=FIXED_PLUCK_POSITION,
            velocity=velocity,
        )
        acc = _string_acceleration(string_excitation)
        harmonics_hz, _ = harmonic_series(frequency_hz, sample_rate)
        defaults_used: List[str] = []
        flags: Dict[str, bool] = {}
        if not band_modes:
            return np.zeros_like(string_excitation)
        body_rad, _, _, _ = synthesize_body_via_transfer_function(
            acc,
            sample_rate,
            band_modes,
            defaults_used=defaults_used,
            flags=flags,
            note_hz=frequency_hz,
            harmonics_hz=harmonics_hz,
            layer_filter="radiation",
        )
        return body_rad


def synthesize_v5_skeleton(
    *,
    frequency_hz: float,
    duration_s: float,
    sample_rate: int,
    modal_data: ModalInput,
    sample_parameters: Optional[Mapping[str, Any]] = None,
    repo_root: Optional[Any] = None,
    sample_id: str = "website",
    velocity: float = DEFAULT_VELOCITY,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    V5 skeleton prototype: excitation → bridge acceleration → admittance filter bank → radiation.

    Reuses V4.2 body-response-first mechanics as a starting point; not production-ready.
    """
    from body_response_first_v4_2 import build_body_transfer_function_v4_2

    f0 = float(frequency_hz)
    string_excitation = synthesize_plucked_string(
        f0,
        duration_s,
        sample_rate,
        pluck_position=FIXED_PLUCK_POSITION,
        velocity=velocity,
    )
    n = len(string_excitation)
    bridge_force = _string_acceleration(string_excitation)

    all_modes, _ = parse_modal_modes(modal_data)
    band_modes = modes_in_validated_band(all_modes)
    params = normalize_sample_parameters(sample_parameters)

    if band_modes:
        H_adm, mode_rows, h_summary = build_body_transfer_function_v4_2(
            sample_rate=sample_rate,
            n_samples=n,
            band_modes=band_modes,
            frequency_hz=f0,
            parameters=params,
            repo_root=repo_root,
            sample_id=sample_id,
        )
        bridge_spec = np.fft.rfft(bridge_force)
        body_spec = bridge_spec * H_adm
        body_audio = np.real(np.fft.irfft(body_spec, n=n))
    else:
        mode_rows = []
        h_summary = {"mode_count": 0}
        body_audio = np.zeros(n, dtype=np.float64)

    # Radiation emphasis on body output (separate stage, not post-EQ on string)
    rad_pool = [float(m.get("radiation_proxy") or 0.0) for m in band_modes]
    rad_mean = float(np.mean(rad_pool)) if rad_pool else 0.0
    rad_scale = 0.85 + 0.35 * min(1.0, rad_mean)
    radiated = body_audio * rad_scale

    # Minimal attack clarity tap — much smaller than current string path
    attack_tap = 0.04 * string_excitation * np.exp(
        -np.arange(n, dtype=np.float64) / max(sample_rate * 0.015, 1.0)
    )
    mixed = radiated + attack_tap

    tapered, _ = apply_anti_click_taper(mixed, sample_rate, duration_s=duration_s)
    final, loudness_info = apply_loudness_finalize(
        tapered,
        sample_rate,
        loudness_normalization_strength=0.22,
        raw_body_variation_preserve=0.62,
    )

    meta = {
        "v5_skeleton_version": V5_SKELETON_VERSION,
        "chain": "pluck→bridge_accel→H_admittance→radiation_scale→tiny_attack→normalize",
        "mode_count": len(mode_rows),
        "H_summary": h_summary,
        "radiation_scale": round(rad_scale, 6),
        "string_rms": round(_rms(string_excitation), 8),
        "body_rms": round(_rms(body_audio), 8),
        "radiated_rms": round(_rms(radiated), 8),
        "output_rms_dbfs": loudness_info.get("output_rms_dbfs"),
        "body_to_string_ratio": round(_rms(body_audio) / max(_rms(string_excitation), 1e-12), 6),
    }
    return final, meta


def synthesize_mode_to_wav(
    *,
    mode: str,
    frequency_hz: float,
    note_name: str,
    duration_s: float,
    sample_rate: int,
    modal_data: ModalInput,
    output_wav: Any,
    sample_parameters: Optional[Mapping[str, Any]] = None,
    repo_root: Optional[Any] = None,
    sample_id: str = "sample_000",
    experiment: str = "",
) -> Dict[str, Any]:
    """Route diagnostic experiment to appropriate renderer."""
    from pathlib import Path

    out = Path(output_wav)
    exp = str(experiment or mode)

    if exp == "string_only":
        layers = decompose_baseline_layers(
            frequency_hz=frequency_hz,
            duration_s=duration_s,
            sample_rate=sample_rate,
            modal_data=modal_data,
        )
        audio = layers["string_path"]
        meta = {
            "experiment": exp,
            "string_rms": layers["string_rms"],
            "body_rms": 0.0,
        }
        write_wav_int16(out, audio, sample_rate, duration_s=duration_s)
    elif exp == "body_only_modal_response":
        layers = decompose_baseline_layers(
            frequency_hz=frequency_hz,
            duration_s=duration_s,
            sample_rate=sample_rate,
            modal_data=modal_data,
        )
        audio = layers["body_raw"]
        meta = {
            "experiment": exp,
            "string_rms": 0.0,
            "body_rms": layers["body_rms_raw"],
        }
        write_wav_int16(out, audio, sample_rate, duration_s=duration_s)
    elif exp == "body_boost_test":
        layers = decompose_baseline_layers(
            frequency_hz=frequency_hz,
            duration_s=duration_s,
            sample_rate=sample_rate,
            modal_data=modal_data,
        )
        audio = 3.5 * layers["body_raw"] + 0.25 * layers["string_path"]
        meta = {"experiment": exp, "body_boost_factor": 3.5, "string_scale": 0.25}
        write_wav_int16(out, audio, sample_rate, duration_s=duration_s)
    elif exp == "string_attenuated_test":
        layers = decompose_baseline_layers(
            frequency_hz=frequency_hz,
            duration_s=duration_s,
            sample_rate=sample_rate,
            modal_data=modal_data,
        )
        audio = layers["body_raw"] + 0.15 * layers["string_path"]
        meta = {"experiment": exp, "string_scale": 0.15}
        write_wav_int16(out, audio, sample_rate, duration_s=duration_s)
    elif exp == "radiation_emphasized_test":
        audio = render_radiation_emphasized_body(
            frequency_hz=frequency_hz,
            duration_s=duration_s,
            sample_rate=sample_rate,
            modal_data=modal_data,
        )
        meta = {"experiment": exp}
        write_wav_int16(out, audio, sample_rate, duration_s=duration_s)
    elif exp == "proposed_v5_skeleton":
        audio, sk_meta = synthesize_v5_skeleton(
            frequency_hz=frequency_hz,
            duration_s=duration_s,
            sample_rate=sample_rate,
            modal_data=modal_data,
            sample_parameters=sample_parameters,
            repo_root=repo_root,
            sample_id=sample_id,
        )
        meta = {"experiment": exp, **sk_meta}
        write_wav_int16(out, audio, sample_rate, duration_s=duration_s)
    elif is_v5_alpha_experiment(exp):
        audio, alpha_meta = synthesize_v5_alpha_body_dominant(
            frequency_hz=frequency_hz,
            duration_s=duration_s,
            sample_rate=sample_rate,
            modal_data=modal_data,
            variant=exp,
            sample_parameters=sample_parameters,
            repo_root=repo_root,
            sample_id=sample_id,
        )
        meta = {"experiment": exp, **alpha_meta}
        write_wav_int16(out, audio, sample_rate, duration_s=duration_s)
    else:
        meta = synthesize_note_with_body_response(
            frequency_hz=frequency_hz,
            note_name=note_name,
            duration_s=duration_s,
            sample_rate=sample_rate,
            modal_data=modal_data,
            output_wav=out,
            diagnostic_mode=mode,
            sample_parameters=sample_parameters,
            repo_root=repo_root,
            sample_id=sample_id,
        )
        meta = dict(meta)
        meta["experiment"] = exp or mode

    from body_response_synth import read_wav_float_mono

    audio_read, _ = read_wav_float_mono(out)
    if meta.get("mix_metrics"):
        mix = meta["mix_metrics"]
        str_rms = float(mix.get("string_rms_in_mix") or 0.0)
        bod_rms = float(mix.get("body_rms_in_mix") or 0.0)
    else:
        layers = decompose_baseline_layers(
            frequency_hz=frequency_hz,
            duration_s=duration_s,
            sample_rate=sample_rate,
            modal_data=modal_data,
        )
        str_rms = float(meta.get("string_rms") or layers["string_rms"])
        bod_rms = float(meta.get("body_rms") or layers["body_rms_raw"])
    metrics = enrich_metrics_with_levels(
        compute_realism_metrics(
            audio_read,
            sample_rate=sample_rate,
            frequency_hz=frequency_hz,
            string_rms=str_rms,
            body_rms=bod_rms,
        ),
        audio_read,
    )
    meta["realism_metrics"] = metrics
    return meta


def singleton_dz_body_quantification(
    *,
    sample_parameters: Mapping[str, Any],
    modal_data: ModalInput,
    frequency_hz: float,
    repo_root: Optional[Any] = None,
    sample_id: str = "website",
) -> Dict[str, Any]:
    """Quantify website single-guitar contrast inactivity for g_30_70."""
    from body_hybrid_v4_1_identity_space import (
        build_batch_contrast_context,
        build_body_identity_vector,
        compute_harmonic_gains,
        STRENGTH_PROFILES,
    )

    z_body = build_body_identity_vector(
        parameters=sample_parameters,
        modal_data=modal_data,
        frequency_hz=frequency_hz,
        repo_root=repo_root,
        sample_id=sample_id,
    )
    ctx = build_batch_contrast_context({sample_id: z_body}).get(sample_id) or {}
    dz = ctx.get("dz_body") or {}
    dz_vec = dz.get("vector") or []
    dz_abs_mean = float(np.mean(np.abs(dz_vec))) if dz_vec else 0.0
    dz_abs_max = float(np.max(np.abs(dz_vec))) if dz_vec else 0.0

    # Residual RMS proxy at harmonic level (no full audio render)
    abs_prof = STRENGTH_PROFILES["strong"]
    con_prof = STRENGTH_PROFILES["contrast_strong"]
    abs_gains = compute_harmonic_gains(z_body, frequency_hz=frequency_hz, profile=abs_prof, contrast=False)
    con_gains = compute_harmonic_gains(dz, frequency_hz=frequency_hz, profile=con_prof, contrast=True)
    abs_gain_rms = float(np.sqrt(np.mean(np.square(abs_gains))))
    con_gain_rms = float(np.sqrt(np.mean(np.square(con_gains))))

    effective_contrast_weight = 0.7 * (con_gain_rms / max(abs_gain_rms, 1e-9))
    effective_absolute_weight = 0.3

    return {
        "sample_count_in_batch": 1,
        "dz_body_abs_mean": round(dz_abs_mean, 6),
        "dz_body_abs_max": round(dz_abs_max, 6),
        "contrast_layer_inactive_on_website": dz_abs_max < 0.01,
        "nominal_blend_absolute": 0.3,
        "nominal_blend_contrast": 0.7,
        "effective_contrast_contribution_ratio": round(
            effective_contrast_weight / max(effective_absolute_weight + effective_contrast_weight, 1e-9),
            4,
        ),
        "interpretation": (
            "Website uses singleton batch: dz_body≈0, so 70% contrast branch is numerically inactive; "
            "only ~30% absolute identity residual applies on top of V4.1 base."
        ),
    }
