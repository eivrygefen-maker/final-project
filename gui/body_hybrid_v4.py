#!/usr/bin/env python3
"""
Stage 5.0 — modal_body_hybrid_v4: f0-blend baseline + radiation v1 + contrast-preserving signature.
"""
from __future__ import annotations

import json
import math
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from body_response_synth import (
    DEFAULT_VELOCITY,
    ModalInput,
    _rms,
    apply_anti_click_taper,
    apply_loudness_finalize,
    modes_in_validated_band,
    parse_modal_modes,
)
from body_signature_cache import (
    DEFAULT_GRID_HZ,
    build_reference_logG,
    interpolate_D_at_frequency,
    load_body_signature_cache,
    write_body_signature_cache,
    _params_hash,
)
from bridge_mobility_proxy import bridge_body_coupling_factor, compute_body_mass_proxies
from string_body_balance import _smoothstep

V4_DMAX_DB = 4.5
V4_MOBILITY_CLAMP = (0.95, 1.05)
V4_BODY_LAYER_DB = -21.0
V4_RAD_CROSSFADE = (160.0, 320.0)
LOG_EPS = 1e-9


@dataclass(frozen=True)
class V4Ablation:
    contrast_imprint: bool = False
    contrast_body_layer: bool = False
    mobility_light: bool = False


V4_MODE_ABLATIONS: Dict[str, V4Ablation] = {
    "modal_body_hybrid_v4": V4Ablation(True, True, True),
    "modal_body_hybrid_v4_full": V4Ablation(True, True, True),
    "stk_body_transfer_v4": V4Ablation(True, True, True),
    "modal_body_hybrid_v4_core": V4Ablation(False, False, False),
    "modal_body_hybrid_v4_contrast_imprint_only": V4Ablation(True, False, False),
    "modal_body_hybrid_v4_contrast_body_layer_only": V4Ablation(False, True, False),
    "modal_body_hybrid_v4_mobility_light_only": V4Ablation(False, False, True),
}


def get_v4_ablation(mode_name: Optional[str]) -> Optional[V4Ablation]:
    if not mode_name:
        return None
    return V4_MODE_ABLATIONS.get(mode_name)


def is_v4_family_mode(mode_name: Optional[str]) -> bool:
    return get_v4_ablation(mode_name) is not None


def radiation_blend_weight_f0(f0: float) -> float:
    return _smoothstep(V4_RAD_CROSSFADE[0], V4_RAD_CROSSFADE[1], max(40.0, float(f0)))


def alpha_contrast_strength(f0: float) -> float:
    f0 = max(40.0, float(f0))
    low = 1.0 - _smoothstep(120.0, 200.0, f0)
    mid = _smoothstep(120.0, 200.0, f0) * (1.0 - _smoothstep(280.0, 420.0, f0))
    high = _smoothstep(380.0, 520.0, f0)
    return max(0.0, min(1.0, 0.85 * low + 0.35 * mid + 0.05 * high))


def beta_harmonic(k: int) -> float:
    if k <= 1:
        return 0.05
    if k <= 8:
        return 0.25 + 0.025 * (k - 2)
    return max(0.08, 0.45 * math.exp(-0.22 * (k - 8)))


def _rank_p95(val: float, pool: Sequence[float]) -> float:
    vals = sorted(float(v) for v in pool if v > 0)
    if not vals or val <= 0:
        return 0.12
    p95 = vals[min(len(vals) - 1, int(math.ceil(0.95 * len(vals))) - 1)]
    return max(0.08, min(1.0, float(val) / max(p95, vals[-1], 1e-12)))


def _light_mobility_factor(
    mode: Mapping[str, Any],
    parameters: Optional[Mapping[str, Any]],
) -> float:
    coupled, _ = bridge_body_coupling_factor(mode, parameters, existing_bridge=1.0)
    lo, hi = V4_MOBILITY_CLAMP
    return max(lo, min(hi, coupled**0.35))


def _lorentzian_mag(f: np.ndarray, f_m: float, q: float) -> np.ndarray:
    r = f / max(f_m, 1e-6)
    qv = max(float(q), 4.0)
    denom = (1.0 - r * r) ** 2 + (r / qv) ** 2
    return 1.0 / np.sqrt(np.maximum(denom, 1e-18))


def _proxy_pools(band_modes: Sequence[Mapping[str, Any]]) -> Dict[str, List[float]]:
    bridges, rads, mics = [], [], []
    for mode in band_modes:
        b = mode.get("bridge_excitation_abs") or mode.get("bridge_excitation_coupling")
        if b is not None and float(b) > 0:
            bridges.append(abs(float(b)))
        r = mode.get("radiation_proxy")
        if r is not None and float(r) > 0:
            rads.append(float(r))
        m = mode.get("mic_output_proxy")
        if m is not None and float(m) > 0:
            mics.append(float(m))
    return {"bridge": bridges, "radiation": rads, "mic": mics}


def _mode_envelope_weight(
    mode: Mapping[str, Any],
    *,
    pools: Mapping[str, List[float]],
    parameters: Optional[Mapping[str, Any]],
    mobility_light: bool,
) -> float:
    bridge = abs(float(mode.get("bridge_excitation_abs") or mode.get("bridge_excitation_coupling") or 0.0))
    if bridge <= 0:
        bridge = 0.12
    rad = float(mode.get("radiation_proxy") or 0.12)
    mic = float(mode.get("mic_output_proxy") or 0.12)
    w = (
        0.45 * _rank_p95(bridge, pools.get("bridge") or [])
        + 0.32 * _rank_p95(rad, pools.get("radiation") or [])
        + 0.23 * _rank_p95(mic, pools.get("mic") or [])
    )
    top = float(mode.get("top_participation") or mode.get("top_share") or 0.25)
    back = float(mode.get("back_participation") or mode.get("back_share") or 0.25)
    air = float(mode.get("air_participation") or mode.get("air_share") or 0.25)
    w *= 0.55 + 0.25 * air + 0.12 * back + 0.08 * top
    if mobility_light:
        w *= _light_mobility_factor(mode, parameters)
    return max(w, 1e-9)


def compute_body_transfer_envelope(
    band_modes: Sequence[Mapping[str, Any]],
    frequencies_hz: np.ndarray,
    *,
    parameters: Optional[Mapping[str, Any]] = None,
    mobility_light: bool = False,
) -> Tuple[np.ndarray, List[float], Dict[str, Any]]:
    f = np.asarray(frequencies_hz, dtype=np.float64)
    G = np.zeros_like(f)
    pools = _proxy_pools(band_modes)
    weights: List[float] = []
    for mode in band_modes:
        f_m = float(mode.get("frequency_hz") or 0.0)
        if f_m <= 0:
            continue
        q = float((mode.get("damping") or {}).get("mode_q") or mode.get("mode_q") or 22.0)
        w_m = _mode_envelope_weight(
            mode, pools=pools, parameters=parameters, mobility_light=mobility_light
        )
        weights.append(w_m)
        G += w_m * _lorentzian_mag(f, f_m, q)
    if float(np.max(G)) > 1e-12:
        G /= float(np.max(G))
    G = 0.35 + 0.65 * G
    G = np.convolve(G, np.ones(7, dtype=np.float64) / 7.0, mode="same")
    return G, weights, {"mode_count": len(weights), "G_peak": round(float(np.max(G)), 6)}


def compute_contrast_D(
    logG_sample: np.ndarray,
    logG_ref: np.ndarray,
    *,
    dmax_db: float = V4_DMAX_DB,
) -> np.ndarray:
    d_natural_max = dmax_db / (20.0 / math.log(10.0))
    delta = np.asarray(logG_sample, dtype=np.float64) - np.asarray(logG_ref, dtype=np.float64)
    return np.clip(delta, -d_natural_max, d_natural_max)


def build_sample_signature_cache(
    repo_root: Path,
    sample_id: str,
    modal_data: ModalInput,
    *,
    parameters: Optional[Mapping[str, Any]],
    logG_ref: Optional[np.ndarray] = None,
    mobility_light: bool = False,
    grid: Tuple[float, float, int] = DEFAULT_GRID_HZ,
) -> Dict[str, Any]:
    lo, hi, n = grid
    freqs = np.linspace(lo, hi, int(n), dtype=np.float64)
    all_modes, _ = parse_modal_modes(modal_data)
    band_modes = modes_in_validated_band(all_modes)
    G, weights, env_meta = compute_body_transfer_envelope(
        band_modes, freqs, parameters=parameters, mobility_light=mobility_light
    )
    logG = np.log(G + LOG_EPS)
    D = compute_contrast_D(logG, logG_ref) if logG_ref is not None else np.zeros_like(logG)
    mass = compute_body_mass_proxies(parameters)
    meta = {
        "params_hash": _params_hash(parameters or {}),
        "envelope_meta": env_meta,
        "dmax_db": V4_DMAX_DB,
        **mass,
    }
    write_body_signature_cache(
        repo_root,
        sample_id,
        frequencies_hz=freqs,
        G_sample=G,
        logG_sample=logG,
        D_sample=D,
        modal_weights=weights,
        metadata=meta,
    )
    return {"frequencies_hz": freqs, "G_sample": G, "logG_sample": logG, "D_sample": D, "modal_weights": weights, **meta}


def precompute_batch_caches(
    repo_root: Path,
    samples: Sequence[Mapping[str, Any]],
    modal_resolver,
) -> Dict[str, Dict[str, Any]]:
    logG_list: List[np.ndarray] = []
    caches: Dict[str, Dict[str, Any]] = {}
    freqs_ref: Optional[np.ndarray] = None
    for sample in samples:
        sid = str(sample["sample_id"])
        modal_data, _ = modal_resolver(sample)
        entry = build_sample_signature_cache(
            repo_root, sid, modal_data, parameters=sample.get("parameters"), logG_ref=None
        )
        caches[sid] = entry
        logG_list.append(entry["logG_sample"])
        freqs_ref = entry["frequencies_hz"]
    logG_ref = build_reference_logG(logG_list)
    for sample in samples:
        sid = str(sample["sample_id"])
        modal_data, _ = modal_resolver(sample)
        caches[sid] = build_sample_signature_cache(
            repo_root,
            sid,
            modal_data,
            parameters=sample.get("parameters"),
            logG_ref=logG_ref,
        )
    if freqs_ref is not None:
        ref_path = repo_root / "ROM" / "classic" / "body_signature_cache" / "_reference_logG.npz"
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(ref_path, frequencies_hz=freqs_ref, logG_ref=logG_ref)
    return caches


def _rms_match(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ra, rb = _rms(a), _rms(b)
    target = math.sqrt(max(ra * rb, 1e-18))
    return a * target / max(ra, 1e-12), b * target / max(rb, 1e-12)


def apply_harmonic_contrast_imprint(
    signal: np.ndarray,
    sample_rate: int,
    f0: float,
    D: np.ndarray,
    freqs_hz: np.ndarray,
) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64)
    n = len(x)
    alpha = alpha_contrast_strength(f0)
    if n < 16 or alpha <= 1e-6:
        return x
    spec = np.fft.rfft(x)
    f_axis = np.fft.rfftfreq(n, d=1.0 / float(sample_rate))
    out = spec.copy()
    for k in range(1, 14):
        fk = k * float(f0)
        if fk >= f_axis[-1]:
            break
        idx = int(np.argmin(np.abs(f_axis - fk)))
        d_val = interpolate_D_at_frequency(D, freqs_hz, fk)
        gain = math.exp(alpha * beta_harmonic(k) * d_val)
        out[idx] *= max(0.82, min(1.22, gain))
    return np.fft.irfft(out, n=n)


def apply_contrast_body_layer(
    string_signal: np.ndarray,
    sample_rate: int,
    f0: float,
    D: np.ndarray,
    freqs_hz: np.ndarray,
    *,
    level_db: float = V4_BODY_LAYER_DB,
) -> np.ndarray:
    layer_strength = (1.0 - radiation_blend_weight_f0(f0)) * 10.0 ** (level_db / 20.0)
    if layer_strength <= 1e-9:
        return np.zeros_like(string_signal)
    n = len(string_signal)
    spec = np.fft.rfft(string_signal)
    f_axis = np.fft.rfftfreq(n, d=1.0 / float(sample_rate))
    filt = np.ones_like(f_axis)
    for i, ff in enumerate(f_axis):
        d_val = interpolate_D_at_frequency(D, freqs_hz, float(ff))
        filt[i] = max(-0.25, min(0.25, math.exp(0.35 * d_val) - 1.0))
    return layer_strength * np.fft.irfft(spec * filt, n=n)


def synthesize_hybrid_v4_note(
    *,
    frequency_hz: float,
    note_name: str,
    duration_s: float,
    sample_rate: int,
    modal_data: ModalInput,
    output_wav: Path,
    output_metadata_json: Optional[Path],
    velocity: float,
    sample_parameters: Optional[Mapping[str, Any]],
    modal_source: Optional[str],
    diagnostic_mode: str,
    synthesis_preset: Optional[str],
    repo_root: Path,
    sample_id: str,
) -> Dict[str, Any]:
    from body_response_synth import _synthesize_note_with_body_response_core
    from build_sample_comparison import read_wav_float_mono
    from diagnostic_synthesis import _spectral_features, use_diagnostic_mode
    from synthesis_presets import use_synthesis_preset
    from timbre_decomposition import compute_note_layers

    ablation = get_v4_ablation(diagnostic_mode) or V4Ablation()
    tmp_dir = output_wav.parent / "_v4_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    def _render(mode: str, path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        with use_synthesis_preset(synthesis_preset):
            with use_diagnostic_mode(mode, sample_parameters=sample_parameters):
                meta = _synthesize_note_with_body_response_core(
                    frequency_hz=frequency_hz,
                    note_name=note_name,
                    duration_s=duration_s,
                    sample_rate=sample_rate,
                    modal_data=modal_data,
                    output_wav=path,
                    velocity=velocity,
                    modal_source=modal_source,
                )
        audio, _ = read_wav_float_mono(path)
        return audio, meta

    baseline, baseline_meta = _render("baseline_current", tmp_dir / f"{sample_id}_{note_name}_base.wav")
    v1, _ = _render("modal_radiation_color_v1", tmp_dir / f"{sample_id}_{note_name}_v1.wav")
    baseline, v1 = _rms_match(baseline, v1)
    w_rad = radiation_blend_weight_f0(frequency_hz)
    hybrid = (1.0 - w_rad) * baseline + w_rad * v1

    cache = load_body_signature_cache(repo_root, sample_id)
    if cache is None:
        cache = build_sample_signature_cache(
            repo_root, sample_id, modal_data, parameters=sample_parameters, logG_ref=None
        )
    if ablation.mobility_light:
        ref_path = repo_root / "ROM" / "classic" / "body_signature_cache" / "_reference_logG.npz"
        logG_ref = None
        if ref_path.is_file():
            with np.load(ref_path) as z:
                logG_ref = z["logG_ref"]
        cache = build_sample_signature_cache(
            repo_root,
            sample_id,
            modal_data,
            parameters=sample_parameters,
            logG_ref=logG_ref,
            mobility_light=True,
        )
    D = np.asarray(cache["D_sample"], dtype=np.float64)
    freqs = np.asarray(cache["frequencies_hz"], dtype=np.float64)
    layers = compute_note_layers(
        frequency_hz, note_name, duration_s, sample_rate, modal_data,
        sample_parameters=sample_parameters, modal_source=modal_source,
    )
    string_only = np.asarray(layers["layers"]["string_only"], dtype=np.float64)
    if len(string_only) != len(hybrid):
        string_only = string_only[: len(hybrid)]
    body_part = hybrid - string_only

    pre_norm = hybrid.copy()
    if ablation.contrast_imprint:
        hybrid = apply_harmonic_contrast_imprint(string_only, sample_rate, frequency_hz, D, freqs) + body_part
    if ablation.contrast_body_layer:
        hybrid = hybrid + apply_contrast_body_layer(string_only, sample_rate, frequency_hz, D, freqs)

    pre_rms, post_rms = _rms(pre_norm), _rms(hybrid)
    if post_rms > 1e-12 and pre_rms > 1e-12:
        hybrid *= pre_rms / post_rms

    spec_before = _spectral_features(pre_norm, sample_rate)
    spec_after = _spectral_features(hybrid, sample_rate)
    tapered, taper_info = apply_anti_click_taper(hybrid, sample_rate, duration_s=duration_s)
    final, loudness_info = apply_loudness_finalize(
        tapered, sample_rate, raw_body_variation_preserve=0.50, loudness_normalization_strength=0.32
    )
    loudness_info.update(taper_info)

    output_wav = Path(output_wav)
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(final * 32767.0, -32767, 32767).astype(np.int16)
    with wave.open(str(output_wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())

    metadata: Dict[str, Any] = {
        **baseline_meta,
        "diagnostic_mode": diagnostic_mode,
        "body_hybrid_v4_active": True,
        "v4_ablation": {
            "contrast_imprint": ablation.contrast_imprint,
            "contrast_body_layer": ablation.contrast_body_layer,
            "mobility_light": ablation.mobility_light,
        },
        "radiation_blend_weight_w_rad": round(w_rad, 6),
        "normalization_diagnostics": {
            "rms_before_post_v4": round(pre_rms, 8),
            "rms_after_post_v4": round(post_rms, 8),
            "spectral_centroid_before": spec_before.get("centroid_hz"),
            "spectral_centroid_after": spec_after.get("centroid_hz"),
        },
        "output_rms_dbfs": loudness_info.get("output_rms_dbfs"),
        "output_peak_dbfs": loudness_info.get("output_peak_dbfs"),
        "limiter_used": loudness_info.get("limiter_used"),
    }
    if output_metadata_json:
        output_metadata_json.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return metadata
