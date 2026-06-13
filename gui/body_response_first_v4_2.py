#!/usr/bin/env python3
"""
Stage 5.2A — body-response-first STK diagnostic (parallel to V4.1, not a replacement).

y = body_response_filter(string_excitation, H_guitar,note)
H_guitar,note(f) = Σ_m A_m,note · Lorentzian(f, f_m, Q_m)

No V1 overlay, no V4.1 hybrid base, no radiation_color_v1 as signal path.
"""
from __future__ import annotations

import json
import math
import wave
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from body_response_synth import (
    FIXED_PLUCK_POSITION,
    ModalInput,
    _complex_mode_response,
    _harmonic_proximity,
    _hf_transfer_envelope,
    _rms,
    _soften_mode_weights,
    _string_acceleration,
    apply_anti_click_taper,
    apply_loudness_finalize,
    harmonic_series,
    modes_in_validated_band,
    parse_modal_modes,
    synthesize_plucked_string,
)
from bridge_mobility_proxy import compute_body_mass_proxies, effective_modal_mass_proxy
from modal_damping import compute_per_mode_damping, normalize_participation_shares
from sample_parameters import normalize_sample_parameters

V4_2_MODES: Tuple[str, ...] = (
    "modal_body_response_first_v4_2",
    "stk_body_response_first_v4_2",
)

V42_BODY_GAIN = 1.05
V42_FUNDAMENTAL_DIRECT_TAP = 0.12
V42_ATTACK_DIRECT_TAP = 0.06
V42_LOUDNESS_STRENGTH = 0.28
V42_VARIATION_PRESERVE = 0.55
PEAK_CLIP_DBFS = -0.5
HARMONIC_IDENTITY_K_MIN = 2
HARMONIC_IDENTITY_K_MAX = 8


def is_v4_2_body_response_first_mode(mode_name: Optional[str]) -> bool:
    return str(mode_name or "") in V4_2_MODES


def _rank_norm(val: float, pool: Sequence[float]) -> float:
    pos = sorted(float(v) for v in pool if v is not None and float(v) > 0)
    if not val or float(val) <= 0 or not pos:
        return 0.12
    rank = sum(1 for v in pos if v <= float(val)) / max(len(pos), 1)
    return max(0.08, min(1.0, rank))


def _mobility_scale(parameters: Optional[Mapping[str, Any]], repo_root: Optional[Path], sample_id: str) -> float:
    p = normalize_sample_parameters(parameters)
    mass = compute_body_mass_proxies(p)
    mob = float(mass.get("bridge_mobility_proxy") or 1.0)
    scale = 0.85 + 0.30 * max(0.0, min(1.0, (mob - 0.72) / 0.56))
    if repo_root and sample_id:
        try:
            from body_signature_cache import load_body_signature_cache

            cache = load_body_signature_cache(Path(repo_root), str(sample_id))
            if cache and cache.get("bridge_mobility_proxy") is not None:
                cm = float(cache["bridge_mobility_proxy"])
                scale = 0.85 + 0.30 * max(0.0, min(1.0, (cm - 0.72) / 0.56))
        except Exception:
            pass
    return max(0.75, min(1.25, scale))


def _harmonic_coupling_weight(
    f_m: float,
    f0: float,
    harmonics_hz: Sequence[float],
    proximity: float,
) -> float:
    """Stronger coupling on harmonics 2–8; fundamental mostly preserved."""
    if not harmonics_hz:
        return 0.5
    nearest_h = min(harmonics_hz, key=lambda h: abs(float(h) - f_m))
    k = max(1, int(round(float(nearest_h) / max(f0, 1e-6))))
    if k == 1:
        return max(0.28, min(0.72, 0.32 + 0.40 * proximity))
    if HARMONIC_IDENTITY_K_MIN <= k <= HARMONIC_IDENTITY_K_MAX:
        return max(0.45, min(1.15, 0.52 + 0.58 * proximity))
    return max(0.10, min(0.45, 0.12 + 0.28 * proximity))


def compute_mode_amplitude_v4_2(
    mode: Mapping[str, Any],
    *,
    f0: float,
    harmonics_hz: Sequence[float],
    parameters: Optional[Mapping[str, Any]],
    rad_pool: Sequence[float],
    mic_pool: Sequence[float],
    bridge_pool: Sequence[float],
    inv_mass_pool: Sequence[float],
    mobility_scale: float,
) -> Dict[str, float]:
    """Bounded A_m,note from bridge, rank-normalized rad/mic, inv mass, participation, harmonic coupling."""
    f_m = float(mode.get("frequency_hz") or 0.0)
    bridge_raw = float(
        mode.get("bridge_excitation_abs")
        or mode.get("bridge_excitation_coupling")
        or 0.0
    )
    bridge_abs = abs(bridge_raw) if bridge_raw else 0.0
    bridge_rank = _rank_norm(bridge_abs, bridge_pool)
    bridge_inj = max(0.08, min(1.0, bridge_rank * mobility_scale))

    rad_rank = _rank_norm(float(mode.get("radiation_proxy") or 0.0), rad_pool)
    mic_rank = _rank_norm(float(mode.get("mic_output_proxy") or 0.0), mic_pool)

    mass = compute_body_mass_proxies(normalize_sample_parameters(parameters))
    eff_m = max(effective_modal_mass_proxy(mode, mass), 1e-8)
    inv_mass = 1.0 / eff_m
    inv_mass_rank = _rank_norm(inv_mass, inv_mass_pool)

    top_s, back_s, air_s, _ = normalize_participation_shares(mode)
    participation = max(0.25, min(1.15, 0.42 + 0.35 * top_s + 0.28 * back_s + 0.18 * air_s))

    proximity = _harmonic_proximity(f_m, harmonics_hz)
    h_weight = _harmonic_coupling_weight(f_m, f0, harmonics_hz, proximity)

    amp = bridge_inj * rad_rank * mic_rank * inv_mass_rank * participation * h_weight
    amp = max(1e-9, min(amp, 2.5))
    return {
        "A_m_note": round(amp, 8),
        "bridge_injection": round(bridge_inj, 6),
        "radiation_rank": round(rad_rank, 6),
        "mic_rank": round(mic_rank, 6),
        "inv_mass_rank": round(inv_mass_rank, 6),
        "participation_weight": round(participation, 6),
        "harmonic_coupling_weight": round(h_weight, 6),
        "harmonic_proximity": round(proximity, 6),
    }


def build_body_transfer_function_v4_2(
    *,
    sample_rate: int,
    n_samples: int,
    band_modes: Sequence[Mapping[str, Any]],
    frequency_hz: float,
    parameters: Optional[Mapping[str, Any]],
    repo_root: Optional[Path],
    sample_id: str,
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    """H_guitar,note(f) on rfft grid."""
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / float(sample_rate))
    f0 = max(40.0, float(frequency_hz))
    harmonics_hz, _ = harmonic_series(f0, sample_rate)

    rad_pool = [float(m.get("radiation_proxy") or 0) for m in band_modes]
    mic_pool = [float(m.get("mic_output_proxy") or 0) for m in band_modes]
    bridge_pool = [
        abs(float(m.get("bridge_excitation_abs") or m.get("bridge_excitation_coupling") or 0))
        for m in band_modes
    ]
    mass = compute_body_mass_proxies(normalize_sample_parameters(parameters))
    inv_mass_pool = [1.0 / max(effective_modal_mass_proxy(m, mass), 1e-8) for m in band_modes]
    mobility_scale = _mobility_scale(parameters, repo_root, sample_id)

    raw_weights: List[float] = []
    mode_rows: List[Dict[str, Any]] = []
    p = normalize_sample_parameters(parameters)

    for mode in band_modes:
        f_m = float(mode.get("frequency_hz") or 0.0)
        if f_m <= 0:
            continue
        amp_meta = compute_mode_amplitude_v4_2(
            mode,
            f0=f0,
            harmonics_hz=harmonics_hz,
            parameters=parameters,
            rad_pool=rad_pool,
            mic_pool=mic_pool,
            bridge_pool=bridge_pool,
            inv_mass_pool=inv_mass_pool,
            mobility_scale=mobility_scale,
        )
        damp = compute_per_mode_damping(mode, f_m, p)
        q_m = max(0.5, float(damp.get("mode_q") or 40.0))
        w = float(amp_meta["A_m_note"])
        raw_weights.append(w)
        mode_rows.append(
            {
                "frequency_hz": f_m,
                "Q_m": round(q_m, 4),
                "amplitude_meta": amp_meta,
                "damping": damp,
                "H_m_peak": None,
            }
        )

    softened, dom_before, dom_after = _soften_mode_weights(raw_weights)
    H = np.zeros_like(freqs, dtype=np.complex128)
    for row, w_eff in zip(mode_rows, softened):
        f_m = float(row["frequency_hz"])
        q_m = float(row["Q_m"])
        H_m = _complex_mode_response(freqs, f_m, q_m)
        row["w_eff"] = round(w_eff, 8)
        row["H_m_peak"] = round(float(np.max(np.abs(H_m))), 6)
        H += w_eff * H_m

    H *= _hf_transfer_envelope(freqs)
    h_peak = float(np.max(np.abs(H))) if len(H) else 0.0
    if h_peak > 1e-12:
        H = H / h_peak

    summary = {
        "mode_count": len(mode_rows),
        "mobility_scale": round(mobility_scale, 6),
        "dominance_before": round(dom_before, 6),
        "dominance_after": round(dom_after, 6),
        "H_peak_normalized": 1.0,
        "harmonics_used": len(harmonics_hz),
    }
    return H, mode_rows, summary


def synthesize_body_response_first_v4_2_note(
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
    """Body-response-first path: string excitation filtered by H_guitar,note."""
    del note_name, synthesis_preset  # f0-continuous; preset not used on this path

    all_modes, parse_defaults = parse_modal_modes(modal_data)
    band_modes = modes_in_validated_band(all_modes)
    f0 = float(frequency_hz)

    string_excitation = synthesize_plucked_string(
        f0,
        duration_s,
        sample_rate,
        pluck_position=FIXED_PLUCK_POSITION,
        velocity=velocity,
    )
    n = len(string_excitation)
    acc = _string_acceleration(string_excitation)

    if band_modes:
        H_body, mode_rows, h_summary = build_body_transfer_function_v4_2(
            sample_rate=sample_rate,
            n_samples=n,
            band_modes=band_modes,
            frequency_hz=f0,
            parameters=sample_parameters,
            repo_root=repo_root,
            sample_id=sample_id,
        )
        acc_spec = np.fft.rfft(acc)
        body_spec = acc_spec * H_body * V42_BODY_GAIN
        body_audio = np.real(np.fft.irfft(body_spec, n=n))
    else:
        mode_rows = []
        h_summary = {"mode_count": 0}
        body_audio = np.zeros(n, dtype=np.float64)

    attack_tap = V42_ATTACK_DIRECT_TAP * string_excitation * np.exp(
        -np.arange(n, dtype=np.float64) / max(sample_rate * 0.012, 1.0)
    )
    fundamental_tap = V42_FUNDAMENTAL_DIRECT_TAP * string_excitation
    mixed = body_audio + attack_tap + fundamental_tap

    tapered, taper_info = apply_anti_click_taper(mixed, sample_rate, duration_s=duration_s)
    final, loudness_info = apply_loudness_finalize(
        tapered,
        sample_rate,
        loudness_normalization_strength=V42_LOUDNESS_STRENGTH,
        raw_body_variation_preserve=V42_VARIATION_PRESERVE,
    )

    peak = float(np.max(np.abs(final)))
    peak_db = 20.0 * math.log10(max(peak, 1e-12))
    if peak >= 1.0:
        final = final * (0.99 / peak)
        peak_db = 20.0 * math.log10(0.99)

    _write_wav(Path(output_wav), final, sample_rate)
    f0_stability = _fundamental_stability_metric(final, sample_rate, f0)

    meta: Dict[str, Any] = {
        "diagnostic_mode": diagnostic_mode,
        "body_response_first_v4_2_active": True,
        "synthesis_model": "body_response_first_H_guitar_note",
        "v4_1_base_preserved": False,
        "v1_overlay_used": False,
        "radiation_color_v1_base": False,
        "formula": "y = IFFT(FFT(string_acc) * H_guitar,note) + bounded_direct_tap",
        "H_guitar_note": h_summary,
        "mode_transfer_rows": mode_rows[:24],
        "fundamental_direct_tap": V42_FUNDAMENTAL_DIRECT_TAP,
        "attack_direct_tap": V42_ATTACK_DIRECT_TAP,
        "body_gain": V42_BODY_GAIN,
        "modal_source": modal_source,
        "modal_mode_count_total": len(all_modes),
        "modal_mode_count_band": len(band_modes),
        "defaults_used": list(parse_defaults),
        "taper_info": taper_info,
        "loudness_info": loudness_info,
        "fundamental_stability_ratio": f0_stability,
        "output_peak_dbfs": round(peak_db, 4),
        "clipping_avoided": peak_db < PEAK_CLIP_DBFS,
    }
    if output_metadata_json:
        output_metadata_json.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return meta


def _fundamental_stability_metric(audio: np.ndarray, sample_rate: int, f0: float) -> Dict[str, float]:
    x = np.asarray(audio, dtype=np.float64)
    if len(x) < 64:
        return {"f0_energy_ratio": 0.0, "f0_to_h2_ratio": 0.0}
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), d=1.0 / float(sample_rate))
    total = float(np.sum(spec**2)) + 1e-18
    f0_idx = int(np.argmin(np.abs(freqs - f0)))
    h2_idx = int(np.argmin(np.abs(freqs - 2.0 * f0)))
    f0_e = float(spec[f0_idx] ** 2)
    h2_e = float(spec[h2_idx] ** 2) if h2_idx < len(spec) else 1e-18
    return {
        "f0_energy_ratio": round(f0_e / total, 6),
        "f0_to_h2_ratio": round(math.sqrt(f0_e / max(h2_e, 1e-18)), 6),
    }


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(audio * 32767.0, -32767, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())
