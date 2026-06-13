#!/usr/bin/env python3
"""
Stage 5.1C — bounded continuous body-identity layer on top of V4.1 (diagnostic only).

y_final = y_v4_1 + epsilon * residual, with harmonic shaping on harmonics 2–8.
Does not replace or alter V4.1 endpoint delegation.
"""
from __future__ import annotations

import json
import math
import tempfile
import wave
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from body_hybrid_v4_1 import synthesize_hybrid_v4_1_note
from body_response_synth import ModalInput, _rms, read_wav_float_mono
from bridge_mobility_proxy import compute_body_mass_proxies, effective_modal_mass_proxy
from modal_damping import WOOD_DAMPING_COEFF, compute_per_mode_damping, normalize_participation_shares
from sample_parameters import normalize_sample_parameters

IDENTITY_SPACE_MODES: Tuple[str, ...] = (
    "modal_body_hybrid_v4_1_identity_space",
    "stk_body_transfer_v4_1_identity_space",
)

# Bounded modifier ranges (diagnostic-only)
IDENTITY_EPSILON = 0.18
HARMONIC_GAIN_MAX = 0.12
FUNDAMENTAL_GAIN_MAX = 0.025
RESIDUAL_GAIN_MAX = 0.08
RMS_GUARD_MAX_DB = 1.5
PEAK_CLIP_DBFS = -0.5

# Typical LHS geometry ranges for feature normalization
_GEOM_RANGES: Dict[str, Tuple[float, float]] = {
    "length": (0.48, 0.58),
    "width": (0.24, 0.44),
    "depth": (0.09, 0.12),
    "top_thickness": (0.0028, 0.0036),
    "back_thickness": (0.0030, 0.0038),
    "hole_radius": (0.040, 0.055),
}

_FREQ_BANDS: Tuple[Tuple[float, float], ...] = (
    (60.0, 120.0),
    (120.0, 180.0),
    (180.0, 280.0),
    (280.0, 420.0),
    (420.0, 650.0),
)

_WOOD_EMBED: Dict[str, Tuple[float, float]] = {
    "spruce": (0.0, 1.0),
    "cedar": (-0.35, 0.85),
    "maple": (0.55, 1.15),
    "mahogany": (0.25, 1.05),
    "rosewood": (0.45, 1.12),
}


def is_v4_1_identity_space_mode(mode_name: Optional[str]) -> bool:
    return str(mode_name or "") in IDENTITY_SPACE_MODES


def _norm_range(val: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (float(val) - lo) / (hi - lo)))


def _to_centered(val01: float) -> float:
    return max(-1.0, min(1.0, 2.0 * float(val01) - 1.0))


def _wood_embed(wood_id: str) -> Tuple[float, float]:
    key = str(wood_id or "spruce").lower()
    return _WOOD_EMBED.get(key, (0.0, 1.0))


def _rank_norm(val: float, pool: Sequence[float]) -> float:
    pos = [float(v) for v in pool if v is not None and float(v) > 0]
    if not val or val <= 0 or not pos:
        return 0.12
    pos.sort()
    rank = sum(1 for v in pos if v <= float(val)) / max(len(pos), 1)
    return max(0.08, min(1.0, rank))


def _modes_from_modal(modal_data: ModalInput) -> List[Dict[str, Any]]:
    from body_response_synth import modes_in_validated_band, parse_modal_modes

    all_modes, _ = parse_modal_modes(modal_data)
    return modes_in_validated_band(all_modes)


def build_body_identity_vector(
    *,
    parameters: Optional[Mapping[str, Any]],
    modal_data: ModalInput,
    frequency_hz: float,
    repo_root: Optional[Path] = None,
    sample_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Construct z_body feature dict + flat vector for distance metrics."""
    p = normalize_sample_parameters(parameters)
    mass = compute_body_mass_proxies(p)
    modes = _modes_from_modal(modal_data)
    f0 = max(40.0, float(frequency_hz))

    from bridge_mobility_proxy import _g

    length = _g(p, "length", 0.52)
    width = _g(p, "width", 0.32)
    depth = _g(p, "depth", 0.10)
    top_t = _g(p, "top_thickness", 0.003)
    back_t = _g(p, "back_thickness", 0.0033)
    hole_r = _g(p, "hole_radius", 0.047)

    top_area = length * width * 0.92
    back_area = length * width * 0.88
    aspect = length / max(width, 1e-6)
    cavity = length * width * depth
    hole_area = math.pi * hole_r * hole_r

    top_w, top_d = _wood_embed(str(p.get("top_wood_id") or "spruce"))
    back_w, back_d = _wood_embed(str(p.get("back_wood_id") or "mahogany"))
    top_damp = WOOD_DAMPING_COEFF.get(str(p.get("top_wood_id") or "spruce").lower(), 1.0)
    back_damp = WOOD_DAMPING_COEFF.get(str(p.get("back_wood_id") or "mahogany").lower(), 1.0)

    rad_pool = [float(m.get("radiation_proxy") or 0) for m in modes]
    mic_pool = [float(m.get("mic_output_proxy") or 0) for m in modes]
    bridge_pool = [
        float(m.get("bridge_excitation_abs") or m.get("bridge_excitation_coupling") or 0) for m in modes
    ]

    band_density: Dict[str, float] = {}
    near_counts: Dict[str, int] = {}
    far_energy: Dict[str, float] = {f"band_{i}": 0.0 for i in range(len(_FREQ_BANDS))}
    q_vals: List[float] = []
    top_shares: List[float] = []
    back_shares: List[float] = []
    air_shares: List[float] = []
    eff_masses: List[float] = []
    dom_regions: Dict[str, int] = {}

    harmonics = [f0 * k for k in range(1, 9)]

    for bi, (lo, hi) in enumerate(_FREQ_BANDS):
        in_band = [m for m in modes if lo <= float(m.get("frequency_hz") or 0) <= hi]
        band_density[f"density_{int(lo)}_{int(hi)}"] = len(in_band) / max(len(modes), 1)
        near = 0
        for m in in_band:
            f_hz = float(m.get("frequency_hz") or 0)
            if any(abs(f_hz - h) / max(h, 1e-6) < 0.05 for h in harmonics):
                near += 1
            rad = float(m.get("radiation_proxy") or 0)
            far_energy[f"band_{bi}"] += rad
        near_counts[f"near_{int(lo)}_{int(hi)}"] = near

    for m in modes:
        top_s, back_s, air_s, _ = normalize_participation_shares(m)
        top_shares.append(top_s)
        back_shares.append(back_s)
        air_shares.append(air_s)
        eff_masses.append(effective_modal_mass_proxy(m, mass))
        q_rec = compute_per_mode_damping(m, float(m.get("frequency_hz") or f0), p)
        q = q_rec.get("mode_q")
        if q and float(q) > 0:
            q_vals.append(float(q))
        dr = str(m.get("dominant_region") or "unknown")
        dom_regions[dr] = dom_regions.get(dr, 0) + 1

    rad_sorted = sorted(rad_pool, reverse=True)
    k_top = max(1, int(math.ceil(len(rad_sorted) * 0.2))) if rad_sorted else 1
    top_rad_mean = float(np.mean(rad_sorted[:k_top])) if rad_sorted else 0.0

    cache_meta: Dict[str, Any] = {}
    if repo_root and sample_id:
        from body_signature_cache import load_body_signature_cache

        cache = load_body_signature_cache(Path(repo_root), str(sample_id))
        if cache:
            cache_meta = {k: cache.get(k) for k in (
                "bridge_mobility_proxy",
                "mixed_body_mass_proxy",
                "top_effective_mass_proxy",
                "back_effective_mass_proxy",
            ) if k in cache}

    features: Dict[str, float] = {
        "geom_length": _to_centered(_norm_range(length, *_GEOM_RANGES["length"])),
        "geom_width": _to_centered(_norm_range(width, *_GEOM_RANGES["width"])),
        "geom_depth": _to_centered(_norm_range(depth, *_GEOM_RANGES["depth"])),
        "geom_top_thickness": _to_centered(_norm_range(top_t, *_GEOM_RANGES["top_thickness"])),
        "geom_back_thickness": _to_centered(_norm_range(back_t, *_GEOM_RANGES["back_thickness"])),
        "geom_hole_radius": _to_centered(_norm_range(hole_r, *_GEOM_RANGES["hole_radius"])),
        "geom_top_area": _to_centered(_norm_range(top_area, 0.10, 0.26)),
        "geom_back_area": _to_centered(_norm_range(back_area, 0.09, 0.24)),
        "geom_air_volume": _to_centered(_norm_range(cavity, 0.008, 0.025)),
        "geom_aspect_ratio": _to_centered(_norm_range(aspect, 1.15, 2.0)),
        "geom_hole_to_area": _to_centered(_norm_range(hole_area / max(top_area, 1e-9), 0.01, 0.08)),
        "mat_top_wood_a": top_w,
        "mat_top_wood_b": top_d,
        "mat_back_wood_a": back_w,
        "mat_back_wood_b": back_d,
        "mat_top_damping": _to_centered(_norm_range(top_damp, 0.75, 1.30)),
        "mat_back_damping": _to_centered(_norm_range(back_damp, 0.75, 1.30)),
        "mass_top": _to_centered(_norm_range(mass["top_effective_mass_proxy"], 0.0003, 0.0007)),
        "mass_back": _to_centered(_norm_range(mass["back_effective_mass_proxy"], 0.0003, 0.0008)),
        "mass_mixed": _to_centered(_norm_range(mass["mixed_body_mass_proxy"], 0.00035, 0.00075)),
        "bridge_mobility": _to_centered(_norm_range(mass["bridge_mobility_proxy"], 0.72, 1.28)),
        "share_top_mean": _to_centered(float(np.mean(top_shares)) if top_shares else 0.0),
        "share_back_mean": _to_centered(float(np.mean(back_shares)) if back_shares else 0.0),
        "share_air_mean": _to_centered(float(np.mean(air_shares)) if air_shares else 0.0),
        "q_fingerprint": _to_centered(_norm_range(float(np.median(q_vals)) if q_vals else 40.0, 25.0, 70.0)),
        "q_spread": _to_centered(_norm_range(float(np.std(q_vals)) if len(q_vals) > 1 else 0.0, 0.0, 12.0)),
        "rad_rank_top": _rank_norm(top_rad_mean, rad_pool),
        "rad_rank_median": _rank_norm(float(np.median(rad_pool)) if rad_pool else 0.0, rad_pool),
        "mic_rank_median": _rank_norm(float(np.median(mic_pool)) if mic_pool else 0.0, mic_pool),
        "bridge_rank_median": _rank_norm(float(np.median(bridge_pool)) if bridge_pool else 0.0, bridge_pool),
        "eff_mass_median": _to_centered(_norm_range(float(np.median(eff_masses)) if eff_masses else 0.0, 0.0002, 0.001)),
        "low_body_color": _to_centered(_norm_range(far_energy.get("band_0", 0.0), 0.0, max(rad_pool or [1.0]))),
        "mid_body_color": _to_centered(_norm_range(far_energy.get("band_2", 0.0), 0.0, max(rad_pool or [1.0]))),
        "high_body_color": _to_centered(_norm_range(far_energy.get("band_4", 0.0), 0.0, max(rad_pool or [1.0]))),
    }
    for k, v in band_density.items():
        features[f"modal_{k}"] = _to_centered(min(1.0, float(v) * 4.0))
    for k, v in near_counts.items():
        features[f"modal_{k}"] = _to_centered(min(1.0, float(v) / 8.0))

    if cache_meta.get("bridge_mobility_proxy") is not None:
        features["cache_bridge_mobility"] = _to_centered(
            _norm_range(float(cache_meta["bridge_mobility_proxy"]), 0.72, 1.28)
        )

    flat = [float(features[k]) for k in sorted(features.keys())]
    return {
        "features": features,
        "vector": flat,
        "field_names": sorted(features.keys()),
        "mass_proxies": mass,
        "cache_meta": cache_meta,
        "dominant_region_histogram": dom_regions,
        "mode_count": len(modes),
    }


def compute_harmonic_gains(z_body: Mapping[str, Any], *, frequency_hz: float) -> List[float]:
    """Bounded log-domain gains per harmonic 1..8."""
    feats = z_body.get("features") or {}
    keys = sorted(feats.keys())
    if not keys:
        return [0.0] * 8
    vec = np.asarray([float(feats[k]) for k in keys], dtype=np.float64)
    gains: List[float] = []
    for k in range(1, 9):
        seed = np.sin(np.arange(len(vec)) * (k + 1) * 0.73 + float(frequency_hz) * 1e-4)
        raw = float(np.dot(vec, seed) / max(len(vec), 1))
        cap = FUNDAMENTAL_GAIN_MAX if k == 1 else HARMONIC_GAIN_MAX
        gains.append(max(-cap, min(cap, raw * 0.35)))
    return gains


def apply_harmonic_identity_shaping(
    audio: np.ndarray,
    *,
    frequency_hz: float,
    sample_rate: int,
    harmonic_gains: Sequence[float],
) -> np.ndarray:
    """Shape harmonics 2–8; fundamental nearly preserved."""
    x = np.asarray(audio, dtype=np.float64)
    n = len(x)
    if n < 64:
        return x.copy()
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sample_rate))
    f0 = float(frequency_hz)
    out = spec.copy()
    for k, gain in enumerate(harmonic_gains[:8], start=1):
        target = f0 * k
        if target >= freqs[-1]:
            continue
        idx = int(np.argmin(np.abs(freqs - target)))
        scale = math.exp(float(gain))
        if k == 1:
            scale = 1.0 + (scale - 1.0) * 0.15
        out[idx] *= scale
        if idx > 0:
            out[idx - 1] *= 1.0 + (scale - 1.0) * 0.25
        if idx + 1 < len(out):
            out[idx + 1] *= 1.0 + (scale - 1.0) * 0.25
    y = np.fft.irfft(out, n=n)
    return np.asarray(y, dtype=np.float64)


def apply_identity_residual(
    base: np.ndarray,
    shaped: np.ndarray,
    *,
    epsilon: float = IDENTITY_EPSILON,
) -> np.ndarray:
    residual = shaped - base
    cap = RESIDUAL_GAIN_MAX * float(np.max(np.abs(base)) + 1e-12)
    residual = np.clip(residual, -cap, cap)
    return base + float(epsilon) * residual


def apply_rms_guard(
    audio: np.ndarray,
    reference: np.ndarray,
    *,
    max_db: float = RMS_GUARD_MAX_DB,
) -> Tuple[np.ndarray, Dict[str, float]]:
    ref_rms = max(_rms(reference), 1e-12)
    out_rms = _rms(audio)
    if out_rms <= 1e-15:
        return audio.copy(), {"rms_guard_gain": 1.0}
    ratio_db = 20.0 * math.log10(out_rms / ref_rms)
    gain = 1.0
    if abs(ratio_db) > max_db:
        target = ref_rms * (10.0 ** (max_db / 20.0)) if ratio_db > 0 else ref_rms / (10.0 ** (max_db / 20.0))
        gain = target / out_rms
    guarded = audio * gain
    peak = float(np.max(np.abs(guarded)))
    if peak >= 1.0:
        gain *= 0.99 / peak
        guarded = guarded * (0.99 / peak)
    return guarded, {"rms_guard_gain": round(gain, 6), "rms_delta_db": round(ratio_db, 4)}


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(audio * 32767.0, -32767, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())


def synthesize_v4_1_identity_space_note(
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
    """V4.1 full base + bounded identity-space harmonic/residual layer."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        base_meta = synthesize_hybrid_v4_1_note(
            frequency_hz=frequency_hz,
            note_name=note_name,
            duration_s=duration_s,
            sample_rate=sample_rate,
            modal_data=modal_data,
            output_wav=tmp_path,
            output_metadata_json=None,
            velocity=velocity,
            sample_parameters=sample_parameters,
            modal_source=modal_source,
            diagnostic_mode="modal_body_hybrid_v4_1_full",
            synthesis_preset=synthesis_preset,
            repo_root=repo_root,
            sample_id=sample_id,
        )
        base_audio, _ = read_wav_float_mono(tmp_path)

        z_body = build_body_identity_vector(
            parameters=sample_parameters,
            modal_data=modal_data,
            frequency_hz=frequency_hz,
            repo_root=repo_root,
            sample_id=sample_id,
        )
        h_gains = compute_harmonic_gains(z_body, frequency_hz=frequency_hz)
        shaped = apply_harmonic_identity_shaping(
            base_audio,
            frequency_hz=frequency_hz,
            sample_rate=sample_rate,
            harmonic_gains=h_gains,
        )
        blended = apply_identity_residual(base_audio, shaped, epsilon=IDENTITY_EPSILON)
        final, guard = apply_rms_guard(blended, base_audio, max_db=RMS_GUARD_MAX_DB)

        peak_db = 20.0 * math.log10(max(float(np.max(np.abs(final))), 1e-12))
        _write_wav(Path(output_wav), final, sample_rate)

        meta = dict(base_meta)
        meta.update(
            {
                "diagnostic_mode": diagnostic_mode,
                "body_hybrid_v4_1_identity_space_active": True,
                "v4_1_base_preserved": True,
                "identity_epsilon": IDENTITY_EPSILON,
                "harmonic_gains": [round(g, 6) for g in h_gains],
                "harmonic_gain_bounds": {
                    "fundamental_max": FUNDAMENTAL_GAIN_MAX,
                    "harmonic_2_8_max": HARMONIC_GAIN_MAX,
                    "residual_max": RESIDUAL_GAIN_MAX,
                },
                "body_identity_vector": z_body,
                "identity_rms_guard": guard,
                "output_peak_dbfs": round(peak_db, 4),
                "clipping_avoided": peak_db < PEAK_CLIP_DBFS,
            }
        )
        if output_metadata_json:
            output_metadata_json.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        return meta
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def physical_distance(z_a: Mapping[str, Any], z_b: Mapping[str, Any]) -> float:
    va = np.asarray(z_a.get("vector") or [], dtype=np.float64)
    vb = np.asarray(z_b.get("vector") or [], dtype=np.float64)
    n = min(len(va), len(vb))
    if n == 0:
        return 0.0
    return float(np.linalg.norm(va[:n] - vb[:n]) / math.sqrt(n))


def audio_timbre_vector(
    audio: np.ndarray,
    *,
    sample_rate: int,
    segment_meta: Optional[Mapping[str, Any]] = None,
) -> List[float]:
    from diagnostic_synthesis import _spectral_features
    from stage48_timbre_decomposition_report import _attack_time_ms, _spectral_flux

    spec = _spectral_features(audio, sample_rate)
    meta = segment_meta or {}
    return [
        float(spec.get("centroid_hz") or 0.0) / 1000.0,
        float(spec.get("low_energy") or 0.0),
        float(spec.get("mid_energy") or 0.0),
        float(spec.get("high_energy") or 0.0),
        _spectral_flux(audio, sample_rate) * 1e-4,
        _attack_time_ms(audio, sample_rate) * 0.01,
        float(meta.get("output_decay_slope_db_per_s") or 0.0) * -0.01,
        float(meta.get("final_rms_dbfs") or 0.0) * 0.01,
    ]


def audio_distance(t_a: Sequence[float], t_b: Sequence[float]) -> float:
    va = np.asarray(t_a, dtype=np.float64)
    vb = np.asarray(t_b, dtype=np.float64)
    n = min(len(va), len(vb))
    if n == 0:
        return 0.0
    return float(np.linalg.norm(va[:n] - vb[:n]) / math.sqrt(n))


def spearman_correlation(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) < 3 or len(xs) != len(ys):
        return None

    def _rank(vals: Sequence[float]) -> List[float]:
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        ranks = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    rx, ry = _rank(list(xs)), _rank(list(ys))
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den_x = math.sqrt(sum((a - mx) ** 2 for a in rx))
    den_y = math.sqrt(sum((b - my) ** 2 for b in ry))
    if den_x <= 0 or den_y <= 0:
        return None
    return round(num / (den_x * den_y), 6)


def distance_consistency_report(
    samples: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Pairwise physical vs audio distance + Spearman correlation."""
    n = len(samples)
    phys: List[float] = []
    audio: List[float] = []
    pairs: List[Dict[str, Any]] = []
    for i in range(n):
        for j in range(i + 1, n):
            si, sj = samples[i], samples[j]
            pd = physical_distance(si.get("z_body") or {}, sj.get("z_body") or {})
            ad = audio_distance(si.get("timbre") or [], sj.get("timbre") or [])
            phys.append(pd)
            audio.append(ad)
            pairs.append(
                {
                    "sample_a": si.get("sample_id"),
                    "sample_b": sj.get("sample_id"),
                    "physical_distance": round(pd, 6),
                    "audio_distance": round(ad, 6),
                }
            )
    rho = spearman_correlation(phys, audio)
    return {
        "pair_count": len(pairs),
        "physical_distances": phys,
        "audio_distances": audio,
        "spearman_rho": rho,
        "pairs_sample": pairs[:15],
    }


def estimate_audible_clusters(audio_distances_flat: Sequence[float], *, threshold: float = 0.12) -> int:
    """Greedy cluster count from pairwise audio distances (diagnostic estimate)."""
    if not audio_distances_flat:
        return 0
    # approximate: count distance bins above threshold relative to median
    med = float(np.median(audio_distances_flat))
    far = sum(1 for d in audio_distances_flat if d > max(threshold, med * 0.85))
    # rough clusters ~ sqrt(far pairs) heuristic capped 2..10
    est = max(2, min(10, int(round(math.sqrt(max(far, 1)) + 1))))
    return est
