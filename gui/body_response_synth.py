#!/usr/bin/env python3
"""
Classic guitar body-response synthesizer (modal transfer-function model).

Signal path (physically motivated):
  plucked-string harmonics -> bridge acceleration
  -> H_body(f) = sum_m W_m H_m(f)  [primary timbre / radiation]
  + small direct attack tap      [pitch anchor / pluck clarity only]

All modes in the validated body band contribute. Final peak normalize once.
"""
from __future__ import annotations

import json
import math
import wave
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

FULL_MODAL_BAND_HZ: Tuple[float, float] = (60.0, 550.0)
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_DURATION_S = 3.0
DEFAULT_VELOCITY = 1.0

# Fixed excitation (body ROM weights are the main guitar-to-guitar variable).
FIXED_PLUCK_POSITION = 0.18
BODY_REFERENCE_GAIN = 1.0
TARGET_BODY_TO_ATTACK_RMS_RATIO = 5.5
DIRECT_ATTACK_GAIN = 0.11
ATTACK_DECAY_S = 0.040
PLUCK_TRANSIENT_MS = 0.006
FIXED_RAD_K = 0.08
MAX_HARMONICS = 48
Q_MIN = 22.0
Q_MAX = 75.0
CONTRIBUTION_THRESHOLD_REL = 1e-5
TOP_CONTRIBUTING_MODES_N = 15
HARMONIC_ROLLOFF_POWER = 1.15

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
    if bridge is None or bridge <= 0:
        bridge = 1.0
        defaults_used.append("bridge_excitation_abs=1.0")
    else:
        flags["bridge_weighting_used"] = True
    bridge_w = bridge

    mic = _safe_float(mode.get("mic_output_proxy"))
    if mic is not None and mic > 0:
        flags["mic_proxy_used"] = True
        mic_w = mic
    else:
        mic_w = 1.0
        defaults_used.append("mic_output_proxy=1.0")

    rad = _safe_float(mode.get("radiation_proxy"))
    if rad is not None and rad > 0:
        flags["radiation_proxy_used"] = True
        rad_w = rad
    else:
        rad_w = 1.0
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
    }


def estimate_mode_q(mode: Mapping[str, Any], f_hz: float, defaults_used: List[str]) -> float:
    for key in ("Q", "q", "modal_q", "quality_factor"):
        q = _safe_float(mode.get(key))
        if q is not None and q > 0:
            return float(max(Q_MIN, min(Q_MAX, q)))

    air = _safe_float(mode.get("air_share")) or 0.22
    wood = (_safe_float(mode.get("top_share")) or 0.33) + (_safe_float(mode.get("back_share")) or 0.33)
    lo, hi = FULL_MODAL_BAND_HZ
    f_norm = min(max((f_hz - lo) / max(hi - lo, 1.0), 0.0), 1.0)
    q_est = (42.0 + 28.0 * air + 10.0 * wood) * (1.0 - 0.32 * f_norm)
    defaults_used.append("Q_estimated_from_frequency_and_participation")
    return float(max(Q_MIN, min(Q_MAX, q_est)))


def _total_q_with_radiation_loss(q_wood: float, f_hz: float, rad_k: float) -> float:
    inv_q = (1.0 / max(q_wood, 0.5)) + rad_k * (f_hz / 1000.0)
    return max(0.5, 1.0 / max(inv_q, 1e-9))


def _complex_mode_response(f_hz: np.ndarray, f_m: float, q: float) -> np.ndarray:
    fm = max(float(f_m), 1.0)
    qv = max(float(q), 0.5)
    r = np.asarray(f_hz, dtype=np.float64) / fm
    denom = (1.0 - r * r) + 1.0j * (r / qv)
    return 1.0 / denom


def _hf_transfer_envelope(f_hz: np.ndarray) -> np.ndarray:
    _, hi = FULL_MODAL_BAND_HZ
    f = np.asarray(f_hz, dtype=np.float64)
    env = np.ones_like(f)
    above = f > hi
    env[above] = np.maximum(0.06, (hi / f[above]) ** 1.15)
    return env


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
        harm_f.append(fk)
        harm_a.append(pluck_factor / (k ** HARMONIC_ROLLOFF_POWER))
    return harm_f, harm_a


def nearest_harmonic_hz(mode_hz: float, f0: float, harmonics_hz: Sequence[float]) -> float:
    if not harmonics_hz:
        return f0
    return min(harmonics_hz, key=lambda h: abs(h - mode_hz))


def _pluck_attack_envelope(n: int, sample_rate: int) -> np.ndarray:
    """Short onset emphasis for pluck realism (deterministic)."""
    t = np.arange(n, dtype=np.float64) / float(sample_rate)
    transient = 1.0 + 1.8 * np.exp(-t / max(PLUCK_TRANSIENT_MS, 1e-4))
    return transient


def _direct_attack_tap(dry: np.ndarray, sample_rate: int) -> np.ndarray:
    """Direct string component: attack/pitch anchor only, not sustained dry string."""
    t = np.arange(len(dry), dtype=np.float64) / float(sample_rate)
    attack_env = np.exp(-t / ATTACK_DECAY_S)
    return dry * attack_env


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
        tau_k = 0.55 + 2.8 / (k ** 0.82)
        phase = 0.0
        signal += amp_k * np.sin(2.0 * math.pi * fk * t + phase) * np.exp(-t / tau_k)
    signal *= _pluck_attack_envelope(n, sample_rate)
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
) -> Tuple[np.ndarray, List[Dict[str, Any]], List[float]]:
    """
    H_body(f) = sum_m W_m H_m(f) on bridge acceleration spectrum.
    Uses stable BODY_REFERENCE_GAIN — no per-guitar H normalization or 1/sqrt(N) scaling.
    """
    n = len(acc)
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sample_rate))
    acc_spec = np.fft.rfft(acc)
    hf_env = _hf_transfer_envelope(freqs)

    H_body = np.zeros_like(freqs, dtype=np.complex128)
    mode_rows: List[Dict[str, Any]] = []
    q_values: List[float] = []

    for mode in band_modes:
        f_m = float(mode["frequency_hz"])
        comp = compute_mode_weight_components(mode, defaults_used=defaults_used, flags=flags)
        w = comp["combined"]
        q_wood = estimate_mode_q(mode, f_m, defaults_used)
        q_total = _total_q_with_radiation_loss(q_wood, f_m, FIXED_RAD_K)
        flags["q_or_damping_used"] = True
        q_values.append(q_total)
        H_m = _complex_mode_response(freqs, f_m, q_total)
        H_body += w * H_m
        mode_rows.append(
            {
                "mode": mode,
                "f_m": f_m,
                "w": w,
                "comp": comp,
                "q": q_total,
                "H_m": H_m,
            }
        )

    H_body *= hf_env
    body_spec = acc_spec * H_body * BODY_REFERENCE_GAIN
    body = np.fft.irfft(body_spec, n=n)

    contributions: List[Dict[str, Any]] = []
    f0 = max(float(note_hz), 1.0)
    for row in mode_rows:
        mode = row["mode"]
        w = row["w"]
        comp = row["comp"]
        q_total = row["q"]
        f_m = row["f_m"]
        H_m = row["H_m"] * hf_env
        mode_spec = acc_spec * w * H_m * BODY_REFERENCE_GAIN
        energy = float(np.sum(np.abs(mode_spec) ** 2))
        nearest_h = nearest_harmonic_hz(f_m, f0, harmonics_hz)
        contributions.append(
            {
                "mode_index": int(mode.get("mode_index", -1)),
                "frequency_hz": round(f_m, 4),
                "contribution_weight": energy,
                "bridge_weight": round(comp["bridge_weight"], 8),
                "mic_weight": round(comp["mic_weight"], 8),
                "radiation_weight": round(comp["radiation_weight"], 8),
                "q": round(q_total, 4),
                "nearest_harmonic_hz": round(nearest_h, 4),
            }
        )

    return body, contributions, q_values


def write_wav_int16(path: Path, samples: np.ndarray, sample_rate: int) -> Tuple[float, float]:
    """Write mono int16 WAV; return (peak_before_normalize, normalization_gain)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(samples, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        raise ValueError("WAV samples contain NaN or Inf")
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    norm_gain = 1.0
    if peak > 0:
        norm_gain = 0.95 / peak
        x = x * norm_gain
    pcm = np.clip(x * 32767.0, -32767, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())
    return peak, norm_gain


def synthesize_note_with_body_response(
    frequency_hz: float,
    note_name: str,
    duration_s: float,
    sample_rate: int,
    modal_data: ModalInput,
    output_wav: Path,
    output_metadata_json: Optional[Path] = None,
    velocity: float = DEFAULT_VELOCITY,
) -> Dict[str, Any]:
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

    if band_modes:
        body_raw, contributions, q_values = synthesize_body_via_transfer_function(
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
        contributions = []
        q_values = []
        defaults_used.append("no_modes_in_validated_band_60_550:body_bypass")

    direct_tap = _direct_attack_tap(string_excitation, sample_rate)
    dry_gain_applied = DIRECT_ATTACK_GAIN
    body_rms_before_calibration = _rms(body_raw)
    dry_rms_before = _rms(direct_tap)

    if body_rms_before_calibration > 1e-15 and dry_rms_before > 1e-15:
        body_gain_applied = (
            TARGET_BODY_TO_ATTACK_RMS_RATIO * dry_rms_before / body_rms_before_calibration
        )
        defaults_used.append(
            f"body_gain_calibration_to_target_ratio={TARGET_BODY_TO_ATTACK_RMS_RATIO}"
        )
    elif body_rms_before_calibration > 0:
        body_gain_applied = BODY_REFERENCE_GAIN
        defaults_used.append("body_gain_calibration_fallback")
    else:
        body_gain_applied = 0.0

    body_signal = body_raw * body_gain_applied
    body_rms_before = _rms(body_signal)
    mixed = body_signal + dry_gain_applied * direct_tap
    final_dry_to_body_rms_ratio = (dry_gain_applied * dry_rms_before) / max(body_rms_before, 1e-12)

    peak_before, final_peak_normalization_gain = write_wav_int16(Path(output_wav), mixed, sample_rate)

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
            f"body_rms_calibration_target_ratio={TARGET_BODY_TO_ATTACK_RMS_RATIO}"
        ),
        "harmonics_used_hz": [round(h, 4) for h in harmonics_hz],
        "top_contributing_modes": top_modes,
        "full_modal_band_hz": list(FULL_MODAL_BAND_HZ),
        "high_frequency_fallback_used": bool(hf_fallback),
        "bridge_weighting_used": flags["bridge_weighting_used"],
        "mic_proxy_used": flags["mic_proxy_used"],
        "radiation_proxy_used": flags["radiation_proxy_used"],
        "q_or_damping_used": flags["q_or_damping_used"],
        "direct_string_role": "attack_pitch_anchor_only",
        "direct_string_gain": round(dry_gain_applied, 6),
        "body_filter_gain": round(BODY_REFERENCE_GAIN, 6),
        "body_rms_before_calibration": round(body_rms_before_calibration, 8),
        "dry_mix": round(dry_gain_applied, 6),
        "wet_mix": round(body_gain_applied, 6),
        "dry_rms_before_mix": round(dry_rms_before, 8),
        "body_rms_before_mix": round(body_rms_before, 8),
        "target_body_to_attack_rms_ratio": TARGET_BODY_TO_ATTACK_RMS_RATIO,
        "dry_gain_applied": round(dry_gain_applied, 6),
        "body_gain_applied": round(body_gain_applied, 6),
        "final_dry_to_body_rms_ratio": round(final_dry_to_body_rms_ratio, 6),
        "final_peak_normalization_gain": round(final_peak_normalization_gain, 6),
        "peak_before_normalize": peak_before,
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
            "rad_k": FIXED_RAD_K,
            "q_clamp": [Q_MIN, Q_MAX],
        },
        "output_wav": str(output_wav),
        "samples_finite": True,
    }
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
    parser.add_argument("--out-dir", type=Path, default=Path("audio/stage1"))
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
            f"{name}: avail={meta['available_modal_count']} eval={meta['evaluated_modal_count']} "
            f"body/dry_rms_ratio={meta['body_rms_before_mix'] / max(meta['dry_rms_before_mix'], 1e-9):.1f} "
            f"hf={meta['high_frequency_fallback_used']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
