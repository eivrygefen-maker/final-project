#!/usr/bin/env python3
"""
Stage-1 classic guitar body-response synthesizer.

Fixed plucked-string excitation -> harmonic series -> modal resonator bank (60–550 Hz)
-> bridge/mic/radiation weighting -> Q/damping -> mixed waveform -> normalized WAV.

Pure NumPy; no FEM solve. Consumes ROM ``predicted_modes`` or legacy ``modes_hz`` JSON.
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

# Fixed excitation (body is the main variable between guitars).
FIXED_PLUCK_POSITION = 0.18
FIXED_STRING_MIX = 0.72
FIXED_WET_GAIN = 8.0
FIXED_RAD_K = 0.06
MAX_MODES_USED = 32
MAX_HARMONICS = 48

ModalInput = Union[Mapping[str, Any], Sequence[Mapping[str, Any]]]


def _safe_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        out = float(val)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def parse_modal_modes(modal_data: ModalInput) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Normalize ROM prediction dict, STK body JSON, or mode list."""
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


def _modes_in_modal_band(modes: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    lo, hi = FULL_MODAL_BAND_HZ
    out = []
    for m in modes:
        f = _safe_float(m.get("frequency_hz"))
        if f is None or f < lo or f > hi:
            continue
        out.append(dict(m))
    out.sort(key=lambda r: float(r["frequency_hz"]))
    return out[:MAX_MODES_USED]


def compute_mode_weight(
    mode: Mapping[str, Any],
    *,
    defaults_used: List[str],
    flags: Dict[str, bool],
) -> float:
    """Bridge × (mic/radiation) with documented fallbacks."""
    combined = _safe_float(mode.get("bridge_to_mic_gain_raw"))
    if combined is not None and combined > 0:
        flags["bridge_weighting_used"] = True
        flags["mic_proxy_used"] = True
        w = combined
    else:
        bridge = _safe_float(mode.get("bridge_excitation_abs"))
        if bridge is None:
            coup = _safe_float(mode.get("bridge_excitation_coupling"))
            bridge = abs(coup) if coup is not None else None
        if bridge is None or bridge <= 0:
            bridge = 1.0
            defaults_used.append("bridge_excitation_abs=1.0")
        else:
            flags["bridge_weighting_used"] = True

        mic = _safe_float(mode.get("mic_output_proxy"))
        rad = _safe_float(mode.get("radiation_proxy"))
        if mic is not None and mic > 0:
            flags["mic_proxy_used"] = True
        else:
            mic = 1.0
            defaults_used.append("mic_output_proxy=1.0")
        if rad is not None and rad > 0:
            flags["radiation_proxy_used"] = True
        else:
            rad = 1.0
            defaults_used.append("radiation_proxy=1.0")
        w = bridge * (0.55 * mic + 0.45 * rad)

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
    return max(w, 1e-12)


def estimate_mode_q(mode: Mapping[str, Any], f_hz: float, defaults_used: List[str]) -> float:
    for key in ("Q", "q", "modal_q", "quality_factor"):
        q = _safe_float(mode.get(key))
        if q is not None and q > 0:
            return float(max(15.0, min(180.0, q)))

    air = _safe_float(mode.get("air_share")) or 0.22
    wood = (_safe_float(mode.get("top_share")) or 0.33) + (_safe_float(mode.get("back_share")) or 0.33)
    lo, hi = FULL_MODAL_BAND_HZ
    f_norm = min(max((f_hz - lo) / max(hi - lo, 1.0), 0.0), 1.0)
    q_est = (48.0 + 38.0 * air + 12.0 * wood) * (1.0 - 0.28 * f_norm)
    defaults_used.append("Q_estimated_from_frequency_and_participation")
    return float(max(20.0, min(120.0, q_est)))


def _biquad_bandpass_coeffs(f0: float, q: float, fs: float) -> Tuple[float, float, float, float, float]:
    f0 = max(1.0, min(f0, 0.49 * fs))
    q = max(0.5, q)
    w0 = 2.0 * math.pi * (f0 / fs)
    alpha = math.sin(w0) / (2.0 * q)
    cosw0 = math.cos(w0)
    b0 = alpha
    b1 = 0.0
    b2 = -alpha
    a0 = 1.0 + alpha
    a1 = -2.0 * cosw0
    a2 = 1.0 - alpha
    return b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0


def _biquad_process(x: np.ndarray, b0: float, b1: float, b2: float, a1: float, a2: float) -> np.ndarray:
    y = np.empty_like(x)
    x1 = x2 = y1 = y2 = 0.0
    for i, xi in enumerate(x):
        yi = b0 * xi + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        x2, x1 = x1, xi
        y2, y1 = y1, yi
        y[i] = yi
    return y


def _total_q_with_radiation_loss(q_wood: float, f_hz: float, rad_k: float) -> float:
    inv_q = (1.0 / max(q_wood, 0.5)) + rad_k * (f_hz / 1000.0)
    return max(0.5, 1.0 / max(inv_q, 1e-9))


def synthesize_plucked_string(
    frequency_hz: float,
    duration_s: float,
    sample_rate: int,
    *,
    pluck_position: float = FIXED_PLUCK_POSITION,
    velocity: float = DEFAULT_VELOCITY,
    amplitude: float = 0.35,
) -> np.ndarray:
    """Fixed pluck position; pitch from ``frequency_hz`` (harmonic series)."""
    n = max(1, int(duration_s * sample_rate))
    t = np.arange(n, dtype=np.float64) / float(sample_rate)
    signal = np.zeros(n, dtype=np.float64)
    f0 = max(1.0, float(frequency_hz))
    max_harm = min(MAX_HARMONICS, int(sample_rate / (2.0 * f0)))
    for k in range(1, max_harm + 1):
        fk = k * f0
        if fk >= sample_rate * 0.49:
            break
        pluck_factor = abs(math.sin(math.pi * pluck_position * k))
        if pluck_factor < 1e-8:
            continue
        amp_k = velocity * amplitude * pluck_factor / k
        tau_k = 0.75 + 2.2 / k
        signal += amp_k * np.sin(2.0 * math.pi * fk * t) * np.exp(-t / tau_k)
    return signal


def _string_acceleration(dry: np.ndarray) -> np.ndarray:
    acc = np.zeros_like(dry)
    if len(dry) >= 3:
        acc[1:-1] = dry[:-2] - 2.0 * dry[1:-1] + dry[2:]
        acc[0] = acc[1]
        acc[-1] = acc[-2]
    return acc


def _hf_string_mix(note_hz: float) -> Tuple[float, bool]:
    """Above 550 Hz: more string-dominated, shorter effective body contribution."""
    _, hi = FULL_MODAL_BAND_HZ
    if note_hz <= hi:
        return FIXED_STRING_MIX, False
    excess = min((note_hz - hi) / hi, 1.5)
    string_mix = min(0.94, FIXED_STRING_MIX + 0.22 * excess)
    return string_mix, True


def _normalize_weights(weights: Sequence[float]) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    if w.size == 0:
        return w
    w = np.abs(w)
    peak = float(np.max(w))
    if peak > 0:
        w /= peak
    return w


def write_wav_int16(path: Path, samples: np.ndarray, sample_rate: int) -> float:
    """Write mono int16 WAV; return peak before normalization."""
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(samples, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        raise ValueError("WAV samples contain NaN or Inf")
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 0:
        x = x * (0.95 / peak)
    pcm = np.clip(x * 32767.0, -32767, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())
    return peak


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
    """
    Synthesize one note with fixed excitation and ROM/FEM modal body filtering.

    Returns metadata dict (also written to ``output_metadata_json`` when set).
    """
    all_modes, parse_defaults = parse_modal_modes(modal_data)
    band_modes = _modes_in_modal_band(all_modes)
    defaults_used: List[str] = list(parse_defaults)
    flags = {
        "bridge_weighting_used": False,
        "mic_proxy_used": False,
        "radiation_proxy_used": False,
        "participation_used": False,
        "q_or_damping_used": False,
    }

    dry = synthesize_plucked_string(
        frequency_hz,
        duration_s,
        sample_rate,
        pluck_position=FIXED_PLUCK_POSITION,
        velocity=velocity,
    )
    acc = _string_acceleration(dry)

    wet = np.zeros_like(dry)
    weights: List[float] = []
    q_values: List[float] = []
    for mode in band_modes:
        f_hz = float(mode["frequency_hz"])
        w = compute_mode_weight(mode, defaults_used=defaults_used, flags=flags)
        q_wood = estimate_mode_q(mode, f_hz, defaults_used)
        q_total = _total_q_with_radiation_loss(q_wood, f_hz, FIXED_RAD_K)
        flags["q_or_damping_used"] = True
        weights.append(w)
        q_values.append(q_total)
        b0, b1, b2, a1, a2 = _biquad_bandpass_coeffs(f_hz, q_total, float(sample_rate))
        wet += w * _biquad_process(acc, b0, b1, b2, a1, a2)

    if weights:
        scale = 1.0 / math.sqrt(len(weights))
        wet *= scale * FIXED_WET_GAIN

    string_mix, hf_fallback = _hf_string_mix(float(frequency_hz))
    body_mix = 1.0 - string_mix
    if not band_modes:
        defaults_used.append("no_modes_in_60_550_band:body_bypass")
        body_mix = 0.0
        string_mix = 1.0

    mixed = string_mix * dry + body_mix * wet
    peak = write_wav_int16(Path(output_wav), mixed, sample_rate)

    used_freqs = [float(m["frequency_hz"]) for m in band_modes]
    metadata: Dict[str, Any] = {
        "note_name": note_name,
        "frequency_hz": float(frequency_hz),
        "duration_s": float(duration_s),
        "sample_rate": int(sample_rate),
        "modal_mode_count_available": len(all_modes),
        "modal_mode_count_used": len(band_modes),
        "modal_frequency_min_hz": min(used_freqs) if used_freqs else None,
        "modal_frequency_max_hz": max(used_freqs) if used_freqs else None,
        "full_modal_band_hz": list(FULL_MODAL_BAND_HZ),
        "high_frequency_fallback_used": bool(hf_fallback),
        "bridge_weighting_used": flags["bridge_weighting_used"],
        "mic_proxy_used": flags["mic_proxy_used"],
        "radiation_proxy_used": flags["radiation_proxy_used"],
        "q_or_damping_used": flags["q_or_damping_used"],
        "defaults_used": sorted(set(defaults_used)),
        "excitation": {
            "pluck_position": FIXED_PLUCK_POSITION,
            "velocity": float(velocity),
            "string_mix": string_mix,
            "wet_gain": FIXED_WET_GAIN,
            "rad_k": FIXED_RAD_K,
        },
        "output_wav": str(output_wav),
        "peak_before_normalize": peak,
        "samples_finite": True,
    }
    if output_metadata_json is not None:
        output_metadata_json.parent.mkdir(parents=True, exist_ok=True)
        output_metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        metadata["output_metadata_json"] = str(output_metadata_json)
    return metadata


def synthetic_classic_body_modes() -> List[Dict[str, Any]]:
    """Deterministic test fixture resembling a classic body modal catalog."""
    freqs = [82.4, 118.5, 156.2, 198.7, 245.1, 288.4, 332.0, 378.5, 421.0, 467.2, 512.8]
    modes: List[Dict[str, Any]] = []
    for i, f in enumerate(freqs):
        modes.append(
            {
                "frequency_hz": f,
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
    """CLI smoke: E2, A4, E5 with synthetic or --modal-json ROM body file."""
    import argparse

    parser = argparse.ArgumentParser(description="Stage-1 body-response note smoke test")
    parser.add_argument("--modal-json", type=Path, default=None, help="ROM STK body JSON path")
    parser.add_argument("--out-dir", type=Path, default=Path("audio/stage1"))
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_S)
    args = parser.parse_args()

    if args.modal_json and args.modal_json.is_file():
        modal_data = load_modal_data_from_path(args.modal_json)
    else:
        modal_data = {"predicted_modes": synthetic_classic_body_modes(), "analysis": "synthetic_fixture"}

    cases = (
        ("E2", 82.41),
        ("A4", 440.0),
        ("E5", 659.25),
    )
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
        print(f"{name}: {wav_path} ({meta['modal_mode_count_used']} modes, hf_fallback={meta['high_frequency_fallback_used']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
