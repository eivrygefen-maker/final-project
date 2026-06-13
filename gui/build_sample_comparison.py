#!/usr/bin/env python3
"""
Build sequential comparison WAVs: N guitars × one note per file (no FEM).

Uses M4 modal surrogate + body-response synthesis when available;
falls back to deterministic synthetic modes per sample for offline tests.
"""
from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gui"))

from body_response_synth import (  # noqa: E402
    DEFAULT_SAMPLE_RATE,
    concatenate_audio_with_crossfade,
    read_wav_float_mono,
    synthesize_note_with_body_response,
    synthetic_classic_body_modes,
)
from diagnostic_synthesis import (  # noqa: E402
    _spectral_features,
    average_spectral_similarity,
    flatten_geometry_parameters,
    get_diagnostic_mode,
    list_diagnostic_modes,
    summarize_diagnostic_mode,
)
from sample_parameters import normalize_sample_parameters  # noqa: E402
from synthesis_presets import DEFAULT_SYNTHESIS_PRESET  # noqa: E402

COMPARISON_NOTES: Tuple[Tuple[str, float], ...] = (
    ("E2", 82.41),
    ("E3", 164.81),
    ("A2", 110.0),
    ("A3", 220.0),
    ("A4", 440.0),
    ("D4", 293.66),
    ("E5", 659.25),
)

NOTE_NAME_TO_HZ: Dict[str, float] = {name: hz for name, hz in COMPARISON_NOTES}


def parse_notes_arg(notes_csv: Optional[str]) -> Tuple[Tuple[str, float], ...]:
    if not notes_csv:
        return COMPARISON_NOTES
    out: List[Tuple[str, float]] = []
    for token in str(notes_csv).split(","):
        name = token.strip().upper()
        if not name:
            continue
        if name not in NOTE_NAME_TO_HZ:
            raise ValueError(f"unknown note: {name!r}; use {list(NOTE_NAME_TO_HZ)}")
        out.append((name, NOTE_NAME_TO_HZ[name]))
    if not out:
        raise ValueError("notes list is empty")
    return tuple(out)


def load_lhs_sample_entries(repo_root: Path, *, max_samples: int = 26) -> List[Dict[str, Any]]:
    pool_path = repo_root / "ROM" / "classic" / "lhs_pool.json"
    if not pool_path.is_file():
        return []
    doc = json.loads(pool_path.read_text(encoding="utf-8"))
    entries = list(doc.get("entries") or [])
    rows: List[Dict[str, Any]] = []
    for entry in entries:
        sid = str(entry.get("id") or "")
        if not sid.startswith("sample_"):
            continue
        params = dict(entry.get("parameters") or {})
        if not params:
            continue
        rows.append(
            {
                "sample_id": sid,
                "run_id": str(entry.get("last_run_id") or ""),
                "parameters": params,
            }
        )
        if len(rows) >= int(max_samples):
            break
    rows.sort(key=lambda r: str(r.get("sample_id") or ""))
    return rows


def _sample_index(sample_id: str) -> int:
    try:
        return int(str(sample_id).split("_")[-1])
    except ValueError:
        return 0


def synthetic_modal_for_sample(sample_id: str, *, n_modes: int = 55) -> Dict[str, Any]:
    """Deterministic offline fixture — frequency scale shifts per sample."""
    idx = _sample_index(sample_id)
    scale = 0.94 + 0.12 * (idx / max(1, 25))
    modes = synthetic_classic_body_modes(n_modes)
    for m in modes:
        m["frequency_hz"] = round(float(m["frequency_hz"]) * scale, 4)
    return {"predicted_modes": modes, "analysis": f"synthetic_fixture_{sample_id}"}


def modal_data_from_prediction(prediction: Mapping[str, Any]) -> Dict[str, Any]:
    freqs = [float(f) for f in (prediction.get("frequencies_hz") or [])]
    weights = [1.0 / (1.0 + 0.25 * i) for i in range(len(freqs))]
    doc: Dict[str, Any] = {
        "analysis": "rom_online_body",
        "modes_hz": freqs,
        "mode_weights": weights,
        "num_modes": len(freqs),
        "full_modal_band_hz": [60.0, 550.0],
        "frequencies_hz": freqs,
    }
    predicted = list(prediction.get("predicted_modes") or [])
    if predicted:
        doc["predicted_modes"] = predicted
    return doc


def m4_surrogate_model_available(repo_root: Path, shape_name: str = "classic") -> bool:
    m4_scripts = repo_root / "FEM/experiments/active_domain_validation/physics_integrity/scripts"
    if not m4_scripts.is_dir():
        return False
    if str(m4_scripts) not in sys.path:
        sys.path.insert(0, str(m4_scripts))
    try:
        from v2_b3_m4_modal_surrogate_lib import surrogate_is_available  # noqa: WPS433

        return bool(surrogate_is_available(repo_root, shape_name))
    except Exception:
        return False


def predict_modal_for_parameters(repo_root: Path, parameters: Mapping[str, Any]) -> Dict[str, Any]:
    if not m4_surrogate_model_available(repo_root):
        return {}
    m4_scripts = repo_root / "FEM/experiments/active_domain_validation/physics_integrity/scripts"
    if str(m4_scripts) not in sys.path:
        sys.path.insert(0, str(m4_scripts))
    try:
        from v2_b3_m4_modal_surrogate_lib import load_surrogate_model, predict_modal_catalog  # noqa: WPS433

        model = load_surrogate_model(repo_root, "classic")
        return dict(predict_modal_catalog(model, dict(parameters), nev=0))
    except Exception:
        return {}


def resolve_modal_data_for_sample(
    repo_root: Path,
    sample: Mapping[str, Any],
    *,
    use_surrogate: bool,
) -> Tuple[Dict[str, Any], str]:
    sid = str(sample["sample_id"])
    if use_surrogate:
        prediction = predict_modal_for_parameters(repo_root, sample.get("parameters") or {})
        if prediction.get("frequencies_hz") or prediction.get("predicted_modes"):
            return modal_data_from_prediction(prediction), "m4_surrogate"
    return synthetic_modal_for_sample(sid), "synthetic_fallback"


def write_wav_mono(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.clip(np.asarray(samples, dtype=np.float64), -1.0, 1.0)
    pcm = (x * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())


def _median_from_per_mode(meta: Mapping[str, Any], key: str) -> Optional[float]:
    vals = [float(r.get(key)) for r in (meta.get("per_mode_damping") or []) if r.get(key) is not None]
    if not vals:
        return None
    vals.sort()
    return round(vals[len(vals) // 2], 6)


def segment_metadata_from_synthesis(
    meta: Mapping[str, Any],
    *,
    sample: Mapping[str, Any],
    note_name: str,
    frequency_hz: float,
    seg_start: float,
    seg_dur: float,
    seg_i: int,
    audio: np.ndarray,
    sample_rate: int,
    diagnostic_mode: Optional[str],
) -> Dict[str, Any]:
    params = normalize_sample_parameters(sample.get("parameters"))
    dqs = dict(meta.get("damping_q_summary") or {})
    spec = _spectral_features(audio, sample_rate)
    row: Dict[str, Any] = {
        "segment_number": seg_i + 1,
        "sample_id": str(sample["sample_id"]),
        "run_id": sample.get("run_id"),
        "note": note_name,
        "frequency_hz": frequency_hz,
        "start_time_s": round(seg_start, 4),
        "duration_s": round(seg_dur, 4),
        "parameters": params,
        "diagnostic_mode": diagnostic_mode or meta.get("diagnostic_mode") or "baseline_current",
        "raw_body_rms_before_normalization": meta.get("raw_body_rms_before_normalization"),
        "final_rms_dbfs": meta.get("final_rms_dbfs"),
        "body_gain_applied": meta.get("body_gain_applied"),
        "string_gain_applied": meta.get("string_gain_applied"),
        "body_to_string_ratio": meta.get("body_to_string_rms_ratio_before_loudness"),
        "broad_signature_band_gains": meta.get("broad_signature_band_gains") or {},
        "damping_q_summary": dqs,
        "mode_q_median": dqs.get("mode_q_median"),
        "mode_q_spread_within_sample": dqs.get("mode_q_spread"),
        "material_damping_median": dqs.get("material_damping_median"),
        "sample_material_damping_fingerprint": meta.get("sample_material_damping_fingerprint"),
        "sample_mode_q_fingerprint": meta.get("sample_mode_q_fingerprint"),
        "material_damping_spread_within_sample": dqs.get("material_damping_spread"),
        "modal_source": meta.get("modal_source"),
        "per_mode_q_used_in_frequency_response": meta.get("per_mode_q_used_in_frequency_response"),
        "per_mode_tau_used_in_time_decay": meta.get("per_mode_tau_used_in_time_decay"),
        "far_mode_weights_sample_specific": meta.get("far_mode_weights_sample_specific"),
        "far_mode_sample_specificity_score": meta.get("far_mode_sample_specificity_score"),
        "mode_amplitude_factor_median": _median_from_per_mode(meta, "mode_amplitude_factor"),
        "mode_radiation_factor_median": _median_from_per_mode(meta, "mode_radiation_factor"),
        "material_amplitude_factor_median": _median_from_per_mode(meta, "material_amplitude_factor"),
        "radiation_color_v1_active": meta.get("radiation_color_v1_active"),
        "modal_radiation_color_v2_active": meta.get("modal_radiation_color_v2_active"),
        "mode_bridge_gate_factor_median": _median_from_per_mode(meta, "mode_bridge_gate_factor"),
        "mode_output_transmittance_factor_median": _median_from_per_mode(meta, "mode_output_transmittance_factor"),
        "mode_final_amplitude_factor_median": _median_from_per_mode(meta, "mode_final_amplitude_factor"),
        "low_body_color_strength": meta.get("low_body_color_strength"),
        "raw_body_rms_before_any_gain": meta.get("raw_body_rms_before_any_gain"),
        "raw_body_rms_after_modal_weighting": meta.get("raw_body_rms_after_modal_weighting"),
        "body_to_string_ratio_before_normalization": meta.get("body_to_string_ratio_before_normalization"),
        "body_to_string_ratio_after_normalization": meta.get("body_to_string_ratio_after_normalization"),
        "normalization_preserved_variation_score": meta.get("normalization_preserved_variation_score"),
        "bridge_proxy_missing_count": meta.get("bridge_proxy_missing_count"),
        "radiation_proxy_missing_count": meta.get("radiation_proxy_missing_count"),
        "mic_proxy_missing_count": meta.get("mic_proxy_missing_count"),
        "note_reward_score": meta.get("note_reward_score"),
        "output_decay_slope_db_per_s": meta.get("output_decay_slope_db_per_s"),
        "near_modal_energy_fraction": meta.get("near_modal_energy_fraction"),
        "broad_body_energy_fraction": meta.get("broad_body_energy_fraction"),
        "mid_modal_energy_fraction": meta.get("mid_modal_energy_fraction"),
        "body_gain_normalization_strength": meta.get("body_gain_normalization_strength"),
        "final_loudness_normalization_strength": meta.get("final_loudness_normalization_strength"),
        "spectral_centroid_hz": round(spec["centroid_hz"], 4),
        "spectral_low_energy": round(spec["low_energy"], 6),
        "spectral_mid_energy": round(spec["mid_energy"], 6),
        "spectral_high_energy": round(spec["high_energy"], 6),
    }
    row.update(flatten_geometry_parameters(params))
    return row


def build_comparison_for_note(
    *,
    note_name: str,
    frequency_hz: float,
    samples: Sequence[Mapping[str, Any]],
    repo_root: Path,
    out_wav: Path,
    duration_s: float,
    silence_s: float,
    sample_rate: int,
    use_surrogate: bool,
    synthesis_preset: str,
    diagnostic_mode: Optional[str] = None,
) -> Dict[str, Any]:
    segments: List[np.ndarray] = []
    segment_audios: List[np.ndarray] = []
    segment_rows: List[Dict[str, Any]] = []
    cursor_s = 0.0

    for seg_i, sample in enumerate(samples):
        sid = str(sample["sample_id"])
        modal_data, modal_source = resolve_modal_data_for_sample(
            repo_root, sample, use_surrogate=use_surrogate
        )
        tmp_wav = out_wav.parent / "_tmp" / f"{note_name}_{sid}.wav"
        meta = synthesize_note_with_body_response(
            frequency_hz=frequency_hz,
            note_name=note_name,
            duration_s=duration_s,
            sample_rate=sample_rate,
            modal_data=modal_data,
            output_wav=tmp_wav,
            synthesis_preset=synthesis_preset,
            diagnostic_mode=diagnostic_mode,
            sample_parameters=normalize_sample_parameters(sample.get("parameters")),
            modal_source=modal_source,
        )
        audio, sr = read_wav_float_mono(tmp_wav)
        if sr != sample_rate:
            raise ValueError(f"sample rate mismatch {sr} != {sample_rate}")
        if seg_i > 0 and silence_s > 0:
            segments.append(np.zeros(int(silence_s * sample_rate), dtype=np.float64))
            cursor_s += silence_s
        seg_start = cursor_s
        segments.append(audio)
        segment_audios.append(audio)
        seg_dur = len(audio) / float(sample_rate)
        segment_rows.append(
            segment_metadata_from_synthesis(
                meta,
                sample=sample,
                note_name=note_name,
                frequency_hz=frequency_hz,
                seg_start=seg_start,
                seg_dur=seg_dur,
                seg_i=seg_i,
                audio=audio,
                sample_rate=sample_rate,
                diagnostic_mode=diagnostic_mode,
            )
        )
        cursor_s += seg_dur

    mixed = concatenate_audio_with_crossfade(segments, sample_rate=sample_rate, crossfade_ms=8.0, silence_ms=0.0)
    write_wav_mono(out_wav, mixed, sample_rate)
    return {
        "note": note_name,
        "frequency_hz": frequency_hz,
        "wav": str(out_wav.name),
        "total_duration_s": round(len(mixed) / float(sample_rate), 4),
        "segment_count": len(segment_rows),
        "segments": segment_rows,
        "average_spectral_similarity": average_spectral_similarity(segment_audios),
    }


def build_sample_comparisons(
    *,
    repo_root: Path,
    out_dir: Path,
    samples: Sequence[Mapping[str, Any]],
    notes: Sequence[Tuple[str, float]] = COMPARISON_NOTES,
    duration_s: float = 2.0,
    silence_s: float = 0.35,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    use_surrogate: bool = True,
    synthesis_preset: str = DEFAULT_SYNTHESIS_PRESET,
    diagnostic_mode: Optional[str] = None,
    write_mode_summary: bool = True,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if diagnostic_mode:
        get_diagnostic_mode(diagnostic_mode)
    note_rows: List[Dict[str, Any]] = []
    for note_name, hz in notes:
        out_wav = out_dir / f"{note_name}_26_guitars.wav"
        row = build_comparison_for_note(
            note_name=note_name,
            frequency_hz=hz,
            samples=samples,
            repo_root=repo_root,
            out_wav=out_wav,
            duration_s=duration_s,
            silence_s=silence_s,
            sample_rate=sample_rate,
            use_surrogate=use_surrogate,
            synthesis_preset=synthesis_preset,
            diagnostic_mode=diagnostic_mode,
        )
        note_rows.append(row)

    mode_name = diagnostic_mode or "baseline_current"
    manifest = {
        "schema_version": "sample_comparison_v2",
        "sample_count": len(samples),
        "sample_ids": [str(s["sample_id"]) for s in samples],
        "synthesis_preset": synthesis_preset,
        "diagnostic_mode": mode_name,
        "silence_between_guitars_s": silence_s,
        "note_duration_s": duration_s,
        "sample_rate": sample_rate,
        "use_surrogate": use_surrogate,
        "notes": note_rows,
    }
    (out_dir / "comparison_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if write_mode_summary:
        summary = summarize_diagnostic_mode(note_rows, diagnostic_mode=mode_name)
        (out_dir / "mode_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        manifest["mode_summary"] = summary
    return manifest


def parse_diagnostic_modes_arg(value: str) -> List[str]:
    modes = [m.strip() for m in str(value or "").split(",") if m.strip()]
    for name in modes:
        get_diagnostic_mode(name)
    return modes


def main() -> int:
    parser = argparse.ArgumentParser(description="26-guitar note comparison WAV builder")
    parser.add_argument("--repo-root", type=Path, default=REPO)
    parser.add_argument("--out-dir", type=Path, default=REPO / "audio" / "comparison_26_samples")
    parser.add_argument("--max-samples", type=int, default=26)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--silence", type=float, default=0.35)
    parser.add_argument("--no-surrogate", action="store_true")
    parser.add_argument("--synthesis-preset", type=str, default=DEFAULT_SYNTHESIS_PRESET)
    parser.add_argument(
        "--diagnostic-mode",
        type=str,
        default=None,
        help="Single diagnostic synthesis mode (see --diagnostic-modes).",
    )
    parser.add_argument(
        "--diagnostic-modes",
        type=str,
        default=None,
        help="Comma-separated modes; writes one subfolder per mode under --out-dir.",
    )
    args = parser.parse_args()

    samples = load_lhs_sample_entries(args.repo_root, max_samples=args.max_samples)
    if not samples:
        samples = [
            {"sample_id": f"sample_{i:03d}", "run_id": "", "parameters": {}}
            for i in range(args.max_samples)
        ]

    if args.diagnostic_modes:
        modes = parse_diagnostic_modes_arg(args.diagnostic_modes)
        for mode in modes:
            mode_dir = args.out_dir / mode
            build_sample_comparisons(
                repo_root=args.repo_root,
                out_dir=mode_dir,
                samples=samples,
                duration_s=args.duration,
                silence_s=args.silence,
                use_surrogate=not args.no_surrogate,
                synthesis_preset=args.synthesis_preset,
                diagnostic_mode=mode,
            )
            print(f"Wrote mode {mode} -> {mode_dir}")
        print(f"Wrote {len(modes)} diagnostic mode folders under {args.out_dir}")
        return 0

    manifest = build_sample_comparisons(
        repo_root=args.repo_root,
        out_dir=args.out_dir,
        samples=samples,
        duration_s=args.duration,
        silence_s=args.silence,
        use_surrogate=not args.no_surrogate,
        synthesis_preset=args.synthesis_preset,
        diagnostic_mode=args.diagnostic_mode,
    )
    print(f"Wrote {len(manifest['notes'])} comparison WAVs to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
