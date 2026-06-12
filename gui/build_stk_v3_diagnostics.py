#!/usr/bin/env python3
"""
Stage 4.9 — STK V3 body-signature diagnostics (M4 surrogate on VM, no FEM).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gui"))

from body_response_synth import (  # noqa: E402
    DEFAULT_SAMPLE_RATE,
    concatenate_audio_with_crossfade,
    synthesize_note_with_body_response,
)
from build_sample_comparison import (  # noqa: E402
    load_lhs_sample_entries,
    m4_surrogate_model_available,
    parse_notes_arg,
    read_wav_float_mono,
    resolve_modal_data_for_sample,
    segment_metadata_from_synthesis,
    write_wav_mono,
)
from diagnostic_synthesis import (  # noqa: E402
    average_spectral_similarity,
    summarize_diagnostic_mode,
)
from sample_parameters import normalize_sample_parameters  # noqa: E402
from stage49_stk_v3_report import build_stage49_report  # noqa: E402
from synthesis_presets import DEFAULT_SYNTHESIS_PRESET  # noqa: E402

DEFAULT_OUT = REPO / "audio" / "stk_v3_body_signature_diagnostics"
DEFAULT_MODES = (
    "baseline_current,modal_radiation_color_v1,modal_body_signature_v3_core,"
    "modal_body_signature_v3_low_f0_imprint_only,modal_body_signature_v3_mobility_only,"
    "modal_body_signature_v3_far_color_only,modal_body_signature_v3_full"
)
DECOMP_LAYERS = (
    "string_only",
    "body_only_v1",
    "body_only_v3_full",
    "body_signature_envelope_only",
    "low_f0_imprint_only",
    "mobility_weighted_body_only",
    "full_mix_baseline",
    "full_mix_radiation_v1",
    "full_mix_v3_full",
)


def _synthesize_layer(
    *,
    layer: str,
    frequency_hz: float,
    note_name: str,
    duration_s: float,
    sample_rate: int,
    modal_data: Mapping[str, Any],
    sample_parameters: Mapping[str, Any],
    modal_source: str,
    band_modes: Sequence[Mapping[str, Any]],
    string_excitation: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    from body_signature_v3 import (
        apply_harmonic_body_imprint,
        build_body_signature_envelope,
        imprint_only_layer,
        low_f0_imprint_strength,
    )
    from timbre_decomposition import compute_note_layers

    tmp = Path("_tmp_layer.wav")
    if layer == "string_only":
        result = compute_note_layers(
            frequency_hz,
            note_name,
            duration_s,
            sample_rate,
            modal_data,
            sample_parameters=sample_parameters,
            use_bridge_mobility_proxy=False,
            modal_source=modal_source,
        )
        return result["layers"]["string_only"], result["metadata"]
    if layer == "body_only_v1":
        meta = synthesize_note_with_body_response(
            frequency_hz=frequency_hz,
            note_name=note_name,
            duration_s=duration_s,
            sample_rate=sample_rate,
            modal_data=modal_data,
            output_wav=tmp,
            diagnostic_mode="modal_radiation_color_v1",
            sample_parameters=sample_parameters,
            modal_source=modal_source,
        )
        audio, _ = read_wav_float_mono(tmp)
        return audio, meta
    if layer == "body_only_v3_full":
        meta = synthesize_note_with_body_response(
            frequency_hz=frequency_hz,
            note_name=note_name,
            duration_s=duration_s,
            sample_rate=sample_rate,
            modal_data=modal_data,
            output_wav=tmp,
            diagnostic_mode="modal_body_signature_v3_full",
            sample_parameters=sample_parameters,
            modal_source=modal_source,
        )
        audio, _ = read_wav_float_mono(tmp)
        return audio, meta
    if layer == "body_signature_envelope_only":
        n = int(duration_s * sample_rate)
        env_spec = build_body_signature_envelope(band_modes, frequency_hz, n, sample_rate)
        env_time = np.fft.irfft(env_spec, n=n)
        t = np.arange(n, dtype=np.float64) / float(sample_rate)
        carrier = np.sin(2.0 * np.pi * frequency_hz * t)
        scale = 0.15 / max(float(np.max(np.abs(env_time))), 1e-9)
        audio = scale * env_time * carrier
        return audio, {"body_signature_envelope_only": True}
    if layer == "low_f0_imprint_only":
        result = compute_note_layers(
            frequency_hz,
            note_name,
            duration_s,
            sample_rate,
            modal_data,
            sample_parameters=sample_parameters,
            modal_source=modal_source,
        )
        string_sig = result["layers"]["string_only"]
        strength = low_f0_imprint_strength(frequency_hz)
        imprint = imprint_only_layer(string_sig, sample_rate, frequency_hz, band_modes, strength=strength)
        return imprint, {"low_f0_imprint_strength": strength}
    if layer == "mobility_weighted_body_only":
        from body_response_synth import (
            _string_acceleration,
            modes_in_validated_band,
            parse_modal_modes,
            synthesize_body_via_transfer_function,
            synthesize_plucked_string,
            FIXED_PLUCK_POSITION,
            use_diagnostic_mode,
        )

        with use_diagnostic_mode("modal_body_signature_v3_mobility_only", sample_parameters=sample_parameters):
            all_modes, defaults = parse_modal_modes(modal_data)
            modes = modes_in_validated_band(all_modes)
            exc = synthesize_plucked_string(
                frequency_hz, duration_s, sample_rate, pluck_position=FIXED_PLUCK_POSITION
            )
            acc = _string_acceleration(exc)
            body, _, _, info = synthesize_body_via_transfer_function(
                acc,
                sample_rate,
                modes,
                defaults_used=list(defaults),
                flags={},
                note_hz=frequency_hz,
                harmonics_hz=[frequency_hz * k for k in range(1, 8)],
                use_bridge_mobility_proxy=True,
            )
        return body, info
    mix_modes = {
        "full_mix_baseline": "baseline_current",
        "full_mix_radiation_v1": "modal_radiation_color_v1",
        "full_mix_v3_full": "modal_body_signature_v3_full",
    }
    if layer in mix_modes:
        meta = synthesize_note_with_body_response(
            frequency_hz=frequency_hz,
            note_name=note_name,
            duration_s=duration_s,
            sample_rate=sample_rate,
            modal_data=modal_data,
            output_wav=tmp,
            diagnostic_mode=mix_modes[layer],
            sample_parameters=sample_parameters,
            modal_source=modal_source,
        )
        audio, _ = read_wav_float_mono(tmp)
        return audio, meta
    raise ValueError(f"unknown layer: {layer}")


def build_v3_diagnostics(
    *,
    repo_root: Path,
    out_dir: Path,
    notes: Sequence[Tuple[str, float]],
    modes: Sequence[str],
    max_samples: int = 10,
    duration_s: float = 1.2,
    silence_s: float = 0.2,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    use_surrogate: bool = True,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = load_lhs_sample_entries(repo_root, max_samples=max_samples)
    if not samples:
        raise RuntimeError("no LHS samples")

    mode_summaries: Dict[str, Any] = {}
    for mode in modes:
        mode_dir = out_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        note_rows: List[Dict[str, Any]] = []
        for note_name, frequency_hz in notes:
            segments: List[np.ndarray] = []
            seg_rows: List[Dict[str, Any]] = []
            for sample in samples:
                sid = str(sample["sample_id"])
                modal_data, modal_source = resolve_modal_data_for_sample(
                    repo_root, sample, use_surrogate=use_surrogate
                )
                wav = mode_dir / f"{note_name}_{sid}.wav"
                meta = synthesize_note_with_body_response(
                    frequency_hz=frequency_hz,
                    note_name=note_name,
                    duration_s=duration_s,
                    sample_rate=sample_rate,
                    modal_data=modal_data,
                    output_wav=wav,
                    diagnostic_mode=mode,
                    sample_parameters=normalize_sample_parameters(sample.get("parameters")),
                    modal_source=modal_source,
                    synthesis_preset=DEFAULT_SYNTHESIS_PRESET,
                )
                audio, _ = read_wav_float_mono(wav)
                segments.append(audio)
                seg_rows.append(
                    segment_metadata_from_synthesis(
                        meta,
                        sample=sample,
                        note_name=note_name,
                        frequency_hz=frequency_hz,
                        seg_start=0.0,
                        seg_dur=len(audio) / float(sample_rate),
                        seg_i=len(seg_rows),
                        audio=audio,
                        sample_rate=sample_rate,
                        diagnostic_mode=mode,
                    )
                )
            stitched = out_dir / f"{note_name}_{mode}_10_guitars.wav"
            write_wav_mono(
                stitched,
                concatenate_audio_with_crossfade(segments, sample_rate=sample_rate, silence_ms=silence_s * 1000),
                sample_rate,
            )
            note_rows.append(
                {
                    "note": note_name,
                    "segments": seg_rows,
                    "average_spectral_similarity": average_spectral_similarity(segments),
                }
            )
        mode_summaries[mode] = summarize_diagnostic_mode(note_rows, diagnostic_mode=mode)

    decomp_dir = out_dir / "timbre_decomposition"
    decomp_dir.mkdir(parents=True, exist_ok=True)
    rep_sample = samples[len(samples) // 2]
    for note_name, frequency_hz in notes:
        modal_data, modal_source = resolve_modal_data_for_sample(
            repo_root, rep_sample, use_surrogate=use_surrogate
        )
        from body_response_synth import modes_in_validated_band, parse_modal_modes

        band_modes = modes_in_validated_band(parse_modal_modes(modal_data)[0])
        sample_dir = decomp_dir / note_name / str(rep_sample["sample_id"])
        sample_dir.mkdir(parents=True, exist_ok=True)
        for layer in DECOMP_LAYERS:
            audio, meta = _synthesize_layer(
                layer=layer,
                frequency_hz=frequency_hz,
                note_name=note_name,
                duration_s=duration_s,
                sample_rate=sample_rate,
                modal_data=modal_data,
                sample_parameters=normalize_sample_parameters(rep_sample.get("parameters")),
                modal_source=modal_source,
                band_modes=band_modes,
            )
            write_wav_mono(sample_dir / f"{layer}.wav", audio, sample_rate)
            (sample_dir / f"{layer}_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    manifest = {
        "stage": "4.9",
        "out_dir": str(out_dir),
        "sample_ids": [str(s["sample_id"]) for s in samples],
        "notes": [n for n, _ in notes],
        "modes": list(modes),
        "use_surrogate": use_surrogate,
        "fem_launched": False,
    }
    (out_dir / "build_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report_dir = repo_root / "audio" / "debug_reports"
    build_stage49_report(
        mode_summaries=mode_summaries,
        notes=[n for n, _ in notes],
        modes=modes,
        build_manifest=manifest,
        out_json=report_dir / "stage49_stk_v3_report.json",
        out_md=report_dir / "stage49_stk_v3_report.md",
    )
    return {**manifest, "mode_summaries": mode_summaries}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 4.9 STK V3 diagnostics")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--notes", type=str, default="A2,A4,E5")
    parser.add_argument("--modes", type=str, default=DEFAULT_MODES)
    parser.add_argument("--duration", type=float, default=1.2)
    parser.add_argument("--silence", type=float, default=0.2)
    parser.add_argument("--no-surrogate", action="store_true")
    args = parser.parse_args(argv)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    notes = parse_notes_arg(args.notes)
    use_surrogate = not args.no_surrogate
    if use_surrogate and not m4_surrogate_model_available(REPO):
        raise RuntimeError("M4 surrogate missing — use --no-surrogate for offline tests")
    manifest = build_v3_diagnostics(
        repo_root=REPO,
        out_dir=args.out_dir,
        notes=notes,
        modes=modes,
        max_samples=args.max_samples,
        duration_s=args.duration,
        silence_s=args.silence,
        use_surrogate=use_surrogate,
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "mode_summaries"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
