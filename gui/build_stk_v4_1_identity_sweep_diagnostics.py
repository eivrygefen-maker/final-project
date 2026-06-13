#!/usr/bin/env python3
"""Stage 5.1D STK V4.1 identity strength sweep diagnostics (A3 default)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gui"))

from body_hybrid_v4_1_identity_space import (  # noqa: E402
    audio_timbre_vector,
    build_body_identity_vector,
    compare_audio_to_reference,
    distance_consistency_report,
    is_v4_1_identity_space_mode,
)
from body_response_synth import DEFAULT_SAMPLE_RATE, concatenate_audio_with_crossfade, synthesize_note_with_body_response  # noqa: E402
from build_sample_comparison import (  # noqa: E402
    load_lhs_sample_entries,
    m4_surrogate_model_available,
    parse_notes_arg,
    read_wav_float_mono,
    resolve_modal_data_for_sample,
    segment_metadata_from_synthesis,
    write_wav_mono,
)
from diagnostic_synthesis import average_spectral_similarity, summarize_diagnostic_mode  # noqa: E402
from sample_parameters import normalize_sample_parameters  # noqa: E402
from stage48_timbre_decomposition_report import _attack_time_ms, _spectral_flux  # noqa: E402
from stage51d_stk_v4_1_identity_sweep_report import build_stage51d_identity_sweep_report  # noqa: E402
from synthesis_presets import DEFAULT_SYNTHESIS_PRESET  # noqa: E402

DEFAULT_OUT = REPO / "audio" / "stk_v4_1_identity_sweep_diagnostics"
DEFAULT_NOTES = "A3"
V41_MODE = "modal_body_hybrid_v4_1_full"
DEFAULT_MODES = (
    "modal_body_hybrid_v4_1_full,"
    "modal_body_hybrid_v4_1_identity_light,"
    "modal_body_hybrid_v4_1_identity_medium,"
    "modal_body_hybrid_v4_1_identity_strong"
)


def _segment_row(
    meta: Mapping[str, Any],
    *,
    sample: Mapping[str, Any],
    note_name: str,
    frequency_hz: float,
    seg_i: int,
    audio: np.ndarray,
    sample_rate: int,
    diagnostic_mode: str,
    z_body: Optional[Mapping[str, Any]] = None,
    vs_v41: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    row = segment_metadata_from_synthesis(
        meta,
        sample=sample,
        note_name=note_name,
        frequency_hz=frequency_hz,
        seg_start=0.0,
        seg_dur=len(audio) / float(sample_rate),
        seg_i=seg_i,
        audio=audio,
        sample_rate=sample_rate,
        diagnostic_mode=diagnostic_mode,
    )
    peak = meta.get("output_peak_dbfs")
    if peak is None:
        pk = float(np.max(np.abs(audio))) if len(audio) else 0.0
        peak = 20.0 * float(np.log10(max(pk, 1e-12)))
    row["output_peak_dbfs"] = peak
    row["clipping_detected"] = float(peak) > -0.5
    row["spectral_flux"] = _spectral_flux(audio, sample_rate)
    row["attack_time_ms"] = _attack_time_ms(audio, sample_rate)
    row["timbre_vector"] = audio_timbre_vector(audio, sample_rate=sample_rate, segment_meta=row)
    if z_body is not None:
        row["body_identity_vector"] = z_body
    if vs_v41 is not None:
        row["vs_v41_reference"] = vs_v41
    if meta.get("identity_vs_v41_reference"):
        row["vs_v41_reference"] = meta["identity_vs_v41_reference"]
    if meta.get("perceptual_axes"):
        row["perceptual_axes"] = meta["perceptual_axes"]
    if meta.get("identity_strength_profile"):
        row["identity_strength_profile"] = meta["identity_strength_profile"]
    return row


def build_v4_1_identity_sweep_diagnostics(
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
    distance_samples_by_mode: Dict[str, List[Dict[str, Any]]] = {m: [] for m in modes}
    stitched: List[str] = []
    v41_audio_cache: Dict[Tuple[str, str], np.ndarray] = {}

    for mode in modes:
        mode_dir = out_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        note_rows: List[Dict[str, Any]] = []
        for note_name, frequency_hz in notes:
            segments: List[np.ndarray] = []
            seg_rows: List[Dict[str, Any]] = []
            for sample in samples:
                sid = str(sample["sample_id"])
                params = normalize_sample_parameters(sample.get("parameters"))
                params = {**params, "sample_id": sid}
                modal_data, modal_source = resolve_modal_data_for_sample(
                    repo_root, sample, use_surrogate=use_surrogate
                )
                z_body = build_body_identity_vector(
                    parameters=params,
                    modal_data=modal_data,
                    frequency_hz=frequency_hz,
                    repo_root=repo_root,
                    sample_id=sid,
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
                    sample_parameters=params,
                    modal_source=modal_source,
                    synthesis_preset=DEFAULT_SYNTHESIS_PRESET,
                    repo_root=repo_root,
                    sample_id=sid,
                )
                audio, _ = read_wav_float_mono(wav)
                segments.append(audio)

                vs_v41: Optional[Dict[str, Any]] = None
                if mode == V41_MODE:
                    v41_audio_cache[(note_name, sid)] = audio.copy()
                elif is_v4_1_identity_space_mode(mode):
                    ref = v41_audio_cache.get((note_name, sid))
                    if ref is not None:
                        vs_v41 = compare_audio_to_reference(audio, ref)

                seg = _segment_row(
                    meta,
                    sample=sample,
                    note_name=note_name,
                    frequency_hz=frequency_hz,
                    seg_i=len(seg_rows),
                    audio=audio,
                    sample_rate=sample_rate,
                    diagnostic_mode=mode,
                    z_body=z_body if is_v4_1_identity_space_mode(mode) else None,
                    vs_v41=vs_v41,
                )
                seg_rows.append(seg)
                distance_samples_by_mode[mode].append(
                    {
                        "sample_id": sid,
                        "z_body": z_body,
                        "timbre": seg.get("timbre_vector") or [],
                    }
                )
            stitch = out_dir / f"{note_name}_{mode}_10_guitars.wav"
            write_wav_mono(
                stitch,
                concatenate_audio_with_crossfade(segments, sample_rate=sample_rate, silence_ms=silence_s * 1000),
                sample_rate,
            )
            stitched.append(str(stitch))
            note_rows.append(
                {
                    "note": note_name,
                    "frequency_hz": frequency_hz,
                    "segments": seg_rows,
                    "average_spectral_similarity": average_spectral_similarity(segments),
                }
            )
        mode_summaries[mode] = summarize_diagnostic_mode(note_rows, diagnostic_mode=mode)
        for nr in note_rows:
            nk = nr["note"]
            if nk in (mode_summaries[mode].get("notes") or {}):
                mode_summaries[mode]["notes"][nk]["segments"] = nr["segments"]

    distance_by_mode = {
        mode: distance_consistency_report(rows) for mode, rows in distance_samples_by_mode.items()
    }

    manifest = {
        "stage": "5.1D",
        "out_dir": str(out_dir),
        "sample_ids": [str(s["sample_id"]) for s in samples],
        "notes": [n for n, _ in notes],
        "note_frequencies_hz": {n: f for n, f in notes},
        "modes": list(modes),
        "use_surrogate": use_surrogate,
        "fem_launched": False,
        "stitched_files": stitched,
        "distance_consistency_by_mode": distance_by_mode,
    }
    (out_dir / "build_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    report_dir = repo_root / "audio" / "debug_reports"
    build_stage51d_identity_sweep_report(
        mode_summaries=mode_summaries,
        distance_by_mode=distance_by_mode,
        notes=[n for n, _ in notes],
        modes=modes,
        build_manifest=manifest,
        out_json=report_dir / "stage51d_stk_v4_1_identity_sweep_report.json",
        out_md=report_dir / "stage51d_stk_v4_1_identity_sweep_report.md",
    )
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 5.1D V4.1 identity strength sweep (A3 default)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--notes", type=str, default=DEFAULT_NOTES)
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
    manifest = build_v4_1_identity_sweep_diagnostics(
        repo_root=REPO,
        out_dir=args.out_dir,
        notes=notes,
        modes=modes,
        max_samples=args.max_samples,
        duration_s=args.duration,
        silence_s=args.silence,
        use_surrogate=use_surrogate,
    )
    print(json.dumps(manifest, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
