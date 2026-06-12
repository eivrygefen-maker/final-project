#!/usr/bin/env python3
"""Stage 5.0 STK V4 hybrid diagnostics builder."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gui"))

from body_hybrid_v4 import precompute_batch_caches  # noqa: E402
from body_response_synth import DEFAULT_SAMPLE_RATE, concatenate_audio_with_crossfade  # noqa: E402
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
from stage50_stk_v4_report import build_stage50_report  # noqa: E402
from synthesis_presets import DEFAULT_SYNTHESIS_PRESET  # noqa: E402
from body_response_synth import synthesize_note_with_body_response  # noqa: E402

DEFAULT_OUT = REPO / "audio" / "stk_v4_body_hybrid_diagnostics"
DEFAULT_MODES = (
    "baseline_current,modal_radiation_color_v1,modal_body_signature_v3_full,"
    "modal_body_hybrid_v4_core,modal_body_hybrid_v4_contrast_imprint_only,"
    "modal_body_hybrid_v4_contrast_body_layer_only,modal_body_hybrid_v4_mobility_light_only,"
    "modal_body_hybrid_v4_full"
)


def build_v4_diagnostics(
    *,
    repo_root: Path,
    out_dir: Path,
    notes: Sequence[tuple],
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

    def modal_resolver(sample):
        return resolve_modal_data_for_sample(repo_root, sample, use_surrogate=use_surrogate)

    precompute_batch_caches(repo_root, samples, modal_resolver)

    mode_summaries: Dict[str, Any] = {}
    stitched: List[str] = []

    for mode in modes:
        mode_dir = out_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        note_rows: List[Dict[str, Any]] = []
        for note_name, frequency_hz in notes:
            segments = []
            seg_rows = []
            for sample in samples:
                sid = str(sample["sample_id"])
                params = normalize_sample_parameters(sample.get("parameters"))
                params = {**params, "sample_id": sid}
                modal_data, modal_source = modal_resolver(sample)
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
                    "segments": seg_rows,
                    "average_spectral_similarity": average_spectral_similarity(segments),
                }
            )
        mode_summaries[mode] = summarize_diagnostic_mode(note_rows, diagnostic_mode=mode)

    manifest = {
        "stage": "5.0",
        "out_dir": str(out_dir),
        "sample_ids": [str(s["sample_id"]) for s in samples],
        "notes": [n for n, _ in notes],
        "modes": list(modes),
        "use_surrogate": use_surrogate,
        "fem_launched": False,
        "stitched_files": stitched,
        "cache_dir": str(repo_root / "ROM" / "classic" / "body_signature_cache"),
    }
    (out_dir / "build_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    build_stage50_report(
        mode_summaries=mode_summaries,
        notes=[n for n, _ in notes],
        modes=modes,
        build_manifest=manifest,
        out_json=repo_root / "audio" / "debug_reports" / "stage50_stk_v4_report.json",
        out_md=repo_root / "audio" / "debug_reports" / "stage50_stk_v4_report.md",
    )
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 5.0 STK V4 diagnostics")
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
    manifest = build_v4_diagnostics(
        repo_root=REPO,
        out_dir=args.out_dir,
        notes=notes,
        modes=modes,
        max_samples=args.max_samples,
        duration_s=args.duration,
        silence_s=args.silence,
        use_surrogate=use_surrogate,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
