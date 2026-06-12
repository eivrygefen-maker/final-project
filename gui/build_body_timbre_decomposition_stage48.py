#!/usr/bin/env python3
"""
Stage 4.8 — Generate layer-separated timbre decomposition audio pack.

Uses M4 surrogate on VM (no FEM, no ROM retrain).
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

from build_sample_comparison import (  # noqa: E402
    DEFAULT_SAMPLE_RATE,
    NOTE_NAME_TO_HZ,
    concatenate_audio_with_crossfade,
    load_lhs_sample_entries,
    parse_notes_arg,
    resolve_modal_data_for_sample,
    write_wav_mono,
)
from stage48_timbre_decomposition_report import (  # noqa: E402
    STITCH_LAYER_ALIASES,
    build_stage48_report,
)
from timbre_decomposition import (  # noqa: E402
    LAYER_NAMES,
    compute_note_layers,
    layer_segment_row,
)

DEFAULT_OUT = REPO / "audio" / "body_timbre_decomposition_stage48"
DEFAULT_DURATION_S = 2.5
DEFAULT_SILENCE_S = 0.35


def _layer_wav_name(layer: str) -> str:
    return f"{layer}.wav"


def build_stage48_pack(
    *,
    repo_root: Path,
    out_dir: Path,
    notes: Sequence[Tuple[str, float]],
    max_samples: int = 10,
    duration_s: float = DEFAULT_DURATION_S,
    silence_s: float = DEFAULT_SILENCE_S,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    use_surrogate: bool = True,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = load_lhs_sample_entries(repo_root, max_samples=max_samples)
    if not samples:
        raise RuntimeError("no LHS samples found in ROM/classic/lhs_pool.json")

    all_rows: List[Dict[str, Any]] = []
    stitched: Dict[str, Path] = {}
    per_note_layers: Dict[str, Dict[str, List[np.ndarray]]] = {
        note: {layer: [] for layer in STITCH_LAYER_ALIASES} for note, _ in notes
    }

    for sample in samples:
        sid = str(sample["sample_id"])
        sample_dir_root = out_dir
        for note_name, frequency_hz in notes:
            modal_data, modal_source = resolve_modal_data_for_sample(
                repo_root, sample, use_surrogate=use_surrogate
            )
            result = compute_note_layers(
                frequency_hz=frequency_hz,
                note_name=note_name,
                duration_s=duration_s,
                sample_rate=sample_rate,
                modal_data=modal_data,
                sample_parameters=sample.get("parameters"),
                use_bridge_mobility_proxy=True,
                modal_source=modal_source,
            )
            note_out = out_dir / note_name / sid
            note_out.mkdir(parents=True, exist_ok=True)
            meta = dict(result["metadata"])
            meta["sample_id"] = sid
            meta["modal_source"] = modal_source
            meta["layers_written"] = []

            for layer_name in LAYER_NAMES:
                audio = np.asarray(result["layers"][layer_name], dtype=np.float64)
                wav_path = note_out / _layer_wav_name(layer_name)
                write_wav_mono(wav_path, audio, sample_rate)
                meta["layers_written"].append(layer_name)
                row = layer_segment_row(
                    sample_id=sid,
                    note_name=note_name,
                    frequency_hz=frequency_hz,
                    layer_name=layer_name,
                    audio=audio,
                    sample_rate=sample_rate,
                    metadata=meta,
                    sample_parameters=sample.get("parameters") or {},
                )
                row["audio"] = audio
                all_rows.append(row)
                if layer_name in STITCH_LAYER_ALIASES:
                    per_note_layers[note_name][layer_name].append(audio)

            (note_out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    for note_name, _ in notes:
        for layer_name, alias in STITCH_LAYER_ALIASES.items():
            segments = per_note_layers[note_name][layer_name]
            if not segments:
                continue
            stitched_name = f"{note_name}_{alias}_10_guitars.wav"
            stitched_path = out_dir / stitched_name
            combined = concatenate_audio_with_crossfade(
                segments,
                sample_rate=sample_rate,
                silence_ms=silence_s * 1000.0,
            )
            write_wav_mono(stitched_path, combined, sample_rate)
            stitched[stitched_name] = stitched_path

    listening_dir = out_dir / "listening_packs"
    listening_dir.mkdir(parents=True, exist_ok=True)
    listening_packs = _build_listening_packs(
        out_dir=out_dir,
        listening_dir=listening_dir,
        notes=[n for n, _ in notes],
        samples=samples,
        sample_rate=sample_rate,
        silence_s=silence_s,
    )

    manifest = {
        "stage": "4.8",
        "out_dir": str(out_dir),
        "max_samples": max_samples,
        "sample_ids": [str(s["sample_id"]) for s in samples],
        "notes": [n for n, _ in notes],
        "use_surrogate": use_surrogate,
        "fem_launched": False,
        "stitched_files": [str(p) for p in stitched.values()],
        "listening_packs": listening_packs,
        "layer_count": len(LAYER_NAMES),
    }
    (out_dir / "build_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    report_dir = repo_root / "audio" / "debug_reports"
    build_stage48_report(
        build_manifest=manifest,
        analysis_rows=all_rows,
        notes=[n for n, _ in notes],
        out_json=report_dir / "stage48_timbre_decomposition_report.json",
        out_md=report_dir / "stage48_timbre_decomposition_report.md",
    )
    return manifest


def _build_listening_packs(
    *,
    out_dir: Path,
    listening_dir: Path,
    notes: Sequence[str],
    samples: Sequence[Mapping[str, Any]],
    sample_rate: int,
    silence_s: float,
) -> Dict[str, str]:
    packs: Dict[str, str] = {}
    body_layers = ["body_only_raw_pre_norm", "body_only_final_norm"]
    for note in notes:
        for bl in body_layers:
            key = f"ab_same_note_{note}_{bl}"
            segs = []
            for sample in samples:
                sid = str(sample["sample_id"])
                wav = out_dir / note / sid / f"{bl}.wav"
                if wav.is_file():
                    from build_sample_comparison import read_wav_float_mono

                    audio, _ = read_wav_float_mono(wav)
                    segs.append(audio)
            if segs:
                path = listening_dir / f"{key}.wav"
                write_wav_mono(
                    path,
                    concatenate_audio_with_crossfade(segs, sample_rate=sample_rate, silence_ms=silence_s * 1000.0),
                    sample_rate,
                )
                packs[key] = str(path)

    for note in notes:
        for mix_layer in ("full_mix_baseline", "full_mix_candidate_balance"):
            key = f"ab_same_note_{note}_{mix_layer}"
            segs = []
            for sample in samples:
                sid = str(sample["sample_id"])
                wav = out_dir / note / sid / f"{mix_layer}.wav"
                if wav.is_file():
                    from build_sample_comparison import read_wav_float_mono

                    audio, _ = read_wav_float_mono(wav)
                    segs.append(audio)
            if segs:
                path = listening_dir / f"{key}.wav"
                write_wav_mono(
                    path,
                    concatenate_audio_with_crossfade(segs, sample_rate=sample_rate, silence_ms=silence_s * 1000.0),
                    sample_rate,
                )
                packs[key] = str(path)

    if samples:
        rep = str(samples[len(samples) // 2]["sample_id"])
        for note in notes:
            layer_segs = []
            for layer in (
                "string_only",
                "body_only_raw_pre_norm",
                "full_mix_baseline",
                "full_mix_radiation_v1",
                "full_mix_candidate_balance",
            ):
                wav = out_dir / note / rep / f"{layer}.wav"
                if wav.is_file():
                    from build_sample_comparison import read_wav_float_mono

                    audio, _ = read_wav_float_mono(wav)
                    layer_segs.append(audio)
            if layer_segs:
                key = f"ab_same_guitar_{rep}_{note}_layers"
                path = listening_dir / f"{key}.wav"
                write_wav_mono(
                    path,
                    concatenate_audio_with_crossfade(layer_segs, sample_rate=sample_rate, silence_ms=silence_s * 1000.0),
                    sample_rate,
                )
                packs[key] = str(path)
    return packs


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 4.8 timbre decomposition pack")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--notes", type=str, default="A2,A4,E5")
    parser.add_argument("--duration-s", type=float, default=DEFAULT_DURATION_S)
    parser.add_argument("--silence-s", type=float, default=DEFAULT_SILENCE_S)
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument(
        "--no-surrogate",
        action="store_true",
        help="Use synthetic modal fallback (offline tests only)",
    )
    args = parser.parse_args(argv)
    notes = parse_notes_arg(args.notes)
    manifest = build_stage48_pack(
        repo_root=REPO,
        out_dir=args.out_dir,
        notes=notes,
        max_samples=args.max_samples,
        duration_s=args.duration_s,
        silence_s=args.silence_s,
        sample_rate=args.sample_rate,
        use_surrogate=not args.no_surrogate,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
