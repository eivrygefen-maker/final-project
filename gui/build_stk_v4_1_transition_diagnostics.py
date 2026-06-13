#!/usr/bin/env python3
"""Stage 5.1B STK V4.1 transition-band diagnostics builder."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gui"))

from body_hybrid_v4_1 import radiation_blend_weight_f0  # noqa: E402
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
from stage51b_stk_v4_1_transition_report import build_stage51b_transition_report  # noqa: E402
from synthesis_presets import DEFAULT_SYNTHESIS_PRESET  # noqa: E402

DEFAULT_OUT = REPO / "audio" / "stk_v4_1_transition_diagnostics"
DEFAULT_NOTES = "E3,A3,D4"
DEFAULT_MODES = (
    "baseline_current,modal_radiation_color_v1,modal_body_hybrid_v4_full,"
    "modal_body_hybrid_v4_1_core,modal_body_hybrid_v4_1_full"
)


def _peak_dbfs(audio: np.ndarray) -> float:
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak <= 1e-12:
        return -120.0
    return 20.0 * float(np.log10(peak))


def _segment_row_with_clipping(
    meta: Mapping[str, Any],
    *,
    sample: Mapping[str, Any],
    note_name: str,
    frequency_hz: float,
    seg_i: int,
    audio: np.ndarray,
    sample_rate: int,
    diagnostic_mode: str,
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
    peak_db = meta.get("output_peak_dbfs")
    if peak_db is None:
        peak_db = _peak_dbfs(audio)
    row["output_peak_dbfs"] = peak_db
    row["clipping_detected"] = float(peak_db) > -0.5 or float(np.max(np.abs(audio))) >= 0.999
    return row


def _core_full_identity(
    out_dir: Path,
    notes: Sequence[str],
    sample_rate: int,
) -> Dict[str, Any]:
    per_note: Dict[str, Any] = {}
    all_identical = True
    for note in notes:
        core_path = out_dir / f"{note}_modal_body_hybrid_v4_1_core_10_guitars.wav"
        full_path = out_dir / f"{note}_modal_body_hybrid_v4_1_full_10_guitars.wav"
        if not core_path.is_file() or not full_path.is_file():
            per_note[note] = {"identical": None, "reason": "missing_stitch_file"}
            all_identical = False
            continue
        core, _ = read_wav_float_mono(core_path)
        full, _ = read_wav_float_mono(full_path)
        n = min(len(core), len(full))
        if n < 8:
            per_note[note] = {"identical": False, "reason": "too_short"}
            all_identical = False
            continue
        max_diff = float(np.max(np.abs(core[:n] - full[:n])))
        identical = max_diff < 1e-6
        per_note[note] = {
            "identical": identical,
            "max_sample_diff": max_diff,
            "spectral_similarity": average_spectral_similarity([core[:n], full[:n]]),
        }
        if not identical:
            all_identical = False
    return {"core_full_identical": all_identical, "per_note": per_note}


def build_v4_1_transition_diagnostics(
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
    transition_rows: List[Dict[str, Any]] = []
    stitched: List[str] = []

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
                seg_row = _segment_row_with_clipping(
                    meta,
                    sample=sample,
                    note_name=note_name,
                    frequency_hz=frequency_hz,
                    seg_i=len(seg_rows),
                    audio=audio,
                    sample_rate=sample_rate,
                    diagnostic_mode=mode,
                )
                seg_rows.append(seg_row)
                if mode.startswith("modal_body_hybrid_v4_1"):
                    w_rad = meta.get("radiation_blend_weight_w_rad")
                    if w_rad is None:
                        w_rad = radiation_blend_weight_f0(frequency_hz)
                    transition_rows.append(
                        {
                            "sample_id": sid,
                            "note": note_name,
                            "frequency_hz": frequency_hz,
                            "w_rad": w_rad,
                            "v4_1_endpoint": meta.get("v4_1_endpoint"),
                            "diagnostic_mode": mode,
                            "endpoint_equivalence": meta.get("endpoint_equivalence"),
                        }
                    )
            stitch = out_dir / f"{note_name}_{mode}_10_guitars.wav"
            write_wav_mono(
                stitch,
                concatenate_audio_with_crossfade(segments, sample_rate=sample_rate, silence_ms=silence_s * 1000),
                sample_rate,
            )
            stitched.append(str(stitch))
            note_row = {
                "note": note_name,
                "frequency_hz": frequency_hz,
                "segments": seg_rows,
                "average_spectral_similarity": average_spectral_similarity(segments),
            }
            note_rows.append(note_row)
        mode_summaries[mode] = summarize_diagnostic_mode(note_rows, diagnostic_mode=mode)
        for nr in note_rows:
            note_key = nr["note"]
            if note_key in (mode_summaries[mode].get("notes") or {}):
                mode_summaries[mode]["notes"][note_key]["segments"] = nr["segments"]

    core_full = _core_full_identity(out_dir, [n for n, _ in notes], sample_rate)

    manifest = {
        "stage": "5.1B",
        "out_dir": str(out_dir),
        "sample_ids": [str(s["sample_id"]) for s in samples],
        "notes": [n for n, _ in notes],
        "note_frequencies_hz": {n: f for n, f in notes},
        "modes": list(modes),
        "use_surrogate": use_surrogate,
        "fem_launched": False,
        "stitched_files": stitched,
        "core_full_identity": core_full,
    }
    (out_dir / "build_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    report_dir = repo_root / "audio" / "debug_reports"
    build_stage51b_transition_report(
        mode_summaries=mode_summaries,
        transition_rows=transition_rows,
        core_full_identity=core_full,
        notes=[n for n, _ in notes],
        modes=modes,
        build_manifest=manifest,
        out_json=report_dir / "stage51b_stk_v4_1_transition_report.json",
        out_md=report_dir / "stage51b_stk_v4_1_transition_report.md",
    )
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 5.1B STK V4.1 transition-band diagnostics")
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
    manifest = build_v4_1_transition_diagnostics(
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
