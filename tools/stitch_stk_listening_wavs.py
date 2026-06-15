#!/usr/bin/env python3
"""
Stitch rendered STK guitar WAVs into per-note listening comparison files.

Uses only the standard library (wave, struct, json). No audio synthesis.
"""
from __future__ import annotations

import argparse
import json
import struct
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLES = tuple(f"sample_{i:03d}" for i in range(10))
DEFAULT_NOTES = ("A2", "A4", "E5")
FADE_IN_MS = 20.0
FADE_OUT_MS = 80.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _expected_wav_name(sample_id: str, note_name: str) -> str:
    return f"{sample_id}_{note_name}_stk_guitar.wav"


def _read_mono_pcm(path: Path) -> Tuple[List[float], int]:
    with wave.open(str(path), "rb") as wf:
        nchannels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)
    if sampwidth != 2:
        raise ValueError(f"{path}: expected 16-bit PCM, got sampwidth={sampwidth}")
    count = len(raw) // 2
    ints = struct.unpack(f"<{count}h", raw)
    if nchannels == 1:
        samples = [v / 32768.0 for v in ints]
    else:
        samples = []
        for i in range(0, len(ints), nchannels):
            samples.append(sum(ints[i : i + nchannels]) / (32768.0 * nchannels))
    return samples, framerate


def _write_mono_pcm(path: Path, samples: Sequence[float], sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = [max(-1.0, min(1.0, s)) for s in samples]
    pcm = struct.pack(f"<{len(clipped)}h", *[int(round(s * 32767.0)) for s in clipped])
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)


def _apply_fade(
    samples: List[float],
    *,
    fade_in_samples: int,
    fade_out_samples: int,
) -> None:
    n = len(samples)
    if n == 0:
        return
    fi = min(fade_in_samples, n)
    for i in range(fi):
        samples[i] *= (i + 1) / max(fi, 1)
    fo = min(fade_out_samples, n)
    for i in range(fo):
        idx = n - fo + i
        samples[idx] *= (fo - i) / max(fo, 1)


def _segment_audio(
    samples: List[float],
    sample_rate: int,
    segment_seconds: float,
) -> List[float]:
    target = max(1, int(round(segment_seconds * sample_rate)))
    if len(samples) >= target:
        return list(samples[:target])
    out = list(samples)
    out.extend([0.0] * (target - len(samples)))
    return out


def _silence(sample_rate: int, seconds: float) -> List[float]:
    return [0.0] * max(0, int(round(seconds * sample_rate)))


def stitch_note(
    *,
    input_dir: Path,
    output_path: Path,
    note_name: str,
    sample_ids: Sequence[str],
    segment_seconds: float,
    gap_seconds: float,
    fade_in_ms: float,
    fade_out_ms: float,
) -> dict:
    stitched: List[float] = []
    sample_rate = 44100
    sources: List[str] = []
    fade_in_samples = max(1, int(round(fade_in_ms * 0.001 * sample_rate)))
    fade_out_samples = max(1, int(round(fade_out_ms * 0.001 * sample_rate)))
    gap_samples = max(0, int(round(gap_seconds * sample_rate)))

    for idx, sample_id in enumerate(sample_ids):
        src_name = _expected_wav_name(sample_id, note_name)
        src_path = input_dir / src_name
        if not src_path.is_file():
            raise FileNotFoundError(f"missing source WAV: {src_path}")
        pcm, sr = _read_mono_pcm(src_path)
        if stitched and sr != sample_rate:
            raise ValueError(f"sample rate mismatch: {src_path} ({sr} vs {sample_rate})")
        sample_rate = sr
        fade_in_samples = max(1, int(round(fade_in_ms * 0.001 * sample_rate)))
        fade_out_samples = max(1, int(round(fade_out_ms * 0.001 * sample_rate)))
        segment = _segment_audio(pcm, sample_rate, segment_seconds)
        _apply_fade(segment, fade_in_samples=fade_in_samples, fade_out_samples=fade_out_samples)
        stitched.extend(segment)
        sources.append(src_name)
        if idx + 1 < len(sample_ids) and gap_samples > 0:
            stitched.extend([0.0] * gap_samples)

    _write_mono_pcm(output_path, stitched, sample_rate)
    duration_s = len(stitched) / float(sample_rate)
    return {
        "note_name": note_name,
        "output_file": output_path.name,
        "output_path": str(output_path).replace("\\", "/"),
        "source_wavs": sources,
        "source_wav_count": len(sources),
        "sample_order": list(sample_ids),
        "segment_seconds": segment_seconds,
        "gap_seconds": gap_seconds,
        "fade_in_ms": fade_in_ms,
        "fade_out_ms": fade_out_ms,
        "sample_rate_hz": sample_rate,
        "duration_s": round(duration_s, 4),
        "total_frames": len(stitched),
    }


def build_stitch_report(
    *,
    input_dir: Path,
    output_dir: Path,
    stitched_files: Sequence[dict],
    sample_ids: Sequence[str],
    notes: Sequence[str],
    segment_seconds: float,
    gap_seconds: float,
    fade_in_ms: float,
    fade_out_ms: float,
    source_wav_count: int,
) -> dict:
    readiness = "ready_for_listening_comparison"
    if source_wav_count < len(sample_ids) * len(notes):
        readiness = "incomplete_source_wav_set"
    return {
        "generated_at": _utc_now(),
        "utility": "stitch_stk_listening_wavs",
        "input_dir": str(input_dir).replace("\\", "/"),
        "output_dir": str(output_dir).replace("\\", "/"),
        "sample_order": list(sample_ids),
        "notes": list(notes),
        "segment_seconds": segment_seconds,
        "gap_seconds": gap_seconds,
        "fade_in_ms": fade_in_ms,
        "fade_out_ms": fade_out_ms,
        "source_wav_count": source_wav_count,
        "stitched_files": list(stitched_files),
        "readiness_for_listening_comparison": readiness,
    }


def write_report(report: dict, json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# STK v4 Stitched Listening Report",
        "",
        f"- **generated_at**: {report['generated_at']}",
        f"- **input_dir**: `{report['input_dir']}`",
        f"- **output_dir**: `{report['output_dir']}`",
        f"- **source_wav_count**: {report['source_wav_count']}",
        f"- **segment_seconds**: {report['segment_seconds']}",
        f"- **gap_seconds**: {report['gap_seconds']}",
        f"- **fade_in_ms**: {report['fade_in_ms']}",
        f"- **fade_out_ms**: {report['fade_out_ms']}",
        f"- **readiness**: {report['readiness_for_listening_comparison']}",
        "",
        "## Sample order",
        "",
    ]
    for sid in report["sample_order"]:
        lines.append(f"- `{sid}`")
    lines.extend(["", "## Stitched files", ""])
    for row in report["stitched_files"]:
        lines.append(f"- `{row['output_file']}` ({row['duration_s']} s, {row['source_wav_count']} segments)")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stitch STK guitar WAVs for listening comparison.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", nargs="+", default=list(DEFAULT_SAMPLES))
    parser.add_argument("--notes", nargs="+", default=list(DEFAULT_NOTES))
    parser.add_argument("--segment-seconds", type=float, default=3.5)
    parser.add_argument("--gap-seconds", type=float, default=0.3)
    parser.add_argument("--fade-in-ms", type=float, default=FADE_IN_MS)
    parser.add_argument("--fade-out-ms", type=float, default=FADE_OUT_MS)
    parser.add_argument(
        "--report-json",
        type=Path,
        default=REPO_ROOT / "audio/debug_reports/pgsm_stk_guitar_demo_v4_10_samples_stitched_report.json",
    )
    parser.add_argument(
        "--report-md",
        type=Path,
        default=REPO_ROOT / "audio/debug_reports/pgsm_stk_guitar_demo_v4_10_samples_stitched_report.md",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not (3.0 <= args.segment_seconds <= 5.0):
        raise SystemExit("--segment-seconds must be between 3.0 and 5.0")
    if not (0.2 <= args.gap_seconds <= 0.4):
        raise SystemExit("--gap-seconds must be between 0.2 and 0.4")

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_wavs = sorted(input_dir.glob("*.wav"))
    if len(source_wavs) == 0:
        raise SystemExit(f"no WAV files in {input_dir}")

    stitched_rows: List[dict] = []
    for note in args.notes:
        out_name = f"{note}_all_{len(args.samples)}_samples_stitched.wav"
        row = stitch_note(
            input_dir=input_dir,
            output_path=output_dir / out_name,
            note_name=note,
            sample_ids=args.samples,
            segment_seconds=args.segment_seconds,
            gap_seconds=args.gap_seconds,
            fade_in_ms=args.fade_in_ms,
            fade_out_ms=args.fade_out_ms,
        )
        stitched_rows.append(row)
        print(f"Wrote {row['output_path']}")

    report = build_stitch_report(
        input_dir=input_dir,
        output_dir=output_dir,
        stitched_files=stitched_rows,
        sample_ids=args.samples,
        notes=args.notes,
        segment_seconds=args.segment_seconds,
        gap_seconds=args.gap_seconds,
        fade_in_ms=args.fade_in_ms,
        fade_out_ms=args.fade_out_ms,
        source_wav_count=len(source_wavs),
    )
    write_report(report, args.report_json.resolve(), args.report_md.resolve())
    print(f"Report: {args.report_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
