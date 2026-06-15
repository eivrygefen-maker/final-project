#!/usr/bin/env python3
"""Assemble APP melodies from cached STK note WAVs (concatenate only, no synthesis)."""
from __future__ import annotations

import argparse
import json
import struct
import sys
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "gui"))

from stk_app_audio_service import (  # noqa: E402
    MELODY_CACHE_ROOT,
    MELODY_LIBRARY_JSON,
    get_note_wav,
    load_melody_library,
    note_wav_path,
)

FADE_IN_MS = 15.0
FADE_OUT_MS = 60.0
GAP_MS = 25.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_mono_pcm(path: Path) -> Tuple[List[float], int]:
    with wave.open(str(path), "rb") as wf:
        nchannels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    if sampwidth != 2:
        raise ValueError(f"{path}: expected 16-bit PCM")
    ints = struct.unpack(f"<{len(raw) // 2}h", raw)
    if nchannels == 1:
        samples = [v / 32768.0 for v in ints]
    else:
        samples = [sum(ints[i : i + nchannels]) / (32768.0 * nchannels) for i in range(0, len(ints), nchannels)]
    return samples, framerate


def _write_mono_pcm(path: Path, samples: Sequence[float], sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = struct.pack(f"<{len(samples)}h", *[int(round(max(-1.0, min(1.0, s)) * 32767.0)) for s in samples])
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)


def _apply_fade(samples: List[float], *, fade_in: int, fade_out: int) -> None:
    n = len(samples)
    for i in range(min(fade_in, n)):
        samples[i] *= (i + 1) / max(fade_in, 1)
    for i in range(min(fade_out, n)):
        idx = n - fade_out + i
        samples[idx] *= (fade_out - i) / max(fade_out, 1)


def render_melody(
    *,
    sample_id: str,
    melody: Mapping[str, Any],
    note_cache_root: Path,
    output_path: Path,
    instrument: str = "classical",
) -> Dict[str, Any]:
    tempo = float(melody.get("tempo_bpm") or 90)
    beat_s = 60.0 / tempo
    notes_spec = melody.get("notes") or []
    if not notes_spec:
        raise ValueError("melody has no notes")

    stitched: List[float] = []
    sample_rate = 44100
    missing: List[str] = []
    used_notes: List[str] = []

    for idx, step in enumerate(notes_spec):
        note_name = str(step["note"])
        beats = float(step.get("beats") or 1.0)
        target_s = beats * beat_s
        src = get_note_wav(sample_id, note_name, instrument=instrument, output_root=note_cache_root)
        if src is None:
            missing.append(note_name)
            continue
        pcm, sr = _read_mono_pcm(src)
        sample_rate = sr
        target_frames = max(1, int(round(target_s * sample_rate)))
        segment = pcm[:target_frames] if len(pcm) >= target_frames else pcm + [0.0] * (target_frames - len(pcm))
        fi = max(1, int(round(FADE_IN_MS * 0.001 * sample_rate)))
        fo = max(1, int(round(FADE_OUT_MS * 0.001 * sample_rate)))
        _apply_fade(segment, fade_in=fi, fade_out=fo)
        stitched.extend(segment)
        used_notes.append(note_name)
        if idx + 1 < len(notes_spec):
            gap = max(0, int(round(GAP_MS * 0.001 * sample_rate)))
            stitched.extend([0.0] * gap)

    if missing:
        raise FileNotFoundError(
            f"missing cached STK notes {missing} — run tools/build_app_stk_note_library.py first "
            f"(expected under {note_cache_root})"
        )

    _write_mono_pcm(output_path, stitched, sample_rate)
    return {
        "melody_id": melody.get("id"),
        "display_name": melody.get("display_name"),
        "tempo_bpm": tempo,
        "notes_used": used_notes,
        "duration_s": round(len(stitched) / sample_rate, 4),
        "output_path": str(output_path).replace("\\", "/"),
        "sample_rate_hz": sample_rate,
    }


def write_melody_report(report: Dict[str, Any], json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md = [
        f"# APP STK Melody — {report.get('melody_id')}",
        "",
        f"- **sample_id**: {report.get('sample_id')}",
        f"- **duration_s**: {report.get('duration_s')}",
        f"- **output**: `{report.get('output_path')}`",
        f"- **readiness**: {report.get('readiness')}",
        "",
    ]
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render APP melody from cached STK notes.")
    parser.add_argument("--sample-id", default="sample_000")
    parser.add_argument("--melody-id", required=True)
    parser.add_argument("--note-cache-root", type=Path, default=REPO_ROOT / "audio" / "app_stk_note_cache")
    parser.add_argument("--melody-json", type=Path, default=MELODY_LIBRARY_JSON)
    parser.add_argument("--output-dir", type=Path, default=MELODY_CACHE_ROOT / "classical" / "sample_000")
    parser.add_argument("--instrument", default="classical")
    args = parser.parse_args(list(argv) if argv is not None else None)

    lib = load_melody_library(args.melody_json)
    melody = next((m for m in lib.get("melodies") or [] if m.get("id") == args.melody_id), None)
    if melody is None:
        print(f"ERROR: unknown melody_id {args.melody_id!r}", file=sys.stderr)
        return 1

    out_path = Path(args.output_dir) / f"{args.melody_id}.wav"
    try:
        meta = render_melody(
            sample_id=args.sample_id,
            melody=melody,
            note_cache_root=args.note_cache_root,
            output_path=out_path,
            instrument=args.instrument,
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    report = {
        "generated_at": _utc_now(),
        "sample_id": args.sample_id,
        "instrument": args.instrument,
        "melody_id": args.melody_id,
        "renderer": "cached_stk_note_concatenation",
        "python_role": "assembly_only_no_synthesis",
        "readiness": "ready_for_app_playback",
        **meta,
    }
    rj = REPO_ROOT / "audio" / "debug_reports" / f"app_stk_melody_{args.sample_id}_{args.melody_id}_report.json"
    rm = REPO_ROOT / "audio" / "debug_reports" / f"app_stk_melody_{args.sample_id}_{args.melody_id}_report.md"
    write_melody_report(report, rj, rm)
    print(f"Wrote {out_path}")
    print(f"Report: {rj}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
