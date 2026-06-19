#!/usr/bin/env python3
"""Build offline listening comparisons from existing Classical STK note caches.

This diagnostic only reads already-rendered WAV files. It never invokes STK,
FEM, ROM, or website generation.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import wave
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
GUI_ROOT = REPO_ROOT / "gui"
if str(GUI_ROOT) not in sys.path:
    sys.path.insert(0, str(GUI_ROOT))

from classical_guitar_fretboard import list_note_wavs, normalize_note_name  # noqa: E402


DEFAULT_CACHE_ROOT = REPO_ROOT / "audio" / "app_stk_note_cache" / "classical"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "audio" / "diagnostics"
DEFAULT_TARGET_RMS_DBFS = -20.0
DEFAULT_PEAK_CEILING_DBFS = -1.0
DEFAULT_MAX_GAIN_DB = 6.0


@dataclass(frozen=True)
class CacheCandidate:
    ordinal: int
    sample_id: str
    parameter_hash: str
    cache_path: Path
    note_wav_path: Optional[Path]


@dataclass(frozen=True)
class AudioClip:
    candidate: CacheCandidate
    original_samples: np.ndarray
    sample_rate: int
    original_duration_s: float
    original_peak: float
    original_rms: float


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _parse_hash_from_cache_name(name: str) -> str:
    if name.startswith("current_preview_"):
        return name.removeprefix("current_preview_")
    if name.startswith("saved_guitar_"):
        remainder = name.removeprefix("saved_guitar_")
        return remainder.rsplit("_", 1)[0] if "_" in remainder else remainder
    return name


def discover_classical_caches(cache_root: Path, note_name: str, max_samples: int) -> Tuple[List[CacheCandidate], List[CacheCandidate]]:
    cache_root = Path(cache_root)
    if not cache_root.is_dir():
        return [], []
    dirs = sorted(
        [p for p in cache_root.iterdir() if p.is_dir() and not p.name.startswith(".")],
        key=lambda p: (not p.name.startswith("current_preview_"), p.name),
    )
    selected: List[CacheCandidate] = []
    missing: List[CacheCandidate] = []
    for path in dirs:
        note_wavs = list_note_wavs(path)
        note_wav = note_wavs.get(note_name)
        candidate = CacheCandidate(
            ordinal=len(selected) + len(missing),
            sample_id=f"sample_{len(selected):03d}",
            parameter_hash=_parse_hash_from_cache_name(path.name),
            cache_path=path,
            note_wav_path=note_wav,
        )
        if note_wav is None:
            missing.append(candidate)
            continue
        selected.append(candidate)
        if len(selected) >= max_samples:
            break
    return selected, missing


def read_wav_mono_float(path: Path) -> Tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        raw = wav.readframes(wav.getnframes())
    if sample_width == 1:
        samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    elif sample_width == 2:
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    elif sample_width == 4:
        samples = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
    else:
        raise ValueError(f"unsupported WAV sample width {sample_width} for {path}")
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples.astype(np.float64, copy=False), int(sample_rate)


def write_wav_mono_int16(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def rms(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples))))


def db_to_gain(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def fit_duration(samples: np.ndarray, sample_rate: int, duration_s: float) -> np.ndarray:
    target_len = max(1, int(round(float(duration_s) * sample_rate)))
    if samples.size >= target_len:
        return samples[:target_len].copy()
    out = np.zeros(target_len, dtype=np.float64)
    out[: samples.size] = samples
    return out


def load_clips(candidates: Sequence[CacheCandidate]) -> Tuple[List[AudioClip], List[Dict[str, Any]]]:
    clips: List[AudioClip] = []
    skipped: List[Dict[str, Any]] = []
    reference_rate: Optional[int] = None
    for candidate in candidates:
        if candidate.note_wav_path is None:
            continue
        try:
            samples, sample_rate = read_wav_mono_float(candidate.note_wav_path)
        except Exception as exc:  # noqa: BLE001 - diagnostic should keep scanning.
            skipped.append(
                {
                    "sample_id": candidate.sample_id,
                    "cache_path": _repo_relative(candidate.cache_path),
                    "note_wav_path": _repo_relative(candidate.note_wav_path),
                    "reason": f"read_error: {exc}",
                }
            )
            continue
        if reference_rate is None:
            reference_rate = sample_rate
        elif sample_rate != reference_rate:
            skipped.append(
                {
                    "sample_id": candidate.sample_id,
                    "cache_path": _repo_relative(candidate.cache_path),
                    "note_wav_path": _repo_relative(candidate.note_wav_path),
                    "reason": f"sample_rate_mismatch: {sample_rate} != {reference_rate}",
                }
            )
            continue
        clips.append(
            AudioClip(
                candidate=candidate,
                original_samples=samples,
                sample_rate=sample_rate,
                original_duration_s=float(samples.size / sample_rate),
                original_peak=float(np.max(np.abs(samples))) if samples.size else 0.0,
                original_rms=rms(samples),
            )
        )
    return clips, skipped


def render_sequence(
    clips: Sequence[AudioClip],
    *,
    duration_s: float,
    silence_s: float,
    target_rms_dbfs: float,
    peak_ceiling_dbfs: float,
    max_gain_db: float,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    if not clips:
        return np.zeros(0, dtype=np.float64), []
    sample_rate = clips[0].sample_rate
    target_rms = db_to_gain(target_rms_dbfs)
    peak_ceiling = db_to_gain(peak_ceiling_dbfs)
    min_gain = db_to_gain(-abs(max_gain_db))
    max_gain = db_to_gain(abs(max_gain_db))
    silence = np.zeros(max(0, int(round(silence_s * sample_rate))), dtype=np.float64)
    parts: List[np.ndarray] = []
    rows: List[Dict[str, Any]] = []
    for idx, clip in enumerate(clips):
        fitted = fit_duration(clip.original_samples, sample_rate, duration_s)
        original_rms = rms(fitted)
        original_peak = float(np.max(np.abs(fitted))) if fitted.size else 0.0
        gain = 1.0 if original_rms <= 1e-12 else target_rms / original_rms
        gain = min(max(gain, min_gain), max_gain)
        if original_peak > 1e-12 and original_peak * gain > peak_ceiling:
            gain = peak_ceiling / original_peak
        rendered = fitted * gain
        final_peak = float(np.max(np.abs(rendered))) if rendered.size else 0.0
        final_rms = rms(rendered)
        parts.append(rendered)
        if idx < len(clips) - 1 and silence.size:
            parts.append(silence)
        rows.append(
            {
                "order": idx + 1,
                "sample_id": clip.candidate.sample_id,
                "parameter_hash": clip.candidate.parameter_hash,
                "cache_path": _repo_relative(clip.candidate.cache_path),
                "note_wav_path": _repo_relative(clip.candidate.note_wav_path or Path()),
                "source_duration_s": round(clip.original_duration_s, 6),
                "segment_duration_s": round(duration_s, 6),
                "original_peak": round(original_peak, 8),
                "original_rms": round(original_rms, 8),
                "gain_db": round(20.0 * math.log10(max(gain, 1e-12)), 4),
                "final_peak": round(final_peak, 8),
                "final_rms": round(final_rms, 8),
            }
        )
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float64), rows


def write_markdown_report(path: Path, report: Dict[str, Any]) -> None:
    lines = [
        "# Classical STK Contrast Diagnostic",
        "",
        f"- Note: `{report['note']}`",
        f"- Output WAV: `{report.get('output_wav') or 'not written'}`",
        f"- Selected clips: {len(report['clips'])}",
        f"- Missing-note caches: {len(report['missing_notes'])}",
        f"- Skipped clips: {len(report['skipped'])}",
        "",
        "## Normalization",
        "",
        f"- Method: {report['normalization']['method']}",
        f"- Target RMS: {report['normalization']['target_rms_dbfs']} dBFS",
        f"- Peak ceiling: {report['normalization']['peak_ceiling_dbfs']} dBFS",
        f"- Max gain change: +/-{report['normalization']['max_gain_db']} dB",
        "",
        "## Listening Order",
        "",
    ]
    if report["clips"]:
        lines.append("| # | sample_id | hash | gain dB | cache |")
        lines.append("|---:|---|---|---:|---|")
        for row in report["clips"]:
            lines.append(
                f"| {row['order']} | `{row['sample_id']}` | `{row['parameter_hash']}` | "
                f"{row['gain_db']} | `{row['cache_path']}` |"
            )
    else:
        lines.append("No playable clips were found for this note.")
    if report["missing_notes"]:
        lines.extend(["", "## Missing Notes", ""])
        for row in report["missing_notes"]:
            lines.append(f"- `{row['cache_path']}` missing `{report['note']}`")
    if report["skipped"]:
        lines.extend(["", "## Skipped", ""])
        for row in report["skipped"]:
            lines.append(f"- `{row['cache_path']}`: {row['reason']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_report(args: argparse.Namespace) -> Dict[str, Any]:
    note = normalize_note_name(args.note)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"classic_contrast_{note.replace('#', 's')}_{timestamp}"
    output_wav = output_dir / f"{stem}.wav"
    output_json = output_dir / f"{stem}.json"
    output_md = output_dir / f"{stem}.md"

    candidates, missing = discover_classical_caches(Path(args.cache_root), note, args.max_samples)
    clips, skipped = load_clips(candidates)
    sequence, clip_rows = render_sequence(
        clips,
        duration_s=args.duration_s,
        silence_s=args.silence_s,
        target_rms_dbfs=args.target_rms_dbfs,
        peak_ceiling_dbfs=args.peak_ceiling_dbfs,
        max_gain_db=args.max_gain_db,
    )
    sample_rate = clips[0].sample_rate if clips else None
    wav_written = False
    if clips and sample_rate is not None:
        write_wav_mono_int16(output_wav, sequence, sample_rate)
        wav_written = True

    report: Dict[str, Any] = {
        "status": "ok" if wav_written else "no_available_note_wavs",
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
        "note": note,
        "cache_root": _repo_relative(Path(args.cache_root)),
        "output_wav": _repo_relative(output_wav) if wav_written else None,
        "output_json": _repo_relative(output_json),
        "output_markdown": _repo_relative(output_md),
        "sample_rate_hz": sample_rate,
        "duration_s": args.duration_s,
        "silence_s": args.silence_s,
        "normalization": {
            "method": "per-clip RMS target with +/-gain clamp and peak ceiling; no compression or EQ",
            "target_rms_dbfs": args.target_rms_dbfs,
            "peak_ceiling_dbfs": args.peak_ceiling_dbfs,
            "max_gain_db": args.max_gain_db,
        },
        "clips": clip_rows,
        "missing_notes": [
            {
                "sample_id": row.sample_id,
                "parameter_hash": row.parameter_hash,
                "cache_path": _repo_relative(row.cache_path),
                "missing_note": note,
            }
            for row in missing
        ],
        "skipped": skipped,
        "rendering_needed": bool(missing) or not wav_written,
        "vm_render_note_cache_hint": (
            "If required notes are missing, regenerate the website/STK note cache on the VM; "
            "this diagnostic intentionally does not invoke STK rendering."
        ),
    }
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown_report(output_md, report)
    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Concatenate existing Classical STK note-cache WAVs for contrast listening."
    )
    parser.add_argument("--note", default="A3", help="Note to compare, e.g. A3, A2, or E5.")
    parser.add_argument("--max-samples", type=int, default=20, help="Maximum playable caches to include.")
    parser.add_argument("--duration-s", type=float, default=4.5, help="Trim/pad each note to this duration.")
    parser.add_argument("--silence-s", type=float, default=0.5, help="Silence between clips.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--target-rms-dbfs", type=float, default=DEFAULT_TARGET_RMS_DBFS)
    parser.add_argument("--peak-ceiling-dbfs", type=float, default=DEFAULT_PEAK_CEILING_DBFS)
    parser.add_argument("--max-gain-db", type=float, default=DEFAULT_MAX_GAIN_DB)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    print(json.dumps({k: report[k] for k in ("status", "note", "output_wav", "output_json", "output_markdown")}, indent=2))
    return 0 if report["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
