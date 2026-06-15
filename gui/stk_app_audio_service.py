#!/usr/bin/env python3
"""
STK classical guitar integration for the Streamlit APP.

Python exports parameters only; STK/C++ renders WAVs. Melodies concatenate cached notes.
"""
from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from pgsm_stk_parameter_export import (
    DURATION_S,
    NUMERIC_SR,
    _compute_v4_continuous_mix,
    _extended_voicing,
    build_render_entry,
    load_physical_parameters,
)
from pgsm_emergency_guitar_demo_engine import compute_v5_physical_factors
from pgsm_step5l_limited_multiguitar_differentiation import REFERENCE_SAMPLE_ID

from pgsm_stk_parameter_export import SAMPLE_SET_V4

REPO_ROOT = Path(__file__).resolve().parents[1]
STK_BINARY = REPO_ROOT / "cpp" / "stk_pgsm_guitar_demo" / "build" / "stk_pgsm_guitar_demo"
APP_NOTE_CACHE_ROOT = REPO_ROOT / "audio" / "app_stk_note_cache"
GUITAR_STACK_ROOT = REPO_ROOT / "audio" / "app_stk_guitar_stack"
MELODY_LIBRARY_JSON = REPO_ROOT / "audio" / "app_stk_melody_library" / "melodies.json"
MELODY_CACHE_ROOT = REPO_ROOT / "audio" / "app_stk_melody_cache"
RENDER_TMP_ROOT = REPO_ROOT / "audio" / "app_stk_note_cache" / ".render_tmp"
DEBUG_REPORTS = REPO_ROOT / "audio" / "debug_reports"

NOTE_NAMES: Tuple[str, ...] = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
A4_REFERENCE_HZ = 440.0
ACCEPTED_STK_DEMO_VERSION = "v4_10_samples"
MAX_GUITAR_STACK = 3

_NOTE_RE = re.compile(r"^([A-G])(#|b)?(\d+)$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def note_name_to_frequency(note_name: str) -> float:
    m = _NOTE_RE.match(str(note_name).strip())
    if not m:
        raise ValueError(f"invalid note name: {note_name!r}")
    letter, acc, octave_s = m.group(1), m.group(2) or "", int(m.group(3))
    if acc == "b":
        idx = (NOTE_NAMES.index(letter) - 1) % 12
        letter = NOTE_NAMES[idx]
    elif acc == "#":
        pass
    elif acc:
        raise ValueError(f"unsupported accidental in {note_name!r}")
    if letter not in NOTE_NAMES:
        raise ValueError(f"unknown pitch class in {note_name!r}")
    midi = (octave_s + 1) * 12 + NOTE_NAMES.index(letter)
    return A4_REFERENCE_HZ * (2.0 ** ((midi - 69) / 12.0))


def frequency_to_note_name(hz: float) -> str:
    midi = 69.0 + 12.0 * math.log2(max(float(hz), 1e-9) / A4_REFERENCE_HZ)
    midi_round = int(round(midi))
    octave = (midi_round // 12) - 1
    return f"{NOTE_NAMES[midi_round % 12]}{octave}"


def parse_note_range(spec: str) -> List[str]:
    parts = str(spec).split(":")
    if len(parts) != 2:
        raise ValueError(f"note range must be LOW:HIGH, got {spec!r}")
    low_hz = note_name_to_frequency(parts[0].strip())
    high_hz = note_name_to_frequency(parts[1].strip())
    if high_hz < low_hz:
        raise ValueError(f"note range high < low: {spec!r}")
    low_midi = int(round(69 + 12 * math.log2(low_hz / A4_REFERENCE_HZ)))
    high_midi = int(round(69 + 12 * math.log2(high_hz / A4_REFERENCE_HZ)))
    return [frequency_to_note_name(A4_REFERENCE_HZ * (2.0 ** ((m - 69) / 12.0))) for m in range(low_midi, high_midi + 1)]


def stk_binary_path(repo_root: Optional[Path] = None) -> Path:
    return Path(repo_root or REPO_ROOT) / "cpp" / "stk_pgsm_guitar_demo" / "build" / "stk_pgsm_guitar_demo"


def note_cache_dir(
    sample_id: str,
    instrument: str = "classical",
    output_root: Optional[Path] = None,
) -> Path:
    return Path(output_root or APP_NOTE_CACHE_ROOT) / instrument / sample_id


def note_wav_path(
    sample_id: str,
    note_name: str,
    *,
    instrument: str = "classical",
    output_root: Optional[Path] = None,
) -> Path:
    return note_cache_dir(sample_id, instrument, output_root) / f"{note_name}.wav"


def library_report_paths(sample_id: str, instrument: str = "classical") -> Tuple[Path, Path]:
    stem = f"app_stk_note_library_{instrument}_{sample_id}"
    return DEBUG_REPORTS / f"{stem}_report.json", DEBUG_REPORTS / f"{stem}_report.md"


def list_available_samples(repo_root: Optional[Path] = None) -> List[str]:
    root = Path(repo_root or REPO_ROOT)
    lhs = root / "ROM" / "classic" / "lhs_pool.json"
    if lhs.is_file():
        try:
            pool = json.loads(lhs.read_text(encoding="utf-8"))
            ids = [str(e.get("id")) for e in pool.get("entries") or [] if str(e.get("id", "")).startswith("sample_")]
            if ids:
                return sorted(ids)
        except (json.JSONDecodeError, OSError):
            pass
    return list(SAMPLE_SET_V4)


def list_available_notes(sample_id: str, instrument: str = "classical", output_root: Optional[Path] = None) -> List[str]:
    cache = note_cache_dir(sample_id, instrument, output_root)
    if not cache.is_dir():
        return []
    return sorted(p.stem for p in cache.glob("*.wav") if p.is_file())


def get_note_wav(
    sample_id: str,
    note_name: str,
    *,
    instrument: str = "classical",
    output_root: Optional[Path] = None,
) -> Optional[Path]:
    path = note_wav_path(sample_id, note_name, instrument=instrument, output_root=output_root)
    return path if path.is_file() else None


def get_latest_note_library_report(sample_id: str, instrument: str = "classical") -> Optional[Dict[str, Any]]:
    json_path, _ = library_report_paths(sample_id, instrument)
    if not json_path.is_file():
        return None
    return json.loads(json_path.read_text(encoding="utf-8"))


def _stk_render_wav_name(note_name: str) -> str:
    return f"{note_name}_stk_guitar.wav"


def _build_single_note_export(
    *,
    repo_root: Path,
    sample_id: str,
    note_name: str,
    duration_s: float,
    render_subdir: str,
    stk_wav_relpath: str,
) -> Dict[str, Any]:
    physical = load_physical_parameters(sample_id)
    reference_physical = load_physical_parameters(REFERENCE_SAMPLE_ID)
    voicing_table = _extended_voicing((sample_id,))
    factors, _ = compute_v5_physical_factors(
        physical, reference_physical, sample_id=sample_id, voicing=voicing_table
    )
    mix_scales = _compute_v4_continuous_mix(physical, reference_physical, factors)
    freq = note_name_to_frequency(note_name)
    render = build_render_entry(
        sample_id,
        note_name,
        physical=physical,
        reference_physical=reference_physical,
        sample_rate=NUMERIC_SR,
        duration_s=duration_s,
        repo_root=repo_root,
        demo_version=ACCEPTED_STK_DEMO_VERSION,
        perceptual_mix=mix_scales,
        frequency_hz=freq,
        output_wav_relpath=stk_wav_relpath,
    )
    return {
        "export_version": "pgsm_stk_app_note_export_v1",
        "demo_version": "app_stk_note_cache_classical",
        "generated_at": _utc_now(),
        "renderer": "stk_cpp",
        "python_role": "parameter_export_only",
        "repo_root": str(repo_root),
        "audio_output_subdir": render_subdir,
        "sample_id": sample_id,
        "note_name": note_name,
        "physical_source": "audit_or_lhs_fallback",
        "renders": [render],
        "expected_render_count": 1,
    }


def invoke_stk_renderer(params_json: Path, repo_root: Path, binary: Optional[Path] = None) -> None:
    exe = Path(binary or stk_binary_path(repo_root))
    if not exe.is_file():
        raise FileNotFoundError(f"STK renderer binary not found: {exe}")
    cmd = [str(exe), "--params", str(params_json), "--repo-root", str(repo_root)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"STK renderer failed (exit {proc.returncode}): {proc.stderr or proc.stdout}"
        )


def render_single_note(
    *,
    repo_root: Path,
    sample_id: str,
    note_name: str,
    cache_path: Path,
    duration_s: float = DURATION_S,
    binary: Optional[Path] = None,
) -> float:
    """Export params, invoke STK/C++, copy WAV to cache. Returns render time in seconds."""
    tmp_dir = RENDER_TMP_ROOT / sample_id / note_name
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    rel_subdir = str(tmp_dir.relative_to(repo_root)).replace("\\", "/")
    stk_rel = f"{rel_subdir}/{_stk_render_wav_name(note_name)}"
    doc = _build_single_note_export(
        repo_root=repo_root,
        sample_id=sample_id,
        note_name=note_name,
        duration_s=duration_s,
        render_subdir=rel_subdir,
        stk_wav_relpath=stk_rel,
    )
    params_path = tmp_dir / "params.json"
    params_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    t0 = time.perf_counter()
    invoke_stk_renderer(params_path, repo_root, binary=binary)
    elapsed = time.perf_counter() - t0
    stk_out = repo_root / stk_rel
    if not stk_out.is_file():
        raise FileNotFoundError(f"STK did not produce expected WAV: {stk_out}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(stk_out, cache_path)
    return elapsed


def build_note_library(
    sample_id: str,
    *,
    instrument: str = "classical",
    note_range: str = "E2:E5",
    output_root: Optional[Path] = None,
    duration_s: float = DURATION_S,
    force: bool = False,
    repo_root: Optional[Path] = None,
    binary: Optional[Path] = None,
) -> Dict[str, Any]:
    root = Path(repo_root or REPO_ROOT)
    out_root = Path(output_root or APP_NOTE_CACHE_ROOT)
    notes = parse_note_range(note_range)
    cache_dir = note_cache_dir(sample_id, instrument, out_root)
    cache_dir.mkdir(parents=True, exist_ok=True)

    timings: Dict[str, float] = {}
    cache_hits = 0
    cache_misses = 0
    missing: List[str] = []
    physical = load_physical_parameters(sample_id)

    for note_name in notes:
        dest = note_wav_path(sample_id, note_name, instrument=instrument, output_root=out_root)
        if dest.is_file() and not force:
            cache_hits += 1
            timings[note_name] = 0.0
            continue
        cache_misses += 1
        try:
            timings[note_name] = render_single_note(
                repo_root=root,
                sample_id=sample_id,
                note_name=note_name,
                cache_path=dest,
                duration_s=duration_s,
                binary=binary,
            )
        except Exception:
            missing.append(note_name)
            timings[note_name] = -1.0

    rendered_times = {k: v for k, v in timings.items() if v > 0}
    total_render = sum(rendered_times.values())
    avg = total_render / len(rendered_times) if rendered_times else 0.0
    slowest = max(rendered_times, key=rendered_times.get) if rendered_times else None
    fastest = min(rendered_times, key=rendered_times.get) if rendered_times else None

    if missing:
        readiness = "generated_but_missing_notes"
    elif cache_misses == 0 or rendered_times:
        readiness = "ready_for_app_playback"
    else:
        readiness = "failed_renderer_or_export"

    report: Dict[str, Any] = {
        "generated_at": _utc_now(),
        "sample_id": sample_id,
        "instrument": instrument,
        "note_range": note_range,
        "note_count": len(notes),
        "notes_requested": notes,
        "missing_notes": missing,
        "total_render_time_s": round(total_render, 4),
        "average_time_per_note_s": round(avg, 4),
        "slowest_note": slowest,
        "slowest_time_s": round(rendered_times.get(slowest, 0.0), 4) if slowest else None,
        "fastest_note": fastest,
        "fastest_time_s": round(rendered_times.get(fastest, 0.0), 4) if fastest else None,
        "per_note_render_time_s": {k: round(v, 4) for k, v in timings.items()},
        "output_dir": str(cache_dir).replace("\\", "/"),
        "cache_hit_count": cache_hits,
        "cache_miss_count": cache_misses,
        "renderer": "STK/C++",
        "python_role": "parameter_export_only",
        "stk_demo_version": ACCEPTED_STK_DEMO_VERSION,
        "physical_source": "audit_or_lhs_fallback",
        "rom_physical_summary": {
            "body_depth_m": physical.get("body_depth_m"),
            "body_volume_proxy": physical.get("body_volume_proxy"),
            "soundhole_area": physical.get("soundhole_area") or physical.get("soundhole_area_proxy"),
            "bridge_mobility_proxy": physical.get("bridge_mobility_proxy"),
        },
        "readiness": readiness,
    }
    json_path, md_path = library_report_paths(sample_id, instrument)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_library_report_md(report), encoding="utf-8")
    report["report_json"] = str(json_path)
    report["report_md"] = str(md_path)
    return report


def _library_report_md(report: Mapping[str, Any]) -> str:
    lines = [
        f"# APP STK Note Library — {report.get('sample_id')}",
        "",
        f"- **instrument**: {report.get('instrument')}",
        f"- **note_range**: {report.get('note_range')}",
        f"- **note_count**: {report.get('note_count')}",
        f"- **total_render_time_s**: {report.get('total_render_time_s')}",
        f"- **average_time_per_note_s**: {report.get('average_time_per_note_s')}",
        f"- **cache_hits**: {report.get('cache_hit_count')}",
        f"- **cache_misses**: {report.get('cache_miss_count')}",
        f"- **readiness**: {report.get('readiness')}",
        f"- **output_dir**: `{report.get('output_dir')}`",
        "",
    ]
    if report.get("missing_notes"):
        lines.append("## Missing notes")
        for n in report["missing_notes"]:
            lines.append(f"- {n}")
    return "\n".join(lines) + "\n"


def get_melody_wav(
    sample_id: str,
    melody_id: str,
    *,
    instrument: str = "classical",
) -> Optional[Path]:
    path = MELODY_CACHE_ROOT / instrument / sample_id / f"{melody_id}.wav"
    return path if path.is_file() else None


def load_melody_library(path: Optional[Path] = None) -> Dict[str, Any]:
    p = Path(path or MELODY_LIBRARY_JSON)
    if not p.is_file():
        raise FileNotFoundError(f"melody library not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def list_melody_ids(path: Optional[Path] = None) -> List[str]:
    doc = load_melody_library(path)
    return [str(m.get("id")) for m in doc.get("melodies") or [] if m.get("id")]


def _stack_index_path(instrument: str = "classical") -> Path:
    return GUITAR_STACK_ROOT / instrument / "stack_index.json"


def load_guitar_stack(instrument: str = "classical") -> Dict[str, Any]:
    path = _stack_index_path(instrument)
    if not path.is_file():
        return {"max_snapshots": MAX_GUITAR_STACK, "snapshots": []}
    return json.loads(path.read_text(encoding="utf-8"))


def save_guitar_stack(doc: Dict[str, Any], instrument: str = "classical") -> Path:
    path = _stack_index_path(instrument)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return path


def push_guitar_snapshot(
    *,
    sample_id: str,
    display_name: str,
    instrument: str = "classical",
    physical_summary: Optional[Mapping[str, Any]] = None,
    note_cache_path: Optional[str] = None,
    timing_report_path: Optional[str] = None,
) -> Dict[str, Any]:
    """FIFO stack — keep latest 3 guitar configurations for comparison."""
    doc = load_guitar_stack(instrument)
    snapshots: List[Dict[str, Any]] = list(doc.get("snapshots") or [])
    entry = {
        "guitar_id": f"{sample_id}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "sample_id": sample_id,
        "timestamp": _utc_now(),
        "display_name": display_name,
        "instrument": instrument,
        "physical_summary": dict(physical_summary or {}),
        "note_cache_path": note_cache_path,
        "timing_report_path": timing_report_path,
        "stk_parameter_export": "pgsm_stk_app_note_export_v1",
        "renderer": "STK/C++",
    }
    snapshots.append(entry)
    max_n = int(doc.get("max_snapshots") or MAX_GUITAR_STACK)
    while len(snapshots) > max_n:
        snapshots.pop(0)
    doc["snapshots"] = snapshots
    doc["updated_at"] = _utc_now()
    save_guitar_stack(doc, instrument)
    return entry


def list_guitar_stack(instrument: str = "classical") -> List[Dict[str, Any]]:
    return list(load_guitar_stack(instrument).get("snapshots") or [])
