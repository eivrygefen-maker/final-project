#!/usr/bin/env python3
"""
STK classical guitar integration for the Streamlit APP.

Python exports parameters only; STK/C++ renders WAVs.
Background jobs render preview caches after ROM completion.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from pgsm_stk_parameter_export import (
    DURATION_S,
    NUMERIC_SR,
    SAMPLE_SET_V4,
    _compute_v4_continuous_mix,
    _extended_voicing,
    build_render_entry,
    load_physical_parameters,
)
from pgsm_emergency_guitar_demo_engine import compute_v5_physical_factors
from pgsm_step5l_limited_multiguitar_differentiation import REFERENCE_SAMPLE_ID

REPO_ROOT = Path(__file__).resolve().parents[1]
STK_BINARY = REPO_ROOT / "cpp" / "stk_pgsm_guitar_demo" / "build" / "stk_pgsm_guitar_demo"
APP_NOTE_CACHE_ROOT = REPO_ROOT / "audio" / "app_stk_note_cache"
GUITAR_STACK_ROOT = REPO_ROOT / "audio" / "app_stk_guitar_stack"
RENDER_TMP_ROOT = REPO_ROOT / "audio" / "app_stk_note_cache" / ".render_tmp"
DEBUG_REPORTS = REPO_ROOT / "audio" / "debug_reports"
ACTIVE_JOB_FILE = APP_NOTE_CACHE_ROOT / "classical" / ".active_job.json"

NOTE_NAMES: Tuple[str, ...] = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
A4_REFERENCE_HZ = 440.0
ACCEPTED_STK_DEMO_VERSION = "v4_10_samples"
MAX_GUITAR_STACK = 3
DEFAULT_NOTE_RANGE = "E2:E5"
DEFAULT_SOURCE_SAMPLE_ID = "sample_000"
DEFAULT_PRIORITY_NOTES: Tuple[str, ...] = ("A2", "A4", "E5")

STK_JOB_STATUSES = (
    "not_started",
    "waiting_for_rom",
    "running",
    "partial_ready",
    "ready",
    "failed",
    "stale",
)

STACK_ENTRY_STATUSES = (
    "pending_audio",
    "ready",
    "failed_audio",
    "stale",
)

_NOTE_RE = re.compile(r"^([A-G])(#|b)?(\d+)$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def compute_parameter_hash(rom_fp: str, lhs_params: Optional[Mapping[str, Any]] = None) -> str:
    payload: Dict[str, Any] = {"rom_fp": str(rom_fp)}
    if lhs_params:
        payload["lhs"] = lhs_params
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return digest[:16]


def preview_cache_dir(parameter_hash: str, instrument: str = "classical") -> Path:
    return APP_NOTE_CACHE_ROOT / instrument / f"current_preview_{parameter_hash}"


def smoke_test_cache_dir(parameter_hash: str, instrument: str = "classical") -> Path:
    """Isolated preview cache path for site smoke tests (never used by the live APP)."""
    return APP_NOTE_CACHE_ROOT / instrument / f"current_preview_smoke_test_{parameter_hash}"


def smoke_test_artifact_paths(parameter_hash: str, instrument: str = "classical") -> List[Path]:
    json_path, md_path = library_report_paths_for_hash(parameter_hash, instrument)
    return [
        smoke_test_cache_dir(parameter_hash, instrument),
        job_status_path(parameter_hash),
        background_status_path(parameter_hash),
        json_path,
        md_path,
    ]


def cleanup_smoke_test_artifacts(parameter_hash: str, instrument: str = "classical") -> None:
    """Remove smoke-only cache/report/status files for one test hash."""
    for path in smoke_test_artifact_paths(parameter_hash, instrument):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.is_file():
            try:
                path.unlink()
            except OSError:
                pass


def saved_guitar_cache_dir(saved_guitar_id: str, instrument: str = "classical") -> Path:
    return APP_NOTE_CACHE_ROOT / instrument / f"saved_{saved_guitar_id}"


def job_status_path(parameter_hash: str) -> Path:
    return DEBUG_REPORTS / f"app_stk_background_job_{parameter_hash}.json"


def background_status_path(parameter_hash: str) -> Path:
    return DEBUG_REPORTS / f"app_stk_background_status_{parameter_hash}.json"


def library_report_paths_for_hash(parameter_hash: str, instrument: str = "classical") -> Tuple[Path, Path]:
    stem = f"app_stk_note_library_{instrument}_preview_{parameter_hash}"
    return DEBUG_REPORTS / f"{stem}_report.json", DEBUG_REPORTS / f"{stem}_report.md"


def note_name_to_frequency(note_name: str) -> float:
    m = _NOTE_RE.match(str(note_name).strip())
    if not m:
        raise ValueError(f"invalid note name: {note_name!r}")
    letter, acc, octave_s = m.group(1), m.group(2) or "", int(m.group(3))
    if acc == "b":
        letter = NOTE_NAMES[(NOTE_NAMES.index(letter) - 1) % 12]
    if letter not in NOTE_NAMES:
        raise ValueError(f"unknown pitch class in {note_name!r}")
    midi = (octave_s + 1) * 12 + NOTE_NAMES.index(letter)
    return A4_REFERENCE_HZ * (2.0 ** ((midi - 69) / 12.0))


def frequency_to_note_name(hz: float) -> str:
    midi = 69.0 + 12.0 * math.log2(max(float(hz), 1e-9) / A4_REFERENCE_HZ)
    midi_round = int(round(midi))
    octave = (midi_round // 12) - 1
    return f"{NOTE_NAMES[midi_round % 12]}{octave}"


def order_notes_with_priority(
    notes: Sequence[str],
    priority_notes: Sequence[str] = DEFAULT_PRIORITY_NOTES,
) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for note_name in priority_notes:
        if note_name in notes and note_name not in seen:
            ordered.append(note_name)
            seen.add(note_name)
    for note_name in notes:
        if note_name not in seen:
            ordered.append(note_name)
            seen.add(note_name)
    return ordered


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
    return [
        frequency_to_note_name(A4_REFERENCE_HZ * (2.0 ** ((m - 69) / 12.0)))
        for m in range(low_midi, high_midi + 1)
    ]


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


def note_wav_in_cache(cache_dir: Path, note_name: str) -> Path:
    return Path(cache_dir) / f"{note_name}.wav"


def list_notes_in_cache(cache_dir: Path) -> List[str]:
    d = Path(cache_dir)
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob("*.wav") if p.is_file())


def list_available_samples(repo_root: Optional[Path] = None) -> List[str]:
    root = Path(repo_root or REPO_ROOT)
    lhs = root / "ROM" / "classic" / "lhs_pool.json"
    if lhs.is_file():
        try:
            pool = json.loads(lhs.read_text(encoding="utf-8"))
            ids = [
                str(e.get("id"))
                for e in pool.get("entries") or []
                if str(e.get("id", "")).startswith("sample_")
            ]
            if ids:
                return sorted(ids)
        except (json.JSONDecodeError, OSError):
            pass
    return list(SAMPLE_SET_V4)


def get_note_wav(
    sample_id: str,
    note_name: str,
    *,
    instrument: str = "classical",
    output_root: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
) -> Optional[Path]:
    if cache_dir is not None:
        path = note_wav_in_cache(cache_dir, note_name)
    else:
        path = note_wav_path(sample_id, note_name, instrument=instrument, output_root=output_root)
    return path if path.is_file() else None


def list_available_notes(
    sample_id: str,
    instrument: str = "classical",
    output_root: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
) -> List[str]:
    if cache_dir is not None:
        return list_notes_in_cache(cache_dir)
    return list_notes_in_cache(note_cache_dir(sample_id, instrument, output_root))


def count_wavs_in_cache(cache_dir: Path) -> int:
    return len(list_notes_in_cache(cache_dir))


def priority_notes_ready(
    cache_dir: Path,
    priority_notes: Sequence[str] = DEFAULT_PRIORITY_NOTES,
) -> bool:
    d = Path(cache_dir)
    if not d.is_dir():
        return False
    return all(note_wav_in_cache(d, n).is_file() for n in priority_notes)


def cache_is_ready(
    cache_dir: Path,
    *,
    note_range: str = DEFAULT_NOTE_RANGE,
) -> bool:
    notes = parse_note_range(note_range)
    return all(note_wav_in_cache(cache_dir, n).is_file() for n in notes)


def _write_json(path: Path, doc: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(doc), indent=2) + "\n", encoding="utf-8")


def read_job_status(parameter_hash: str) -> Dict[str, Any]:
    path = job_status_path(parameter_hash)
    if not path.is_file():
        return {"parameter_hash": parameter_hash, "status": "not_started"}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"parameter_hash": parameter_hash, "status": "failed", "error": "corrupt_job_status"}


def write_job_status(parameter_hash: str, doc: Mapping[str, Any]) -> Path:
    path = job_status_path(parameter_hash)
    payload = dict(doc)
    payload["parameter_hash"] = parameter_hash
    payload["updated_at"] = _utc_now()
    _write_json(path, payload)
    return path


def read_background_status(parameter_hash: str) -> Dict[str, Any]:
    path = background_status_path(parameter_hash)
    if not path.is_file():
        return {"parameter_hash": parameter_hash, "status": "not_started"}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"parameter_hash": parameter_hash, "status": "failed", "error": "corrupt_background_status"}


def write_background_status(parameter_hash: str, doc: Mapping[str, Any]) -> Path:
    path = background_status_path(parameter_hash)
    payload = dict(doc)
    payload["parameter_hash"] = parameter_hash
    payload["updated_at"] = _utc_now()
    _write_json(path, payload)
    return path


def set_active_job(parameter_hash: str) -> None:
    ACTIVE_JOB_FILE.parent.mkdir(parents=True, exist_ok=True)
    _write_json(
        ACTIVE_JOB_FILE,
        {"parameter_hash": parameter_hash, "updated_at": _utc_now()},
    )


def get_active_job_hash() -> Optional[str]:
    if not ACTIVE_JOB_FILE.is_file():
        return None
    try:
        doc = json.loads(ACTIVE_JOB_FILE.read_text(encoding="utf-8"))
        return str(doc.get("parameter_hash") or "") or None
    except (json.JSONDecodeError, OSError):
        return None


def is_active_job(parameter_hash: str) -> bool:
    return get_active_job_hash() == parameter_hash


def mark_stk_job_stale(parameter_hash: Optional[str] = None) -> None:
    active = parameter_hash or get_active_job_hash()
    if not active:
        return
    status = read_job_status(active)
    if status.get("status") == "running":
        write_job_status(active, {**status, "status": "stale", "stale_reason": "design_or_rom_changed"})
    elif status.get("status") not in ("ready",):
        write_job_status(active, {"status": "stale", "stale_reason": "design_or_rom_changed"})


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
    cache_key: str = "",
) -> float:
    key = cache_key or sample_id
    tmp_dir = RENDER_TMP_ROOT / key / note_name
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
    note_range: str = DEFAULT_NOTE_RANGE,
    output_root: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    duration_s: float = DURATION_S,
    force: bool = False,
    repo_root: Optional[Path] = None,
    binary: Optional[Path] = None,
    parameter_hash: Optional[str] = None,
    job_status_json: Optional[Path] = None,
    priority_notes: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    root = Path(repo_root or REPO_ROOT)
    notes = order_notes_with_priority(
        parse_note_range(note_range),
        priority_notes or DEFAULT_PRIORITY_NOTES,
    )
    target_dir = Path(cache_dir) if cache_dir else note_cache_dir(sample_id, instrument, output_root)
    target_dir.mkdir(parents=True, exist_ok=True)
    cache_key = parameter_hash or sample_id
    bg_status_path = background_status_path(cache_key) if parameter_hash else None
    started_at = _utc_now()

    timings: Dict[str, float] = {}
    cache_hits = 0
    cache_misses = 0
    missing: List[str] = []
    physical = load_physical_parameters(sample_id)
    t_start = time.perf_counter()

    def _write_progress(idx: int, note_name: str, job_status: str) -> None:
        elapsed = round(time.perf_counter() - t_start, 3)
        progress_doc = {
            "parameter_hash": cache_key,
            "status": job_status,
            "rendered_notes": idx + 1,
            "total_notes": len(notes),
            "current_note": note_name,
            "output_dir": str(target_dir).replace("\\", "/"),
            "elapsed_time_s": elapsed,
            "started_at": started_at,
            "cache_hit_count": cache_hits,
            "cache_miss_count": cache_misses,
        }
        if job_status_json is not None:
            _write_json(job_status_json, {**progress_doc, "elapsed_s": elapsed})
        if bg_status_path is not None:
            write_background_status(cache_key, progress_doc)

    if bg_status_path is not None:
        write_background_status(
            cache_key,
            {
                "parameter_hash": cache_key,
                "status": "running",
                "rendered_notes": 0,
                "total_notes": len(notes),
                "output_dir": str(target_dir).replace("\\", "/"),
                "elapsed_time_s": 0.0,
                "started_at": started_at,
            },
        )

    for idx, note_name in enumerate(notes):
        dest = note_wav_in_cache(target_dir, note_name)
        if dest.is_file() and not force:
            cache_hits += 1
            timings[note_name] = 0.0
        else:
            cache_misses += 1
            if not is_active_job(cache_key) and parameter_hash:
                missing.append(note_name)
                timings[note_name] = -1.0
                continue
            try:
                timings[note_name] = render_single_note(
                    repo_root=root,
                    sample_id=sample_id,
                    note_name=note_name,
                    cache_path=dest,
                    duration_s=duration_s,
                    binary=binary,
                    cache_key=cache_key,
                )
            except Exception:
                missing.append(note_name)
                timings[note_name] = -1.0

        job_status = "partial_ready" if priority_notes_ready(target_dir) and not cache_is_ready(
            target_dir, note_range=note_range
        ) else "running"
        _write_progress(idx, note_name, job_status)

    rendered_times = {k: v for k, v in timings.items() if v > 0}
    total_render = sum(rendered_times.values())
    avg = total_render / len(rendered_times) if rendered_times else 0.0
    slowest = max(rendered_times, key=rendered_times.get) if rendered_times else None
    fastest = min(rendered_times, key=rendered_times.get) if rendered_times else None

    if parameter_hash and not is_active_job(parameter_hash):
        readiness = "stale"
        job_status = "stale"
    elif missing:
        readiness = "generated_but_missing_notes"
        job_status = "failed"
    elif cache_is_ready(target_dir, note_range=note_range):
        readiness = "ready_for_app_playback"
        job_status = "ready"
    else:
        readiness = "failed_renderer_or_export"
        job_status = "failed"

    report: Dict[str, Any] = {
        "generated_at": _utc_now(),
        "sample_id": sample_id,
        "parameter_hash": parameter_hash,
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
        "output_dir": str(target_dir).replace("\\", "/"),
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
        "status": job_status,
    }
    if parameter_hash:
        json_path, md_path = library_report_paths_for_hash(parameter_hash, instrument)
    else:
        json_path = DEBUG_REPORTS / f"app_stk_note_library_{instrument}_{sample_id}_report.json"
        md_path = DEBUG_REPORTS / f"app_stk_note_library_{instrument}_{sample_id}_report.md"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_library_report_md(report), encoding="utf-8")
    report["report_json"] = str(json_path)
    report["report_md"] = str(md_path)
    if job_status_json is not None:
        write_job_status(
            cache_key,
            {
                **report,
                "status": job_status,
                "rendered_notes": len(notes) - len(missing),
                "total_notes": len(notes),
            },
        )
    if bg_status_path is not None:
        write_background_status(
            cache_key,
            {
                **report,
                "status": job_status,
                "rendered_notes": len(notes) - len(missing),
                "total_notes": len(notes),
                "elapsed_time_s": round(total_render, 3),
                "report_path": str(json_path),
                "output_dir": str(target_dir).replace("\\", "/"),
            },
        )
    return report


def _library_report_md(report: Mapping[str, Any]) -> str:
    lines = [
        f"# APP STK Note Library — {report.get('sample_id')}",
        "",
        f"- **parameter_hash**: {report.get('parameter_hash')}",
        f"- **note_range**: {report.get('note_range')}",
        f"- **total_render_time_s**: {report.get('total_render_time_s')}",
        f"- **readiness**: {report.get('readiness')}",
        f"- **output_dir**: `{report.get('output_dir')}`",
        "",
    ]
    return "\n".join(lines) + "\n"


def _is_process_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        if sys.platform == "win32":
            import ctypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not handle:
                return False
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        import os

        os.kill(pid, 0)
        return True
    except (OSError, AttributeError):
        return False


def poll_background_job(parameter_hash: str) -> Dict[str, Any]:
    """Backward-compatible alias — always refreshes from disk/subprocess state."""
    return refresh_stk_background_job_status(parameter_hash)


def _report_ready_for_hash(
    report: Optional[Mapping[str, Any]],
    parameter_hash: str,
    output_dir: Path,
    *,
    note_range: str = DEFAULT_NOTE_RANGE,
) -> bool:
    """True when library report + WAV count confirm ready for the requested hash."""
    if not report:
        return False
    rep_hash = str(report.get("parameter_hash") or parameter_hash)
    if rep_hash != parameter_hash:
        return False
    if report.get("readiness") != "ready_for_app_playback":
        return False
    if report.get("status") != "ready":
        return False
    if not output_dir.is_dir():
        return False
    expected = int(report.get("note_count") or 0)
    actual = count_wavs_in_cache(output_dir)
    if expected > 0:
        return actual >= expected
    return cache_is_ready(output_dir, note_range=note_range)


def refresh_stk_background_job_status(
    parameter_hash: str,
    *,
    note_range: str = DEFAULT_NOTE_RANGE,
    priority_notes: Sequence[str] = DEFAULT_PRIORITY_NOTES,
    instrument: str = "classical",
    promote_stack: bool = True,
    cache_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Reconcile subprocess, progress JSON, library report, and WAV files into APP state."""
    preview = Path(cache_dir) if cache_dir is not None else preview_cache_dir(parameter_hash, instrument)
    expected_notes = parse_note_range(note_range)
    total_notes = len(expected_notes)

    job_doc = read_job_status(parameter_hash)
    bg_doc = read_background_status(parameter_hash)
    report = get_latest_note_library_report(
        DEFAULT_SOURCE_SAMPLE_ID, instrument, parameter_hash=parameter_hash
    )
    report_json_path, _ = library_report_paths_for_hash(parameter_hash, instrument)
    latest_report_path = ""
    if report and report.get("report_json"):
        latest_report_path = str(report["report_json"])
    elif report_json_path.is_file():
        latest_report_path = str(report_json_path)

    pid = int(job_doc.get("pid") or bg_doc.get("pid") or 0)
    proc_alive = _is_process_running(pid) if pid else False
    output_dir = Path(
        str(
            (report or {}).get("output_dir")
            or bg_doc.get("output_dir")
            or job_doc.get("output_dir")
            or preview
        )
    )
    scan_dir = output_dir if output_dir.is_dir() else preview
    actual_wav_count = count_wavs_in_cache(scan_dir)
    reported_rendered = int(bg_doc.get("rendered_notes") or job_doc.get("rendered_notes") or 0)

    active_hash = get_active_job_hash()
    is_active_hash = active_hash is None or active_hash == parameter_hash

    result: Dict[str, Any] = {
        "parameter_hash": parameter_hash,
        "preview_cache_path": str(preview).replace("\\", "/"),
        "output_dir": str(scan_dir).replace("\\", "/"),
        "actual_wav_count": actual_wav_count,
        "reported_rendered_notes": reported_rendered,
        "wav_count": actual_wav_count,
        "note_count": total_notes,
        "total_notes": total_notes,
        "rendered_notes": reported_rendered if reported_rendered > 0 else actual_wav_count,
        "current_note": bg_doc.get("current_note") or job_doc.get("current_note"),
        "elapsed_time_s": bg_doc.get("elapsed_time_s") or job_doc.get("elapsed_s"),
        "elapsed_s": bg_doc.get("elapsed_time_s") or job_doc.get("elapsed_s"),
        "started_at": job_doc.get("started_at") or bg_doc.get("started_at"),
        "latest_report_path": latest_report_path,
        "preview_cache_ready": False,
        "is_active_hash": is_active_hash,
        "cache_hit_count": (report or {}).get("cache_hit_count", job_doc.get("cache_hit_count", 0)),
        "cache_miss_count": (report or {}).get("cache_miss_count", job_doc.get("cache_miss_count", 0)),
    }

    def _sync_counts(ready_dir: Path) -> None:
        actual = count_wavs_in_cache(ready_dir)
        result["actual_wav_count"] = actual
        result["wav_count"] = actual
        result["rendered_notes"] = total_notes
        result["reported_rendered_notes"] = reported_rendered

    def _finalize_ready(ready_dir: Path) -> Dict[str, Any]:
        _sync_counts(ready_dir)
        result.update(
            {
                "status": "ready",
                "preview_cache_ready": True,
                "readiness": "ready_for_app_playback",
                "note_count": total_notes,
                "output_dir": str(ready_dir).replace("\\", "/"),
                "preview_cache_path": str(ready_dir).replace("\\", "/"),
                "report_path": latest_report_path,
                "pid": None,
            }
        )
        write_job_status(parameter_hash, {**job_doc, **result})
        write_background_status(parameter_hash, {**result, "report_path": latest_report_path})
        if promote_stack:
            promote_pending_stack_entries(parameter_hash, instrument=instrument)
        return result

    # 1) Ready promotion from this hash's own report/WAVs (independent of active session job).
    report_out = Path(str((report or {}).get("output_dir") or preview))
    if _report_ready_for_hash(report, parameter_hash, report_out, note_range=note_range):
        return _finalize_ready(report_out)

    if cache_is_ready(preview, note_range=note_range):
        return _finalize_ready(preview)

    if cache_is_ready(scan_dir, note_range=note_range):
        return _finalize_ready(scan_dir)

    # 2) Explicit failed report for this hash.
    if report and str(report.get("parameter_hash") or parameter_hash) == parameter_hash:
        if report.get("status") in ("failed",):
            result["status"] = "failed"
            result["error"] = report.get("missing_notes") or report.get("error")
            return result

    # 3) Subprocess exited — re-check report/files before failing.
    if job_doc.get("status") == "running" and pid and not proc_alive:
        if _report_ready_for_hash(report, parameter_hash, report_out, note_range=note_range):
            return _finalize_ready(report_out)
        if cache_is_ready(preview, note_range=note_range):
            return _finalize_ready(preview)
        if actual_wav_count > 0 and is_active_hash:
            pass  # fall through to partial/running handling
        else:
            result["status"] = "failed"
            result["error"] = "subprocess_exited_incomplete"
            write_job_status(parameter_hash, {**job_doc, **result, "pid": None})
            write_background_status(parameter_hash, result)
            return result

    # 4) Partial / in-progress on matching active hash (or no competing active job).
    priority_ok = priority_notes_ready(scan_dir, priority_notes)
    full_ready = cache_is_ready(preview, note_range=note_range)

    if priority_ok and not full_ready and (is_active_hash or actual_wav_count > 0):
        result["status"] = "partial_ready"
        result["priority_notes_ready"] = True
        result["preview_cache_ready"] = False
        result["rendered_notes"] = reported_rendered if reported_rendered > 0 else actual_wav_count
        result["wav_count"] = actual_wav_count
        return result

    if actual_wav_count > 0 and not full_ready and is_active_hash:
        result["status"] = "running" if (proc_alive or job_doc.get("status") == "running") else "partial_ready"
        result["rendered_notes"] = reported_rendered if reported_rendered > 0 else actual_wav_count
        result["wav_count"] = actual_wav_count
        return result

    if proc_alive or (job_doc.get("status") == "running" and is_active_hash):
        result["status"] = "running"
        result["rendered_notes"] = reported_rendered if reported_rendered > 0 else actual_wav_count
        result["wav_count"] = actual_wav_count
        return result

    # 5) Abandoned partial cache while another hash is active → stale (informational).
    if active_hash and active_hash != parameter_hash:
        if actual_wav_count > 0 or job_doc.get("status") in ("running", "partial_ready"):
            result["status"] = "stale"
            result["stale_reason"] = "hash_mismatch_active_job"
            result["active_job_hash"] = active_hash
            return result
        result["status"] = "stale"
        result["stale_reason"] = "hash_mismatch_active_job"
        result["active_job_hash"] = active_hash
        return result

    if job_doc.get("status") == "ready" and cache_is_ready(preview, note_range=note_range):
        return _finalize_ready(preview)

    result["status"] = str(job_doc.get("status") or bg_doc.get("status") or "not_started")
    return result


def start_background_note_library_job(
    *,
    parameter_hash: str,
    repo_root: Optional[Path] = None,
    sample_id: str = DEFAULT_SOURCE_SAMPLE_ID,
    note_range: str = DEFAULT_NOTE_RANGE,
    instrument: str = "classical",
) -> Dict[str, Any]:
    root = Path(repo_root or REPO_ROOT)
    preview = preview_cache_dir(parameter_hash, instrument)
    preview.mkdir(parents=True, exist_ok=True)
    set_active_job(parameter_hash)

    if cache_is_ready(preview, note_range=note_range):
        report = {
            "status": "ready",
            "parameter_hash": parameter_hash,
            "cache_hit_count": len(parse_note_range(note_range)),
            "cache_miss_count": 0,
            "output_dir": str(preview),
            "readiness": "ready_for_app_playback",
        }
        write_job_status(parameter_hash, report)
        return report

    existing = read_job_status(parameter_hash)
    pid = int(existing.get("pid") or 0)
    if existing.get("status") == "running" and pid and _is_process_running(pid):
        return existing

    script = root / "tools" / "build_app_stk_note_library.py"
    job_json = job_status_path(parameter_hash)
    cmd = [
        sys.executable,
        str(script),
        "--sample-id",
        sample_id,
        "--instrument",
        instrument,
        "--note-range",
        note_range,
        "--cache-dir",
        str(preview),
        "--parameter-hash",
        parameter_hash,
        "--job-status-json",
        str(job_json),
        "--repo-root",
        str(root),
        "--priority-notes",
        *DEFAULT_PRIORITY_NOTES,
    ]
    write_job_status(
        parameter_hash,
        {
            "status": "running",
            "started_at": _utc_now(),
            "output_dir": str(preview),
            "source_sample_id": sample_id,
            "rendered_notes": 0,
            "total_notes": len(parse_note_range(note_range)),
        },
    )
    write_background_status(
        parameter_hash,
        {
            "status": "running",
            "started_at": _utc_now(),
            "output_dir": str(preview),
            "rendered_notes": 0,
            "total_notes": len(parse_note_range(note_range)),
            "elapsed_time_s": 0.0,
        },
    )
    proc = subprocess.Popen(cmd, cwd=str(root))
    write_job_status(parameter_hash, {**read_job_status(parameter_hash), "pid": proc.pid})
    return read_job_status(parameter_hash)


def schedule_stk_after_rom(
    *,
    rom_fp: str,
    lhs_params: Mapping[str, Any],
    repo_root: Optional[Path] = None,
    sample_id: str = DEFAULT_SOURCE_SAMPLE_ID,
) -> Dict[str, Any]:
    parameter_hash = compute_parameter_hash(rom_fp, lhs_params)
    mark_stk_job_stale()
    set_active_job(parameter_hash)
    return start_background_note_library_job(
        parameter_hash=parameter_hash,
        repo_root=repo_root,
        sample_id=sample_id,
    )


def get_latest_note_library_report(
    sample_id: str,
    instrument: str = "classical",
    parameter_hash: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if parameter_hash:
        json_path, _ = library_report_paths_for_hash(parameter_hash, instrument)
    else:
        json_path = DEBUG_REPORTS / f"app_stk_note_library_{instrument}_{sample_id}_report.json"
    if not json_path.is_file():
        return None
    return json.loads(json_path.read_text(encoding="utf-8"))


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


def _remove_stack_cache(path_str: Optional[str]) -> None:
    if not path_str:
        return
    p = Path(path_str)
    if p.is_dir() and "saved_" in p.name:
        shutil.rmtree(p, ignore_errors=True)


def _ready_cache_source(
    state: Mapping[str, Any],
    parameter_hash: str,
    instrument: str = "classical",
) -> Path:
    """Resolve the directory that holds ready WAVs for a hash."""
    for key in ("preview_cache_path", "output_dir"):
        candidate = Path(str(state.get(key) or ""))
        if candidate.is_dir() and count_wavs_in_cache(candidate) > 0:
            return candidate
    return preview_cache_dir(parameter_hash, instrument)


def promote_pending_stack_entries(
    parameter_hash: Optional[str] = None,
    instrument: str = "classical",
) -> List[Dict[str, Any]]:
    """Attach preview cache to pending FIFO entries when STK becomes ready."""
    doc = load_guitar_stack(instrument)
    snapshots: List[Dict[str, Any]] = list(doc.get("snapshots") or [])
    promoted: List[Dict[str, Any]] = []
    changed = False

    for entry in snapshots:
        if str(entry.get("status") or "") != "pending_audio":
            continue
        entry_hash = str(entry.get("parameter_hash") or "")
        if parameter_hash and entry_hash != parameter_hash:
            continue
        if not entry_hash:
            entry["status"] = "stale"
            changed = True
            continue

        state = refresh_stk_background_job_status(
            entry_hash, instrument=instrument, promote_stack=False
        )
        if entry_hash != str(state.get("parameter_hash") or ""):
            entry["status"] = "stale"
            changed = True
            continue

        stk_status = str(state.get("status") or "")
        if stk_status == "failed":
            entry["status"] = "failed_audio"
            entry["error"] = state.get("error")
            changed = True
            continue
        if stk_status != "ready" or not state.get("preview_cache_ready"):
            continue

        source = _ready_cache_source(state, entry_hash, instrument)
        if not source.is_dir() or count_wavs_in_cache(source) == 0:
            continue

        saved_id = str(entry.get("saved_guitar_id") or "")
        if not saved_id:
            continue
        saved_dir = saved_guitar_cache_dir(saved_id, instrument)
        if saved_dir.exists():
            shutil.rmtree(saved_dir)
        shutil.copytree(source, saved_dir)

        report = get_latest_note_library_report(
            str(entry.get("sample_id") or DEFAULT_SOURCE_SAMPLE_ID),
            instrument,
            parameter_hash=entry_hash,
        )
        entry["status"] = "ready"
        entry["note_cache_path"] = str(saved_dir).replace("\\", "/")
        entry["timing_report_path"] = (
            state.get("latest_report_path")
            or (report.get("report_json") if report else None)
        )
        entry["promoted_at"] = _utc_now()
        promoted.append(entry)
        changed = True

    if changed:
        doc["snapshots"] = snapshots
        doc["updated_at"] = _utc_now()
        save_guitar_stack(doc, instrument)
    return promoted


def save_guitar_to_stack(
    *,
    parameter_hash: str,
    display_name: str,
    instrument: str = "classical",
    source_sample_id: str = DEFAULT_SOURCE_SAMPLE_ID,
    geometry_summary: Optional[Mapping[str, Any]] = None,
    rom_physical_summary_path: Optional[str] = None,
    repo_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Save current guitar to FIFO stack as ready or pending_audio."""
    _ = repo_root
    state = refresh_stk_background_job_status(parameter_hash, instrument=instrument, promote_stack=False)
    if parameter_hash != str(state.get("parameter_hash") or ""):
        raise RuntimeError("STK parameter hash mismatch — Save & Sync again.")

    stk_status = str(state.get("status") or "not_started")

    if stk_status == "failed":
        raise RuntimeError(
            f"STK audio rendering failed: {state.get('error') or 'see debug report'}"
        )
    if stk_status == "stale":
        raise RuntimeError("STK cache is stale for this design — Save & Sync again.")

    saved_id = f"guitar_{parameter_hash}_{datetime.now(timezone.utc).strftime('%H%M%S')}"
    report = get_latest_note_library_report(source_sample_id, instrument, parameter_hash=parameter_hash)

    entry: Dict[str, Any] = {
        "saved_guitar_id": saved_id,
        "guitar_id": saved_id,
        "sample_id": source_sample_id,
        "timestamp": _utc_now(),
        "display_name": display_name,
        "instrument": instrument,
        "parameter_hash": parameter_hash,
        "geometry_summary": dict(geometry_summary or {}),
        "rom_physical_summary_path": rom_physical_summary_path,
        "stk_parameter_export": "pgsm_stk_app_note_export_v1",
        "timing_report_path": state.get("latest_report_path")
        or (report.get("report_json") if report else None),
        "renderer": "STK/C++",
        "python_role": "parameter_export_only",
    }

    if stk_status == "ready" and state.get("preview_cache_ready"):
        source = _ready_cache_source(state, parameter_hash, instrument)
        saved_dir = saved_guitar_cache_dir(saved_id, instrument)
        if saved_dir.exists():
            shutil.rmtree(saved_dir)
        shutil.copytree(source, saved_dir)
        entry["status"] = "ready"
        entry["note_cache_path"] = str(saved_dir).replace("\\", "/")
    elif stk_status == "partial_ready":
        entry["status"] = "partial_audio"
        entry["note_cache_path"] = None
    elif stk_status in ("running", "not_started"):
        entry["status"] = "pending_audio"
        entry["note_cache_path"] = None
    else:
        raise RuntimeError(
            f"Cannot save guitar while STK status is {stk_status!r}."
        )

    doc = load_guitar_stack(instrument)
    snapshots: List[Dict[str, Any]] = list(doc.get("snapshots") or [])
    snapshots.append(entry)
    max_n = int(doc.get("max_snapshots") or MAX_GUITAR_STACK)
    while len(snapshots) > max_n:
        evicted = snapshots.pop(0)
        if str(evicted.get("status") or "") == "ready":
            _remove_stack_cache(evicted.get("note_cache_path"))

    doc["snapshots"] = snapshots
    doc["updated_at"] = _utc_now()
    doc["active_saved_guitar_id"] = saved_id
    save_guitar_stack(doc, instrument)
    return entry


def list_guitar_stack(instrument: str = "classical") -> List[Dict[str, Any]]:
    return list(load_guitar_stack(instrument).get("snapshots") or [])


def build_note_library_for_cli(**kwargs) -> Dict[str, Any]:
    """Backward-compatible alias used by CLI tools."""
    return build_note_library(**kwargs)
