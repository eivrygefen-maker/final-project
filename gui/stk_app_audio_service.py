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
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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

from app_stk_config import load_app_stk_config, priority_notes_from_config
from app_stk_fretboard import (
    build_fretboard_note_mapping,
    build_required_note_set_from_fretboard,
    get_fret_count,
    list_ignored_non_note_wavs,
    list_note_wavs,
    normalize_note_name,
    note_range_label_from_required,
    note_to_midi,
    player_fretboard_metadata,
    run_fretboard_mapping_audit,
)

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
DEFAULT_NOTE_RANGE = "E2:E5"  # legacy chromatic range; player uses fretboard-derived set
DEFAULT_SOURCE_SAMPLE_ID = "sample_000"
DEFAULT_PRIORITY_NOTES: Tuple[str, ...] = ("A2", "A4", "E5")
CACHE_SPEC_FILE = ".cache_spec.json"
NOTE_CACHE_VERSION = "app_stk_v3_fretboard_json"
APP_STK_PARALLEL_WORKERS_ENABLED = True
DEFAULT_APP_STK_WORKERS = 3
AUDIT_INCOMPLETE_MSG = (
    "STK note cache is incomplete for this fretboard; rebuilding required."
)

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
    return Path(cache_dir) / f"{normalize_note_name(note_name)}.wav"


def list_notes_in_cache(cache_dir: Path) -> List[str]:
    return sorted(list_note_wavs(cache_dir).keys(), key=note_to_midi)


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


def parallel_workers_from_config(cfg: Optional[Mapping[str, Any]] = None) -> int:
    """Worker count for parallel STK note-cache rendering (safe fallback to 1)."""
    if not APP_STK_PARALLEL_WORKERS_ENABLED:
        return 1
    c = dict(cfg or load_app_stk_config())
    raw = c.get("parallel_workers", c.get("stk_note_render_workers", DEFAULT_APP_STK_WORKERS))
    try:
        count = int(raw)
    except (TypeError, ValueError):
        count = DEFAULT_APP_STK_WORKERS
    return max(1, count)


def split_notes_for_workers(notes: Sequence[str], worker_count: int) -> List[List[str]]:
    """Deterministic round-robin split — every note once, no duplicates."""
    normalized = [normalize_note_name(n) for n in notes]
    if not normalized:
        return []
    count = max(1, min(int(worker_count), len(normalized)))
    if count <= 1:
        return [list(normalized)]
    chunks: List[List[str]] = [[] for _ in range(count)]
    for idx, note in enumerate(normalized):
        chunks[idx % count].append(note)
    return chunks


def worker_render_tmp_dir(cache_key: str, worker_id: int) -> Path:
    return RENDER_TMP_ROOT / cache_key / f"worker_{worker_id}"


def parallel_staging_dir(cache_key: str) -> Path:
    return RENDER_TMP_ROOT / cache_key / "staging"


def _count_rendered_notes_in_worker_dirs(cache_key: str, worker_count: int) -> Tuple[int, List[Dict[str, Any]]]:
    """Count note WAVs per worker temp dir for progress reporting."""
    workers: List[Dict[str, Any]] = []
    total = 0
    for worker_id in range(worker_count):
        worker_dir = worker_render_tmp_dir(cache_key, worker_id)
        rendered = len(list_notes_in_cache(worker_dir)) if worker_dir.is_dir() else 0
        total += rendered
        workers.append(
            {
                "worker_id": worker_id,
                "status": "running" if worker_dir.is_dir() else "not_started",
                "rendered_notes": rendered,
                "output_dir": str(worker_dir).replace("\\", "/"),
            }
        )
    return total, workers


def _priority_notes_in_worker_dirs(cache_key: str, worker_count: int, priority_notes: Sequence[str]) -> bool:
    base = RENDER_TMP_ROOT / cache_key
    for note in priority_notes:
        found = any(note_wav_in_cache(base / f"worker_{wid}", note).is_file() for wid in range(worker_count))
        if not found:
            return False
    return True


def priority_notes_ready(
    cache_dir: Path,
    priority_notes: Sequence[str] = DEFAULT_PRIORITY_NOTES,
) -> bool:
    d = Path(cache_dir)
    if not d.is_dir():
        return False
    return all(note_wav_in_cache(d, n).is_file() for n in priority_notes)


def duration_for_note(note_name: str, cfg: Optional[Mapping[str, Any]] = None) -> float:
    """Per-note render duration from APP config (longer low notes, shorter high)."""
    c = dict(cfg or load_app_stk_config())
    midi = note_to_midi(normalize_note_name(note_name))
    if midi <= 45:
        return float(c.get("low_note_duration_s", 5.0))
    if midi >= 76:
        return float(c.get("high_note_duration_s", 3.8))
    return float(c.get("default_duration_s", 4.5))


def durations_fingerprint(notes: Sequence[str], cfg: Optional[Mapping[str, Any]] = None) -> str:
    ordered = sorted({normalize_note_name(n) for n in notes}, key=note_to_midi)
    parts = [f"{n}:{duration_for_note(n, cfg):.2f}" for n in ordered]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]


def compute_cache_spec_hash(
    parameter_hash: str,
    required_notes: Sequence[str],
    *,
    render_mode: str,
    durations_fp: str,
    parallel_workers: int = 1,
) -> str:
    payload = {
        "parameter_hash": parameter_hash,
        "renderer_version": ACCEPTED_STK_DEMO_VERSION,
        "note_cache_version": NOTE_CACHE_VERSION,
        "render_mode": render_mode,
        "parallel_workers": int(parallel_workers) if render_mode == "parallel_batch" else 1,
        "required_notes": sorted({normalize_note_name(n) for n in required_notes}, key=note_to_midi),
        "durations_fingerprint": durations_fp,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def build_cache_spec_for_hash(
    parameter_hash: str,
    cfg: Optional[Mapping[str, Any]] = None,
    fret_count: int = 19,
    *,
    render_mode: str = "",
) -> Dict[str, Any]:
    c = dict(cfg or load_app_stk_config())
    required = build_required_note_set_from_fretboard(fret_count)
    mode = str(render_mode or c.get("render_mode") or "batch")
    workers = parallel_workers_from_config(c) if mode == "parallel_batch" else 1
    durations_fp = durations_fingerprint(required, c)
    spec_hash = compute_cache_spec_hash(
        parameter_hash, required, render_mode=mode, durations_fp=durations_fp, parallel_workers=workers
    )
    return {
        "cache_spec_hash": spec_hash,
        "parameter_hash": parameter_hash,
        "renderer_version": ACCEPTED_STK_DEMO_VERSION,
        "note_cache_version": NOTE_CACHE_VERSION,
        "render_mode": mode,
        "parallel_workers": workers,
        "required_notes": required,
        "durations_fingerprint": durations_fp,
        "fretboard_required_note_count": len(required),
        "lowest_required_note": required[0] if required else "",
        "highest_required_note": required[-1] if required else "",
        "default_duration_s": c.get("default_duration_s"),
        "low_note_duration_s": c.get("low_note_duration_s"),
        "high_note_duration_s": c.get("high_note_duration_s"),
    }


def read_cache_spec(cache_dir: Path) -> Optional[Dict[str, Any]]:
    path = Path(cache_dir) / CACHE_SPEC_FILE
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def write_cache_spec(cache_dir: Path, spec: Mapping[str, Any]) -> Path:
    path = Path(cache_dir) / CACHE_SPEC_FILE
    _write_json(path, spec)
    return path


def cache_spec_is_compatible(cache_dir: Path, expected_spec_hash: str) -> bool:
    spec = read_cache_spec(cache_dir)
    if not spec:
        return False
    return str(spec.get("cache_spec_hash") or "") == str(expected_spec_hash)


def cache_is_ready_for_fretboard(
    cache_dir: Path,
    parameter_hash: str = "",
    *,
    cfg: Optional[Mapping[str, Any]] = None,
) -> bool:
    """True when cache spec matches and every fretboard note has a WAV."""
    cache_dir = Path(cache_dir)
    c = dict(cfg or load_app_stk_config())
    fret_count = int(c.get("fret_count") or 19)
    required = build_required_note_set_from_fretboard(fret_count)
    for note in required:
        if resolve_stk_note_wav(cache_dir, note) is None:
            return False
    if parameter_hash:
        expected = build_cache_spec_for_hash(parameter_hash, c, fret_count)
        if not cache_spec_is_compatible(cache_dir, expected["cache_spec_hash"]):
            # Repair spec when all required WAVs exist (e.g. render_mode migration).
            write_cache_spec(cache_dir, expected)
    return True


def preview_cache_dir_has_required_notes(
    cache_dir: Path,
    cfg: Optional[Mapping[str, Any]] = None,
) -> bool:
    """True when every fretboard-required note WAV exists (ignores cache spec)."""
    cache_dir = Path(cache_dir)
    c = dict(cfg or load_app_stk_config())
    fret_count = int(c.get("fret_count") or 19)
    required = build_required_note_set_from_fretboard(fret_count)
    return all(resolve_stk_note_wav(cache_dir, note) is not None for note in required)


def ensure_preview_cache_spec(
    cache_dir: Path,
    parameter_hash: str,
    *,
    cfg: Optional[Mapping[str, Any]] = None,
    render_mode: str = "",
) -> None:
    """Write or refresh ``.cache_spec.json`` for a complete preview cache directory."""
    c = dict(cfg or load_app_stk_config())
    fret_count = int(c.get("fret_count") or 19)
    mode = str(render_mode or c.get("render_mode") or "parallel_batch")
    spec = build_cache_spec_for_hash(parameter_hash, c, fret_count, render_mode=mode)
    write_cache_spec(cache_dir, spec)


def resolve_preview_cache_ready_state(
    parameter_hash: str,
    *,
    instrument: str = "classical",
    promote_stack: bool = True,
    repo_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Reconcile STK job status; promote to ready when fretboard WAVs exist on disk."""
    root = Path(repo_root or REPO_ROOT)
    cfg = load_app_stk_config(root)
    preview = preview_cache_dir(parameter_hash, instrument)
    state = refresh_stk_background_job_status(
        parameter_hash,
        instrument=instrument,
        promote_stack=promote_stack,
    )
    if state.get("preview_cache_ready"):
        return state
    if not preview.is_dir() or not preview_cache_dir_has_required_notes(preview, cfg):
        return state
    report = get_latest_note_library_report(
        DEFAULT_SOURCE_SAMPLE_ID, instrument, parameter_hash=parameter_hash
    )
    render_mode = str((report or {}).get("render_mode") or cfg.get("render_mode") or "parallel_batch")
    ensure_preview_cache_spec(
        preview,
        parameter_hash,
        cfg=cfg,
        render_mode=render_mode,
    )
    return refresh_stk_background_job_status(
        parameter_hash,
        instrument=instrument,
        promote_stack=promote_stack,
    )


def cache_is_ready(
    cache_dir: Path,
    *,
    note_range: str = DEFAULT_NOTE_RANGE,
    parameter_hash: str = "",
) -> bool:
    if parameter_hash:
        return cache_is_ready_for_fretboard(cache_dir, parameter_hash)
    notes = parse_note_range(note_range)
    return all(resolve_stk_note_wav(cache_dir, n) is not None for n in notes)


def note_mapping_audit_paths(parameter_hash: str) -> Tuple[Path, Path]:
    stem = f"app_stk_note_mapping_audit_{parameter_hash}"
    return DEBUG_REPORTS / f"{stem}.json", DEBUG_REPORTS / f"{stem}.md"


def run_note_mapping_audit(
    cache_dir: Path,
    parameter_hash: str = "",
    *,
    cfg: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Audit fretboard positions vs STK cache; writes JSON/MD report."""
    cache_dir = Path(cache_dir)
    c = dict(cfg or load_app_stk_config())
    fret_count = int(c.get("fret_count") or 19)
    mapping = build_fretboard_note_mapping(fret_count)
    required = build_required_note_set_from_fretboard(fret_count)
    note_wavs = list_note_wavs(cache_dir)
    generated = sorted(note_wavs.keys(), key=note_to_midi)
    ignored_non_note = list_ignored_non_note_wavs(cache_dir)
    missing_required: List[str] = []
    missing_positions: List[Dict[str, Any]] = []
    for row in mapping:
        note = str(row["note_name"])
        if resolve_stk_note_wav(cache_dir, note) is None:
            missing_positions.append(
                {
                    "string": row["string_number"],
                    "fret": row["fret"],
                    "note_name": note,
                }
            )
            if note not in missing_required:
                missing_required.append(note)
    required_set = set(required)
    generated_set = set(generated)
    audit: Dict[str, Any] = {
        "generated_at": _utc_now(),
        "parameter_hash": parameter_hash,
        "cache_dir": str(cache_dir).replace("\\", "/"),
        "passed": not missing_positions,
        "fretboard_required_note_count": len(required),
        "valid_note_wav_count": len(generated),
        "generated_note_count": len(generated),
        "ignored_non_note_wavs": ignored_non_note,
        "missing_required_notes": sorted(missing_required, key=note_to_midi),
        "extra_valid_notes": sorted(generated_set - required_set, key=note_to_midi),
        "extra_generated_notes": sorted(generated_set - required_set, key=note_to_midi),
        "lowest_required_note": required[0] if required else "",
        "highest_required_note": required[-1] if required else "",
        "s1_frets_13_19": [
            {"fret": r["fret"], "note_name": r["note_name"]}
            for r in mapping
            if int(r["string_number"]) == 1 and int(r["fret"]) >= 13
        ],
        "s2_frets_18_19": [
            {"fret": r["fret"], "note_name": r["note_name"]}
            for r in mapping
            if int(r["string_number"]) == 2 and int(r["fret"]) >= 18
        ],
        "missing_positions": missing_positions,
        "all_notes_preview_exists": (cache_dir / "all_notes_preview.wav").is_file(),
    }
    if parameter_hash:
        json_path, md_path = note_mapping_audit_paths(parameter_hash)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
        md_lines = [
            f"# APP STK Note Mapping Audit — `{parameter_hash}`",
            "",
            f"- **passed**: {audit['passed']}",
            f"- **fretboard_required_note_count**: {audit['fretboard_required_note_count']}",
            f"- **valid_note_wav_count**: {audit['valid_note_wav_count']}",
            f"- **ignored_non_note_wavs**: {audit['ignored_non_note_wavs']}",
            f"- **lowest_required_note**: {audit['lowest_required_note']}",
            f"- **highest_required_note**: {audit['highest_required_note']}",
            f"- **missing_required_notes**: {audit['missing_required_notes']}",
            "",
        ]
        md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
        audit["report_json"] = str(json_path).replace("\\", "/")
        audit["report_md"] = str(md_path).replace("\\", "/")
    return audit


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


def _build_batch_note_export(
    *,
    repo_root: Path,
    sample_id: str,
    notes: Sequence[str],
    durations_by_note: Mapping[str, float],
    render_subdir: str,
) -> Dict[str, Any]:
    physical = load_physical_parameters(sample_id)
    reference_physical = load_physical_parameters(REFERENCE_SAMPLE_ID)
    voicing_table = _extended_voicing((sample_id,))
    factors, _ = compute_v5_physical_factors(
        physical, reference_physical, sample_id=sample_id, voicing=voicing_table
    )
    mix_scales = _compute_v4_continuous_mix(physical, reference_physical, factors)
    renders: List[Dict[str, Any]] = []
    for note_name in notes:
        normalized = normalize_note_name(note_name)
        stk_rel = f"{render_subdir}/{_stk_render_wav_name(normalized)}"
        freq = note_name_to_frequency(normalized)
        duration_s = float(durations_by_note.get(normalized) or durations_by_note.get(note_name) or 4.5)
        renders.append(
            build_render_entry(
                sample_id,
                normalized,
                physical=physical,
                reference_physical=reference_physical,
                sample_rate=NUMERIC_SR,
                duration_s=duration_s,
                repo_root=repo_root,
                demo_version=ACCEPTED_STK_DEMO_VERSION,
                perceptual_mix=mix_scales,
                frequency_hz=freq,
                output_wav_relpath=stk_rel,
            )
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
        "physical_source": "audit_or_lhs_fallback",
        "renders": renders,
        "expected_render_count": len(renders),
    }


def render_notes_batch(
    *,
    repo_root: Path,
    sample_id: str,
    notes_to_render: Sequence[str],
    durations_by_note: Mapping[str, float],
    target_dir: Path,
    binary: Optional[Path] = None,
    cache_key: str = "",
) -> float:
    """Render all notes in one STK/C++ invocation."""
    if not notes_to_render:
        return 0.0
    root = Path(repo_root)
    key = cache_key or sample_id
    tmp_dir = RENDER_TMP_ROOT / key / "batch"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    rel_subdir = str(tmp_dir.relative_to(root)).replace("\\", "/")
    normalized_notes = [normalize_note_name(n) for n in notes_to_render]
    doc = _build_batch_note_export(
        repo_root=root,
        sample_id=sample_id,
        notes=normalized_notes,
        durations_by_note=durations_by_note,
        render_subdir=rel_subdir,
    )
    params_path = tmp_dir / "params.json"
    params_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    t0 = time.perf_counter()
    invoke_stk_renderer(params_path, root, binary=binary)
    elapsed = time.perf_counter() - t0
    target_dir.mkdir(parents=True, exist_ok=True)
    for note_name in normalized_notes:
        stk_rel = f"{rel_subdir}/{_stk_render_wav_name(note_name)}"
        stk_out = root / stk_rel
        if not stk_out.is_file():
            raise FileNotFoundError(f"STK did not produce expected WAV: {stk_out}")
        shutil.copy2(stk_out, note_wav_in_cache(target_dir, note_name))
    return elapsed


def render_notes_batch_to_worker_dir(
    *,
    repo_root: Path,
    sample_id: str,
    notes_to_render: Sequence[str],
    durations_by_note: Mapping[str, float],
    worker_dir: Path,
    binary: Optional[Path] = None,
) -> Tuple[float, int]:
    """Render a worker chunk into an isolated temp dir (flat ``{note}.wav`` files)."""
    if not notes_to_render:
        return 0.0, 0
    root = Path(repo_root)
    worker_dir = Path(worker_dir)
    if worker_dir.exists():
        shutil.rmtree(worker_dir)
    worker_dir.mkdir(parents=True, exist_ok=True)
    stk_work = worker_dir / "stk_work"
    stk_work.mkdir(parents=True, exist_ok=True)
    rel_subdir = str(stk_work.relative_to(root)).replace("\\", "/")
    normalized_notes = [normalize_note_name(n) for n in notes_to_render]
    doc = _build_batch_note_export(
        repo_root=root,
        sample_id=sample_id,
        notes=normalized_notes,
        durations_by_note=durations_by_note,
        render_subdir=rel_subdir,
    )
    params_path = stk_work / "params.json"
    params_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    t0 = time.perf_counter()
    invoke_stk_renderer(params_path, root, binary=binary)
    elapsed = time.perf_counter() - t0
    rendered = 0
    for note_name in normalized_notes:
        stk_rel = f"{rel_subdir}/{_stk_render_wav_name(note_name)}"
        stk_out = root / stk_rel
        if not stk_out.is_file():
            raise FileNotFoundError(f"STK did not produce expected WAV: {stk_out}")
        shutil.copy2(stk_out, note_wav_in_cache(worker_dir, note_name))
        rendered += 1
    return elapsed, rendered


def _merge_worker_outputs_to_staging(
    *,
    staging_dir: Path,
    cache_key: str,
    worker_count: int,
    target_dir: Path,
    cache_hit_notes: Sequence[str],
) -> None:
    """Copy cache hits + worker WAVs into staging (no concurrent writes to final cache)."""
    staging_dir = Path(staging_dir)
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    for note_name in cache_hit_notes:
        src = note_wav_in_cache(target_dir, note_name)
        if src.is_file():
            shutil.copy2(src, note_wav_in_cache(staging_dir, note_name))
    for worker_id in range(worker_count):
        worker_dir = worker_render_tmp_dir(cache_key, worker_id)
        if not worker_dir.is_dir():
            continue
        for note_name in list_notes_in_cache(worker_dir):
            src = note_wav_in_cache(worker_dir, note_name)
            dest = note_wav_in_cache(staging_dir, note_name)
            if src.is_file():
                shutil.copy2(src, dest)


def _promote_staging_to_target(staging_dir: Path, target_dir: Path) -> None:
    """Atomically promote validated staging cache into the final preview directory."""
    staging_dir = Path(staging_dir)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    for path in staging_dir.iterdir():
        if path.is_file():
            shutil.copy2(path, target_dir / path.name)


def render_notes_parallel_batch(
    *,
    repo_root: Path,
    sample_id: str,
    notes_to_render: Sequence[str],
    durations_by_note: Mapping[str, float],
    target_dir: Path,
    binary: Optional[Path] = None,
    cache_key: str = "",
    worker_count: int = DEFAULT_APP_STK_WORKERS,
    cache_hit_notes: Sequence[str] = (),
    parameter_hash: str = "",
    job_status_json: Optional[Path] = None,
    bg_status_path: Optional[Path] = None,
    total_notes: int = 0,
    cache_hits: int = 0,
    started_at: str = "",
    priority_notes: Sequence[str] = DEFAULT_PRIORITY_NOTES,
    t_start: float = 0.0,
) -> Tuple[float, List[str], List[Dict[str, Any]]]:
    """Render note chunks in parallel worker dirs, merge to staging, promote to target."""
    root = Path(repo_root)
    key = cache_key or sample_id
    worker_count = max(1, min(int(worker_count), len(notes_to_render) or 1))
    chunks = split_notes_for_workers(notes_to_render, worker_count)
    assigned_counts = [len(c) for c in chunks]

    workers_state: List[Dict[str, Any]] = [
        {
            "worker_id": idx,
            "status": "pending",
            "assigned_notes": assigned_counts[idx] if idx < len(assigned_counts) else 0,
            "rendered_notes": 0,
            "output_dir": str(worker_render_tmp_dir(key, idx)).replace("\\", "/"),
            "exit_code": None,
            "elapsed_s": 0.0,
            "error": "",
        }
        for idx in range(worker_count)
    ]
    progress_lock = threading.Lock()
    stop_progress = threading.Event()

    def _write_parallel_progress(job_status: str = "running") -> None:
        rendered_total, live_workers = _count_rendered_notes_in_worker_dirs(key, worker_count)
        with progress_lock:
            for idx, live in enumerate(live_workers):
                if idx < len(workers_state):
                    workers_state[idx]["rendered_notes"] = int(live.get("rendered_notes") or 0)
                    if workers_state[idx].get("status") == "running":
                        workers_state[idx]["status"] = "running"
            elapsed = round(time.perf_counter() - t_start, 3)
            if priority_notes and _priority_notes_in_worker_dirs(key, worker_count, priority_notes):
                if job_status == "running" and rendered_total < len(notes_to_render):
                    job_status = "partial_ready"
            progress_doc: Dict[str, Any] = {
                "parameter_hash": key,
                "status": job_status,
                "render_mode": "parallel_batch",
                "worker_count": worker_count,
                "rendered_notes": cache_hits + rendered_total,
                "total_notes": total_notes or (cache_hits + len(notes_to_render)),
                "output_dir": str(target_dir).replace("\\", "/"),
                "elapsed_time_s": elapsed,
                "elapsed_s": elapsed,
                "started_at": started_at,
                "updated_at": _utc_now(),
                "cache_hit_count": cache_hits,
                "cache_miss_count": len(notes_to_render),
                "workers": [dict(w) for w in workers_state],
            }
            if job_status_json is not None:
                _write_json(job_status_json, progress_doc)
            if bg_status_path is not None:
                write_background_status(key, progress_doc)
            if parameter_hash:
                write_job_status(key, {**read_job_status(key), **progress_doc})

    def _progress_loop() -> None:
        while not stop_progress.wait(5.0):
            _write_parallel_progress("running")

    def _run_worker(worker_id: int, chunk: Sequence[str]) -> Dict[str, Any]:
        worker_dir = worker_render_tmp_dir(key, worker_id)
        t_worker = time.perf_counter()
        with progress_lock:
            workers_state[worker_id]["status"] = "running"
        try:
            elapsed, rendered = render_notes_batch_to_worker_dir(
                repo_root=root,
                sample_id=sample_id,
                notes_to_render=chunk,
                durations_by_note=durations_by_note,
                worker_dir=worker_dir,
                binary=binary,
            )
            result = {
                "worker_id": worker_id,
                "status": "ready",
                "assigned_notes": len(chunk),
                "rendered_notes": rendered,
                "output_dir": str(worker_dir).replace("\\", "/"),
                "exit_code": 0,
                "elapsed_s": round(elapsed, 3),
                "error": "",
            }
        except Exception as exc:
            result = {
                "worker_id": worker_id,
                "status": "failed",
                "assigned_notes": len(chunk),
                "rendered_notes": len(list_notes_in_cache(worker_dir)) if worker_dir.is_dir() else 0,
                "output_dir": str(worker_dir).replace("\\", "/"),
                "exit_code": 1,
                "elapsed_s": round(time.perf_counter() - t_worker, 3),
                "error": str(exc),
            }
            raise
        finally:
            with progress_lock:
                workers_state[worker_id].update(result)
            _write_parallel_progress("running")
        return result

    _write_parallel_progress("running")
    progress_thread = threading.Thread(target=_progress_loop, daemon=True)
    progress_thread.start()
    total_elapsed = 0.0
    failed_workers: List[Dict[str, Any]] = []
    try:
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            futures = {
                pool.submit(_run_worker, idx, chunk): idx
                for idx, chunk in enumerate(chunks)
                if chunk
            }
            for fut in as_completed(futures):
                worker_id = futures[fut]
                try:
                    fut.result()
                except Exception as exc:
                    failed_workers.append({**workers_state[worker_id], "error": str(exc)})
    finally:
        stop_progress.set()
        progress_thread.join(timeout=2.0)

    if failed_workers:
        _write_parallel_progress("failed")
        missing = list(notes_to_render)
        return total_elapsed, missing, workers_state

    staging = parallel_staging_dir(key)
    _merge_worker_outputs_to_staging(
        staging_dir=staging,
        cache_key=key,
        worker_count=worker_count,
        target_dir=target_dir,
        cache_hit_notes=cache_hit_notes,
    )
    missing: List[str] = []
    for note_name in notes_to_render:
        if not note_wav_in_cache(staging, note_name).is_file():
            missing.append(note_name)
    if missing:
        _write_parallel_progress("failed")
        return total_elapsed, missing, workers_state

    cfg = load_app_stk_config(root)
    audit = run_note_mapping_audit(staging, parameter_hash or cache_key, cfg=cfg)
    if not audit.get("passed"):
        missing = sorted(audit.get("missing_required_notes") or missing, key=note_to_midi)
        _write_parallel_progress("failed")
        return total_elapsed, missing, workers_state

    _write_stk_preview_wav(staging, staging)
    _promote_staging_to_target(staging, target_dir)
    total_elapsed = round(time.perf_counter() - t_start, 3)
    for worker in workers_state:
        if worker.get("status") != "failed":
            worker["status"] = "ready"
    _write_parallel_progress("running")
    return total_elapsed, missing, workers_state


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
    duration_s: Optional[float] = None,
    binary: Optional[Path] = None,
    cache_key: str = "",
) -> float:
    normalized = normalize_note_name(note_name)
    dur = float(duration_s if duration_s is not None else duration_for_note(normalized))
    key = cache_key or sample_id
    tmp_dir = RENDER_TMP_ROOT / key / normalized
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    rel_subdir = str(tmp_dir.relative_to(repo_root)).replace("\\", "/")
    stk_rel = f"{rel_subdir}/{_stk_render_wav_name(normalized)}"
    doc = _build_single_note_export(
        repo_root=repo_root,
        sample_id=sample_id,
        note_name=normalized,
        duration_s=dur,
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
    note_range: str = "",
    output_root: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    duration_s: Optional[float] = None,
    force: bool = False,
    repo_root: Optional[Path] = None,
    binary: Optional[Path] = None,
    parameter_hash: Optional[str] = None,
    job_status_json: Optional[Path] = None,
    priority_notes: Optional[Sequence[str]] = None,
    render_mode: Optional[str] = None,
    parallel_workers: Optional[int] = None,
) -> Dict[str, Any]:
    root = Path(repo_root or REPO_ROOT)
    cfg = load_app_stk_config(root)
    if parallel_workers is not None:
        cfg = dict(cfg)
        cfg["parallel_workers"] = max(1, int(parallel_workers))
    fret_count = int(cfg.get("fret_count") or 19)
    prio = list(priority_notes or priority_notes_from_config(cfg))
    required = build_required_note_set_from_fretboard(fret_count)
    notes = order_notes_with_priority(required, prio)
    note_range = note_range or note_range_label_from_required(required)
    mode = str(render_mode or cfg.get("render_mode") or "batch")
    worker_count = parallel_workers_from_config(cfg)
    if mode == "parallel_batch" and worker_count <= 1:
        mode = "batch"
    elif mode == "batch" and worker_count > 1 and APP_STK_PARALLEL_WORKERS_ENABLED:
        mode = "parallel_batch"
    target_dir = Path(cache_dir) if cache_dir else note_cache_dir(sample_id, instrument, output_root)
    target_dir.mkdir(parents=True, exist_ok=True)
    cache_key = parameter_hash or sample_id
    spec_doc = build_cache_spec_for_hash(cache_key, cfg, fret_count, render_mode=mode)
    spec_hash = str(spec_doc["cache_spec_hash"])
    effective_workers = int(spec_doc.get("parallel_workers") or worker_count)
    bg_status_path = background_status_path(cache_key) if parameter_hash else None
    started_at = _utc_now()
    target_runtime_s = float(cfg.get("target_runtime_s") or 180)

    timings: Dict[str, float] = {}
    cache_hits = 0
    cache_misses = 0
    missing: List[str] = []
    physical = load_physical_parameters(sample_id)
    t_start = time.perf_counter()
    spec_ok = cache_spec_is_compatible(target_dir, spec_hash) and not force

    to_render: List[str] = []
    for note_name in notes:
        dest = note_wav_in_cache(target_dir, note_name)
        if spec_ok and dest.is_file():
            cache_hits += 1
            timings[note_name] = 0.0
        else:
            cache_misses += 1
            to_render.append(note_name)

    def _write_progress(rendered: int, job_status: str, current_note: str = "") -> None:
        elapsed = round(time.perf_counter() - t_start, 3)
        progress_doc = {
            "parameter_hash": cache_key,
            "status": job_status,
            "rendered_notes": rendered,
            "total_notes": len(notes),
            "current_note": current_note,
            "output_dir": str(target_dir).replace("\\", "/"),
            "elapsed_time_s": elapsed,
            "started_at": started_at,
            "cache_hit_count": cache_hits,
            "cache_miss_count": cache_misses,
            "render_mode": mode,
        }
        if job_status_json is not None:
            _write_json(job_status_json, {**progress_doc, "elapsed_s": elapsed})
        if bg_status_path is not None:
            write_background_status(cache_key, progress_doc)

    if bg_status_path is not None:
        initial_status: Dict[str, Any] = {
            "parameter_hash": cache_key,
            "status": "running",
            "rendered_notes": cache_hits,
            "total_notes": len(notes),
            "output_dir": str(target_dir).replace("\\", "/"),
            "elapsed_time_s": 0.0,
            "started_at": started_at,
            "render_mode": mode,
        }
        if mode == "parallel_batch" and to_render:
            initial_status["worker_count"] = effective_workers
            initial_status["workers"] = [
                {
                    "worker_id": idx,
                    "status": "pending",
                    "assigned_notes": len(chunk),
                    "rendered_notes": 0,
                    "output_dir": str(worker_render_tmp_dir(cache_key, idx)).replace("\\", "/"),
                }
                for idx, chunk in enumerate(split_notes_for_workers(to_render, effective_workers))
            ]
        write_background_status(cache_key, initial_status)
        if parameter_hash:
            write_job_status(cache_key, {**read_job_status(cache_key), **initial_status})

    workers_state: List[Dict[str, Any]] = []

    if to_render:
        if not is_active_job(cache_key) and parameter_hash:
            missing.extend(to_render)
            for note_name in to_render:
                timings[note_name] = -1.0
        elif mode == "parallel_batch":
            durations_by_note = {
                normalize_note_name(n): (
                    float(duration_s) if duration_s is not None else duration_for_note(n, cfg)
                )
                for n in to_render
            }
            cache_hit_notes = [n for n in notes if n not in to_render]
            try:
                parallel_elapsed, parallel_missing, workers_state = render_notes_parallel_batch(
                    repo_root=root,
                    sample_id=sample_id,
                    notes_to_render=to_render,
                    durations_by_note=durations_by_note,
                    target_dir=target_dir,
                    binary=binary,
                    cache_key=cache_key,
                    worker_count=effective_workers,
                    cache_hit_notes=cache_hit_notes,
                    parameter_hash=parameter_hash or cache_key,
                    job_status_json=job_status_json,
                    bg_status_path=bg_status_path,
                    total_notes=len(notes),
                    cache_hits=cache_hits,
                    started_at=started_at,
                    priority_notes=prio,
                    t_start=t_start,
                )
                per_note = parallel_elapsed / max(len(to_render), 1)
                for note_name in to_render:
                    dest = note_wav_in_cache(target_dir, note_name)
                    if dest.is_file():
                        timings[note_name] = per_note
                    else:
                        if note_name not in parallel_missing:
                            parallel_missing.append(note_name)
                        timings[note_name] = -1.0
                for note_name in parallel_missing:
                    if note_name not in missing:
                        missing.append(note_name)
            except Exception:
                for note_name in to_render:
                    if note_name not in missing:
                        missing.append(note_name)
                    timings[note_name] = -1.0
            _write_progress(len(notes) - len(missing), "running", "parallel_batch")
        elif mode == "batch":
            durations_by_note = {
                normalize_note_name(n): (
                    float(duration_s) if duration_s is not None else duration_for_note(n, cfg)
                )
                for n in to_render
            }
            try:
                batch_elapsed = render_notes_batch(
                    repo_root=root,
                    sample_id=sample_id,
                    notes_to_render=to_render,
                    durations_by_note=durations_by_note,
                    target_dir=target_dir,
                    binary=binary,
                    cache_key=cache_key,
                )
                per_note = batch_elapsed / max(len(to_render), 1)
                for note_name in to_render:
                    dest = note_wav_in_cache(target_dir, note_name)
                    if dest.is_file():
                        timings[note_name] = per_note
                    else:
                        missing.append(note_name)
                        timings[note_name] = -1.0
            except Exception:
                for note_name in to_render:
                    try:
                        dest = note_wav_in_cache(target_dir, note_name)
                        note_dur = float(duration_s) if duration_s is not None else duration_for_note(
                            note_name, cfg
                        )
                        timings[note_name] = render_single_note(
                            repo_root=root,
                            sample_id=sample_id,
                            note_name=note_name,
                            cache_path=dest,
                            duration_s=note_dur,
                            binary=binary,
                            cache_key=cache_key,
                        )
                    except Exception:
                        missing.append(note_name)
                        timings[note_name] = -1.0
            _write_progress(len(notes) - len(missing), "running", "batch")
        else:
            for idx, note_name in enumerate(to_render):
                dest = note_wav_in_cache(target_dir, note_name)
                note_dur = float(duration_s) if duration_s is not None else duration_for_note(
                    note_name, cfg
                )
                try:
                    timings[note_name] = render_single_note(
                        repo_root=root,
                        sample_id=sample_id,
                        note_name=note_name,
                        cache_path=dest,
                        duration_s=note_dur,
                        binary=binary,
                        cache_key=cache_key,
                    )
                except Exception:
                    missing.append(note_name)
                    timings[note_name] = -1.0
                job_status = (
                    "partial_ready"
                    if priority_notes_ready(target_dir, prio)
                    and not cache_is_ready_for_fretboard(target_dir, cache_key, cfg=cfg)
                    else "running"
                )
                _write_progress(cache_hits + idx + 1, job_status, note_name)

    if missing or not cache_is_ready_for_fretboard(target_dir, cache_key, cfg=cfg):
        pass  # skip preview until cache complete
    elif not (target_dir / "all_notes_preview.wav").is_file():
        _write_stk_preview_wav(target_dir, target_dir)

    write_cache_spec(target_dir, spec_doc)
    generated_in_cache = sorted(list_note_wavs(target_dir).keys(), key=note_to_midi)
    required_set = set(required)
    generated_set = set(generated_in_cache)

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
    elif cache_is_ready_for_fretboard(target_dir, cache_key, cfg=cfg):
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
        "render_mode": mode,
        "worker_count": effective_workers if mode == "parallel_batch" else 1,
        "workers": workers_state if workers_state else None,
        "finished_at": _utc_now(),
        "elapsed_s": round(time.perf_counter() - t_start, 3),
        "cache_spec_hash": spec_hash,
        "fretboard_required_note_count": len(required),
        "generated_note_count": len(generated_set),
        "missing_required_notes": sorted(required_set - generated_set, key=note_to_midi),
        "extra_generated_notes": sorted(generated_set - required_set, key=note_to_midi),
        "lowest_required_note": required[0] if required else "",
        "highest_required_note": required[-1] if required else "",
        "required_note_count": len(required),
        "target_runtime_s": target_runtime_s,
        "achieved_target": total_render <= target_runtime_s if total_render > 0 else True,
        "default_duration_s": cfg.get("default_duration_s"),
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
        f"- **render_mode**: {report.get('render_mode')}",
        f"- **fretboard_required_note_count**: {report.get('fretboard_required_note_count')}",
        f"- **highest_required_note**: {report.get('highest_required_note')}",
        f"- **total_render_time_s**: {report.get('total_render_time_s')}",
        f"- **achieved_target**: {report.get('achieved_target')}",
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
    note_range: str = "",
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
    if not preview_cache_dir_has_required_notes(output_dir):
        return False
    if report.get("cache_spec_hash"):
        expected = build_cache_spec_for_hash(parameter_hash)
        if str(report.get("cache_spec_hash")) != expected["cache_spec_hash"]:
            ensure_preview_cache_spec(
                output_dir,
                parameter_hash,
                render_mode=str(report.get("render_mode") or expected.get("render_mode") or ""),
            )
    if parameter_hash and not cache_spec_is_compatible(output_dir, build_cache_spec_for_hash(parameter_hash)["cache_spec_hash"]):
        ensure_preview_cache_spec(
            output_dir,
            parameter_hash,
            render_mode=str(report.get("render_mode") or ""),
        )
    expected = int(report.get("fretboard_required_note_count") or report.get("note_count") or 0)
    actual = count_wavs_in_cache(output_dir)
    if expected > 0:
        return actual >= expected
    return True


def refresh_stk_background_job_status(
    parameter_hash: str,
    *,
    note_range: str = "",
    priority_notes: Optional[Sequence[str]] = None,
    instrument: str = "classical",
    promote_stack: bool = True,
    cache_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Reconcile subprocess, progress JSON, library report, and WAV files into APP state."""
    cfg = load_app_stk_config()
    prio = list(priority_notes or priority_notes_from_config(cfg))
    fret_count = int(cfg.get("fret_count") or 19)
    required_notes = build_required_note_set_from_fretboard(fret_count)
    note_range = note_range or note_range_label_from_required(required_notes)
    total_notes = len(required_notes)

    preview = Path(cache_dir) if cache_dir is not None else preview_cache_dir(parameter_hash, instrument)

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
    reported_rendered = int(bg_doc.get("rendered_notes") or job_doc.get("rendered_notes") or 0)
    workers = job_doc.get("workers") or bg_doc.get("workers")
    worker_count = int(job_doc.get("worker_count") or bg_doc.get("worker_count") or 0)
    if isinstance(workers, list) and workers:
        worker_rendered = sum(int(w.get("rendered_notes") or 0) for w in workers)
        cache_hits_from_job = int(job_doc.get("cache_hit_count") or bg_doc.get("cache_hit_count") or 0)
        if worker_rendered > 0:
            reported_rendered = max(reported_rendered, cache_hits_from_job + worker_rendered)
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
    if isinstance(workers, list) and workers:
        result["workers"] = workers
        result["worker_count"] = worker_count or len(workers)
        result["render_mode"] = (
            job_doc.get("render_mode") or bg_doc.get("render_mode") or (report or {}).get("render_mode")
        )

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

    if cache_is_ready_for_fretboard(preview, parameter_hash, cfg=cfg):
        return _finalize_ready(preview)

    if cache_is_ready_for_fretboard(scan_dir, parameter_hash, cfg=cfg):
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
        if cache_is_ready_for_fretboard(preview, parameter_hash, cfg=cfg):
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
    priority_ok = priority_notes_ready(scan_dir, prio)
    full_ready = cache_is_ready_for_fretboard(preview, parameter_hash, cfg=cfg)

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

    if job_doc.get("status") == "ready" and cache_is_ready_for_fretboard(preview, parameter_hash, cfg=cfg):
        return _finalize_ready(preview)

    if (
        str(job_doc.get("status") or "") == "ready"
        or (
            report
            and str(report.get("parameter_hash") or parameter_hash) == parameter_hash
            and report.get("readiness") == "ready_for_app_playback"
            and report.get("status") == "ready"
        )
    ):
        for candidate in (report_out, preview, scan_dir):
            if candidate.is_dir() and preview_cache_dir_has_required_notes(candidate, cfg):
                if parameter_hash:
                    ensure_preview_cache_spec(
                        candidate,
                        parameter_hash,
                        cfg=cfg,
                        render_mode=str((report or {}).get("render_mode") or cfg.get("render_mode") or ""),
                    )
                return _finalize_ready(candidate)

    result["status"] = str(job_doc.get("status") or bg_doc.get("status") or "not_started")
    if result["status"] == "ready" and preview_cache_dir_has_required_notes(preview, cfg):
        return _finalize_ready(preview)
    return result


def start_background_note_library_job(
    *,
    parameter_hash: str,
    repo_root: Optional[Path] = None,
    sample_id: str = DEFAULT_SOURCE_SAMPLE_ID,
    note_range: str = "",
    instrument: str = "classical",
    render_mode: Optional[str] = None,
) -> Dict[str, Any]:
    root = Path(repo_root or REPO_ROOT)
    cfg = load_app_stk_config(root)
    fret_count = int(cfg.get("fret_count") or 19)
    required = build_required_note_set_from_fretboard(fret_count)
    note_range = note_range or note_range_label_from_required(required)
    mode = str(render_mode or cfg.get("render_mode") or "batch")
    if mode == "batch" and parallel_workers_from_config(cfg) > 1 and APP_STK_PARALLEL_WORKERS_ENABLED:
        mode = "parallel_batch"
    preview = preview_cache_dir(parameter_hash, instrument)
    preview.mkdir(parents=True, exist_ok=True)
    set_active_job(parameter_hash)

    if cache_is_ready_for_fretboard(preview, parameter_hash, cfg=cfg):
        report = {
            "status": "ready",
            "parameter_hash": parameter_hash,
            "cache_hit_count": len(required),
            "cache_miss_count": 0,
            "output_dir": str(preview),
            "readiness": "ready_for_app_playback",
            "fretboard_required_note_count": len(required),
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
        "--cache-dir",
        str(preview),
        "--parameter-hash",
        parameter_hash,
        "--job-status-json",
        str(job_json),
        "--repo-root",
        str(root),
        "--render-mode",
        mode,
        "--priority-notes",
        *priority_notes_from_config(cfg),
    ]
    write_job_status(
        parameter_hash,
        {
            "status": "running",
            "started_at": _utc_now(),
            "output_dir": str(preview),
            "source_sample_id": sample_id,
            "rendered_notes": 0,
            "total_notes": len(required),
            "render_mode": mode,
            "worker_count": parallel_workers_from_config(cfg) if mode == "parallel_batch" else 1,
        },
    )
    write_background_status(
        parameter_hash,
        {
            "status": "running",
            "started_at": _utc_now(),
            "output_dir": str(preview),
            "rendered_notes": 0,
            "total_notes": len(required),
            "elapsed_time_s": 0.0,
            "render_mode": mode,
            "worker_count": parallel_workers_from_config(cfg) if mode == "parallel_batch" else 1,
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
    """Save a fully ready guitar to the FIFO stack (ready entries only, no duplicates)."""
    root = Path(repo_root or REPO_ROOT)
    cfg = load_app_stk_config(root)
    if not cfg.get("enable_ready_fifo_stack", True):
        raise RuntimeError("FIFO stack is disabled in APP STK config.")
    existing = find_stack_entry_by_hash(parameter_hash, instrument)
    if existing:
        return {**existing, "_duplicate": True}

    state = resolve_preview_cache_ready_state(parameter_hash, instrument=instrument, promote_stack=False, repo_root=root)
    if parameter_hash != str(state.get("parameter_hash") or ""):
        raise RuntimeError("STK parameter hash mismatch — Save & Sync again.")

    stk_status = str(state.get("status") or "not_started")
    if stk_status == "failed":
        raise RuntimeError(
            f"STK audio rendering failed: {state.get('error') or 'see debug report'}"
        )
    if stk_status == "stale":
        raise RuntimeError("STK cache is stale for this design — Save & Sync again.")
    if stk_status != "ready" or not state.get("preview_cache_ready"):
        raise RuntimeError(
            "Guitar sound is still being prepared. Please wait a little longer."
        )

    saved_id = f"guitar_{parameter_hash}_{datetime.now(timezone.utc).strftime('%H%M%S')}"
    report = get_latest_note_library_report(source_sample_id, instrument, parameter_hash=parameter_hash)
    source = _ready_cache_source(state, parameter_hash, instrument)
    saved_dir = saved_guitar_cache_dir(saved_id, instrument)
    if saved_dir.exists():
        shutil.rmtree(saved_dir)
    shutil.copytree(source, saved_dir)

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
        "stk_parameter_json_path": state.get("latest_report_path")
        or (report.get("report_json") if report else None),
        "timing_report_path": state.get("latest_report_path")
        or (report.get("report_json") if report else None),
        "note_cache_path": str(saved_dir).replace("\\", "/"),
        "preview_image_path": None,
        "renderer": "STK/C++",
        "python_role": "parameter_export_only",
        "status": "ready",
    }

    doc = load_guitar_stack(instrument)
    snapshots: List[Dict[str, Any]] = [
        e for e in (doc.get("snapshots") or [])
        if str(e.get("status") or "ready") == "ready"
    ]
    snapshots.append(entry)
    max_n = int(doc.get("max_snapshots") or cfg.get("fifo_max_guitars") or MAX_GUITAR_STACK)
    while len(snapshots) > max_n:
        evicted = snapshots.pop(0)
        _remove_stack_cache(evicted.get("note_cache_path"))

    doc["snapshots"] = snapshots
    doc["updated_at"] = _utc_now()
    doc["active_saved_guitar_id"] = saved_id
    save_guitar_stack(doc, instrument)
    return entry


def find_stack_entry_by_hash(
    parameter_hash: str,
    instrument: str = "classical",
) -> Optional[Dict[str, Any]]:
    for entry in reversed(list_guitar_stack(instrument)):
        if str(entry.get("parameter_hash") or "") != parameter_hash:
            continue
        if str(entry.get("status") or "ready") != "ready":
            continue
        if not entry.get("note_cache_path"):
            continue
        return entry
    return None


def get_stack_entry(
    saved_guitar_id: str,
    instrument: str = "classical",
) -> Optional[Dict[str, Any]]:
    for entry in list_guitar_stack(instrument):
        if str(entry.get("saved_guitar_id") or "") == saved_guitar_id:
            return entry
    return None


def list_ready_guitar_stack(instrument: str = "classical") -> List[Dict[str, Any]]:
    """FIFO stack entries that are fully ready (user-facing list)."""
    return [
        e
        for e in list_guitar_stack(instrument)
        if str(e.get("status") or "ready") == "ready" and e.get("note_cache_path")
    ]


def user_facing_stk_status(internal_status: str) -> str:
    """Map internal STK job status to a short user-facing message."""
    status = str(internal_status or "not_started")
    if status in ("waiting_for_rom",):
        return "Waiting for guitar simulation"
    if status in ("not_started",):
        return "Click Generate Sound to build your guitar audio"
    if status in ("stale",):
        return "Design changed — click Generate Sound to rebuild audio"
    if status in ("running", "partial_ready"):
        return "Preparing guitar sound…"
    if status == "ready":
        return "Guitar sound is ready"
    if status == "failed":
        return "Sound preparation failed — retry Save & Sync"
    return "Preparing guitar sound…"


_ENHARMONIC_EQUIV: Dict[str, str] = {
    "Bb": "A#",
    "Db": "C#",
    "Eb": "D#",
    "Gb": "F#",
    "Ab": "G#",
    "A#": "Bb",
    "C#": "Db",
    "D#": "Eb",
    "F#": "Gb",
    "G#": "Ab",
}


def _note_name_variants(note_name: str) -> List[str]:
    """Return enharmonic spellings for STK cache lookup (Bb <-> A#, etc.)."""
    variants: List[str] = [str(note_name).strip()]
    m = _NOTE_RE.match(str(note_name).strip())
    if not m:
        return variants
    letter, acc, octave = m.group(1), m.group(2) or "", m.group(3)
    pitch = f"{letter}{acc}"
    alt = _ENHARMONIC_EQUIV.get(pitch)
    if alt:
        variants.append(f"{alt}{octave}")
    return list(dict.fromkeys(variants))


def resolve_stk_note_wav(cache_dir: Path, note_name: str) -> Optional[Path]:
    """Resolve a STK source WAV, trying sharp/flat spellings."""
    root = Path(cache_dir)
    candidates = [normalize_note_name(note_name)]
    candidates.extend(_note_name_variants(note_name))
    for variant in dict.fromkeys(candidates):
        path = root / f"{variant}.wav"
        if path.is_file():
            return path
    return None


def write_minimal_silent_wav(
    path: Path,
    *,
    duration_s: float = 0.25,
    sample_rate: int = 44100,
) -> Path:
    """Write a tiny valid mono PCM WAV (used for preview fallback / tests)."""
    import wave

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nframes = max(1, int(sample_rate * duration_s))
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * nframes)
    return path


def _stitch_preview_wav(sources: Sequence[Path], dest: Path) -> Path:
    """Concatenate WAVs into all_notes_preview.wav; silent fallback on failure."""
    import wave

    dest = Path(dest)
    usable = [Path(s) for s in sources if Path(s).is_file()]
    if not usable:
        return write_minimal_silent_wav(dest)

    try:
        with wave.open(str(usable[0]), "rb") as first:
            params = first.getparams()
            frames = [first.readframes(first.getnframes())]
        for src in usable[1:]:
            with wave.open(str(src), "rb") as wf:
                if wf.getparams()[:3] != params[:3]:
                    continue
                frames.append(wf.readframes(wf.getnframes()))
        dest.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(dest), "wb") as out:
            out.setparams(params)
            out.writeframes(b"".join(frames))
        return dest
    except (wave.Error, OSError):
        return write_minimal_silent_wav(dest)


def _write_stk_preview_wav(dest_dir: Path, cache_dir: Path) -> Path:
    """Create all_notes_preview.wav for the HTML guitar player."""
    preview_path = Path(dest_dir) / "all_notes_preview.wav"
    stitch_sources: List[Path] = []
    for note_name in DEFAULT_PRIORITY_NOTES:
        src = resolve_stk_note_wav(cache_dir, note_name)
        if src is not None:
            stitch_sources.append(src)
    if not stitch_sources:
        for path in sorted(Path(cache_dir).glob("*.wav"))[:3]:
            stitch_sources.append(path)
    return _stitch_preview_wav(stitch_sources, preview_path)


def prepare_stk_player_assets(cache_dir: Path, fingerprint: str) -> Path:
    """Copy STK note WAVs into the guitar_player runtime folder for iframe playback."""
    from note_cache_ui import RUNTIME_CACHE_DIR  # noqa: WPS433

    cache_dir = Path(cache_dir)
    dest = RUNTIME_CACHE_DIR / fingerprint
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    copied_ids: set[str] = set()
    for row in build_fretboard_note_mapping():
        note_name = str(row["note_name"])
        note_id = str(row["note_id"])
        src = resolve_stk_note_wav(cache_dir, note_name)
        if src is not None and note_id not in copied_ids:
            shutil.copy2(src, dest / f"{note_id}.wav")
            copied_ids.add(note_id)

    _write_stk_preview_wav(dest, cache_dir)
    return dest


def build_stk_player_payload(
    cache_dir: Path,
    *,
    fingerprint: str,
    ui_status: str = "ready",
    fret_count: Optional[int] = None,
) -> Dict[str, Any]:
    """Build guitar_player component payload from an STK note cache directory."""
    cfg = load_app_stk_config()
    fc = int(fret_count if fret_count is not None else get_fret_count())
    if ui_status not in ("ready", "building"):
        return {"status": ui_status, "positions": [], "fingerprint": ""}

    cache_dir = Path(cache_dir)
    positions: List[Dict[str, Any]] = []
    unique_ids: set[str] = set()
    for row in build_fretboard_note_mapping(fc):
        note_name = str(row["note_name"])
        note_id = str(row["note_id"])
        if resolve_stk_note_wav(cache_dir, note_name) is None:
            continue
        positions.append(
            {
                "string": int(row["string_number"]),
                "fret": int(row["fret"]),
                "note_name": note_name,
                "note_id": note_id,
                "wav": f"{note_id}.wav",
            }
        )
        unique_ids.add(note_id)

    if not positions:
        return {"status": "hidden", "positions": [], "fingerprint": ""}

    fretboard_meta = player_fretboard_metadata()
    return {
        "status": "ready",
        "fingerprint": fingerprint,
        "fret_count": fc,
        "unique_note_count": len(unique_ids),
        "playable_position_count": len(positions),
        "positions": positions,
        "enable_overlapping_playback": bool(cfg.get("enable_overlapping_playback", True)),
        "fretboard": fretboard_meta,
        "string_visual_order_numbers": fretboard_meta.get("string_visual_order_numbers"),
        "preview_wav": "all_notes_preview.wav",
    }


def validate_stk_player_runtime_cache(
    payload: Mapping[str, Any],
    *,
    runtime_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Verify runtime cache files exist for mapped playable notes (not helper previews)."""
    from classical_guitar_fretboard import is_note_wav_path  # noqa: WPS433
    from note_cache_ui import RUNTIME_CACHE_DIR  # noqa: WPS433

    fingerprint = str(payload.get("fingerprint") or "")
    runtime = Path(runtime_dir) if runtime_dir is not None else RUNTIME_CACHE_DIR / fingerprint
    positions = list(payload.get("positions") or [])
    errors: List[str] = []
    playable_errors: List[str] = []

    if not runtime.is_dir():
        errors.append(f"runtime cache dir missing: {runtime}")

    for pos in positions:
        wav_name = str(pos.get("wav") or "")
        if not wav_name:
            playable_errors.append("position missing wav mapping")
            continue
        wav_path = runtime / wav_name
        if not is_note_wav_path(wav_path):
            playable_errors.append(f"mapped wav is not a note file: {wav_name}")
            continue
        if not wav_path.is_file():
            playable_errors.append(f"missing mapped wav: {wav_name}")

    errors.extend(playable_errors)

    preview_wav = payload.get("preview_wav")
    preview_path = runtime / str(preview_wav) if preview_wav else None
    preview_missing = False
    if preview_wav:
        if preview_path is None or not preview_path.is_file():
            errors.append(f"missing preview wav: {preview_wav}")
            preview_missing = True

    return {
        "ok": not errors and len(positions) > 0,
        "errors": errors,
        "playable_errors": playable_errors,
        "runtime_dir": str(runtime).replace("\\", "/"),
        "position_count": len(positions),
        "preview_wav": str(preview_wav) if preview_wav else "",
        "preview_path": str(preview_path).replace("\\", "/")
        if preview_path is not None and preview_path.is_file()
        else "",
        "preview_missing": preview_missing,
    }


def activate_stk_guitar_for_player(
    *,
    cache_dir: Path,
    parameter_hash: str,
    saved_guitar_id: str = "",
) -> Dict[str, Any]:
    """Stage STK cache for the HTML fretboard player."""
    cache_dir = Path(cache_dir)
    mapping_audit = run_fretboard_mapping_audit(cache_dir=cache_dir)
    if mapping_audit.get("readiness") == "failed_wrong_note_mapping":
        validation = {
            "ok": False,
            "errors": ["Fretboard mapping validation failed — see app_stk_fretboard_mapping_audit.json"],
            "runtime_dir": "",
            "position_count": 0,
            "preview_path": "",
            "fretboard_audit": mapping_audit,
        }
        return {
            "cache_path": str(cache_dir).replace("\\", "/"),
            "parameter_hash": parameter_hash,
            "saved_guitar_id": saved_guitar_id,
            "player_fingerprint": saved_guitar_id or f"stk_{parameter_hash}",
            "player_payload": {"status": "hidden", "positions": [], "fingerprint": ""},
            "validation": validation,
            "runtime_dir": "",
            "fretboard_audit": mapping_audit,
        }

    audit = run_note_mapping_audit(cache_dir, parameter_hash)
    player_fp = saved_guitar_id or f"stk_{parameter_hash}"
    if not audit.get("passed"):
        validation = {
            "ok": False,
            "errors": [AUDIT_INCOMPLETE_MSG],
            "runtime_dir": "",
            "position_count": 0,
            "preview_path": "",
            "audit": audit,
        }
        return {
            "cache_path": str(cache_dir).replace("\\", "/"),
            "parameter_hash": parameter_hash,
            "saved_guitar_id": saved_guitar_id,
            "player_fingerprint": player_fp,
            "player_payload": {"status": "hidden", "positions": [], "fingerprint": player_fp},
            "validation": validation,
            "runtime_dir": "",
            "audit": audit,
        }

    runtime_dir = prepare_stk_player_assets(cache_dir, player_fp)
    payload = build_stk_player_payload(cache_dir, fingerprint=player_fp, ui_status="ready")
    run_fretboard_mapping_audit(cache_dir=cache_dir, player_payload=payload)
    validation = validate_stk_player_runtime_cache(payload, runtime_dir=runtime_dir)
    if not validation.get("ok"):
        payload = {"status": "hidden", "positions": [], "fingerprint": player_fp}
        validation = {**validation, "errors": validation.get("errors", []) + [AUDIT_INCOMPLETE_MSG]}
    return {
        "cache_path": str(cache_dir).replace("\\", "/"),
        "parameter_hash": parameter_hash,
        "saved_guitar_id": saved_guitar_id,
        "player_fingerprint": player_fp,
        "player_payload": payload,
        "validation": validation,
        "runtime_dir": str(runtime_dir).replace("\\", "/"),
        "audit": audit,
    }


def load_stack_guitar_for_player(
    saved_guitar_id: str,
    instrument: str = "classical",
) -> Dict[str, Any]:
    """Load a saved FIFO guitar into the fretboard player."""
    entry = get_stack_entry(saved_guitar_id, instrument)
    if not entry:
        raise RuntimeError("Saved guitar not found.")
    if str(entry.get("status") or "") != "ready":
        raise RuntimeError("Saved guitar is not ready for playback.")
    cache_path = Path(str(entry.get("note_cache_path") or ""))
    if not cache_path.is_dir():
        raise RuntimeError("Saved guitar cache is missing.")
    return activate_stk_guitar_for_player(
        cache_dir=cache_path,
        parameter_hash=str(entry.get("parameter_hash") or ""),
        saved_guitar_id=saved_guitar_id,
    )


def list_guitar_stack(instrument: str = "classical") -> List[Dict[str, Any]]:
    return list(load_guitar_stack(instrument).get("snapshots") or [])


def build_note_library_for_cli(**kwargs) -> Dict[str, Any]:
    """Backward-compatible alias used by CLI tools."""
    return build_note_library(**kwargs)
