#!/usr/bin/env python3
"""
Note-cache helpers for Stage 4 interactive guitar player (manifest, assets, no synthesis on click).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from build_note_cache import (
    DEFAULT_DURATION_S,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_TUNING,
    build_note_cache,
    compute_guitar_fingerprint,
    file_sha256,
    optional_geometry_fingerprint,
    tuning_open_frequencies,
)

from classical_guitar_fretboard import (
    get_fret_count,
    get_tuning,
    string_key_to_number,
    string_visual_order_numbers,
)

NOTE_CACHE_ROOT_NAME = "note_cache"
DEFAULT_FRET_COUNT = get_fret_count()

# Display order: low E (string 6) at top; nut / open strings on the right column.
FRETBOARD_DISPLAY_STRING_ORDER: Tuple[int, ...] = tuple(string_visual_order_numbers())
OPEN_STRING_NOTE_IDS: Dict[int, str] = {
    string_key_to_number(k): v for k, v in get_tuning().items()
}

_GUITAR_PLAYER_COMPONENT_DIR = Path(__file__).resolve().parent / "components" / "guitar_player"
RUNTIME_CACHE_DIR = _GUITAR_PLAYER_COMPONENT_DIR / "runtime_cache"


def note_cache_root(repo_root: Path) -> Path:
    return Path(repo_root) / "audio" / NOTE_CACHE_ROOT_NAME


def list_manifest_paths(out_root: Path) -> List[Path]:
    root = Path(out_root)
    if not root.is_dir():
        return []
    manifests = [p for p in root.glob("*/note_manifest.json") if p.is_file()]
    manifests.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return manifests


def load_note_manifest(manifest_path: Path) -> Dict[str, Any]:
    path = Path(manifest_path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ValueError(f"expected object in {path}")
    doc["_manifest_path"] = str(path.resolve())
    doc["_cache_root"] = str(path.parent.resolve())
    return doc


def build_position_lookup(manifest: Mapping[str, Any]) -> Dict[Tuple[int, int], Dict[str, Any]]:
    lookup: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for pos in manifest.get("positions") or []:
        key = (int(pos["string_number"]), int(pos["fret"]))
        lookup[key] = dict(pos)
    return lookup


def resolve_wav_path(cache_root: Path, wav_rel: str) -> Path:
    return (Path(cache_root) / wav_rel).resolve()


def expected_note_cache_fingerprint(
    *,
    modal_json: Path,
    geometry_config: Optional[Path] = None,
    fret_count: int = DEFAULT_FRET_COUNT,
    duration_s: float = DEFAULT_DURATION_S,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    stk_mode_alias: Optional[str] = None,
) -> Optional[str]:
    modal_json = Path(modal_json)
    if not modal_json.is_file():
        return None
    modal_sha = file_sha256(modal_json)
    geom_fp = optional_geometry_fingerprint(geometry_config)
    return compute_guitar_fingerprint(
        modal_json_sha256=modal_sha,
        fret_count=fret_count,
        tuning_hz=tuning_open_frequencies(DEFAULT_TUNING),
        duration_s=duration_s,
        sample_rate=sample_rate,
        geometry_fingerprint=geom_fp,
        stk_model_alias=stk_mode_alias,
    )


def resolve_note_cache(
    out_root: Path,
    *,
    expected_fingerprint: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Pick manifest: exact fingerprint match preferred, else most recent folder.

    Returns status: ready | stale | missing
    """
    out_root = Path(out_root)
    result: Dict[str, Any] = {
        "status": "missing",
        "manifest": None,
        "manifest_path": None,
        "cache_root": None,
        "guitar_fingerprint": None,
        "expected_fingerprint": expected_fingerprint,
        "position_lookup": {},
    }

    if expected_fingerprint:
        exact = out_root / expected_fingerprint / "note_manifest.json"
        if exact.is_file():
            manifest = load_note_manifest(exact)
            result.update(
                {
                    "status": "ready",
                    "manifest": manifest,
                    "manifest_path": exact,
                    "cache_root": exact.parent,
                    "guitar_fingerprint": manifest.get("guitar_fingerprint"),
                    "position_lookup": build_position_lookup(manifest),
                }
            )
            return result

    manifests = list_manifest_paths(out_root)
    if not manifests:
        return result

    manifest_path = manifests[0]
    manifest = load_note_manifest(manifest_path)
    fp = str(manifest.get("guitar_fingerprint") or manifest_path.parent.name)
    status = "ready"
    if expected_fingerprint and fp != expected_fingerprint:
        status = "stale"

    result.update(
        {
            "status": status,
            "manifest": manifest,
            "manifest_path": manifest_path,
            "cache_root": manifest_path.parent,
            "guitar_fingerprint": fp,
            "position_lookup": build_position_lookup(manifest),
        }
    )
    return result


def lookup_position(
    position_lookup: Mapping[Tuple[int, int], Mapping[str, Any]],
    string_number: int,
    fret: int,
) -> Optional[Dict[str, Any]]:
    key = (int(string_number), int(fret))
    if key not in position_lookup:
        return None
    return dict(position_lookup[key])


def preview_wav_path(cache_root: Path) -> Path:
    return Path(cache_root) / "all_notes_preview.wav"


def build_cache_safe(
    *,
    modal_json: Path,
    out_root: Path,
    fret_count: int = DEFAULT_FRET_COUNT,
    geometry_config: Optional[Path] = None,
    force: bool = False,
    stk_mode_alias: Optional[str] = None,
    sample_parameters: Optional[Mapping[str, Any]] = None,
    precompute_bundle: Optional[Mapping[str, Any]] = None,
    repo_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Build note cache (no FEM) — used by Generate Sound."""
    geom = geometry_config if geometry_config and geometry_config.is_file() else None
    return build_note_cache(
        modal_json=modal_json,
        out_root=out_root,
        fret_count=fret_count,
        geometry_config=geom,
        force=force,
        stk_mode_alias=stk_mode_alias,
        sample_parameters=sample_parameters,
        precompute_bundle=precompute_bundle,
        repo_root=repo_root,
    )


def prepare_player_assets(cache_root: Path, manifest: Mapping[str, Any]) -> Path:
    """
    Copy unique note WAVs (+ preview) into the guitar_player component runtime folder
    so the iframe can fetch them without Streamlit reruns.
    """
    cache_root = Path(cache_root)
    fingerprint = str(manifest.get("guitar_fingerprint") or cache_root.name)
    dest = RUNTIME_CACHE_DIR / fingerprint
    dest.mkdir(parents=True, exist_ok=True)

    for note in manifest.get("notes") or []:
        note_id = str(note.get("note_id") or "")
        wav_rel = str(note.get("wav_path") or "")
        if not note_id or not wav_rel:
            continue
        src = cache_root / wav_rel
        if src.is_file():
            shutil.copy2(src, dest / f"{note_id}.wav")

    preview_src = preview_wav_path(cache_root)
    if preview_src.is_file():
        shutil.copy2(preview_src, dest / "all_notes_preview.wav")

    return dest


def fretboard_display_fret_order(fret_count: int = DEFAULT_FRET_COUNT) -> List[int]:
    """Fret columns left→right: highest fret to open (nut on the right)."""
    n = int(fret_count)
    return list(range(n, -1, -1))


def fretboard_screen_position(
    string_number: int,
    fret: int,
    *,
    fret_count: int = DEFAULT_FRET_COUNT,
) -> Tuple[int, int]:
    """
    Map manifest (string_number, fret) to display grid (row, col).
    Row 0 = string 6 (top); col = fret_count = open string at nut (right).
    """
    row = FRETBOARD_DISPLAY_STRING_ORDER.index(int(string_number))
    col = int(fret_count) - int(fret)
    return row, col


def build_player_payload(
    resolved: Mapping[str, Any],
    *,
    ui_status: str,
) -> Dict[str, Any]:
    """
    JSON passed to the guitar_player Streamlit component.

    ui_status: hidden | building | ready | stale | missing
    """
    status = str(ui_status or "hidden").lower()
    manifest = resolved.get("manifest")
    if status not in ("ready", "building") or not manifest:
        return {"status": status, "positions": [], "fingerprint": ""}

    fingerprint = str(manifest.get("guitar_fingerprint") or resolved.get("guitar_fingerprint") or "")
    positions: List[Dict[str, Any]] = []
    for pos in manifest.get("positions") or []:
        note_id = str(pos.get("note_id") or "")
        note_name = str(pos.get("note_name") or "")
        positions.append(
            {
                "string": int(pos["string_number"]),
                "fret": int(pos["fret"]),
                "note_id": note_id,
                "note_name": note_name,
                "wav": f"{note_id}.wav",
            }
        )

    return {
        "status": status,
        "fingerprint": fingerprint,
        "fret_count": int(manifest.get("fret_count") or DEFAULT_FRET_COUNT),
        "unique_note_count": manifest.get("unique_note_count"),
        "playable_position_count": manifest.get("playable_position_count"),
        "positions": positions,
    }


def note_cache_ui_status(
    *,
    sound_stale: bool,
    note_cache_ready_fp: str,
    expected_fingerprint: Optional[str],
    resolved: Mapping[str, Any],
    building: bool = False,
) -> str:
    """Derive player UI state for the current guitar session."""
    if building:
        return "building"
    if sound_stale or not note_cache_ready_fp:
        return "hidden"
    if not expected_fingerprint:
        return "hidden"
    if note_cache_ready_fp != expected_fingerprint:
        return "stale"
    cache_status = str(resolved.get("status") or "missing")
    if cache_status == "missing":
        return "hidden"
    if cache_status == "stale":
        return "stale"
    return "ready"
