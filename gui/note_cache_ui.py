#!/usr/bin/env python3
"""
Note-cache helpers for Stage 4 fretboard UI (manifest load, position lookup, no synthesis).
"""
from __future__ import annotations

import json
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

NOTE_CACHE_ROOT_NAME = "note_cache"
DEFAULT_FRET_COUNT = 19


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


def read_wav_bytes(wav_path: Path) -> bytes:
    path = Path(wav_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.read_bytes()


def preview_wav_path(cache_root: Path) -> Path:
    return Path(cache_root) / "all_notes_preview.wav"


def build_cache_safe(
    *,
    modal_json: Path,
    out_root: Path,
    fret_count: int = DEFAULT_FRET_COUNT,
    geometry_config: Optional[Path] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Build note cache (no FEM) — wrapper for GUI button."""
    geom = geometry_config if geometry_config and geometry_config.is_file() else None
    return build_note_cache(
        modal_json=modal_json,
        out_root=out_root,
        fret_count=fret_count,
        geometry_config=geom,
        force=force,
    )


def fretboard_note_label(position: Optional[Mapping[str, Any]]) -> str:
    if not position:
        return "—"
    note_id = str(position.get("note_id") or "")
    return note_id if len(note_id) <= 4 else note_id[:4]


def render_fretboard_panel(
    st_module: Any,
    *,
    repo_root: Path,
    modal_json: Path,
    geometry_config: Optional[Path] = None,
    rom_body_ready: bool,
) -> None:
    """Streamlit fretboard section — plays cached WAVs only."""
    st_module.subheader("Fretboard (note cache)")
    cache_root = note_cache_root(repo_root)
    expected_fp = None
    if modal_json.is_file():
        expected_fp = expected_note_cache_fingerprint(
            modal_json=modal_json,
            geometry_config=geometry_config,
        )
    resolved = resolve_note_cache(cache_root, expected_fingerprint=expected_fp)
    status = resolved["status"]
    manifest = resolved.get("manifest")
    cache_dir = resolved.get("cache_root")

    if status == "missing":
        st_module.caption("Note cache: **missing**")
        st_module.info("Generate the note cache first (requires ROM body response).")
        if rom_body_ready and modal_json.is_file():
            if st_module.button("Build note cache", key="btn_build_note_cache", use_container_width=True):
                with st_module.spinner("Building note cache (no live synthesis on fretboard clicks)…"):
                    try:
                        build_cache_safe(
                            modal_json=modal_json,
                            out_root=cache_root,
                            geometry_config=geometry_config,
                            force=True,
                        )
                        st_module.session_state.pop("fretboard_wav_bytes", None)
                        st_module.success("Note cache built.")
                        st_module.rerun()
                    except Exception as exc:
                        st_module.error(f"Note cache build failed: {exc}")
        return

    fp = resolved.get("guitar_fingerprint") or "?"
    if status == "stale":
        st_module.warning(
            f"Note cache: **stale** — loaded `{fp[:12]}…` "
            f"(expected `{str(expected_fp)[:12]}…`). Rebuild after Save & Sync."
        )
    else:
        st_module.caption(f"Note cache: **ready** · fingerprint `{fp[:16]}…`")

    if manifest:
        st_module.caption(
            f"{manifest.get('unique_note_count', '?')} unique notes · "
            f"{manifest.get('playable_position_count', '?')} positions · "
            f"frets 0–{manifest.get('fret_count', '?')}"
        )

    preview = preview_wav_path(Path(cache_dir)) if cache_dir else None
    if preview and preview.is_file():
        if st_module.button("Play all notes preview", key="btn_play_note_cache_preview", use_container_width=True):
            st_module.session_state["fretboard_wav_bytes"] = read_wav_bytes(preview)
            st_module.session_state["fretboard_play_label"] = "all_notes_preview"

    lookup = resolved.get("position_lookup") or {}
    fret_count = int((manifest or {}).get("fret_count") or DEFAULT_FRET_COUNT)
    frets = list(range(fret_count + 1))

    header_cols = st_module.columns([1] + [1] * len(frets))
    with header_cols[0]:
        st_module.caption("Str")
    for i, fret in enumerate(frets):
        with header_cols[i + 1]:
            st_module.caption(str(fret))

    for string_number in (6, 5, 4, 3, 2, 1):
        row = st_module.columns([1] + [1] * len(frets))
        with row[0]:
            st_module.markdown(f"**{string_number}**")
        for i, fret in enumerate(frets):
            pos = lookup_position(lookup, string_number, fret)
            label = fretboard_note_label(pos)
            with row[i + 1]:
                if st_module.button(
                    label,
                    key=f"fret_s{string_number}_f{fret}",
                    use_container_width=True,
                    disabled=pos is None,
                ):
                    if pos and cache_dir:
                        wav = resolve_wav_path(Path(cache_dir), str(pos["wav_path"]))
                        st_module.session_state["fretboard_wav_bytes"] = read_wav_bytes(wav)
                        st_module.session_state["fretboard_play_label"] = (
                            f"S{string_number} F{fret} · {pos.get('note_id', '')}"
                        )

    wav_bytes = st_module.session_state.get("fretboard_wav_bytes")
    if wav_bytes:
        label = st_module.session_state.get("fretboard_play_label") or "note"
        st_module.caption(f"Playing: {label}")
        st_module.audio(wav_bytes, format="audio/wav")

    if status == "stale" and rom_body_ready and modal_json.is_file():
        if st_module.button("Rebuild note cache", key="btn_rebuild_note_cache", use_container_width=True):
            with st_module.spinner("Rebuilding note cache…"):
                try:
                    build_cache_safe(
                        modal_json=modal_json,
                        out_root=cache_root,
                        geometry_config=geometry_config,
                        force=True,
                    )
                    st_module.session_state.pop("fretboard_wav_bytes", None)
                    st_module.success("Note cache rebuilt.")
                    st_module.rerun()
                except Exception as exc:
                    st_module.error(f"Rebuild failed: {exc}")
