#!/usr/bin/env python3
"""Static melody/chord library loader for the ready Classical guitar player."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Set

from classical_guitar_fretboard import (
    get_fret_count,
    lookup_note,
    normalize_note_name,
    position_runtime_wav_name,
)

LIBRARY_DIR = Path(__file__).resolve().parent / "data" / "guitar_library"
CHORDS_PATH = LIBRARY_DIR / "chords.json"
MELODIES_PATH = LIBRARY_DIR / "melodies.json"


def _read_json(path: Path) -> Dict[str, Any]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return doc


def _require_unique_ids(items: Iterable[Mapping[str, Any]], label: str) -> None:
    seen: Set[str] = set()
    for item in items:
        item_id = str(item.get("id") or "").strip()
        if not item_id:
            raise ValueError(f"{label} entry missing id")
        if item_id in seen:
            raise ValueError(f"duplicate {label} id: {item_id}")
        seen.add(item_id)


def _validate_string_number(value: Any, *, context: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: invalid string number {value!r}") from exc
    if number < 1 or number > 6:
        raise ValueError(f"{context}: string must be 1..6, got {number}")
    return number


def _validate_fret(value: Any, *, context: str, allow_muted: bool = False) -> int:
    try:
        fret = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: invalid fret {value!r}") from exc
    if allow_muted and fret == -1:
        return fret
    fret_count = get_fret_count()
    if fret < 0 or fret > fret_count:
        raise ValueError(f"{context}: fret must be 0..{fret_count}, got {fret}")
    return fret


def validate_chord_library(doc: Mapping[str, Any]) -> None:
    chords = doc.get("chords")
    if not isinstance(chords, list) or not chords:
        raise ValueError("chords.json must contain non-empty chords list")
    _require_unique_ids(chords, "chord")
    for chord in chords:
        chord_id = str(chord.get("id") or "")
        fingering = chord.get("fingering")
        if not isinstance(fingering, dict):
            raise ValueError(f"{chord_id}: fingering must be an object")
        expected_keys = {f"string_{n}" for n in range(1, 7)}
        if set(fingering.keys()) != expected_keys:
            raise ValueError(f"{chord_id}: fingering must define exactly six strings")
        for key, value in fingering.items():
            string_number = _validate_string_number(key.split("_", 1)[1], context=f"{chord_id}.{key}")
            _ = string_number
            _validate_fret(value, context=f"{chord_id}.{key}", allow_muted=True)
        strum = chord.get("default_strum") or {}
        strings = strum.get("strings")
        if not isinstance(strings, list) or not strings:
            raise ValueError(f"{chord_id}: default_strum.strings must be non-empty")
        for string_number in strings:
            sn = _validate_string_number(string_number, context=f"{chord_id}.default_strum")
            fret = int(fingering[f"string_{sn}"])
            if fret == -1:
                raise ValueError(f"{chord_id}: muted string {sn} included in default strum")


def validate_melody_library(doc: Mapping[str, Any]) -> None:
    melodies = doc.get("melodies")
    if not isinstance(melodies, list) or not melodies:
        raise ValueError("melodies.json must contain non-empty melodies list")
    _require_unique_ids(melodies, "melody")
    playable_wavs = {
        position_runtime_wav_name(row_string, row_fret)
        for row_string in range(1, 7)
        for row_fret in range(get_fret_count() + 1)
    }
    for melody in melodies:
        melody_id = str(melody.get("id") or "")
        try:
            bpm = float(melody.get("tempo_bpm"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{melody_id}: tempo_bpm must be numeric") from exc
        if bpm <= 0:
            raise ValueError(f"{melody_id}: tempo_bpm must be positive")
        events = melody.get("events")
        if not isinstance(events, list) or not events:
            raise ValueError(f"{melody_id}: events must be non-empty")
        for index, event in enumerate(events):
            context = f"{melody_id}.events[{index}]"
            if not isinstance(event, list) or len(event) != 6:
                raise ValueError(f"{context}: event must have six values")
            note, string_number, fret, start_beats, duration_beats, velocity = event
            sn = _validate_string_number(string_number, context=context)
            fr = _validate_fret(fret, context=context)
            expected_note = lookup_note(sn, fr)
            actual_note = normalize_note_name(str(note))
            if actual_note != expected_note:
                raise ValueError(f"{context}: note {actual_note} does not match S{sn} F{fr} ({expected_note})")
            try:
                start = float(start_beats)
                duration = float(duration_beats)
                vel = float(velocity)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{context}: timing/velocity values must be numeric") from exc
            if start < 0:
                raise ValueError(f"{context}: start_beats must be >= 0")
            if duration <= 0:
                raise ValueError(f"{context}: duration_beats must be > 0")
            if vel < 0.0 or vel > 1.0:
                raise ValueError(f"{context}: velocity must be 0.0..1.0")
            if position_runtime_wav_name(sn, fr) not in playable_wavs:
                raise ValueError(f"{context}: no playable cached-WAV mapping for S{sn} F{fr}")


def load_guitar_library() -> Dict[str, Any]:
    """Load and validate static player-library JSON without touching audio generation."""
    chords = _read_json(CHORDS_PATH)
    melodies = _read_json(MELODIES_PATH)
    validate_chord_library(chords)
    validate_melody_library(melodies)
    return {
        "status": "ready",
        "chords": chords,
        "melodies": melodies,
        "fret_count": get_fret_count(),
    }


if __name__ == "__main__":
    loaded = load_guitar_library()
    print(
        json.dumps(
            {
                "status": loaded["status"],
                "chord_count": len(loaded["chords"]["chords"]),
                "melody_count": len(loaded["melodies"]["melodies"]),
            },
            sort_keys=True,
        )
    )
