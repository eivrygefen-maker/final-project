#!/usr/bin/env python3
"""Backward-compatible re-exports — use classical_guitar_fretboard as source of truth."""
from __future__ import annotations

from classical_guitar_fretboard import (  # noqa: F401
    build_classical_fretboard_map,
    build_fretboard_note_mapping,
    build_required_note_set_from_fretboard,
    ensure_fretboard_config_file,
    get_fret_count,
    is_note_wav_path,
    is_position_runtime_wav_stem,
    is_valid_note_name,
    list_ignored_non_note_wavs,
    list_note_wavs,
    list_position_wav_files,
    load_fretboard_config,
    midi_to_frequency_hz,
    midi_to_note_name,
    normalize_note_name,
    lookup_note,
    note_at_fret,
    note_id_from_note_name,
    note_name_to_frequency_hz,
    note_range_label_from_required,
    note_to_midi,
    build_note_frequency_hz_table,
    player_fretboard_metadata,
    position_runtime_wav_name,
    required_notes_cover_high_frets,
    run_fretboard_mapping_audit,
    string_visual_order_numbers,
    validate_explicit_fretboard_checks,
    validate_player_payload_positions,
    validate_sharp_note_frequency_checks,
)

# Legacy alias used by note_cache_ui / tests
DEFAULT_FRET_COUNT = 19
