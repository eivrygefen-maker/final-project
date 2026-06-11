#!/usr/bin/env python3
"""
Build playable guitar note cache from ROM/STK body-response synthesis.

Reuses ``body_response_synth.synthesize_note_with_body_response`` — no duplicated DSP.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from body_response_synth import (
    BODY_MODAL_RICHNESS_GAIN,
    DEFAULT_DURATION_S,
    DEFAULT_SAMPLE_RATE,
    FINAL_PEAK_CEILING_DBFS,
    HARMONIC_DECAY_MODEL,
    TARGET_RMS_DBFS,
    load_modal_data_from_path,
    synthesize_note_with_body_response,
    synthetic_classic_body_modes,
)

NOTE_CACHE_SCHEMA_VERSION = "note_cache_v1"
NOTE_CACHE_BUILDER_VERSION = "stage3_v1"

# String 6 (low E) .. string 1 (high E); open frequencies in Hz.
DEFAULT_TUNING: Tuple[Tuple[int, float, str], ...] = (
    (6, 82.41, "E2"),
    (5, 110.00, "A2"),
    (4, 146.83, "D3"),
    (3, 196.00, "G3"),
    (2, 246.94, "B3"),
    (1, 329.63, "E4"),
)

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
A4_REFERENCE_HZ = 440.0


def fret_frequency(open_hz: float, fret: int) -> float:
    return float(open_hz) * (2.0 ** (int(fret) / 12.0))


def frequency_to_note_name(hz: float) -> str:
    """Equal-temperament note name (e.g. E2, A#3) from frequency."""
    midi = 69.0 + 12.0 * math.log2(max(float(hz), 1e-9) / A4_REFERENCE_HZ)
    midi_round = int(round(midi))
    octave = (midi_round // 12) - 1
    return f"{NOTE_NAMES[midi_round % 12]}{octave}"


def note_id_from_frequency(hz: float) -> str:
    """Filesystem-safe note id shared by duplicate fretboard pitches."""
    return frequency_to_note_name(hz).replace("#", "s")


def pitch_dedup_key(hz: float) -> str:
    """Stable pitch identity for cross-string deduplication (equal temperament)."""
    return note_id_from_frequency(hz)


def tuning_open_frequencies(
    tuning: Sequence[Tuple[int, float, str]] = DEFAULT_TUNING,
) -> List[float]:
    return [float(row[1]) for row in tuning]


def enumerate_fretboard_positions(
    fret_count: int,
    tuning: Sequence[Tuple[int, float, str]] = DEFAULT_TUNING,
) -> List[Dict[str, Any]]:
    """All (string, fret) playable positions with frequency."""
    positions: List[Dict[str, Any]] = []
    for string_number, open_hz, open_name in tuning:
        for fret in range(int(fret_count) + 1):
            hz = fret_frequency(open_hz, fret)
            positions.append(
                {
                    "string_number": int(string_number),
                    "fret": int(fret),
                    "frequency_hz": round(hz, 6),
                    "open_string_name": open_name,
                }
            )
    return positions


def position_frequency(string_number: int, fret: int, tuning=DEFAULT_TUNING) -> float:
    for sn, open_hz, _ in tuning:
        if int(sn) == int(string_number):
            return fret_frequency(open_hz, fret)
    raise ValueError(f"unknown string_number={string_number}")


def group_unique_pitches(
    positions: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Map note_id -> representative note entry (duplicate pitches share one WAV)."""
    unique: Dict[str, Dict[str, Any]] = {}
    for pos in positions:
        hz = float(pos["frequency_hz"])
        note_id = pitch_dedup_key(hz)
        if note_id not in unique:
            unique[note_id] = {
                "frequency_hz": hz,
                "note_id": note_id,
                "note_name": frequency_to_note_name(hz),
            }
    return unique


def synthesis_version_payload(
    *,
    duration_s: float,
    sample_rate: int,
) -> Dict[str, Any]:
    return {
        "builder_version": NOTE_CACHE_BUILDER_VERSION,
        "synthesis_model": "modal_transfer_function_H_body_sum_m_Wm_Hm",
        "body_modal_richness_gain": BODY_MODAL_RICHNESS_GAIN,
        "target_rms_dbfs": TARGET_RMS_DBFS,
        "final_peak_ceiling_dbfs": FINAL_PEAK_CEILING_DBFS,
        "harmonic_decay_model": HARMONIC_DECAY_MODEL,
        "duration_s": float(duration_s),
        "sample_rate": int(sample_rate),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_guitar_fingerprint(
    *,
    modal_json_sha256: str,
    fret_count: int,
    tuning_hz: Sequence[float],
    duration_s: float,
    sample_rate: int,
    geometry_fingerprint: Optional[str] = None,
) -> str:
    payload = {
        "modal_json_sha256": modal_json_sha256,
        "fret_count": int(fret_count),
        "tuning_hz": [round(float(f), 6) for f in tuning_hz],
        "geometry_fingerprint": geometry_fingerprint or "",
        **synthesis_version_payload(duration_s=duration_s, sample_rate=sample_rate),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def optional_geometry_fingerprint(config_path: Optional[Path]) -> Optional[str]:
    if config_path is None or not config_path.is_file():
        return None
    return file_sha256(config_path)


def _relative(cache_root: Path, path: Path) -> str:
    return path.relative_to(cache_root).as_posix()


def build_note_cache(
    modal_json: Path,
    out_root: Path,
    *,
    fret_count: int = 19,
    duration_s: float = DEFAULT_DURATION_S,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    tuning: Sequence[Tuple[int, float, str]] = DEFAULT_TUNING,
    geometry_config: Optional[Path] = None,
    force: bool = False,
) -> Dict[str, Any]:
    modal_json = Path(modal_json).resolve()
    out_root = Path(out_root).resolve()

    if modal_json.is_file():
        modal_data = load_modal_data_from_path(modal_json)
        modal_sha = file_sha256(modal_json)
    else:
        modal_data = {"predicted_modes": synthetic_classic_body_modes(), "analysis": "synthetic_fixture"}
        modal_sha = hashlib.sha256(
            json.dumps(modal_data, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

    tuning_hz = tuning_open_frequencies(tuning)
    geom_fp = optional_geometry_fingerprint(geometry_config)
    guitar_fp = compute_guitar_fingerprint(
        modal_json_sha256=modal_sha,
        fret_count=fret_count,
        tuning_hz=tuning_hz,
        duration_s=duration_s,
        sample_rate=sample_rate,
        geometry_fingerprint=geom_fp,
    )

    cache_root = out_root / guitar_fp
    wav_dir = cache_root / "wav"
    meta_dir = cache_root / "metadata"
    wav_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    positions_raw = enumerate_fretboard_positions(fret_count, tuning=tuning)
    unique_by_key = group_unique_pitches(positions_raw)

    notes_out: List[Dict[str, Any]] = []
    note_lookup: Dict[str, Dict[str, Any]] = {}

    for note_id in sorted(unique_by_key.keys()):
        entry = unique_by_key[note_id]
        hz = float(entry["frequency_hz"])
        wav_path = wav_dir / f"{note_id}.wav"
        meta_path = meta_dir / f"{note_id}.json"

        if force or not wav_path.is_file() or not meta_path.is_file():
            synth_meta = synthesize_note_with_body_response(
                frequency_hz=hz,
                note_name=entry["note_name"],
                duration_s=duration_s,
                sample_rate=sample_rate,
                modal_data=modal_data,
                output_wav=wav_path,
                output_metadata_json=meta_path,
            )
        else:
            synth_meta = json.loads(meta_path.read_text(encoding="utf-8"))

        note_row = {
            "note_id": note_id,
            "note_name": entry["note_name"],
            "frequency_hz": round(hz, 6),
            "wav_path": _relative(cache_root, wav_path),
            "metadata_path": _relative(cache_root, meta_path),
            "high_frequency_fallback_used": bool(synth_meta.get("high_frequency_fallback_used")),
            "output_rms_dbfs": synth_meta.get("output_rms_dbfs"),
            "output_peak_dbfs": synth_meta.get("output_peak_dbfs"),
        }
        notes_out.append(note_row)
        note_lookup[note_id] = note_row

    positions_out: List[Dict[str, Any]] = []
    for pos in positions_raw:
        nid = pitch_dedup_key(float(pos["frequency_hz"]))
        note_row = note_lookup[nid]
        positions_out.append(
            {
                "string_number": pos["string_number"],
                "fret": pos["fret"],
                "frequency_hz": round(float(pos["frequency_hz"]), 6),
                "note_id": note_row["note_id"],
                "wav_path": note_row["wav_path"],
            }
        )

    manifest: Dict[str, Any] = {
        "schema_version": NOTE_CACHE_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "guitar_fingerprint": guitar_fp,
        "modal_json_path": str(modal_json),
        "modal_json_sha256": modal_sha,
        "geometry_fingerprint": geom_fp,
        "synthesis_model": "modal_transfer_function_H_body_sum_m_Wm_Hm",
        "body_modal_richness_gain": BODY_MODAL_RICHNESS_GAIN,
        "builder_version": NOTE_CACHE_BUILDER_VERSION,
        "synthesis_constants": synthesis_version_payload(
            duration_s=duration_s,
            sample_rate=sample_rate,
        ),
        "fret_count": int(fret_count),
        "tuning": [
            {"string_number": sn, "open_hz": hz, "open_name": name}
            for sn, hz, name in tuning
        ],
        "duration_s": float(duration_s),
        "sample_rate": int(sample_rate),
        "unique_note_count": len(notes_out),
        "playable_position_count": len(positions_out),
        "notes": notes_out,
        "positions": positions_out,
    }

    manifest_path = cache_root / "note_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["cache_root"] = str(cache_root)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build playable guitar note cache")
    parser.add_argument(
        "--modal-json",
        type=Path,
        default=repo / "FEM" / "outputs" / "rom_stk_body.json",
    )
    parser.add_argument("--out-root", type=Path, default=repo / "audio" / "note_cache")
    parser.add_argument("--frets", type=int, default=19, help="Highest fret index (0..N)")
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_S)
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument(
        "--geometry-config",
        type=Path,
        default=repo / "FEM" / "configs" / "guitar_3d.json",
        help="Optional guitar geometry config for fingerprint (if file exists)",
    )
    parser.add_argument("--force", action="store_true", help="Regenerate all WAV/metadata")
    args = parser.parse_args()

    geom = args.geometry_config if args.geometry_config.is_file() else None
    manifest = build_note_cache(
        modal_json=args.modal_json,
        out_root=args.out_root,
        fret_count=args.frets,
        duration_s=args.duration,
        sample_rate=args.sample_rate,
        geometry_config=geom,
        force=args.force,
    )
    print(f"guitar_fingerprint={manifest['guitar_fingerprint']}")
    print(f"cache_root={manifest['cache_root']}")
    print(
        f"unique_notes={manifest['unique_note_count']} "
        f"positions={manifest['playable_position_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
