#!/usr/bin/env python3
"""
Classical guitar fretboard — single source of truth for APP / HTML / STK.

Loads ``config/classical_guitar_fretboard.json`` (generated on first use if missing).
Note at fret = open_string_midi + fret_number (equal temperament, sharp names).
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config" / "classical_guitar_fretboard.json"
AUDIT_JSON_PATH = REPO_ROOT / "audio" / "debug_reports" / "app_stk_fretboard_mapping_audit.json"
AUDIT_MD_PATH = REPO_ROOT / "audio" / "debug_reports" / "app_stk_fretboard_mapping_audit.md"

NOTE_NAMES: Tuple[str, ...] = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
A4_REFERENCE_HZ = 440.0
_NOTE_RE = re.compile(r"^([A-G])(#|b)?(\d+)$")
_NOTE_ID_RE = re.compile(r"^([A-G])(s)?(\d+)$")

# WAV stems that are helpers/previews — not playable fretboard notes.
HELPER_WAV_STEMS = frozenset(
    {
        "all_notes_preview",
    }
)

_FLAT_TO_SHARP = {
    "Bb": "A#",
    "Db": "C#",
    "Eb": "D#",
    "Gb": "F#",
    "Ab": "G#",
}

_DEFAULT_TUNING: Dict[str, str] = {
    "S6": "E2",
    "S5": "A2",
    "S4": "D3",
    "S3": "G3",
    "S2": "B3",
    "S1": "E4",
}

_DEFAULT_CONFIG: Dict[str, Any] = {
    "instrument": "classical_guitar",
    "tuning": dict(_DEFAULT_TUNING),
    "string_visual_order_top_to_bottom": ["S6", "S5", "S4", "S3", "S2", "S1"],
    "open_strings_side": "right",
    "higher_frets_direction": "left",
    "fret_count": 19,
    "note_naming": "sharps",
    "flat_aliases": dict(_FLAT_TO_SHARP),
}

EXPLICIT_VALIDATION_CHECKS: Tuple[Tuple[str, int, str], ...] = (
    ("S6", 0, "E2"),
    ("S6", 1, "F2"),
    ("S6", 2, "F#2"),
    ("S6", 3, "G2"),
    ("S5", 0, "A2"),
    ("S5", 1, "A#2"),
    ("S5", 2, "B2"),
    ("S5", 3, "C3"),
    ("S4", 0, "D3"),
    ("S3", 0, "G3"),
    ("S2", 0, "B3"),
    ("S2", 1, "C4"),
    ("S1", 0, "E4"),
    ("S1", 12, "E5"),
    ("S1", 19, "B5"),
)


def normalize_note_name(note_name: str) -> str:
    """Canonical sharp spelling (e.g. Bb4 -> A#4)."""
    m = _NOTE_RE.match(str(note_name).strip())
    if not m:
        return str(note_name).strip()
    letter, acc, octave_s = m.group(1), m.group(2) or "", int(m.group(3))
    if acc == "b":
        sharp_letter = _FLAT_TO_SHARP.get(f"{letter}b")
        if sharp_letter:
            return f"{sharp_letter}{octave_s}"
    if acc == "#":
        return f"{letter}#{octave_s}"
    return f"{letter}{octave_s}"


def is_valid_note_name(name: str) -> bool:
    """True when ``name`` matches A–G with optional #/b and octave (no exceptions)."""
    return bool(_NOTE_RE.match(str(name).strip()))


def note_id_stem_to_note_name(stem: str) -> Optional[str]:
    """Map runtime note-id stems (e.g. Fs2) to normalized note names (F#2)."""
    s = str(stem).strip()
    if is_valid_note_name(s):
        return normalize_note_name(s)
    m = _NOTE_ID_RE.match(s)
    if not m:
        return None
    letter, sharp, octave_s = m.group(1), m.group(2), m.group(3)
    candidate = f"{letter}#{octave_s}" if sharp else f"{letter}{octave_s}"
    if not is_valid_note_name(candidate):
        return None
    return normalize_note_name(candidate)


def is_helper_wav_stem(stem: str) -> bool:
    return str(stem).strip() in HELPER_WAV_STEMS


def is_note_wav_path(path: Path) -> bool:
    """True for playable note WAV paths; false for preview/helper/debug WAVs."""
    stem = Path(path).stem
    if is_helper_wav_stem(stem):
        return False
    if is_position_runtime_wav_stem(stem):
        return True
    if is_valid_note_name(stem):
        return True
    return note_id_stem_to_note_name(stem) is not None


def wav_stem_to_note_name(stem: str) -> Optional[str]:
    """Return normalized note name for a WAV stem, or None if not a note file."""
    s = str(stem).strip()
    if is_helper_wav_stem(s):
        return None
    if is_valid_note_name(s):
        return normalize_note_name(s)
    return note_id_stem_to_note_name(s)


def list_note_wavs(cache_dir: Path) -> Dict[str, Path]:
    """Map normalized note names to WAV paths (excludes helper/preview files)."""
    out: Dict[str, Path] = {}
    d = Path(cache_dir)
    if not d.is_dir():
        return out
    for path in d.glob("*.wav"):
        if not path.is_file():
            continue
        note_name = wav_stem_to_note_name(path.stem)
        if note_name is None:
            continue
        out[note_name] = path
    return out


def list_ignored_non_note_wavs(cache_dir: Path) -> List[str]:
    """WAV filenames in ``cache_dir`` that are not valid note files."""
    d = Path(cache_dir)
    if not d.is_dir():
        return []
    ignored: List[str] = []
    for path in d.glob("*.wav"):
        if path.is_file() and not is_note_wav_path(path):
            ignored.append(path.name)
    return sorted(ignored)


def note_to_midi(note_name: str) -> int:
    normalized = normalize_note_name(note_name)
    m = _NOTE_RE.match(normalized)
    if not m:
        raise ValueError(f"invalid note name: {note_name!r}")
    letter = m.group(1)
    if letter not in NOTE_NAMES:
        raise ValueError(f"unknown pitch class in {note_name!r}")
    octave_s = int(m.group(3))
    return (octave_s + 1) * 12 + NOTE_NAMES.index(letter)


def midi_to_note_name(midi: int, *, use_sharps: bool = True) -> str:
    _ = use_sharps
    octave = (int(midi) // 12) - 1
    return f"{NOTE_NAMES[int(midi) % 12]}{octave}"


def midi_to_frequency_hz(midi: int) -> float:
    return A4_REFERENCE_HZ * (2.0 ** ((int(midi) - 69) / 12.0))


def note_id_from_note_name(note_name: str) -> str:
    """Filesystem-safe id (# -> s) shared by duplicate pitches."""
    return normalize_note_name(note_name).replace("#", "s")


def position_runtime_wav_name(string_number: int, fret: int) -> str:
    """Runtime player WAV filename — one file per physical string/fret cell."""
    return f"S{int(string_number)}_f{int(fret)}.wav"


def is_position_runtime_wav_stem(stem: str) -> bool:
    """True for per-cell runtime WAV stems like ``S6_f1``."""
    return bool(re.match(r"^S[1-6]_f\d+$", str(stem).strip()))


def string_key_to_number(string_key: str) -> int:
    return int(str(string_key).lstrip("Ss"))


def string_number_to_key(string_number: int) -> str:
    return f"S{int(string_number)}"


def note_at_fret(open_note: str, fret: int) -> str:
    """Musical rule: pitch midi = open_string_midi + fret."""
    return midi_to_note_name(note_to_midi(open_note) + int(fret))


def build_classical_fretboard_map(
    tuning: Optional[Mapping[str, str]] = None,
    fret_count: int = 19,
) -> Dict[str, Dict[str, str]]:
    """Build S6..S1 -> fret -> note_name map from tuning + formula."""
    t = dict(tuning or _DEFAULT_TUNING)
    out: Dict[str, Dict[str, str]] = {}
    for string_key in ("S6", "S5", "S4", "S3", "S2", "S1"):
        open_note = normalize_note_name(str(t.get(string_key, _DEFAULT_TUNING[string_key])))
        frets: Dict[str, str] = {}
        for fret in range(int(fret_count) + 1):
            frets[str(fret)] = note_at_fret(open_note, fret)
        out[string_key] = frets
    return out


def build_fretboard_config_document(
    *,
    fret_count: int = 19,
    tuning: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    t = dict(tuning or _DEFAULT_TUNING)
    doc = dict(_DEFAULT_CONFIG)
    doc["tuning"] = {k: normalize_note_name(str(v)) for k, v in t.items()}
    doc["fret_count"] = int(fret_count)
    doc["generated_fretboard"] = build_classical_fretboard_map(t, fret_count)
    return doc


def ensure_fretboard_config_file(path: Optional[Path] = None) -> Path:
    """Write config JSON if missing; refresh generated_fretboard if stale."""
    cfg_path = Path(path or CONFIG_PATH)
    if cfg_path.is_file():
        try:
            loaded = json.loads(cfg_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            loaded = {}
    else:
        loaded = {}
    fret_count = int(loaded.get("fret_count") or _DEFAULT_CONFIG["fret_count"])
    tuning = loaded.get("tuning") or _DEFAULT_TUNING
    fresh = build_fretboard_config_document(fret_count=fret_count, tuning=tuning)
    for key in _DEFAULT_CONFIG:
        fresh.setdefault(key, loaded.get(key, _DEFAULT_CONFIG.get(key)))
    if loaded.get("generated_fretboard") != fresh["generated_fretboard"]:
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(json.dumps(fresh, indent=2) + "\n", encoding="utf-8")
    elif not cfg_path.is_file():
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(json.dumps(fresh, indent=2) + "\n", encoding="utf-8")
    return cfg_path


def load_fretboard_config(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    root = Path(repo_root or REPO_ROOT)
    cfg_path = root / "config" / "classical_guitar_fretboard.json"
    ensure_fretboard_config_file(cfg_path)
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def get_fret_count(cfg: Optional[Mapping[str, Any]] = None) -> int:
    c = dict(cfg or load_fretboard_config())
    return int(c.get("fret_count") or 19)


def get_tuning(cfg: Optional[Mapping[str, Any]] = None) -> Dict[str, str]:
    c = dict(cfg or load_fretboard_config())
    return {k: normalize_note_name(str(v)) for k, v in dict(c.get("tuning") or _DEFAULT_TUNING).items()}


def string_visual_order_numbers(cfg: Optional[Mapping[str, Any]] = None) -> List[int]:
    c = dict(cfg or load_fretboard_config())
    order = list(c.get("string_visual_order_top_to_bottom") or _DEFAULT_CONFIG["string_visual_order_top_to_bottom"])
    return [string_key_to_number(k) for k in order]


def get_default_tuning_tuple(
    cfg: Optional[Mapping[str, Any]] = None,
) -> Tuple[Tuple[int, float, str], ...]:
    """(string_number, open_hz, open_note) for build_note_cache compatibility."""
    tuning = get_tuning(cfg)
    rows: List[Tuple[int, float, str]] = []
    for string_key in ("S6", "S5", "S4", "S3", "S2", "S1"):
        sn = string_key_to_number(string_key)
        open_note = tuning[string_key]
        open_midi = note_to_midi(open_note)
        open_hz = midi_to_frequency_hz(open_midi)
        rows.append((sn, round(open_hz, 6), open_note))
    return tuple(rows)


def build_fretboard_note_mapping(
    fret_count: Optional[int] = None,
    cfg: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """All clickable (string, fret) positions — canonical for APP/HTML/STK."""
    c = dict(cfg or load_fretboard_config())
    fc = int(fret_count if fret_count is not None else c.get("fret_count") or 19)
    tuning = get_tuning(c)
    generated = c.get("generated_fretboard") or build_classical_fretboard_map(tuning, fc)
    mapping: List[Dict[str, Any]] = []
    for string_key in c.get("string_visual_order_top_to_bottom") or _DEFAULT_CONFIG["string_visual_order_top_to_bottom"]:
        string_number = string_key_to_number(string_key)
        open_note = tuning[string_key]
        open_midi = note_to_midi(open_note)
        fret_map = generated.get(string_key) or build_classical_fretboard_map(tuning, fc)[string_key]
        for fret_s, note_name in sorted(fret_map.items(), key=lambda kv: int(kv[0])):
            fret = int(fret_s)
            midi = open_midi + fret
            hz = round(midi_to_frequency_hz(midi), 6)
            note_name = normalize_note_name(note_name)
            mapping.append(
                {
                    "string_key": string_key,
                    "string_number": string_number,
                    "fret": fret,
                    "note_name": note_name,
                    "note_id": note_id_from_note_name(note_name),
                    "frequency_hz": hz,
                    "open_string_name": open_note,
                    "midi": midi,
                }
            )
    return mapping


def build_required_note_set_from_fretboard(
    fret_count: Optional[int] = None,
    cfg: Optional[Mapping[str, Any]] = None,
) -> List[str]:
    names = {str(row["note_name"]) for row in build_fretboard_note_mapping(fret_count, cfg)}
    return sorted(names, key=note_to_midi)


def note_range_label_from_required(required_notes: Sequence[str]) -> str:
    if not required_notes:
        return "E2:B5"
    ordered = sorted(required_notes, key=note_to_midi)
    return f"{ordered[0]}:{ordered[-1]}"


def required_notes_cover_high_frets(
    fret_count: Optional[int] = None,
    *,
    string_number: int = 1,
    min_fret: int = 13,
    cfg: Optional[Mapping[str, Any]] = None,
) -> List[str]:
    return [
        str(row["note_name"])
        for row in build_fretboard_note_mapping(fret_count, cfg)
        if int(row["string_number"]) == int(string_number) and int(row["fret"]) >= int(min_fret)
    ]


def lookup_note(string_number: int, fret: int, cfg: Optional[Mapping[str, Any]] = None) -> str:
    sk = string_number_to_key(string_number)
    c = dict(cfg or load_fretboard_config())
    fc = get_fret_count(c)
    generated = c.get("generated_fretboard") or build_classical_fretboard_map(get_tuning(c), fc)
    return normalize_note_name(str(generated[sk][str(int(fret))]))


def validate_explicit_fretboard_checks(
    cfg: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    c = dict(cfg or load_fretboard_config())
    results: List[Dict[str, Any]] = []
    for string_key, fret, expected in EXPLICIT_VALIDATION_CHECKS:
        sn = string_key_to_number(string_key)
        actual = lookup_note(sn, fret, c)
        results.append(
            {
                "string": string_key,
                "string_number": sn,
                "fret": fret,
                "expected": expected,
                "actual": actual,
                "passed": actual == normalize_note_name(expected),
            }
        )
    return results


def validate_player_payload_positions(
    player_payload: Mapping[str, Any],
    cfg: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Verify player payload maps each required string/fret to the fretboard note."""
    c = dict(cfg or load_fretboard_config())
    lookup = {
        (int(p["string"]), int(p["fret"])): p
        for p in (player_payload.get("positions") or [])
    }
    results: List[Dict[str, Any]] = []
    for string_key, fret, expected in EXPLICIT_VALIDATION_CHECKS:
        sn = string_key_to_number(string_key)
        pos = lookup.get((sn, int(fret)))
        actual = normalize_note_name(str((pos or {}).get("note_name") or "")) if pos else ""
        wav = str((pos or {}).get("wav") or "")
        results.append(
            {
                "string": string_key,
                "string_number": sn,
                "fret": fret,
                "expected": expected,
                "actual": actual,
                "wav": wav,
                "passed": actual == normalize_note_name(expected) and bool(pos),
            }
        )
    return results


def player_fretboard_metadata(cfg: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    c = dict(cfg or load_fretboard_config())
    return {
        "instrument": c.get("instrument", "classical_guitar"),
        "string_visual_order_top_to_bottom": list(
            c.get("string_visual_order_top_to_bottom") or _DEFAULT_CONFIG["string_visual_order_top_to_bottom"]
        ),
        "string_visual_order_numbers": string_visual_order_numbers(c),
        "open_strings_side": c.get("open_strings_side", "right"),
        "higher_frets_direction": c.get("higher_frets_direction", "left"),
        "fret_count": get_fret_count(c),
        "tuning": get_tuning(c),
    }


def run_fretboard_mapping_audit(
    *,
    cache_dir: Optional[Path] = None,
    player_payload: Optional[Mapping[str, Any]] = None,
    repo_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Write app_stk_fretboard_mapping_audit JSON/MD; return audit document."""
    root = Path(repo_root or REPO_ROOT)
    cfg = load_fretboard_config(root)
    fc = get_fret_count(cfg)
    tuning = get_tuning(cfg)
    required = build_required_note_set_from_fretboard(fc, cfg)
    mapping = build_fretboard_note_mapping(fc, cfg)
    explicit_checks = validate_explicit_fretboard_checks(cfg)
    explicit_ok = all(row["passed"] for row in explicit_checks)

    generated_map: Dict[str, Dict[str, str]] = {}
    for row in mapping:
        sk = str(row["string_key"])
        generated_map.setdefault(sk, {})[str(row["fret"])] = str(row["note_name"])

    missing_cache_notes: List[str] = []
    if cache_dir is not None:
        cache_path = Path(cache_dir)
        for note in required:
            normalized = normalize_note_name(note)
            found = (cache_path / f"{normalized}.wav").is_file()
            if not found:
                alt = note_id_from_note_name(note) + ".wav"
                found = (cache_path / alt).is_file()
            if not found:
                missing_cache_notes.append(note)

    broken_payload_refs: List[str] = []
    payload_mismatches: List[Dict[str, Any]] = []
    if player_payload:
        lookup = {
            (int(p["string"]), int(p["fret"])): p
            for p in (player_payload.get("positions") or [])
        }
        for row in mapping:
            key = (int(row["string_number"]), int(row["fret"]))
            pos = lookup.get(key)
            if not pos:
                broken_payload_refs.append(f"S{row['string_number']} fret {row['fret']}: missing position")
                continue
            expected_note = str(row["note_name"])
            actual_note = normalize_note_name(str(pos.get("note_name") or pos.get("note_id", "")))
            if actual_note != expected_note and note_id_from_note_name(expected_note) != str(pos.get("note_id")):
                payload_mismatches.append(
                    {
                        "string": row["string_number"],
                        "fret": row["fret"],
                        "expected_note": expected_note,
                        "payload_note": actual_note,
                        "note_id": pos.get("note_id"),
                    }
                )

    if not explicit_ok:
        readiness = "failed_wrong_note_mapping"
    elif missing_cache_notes:
        readiness = "failed_missing_required_notes"
    elif broken_payload_refs or payload_mismatches:
        readiness = "failed_wrong_note_mapping"
    else:
        readiness = "ready_fretboard_mapping"

    audit: Dict[str, Any] = {
        "readiness": readiness,
        "instrument": cfg.get("instrument"),
        "tuning": tuning,
        "fret_count": fc,
        "string_visual_order_top_to_bottom": cfg.get("string_visual_order_top_to_bottom"),
        "open_strings_side": cfg.get("open_strings_side"),
        "higher_frets_direction": cfg.get("higher_frets_direction"),
        "lowest_required_note": required[0] if required else "",
        "highest_required_note": required[-1] if required else "",
        "total_clickable_positions": 6 * (fc + 1),
        "unique_required_note_count": len(required),
        "generated_fretboard": generated_map,
        "explicit_checks": explicit_checks,
        "explicit_checks_passed": explicit_ok,
        "missing_cache_notes": sorted(missing_cache_notes, key=note_to_midi),
        "broken_player_references": broken_payload_refs,
        "payload_note_mismatches": payload_mismatches,
        "config_path": str(root / "config" / "classical_guitar_fretboard.json").replace("\\", "/"),
    }

    json_path = root / "audio" / "debug_reports" / "app_stk_fretboard_mapping_audit.json"
    md_path = root / "audio" / "debug_reports" / "app_stk_fretboard_mapping_audit.md"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    md_lines = [
        "# APP STK Fretboard Mapping Audit",
        "",
        f"- **readiness**: {readiness}",
        f"- **tuning**: {tuning}",
        f"- **fret_count**: {fc}",
        f"- **lowest_required_note**: {audit['lowest_required_note']}",
        f"- **highest_required_note**: {audit['highest_required_note']}",
        f"- **unique_required_note_count**: {audit['unique_required_note_count']}",
        f"- **total_clickable_positions**: {audit['total_clickable_positions']}",
        "",
        "## Explicit checks",
        "",
    ]
    for row in explicit_checks:
        mark = "ok" if row["passed"] else "FAIL"
        md_lines.append(
            f"- [{mark}] {row['string']} fret {row['fret']}: expected {row['expected']}, got {row['actual']}"
        )
    if missing_cache_notes:
        md_lines.extend(["", "## Missing cache notes", "", f"{missing_cache_notes}"])
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    audit["report_json"] = str(json_path).replace("\\", "/")
    audit["report_md"] = str(md_path).replace("\\", "/")
    return audit


if __name__ == "__main__":
    path = ensure_fretboard_config_file()
    print(f"Wrote {path}")
