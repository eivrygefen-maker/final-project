#!/usr/bin/env python3
"""Stage 4 interactive guitar player helpers (no FEM, no Streamlit runtime)."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gui"))

from build_note_cache import build_note_cache  # noqa: E402
from note_cache_ui import (  # noqa: E402
    FRETBOARD_DISPLAY_STRING_ORDER,
    OPEN_STRING_NOTE_IDS,
    build_player_payload,
    build_position_lookup,
    fretboard_display_fret_order,
    fretboard_screen_position,
    list_manifest_paths,
    lookup_position,
    note_cache_ui_status,
    prepare_player_assets,
    resolve_note_cache,
    resolve_wav_path,
)


class FretboardUiTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.out_root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_list_manifests_and_resolve_wav_path(self) -> None:
        manifest = build_note_cache(
            modal_json=Path("__missing_modal__.json"),
            out_root=self.out_root,
            fret_count=5,
            duration_s=0.15,
            force=True,
        )
        cache_root = Path(manifest["cache_root"])
        manifests = list_manifest_paths(self.out_root)
        self.assertEqual(len(manifests), 1)
        loaded = json.loads(manifests[0].read_text(encoding="utf-8"))
        note = loaded["notes"][0]
        wav = resolve_wav_path(cache_root, note["wav_path"])
        self.assertTrue(wav.is_file())

    def test_position_lookup_a2_duplicate_pitches(self) -> None:
        manifest = build_note_cache(
            modal_json=Path("__missing_modal__.json"),
            out_root=self.out_root,
            fret_count=5,
            duration_s=0.15,
            force=True,
        )
        doc = json.loads((Path(manifest["cache_root"]) / "note_manifest.json").read_text(encoding="utf-8"))
        lookup = build_position_lookup(doc)
        s6f5 = lookup_position(lookup, 6, 5)
        s5f0 = lookup_position(lookup, 5, 0)
        self.assertIsNotNone(s6f5)
        self.assertIsNotNone(s5f0)
        self.assertEqual(s6f5["note_id"], "A2")
        self.assertEqual(s5f0["note_id"], "A2")
        self.assertEqual(s6f5["wav_path"], s5f0["wav_path"])
        wav = resolve_wav_path(Path(manifest["cache_root"]), s6f5["wav_path"])
        self.assertTrue(wav.is_file())

    def test_missing_cache_no_crash(self) -> None:
        empty = self.out_root / "empty"
        empty.mkdir()
        resolved = resolve_note_cache(empty, expected_fingerprint="deadbeef")
        self.assertEqual(resolved["status"], "missing")
        self.assertIsNone(resolved["manifest"])

    def test_stale_when_fingerprint_mismatch(self) -> None:
        build_note_cache(
            modal_json=Path("__missing_modal__.json"),
            out_root=self.out_root,
            fret_count=3,
            duration_s=0.12,
            force=True,
        )
        resolved = resolve_note_cache(
            self.out_root,
            expected_fingerprint="0" * 64,
        )
        self.assertEqual(resolved["status"], "stale")
        self.assertIsNotNone(resolved["manifest"])

    def test_build_player_payload_ready(self) -> None:
        manifest = build_note_cache(
            modal_json=Path("__missing_modal__.json"),
            out_root=self.out_root,
            fret_count=3,
            duration_s=0.12,
            force=True,
        )
        cache_root = Path(manifest["cache_root"])
        resolved = resolve_note_cache(self.out_root, expected_fingerprint=manifest["guitar_fingerprint"])
        payload = build_player_payload(resolved, ui_status="ready")
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["fingerprint"], manifest["guitar_fingerprint"])
        self.assertGreater(len(payload["positions"]), 0)
        self.assertTrue(all("wav" in p and p["wav"].endswith(".wav") for p in payload["positions"]))

    def test_prepare_player_assets_copies_wavs(self) -> None:
        manifest = build_note_cache(
            modal_json=Path("__missing_modal__.json"),
            out_root=self.out_root,
            fret_count=3,
            duration_s=0.12,
            force=True,
        )
        cache_root = Path(manifest["cache_root"])
        doc = json.loads((cache_root / "note_manifest.json").read_text(encoding="utf-8"))
        dest = prepare_player_assets(cache_root, doc)
        self.assertTrue(dest.is_dir())
        first_note = doc["notes"][0]["note_id"]
        self.assertTrue((dest / f"{first_note}.wav").is_file())

    def test_fretboard_orientation_open_low_e_upper_right(self) -> None:
        fret_count = 19
        row, col = fretboard_screen_position(6, 0, fret_count=fret_count)
        self.assertEqual(row, 0)
        self.assertEqual(col, fret_count)
        self.assertEqual(FRETBOARD_DISPLAY_STRING_ORDER[0], 6)
        self.assertEqual(fretboard_display_fret_order(fret_count)[-1], 0)
        row_e4, col_e4 = fretboard_screen_position(1, 0, fret_count=fret_count)
        self.assertEqual(row_e4, 5)
        self.assertEqual(col_e4, fret_count)

    def test_open_string_note_mapping(self) -> None:
        manifest = build_note_cache(
            modal_json=Path("__missing_modal__.json"),
            out_root=self.out_root,
            fret_count=5,
            duration_s=0.12,
            force=True,
        )
        doc = json.loads((Path(manifest["cache_root"]) / "note_manifest.json").read_text(encoding="utf-8"))
        lookup = build_position_lookup(doc)
        for sn, expected_note in OPEN_STRING_NOTE_IDS.items():
            pos = lookup_position(lookup, sn, 0)
            self.assertIsNotNone(pos, f"string {sn} open")
            self.assertEqual(pos["note_id"], expected_note)

    def test_player_payload_preserves_manifest_positions(self) -> None:
        manifest = build_note_cache(
            modal_json=Path("__missing_modal__.json"),
            out_root=self.out_root,
            fret_count=5,
            duration_s=0.12,
            force=True,
        )
        resolved = resolve_note_cache(self.out_root, expected_fingerprint=manifest["guitar_fingerprint"])
        payload = build_player_payload(resolved, ui_status="ready")
        s6f5 = next(p for p in payload["positions"] if p["string"] == 6 and p["fret"] == 5)
        self.assertEqual(s6f5["note_id"], "A2")
        self.assertTrue(s6f5["wav"].endswith(".wav"))

    def test_note_cache_ui_status_hidden_until_generate(self) -> None:
        manifest = build_note_cache(
            modal_json=Path("__missing_modal__.json"),
            out_root=self.out_root,
            fret_count=3,
            duration_s=0.12,
            force=True,
        )
        resolved = resolve_note_cache(self.out_root, expected_fingerprint=manifest["guitar_fingerprint"])
        self.assertEqual(
            note_cache_ui_status(
                sound_stale=True,
                note_cache_ready_fp=manifest["guitar_fingerprint"],
                expected_fingerprint=manifest["guitar_fingerprint"],
                resolved=resolved,
            ),
            "hidden",
        )
        self.assertEqual(
            note_cache_ui_status(
                sound_stale=False,
                note_cache_ready_fp="",
                expected_fingerprint=manifest["guitar_fingerprint"],
                resolved=resolved,
            ),
            "hidden",
        )
        self.assertEqual(
            note_cache_ui_status(
                sound_stale=False,
                note_cache_ready_fp=manifest["guitar_fingerprint"],
                expected_fingerprint=manifest["guitar_fingerprint"],
                resolved=resolved,
            ),
            "ready",
        )


if __name__ == "__main__":
    unittest.main()
