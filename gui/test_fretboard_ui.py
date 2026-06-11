#!/usr/bin/env python3
"""Stage 4 fretboard / note-cache UI helpers (no FEM, no Streamlit runtime)."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gui"))

from build_note_cache import build_note_cache  # noqa: E402
from note_cache_ui import (  # noqa: E402
    build_position_lookup,
    list_manifest_paths,
    lookup_position,
    resolve_note_cache,
    resolve_wav_path,
)


def _write_silent_wav(path: Path, *, duration_s: float = 0.05, sample_rate: int = 44100) -> None:
    n = max(1, int(duration_s * sample_rate))
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = b"\x00\x00" * n
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)


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


if __name__ == "__main__":
    unittest.main()
