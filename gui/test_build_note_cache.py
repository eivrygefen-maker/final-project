#!/usr/bin/env python3
"""Note-cache builder tests (no FEM)."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gui"))

from build_note_cache import (  # noqa: E402
    DEFAULT_TUNING,
    build_frequency_ordered_preview,
    build_note_cache,
    enumerate_fretboard_positions,
    group_unique_pitches,
    note_id_from_frequency,
    pitch_dedup_key,
    position_frequency,
)
from body_response_synth import read_wav_float_mono  # noqa: E402
from body_response_synth import BODY_MODAL_RICHNESS_GAIN  # noqa: E402


class NoteCacheBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.out_root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_frequency_mapping(self) -> None:
        e2 = position_frequency(6, 0)
        self.assertAlmostEqual(e2, 82.41, places=2)
        self.assertEqual(note_id_from_frequency(e2), "E2")

        a2_s6_f5 = position_frequency(6, 5)
        a2_s5_f0 = position_frequency(5, 0)
        self.assertEqual(pitch_dedup_key(a2_s6_f5), pitch_dedup_key(a2_s5_f0))
        self.assertAlmostEqual(a2_s6_f5, 110.0, places=1)
        self.assertEqual(note_id_from_frequency(a2_s6_f5), "A2")

        e4 = position_frequency(1, 0)
        self.assertAlmostEqual(e4, 329.63, places=1)
        self.assertEqual(note_id_from_frequency(e4), "E4")

    def test_deduplication_groups_duplicate_pitches(self) -> None:
        positions = enumerate_fretboard_positions(5, tuning=DEFAULT_TUNING)
        unique = group_unique_pitches(positions)
        self.assertLess(len(unique), len(positions))
        self.assertIn("A2", unique)
        self.assertEqual(unique["A2"]["note_id"], "A2")

    def test_build_note_cache_manifest_and_files(self) -> None:
        manifest = build_note_cache(
            modal_json=Path("__missing_modal_for_synthetic__.json"),
            out_root=self.out_root,
            fret_count=5,
            duration_s=0.22,
            sample_rate=44100,
            force=True,
        )
        cache_root = Path(manifest["cache_root"])
        manifest_path = cache_root / "note_manifest.json"
        self.assertTrue(manifest_path.is_file())

        with open(manifest_path, encoding="utf-8") as fh:
            on_disk = json.load(fh)

        self.assertEqual(on_disk["schema_version"], "note_cache_v1")
        self.assertEqual(on_disk["fret_count"], 5)
        self.assertEqual(on_disk["playable_position_count"], 6 * 6)
        self.assertLess(on_disk["unique_note_count"], on_disk["playable_position_count"])
        self.assertEqual(on_disk["body_modal_richness_gain"], BODY_MODAL_RICHNESS_GAIN)

        for note in on_disk["notes"]:
            wav = cache_root / note["wav_path"]
            meta = cache_root / note["metadata_path"]
            self.assertTrue(wav.is_file() and wav.stat().st_size > 44, note["note_id"])
            self.assertTrue(meta.is_file(), note["note_id"])
            meta_doc = json.loads(meta.read_text(encoding="utf-8"))
            self.assertEqual(meta_doc["body_modal_richness_gain"], BODY_MODAL_RICHNESS_GAIN)
            self.assertIn("output_rms_dbfs", meta_doc)
            self.assertIn("output_decay_slope_db_per_s", meta_doc)
            self.assertIn("late_to_early_rms_db", meta_doc)
            self.assertTrue(meta_doc.get("anti_click_taper_applied"))
            self.assertIn("body_to_string_rms_ratio_before_loudness", meta_doc)
            self.assertIn("body_modal_bandwidth_widening", meta_doc)

        for pos in on_disk["positions"]:
            wav = cache_root / pos["wav_path"]
            self.assertTrue(wav.is_file(), f"s{pos['string_number']} f{pos['fret']}")

        # Duplicate pitch: string 6 fret 5 and string 5 fret 0 -> same note_id/wav
        s6f5 = next(p for p in on_disk["positions"] if p["string_number"] == 6 and p["fret"] == 5)
        s5f0 = next(p for p in on_disk["positions"] if p["string_number"] == 5 and p["fret"] == 0)
        self.assertEqual(s6f5["note_id"], "A2")
        self.assertEqual(s5f0["note_id"], "A2")
        self.assertEqual(s6f5["wav_path"], s5f0["wav_path"])

    def test_preview_concatenation_near_zero_boundaries(self) -> None:
        manifest = build_note_cache(
            modal_json=Path("__missing_modal_for_synthetic__.json"),
            out_root=self.out_root / "preview",
            fret_count=3,
            duration_s=0.2,
            sample_rate=44100,
            force=True,
        )
        preview = build_frequency_ordered_preview(
            Path(manifest["cache_root"]),
            Path(manifest["cache_root"]) / "all_notes_preview.wav",
        )
        samples, _ = read_wav_float_mono(Path(preview["preview_wav"]))
        self.assertGreater(samples.size, 0)
        self.assertLess(abs(float(samples[0])), 0.05)
        self.assertLess(abs(float(samples[-1])), 0.01)

    def test_high_note_uses_hf_fallback_in_cache(self) -> None:
        manifest = build_note_cache(
            modal_json=Path("__missing_modal_for_synthetic__.json"),
            out_root=self.out_root / "hf",
            fret_count=9,
            duration_s=0.18,
            sample_rate=44100,
            force=True,
        )
        high_notes = [n for n in manifest["notes"] if n.get("high_frequency_fallback_used")]
        self.assertGreater(len(high_notes), 0)
        cache_root = Path(manifest["cache_root"])
        meta = json.loads((cache_root / high_notes[0]["metadata_path"]).read_text(encoding="utf-8"))
        self.assertTrue(meta["high_frequency_fallback_used"])


if __name__ == "__main__":
    unittest.main()
