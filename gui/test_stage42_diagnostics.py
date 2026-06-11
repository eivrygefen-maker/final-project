#!/usr/bin/env python3
"""Stage 4.2 diagnostic synthesis + UI structure tests (no FEM)."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gui"))

from build_sample_comparison import (  # noqa: E402
    build_sample_comparisons,
    parse_diagnostic_modes_arg,
)
from diagnostic_synthesis import (  # noqa: E402
    get_diagnostic_mode,
    list_diagnostic_modes,
)


class Stage42DiagnosticTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.out_dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_diagnostic_modes_accepted(self) -> None:
        modes = list_diagnostic_modes()
        self.assertIn("baseline_current", modes)
        self.assertIn("preserve_raw_body_variation", modes)
        parsed = parse_diagnostic_modes_arg(",".join(modes))
        self.assertEqual(len(parsed), len(modes))
        with self.assertRaises(ValueError):
            get_diagnostic_mode("not_a_real_mode")

    def test_output_folder_structure_and_manifest(self) -> None:
        samples = [
            {"sample_id": f"sample_{i:03d}", "run_id": "", "parameters": {"top_wood_id": "spruce"}}
            for i in range(3)
        ]
        mode_dir = self.out_dir / "preserve_raw_body_variation"
        manifest = build_sample_comparisons(
            repo_root=REPO,
            out_dir=mode_dir,
            samples=samples,
            notes=(("A2", 110.0),),
            duration_s=0.1,
            silence_s=0.03,
            use_surrogate=False,
            diagnostic_mode="preserve_raw_body_variation",
        )
        self.assertTrue((mode_dir / "A2_26_guitars.wav").is_file())
        self.assertTrue((mode_dir / "comparison_manifest.json").is_file())
        self.assertTrue((mode_dir / "mode_summary.json").is_file())
        segs = manifest["notes"][0]["segments"]
        self.assertEqual(len(segs), 3)
        for seg in segs:
            self.assertEqual(seg["diagnostic_mode"], "preserve_raw_body_variation")
            self.assertIn("note_reward_score", seg)
            self.assertIn("raw_body_rms_before_normalization", seg)

    def test_deterministic_generation(self) -> None:
        samples = [{"sample_id": "sample_000", "run_id": "", "parameters": {}}]
        a_dir = self.out_dir / "a"
        b_dir = self.out_dir / "b"
        m1 = build_sample_comparisons(
            repo_root=REPO,
            out_dir=a_dir,
            samples=samples,
            notes=(("E5", 659.25),),
            duration_s=0.08,
            silence_s=0.0,
            use_surrogate=False,
            diagnostic_mode="baseline_current",
        )
        m2 = build_sample_comparisons(
            repo_root=REPO,
            out_dir=b_dir,
            samples=samples,
            notes=(("E5", 659.25),),
            duration_s=0.08,
            silence_s=0.0,
            use_surrogate=False,
            diagnostic_mode="baseline_current",
        )
        self.assertEqual(
            m1["notes"][0]["segments"][0]["note_reward_score"],
            m2["notes"][0]["segments"][0]["note_reward_score"],
        )

    def test_no_fem_import_in_comparison_builder(self) -> None:
        src = (REPO / "gui" / "build_sample_comparison.py").read_text(encoding="utf-8")
        self.assertNotIn("run_fem", src.lower())
        self.assertNotIn("fem_solve", src.lower())


class Stage42UiStructureTests(unittest.TestCase):
    def test_run_rom_removed_from_fast_preview(self) -> None:
        html = (REPO / "gui" / "components/fast_preview/index.html").read_text(encoding="utf-8")
        self.assertNotIn('id="btnRom"', html)
        self.assertNotIn("run_rom", html)

    def test_guitar_player_bridge_and_strings(self) -> None:
        html = (REPO / "gui" / "components/guitar_player/index.html").read_text(encoding="utf-8")
        self.assertIn("bridge-side", html)
        self.assertIn("bridge-bar", html)
        self.assertIn("string-row::after", html)
        self.assertIn("fret-cell", html)
        self.assertIn("open-string-hit", html)

    def test_step_headings_enlarged(self) -> None:
        app_src = (REPO / "gui" / "app.py").read_text(encoding="utf-8")
        self.assertIn("user-step-heading", app_src)
        self.assertIn("3.6rem", app_src)
        self.assertIn("user-gen-sound-heading", app_src)


if __name__ == "__main__":
    unittest.main()
