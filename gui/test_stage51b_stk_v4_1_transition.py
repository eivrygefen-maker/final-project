#!/usr/bin/env python3
"""Stage 5.1B STK V4.1 transition-band validation tests."""
from __future__ import annotations

import json
import shutil
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gui"))

from body_hybrid_v4_1 import radiation_blend_weight_f0  # noqa: E402
from build_sample_comparison import parse_notes_arg, synthetic_modal_for_sample  # noqa: E402
from stage51b_stk_v4_1_transition_report import (  # noqa: E402
    build_stage51b_transition_report,
    classify_crossfade_mode,
)


class TestStage51BTransition(unittest.TestCase):
    def test_transition_notes_parse(self) -> None:
        notes = parse_notes_arg("E3,A3,D4")
        self.assertEqual([n for n, _ in notes], ["E3", "A3", "D4"])
        freqs = {n: f for n, f in notes}
        self.assertAlmostEqual(freqs["E3"], 164.81, places=1)
        self.assertAlmostEqual(freqs["A3"], 220.0, places=1)
        self.assertAlmostEqual(freqs["D4"], 293.66, places=1)

    def test_w_rad_in_transition_band(self) -> None:
        w_e3 = radiation_blend_weight_f0(164.81)
        w_a3 = radiation_blend_weight_f0(220.0)
        w_d4 = radiation_blend_weight_f0(293.66)
        self.assertLess(w_e3, 0.25)
        self.assertGreater(w_a3, w_e3)
        self.assertLess(w_a3, w_d4)
        self.assertEqual(classify_crossfade_mode(w_e3), "near_baseline")
        self.assertEqual(classify_crossfade_mode(w_a3), "transition")
        self.assertIn(classify_crossfade_mode(w_d4), ("transition", "near_v1"))

    def test_build_transition_no_fem(self) -> None:
        from build_stk_v4_1_transition_diagnostics import build_v4_1_transition_diagnostics

        out = REPO / "audio" / "_test_stk_v41_transition"
        if out.exists():
            shutil.rmtree(out)
        notes = parse_notes_arg("E3,A3,D4")
        with patch(
            "build_stk_v4_1_transition_diagnostics.resolve_modal_data_for_sample",
            side_effect=lambda repo, sample, use_surrogate: (
                synthetic_modal_for_sample(str(sample["sample_id"])),
                "synthetic_fallback",
            ),
        ):
            manifest = build_v4_1_transition_diagnostics(
                repo_root=REPO,
                out_dir=out,
                notes=notes,
                modes=[
                    "baseline_current",
                    "modal_radiation_color_v1",
                    "modal_body_hybrid_v4_1_core",
                    "modal_body_hybrid_v4_1_full",
                ],
                max_samples=3,
                duration_s=0.2,
                use_surrogate=False,
            )
        self.assertFalse(manifest["fem_launched"])
        report_path = REPO / "audio" / "debug_reports" / "stage51b_stk_v4_1_transition_report.json"
        self.assertTrue(report_path.is_file())
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report.get("stage"), "5.1B")
        self.assertFalse(report.get("uses_v3_v2_as_base"))
        self.assertIn("E3", report.get("per_note_comparison") or {})
        self.assertTrue(report.get("core_full_identity", {}).get("core_full_identical"))


if __name__ == "__main__":
    unittest.main()
