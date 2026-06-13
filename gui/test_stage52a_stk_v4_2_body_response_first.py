#!/usr/bin/env python3
"""Stage 5.2A STK body-response-first V4.2 tests."""
from __future__ import annotations

import inspect
import json
import shutil
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gui"))

from body_response_first_v4_2 import (  # noqa: E402
    build_body_transfer_function_v4_2,
    is_v4_2_body_response_first_mode,
    synthesize_body_response_first_v4_2_note,
)
from build_sample_comparison import synthetic_modal_for_sample  # noqa: E402
from diagnostic_synthesis import list_diagnostic_modes  # noqa: E402


class TestStage52ABodyResponseFirst(unittest.TestCase):
    def test_modes_exist(self) -> None:
        for mode in ("modal_body_response_first_v4_2", "stk_body_response_first_v4_2"):
            self.assertIn(mode, list_diagnostic_modes())
            self.assertTrue(is_v4_2_body_response_first_mode(mode))

    def test_H_guitar_bounded(self) -> None:
        modal = synthetic_modal_for_sample("sample_002")
        from body_response_synth import modes_in_validated_band, parse_modal_modes

        all_modes, _ = parse_modal_modes(modal)
        band = modes_in_validated_band(all_modes)
        H, rows, summary = build_body_transfer_function_v4_2(
            sample_rate=44100,
            n_samples=4410,
            band_modes=band,
            frequency_hz=220.0,
            parameters={"top_wood_id": "spruce", "back_wood_id": "mahogany"},
            repo_root=REPO,
            sample_id="sample_002",
        )
        self.assertGreater(len(rows), 0)
        self.assertAlmostEqual(summary["H_peak_normalized"], 1.0, places=4)
        for row in rows:
            amp = row["amplitude_meta"]["A_m_note"]
            self.assertGreater(amp, 0)
            self.assertLessEqual(amp, 2.5)
        self.assertGreater(float(np.max(np.abs(H))), 0)

    def test_no_v1_in_module(self) -> None:
        import body_response_first_v4_2 as mod

        src = inspect.getsource(mod)
        self.assertNotIn("modal_radiation_color_v1", src)
        self.assertNotIn("synthesize_hybrid_v4_1", src)

    def test_no_note_name_branch(self) -> None:
        import body_response_first_v4_2 as mod

        src = inspect.getsource(mod)
        for token in ("A3", "E3", "D4", "A2"):
            self.assertNotIn(f'"{token}"', src)

    def test_synthesize_produces_audio(self) -> None:
        modal = synthetic_modal_for_sample("sample_001")
        out = REPO / "audio" / "_test_v42_single.wav"
        meta = synthesize_body_response_first_v4_2_note(
            frequency_hz=220.0,
            note_name="A3",
            duration_s=0.25,
            sample_rate=44100,
            modal_data=modal,
            output_wav=out,
            output_metadata_json=None,
            velocity=1.0,
            sample_parameters={"top_wood_id": "spruce"},
            modal_source="synthetic",
            diagnostic_mode="modal_body_response_first_v4_2",
            synthesis_preset=None,
            repo_root=REPO,
            sample_id="sample_001",
        )
        self.assertTrue(out.is_file())
        self.assertTrue(meta.get("body_response_first_v4_2_active"))
        self.assertFalse(meta.get("v1_overlay_used"))

    def test_build_no_fem(self) -> None:
        from build_stk_v4_2_body_response_first_diagnostics import build_v4_2_body_response_first_diagnostics

        out = REPO / "audio" / "_test_stk_v42_body_first"
        if out.exists():
            shutil.rmtree(out)
        with patch(
            "build_stk_v4_2_body_response_first_diagnostics.resolve_modal_data_for_sample",
            side_effect=lambda repo, sample, use_surrogate: (
                synthetic_modal_for_sample(str(sample["sample_id"])),
                "synthetic_fallback",
            ),
        ):
            manifest = build_v4_2_body_response_first_diagnostics(
                repo_root=REPO,
                out_dir=out,
                notes=[("A3", 220.0)],
                modes=[
                    "modal_body_hybrid_v4_1_full",
                    "modal_body_hybrid_v4_1_identity_contrast_strong",
                    "modal_body_response_first_v4_2",
                ],
                max_samples=3,
                duration_s=0.2,
                use_surrogate=False,
            )
        self.assertFalse(manifest["fem_launched"])
        report_path = REPO / "audio" / "debug_reports" / "stage52a_stk_v4_2_body_response_first_report.json"
        self.assertTrue(report_path.is_file())
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report.get("stage"), "5.2A")
        self.assertTrue(report.get("v4_1_unchanged"))


if __name__ == "__main__":
    unittest.main()
