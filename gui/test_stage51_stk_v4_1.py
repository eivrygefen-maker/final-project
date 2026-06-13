#!/usr/bin/env python3
"""Stage 5.1 STK V4.1 strict hybrid and ROM readiness tests."""
from __future__ import annotations

import inspect
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gui"))

from body_hybrid_v4_1 import (  # noqa: E402
    V4_1_F_HIGH,
    V4_1_F_LOW,
    equal_loudness_crossfade,
    get_v4_1_ablation,
    is_v4_1_family_mode,
    radiation_blend_weight_f0,
)
from build_sample_comparison import synthetic_modal_for_sample  # noqa: E402
from diagnostic_synthesis import list_diagnostic_modes  # noqa: E402
from stage51_rom_readiness_report import build_rom_readiness_report, load_lhs_pool_entries  # noqa: E402


class TestStage51StkV41(unittest.TestCase):
    def test_v4_1_modes_exist(self) -> None:
        for mode in (
            "modal_body_hybrid_v4_1",
            "modal_body_hybrid_v4_1_core",
            "modal_body_hybrid_v4_1_full",
            "stk_body_transfer_v4_1",
        ):
            self.assertIn(mode, list_diagnostic_modes())
            self.assertTrue(is_v4_1_family_mode(mode))

    def test_w_rad_continuous_and_endpoints(self) -> None:
        vals = [radiation_blend_weight_f0(f) for f in [80, 110, 160, 240, 320, 440, 659]]
        for a, b in zip(vals, vals[1:]):
            self.assertLessEqual(a - 1e-9, b)
        self.assertLess(radiation_blend_weight_f0(110.0), 0.02)
        self.assertGreaterEqual(radiation_blend_weight_f0(160.0), 0.0)
        self.assertGreaterEqual(radiation_blend_weight_f0(320.0), 1.0 - 1e-6)
        self.assertGreaterEqual(radiation_blend_weight_f0(440.0), 1.0 - 1e-6)

    def test_no_note_name_branch(self) -> None:
        import body_hybrid_v4_1 as mod

        src = inspect.getsource(mod)
        for token in ("A2", "A4", "E5"):
            self.assertNotIn(f'"{token}"', src)

    def test_low_endpoint_crossfade_is_baseline(self) -> None:
        b = np.sin(np.linspace(0, 4 * np.pi, 512))
        v = np.cos(np.linspace(0, 4 * np.pi, 512))
        out, info = equal_loudness_crossfade(b, v, 0.0)
        np.testing.assert_array_almost_equal(out, b)
        self.assertEqual(info["crossfade_mode"], "baseline_only")

    def test_high_endpoint_crossfade_is_v1(self) -> None:
        b = np.sin(np.linspace(0, 4 * np.pi, 512))
        v = np.cos(np.linspace(0, 4 * np.pi, 512))
        out, info = equal_loudness_crossfade(b, v, 1.0)
        np.testing.assert_array_almost_equal(out, v)
        self.assertEqual(info["crossfade_mode"], "v1_only")

    def test_v4_1_a2_delegates_to_baseline(self) -> None:
        from body_hybrid_v4_1 import synthesize_hybrid_v4_1_note

        out = REPO / "audio" / "_test_v41_a2.wav"
        modal = synthetic_modal_for_sample("sample_003")
        meta = synthesize_hybrid_v4_1_note(
            frequency_hz=110.0,
            note_name="A2",
            duration_s=0.25,
            sample_rate=44100,
            modal_data=modal,
            output_wav=out,
            output_metadata_json=None,
            velocity=0.5,
            sample_parameters={"geometry.length": 0.46},
            modal_source="synthetic_fallback",
            diagnostic_mode="modal_body_hybrid_v4_1_core",
            synthesis_preset=None,
            repo_root=REPO,
            sample_id="sample_003",
        )
        self.assertEqual(meta.get("v4_1_endpoint"), "baseline_delegate")
        self.assertLess(float(meta.get("radiation_blend_weight_w_rad", 999)), 0.02)
        eq = meta.get("endpoint_equivalence") or {}
        self.assertEqual(eq.get("endpoint_equivalence_error_to_baseline"), 0.0)

    def test_v4_1_e5_delegates_to_v1(self) -> None:
        from body_hybrid_v4_1 import synthesize_hybrid_v4_1_note

        out = REPO / "audio" / "_test_v41_e5.wav"
        modal = synthetic_modal_for_sample("sample_004")
        meta = synthesize_hybrid_v4_1_note(
            frequency_hz=659.25,
            note_name="E5",
            duration_s=0.25,
            sample_rate=44100,
            modal_data=modal,
            output_wav=out,
            output_metadata_json=None,
            velocity=0.5,
            sample_parameters={},
            modal_source="synthetic_fallback",
            diagnostic_mode="modal_body_hybrid_v4_1_core",
            synthesis_preset=None,
            repo_root=REPO,
            sample_id="sample_004",
        )
        self.assertEqual(meta.get("v4_1_endpoint"), "v1_delegate")
        self.assertGreater(float(meta.get("radiation_blend_weight_w_rad") or 0), 0.99)

    def test_rom_readiness_parser(self) -> None:
        report = build_rom_readiness_report(REPO)
        self.assertIn("official_rom_dataset_jsonl", report)
        self.assertIn("rom_model_manifest", report)
        self.assertIn("rom_retrain_state", report)
        self.assertIn("holdout_split", report)
        self.assertIn("readiness_status", report)
        entries = load_lhs_pool_entries(REPO)
        self.assertGreater(len(entries), 0)

    def test_build_no_fem(self) -> None:
        from build_stk_v4_1_diagnostics import build_v4_1_diagnostics

        out = REPO / "audio" / "_test_stk_v41_pack"
        if out.exists():
            import shutil

            shutil.rmtree(out)
        with patch(
            "build_stk_v4_1_diagnostics.resolve_modal_data_for_sample",
            side_effect=lambda repo, sample, use_surrogate: (
                synthetic_modal_for_sample(str(sample["sample_id"])),
                "synthetic_fallback",
            ),
        ):
            manifest = build_v4_1_diagnostics(
                repo_root=REPO,
                out_dir=out,
                notes=[("A2", 110.0), ("A4", 440.0)],
                modes=["baseline_current", "modal_body_hybrid_v4_1_core"],
                max_samples=3,
                duration_s=0.2,
                use_surrogate=False,
            )
        self.assertFalse(manifest["fem_launched"])
        report = REPO / "audio" / "debug_reports" / "stage51_stk_v4_1_report.json"
        self.assertTrue(report.is_file())
        doc = json.loads(report.read_text(encoding="utf-8"))
        self.assertFalse(doc.get("uses_v3_v2_as_base"))


if __name__ == "__main__":
    unittest.main()
