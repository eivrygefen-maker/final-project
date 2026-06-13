#!/usr/bin/env python3
"""Stage 5.1F STK V4.1 identity+contrast hybrid tests."""
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

from body_hybrid_v4_1_identity_space import (  # noqa: E402
    HYBRID_BLEND_RATIOS,
    apply_hybrid_audibility_floor,
    build_batch_contrast_context,
    compute_identity_layer_residual,
    hybrid_blend_for_mode,
    is_hybrid_identity_mode,
    is_v4_1_identity_space_mode,
    requires_identity_contrast_context,
    STRENGTH_PROFILES,
)
from build_sample_comparison import synthetic_modal_for_sample  # noqa: E402
from diagnostic_synthesis import list_diagnostic_modes  # noqa: E402


class TestStage51FIdentityContrastHybrid(unittest.TestCase):
    def test_modes_exist(self) -> None:
        for mode in (
            "modal_body_hybrid_v4_1_identity_contrast_hybrid",
            "stk_body_transfer_v4_1_identity_contrast_hybrid",
            "modal_body_hybrid_v4_1_identity_contrast_hybrid_25_75",
            "modal_body_hybrid_v4_1_identity_contrast_hybrid_40_60",
            "modal_body_hybrid_v4_1_identity_contrast_hybrid_50_50",
        ):
            self.assertIn(mode, list_diagnostic_modes())
            self.assertTrue(is_v4_1_identity_space_mode(mode))
            self.assertTrue(is_hybrid_identity_mode(mode))
            self.assertTrue(requires_identity_contrast_context(mode))

    def test_hybrid_blend_ratios(self) -> None:
        self.assertEqual(hybrid_blend_for_mode("modal_body_hybrid_v4_1_identity_contrast_hybrid_25_75"), (0.25, 0.75))
        self.assertEqual(hybrid_blend_for_mode("modal_body_hybrid_v4_1_identity_contrast_hybrid_40_60"), (0.40, 0.60))
        self.assertEqual(hybrid_blend_for_mode("modal_body_hybrid_v4_1_identity_contrast_hybrid_50_50"), (0.50, 0.50))
        self.assertEqual(hybrid_blend_for_mode("modal_body_hybrid_v4_1_identity_contrast_hybrid"), (0.40, 0.60))

    def test_layer_residual_bounded(self) -> None:
        from body_hybrid_v4_1_identity_space import build_body_identity_vector

        modal = synthetic_modal_for_sample("sample_002")
        z = build_body_identity_vector(
            parameters={"top_wood_id": "spruce"},
            modal_data=modal,
            frequency_hz=220.0,
        )
        base = np.random.randn(4096) * 0.02
        res, _ = compute_identity_layer_residual(
            base,
            frequency_hz=220.0,
            sample_rate=44100,
            feature_source=z,
            profile=STRENGTH_PROFILES["strong"],
        )
        cap = STRENGTH_PROFILES["strong"].residual_gain_max * float(np.max(np.abs(base)))
        self.assertLessEqual(float(np.max(np.abs(res))), cap + 0.01)

    def test_audibility_floor_raises_quiet_residual(self) -> None:
        base = np.ones(1000) * 0.1
        quiet = base + 1e-5 * np.random.randn(1000)
        out, info = apply_hybrid_audibility_floor(quiet, base, min_db=-40.0, target_db=-26.0)
        self.assertTrue(info.get("audibility_adjusted"))
        self.assertGreater(info.get("rms_diff_db_after", -100), info.get("rms_diff_db_before", -100))

    def test_contrast_not_hybrid(self) -> None:
        self.assertTrue(requires_identity_contrast_context("modal_body_hybrid_v4_1_identity_contrast_strong"))
        self.assertFalse(is_hybrid_identity_mode("modal_body_hybrid_v4_1_identity_contrast_strong"))

    def test_no_note_name_branch(self) -> None:
        import body_hybrid_v4_1_identity_space as mod

        src = inspect.getsource(mod)
        for token in ("A3", "E3", "D4", "A2"):
            self.assertNotIn(f'"{token}"', src)

    def test_build_no_fem(self) -> None:
        from build_stk_v4_1_identity_contrast_hybrid_diagnostics import build_v4_1_identity_contrast_hybrid_diagnostics

        out = REPO / "audio" / "_test_stk_v41_hybrid"
        if out.exists():
            shutil.rmtree(out)
        with patch(
            "build_stk_v4_1_identity_contrast_hybrid_diagnostics.resolve_modal_data_for_sample",
            side_effect=lambda repo, sample, use_surrogate: (
                synthetic_modal_for_sample(str(sample["sample_id"])),
                "synthetic_fallback",
            ),
        ):
            manifest = build_v4_1_identity_contrast_hybrid_diagnostics(
                repo_root=REPO,
                out_dir=out,
                notes=[("A3", 220.0)],
                modes=[
                    "modal_body_hybrid_v4_1_full",
                    "modal_body_hybrid_v4_1_identity_contrast_hybrid_40_60",
                ],
                max_samples=3,
                duration_s=0.2,
                use_surrogate=False,
            )
        self.assertFalse(manifest["fem_launched"])
        report_path = REPO / "audio" / "debug_reports" / "stage51f_stk_v4_1_identity_contrast_hybrid_report.json"
        self.assertTrue(report_path.is_file())
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report.get("stage"), "5.1F")


if __name__ == "__main__":
    unittest.main()
