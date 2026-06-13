#!/usr/bin/env python3
"""Stage 5.1G STK V4.1 physical differentiation tests."""
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
    G_BLEND_RATIOS,
    apply_bridge_coupling_to_residual,
    apply_decay_differentiation_to_residual,
    build_batch_contrast_context,
    compute_bridge_axis,
    compute_decay_axis,
    compose_hybrid_contrast_residual,
    g_config_for_mode,
    is_g_identity_mode,
    is_v4_1_identity_space_mode,
    requires_identity_contrast_context,
    STRENGTH_PROFILES,
)
from build_sample_comparison import synthetic_modal_for_sample  # noqa: E402
from diagnostic_synthesis import list_diagnostic_modes  # noqa: E402


class TestStage51GIdentityContrastG(unittest.TestCase):
    def test_modes_exist(self) -> None:
        modes = (
            "modal_body_hybrid_v4_1_identity_contrast_g_20_80",
            "modal_body_hybrid_v4_1_identity_contrast_g_25_75",
            "modal_body_hybrid_v4_1_identity_contrast_g_30_70",
            "modal_body_hybrid_v4_1_identity_contrast_g_25_75_decay",
            "modal_body_hybrid_v4_1_identity_contrast_g_25_75_bridge",
            "modal_body_hybrid_v4_1_identity_contrast_g_25_75_full",
            "stk_body_transfer_v4_1_identity_contrast_g",
        )
        for mode in modes:
            self.assertIn(mode, list_diagnostic_modes())
            self.assertTrue(is_v4_1_identity_space_mode(mode))
            self.assertTrue(is_g_identity_mode(mode))
            self.assertTrue(requires_identity_contrast_context(mode))

    def test_g_blend_ratios(self) -> None:
        self.assertEqual(g_config_for_mode("modal_body_hybrid_v4_1_identity_contrast_g_20_80")["absolute_weight"], 0.20)
        self.assertEqual(g_config_for_mode("modal_body_hybrid_v4_1_identity_contrast_g_25_75")["contrast_weight"], 0.75)
        self.assertEqual(g_config_for_mode("modal_body_hybrid_v4_1_identity_contrast_g_30_70")["absolute_weight"], 0.30)
        self.assertTrue(g_config_for_mode("modal_body_hybrid_v4_1_identity_contrast_g_25_75_decay")["decay_active"])
        self.assertFalse(g_config_for_mode("modal_body_hybrid_v4_1_identity_contrast_g_25_75_decay")["bridge_active"])
        self.assertTrue(g_config_for_mode("modal_body_hybrid_v4_1_identity_contrast_g_25_75_bridge")["bridge_active"])
        self.assertTrue(g_config_for_mode("modal_body_hybrid_v4_1_identity_contrast_g_25_75_full")["decay_active"])
        self.assertTrue(g_config_for_mode("modal_body_hybrid_v4_1_identity_contrast_g_25_75_full")["bridge_active"])
        alias = g_config_for_mode("stk_body_transfer_v4_1_identity_contrast_g")
        self.assertTrue(alias["decay_active"] and alias["bridge_active"])

    def test_physical_axes_bounded(self) -> None:
        from body_hybrid_v4_1_identity_space import build_body_identity_vector

        modal = synthetic_modal_for_sample("sample_003")
        z = build_body_identity_vector(
            parameters={"top_wood_id": "spruce", "back_wood_id": "mahogany"},
            modal_data=modal,
            frequency_hz=220.0,
        )
        self.assertGreaterEqual(compute_decay_axis(z), -1.0)
        self.assertLessEqual(compute_decay_axis(z), 1.0)
        self.assertGreaterEqual(compute_bridge_axis(z), -1.0)
        self.assertLessEqual(compute_bridge_axis(z), 1.0)

    def test_decay_and_bridge_change_residual(self) -> None:
        from body_hybrid_v4_1_identity_space import build_body_identity_vector

        modal = synthetic_modal_for_sample("sample_001")
        z = build_body_identity_vector(parameters={}, modal_data=modal, frequency_hz=220.0)
        base = np.random.randn(8192) * 0.02
        res, _ = compose_hybrid_contrast_residual(
            base,
            frequency_hz=220.0,
            sample_rate=44100,
            z_body=z,
            dz_body=z,
            abs_w=0.25,
            contrast_w=0.75,
        )
        dec, _ = apply_decay_differentiation_to_residual(res, z, sample_rate=44100)
        br, _ = apply_bridge_coupling_to_residual(
            res, frequency_hz=220.0, sample_rate=44100, z_body=z
        )
        self.assertFalse(np.allclose(res, dec))
        self.assertFalse(np.allclose(res, br))

    def test_hybrid_not_g(self) -> None:
        self.assertFalse(is_g_identity_mode("modal_body_hybrid_v4_1_identity_contrast_hybrid_25_75"))
        self.assertTrue(requires_identity_contrast_context("modal_body_hybrid_v4_1_identity_contrast_hybrid_25_75"))

    def test_no_note_name_branch(self) -> None:
        import body_hybrid_v4_1_identity_space as mod

        src = inspect.getsource(mod)
        for token in ("A3", "E3", "D4", "A2"):
            self.assertNotIn(f'"{token}"', src)

    def test_build_no_fem(self) -> None:
        from build_stk_v4_1_identity_contrast_g_diagnostics import build_v4_1_identity_contrast_g_diagnostics

        out = REPO / "audio" / "_test_stk_v41_g"
        if out.exists():
            shutil.rmtree(out)
        with patch(
            "build_stk_v4_1_identity_contrast_g_diagnostics.resolve_modal_data_for_sample",
            side_effect=lambda repo, sample, use_surrogate: (
                synthetic_modal_for_sample(str(sample["sample_id"])),
                "synthetic_fallback",
            ),
        ):
            manifest = build_v4_1_identity_contrast_g_diagnostics(
                repo_root=REPO,
                out_dir=out,
                notes=[("A3", 220.0)],
                modes=[
                    "modal_body_hybrid_v4_1_full",
                    "modal_body_hybrid_v4_1_identity_contrast_g_25_75_full",
                ],
                max_samples=3,
                duration_s=0.2,
                use_surrogate=False,
            )
        self.assertFalse(manifest["fem_launched"])
        report_path = REPO / "audio" / "debug_reports" / "stage51g_stk_v4_1_identity_contrast_g_report.json"
        self.assertTrue(report_path.is_file())
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report.get("stage"), "5.1G")


if __name__ == "__main__":
    unittest.main()
