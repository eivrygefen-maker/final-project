#!/usr/bin/env python3
"""Stage 5.1E STK V4.1 sample-relative identity contrast tests."""
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
    DZ_BODY_CLIP,
    build_batch_contrast_context,
    build_body_identity_vector,
    compute_dz_body,
    compute_perceptual_axes,
    compute_robust_identity_reference,
    is_contrast_identity_mode,
    is_v4_1_identity_space_mode,
    strength_profile_for_mode,
)
from build_sample_comparison import synthetic_modal_for_sample  # noqa: E402
from diagnostic_synthesis import list_diagnostic_modes  # noqa: E402


class TestStage51EIdentityContrast(unittest.TestCase):
    def test_modes_exist(self) -> None:
        for mode in (
            "modal_body_hybrid_v4_1_identity_contrast",
            "stk_body_transfer_v4_1_identity_contrast",
            "modal_body_hybrid_v4_1_identity_contrast_medium",
            "modal_body_hybrid_v4_1_identity_contrast_strong",
        ):
            self.assertIn(mode, list_diagnostic_modes())
            self.assertTrue(is_v4_1_identity_space_mode(mode))
            self.assertTrue(is_contrast_identity_mode(mode))

    def test_dz_body_robust_reference(self) -> None:
        z_bodies = []
        for i, sid in enumerate(("sample_000", "sample_001", "sample_002")):
            modal = synthetic_modal_for_sample(sid)
            z = build_body_identity_vector(
                parameters={"top_wood_id": "spruce", "geometry.length": 0.50 + i * 0.02},
                modal_data=modal,
                frequency_hz=220.0,
            )
            z_bodies.append(z)
        ref = compute_robust_identity_reference(z_bodies)
        self.assertIn("median_features", ref)
        self.assertIn("iqr_features", ref)
        dz = compute_dz_body(z_bodies[0], ref)
        for v in (dz.get("features") or {}).values():
            self.assertGreaterEqual(v, -DZ_BODY_CLIP)
            self.assertLessEqual(v, DZ_BODY_CLIP)

    def test_contrast_axes_differ_across_samples(self) -> None:
        z_map = {}
        for sid in ("sample_000", "sample_001", "sample_002"):
            modal = synthetic_modal_for_sample(sid)
            z_map[sid] = build_body_identity_vector(
                parameters={"top_wood_id": "cedar" if sid == "sample_001" else "spruce"},
                modal_data=modal,
                frequency_hz=220.0,
            )
        ctx = build_batch_contrast_context(z_map)
        axes_a = compute_perceptual_axes(ctx["sample_000"]["dz_body"], contrast=True)
        axes_b = compute_perceptual_axes(ctx["sample_001"]["dz_body"], contrast=True)
        self.assertNotEqual(axes_a, axes_b)

    def test_contrast_profiles(self) -> None:
        med = strength_profile_for_mode("modal_body_hybrid_v4_1_identity_contrast_medium")
        strong = strength_profile_for_mode("modal_body_hybrid_v4_1_identity_contrast_strong")
        assert med and strong
        self.assertEqual(med.identity_epsilon, 0.45)
        self.assertEqual(strong.identity_epsilon, 0.65)
        self.assertLessEqual(strong.fundamental_gain_max, 0.04)

    def test_no_note_name_branch(self) -> None:
        import body_hybrid_v4_1_identity_space as mod

        src = inspect.getsource(mod)
        for token in ("A3", "E3", "D4", "A2"):
            self.assertNotIn(f'"{token}"', src)

    def test_build_no_fem(self) -> None:
        from build_stk_v4_1_identity_contrast_diagnostics import build_v4_1_identity_contrast_diagnostics

        out = REPO / "audio" / "_test_stk_v41_identity_contrast"
        if out.exists():
            shutil.rmtree(out)
        with patch(
            "build_stk_v4_1_identity_contrast_diagnostics.resolve_modal_data_for_sample",
            side_effect=lambda repo, sample, use_surrogate: (
                synthetic_modal_for_sample(str(sample["sample_id"])),
                "synthetic_fallback",
            ),
        ):
            manifest = build_v4_1_identity_contrast_diagnostics(
                repo_root=REPO,
                out_dir=out,
                notes=[("A3", 220.0)],
                modes=[
                    "modal_body_hybrid_v4_1_full",
                    "modal_body_hybrid_v4_1_identity_strong",
                    "modal_body_hybrid_v4_1_identity_contrast_medium",
                    "modal_body_hybrid_v4_1_identity_contrast_strong",
                ],
                max_samples=3,
                duration_s=0.2,
                use_surrogate=False,
            )
        self.assertFalse(manifest["fem_launched"])
        report_path = REPO / "audio" / "debug_reports" / "stage51e_stk_v4_1_identity_contrast_report.json"
        self.assertTrue(report_path.is_file())
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report.get("stage"), "5.1E")
        self.assertTrue(report.get("v4_1_base_preserved"))


if __name__ == "__main__":
    unittest.main()
