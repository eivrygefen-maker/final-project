#!/usr/bin/env python3
"""Stage 5.1D STK V4.1 identity strength sweep tests."""
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
    PERCEPTUAL_AXIS_NAMES,
    STRENGTH_PROFILES,
    apply_perceptual_band_shaping,
    compare_audio_to_reference,
    compute_harmonic_gains,
    compute_perceptual_axes,
    is_v4_1_identity_space_mode,
    strength_profile_for_mode,
)
from build_sample_comparison import synthetic_modal_for_sample  # noqa: E402
from diagnostic_synthesis import list_diagnostic_modes  # noqa: E402


class TestStage51DIdentitySweep(unittest.TestCase):
    def test_modes_exist(self) -> None:
        for mode in (
            "modal_body_hybrid_v4_1_identity_light",
            "modal_body_hybrid_v4_1_identity_medium",
            "modal_body_hybrid_v4_1_identity_strong",
            "stk_body_transfer_v4_1_identity_sweep",
        ):
            self.assertIn(mode, list_diagnostic_modes())
            self.assertTrue(is_v4_1_identity_space_mode(mode))

    def test_strength_profiles_ordered(self) -> None:
        light = STRENGTH_PROFILES["light"]
        medium = STRENGTH_PROFILES["medium"]
        strong = STRENGTH_PROFILES["strong"]
        self.assertLess(light.identity_epsilon, medium.identity_epsilon)
        self.assertLess(medium.identity_epsilon, strong.identity_epsilon)
        self.assertLessEqual(strong.fundamental_gain_max, 0.04)

    def test_perceptual_axes_bounded(self) -> None:
        modal = synthetic_modal_for_sample("sample_002")
        from body_hybrid_v4_1_identity_space import build_body_identity_vector

        z = build_body_identity_vector(
            parameters={"top_wood_id": "spruce", "back_wood_id": "rosewood", "geometry.length": 0.52},
            modal_data=modal,
            frequency_hz=220.0,
        )
        axes = compute_perceptual_axes(z)
        for name in PERCEPTUAL_AXIS_NAMES:
            self.assertIn(name, axes)
            self.assertGreaterEqual(axes[name], -1.0)
            self.assertLessEqual(axes[name], 1.0)

    def test_strong_produces_larger_diff_than_light(self) -> None:
        modal = synthetic_modal_for_sample("sample_002")
        from body_hybrid_v4_1_identity_space import build_body_identity_vector

        z = build_body_identity_vector(
            parameters={"top_wood_id": "cedar", "back_wood_id": "maple", "geometry.length": 0.55},
            modal_data=modal,
            frequency_hz=220.0,
        )
        sr = 44100
        t = np.arange(sr // 2) / sr
        audio = 0.3 * np.sin(2 * np.pi * 220 * t) + 0.1 * np.sin(2 * np.pi * 440 * t)
        axes = compute_perceptual_axes(z)
        light = strength_profile_for_mode("modal_body_hybrid_v4_1_identity_light")
        strong = strength_profile_for_mode("modal_body_hybrid_v4_1_identity_strong")
        assert light and strong
        h_light = compute_harmonic_gains(z, frequency_hz=220.0, profile=light)
        h_strong = compute_harmonic_gains(z, frequency_hz=220.0, profile=strong)
        self.assertGreater(max(map(abs, h_strong[1:])), max(map(abs, h_light[1:])))
        shaped_light = apply_perceptual_band_shaping(
            audio, frequency_hz=220.0, sample_rate=sr, axes=axes, profile=light
        )
        shaped_strong = apply_perceptual_band_shaping(
            audio, frequency_hz=220.0, sample_rate=sr, axes=axes, profile=strong
        )
        diff_light = compare_audio_to_reference(shaped_light, audio)
        diff_strong = compare_audio_to_reference(shaped_strong, audio)
        self.assertGreater(
            float(diff_strong["rms_diff_db_vs_reference"]),
            float(diff_light["rms_diff_db_vs_reference"]),
        )

    def test_no_note_name_branch(self) -> None:
        import body_hybrid_v4_1_identity_space as mod

        src = inspect.getsource(mod)
        for token in ("A3", "E3", "D4", "A2"):
            self.assertNotIn(f'"{token}"', src)

    def test_build_no_fem(self) -> None:
        from build_stk_v4_1_identity_sweep_diagnostics import build_v4_1_identity_sweep_diagnostics

        out = REPO / "audio" / "_test_stk_v41_identity_sweep"
        if out.exists():
            shutil.rmtree(out)
        with patch(
            "build_stk_v4_1_identity_sweep_diagnostics.resolve_modal_data_for_sample",
            side_effect=lambda repo, sample, use_surrogate: (
                synthetic_modal_for_sample(str(sample["sample_id"])),
                "synthetic_fallback",
            ),
        ):
            manifest = build_v4_1_identity_sweep_diagnostics(
                repo_root=REPO,
                out_dir=out,
                notes=[("A3", 220.0)],
                modes=[
                    "modal_body_hybrid_v4_1_full",
                    "modal_body_hybrid_v4_1_identity_light",
                    "modal_body_hybrid_v4_1_identity_medium",
                    "modal_body_hybrid_v4_1_identity_strong",
                ],
                max_samples=3,
                duration_s=0.2,
                use_surrogate=False,
            )
        self.assertFalse(manifest["fem_launched"])
        report_path = REPO / "audio" / "debug_reports" / "stage51d_stk_v4_1_identity_sweep_report.json"
        self.assertTrue(report_path.is_file())
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report.get("stage"), "5.1D")
        self.assertTrue(report.get("v4_1_base_preserved"))


if __name__ == "__main__":
    unittest.main()
