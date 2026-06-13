#!/usr/bin/env python3
"""Lightweight STK V5 design helper tests (no FEM/ROM batch)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gui"))

from body_hybrid_v4_1_identity_space import STK_BODY_TRANSFER_FINAL_V1  # noqa: E402
from body_response_synth import synthetic_classic_body_modes  # noqa: E402
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE  # noqa: E402
from stk_v5_design_helpers import (  # noqa: E402
    compute_realism_metrics,
    decompose_baseline_layers,
    singleton_dz_body_quantification,
    synthesize_v5_skeleton,
)


class TestStkV5DesignHelpers(unittest.TestCase):
    def test_website_default_unchanged(self) -> None:
        self.assertEqual(DEFAULT_WEBSITE_STK_MODE, STK_BODY_TRANSFER_FINAL_V1)

    def test_realism_metrics_keys(self) -> None:
        import numpy as np

        audio = 0.05 * np.sin(2 * np.pi * 110 * np.arange(4410) / 44100)
        m = compute_realism_metrics(audio, sample_rate=44100, frequency_hz=110.0)
        for key in (
            "metallicity_index",
            "body_audibility_index",
            "string_dominance_ratio",
            "guitar_realism_sanity_score",
        ):
            self.assertIn(key, m)

    def test_singleton_contrast_inactive(self) -> None:
        modal_data = {"predicted_modes": synthetic_classic_body_modes(), "analysis": "test"}
        params = {
            "geometry.length": 0.52,
            "geometry.width": 0.32,
            "materials.top.wood_id": "spruce",
            "materials.back.wood_id": "mahogany",
        }
        q = singleton_dz_body_quantification(
            sample_parameters=params,
            modal_data=modal_data,
            frequency_hz=220.0,
            sample_id="website",
        )
        self.assertTrue(q["contrast_layer_inactive_on_website"])
        self.assertLess(q["dz_body_abs_max"], 0.01)

    def test_baseline_decomposition(self) -> None:
        modal_data = {"predicted_modes": synthetic_classic_body_modes(), "analysis": "test"}
        layers = decompose_baseline_layers(
            frequency_hz=110.0,
            duration_s=0.2,
            sample_rate=44100,
            modal_data=modal_data,
        )
        self.assertGreater(layers["string_rms"], 0.0)
        self.assertIn("body_raw", layers)

    def test_v5_skeleton_runs(self) -> None:
        modal_data = {"predicted_modes": synthetic_classic_body_modes(), "analysis": "test"}
        audio, meta = synthesize_v5_skeleton(
            frequency_hz=110.0,
            duration_s=0.2,
            sample_rate=44100,
            modal_data=modal_data,
            sample_parameters={"geometry.length": 0.52},
        )
        self.assertGreater(len(audio), 0)
        self.assertEqual(meta["v5_skeleton_version"], "stk_v5_skeleton_v0")

    def test_no_fem_rom_imports_in_helpers(self) -> None:
        import stk_v5_design_helpers as h

        src = Path(h.__file__).read_text(encoding="utf-8")
        self.assertNotIn("run_fom_acoustics", src)
        self.assertNotIn("run_rom_batch", src)


if __name__ == "__main__":
    unittest.main()
