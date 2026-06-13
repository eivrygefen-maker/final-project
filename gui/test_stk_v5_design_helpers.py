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
from diagnostic_synthesis import list_diagnostic_modes  # noqa: E402
from stk_v5_design_helpers import (  # noqa: E402
    STK_V5_ALPHA_BODY_DOMINANT,
    STK_V5_ALPHA_GUI_LABEL,
    V5_ALPHA_VARIANTS,
    compute_realism_metrics,
    decompose_baseline_layers,
    list_v5_alpha_mode_names,
    rms_matched_body_dominant_mix,
    singleton_dz_body_quantification,
    synthesize_v5_alpha_body_dominant,
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

    def test_v5_alpha_modes_registered(self) -> None:
        modes = list_diagnostic_modes()
        for name in list_v5_alpha_mode_names():
            self.assertIn(name, modes)
            self.assertIn(name, V5_ALPHA_VARIANTS)
        self.assertEqual(STK_V5_ALPHA_GUI_LABEL, "STK V5 alpha — body dominant")

    def test_rms_matched_mix_weights(self) -> None:
        import numpy as np

        body = np.ones(1000) * 0.5
        string = np.ones(1000) * 0.1
        mixed, mix = rms_matched_body_dominant_mix(body, string, body_weight=0.9, string_weight=0.1)
        self.assertAlmostEqual(mix["string_dominance_in_mix"], 0.1, places=2)
        self.assertGreater(mix["body_to_string_in_mix_ratio"], 5.0)
        self.assertEqual(len(mixed), 1000)

    def test_v5_alpha_reduces_string_dominance_vs_final_v1(self) -> None:
        from stk_v5_design_helpers import synthesize_mode_to_wav
        import tempfile

        modal_data = {"predicted_modes": synthetic_classic_body_modes(), "analysis": "test"}
        params = {"geometry.length": 0.52, "materials.top.wood_id": "spruce"}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            final_wav = root / "final.wav"
            alpha_wav = root / "alpha.wav"
            final_meta = synthesize_mode_to_wav(
                mode=STK_BODY_TRANSFER_FINAL_V1,
                frequency_hz=110.0,
                note_name="A2",
                duration_s=0.2,
                sample_rate=44100,
                modal_data=modal_data,
                output_wav=final_wav,
                sample_parameters=params,
                experiment="current_final_v1",
            )
            alpha_meta = synthesize_mode_to_wav(
                mode=STK_V5_ALPHA_BODY_DOMINANT,
                frequency_hz=110.0,
                note_name="A2",
                duration_s=0.2,
                sample_rate=44100,
                modal_data=modal_data,
                output_wav=alpha_wav,
                sample_parameters=params,
                experiment="v5_alpha_s10_b90",
            )
        f_met = final_meta["realism_metrics"]
        a_met = alpha_meta["realism_metrics"]
        self.assertLess(
            a_met["string_dominance_ratio"],
            f_met["string_dominance_ratio"],
        )
        self.assertGreater(
            a_met["body_to_string_energy_ratio"],
            f_met["body_to_string_energy_ratio"],
        )

    def test_v5_alpha_no_clipping(self) -> None:
        modal_data = {"predicted_modes": synthetic_classic_body_modes(), "analysis": "test"}
        for variant in ("v5_alpha_s10_b90", "v5_alpha_s20_b80", "v5_alpha_s35_b65"):
            audio, meta = synthesize_v5_alpha_body_dominant(
                frequency_hz=220.0,
                duration_s=0.2,
                sample_rate=44100,
                modal_data=modal_data,
                variant=variant,
            )
            self.assertTrue(meta["clipping_avoided"], variant)
            self.assertLess(meta["peak_dbfs"], 0.0, variant)
            self.assertGreater(len(audio), 0)


if __name__ == "__main__":
    unittest.main()
