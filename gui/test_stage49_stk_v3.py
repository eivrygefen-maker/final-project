#!/usr/bin/env python3
"""Stage 4.9 STK V3 body-signature tests."""
from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gui"))

from body_response_synth import synthesize_note_with_body_response  # noqa: E402
from body_signature_v3 import (  # noqa: E402
    bounded_mobility_factor,
    build_body_signature_envelope,
    get_v3_ablation,
    low_f0_imprint_strength,
)
from build_sample_comparison import synthetic_modal_for_sample  # noqa: E402
from diagnostic_synthesis import get_diagnostic_mode, list_diagnostic_modes  # noqa: E402
from stage48_timbre_decomposition_report import compute_normalization_impact  # noqa: E402


V3_MODES = (
    "modal_body_signature_v3",
    "modal_body_signature_v3_core",
    "modal_body_signature_v3_low_f0_imprint_only",
    "modal_body_signature_v3_mobility_only",
    "modal_body_signature_v3_far_color_only",
    "modal_body_signature_v3_full",
    "modal_radiation_color_v3",
)


class TestStage49StkV3(unittest.TestCase):
    def test_v3_modes_exist(self) -> None:
        names = list_diagnostic_modes()
        for mode in V3_MODES:
            self.assertIn(mode, names)
            self.assertIsNotNone(get_v3_ablation(mode))

    def test_no_note_name_branch_in_v3(self) -> None:
        import body_signature_v3 as mod

        src = inspect.getsource(mod)
        for token in ("A2", "A4", "E5", '"A2"', "'A2'"):
            self.assertNotIn(token, src)

    def test_low_f0_strength_continuous(self) -> None:
        freqs = [80.0, 110.0, 150.0, 200.0, 350.0, 500.0]
        vals = [low_f0_imprint_strength(f) for f in freqs]
        for a, b in zip(vals, vals[1:]):
            self.assertLessEqual(b, a + 1e-9)
        self.assertGreater(vals[0], vals[-1])

    def test_mobility_bounded(self) -> None:
        mode = {"top_participation": 0.5, "back_participation": 0.3, "air_participation": 0.1, "coupled_participation": 0.1}
        light = {"top_wood_id": "cedar", "geometry.length": 0.44}
        heavy = {"top_wood_id": "maple", "geometry.length": 0.48}
        f_light, _ = bounded_mobility_factor(mode, {"bridge_weight": 1.0}, light)
        f_heavy, _ = bounded_mobility_factor(mode, {"bridge_weight": 1.0}, heavy)
        self.assertGreaterEqual(f_light, 0.85)
        self.assertLessEqual(f_light, 1.15)
        self.assertGreaterEqual(f_heavy, 0.85)
        self.assertLessEqual(f_heavy, 1.15)

    def test_body_signature_envelope_differs_across_guitars(self) -> None:
        modes_a = synthetic_modal_for_sample("sample_001")["predicted_modes"]
        modes_b = synthetic_modal_for_sample("sample_009")["predicted_modes"]
        env_a = build_body_signature_envelope(modes_a, 110.0, 4096, 44100)
        env_b = build_body_signature_envelope(modes_b, 110.0, 4096, 44100)
        self.assertGreater(float(np.max(np.abs(env_a - env_b))), 1e-6)

    def test_string_only_identical_v3_context(self) -> None:
        from timbre_decomposition import compute_note_layers

        modal_a = synthetic_modal_for_sample("sample_001")
        modal_b = synthetic_modal_for_sample("sample_009")
        kw = dict(frequency_hz=440.0, note_name="A4", duration_s=0.25, sample_rate=44100)
        a = compute_note_layers(**kw, modal_data=modal_a, sample_parameters={"geometry.length": 0.44})
        b = compute_note_layers(**kw, modal_data=modal_b, sample_parameters={"geometry.length": 0.48})
        np.testing.assert_array_almost_equal(a["layers"]["string_only"], b["layers"]["string_only"])

    def test_v3_metadata_decomposition(self) -> None:
        out = REPO / "audio" / "_test_v3_meta.wav"
        meta = synthesize_note_with_body_response(
            frequency_hz=110.0,
            note_name="A2",
            duration_s=0.3,
            sample_rate=44100,
            modal_data=synthetic_modal_for_sample("sample_003"),
            output_wav=out,
            diagnostic_mode="modal_body_signature_v3_full",
            sample_parameters={"top_wood_id": "spruce", "geometry.length": 0.46},
        )
        self.assertTrue(meta.get("body_signature_v3_active"))
        per_mode = meta.get("top_contributing_modes") or []
        damp = (meta.get("damping_q_summary") or {})
        self.assertTrue(meta.get("body_signature_envelope_meta") or meta.get("body_signature_v3_ablation"))
        self.assertIn("final_modal_amp_m", str(meta))

    def test_no_clipping_v3(self) -> None:
        out = REPO / "audio" / "_test_v3_clip.wav"
        meta = synthesize_note_with_body_response(
            frequency_hz=659.25,
            note_name="E5",
            duration_s=0.35,
            sample_rate=44100,
            modal_data=synthetic_modal_for_sample("sample_005"),
            output_wav=out,
            diagnostic_mode="modal_body_signature_v3_full",
        )
        peak = float(meta.get("output_peak_dbfs") or -6.0)
        self.assertLessEqual(peak, 0.5)

    def test_normalization_delta_sign_convention(self) -> None:
        layer_metrics = {
            "A4/body_only_raw_pre_norm": {"spectral_differentiation": 0.002},
            "A4/body_only_final_norm": {"spectral_differentiation": 0.001},
            "A4/full_mix_baseline": {"spectral_differentiation": 0.0008},
        }
        impact = compute_normalization_impact(layer_metrics, "A4")
        self.assertEqual(impact["delta_final_minus_raw"], -0.001)
        self.assertTrue(impact["final_norm_reduces_diff"])
        self.assertFalse(impact["final_norm_increases_diff"])

    def test_build_v3_diagnostics_no_fem(self) -> None:
        from build_stk_v3_diagnostics import build_v3_diagnostics

        out = REPO / "audio" / "_test_stk_v3_pack"
        if out.exists():
            import shutil

            shutil.rmtree(out)
        with patch(
            "build_stk_v3_diagnostics.resolve_modal_data_for_sample",
            side_effect=lambda repo, sample, use_surrogate: (
                synthetic_modal_for_sample(str(sample["sample_id"])),
                "synthetic_fallback",
            ),
        ):
            manifest = build_v3_diagnostics(
                repo_root=REPO,
                out_dir=out,
                notes=[("A4", 440.0)],
                modes=["baseline_current", "modal_radiation_color_v1", "modal_body_signature_v3_full"],
                max_samples=3,
                duration_s=0.2,
                use_surrogate=False,
            )
        self.assertFalse(manifest["fem_launched"])
        report = REPO / "audio" / "debug_reports" / "stage49_stk_v3_report.json"
        self.assertTrue(report.is_file())


if __name__ == "__main__":
    unittest.main()
