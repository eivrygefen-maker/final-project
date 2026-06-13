#!/usr/bin/env python3
"""Stage 5.1C STK V4.1 identity-space tests."""
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
    FUNDAMENTAL_GAIN_MAX,
    HARMONIC_GAIN_MAX,
    IDENTITY_EPSILON,
    RESIDUAL_GAIN_MAX,
    apply_harmonic_identity_shaping,
    apply_identity_residual,
    apply_rms_guard,
    audio_distance,
    build_body_identity_vector,
    compute_harmonic_gains,
    distance_consistency_report,
    is_v4_1_identity_space_mode,
    physical_distance,
    spearman_correlation,
    synthesize_v4_1_identity_space_note,
)
from build_sample_comparison import synthetic_modal_for_sample  # noqa: E402
from diagnostic_synthesis import list_diagnostic_modes  # noqa: E402


class TestStage51CIdentitySpace(unittest.TestCase):
    def test_modes_exist(self) -> None:
        for mode in (
            "modal_body_hybrid_v4_1_identity_space",
            "stk_body_transfer_v4_1_identity_space",
        ):
            self.assertIn(mode, list_diagnostic_modes())
            self.assertTrue(is_v4_1_identity_space_mode(mode))

    def test_no_note_name_branch(self) -> None:
        import body_hybrid_v4_1_identity_space as mod

        src = inspect.getsource(mod)
        for token in ("A3", "E3", "D4", "A2"):
            self.assertNotIn(f'"{token}"', src)

    def test_identity_modifiers_bounded(self) -> None:
        modal = synthetic_modal_for_sample("sample_002")
        z = build_body_identity_vector(
            parameters={"top_wood_id": "spruce", "back_wood_id": "rosewood", "geometry.length": 0.52},
            modal_data=modal,
            frequency_hz=220.0,
        )
        gains = compute_harmonic_gains(z, frequency_hz=220.0)
        self.assertEqual(len(gains), 8)
        self.assertLessEqual(abs(gains[0]), FUNDAMENTAL_GAIN_MAX + 1e-9)
        for g in gains[1:]:
            self.assertLessEqual(abs(g), HARMONIC_GAIN_MAX + 1e-9)
        for v in z.get("vector") or []:
            self.assertGreaterEqual(v, -1.5)
            self.assertLessEqual(v, 1.5)

    def test_fundamental_mostly_preserved(self) -> None:
        sr = 44100
        t = np.arange(sr // 4) / sr
        audio = 0.5 * np.sin(2 * np.pi * 220.0 * t)
        gains = [0.02] + [0.10] * 7
        out = apply_harmonic_identity_shaping(audio, frequency_hz=220.0, sample_rate=sr, harmonic_gains=gains)
        f0_bin = int(np.argmin(np.abs(np.fft.rfftfreq(len(out), 1 / sr) - 220.0)))
        spec_in = np.abs(np.fft.rfft(audio))
        spec_out = np.abs(np.fft.rfft(out))
        ratio = spec_out[f0_bin] / max(spec_in[f0_bin], 1e-12)
        self.assertGreater(ratio, 0.85)
        self.assertLess(ratio, 1.20)

    def test_physical_audio_distance_metrics(self) -> None:
        z1 = {"vector": [0.0, 0.1, 0.2]}
        z2 = {"vector": [0.0, 0.5, 0.9]}
        self.assertGreater(physical_distance(z1, z2), 0.0)
        self.assertGreater(audio_distance([0, 1, 2], [1, 2, 3]), 0.0)
        rho = spearman_correlation([1, 2, 3, 4, 5], [1, 2, 4, 4.5, 6])
        self.assertIsNotNone(rho)
        self.assertGreater(float(rho), 0.5)

    def test_rms_guard(self) -> None:
        ref = np.ones(1000) * 0.1
        loud = ref * 4.0
        guarded, info = apply_rms_guard(loud, ref, max_db=1.5)
        self.assertLess(float(np.max(np.abs(guarded))), 1.0)
        self.assertIn("rms_guard_gain", info)

    def test_residual_bounded(self) -> None:
        base = np.random.randn(512) * 0.01
        shaped = base + 0.5
        out = apply_identity_residual(base, shaped, epsilon=IDENTITY_EPSILON)
        self.assertLessEqual(float(np.max(np.abs(out - base))), RESIDUAL_GAIN_MAX + 0.01)

    def test_build_no_fem(self) -> None:
        from build_stk_v4_1_identity_space_diagnostics import build_v4_1_identity_space_diagnostics

        out = REPO / "audio" / "_test_stk_v41_identity"
        if out.exists():
            shutil.rmtree(out)
        with patch(
            "build_stk_v4_1_identity_space_diagnostics.resolve_modal_data_for_sample",
            side_effect=lambda repo, sample, use_surrogate: (
                synthetic_modal_for_sample(str(sample["sample_id"])),
                "synthetic_fallback",
            ),
        ):
            manifest = build_v4_1_identity_space_diagnostics(
                repo_root=REPO,
                out_dir=out,
                notes=[("A3", 220.0)],
                modes=[
                    "modal_body_hybrid_v4_1_full",
                    "modal_body_hybrid_v4_1_identity_space",
                ],
                max_samples=3,
                duration_s=0.2,
                use_surrogate=False,
            )
        self.assertFalse(manifest["fem_launched"])
        report_path = REPO / "audio" / "debug_reports" / "stage51c_stk_v4_1_identity_space_report.json"
        self.assertTrue(report_path.is_file())
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report.get("stage"), "5.1C")
        self.assertTrue(report.get("v4_1_base_preserved"))

    def test_distance_consistency_report(self) -> None:
        samples = [
            {"sample_id": "a", "z_body": {"vector": [0, 0]}, "timbre": [0, 0]},
            {"sample_id": "b", "z_body": {"vector": [1, 0]}, "timbre": [0.5, 0.2]},
            {"sample_id": "c", "z_body": {"vector": [2, 1]}, "timbre": [1.0, 0.5]},
        ]
        rep = distance_consistency_report(samples)
        self.assertEqual(rep["pair_count"], 3)
        self.assertIsNotNone(rep.get("spearman_rho"))


if __name__ == "__main__":
    unittest.main()
