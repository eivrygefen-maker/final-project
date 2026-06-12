#!/usr/bin/env python3
"""Stage 5.0 STK V4 hybrid body-transfer tests."""
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

from body_hybrid_v4 import (  # noqa: E402
    V4_DMAX_DB,
    V4_MOBILITY_CLAMP,
    apply_harmonic_contrast_imprint,
    beta_harmonic,
    compute_body_transfer_envelope,
    compute_contrast_D,
    get_v4_ablation,
    is_v4_family_mode,
    radiation_blend_weight_f0,
)
from body_signature_cache import cache_paths, load_body_signature_cache  # noqa: E402
from build_sample_comparison import synthetic_modal_for_sample  # noqa: E402
from diagnostic_synthesis import list_diagnostic_modes  # noqa: E402


class TestStage50StkV4(unittest.TestCase):
    def test_v4_modes_exist(self) -> None:
        for mode in (
            "modal_body_hybrid_v4",
            "modal_body_hybrid_v4_core",
            "modal_body_hybrid_v4_contrast_imprint_only",
            "modal_body_hybrid_v4_contrast_body_layer_only",
            "modal_body_hybrid_v4_mobility_light_only",
            "modal_body_hybrid_v4_full",
            "stk_body_transfer_v4",
        ):
            self.assertIn(mode, list_diagnostic_modes())
            self.assertTrue(is_v4_family_mode(mode))

    def test_crossfade_continuous(self) -> None:
        freqs = [80.0, 120.0, 160.0, 200.0, 260.0, 320.0, 440.0, 659.0]
        vals = [radiation_blend_weight_f0(f) for f in freqs]
        for a, b in zip(vals, vals[1:]):
            self.assertLessEqual(a - 1e-9, b)
        self.assertLess(radiation_blend_weight_f0(110.0), 0.15)
        self.assertGreater(radiation_blend_weight_f0(440.0), 0.85)

    def test_no_note_name_branch(self) -> None:
        import body_hybrid_v4 as mod

        src = inspect.getsource(mod)
        for token in ("A2", "A4", "E5"):
            self.assertNotIn(f'"{token}"', src)

    def test_contrast_bounded_and_differs(self) -> None:
        modes_a = synthetic_modal_for_sample("sample_001")["predicted_modes"]
        modes_b = synthetic_modal_for_sample("sample_009")["predicted_modes"]
        freqs = np.linspace(60, 1000, 128)
        Ga, _, _ = compute_body_transfer_envelope(modes_a, freqs)
        Gb, _, _ = compute_body_transfer_envelope(modes_b, freqs)
        logGa, logGb = np.log(Ga + 1e-9), np.log(Gb + 1e-9)
        logG_ref = (logGa + logGb) / 2.0
        Da = compute_contrast_D(logGa, logG_ref)
        Db = compute_contrast_D(logGb, logG_ref)
        dmax = V4_DMAX_DB / (20.0 / np.log(10.0))
        self.assertLessEqual(float(np.max(np.abs(Da))), dmax + 1e-6)
        self.assertGreater(float(np.max(np.abs(Da - Db))), 1e-6)

    def test_fundamental_preserved_more_than_higher_harmonics(self) -> None:
        self.assertLess(beta_harmonic(1), beta_harmonic(3))
        self.assertLess(beta_harmonic(1), beta_harmonic(8))

    def test_harmonic_imprint_affects_h2_more_than_f0(self) -> None:
        sr = 44100
        f0 = 110.0
        t = np.arange(sr // 4) / sr
        sig = np.sin(2 * np.pi * f0 * t) + 0.5 * np.sin(2 * np.pi * 2 * f0 * t)
        freqs = np.linspace(60, 1000, 128)
        D = np.linspace(-0.1, 0.1, 128)
        out = apply_harmonic_contrast_imprint(sig, sr, f0, D, freqs)
        spec_in = np.abs(np.fft.rfft(sig))
        spec_out = np.abs(np.fft.rfft(out))
        f_axis = np.fft.rfftfreq(len(sig), d=1.0 / sr)
        i_f0 = int(np.argmin(np.abs(f_axis - f0)))
        i_h2 = int(np.argmin(np.abs(f_axis - 2 * f0)))
        rel_f0 = abs(spec_out[i_f0] - spec_in[i_f0]) / max(spec_in[i_f0], 1e-9)
        rel_h2 = abs(spec_out[i_h2] - spec_in[i_h2]) / max(spec_in[i_h2], 1e-9)
        self.assertGreater(rel_h2, rel_f0)

    def test_mobility_tight_clamp(self) -> None:
        lo, hi = V4_MOBILITY_CLAMP
        self.assertGreaterEqual(lo, 0.95)
        self.assertLessEqual(hi, 1.05)

    def test_cache_deterministic(self) -> None:
        from body_hybrid_v4 import build_sample_signature_cache

        modal = synthetic_modal_for_sample("sample_002")
        params = {"top_wood_id": "spruce", "geometry.length": 0.46}
        out = REPO / "ROM" / "classic" / "body_signature_cache"
        build_sample_signature_cache(REPO, "sample_002", modal, parameters=params, logG_ref=None)
        a = load_body_signature_cache(REPO, "sample_002")
        build_sample_signature_cache(REPO, "sample_002", modal, parameters=params, logG_ref=None)
        b = load_body_signature_cache(REPO, "sample_002")
        np.testing.assert_array_almost_equal(a["G_sample"], b["G_sample"])
        jpath, _ = cache_paths(REPO, "sample_002")
        self.assertTrue(jpath.is_file())

    def test_v4_core_blend_weights(self) -> None:
        w_low = radiation_blend_weight_f0(110.0)
        w_high = radiation_blend_weight_f0(440.0)
        self.assertLess(w_low, 0.2)
        self.assertGreater(w_high, 0.8)

    def test_build_no_fem(self) -> None:
        from build_stk_v4_diagnostics import build_v4_diagnostics

        out = REPO / "audio" / "_test_stk_v4_pack"
        if out.exists():
            import shutil

            shutil.rmtree(out)
        with patch(
            "build_stk_v4_diagnostics.resolve_modal_data_for_sample",
            side_effect=lambda repo, sample, use_surrogate: (
                synthetic_modal_for_sample(str(sample["sample_id"])),
                "synthetic_fallback",
            ),
        ):
            manifest = build_v4_diagnostics(
                repo_root=REPO,
                out_dir=out,
                notes=[("A4", 440.0)],
                modes=["baseline_current", "modal_body_hybrid_v4_core", "modal_body_hybrid_v4_full"],
                max_samples=3,
                duration_s=0.2,
                use_surrogate=False,
            )
        self.assertFalse(manifest["fem_launched"])
        report = REPO / "audio" / "debug_reports" / "stage50_stk_v4_report.json"
        self.assertTrue(report.is_file())
        doc = json.loads(report.read_text(encoding="utf-8"))
        self.assertTrue(doc.get("starts_from_baseline_and_v1_not_v3_v2"))


if __name__ == "__main__":
    unittest.main()
