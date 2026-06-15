#!/usr/bin/env python3
"""Lightweight tests for PGSM final guitar demo engine v9 (no synthesis runtime)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gui"))

from pgsm_emergency_guitar_demo_engine import (  # noqa: E402
    AUDIO_DIR_V9,
    FINAL_DEMO_VERSION,
    ONSET_REPAIR_CROSSFADE_END_MS,
    ONSET_REPAIR_WINDOW_MS,
    SAMPLE_SET,
    V8_VOICING,
    build_emergency_demo_config,
    build_readiness_emergency_demo,
    enforce_single_attack_peak,
    READINESS_DOUBLE_PLUCK,
    READINESS_OK,
)
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE  # noqa: E402


class TestFinalDemoV9Config(unittest.TestCase):
    def test_v9_config(self) -> None:
        cfg = build_emergency_demo_config()
        self.assertEqual(cfg.get("final_demo_version"), FINAL_DEMO_VERSION)
        self.assertEqual(cfg.get("final_demo_version"), "v9_single_attack_enforced_guitar_demo")
        self.assertIn("pgsm_final_guitar_demo_v9", str(AUDIO_DIR_V9))

    def test_v8_voicing_preserved(self) -> None:
        self.assertIn("factor_multipliers", V8_VOICING["sample_001"])
        self.assertNotEqual(
            V8_VOICING["sample_001"]["factor_multipliers"]["bridge_mobility_factor"],
            V8_VOICING["sample_002"]["factor_multipliers"]["bridge_mobility_factor"],
        )


class TestEnforceSingleAttackPeak(unittest.TestCase):
    def test_function_exists_and_returns_diagnostics(self) -> None:
        sr = 44100
        t = np.arange(int(0.05 * sr)) / sr
        wave = np.zeros_like(t)
        wave[int(0.002 * sr)] = 1.0
        wave[int(0.012 * sr)] = 0.6
        wave[int(0.020 * sr)] = 0.5
        out, diag = enforce_single_attack_peak(wave.astype(np.float64), sr, "E5", "sample_001")
        self.assertEqual(len(out), len(wave))
        self.assertTrue(diag.get("onset_repair_applied"))
        self.assertIn("early_attack_peak_times_ms_before", diag)
        self.assertIn("early_attack_peak_times_ms_after", diag)
        self.assertIn("early_attack_peak_ratios_before", diag)
        self.assertIn("early_attack_peak_ratios_after", diag)
        self.assertIn("early_double_attack_risk_before", diag)
        self.assertIn("early_double_attack_risk_after", diag)
        self.assertIn("onset_repair_gain_min", diag)

    def test_repair_window_limited(self) -> None:
        self.assertLessEqual(ONSET_REPAIR_WINDOW_MS, 35.0)
        self.assertLessEqual(ONSET_REPAIR_CROSSFADE_END_MS, 45.0)

    def test_readiness_requires_after_risk_clear(self) -> None:
        ok = build_readiness_emergency_demo(
            files_generated=9,
            peaks_controlled=True,
            family_metrics={"pass": True, "too_similar": False},
            early_attack_ok=True,
        )
        self.assertEqual(ok.get("current_status"), READINESS_OK)
        bad = build_readiness_emergency_demo(
            files_generated=9,
            peaks_controlled=True,
            family_metrics={"pass": True},
            early_attack_ok=False,
        )
        self.assertEqual(bad.get("current_status"), READINESS_DOUBLE_PLUCK)


class TestV9SourceGuards(unittest.TestCase):
    def test_enforce_single_attack_peak_in_source(self) -> None:
        import pgsm_emergency_guitar_demo_engine as mod

        src = Path(mod.__file__).read_text(encoding="utf-8")
        self.assertIn("def enforce_single_attack_peak", src)
        self.assertIn("enforce_single_attack_peak(y, SR, note, state.sample_id)", src)
        self.assertIn("early_double_attack_risk_after", src)
        self.assertIn("onset_repair_per_file", src)
        self.assertIn("pgsm_final_guitar_demo_v9", src)
        self.assertIn("no_separate_contact_mix_layer", src)
        self.assertNotIn("subprocess", src)
        self.assertNotIn("rom_manager", src)

        with patch("subprocess.run") as mock_run, patch("subprocess.Popen") as mock_popen:
            build_emergency_demo_config()
            mock_run.assert_not_called()
            mock_popen.assert_not_called()

    def test_website_default_unchanged(self) -> None:
        self.assertIsNotNone(DEFAULT_WEBSITE_STK_MODE)


if __name__ == "__main__":
    unittest.main()
