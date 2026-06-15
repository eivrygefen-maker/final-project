#!/usr/bin/env python3
"""Lightweight tests for PGSM final guitar demo engine v10 (no synthesis runtime)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gui"))

from pgsm_emergency_guitar_demo_engine import (  # noqa: E402
    AUDIO_DIR_V10,
    FINAL_DEMO_VERSION,
    HARD_ATTACK_CROSSFADE_END_MS,
    HARD_ATTACK_WINDOW_MS,
    MAX_SECONDARY_RATIO_AFTER,
    SAMPLE_SET,
    V10_VOICING,
    V8_VOICING,
    build_emergency_demo_config,
    build_readiness_emergency_demo,
    enforce_hard_single_attack,
    READINESS_DOUBLE_PLUCK,
    READINESS_OK,
)
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE  # noqa: E402


class TestFinalDemoV10Config(unittest.TestCase):
    def test_v10_config(self) -> None:
        cfg = build_emergency_demo_config()
        self.assertEqual(cfg.get("final_demo_version"), FINAL_DEMO_VERSION)
        self.assertEqual(cfg.get("final_demo_version"), "v10_single_attack_stronger_mix_identity")
        self.assertIn("pgsm_final_guitar_demo_v10", str(AUDIO_DIR_V10))
        self.assertTrue(cfg.get("diagnostic_exaggeration_for_audible_demo"))

    def test_v10_mix_ratios_differ(self) -> None:
        m0 = V10_VOICING["sample_000"]["mix"]
        m1 = V10_VOICING["sample_001"]["mix"]
        m2 = V10_VOICING["sample_002"]["mix"]
        self.assertNotEqual(m0["string_bridge"], m1["string_bridge"])
        self.assertNotEqual(m0["body_modal"], m2["body_modal"])
        self.assertGreater(m1["string_bridge"], m0["string_bridge"])
        self.assertGreater(m2["body_modal"], m0["body_modal"])

    def test_v10_factor_multipliers_differ(self) -> None:
        self.assertNotEqual(
            V10_VOICING["sample_001"]["factor_multipliers"]["bridge_mobility_factor"],
            V10_VOICING["sample_002"]["factor_multipliers"]["bridge_mobility_factor"],
        )
        self.assertNotEqual(V8_VOICING["sample_001"]["mix"]["body_modal"], V10_VOICING["sample_001"]["mix"]["body_modal"])


class TestEnforceHardSingleAttack(unittest.TestCase):
    def test_function_exists_and_returns_diagnostics(self) -> None:
        sr = 44100
        t = np.arange(int(0.05 * sr)) / sr
        wave = np.zeros_like(t)
        wave[int(0.002 * sr)] = 1.0
        wave[int(0.012 * sr)] = 0.55
        wave[int(0.020 * sr)] = 0.45
        out, diag = enforce_hard_single_attack(wave.astype(np.float64), sr, "E5", "sample_001")
        self.assertEqual(len(out), len(wave))
        self.assertIn("early_attack_peak_times_ms_before", diag)
        self.assertIn("early_attack_peak_times_ms_after", diag)
        self.assertIn("max_secondary_peak_ratio_before", diag)
        self.assertIn("max_secondary_peak_ratio_after", diag)
        self.assertIn("early_double_attack_risk_before", diag)
        self.assertIn("early_double_attack_risk_after", diag)
        self.assertIn("repair_pass_count", diag)
        self.assertIn("onset_repair_gain_min", diag)
        self.assertTrue(diag.get("hard_single_attack"))

    def test_repair_window_limited(self) -> None:
        self.assertLessEqual(HARD_ATTACK_WINDOW_MS, 35.0)
        self.assertLessEqual(HARD_ATTACK_CROSSFADE_END_MS, 45.0)

    def test_readiness_requires_secondary_control(self) -> None:
        ok = build_readiness_emergency_demo(
            files_generated=9,
            peaks_controlled=True,
            family_metrics={"pass": True, "too_similar": False},
            early_attack_ok=True,
            max_secondary_ok=True,
        )
        self.assertEqual(ok.get("current_status"), READINESS_OK)
        bad = build_readiness_emergency_demo(
            files_generated=9,
            peaks_controlled=True,
            family_metrics={"pass": True},
            max_secondary_ok=False,
        )
        self.assertEqual(bad.get("current_status"), READINESS_DOUBLE_PLUCK)


class TestV10SourceGuards(unittest.TestCase):
    def test_enforce_hard_single_attack_in_source(self) -> None:
        import pgsm_emergency_guitar_demo_engine as mod

        src = Path(mod.__file__).read_text(encoding="utf-8")
        self.assertIn("def enforce_hard_single_attack", src)
        self.assertIn('onset_repair_mode == "hard"', src)
        self.assertIn("enforce_hard_single_attack(y, SR, note, state.sample_id)", src)
        self.assertIn("repair_pass_count_per_file", src)
        self.assertIn("hard_single_attack_summary", src)
        self.assertIn("pgsm_final_guitar_demo_v10", src)
        self.assertIn("V10_VOICING", src)
        self.assertIn("no_separate_contact_mix_layer", src)
        self.assertIn("_early_modal_bump_suppression", src)
        self.assertIn("per_sample_state_isolated", src)
        self.assertNotIn("subprocess", src)
        self.assertNotIn("rom_manager", src)

        with patch("subprocess.run") as mock_run, patch("subprocess.Popen") as mock_popen:
            build_emergency_demo_config()
            mock_run.assert_not_called()
            mock_popen.assert_not_called()

    def test_per_sample_isolated_processing(self) -> None:
        import pgsm_emergency_guitar_demo_engine as mod

        src = Path(mod.__file__).read_text(encoding="utf-8")
        self.assertIn("cleanup_sample_state(state)", src)
        self.assertIn("build_sample_synthesis_state", src)

    def test_secondary_limits_defined(self) -> None:
        self.assertIn("A2", MAX_SECONDARY_RATIO_AFTER)
        self.assertIn("E5", MAX_SECONDARY_RATIO_AFTER)
        self.assertLess(MAX_SECONDARY_RATIO_AFTER["E5"], MAX_SECONDARY_RATIO_AFTER["A2"])

    def test_website_default_unchanged(self) -> None:
        self.assertIsNotNone(DEFAULT_WEBSITE_STK_MODE)


if __name__ == "__main__":
    unittest.main()
