#!/usr/bin/env python3
"""Lightweight tests for PGSM final guitar demo engine v11 (no synthesis runtime)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gui"))

from pgsm_emergency_guitar_demo_engine import (  # noqa: E402
    AUDIO_DIR_V11,
    BODY_BLOOM_ATTACK_MS,
    FINAL_DEMO_VERSION,
    HIGH_NOTE_LOW_MID_BOOST,
    PHYSICAL_FACTOR_KEYS,
    SAMPLE_SET,
    V11_VOICING,
    build_body_drive_from_bridge,
    build_emergency_demo_config,
    build_readiness_emergency_demo,
    build_sample_synthesis_state,
    READINESS_BODY_SECOND_PLUCK,
    READINESS_OK,
)
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE  # noqa: E402


class TestFinalDemoV11Config(unittest.TestCase):
    def test_v11_config(self) -> None:
        cfg = build_emergency_demo_config()
        self.assertEqual(cfg.get("final_demo_version"), FINAL_DEMO_VERSION)
        self.assertEqual(cfg.get("final_demo_version"), "v11_body_bloom_physical_guitar_demo")
        self.assertIn("pgsm_final_guitar_demo_v11", str(AUDIO_DIR_V11))

    def test_soundhole_factor_exists(self) -> None:
        self.assertIn("soundhole_radiation_factor", PHYSICAL_FACTOR_KEYS)
        self.assertIn("soundhole_radiation_factor", V11_VOICING["sample_002"]["factor_multipliers"])

    def test_v11_mix_ratios_differ(self) -> None:
        m0 = V11_VOICING["sample_000"]["mix"]
        m1 = V11_VOICING["sample_001"]["mix"]
        m2 = V11_VOICING["sample_002"]["mix"]
        self.assertGreater(m1["string_bridge"], m0["string_bridge"])
        self.assertGreater(m2["body_modal"], m0["body_modal"])
        self.assertLess(m1["air_share"], m2["air_share"])


class TestBodyDriveFromBridge(unittest.TestCase):
    def test_helper_exists_and_starts_at_zero_delay(self) -> None:
        import pgsm_emergency_guitar_demo_engine as mod
        from pgsm_step5l_limited_multiguitar_differentiation import extract_per_sample_physical_parameters
        from stk_v6_2_audit_features import load_audit_report

        audit = load_audit_report()
        ref = extract_per_sample_physical_parameters("sample_000", audit)
        phys = extract_per_sample_physical_parameters("sample_001", audit)
        state = build_sample_synthesis_state(
            "sample_001",
            physical=phys,
            reference_physical=ref,
            readonly_modes=mod._load_readonly_reference_modes(),
            voicing=V11_VOICING,
        )
        sr = 44100
        n = int(0.05 * sr)
        bridge = np.zeros(n)
        bridge[int(0.002 * sr)] = 1.0
        bridge[int(0.008 * sr)] = 0.4
        drive, diag = build_body_drive_from_bridge(bridge.astype(np.float64), sr, state, "A4")
        self.assertEqual(len(drive), n)
        self.assertTrue(diag.get("body_drive_from_bridge"))
        self.assertTrue(diag.get("no_pre_delay"))
        self.assertTrue(diag.get("no_independent_body_pluck"))
        self.assertIn("body_bloom_attack_ms", diag)

    def test_body_bloom_attack_differs_per_sample(self) -> None:
        self.assertLess(BODY_BLOOM_ATTACK_MS["sample_001"][0], BODY_BLOOM_ATTACK_MS["sample_002"][0])
        self.assertLess(BODY_BLOOM_ATTACK_MS["sample_000"][0], BODY_BLOOM_ATTACK_MS["sample_002"][0])

    def test_high_note_low_mid_support_differs(self) -> None:
        self.assertLess(
            HIGH_NOTE_LOW_MID_BOOST["E5"]["sample_001"],
            HIGH_NOTE_LOW_MID_BOOST["E5"]["sample_002"],
        )
        self.assertLess(
            HIGH_NOTE_LOW_MID_BOOST["A4"]["sample_001"],
            HIGH_NOTE_LOW_MID_BOOST["A4"]["sample_002"],
        )


class TestV11Readiness(unittest.TestCase):
    def test_body_second_pluck_readiness(self) -> None:
        ok = build_readiness_emergency_demo(
            files_generated=9,
            peaks_controlled=True,
            family_metrics={"pass": True, "too_similar": False},
            body_second_pluck_ok=True,
        )
        self.assertEqual(ok.get("current_status"), READINESS_OK)
        bad = build_readiness_emergency_demo(
            files_generated=9,
            peaks_controlled=True,
            family_metrics={"pass": True},
            body_second_pluck_ok=False,
        )
        self.assertEqual(bad.get("current_status"), READINESS_BODY_SECOND_PLUCK)


class TestV11SourceGuards(unittest.TestCase):
    def test_body_bloom_in_source(self) -> None:
        import pgsm_emergency_guitar_demo_engine as mod

        src = Path(mod.__file__).read_text(encoding="utf-8")
        self.assertIn("def build_body_drive_from_bridge", src)
        self.assertIn('onset_repair_mode == "bloom"', src)
        self.assertIn("body_second_pluck_risk", src)
        self.assertIn("body_drive_summary", src)
        self.assertIn("pgsm_final_guitar_demo_v11", src)
        self.assertIn("V11_VOICING", src)
        self.assertIn("cleanup_sample_state(state)", src)
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
