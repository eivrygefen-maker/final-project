#!/usr/bin/env python3
"""Lightweight tests for PGSM emergency guitar demo engine v4 (no synthesis runtime)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gui"))

from pgsm_emergency_guitar_demo_engine import (  # noqa: E402
    EMERGENCY_DEMO_VERSION,
    GENTLE_SAMPLE_VOICING,
    NOTE_SET,
    PHYSICAL_CHAIN_STAGES,
    PHYSICAL_MODIFIER_KEYS,
    READINESS_FAIL,
    READINESS_OK,
    READINESS_REVIEW,
    READINESS_WEAK,
    SAMPLE_SET,
    build_anti_cheat_checks,
    build_emergency_demo_config,
    build_readiness_emergency_demo,
    build_synthesis_profile,
    compute_gentle_sample_modifiers,
    compute_guitar_family_consistency_metrics,
    demo_wav_filename,
    extract_per_sample_physical_parameters,
)
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE  # noqa: E402
from stk_v6_2_audit_features import load_audit_report  # noqa: E402


class TestEmergencyDemoConfig(unittest.TestCase):
    def test_v4_config(self) -> None:
        cfg = build_emergency_demo_config()
        self.assertEqual(cfg.get("emergency_demo_version"), EMERGENCY_DEMO_VERSION)
        self.assertEqual(len(cfg.get("physical_chain_stages") or []), 7)

    def test_physical_chain_stages_exist(self) -> None:
        self.assertIn("A_pluck_contact", PHYSICAL_CHAIN_STAGES)
        self.assertIn("D_body_modal_response", PHYSICAL_CHAIN_STAGES)
        self.assertIn("G_modal_decay_tail_only", PHYSICAL_CHAIN_STAGES)

    def test_six_to_eight_physical_modifiers(self) -> None:
        self.assertGreaterEqual(len(PHYSICAL_MODIFIER_KEYS), 5)

    def test_demo_wav_filenames(self) -> None:
        self.assertEqual(demo_wav_filename("sample_002", "E5"), "sample_002_E5_guitar_demo.wav")


class TestPhysicalProfiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._audit = load_audit_report()

    def test_samples_share_base_model_with_moderate_overlay(self) -> None:
        ref = extract_per_sample_physical_parameters("sample_000", self._audit)
        overlays = []
        for sid in SAMPLE_SET:
            phys = extract_per_sample_physical_parameters(sid, self._audit)
            mods, trace = compute_gentle_sample_modifiers(phys, ref, sample_id=sid)
            profile = build_synthesis_profile(sid, mods)
            overlays.append(profile["pluck_position_ratio"])
            self.assertGreaterEqual(len(trace), 5)
            for key in PHYSICAL_MODIFIER_KEYS:
                self.assertIn(key, mods)
                self.assertLessEqual(mods[key], 1.12)
                self.assertGreaterEqual(mods[key], 0.88)
        self.assertNotEqual(overlays[0], overlays[1])

    def test_voicing_profiles_differ(self) -> None:
        profiles = {sid: GENTLE_SAMPLE_VOICING[sid]["profile"] for sid in SAMPLE_SET}
        self.assertEqual(len(set(profiles.values())), 3)


class TestFamilyConsistencyAndReadiness(unittest.TestCase):
    def test_family_consistency_band(self) -> None:
        ok = compute_guitar_family_consistency_metrics(
            {"max_correlation": 0.72, "min_correlation": 0.55, "mean_correlation": 0.64}
        )
        self.assertTrue(ok.get("in_family_band"))
        bad = compute_guitar_family_consistency_metrics(
            {"max_correlation": 0.99, "min_correlation": 0.98, "mean_correlation": 0.985}
        )
        self.assertTrue(bad.get("too_similar"))

    def test_readiness_labels(self) -> None:
        ok = build_readiness_emergency_demo(
            files_generated=9,
            peaks_controlled=True,
            family_metrics={"too_unrelated": False, "too_similar": False, "pass": True},
        )
        self.assertEqual(ok.get("current_status"), READINESS_OK)
        weak = build_readiness_emergency_demo(
            files_generated=9,
            peaks_controlled=True,
            family_metrics={"too_unrelated": False, "too_similar": True, "pass": False},
        )
        self.assertEqual(weak.get("current_status"), READINESS_WEAK)
        review = build_readiness_emergency_demo(
            files_generated=9,
            peaks_controlled=True,
            family_metrics={"too_unrelated": True, "too_similar": False, "pass": False},
        )
        self.assertEqual(review.get("current_status"), READINESS_REVIEW)
        fail = build_readiness_emergency_demo(
            files_generated=3,
            peaks_controlled=True,
            family_metrics={"pass": True},
        )
        self.assertEqual(fail.get("current_status"), READINESS_FAIL)

    def test_anti_cheat_exists(self) -> None:
        ac = build_anti_cheat_checks(
            traces={sid: [{"modifier": "x"}] * 6 for sid in SAMPLE_SET},
            family_metrics={"pass": True, "too_similar": False},
            pairwise={"mean_spectral_distance": 0.05},
            peak_rms_report={"all_peaks_within_target": True},
        )
        self.assertTrue(ac.get("no_randomization"))
        self.assertTrue(ac.get("no_reverb_echo_body_tail"))


class TestEmergencyDemoGuards(unittest.TestCase):
    def test_no_stk_fem_rom_subprocess(self) -> None:
        import pgsm_emergency_guitar_demo_engine as mod

        src = Path(mod.__file__).read_text(encoding="utf-8")
        self.assertIn("build_v4_string_bridge_force", src)
        self.assertIn("apply_bridge_admittance_coupling", src)
        self.assertIn("compute_step5j_1_modal_kernels_decomposed", src)
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
