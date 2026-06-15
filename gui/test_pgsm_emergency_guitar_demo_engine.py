#!/usr/bin/env python3
"""Lightweight tests for PGSM final guitar demo engine v5 (no synthesis runtime)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gui"))

from pgsm_emergency_guitar_demo_engine import (  # noqa: E402
    AUDIO_DIR_V5,
    FINAL_DEMO_VERSION,
    GENTLE_SAMPLE_VOICING,
    NOTE_SET,
    PHYSICAL_CHAIN_STAGES,
    PHYSICAL_FACTOR_KEYS,
    PHYSICAL_MODIFIER_KEYS,
    READINESS_FAIL,
    READINESS_OK,
    READINESS_REVIEW,
    READINESS_WEAK,
    SAMPLE_SET,
    SampleSynthesisState,
    build_anti_cheat_checks,
    build_emergency_demo_config,
    build_readiness_emergency_demo,
    build_sample_synthesis_state,
    build_synthesis_profile,
    cleanup_sample_state,
    compute_gentle_sample_modifiers,
    compute_guitar_family_consistency_metrics,
    demo_wav_filename,
    extract_per_sample_physical_parameters,
    final_wav_filename,
)
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE  # noqa: E402
from stk_v6_2_audit_features import load_audit_report  # noqa: E402


class TestFinalDemoV5Config(unittest.TestCase):
    def test_v5_config(self) -> None:
        cfg = build_emergency_demo_config()
        self.assertEqual(cfg.get("final_demo_version"), FINAL_DEMO_VERSION)
        self.assertEqual(cfg.get("final_demo_version"), "v5_ordered_physical_transfer_chain")
        self.assertEqual(len(cfg.get("physical_chain_stages") or []), 7)
        self.assertTrue(cfg.get("diagnostic_exaggeration_for_audible_demo"))

    def test_physical_chain_stages_exist(self) -> None:
        self.assertIn("A_pluck_contact", PHYSICAL_CHAIN_STAGES)
        self.assertIn("D_body_modal_transfer", PHYSICAL_CHAIN_STAGES)
        self.assertIn("G_modal_decay_only", PHYSICAL_CHAIN_STAGES)

    def test_eight_physical_factors(self) -> None:
        self.assertEqual(len(PHYSICAL_FACTOR_KEYS), 8)
        self.assertEqual(PHYSICAL_MODIFIER_KEYS, PHYSICAL_FACTOR_KEYS)

    def test_v5_output_filenames_and_folder(self) -> None:
        self.assertEqual(
            final_wav_filename("sample_002", "E5"),
            "sample_002_E5_final_guitar.wav",
        )
        self.assertEqual(demo_wav_filename("sample_000", "A2"), "sample_000_A2_final_guitar.wav")
        self.assertIn("pgsm_final_guitar_demo_v5", str(AUDIO_DIR_V5))


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
            self.assertGreaterEqual(len(trace), 8)
            for key in PHYSICAL_MODIFIER_KEYS:
                self.assertIn(key, mods)
                self.assertLessEqual(mods[key], 1.14)
                self.assertGreaterEqual(mods[key], 0.86)
        self.assertNotEqual(overlays[0], overlays[1])

    def test_voicing_profiles_differ(self) -> None:
        profiles = {sid: GENTLE_SAMPLE_VOICING[sid]["profile"] for sid in SAMPLE_SET}
        self.assertEqual(len(set(profiles.values())), 3)


class TestIsolatedSampleState(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._audit = load_audit_report()
        cls._ref = extract_per_sample_physical_parameters("sample_000", cls._audit)

    def test_build_sample_synthesis_state_is_independent(self) -> None:
        from pgsm_emergency_guitar_demo_engine import _load_readonly_reference_modes

        ref_modes = _load_readonly_reference_modes()
        mode_freqs: list[float] = []
        factor_snapshots: list[dict] = []
        for sid in SAMPLE_SET:
            phys = extract_per_sample_physical_parameters(sid, self._audit)
            state = build_sample_synthesis_state(
                sid,
                physical=phys,
                reference_physical=self._ref,
                readonly_modes=ref_modes,
            )
            factor_snapshots.append(dict(state.factors))
            mode_freqs.append(float(state.modes[0]["frequency_hz"]))
            self.assertIsInstance(state, SampleSynthesisState)
            self.assertTrue(state.isolation_meta.get("independent_state_created"))
            self.assertGreaterEqual(state.isolation_meta.get("derived_mode_count", 0), 5)
            self.assertIn("bridge_transfer_hash", state.isolation_meta)
            self.assertIn("modal_transfer_hash", state.isolation_meta)
            cleanup_sample_state(state)
            self.assertTrue(state.isolation_meta.get("cleanup_completed"))

        self.assertNotEqual(factor_snapshots[0], factor_snapshots[1])
        self.assertNotEqual(mode_freqs[0], mode_freqs[1])


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
        isolation = [
            {"independent_state_created": True, "cleanup_completed": True},
            {"independent_state_created": True, "cleanup_completed": True},
            {"independent_state_created": True, "cleanup_completed": True},
        ]
        ac = build_anti_cheat_checks(
            isolation_reports=isolation,
            family_metrics={"pass": True, "too_similar": False},
            pairwise={"mean_spectral_distance": 0.05},
            peak_rms_report={"all_peaks_within_target": True, "v5_folder_cleared": True},
        )
        self.assertTrue(ac.get("no_randomization"))
        self.assertTrue(ac.get("no_reverb_echo_body_tail"))
        self.assertTrue(ac.get("per_sample_state_isolated"))
        self.assertTrue(ac.get("no_cross_sample_mutable_state"))
        self.assertTrue(ac.get("no_stale_wav_mix"))


class TestFinalDemoGuards(unittest.TestCase):
    def test_transfer_chain_and_isolation_in_source(self) -> None:
        import pgsm_emergency_guitar_demo_engine as mod

        src = Path(mod.__file__).read_text(encoding="utf-8")
        self.assertIn("build_sample_synthesis_state", src)
        self.assertIn("F_bridge_eff", src)
        self.assertIn("cleanup_sample_state", src)
        self.assertIn("pgsm_final_guitar_demo_v5", src)
        self.assertNotIn("build_v4_string_bridge_force", src)
        self.assertNotIn("apply_listening_render_step5j_1", src)
        self.assertNotIn("subprocess", src)
        self.assertNotIn("rom_manager", src)

        with patch("subprocess.run") as mock_run, patch("subprocess.Popen") as mock_popen:
            build_emergency_demo_config()
            mock_run.assert_not_called()
            mock_popen.assert_not_called()

    def test_modal_transfer_driven_by_bridge_force(self) -> None:
        import pgsm_emergency_guitar_demo_engine as mod

        src = Path(mod.__file__).read_text(encoding="utf-8")
        self.assertIn("_mode_transfer(f_bridge_eff", src)
        self.assertIn("driven_by", src)
        self.assertIn("F_bridge_eff", src)

    def test_website_default_unchanged(self) -> None:
        self.assertIsNotNone(DEFAULT_WEBSITE_STK_MODE)


if __name__ == "__main__":
    unittest.main()
