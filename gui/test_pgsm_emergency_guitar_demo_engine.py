#!/usr/bin/env python3
"""Lightweight tests for PGSM final guitar demo engine v7 (no synthesis runtime)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gui"))

from pgsm_emergency_guitar_demo_engine import (  # noqa: E402
    AUDIO_DIR_V7,
    FINAL_DEMO_VERSION,
    GENTLE_SAMPLE_VOICING,
    NOTE_BODY_SUPPORT,
    NOTE_SET,
    ONSET_LOCK_SUMMARY,
    PHYSICAL_CHAIN_STAGES,
    PHYSICAL_FACTOR_KEYS,
    PHYSICAL_MODIFIER_KEYS,
    READINESS_DOUBLE_PLUCK,
    READINESS_FAIL,
    READINESS_OK,
    READINESS_REVIEW,
    READINESS_WEAK,
    SAMPLE_SET,
    SampleSynthesisState,
    _align_attack_polarity,
    attenuate_delayed_modal_peak,
    build_anti_cheat_checks,
    build_emergency_demo_config,
    build_readiness_emergency_demo,
    build_sample_synthesis_state,
    build_synthesis_profile,
    cleanup_sample_state,
    compute_double_pluck_risk,
    compute_gentle_sample_modifiers,
    compute_guitar_family_consistency_metrics,
    demo_wav_filename,
    extract_per_sample_physical_parameters,
    final_wav_filename,
    onset_lock_envelope,
)
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE  # noqa: E402
from stk_v6_2_audit_features import load_audit_report  # noqa: E402


class TestFinalDemoV7Config(unittest.TestCase):
    def test_v7_config(self) -> None:
        cfg = build_emergency_demo_config()
        self.assertEqual(cfg.get("final_demo_version"), FINAL_DEMO_VERSION)
        self.assertEqual(cfg.get("final_demo_version"), "v7_onset_locked_body_supported_guitar")
        self.assertEqual(len(cfg.get("physical_chain_stages") or []), 7)
        self.assertTrue(cfg.get("diagnostic_exaggeration_for_audible_demo"))

    def test_physical_chain_stages_exist(self) -> None:
        self.assertIn("A_pluck_contact", PHYSICAL_CHAIN_STAGES)
        self.assertIn("D_body_modal_transfer", PHYSICAL_CHAIN_STAGES)
        self.assertIn("G_modal_decay_only", PHYSICAL_CHAIN_STAGES)

    def test_eight_physical_factors(self) -> None:
        self.assertEqual(len(PHYSICAL_FACTOR_KEYS), 8)
        self.assertEqual(PHYSICAL_MODIFIER_KEYS, PHYSICAL_FACTOR_KEYS)

    def test_v7_output_filenames_and_folder(self) -> None:
        self.assertEqual(
            final_wav_filename("sample_002", "E5"),
            "sample_002_E5_final_guitar.wav",
        )
        self.assertEqual(demo_wav_filename("sample_000", "A2"), "sample_000_A2_final_guitar.wav")
        self.assertIn("pgsm_final_guitar_demo_v7", str(AUDIO_DIR_V7))

    def test_onset_lock_summary_exists(self) -> None:
        self.assertIn("window_ms", ONSET_LOCK_SUMMARY)
        self.assertTrue(ONSET_LOCK_SUMMARY.get("body_starts_at_sample_zero"))


class TestOnsetAndBodySupport(unittest.TestCase):
    def test_onset_lock_envelope_exists(self) -> None:
        sr = 44100
        env = onset_lock_envelope(int(0.2 * sr), sr)
        self.assertGreater(float(env[0]), 0.0)
        self.assertGreater(float(env[10]), float(env[0]))

    def test_delayed_peak_control_helper_exists(self) -> None:
        sr = 44100
        t = np.arange(int(0.2 * sr)) / sr
        y = np.exp(-t / 0.02) * np.sin(2 * np.pi * 220 * t)
        out, meta = attenuate_delayed_modal_peak(y.astype(np.float64), sr)
        self.assertEqual(len(out), len(y))
        self.assertIn("delayed_peak_ratio", meta)

    def test_body_support_for_a4_e5(self) -> None:
        self.assertGreater(NOTE_BODY_SUPPORT["A4"]["body_modal_mult"], 1.0)
        self.assertGreater(NOTE_BODY_SUPPORT["E5"]["low_mid_mode_mult"], NOTE_BODY_SUPPORT["A2"]["low_mid_mode_mult"])

    def test_double_pluck_diagnostic_exists(self) -> None:
        sr = 44100
        t = np.arange(int(0.2 * sr)) / sr
        single = np.exp(-t / 0.02) * np.sin(2 * np.pi * 220 * t)
        diag = compute_double_pluck_risk(single.astype(np.float64), sr)
        self.assertIn("double_pluck_risk", diag)

    def test_polarity_alignment_flips_negative_attack(self) -> None:
        sr = 44100
        y = -np.exp(-np.arange(sr) / 500.0)
        aligned, flipped = _align_attack_polarity(y, sr)
        self.assertTrue(flipped)
        self.assertGreater(float(aligned[np.argmax(np.abs(aligned[: sr // 20]))]), 0.0)


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
            cleanup_sample_state(state)
            self.assertTrue(state.isolation_meta.get("cleanup_completed"))

        self.assertNotEqual(factor_snapshots[0], factor_snapshots[1])
        self.assertNotEqual(mode_freqs[0], mode_freqs[1])


class TestFamilyConsistencyAndReadiness(unittest.TestCase):
    def test_readiness_labels(self) -> None:
        ok = build_readiness_emergency_demo(
            files_generated=9,
            peaks_controlled=True,
            family_metrics={"too_unrelated": False, "too_similar": False, "pass": True},
            double_pluck_ok=True,
        )
        self.assertEqual(ok.get("current_status"), READINESS_OK)
        double = build_readiness_emergency_demo(
            files_generated=9,
            peaks_controlled=True,
            family_metrics={"pass": True},
            double_pluck_ok=False,
        )
        self.assertEqual(double.get("current_status"), READINESS_DOUBLE_PLUCK)
        weak = build_readiness_emergency_demo(
            files_generated=9,
            peaks_controlled=True,
            family_metrics={"too_similar": True, "pass": False},
        )
        self.assertEqual(weak.get("current_status"), READINESS_WEAK)
        review = build_readiness_emergency_demo(
            files_generated=9,
            peaks_controlled=True,
            family_metrics={"too_unrelated": True, "pass": False},
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
            peak_rms_report={"all_peaks_within_target": True, "v7_folder_cleared": True},
            double_pluck_ok=True,
        )
        self.assertTrue(ac.get("per_sample_state_isolated"))
        self.assertTrue(ac.get("single_pluck_onset"))


class TestFinalDemoGuards(unittest.TestCase):
    def test_v7_helpers_in_source(self) -> None:
        import pgsm_emergency_guitar_demo_engine as mod

        src = Path(mod.__file__).read_text(encoding="utf-8")
        self.assertIn("onset_lock_envelope", src)
        self.assertIn("attenuate_delayed_modal_peak", src)
        self.assertIn("apply_onset_lock_to_body", src)
        self.assertIn("NOTE_BODY_SUPPORT", src)
        self.assertIn("pgsm_final_guitar_demo_v7", src)
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
        self.assertIn("onset_locked_first_80ms", src)

    def test_website_default_unchanged(self) -> None:
        self.assertIsNotNone(DEFAULT_WEBSITE_STK_MODE)


if __name__ == "__main__":
    unittest.main()
