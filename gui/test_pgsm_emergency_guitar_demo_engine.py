#!/usr/bin/env python3
"""Lightweight tests for PGSM final guitar demo engine v8 (no synthesis runtime)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gui"))

from pgsm_emergency_guitar_demo_engine import (  # noqa: E402
    AUDIO_DIR_V8,
    FINAL_DEMO_VERSION,
    GENTLE_SAMPLE_VOICING,
    NOTE_SET,
    PHYSICAL_CHAIN_STAGES,
    PHYSICAL_FACTOR_KEYS,
    PHYSICAL_MODIFIER_KEYS,
    READINESS_DOUBLE_PLUCK,
    READINESS_FAIL,
    READINESS_OK,
    READINESS_OK_LIMITED,
    SAMPLE_SET,
    SampleSynthesisState,
    V8_VOICING,
    _align_attack_polarity,
    _per_sample_body_support,
    _pluck_contact_shape,
    build_anti_cheat_checks,
    build_emergency_demo_config,
    build_readiness_emergency_demo,
    build_sample_synthesis_state,
    build_synthesis_profile,
    cleanup_sample_state,
    compute_early_double_attack_risk,
    compute_gentle_sample_modifiers,
    local_limited_delayed_peak_correction,
    demo_wav_filename,
    extract_per_sample_physical_parameters,
    final_wav_filename,
)
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE  # noqa: E402
from stk_v6_2_audit_features import load_audit_report  # noqa: E402


class TestFinalDemoV8Config(unittest.TestCase):
    def test_v8_config(self) -> None:
        cfg = build_emergency_demo_config()
        self.assertEqual(cfg.get("final_demo_version"), FINAL_DEMO_VERSION)
        self.assertEqual(cfg.get("final_demo_version"), "v8_final_physical_guitar_demo")
        self.assertIn("pgsm_final_guitar_demo_v8", str(AUDIO_DIR_V8))
        self.assertEqual(len(cfg.get("physical_chain_stages") or []), 7)

    def test_eight_physical_factors(self) -> None:
        self.assertEqual(len(PHYSICAL_FACTOR_KEYS), 8)
        self.assertEqual(PHYSICAL_MODIFIER_KEYS, PHYSICAL_FACTOR_KEYS)

    def test_v8_output_filenames(self) -> None:
        self.assertEqual(final_wav_filename("sample_002", "E5"), "sample_002_E5_final_guitar.wav")
        self.assertEqual(demo_wav_filename("sample_000", "A2"), "sample_000_A2_final_guitar.wav")

    def test_per_sample_factor_multipliers_differ(self) -> None:
        m0 = V8_VOICING["sample_000"]["factor_multipliers"]
        m1 = V8_VOICING["sample_001"]["factor_multipliers"]
        m2 = V8_VOICING["sample_002"]["factor_multipliers"]
        self.assertNotEqual(m1["bridge_mobility_factor"], m2["bridge_mobility_factor"])
        self.assertNotEqual(m1["radiation_brightness_factor"], m2["radiation_brightness_factor"])


class TestV8SynthesisHelpers(unittest.TestCase):
    def test_pluck_contact_shape_embedded(self) -> None:
        sr = 44100
        shape = _pluck_contact_shape(int(0.05 * sr), sr)
        self.assertGreater(float(shape[0]), 0.0)

    def test_local_delayed_peak_limited(self) -> None:
        sr = 44100
        t = np.arange(int(0.2 * sr)) / sr
        y = np.exp(-t / 0.02) * np.sin(2 * np.pi * 220 * t)
        _, meta = local_limited_delayed_peak_correction(y.astype(np.float64), sr)
        self.assertIn("attenuation_amount", meta)
        if meta.get("attenuation_applied"):
            self.assertLessEqual(float(meta["attenuation_amount"]), 0.25)

    def test_early_double_attack_diagnostic(self) -> None:
        sr = 44100
        y = np.zeros(int(0.03 * sr))
        y[0] = 1.0
        y[50] = 0.5
        diag = compute_early_double_attack_risk(y, sr)
        self.assertIn("early_double_attack_risk", diag)

    def test_per_sample_body_support_differs(self) -> None:
        a4_bright = _per_sample_body_support("sample_001", "A4")
        a4_warm = _per_sample_body_support("sample_002", "A4")
        self.assertLess(a4_bright["low_mid_mode_mult"], a4_warm["low_mid_mode_mult"])


class TestPhysicalProfiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._audit = load_audit_report()

    def test_samples_have_distinct_factors(self) -> None:
        ref = extract_per_sample_physical_parameters("sample_000", self._audit)
        factors = []
        for sid in SAMPLE_SET:
            phys = extract_per_sample_physical_parameters(sid, cls._audit)
            mods, trace = compute_gentle_sample_modifiers(phys, ref, sample_id=sid)
            factors.append(mods)
            profile = build_synthesis_profile(sid, mods)
            self.assertGreaterEqual(len(trace), 8)
            self.assertIn("voicing_profile", profile)
        self.assertNotEqual(factors[0]["bridge_mobility_factor"], factors[1]["bridge_mobility_factor"])

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
        snaps: list[dict] = []
        for sid in SAMPLE_SET:
            phys = extract_per_sample_physical_parameters(sid, cls._audit)
            state = build_sample_synthesis_state(
                sid,
                physical=phys,
                reference_physical=cls._ref,
                readonly_modes=ref_modes,
            )
            snaps.append(dict(state.factors))
            self.assertIsInstance(state, SampleSynthesisState)
            self.assertTrue(state.isolation_meta.get("independent_state_created"))
            cleanup_sample_state(state)
            self.assertTrue(state.isolation_meta.get("cleanup_completed"))
        self.assertNotEqual(snaps[0], snaps[1])


class TestReadinessAndGuards(unittest.TestCase):
    def test_readiness_labels(self) -> None:
        ok = build_readiness_emergency_demo(
            files_generated=9,
            peaks_controlled=True,
            family_metrics={"too_unrelated": False, "too_similar": False, "pass": True},
            double_pluck_ok=True,
            early_attack_ok=True,
        )
        self.assertEqual(ok.get("current_status"), READINESS_OK)
        limited = build_readiness_emergency_demo(
            files_generated=9,
            peaks_controlled=True,
            family_metrics={"too_similar": False, "pass": False},
            usable_with_limitations=True,
        )
        self.assertEqual(limited.get("current_status"), READINESS_OK_LIMITED)
        double = build_readiness_emergency_demo(
            files_generated=9,
            peaks_controlled=True,
            family_metrics={"pass": True},
            early_attack_ok=False,
        )
        self.assertEqual(double.get("current_status"), READINESS_DOUBLE_PLUCK)
        fail = build_readiness_emergency_demo(files_generated=3, peaks_controlled=True, family_metrics={"pass": True})
        self.assertEqual(fail.get("current_status"), READINESS_FAIL)

    def test_v8_source_guards(self) -> None:
        import pgsm_emergency_guitar_demo_engine as mod

        src = Path(mod.__file__).read_text(encoding="utf-8")
        self.assertIn("_pluck_contact_shape", src)
        self.assertIn("local_limited_delayed_peak_correction", src)
        self.assertIn("compute_early_double_attack_risk", src)
        self.assertIn("F_excitation_embedded_contact", src)
        self.assertIn("no_separate_contact_mix_layer", src)
        self.assertIn("global_onset_lock_disabled", src)
        self.assertNotIn("apply_onset_lock_to_body(y_body", src.split("def synthesize_note_for_sample")[1])
        self.assertNotIn("subprocess", src)
        self.assertNotIn("rom_manager", src)

        with patch("subprocess.run") as mock_run, patch("subprocess.Popen") as mock_popen:
            build_emergency_demo_config()
            mock_run.assert_not_called()
            mock_popen.assert_not_called()

    def test_anti_cheat_no_global_onset_lock(self) -> None:
        isolation = [{"independent_state_created": True, "cleanup_completed": True}] * 3
        ac = build_anti_cheat_checks(
            isolation_reports=isolation,
            family_metrics={"pass": True, "too_similar": False},
            pairwise={"mean_spectral_distance": 0.05},
            peak_rms_report={"all_peaks_within_target": True, "v8_folder_cleared": True},
        )
        self.assertTrue(ac.get("no_global_onset_lock"))

    def test_website_default_unchanged(self) -> None:
        self.assertIsNotNone(DEFAULT_WEBSITE_STK_MODE)

    def test_polarity_alignment(self) -> None:
        sr = 44100
        y = -np.exp(-np.arange(sr) / 500.0)
        aligned, flipped = _align_attack_polarity(y, sr)
        self.assertTrue(flipped)
        self.assertGreater(float(aligned[0]), 0.0)


if __name__ == "__main__":
    unittest.main()
