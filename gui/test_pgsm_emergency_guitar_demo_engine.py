#!/usr/bin/env python3
"""Lightweight tests for PGSM emergency guitar demo engine v2 (no synthesis runtime)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gui"))

from pgsm_emergency_guitar_demo_engine import (  # noqa: E402
    EMERGENCY_DEMO_VERSION,
    NOTE_SET,
    PHYSICAL_FACTOR_GROUPS,
    READINESS_FAIL,
    READINESS_OK,
    READINESS_WEAK,
    SAMPLE_SET,
    SAMPLE_VOICING_CALIBRATION,
    build_anti_cheat_checks,
    build_emergency_demo_config,
    build_readiness_emergency_demo,
    compute_physical_factors,
    demo_wav_filename,
    extract_physical_parameters,
)
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE  # noqa: E402
from stk_v6_2_audit_features import load_audit_report  # noqa: E402


class TestEmergencyDemoConfig(unittest.TestCase):
    def test_build_emergency_demo_config(self) -> None:
        cfg = build_emergency_demo_config()
        self.assertEqual(cfg.get("emergency_demo_version"), EMERGENCY_DEMO_VERSION)
        self.assertEqual(cfg.get("validation_mode"), "emergency_demo")
        self.assertEqual(len(cfg.get("physical_factor_groups") or []), 10)

    def test_eight_core_physical_factor_groups(self) -> None:
        core = PHYSICAL_FACTOR_GROUPS[:8]
        self.assertEqual(len(core), 8)
        self.assertIn("body_size_cavity_factor", core)
        self.assertIn("radiation_brightness_factor", core)

    def test_demo_wav_filename_pattern(self) -> None:
        self.assertEqual(demo_wav_filename("sample_000", "A2"), "sample_000_A2_guitar_demo.wav")
        self.assertEqual(demo_wav_filename("sample_002", "E5"), "sample_002_E5_guitar_demo.wav")

    def test_sample_voicing_profiles_differ(self) -> None:
        profiles = {sid: SAMPLE_VOICING_CALIBRATION[sid]["voicing_profile"] for sid in SAMPLE_SET}
        self.assertEqual(len(set(profiles.values())), 3)


class TestPhysicalTrace(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._audit = load_audit_report()

    def test_physical_parameters_per_sample(self) -> None:
        for sid in SAMPLE_SET:
            row = extract_physical_parameters(sid, self._audit)
            self.assertEqual(row.get("sample_id"), sid)
            self.assertIn("bridge_mobility_proxy", row)

    def test_physical_factors_and_trace(self) -> None:
        ref = extract_physical_parameters("sample_000", self._audit)
        for sid in SAMPLE_SET:
            phys = extract_physical_parameters(sid, self._audit)
            factors, trace, exag = compute_physical_factors(phys, ref, sample_id=sid)
            self.assertGreaterEqual(len(trace), 8)
            self.assertTrue(exag)
            for key in PHYSICAL_FACTOR_GROUPS:
                self.assertIn(key, factors)
            if sid != "sample_000":
                self.assertNotEqual(
                    factors["radiation_brightness_factor"],
                    SAMPLE_VOICING_CALIBRATION["sample_000"]["radiation_brightness_factor"],
                )


class TestAntiCheatAndReadiness(unittest.TestCase):
    def test_anti_cheat_rejects_loudness_only(self) -> None:
        traces = {sid: [{"factor": f"f{i}"} for i in range(9)] for sid in SAMPLE_SET}
        spectral = {
            sid: {note: {"rms_dbfs": -20.0, "peak_dbfs": -6.0} for note in NOTE_SET}
            for sid in SAMPLE_SET
        }
        pairwise = {"mean_overall_differentiation_score": 0.01}
        correlation = {"max_correlation": 0.999}
        ac = build_anti_cheat_checks(
            traces=traces,
            pairwise=pairwise,
            correlation=correlation,
            spectral_metrics=spectral,
            sample_ids=SAMPLE_SET,
        )
        self.assertFalse(ac.get("differences_not_only_loudness"))

    def test_readiness_labels(self) -> None:
        ok = build_readiness_emergency_demo(
            files_generated=9,
            expected_files=9,
            mean_differentiation=0.06,
            max_correlation=0.94,
            peaks_controlled=True,
        )
        self.assertEqual(ok.get("current_status"), READINESS_OK)
        weak = build_readiness_emergency_demo(
            files_generated=9,
            expected_files=9,
            mean_differentiation=0.06,
            max_correlation=0.985,
            peaks_controlled=True,
        )
        self.assertEqual(weak.get("current_status"), READINESS_WEAK)
        fail = build_readiness_emergency_demo(
            files_generated=3,
            expected_files=9,
            mean_differentiation=0.2,
            max_correlation=0.5,
            peaks_controlled=True,
        )
        self.assertEqual(fail.get("current_status"), READINESS_FAIL)


class TestEmergencyDemoGuards(unittest.TestCase):
    def test_no_stk_fem_rom_in_module(self) -> None:
        import pgsm_emergency_guitar_demo_engine as mod

        src = Path(mod.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import diagnostic_synthesis", src)
        self.assertNotIn("subprocess", src)
        self.assertNotIn("rom_manager", src)

    def test_no_fem_rom_subprocess(self) -> None:
        with patch("subprocess.run") as mock_run, patch("subprocess.Popen") as mock_popen:
            cfg = build_emergency_demo_config()
            self.assertEqual(cfg.get("emergency_demo_version"), EMERGENCY_DEMO_VERSION)
            mock_run.assert_not_called()
            mock_popen.assert_not_called()

    def test_website_default_unchanged(self) -> None:
        self.assertEqual(DEFAULT_WEBSITE_STK_MODE, DEFAULT_WEBSITE_STK_MODE)


if __name__ == "__main__":
    unittest.main()
