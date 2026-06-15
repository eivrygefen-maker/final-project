#!/usr/bin/env python3
"""Lightweight tests for PGSM emergency guitar demo engine v3 (no synthesis runtime)."""
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
    READINESS_MODERATE,
    READINESS_OK,
    READINESS_WEAK,
    SAMPLE_SET,
    SAMPLE_VOICING_CALIBRATION,
    build_anti_cheat_checks,
    build_body_resonator_bank,
    build_emergency_demo_config,
    build_readiness_emergency_demo,
    build_synthesis_profile,
    compute_physical_factors,
    demo_wav_filename,
    extract_physical_parameters,
)
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE  # noqa: E402
from stk_v6_2_audit_features import load_audit_report  # noqa: E402


class TestEmergencyDemoConfig(unittest.TestCase):
    def test_v3_config(self) -> None:
        cfg = build_emergency_demo_config()
        self.assertEqual(cfg.get("emergency_demo_version"), EMERGENCY_DEMO_VERSION)
        self.assertEqual(len(cfg.get("physical_factor_groups") or []), 10)

    def test_ten_physical_factor_groups(self) -> None:
        self.assertIn("effective_pluck_position_factor", PHYSICAL_FACTOR_GROUPS)
        self.assertIn("bridge_transfer_attack_factor", PHYSICAL_FACTOR_GROUPS)
        self.assertIn("radiation_brightness_factor", PHYSICAL_FACTOR_GROUPS)

    def test_demo_wav_filename_pattern(self) -> None:
        self.assertEqual(demo_wav_filename("sample_001", "A4"), "sample_001_A4_guitar_demo.wav")


class TestPhysicalTrace(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._audit = load_audit_report()

    def test_sample_profiles_differ_in_at_least_eight_factors(self) -> None:
        diffs = 0
        ref = SAMPLE_VOICING_CALIBRATION["sample_000"]
        for sid in ("sample_001", "sample_002"):
            cal = SAMPLE_VOICING_CALIBRATION[sid]
            for key in PHYSICAL_FACTOR_GROUPS:
                if cal[key] != ref[key]:
                    diffs += 1
        self.assertGreaterEqual(diffs, 16)

    def test_pluck_positions_differ_across_samples(self) -> None:
        ref = extract_physical_parameters("sample_000", self._audit)
        positions = []
        for sid in SAMPLE_SET:
            phys = extract_physical_parameters(sid, self._audit)
            factors, _, _ = compute_physical_factors(phys, ref, sample_id=sid)
            profile = build_synthesis_profile(sid, factors, phys)
            positions.append(profile["pluck_position_ratio"])
        self.assertEqual(len(set(positions)), 3)

    def test_body_resonator_banks_differ(self) -> None:
        ref = extract_physical_parameters("sample_000", self._audit)
        banks = []
        for sid in SAMPLE_SET:
            phys = extract_physical_parameters(sid, self._audit)
            factors, _, _ = compute_physical_factors(phys, ref, sample_id=sid)
            bank = build_body_resonator_bank(sid, factors, phys)
            banks.append(tuple(r["frequency_hz"] for r in bank))
        self.assertNotEqual(banks[0], banks[1])
        self.assertNotEqual(banks[1], banks[2])

    def test_physical_trace_has_synthesis_path(self) -> None:
        ref = extract_physical_parameters("sample_000", self._audit)
        phys = extract_physical_parameters("sample_001", self._audit)
        _, trace, _ = compute_physical_factors(phys, ref, sample_id="sample_001")
        self.assertGreaterEqual(len(trace), 10)
        self.assertIn("synthesis_path", trace[0])


class TestAntiCheatAndReadiness(unittest.TestCase):
    def test_anti_cheat_rejects_loudness_only(self) -> None:
        traces = {sid: [{"factor": f"f{i}"} for i in range(11)] for sid in SAMPLE_SET}
        spectral = {
            sid: {note: {"rms_dbfs": -20.0, "peak_dbfs": -6.0} for note in NOTE_SET}
            for sid in SAMPLE_SET
        }
        ac = build_anti_cheat_checks(
            traces=traces,
            pairwise={"mean_overall_differentiation_score": 0.01, "pairwise_guitar_difference_metrics": {}},
            correlation={"max_correlation": 0.998},
            spectral_metrics=spectral,
            rms_spread_by_note={n: 0.3 for n in NOTE_SET},
            sample_ids=SAMPLE_SET,
        )
        self.assertFalse(ac.get("differences_not_only_loudness"))

    def test_readiness_labels(self) -> None:
        ok = build_readiness_emergency_demo(
            files_generated=9, expected_files=9, max_correlation=0.84, peaks_controlled=True
        )
        self.assertEqual(ok.get("current_status"), READINESS_OK)
        mod = build_readiness_emergency_demo(
            files_generated=9, expected_files=9, max_correlation=0.92, peaks_controlled=True
        )
        self.assertEqual(mod.get("current_status"), READINESS_MODERATE)
        weak = build_readiness_emergency_demo(
            files_generated=9, expected_files=9, max_correlation=0.98, peaks_controlled=True
        )
        self.assertEqual(weak.get("current_status"), READINESS_WEAK)
        fail = build_readiness_emergency_demo(
            files_generated=3, expected_files=9, max_correlation=0.5, peaks_controlled=True
        )
        self.assertEqual(fail.get("current_status"), READINESS_FAIL)


class TestEmergencyDemoGuards(unittest.TestCase):
    def test_no_stk_fem_rom(self) -> None:
        import pgsm_emergency_guitar_demo_engine as mod

        src = Path(mod.__file__).read_text(encoding="utf-8")
        self.assertNotIn("subprocess", src)
        self.assertNotIn("rom_manager", src)

    def test_no_subprocess(self) -> None:
        with patch("subprocess.run") as mock_run, patch("subprocess.Popen") as mock_popen:
            build_emergency_demo_config()
            mock_run.assert_not_called()
            mock_popen.assert_not_called()

    def test_website_default_unchanged(self) -> None:
        self.assertIsNotNone(DEFAULT_WEBSITE_STK_MODE)


if __name__ == "__main__":
    unittest.main()
