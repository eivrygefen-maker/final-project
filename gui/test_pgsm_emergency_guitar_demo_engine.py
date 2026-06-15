#!/usr/bin/env python3
"""Lightweight tests for PGSM emergency guitar demo engine (no synthesis runtime)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gui"))

from pgsm_emergency_guitar_demo_engine import (  # noqa: E402
    NOTE_SET,
    READINESS_FAIL,
    READINESS_OK,
    READINESS_WEAK,
    SAMPLE_SET,
    build_anti_cheat_checks,
    build_emergency_demo_config,
    build_readiness_emergency_demo,
    compute_demo_modifiers,
    demo_wav_filename,
    extract_physical_parameters,
)
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE  # noqa: E402
from stk_v6_2_audit_features import load_audit_report  # noqa: E402


class TestEmergencyDemoConfig(unittest.TestCase):
    def test_build_emergency_demo_config(self) -> None:
        cfg = build_emergency_demo_config()
        self.assertEqual(cfg.get("validation_mode"), "emergency_demo")
        self.assertEqual(cfg.get("sample_set"), list(SAMPLE_SET))
        self.assertEqual(cfg.get("note_set"), list(NOTE_SET))
        self.assertEqual(cfg.get("duration_s"), 2.5)

    def test_demo_wav_filename_pattern(self) -> None:
        self.assertEqual(demo_wav_filename("sample_000", "A2"), "sample_000_A2_guitar_demo.wav")
        self.assertEqual(demo_wav_filename("sample_002", "E5"), "sample_002_E5_guitar_demo.wav")

    def test_expected_file_count(self) -> None:
        self.assertEqual(len(SAMPLE_SET) * len(NOTE_SET), 9)


class TestPhysicalTrace(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._audit = load_audit_report()

    def test_physical_parameters_per_sample(self) -> None:
        for sid in SAMPLE_SET:
            row = extract_physical_parameters(sid, self._audit)
            self.assertEqual(row.get("sample_id"), sid)
            self.assertIn("bridge_mobility_proxy", row)
            self.assertIn("body_depth_m", row)

    def test_differentiation_trace_and_modifiers(self) -> None:
        ref = extract_physical_parameters("sample_000", self._audit)
        for sid in SAMPLE_SET:
            phys = extract_physical_parameters(sid, self._audit)
            mods, trace, _exag = compute_demo_modifiers(phys, ref)
            self.assertGreaterEqual(len(trace), 4)
            self.assertIn("bridge_coupling_strength", mods)
            self.assertIn("brightness_scale", mods)


class TestAntiCheatAndReadiness(unittest.TestCase):
    def test_anti_cheat_logic_exists(self) -> None:
        traces = {
            sid: [{"driver": "bridge_mobility_proxy", "effect": "x"} for _ in range(5)]
            for sid in SAMPLE_SET
        }
        spectral = {
            sid: {note: {"rms_dbfs": -20.0, "crest_factor": 4.0} for note in NOTE_SET}
            for sid in SAMPLE_SET
        }
        pairwise = {"mean_overall_differentiation_score": 0.06}
        ac = build_anti_cheat_checks(
            traces=traces,
            pairwise=pairwise,
            spectral_metrics=spectral,
            sample_ids=SAMPLE_SET,
        )
        self.assertTrue(ac.get("physical_driver_trace_per_sample"))
        self.assertTrue(ac.get("no_randomization"))
        self.assertIn("differences_not_only_loudness", ac)

    def test_readiness_labels(self) -> None:
        ok = build_readiness_emergency_demo(
            files_generated=9, expected_files=9, mean_differentiation=0.08
        )
        self.assertEqual(ok.get("current_status"), READINESS_OK)
        weak = build_readiness_emergency_demo(
            files_generated=9, expected_files=9, mean_differentiation=0.01
        )
        self.assertEqual(weak.get("current_status"), READINESS_WEAK)
        fail = build_readiness_emergency_demo(
            files_generated=3, expected_files=9, mean_differentiation=0.2
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
            self.assertEqual(cfg.get("validation_mode"), "emergency_demo")
            mock_run.assert_not_called()
            mock_popen.assert_not_called()

    def test_website_default_unchanged(self) -> None:
        self.assertEqual(DEFAULT_WEBSITE_STK_MODE, DEFAULT_WEBSITE_STK_MODE)


if __name__ == "__main__":
    unittest.main()
