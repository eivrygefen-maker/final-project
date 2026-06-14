#!/usr/bin/env python3
"""PGSM Step 3B — modal response validation tests (no audio, no FEM/ROM, no STK)."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gui"))

from body_hybrid_v4_1_identity_space import STK_BODY_TRANSFER_FINAL_V1  # noqa: E402
from pgsm_step2_1_parameter_targets import load_step_report  # noqa: E402
from pgsm_step3b_modal_response_validation import (  # noqa: E402
    PGSM_STEP3B_VERSION,
    READINESS_VALUES,
    build_pgsm_step3b_report,
    validate_string_consistency,
    write_pgsm_step3b_reports,
)
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE  # noqa: E402


class TestPgsmStep3bModalResponseValidation(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)
        self.step3a = load_step_report(
            REPO / "audio" / "debug_reports" / "pgsm_step3a_numerical_ir_testbench.json"
        )

    def tearDown(self) -> None:
        self._td.cleanup()

    def test_website_default_unchanged(self) -> None:
        self.assertEqual(DEFAULT_WEBSITE_STK_MODE, STK_BODY_TRANSFER_FINAL_V1)

    def test_report_files_created(self) -> None:
        report = write_pgsm_step3b_reports(
            repo_root=REPO,
            json_path=self.tmp / "step3b.json",
            md_path=self.tmp / "step3b.md",
            write_figures=False,
        )
        self.assertTrue((self.tmp / "step3b.json").is_file())
        self.assertTrue((self.tmp / "step3b.md").is_file())
        doc = json.loads((self.tmp / "step3b.json").read_text(encoding="utf-8"))
        self.assertEqual(doc["report_version"], PGSM_STEP3B_VERSION)
        self.assertTrue(doc["no_audio_generated"])
        self.assertTrue(doc["no_wav_generated"])
        self.assertIn("does not synthesize sound", doc["explicit_statement"])
        self.assertEqual(report["status"], "pgsm_step3b_modal_response_validation_complete")

    def test_no_wav_files_created(self) -> None:
        write_pgsm_step3b_reports(
            repo_root=REPO,
            json_path=self.tmp / "s.json",
            md_path=self.tmp / "s.md",
            write_figures=False,
        )
        self.assertEqual(list(self.tmp.rglob("*.wav")), [])

    def test_no_fem_rom_subprocess(self) -> None:
        with patch("subprocess.run") as mock_run, patch("subprocess.Popen") as mock_popen:
            build_pgsm_step3b_report(repo_root=REPO, write_figures=False)
            mock_run.assert_not_called()
            mock_popen.assert_not_called()

    def test_no_stk_integration(self) -> None:
        import pgsm_step3b_modal_response_validation as mod

        src = Path(mod.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import diagnostic_synthesis", src)
        self.assertNotIn("from diagnostic_synthesis", src)

    def test_step3a_report_loads(self) -> None:
        report = build_pgsm_step3b_report(repo_root=REPO, write_figures=False)
        self.assertIsNotNone(report.get("step3a_report_loaded"))

    def test_string_consistency_detects_unrealistic_588N(self) -> None:
        s = validate_string_consistency(self.step3a)
        self.assertTrue(s.get("unrealistic_588N_case_detected"))
        self.assertGreater(s.get("step3a_inferred_tension_N", 0), 120.0)
        self.assertFalse(s.get("open_length_tension_realistic"))

    def test_string_consistency_safer_interpretation(self) -> None:
        s = validate_string_consistency(self.step3a)
        rec = s.get("recommended_string_interpretation", "")
        self.assertTrue(
            "fretted" in rec
            or "reference_only" in rec
            or "A4_reference" in rec
        )
        self.assertGreater(len(s.get("effective_length_options_m") or []), 0)

    def test_exact_string_claims_blocked(self) -> None:
        s = validate_string_consistency(self.step3a)
        self.assertFalse(s.get("exact_string_physical_claims_allowed"))
        self.assertGreater(len(s.get("blocked_claims") or []), 0)

    def test_q_tau_validation_by_band(self) -> None:
        report = build_pgsm_step3b_report(repo_root=REPO, write_figures=False)
        qt = report["modal_q_tau_validation"]
        bands = qt.get("bands") or {}
        self.assertIn("sub_body", bands)
        self.assertIn("mid_body", bands)
        for band in bands.values():
            if band.get("mode_count", 0) > 0:
                self.assertIn("median_Q", band)
                self.assertIn("median_tau_s", band)
                self.assertIn("status", band)

    def test_admittance_scientific_notation(self) -> None:
        report = build_pgsm_step3b_report(repo_root=REPO, write_figures=False)
        adm = report["admittance_quality_validation"]
        self.assertIn("e", adm.get("max_abs_Y_sci", "").lower())
        self.assertGreater(adm.get("max_abs_Y", 0.0), 0.0)
        self.assertGreater(adm.get("dynamic_range_dB", 0.0), 0.0)
        peaks = adm.get("peaks") or []
        self.assertGreater(len(peaks), 0)
        self.assertIn("e", peaks[0].get("abs_Y_sci", "").lower())

    def test_region_contribution_flags(self) -> None:
        report = build_pgsm_step3b_report(repo_root=REPO, write_figures=False)
        reg = report["region_contribution_validation"]
        self.assertTrue(reg.get("reference_shared_modal_catalog"))
        self.assertFalse(reg.get("multi_guitar_differentiation_allowed"))
        self.assertGreater(len(reg.get("warnings") or []), 0)

    def test_decay_validation_no_end_rise(self) -> None:
        report = build_pgsm_step3b_report(repo_root=REPO, write_figures=False)
        dec = report["decay_envelope_validation"]
        self.assertTrue(dec.get("no_artificial_end_rise"))
        self.assertTrue(dec.get("no_delayed_body_event"))

    def test_calibration_warnings_include_missing_modal_data(self) -> None:
        report = build_pgsm_step3b_report(repo_root=REPO, write_figures=False)
        codes = {w["code"] for w in report.get("calibration_warnings") or []}
        self.assertIn("modal_mass_stiffness_missing", codes)
        self.assertIn("reference_shared_modal_catalog", codes)

    def test_readiness_no_musical_wav(self) -> None:
        report = build_pgsm_step3b_report(repo_root=REPO, write_figures=False)
        rg = report["readiness_after_step3b"]
        self.assertFalse(rg["musical_wav_synthesis_allowed"])
        self.assertFalse(rg["stk_integration_allowed"])
        self.assertNotEqual(rg["current_status"], "ready_for_step3c_limited_synthesis_precheck")

    def test_readiness_no_stk_integration(self) -> None:
        report = build_pgsm_step3b_report(repo_root=REPO, write_figures=False)
        self.assertFalse(report["readiness_after_step3b"]["stk_integration_allowed"])

    def test_safe_next_step_numeric_calibration_only(self) -> None:
        report = build_pgsm_step3b_report(repo_root=REPO, write_figures=False)
        nxt = report["safe_next_step"].lower()
        self.assertIn("step 3c", nxt)
        self.assertIn("calibration", nxt)
        self.assertTrue("no musical" in nxt or "no audio" in nxt)
        rg = report["readiness_after_step3b"]
        self.assertIn(rg["current_status"], READINESS_VALUES)
        self.assertEqual(rg["current_status"], "ready_for_step3c_numeric_calibration_only")


if __name__ == "__main__":
    unittest.main()
