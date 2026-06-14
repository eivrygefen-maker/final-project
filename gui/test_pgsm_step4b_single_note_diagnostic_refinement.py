#!/usr/bin/env python3
"""PGSM Step 4B — diagnostic refinement review tests."""
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
from pgsm_step4a_single_note_diagnostic_audio import READINESS_STEP4B  # noqa: E402
from pgsm_step4b_single_note_diagnostic_refinement import (  # noqa: E402
    PGSM_STEP4B_VERSION,
    READINESS_STEP4C,
    STEP4A_AUDIO_DIR,
    analyze_onset,
    analyze_spectral_modal,
    analyze_stem_balance,
    analyze_waveform_envelope,
    build_pgsm_step4b_report,
    compare_step3c_consistency,
    load_wav_mono,
    write_pgsm_step4b_reports,
)
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE  # noqa: E402

MAX_MODES_TEST = 100


class TestPgsmStep4bSingleNoteDiagnosticRefinement(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)
        self._report_cache: dict | None = None

    def tearDown(self) -> None:
        self._td.cleanup()

    def _report(self) -> dict:
        if self._report_cache is None:
            self._report_cache = build_pgsm_step4b_report(
                repo_root=REPO, max_modes=MAX_MODES_TEST
            )
        return self._report_cache

    def test_step4a_report_loads(self) -> None:
        report = self._report()
        self.assertIsNotNone(report.get("step4a_loaded"))
        verify = report.get("step4a_readiness_verification") or {}
        self.assertEqual(verify.get("readiness"), READINESS_STEP4B)

    def test_step3c_and_step3d_reports_load(self) -> None:
        report = self._report()
        self.assertIsNotNone(report.get("step3c_loaded"))
        self.assertIsNotNone(report.get("step3d_loaded"))

    def test_step4a_wav_and_stems_exist(self) -> None:
        self.assertTrue((STEP4A_AUDIO_DIR / "sample_000_A4_diagnostic.wav").is_file())
        self.assertTrue((STEP4A_AUDIO_DIR / "sample_000_A4_body_stem.wav").is_file())
        self.assertTrue((STEP4A_AUDIO_DIR / "sample_000_A4_excitation_stem.wav").is_file())

    def test_website_default_unchanged(self) -> None:
        self.assertEqual(DEFAULT_WEBSITE_STK_MODE, STK_BODY_TRANSFER_FINAL_V1)

    def test_no_stk_integration(self) -> None:
        import pgsm_step4b_single_note_diagnostic_refinement as mod

        src = Path(mod.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import diagnostic_synthesis", src)

    def test_no_fem_rom_subprocess(self) -> None:
        with patch("subprocess.run") as mock_run, patch("subprocess.Popen") as mock_popen:
            build_pgsm_step4b_report(repo_root=REPO, max_modes=MAX_MODES_TEST)
            mock_run.assert_not_called()
            mock_popen.assert_not_called()

    def test_waveform_envelope_metrics_computed(self) -> None:
        report = self._report()
        w = report.get("waveform_envelope_metrics") or {}
        self.assertIn("peak_amplitude_fs", w)
        self.assertIn("decay_ms", w)

    def test_onset_metrics_computed(self) -> None:
        report = self._report()
        o = report.get("onset_metrics") or {}
        self.assertIn("single_onset_detected", o)
        self.assertTrue(o.get("no_delayed_body_onset"))

    def test_spectral_modal_metrics_computed(self) -> None:
        report = self._report()
        s = report.get("spectral_modal_metrics") or {}
        self.assertIn("modal_peaks_aligned_count", s)
        self.assertTrue(s.get("pass"))

    def test_stem_balance_metrics_computed(self) -> None:
        report = self._report()
        st = report.get("stem_balance_metrics") or {}
        self.assertIn("body_to_excitation_energy_ratio", st)
        self.assertTrue(st.get("pass"))

    def test_step3c_consistency_computed(self) -> None:
        report = self._report()
        c = report.get("step3c_consistency_metrics") or {}
        self.assertIn("envelope_correlation_vs_step3c_ir", c)
        self.assertIn(c.get("overall_status"), ("pass", "warn", "fail"))

    def test_no_delayed_body_onset(self) -> None:
        report = self._report()
        self.assertTrue((report.get("onset_metrics") or {}).get("no_delayed_body_onset"))

    def test_no_end_rise(self) -> None:
        report = self._report()
        self.assertTrue((report.get("waveform_envelope_metrics") or {}).get("no_end_rise"))

    def test_no_hard_gate(self) -> None:
        report = self._report()
        self.assertTrue((report.get("waveform_envelope_metrics") or {}).get("no_hard_gate"))

    def test_no_second_onset(self) -> None:
        report = self._report()
        self.assertTrue((report.get("onset_metrics") or {}).get("no_second_pluck_event"))

    def test_artifact_guard_no_forbidden_layers(self) -> None:
        report = self._report()
        a = report.get("artifact_guard_results") or {}
        self.assertFalse(a.get("reverb_added"))
        self.assertFalse(a.get("body_tail_stem_added"))
        self.assertFalse(a.get("helmholtz_echo_added"))
        self.assertTrue(a.get("pass"))

    def test_no_corrected_candidate_when_checks_pass(self) -> None:
        report = self._report()
        self.assertFalse(report.get("corrected_candidate_generated"))
        self.assertTrue(report.get("step4a_outputs_preserved"))

    def test_no_new_wav_in_step4b_dir_by_default(self) -> None:
        step4b_dir = REPO / "audio" / "pgsm_step4b_diagnostic_audio"
        before = set(step4b_dir.glob("*.wav")) if step4b_dir.is_dir() else set()
        self._report()
        after = set(step4b_dir.glob("*.wav")) if step4b_dir.is_dir() else set()
        self.assertEqual(before, after)

    def test_readiness_not_final_stk_website_multiguitar(self) -> None:
        report = self._report()
        rg = report.get("readiness_after_step4b") or {}
        self.assertFalse(rg.get("final_synthesis_ready"))
        self.assertFalse(rg.get("stk_integration_allowed"))
        self.assertFalse(rg.get("website_production_replacement_allowed"))
        self.assertFalse(rg.get("multi_guitar_comparison_allowed"))

    def test_readiness_step4c_extended_validation(self) -> None:
        report = self._report()
        rg = report.get("readiness_after_step4b") or {}
        self.assertEqual(rg.get("current_status"), READINESS_STEP4C)

    def test_objective_not_listening_based(self) -> None:
        report = self._report()
        self.assertFalse(report.get("listening_based_tuning_used"))
        self.assertTrue(report.get("objective_analysis_only"))

    def test_report_files_created(self) -> None:
        write_pgsm_step4b_reports(
            repo_root=REPO,
            json_path=self.tmp / "step4b.json",
            md_path=self.tmp / "step4b.md",
            max_modes=MAX_MODES_TEST,
        )
        doc = json.loads((self.tmp / "step4b.json").read_text(encoding="utf-8"))
        self.assertEqual(doc["report_version"], PGSM_STEP4B_VERSION)
        self.assertIn("diagnostic refinement analysis only", doc["explicit_statement"])


if __name__ == "__main__":
    unittest.main()
