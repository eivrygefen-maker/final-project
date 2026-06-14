#!/usr/bin/env python3
"""PGSM Step 4C — extended diagnostic validation tests."""
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
from pgsm_step4b_single_note_diagnostic_refinement import (  # noqa: E402
    READINESS_STEP4C,
    STEP4A_AUDIO_DIR,
)
from pgsm_step4c_single_note_extended_validation import (  # noqa: E402
    PGSM_STEP4C_VERSION,
    READINESS_STEP5A,
    build_pgsm_step4c_report,
    write_pgsm_step4c_reports,
)
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE  # noqa: E402

MAX_MODES_TEST = 100


class TestPgsmStep4cSingleNoteExtendedValidation(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)
        self._report_cache: dict | None = None

    def tearDown(self) -> None:
        self._td.cleanup()

    def _report(self) -> dict:
        if self._report_cache is None:
            self._report_cache = build_pgsm_step4c_report(
                repo_root=REPO, max_modes=MAX_MODES_TEST
            )
        return self._report_cache

    def test_step4b_report_loads(self) -> None:
        report = self._report()
        self.assertIsNotNone(report.get("step4b_loaded"))
        upstream = report.get("upstream_readiness") or {}
        self.assertEqual(upstream.get("step4b_readiness"), READINESS_STEP4C)
        self.assertTrue(upstream.get("step4b_pass"))

    def test_step4a_wav_and_stems_exist(self) -> None:
        self.assertTrue((STEP4A_AUDIO_DIR / "sample_000_A4_diagnostic.wav").is_file())
        self.assertTrue((STEP4A_AUDIO_DIR / "sample_000_A4_body_stem.wav").is_file())
        self.assertTrue((STEP4A_AUDIO_DIR / "sample_000_A4_excitation_stem.wav").is_file())

    def test_no_new_wav_generated(self) -> None:
        step4a_dir = STEP4A_AUDIO_DIR
        before = set(step4a_dir.glob("*.wav"))
        self._report()
        after = set(step4a_dir.glob("*.wav"))
        self.assertEqual(before, after)
        self.assertTrue(self._report().get("no_new_wav_generated"))

    def test_no_stk_integration(self) -> None:
        import pgsm_step4c_single_note_extended_validation as mod

        src = Path(mod.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import diagnostic_synthesis", src)
        self.assertTrue(self._report().get("no_stk_integration"))

    def test_no_fem_rom_subprocess(self) -> None:
        with patch("subprocess.run") as mock_run, patch("subprocess.Popen") as mock_popen:
            build_pgsm_step4c_report(repo_root=REPO, max_modes=MAX_MODES_TEST)
            mock_run.assert_not_called()
            mock_popen.assert_not_called()

    def test_website_default_unchanged(self) -> None:
        self.assertEqual(DEFAULT_WEBSITE_STK_MODE, STK_BODY_TRANSFER_FINAL_V1)
        self.assertTrue(self._report().get("website_default_unchanged"))

    def test_extended_envelope_metrics_computed(self) -> None:
        report = self._report()
        e = report.get("extended_envelope_metrics") or {}
        self.assertIn("attack_time_ms", e)
        self.assertIn("decay_ms", e)
        self.assertIn("cumulative_energy_final", e)

    def test_no_end_rise(self) -> None:
        report = self._report()
        self.assertTrue((report.get("extended_envelope_metrics") or {}).get("no_late_energy_rise"))

    def test_no_hard_gate(self) -> None:
        report = self._report()
        self.assertTrue((report.get("extended_envelope_metrics") or {}).get("no_hard_cut"))

    def test_no_unexplained_late_bump(self) -> None:
        report = self._report()
        self.assertTrue(
            (report.get("extended_envelope_metrics") or {}).get("no_unexplained_late_bump")
        )

    def test_extended_spectral_metrics_computed(self) -> None:
        report = self._report()
        s = report.get("extended_spectral_metrics") or {}
        self.assertIn("spectral_centroid_hz", s)
        self.assertIn("aligned_modal_peak_count", s)

    def test_no_hf_spike(self) -> None:
        report = self._report()
        self.assertTrue((report.get("extended_spectral_metrics") or {}).get("no_artificial_hf_spike"))

    def test_no_comb_echo_signature(self) -> None:
        report = self._report()
        self.assertTrue((report.get("extended_spectral_metrics") or {}).get("no_echo_comb_pattern"))

    def test_modal_peak_alignment_preserved(self) -> None:
        report = self._report()
        s = report.get("extended_spectral_metrics") or {}
        self.assertTrue(s.get("modal_peak_alignment_strong"))
        self.assertGreaterEqual(s.get("aligned_modal_peak_count", 0), 5)

    def test_time_frequency_metrics_computed(self) -> None:
        report = self._report()
        tf = report.get("time_frequency_metrics") or {}
        self.assertIn("early_window", tf)
        self.assertIn("spectral_centroid_over_time", tf)

    def test_hf_energy_decays_over_time(self) -> None:
        report = self._report()
        self.assertTrue(
            (report.get("time_frequency_metrics") or {}).get("high_frequency_decays_over_time")
        )

    def test_stem_coherence_metrics_computed(self) -> None:
        report = self._report()
        st = report.get("stem_coherence_metrics") or {}
        self.assertIn("main_vs_stem_model_correlation", st)
        self.assertIn("body_excitation_energy_ratio", st)

    def test_excitation_not_dominant_as_click(self) -> None:
        report = self._report()
        st = report.get("stem_coherence_metrics") or {}
        self.assertTrue(st.get("excitation_not_dominant_click"))

    def test_contract_compliance_passes(self) -> None:
        report = self._report()
        c = report.get("contract_compliance") or {}
        self.assertTrue(c.get("pass"))
        self.assertFalse(c.get("no_stk_integration") is False)

    def test_readiness_not_final_stk_website_multiguitar(self) -> None:
        report = self._report()
        rg = report.get("readiness_after_step4c") or {}
        self.assertFalse(rg.get("final_synthesis_ready"))
        self.assertFalse(rg.get("stk_integration_allowed"))
        self.assertFalse(rg.get("website_production_replacement_allowed"))
        self.assertFalse(rg.get("multi_guitar_comparison_allowed"))

    def test_safe_next_step_limited_note_set_only(self) -> None:
        report = self._report()
        rg = report.get("readiness_after_step4c") or {}
        self.assertEqual(rg.get("current_status"), READINESS_STEP5A)
        self.assertTrue(rg.get("step5a_limited_note_set_allowed"))
        self.assertIn("Step 5A", report.get("safe_next_step", ""))
        blocked = report.get("blocked_next_steps") or []
        self.assertIn("Final synthesis", blocked)
        self.assertIn("STK integration", blocked)

    def test_report_files_created(self) -> None:
        write_pgsm_step4c_reports(
            repo_root=REPO,
            json_path=self.tmp / "step4c.json",
            md_path=self.tmp / "step4c.md",
            max_modes=MAX_MODES_TEST,
        )
        doc = json.loads((self.tmp / "step4c.json").read_text(encoding="utf-8"))
        self.assertEqual(doc["report_version"], PGSM_STEP4C_VERSION)
        self.assertIn("does not generate new audio", doc["explicit_statement"])
        self.assertTrue(doc.get("objective_test_results", {}).get("all_pass"))


if __name__ == "__main__":
    unittest.main()
