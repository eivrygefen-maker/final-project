#!/usr/bin/env python3
"""PGSM Step 5A — limited note-set diagnostic audio tests."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gui"))

from body_hybrid_v4_1_identity_space import STK_BODY_TRANSFER_FINAL_V1  # noqa: E402
from pgsm_step4b_single_note_diagnostic_refinement import STEP4A_AUDIO_DIR  # noqa: E402
from pgsm_step4c_single_note_extended_validation import READINESS_STEP5A  # noqa: E402
from pgsm_step5a_limited_note_set_diagnostic_audio import (  # noqa: E402
    AUDIO_DIR,
    NOTE_SET,
    PGSM_STEP5A_VERSION,
    READINESS_STEP5B,
    build_pgsm_step5a_report,
    step4a_output_fingerprints,
    write_pgsm_step5a_reports,
)
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE  # noqa: E402

MAX_MODES_TEST = 100


class TestPgsmStep5aLimitedNoteSetDiagnosticAudio(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)
        self.audio_dir = self.tmp / "pgsm_step5a_limited_note_set"
        self._report_cache: dict | None = None
        self._step4a_fp = step4a_output_fingerprints(REPO)

    def tearDown(self) -> None:
        self._td.cleanup()

    def _report(self) -> dict:
        if self._report_cache is None:
            self._report_cache = build_pgsm_step5a_report(
                repo_root=REPO,
                audio_dir=self.audio_dir,
                write_wav=True,
                max_modes=MAX_MODES_TEST,
            )
        return self._report_cache

    def test_step4c_report_loads_and_readiness_verified(self) -> None:
        report = self._report()
        upstream = report.get("upstream_readiness") or {}
        self.assertIsNotNone(report.get("step4c_loaded"))
        self.assertEqual(upstream.get("step4c_readiness"), READINESS_STEP5A)
        self.assertTrue(upstream.get("step4c_pass"))

    def test_exactly_four_main_diagnostic_wavs_generated(self) -> None:
        self._report()
        mains = list(self.audio_dir.glob("*_diagnostic.wav"))
        self.assertEqual(len(mains), 4)
        for note in NOTE_SET:
            self.assertTrue((self.audio_dir / f"sample_000_{note}_diagnostic.wav").is_file())

    def test_no_wav_outside_step5a_dir_in_test_run(self) -> None:
        self._report()
        wavs = list(self.audio_dir.glob("*.wav"))
        self.assertTrue(all(p.parent.resolve() == self.audio_dir.resolve() for p in wavs))

    def test_step4a_outputs_preserved(self) -> None:
        report = self._report()
        self.assertTrue(report.get("step4a_outputs_preserved"))
        after = step4a_output_fingerprints(REPO)
        self.assertEqual(self._step4a_fp, after)
        self.assertTrue((STEP4A_AUDIO_DIR / "sample_000_A4_diagnostic.wav").is_file())

    def test_no_stk_integration(self) -> None:
        import pgsm_step5a_limited_note_set_diagnostic_audio as mod

        src = Path(mod.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import diagnostic_synthesis", src)
        self.assertFalse(self._report().get("stk_integration_allowed"))

    def test_no_fem_rom_subprocess(self) -> None:
        with patch("subprocess.run") as mock_run, patch("subprocess.Popen") as mock_popen:
            build_pgsm_step5a_report(
                repo_root=REPO,
                audio_dir=self.audio_dir,
                write_wav=True,
                max_modes=MAX_MODES_TEST,
            )
            mock_run.assert_not_called()
            mock_popen.assert_not_called()

    def test_website_default_unchanged(self) -> None:
        self.assertEqual(DEFAULT_WEBSITE_STK_MODE, STK_BODY_TRANSFER_FINAL_V1)
        self.assertTrue(self._report().get("website_default_unchanged"))

    def test_final_synthesis_ready_false(self) -> None:
        self.assertFalse(self._report().get("final_synthesis_ready"))

    def test_multi_guitar_comparison_blocked(self) -> None:
        rg = self._report().get("readiness_after_step5a") or {}
        self.assertFalse(rg.get("multi_guitar_comparison_allowed"))

    def test_melody_chords_blocked(self) -> None:
        rg = self._report().get("readiness_after_step5a") or {}
        self.assertFalse(rg.get("melody_chord_playback_allowed"))

    def test_exact_open_string_claims_blocked(self) -> None:
        contracts = self._report().get("per_note_contracts") or {}
        for note in NOTE_SET:
            self.assertFalse(contracts[note].get("exact_open_string_claim_allowed"))

    def test_per_note_contracts_exist(self) -> None:
        contracts = self._report().get("per_note_contracts") or {}
        for note in NOTE_SET:
            self.assertIn(note, contracts)
            self.assertEqual(contracts[note]["note"], note)

    def test_every_note_has_source_fallback_level(self) -> None:
        contracts = self._report().get("per_note_contracts") or {}
        for note in NOTE_SET:
            self.assertTrue(contracts[note].get("source_fallback_level"))

    def test_peak_amplitude_below_0p3_fs_every_note(self) -> None:
        metrics = self._report().get("per_note_audio_metrics") or {}
        for note in NOTE_SET:
            self.assertLessEqual(metrics[note]["peak_fs"], 0.3)

    def test_no_clipping_every_note(self) -> None:
        metrics = self._report().get("per_note_audio_metrics") or {}
        for note in NOTE_SET:
            self.assertFalse(metrics[note]["clipping"])

    def test_no_delayed_onset_every_note(self) -> None:
        metrics = self._report().get("per_note_audio_metrics") or {}
        for note in NOTE_SET:
            self.assertTrue(metrics[note]["no_delayed_onset"])

    def test_no_second_onset_every_note(self) -> None:
        metrics = self._report().get("per_note_audio_metrics") or {}
        for note in NOTE_SET:
            self.assertTrue(metrics[note]["no_second_onset"])

    def test_no_end_rise_every_note(self) -> None:
        metrics = self._report().get("per_note_audio_metrics") or {}
        for note in NOTE_SET:
            self.assertTrue(metrics[note]["no_end_rise"])

    def test_no_hard_gate_every_note(self) -> None:
        metrics = self._report().get("per_note_audio_metrics") or {}
        for note in NOTE_SET:
            self.assertTrue(metrics[note]["no_hard_gate"])

    def test_spectral_modal_checks_every_note(self) -> None:
        metrics = self._report().get("per_note_audio_metrics") or {}
        for note in NOTE_SET:
            m = metrics[note]
            self.assertIn("modal_peaks_aligned_count", m)
            self.assertTrue(m["no_artificial_hf_spike"])
            self.assertTrue(m["no_echo_comb_pattern"])

    def test_cross_note_sanity_checks_computed(self) -> None:
        cross = self._report().get("cross_note_sanity_checks") or {}
        self.assertIn("peak_fs_by_note", cross)
        self.assertIn("decay_minus_40_db_ms_by_note", cross)
        self.assertTrue(cross.get("pass"))

    def test_readiness_remains_diagnostic_only(self) -> None:
        report = self._report()
        rg = report.get("readiness_after_step5a") or {}
        self.assertEqual(rg.get("current_status"), READINESS_STEP5B)
        self.assertFalse(rg.get("final_synthesis_ready"))
        self.assertFalse(rg.get("stk_integration_allowed"))
        self.assertFalse(rg.get("subjective_tuning_allowed"))
        obj = report.get("objective_test_results") or {}
        self.assertTrue(obj.get("all_pass"))

    def test_report_files_created(self) -> None:
        write_pgsm_step5a_reports(
            repo_root=REPO,
            audio_dir=self.audio_dir,
            json_path=self.tmp / "step5a.json",
            md_path=self.tmp / "step5a.md",
            max_modes=MAX_MODES_TEST,
        )
        doc = json.loads((self.tmp / "step5a.json").read_text(encoding="utf-8"))
        self.assertEqual(doc["report_version"], PGSM_STEP5A_VERSION)
        self.assertIn("not final guitar synthesis", doc["explicit_statement"])


if __name__ == "__main__":
    unittest.main()
