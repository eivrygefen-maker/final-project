#!/usr/bin/env python3
"""PGSM Step 5C — note-set extended validation tests."""
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
from pgsm_step5a_limited_note_set_diagnostic_audio import (  # noqa: E402
    AUDIO_DIR as STEP5A_AUDIO_DIR,
    NOTE_SET,
)
from pgsm_step5b_limited_note_set_refinement import READINESS_STEP5C  # noqa: E402
from pgsm_step5b_limited_note_set_refinement import step5a_output_fingerprints  # noqa: E402
from pgsm_step5c_note_set_extended_validation import (  # noqa: E402
    PGSM_STEP5C_VERSION,
    READINESS_STEP6A,
    build_pgsm_step5c_report,
    write_pgsm_step5c_reports,
)
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE  # noqa: E402

MAX_MODES_TEST = 100


class TestPgsmStep5cNoteSetExtendedValidation(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)
        self._report_cache: dict | None = None
        self._step5a_fp = step5a_output_fingerprints(REPO)

    def tearDown(self) -> None:
        self._td.cleanup()

    def _report(self) -> dict:
        if self._report_cache is None:
            self._report_cache = build_pgsm_step5c_report(
                repo_root=REPO, max_modes=MAX_MODES_TEST
            )
        return self._report_cache

    def test_step5b_report_loads(self) -> None:
        report = self._report()
        self.assertIsNotNone(report.get("step5b_loaded"))

    def test_step5b_readiness_verified(self) -> None:
        upstream = self._report().get("upstream_readiness") or {}
        self.assertEqual(upstream.get("step5b_readiness"), READINESS_STEP5C)
        self.assertTrue(upstream.get("step5b_pass"))
        self.assertFalse(upstream.get("step5b_corrected_candidate_generated"))

    def test_exactly_four_notes_analyzed(self) -> None:
        report = self._report()
        self.assertEqual(report.get("analyzed_note_set"), list(NOTE_SET))
        self.assertTrue((report.get("upstream_readiness") or {}).get("four_notes_a2_a3_a4_e5"))

    def test_all_main_wavs_and_stems_exist(self) -> None:
        for note in NOTE_SET:
            self.assertTrue((STEP5A_AUDIO_DIR / f"sample_000_{note}_diagnostic.wav").is_file())
            self.assertTrue((STEP5A_AUDIO_DIR / f"sample_000_{note}_body_stem.wav").is_file())
            self.assertTrue((STEP5A_AUDIO_DIR / f"sample_000_{note}_excitation_stem.wav").is_file())

    def test_no_new_wav_generated(self) -> None:
        before = step5a_output_fingerprints(REPO)
        self._report()
        after = step5a_output_fingerprints(REPO)
        self.assertEqual(before, after)
        self.assertTrue(self._report().get("no_new_wav_generated"))

    def test_no_stk_integration(self) -> None:
        import pgsm_step5c_note_set_extended_validation as mod

        src = Path(mod.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import diagnostic_synthesis", src)
        self.assertTrue(self._report().get("no_stk_integration"))

    def test_no_fem_rom_subprocess(self) -> None:
        with patch("subprocess.run") as mock_run, patch("subprocess.Popen") as mock_popen:
            build_pgsm_step5c_report(repo_root=REPO, max_modes=MAX_MODES_TEST)
            mock_run.assert_not_called()
            mock_popen.assert_not_called()

    def test_website_default_unchanged(self) -> None:
        self.assertEqual(DEFAULT_WEBSITE_STK_MODE, STK_BODY_TRANSFER_FINAL_V1)
        self.assertTrue(self._report().get("website_default_unchanged"))

    def test_final_synthesis_ready_false(self) -> None:
        rg = self._report().get("readiness_after_step5c") or {}
        self.assertFalse(rg.get("final_synthesis_ready"))

    def test_multi_guitar_comparison_blocked(self) -> None:
        rg = self._report().get("readiness_after_step5c") or {}
        self.assertFalse(rg.get("multi_guitar_comparison_allowed"))

    def test_melody_chords_blocked(self) -> None:
        rg = self._report().get("readiness_after_step5c") or {}
        self.assertFalse(rg.get("melody_chord_playback_allowed"))

    def test_subjective_tuning_blocked(self) -> None:
        rg = self._report().get("readiness_after_step5c") or {}
        self.assertFalse(rg.get("subjective_tuning_allowed"))

    def test_cross_note_envelope_metrics_computed(self) -> None:
        env = self._report().get("cross_note_envelope_metrics") or {}
        self.assertIn("per_note", env)
        self.assertTrue(env.get("body_stem_tail_consistent_all_notes"))

    def test_decay_separated_raw_smoothed_body_excitation(self) -> None:
        per = (self._report().get("cross_note_envelope_metrics") or {}).get("per_note") or {}
        for note in NOTE_SET:
            e = per[note]
            self.assertIsNotNone(e.get("main_raw_minus_40_db_ms"))
            self.assertIsNotNone(e.get("main_smoothed_minus_40_db_ms"))
            self.assertIsNotNone(e.get("body_stem_minus_40_db_ms"))
            self.assertIsNotNone(e.get("excitation_minus_40_db_ms"))
            self.assertLess(float(e["main_raw_minus_40_db_ms"]), 50.0)
            self.assertGreater(float(e["body_stem_minus_40_db_ms"]), 50.0)

    def test_no_end_rise_every_note(self) -> None:
        per = (self._report().get("cross_note_envelope_metrics") or {}).get("per_note") or {}
        for note in NOTE_SET:
            self.assertTrue(per[note]["no_end_rise"])

    def test_no_hard_gate_every_note(self) -> None:
        per = (self._report().get("cross_note_envelope_metrics") or {}).get("per_note") or {}
        for note in NOTE_SET:
            self.assertTrue(per[note]["no_hard_gate"])

    def test_no_second_onset_every_note(self) -> None:
        per = (self._report().get("cross_note_envelope_metrics") or {}).get("per_note") or {}
        for note in NOTE_SET:
            self.assertTrue(per[note]["no_second_onset"])

    def test_cross_note_spectral_metrics_computed(self) -> None:
        spec = self._report().get("cross_note_spectral_metrics") or {}
        self.assertIn("spectral_centroid_hz_by_note", spec)
        self.assertTrue(spec.get("modal_alignment_strong_all_notes"))

    def test_harmonic_shaping_metrics_computed(self) -> None:
        harm = self._report().get("harmonic_shaping_metrics") or {}
        self.assertIn("H1_energy_fraction_by_note", harm)
        self.assertTrue(harm.get("pass"))

    def test_modal_alignment_preserved(self) -> None:
        spec = self._report().get("cross_note_spectral_metrics") or {}
        aligned = spec.get("aligned_modal_peak_count_by_note") or {}
        for note in NOTE_SET:
            self.assertGreaterEqual(aligned.get(note, 0), 5)

    def test_no_hf_spike(self) -> None:
        self.assertTrue(
            (self._report().get("cross_note_spectral_metrics") or {}).get("no_hf_spike_all_notes")
        )

    def test_no_comb_echo_signature(self) -> None:
        self.assertTrue(
            (self._report().get("cross_note_spectral_metrics") or {}).get("no_comb_echo_all_notes")
        )

    def test_shared_body_ir_limitation_explicitly_reported(self) -> None:
        shared = self._report().get("shared_body_ir_limitation") or {}
        self.assertTrue(shared.get("shared_modal_ir_across_all_notes"))
        self.assertTrue(shared.get("is_limitation_not_failure"))
        self.assertTrue(shared.get("real_guitar_equivalence_claim_blocked"))
        self.assertIn("explicit_label", shared)

    def test_readiness_reference_guided_only_not_proof(self) -> None:
        report = self._report()
        rg = report.get("readiness_after_step5c") or {}
        self.assertEqual(rg.get("current_status"), READINESS_STEP6A)
        self.assertFalse(rg.get("real_guitar_validation_proof_allowed"))
        self.assertTrue(rg.get("step6a_reference_guided_diagnostic_comparison_allowed"))
        self.assertIn("not validation", report.get("safe_next_step", "").lower())
        self.assertTrue(report.get("objective_test_results", {}).get("real_guitar_equivalence_blocked"))

    def test_report_files_created(self) -> None:
        write_pgsm_step5c_reports(
            repo_root=REPO,
            json_path=self.tmp / "step5c.json",
            md_path=self.tmp / "step5c.md",
            max_modes=MAX_MODES_TEST,
        )
        doc = json.loads((self.tmp / "step5c.json").read_text(encoding="utf-8"))
        self.assertEqual(doc["report_version"], PGSM_STEP5C_VERSION)
        self.assertIn("does not prove realism", doc["explicit_statement"])


if __name__ == "__main__":
    unittest.main()
