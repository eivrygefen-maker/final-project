#!/usr/bin/env python3
"""PGSM Step 5B — limited note-set diagnostic refinement tests."""
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
    READINESS_STEP5B,
    step4a_output_fingerprints,
)
from pgsm_step5b_limited_note_set_refinement import (  # noqa: E402
    PGSM_STEP5B_VERSION,
    READINESS_STEP5C,
    build_pgsm_step5b_report,
    interpret_per_note_decay,
    step5a_output_fingerprints,
    write_pgsm_step5b_reports,
)
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE  # noqa: E402

MAX_MODES_TEST = 100


class TestPgsmStep5bLimitedNoteSetRefinement(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)
        self._report_cache: dict | None = None
        self._step5a_fp = step5a_output_fingerprints(REPO)
        self._step4a_fp = step4a_output_fingerprints(REPO)

    def tearDown(self) -> None:
        self._td.cleanup()

    def _report(self) -> dict:
        if self._report_cache is None:
            self._report_cache = build_pgsm_step5b_report(
                repo_root=REPO, max_modes=MAX_MODES_TEST
            )
        return self._report_cache

    def test_step5a_report_loads(self) -> None:
        report = self._report()
        self.assertIsNotNone(report.get("step5a_loaded"))
        verify = report.get("step5a_readiness_verification") or {}
        self.assertEqual(verify.get("step5a_readiness"), READINESS_STEP5B)
        self.assertTrue(verify.get("step5a_pass"))

    def test_exactly_four_notes_analyzed(self) -> None:
        report = self._report()
        self.assertEqual(report.get("analyzed_note_set"), list(NOTE_SET))
        self.assertEqual(
            (report.get("step5a_readiness_verification") or {}).get("main_wav_count"), 4
        )

    def test_all_main_wavs_and_stems_exist(self) -> None:
        for note in NOTE_SET:
            self.assertTrue((STEP5A_AUDIO_DIR / f"sample_000_{note}_diagnostic.wav").is_file())
            self.assertTrue((STEP5A_AUDIO_DIR / f"sample_000_{note}_body_stem.wav").is_file())
            self.assertTrue((STEP5A_AUDIO_DIR / f"sample_000_{note}_excitation_stem.wav").is_file())

    def test_no_new_wav_generated_by_default(self) -> None:
        before = step5a_output_fingerprints(REPO)
        self._report()
        after = step5a_output_fingerprints(REPO)
        self.assertEqual(before, after)
        self.assertFalse(self._report().get("corrected_candidate_generated"))

    def test_no_wav_outside_allowed_folders(self) -> None:
        candidates = REPO / "audio" / "pgsm_step5b_limited_note_set_candidates"
        before_c = set(candidates.glob("*.wav")) if candidates.is_dir() else set()
        self._report()
        after_c = set(candidates.glob("*.wav")) if candidates.is_dir() else set()
        self.assertEqual(before_c, after_c)

    def test_no_stk_integration(self) -> None:
        import pgsm_step5b_limited_note_set_refinement as mod

        src = Path(mod.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import diagnostic_synthesis", src)
        self.assertTrue(self._report().get("no_stk_integration"))

    def test_no_fem_rom_subprocess(self) -> None:
        with patch("subprocess.run") as mock_run, patch("subprocess.Popen") as mock_popen:
            build_pgsm_step5b_report(repo_root=REPO, max_modes=MAX_MODES_TEST)
            mock_run.assert_not_called()
            mock_popen.assert_not_called()

    def test_website_default_unchanged(self) -> None:
        self.assertEqual(DEFAULT_WEBSITE_STK_MODE, STK_BODY_TRANSFER_FINAL_V1)
        self.assertTrue(self._report().get("website_default_unchanged"))

    def test_final_synthesis_ready_false(self) -> None:
        rg = self._report().get("readiness_after_step5b") or {}
        self.assertFalse(rg.get("final_synthesis_ready"))

    def test_multi_guitar_comparison_blocked(self) -> None:
        rg = self._report().get("readiness_after_step5b") or {}
        self.assertFalse(rg.get("multi_guitar_comparison_allowed"))

    def test_melody_chords_blocked(self) -> None:
        rg = self._report().get("readiness_after_step5b") or {}
        self.assertFalse(rg.get("melody_chord_playback_allowed"))

    def test_subjective_tuning_blocked(self) -> None:
        report = self._report()
        self.assertFalse(report.get("listening_based_tuning_used"))
        rg = report.get("readiness_after_step5b") or {}
        self.assertFalse(rg.get("subjective_tuning_allowed"))

    def test_per_note_stem_separated_decay_metrics_computed(self) -> None:
        stem = self._report().get("per_note_stem_decay_metrics") or {}
        for note in NOTE_SET:
            block = stem[note]
            self.assertIn("main", block)
            self.assertIn("body_stem", block)
            self.assertIn("excitation_stem", block)

    def test_main_body_excitation_decay_separated(self) -> None:
        stem = self._report().get("per_note_stem_decay_metrics") or {}
        for note in NOTE_SET:
            main_d = stem[note]["main"]["decay_ms"]["minus_40_dB"]
            body_d = stem[note]["body_stem"]["decay_ms"]["minus_40_dB"]
            exc_d = stem[note]["excitation_stem"]["decay_ms"]["minus_40_dB"]
            self.assertIsNotNone(main_d)
            self.assertIsNotNone(body_d)
            self.assertIsNotNone(exc_d)

    def test_short_main_decay_interpreted_not_blindly_accepted(self) -> None:
        decay = self._report().get("decay_interpretation") or {}
        self.assertTrue(decay.get("step5a_raw_main_decay_reconciled"))
        self.assertFalse(decay.get("any_modal_tail_issue"))
        per = decay.get("per_note") or {}
        for note in NOTE_SET:
            interp = per[note]
            self.assertTrue(interp.get("main_decay_short_raw_envelope"))
            self.assertTrue(interp.get("body_modal_tail_long_enough"))
            self.assertIn(
                interp.get("classification"),
                (
                    "force_dominant_main_envelope_not_body_failure",
                    "main_and_body_decay_consistent",
                ),
            )

    def test_body_stem_decay_evaluated_separately(self) -> None:
        stem = self._report().get("per_note_stem_decay_metrics") or {}
        for note in NOTE_SET:
            body_d40 = stem[note]["body_stem"]["decay_ms"]["minus_40_dB"]
            main_raw = stem[note]["main"]["raw_envelope_decay_ms"]["minus_40_dB"]
            self.assertGreater(float(body_d40), 50.0)
            self.assertLess(float(main_raw), 50.0)

    def test_excitation_dominance_check_computed(self) -> None:
        stem = self._report().get("per_note_stem_decay_metrics") or {}
        for note in NOTE_SET:
            interp = stem[note]["decay_interpretation"]
            self.assertIn("excitation_dominance_issue", interp)
            self.assertFalse(interp["excitation_dominance_issue"])

    def test_cross_note_consistency_computed(self) -> None:
        cross = self._report().get("cross_note_consistency") or {}
        self.assertIn("main_peak_fs_by_note", cross)
        self.assertIn("body_minus_40_db_ms_by_note", cross)
        self.assertTrue(cross.get("pass"))

    def test_spectral_harmonic_diagnostics_computed(self) -> None:
        spec = self._report().get("spectral_harmonic_diagnostics") or {}
        for note in NOTE_SET:
            self.assertIn("harmonic_energy_fraction", spec[note])
            self.assertIn("note_dependent_spectral_shaping_visible", spec[note])

    def test_shared_body_ir_limitation_explicitly_reported(self) -> None:
        shared = self._report().get("shared_body_ir_limitation") or {}
        self.assertTrue(shared.get("shared_modal_ir_across_all_notes"))
        self.assertTrue(shared.get("note_realism_claim_blocked"))
        self.assertIn("explicit_label", shared)

    def test_artifact_guard_passes(self) -> None:
        artifact = self._report().get("artifact_guard_results") or {}
        self.assertTrue(artifact.get("pass"))
        self.assertTrue(artifact.get("no_listening_based_acceptance"))

    def test_step5a_and_step4a_outputs_preserved(self) -> None:
        report = self._report()
        self.assertTrue(report.get("step5a_outputs_preserved"))
        self.assertTrue(report.get("step4a_outputs_preserved"))
        self.assertEqual(self._step5a_fp, step5a_output_fingerprints(REPO))
        self.assertEqual(self._step4a_fp, step4a_output_fingerprints(REPO))

    def test_readiness_remains_diagnostic_only(self) -> None:
        report = self._report()
        rg = report.get("readiness_after_step5b") or {}
        self.assertEqual(rg.get("current_status"), READINESS_STEP5C)
        self.assertFalse(rg.get("final_synthesis_ready"))
        self.assertFalse(rg.get("stk_integration_allowed"))
        obj = report.get("objective_test_results") or {}
        self.assertTrue(obj.get("all_pass"))

    def test_interpret_decay_unit_logic(self) -> None:
        main = {
            "decay_ms": {"minus_40_dB": 115.0},
            "raw_envelope_decay_ms": {"minus_40_dB": 3.0},
        }
        body = {"decay_ms": {"minus_40_dB": 120.0}}
        exc = {"decay_ms": {"minus_40_dB": 2.0}}
        stems = {"excitation_not_dominating_click": True}
        out = interpret_per_note_decay(main, body, exc, stems=stems, step5a_main_decay_ms=3.0)
        self.assertEqual(out["classification"], "force_dominant_main_envelope_not_body_failure")
        self.assertTrue(out["main_decay_short_raw_envelope"])
        self.assertTrue(out["pass"])

    def test_report_files_created(self) -> None:
        write_pgsm_step5b_reports(
            repo_root=REPO,
            json_path=self.tmp / "step5b.json",
            md_path=self.tmp / "step5b.md",
            max_modes=MAX_MODES_TEST,
        )
        doc = json.loads((self.tmp / "step5b.json").read_text(encoding="utf-8"))
        self.assertEqual(doc["report_version"], PGSM_STEP5B_VERSION)
        self.assertIn("not final guitar synthesis", doc["explicit_statement"])


if __name__ == "__main__":
    unittest.main()
