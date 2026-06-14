#!/usr/bin/env python3
"""PGSM Step 6A — reference-guided diagnostic comparison tests."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gui"))

from body_hybrid_v4_1_identity_space import STK_BODY_TRANSFER_FINAL_V1  # noqa: E402
from pgsm_step5a_limited_note_set_diagnostic_audio import (  # noqa: E402
    AUDIO_DIR as STEP5A_AUDIO_DIR,
    NOTE_FREQUENCY_HZ,
    NOTE_SET,
)
from pgsm_step5c_note_set_extended_validation import READINESS_STEP6A  # noqa: E402
from pgsm_step5b_limited_note_set_refinement import step5a_output_fingerprints  # noqa: E402
from pgsm_step6a_reference_guided_diagnostic_comparison import (  # noqa: E402
    FORBIDDEN_RECOMMENDATION_KEYWORDS,
    PGSM_STEP6A_VERSION,
    READINESS_STEP6B,
    build_pgsm_step6a_report,
    discover_reference_files,
    write_pgsm_step6a_reports,
)
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE  # noqa: E402

MAX_MODES_TEST = 100
SR = 44100


def _write_reference_wav(path: Path, note: str, duration_s: float = 2.5) -> None:
    hz = NOTE_FREQUENCY_HZ[note]
    t = np.arange(int(SR * duration_s), dtype=np.float64) / SR
    y = 0.25 * np.sin(2.0 * np.pi * hz * t) * np.exp(-t * 1.5)
    y += 0.05 * np.sin(2.0 * np.pi * hz * 2 * t) * np.exp(-t * 2.0)
    pcm = (np.clip(y, -1.0, 1.0) * 32767.0).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm.tobytes())


class TestPgsmStep6aReferenceGuidedDiagnosticComparison(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)
        self.ref_dir = self.tmp / "reference_guitar"
        self._step5a_fp = step5a_output_fingerprints(REPO)
        self._report_missing: dict | None = None
        self._report_with_ref: dict | None = None

    def tearDown(self) -> None:
        self._td.cleanup()

    def _report_missing_refs(self) -> dict:
        if self._report_missing is None:
            self._report_missing = build_pgsm_step6a_report(
                repo_root=REPO,
                reference_dirs=[self.ref_dir],
                max_modes=MAX_MODES_TEST,
            )
        return self._report_missing

    def _report_with_refs(self) -> dict:
        if self._report_with_ref is None:
            for note in ("A4",):
                _write_reference_wav(self.ref_dir / f"reference_{note}.wav", note)
            self._report_with_ref = build_pgsm_step6a_report(
                repo_root=REPO,
                reference_dirs=[self.ref_dir],
                max_modes=MAX_MODES_TEST,
            )
        return self._report_with_ref

    def test_step5c_report_loads_and_readiness_verified(self) -> None:
        report = self._report_missing_refs()
        upstream = report.get("upstream_readiness") or {}
        self.assertIsNotNone(report.get("step5c_loaded"))
        self.assertEqual(upstream.get("step5c_readiness"), READINESS_STEP6A)
        self.assertTrue(upstream.get("step5c_pass"))
        self.assertTrue(upstream.get("shared_body_ir_limitation_present"))

    def test_pgsm_wavs_exist(self) -> None:
        for note in NOTE_SET:
            self.assertTrue((STEP5A_AUDIO_DIR / f"sample_000_{note}_diagnostic.wav").is_file())

    def test_reference_discovery_handles_missing_gracefully(self) -> None:
        disc = discover_reference_files([self.ref_dir])
        self.assertFalse(disc.get("any_found"))
        self.assertEqual(len(disc.get("missing_notes") or []), 4)
        report = self._report_missing_refs()
        self.assertEqual(
            (report.get("readiness_after_step6a") or {}).get("current_status"),
            "blocked_due_to_missing_reference_audio",
        )

    def test_no_new_pgsm_wav_generated(self) -> None:
        before = step5a_output_fingerprints(REPO)
        self._report_missing_refs()
        after = step5a_output_fingerprints(REPO)
        self.assertEqual(before, after)
        self.assertTrue(self._report_missing_refs().get("no_new_pgsm_wav_generated"))

    def test_no_pgsm_wav_modified(self) -> None:
        report = self._report_with_refs()
        self.assertTrue(report.get("no_audio_modified"))
        self.assertTrue(report.get("pgsm_wav_fingerprints_unchanged"))

    def test_no_stk_integration(self) -> None:
        import pgsm_step6a_reference_guided_diagnostic_comparison as mod

        src = Path(mod.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import diagnostic_synthesis", src)
        self.assertTrue(self._report_missing_refs().get("no_stk_integration"))

    def test_no_fem_rom_subprocess(self) -> None:
        with patch("subprocess.run") as mock_run, patch("subprocess.Popen") as mock_popen:
            build_pgsm_step6a_report(
                repo_root=REPO, reference_dirs=[self.ref_dir], max_modes=MAX_MODES_TEST
            )
            mock_run.assert_not_called()
            mock_popen.assert_not_called()

    def test_website_default_unchanged(self) -> None:
        self.assertEqual(DEFAULT_WEBSITE_STK_MODE, STK_BODY_TRANSFER_FINAL_V1)
        self.assertTrue(self._report_missing_refs().get("website_default_unchanged"))

    def test_final_synthesis_ready_false(self) -> None:
        rg = self._report_missing_refs().get("readiness_after_step6a") or {}
        self.assertFalse(rg.get("final_synthesis_ready"))

    def test_multi_guitar_comparison_blocked(self) -> None:
        rg = self._report_missing_refs().get("readiness_after_step6a") or {}
        self.assertFalse(rg.get("multi_guitar_comparison_allowed"))

    def test_melody_chords_blocked(self) -> None:
        rg = self._report_missing_refs().get("readiness_after_step6a") or {}
        self.assertFalse(rg.get("melody_chord_playback_allowed"))

    def test_subjective_tuning_blocked(self) -> None:
        rg = self._report_missing_refs().get("readiness_after_step6a") or {}
        self.assertFalse(rg.get("subjective_tuning_allowed"))

    def test_comparison_diagnostic_not_validation_proof(self) -> None:
        report = self._report_missing_refs()
        self.assertIn("not validation", report.get("explicit_statement", "").lower())
        rg = report.get("readiness_after_step6a") or {}
        self.assertFalse(rg.get("validation_proof_allowed"))
        self.assertFalse(rg.get("real_guitar_equivalence_allowed"))
        self.assertTrue((report.get("objective_test_results") or {}).get("diagnostic_not_validation_proof"))

    def test_if_reference_exists_matched_notes_computed(self) -> None:
        report = self._report_with_refs()
        self.assertIn("A4", report.get("matched_notes") or [])

    def test_onset_alignment_computed(self) -> None:
        norm = self._report_with_refs().get("comparison_normalization") or {}
        self.assertIn("A4", norm)
        self.assertIn("alignment", norm["A4"])

    def test_normalization_factors_reported(self) -> None:
        norm = (self._report_with_refs().get("comparison_normalization") or {}).get("A4") or {}
        self.assertIn("pgsm_scale", norm)
        self.assertIn("reference_scale", norm)
        self.assertFalse(norm.get("loudness_matching_as_proof"))

    def test_envelope_comparison_metrics_computed(self) -> None:
        env = self._report_with_refs().get("envelope_decay_comparison") or {}
        self.assertIn("A4", env)
        self.assertIn("smoothed_envelope_correlation", env["A4"])

    def test_spectral_comparison_metrics_computed(self) -> None:
        spec = self._report_with_refs().get("spectral_comparison") or {}
        self.assertIn("A4", spec)
        self.assertIn("pgsm", spec["A4"])
        self.assertIn("reference", spec["A4"])

    def test_gap_labels_generated(self) -> None:
        gaps = self._report_with_refs().get("gap_labels") or {}
        self.assertIn("A4", gaps)
        self.assertIsInstance(gaps["A4"], list)

    def test_reference_caveats_reported(self) -> None:
        caveats = self._report_with_refs().get("reference_caveats") or {}
        self.assertTrue(caveats.get("not_validation_proof"))
        self.assertTrue(caveats.get("not_realism_proof"))

    def test_recommendations_no_forbidden_categories(self) -> None:
        for report in (self._report_missing_refs(), self._report_with_refs()):
            for rec in report.get("diagnostic_recommendations") or []:
                text = f"{rec.get('category', '')} {rec.get('reason', '')}".lower()
                for forbidden in FORBIDDEN_RECOMMENDATION_KEYWORDS:
                    self.assertNotIn(forbidden, text)

    def test_readiness_diagnostic_only_with_reference(self) -> None:
        report = self._report_with_refs()
        rg = report.get("readiness_after_step6a") or {}
        self.assertEqual(rg.get("current_status"), READINESS_STEP6B)
        self.assertFalse(rg.get("final_synthesis_ready"))
        self.assertTrue((report.get("objective_test_results") or {}).get("all_pass"))

    def test_report_files_created(self) -> None:
        write_pgsm_step6a_reports(
            repo_root=REPO,
            reference_dirs=[self.ref_dir],
            json_path=self.tmp / "step6a.json",
            md_path=self.tmp / "step6a.md",
            max_modes=MAX_MODES_TEST,
        )
        doc = json.loads((self.tmp / "step6a.json").read_text(encoding="utf-8"))
        self.assertEqual(doc["report_version"], PGSM_STEP6A_VERSION)
        self.assertIn("not validation", doc["explicit_statement"])


if __name__ == "__main__":
    unittest.main()
