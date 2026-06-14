#!/usr/bin/env python3
"""PGSM Step 5D — audible diagnostic render repair tests."""
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
from pgsm_step4b_single_note_diagnostic_refinement import load_wav_mono  # noqa: E402
from pgsm_step5a_limited_note_set_diagnostic_audio import (  # noqa: E402
    AUDIO_DIR as STEP5A_AUDIO_DIR,
    NOTE_SET,
    step4a_output_fingerprints,
)
from pgsm_step5b_limited_note_set_refinement import step5a_output_fingerprints  # noqa: E402
from pgsm_step5c_note_set_extended_validation import READINESS_STEP6A  # noqa: E402
from pgsm_step5d_audible_diagnostic_render_repair import (  # noqa: E402
    INAUDIBLE_RMS_DBFS,
    PEAK_CAP_DBFS,
    PGSM_STEP5D_VERSION,
    READINESS_AFTER,
    RENDER_DIR,
    TARGET_RMS_DBFS_MAX,
    TARGET_RMS_DBFS_MIN,
    analyze_audibility,
    apply_listening_render,
    build_pgsm_step5d_report,
    write_pgsm_step5d_reports,
)
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE  # noqa: E402


class TestPgsmStep5dAudibleDiagnosticRenderRepair(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)
        self._report_cache: dict | None = None
        self._step5a_fp = step5a_output_fingerprints(REPO)
        self._step4a_fp = step4a_output_fingerprints(REPO)

    def tearDown(self) -> None:
        self._td.cleanup()

    def _report(self, *, write_wav: bool = False) -> dict:
        if self._report_cache is None:
            self._report_cache = build_pgsm_step5d_report(
                repo_root=REPO,
                render_dir=self.tmp / "renders",
                write_wav=write_wav,
            )
        return self._report_cache

    def test_step5c_report_loads(self) -> None:
        report = self._report()
        self.assertEqual(report.get("step5c_loaded"), "pgsm_step5c_note_set_extended_validation_v1")
        obj = report.get("objective_test_results") or {}
        self.assertTrue(obj.get("step5c_loaded"))

    def test_original_step5a_wavs_exist(self) -> None:
        for note in NOTE_SET:
            self.assertTrue((STEP5A_AUDIO_DIR / f"sample_000_{note}_diagnostic.wav").is_file())
            self.assertTrue((STEP5A_AUDIO_DIR / f"sample_000_{note}_body_stem.wav").is_file())
            self.assertTrue((STEP5A_AUDIO_DIR / f"sample_000_{note}_excitation_stem.wav").is_file())

    def test_original_audibility_metrics_computed(self) -> None:
        report = self._report()
        metrics = report.get("original_audibility_metrics") or {}
        for note in NOTE_SET:
            main = (metrics.get(note) or {}).get("main") or {}
            self.assertIn("peak_dbfs", main)
            self.assertIn("rms_dbfs", main)
            self.assertIn("crest_factor", main)
            self.assertIn("active_duration_above_1pct_peak_ms", main)

    def test_inaudible_rms_detected_for_originals(self) -> None:
        report = self._report()
        obj = report.get("objective_test_results") or {}
        self.assertTrue(obj.get("original_inaudible_detected"))
        for note in NOTE_SET:
            main = (report.get("original_audibility_metrics") or {}).get(note, {}).get("main") or {}
            self.assertTrue(main.get("inaudible_rms"))
            self.assertLess(main.get("rms_dbfs"), INAUDIBLE_RMS_DBFS)

    def test_listening_renders_generated_exactly_four_files(self) -> None:
        render_dir = self.tmp / "renders_out"
        report = build_pgsm_step5d_report(
            repo_root=REPO, render_dir=render_dir, write_wav=True
        )
        files = report.get("listening_render_files") or {}
        self.assertEqual(len(files), 4)
        for note in NOTE_SET:
            path = render_dir / f"sample_000_{note}_listening_diagnostic.wav"
            self.assertTrue(path.is_file(), msg=f"missing {path}")

    def test_original_files_preserved_by_sha256(self) -> None:
        before5a = step5a_output_fingerprints(REPO)
        before4a = step4a_output_fingerprints(REPO)
        build_pgsm_step5d_report(
            repo_root=REPO, render_dir=self.tmp / "fp_test", write_wav=True
        )
        self.assertEqual(before5a, step5a_output_fingerprints(REPO))
        self.assertEqual(before4a, step4a_output_fingerprints(REPO))
        report = build_pgsm_step5d_report(
            repo_root=REPO, render_dir=self.tmp / "fp_test2", write_wav=False
        )
        self.assertTrue(report.get("original_file_fingerprints_preserved"))

    def test_no_stk_integration(self) -> None:
        import pgsm_step5d_audible_diagnostic_render_repair as mod

        src = Path(mod.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import diagnostic_synthesis", src)
        self.assertTrue(self._report().get("no_stk_integration"))

    def test_no_fem_rom_subprocess(self) -> None:
        with patch("subprocess.run") as mock_run, patch("subprocess.Popen") as mock_popen:
            build_pgsm_step5d_report(
                repo_root=REPO, render_dir=self.tmp / "subproc", write_wav=False
            )
            mock_run.assert_not_called()
            mock_popen.assert_not_called()

    def test_website_default_unchanged(self) -> None:
        report = self._report()
        self.assertEqual(report.get("website_default"), DEFAULT_WEBSITE_STK_MODE)
        self.assertEqual(report.get("website_default"), STK_BODY_TRANSFER_FINAL_V1)
        self.assertTrue(report.get("website_default_unchanged"))

    def test_rms_target_reached_for_listening_renders(self) -> None:
        report = self._report()
        val = report.get("render_validation") or {}
        for note in NOTE_SET:
            v = val.get(note) or {}
            self.assertTrue(v.get("rms_in_target_range"), msg=f"{note} rms={v.get('rms_dbfs')}")
            self.assertGreaterEqual(v.get("rms_dbfs"), TARGET_RMS_DBFS_MIN)
            self.assertLessEqual(v.get("rms_dbfs"), TARGET_RMS_DBFS_MAX)

    def test_peak_below_minus_1_dbfs(self) -> None:
        report = self._report()
        for note in NOTE_SET:
            v = (report.get("render_validation") or {}).get(note) or {}
            self.assertTrue(v.get("peak_below_minus_1_dbfs"))
            self.assertLessEqual(v.get("peak_dbfs"), PEAK_CAP_DBFS + 0.01)

    def test_no_clipping(self) -> None:
        report = self._report()
        for note in NOTE_SET:
            v = (report.get("render_validation") or {}).get(note) or {}
            self.assertTrue(v.get("no_clipping"))

    def test_no_second_onset(self) -> None:
        report = self._report()
        for note in NOTE_SET:
            v = (report.get("render_validation") or {}).get(note) or {}
            self.assertTrue(v.get("no_second_onset"))

    def test_no_end_rise(self) -> None:
        report = self._report()
        for note in NOTE_SET:
            v = (report.get("render_validation") or {}).get(note) or {}
            self.assertTrue(v.get("no_end_rise"))

    def test_no_hard_gate(self) -> None:
        report = self._report()
        for note in NOTE_SET:
            v = (report.get("render_validation") or {}).get(note) or {}
            self.assertTrue(v.get("no_hard_gate"))

    def test_no_reverb_echo_body_tail_added(self) -> None:
        report = self._report()
        art = report.get("artifact_guard_results") or {}
        self.assertTrue(art.get("no_reverb"))
        self.assertTrue(art.get("no_echo"))
        self.assertTrue(art.get("no_body_tail_layer"))
        self.assertTrue(art.get("pass"))

    def test_gain_reported_separately_from_physics(self) -> None:
        report = self._report()
        gain = report.get("gain_applied_db_by_note") or {}
        details = report.get("render_gain_details") or {}
        self.assertEqual(len(gain), 4)
        for note in NOTE_SET:
            self.assertIn(note, gain)
            d = details.get(note) or {}
            self.assertTrue(d.get("gain_separate_from_physics"))
            self.assertFalse(d.get("physics_changed"))
        obj = report.get("objective_test_results") or {}
        self.assertTrue(obj.get("gain_reported_separately"))

    def test_no_physics_parameters_changed(self) -> None:
        report = self._report()
        self.assertTrue(report.get("no_physics_changed"))
        art = report.get("artifact_guard_results") or {}
        self.assertTrue(art.get("no_physics_change"))
        self.assertTrue(art.get("no_decay_stretch"))

    def test_readiness_remains_diagnostic_only(self) -> None:
        report = self._report()
        rg = report.get("readiness_after_step5d") or {}
        self.assertFalse(rg.get("final_synthesis_ready"))
        self.assertFalse(rg.get("stk_integration_allowed"))
        self.assertFalse(rg.get("website_production_replacement_allowed"))
        self.assertFalse(rg.get("multi_guitar_comparison_allowed"))
        self.assertFalse(rg.get("melody_chord_playback_allowed"))
        self.assertFalse(rg.get("subjective_tuning_allowed"))
        self.assertFalse(rg.get("real_guitar_equivalence_allowed"))

    def test_all_renders_pass_objective_validation(self) -> None:
        report = self._report()
        obj = report.get("objective_test_results") or {}
        self.assertTrue(obj.get("all_renders_pass"))
        self.assertTrue(obj.get("artifact_guard_pass"))
        self.assertTrue(obj.get("all_pass"))

    def test_readiness_after_pass(self) -> None:
        report = self._report()
        rg = report.get("readiness_after_step5d") or {}
        self.assertEqual(rg.get("current_status"), READINESS_AFTER)
        self.assertTrue(rg.get("step6a_with_audible_renders_allowed"))

    def test_explicit_statement_present(self) -> None:
        report = self._report()
        stmt = report.get("explicit_statement") or ""
        self.assertIn("audible diagnostic listening renders", stmt)
        self.assertIn("does not change the physical model", stmt)

    def test_write_reports_to_disk(self) -> None:
        jpath = self.tmp / "step5d.json"
        mpath = self.tmp / "step5d.md"
        report = write_pgsm_step5d_reports(
            repo_root=REPO,
            render_dir=self.tmp / "disk_renders",
            json_path=jpath,
            md_path=mpath,
        )
        self.assertTrue(jpath.is_file())
        self.assertTrue(mpath.is_file())
        loaded = json.loads(jpath.read_text(encoding="utf-8"))
        self.assertEqual(loaded.get("report_version"), PGSM_STEP5D_VERSION)
        self.assertEqual(report.get("report_version"), PGSM_STEP5D_VERSION)

    def test_analyze_audibility_on_real_wav(self) -> None:
        y, sr = load_wav_mono(STEP5A_AUDIO_DIR / "sample_000_A2_diagnostic.wav")
        m = analyze_audibility(y, sr, role="main")
        self.assertTrue(m.get("transient_dominant_peak"))
        self.assertTrue(m.get("too_short_active_duration"))

    def test_apply_listening_render_waveform_correlation(self) -> None:
        y, sr = load_wav_mono(STEP5A_AUDIO_DIR / "sample_000_A2_diagnostic.wav")
        rendered, trimmed, info = apply_listening_render(y, sr)
        val = __import__(
            "pgsm_step5d_audible_diagnostic_render_repair", fromlist=["validate_render"]
        ).validate_render(trimmed, rendered, sr, info)
        self.assertGreaterEqual(val.get("waveform_correlation_after_gain_compensation"), 0.98)


if __name__ == "__main__":
    unittest.main()
