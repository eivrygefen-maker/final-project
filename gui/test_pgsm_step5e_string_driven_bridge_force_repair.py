#!/usr/bin/env python3
"""PGSM Step 5E — string-driven bridge force repair tests."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gui"))

from body_hybrid_v4_1_identity_space import STK_BODY_TRANSFER_FINAL_V1  # noqa: E402
from pgsm_step3a_numerical_ir_testbench import DURATION_S, NUMERIC_SR  # noqa: E402
from pgsm_step4b_single_note_diagnostic_refinement import load_wav_mono  # noqa: E402
from pgsm_step5a_limited_note_set_diagnostic_audio import (  # noqa: E402
    NOTE_SET,
    step4a_output_fingerprints,
)
from pgsm_step5b_limited_note_set_refinement import step5a_output_fingerprints  # noqa: E402
from pgsm_step5d_audible_diagnostic_render_repair import (  # noqa: E402
    READINESS_AFTER as READINESS_STEP5D,
    RENDER_DIR as STEP5D_RENDER_DIR,
)
from pgsm_step5e_string_driven_bridge_force_repair import (  # noqa: E402
    AUDIO_DIR,
    ENERGY_FIRST_10MS_MAX,
    OUTPUT_DURATION_S,
    PGSM_STEP5E_VERSION,
    READINESS_AFTER,
    STRING_FORCE_LABEL,
    build_pgsm_step5e_report,
    build_string_driven_bridge_force,
    collect_previous_audio_fingerprints,
    write_pgsm_step5e_reports,
)
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE  # noqa: E402

MAX_MODES_TEST = 100


class TestPgsmStep5eStringDrivenBridgeForceRepair(unittest.TestCase):
    _shared_report: dict | None = None
    _shared_wav_dir: Path | None = None

    @classmethod
    def setUpClass(cls) -> None:
        cls._shared_wav_dir = REPO / "audio" / "pgsm_step5e_string_driven_bridge_force"
        cls._shared_report = build_pgsm_step5e_report(
            repo_root=REPO,
            audio_dir=cls._shared_wav_dir,
            write_wav=True,
            max_modes=MAX_MODES_TEST,
        )

    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)
        self._prev_fp = collect_previous_audio_fingerprints(REPO)

    def tearDown(self) -> None:
        self._td.cleanup()

    def _report(self) -> dict:
        assert self._shared_report is not None
        return self._shared_report

    def test_step5d_and_step5c_reports_load(self) -> None:
        report = self._report()
        self.assertIsNotNone(report.get("step5d_loaded"))
        self.assertIsNotNone(report.get("step5c_loaded"))
        upstream = report.get("upstream_readiness") or {}
        self.assertTrue(upstream.get("step5d_pass"))
        self.assertEqual(upstream.get("step5d_readiness"), READINESS_STEP5D)

    def test_no_stk_integration(self) -> None:
        import pgsm_step5e_string_driven_bridge_force_repair as mod

        src = Path(mod.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import diagnostic_synthesis", src)
        self.assertTrue(self._report().get("no_stk_integration"))

    def test_no_fem_rom_subprocess(self) -> None:
        with patch("subprocess.run") as mock_run, patch("subprocess.Popen") as mock_popen:
            build_pgsm_step5e_report(
                repo_root=REPO,
                audio_dir=self.tmp / "subproc",
                write_wav=False,
                max_modes=MAX_MODES_TEST,
            )
            mock_run.assert_not_called()
            mock_popen.assert_not_called()

    def test_website_default_unchanged(self) -> None:
        report = self._report()
        self.assertEqual(report.get("website_default"), DEFAULT_WEBSITE_STK_MODE)
        self.assertEqual(report.get("website_default"), STK_BODY_TRANSFER_FINAL_V1)
        self.assertTrue(report.get("website_default_unchanged"))

    def test_previous_step5a_5d_files_preserved(self) -> None:
        after = collect_previous_audio_fingerprints(REPO)
        self.assertEqual(self._prev_fp, after)
        self.assertTrue(self._report().get("no_previous_audio_modified"))

    def test_exactly_four_string_driven_diagnostic_wavs_generated(self) -> None:
        report = self._report()
        notes = (report.get("output_files") or {}).get("notes") or {}
        self.assertEqual(len(notes), 4)
        out = self._shared_wav_dir
        assert out is not None
        for note in NOTE_SET:
            path = out / f"sample_000_{note}_string_driven_diagnostic.wav"
            self.assertTrue(path.is_file(), msg=str(path))

    def test_string_force_stems_generated(self) -> None:
        out = self._shared_wav_dir
        assert out is not None
        for note in NOTE_SET:
            self.assertTrue((out / f"sample_000_{note}_string_force_stem.wav").is_file())

    def test_body_stems_generated(self) -> None:
        out = self._shared_wav_dir
        assert out is not None
        for note in NOTE_SET:
            self.assertTrue((out / f"sample_000_{note}_body_stem.wav").is_file())

    def test_output_duration_not_trimmed_to_0_22s(self) -> None:
        report = self._report()
        obj = report.get("objective_test_results") or {}
        self.assertTrue(obj.get("full_duration_not_trimmed"))
        for note in NOTE_SET:
            m = (report.get("per_note_metrics") or {}).get(note) or {}
            self.assertGreater(m.get("duration_s", 0), 1.0)
            self.assertAlmostEqual(m.get("duration_s"), OUTPUT_DURATION_S, delta=0.05)

    def test_energy_first_10ms_reduced_versus_step5d(self) -> None:
        report = self._report()
        obj = report.get("objective_test_results") or {}
        self.assertTrue(obj.get("energy_first_10ms_reduced_vs_step5d"))
        for note in NOTE_SET:
            m = (report.get("per_note_metrics") or {}).get(note) or {}
            self.assertLess(m.get("energy_first_10ms"), ENERGY_FIRST_10MS_MAX)
            self.assertTrue(m.get("improved_vs_step5d_energy_10ms"))

    def test_active_duration_increased_versus_step5d(self) -> None:
        report = self._report()
        obj = report.get("objective_test_results") or {}
        self.assertTrue(obj.get("active_duration_increased_vs_step5d"))
        for note in NOTE_SET:
            m = (report.get("per_note_metrics") or {}).get(note) or {}
            self.assertTrue(m.get("improved_vs_step5d_active_duration"))
            if note in ("A2", "A3", "A4"):
                self.assertGreater(m.get("active_duration_minus_60_dbfs_ms"), 1000.0)

    def test_pitch_salience_computed_for_every_note(self) -> None:
        report = self._report()
        obj = report.get("objective_test_results") or {}
        self.assertTrue(obj.get("pitch_salience_all_notes"))
        for note in NOTE_SET:
            m = (report.get("per_note_metrics") or {}).get(note) or {}
            self.assertIn("pitch_salience_f0", m)
            self.assertTrue(m.get("pitch_salience_detectable"))

    def test_harmonic_energies_h1_h8_computed(self) -> None:
        report = self._report()
        for note in NOTE_SET:
            h = (report.get("per_note_metrics") or {}).get(note, {}).get("harmonic_energy_fraction") or {}
            for k in range(1, 9):
                self.assertIn(f"H{k}", h)

    def test_no_second_onset(self) -> None:
        report = self._report()
        for note in NOTE_SET:
            m = (report.get("per_note_metrics") or {}).get(note) or {}
            self.assertTrue(m.get("no_second_onset"))

    def test_no_hard_gate(self) -> None:
        report = self._report()
        for note in NOTE_SET:
            m = (report.get("per_note_metrics") or {}).get(note) or {}
            self.assertTrue(m.get("no_hard_gate"))

    def test_no_end_rise(self) -> None:
        report = self._report()
        for note in NOTE_SET:
            m = (report.get("per_note_metrics") or {}).get(note) or {}
            self.assertTrue(m.get("no_end_rise"))

    def test_no_reverb_echo_body_tail_added(self) -> None:
        report = self._report()
        art = report.get("artifact_guard_results") or {}
        cavity = report.get("cavity_response_summary") or {}
        self.assertTrue(art.get("no_reverb"))
        self.assertTrue(art.get("no_echo"))
        self.assertTrue(art.get("no_body_tail_layer"))
        self.assertTrue(cavity.get("no_artificial_echo_or_reverb"))

    def test_gain_reported_separately_from_physics(self) -> None:
        report = self._report()
        details = report.get("listening_render_details") or {}
        for note in NOTE_SET:
            d = details.get(note) or {}
            self.assertTrue(d.get("gain_separate_from_physics"))
            self.assertFalse(d.get("physics_changed"))
        obj = report.get("objective_test_results") or {}
        self.assertTrue(obj.get("gain_reported_separately"))

    def test_readiness_remains_diagnostic_only(self) -> None:
        report = self._report()
        rg = report.get("readiness_after_step5e") or {}
        self.assertFalse(rg.get("final_synthesis_ready"))
        self.assertFalse(rg.get("stk_integration_allowed"))
        self.assertFalse(rg.get("website_production_replacement_allowed"))
        self.assertFalse(rg.get("multi_guitar_comparison_allowed"))
        self.assertFalse(rg.get("melody_chord_playback_allowed"))
        self.assertFalse(rg.get("subjective_tuning_allowed"))
        self.assertFalse(rg.get("real_guitar_equivalence_allowed"))

    def test_all_objective_tests_pass(self) -> None:
        report = self._report()
        obj = report.get("objective_test_results") or {}
        self.assertTrue(obj.get("all_notes_pass"))
        self.assertTrue(obj.get("artifact_guard_pass"))
        self.assertTrue(obj.get("all_pass"))

    def test_readiness_after_pass(self) -> None:
        report = self._report()
        rg = report.get("readiness_after_step5e") or {}
        self.assertEqual(rg.get("current_status"), READINESS_AFTER)
        self.assertTrue(rg.get("step5f_extended_validation_allowed"))

    def test_string_force_contract_label(self) -> None:
        report = self._report()
        contracts = report.get("per_note_string_force_contract") or {}
        for note in NOTE_SET:
            c = contracts.get(note) or {}
            self.assertEqual(c.get("label"), STRING_FORCE_LABEL)

    def test_explicit_statement_present(self) -> None:
        report = self._report()
        stmt = report.get("explicit_statement") or ""
        self.assertIn("sustained diagnostic string-driven bridge-force proxy", stmt)
        self.assertIn("does not prove realism", stmt)

    def test_write_reports_to_disk(self) -> None:
        jpath = self.tmp / "step5e.json"
        mpath = self.tmp / "step5e.md"
        report = write_pgsm_step5e_reports(
            repo_root=REPO,
            audio_dir=self.tmp / "disk_out",
            json_path=jpath,
            md_path=mpath,
            max_modes=MAX_MODES_TEST,
        )
        self.assertTrue(jpath.is_file())
        self.assertTrue(mpath.is_file())
        loaded = json.loads(jpath.read_text(encoding="utf-8"))
        self.assertEqual(loaded.get("report_version"), PGSM_STEP5E_VERSION)
        self.assertEqual(report.get("report_version"), PGSM_STEP5E_VERSION)

    def test_string_force_sustained_not_impulse(self) -> None:
        sr = NUMERIC_SR
        n = int(DURATION_S * sr)
        f = build_string_driven_bridge_force(n, sr, 110.0)
        e10 = float(np.sum(f[: int(0.01 * sr)] ** 2) / max(np.sum(f**2), 1e-12))
        self.assertLess(e10, 0.05)


if __name__ == "__main__":
    unittest.main()
