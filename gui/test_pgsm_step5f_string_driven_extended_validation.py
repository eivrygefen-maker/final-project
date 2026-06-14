#!/usr/bin/env python3
"""PGSM Step 5F — string-driven extended validation tests."""
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
from pgsm_step5e_string_driven_bridge_force_repair import (  # noqa: E402
    AUDIO_DIR as STEP5E_AUDIO_DIR,
    NOTE_SET,
    READINESS_AFTER as READINESS_STEP5E,
)
from pgsm_step5f_string_driven_extended_validation import (  # noqa: E402
    ACTIVE_DURATION_MIN_MS,
    CLICK_ENERGY_10MS_MAX,
    PGSM_STEP5F_VERSION,
    READINESS_AFTER,
    build_pgsm_step5f_report,
    collect_previous_audio_fingerprints,
    collect_step5e_fingerprints,
    write_pgsm_step5f_reports,
)
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE  # noqa: E402

MAX_MODES_TEST = 100


class TestPgsmStep5fStringDrivenExtendedValidation(unittest.TestCase):
    _shared_report: dict | None = None

    @classmethod
    def setUpClass(cls) -> None:
        cls._shared_report = build_pgsm_step5f_report(
            repo_root=REPO, write_figures=False, max_modes=MAX_MODES_TEST
        )

    def setUp(self) -> None:
        self._step5e_fp = collect_step5e_fingerprints(REPO)
        self._prev_fp = collect_previous_audio_fingerprints(REPO)

    def _report(self) -> dict:
        assert self._shared_report is not None
        return self._shared_report

    def test_step5e_report_loads(self) -> None:
        report = self._report()
        self.assertIsNotNone(report.get("step5e_loaded"))

    def test_step5e_readiness_verified(self) -> None:
        upstream = self._report().get("upstream_readiness") or {}
        self.assertEqual(upstream.get("step5e_readiness"), READINESS_STEP5E)
        self.assertTrue(upstream.get("step5e_pass"))

    def test_exactly_four_step5e_main_wavs_exist(self) -> None:
        upstream = self._report().get("upstream_readiness") or {}
        self.assertTrue(upstream.get("four_main_wavs_exist"))
        for note in NOTE_SET:
            path = STEP5E_AUDIO_DIR / f"sample_000_{note}_string_driven_diagnostic.wav"
            self.assertTrue(path.is_file(), msg=str(path))

    def test_body_stems_and_string_force_stems_exist(self) -> None:
        upstream = self._report().get("upstream_readiness") or {}
        self.assertTrue(upstream.get("stems_exist_all_notes"))
        for note in NOTE_SET:
            self.assertTrue((STEP5E_AUDIO_DIR / f"sample_000_{note}_body_stem.wav").is_file())
            self.assertTrue((STEP5E_AUDIO_DIR / f"sample_000_{note}_string_force_stem.wav").is_file())

    def test_no_new_wav_generated(self) -> None:
        after5e = collect_step5e_fingerprints(REPO)
        after_prev = collect_previous_audio_fingerprints(REPO)
        self.assertEqual(self._step5e_fp, after5e)
        self.assertEqual(self._prev_fp, after_prev)
        self.assertTrue(self._report().get("no_new_wav_generated"))

    def test_no_stk_integration(self) -> None:
        import pgsm_step5f_string_driven_extended_validation as mod

        src = Path(mod.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import diagnostic_synthesis", src)
        self.assertTrue(self._report().get("no_stk_integration"))

    def test_no_fem_rom_subprocess(self) -> None:
        with patch("subprocess.run") as mock_run, patch("subprocess.Popen") as mock_popen:
            build_pgsm_step5f_report(repo_root=REPO, max_modes=MAX_MODES_TEST)
            mock_run.assert_not_called()
            mock_popen.assert_not_called()

    def test_website_default_unchanged(self) -> None:
        report = self._report()
        self.assertEqual(report.get("website_default"), DEFAULT_WEBSITE_STK_MODE)
        self.assertEqual(report.get("website_default"), STK_BODY_TRANSFER_FINAL_V1)
        self.assertTrue(report.get("website_default_unchanged"))

    def test_per_note_extended_metrics_computed(self) -> None:
        metrics = self._report().get("per_note_extended_metrics") or {}
        self.assertEqual(len(metrics), 4)
        for note in NOTE_SET:
            m = metrics.get(note) or {}
            self.assertIn("spectral_centroid_hz", m)
            self.assertIn("harmonic_energy_fraction", m)
            self.assertIn("partial_decay_analysis", m)

    def test_pitch_salience_present_for_every_note(self) -> None:
        obj = self._report().get("objective_test_results") or {}
        self.assertTrue(obj.get("pitch_salience_all_notes"))
        for note in NOTE_SET:
            m = (self._report().get("per_note_extended_metrics") or {}).get(note) or {}
            self.assertTrue(m.get("pitch_salience_present"))

    def test_energy_first_10ms_below_click_threshold(self) -> None:
        obj = self._report().get("objective_test_results") or {}
        self.assertTrue(obj.get("energy_first_10ms_below_click_threshold"))
        for note in NOTE_SET:
            m = (self._report().get("per_note_extended_metrics") or {}).get(note) or {}
            self.assertLess(m.get("energy_first_10ms"), CLICK_ENERGY_10MS_MAX)

    def test_active_duration_above_1s(self) -> None:
        obj = self._report().get("objective_test_results") or {}
        self.assertTrue(obj.get("active_duration_sufficient"))
        for note in ("A2", "A3", "A4"):
            m = (self._report().get("per_note_extended_metrics") or {}).get(note) or {}
            self.assertGreater(m.get("active_duration_minus_60_dbfs_ms"), ACTIVE_DURATION_MIN_MS)

    def test_no_second_onset(self) -> None:
        for note in NOTE_SET:
            m = (self._report().get("per_note_extended_metrics") or {}).get(note) or {}
            self.assertTrue(m.get("no_second_onset"))

    def test_no_end_rise(self) -> None:
        for note in NOTE_SET:
            m = (self._report().get("per_note_extended_metrics") or {}).get(note) or {}
            self.assertTrue(m.get("no_end_rise"))

    def test_no_hard_gate(self) -> None:
        for note in NOTE_SET:
            m = (self._report().get("per_note_extended_metrics") or {}).get(note) or {}
            self.assertTrue(m.get("no_hard_gate"))

    def test_no_reverb_echo_body_tail(self) -> None:
        art = self._report().get("artifact_guard_results") or {}
        self.assertTrue(art.get("no_reverb"))
        self.assertTrue(art.get("no_echo"))
        self.assertTrue(art.get("no_body_tail_layer"))

    def test_robotic_tone_diagnosis_computed(self) -> None:
        robotic = self._report().get("robotic_tone_diagnosis") or {}
        self.assertIn("global_robotic_tone_labels", robotic)
        labels = robotic.get("global_robotic_tone_labels") or {}
        self.assertIn("excessive_harmonic_purity", labels)
        self.assertIn("weak_pluck_noise_component", labels)
        self.assertIn("over_regular_partial_decay", labels)

    def test_string_body_air_decay_decomposition_reported(self) -> None:
        decay = self._report().get("string_body_air_decay_decomposition") or {}
        self.assertIn("string_decay_summary", decay)
        self.assertIn("body_top_decay_summary", decay)
        self.assertIn("air_cavity_decay_summary", decay)
        self.assertIn("radiation_decay_summary", decay)

    def test_string_force_vs_body_metrics_computed(self) -> None:
        sb = self._report().get("string_force_vs_body_metrics") or {}
        self.assertEqual(len(sb), 4)
        for note in NOTE_SET:
            self.assertIn("envelope_correlation", sb.get(note) or {})

    def test_body_causally_driven_by_string_force(self) -> None:
        for note in NOTE_SET:
            b = (self._report().get("string_force_vs_body_metrics") or {}).get(note) or {}
            self.assertTrue(b.get("causally_driven_no_delayed_body_onset"))
            self.assertTrue(b.get("body_not_copied_string_waveform"))

    def test_cavity_internal_response_proxy_only(self) -> None:
        cavity = self._report().get("cavity_response_summary") or {}
        self.assertTrue(cavity.get("proxy_only_not_measured"))
        self.assertTrue(cavity.get("no_artificial_echo_or_reverb"))

    def test_shared_body_ir_limitation_reported(self) -> None:
        shared = self._report().get("shared_body_ir_limitation") or {}
        self.assertTrue(shared.get("shared_modal_ir_across_all_notes"))
        self.assertIn("explicit_label", shared)

    def test_step5g_model_update_plan_generated(self) -> None:
        plan = self._report().get("recommended_step5g_model_update_plan") or {}
        self.assertTrue(plan.get("recommended_categories"))
        self.assertTrue(plan.get("no_implementation_in_step5f"))
        obj = self._report().get("objective_test_results") or {}
        self.assertTrue(obj.get("step5g_plan_generated"))

    def test_readiness_remains_diagnostic_only(self) -> None:
        rg = self._report().get("readiness_after_step5f") or {}
        self.assertFalse(rg.get("final_synthesis_ready"))
        self.assertFalse(rg.get("stk_integration_allowed"))
        self.assertFalse(rg.get("real_guitar_equivalence_allowed"))

    def test_all_objective_tests_pass(self) -> None:
        obj = self._report().get("objective_test_results") or {}
        self.assertTrue(obj.get("all_notes_extended_pass"))
        self.assertTrue(obj.get("artifact_guard_pass"))
        self.assertTrue(obj.get("all_pass"))

    def test_readiness_after_pass(self) -> None:
        rg = self._report().get("readiness_after_step5f") or {}
        self.assertEqual(rg.get("current_status"), READINESS_AFTER)
        self.assertTrue(rg.get("step5g_physical_tone_update_allowed"))

    def test_explicit_statement_present(self) -> None:
        stmt = self._report().get("explicit_statement") or ""
        self.assertIn("validates and diagnoses", stmt)
        self.assertIn("does not generate new audio", stmt)

    def test_harmonic_energies_h1_h12_computed(self) -> None:
        for note in NOTE_SET:
            h = (self._report().get("per_note_extended_metrics") or {}).get(note, {}).get(
                "harmonic_energy_fraction"
            ) or {}
            for k in range(1, 13):
                self.assertIn(f"H{k}", h)

    def test_write_reports_to_disk(self) -> None:
        td = tempfile.TemporaryDirectory()
        tmp = Path(td.name)
        jpath = tmp / "step5f.json"
        mpath = tmp / "step5f.md"
        report = write_pgsm_step5f_reports(
            repo_root=REPO,
            json_path=jpath,
            md_path=mpath,
            max_modes=MAX_MODES_TEST,
        )
        self.assertTrue(jpath.is_file())
        self.assertTrue(mpath.is_file())
        loaded = json.loads(jpath.read_text(encoding="utf-8"))
        self.assertEqual(loaded.get("report_version"), PGSM_STEP5F_VERSION)
        self.assertEqual(report.get("report_version"), PGSM_STEP5F_VERSION)
        td.cleanup()


if __name__ == "__main__":
    unittest.main()
