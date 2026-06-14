#!/usr/bin/env python3
"""PGSM Step 5I — string partial damping refinement tests."""
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
from pgsm_step5a_limited_note_set_diagnostic_audio import NOTE_SET  # noqa: E402
from pgsm_step5e_string_driven_bridge_force_repair import (  # noqa: E402
    ENERGY_FIRST_10MS_MAX,
    OUTPUT_DURATION_S,
    collect_previous_audio_fingerprints,
)
from pgsm_step5f_string_driven_extended_validation import collect_step5e_fingerprints  # noqa: E402
from pgsm_step5h_note_string_fret_contract import READINESS_AFTER as READINESS_STEP5H  # noqa: E402
from pgsm_step5i_string_partial_damping_refinement import (  # noqa: E402
    AUDIO_DIR,
    HARMONIC_ORDER_EXPONENT,
    PGSM_STEP5I_VERSION,
    READINESS_AFTER,
    build_partial_tau_summary,
    build_pgsm_step5i_report,
    build_refined_string_bridge_force,
    build_string_partial_damping_contract,
    collect_all_previous_audio_fingerprints,
    compute_refined_partial_tau_k,
    compute_step5e_partial_tau_k,
    write_pgsm_step5i_reports,
)
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE  # noqa: E402

MAX_MODES_TEST = 100
REQUIRED_TERMS = (
    "base_string_decay_by_string_id",
    "harmonic_order_loss",
    "frequency_dependent_loss",
    "fret_contact_loss_proxy",
    "material_internal_loss_proxy",
    "bridge_nut_boundary_loss_proxy",
    "combined_partial_tau",
)


class TestPgsmStep5iStringPartialDampingRefinement(unittest.TestCase):
    _shared_report: dict | None = None
    _shared_wav_dir: Path | None = None

    @classmethod
    def setUpClass(cls) -> None:
        cls._shared_wav_dir = REPO / "audio" / "pgsm_step5i_string_partial_damping_refinement"
        cls._shared_report = build_pgsm_step5i_report(
            repo_root=REPO,
            audio_dir=cls._shared_wav_dir,
            write_wav=True,
            max_modes=MAX_MODES_TEST,
        )

    def setUp(self) -> None:
        self._prev_fp = collect_all_previous_audio_fingerprints(REPO)

    def _report(self) -> dict:
        assert self._shared_report is not None
        return self._shared_report

    def test_step5h_contract_loads(self) -> None:
        self.assertIsNotNone(self._report().get("step5h_loaded"))

    def test_step5h_readiness_verified(self) -> None:
        upstream = self._report().get("upstream_readiness") or {}
        self.assertEqual(upstream.get("step5h_readiness"), READINESS_STEP5H)
        self.assertTrue(upstream.get("step5h_pass"))
        self.assertTrue(upstream.get("step5h_all_pass"))

    def test_preferred_mappings_loaded(self) -> None:
        mapping = self._report().get("note_string_fret_mapping_used") or {}
        for note in NOTE_SET:
            self.assertIn(note, mapping)
            self.assertIn("string_id", mapping[note])
            self.assertIn("fret", mapping[note])
            self.assertIn("effective_length_m", mapping[note])

    def test_no_stk_integration(self) -> None:
        import pgsm_step5i_string_partial_damping_refinement as mod

        src = Path(mod.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import diagnostic_synthesis", src)
        self.assertTrue(self._report().get("no_stk_integration"))

    def test_no_fem_rom_subprocess(self) -> None:
        with patch("subprocess.run") as mock_run, patch("subprocess.Popen") as mock_popen:
            build_pgsm_step5i_report(
                repo_root=REPO,
                audio_dir=REPO / "audio" / "_tmp_step5i_subproc",
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

    def test_previous_audio_files_preserved(self) -> None:
        after = collect_all_previous_audio_fingerprints(REPO)
        self.assertEqual(self._prev_fp, after)
        self.assertTrue(self._report().get("no_previous_audio_modified"))

    def test_exactly_four_step5i_diagnostic_wavs_generated(self) -> None:
        notes = (self._report().get("generated_files") or {}).get("notes") or {}
        self.assertEqual(len(notes), 4)
        out = self._shared_wav_dir
        assert out is not None
        for note in NOTE_SET:
            path = out / f"sample_000_{note}_damping_refined_diagnostic.wav"
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

    def test_damping_contract_includes_all_terms(self) -> None:
        contract = self._report().get("string_partial_damping_contract") or {}
        names = {t.get("term") for t in contract.get("terms") or []}
        for term in REQUIRED_TERMS:
            self.assertIn(term, names)

    def test_tau_differs_by_harmonic_order(self) -> None:
        tau1 = compute_refined_partial_tau_k(1, 110.0, string_id="string_5", fret=0)[0]
        tau8 = compute_refined_partial_tau_k(8, 880.0, string_id="string_5", fret=0)[0]
        self.assertGreater(tau1, tau8)

    def test_tau_differs_by_string_context(self) -> None:
        tau_bass = compute_refined_partial_tau_k(3, 330.0, string_id="string_5", fret=0)[0]
        tau_treble = compute_refined_partial_tau_k(3, 330.0, string_id="string_1", fret=5)[0]
        self.assertNotAlmostEqual(tau_bass, tau_treble, places=4)

    def test_high_harmonics_decay_faster_than_low(self) -> None:
        summary = build_partial_tau_summary(string_id="string_4", fret=7, f0=220.0)
        self.assertTrue(summary.get("high_harmonics_decay_faster_than_low"))
        obj = self._report().get("objective_test_results") or {}
        self.assertTrue(obj.get("high_harmonics_decay_faster"))

    def test_pitch_salience_remains_present(self) -> None:
        obj = self._report().get("objective_test_results") or {}
        self.assertTrue(obj.get("pitch_salience_all_notes"))
        for note in NOTE_SET:
            m = (self._report().get("per_note_metrics") or {}).get(note) or {}
            self.assertTrue(m.get("pitch_salience_detectable"))

    def test_active_duration_above_one_second(self) -> None:
        obj = self._report().get("objective_test_results") or {}
        self.assertTrue(obj.get("active_duration_all_notes"))
        for note in ("A2", "A3", "A4"):
            m = (self._report().get("per_note_metrics") or {}).get(note) or {}
            self.assertGreater(m.get("active_duration_minus_60_dbfs_ms", 0), 1000.0)

    def test_energy_first_10ms_below_click_threshold(self) -> None:
        for note in NOTE_SET:
            m = (self._report().get("per_note_metrics") or {}).get(note) or {}
            self.assertLess(m.get("energy_first_10ms"), ENERGY_FIRST_10MS_MAX)

    def test_harmonic_purity_proxy_computed(self) -> None:
        obj = self._report().get("objective_test_results") or {}
        self.assertTrue(obj.get("harmonic_purity_proxy_computed"))
        for note in NOTE_SET:
            m = (self._report().get("per_note_metrics") or {}).get(note) or {}
            self.assertIn("harmonic_purity_proxy_db", m)

    def test_comparison_vs_step5e_computed(self) -> None:
        comp = self._report().get("comparison_vs_step5e") or {}
        self.assertTrue(comp.get("all_notes_pitch_salience_maintained"))
        for note in NOTE_SET:
            self.assertIn(note, comp)
            self.assertIn("harmonic_purity_delta_db", comp[note])

    def test_no_second_onset(self) -> None:
        for note in NOTE_SET:
            m = (self._report().get("per_note_metrics") or {}).get(note) or {}
            self.assertTrue(m.get("no_second_onset"))

    def test_no_end_rise(self) -> None:
        for note in NOTE_SET:
            m = (self._report().get("per_note_metrics") or {}).get(note) or {}
            self.assertTrue(m.get("no_end_rise"))

    def test_no_hard_gate(self) -> None:
        for note in NOTE_SET:
            m = (self._report().get("per_note_metrics") or {}).get(note) or {}
            self.assertTrue(m.get("no_hard_gate"))

    def test_no_reverb_echo_body_tail(self) -> None:
        art = self._report().get("artifact_guard_results") or {}
        self.assertTrue(art.get("no_reverb"))
        self.assertTrue(art.get("no_echo"))
        self.assertTrue(art.get("no_body_tail_layer"))

    def test_gain_reported_separately_from_physics(self) -> None:
        obj = self._report().get("objective_test_results") or {}
        self.assertTrue(obj.get("gain_reported_separately"))
        details = self._report().get("listening_render_details") or {}
        for note in NOTE_SET:
            self.assertTrue((details.get(note) or {}).get("gain_separate_from_physics"))

    def test_readiness_diagnostic_only(self) -> None:
        rg = self._report().get("readiness_after_step5i") or {}
        self.assertEqual(rg.get("current_status"), READINESS_AFTER)
        self.assertFalse(rg.get("final_synthesis_ready"))
        self.assertFalse(rg.get("stk_integration_allowed"))
        self.assertTrue(rg.get("contract_only_not_final"))

    def test_objective_all_pass(self) -> None:
        obj = self._report().get("objective_test_results") or {}
        self.assertTrue(obj.get("all_pass"))

    def test_refined_force_steeper_than_step5e(self) -> None:
        n = int(DURATION_S * NUMERIC_SR)
        y_ref, _ = build_refined_string_bridge_force(
            n, NUMERIC_SR, 110.0, string_id="string_5", fret=0
        )
        tau_ref = compute_refined_partial_tau_k(6, 660.0, string_id="string_5", fret=0)[0]
        tau_e = compute_step5e_partial_tau_k(6)
        self.assertLess(tau_ref, tau_e)
        self.assertGreater(HARMONIC_ORDER_EXPONENT, 0.65)
        self.assertGreater(np.max(np.abs(y_ref)), 0.0)

    def test_write_reports_creates_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "audio" / "debug_reports").mkdir(parents=True)
            (root / "data").mkdir(parents=True)
            for name in (
                "pgsm_step5h_note_string_fret_contract.json",
                "pgsm_step5g_physical_tone_model_update_plan.json",
                "pgsm_step5f_string_driven_extended_validation.json",
                "pgsm_step5e_string_driven_bridge_force_repair.json",
                "pgsm_step3c_numeric_calibration.json",
            ):
                src = REPO / "audio" / "debug_reports" / name
                if src.is_file():
                    (root / "audio" / "debug_reports" / name).write_text(
                        src.read_text(encoding="utf-8"), encoding="utf-8"
                    )
            (root / "data" / "pgsm_classical_guitar_note_string_fret_contract.json").write_text(
                (REPO / "data" / "pgsm_classical_guitar_note_string_fret_contract.json").read_text(
                    encoding="utf-8"
                ),
                encoding="utf-8",
            )
            report = write_pgsm_step5i_reports(
                repo_root=REPO,
                json_path=root / "audio" / "debug_reports" / "out.json",
                md_path=root / "audio" / "debug_reports" / "out.md",
                data_path=root / "data" / "contract.json",
                audio_dir=root / "audio" / "step5i_test",
                write_wav=True,
                max_modes=MAX_MODES_TEST,
            )
            self.assertEqual(report.get("report_version"), PGSM_STEP5I_VERSION)
            self.assertTrue((root / "data" / "contract.json").is_file())

    def test_output_duration_full(self) -> None:
        out = self._shared_wav_dir
        assert out is not None
        for note in NOTE_SET:
            y, sr = load_wav_mono(out / f"sample_000_{note}_damping_refined_diagnostic.wav")
            self.assertAlmostEqual(len(y) / sr, OUTPUT_DURATION_S, delta=0.05)


if __name__ == "__main__":
    unittest.main()
