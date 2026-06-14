#!/usr/bin/env python3
"""PGSM Step 5I.1 — string damping duration and treble harshness repair tests."""
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
from pgsm_step5a_limited_note_set_diagnostic_audio import NOTE_SET  # noqa: E402
from pgsm_step5e_string_driven_bridge_force_repair import ENERGY_FIRST_10MS_MAX  # noqa: E402
from pgsm_step5i_string_partial_damping_refinement import READINESS_AFTER as READINESS_STEP5I  # noqa: E402
from pgsm_step5i_1_string_damping_duration_harshness_repair import (  # noqa: E402
    ALLOWED_DURATION_MAX_S,
    ALLOWED_DURATION_MIN_S,
    DEFAULT_DURATION_S,
    HARMONIC_ORDER_EXPONENT,
    PGSM_STEP5I_1_VERSION,
    READINESS_AFTER,
    assess_harmonic_purity_change,
    build_pgsm_step5i_1_report,
    build_partial_tau_summary_v2,
    build_v2_string_bridge_force,
    collect_all_previous_audio_fingerprints,
    compute_v2_partial_tau_k,
    write_pgsm_step5i_1_reports,
)
from pgsm_step3a_numerical_ir_testbench import NUMERIC_SR  # noqa: E402
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


class TestPgsmStep5i1StringDampingDurationHarshnessRepair(unittest.TestCase):
    _shared_report: dict | None = None
    _shared_wav_dir: Path | None = None

    @classmethod
    def setUpClass(cls) -> None:
        cls._shared_wav_dir = REPO / "audio" / "pgsm_step5i_1_string_damping_duration_harshness_repair"
        cls._shared_report = build_pgsm_step5i_1_report(
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

    def test_step5i_report_loads(self) -> None:
        self.assertIsNotNone(self._report().get("step5i_loaded"))

    def test_step5h_contract_loads(self) -> None:
        self.assertIsNotNone(self._report().get("step5h_loaded"))

    def test_no_stk_integration(self) -> None:
        import pgsm_step5i_1_string_damping_duration_harshness_repair as mod

        src = Path(mod.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import diagnostic_synthesis", src)
        self.assertTrue(self._report().get("no_stk_integration"))

    def test_no_fem_rom_subprocess(self) -> None:
        with patch("subprocess.run") as mock_run, patch("subprocess.Popen") as mock_popen:
            build_pgsm_step5i_1_report(
                repo_root=REPO,
                audio_dir=REPO / "audio" / "_tmp_step5i1_subproc",
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

    def test_exactly_four_step5i1_diagnostic_wavs_generated(self) -> None:
        notes = (self._report().get("generated_files") or {}).get("notes") or {}
        self.assertEqual(len(notes), 4)
        out = self._shared_wav_dir
        assert out is not None
        for note in NOTE_SET:
            self.assertTrue((out / f"sample_000_{note}_damping_v2_diagnostic.wav").is_file())

    def test_duration_between_3_and_6_seconds(self) -> None:
        obj = self._report().get("objective_test_results") or {}
        self.assertTrue(obj.get("duration_in_range_3_to_6s"))
        for note in NOTE_SET:
            m = (self._report().get("per_note_metrics") or {}).get(note) or {}
            d = m.get("duration_s", 0)
            self.assertGreaterEqual(d, ALLOWED_DURATION_MIN_S)
            self.assertLessEqual(d, ALLOWED_DURATION_MAX_S + 0.05)
        self.assertAlmostEqual(
            (self._report().get("duration_policy") or {}).get("applied_duration_s"),
            DEFAULT_DURATION_S,
        )

    def test_string_force_and_body_stems_generated(self) -> None:
        out = self._shared_wav_dir
        assert out is not None
        for note in NOTE_SET:
            self.assertTrue((out / f"sample_000_{note}_string_force_stem.wav").is_file())
            self.assertTrue((out / f"sample_000_{note}_body_stem.wav").is_file())

    def test_damping_v2_contract_includes_all_terms(self) -> None:
        contract = self._report().get("string_partial_damping_contract_v2") or {}
        names = {t.get("term") for t in contract.get("terms") or []}
        for term in REQUIRED_TERMS:
            self.assertIn(term, names)

    def test_tau_differs_by_harmonic_order_and_string(self) -> None:
        tau1 = compute_v2_partial_tau_k(1, 110.0, string_id="string_5", fret=0)[0]
        tau10 = compute_v2_partial_tau_k(10, 1100.0, string_id="string_5", fret=0)[0]
        self.assertGreater(tau1, tau10)
        bass = compute_v2_partial_tau_k(3, 330.0, string_id="string_5", fret=0)[0]
        treble = compute_v2_partial_tau_k(3, 330.0, string_id="string_1", fret=5)[0]
        self.assertNotAlmostEqual(bass, treble, places=4)

    def test_high_harmonics_decay_faster_than_low(self) -> None:
        summary = build_partial_tau_summary_v2(string_id="string_1", fret=5, f0=440.0)
        self.assertTrue(summary.get("high_harmonics_decay_faster_than_low"))
        obj = self._report().get("objective_test_results") or {}
        self.assertTrue(obj.get("tau_differs_by_harmonic_order"))

    def test_high_vs_low_decay_slope_ratio_computed(self) -> None:
        obj = self._report().get("objective_test_results") or {}
        self.assertTrue(obj.get("high_vs_low_decay_slope_ratio_computed"))
        for note in NOTE_SET:
            m = (self._report().get("per_note_metrics") or {}).get(note) or {}
            self.assertIn("high_vs_low_decay_slope_ratio", m)
            self.assertIn("high_partial_late_energy_ratio", m)

    def test_treble_harshness_proxy_computed(self) -> None:
        obj = self._report().get("objective_test_results") or {}
        self.assertTrue(obj.get("treble_harshness_proxy_computed"))
        for note in NOTE_SET:
            m = (self._report().get("per_note_metrics") or {}).get(note) or {}
            self.assertIn("treble_harshness_proxy", m)

    def test_treble_A4_E5_improved_or_flagged(self) -> None:
        obj = self._report().get("objective_test_results") or {}
        self.assertTrue(obj.get("treble_A4_E5_improved_or_flagged"))
        for note in ("A4", "E5"):
            m = (self._report().get("per_note_metrics") or {}).get(note) or {}
            improved = m.get("treble_harshness_improved_vs_step5i")
            flagged = m.get("treble_harshness_not_improved_flagged")
            self.assertTrue(improved or flagged)

    def test_hnr_comparison_logic_honest(self) -> None:
        increased = assess_harmonic_purity_change(50.0, 55.0)
        self.assertFalse(increased["harmonic_purity_reduced"])
        self.assertTrue(increased["not_improved_flag"])
        decreased = assess_harmonic_purity_change(55.0, 48.0)
        self.assertTrue(decreased["harmonic_purity_reduced"])
        self.assertFalse(decreased["not_improved_flag"])
        obj = self._report().get("objective_test_results") or {}
        self.assertTrue(obj.get("honest_hnr_purity_logic"))
        comp = self._report().get("comparison_vs_step5i") or {}
        for note in NOTE_SET:
            c = comp.get(note) or {}
            if c.get("harmonic_purity_not_improved_flag"):
                self.assertFalse(c.get("harmonic_purity_reduced"))

    def test_pitch_salience_and_active_duration(self) -> None:
        obj = self._report().get("objective_test_results") or {}
        self.assertTrue(obj.get("pitch_salience_all_notes"))
        self.assertTrue(obj.get("active_duration_all_notes"))
        for note in ("A2", "A3", "A4"):
            m = (self._report().get("per_note_metrics") or {}).get(note) or {}
            self.assertGreater(m.get("active_duration_minus_60_dbfs_ms", 0), 1000.0)

    def test_energy_first_10ms_below_click_threshold(self) -> None:
        for note in NOTE_SET:
            m = (self._report().get("per_note_metrics") or {}).get(note) or {}
            self.assertLess(m.get("energy_first_10ms"), ENERGY_FIRST_10MS_MAX)

    def test_no_forbidden_artifacts(self) -> None:
        art = self._report().get("artifact_guard_results") or {}
        self.assertTrue(art.get("pass"))
        for note in NOTE_SET:
            m = (self._report().get("per_note_metrics") or {}).get(note) or {}
            self.assertTrue(m.get("no_second_onset"))
            self.assertTrue(m.get("no_end_rise"))
            self.assertTrue(m.get("no_hard_gate"))

    def test_gain_reported_separately_from_physics(self) -> None:
        obj = self._report().get("objective_test_results") or {}
        self.assertTrue(obj.get("gain_reported_separately"))

    def test_readiness_diagnostic_only(self) -> None:
        rg = self._report().get("readiness_after_step5i_1") or {}
        self.assertEqual(rg.get("current_status"), READINESS_AFTER)
        self.assertFalse(rg.get("stk_integration_allowed"))
        self.assertTrue(rg.get("contract_only_not_final"))

    def test_objective_all_pass(self) -> None:
        self.assertTrue((self._report().get("objective_test_results") or {}).get("all_pass"))

    def test_v2_harmonic_exponent_stronger_than_step5i(self) -> None:
        self.assertGreaterEqual(HARMONIC_ORDER_EXPONENT, 1.10)
        n = int(DEFAULT_DURATION_S * NUMERIC_SR)
        y, _ = build_v2_string_bridge_force(n, NUMERIC_SR, 440.0, string_id="string_1", fret=5)
        self.assertGreater(float(abs(y).max()), 0.0)

    def test_write_reports_creates_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "audio" / "debug_reports").mkdir(parents=True)
            (root / "data").mkdir(parents=True)
            for name in (
                "pgsm_step5i_string_partial_damping_refinement.json",
                "pgsm_step5h_note_string_fret_contract.json",
                "pgsm_step5g_physical_tone_model_update_plan.json",
                "pgsm_step5f_string_driven_extended_validation.json",
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
            report = write_pgsm_step5i_1_reports(
                repo_root=REPO,
                json_path=root / "audio" / "debug_reports" / "out.json",
                md_path=root / "audio" / "debug_reports" / "out.md",
                data_path=root / "data" / "contract_v2.json",
                audio_dir=root / "audio" / "step5i1_test",
                write_wav=True,
                max_modes=MAX_MODES_TEST,
            )
            self.assertEqual(report.get("report_version"), PGSM_STEP5I_1_VERSION)
            self.assertTrue((root / "data" / "contract_v2.json").is_file())

    def test_upstream_step5i_readiness(self) -> None:
        upstream = self._report().get("upstream_readiness") or {}
        self.assertEqual(upstream.get("step5i_readiness"), READINESS_STEP5I)
        self.assertTrue(upstream.get("step5i_pass"))


if __name__ == "__main__":
    unittest.main()
