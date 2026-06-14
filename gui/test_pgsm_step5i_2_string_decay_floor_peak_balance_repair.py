#!/usr/bin/env python3
"""PGSM Step 5I.2 — string decay floor and peak balance repair tests."""
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
from pgsm_step5a_limited_note_set_diagnostic_audio import NOTE_SET  # noqa: E402
from pgsm_step5e_string_driven_bridge_force_repair import ENERGY_FIRST_10MS_MAX  # noqa: E402
from pgsm_step5i_1_string_damping_duration_harshness_repair import READINESS_AFTER as READINESS_STEP5I_1  # noqa: E402
from pgsm_step5i_2_string_decay_floor_peak_balance_repair import (  # noqa: E402
    ALLOWED_DURATION_MAX_S,
    ALLOWED_DURATION_MIN_S,
    DEFAULT_DURATION_S,
    HARMONIC_ORDER_EXPONENT,
    PGSM_STEP5I_2_VERSION,
    READINESS_AFTER,
    assess_harmonic_purity_change,
    build_pgsm_step5i_2_report,
    build_partial_tau_summary_v3,
    build_v3_string_bridge_force,
    collect_all_previous_audio_fingerprints,
    compute_decay_metrics,
    compute_v3_partial_tau_k,
    write_pgsm_step5i_2_reports,
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
    "low_partial_long_decay_control",
    "late_decay_leakage_proxy",
    "high_note_peak_balance_proxy",
    "combined_partial_tau",
)


class TestPgsmStep5i2StringDecayFloorPeakBalanceRepair(unittest.TestCase):
    _shared_report: dict | None = None
    _shared_wav_dir: Path | None = None

    @classmethod
    def setUpClass(cls) -> None:
        cls._shared_wav_dir = REPO / "audio" / "pgsm_step5i_2_string_decay_floor_peak_balance_repair"
        cls._shared_report = build_pgsm_step5i_2_report(
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

    def test_step5i_1_report_loads(self) -> None:
        self.assertIsNotNone(self._report().get("step5i_1_loaded"))

    def test_step5h_contract_loads(self) -> None:
        self.assertIsNotNone(self._report().get("step5h_loaded"))

    def test_no_stk_integration(self) -> None:
        import pgsm_step5i_2_string_decay_floor_peak_balance_repair as mod

        src = Path(mod.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import diagnostic_synthesis", src)
        self.assertTrue(self._report().get("no_stk_integration"))

    def test_no_fem_rom_subprocess(self) -> None:
        with patch("subprocess.run") as mock_run, patch("subprocess.Popen") as mock_popen:
            build_pgsm_step5i_2_report(
                repo_root=REPO,
                audio_dir=REPO / "audio" / "_tmp_step5i2_subproc",
                write_wav=False,
                max_modes=MAX_MODES_TEST,
            )
            mock_run.assert_not_called()
            mock_popen.assert_not_called()

    def test_website_default_unchanged(self) -> None:
        report = self._report()
        self.assertEqual(report.get("website_default"), DEFAULT_WEBSITE_STK_MODE)
        self.assertTrue(report.get("website_default_unchanged"))

    def test_previous_audio_preserved(self) -> None:
        after = collect_all_previous_audio_fingerprints(REPO)
        self.assertEqual(self._prev_fp, after)
        self.assertTrue(self._report().get("no_previous_audio_modified"))

    def test_four_v3_wavs_generated(self) -> None:
        out = self._shared_wav_dir
        assert out is not None
        for note in NOTE_SET:
            self.assertTrue((out / f"sample_000_{note}_damping_v3_diagnostic.wav").is_file())

    def test_stems_generated(self) -> None:
        out = self._shared_wav_dir
        assert out is not None
        for note in NOTE_SET:
            self.assertTrue((out / f"sample_000_{note}_string_force_stem.wav").is_file())
            self.assertTrue((out / f"sample_000_{note}_body_stem.wav").is_file())

    def test_duration_in_allowed_range(self) -> None:
        obj = self._report().get("objective_test_results") or {}
        self.assertTrue(obj.get("duration_in_range"))
        applied = (self._report().get("duration_policy") or {}).get("applied_duration_s")
        self.assertGreaterEqual(applied, ALLOWED_DURATION_MIN_S)
        self.assertLessEqual(applied, ALLOWED_DURATION_MAX_S)
        self.assertAlmostEqual(applied, DEFAULT_DURATION_S)

    def test_v3_contract_terms(self) -> None:
        contract = self._report().get("string_partial_damping_contract_v3") or {}
        names = {t.get("term") for t in contract.get("terms") or []}
        for term in REQUIRED_TERMS:
            self.assertIn(term, names)

    def test_high_harmonics_decay_faster(self) -> None:
        summary = build_partial_tau_summary_v3(string_id="string_5", fret=0, f0=110.0)
        self.assertTrue(summary.get("high_harmonics_decay_faster_than_low"))
        t1 = compute_v3_partial_tau_k(1, 110.0, string_id="string_5", fret=0)[0]
        t10 = compute_v3_partial_tau_k(10, 1100.0, string_id="string_5", fret=0)[0]
        self.assertGreater(t1, t10)

    def test_A2_A3_measurable_decay(self) -> None:
        obj = self._report().get("objective_test_results") or {}
        self.assertTrue(obj.get("A2_A3_measurable_decay"))
        for note in ("A2", "A3"):
            m = (self._report().get("per_note_metrics") or {}).get(note) or {}
            self.assertTrue(m.get("measurable_decay_over_window"))

    def test_A4_E5_decay_faster_than_A2(self) -> None:
        obj = self._report().get("objective_test_results") or {}
        self.assertTrue(obj.get("A4_E5_decay_faster_than_A2"))

    def test_decay_metrics_computed(self) -> None:
        decay = self._report().get("per_note_decay_metrics") or {}
        for note in NOTE_SET:
            d = decay.get(note) or {}
            self.assertIn("final_0p5s_to_initial_0p5s_energy_ratio", d)
            self.assertIn("t_minus_20_db_ms", d)
            self.assertIn("minus_20db_honestly_flagged_not_reached", d)
            self.assertIn("minus_40db_honestly_flagged_not_reached", d)

    def test_minus_20_reached_or_honestly_flagged(self) -> None:
        obj = self._report().get("objective_test_results") or {}
        self.assertTrue(obj.get("minus_20db_reached_or_flagged"))

    def test_peak_harshness_proxies_computed(self) -> None:
        for note in NOTE_SET:
            m = (self._report().get("per_note_metrics") or {}).get(note) or {}
            self.assertIn("high_note_piercing_proxy", m)
            self.assertIn("upper_mid_dominance_proxy", m)
            self.assertIn("high_vs_low_decay_slope_ratio", m)

    def test_A4_E5_peak_improved_or_flagged(self) -> None:
        obj = self._report().get("objective_test_results") or {}
        self.assertTrue(obj.get("A4_E5_peak_improved_or_flagged"))

    def test_hnr_logic_honest(self) -> None:
        inc = assess_harmonic_purity_change(50.0, 55.0)
        self.assertFalse(inc["harmonic_purity_reduced"])
        self.assertTrue(inc["not_improved_flag"])
        comp = self._report().get("comparison_vs_step5i_1") or {}
        for note in NOTE_SET:
            c = comp.get(note) or {}
            if c.get("harmonic_purity_not_improved_flag"):
                self.assertFalse(c.get("harmonic_purity_reduced"))

    def test_no_forbidden_artifacts(self) -> None:
        art = self._report().get("artifact_guard_results") or {}
        self.assertTrue(art.get("pass"))
        for note in NOTE_SET:
            m = (self._report().get("per_note_metrics") or {}).get(note) or {}
            self.assertTrue(m.get("no_second_onset"))
            self.assertTrue(m.get("no_end_rise"))
            self.assertTrue(m.get("no_hard_gate"))

    def test_pitch_salience_and_active_duration(self) -> None:
        obj = self._report().get("objective_test_results") or {}
        self.assertTrue(obj.get("pitch_salience_all_notes"))
        for note in ("A2", "A3", "A4"):
            d = (self._report().get("per_note_decay_metrics") or {}).get(note) or {}
            self.assertGreater(d.get("active_duration_minus_60_dbfs_ms", 0), 1000.0)

    def test_energy_first_10ms_below_threshold(self) -> None:
        for note in NOTE_SET:
            m = (self._report().get("per_note_metrics") or {}).get(note) or {}
            self.assertLess(m.get("energy_first_10ms"), ENERGY_FIRST_10MS_MAX)

    def test_gain_separate_from_physics(self) -> None:
        for note in NOTE_SET:
            m = (self._report().get("per_note_metrics") or {}).get(note) or {}
            self.assertTrue(m.get("gain_separate_from_physics"))

    def test_readiness_diagnostic_only(self) -> None:
        rg = self._report().get("readiness_after_step5i_2") or {}
        self.assertEqual(rg.get("current_status"), READINESS_AFTER)
        self.assertFalse(rg.get("stk_integration_allowed"))

    def test_objective_all_pass(self) -> None:
        self.assertTrue((self._report().get("objective_test_results") or {}).get("all_pass"))

    def test_v3_force_with_leakage(self) -> None:
        n = int(DEFAULT_DURATION_S * NUMERIC_SR)
        y, meta = build_v3_string_bridge_force(n, NUMERIC_SR, 110.0, string_id="string_5", fret=0)
        self.assertTrue(meta.get("late_decay_leakage_applied"))
        self.assertGreater(HARMONIC_ORDER_EXPONENT, 1.18)
        tail = float(abs(y[-n // 10:]).max())
        mid = float(abs(y[n // 4 : n // 2]).max())
        self.assertLess(tail, mid)

    def test_write_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "audio" / "debug_reports").mkdir(parents=True)
            (root / "data").mkdir(parents=True)
            for name in (
                "pgsm_step5i_1_string_damping_duration_harshness_repair.json",
                "pgsm_step5i_string_partial_damping_refinement.json",
                "pgsm_step5h_note_string_fret_contract.json",
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
            report = write_pgsm_step5i_2_reports(
                repo_root=REPO,
                json_path=root / "audio" / "debug_reports" / "out.json",
                md_path=root / "audio" / "debug_reports" / "out.md",
                data_path=root / "data" / "contract_v3.json",
                audio_dir=root / "audio" / "step5i2_test",
                write_wav=True,
                max_modes=MAX_MODES_TEST,
            )
            self.assertEqual(report.get("report_version"), PGSM_STEP5I_2_VERSION)


if __name__ == "__main__":
    unittest.main()
