#!/usr/bin/env python3
"""PGSM Step 5I.3 — absolute-frequency damping and pluck attack balance tests."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gui"))

from pgsm_step5a_limited_note_set_diagnostic_audio import NOTE_SET  # noqa: E402
from pgsm_step5e_string_driven_bridge_force_repair import ENERGY_FIRST_10MS_MAX  # noqa: E402
from pgsm_step5i_2_string_decay_floor_peak_balance_repair import READINESS_AFTER as READINESS_STEP5I_2  # noqa: E402
from pgsm_step5i_3_absolute_frequency_damping_pluck_balance import (  # noqa: E402
    CLICK_DOMINANCE_MAX,
    DEFAULT_DURATION_S,
    PGSM_STEP5I_3_VERSION,
    PLUCK_ATTACK_ENABLED,
    READINESS_AFTER,
    build_pgsm_step5i_3_report,
    build_partial_tau_summary_v4,
    build_pluck_attack_component,
    build_v4_string_bridge_force,
    collect_all_previous_audio_fingerprints,
    compute_v4_partial_sigma_tau,
    write_pgsm_step5i_3_reports,
)
from pgsm_step3a_numerical_ir_testbench import NUMERIC_SR  # noqa: E402
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE  # noqa: E402

MAX_MODES_TEST = 100
REQUIRED_TERMS = (
    "base_string_decay_by_string_id",
    "harmonic_order_loss",
    "absolute_frequency_loss",
    "frequency_band_loss",
    "fret_contact_loss_proxy",
    "material_internal_loss_proxy",
    "bridge_nut_boundary_loss_proxy",
    "late_decay_leakage_proxy",
    "high_note_peak_balance_proxy",
    "combined_partial_tau",
)


class TestPgsmStep5i3AbsoluteFrequencyDampingPluckBalance(unittest.TestCase):
    _shared_report: dict | None = None
    _shared_wav_dir: Path | None = None

    @classmethod
    def setUpClass(cls) -> None:
        cls._shared_wav_dir = REPO / "audio" / "pgsm_step5i_3_absolute_frequency_damping_pluck_balance"
        cls._shared_report = build_pgsm_step5i_3_report(
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

    def test_step5i_2_report_loads(self) -> None:
        self.assertIsNotNone(self._report().get("step5i_2_loaded"))
        upstream = self._report().get("upstream_readiness") or {}
        self.assertTrue(upstream.get("step5i_2_pass"))

    def test_step5h_contract_loads(self) -> None:
        self.assertIsNotNone(self._report().get("step5h_loaded"))

    def test_no_stk_integration(self) -> None:
        import pgsm_step5i_3_absolute_frequency_damping_pluck_balance as mod

        src = Path(mod.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import diagnostic_synthesis", src)
        self.assertTrue(self._report().get("no_stk_integration"))

    def test_no_fem_rom_subprocess(self) -> None:
        with patch("subprocess.run") as mock_run, patch("subprocess.Popen") as mock_popen:
            build_pgsm_step5i_3_report(
                repo_root=REPO,
                audio_dir=REPO / "audio" / "_tmp_step5i3_subproc",
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

    def test_four_v4_wavs_generated(self) -> None:
        out = self._shared_wav_dir
        assert out is not None
        for note in NOTE_SET:
            self.assertTrue((out / f"sample_000_{note}_damping_v4_diagnostic.wav").is_file())

    def test_stems_generated(self) -> None:
        out = self._shared_wav_dir
        assert out is not None
        for note in NOTE_SET:
            self.assertTrue((out / f"sample_000_{note}_string_force_stem.wav").is_file())
            self.assertTrue((out / f"sample_000_{note}_body_stem.wav").is_file())

    def test_pluck_attack_stems_generated(self) -> None:
        out = self._shared_wav_dir
        assert out is not None
        if PLUCK_ATTACK_ENABLED:
            for note in NOTE_SET:
                self.assertTrue((out / f"sample_000_{note}_pluck_attack_stem.wav").is_file())
        else:
            pluck = self._report().get("pluck_attack_balance_summary") or {}
            self.assertIsNotNone(pluck.get("rationale_if_disabled"))

    def test_duration_5p5_seconds(self) -> None:
        obj = self._report().get("objective_test_results") or {}
        self.assertTrue(obj.get("duration_in_range"))
        applied = (self._report().get("duration_policy") or {}).get("applied_duration_s")
        self.assertAlmostEqual(applied, DEFAULT_DURATION_S)

    def test_v4_contract_terms(self) -> None:
        contract = self._report().get("string_partial_damping_contract_v4") or {}
        names = {t.get("term") for t in contract.get("terms") or []}
        for term in REQUIRED_TERMS:
            self.assertIn(term, names)

    def test_tau_depends_on_absolute_frequency(self) -> None:
        _, tau_low, bd_low = compute_v4_partial_sigma_tau(1, 110.0, string_id="string_5", fret=0)
        _, tau_high, bd_high = compute_v4_partial_sigma_tau(1, 659.0, string_id="string_1", fret=12)
        self.assertGreater(tau_low, tau_high)
        self.assertIn("f_k_hz", bd_low)
        self.assertIn("freq_lin_factor", bd_low)

    def test_h1_tau_cross_note_order(self) -> None:
        prop = self._report().get("decay_proportionality_metrics") or {}
        self.assertTrue(prop.get("A3_H1_tau_lt_A2"))
        self.assertTrue(prop.get("A4_H1_tau_lt_A3"))
        self.assertTrue(prop.get("E5_H1_tau_lt_A4"))
        self.assertTrue(prop.get("cross_note_h1_tau_monotonic"))
        obj = self._report().get("objective_test_results") or {}
        self.assertTrue(obj.get("h1_tau_cross_note_order"))

    def test_high_harmonics_decay_faster(self) -> None:
        summary = build_partial_tau_summary_v4(string_id="string_5", fret=0, f0=110.0)
        self.assertTrue(summary.get("high_harmonics_decay_faster_than_low"))
        _, t1, _ = compute_v4_partial_sigma_tau(1, 110.0, string_id="string_5", fret=0)
        _, t10, _ = compute_v4_partial_sigma_tau(10, 1100.0, string_id="string_5", fret=0)
        self.assertGreater(t1, t10)

    def test_A2_A3_measurable_decay(self) -> None:
        obj = self._report().get("objective_test_results") or {}
        self.assertTrue(obj.get("A2_A3_measurable_decay"))

    def test_A4_E5_decay_faster_than_A2(self) -> None:
        obj = self._report().get("objective_test_results") or {}
        self.assertTrue(obj.get("A4_E5_decay_faster_than_A2"))

    def test_decay_metrics_computed(self) -> None:
        prop = self._report().get("decay_proportionality_metrics") or {}
        for note in NOTE_SET:
            p = (prop.get("per_note") or {}).get(note) or {}
            self.assertIn("h1_tau_contract_s", p)
            self.assertIn("final_0p5s_to_initial_0p5s_energy_ratio", p)

    def test_peak_harshness_proxies_computed(self) -> None:
        for note in NOTE_SET:
            m = (self._report().get("per_note_metrics") or {}).get(note) or {}
            self.assertIn("high_note_piercing_proxy", m)
            self.assertIn("upper_mid_dominance_proxy", m)

    def test_attack_clarity_proxy_computed(self) -> None:
        attack = self._report().get("attack_clarity_analysis") or {}
        self.assertTrue(attack.get("attack_clarity_improved_or_flagged"))
        for note in NOTE_SET:
            m = (self._report().get("per_note_metrics") or {}).get(note) or {}
            self.assertIn("attack_clarity_proxy", m)

    def test_A4_E5_peak_improved_or_flagged(self) -> None:
        obj = self._report().get("objective_test_results") or {}
        self.assertTrue(obj.get("A4_E5_peak_improved_or_flagged"))

    def test_attack_not_click_dominant(self) -> None:
        obj = self._report().get("objective_test_results") or {}
        self.assertTrue(obj.get("attack_not_click_dominant"))
        for note in NOTE_SET:
            m = (self._report().get("per_note_metrics") or {}).get(note) or {}
            self.assertLess(m.get("click_dominance_score"), CLICK_DOMINANCE_MAX)
            self.assertTrue(m.get("not_click_dominant"))

    def test_energy_first_10ms_below_threshold(self) -> None:
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

    def test_gain_separate_from_physics(self) -> None:
        for note in NOTE_SET:
            m = (self._report().get("per_note_metrics") or {}).get(note) or {}
            self.assertTrue(m.get("gain_separate_from_physics"))

    def test_readiness_diagnostic_only(self) -> None:
        rg = self._report().get("readiness_after_step5i_3") or {}
        self.assertEqual(rg.get("current_status"), READINESS_AFTER)
        self.assertFalse(rg.get("stk_integration_allowed"))

    def test_objective_all_pass(self) -> None:
        self.assertTrue((self._report().get("objective_test_results") or {}).get("all_pass"))

    def test_v4_force_with_pluck_attack(self) -> None:
        n = int(DEFAULT_DURATION_S * NUMERIC_SR)
        y, pluck, meta = build_v4_string_bridge_force(
            n, NUMERIC_SR, 110.0, string_id="string_5", fret=0, note="A2"
        )
        self.assertTrue(meta.get("absolute_frequency_damping_applied"))
        self.assertTrue(meta.get("pluck_attack_enabled"))
        self.assertGreater(float(abs(pluck).max()), 0.0)
        e10 = float(abs(y[: int(0.01 * NUMERIC_SR)]).max())
        e_mid = float(abs(y[n // 4 : n // 2]).max())
        self.assertGreater(e10, 0.0)
        self.assertGreater(e_mid, e10 * 0.1)

    def test_pluck_attack_component(self) -> None:
        n = int(DEFAULT_DURATION_S * NUMERIC_SR)
        attack, meta = build_pluck_attack_component(n, NUMERIC_SR, 440.0, note="A4", fret=5)
        self.assertTrue(meta.get("pluck_attack_enabled"))
        self.assertTrue(meta.get("treble_amplitude_scaled"))
        self.assertGreater(float(abs(attack).max()), 0.0)

    def test_write_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "audio" / "debug_reports").mkdir(parents=True)
            (root / "data").mkdir(parents=True)
            for name in (
                "pgsm_step5i_2_string_decay_floor_peak_balance_repair.json",
                "pgsm_step5i_1_string_damping_duration_harshness_repair.json",
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
            report = write_pgsm_step5i_3_reports(
                repo_root=REPO,
                json_path=root / "audio" / "debug_reports" / "out.json",
                md_path=root / "audio" / "debug_reports" / "out.md",
                data_path=root / "data" / "contract_v4.json",
                audio_dir=root / "audio" / "step5i3_test",
                write_wav=True,
                max_modes=MAX_MODES_TEST,
            )
            self.assertEqual(report.get("report_version"), PGSM_STEP5I_3_VERSION)


if __name__ == "__main__":
    unittest.main()
