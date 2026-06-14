#!/usr/bin/env python3
"""PGSM Step 5J — top/back/air/radiation weighting refinement tests."""
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
from pgsm_step5i_3_absolute_frequency_damping_pluck_balance import (  # noqa: E402
    DEFAULT_DURATION_S,
    READINESS_AFTER as READINESS_STEP5I_3,
)
from pgsm_step5j_top_back_air_radiation_weighting_refinement import (  # noqa: E402
    PGSM_STEP5J_VERSION,
    READINESS_AFTER,
    build_pgsm_step5j_report,
    build_top_back_air_radiation_weighting_contract,
    collect_all_previous_audio_fingerprints,
    compute_step5j_modal_kernels_decomposed,
    radiation_band_weight,
    write_pgsm_step5j_reports,
)
from pgsm_step4a_single_note_diagnostic_audio import build_calibrated_modal_state  # noqa: E402
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE  # noqa: E402

MAX_MODES_TEST = 100
REQUIRED_TERMS = (
    "top_plate_modal_weight",
    "back_plate_modal_weight",
    "air_cavity_modal_weight",
    "radiation_band_weight",
    "combined_body_radiation_weight",
)


class TestPgsmStep5jTopBackAirRadiationWeightingRefinement(unittest.TestCase):
    _shared_report: dict | None = None
    _shared_wav_dir: Path | None = None

    @classmethod
    def setUpClass(cls) -> None:
        cls._shared_wav_dir = REPO / "audio" / "pgsm_step5j_top_back_air_radiation_weighting_refinement"
        cls._shared_report = build_pgsm_step5j_report(
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

    def test_step5i_3_report_loads(self) -> None:
        self.assertIsNotNone(self._report().get("step5i_3_loaded"))

    def test_step5i_3_readiness_verified(self) -> None:
        upstream = self._report().get("upstream_readiness") or {}
        self.assertTrue(upstream.get("step5i_3_pass"))
        self.assertEqual(upstream.get("step5i_3_readiness"), READINESS_STEP5I_3)

    def test_step5h_contract_loads(self) -> None:
        self.assertIsNotNone(self._report().get("step5h_loaded"))

    def test_no_stk_integration(self) -> None:
        import pgsm_step5j_top_back_air_radiation_weighting_refinement as mod

        src = Path(mod.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import diagnostic_synthesis", src)
        self.assertTrue(self._report().get("no_stk_integration"))

    def test_no_fem_rom_subprocess(self) -> None:
        with patch("subprocess.run") as mock_run, patch("subprocess.Popen") as mock_popen:
            build_pgsm_step5j_report(
                repo_root=REPO,
                audio_dir=REPO / "audio" / "_tmp_step5j_subproc",
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

    def test_four_body_weighted_wavs_generated(self) -> None:
        out = self._shared_wav_dir
        assert out is not None
        for note in NOTE_SET:
            self.assertTrue((out / f"sample_000_{note}_body_weighted_diagnostic.wav").is_file())

    def test_all_stems_generated(self) -> None:
        out = self._shared_wav_dir
        assert out is not None
        stem_names = (
            "string_force_stem",
            "pluck_attack_stem",
            "top_plate_stem",
            "back_plate_stem",
            "air_cavity_stem",
            "radiation_sum_stem",
            "final_body_weighted_stem",
        )
        for note in NOTE_SET:
            for stem in stem_names:
                self.assertTrue((out / f"sample_000_{note}_{stem}.wav").is_file(), f"missing {note} {stem}")

    def test_weighting_contract_terms(self) -> None:
        contract = self._report().get("top_back_air_radiation_weighting_contract") or {}
        names = {t.get("term") for t in contract.get("terms") or []}
        for term in REQUIRED_TERMS:
            self.assertIn(term, names)
        self.assertTrue(contract.get("modal_frequencies_unchanged"))
        self.assertTrue(contract.get("modal_q_tau_unchanged"))

    def test_modal_frequencies_unchanged(self) -> None:
        obj = self._report().get("objective_test_results") or {}
        self.assertTrue(obj.get("step3c_frequencies_unchanged"))
        meta = self._report().get("modal_kernel_meta") or {}
        self.assertTrue(meta.get("frequencies_unchanged"))
        self.assertTrue(meta.get("q_tau_unchanged"))

    def test_radiation_band_weight_reduces_high_freq(self) -> None:
        low = radiation_band_weight(400.0, 0.05, w_rad_median=0.05)
        high = radiation_band_weight(1800.0, 0.05, w_rad_median=0.05)
        self.assertGreater(low, high)

    def test_decomposed_kernels_causal(self) -> None:
        state = build_calibrated_modal_state(REPO, max_modes=MAX_MODES_TEST)
        h_combined, h_top, h_back, h_air, h_rad, meta = compute_step5j_modal_kernels_decomposed(
            state["modal_weights"], duration_s=DEFAULT_DURATION_S
        )
        self.assertGreater(float(np_max_abs(h_top)), 0.0)
        self.assertGreater(float(np_max_abs(h_back)), 0.0)
        self.assertTrue(meta.get("h0_causal_near_zero"))
        self.assertAlmostEqual(
            float(np_max_abs(h_combined)),
            float(np_max_abs(h_top + h_back + h_air)),
            places=6,
        )

    def test_body_string_ratio_computed(self) -> None:
        stems = self._report().get("per_note_stem_energy_summary") or {}
        for note in NOTE_SET:
            s = stems.get(note) or {}
            self.assertIn("body_to_string_energy_ratio", s)
            self.assertIn("top_share", s)
            self.assertIn("back_share", s)
            self.assertIn("air_share", s)

    def test_cavity_imprint_computed(self) -> None:
        body = self._report().get("per_note_body_identity_metrics") or {}
        for note in NOTE_SET:
            b = body.get(note) or {}
            self.assertIn("cavity_air_imprint_score", b)
            self.assertIn("radiation_balance_score", b)

    def test_E5_peak_source_analysis(self) -> None:
        e5 = self._report().get("E5_peak_source_analysis") or {}
        self.assertTrue(e5.get("applicable"))
        self.assertIn("peak_by_stem_dbfs", e5)
        self.assertIn("dominant_peak_stem", e5)
        obj = self._report().get("objective_test_results") or {}
        self.assertTrue(obj.get("E5_peak_analysis_computed"))

    def test_E5_peak_improved_or_flagged(self) -> None:
        obj = self._report().get("objective_test_results") or {}
        self.assertTrue(obj.get("E5_peak_improved_or_flagged"))
        e5_metrics = (self._report().get("per_note_peak_harshness_analysis") or {}).get("E5") or {}
        self.assertTrue(
            e5_metrics.get("peak_improved_vs_step5i_3") or e5_metrics.get("peak_flagged")
        )

    def test_no_forbidden_artifacts(self) -> None:
        art = self._report().get("artifact_guard_results") or {}
        self.assertTrue(art.get("pass"))
        for note in NOTE_SET:
            m = (self._report().get("per_note_metrics") or {}).get(note) or {}
            self.assertTrue(m.get("no_second_onset"))
            self.assertTrue(m.get("no_end_rise"))
            self.assertTrue(m.get("no_hard_gate"))

    def test_energy_first_10ms_below_threshold(self) -> None:
        for note in NOTE_SET:
            m = (self._report().get("per_note_metrics") or {}).get(note) or {}
            self.assertLess(m.get("energy_first_10ms"), ENERGY_FIRST_10MS_MAX)

    def test_gain_separate_from_physics(self) -> None:
        for note in NOTE_SET:
            m = (self._report().get("per_note_metrics") or {}).get(note) or {}
            self.assertTrue(m.get("gain_separate_from_physics"))

    def test_readiness_diagnostic_only(self) -> None:
        rg = self._report().get("readiness_after_step5j") or {}
        self.assertEqual(rg.get("current_status"), READINESS_AFTER)
        self.assertFalse(rg.get("stk_integration_allowed"))
        self.assertFalse(rg.get("final_synthesis_ready"))

    def test_objective_all_pass(self) -> None:
        self.assertTrue((self._report().get("objective_test_results") or {}).get("all_pass"))

    def test_write_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "audio" / "debug_reports").mkdir(parents=True)
            (root / "data").mkdir(parents=True)
            for name in (
                "pgsm_step5i_3_absolute_frequency_damping_pluck_balance.json",
                "pgsm_step5h_note_string_fret_contract.json",
                "pgsm_step5g_physical_tone_model_update_plan.json",
                "pgsm_step3c_numeric_calibration.json",
            ):
                src = REPO / "audio" / "debug_reports" / name
                if src.is_file():
                    (root / "audio" / "debug_reports" / name).write_text(
                        src.read_text(encoding="utf-8"), encoding="utf-8"
                    )
            cd = REPO / "data" / "pgsm_classical_guitar_note_string_fret_contract.json"
            if cd.is_file():
                (root / "data" / "pgsm_classical_guitar_note_string_fret_contract.json").write_text(
                    cd.read_text(encoding="utf-8"), encoding="utf-8"
                )
            report = write_pgsm_step5j_reports(
                repo_root=REPO,
                json_path=root / "audio" / "debug_reports" / "out.json",
                md_path=root / "audio" / "debug_reports" / "out.md",
                data_path=root / "data" / "contract.json",
                audio_dir=root / "audio" / "step5j_test",
                write_wav=True,
                max_modes=MAX_MODES_TEST,
            )
            self.assertEqual(report.get("report_version"), PGSM_STEP5J_VERSION)


def np_max_abs(arr) -> float:
    import numpy as np

    return float(np.max(np.abs(arr)))


if __name__ == "__main__":
    unittest.main()
