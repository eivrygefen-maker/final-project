#!/usr/bin/env python3
"""PGSM Step 5J.1 — fast diagnostic unittest (no WAV, no tracked contract mutation)."""
from __future__ import annotations

import hashlib
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
from pgsm_step5j_top_back_air_radiation_weighting_refinement import READINESS_AFTER as READINESS_STEP5J  # noqa: E402
from pgsm_step5j_1_guitar_articulation_body_balance_repair import (  # noqa: E402
    AIR_CAVITY_MODAL_GAIN,
    FAST_VALIDATION_MAX_MODES,
    PGSM_STEP5J_1_VERSION,
    READINESS_AFTER,
    SOURCE_CONTRACT_JSON,
    build_body_weighting_v2_contract,
    build_pgsm_step5j_1_report,
    collect_all_previous_audio_fingerprints,
    compute_step5j_1_modal_kernels_decomposed,
    radiation_band_weight_v2,
    validate_report_internal_consistency,
    write_pgsm_step5j_1_reports,
)
from pgsm_step5j_top_back_air_radiation_weighting_refinement import (  # noqa: E402
    AIR_CAVITY_MODAL_GAIN as AIR_GAIN_V1,
)
from pgsm_step4a_single_note_diagnostic_audio import build_calibrated_modal_state  # noqa: E402
from pgsm_step5i_3_absolute_frequency_damping_pluck_balance import DEFAULT_DURATION_S  # noqa: E402
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE  # noqa: E402

REQUIRED_TERMS = (
    "top_plate_modal_weight",
    "back_plate_modal_weight",
    "air_cavity_modal_weight",
    "radiation_band_weight",
    "combined_body_radiation_weight",
)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestPgsmStep5j1GuitarArticulationBodyBalanceRepair(unittest.TestCase):
    _shared_report: dict | None = None
    _source_contract_hash: str | None = None

    @classmethod
    def setUpClass(cls) -> None:
        if SOURCE_CONTRACT_JSON.is_file():
            cls._source_contract_hash = _file_sha256(SOURCE_CONTRACT_JSON)
        cls._shared_report = build_pgsm_step5j_1_report(
            repo_root=REPO,
            render_audio=False,
            write_outputs=False,
            fast_validation=True,
        )

    def setUp(self) -> None:
        self._prev_fp = collect_all_previous_audio_fingerprints(REPO)

    def _report(self) -> dict:
        assert self._shared_report is not None
        return self._shared_report

    def test_validation_mode_fast(self) -> None:
        vcfg = self._report().get("validation_config") or {}
        self.assertEqual(vcfg.get("validation_mode"), "fast")
        self.assertFalse(vcfg.get("render_audio"))
        self.assertFalse(vcfg.get("write_outputs"))
        self.assertFalse(vcfg.get("tracked_source_files_modified"))

    def test_validation_max_modes_documented(self) -> None:
        self.assertEqual(self._report().get("validation_max_modes"), FAST_VALIDATION_MAX_MODES)

    def test_tracked_source_contract_unmodified(self) -> None:
        if self._source_contract_hash is None:
            self.skipTest("source contract file missing")
        self.assertEqual(_file_sha256(SOURCE_CONTRACT_JSON), self._source_contract_hash)

    def test_report_internal_consistency(self) -> None:
        check = validate_report_internal_consistency(self._report())
        self.assertTrue(check.get("pass"), msg=str(check.get("issues")))

    def test_step5j_report_loads(self) -> None:
        self.assertIsNotNone(self._report().get("step5j_loaded"))

    def test_step5j_readiness_verified(self) -> None:
        upstream = self._report().get("upstream_readiness") or {}
        self.assertTrue(upstream.get("step5j_pass"))
        self.assertEqual(upstream.get("step5j_readiness"), READINESS_STEP5J)

    def test_step5i_3_report_loads(self) -> None:
        self.assertIsNotNone(self._report().get("step5i_3_loaded"))

    def test_step5h_contract_loads(self) -> None:
        self.assertIsNotNone(self._report().get("step5h_loaded"))

    def test_no_stk_integration(self) -> None:
        import pgsm_step5j_1_guitar_articulation_body_balance_repair as mod

        src = Path(mod.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import diagnostic_synthesis", src)
        self.assertTrue(self._report().get("no_stk_integration"))

    def test_no_fem_rom_subprocess(self) -> None:
        with patch("subprocess.run") as mock_run, patch("subprocess.Popen") as mock_popen:
            build_pgsm_step5j_1_report(
                repo_root=REPO,
                render_audio=False,
                write_outputs=False,
                fast_validation=True,
            )
            mock_run.assert_not_called()
            mock_popen.assert_not_called()

    def test_website_default_unchanged(self) -> None:
        self.assertEqual(self._report().get("website_default"), DEFAULT_WEBSITE_STK_MODE)
        self.assertTrue(self._report().get("website_default_unchanged"))

    def test_previous_audio_preserved(self) -> None:
        after = collect_all_previous_audio_fingerprints(REPO)
        self.assertEqual(self._prev_fp, after)
        self.assertTrue(self._report().get("no_previous_audio_modified"))

    def test_weighting_v2_contract(self) -> None:
        contract = self._report().get("body_weighting_v2_contract") or {}
        names = {t.get("term") for t in contract.get("terms") or []}
        for term in REQUIRED_TERMS:
            self.assertIn(term, names)
        self.assertTrue(contract.get("string_damping_unchanged"))

    def test_air_gain_reduced_vs_v1(self) -> None:
        self.assertLess(AIR_CAVITY_MODAL_GAIN, AIR_GAIN_V1)

    def test_modal_unchanged(self) -> None:
        obj = self._report().get("objective_test_results") or {}
        self.assertTrue(obj.get("step3c_frequencies_unchanged"))
        self.assertTrue(obj.get("step3c_q_tau_unchanged"))
        self.assertTrue(obj.get("string_damping_unchanged"))

    def test_organ_like_diagnosis_computed(self) -> None:
        organ = self._report().get("organ_like_diagnosis") or {}
        for note in NOTE_SET:
            o = organ.get(note) or {}
            self.assertIn("organ_like_purity_flag", o)
            self.assertIn("air_dominance_flag", o)
            self.assertIn("weak_guitar_articulation_flag", o)
            self.assertIn("top_dominance_ratio", o)
            self.assertIn("back_balance_ratio", o)
            self.assertIn("air_balance_ratio", o)
            self.assertIn("h2_h8_energy_ratio", o)

    def test_comb_echo_diagnostics_present(self) -> None:
        art = self._report().get("artifact_guard_results") or {}
        self.assertIn("comb_echo_score_by_note", art)
        self.assertIn("comb_echo_score_by_stem", art)
        self.assertIn("dominant_comb_echo_stem", art)
        for note in NOTE_SET:
            self.assertIn(note, art.get("comb_echo_score_by_note") or {})
            self.assertIn(note, art.get("comb_echo_score_by_stem") or {})

    def test_e5_radiation_guard_diagnostics_present(self) -> None:
        e5g = self._report().get("E5_radiation_guard_analysis") or {}
        self.assertTrue(e5g.get("applicable"))
        self.assertIn("e5_radiation_guard_applied", e5g)
        self.assertIn("e5_guarded_mode_count", e5g)
        self.assertIn("e5_guard_weight_before_after_summary", e5g)
        self.assertIn("e5_radiation_sum_comb_score_before_guard", e5g)
        self.assertIn("e5_radiation_sum_comb_score_after_guard", e5g)
        self.assertIn("e5_guard_selection_diagnostics", e5g)
        sel = e5g.get("e5_guard_selection_diagnostics") or {}
        self.assertIn("top10_e5_radiation_contributors", sel)
        self.assertIn("candidate_mode_count", sel)
        meta = self._report().get("modal_kernel_meta") or {}
        self.assertIn("e5_radiation_guard_applied", meta)
        self.assertIn("modal_frequency_min_hz", meta)
        self.assertIn("modal_frequency_max_hz", meta)
        self.assertEqual(self._report().get("validation_max_modes"), FAST_VALIDATION_MAX_MODES)

    def test_articulation_metrics_computed(self) -> None:
        art = self._report().get("per_note_articulation_metrics") or {}
        for note in NOTE_SET:
            a = art.get(note) or {}
            self.assertIn("h1_dominance_ratio", a)
            self.assertIn("h2_h8_total_ratio", a)
            self.assertIn("h2_h8_energy_ratio", a)
            self.assertIn("top_plate_attack_share", a)
            self.assertIn("air_cavity_sustain_share", a)
            self.assertIn("transient_to_sustain_ratio", a)

    def test_air_dominance_reduced_or_flagged(self) -> None:
        obj = self._report().get("objective_test_results") or {}
        self.assertTrue(obj.get("air_dominance_reduced_or_flagged"))
        comp = self._report().get("comparison_vs_step5j") or {}
        for note in ("A2", "A3", "E5"):
            c = comp.get(note) or {}
            if c.get("step5j_air_share") is not None:
                self.assertLessEqual(
                    float(c.get("step5j_1_air_share") or 1),
                    float(c.get("step5j_air_share") or 0) + 0.01,
                )

    def test_top_attack_improved_or_flagged(self) -> None:
        obj = self._report().get("objective_test_results") or {}
        self.assertTrue(obj.get("top_attack_improved_or_flagged"))

    def test_E5_peak_analysis(self) -> None:
        e5 = self._report().get("E5_peak_source_analysis") or {}
        self.assertTrue(e5.get("applicable"))
        self.assertIn("dominant_peak_stem", e5)
        obj = self._report().get("objective_test_results") or {}
        self.assertTrue(obj.get("E5_peak_improved_or_flagged"))

    def test_radiation_v2_preserves_mid_harmonics(self) -> None:
        mid = radiation_band_weight_v2(700.0, 0.05, w_rad_median=0.05)
        high = radiation_band_weight_v2(2200.0, 0.05, w_rad_median=0.05)
        self.assertGreater(mid, high * 0.5)

    def test_kernels_causal(self) -> None:
        state = build_calibrated_modal_state(REPO, max_modes=FAST_VALIDATION_MAX_MODES)
        h, _, _, _, _, meta = compute_step5j_1_modal_kernels_decomposed(
            state["modal_weights"], duration_s=DEFAULT_DURATION_S
        )
        self.assertTrue(meta.get("h0_causal_near_zero"))
        self.assertGreater(float(abs(h).max()), 0.0)

    def test_objective_artifact_guard_consistent(self) -> None:
        obj = self._report().get("objective_test_results") or {}
        art = self._report().get("artifact_guard_results") or {}
        self.assertEqual(art.get("pass"), obj.get("artifact_guard_pass"))

    def test_no_forbidden_artifacts(self) -> None:
        art = self._report().get("artifact_guard_results") or {}
        self.assertTrue(art.get("pass"), msg=str(art.get("failed_guard_fields")))

    def test_energy_first_10ms(self) -> None:
        for note in NOTE_SET:
            m = (self._report().get("per_note_metrics") or {}).get(note) or {}
            self.assertLess(m.get("energy_first_10ms"), ENERGY_FIRST_10MS_MAX)

    def test_gain_separate_from_physics(self) -> None:
        for note in NOTE_SET:
            m = (self._report().get("per_note_metrics") or {}).get(note) or {}
            self.assertTrue(m.get("gain_separate_from_physics"))

    def test_readiness_diagnostic_only(self) -> None:
        rg = self._report().get("readiness_after_step5j_1") or {}
        obj = self._report().get("objective_test_results") or {}
        if obj.get("all_pass"):
            self.assertEqual(rg.get("current_status"), READINESS_AFTER)
        else:
            self.assertEqual(rg.get("current_status"), "failed_guitar_articulation_body_balance_repair")
        self.assertFalse(rg.get("stk_integration_allowed"))

    def test_objective_all_pass(self) -> None:
        self.assertTrue((self._report().get("objective_test_results") or {}).get("all_pass"))

    def test_write_reports_to_temp_without_touching_source(self) -> None:
        source_hash_before = _file_sha256(SOURCE_CONTRACT_JSON) if SOURCE_CONTRACT_JSON.is_file() else None
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "audio" / "debug_reports" / "generated_contracts").mkdir(parents=True)
            (root / "data").mkdir(parents=True)
            for name in (
                "pgsm_step5j_top_back_air_radiation_weighting_refinement.json",
                "pgsm_step5i_3_absolute_frequency_damping_pluck_balance.json",
                "pgsm_step5h_note_string_fret_contract.json",
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
            generated = root / "audio" / "debug_reports" / "generated_contracts" / "contract.generated.json"
            report = write_pgsm_step5j_1_reports(
                repo_root=REPO,
                json_path=root / "audio" / "debug_reports" / "out.json",
                md_path=root / "audio" / "debug_reports" / "out.md",
                data_path=generated,
                audio_dir=root / "audio" / "step5j1_test",
                render_audio=False,
                write_outputs=True,
                fast_validation=True,
            )
            self.assertEqual(report.get("report_version"), PGSM_STEP5J_1_VERSION)
            self.assertTrue(generated.is_file())
            self.assertFalse((report.get("validation_config") or {}).get("tracked_source_files_modified"))
        if source_hash_before is not None:
            self.assertEqual(_file_sha256(SOURCE_CONTRACT_JSON), source_hash_before)

    def test_built_contract_matches_source_gains(self) -> None:
        if not SOURCE_CONTRACT_JSON.is_file():
            self.skipTest("source contract file missing")
        source = json.loads(SOURCE_CONTRACT_JSON.read_text(encoding="utf-8"))
        src_gains = (source.get("body_weighting_v2_contract") or {}).get("gains") or {}
        built_gains = build_body_weighting_v2_contract().get("gains") or {}
        for key in ("top_plate_modal_gain", "back_plate_modal_gain", "air_cavity_modal_gain"):
            self.assertEqual(built_gains.get(key), src_gains.get(key))


if __name__ == "__main__":
    unittest.main()
