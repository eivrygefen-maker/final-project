#!/usr/bin/env python3
"""PGSM Step 3C — numeric calibration tests (no audio, no FEM/ROM, no STK)."""
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
from pgsm_step2_1_parameter_targets import load_step_report  # noqa: E402
from pgsm_step3c_numeric_calibration import (  # noqa: E402
    PGSM_STEP3C_VERSION,
    Q_MAX_CALIBRATED,
    Q_MIN_CALIBRATED,
    REGION_BOUNDS,
    apply_material_policy_sample,
    build_pgsm_step3c_report,
    calibrate_q_tau_modes,
    load_fem_woods_ortho,
    load_pgsm_library,
    normalize_admittance_output,
    project_region_weights,
    verify_damping_monotonicity,
    write_pgsm_step3c_reports,
)
from pgsm_physical_factor_registry import load_audit_report  # noqa: E402
from pgsm_step3a_numerical_ir_testbench import compute_admittance_curve, compute_modal_weights, load_rom_modal_catalog  # noqa: E402
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE  # noqa: E402

MAX_MODES_TEST = 100


class TestPgsmStep3cNumericCalibration(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)
        self.audit = load_audit_report()
        self.fem = load_fem_woods_ortho(REPO / "FEM" / "materials" / "woods_ortho.json")
        self.pgsm = load_pgsm_library(REPO / "data" / "pgsm_tonewood_material_library.json")
        self._report_cache: dict | None = None

    def tearDown(self) -> None:
        self._td.cleanup()

    def _report(self) -> dict:
        if self._report_cache is None:
            self._report_cache = build_pgsm_step3c_report(repo_root=REPO, max_modes=MAX_MODES_TEST)
        return self._report_cache

    def test_website_default_unchanged(self) -> None:
        self.assertEqual(DEFAULT_WEBSITE_STK_MODE, STK_BODY_TRANSFER_FINAL_V1)

    def test_report_files_created(self) -> None:
        report = write_pgsm_step3c_reports(
            repo_root=REPO,
            json_path=self.tmp / "step3c.json",
            md_path=self.tmp / "step3c.md",
        )
        self.assertTrue((self.tmp / "step3c.json").is_file())
        doc = json.loads((self.tmp / "step3c.json").read_text(encoding="utf-8"))
        self.assertEqual(doc["report_version"], PGSM_STEP3C_VERSION)
        self.assertTrue(doc["no_wav_generated"])
        self.assertIn("does not synthesize sound", doc["explicit_statement"])

    def test_no_wav_files_created(self) -> None:
        write_pgsm_step3c_reports(
            repo_root=REPO,
            json_path=self.tmp / "s.json",
            md_path=self.tmp / "s.md",
        )
        self.assertEqual(list(self.tmp.rglob("*.wav")), [])

    def test_no_fem_rom_subprocess(self) -> None:
        with patch("subprocess.run") as mock_run, patch("subprocess.Popen") as mock_popen:
            build_pgsm_step3c_report(repo_root=REPO, max_modes=MAX_MODES_TEST)
            mock_run.assert_not_called()
            mock_popen.assert_not_called()

    def test_no_stk_integration(self) -> None:
        import pgsm_step3c_numeric_calibration as mod

        src = Path(mod.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import diagnostic_synthesis", src)

    def test_step3b_and_step22b_reports_load(self) -> None:
        report = self._report()
        self.assertIsNotNone(report.get("step3b_report_loaded"))
        self.assertIsNotNone(report.get("step22b_report_loaded"))
        self.assertEqual(
            report.get("step22b_policy_loaded"),
            "use_fem_values_as_primary_for_pgsm_calibration",
        )

    def test_written_json_includes_step22b_policy(self) -> None:
        write_pgsm_step3c_reports(
            repo_root=REPO,
            json_path=self.tmp / "contract.json",
            md_path=self.tmp / "contract.md",
        )
        doc = json.loads((self.tmp / "contract.json").read_text(encoding="utf-8"))
        self.assertEqual(
            doc.get("step22b_policy_loaded"),
            "use_fem_values_as_primary_for_pgsm_calibration",
        )
        md = (self.tmp / "contract.md").read_text(encoding="utf-8")
        self.assertIn(
            "Step 2.2b material policy loaded: use_fem_values_as_primary_for_pgsm_calibration",
            md,
        )

    def test_fem_primary_material_policy(self) -> None:
        mat = apply_material_policy_sample(self.audit, self.fem, self.pgsm)
        et = mat["top"]["fields"]["young_modulus_tangential_gpa"]
        self.assertEqual(et["chosen_source"], "FEM_primary")
        self.assertTrue(mat["pgsm_override_blocked"])

    def test_pgsm_does_not_override_fem(self) -> None:
        mat = apply_material_policy_sample(self.audit, self.fem, self.pgsm)
        for key in ("density_kg_m3", "young_modulus_longitudinal_gpa", "young_modulus_tangential_gpa"):
            self.assertEqual(mat["top"]["fields"][key]["chosen_source"], "FEM_primary")

    def test_E_T_mismatch_fem_primary(self) -> None:
        mat = apply_material_policy_sample(self.audit, self.fem, self.pgsm)
        et = mat["top"]["fields"]["young_modulus_tangential_gpa"]
        diffs = mat["top"].get("material_differences_vs_pgsm") or []
        et_diff = any(d["field"] == "young_modulus_tangential_gpa" for d in diffs)
        self.assertTrue(et_diff or et["fem_value"] != et.get("pgsm_typical"))
        self.assertEqual(et["chosen_source"], "FEM_primary")

    def test_Q_calibration_increases_and_bounded(self) -> None:
        rom = load_rom_modal_catalog()
        from pgsm_step3a_numerical_ir_testbench import build_parameter_pack

        pack = build_parameter_pack(self.audit)
        raw = compute_modal_weights(rom["predicted_modes"][:MAX_MODES_TEST], pack)
        mat = apply_material_policy_sample(self.audit, self.fem, self.pgsm)
        cal_modes, summary = calibrate_q_tau_modes(raw["modes"], mat)
        self.assertTrue(summary["mean_Q_increase"])
        self.assertGreater(summary["after"]["mean_Q"], summary["before"]["mean_Q"])
        for m in cal_modes:
            self.assertGreaterEqual(m["Q_calibrated"], Q_MIN_CALIBRATED)
            self.assertLessEqual(m["Q_calibrated"], Q_MAX_CALIBRATED)
            self.assertGreaterEqual(m["tau_s"], m["tau_raw_s"])

    def test_damping_monotonicity(self) -> None:
        rom = load_rom_modal_catalog()
        from pgsm_step3a_numerical_ir_testbench import build_parameter_pack

        pack = build_parameter_pack(self.audit)
        mat = apply_material_policy_sample(self.audit, self.fem, self.pgsm)
        mono = verify_damping_monotonicity(
            rom["predicted_modes"][:80], pack, mat
        )
        self.assertTrue(mono["pass"], mono)
        by = mono["by_damping_scale"]
        low, nom, high = by["low_damping"], by["nominal"], by["high_damping"]
        self.assertGreater(low["mean_Q_clamped"], nom["mean_Q_clamped"])
        self.assertGreater(nom["mean_Q_clamped"], high["mean_Q_clamped"])
        self.assertGreater(low["mean_tau_s_clamped"], nom["mean_tau_s_clamped"])
        self.assertGreater(nom["mean_tau_s_clamped"], high["mean_tau_s_clamped"])
        self.assertTrue(mono["Q_monotonic_unclamped"])
        self.assertTrue(mono["tau_monotonic_unclamped"])

    def test_region_weights_sum_and_bounds(self) -> None:
        reg = project_region_weights(0.214, 0.766, 0.019)
        cal = reg["calibrated"]
        self.assertAlmostEqual(cal["top"] + cal["back"] + cal["air"], 1.0, places=3)
        self.assertTrue(reg["within_bounds"])
        t_lo, t_hi = REGION_BOUNDS["top"]
        self.assertGreaterEqual(cal["top"], t_lo)
        self.assertLessEqual(cal["top"], t_hi)

    def test_raw_region_imbalance_reported(self) -> None:
        reg = project_region_weights(0.214, 0.766, 0.019)
        self.assertTrue(reg["raw_imbalance_flag"])
        self.assertTrue(reg["not_measured_radiation"])

    def test_normalized_admittance_max_one(self) -> None:
        rom = load_rom_modal_catalog()
        from pgsm_step3a_numerical_ir_testbench import build_parameter_pack

        pack = build_parameter_pack(self.audit)
        mw = compute_modal_weights(rom["predicted_modes"][:80], pack)
        adm = compute_admittance_curve(mw)
        norm = normalize_admittance_output(adm)
        self.assertEqual(norm["Y_abs_normalized_peak1_max"], 1.0)
        self.assertTrue(norm["pass"])

    def test_objective_all_pass(self) -> None:
        report = self._report()
        self.assertTrue(report["objective_test_results"]["all_pass"])

    def test_readiness_no_musical_wav_or_stk(self) -> None:
        report = self._report()
        rg = report["readiness_after_step3c"]
        self.assertFalse(rg["musical_wav_synthesis_allowed"])
        self.assertFalse(rg["stk_integration_allowed"])
        self.assertNotEqual(rg["current_status"], "ready_for_musical_audio")

    def test_calibrated_ir_causal_no_end_rise(self) -> None:
        report = self._report()
        ir = report.get("calibrated_ir_summary") or {}
        self.assertEqual(ir.get("h_at_t0"), 0.0)
        self.assertTrue(ir.get("no_artificial_end_rise"))
        self.assertTrue(ir.get("no_delayed_body_event"))

    def test_admittance_peaks_aligned(self) -> None:
        report = self._report()
        adm = report.get("admittance_normalization") or {}
        peaks = adm.get("peaks") or []
        self.assertGreater(len(peaks), 0)
        for pk in peaks[:5]:
            f_pk = float(pk.get("frequency_hz", 0))
            nearest = pk.get("nearest_mode_hz")
            if nearest is not None:
                self.assertLess(abs(f_pk - float(nearest)), 50.0)

    def test_calibrated_region_not_claimed_measured(self) -> None:
        report = self._report()
        reg = report.get("region_weight_calibration") or {}
        self.assertTrue(reg.get("not_measured_radiation"))
        self.assertTrue(reg.get("reference_shared_limitation"))

    def test_safe_next_step_pre_synthesis_contract(self) -> None:
        report = self._report()
        nxt = report["safe_next_step"].lower()
        self.assertIn("step 3d", nxt)
        self.assertIn("pre-synthesis", nxt)
        self.assertTrue("no musical" in nxt or "no stk" in nxt)


if __name__ == "__main__":
    unittest.main()
