#!/usr/bin/env python3
"""PGSM Step 3A — numerical IR testbench tests (no audio, no FEM/ROM, no STK)."""
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
from pgsm_step3a_numerical_ir_testbench import (  # noqa: E402
    L3_BLOCKED_NAMES,
    PGSM_STEP3A_VERSION,
    SAMPLE_ID,
    build_parameter_pack,
    build_pgsm_step3a_report,
    compute_admittance_curve,
    compute_impulse_response,
    compute_modal_weights,
    load_rom_modal_catalog,
    write_pgsm_step3a_reports,
)
from pgsm_step2_1_parameter_targets import load_step_report  # noqa: E402
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE  # noqa: E402
from stk_v6_2_audit_features import load_audit_report  # noqa: E402

REQUIRED_PARAM_KEYS = (
    "name",
    "value",
    "source",
    "fallback_level",
    "units",
    "confidence",
    "allowed_use",
)


class TestPgsmStep3aNumericalIrTestbench(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)
        self.audit = load_audit_report()
        self.rom = load_rom_modal_catalog(REPO / "FEM" / "outputs" / "rom_stk_body.json")

    def tearDown(self) -> None:
        self._td.cleanup()

    def test_website_default_unchanged(self) -> None:
        self.assertEqual(DEFAULT_WEBSITE_STK_MODE, STK_BODY_TRANSFER_FINAL_V1)

    def test_report_files_created(self) -> None:
        report = write_pgsm_step3a_reports(
            repo_root=REPO,
            json_path=self.tmp / "step3a.json",
            md_path=self.tmp / "step3a.md",
            write_figures=False,
        )
        self.assertTrue((self.tmp / "step3a.json").is_file())
        self.assertTrue((self.tmp / "step3a.md").is_file())
        doc = json.loads((self.tmp / "step3a.json").read_text(encoding="utf-8"))
        self.assertEqual(doc["report_version"], PGSM_STEP3A_VERSION)
        self.assertTrue(doc["no_audio_generated"])
        self.assertTrue(doc["no_wav_generated"])
        self.assertIn("does not synthesize musical sound", doc["explicit_statement"])
        self.assertEqual(report["sample_id"], SAMPLE_ID)

    def test_no_wav_files_created(self) -> None:
        write_pgsm_step3a_reports(
            repo_root=REPO,
            json_path=self.tmp / "s.json",
            md_path=self.tmp / "s.md",
            write_figures=False,
        )
        self.assertEqual(list(self.tmp.rglob("*.wav")), [])

    def test_no_fem_rom_subprocess(self) -> None:
        with patch("subprocess.run") as mock_run, patch("subprocess.Popen") as mock_popen:
            build_pgsm_step3a_report(repo_root=REPO, write_figures=False)
            mock_run.assert_not_called()
            mock_popen.assert_not_called()

    def test_no_stk_diagnostic_synthesis_import(self) -> None:
        import pgsm_step3a_numerical_ir_testbench as mod

        src = Path(mod.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import diagnostic_synthesis", src)
        self.assertNotIn("from diagnostic_synthesis", src)

    def test_step1_step2_step21_reports_load(self) -> None:
        report = build_pgsm_step3a_report(repo_root=REPO, write_figures=False)
        self.assertIsNotNone(report.get("step1_report_loaded"))
        self.assertIsNotNone(report.get("step2_report_loaded"))
        self.assertIsNotNone(report.get("step21_report_loaded"))
        load_step_report(REPO / "audio" / "debug_reports" / "pgsm_step1_physical_factor_registry.json")
        load_step_report(REPO / "audio" / "debug_reports" / "pgsm_step2_physical_interaction_map.json")
        load_step_report(REPO / "audio" / "debug_reports" / "pgsm_step2_1_parameter_targets.json")

    def test_parameter_pack_includes_fallback_levels(self) -> None:
        pack = build_parameter_pack(self.audit)
        for section in ("excitation", "string", "geometry_material"):
            for entry in pack.get(section) or []:
                for key in REQUIRED_PARAM_KEYS:
                    self.assertIn(key, entry, msg=f"{section}/{entry.get('name')} missing {key}")
                self.assertTrue(str(entry["fallback_level"]).startswith("L"))

    def test_l3_parameters_blocked_from_computation(self) -> None:
        pack = build_parameter_pack(self.audit)
        blocked = {e["name"] for e in pack.get("L3_blocked_not_used") or []}
        self.assertEqual(blocked, set(L3_BLOCKED_NAMES))
        for e in pack.get("L3_blocked_not_used") or []:
            self.assertEqual(e["allowed_use"], "reported_missing_only")
            self.assertIsNone(e["value"])

    def test_modal_weights_computed(self) -> None:
        pack = build_parameter_pack(self.audit)
        modes = self.rom.get("predicted_modes") or []
        mw = compute_modal_weights(modes[:50], pack)
        self.assertEqual(mw["status"], "ok")
        self.assertGreater(len(mw["modes"]), 0)
        row = mw["modes"][0]
        for key in ("W_exc", "W_rad", "Q_total", "tau_s"):
            self.assertIn(key, row)

    def test_admittance_curve_computed(self) -> None:
        pack = build_parameter_pack(self.audit)
        mw = compute_modal_weights(self.rom.get("predicted_modes")[:80], pack)
        adm = compute_admittance_curve(mw)
        self.assertEqual(adm["status"], "ok")
        self.assertGreater(len(adm["frequency_hz"]), 0)
        self.assertGreater(adm["max_abs_Y"], 0)

    def test_impulse_response_numeric_summary_only(self) -> None:
        pack = build_parameter_pack(self.audit)
        mw = compute_modal_weights(self.rom.get("predicted_modes")[:80], pack)
        ir = compute_impulse_response(mw)
        self.assertEqual(ir["status"], "ok")
        self.assertTrue(ir["not_audio_output"])
        self.assertIn("h_downsampled", ir)
        self.assertLessEqual(len(ir["h_downsampled"]), 200)

    def test_h_causal_from_t0(self) -> None:
        pack = build_parameter_pack(self.audit)
        mw = compute_modal_weights(self.rom.get("predicted_modes")[:100], pack)
        ir = compute_impulse_response(mw)
        self.assertAlmostEqual(ir["h_at_t0"], 0.0, places=5)

    def test_no_delayed_body_tail_artifact(self) -> None:
        report = build_pgsm_step3a_report(repo_root=REPO, write_figures=False)
        ag = report["artifact_guard_results"]
        self.assertTrue(ag.get("independent_body_tail_forbidden"))
        self.assertTrue(ag.get("delayed_body_event_risk_false"))

    def test_no_end_rise(self) -> None:
        report = build_pgsm_step3a_report(repo_root=REPO, write_figures=False)
        decay = report["objective_test_results"]["decay"]
        self.assertTrue(decay.get("no_artificial_end_rise"))

    def test_damping_sensitivity_monotonic(self) -> None:
        report = build_pgsm_step3a_report(repo_root=REPO, write_figures=False)
        d = report["sensitivity_results"]["damping_scale"]
        self.assertTrue(d["pass"])
        self.assertLess(d["tau_at_1p3x_damping"], d["tau_at_1p0x"])
        self.assertLess(d["tau_at_1p0x"], d["tau_at_0p7x_damping"])

    def test_mobility_sensitivity_monotonic(self) -> None:
        report = build_pgsm_step3a_report(repo_root=REPO, write_figures=False)
        m = report["sensitivity_results"]["bridge_mobility"]
        self.assertTrue(m["pass"])
        self.assertLess(m["max_Y_at_0p8x"], m["max_Y_at_1p0x"])
        self.assertLess(m["max_Y_at_1p0x"], m["max_Y_at_1p2x"])

    def test_helmholtz_sensitivity_monotonic(self) -> None:
        report = build_pgsm_step3a_report(repo_root=REPO, write_figures=False)
        h = report["sensitivity_results"]["helmholtz_L_eff"]
        self.assertTrue(h["pass"])
        self.assertGreater(h["f_H_low_L"], h["f_H_nominal"])
        self.assertGreater(h["f_H_nominal"], h["f_H_high_L"])

    def test_pluck_position_changes_excitation(self) -> None:
        report = build_pgsm_step3a_report(repo_root=REPO, write_figures=False)
        p = report["sensitivity_results"]["pluck_position_ratio"]
        self.assertTrue(p["pass"])

    def test_readiness_does_not_allow_musical_wav(self) -> None:
        report = build_pgsm_step3a_report(repo_root=REPO, write_figures=False)
        rg = report["readiness_after_step3a"]
        self.assertFalse(rg["musical_wav_synthesis_allowed"])
        self.assertFalse(rg["stk_integration_allowed"])
        blocked = " ".join(report.get("blocked_next_steps") or []).lower()
        self.assertIn("wav", blocked)

    def test_objective_and_sensitivity_pass(self) -> None:
        report = build_pgsm_step3a_report(repo_root=REPO, write_figures=False)
        self.assertTrue(report["objective_test_results"]["all_pass"])
        self.assertTrue(report["sensitivity_results"]["all_pass"])


if __name__ == "__main__":
    unittest.main()
