#!/usr/bin/env python3
"""PGSM Step 2.1 — parameter targets tests (no audio, no FEM/ROM)."""
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
from pgsm_step2_1_parameter_targets import (  # noqa: E402
    PGSM_STEP2_1_VERSION,
    READINESS_VALUES,
    build_literature_alignment_checklist,
    build_parameter_target_table,
    build_pgsm_step2_1_report,
    load_step_report,
    write_pgsm_step2_1_reports,
)
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE  # noqa: E402

CRITICAL_PARAMS = (
    "scale_length",
    "bridge_position",
    "string_tension",
    "linear_density",
    "inharmonicity",
    "pluck_duration_ms",
    "top_elastic_moduli",
    "back_elastic_moduli",
    "wood_anisotropy",
    "modal_mass",
    "modal_stiffness",
    "exact_modal_Q",
)

REQUIRED_PARAM_KEYS = (
    "name",
    "physical_role",
    "required_for_step3",
    "required_for_multi_guitar",
    "current_status",
    "source_in_project",
    "proposed_strategy",
    "fallback_level",
    "safe_range",
    "units",
    "confidence",
    "risk_if_wrong",
    "affects",
)

MANDATORY_CHECKLIST_IDS = (
    "bridge_admittance_central",
    "modal_peaks_via_bridge",
    "causal_not_delayed",
    "cavity_coupled_not_echo",
    "radiation_damping_interaction",
)


class TestPgsmStep21ParameterTargets(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)

    def tearDown(self) -> None:
        self._td.cleanup()

    def test_website_default_unchanged(self) -> None:
        self.assertEqual(DEFAULT_WEBSITE_STK_MODE, STK_BODY_TRANSFER_FINAL_V1)

    def test_report_files_created(self) -> None:
        report = write_pgsm_step2_1_reports(
            repo_root=REPO,
            json_path=self.tmp / "step2_1.json",
            md_path=self.tmp / "step2_1.md",
        )
        self.assertTrue((self.tmp / "step2_1.json").is_file())
        self.assertTrue((self.tmp / "step2_1.md").is_file())
        doc = json.loads((self.tmp / "step2_1.json").read_text(encoding="utf-8"))
        self.assertEqual(doc["report_version"], PGSM_STEP2_1_VERSION)
        self.assertTrue(doc["no_audio_generated"])
        self.assertIn("does not synthesize sound", doc["explicit_statement"])
        self.assertEqual(report["status"], "pgsm_step2_1_parameter_targets_complete")

    def test_no_wav_files_created(self) -> None:
        write_pgsm_step2_1_reports(
            repo_root=REPO,
            json_path=self.tmp / "s.json",
            md_path=self.tmp / "s.md",
        )
        self.assertEqual(list(self.tmp.rglob("*.wav")), [])

    def test_no_fem_rom_subprocess(self) -> None:
        with patch("subprocess.run") as mock_run, patch("subprocess.Popen") as mock_popen:
            build_pgsm_step2_1_report(repo_root=REPO)
            mock_run.assert_not_called()
            mock_popen.assert_not_called()

    def test_every_critical_missing_parameter_covered(self) -> None:
        table = build_parameter_target_table()
        names = {p["name"] for p in table}
        for name in CRITICAL_PARAMS:
            self.assertIn(name, names, msg=f"missing parameter {name}")

    def test_every_parameter_has_required_fields(self) -> None:
        for p in build_parameter_target_table():
            for key in REQUIRED_PARAM_KEYS:
                self.assertIn(key, p, msg=f"{p.get('name')} missing {key}")
                self.assertTrue(str(p[key]).strip() != "" if p[key] is not None else False)

    def test_fallback_levels_assigned(self) -> None:
        valid = {"L0_measured", "L1_derived", "L2_literature_fallback", "L3_blocked"}
        for p in build_parameter_target_table():
            self.assertIn(p["fallback_level"], valid)

    def test_unsafe_L3_blocked(self) -> None:
        l3 = [p for p in build_parameter_target_table() if p["fallback_level"] == "L3_blocked"]
        self.assertGreater(len(l3), 0)
        l3_names = {p["name"] for p in l3}
        self.assertIn("top_elastic_moduli", l3_names)
        self.assertIn("wood_anisotropy", l3_names)

    def test_literature_alignment_mandatory_pass(self) -> None:
        checklist = build_literature_alignment_checklist()
        ids = {c["id"]: c for c in checklist}
        for mid in MANDATORY_CHECKLIST_IDS:
            self.assertIn(mid, ids)
            self.assertTrue(ids[mid]["aligned"])
        report = build_pgsm_step2_1_report(repo_root=REPO)
        self.assertTrue(report["literature_alignment_all_pass"])

    def test_readiness_not_multi_guitar(self) -> None:
        report = build_pgsm_step2_1_report(repo_root=REPO)
        rg = report["readiness_gate"]
        self.assertNotEqual(rg["current_status"], "ready_for_multi_guitar_comparison")
        self.assertFalse(rg["multi_guitar_comparison_allowed"])

    def test_readiness_blocks_musical_audio(self) -> None:
        report = build_pgsm_step2_1_report(repo_root=REPO)
        rg = report["readiness_gate"]
        self.assertFalse(rg["musical_audio_synthesis_allowed"])
        blocked = " ".join(rg.get("blocked_now") or []).lower()
        self.assertIn("wav", blocked)

    def test_safe_next_step_is_numerical_ir_only(self) -> None:
        report = build_pgsm_step2_1_report(repo_root=REPO)
        nxt = report["safe_next_step"].lower()
        self.assertIn("impulse", nxt)
        self.assertIn("no", nxt)
        self.assertIn("stk", nxt)
        rg = report["readiness_gate"]
        self.assertIn(rg["current_status"], READINESS_VALUES)
        self.assertEqual(rg["current_status"], "ready_for_numerical_impulse_response_only")

    def test_sensitivity_plan_passes(self) -> None:
        report = build_pgsm_step2_1_report(repo_root=REPO)
        self.assertTrue(report["sensitivity_plan"]["tests"]["all_pass"])

    def test_step1_and_step2_reports_load(self) -> None:
        report = build_pgsm_step2_1_report(repo_root=REPO)
        self.assertIsNotNone(report.get("step1_report_loaded"))
        self.assertIsNotNone(report.get("step2_report_loaded"))
        load_step_report(REPO / "audio" / "debug_reports" / "pgsm_step1_physical_factor_registry.json")
        load_step_report(REPO / "audio" / "debug_reports" / "pgsm_step2_physical_interaction_map.json")

    def test_blocked_claims_present(self) -> None:
        report = build_pgsm_step2_1_report(repo_root=REPO)
        self.assertGreater(len(report["blocked_claims"]), 0)


if __name__ == "__main__":
    unittest.main()
