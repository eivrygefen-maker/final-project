#!/usr/bin/env python3
"""PGSM Step 5G — physical tone model update plan tests."""
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
from pgsm_step5e_string_driven_bridge_force_repair import collect_previous_audio_fingerprints  # noqa: E402
from pgsm_step5f_string_driven_extended_validation import (  # noqa: E402
    READINESS_AFTER as READINESS_STEP5F,
    collect_step5e_fingerprints,
)
from pgsm_step5g_physical_tone_model_update_plan import (  # noqa: E402
    PGSM_STEP5G_VERSION,
    READINESS_AFTER,
    build_pgsm_step5g_report,
    write_pgsm_step5g_reports,
)
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE  # noqa: E402


class TestPgsmStep5gPhysicalToneModelUpdatePlan(unittest.TestCase):
    _shared_report: dict | None = None

    @classmethod
    def setUpClass(cls) -> None:
        cls._shared_report = build_pgsm_step5g_report(repo_root=REPO)

    def setUp(self) -> None:
        self._fp = {
            **collect_step5e_fingerprints(REPO),
            **collect_previous_audio_fingerprints(REPO),
        }

    def _report(self) -> dict:
        assert self._shared_report is not None
        return self._shared_report

    def test_step5f_report_loads(self) -> None:
        self.assertIsNotNone(self._report().get("step5f_loaded"))

    def test_step5f_readiness_verified(self) -> None:
        upstream = self._report().get("upstream_readiness") or {}
        self.assertEqual(upstream.get("step5f_readiness"), READINESS_STEP5F)
        self.assertTrue(upstream.get("step5f_pass"))
        self.assertTrue(upstream.get("step5f_all_pass"))

    def test_no_new_wav_generated(self) -> None:
        self.assertTrue(self._report().get("no_audio_generated"))

    def test_no_audio_modified(self) -> None:
        after = {
            **collect_step5e_fingerprints(REPO),
            **collect_previous_audio_fingerprints(REPO),
        }
        self.assertEqual(self._fp, after)
        self.assertTrue(self._report().get("no_audio_modified"))

    def test_no_stk_integration(self) -> None:
        import pgsm_step5g_physical_tone_model_update_plan as mod

        src = Path(mod.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import diagnostic_synthesis", src)
        self.assertTrue(self._report().get("no_stk_integration"))

    def test_no_fem_rom_subprocess(self) -> None:
        with patch("subprocess.run") as mock_run, patch("subprocess.Popen") as mock_popen:
            build_pgsm_step5g_report(repo_root=REPO)
            mock_run.assert_not_called()
            mock_popen.assert_not_called()

    def test_website_default_unchanged(self) -> None:
        report = self._report()
        self.assertEqual(report.get("website_default"), DEFAULT_WEBSITE_STK_MODE)
        self.assertEqual(report.get("website_default"), STK_BODY_TRANSFER_FINAL_V1)
        self.assertTrue(report.get("website_default_unchanged"))

    def test_diagnosis_to_target_mapping_exists(self) -> None:
        diag = self._report().get("diagnosis_to_update_targets") or {}
        self.assertTrue(diag.get("mappings"))
        self.assertTrue(diag.get("all_required_areas_covered"))

    def test_excessive_harmonic_purity_maps_to_string_damping(self) -> None:
        obj = self._report().get("objective_test_results") or {}
        self.assertTrue(obj.get("harmonic_purity_maps_to_damping"))

    def test_weak_cavity_maps_to_weighting(self) -> None:
        obj = self._report().get("objective_test_results") or {}
        self.assertTrue(obj.get("cavity_maps_to_weighting"))

    def test_insufficient_feedback_maps_to_coupling(self) -> None:
        obj = self._report().get("objective_test_results") or {}
        self.assertTrue(obj.get("feedback_maps_to_coupling"))

    def test_shared_ir_maps_to_contract(self) -> None:
        obj = self._report().get("objective_test_results") or {}
        self.assertTrue(obj.get("shared_ir_maps_to_contract"))

    def test_multi_decay_contract_complete(self) -> None:
        obj = self._report().get("objective_test_results") or {}
        self.assertTrue(obj.get("multi_decay_terms_complete"))
        terms = (self._report().get("multi_decay_model_contract") or {}).get("terms") or []
        names = {t.get("term") for t in terms}
        for name in (
            "string_partial_decay",
            "top_plate_decay",
            "back_plate_decay",
            "air_cavity_decay",
            "radiation_decay",
            "bridge_coupling_loss",
            "combined_observed_decay",
        ):
            self.assertIn(name, names)

    def test_no_forbidden_recommendations(self) -> None:
        audit = self._report().get("forbidden_recommendation_audit") or {}
        self.assertTrue(audit.get("no_forbidden_recommendations"))
        obj = self._report().get("objective_test_results") or {}
        self.assertTrue(obj.get("no_forbidden_recommendations"))

    def test_proposed_step5h_contract_exists(self) -> None:
        step5h = self._report().get("proposed_step5h_contract") or {}
        self.assertIn("PGSM Step 5H", step5h.get("step_id", ""))
        self.assertTrue(step5h.get("no_audio_generation"))
        self.assertIn("A2", step5h.get("note_mapping_candidates") or {})

    def test_implementation_order_defined(self) -> None:
        order = self._report().get("implementation_order") or {}
        steps = order.get("ordered_steps") or []
        self.assertEqual(len(steps), 6)
        self.assertEqual(steps[0].get("target"), "note_string_fret_contract_repair")

    def test_readiness_remains_diagnostic_planning_only(self) -> None:
        rg = self._report().get("readiness_after_step5g") or {}
        self.assertFalse(rg.get("final_synthesis_ready"))
        self.assertFalse(rg.get("stk_integration_allowed"))
        self.assertFalse(rg.get("real_guitar_equivalence_allowed"))

    def test_all_objective_tests_pass(self) -> None:
        obj = self._report().get("objective_test_results") or {}
        self.assertTrue(obj.get("all_pass"))

    def test_readiness_after_pass(self) -> None:
        rg = self._report().get("readiness_after_step5g") or {}
        self.assertEqual(rg.get("current_status"), READINESS_AFTER)
        self.assertTrue(rg.get("step5h_contract_repair_allowed"))

    def test_explicit_statement_present(self) -> None:
        stmt = self._report().get("explicit_statement") or ""
        self.assertIn("update plan only", stmt)
        self.assertIn("does not generate audio", stmt)

    def test_write_reports_to_disk(self) -> None:
        td = tempfile.TemporaryDirectory()
        tmp = Path(td.name)
        jpath = tmp / "step5g.json"
        mpath = tmp / "step5g.md"
        report = write_pgsm_step5g_reports(
            repo_root=REPO, json_path=jpath, md_path=mpath
        )
        self.assertTrue(jpath.is_file())
        self.assertTrue(mpath.is_file())
        loaded = json.loads(jpath.read_text(encoding="utf-8"))
        self.assertEqual(loaded.get("report_version"), PGSM_STEP5G_VERSION)
        self.assertEqual(report.get("report_version"), PGSM_STEP5G_VERSION)
        td.cleanup()


if __name__ == "__main__":
    unittest.main()
