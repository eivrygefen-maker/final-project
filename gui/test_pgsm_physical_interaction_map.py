#!/usr/bin/env python3
"""PGSM Step 2 — physical interaction map tests (no audio, no FEM/ROM)."""
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
from diagnostic_synthesis import DIAGNOSTIC_MODES  # noqa: E402
from pgsm_physical_factor_registry import (  # noqa: E402
    build_factor_registry,
    write_pgsm_step1_reports,
)
from pgsm_physical_interaction_map import (  # noqa: E402
    EDGE_CATEGORIES,
    FACTOR_TYPES,
    PGSM_STEP2_VERSION,
    build_forbidden_paths,
    build_interaction_graph,
    build_pgsm_step2_report,
    build_weighting_equations,
    classify_all_factors,
    load_step1_registry,
    write_pgsm_step2_reports,
)

from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE  # noqa: E402

STEP1_GROUPS = (
    "string_factors",
    "pluck_factors",
    "bridge_factors",
    "body_modal_factors",
    "geometry_factors",
    "material_factors",
    "cavity_air_factors",
    "radiation_factors",
    "energy_decay_factors",
    "artifact_guard_factors",
)

REJECTED_V6_MODES = (
    "stk_v6_2_physical_routing_alpha",
    "stk_v6_3_clean_pluck_body_alpha",
    "stk_v5_alpha_body_dominant",
)

REQUIRED_WEIGHTING_IDS = (
    "modal_excitation_weight",
    "combined_modal_Q",
    "amplitude_decay_tau",
    "radiation_weight",
    "cavity_air_weight",
    "causal_radiation_sum",
)

FORBIDDEN_PATH_IDS = (
    "independent_body_tail_stem",
    "delayed_helmholtz_ir",
    "delayed_body_onset",
    "post_hoc_echo_layer",
)

CLASSIFICATION_KEYS = (
    "factor_name",
    "factor_type",
    "physical_role",
    "source_status",
    "per_sample",
    "units",
    "allowed_to_drive_multi_guitar_difference",
    "allowed_to_affect_audio_directly",
)


class TestPgsmStep2PhysicalInteractionMap(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)

    def tearDown(self) -> None:
        self._td.cleanup()

    def test_website_default_unchanged(self) -> None:
        self.assertEqual(DEFAULT_WEBSITE_STK_MODE, STK_BODY_TRANSFER_FINAL_V1)

    def test_report_files_created(self) -> None:
        report = write_pgsm_step2_reports(
            repo_root=REPO,
            json_path=self.tmp / "step2.json",
            md_path=self.tmp / "step2.md",
        )
        self.assertTrue((self.tmp / "step2.json").is_file())
        self.assertTrue((self.tmp / "step2.md").is_file())
        doc = json.loads((self.tmp / "step2.json").read_text(encoding="utf-8"))
        self.assertEqual(doc["report_version"], PGSM_STEP2_VERSION)
        self.assertTrue(doc["no_audio_generated"])
        self.assertTrue(doc["no_fem_run"])
        self.assertTrue(doc["no_rom_run"])
        self.assertTrue(doc["website_default_unchanged"])
        self.assertIn("does not synthesize sound", doc["explicit_statement"])
        self.assertEqual(report["status"], "pgsm_step2_interaction_map_complete")

    def test_no_wav_files_created(self) -> None:
        write_pgsm_step2_reports(
            repo_root=REPO,
            json_path=self.tmp / "s2.json",
            md_path=self.tmp / "s2.md",
        )
        self.assertEqual(list(self.tmp.rglob("*.wav")), [])

    def test_no_fem_rom_subprocess(self) -> None:
        with patch("subprocess.run") as mock_run, patch("subprocess.Popen") as mock_popen:
            build_pgsm_step2_report(repo_root=REPO)
            mock_run.assert_not_called()
            mock_popen.assert_not_called()

    def test_all_step1_factor_groups_loaded(self) -> None:
        registry = load_step1_registry()
        for group in STEP1_GROUPS:
            self.assertIn(group, registry)

    def test_every_factor_has_type_classification(self) -> None:
        registry = build_factor_registry()
        classification = classify_all_factors(registry)
        total = sum(len(v) for v in registry.values())
        self.assertEqual(len(classification), total)
        for uid, c in classification.items():
            for key in CLASSIFICATION_KEYS:
                self.assertIn(key, c, msg=f"{uid} missing {key}")
            self.assertIn(c["factor_type"], FACTOR_TYPES)

    def test_every_edge_has_category_and_explanation(self) -> None:
        for e in build_interaction_graph():
            self.assertIn(e["category"], EDGE_CATEGORIES)
            self.assertTrue(e.get("physical_explanation"))

    def test_forbidden_paths_include_v6_artifacts(self) -> None:
        fps = {f["path_id"]: f for f in build_forbidden_paths()}
        for pid in FORBIDDEN_PATH_IDS:
            self.assertIn(pid, fps)
            self.assertIn("forbidden", fps[pid].get("status", ""))

    def test_weighting_equations_exist(self) -> None:
        ids = {e["id"] for e in build_weighting_equations()}
        for wid in REQUIRED_WEIGHTING_IDS:
            self.assertIn(wid, ids)

    def test_causality_routes_through_F_bridge(self) -> None:
        report = build_pgsm_step2_report(repo_root=REPO)
        causality = report["objective_test_results"]["causality"]
        self.assertTrue(causality["body_modal_depends_on_F_bridge"])
        self.assertTrue(causality["independent_body_tail_marked_forbidden"])
        graph = report["interaction_graph"]
        self.assertTrue(any(
            e["from"] == "bridge_factors.bridge_force" and e["to"] == "radiation_factors.causal_radiation_sum"
            for e in graph
        ))

    def test_reference_shared_blocked_from_multi_guitar(self) -> None:
        report = build_pgsm_step2_report(repo_root=REPO)
        mg = report["objective_test_results"]["multi_guitar_guard"]
        self.assertTrue(mg["reference_shared_blocked_from_differentiation"])
        self.assertTrue(report["multi_guitar_limitations"]["differentiation_limited"])

    def test_missing_critical_fields_reported(self) -> None:
        report = build_pgsm_step2_report(repo_root=REPO)
        missing = report["missing_critical_data"]
        for field in ("scale_length", "bridge_position", "elastic_moduli", "anisotropy", "string_tension"):
            self.assertIn(field, missing)

    def test_monotonic_interaction_tests_pass(self) -> None:
        report = build_pgsm_step2_report(repo_root=REPO)
        mono = report["objective_test_results"]["monotonic_interaction"]
        self.assertTrue(mono["all_pass"])

    def test_step3_not_ready_for_synthesis(self) -> None:
        report = build_pgsm_step2_report(repo_root=REPO)
        self.assertIn(report["step3_readiness_status"], (
            "not_ready_for_synthesis",
            "limited_mapping_only_critical_data_missing",
        ))
        self.assertGreater(len(report["blocked_step3_inputs"]), 0)
        self.assertTrue(any("STK" in b or "synthesis" in b.lower() for b in report["blocked_step3_inputs"]))

    def test_rejected_v6_modes_not_registered(self) -> None:
        for mode in REJECTED_V6_MODES:
            self.assertNotIn(mode, DIAGNOSTIC_MODES)

    def test_pgsm_step1_still_regenerates(self) -> None:
        report = write_pgsm_step1_reports(
            repo_root=REPO,
            json_path=self.tmp / "step1.json",
            md_path=self.tmp / "step1.md",
        )
        self.assertTrue((self.tmp / "step1.json").is_file())
        self.assertTrue(report["no_audio_generated"])

    def test_objective_tests_all_pass(self) -> None:
        report = build_pgsm_step2_report(repo_root=REPO)
        self.assertTrue(report["objective_test_results"]["all_pass"])


if __name__ == "__main__":
    unittest.main()
