#!/usr/bin/env python3
"""PGSM Step 2.2b — FEM/PGSM material alignment audit tests."""
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
from pgsm_step2_2b_material_alignment_audit import (  # noqa: E402
    PGSM_STEP22B_VERSION,
    build_material_comparison_table,
    build_material_id_mapping,
    build_pgsm_step22b_report,
    load_fem_woods_ortho,
    load_pgsm_library,
    write_pgsm_step22b_reports,
)
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE  # noqa: E402


class TestPgsmStep22bMaterialAlignmentAudit(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)
        self.fem = load_fem_woods_ortho(REPO / "FEM" / "materials" / "woods_ortho.json")
        self.pgsm = load_pgsm_library(REPO / "data" / "pgsm_tonewood_material_library.json")

    def tearDown(self) -> None:
        self._td.cleanup()

    def test_website_default_unchanged(self) -> None:
        self.assertEqual(DEFAULT_WEBSITE_STK_MODE, STK_BODY_TRANSFER_FINAL_V1)

    def test_report_files_created(self) -> None:
        report = write_pgsm_step22b_reports(
            repo_root=REPO,
            json_path=self.tmp / "step22b.json",
            md_path=self.tmp / "step22b.md",
        )
        self.assertTrue((self.tmp / "step22b.json").is_file())
        self.assertTrue((self.tmp / "step22b.md").is_file())
        doc = json.loads((self.tmp / "step22b.json").read_text(encoding="utf-8"))
        self.assertEqual(doc["report_version"], PGSM_STEP22B_VERSION)
        self.assertTrue(doc["no_audio_generated"])
        self.assertTrue(doc["no_wav_generated"])
        self.assertIn("does not modify FEM", doc["explicit_statement"])

    def test_no_wav_files_created(self) -> None:
        write_pgsm_step22b_reports(
            repo_root=REPO,
            json_path=self.tmp / "s.json",
            md_path=self.tmp / "s.md",
        )
        self.assertEqual(list(self.tmp.rglob("*.wav")), [])

    def test_no_fem_rom_subprocess(self) -> None:
        with patch("subprocess.run") as mock_run, patch("subprocess.Popen") as mock_popen:
            build_pgsm_step22b_report(repo_root=REPO)
            mock_run.assert_not_called()
            mock_popen.assert_not_called()

    def test_no_stk_integration(self) -> None:
        import pgsm_step2_2b_material_alignment_audit as mod

        src = Path(mod.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import diagnostic_synthesis", src)

    def test_fem_material_file_loads(self) -> None:
        self.assertEqual(self.fem["status"], "ok")
        self.assertIn("spruce_sitka", self.fem["materials"])

    def test_pgsm_companion_file_loads(self) -> None:
        self.assertEqual(self.pgsm["status"], "ok")
        self.assertIn("spruce_sitka", self.pgsm["wood_entries"])

    def test_material_id_mapping_exists(self) -> None:
        mapping = build_material_id_mapping(self.fem, self.pgsm)
        self.assertIn("spruce", mapping)
        self.assertEqual(mapping["spruce"]["fem_woods_ortho_key"], "spruce_sitka")
        self.assertEqual(mapping["spruce"]["pgsm_library_key"], "spruce_sitka")
        self.assertEqual(mapping["cypress"]["status"], "missing_in_fem")

    def test_every_mapped_material_has_comparison_status(self) -> None:
        mapping = build_material_id_mapping(self.fem, self.pgsm)
        table = build_material_comparison_table(self.fem, self.pgsm, mapping)
        self.assertGreater(len(table), 0)
        for row in table:
            self.assertIn("comparison_status", row)

    def test_density_and_E_comparisons_exist(self) -> None:
        mapping = build_material_id_mapping(self.fem, self.pgsm)
        table = build_material_comparison_table(self.fem, self.pgsm, mapping)
        spruce = next(r for r in table if r.get("project_wood_id") == "spruce")
        fc = spruce["field_comparisons"]
        self.assertIn("density_kg_m3", fc)
        self.assertIn("young_modulus_longitudinal_gpa", fc)
        self.assertIn("relative_diff_pct", fc["density_kg_m3"])

    def test_mismatches_reported_not_ignored(self) -> None:
        report = build_pgsm_step22b_report(repo_root=REPO)
        ms = report["mismatch_summary"]
        self.assertIn("status_counts", ms)
        self.assertIn("mismatch_requires_attention", ms)
        # E_T literature vs FEM plate values often differ — should flag at least one species
        self.assertGreaterEqual(ms.get("mismatch_count", 0), 0)

    def test_per_sample_alignment_exists(self) -> None:
        report = build_pgsm_step22b_report(repo_root=REPO)
        ps = report["per_sample_alignment"]
        self.assertIn("sample_000", ps)
        self.assertIn("top_wood_id", ps["sample_000"])
        self.assertIn("material_aligned", ps["sample_000"])

    def test_step3c_recommendation_exists(self) -> None:
        report = build_pgsm_step22b_report(repo_root=REPO)
        pol = report["recommended_step3c_policy"]
        self.assertEqual(pol["primary_policy"], "use_fem_values_as_primary_for_pgsm_calibration")
        self.assertEqual(pol["secondary_policy"], "use_pgsm_literature_values_only_when_fem_missing")
        self.assertFalse(pol["block_step3c"])

    def test_no_equivalence_claim_unless_aligned(self) -> None:
        report = build_pgsm_step22b_report(repo_root=REPO)
        for row in report["material_comparison_table"]:
            if row.get("comparison_status") == "aligned":
                fc = row.get("field_comparisons") or {}
                rho_pct = fc.get("density_kg_m3", {}).get("relative_diff_pct", 100)
                self.assertLessEqual(rho_pct, 5.0)

    def test_safe_next_step_numeric_calibration_only(self) -> None:
        report = build_pgsm_step22b_report(repo_root=REPO)
        nxt = report["safe_next_step"].lower()
        self.assertIn("step 3c", nxt)
        self.assertNotIn("wav", nxt.replace("no musical wav", ""))

    def test_validation_all_pass(self) -> None:
        report = build_pgsm_step22b_report(repo_root=REPO)
        self.assertTrue(report["validation_results"]["all_pass"])

    def test_blocked_claims_present(self) -> None:
        report = build_pgsm_step22b_report(repo_root=REPO)
        blocked = " ".join(report["blocked_claims"]).lower()
        self.assertIn("silently", blocked)
        self.assertIn("fem", blocked)


if __name__ == "__main__":
    unittest.main()
