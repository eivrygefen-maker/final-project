#!/usr/bin/env python3
"""PGSM Step 2.2 — tonewood material library tests (no audio, no FEM/ROM, no STK)."""
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
from pgsm_tonewood_material_library import (  # noqa: E402
    PGSM_LIBRARY_JSON,
    PGSM_STEP22_VERSION,
    PROJECT_WOOD_IDS,
    build_pgsm_step22_report,
    build_wood_entries,
    build_source_reference_registry,
    discover_material_files,
    assess_existing_file_strategy,
    compute_derived_proxies,
    build_parameter_status_proposal,
    write_pgsm_step22_reports,
)
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE  # noqa: E402

REQUIRED_WOOD_IDS = (
    "spruce_sitka",
    "cedar_western",
    "rosewood_indian",
    "mahogany_honduran",
    "maple_hard",
    "cypress_mediterranean",
    "generic_top_wood",
    "generic_back_wood",
)

NUMERIC_FIELDS = (
    "density_kg_m3",
    "young_modulus_longitudinal_gpa",
    "young_modulus_radial_gpa",
    "young_modulus_tangential_gpa",
    "anisotropy_ratio_longitudinal_to_radial",
    "damping_loss_factor",
    "speed_of_sound_longitudinal_m_s",
)


class TestPgsmTonewoodMaterialLibrary(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)

    def tearDown(self) -> None:
        self._td.cleanup()

    def test_website_default_unchanged(self) -> None:
        self.assertEqual(DEFAULT_WEBSITE_STK_MODE, STK_BODY_TRANSFER_FINAL_V1)

    def test_existing_material_files_discovered(self) -> None:
        found = discover_material_files(REPO)
        paths = {f["path"] for f in found}
        self.assertIn("FEM/materials/woods_ortho.json", paths)
        strat = assess_existing_file_strategy(found)
        self.assertEqual(strat["selected_existing_material_file"], "FEM/materials/woods_ortho.json")
        self.assertFalse(strat["extend_existing_file"])
        self.assertTrue(strat["pgsm_companion_file_required"])

    def test_report_files_created(self) -> None:
        report = write_pgsm_step22_reports(
            repo_root=REPO,
            json_path=self.tmp / "step22.json",
            md_path=self.tmp / "step22.md",
        )
        self.assertTrue((self.tmp / "step22.json").is_file())
        self.assertTrue((self.tmp / "step22.md").is_file())
        doc = json.loads((self.tmp / "step22.json").read_text(encoding="utf-8"))
        self.assertEqual(doc["report_version"], PGSM_STEP22_VERSION)
        self.assertTrue(doc["no_audio_generated"])
        self.assertTrue(doc["no_wav_generated"])
        self.assertIn("does not synthesize sound", doc["explicit_statement"])

    def test_library_json_exists_after_report(self) -> None:
        write_pgsm_step22_reports(
            repo_root=REPO,
            json_path=self.tmp / "r.json",
            md_path=self.tmp / "r.md",
        )
        self.assertTrue(PGSM_LIBRARY_JSON.is_file())
        doc = json.loads(PGSM_LIBRARY_JSON.read_text(encoding="utf-8"))
        self.assertIn("_pgsm_companion_notice", doc)
        self.assertIn("wood_entries", doc)

    def test_no_wav_files_created(self) -> None:
        write_pgsm_step22_reports(
            repo_root=REPO,
            json_path=self.tmp / "s.json",
            md_path=self.tmp / "s.md",
        )
        self.assertEqual(list(self.tmp.rglob("*.wav")), [])

    def test_no_fem_rom_subprocess(self) -> None:
        with patch("subprocess.run") as mock_run, patch("subprocess.Popen") as mock_popen:
            build_pgsm_step22_report(repo_root=REPO)
            mock_run.assert_not_called()
            mock_popen.assert_not_called()

    def test_no_stk_integration(self) -> None:
        import pgsm_tonewood_material_library as mod

        src = Path(mod.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import diagnostic_synthesis", src)
        self.assertNotIn("from diagnostic_synthesis", src)

    def test_required_wood_entries_exist(self) -> None:
        woods = build_wood_entries()
        for wid in REQUIRED_WOOD_IDS:
            self.assertIn(wid, woods)

    def test_every_wood_has_required_fields(self) -> None:
        woods = build_wood_entries()
        for wid, w in woods.items():
            for field in NUMERIC_FIELDS:
                self.assertIn(field, w, msg=f"{wid} missing {field}")
                rec = w[field]
                self.assertIn("typical", rec)
                self.assertIn("source_reference_id", rec)

    def test_every_numerical_field_has_source_reference(self) -> None:
        sources = {s["source_reference_id"] for s in build_source_reference_registry()}
        for wid, w in build_wood_entries().items():
            for field in NUMERIC_FIELDS:
                sid = w[field]["source_reference_id"]
                self.assertIn(sid, sources, msg=f"{wid}.{field} bad source {sid}")

    def test_source_reference_registry_exists(self) -> None:
        reg = build_source_reference_registry()
        self.assertGreater(len(reg), 0)
        self.assertTrue(any(s["source_reference_id"] == "SRC_USDA_WHB_2010" for s in reg))

    def test_E_longitudinal_ordering(self) -> None:
        for wid, w in build_wood_entries().items():
            e_l = w["young_modulus_longitudinal_gpa"]["typical"]
            e_r = w["young_modulus_radial_gpa"]["typical"]
            e_t = w["young_modulus_tangential_gpa"]["typical"]
            self.assertGreater(e_l, e_r, msg=wid)
            self.assertGreater(e_l, e_t, msg=wid)

    def test_density_and_damping_positive(self) -> None:
        for w in build_wood_entries().values():
            self.assertGreater(w["density_kg_m3"]["typical"], 0)
            self.assertGreater(w["anisotropy_ratio_longitudinal_to_radial"]["typical"], 1.0)
            self.assertGreater(w["damping_loss_factor"]["typical"], 0)

    def test_derived_proxies_exist(self) -> None:
        woods = build_wood_entries()
        px = compute_derived_proxies(woods["spruce_sitka"])
        for key in (
            "stiffness_to_weight_proxy",
            "speed_of_sound_proxy_m_s",
            "radiation_ratio_proxy",
            "mass_loading_proxy_factor",
        ):
            self.assertIn(key, px)
            self.assertIn("formula", px[key])

    def test_project_wood_ids_map(self) -> None:
        report = build_pgsm_step22_report(repo_root=REPO)
        mapping = report["project_wood_id_mapping"]
        for pid in PROJECT_WOOD_IDS:
            self.assertIn(pid, mapping)
        self.assertTrue(mapping["spruce"]["resolved"])
        self.assertEqual(mapping["spruce"]["library_wood_id"], "spruce_sitka")

    def test_parameter_status_L2_proposal(self) -> None:
        prop = build_parameter_status_proposal()
        before = prop["before_library"]
        after = prop["after_library_with_sources"]
        self.assertEqual(before["top_elastic_moduli"], "L3_blocked")
        self.assertEqual(after["top_elastic_moduli"], "L2_literature_fallback")
        self.assertEqual(after["wood_anisotropy"], "L2_literature_fallback")
        self.assertIn("numeric_calibration", after["allowed_use"])

    def test_exact_material_claims_blocked(self) -> None:
        report = build_pgsm_step22_report(repo_root=REPO)
        blocked = " ".join(report.get("blocked_claims") or []).lower()
        self.assertIn("exact", blocked)
        self.assertIn("multi-guitar", blocked)

    def test_no_direct_audio_gain(self) -> None:
        report = build_pgsm_step22_report(repo_root=REPO)
        self.assertTrue(report["validation_results"]["no_arbitrary_audio_gain_in_proxies"])

    def test_validation_all_pass(self) -> None:
        report = build_pgsm_step22_report(repo_root=REPO)
        self.assertTrue(report["validation_results"]["all_pass"])

    def test_safe_next_step_numeric_calibration_only(self) -> None:
        report = build_pgsm_step22_report(repo_root=REPO)
        nxt = report["safe_next_step"].lower()
        self.assertIn("step 3c", nxt)
        self.assertIn("calibration", nxt)
        self.assertTrue("no musical" in nxt or "no stk" in nxt)

    def test_step3b_readiness_loaded(self) -> None:
        report = build_pgsm_step22_report(repo_root=REPO)
        self.assertEqual(
            report.get("step3b_prior_readiness"),
            "ready_for_step3c_numeric_calibration_only",
        )

    def test_per_sample_mapping_sample_000(self) -> None:
        report = build_pgsm_step22_report(repo_root=REPO)
        s0 = report["per_sample_material_mapping"]["sample_000"]
        self.assertEqual(s0["top_wood_id_project"], "spruce")
        self.assertEqual(s0["back_wood_id_project"], "rosewood")
        self.assertGreater(len(s0["source_reference_ids_used"]), 0)


if __name__ == "__main__":
    unittest.main()
