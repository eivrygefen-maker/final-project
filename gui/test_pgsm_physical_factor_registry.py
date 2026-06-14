#!/usr/bin/env python3
"""PGSM Step 1 — physical factor registry tests (no audio, no FEM/ROM)."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gui"))

from body_hybrid_v4_1_identity_space import STK_BODY_TRANSFER_FINAL_V1  # noqa: E402
from pgsm_physical_factor_registry import (  # noqa: E402
    PGSM_STEP1_VERSION,
    build_artifact_guard_rules,
    build_equations_registry,
    build_factor_registry,
    build_pgsm_step1_report,
    write_pgsm_step1_reports,
)
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE  # noqa: E402

REQUIRED_GROUPS = (
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

REQUIRED_EQUATION_IDS = (
    "string_harmonics",
    "pluck_harmonic_shape",
    "helmholtz_proxy",
    "modal_oscillator",
    "damping_from_Q",
    "amplitude_tau",
    "bridge_admittance",
    "radiation_sum",
    "combined_Q",
)

REQUIRED_FACTOR_KEYS = ("units", "data_source_path", "availability", "per_sample", "confidence")


class TestPgsmStep1PhysicalFactorRegistry(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)
        self.json_path = self.tmp / "pgsm_step1.json"
        self.md_path = self.tmp / "pgsm_step1.md"

    def tearDown(self) -> None:
        self._td.cleanup()

    def test_website_default_unchanged(self) -> None:
        self.assertEqual(DEFAULT_WEBSITE_STK_MODE, STK_BODY_TRANSFER_FINAL_V1)
        self.assertEqual(DEFAULT_WEBSITE_STK_MODE, "stk_body_transfer_final_v1")

    def test_all_required_factor_groups_exist(self) -> None:
        registry = build_factor_registry()
        for group in REQUIRED_GROUPS:
            self.assertIn(group, registry, msg=f"missing group {group}")
            self.assertGreater(len(registry[group]), 0)

    def test_every_factor_has_required_metadata(self) -> None:
        registry = build_factor_registry()
        for group, factors in registry.items():
            for fname, frec in factors.items():
                for key in REQUIRED_FACTOR_KEYS:
                    self.assertIn(key, frec, msg=f"{group}.{fname} missing {key}")
                    self.assertTrue(
                        frec[key] is not None and str(frec[key]).strip() != "",
                        msg=f"{group}.{fname}.{key} empty",
                    )

    def test_required_equations_present(self) -> None:
        eqs = build_equations_registry()
        ids = {e["id"] for e in eqs}
        for eid in REQUIRED_EQUATION_IDS:
            self.assertIn(eid, ids)

    def test_monotonic_helmholtz_and_damping_sanity(self) -> None:
        report = build_pgsm_step1_report(repo_root=REPO)
        mono = report["monotonic_sanity_results"]
        self.assertTrue(mono["volume_up_lowers_helmholtz"]["pass"])
        self.assertTrue(mono["soundhole_up_raises_helmholtz"]["pass"])
        self.assertTrue(mono["damping_up_lowers_Q"]["pass"])
        self.assertTrue(mono["damping_up_lowers_tau"]["pass"])
        self.assertTrue(mono["mass_up_lowers_mobility"]["pass"])
        self.assertTrue(mono["coupling_up_raises_excitation"]["pass"])
        self.assertTrue(mono["radiation_up_can_lower_Q"]["pass"])
        self.assertTrue(mono["all_pass"])

    def test_reference_shared_not_safe_for_multi_guitar(self) -> None:
        report = build_pgsm_step1_report(repo_root=REPO)
        ref = report["data_mapping_by_sample"]["reference_shared_features"]
        self.assertGreater(len(ref), 0)
        blocked = report["blocked_step2_items"]
        self.assertTrue(any("reference_shared" in b.lower() or "multi-guitar" in b.lower() for b in blocked))

    def test_artifact_guard_includes_no_delayed_body_tail(self) -> None:
        rules = build_artifact_guard_rules()
        ids = {r["rule_id"] for r in rules}
        self.assertIn("no_body_tail_stem", ids)
        self.assertIn("no_helmholtz_ir_late", ids)
        registry = build_factor_registry()
        self.assertIn("independent_body_tail_forbidden", registry["artifact_guard_factors"])
        forbidden = registry["artifact_guard_factors"]["independent_body_tail_forbidden"]
        self.assertIn("FORBIDDEN", forbidden["physical_meaning"])

    def test_missing_critical_fields_reported(self) -> None:
        report = build_pgsm_step1_report(repo_root=REPO)
        missing = report["missing_critical_data"]
        for field in ("scale_length", "bridge_position", "elastic_moduli", "anisotropy"):
            self.assertIn(field, missing)

    def test_report_files_created(self) -> None:
        report = write_pgsm_step1_reports(
            repo_root=REPO,
            json_path=self.json_path,
            md_path=self.md_path,
        )
        self.assertTrue(self.json_path.is_file())
        self.assertTrue(self.md_path.is_file())
        doc = json.loads(self.json_path.read_text(encoding="utf-8"))
        self.assertEqual(doc["report_version"], PGSM_STEP1_VERSION)
        self.assertTrue(doc["no_audio_generated"])
        self.assertTrue(doc["no_fem_run"])
        self.assertTrue(doc["no_rom_run"])
        self.assertTrue(doc["website_default_unchanged"])
        self.assertIn("factor_registry", doc)
        self.assertIn("explicit_statement", doc)
        self.assertIn("does not synthesize sound", doc["explicit_statement"])
        self.assertIn("Physical sound chain", self.md_path.read_text(encoding="utf-8"))
        self.assertIn("delayed echo", self.md_path.read_text(encoding="utf-8").lower())

    def test_no_wav_files_created_in_temp_output(self) -> None:
        write_pgsm_step1_reports(
            repo_root=REPO,
            json_path=self.tmp / "out.json",
            md_path=self.tmp / "out.md",
        )
        wavs = list(self.tmp.rglob("*.wav"))
        self.assertEqual(wavs, [])

    def test_no_fem_rom_subprocess_calls(self) -> None:
        with patch("subprocess.run") as mock_run, patch("subprocess.Popen") as mock_popen:
            build_pgsm_step1_report(repo_root=REPO)
            write_pgsm_step1_reports(
                repo_root=REPO,
                json_path=self.json_path,
                md_path=self.md_path,
            )
            mock_run.assert_not_called()
            mock_popen.assert_not_called()

    def test_data_mapping_covers_samples_000_009(self) -> None:
        report = build_pgsm_step1_report(repo_root=REPO)
        per = report["data_mapping_by_sample"]["per_sample"]
        for i in range(10):
            sid = f"sample_{i:03d}"
            self.assertIn(sid, per)
            self.assertIn("geometry", per[sid])
            self.assertIn("body_signature_cache", per[sid])
            self.assertIn("json_on_disk", per[sid]["body_signature_cache"])

    def test_causality_guard_checks(self) -> None:
        report = build_pgsm_step1_report(repo_root=REPO)
        cg = report["causality_guard_checks"]
        self.assertTrue(cg["body_starts_at_t0"])
        self.assertTrue(cg["no_independent_delayed_body_tail"])
        self.assertTrue(cg["no_delayed_resonator_ramp_second_onset"])

    def test_dimensional_sanity(self) -> None:
        report = build_pgsm_step1_report(repo_root=REPO)
        self.assertTrue(report["dimensional_sanity"]["pass"])


if __name__ == "__main__":
    unittest.main()
