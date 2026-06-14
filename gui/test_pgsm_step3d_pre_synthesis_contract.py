#!/usr/bin/env python3
"""PGSM Step 3D — pre-synthesis contract tests (no audio, no FEM/ROM, no STK)."""
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
from pgsm_step3d_pre_synthesis_contract import (  # noqa: E402
    PGSM_STEP3D_VERSION,
    STEP22B_POLICY_PRIMARY,
    STEP3C_READINESS_REQUIRED,
    STEP4A_READINESS,
    build_artifact_guard_contract,
    build_pgsm_step3d_report,
    build_synthesis_input_contract,
    load_upstream_reports,
    write_pgsm_step3d_reports,
)
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE  # noqa: E402


class TestPgsmStep3dPreSynthesisContract(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)
        self._report_cache: dict | None = None

    def tearDown(self) -> None:
        self._td.cleanup()

    def _report(self) -> dict:
        if self._report_cache is None:
            self._report_cache = build_pgsm_step3d_report(repo_root=REPO)
        return self._report_cache

    def test_website_default_unchanged(self) -> None:
        self.assertEqual(DEFAULT_WEBSITE_STK_MODE, STK_BODY_TRANSFER_FINAL_V1)

    def test_report_files_created(self) -> None:
        write_pgsm_step3d_reports(
            repo_root=REPO,
            json_path=self.tmp / "step3d.json",
            md_path=self.tmp / "step3d.md",
        )
        self.assertTrue((self.tmp / "step3d.json").is_file())
        doc = json.loads((self.tmp / "step3d.json").read_text(encoding="utf-8"))
        self.assertEqual(doc["report_version"], PGSM_STEP3D_VERSION)
        self.assertTrue(doc["no_wav_generated"])
        self.assertIn("does not synthesize sound", doc["explicit_statement"])

    def test_no_wav_files_created(self) -> None:
        write_pgsm_step3d_reports(
            repo_root=REPO,
            json_path=self.tmp / "s.json",
            md_path=self.tmp / "s.md",
        )
        self.assertEqual(list(self.tmp.rglob("*.wav")), [])

    def test_no_fem_rom_subprocess(self) -> None:
        with patch("subprocess.run") as mock_run, patch("subprocess.Popen") as mock_popen:
            build_pgsm_step3d_report(repo_root=REPO)
            mock_run.assert_not_called()
            mock_popen.assert_not_called()

    def test_no_stk_integration(self) -> None:
        import pgsm_step3d_pre_synthesis_contract as mod

        src = Path(mod.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import diagnostic_synthesis", src)
        self.assertNotIn("stk_body_transfer", src.lower().replace("stk_body_transfer_final_v1", ""))

    def test_all_upstream_reports_load(self) -> None:
        reports = load_upstream_reports(REPO)
        for key in ("step1", "step2", "step2_1", "step2_2", "step2_2b", "step3a", "step3b", "step3c"):
            self.assertIn(key, reports)
            self.assertIn("report_version", reports[key])

    def test_step3c_readiness_verified(self) -> None:
        report = self._report()
        checks = (report.get("upstream_readiness_summary") or {}).get("checks") or {}
        self.assertTrue(checks["step3c"]["pass"])
        self.assertEqual(checks["step3c"]["readiness"], STEP3C_READINESS_REQUIRED)

    def test_step22b_fem_primary_policy_verified(self) -> None:
        report = self._report()
        checks = (report.get("upstream_readiness_summary") or {}).get("checks") or {}
        self.assertTrue(checks["step2_2b"]["fem_primary"])
        self.assertEqual(checks["step2_2b"]["primary_policy"], STEP22B_POLICY_PRIMARY)

    def test_synthesis_contract_includes_required_field_groups(self) -> None:
        reports = load_upstream_reports(REPO)
        contract = build_synthesis_input_contract(reports)
        body_names = {f["field"] for f in contract["guitar_body_numeric_fields"]}
        string_names = {f["field"] for f in contract["string_excitation_fields"]}
        for name in (
            "modal_frequencies_hz",
            "calibrated_Q",
            "calibrated_tau_s",
            "normalized_bridge_admittance",
            "calibrated_region_weights",
            "fem_primary_material_values",
        ):
            self.assertIn(name, body_names)
        for name in (
            "note_frequency_hz",
            "pluck_position_ratio",
            "F_bridge_proxy",
            "exact_open_string_claim_allowed",
        ):
            self.assertIn(name, string_names)

    def test_every_field_has_source_fallback_level(self) -> None:
        reports = load_upstream_reports(REPO)
        contract = build_synthesis_input_contract(reports)
        all_fields = contract["guitar_body_numeric_fields"] + contract["string_excitation_fields"]
        for field in all_fields:
            self.assertTrue(field.get("source_fallback_level"), field["field"])

    def test_exact_open_string_claims_blocked(self) -> None:
        reports = load_upstream_reports(REPO)
        contract = build_synthesis_input_contract(reports)
        open_field = next(
            f for f in contract["string_excitation_fields"] if f["field"] == "exact_open_string_claim_allowed"
        )
        self.assertFalse(open_field["value_summary"])
        restrictions = contract["output_restrictions"]
        self.assertTrue(restrictions["no_final_physical_accuracy_claim"])

    def test_artifact_guard_forbids_v6_patterns(self) -> None:
        guard = build_artifact_guard_contract()
        names = {row["artifact"] for row in guard["forbidden_artifacts"]}
        for artifact in (
            "delayed_body_tail_stem",
            "helmholtz_echo_ir",
            "independent_delayed_body_onset",
            "hard_gate_tail_cut",
            "end_rise_noise",
        ):
            self.assertIn(artifact, names)

    def test_allowed_step4a_diagnostic_single_note_only(self) -> None:
        report = self._report()
        allowed = report.get("allowed_step4a_scope") or {}
        self.assertIn("single_note", " ".join(allowed.get("allowed") or []).lower())
        self.assertIn("diagnostic", allowed.get("label", "").lower())
        self.assertTrue(allowed.get("not_final_synthesis"))

    def test_multi_guitar_comparison_blocked(self) -> None:
        report = self._report()
        readiness = report.get("readiness_after_step3d") or {}
        self.assertFalse(readiness.get("multi_guitar_comparison_allowed"))
        blocked = report.get("blocked_steps") or []
        self.assertTrue(any("multi-guitar" in b.lower() for b in blocked))

    def test_final_stk_integration_blocked(self) -> None:
        report = self._report()
        readiness = report.get("readiness_after_step3d") or {}
        self.assertFalse(readiness.get("stk_integration_allowed"))
        blocked = report.get("blocked_steps") or []
        self.assertTrue(any("STK" in b for b in blocked))

    def test_readiness_not_final_synthesis(self) -> None:
        report = self._report()
        readiness = report.get("readiness_after_step3d") or {}
        self.assertFalse(readiness.get("final_synthesis_ready"))
        self.assertFalse(readiness.get("musical_wav_synthesis_allowed"))
        self.assertNotIn("ready_for_final", readiness.get("current_status", ""))

    def test_readiness_step4a_diagnostic_only(self) -> None:
        report = self._report()
        readiness = report.get("readiness_after_step3d") or {}
        self.assertEqual(readiness.get("current_status"), STEP4A_READINESS)
        self.assertTrue(readiness.get("step4a_diagnostic_single_note_allowed"))

    def test_safe_next_step_step4a_not_production(self) -> None:
        report = self._report()
        nxt = report.get("safe_next_step", "").lower()
        self.assertIn("step 4a", nxt)
        self.assertIn("diagnostic", nxt)
        self.assertTrue("no stk" in nxt or "no stK" in nxt.lower())
        self.assertNotIn("website production", nxt)

    def test_upstream_no_musical_audio_readiness(self) -> None:
        report = self._report()
        upstream = report.get("upstream_readiness_summary") or {}
        self.assertTrue(upstream.get("no_musical_audio_readiness"))
        self.assertEqual(upstream.get("musical_audio_readiness_claims_found"), [])


if __name__ == "__main__":
    unittest.main()
