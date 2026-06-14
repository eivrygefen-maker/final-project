#!/usr/bin/env python3
"""PGSM Step 1.1 cleanup verification tests."""
from __future__ import annotations

import importlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gui"))

from body_hybrid_v4_1_identity_space import STK_BODY_TRANSFER_FINAL_V1  # noqa: E402
from diagnostic_synthesis import DIAGNOSTIC_MODES  # noqa: E402
from pgsm_physical_factor_registry import write_pgsm_step1_reports  # noqa: E402
from pgsm_step1_1_cleanup import (  # noqa: E402
    DELETED_AUDIO_DIRS,
    DELETED_CODE_FILES,
    DELETED_DEBUG_REPORT_FILES,
    KEPT_FILES,
    REJECTED_DIAGNOSTIC_MODES,
    delete_obsolete_artifacts,
    verify_cleanup_status,
    write_cleanup_reports,
)
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE  # noqa: E402


class TestPgsmStep11Cleanup(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        delete_obsolete_artifacts(repo_root=REPO)

    def test_delete_obsolete_artifacts_idempotent(self) -> None:
        delete_obsolete_artifacts(repo_root=REPO)
        second = delete_obsolete_artifacts(repo_root=REPO)
        self.assertEqual(second["removed_audio_dirs"], [])
        self.assertEqual(second["removed_debug_reports"], [])
        self.assertEqual(second["removed_code_files"], [])
        status = verify_cleanup_status(repo_root=REPO)
        self.assertEqual(status["audio_dirs_remaining"], [])
        self.assertEqual(status["debug_reports_remaining"], [])

    def test_delete_removes_staged_leftover_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "audio" / "stk_v6_3_review_audio").mkdir(parents=True)
            (root / "audio" / "debug_reports").mkdir(parents=True)
            (root / "audio" / "debug_reports" / "stk_v6_3_artifact_quarantine_report.json").write_text(
                "{}", encoding="utf-8"
            )
            (root / "audio" / "debug_reports" / "stk_v6_physical_dof_audit.json").write_text("{}", encoding="utf-8")
            result = delete_obsolete_artifacts(repo_root=root)
            self.assertIn("audio/stk_v6_3_review_audio", result["removed_audio_dirs"])
            self.assertTrue(
                any("stk_v6_3_artifact_quarantine" in p for p in result["removed_debug_reports"])
            )
            self.assertTrue((root / "audio" / "debug_reports" / "stk_v6_physical_dof_audit.json").is_file())

    def test_pgsm_step1_tests_still_pass(self) -> None:
        proc = subprocess.run(
            [sys.executable, "-m", "unittest", "test_pgsm_physical_factor_registry", "-v"],
            cwd=str(REPO / "gui"),
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)

    def test_pgsm_step1_regenerates(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            report = write_pgsm_step1_reports(
                repo_root=REPO,
                json_path=tmp / "pgsm.json",
                md_path=tmp / "pgsm.md",
            )
            self.assertTrue((tmp / "pgsm.json").is_file())
            self.assertTrue(report["no_audio_generated"])

    def test_cleanup_report_exists(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            report = write_cleanup_reports(
                repo_root=REPO,
                json_path=tmp / "cleanup.json",
                md_path=tmp / "cleanup.md",
                tests_run=["test_pgsm_step1_1_cleanup"],
            )
            self.assertTrue((tmp / "cleanup.json").is_file())
            self.assertTrue((tmp / "cleanup.md").is_file())
            self.assertIn("deleted", report["explicit_statement"].lower())

    def test_no_obsolete_v5_v6_builders_remain(self) -> None:
        status = verify_cleanup_status(repo_root=REPO)
        self.assertEqual(status["code_obsolete_remaining"], [])

    def test_no_obsolete_v5_v6_tests_remain(self) -> None:
        gui = REPO / "gui"
        for pattern in ("test_stk_v5*.py", "test_stk_v6*.py", "test_stage51h*.py"):
            matches = [p for p in gui.glob(pattern)]
            self.assertEqual(matches, [], msg=f"found {matches} for {pattern}")

    def test_no_obsolete_review_audio_dirs(self) -> None:
        for rel in DELETED_AUDIO_DIRS:
            self.assertFalse((REPO / rel).exists(), msg=rel)

    def test_no_obsolete_v5_v6_debug_reports_except_allowed(self) -> None:
        status = verify_cleanup_status(repo_root=REPO)
        self.assertEqual(status["debug_reports_remaining"], [])
        self.assertTrue((REPO / "audio/debug_reports/stk_v6_physical_dof_audit.json").is_file())

    def test_rejected_v6_modes_not_registered(self) -> None:
        for mode in REJECTED_DIAGNOSTIC_MODES:
            self.assertNotIn(mode, DIAGNOSTIC_MODES, msg=mode)

    def test_website_imports_do_not_fail(self) -> None:
        status = verify_cleanup_status(repo_root=REPO)
        self.assertTrue(status["import_check"]["pass"], msg=status["import_check"])

    def test_website_default_unchanged(self) -> None:
        self.assertEqual(DEFAULT_WEBSITE_STK_MODE, STK_BODY_TRANSFER_FINAL_V1)
        self.assertEqual(DEFAULT_WEBSITE_STK_MODE, "stk_body_transfer_final_v1")

    def test_no_fem_rom_subprocess_calls_in_cleanup(self) -> None:
        with patch("subprocess.run") as mock_run, patch("subprocess.Popen") as mock_popen:
            verify_cleanup_status(repo_root=REPO)
            write_cleanup_reports(repo_root=REPO)
            mock_run.assert_not_called()
            mock_popen.assert_not_called()

    def test_no_new_wavs_generated_by_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            write_cleanup_reports(
                repo_root=REPO,
                json_path=tmp / "cleanup.json",
                md_path=tmp / "cleanup.md",
            )
            self.assertEqual(list(tmp.rglob("*.wav")), [])

    def test_pgsm_step1_files_preserved(self) -> None:
        for rel, _reason in KEPT_FILES:
            self.assertTrue((REPO / rel).is_file(), msg=rel)

    def test_deleted_code_files_gone(self) -> None:
        for rel in DELETED_CODE_FILES:
            self.assertFalse((REPO / rel).is_file(), msg=rel)

    def test_app_dependency_import(self) -> None:
        importlib.import_module("body_response_synth")
        importlib.import_module("build_note_cache")


if __name__ == "__main__":
    unittest.main()
