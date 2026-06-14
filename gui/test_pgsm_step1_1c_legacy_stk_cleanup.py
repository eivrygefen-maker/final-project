#!/usr/bin/env python3
"""PGSM Step 1.1c — legacy STK V3/V4 / Stage 42–52 cleanup tests."""
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
from pgsm_physical_factor_registry import write_pgsm_step1_reports  # noqa: E402
from pgsm_physical_interaction_map import write_pgsm_step2_reports  # noqa: E402
from pgsm_step1_1c_legacy_stk_cleanup import (  # noqa: E402
    DELETED_CODE_FILES,
    KEPT_PRODUCTION_FILES,
    OBSOLETE_CODE_GLOBS,
    delete_legacy_stk_artifacts,
    verify_cleanup_status,
    write_cleanup_reports,
)
from stk_pipeline_defaults import DEFAULT_WEBSITE_STK_MODE  # noqa: E402


class TestPgsmStep11cLegacyStkCleanup(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        delete_legacy_stk_artifacts(repo_root=REPO)

    def test_pgsm_step1_tests_still_pass(self) -> None:
        proc = subprocess.run(
            [sys.executable, "-m", "unittest", "test_pgsm_physical_factor_registry", "-v"],
            cwd=str(REPO / "gui"),
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)

    def test_pgsm_step2_tests_still_pass(self) -> None:
        proc = subprocess.run(
            [sys.executable, "-m", "unittest", "test_pgsm_physical_interaction_map", "-v"],
            cwd=str(REPO / "gui"),
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)

    def test_no_listed_legacy_files_remain(self) -> None:
        for rel in DELETED_CODE_FILES:
            self.assertFalse((REPO / rel).is_file(), msg=rel)

    def test_no_obsolete_code_globs_remain(self) -> None:
        status = verify_cleanup_status(repo_root=REPO)
        self.assertEqual(status["code_obsolete_remaining"], [])

    def test_no_obsolete_stage_tests_remain(self) -> None:
        gui = REPO / "gui"
        for pattern in OBSOLETE_CODE_GLOBS:
            if not pattern.startswith("test_stage"):
                continue
            matches = list(gui.glob(pattern))
            self.assertEqual(matches, [], msg=f"{pattern}: {matches}")

    def test_no_obsolete_debug_reports_remain(self) -> None:
        status = verify_cleanup_status(repo_root=REPO)
        self.assertEqual(status["debug_reports_remaining"], [])

    def test_no_obsolete_audio_dirs_remain(self) -> None:
        status = verify_cleanup_status(repo_root=REPO)
        self.assertEqual(status["audio_dirs_remaining"], [])

    def test_kept_production_files_exist(self) -> None:
        for rel, _reason in KEPT_PRODUCTION_FILES:
            self.assertTrue((REPO / rel).is_file(), msg=rel)

    def test_website_imports_ok(self) -> None:
        status = verify_cleanup_status(repo_root=REPO)
        self.assertTrue(status["import_check"]["pass"], msg=status["import_check"])

    def test_website_default_unchanged(self) -> None:
        self.assertEqual(DEFAULT_WEBSITE_STK_MODE, STK_BODY_TRANSFER_FINAL_V1)
        self.assertEqual(DEFAULT_WEBSITE_STK_MODE, "stk_body_transfer_final_v1")

    def test_no_fem_rom_subprocess(self) -> None:
        with patch("subprocess.run") as mock_run, patch("subprocess.Popen") as mock_popen:
            verify_cleanup_status(repo_root=REPO)
            write_cleanup_reports(repo_root=REPO)
            mock_run.assert_not_called()
            mock_popen.assert_not_called()

    def test_no_wav_generation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            write_cleanup_reports(
                repo_root=REPO,
                json_path=tmp / "cleanup.json",
                md_path=tmp / "cleanup.md",
            )
            self.assertEqual(list(tmp.rglob("*.wav")), [])

    def test_cleanup_report_complete(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            report = write_cleanup_reports(
                repo_root=REPO,
                json_path=tmp / "cleanup.json",
                md_path=tmp / "cleanup.md",
            )
            self.assertTrue((tmp / "cleanup.json").is_file())
            doc = json.loads((tmp / "cleanup.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "pgsm_step1_1c_legacy_stk_cleanup_complete")
            self.assertTrue(doc["verification"]["all_clean"])
            self.assertIn("deleted", doc["explicit_statement"].lower())

    def test_pgsm_step1_and_step2_regenerate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            write_pgsm_step1_reports(
                repo_root=REPO,
                json_path=tmp / "step1.json",
                md_path=tmp / "step1.md",
            )
            write_pgsm_step2_reports(
                repo_root=REPO,
                json_path=tmp / "step2.json",
                md_path=tmp / "step2.md",
            )
            self.assertTrue((tmp / "step1.json").is_file())
            self.assertTrue((tmp / "step2.json").is_file())

    def test_body_hybrid_no_stage48_import(self) -> None:
        text = (REPO / "gui" / "body_hybrid_v4_1_identity_space.py").read_text(encoding="utf-8")
        self.assertNotIn("stage48_timbre_decomposition_report", text)


if __name__ == "__main__":
    unittest.main()
