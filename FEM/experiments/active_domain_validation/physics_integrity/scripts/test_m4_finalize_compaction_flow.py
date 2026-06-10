#!/usr/bin/env python3
"""Integration tests for explicit finalization compaction flow (no FEM)."""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compact_completed_m4_runs import compact_one_completed_run, production_compaction_preconditions  # noqa: E402
from test_m4_compaction_run_selection import _make_strict_completed_run  # noqa: E402
from v2_b3_m4_finalize_completed_run import (  # noqa: E402
    build_compaction_production_row,
    diagnose_finalization_state,
    finalize_completed_run,
    require_compaction_completed,
)
from v2_b3_m4_physics_identity_lib import FORBIDDEN_HEAVY_REL_DIRS  # noqa: E402
from v2_b3_m4_shared_export import APPROVED_SHARED_PLOT_NAMES  # noqa: E402
from v2_b3_petsc_util import write_json_atomic  # noqa: E402


class FinalizeCompactionFlowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        self.shared = self.repo / "sf_gmar"
        self.shared.mkdir()
        self.sample_id = "sample_000"
        self.run_id = "sample_000_rom_official_v1"
        self.run_root = (
            self.repo
            / "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars"
            / self.sample_id
            / "runs"
            / self.run_id
        )
        self.lhs = self.repo / "ROM/classic/lhs_pool.json"
        self.lhs.parent.mkdir(parents=True, exist_ok=True)
        self.lhs.write_text(
            json.dumps(
                {
                    "shape_name": "classic",
                    "entries": [
                        {
                            "id": self.sample_id,
                            "status": "PENDING",
                            "lhs_row_index": 0,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        _make_strict_completed_run(
            self.run_root,
            sample_id=self.sample_id,
            run_id=self.run_id,
            production_acceptance_pass=False,
            production_acceptance_failures=["missing_acceptance"],
        )
        for name in APPROVED_SHARED_PLOT_NAMES:
            (self.run_root / "aggregation" / name).write_bytes(b"png")
        write_json_atomic(self.run_root / "aggregation" / "runtime_summary.json", {"status": "ok"})
        write_json_atomic(
            self.run_root / "pipeline_run_manifest.json",
            {
                "terminal_status": "COMPLETED",
                "production_acceptance_pass": True,
            },
        )
        write_json_atomic(
            self.run_root / "freeze" / "freeze_manifest.json",
            {"production_acceptance_pass": True, "production_acceptance_failures": []},
        )
        write_json_atomic(
            self.run_root / "freeze" / "physics_identity_manifest.json",
            {
                "schema": "m4_physics_identity_v1",
                "sample_id": self.sample_id,
                "run_id": self.run_id,
                "production_acceptance_pass": True,
                "production_acceptance_failures": [],
                "generated_mesh_sha256": "abc",
                "operator_mesh_matches_generated": True,
                "active_dimension": 100,
                "masks": {"p_idx_aperture_count": 4},
                "fallback_flags": {"cross_sample_reuse": False},
                "path_contamination": {"contamination_detected": False},
            },
        )
        for rel in FORBIDDEN_HEAVY_REL_DIRS:
            path = self.run_root / rel
            path.mkdir(parents=True, exist_ok=True)
            (path / "artifact.bin").write_bytes(b"x" * 128)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _acceptance_report(self) -> dict:
        return {
            "acceptance_pass": True,
            "production_acceptance_pass": True,
            "production_acceptance_failures": [],
            "manifests_repaired": True,
        }

    def _export_manifest(self) -> dict:
        return {
            "export_status": "EXPORTED",
            "summary_export_path": str(self.shared / "classic/summaries/summary.json"),
            "graph_manifest_export_path": str(self.shared / "classic/summaries/graph_manifest.json"),
            "graph_export_entries": [{"copy_status": "copied"}],
        }

    def _summary(self) -> dict:
        return {
            "terminal_status": "COMPLETED",
            "aggregation_status": "AGGREGATION_PASS",
            "final_aggregation_ready": True,
        }

    def test_stale_row_without_acceptance_blocks_compaction_gate(self) -> None:
        stale_row = build_compaction_production_row(
            sample_id=self.sample_id,
            run_id=self.run_id,
            acceptance_report={"production_acceptance_pass": False, "production_acceptance_failures": ["x"]},
            export_manifest=self._export_manifest(),
            summary=self._summary(),
        )
        pool_entry = {"status": "PENDING"}
        ok, reason, _ = production_compaction_preconditions(
            row=stale_row,
            pool_entry=pool_entry,
            run_rom_compare=False,
        )
        self.assertFalse(ok)
        self.assertIn("production_acceptance_pass", reason)

    def test_refreshed_row_passes_compaction_gate_with_pending_lhs(self) -> None:
        row = build_compaction_production_row(
            sample_id=self.sample_id,
            run_id=self.run_id,
            acceptance_report=self._acceptance_report(),
            export_manifest=self._export_manifest(),
            summary=self._summary(),
        )
        ok, reason, _ = production_compaction_preconditions(
            row=row,
            pool_entry={"status": "PENDING"},
            run_rom_compare=False,
        )
        self.assertTrue(ok, reason)
        self.assertTrue(row["production_acceptance_pass"])

    def test_pending_lhs_blocks_compaction_without_allow_pending(self) -> None:
        row = build_compaction_production_row(
            sample_id=self.sample_id,
            run_id=self.run_id,
            acceptance_report=self._acceptance_report(),
            export_manifest=self._export_manifest(),
            summary=self._summary(),
        )
        pool = json.loads(self.lhs.read_text(encoding="utf-8"))
        out = compact_one_completed_run(
            repo_root=self.repo,
            pool=pool,
            sample_id=self.sample_id,
            run_id=self.run_id,
            production_row=row,
            production_trigger=True,
            allow_pending_lhs=False,
        )
        self.assertNotEqual(out.status, "completed")
        self.assertIn("lhs_status", out.skip_reason)

    def test_pending_lhs_compacts_with_allow_pending_lhs(self) -> None:
        row = build_compaction_production_row(
            sample_id=self.sample_id,
            run_id=self.run_id,
            acceptance_report=self._acceptance_report(),
            export_manifest=self._export_manifest(),
            summary=self._summary(),
        )
        pool = json.loads(self.lhs.read_text(encoding="utf-8"))
        out = compact_one_completed_run(
            repo_root=self.repo,
            pool=pool,
            sample_id=self.sample_id,
            run_id=self.run_id,
            production_row=row,
            production_trigger=True,
            allow_pending_lhs=True,
        )
        self.assertEqual(out.status, "completed", out.skip_reason)
        self.assertTrue((self.run_root / "compaction" / "compaction_manifest.json").is_file())
        for rel in FORBIDDEN_HEAVY_REL_DIRS:
            self.assertFalse((self.run_root / rel).exists(), rel)

    def test_skipped_compaction_stops_before_cleanup(self) -> None:
        acceptance_report = self._acceptance_report()
        with patch(
            "v2_b3_m4_finalize_completed_run.ensure_production_acceptance_for_finalization",
            return_value=acceptance_report,
        ):
            with patch(
                "v2_b3_m4_finalize_completed_run.try_export_sample_to_shared",
                return_value=(self._export_manifest(), None),
            ):
                with patch(
                    "v2_b3_m4_finalize_completed_run.compact_one_completed_run",
                ) as compact_mock:
                    from compact_completed_m4_runs import CompactionOutcome  # noqa: WPS433

                    compact_mock.return_value = CompactionOutcome(
                        sample_id=self.sample_id,
                        run_id=self.run_id,
                        status="planned",
                        skip_reason="lhs_status=PENDING",
                    )
                    with patch(
                        "v2_b3_m4_finalize_completed_run.run_sample_cleanup_barrier",
                    ) as barrier_mock:
                        with self.assertRaises(RuntimeError) as ctx:
                            finalize_completed_run(
                                repo_root=self.repo,
                                sample_id=self.sample_id,
                                run_id=self.run_id,
                                lhs_path=self.lhs,
                                shared_root=self.shared,
                                reconcile_bookkeeping=False,
                            )
        self.assertIn("COMPACTION_NOT_COMPLETED", str(ctx.exception))
        barrier_mock.assert_not_called()

    def test_diagnose_reports_compaction_row_with_acceptance(self) -> None:
        report = diagnose_finalization_state(
            repo_root=self.repo,
            sample_id=self.sample_id,
            run_id=self.run_id,
            lhs_path=self.lhs,
            shared_root=self.shared,
        )
        row = report.get("compaction_row") or {}
        self.assertTrue(row.get("production_acceptance_pass"))
        self.assertEqual(report.get("lhs_entry_status"), "PENDING")
        gate = report.get("production_compaction_gate") or {}
        self.assertTrue(gate.get("ok"), gate)

    def test_finalize_integration_deletes_heavy_paths(self) -> None:
        acceptance_report = self._acceptance_report()
        with patch(
            "v2_b3_m4_finalize_completed_run.ensure_production_acceptance_for_finalization",
            return_value=acceptance_report,
        ):
            with patch(
                "v2_b3_m4_finalize_completed_run.try_export_sample_to_shared",
                return_value=(self._export_manifest(), None),
            ):
                with patch(
                    "v2_b3_m4_finalize_completed_run.run_sample_cleanup_barrier",
                ) as barrier_mock:
                    from v2_b3_m4_sample_cleanup_barrier import CleanupBarrierOutcome  # noqa: WPS433

                    barrier_mock.return_value = CleanupBarrierOutcome(
                        sample_id=self.sample_id,
                        run_id=self.run_id,
                        status="completed",
                        sample_success=True,
                        verification_pass=True,
                    )
                    report = finalize_completed_run(
                        repo_root=self.repo,
                        sample_id=self.sample_id,
                        run_id=self.run_id,
                        lhs_path=self.lhs,
                        shared_root=self.shared,
                        reconcile_bookkeeping=False,
                    )
        self.assertEqual(report["compaction_status"], "completed")
        self.assertTrue(report["production_acceptance_pass"])
        for rel in FORBIDDEN_HEAVY_REL_DIRS:
            self.assertFalse((self.run_root / rel).exists(), rel)
        stages = report.get("stages") or {}
        self.assertTrue(stages.get("compaction_invoked"))
        self.assertEqual(stages.get("cleanup_status"), "completed")


if __name__ == "__main__":
    unittest.main()
