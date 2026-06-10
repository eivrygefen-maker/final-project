#!/usr/bin/env python3
"""Integration tests for explicit finalization compaction flow (no FEM)."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compact_completed_m4_runs import (  # noqa: E402
    compact_one_completed_run,
    probe_compaction_eligibility,
    production_compaction_preconditions,
)
from test_m4_compaction_run_selection import _make_strict_completed_run  # noqa: E402
from v2_b3_m4_finalize_completed_run import (  # noqa: E402
    build_compaction_production_row,
    diagnose_finalization_state,
    finalize_completed_run,
)
from v2_b3_m4_lhs_pool_bridge import (  # noqa: E402
    LHS_COMPLETED,
    LHS_FAILED,
    LHS_RUNNING,
    load_lhs_pool,
    write_lhs_pool,
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
        self._write_lhs(status="PENDING", last_run_id=None)
        _make_strict_completed_run(
            self.run_root,
            sample_id=self.sample_id,
            run_id=self.run_id,
            production_acceptance_pass=True,
        )
        for name in APPROVED_SHARED_PLOT_NAMES:
            (self.run_root / "aggregation" / name).write_bytes(b"png")
        write_json_atomic(self.run_root / "aggregation" / "runtime_summary.json", {"status": "ok"})
        write_json_atomic(
            self.run_root / "pipeline_run_manifest.json",
            {"terminal_status": "COMPLETED", "production_acceptance_pass": True},
        )
        for rel in FORBIDDEN_HEAVY_REL_DIRS:
            path = self.run_root / rel
            path.mkdir(parents=True, exist_ok=True)
            (path / "artifact.bin").write_bytes(b"x" * 128)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _write_lhs(self, *, status: str, last_run_id: str | None) -> None:
        entry: dict = {"id": self.sample_id, "status": status, "lhs_row_index": 0}
        if last_run_id is not None:
            entry["last_run_id"] = last_run_id
        self.lhs.write_text(
            json.dumps({"shape_name": "classic", "entries": [entry]}),
            encoding="utf-8",
        )

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
            "failed_chunks": 0,
            "missing_chunks": 0,
        }

    def _compaction_row(self) -> dict:
        return build_compaction_production_row(
            sample_id=self.sample_id,
            run_id=self.run_id,
            acceptance_report=self._acceptance_report(),
            export_manifest=self._export_manifest(),
            summary=self._summary(),
        )

    def _pool(self) -> dict:
        return json.loads(self.lhs.read_text(encoding="utf-8"))

    def _compact(self, *, allow_transitional_lhs: bool) -> object:
        return compact_one_completed_run(
            repo_root=self.repo,
            pool=self._pool(),
            sample_id=self.sample_id,
            run_id=self.run_id,
            production_row=self._compaction_row(),
            production_trigger=True,
            allow_transitional_lhs=allow_transitional_lhs,
        )

    def test_pending_matching_completed_run_compaction_allowed(self) -> None:
        self._write_lhs(status="PENDING", last_run_id=None)
        out = self._compact(allow_transitional_lhs=True)
        self.assertEqual(out.status, "completed", out.skip_reason)
        probe = probe_compaction_eligibility(
            repo_root=self.repo,
            pool=self._pool(),
            sample_id=self.sample_id,
            run_id=self.run_id,
            production_row=self._compaction_row(),
        )
        self.assertTrue(probe["transitional_lhs_allowed"])

    def test_running_matching_completed_run_compaction_allowed(self) -> None:
        self._write_lhs(status=LHS_RUNNING, last_run_id=self.run_id)
        out = self._compact(allow_transitional_lhs=True)
        self.assertEqual(out.status, "completed", out.skip_reason)

    def test_running_different_last_run_id_blocked(self) -> None:
        self._write_lhs(status=LHS_RUNNING, last_run_id="sample_000_other_run")
        out = self._compact(allow_transitional_lhs=True)
        self.assertNotEqual(out.status, "completed")
        self.assertIn("last_run_id_mismatch", out.skip_reason)

    def test_failed_lhs_blocked(self) -> None:
        self._write_lhs(status=LHS_FAILED, last_run_id=self.run_id)
        out = self._compact(allow_transitional_lhs=True)
        self.assertNotEqual(out.status, "completed")
        self.assertIn("lhs_status_blocked", out.skip_reason)

    def test_completed_lhs_allowed_idempotent(self) -> None:
        self._write_lhs(status=LHS_COMPLETED, last_run_id=self.run_id)
        out = self._compact(allow_transitional_lhs=True)
        self.assertEqual(out.status, "completed", out.skip_reason)

    def test_transitional_lhs_blocked_without_flag(self) -> None:
        self._write_lhs(status=LHS_RUNNING, last_run_id=self.run_id)
        out = self._compact(allow_transitional_lhs=False)
        self.assertNotEqual(out.status, "completed")
        self.assertIn("lhs_status", out.skip_reason)

    def test_stale_row_without_acceptance_blocks_gate(self) -> None:
        row = build_compaction_production_row(
            sample_id=self.sample_id,
            run_id=self.run_id,
            acceptance_report={"production_acceptance_pass": False, "production_acceptance_failures": ["x"]},
            export_manifest=self._export_manifest(),
            summary=self._summary(),
        )
        ok, reason, _ = production_compaction_preconditions(
            row=row,
            pool_entry={"status": "PENDING"},
            run_rom_compare=False,
        )
        self.assertFalse(ok)
        self.assertIn("production_acceptance_pass", reason)

    def test_skipped_compaction_stops_before_cleanup(self) -> None:
        with patch(
            "v2_b3_m4_finalize_completed_run.ensure_production_acceptance_for_finalization",
            return_value=self._acceptance_report(),
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
                        skip_reason="lhs_status=RUNNING",
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

    def test_diagnose_reports_lhs_eligibility_fields(self) -> None:
        self._write_lhs(status=LHS_RUNNING, last_run_id=self.run_id)
        report = diagnose_finalization_state(
            repo_root=self.repo,
            sample_id=self.sample_id,
            run_id=self.run_id,
            lhs_path=self.lhs,
            shared_root=self.shared,
        )
        for key in (
            "lhs_entry_status",
            "lhs_entry_last_run_id",
            "finalizing_run_id",
            "lhs_run_ownership_match",
            "transitional_lhs_allowed",
            "compaction_eligible",
        ):
            self.assertIn(key, report, key)
        self.assertTrue(report["lhs_run_ownership_match"])
        self.assertTrue(report["transitional_lhs_allowed"])

    def test_finalize_marks_lhs_completed_after_successful_cleanup(self) -> None:
        self._write_lhs(status=LHS_RUNNING, last_run_id=self.run_id)

        def _reconcile_bookkeeping(**kwargs: object) -> dict:
            pool = load_lhs_pool(self.lhs)
            entry = next(e for e in pool["entries"] if e["id"] == self.sample_id)
            entry["status"] = LHS_COMPLETED
            entry["last_run_id"] = self.run_id
            write_lhs_pool(self.lhs, pool)
            return {"outcome": "pass"}

        with patch(
            "v2_b3_m4_finalize_completed_run.ensure_production_acceptance_for_finalization",
            return_value=self._acceptance_report(),
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
                    with patch(
                        "v2_b3_m4_finalize_completed_run.reconcile_run_bookkeeping",
                        side_effect=_reconcile_bookkeeping,
                    ):
                        report = finalize_completed_run(
                            repo_root=self.repo,
                            sample_id=self.sample_id,
                            run_id=self.run_id,
                            lhs_path=self.lhs,
                            shared_root=self.shared,
                            reconcile_bookkeeping=True,
                        )
        pool = load_lhs_pool(self.lhs)
        entry = next(e for e in pool["entries"] if e["id"] == self.sample_id)
        self.assertEqual(entry["status"], LHS_COMPLETED)
        self.assertEqual(entry["last_run_id"], self.run_id)
        self.assertEqual(report["outcome"], "pass")

    def test_cleanup_failure_leaves_lhs_non_completed(self) -> None:
        self._write_lhs(status=LHS_RUNNING, last_run_id=self.run_id)
        with patch(
            "v2_b3_m4_finalize_completed_run.ensure_production_acceptance_for_finalization",
            return_value=self._acceptance_report(),
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
                        status="failed",
                        sample_success=True,
                        verification_pass=False,
                        forbidden_heavy_artifact_count=6,
                    )
                    with self.assertRaises(RuntimeError):
                        finalize_completed_run(
                            repo_root=self.repo,
                            sample_id=self.sample_id,
                            run_id=self.run_id,
                            lhs_path=self.lhs,
                            shared_root=self.shared,
                            reconcile_bookkeeping=True,
                        )
        pool = load_lhs_pool(self.lhs)
        entry = next(e for e in pool["entries"] if e["id"] == self.sample_id)
        self.assertEqual(entry["status"], LHS_RUNNING)


if __name__ == "__main__":
    unittest.main()
