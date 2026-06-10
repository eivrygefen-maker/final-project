#!/usr/bin/env python3
"""Regression tests for acceptance-only finalization recovery (no FEM)."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compact_completed_m4_runs import CompactionOutcome  # noqa: E402
from test_m4_compaction_run_selection import _make_strict_completed_run  # noqa: E402
from v2_b3_m4_finalize_completed_run import finalize_completed_run  # noqa: E402
from v2_b3_m4_physics_identity_lib import mesh_component_hashes  # noqa: E402
from v2_b3_m4_production_freeze import (  # noqa: E402
    ensure_production_acceptance_for_finalization,
    physics_identity_manifest_needs_repair,
    read_production_acceptance_status,
)
from v2_b3_m4_production_freeze_test import _write_minimal_production_aggregated_run  # noqa: E402
from v2_b3_m4_sample_cleanup_barrier import FAILURE_REPORT_REL  # noqa: E402
from v2_b3_m4_shared_export import APPROVED_SHARED_PLOT_NAMES  # noqa: E402
from v2_b3_petsc_util import write_json_atomic  # noqa: E402


def _acceptance_pass_payload() -> dict:
    return {
        "acceptance_pass": True,
        "failures": [],
        "dataset_version": "m4_geometry_corrected_rommesh_v1",
        "mesh_profile": "rom",
        "mesh_level_id": "L_rom_prod",
        "p_idx_aperture_count": 4,
    }


class FinalizeAcceptanceRecoveryTests(unittest.TestCase):
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
                    "entries": [{"id": self.sample_id, "status": "PENDING", "lhs_row_index": 0}],
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _write_completed_run_missing_acceptance(self) -> None:
        _write_minimal_production_aggregated_run(self.run_root)
        pipeline = json.loads((self.run_root / "pipeline_run_manifest.json").read_text(encoding="utf-8"))
        pipeline["terminal_status"] = "COMPLETED"
        write_json_atomic(self.run_root / "pipeline_run_manifest.json", pipeline)
        for rel in (
            "freeze/freeze_manifest.json",
            "freeze/physics_identity_manifest.json",
        ):
            path = self.run_root / rel
            if path.is_file():
                path.unlink()

    def test_mesh_component_hashes_avoids_geometry_dofmap(self) -> None:
        class _Conn:
            def links(self, c: int) -> list[int]:
                return [c, c + 1, c + 2]

        class _Topology:
            dim = 3

            def create_connectivity(self, a: int, b: int) -> None:
                return None

            def connectivity(self, a: int, b: int) -> _Conn:
                return _Conn()

            def index_map(self, dim: int) -> MagicMock:
                m = MagicMock()
                m.size_local = 2
                return m

        class _Mesh:
            geometry = MagicMock()
            geometry.x = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
            geometry.dofmap = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32)
            topology = _Topology()

        class _Tags:
            values = [1, 2]

        mesh_path = self.repo / "fixture.msh"
        mesh_path.write_bytes(b"mesh")

        with patch("v2_b3_m4_physics_identity_lib._sha256_file", return_value="meshsha"):
            with patch("v2_b3_synthesis_export.import_fem_main_3d") as imp:
                fem = MagicMock()
                fem._load_mesh_and_tags.return_value = (_Mesh(), _Tags(), None)
                fem._locate_air_volume_pressure_dofs.return_value = np.array([], dtype=np.int32)
                imp.return_value = (fem, None)
                doc = mesh_component_hashes(mesh_path, built_meta={"p_idx": [], "n_u_b3": 0})
        self.assertEqual(doc.get("status"), "ok")
        self.assertNotIn("links", str(doc.get("error") or ""))
        self.assertIn("full_mesh_topology_sha256", doc)

    def test_physics_identity_diagnostic_error_needs_repair(self) -> None:
        _write_minimal_production_aggregated_run(self.run_root)
        identity = {
            "schema": "m4_physics_identity_v1",
            "production_acceptance_pass": True,
            "mesh_components": {
                "status": "error:AttributeError",
                "error": "'numpy.ndarray' object has no attribute 'links'",
            },
        }
        write_json_atomic(self.run_root / "freeze" / "physics_identity_manifest.json", identity)
        self.assertTrue(physics_identity_manifest_needs_repair(self.run_root))

    def test_acceptance_only_recomputation_succeeds(self) -> None:
        self._write_completed_run_missing_acceptance()
        with patch(
            "v2_b3_m4_production_contracts.evaluate_production_acceptance",
            return_value=_acceptance_pass_payload(),
        ):
            with patch(
                "v2_b3_m4_production_freeze.replay_production_freeze",
                return_value=(0, "production freeze finalized"),
            ) as replay_mock:
                report = ensure_production_acceptance_for_finalization(
                    repo_root=self.repo,
                    run_root=self.run_root,
                )
        self.assertTrue(report["acceptance_pass"])
        self.assertTrue(report["production_acceptance_pass"])
        self.assertTrue(report["manifests_repaired"])
        replay_mock.assert_called_once()

    def test_real_acceptance_failure_blocks_finalization(self) -> None:
        self._write_completed_run_missing_acceptance()
        (self.run_root / "aggregation" / "aggregation_result.json").write_text(
            json.dumps(
                {
                    "status": "AGGREGATION_PARTIAL",
                    "final_aggregation_ready": False,
                    "completed_chunk_count": 1,
                    "planned_chunk_count": 12,
                    "missing_chunk_count": 11,
                    "failed_chunk_count": 0,
                }
            ),
            encoding="utf-8",
        )
        report = ensure_production_acceptance_for_finalization(
            repo_root=self.repo,
            run_root=self.run_root,
        )
        self.assertFalse(report["acceptance_pass"])

    def test_compaction_runs_only_after_acceptance_passes(self) -> None:
        from compact_completed_m4_runs import production_compaction_preconditions  # noqa: WPS433

        row_missing = {
            "outcome": "pass",
            "aggregation_status": "AGGREGATION_PASS",
            "final_aggregation_ready": True,
            "terminal_status": "COMPLETED",
        }
        ok, reason, _ = production_compaction_preconditions(
            row=row_missing,
            pool_entry={"status": "PENDING"},
            run_rom_compare=False,
        )
        self.assertFalse(ok)
        self.assertIn("production_acceptance_pass", reason)

        row_ok = dict(row_missing, production_acceptance_pass=True)
        ok2, reason2, _ = production_compaction_preconditions(
            row=row_ok,
            pool_entry={"status": "PENDING"},
            run_rom_compare=False,
        )
        self.assertTrue(ok2, reason2)

    def test_finalize_recovery_executes_no_fem_stages(self) -> None:
        self._write_completed_run_missing_acceptance()
        for name in APPROVED_SHARED_PLOT_NAMES:
            (self.run_root / "aggregation" / name).write_bytes(b"png")

        from v2_b3_m4_sample_cleanup_barrier import CleanupBarrierOutcome  # noqa: WPS433

        barrier = CleanupBarrierOutcome(
            sample_id=self.sample_id,
            run_id=self.run_id,
            status="completed",
            sample_success=True,
            verification_pass=True,
            compaction={"status": "completed"},
        )
        acceptance_report = {
            "acceptance_pass": True,
            "production_acceptance_pass": True,
            "production_acceptance_failures": [],
            "fem_stages_executed": False,
        }
        with patch(
            "v2_b3_m4_finalize_completed_run.ensure_production_acceptance_for_finalization",
            return_value=acceptance_report,
        ):
            with patch(
                "v2_b3_m4_finalize_completed_run.run_sample_cleanup_barrier",
                return_value=barrier,
            ):
                report = finalize_completed_run(
                    repo_root=self.repo,
                    sample_id=self.sample_id,
                    run_id=self.run_id,
                    lhs_path=self.lhs,
                    shared_root=self.shared,
                    reconcile_bookkeeping=False,
                )
        self.assertFalse(report["fem_stages_executed"])
        self.assertTrue(report["production_acceptance_pass"])
        self.assertEqual(report["outcome"], "pass")

    def test_cleanup_removes_stale_failure_report_on_success(self) -> None:
        failure = self.run_root / FAILURE_REPORT_REL
        failure.parent.mkdir(parents=True, exist_ok=True)
        failure.write_text("{}", encoding="utf-8")

        row = {
            "sample_id": self.sample_id,
            "run_id": self.run_id,
            "outcome": "pass",
            "aggregation_status": "AGGREGATION_PASS",
            "final_aggregation_ready": True,
            "terminal_status": "COMPLETED",
            "production_acceptance_pass": True,
        }
        verify_report = {
            "pass": True,
            "errors": [],
            "forbidden_heavy_artifact_count": 0,
            "forbidden_heavy_artifacts_present": [],
            "shared_sample_artifact_count": 0,
            "shared_sample_artifacts_present": [],
        }
        with patch(
            "v2_b3_m4_sample_cleanup_barrier.verify_cleanup_barrier",
            return_value=verify_report,
        ):
            with patch(
                "v2_b3_m4_sample_cleanup_barrier._delete_shared_only",
                return_value=([], []),
            ):
                from v2_b3_m4_sample_cleanup_barrier import run_sample_cleanup_barrier  # noqa: WPS433

                outcome = run_sample_cleanup_barrier(
                    repo_root=self.repo,
                    run_root=self.run_root,
                    sample_id=self.sample_id,
                    run_id=self.run_id,
                    row=row,
                    pool={"entries": [{"id": self.sample_id}]},
                    keep_full=True,
                )
        self.assertEqual(outcome.status, "completed")
        self.assertFalse(failure.is_file())

    def test_completed_run_missing_acceptance_field_blocks_without_repair(self) -> None:
        self._write_completed_run_missing_acceptance()
        recorded = read_production_acceptance_status(self.run_root)
        self.assertIsNone(recorded["production_acceptance_pass"])


if __name__ == "__main__":
    unittest.main()
