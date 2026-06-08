#!/usr/bin/env python3
"""Regression tests for mandatory per-sample cleanup barrier."""
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

from compact_completed_m4_runs import AGG_PASS, LHS_COMPLETED  # noqa: E402
from test_m4_compaction_run_selection import _make_strict_completed_run  # noqa: E402
from v2_b3_m4_lhs_production_batch import _run_sample_cleanup_barrier_for_batch  # noqa: E402
from v2_b3_m4_physics_identity_lib import count_forbidden_heavy_artifacts  # noqa: E402
from v2_b3_m4_sample_cleanup_barrier import (  # noqa: E402
    collect_shared_sample_artifact_paths,
    mesh_build_config_dir,
    mesh_convergence_root,
    pipeline_runs_root,
    run_sample_cleanup_barrier,
    verify_success_durable_outputs,
)
from v2_b3_m4_lhs_pool_bridge import specs_generated_dir  # noqa: E402


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_shared_artifacts(
    *,
    repo_root: Path,
    sample_id: str,
    run_id: str,
) -> list[Path]:
    mesh_root = mesh_convergence_root(repo_root) / "mesh"
    created: list[Path] = []
    for level in ("L_scout_coarse", "L_prod"):
        level_dir = mesh_root / level
        level_dir.mkdir(parents=True, exist_ok=True)
        for name in (
            f"{sample_id}.msh",
            f"{sample_id}_mesh_build_summary.json",
            f"{sample_id}_mesh_audit.json",
            f"{sample_id}_build.log",
        ):
            path = level_dir / name
            path.write_text("shared", encoding="utf-8")
            created.append(path)
        cfg = mesh_build_config_dir(repo_root) / f"{level}_{sample_id}.json"
        cfg.parent.mkdir(parents=True, exist_ok=True)
        _write_json(cfg, {"sample_id": sample_id})
        created.append(cfg)

    overlay = pipeline_runs_root(repo_root) / "config_overlays" / sample_id
    overlay.mkdir(parents=True, exist_ok=True)
    _write_json(overlay / "resolved_core_config.json", {"sample_id": sample_id})
    created.append(overlay)

    generated = specs_generated_dir(repo_root) / f"{run_id}.json"
    generated.parent.mkdir(parents=True, exist_ok=True)
    _write_json(generated, {"sample_id": sample_id, "run_id": run_id})
    created.append(generated)
    return created


def _make_partial_failed_run(
    run_root: Path,
    *,
    sample_id: str,
    run_id: str,
) -> None:
    _write_json(run_root / "pipeline_run_manifest.json", {"terminal_status": "FAILED"})
    _write_json(run_root / "sample" / "sample_input.json", {"sample_id": sample_id})
    heavy = run_root / "lprod" / "checkpoint"
    heavy.mkdir(parents=True, exist_ok=True)
    (heavy / "built_metadata.json").write_text("{}", encoding="utf-8")
    (heavy / "A_active_csr.npz").write_bytes(b"x" * 1024)
    scout_mesh = run_root / "scout" / "mesh" / "L_scout_coarse" / f"{sample_id}.msh"
    scout_mesh.parent.mkdir(parents=True, exist_ok=True)
    scout_mesh.write_text("mesh", encoding="utf-8")
    (run_root / "worker_results" / "chunk_01").mkdir(parents=True, exist_ok=True)
    (run_root / "worker_results" / "chunk_01" / "worker_result.json").write_text("{}", encoding="utf-8")


class SampleCleanupBarrierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.repo_root = Path(self.tmp.name)
        self.guitars = (
            self.repo_root
            / "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars"
        )
        self.sample_id = "sample_088"
        self.run_id = "sample_088_m4prod2_strict"
        self.run_root = self.guitars / self.sample_id / "runs" / self.run_id
        _make_strict_completed_run(
            self.run_root,
            sample_id=self.sample_id,
            run_id=self.run_id,
        )
        heavy = self.run_root / "lprod" / "checkpoint"
        heavy.mkdir(parents=True, exist_ok=True)
        (heavy / "built_metadata.json").write_text(
            json.dumps({"dataset_version": "m4_geometry_corrected_v1"}),
            encoding="utf-8",
        )
        (heavy / "A_active_csr.npz").write_bytes(b"x" * 4096)
        (self.run_root / "worker_results" / "chunk_01").mkdir(parents=True, exist_ok=True)
        (self.run_root / "worker_results" / "chunk_01" / "worker_result.json").write_text(
            "{}", encoding="utf-8"
        )
        _write_json(self.run_root / "aggregation" / "runtime_summary.json", {"status": "ok"})
        self.shared_paths = _make_shared_artifacts(
            repo_root=self.repo_root,
            sample_id=self.sample_id,
            run_id=self.run_id,
        )
        self.pool = {
            "shape_name": "classic",
            "entries": [
                {
                    "id": self.sample_id,
                    "status": LHS_COMPLETED,
                    "last_run_id": self.run_id,
                    "last_aggregation_status": AGG_PASS,
                }
            ],
        }
        self.prod_row = {
            "sample_id": self.sample_id,
            "run_id": self.run_id,
            "run_root_abs": str(self.run_root.resolve()),
            "outcome": "pass",
            "aggregation_status": AGG_PASS,
            "final_aggregation_ready": True,
            "terminal_status": "COMPLETED",
            "production_acceptance_pass": True,
            "shared_export": {"export_status": "EXPORTED"},
            "rom_compare": {"status": "COMPLETED"},
        }

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_success_cleanup_removes_shared_artifacts(self) -> None:
        outcome = run_sample_cleanup_barrier(
            repo_root=self.repo_root,
            run_root=self.run_root,
            sample_id=self.sample_id,
            run_id=self.run_id,
            row=self.prod_row,
            pool=self.pool,
            blocking=True,
        )
        self.assertEqual(outcome.status, "completed")
        self.assertEqual(outcome.shared_sample_artifact_count, 0)
        self.assertEqual(outcome.forbidden_heavy_artifact_count, 0)
        remaining = collect_shared_sample_artifact_paths(
            repo_root=self.repo_root,
            sample_id=self.sample_id,
            run_id=self.run_id,
        )
        self.assertFalse(any(p.exists() for p in remaining))

    def test_failed_sample_cleanup_removes_partial_shared_artifacts(self) -> None:
        failed_root = self.guitars / "sample_089" / "runs" / "sample_089_m4prod2_strict"
        _make_partial_failed_run(
            failed_root,
            sample_id="sample_089",
            run_id="sample_089_m4prod2_strict",
        )
        _make_shared_artifacts(
            repo_root=self.repo_root,
            sample_id="sample_089",
            run_id="sample_089_m4prod2_strict",
        )
        row = {
            "sample_id": "sample_089",
            "run_id": "sample_089_m4prod2_strict",
            "outcome": "fail",
            "error_message": "scout_failed",
            "return_code": 2,
            "aggregation_status": "AGGREGATION_PARTIAL",
        }
        outcome = run_sample_cleanup_barrier(
            repo_root=self.repo_root,
            run_root=failed_root,
            sample_id="sample_089",
            run_id="sample_089_m4prod2_strict",
            row=row,
            pool=self.pool,
            blocking=True,
        )
        self.assertEqual(outcome.status, "completed")
        self.assertEqual(outcome.forbidden_heavy_artifact_count, 0)
        self.assertEqual(outcome.shared_sample_artifact_count, 0)
        self.assertTrue((failed_root / "cleanup" / "sample_failure_retention.json").is_file())
        self.assertTrue((failed_root / "logs" / "sample_failure_diagnostic.log").is_file())
        count, _ = count_forbidden_heavy_artifacts(failed_root)
        self.assertEqual(count, 0)

    def test_next_sample_blocked_when_forbidden_path_remains(self) -> None:
        row = dict(self.prod_row)
        allowed = _run_sample_cleanup_barrier_for_batch(
            row=row,
            repo_root=self.repo_root,
            pool=self.pool,
            compact_after_sample=True,
            compact_keep_full_samples={self.sample_id},
            compact_nonblocking=False,
            run_rom_compare=True,
            strict_production=True,
        )
        self.assertFalse(allowed)
        self.assertEqual(row["cleanup_barrier"]["status"], "failed")
        self.assertGreater(row["cleanup_barrier"]["forbidden_heavy_artifact_count"], 0)
        self.assertTrue(
            (self.run_root / "cleanup" / "sample_cleanup_failure_report.json").is_file()
        )

    def test_rom_provenance_plots_summaries_retained(self) -> None:
        outcome = run_sample_cleanup_barrier(
            repo_root=self.repo_root,
            run_root=self.run_root,
            sample_id=self.sample_id,
            run_id=self.run_id,
            row=self.prod_row,
            pool=self.pool,
            blocking=True,
        )
        self.assertEqual(outcome.status, "completed")
        ok, errors = verify_success_durable_outputs(self.run_root)
        self.assertTrue(ok, errors)
        self.assertTrue((self.run_root / "aggregation" / "mode_provenance.jsonl").is_file())
        self.assertTrue((self.run_root / "aggregation" / "mode_frequency_plot.png").is_file())
        self.assertTrue((self.run_root / "freeze" / "physics_identity_manifest.json").is_file())

    def test_generated_specs_and_shared_mesh_config_removed(self) -> None:
        run_sample_cleanup_barrier(
            repo_root=self.repo_root,
            run_root=self.run_root,
            sample_id=self.sample_id,
            run_id=self.run_id,
            row=self.prod_row,
            pool=self.pool,
            blocking=True,
        )
        self.assertFalse(
            (specs_generated_dir(self.repo_root) / f"{self.run_id}.json").exists()
        )
        self.assertFalse(
            (
                pipeline_runs_root(self.repo_root) / "config_overlays" / self.sample_id
            ).exists()
        )
        for level in ("L_scout_coarse", "L_prod"):
            mesh_dir = mesh_convergence_root(self.repo_root) / "mesh" / level
            self.assertFalse((mesh_dir / f"{self.sample_id}.msh").exists())
            self.assertFalse(
                (mesh_build_config_dir(self.repo_root) / f"{level}_{self.sample_id}.json").exists()
            )

    def test_cleanup_failure_not_pass_with_warning(self) -> None:
        with patch(
            "v2_b3_m4_sample_cleanup_barrier.verify_cleanup_barrier",
            return_value={
                "pass": False,
                "errors": ["shared_sample_artifacts:['/tmp/stale.msh']"],
                "forbidden_heavy_artifact_count": 1,
                "shared_sample_artifact_count": 1,
                "forbidden_heavy_artifacts_present": ["lprod/checkpoint"],
                "shared_sample_artifacts_present": ["/tmp/stale.msh"],
            },
        ):
            outcome = run_sample_cleanup_barrier(
                repo_root=self.repo_root,
                run_root=self.run_root,
                sample_id=self.sample_id,
                run_id=self.run_id,
                row=self.prod_row,
                pool=self.pool,
                blocking=True,
            )
        self.assertEqual(outcome.status, "failed")
        self.assertNotEqual(outcome.status, "PASS_WITH_WARNING")
        self.assertNotIn("PASS_WITH_WARNING", outcome.errors)
        self.assertFalse(outcome.verification_pass)
        barrier_doc = json.loads(
            (self.run_root / "cleanup" / "sample_cleanup_barrier.json").read_text(encoding="utf-8")
        )
        self.assertEqual(barrier_doc["status"], "failed")
        self.assertNotEqual(barrier_doc.get("verification", {}).get("status"), "PASS_WITH_WARNING")


if __name__ == "__main__":
    unittest.main()
