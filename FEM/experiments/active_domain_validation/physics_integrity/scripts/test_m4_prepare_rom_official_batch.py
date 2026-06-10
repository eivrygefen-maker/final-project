#!/usr/bin/env python3
"""Tests for official ROM batch preparation helpers (no FEM)."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_m4_full_lhs_pool_reset import (  # noqa: E402
    LHS_PENDING,
    apply_full_lhs_pool_reset,
    plan_full_lhs_pool_reset,
    reset_pool_entries,
    verify_all_entries_pending,
)
from v2_b3_m4_lhs_pool_bridge import (  # noqa: E402
    LHS_COMPLETED,
    LHS_RUNNING,
    classify_batch_sample_outcome,
    load_lhs_pool,
)
from v2_b3_m4_mesh_profile_lib import (  # noqa: E402
    DATASET_VERSION_ROM,
    LEVEL_ROM_PROD,
    MESH_PROFILE_ROM,
    checkpoint_export_mesh_level,
    resolve_mesh_profile,
)
from v2_b3_m4_prepare_rom_official_batch import (  # noqa: E402
    OFFICIAL_MAX_SAMPLES,
    OFFICIAL_RUN_ID_SUFFIX,
    build_prepare_report,
    simulate_post_reset_selection,
    verify_unique_run_roots,
)
from v2_b3_m4_shared_export import (  # noqa: E402
    GRAPH_EXPORT_MANIFEST_SCHEMA,
    export_graphs_fixture,
    graphs_destination_dir,
)
from v2_b3_petsc_util import write_json_atomic  # noqa: E402


def _mini_pool() -> dict:
    entries = []
    for i in range(12):
        row = {
            "id": f"sample_{i:03d}",
            "parameters": {"geometry.length": 0.4 + i * 0.01},
            "status": LHS_PENDING if i >= 5 else LHS_COMPLETED,
        }
        if i == 0:
            row["status"] = LHS_RUNNING
            row["last_run_id"] = "sample_000_old"
        if i == 2:
            row["last_run_id"] = "sample_002_rom_prod_004"
        entries.append(row)
    return {"shape_name": "classic", "entries": entries}


class PrepareRomOfficialBatchTests(unittest.TestCase):
    def test_default_mesh_profile_is_rom(self) -> None:
        resolved = resolve_mesh_profile()
        self.assertEqual(resolved.mesh_profile, MESH_PROFILE_ROM)
        self.assertEqual(resolved.mesh_level_id, LEVEL_ROM_PROD)
        self.assertEqual(resolved.dataset_version, DATASET_VERSION_ROM)

    def test_stage_a_checkpoint_export_level_is_l_prod(self) -> None:
        self.assertEqual(checkpoint_export_mesh_level(), "L_prod")

    def test_full_reset_plan_covers_all_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            lhs = repo / "ROM/classic/lhs_pool.json"
            lhs.parent.mkdir(parents=True)
            write_json_atomic(lhs, _mini_pool())
            plan = plan_full_lhs_pool_reset(repo_root=repo, lhs_path=lhs)
            self.assertEqual(plan["total_lhs_entries"], 12)
            self.assertEqual(len(plan["lhs_entries_to_reset"]), 12)
            self.assertTrue(plan["run_trees_are_not_deleted"])
            self.assertEqual(plan["run_directory_deletions"], 0)

    def test_full_reset_execute_makes_all_pending(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            lhs = repo / "ROM/classic/lhs_pool.json"
            lhs.parent.mkdir(parents=True)
            write_json_atomic(lhs, _mini_pool())
            result = apply_full_lhs_pool_reset(repo_root=repo, lhs_path=lhs)
            pool = load_lhs_pool(lhs)
            self.assertTrue(result["all_entries_pending"])
            self.assertTrue(verify_all_entries_pending(pool))
            entry2 = pool["entries"][2]
            self.assertEqual(entry2["status"], LHS_PENDING)
            self.assertNotIn("last_run_id", entry2)

    def test_post_reset_selection_first_five_in_order(self) -> None:
        pool = _mini_pool()
        selection = simulate_post_reset_selection(
            pool=pool,
            max_samples=OFFICIAL_MAX_SAMPLES,
            run_id_suffix=OFFICIAL_RUN_ID_SUFFIX,
        )
        self.assertEqual(len(selection), 5)
        self.assertEqual([row["lhs_row_index"] for row in selection], [0, 1, 2, 3, 4])
        self.assertEqual(
            [row["sample_id"] for row in selection],
            [f"sample_{i:03d}" for i in range(5)],
        )

    def test_unique_run_root_checks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            selection = simulate_post_reset_selection(
                pool=_mini_pool(),
                max_samples=5,
                run_id_suffix=OFFICIAL_RUN_ID_SUFFIX,
            )
            checks = verify_unique_run_roots(repo_root=repo, selection=selection)
            self.assertEqual(len(checks), 5)
            self.assertTrue(all(c["ok"] for c in checks))

    def test_graph_export_fixture_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            shared = repo / "shared"
            shared.mkdir()
            run_root = repo / "run"
            manifest = export_graphs_fixture(
                run_root=run_root,
                sample_id="sample_000",
                run_id="sample_000_rom_official_v1",
                shared_root=shared,
            )
            self.assertEqual(manifest.get("export_status"), "EXPORTED")
            graphs_dir = graphs_destination_dir(
                shared_root=shared,
                shape_name="classic",
                sample_id="sample_000",
                run_id="sample_000_rom_official_v1",
            )
            graph_manifest = json.loads((graphs_dir / "graph_export_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(graph_manifest["schema"], GRAPH_EXPORT_MANIFEST_SCHEMA)

    def test_graph_export_failure_blocks_classification(self) -> None:
        outcome, err = classify_batch_sample_outcome(
            return_code=0,
            summary={
                "terminal_status": "COMPLETED",
                "aggregation_status": "AGGREGATION_PASS",
                "failed_chunks": 0,
                "missing_chunks": 0,
                "final_aggregation_ready": True,
            },
            cleanup_barrier={
                "status": "completed",
                "verification_pass": True,
                "forbidden_heavy_artifact_count": 0,
                "shared_sample_artifact_count": 0,
            },
            require_cleanup_barrier=True,
            shared_export={"export_status": "FAILED"},
            require_graph_export=True,
        )
        self.assertEqual(outcome, "fail")
        self.assertIn("graph_export_status=FAILED", err or "")

    def test_prepare_report_uses_normal_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            lhs = repo / "ROM/classic/lhs_pool.json"
            lhs.parent.mkdir(parents=True)
            write_json_atomic(lhs, _mini_pool())
            report = build_prepare_report(
                repo_root=repo,
                lhs_path=lhs,
                batch_id="lhs_rom_official_v1_test",
                run_id_suffix=OFFICIAL_RUN_ID_SUFFIX,
                max_samples=5,
                execute_reset=False,
            )
            self.assertFalse(report["bounded_index_selection"])
            self.assertEqual(report["post_reset_selection_count"], 5)
            self.assertEqual(
                [row["lhs_index"] for row in report["post_reset_pipeline_selection"]],
                [0, 1, 2, 3, 4],
            )
            self.assertFalse(report["fem_launched"])


if __name__ == "__main__":
    unittest.main()
