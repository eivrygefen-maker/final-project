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

from v2_b3_m4_bounded_lhs_reset import (  # noqa: E402
    apply_bounded_lhs_reset,
    plan_bounded_lhs_reset,
)
from v2_b3_m4_lhs_pool_bridge import (  # noqa: E402
    LHS_COMPLETED,
    LHS_PENDING,
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
    OFFICIAL_END_INDEX,
    OFFICIAL_RUN_ID_SUFFIX,
    OFFICIAL_START_INDEX,
    build_official_sample_rows,
    build_prepare_report,
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
    for i in range(8):
        entries.append(
            {
                "id": f"sample_{i:03d}",
                "parameters": {"geometry.length": 0.4 + i * 0.01},
                "status": LHS_COMPLETED if i < 5 else LHS_PENDING,
                "last_run_id": f"sample_{i:03d}_old_run",
            }
        )
    entries[2]["last_run_id"] = "sample_002_rom_prod_004"
    return {"shape_name": "classic", "entries": entries}


class PrepareRomOfficialBatchTests(unittest.TestCase):
    def test_default_mesh_profile_is_rom(self) -> None:
        resolved = resolve_mesh_profile()
        self.assertEqual(resolved.mesh_profile, MESH_PROFILE_ROM)
        self.assertEqual(resolved.mesh_level_id, LEVEL_ROM_PROD)
        self.assertEqual(resolved.dataset_version, DATASET_VERSION_ROM)

    def test_stage_a_checkpoint_export_level_is_l_prod(self) -> None:
        self.assertEqual(checkpoint_export_mesh_level(), "L_prod")

    def test_bounded_reset_plan_only_indexes_0_to_4(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            lhs = repo / "ROM/classic/lhs_pool.json"
            lhs.parent.mkdir(parents=True)
            write_json_atomic(lhs, _mini_pool())
            plan = plan_bounded_lhs_reset(
                repo_root=repo,
                lhs_path=lhs,
                start_index=0,
                end_index=4,
                preserved_run_ids=["sample_002_rom_prod_004"],
            )
            indexes = {row["lhs_index"] for row in plan["lhs_entries_to_reset"]}
            self.assertEqual(indexes, {0, 1, 2, 3, 4})
            untouched = plan.get("lhs_entries_untouched_after_end") or []
            self.assertTrue(any(row["lhs_index"] == 5 for row in untouched))

    def test_bounded_reset_execute_preserves_validation_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            lhs = repo / "ROM/classic/lhs_pool.json"
            lhs.parent.mkdir(parents=True)
            write_json_atomic(lhs, _mini_pool())
            apply_bounded_lhs_reset(
                repo_root=repo,
                lhs_path=lhs,
                start_index=0,
                end_index=4,
                preserved_run_ids=["sample_002_rom_prod_004"],
            )
            pool = load_lhs_pool(lhs)
            entry2 = pool["entries"][2]
            self.assertEqual(entry2["status"], LHS_PENDING)
            self.assertNotIn("last_run_id", entry2)

    def test_unique_run_root_checks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            rows = build_official_sample_rows(
                pool=_mini_pool(),
                batch_id="b1",
                lhs_source_path="ROM/classic/lhs_pool.json",
                start_index=0,
                end_index=4,
                run_id_suffix=OFFICIAL_RUN_ID_SUFFIX,
            )
            checks = verify_unique_run_roots(repo_root=repo, rows=rows)
            self.assertEqual(len(checks), 5)
            self.assertTrue(all(c["ok"] for c in checks))
            self.assertTrue(all(c["unique_run_id"] for c in checks))

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
            entries = graph_manifest.get("entries") or []
            self.assertTrue(entries)
            self.assertTrue(all(e.get("sha256") for e in entries))
            self.assertTrue(all(e.get("copy_status") == "copied" for e in entries))

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

    def test_prepare_report_indexes_0_to_4_only(self) -> None:
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
                start_index=OFFICIAL_START_INDEX,
                end_index=OFFICIAL_END_INDEX,
                execute_reset=False,
            )
            mapping = report["sample_mapping"]
            self.assertEqual(len(mapping), 5)
            self.assertEqual([row["lhs_index"] for row in mapping], [0, 1, 2, 3, 4])
            self.assertTrue(report["index_5_plus_excluded"])
            self.assertFalse(report["fem_launched"])


if __name__ == "__main__":
    unittest.main()
