#!/usr/bin/env python3
"""Tests for simplified shared export policy (no FEM)."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from test_m4_compaction_run_selection import _make_strict_completed_run  # noqa: E402
from v2_b3_m4_finalize_completed_run import finalize_completed_run  # noqa: E402
from v2_b3_m4_lhs_pool_bridge import classify_batch_sample_outcome  # noqa: E402
from v2_b3_m4_shared_export import (  # noqa: E402
    APPROVED_SHARED_PLOT_NAMES,
    EXCLUDED_SHARED_PLOT_NAMES,
    GRAPH_EXPORT_MANIFEST_SCHEMA,
    SUMMARY_SCHEMA,
    export_graphs_fixture,
    export_sample_to_shared,
    graph_manifest_filename,
    remove_stale_shared_exports_for_run,
    run_plot_filename,
    sample_plots_destination_dir,
    summaries_destination_dir,
    summary_json_filename,
)
from unittest.mock import patch  # noqa: E402


class SharedExportPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.shared = Path(self.tmp.name) / "sf_gmar"
        self.shared.mkdir()
        self.run_root = Path(self.tmp.name) / "run"
        self.sample_id = "sample_000"
        self.run_id = "sample_000_rom_official_v1"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _write_approved_plots(self) -> None:
        agg = self.run_root / "aggregation"
        agg.mkdir(parents=True, exist_ok=True)
        for name in APPROVED_SHARED_PLOT_NAMES:
            (agg / name).write_bytes(b"\x89PNG\r\n\x1a\nfixture\n")
        (agg / "mode_frequency_plot.png").write_bytes(b"\x89PNG\r\n\x1a\nexcluded\n")

    def test_png_destination_is_sample_subfolder(self) -> None:
        self._write_approved_plots()
        manifest = export_sample_to_shared(
            run_root=self.run_root,
            sample_id=self.sample_id,
            run_id=self.run_id,
            shared_root=self.shared,
        )
        plots_dir = sample_plots_destination_dir(
            shared_root=self.shared,
            shape_name="classic",
            sample_id=self.sample_id,
        )
        self.assertEqual(manifest["plots_dir"], str(plots_dir))
        self.assertEqual(plots_dir, self.shared / "classic" / "plots" / self.sample_id)
        for name in APPROVED_SHARED_PLOT_NAMES:
            dest = plots_dir / run_plot_filename(self.run_id, name)
            self.assertTrue(dest.is_file(), dest)

    def test_json_destination_is_summaries_root(self) -> None:
        self._write_approved_plots()
        manifest = export_sample_to_shared(
            run_root=self.run_root,
            sample_id=self.sample_id,
            run_id=self.run_id,
            shared_root=self.shared,
        )
        summaries = summaries_destination_dir(shared_root=self.shared, shape_name="classic")
        self.assertEqual(manifest["summaries_dir"], str(summaries))
        summary_path = summaries / summary_json_filename(self.sample_id, self.run_id)
        manifest_path = summaries / graph_manifest_filename(self.sample_id, self.run_id)
        self.assertTrue(summary_path.is_file())
        self.assertTrue(manifest_path.is_file())
        self.assertFalse(any(summaries.joinpath(p).suffix == ".png" for p in summaries.iterdir()))

    def test_mode_frequency_plot_excluded(self) -> None:
        self._write_approved_plots()
        export_sample_to_shared(
            run_root=self.run_root,
            sample_id=self.sample_id,
            run_id=self.run_id,
            shared_root=self.shared,
        )
        plots_dir = self.shared / "classic" / "plots" / self.sample_id
        self.assertFalse((plots_dir / run_plot_filename(self.run_id, "mode_frequency_plot.png")).exists())
        self.assertIn("mode_frequency_plot.png", EXCLUDED_SHARED_PLOT_NAMES)

    def test_run_id_safe_filenames_do_not_overwrite_other_runs(self) -> None:
        self._write_approved_plots()
        other_run = "sample_000_rom_official_v0"
        export_sample_to_shared(
            run_root=self.run_root,
            sample_id=self.sample_id,
            run_id=other_run,
            shared_root=self.shared,
        )
        export_sample_to_shared(
            run_root=self.run_root,
            sample_id=self.sample_id,
            run_id=self.run_id,
            shared_root=self.shared,
        )
        plots_dir = self.shared / "classic" / "plots" / self.sample_id
        self.assertTrue((plots_dir / run_plot_filename(other_run, APPROVED_SHARED_PLOT_NAMES[0])).is_file())
        self.assertTrue((plots_dir / run_plot_filename(self.run_id, APPROVED_SHARED_PLOT_NAMES[0])).is_file())

    def test_graph_manifest_fields(self) -> None:
        self._write_approved_plots()
        export_sample_to_shared(
            run_root=self.run_root,
            sample_id=self.sample_id,
            run_id=self.run_id,
            shared_root=self.shared,
        )
        manifest_path = (
            self.shared / "classic" / "summaries" / graph_manifest_filename(self.sample_id, self.run_id)
        )
        doc = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(doc["schema"], GRAPH_EXPORT_MANIFEST_SCHEMA)
        for entry in doc["entries"]:
            self.assertIn("source_path", entry)
            self.assertIn("destination_path", entry)
            self.assertIn("sha256", entry)
            self.assertIn("size_bytes", entry)
            self.assertIn("copy_status", entry)

    def test_summary_json_schema(self) -> None:
        self._write_approved_plots()
        export_sample_to_shared(
            run_root=self.run_root,
            sample_id=self.sample_id,
            run_id=self.run_id,
            shared_root=self.shared,
        )
        summary_path = self.shared / "classic" / "summaries" / summary_json_filename(
            self.sample_id, self.run_id
        )
        doc = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(doc["schema"], SUMMARY_SCHEMA)
        for key in (
            "sample_id",
            "run_id",
            "mesh_profile",
            "aggregation_status",
            "graph_export_status",
            "graph_destination_paths",
        ):
            self.assertIn(key, doc)

    def test_no_json_inside_plots_directory(self) -> None:
        self._write_approved_plots()
        export_sample_to_shared(
            run_root=self.run_root,
            sample_id=self.sample_id,
            run_id=self.run_id,
            shared_root=self.shared,
        )
        plots_dir = self.shared / "classic" / "plots" / self.sample_id
        self.assertFalse(any(p.suffix == ".json" for p in plots_dir.iterdir()))

    def test_legacy_nested_layout_removed(self) -> None:
        nested = self.shared / "classic" / "m4_production" / self.sample_id / self.run_id / "graphs"
        nested.mkdir(parents=True)
        (nested / "old.png").write_bytes(b"x")
        removed = remove_stale_shared_exports_for_run(
            shared_root=self.shared,
            shape_name="classic",
            sample_id=self.sample_id,
            run_id=self.run_id,
        )
        self.assertTrue(removed)
        self.assertFalse(nested.exists())

    def test_missing_present_plot_fails_export(self) -> None:
        agg = self.run_root / "aggregation"
        agg.mkdir(parents=True, exist_ok=True)
        (agg / APPROVED_SHARED_PLOT_NAMES[0]).write_bytes(b"png")
        manifest = export_sample_to_shared(
            run_root=self.run_root,
            sample_id=self.sample_id,
            run_id=self.run_id,
            shared_root=self.shared,
        )
        self.assertEqual(manifest["export_status"], "EXPORTED")

    def test_finalize_recovery_does_not_run_fem(self) -> None:
        repo = Path(self.tmp.name) / "repo"
        guitars = (
            repo
            / "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars"
            / self.sample_id
            / "runs"
            / self.run_id
        )
        _make_strict_completed_run(guitars, sample_id=self.sample_id, run_id=self.run_id)
        for name in APPROVED_SHARED_PLOT_NAMES:
            (guitars / "aggregation" / name).write_bytes(b"png")
        lhs = repo / "ROM/classic/lhs_pool.json"
        lhs.parent.mkdir(parents=True, exist_ok=True)
        lhs.write_text(
            json.dumps({"shape_name": "classic", "entries": [{"id": self.sample_id, "status": "PENDING"}]}),
            encoding="utf-8",
        )
        from v2_b3_m4_sample_cleanup_barrier import CleanupBarrierOutcome  # noqa: WPS433

        barrier = CleanupBarrierOutcome(
            sample_id=self.sample_id,
            run_id=self.run_id,
            status="completed",
            sample_success=True,
            verification_pass=True,
            compaction={"status": "completed"},
        )
        with patch(
            "v2_b3_m4_finalize_completed_run.run_sample_cleanup_barrier",
            return_value=barrier,
        ):
            report = finalize_completed_run(
                repo_root=repo,
                sample_id=self.sample_id,
                run_id=self.run_id,
                lhs_path=lhs,
                shared_root=self.shared,
                reconcile_bookkeeping=False,
            )
        self.assertFalse(report["fem_stages_executed"])
        self.assertEqual(report["outcome"], "pass")

    def test_compaction_blocks_only_after_export_success(self) -> None:
        outcome, err = classify_batch_sample_outcome(
            return_code=0,
            summary={
                "terminal_status": "COMPLETED",
                "aggregation_status": "AGGREGATION_PASS",
                "failed_chunks": 0,
                "missing_chunks": 0,
                "final_aggregation_ready": True,
            },
            cleanup_barrier={"status": "failed"},
            require_cleanup_barrier=True,
            shared_export={"export_status": "EXPORTED", "graph_export_entries": [{"copy_status": "copied"}],
                           "summary_export_path": "/tmp/s.json", "graph_manifest_export_path": "/tmp/g.json"},
            require_graph_export=True,
        )
        self.assertEqual(outcome, "fail")
        self.assertIn("cleanup_barrier_status=failed", err or "")


if __name__ == "__main__":
    unittest.main()
