#!/usr/bin/env python3
"""Integration tests for production compaction hooks (synthetic run trees)."""
from __future__ import annotations

import io
import json
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compact_completed_m4_runs import (  # noqa: E402
    AGG_PASS,
    LHS_COMPLETED,
    MODE_DELETE_WITHOUT_ARCHIVE,
    compact_one_completed_run,
    compact_runs_for_samples,
    production_compaction_preconditions,
)
from run_m4_production_pipeline import _parse_args  # noqa: E402

COMPACTION_ARG_ATTRS = (
    "compact_after_sample",
    "compact_after_batch",
    "compact_keep_full_latest",
    "compact_keep_full_samples",
    "compact_nonblocking",
)
COMPACTION_CLI_FLAGS = (
    "--compact-after-sample",
    "--compact-after-batch",
    "--compact-keep-full-latest",
    "--compact-keep-full-samples",
    "--compact-nonblocking",
    "--compact-blocking",
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_completed_run(run_root: Path, *, sample_id: str, run_id: str) -> None:
    _write_json(
        run_root / "aggregation" / "aggregation_result.json",
        {
            "status": AGG_PASS,
            "final_aggregation_ready": True,
            "deduped_mode_count": 12,
            "raw_mode_count": 14,
            "completed_chunk_count": 3,
            "planned_chunk_count": 3,
            "missing_chunk_count": 0,
            "failed_chunk_count": 0,
        },
    )
    _write_json(run_root / "aggregation" / "modes_summary.json", {"deduped_mode_count": 12})
    (run_root / "aggregation" / "modes_catalog.jsonl").write_text(
        '{"frequency_hz": 120.0, "mic_output_proxy": 0.01}\n',
        encoding="utf-8",
    )
    (run_root / "aggregation" / "modes_catalog_deduped.jsonl").write_text(
        '{"frequency_hz": 120.0, "mic_output_proxy": 0.01}\n',
        encoding="utf-8",
    )
    (run_root / "aggregation" / "mode_provenance.jsonl").write_text(
        '{"mode_index": 0, "frequency_hz": 120.0}\n',
        encoding="utf-8",
    )
    (run_root / "aggregation" / "mode_frequency_plot.png").write_bytes(b"png")
    _write_json(run_root / "rom" / "rom_fom_comparison.json", {"status": "COMPLETED"})
    _write_json(run_root / "freeze" / "sample_e2e_run_manifest.json", {"status": "ok"})
    _write_json(
        run_root / "freeze" / "physics_identity_manifest.json",
        {
            "schema": "m4_physics_identity_v1",
            "sample_id": sample_id,
            "run_id": run_id,
            "production_acceptance_pass": True,
            "generated_mesh_sha256": "abc",
            "operator_mesh_matches_generated": True,
            "active_dimension": 100,
            "masks": {"p_idx_aperture_count": 4},
            "fallback_flags": {"cross_sample_reuse": False},
            "path_contamination": {"contamination_detected": False},
        },
    )
    _write_json(run_root / "pipeline_run_manifest.json", {"terminal_status": "COMPLETED"})
    _write_json(run_root / "sample" / "sample_input.json", {"sample_id": sample_id})
    (run_root / "logs").mkdir(parents=True, exist_ok=True)
    heavy = run_root / "lprod" / "checkpoint"
    heavy.mkdir(parents=True, exist_ok=True)
    (heavy / "built_metadata.json").write_text(
        json.dumps({"dataset_version": "m4_geometry_corrected_v1"}),
        encoding="utf-8",
    )
    (heavy / "A_active_csr.npz").write_bytes(b"x" * 4096)
    (run_root / "worker_results" / "chunk_01").mkdir(parents=True, exist_ok=True)
    (run_root / "worker_results" / "chunk_01" / "worker_result.json").write_text("{}", encoding="utf-8")


class ProductionPipelineParserTests(unittest.TestCase):
    def test_parse_args_empty_has_compaction_defaults(self) -> None:
        args = _parse_args([])
        for attr in COMPACTION_ARG_ATTRS:
            self.assertTrue(hasattr(args, attr), f"missing attribute: {attr}")
        self.assertFalse(args.compact_after_sample)
        self.assertFalse(args.compact_after_batch)
        self.assertEqual(args.compact_keep_full_latest, 0)
        self.assertEqual(args.compact_keep_full_samples, "")
        self.assertTrue(args.compact_nonblocking)

    def test_parse_args_compact_blocking_sets_nonblocking_false(self) -> None:
        args = _parse_args(["--compact-blocking"])
        self.assertFalse(args.compact_nonblocking)

    def test_help_lists_compaction_flags(self) -> None:
        buf = io.StringIO()
        with self.assertRaises(SystemExit):
            with redirect_stdout(buf):
                _parse_args(["--help"])
        help_text = buf.getvalue()
        for flag in COMPACTION_CLI_FLAGS:
            self.assertIn(flag, help_text, f"missing from --help: {flag}")


class CompactionProductionIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.repo_root = Path(self.tmp.name)
        self.guitars = (
            self.repo_root
            / "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars"
        )
        self.sample_id = "sample_099"
        self.run_id = "sample_099_m4prod1"
        self.run_root = self.guitars / self.sample_id / "runs" / self.run_id
        _make_completed_run(self.run_root, sample_id=self.sample_id, run_id=self.run_id)
        self.pool = {
            "shape_name": "classic",
            "entries": [
                {
                    "id": self.sample_id,
                    "status": LHS_COMPLETED,
                    "last_run_id": self.run_id,
                    "last_aggregation_status": AGG_PASS,
                    "parameters": {"top_wood_id": "spruce", "back_wood_id": "rosewood"},
                }
            ],
        }
        self.prod_row = {
            "sample_id": self.sample_id,
            "run_id": self.run_id,
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

    def test_production_gates_pass(self) -> None:
        ok, reason, _ = production_compaction_preconditions(
            row=self.prod_row,
            pool_entry=self.pool["entries"][0],
            run_rom_compare=True,
        )
        self.assertTrue(ok, reason)

    def test_production_gates_reject_partial(self) -> None:
        row = dict(self.prod_row, outcome="fail", final_aggregation_ready=False)
        ok, reason, _ = production_compaction_preconditions(
            row=row,
            pool_entry=self.pool["entries"][0],
            run_rom_compare=True,
        )
        self.assertFalse(ok)
        self.assertIn("outcome", reason)

    def test_compact_one_dry_run(self) -> None:
        out = compact_one_completed_run(
            repo_root=self.repo_root,
            pool=self.pool,
            sample_id=self.sample_id,
            run_id=self.run_id,
            dry_run=True,
            production_row=self.prod_row,
            run_rom_compare=True,
        )
        self.assertEqual(out.status, "dry_run_planned_delete")
        self.assertTrue((self.run_root / "lprod" / "checkpoint").is_dir())

    def test_compact_one_deletes_heavy(self) -> None:
        out = compact_one_completed_run(
            repo_root=self.repo_root,
            pool=self.pool,
            sample_id=self.sample_id,
            run_id=self.run_id,
            production_row=self.prod_row,
            run_rom_compare=True,
            production_trigger=True,
        )
        self.assertEqual(out.status, "completed")
        self.assertFalse((self.run_root / "lprod" / "checkpoint").exists())
        self.assertTrue((self.run_root / "aggregation" / "modes_catalog.jsonl").is_file())
        manifest = json.loads((self.run_root / "compaction" / "compaction_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest.get("mode"), MODE_DELETE_WITHOUT_ARCHIVE)
        self.assertTrue(manifest.get("production_trigger"))

    def test_idempotent_second_compact(self) -> None:
        compact_one_completed_run(
            repo_root=self.repo_root,
            pool=self.pool,
            sample_id=self.sample_id,
            run_id=self.run_id,
            production_row=self.prod_row,
            run_rom_compare=True,
        )
        out2 = compact_one_completed_run(
            repo_root=self.repo_root,
            pool=self.pool,
            sample_id=self.sample_id,
            run_id=self.run_id,
            production_row=self.prod_row,
            run_rom_compare=True,
        )
        self.assertIn(out2.status, ("already_compacted", "no_heavy_artifacts_present", "skipped"))

    def test_keep_full_sample_untouched(self) -> None:
        out = compact_one_completed_run(
            repo_root=self.repo_root,
            pool=self.pool,
            sample_id=self.sample_id,
            run_id=self.run_id,
            keep_full=True,
            production_row=self.prod_row,
            run_rom_compare=True,
        )
        self.assertEqual(out.status, "keep_full")
        self.assertTrue((self.run_root / "lprod" / "checkpoint").is_dir())

    def test_batch_keep_full_latest(self) -> None:
        other_id = "sample_098"
        other_run = "sample_098_m4prod1"
        other_root = self.guitars / other_id / "runs" / other_run
        _make_completed_run(other_root, sample_id=other_id, run_id=other_run)
        pool = {
            "shape_name": "classic",
            "entries": [
                {
                    "id": self.sample_id,
                    "status": LHS_COMPLETED,
                    "last_run_id": self.run_id,
                    "last_completed_at": "2026-06-01T00:00:00Z",
                    "last_aggregation_status": AGG_PASS,
                },
                {
                    "id": other_id,
                    "status": LHS_COMPLETED,
                    "last_run_id": other_run,
                    "last_completed_at": "2026-06-02T00:00:00Z",
                    "last_aggregation_status": AGG_PASS,
                },
            ],
        }
        rows = {
            self.sample_id: self.prod_row,
            other_id: {
                **self.prod_row,
                "sample_id": other_id,
                "run_id": other_run,
            },
        }
        summary = compact_runs_for_samples(
            repo_root=self.repo_root,
            pool=pool,
            sample_specs=[(self.sample_id, self.run_id), (other_id, other_run)],
            keep_full_latest=1,
            production_rows_by_sid=rows,
            run_rom_compare=True,
            production_trigger=True,
        )
        self.assertEqual(summary["compaction_sample_count"], 1)
        self.assertFalse((self.run_root / "lprod" / "checkpoint").exists())
        self.assertTrue((other_root / "lprod" / "checkpoint").is_dir())


if __name__ == "__main__":
    unittest.main()
