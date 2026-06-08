#!/usr/bin/env python3
"""Regression tests for precise run selection in compact_completed_m4_runs.py."""
from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compact_completed_m4_runs import (  # noqa: E402
    AGG_PASS,
    LHS_COMPLETED,
    _process_delete_without_archive,
    _strict_precheck_run_root,
    guitars_root,
    main,
    resolve_compaction_run_selections,
)
from v2_b3_m4_physics_identity_lib import (  # noqa: E402
    count_forbidden_heavy_artifacts,
    verify_post_compaction_contract,
)

RUN_A = "sample_000_m4prod2"
RUN_B = "sample_000_m4prod2_strict_val"


def _selected_paths_from_output(out: str) -> list[str]:
    paths: list[str] = []
    in_section = False
    for line in out.splitlines():
        if line.strip() == "selected_run_paths:":
            in_section = True
            continue
        if in_section:
            if not line.startswith("  "):
                break
            paths.append(line.strip())
    return paths


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_strict_completed_run(
    run_root: Path,
    *,
    sample_id: str,
    run_id: str,
    terminal_status: str = "COMPLETED",
    production_acceptance_pass: bool = True,
    include_physics_manifest: bool = True,
    aggregation_pass: bool = True,
) -> None:
    agg_status = AGG_PASS if aggregation_pass else "AGGREGATION_PARTIAL"
    _write_json(
        run_root / "aggregation" / "aggregation_result.json",
        {
            "status": agg_status,
            "final_aggregation_ready": aggregation_pass,
            "deduped_mode_count": 12,
            "raw_mode_count": 14,
            "completed_chunk_count": 3 if aggregation_pass else 1,
            "planned_chunk_count": 3,
            "missing_chunk_count": 0 if aggregation_pass else 2,
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
    _write_json(run_root / "freeze" / "freeze_manifest.json", {"status": "ok"})
    _write_json(run_root / "pipeline_run_manifest.json", {"terminal_status": terminal_status})
    _write_json(run_root / "sample" / "sample_input.json", {"sample_id": sample_id})
    if include_physics_manifest:
        _write_json(
            run_root / "freeze" / "physics_identity_manifest.json",
            {
                "schema": "m4_physics_identity_v1",
                "sample_id": sample_id,
                "run_id": run_id,
                "production_acceptance_pass": production_acceptance_pass,
                "generated_mesh_sha256": "abc",
                "operator_mesh_matches_generated": True,
                "active_dimension": 100,
                "masks": {"p_idx_aperture_count": 4},
                "fallback_flags": {"cross_sample_reuse": False},
                "path_contamination": {"contamination_detected": False},
            },
        )
    (run_root / "logs").mkdir(parents=True, exist_ok=True)
    heavy = run_root / "lprod" / "checkpoint"
    heavy.mkdir(parents=True, exist_ok=True)
    (heavy / "built_metadata.json").write_text(
        json.dumps({"dataset_version": "m4_geometry_corrected_v1"}),
        encoding="utf-8",
    )
    (heavy / "A_active_csr.npz").write_bytes(b"x" * 4096)
    (run_root / "scout" / "checkpoint").mkdir(parents=True, exist_ok=True)
    (run_root / "scout" / "mesh").mkdir(parents=True, exist_ok=True)
    (run_root / "worker_results" / "chunk_01").mkdir(parents=True, exist_ok=True)
    (run_root / "worker_results" / "chunk_01" / "worker_result.json").write_text("{}", encoding="utf-8")


class CompactionRunSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.repo_root = Path(self.tmp.name)
        self.guitars = guitars_root(self.repo_root)
        self.sample_id = "sample_000"
        self.run_a_root = self.guitars / self.sample_id / "runs" / RUN_A
        self.run_b_root = self.guitars / self.sample_id / "runs" / RUN_B
        _make_strict_completed_run(self.run_a_root, sample_id=self.sample_id, run_id=RUN_A)
        _make_strict_completed_run(self.run_b_root, sample_id=self.sample_id, run_id=RUN_B)
        self.lhs_path = self.repo_root / "ROM" / "classic" / "lhs_pool.json"
        self.lhs_path.parent.mkdir(parents=True, exist_ok=True)
        self.lhs_path.write_text(
            json.dumps(
                {
                    "shape_name": "classic",
                    "entries": [
                        {
                            "id": self.sample_id,
                            "status": LHS_COMPLETED,
                            "last_run_id": RUN_B,
                            "last_aggregation_status": AGG_PASS,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_two_runs_exist_for_sample_000(self) -> None:
        self.assertTrue(self.run_a_root.is_dir())
        self.assertTrue(self.run_b_root.is_dir())

    def test_run_id_suffix_selects_strict_val_only(self) -> None:
        selections, errors = resolve_compaction_run_selections(
            self.repo_root,
            sample_ids=[self.sample_id],
            run_id_suffix="m4prod2_strict_val",
        )
        self.assertEqual(errors, [])
        self.assertEqual(len(selections), 1)
        self.assertEqual(selections[0].run_id, RUN_B)
        self.assertEqual(selections[0].run_root, self.run_b_root)
        self.assertFalse(selections[0].ambiguous)

    def test_run_dir_selects_exactly_one_run(self) -> None:
        selections, errors = resolve_compaction_run_selections(
            self.repo_root,
            sample_ids=[],
            run_dir=self.run_b_root,
        )
        self.assertEqual(errors, [])
        self.assertEqual(len(selections), 1)
        self.assertEqual(selections[0].run_id, RUN_B)
        self.assertEqual(selections[0].run_root.resolve(), self.run_b_root.resolve())

    def test_ambiguous_sample_range_refuses_execute(self) -> None:
        buf_err = io.StringIO()
        with patch("compact_completed_m4_runs.detect_repo_root", return_value=self.repo_root):
            with redirect_stderr(buf_err):
                code = main(
                    [
                        "--lhs-json",
                        str(self.lhs_path),
                        "--sample-range",
                        "0",
                        "--delete-heavy-without-archive",
                        "--execute",
                    ]
                )
        self.assertEqual(code, 2)
        self.assertIn("ambiguous", buf_err.getvalue().lower())
        self.assertTrue((self.run_a_root / "lprod" / "checkpoint").is_dir())
        self.assertTrue((self.run_b_root / "lprod" / "checkpoint").is_dir())

    def test_incomplete_run_refuses_compaction(self) -> None:
        incomplete = self.guitars / self.sample_id / "runs" / "sample_000_incomplete"
        _make_strict_completed_run(
            incomplete,
            sample_id=self.sample_id,
            run_id="sample_000_incomplete",
            terminal_status="RUNNING",
            aggregation_pass=False,
        )
        ok, reason = _strict_precheck_run_root(incomplete)
        self.assertFalse(ok)
        self.assertIn("terminal_status", reason)

    def test_missing_physics_identity_refuses_compaction(self) -> None:
        no_phys = self.guitars / self.sample_id / "runs" / "sample_000_no_phys"
        _make_strict_completed_run(
            no_phys,
            sample_id=self.sample_id,
            run_id="sample_000_no_phys",
            include_physics_manifest=False,
        )
        ok, reason = _strict_precheck_run_root(no_phys)
        self.assertFalse(ok)
        self.assertIn("physics_identity", reason)

    def test_dry_run_prints_exact_selected_path(self) -> None:
        buf_out = io.StringIO()
        with patch("compact_completed_m4_runs.detect_repo_root", return_value=self.repo_root):
            with redirect_stdout(buf_out):
                code = main(
                    [
                        "--lhs-json",
                        str(self.lhs_path),
                        "--sample-range",
                        "0",
                        "--run-id-suffix",
                        "m4prod2_strict_val",
                        "--keep-full-latest",
                        "0",
                        "--delete-heavy-without-archive",
                        "--dry-run",
                    ]
                )
        self.assertEqual(code, 0)
        out = buf_out.getvalue()
        self.assertIn("selected_run_paths:", out)
        selected = _selected_paths_from_output(out)
        self.assertEqual(selected, [str(self.run_b_root)])

    def test_post_delete_verification_zero_forbidden_heavy(self) -> None:
        from compact_completed_m4_runs import _eligible_run  # noqa: WPS433

        rec = _eligible_run(
            repo_root=self.repo_root,
            entry={"id": self.sample_id, "status": LHS_COMPLETED},
            sample_id=self.sample_id,
            run_id=RUN_B,
            run_root=self.run_b_root,
            explicit_selection=True,
        )
        self.assertTrue(rec.eligible, rec.skip_reason)
        _process_delete_without_archive(rec, dry_run=False, repo_root=self.repo_root)
        self.assertEqual(rec.status, "completed")

        count, paths = count_forbidden_heavy_artifacts(self.run_b_root)
        self.assertEqual(count, 0, paths)

        report = verify_post_compaction_contract(self.run_b_root)
        self.assertEqual(report.get("forbidden_heavy_artifact_count"), 0)
        self.assertTrue(report.get("compaction_manifest_present"))


if __name__ == "__main__":
    unittest.main()
