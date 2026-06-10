#!/usr/bin/env python3
"""Tests for normal batch execute path: compaction before cleanup (no FEM)."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compact_completed_m4_runs import AGG_PASS, LHS_COMPLETED  # noqa: E402
from test_m4_compaction_run_selection import _make_strict_completed_run, _write_json  # noqa: E402
from test_m4_rom_shadow_pipeline import (  # noqa: E402
    _guitars_root,
    _seed_five_official_runs,
    _strip_post_cleanup_gates,
)
from v2_b3_m4_lhs_production_batch import (  # noqa: E402
    _assert_compaction_ready_before_cleanup,
    _run_sample_post_export_finalization,
    run_production_batch,
)
from v2_b3_m4_mesh_profile_lib import DATASET_VERSION_ROM, LEVEL_ROM_PROD, MESH_PROFILE_ROM  # noqa: E402
from v2_b3_m4_official_rom_dataset_lib import OFFICIAL_INITIAL_RUN_IDS  # noqa: E402
from v2_b3_m4_rom_shadow_pipeline_lib import (  # noqa: E402
    build_official_rom_surrogate_from_runs,
    mark_fom_pipeline_started,
    run_shadow_rom_compare_nonblocking,
    run_shadow_rom_prepredict_nonblocking,
)
from v2_b3_m4_sample_cleanup_barrier import run_sample_cleanup_barrier  # noqa: E402


def _make_shadow_batch_run(
    repo: Path,
    *,
    sample_id: str,
    run_id: str,
    lhs_row_index: int,
) -> Path:
    run_root = _guitars_root(repo) / sample_id / "runs" / run_id
    _make_strict_completed_run(run_root, sample_id=sample_id, run_id=run_id)
    _write_json(
        run_root / "sample" / "sample_input.json",
        {
            "sample_id": sample_id,
            "run_id": run_id,
            "shape_name": "classic",
            "lhs_row_index": lhs_row_index,
            "mesh_profile": MESH_PROFILE_ROM,
            "mesh_level_id": LEVEL_ROM_PROD,
            "dataset_version": DATASET_VERSION_ROM,
            "parameters": {
                "geometry.length": 0.5,
                "geometry.width": 0.33,
                "geometry.depth": 0.1,
                "geometry.top_thickness": 0.003,
                "geometry.hole_radius": 0.046,
                "geometry.back_thickness": 0.0033,
                "top_wood_id": "maple",
                "back_wood_id": "cedar",
            },
        },
    )
    _write_json(
        run_root / "aggregation" / "runtime_summary.json",
        {"status": "ok"},
    )
    (run_root / "scout" / "discovery").mkdir(parents=True, exist_ok=True)
    (run_root / "scout" / "discovery" / "scout_summary.json").write_text("{}", encoding="utf-8")
    (run_root / "lprod" / "mesh").mkdir(parents=True, exist_ok=True)
    return run_root


def _write_lhs_pool(repo: Path, *, sample_id: str, run_id: str, lhs_row_index: int) -> Path:
    lhs_path = repo / "ROM" / "classic" / "lhs_pool.json"
    lhs_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(
        lhs_path,
        {
            "shape_name": "classic",
            "dataset_version": DATASET_VERSION_ROM,
            "entries": [
                {
                    "id": sample_id,
                    "status": LHS_COMPLETED,
                    "last_run_id": run_id,
                    "last_aggregation_status": AGG_PASS,
                    "lhs_row_index": lhs_row_index,
                    "parameters": {
                        "geometry.length": 0.5,
                        "geometry.width": 0.33,
                        "geometry.depth": 0.1,
                        "geometry.top_thickness": 0.003,
                        "geometry.hole_radius": 0.046,
                        "geometry.back_thickness": 0.0033,
                        "top_wood_id": "maple",
                        "back_wood_id": "cedar",
                    },
                }
            ],
        },
    )
    return lhs_path


def _batch_sample_entry(
    *,
    sample_id: str,
    run_id: str,
    lhs_row_index: int,
    run_root: Path,
) -> dict:
    sample_input = json.loads((run_root / "sample" / "sample_input.json").read_text(encoding="utf-8"))
    return {
        "sample_id": sample_id,
        "run_id": run_id,
        "lhs_row_index": lhs_row_index,
        "sample_input": sample_input,
    }


def _batch_spec(
    *,
    batch_id: str,
    sample_id: str,
    run_id: str,
    lhs_row_index: int,
    run_root: Path,
) -> dict:
    return {
        "batch_id": batch_id,
        "mesh_profile": MESH_PROFILE_ROM,
        "mesh_level_id": LEVEL_ROM_PROD,
        "dataset_version": DATASET_VERSION_ROM,
        "frequency_policy": {"band_hz": [60.0, 550.0]},
        "samples": [_batch_sample_entry(
            sample_id=sample_id,
            run_id=run_id,
            lhs_row_index=lhs_row_index,
            run_root=run_root,
        )],
    }


def _production_row(
    *,
    run_root: Path,
    sample_id: str,
    run_id: str,
    lhs_row_index: int,
) -> dict:
    return {
        "sample_id": sample_id,
        "run_id": run_id,
        "run_root_abs": str(run_root.resolve()),
        "outcome": "pass",
        "return_code": 0,
        "aggregation_status": AGG_PASS,
        "final_aggregation_ready": True,
        "terminal_status": "COMPLETED",
        "production_acceptance_pass": True,
        "shared_export": {"export_status": "EXPORTED"},
        "rom_shadow_compare": {"status": "COMPLETED", "matched_mode_count": 593},
        "lhs_row_index": lhs_row_index,
    }


def _acceptance_pass(**kwargs):  # type: ignore[no-untyped-def]
    return {"acceptance_pass": True, "production_acceptance_pass": True, "failures": []}


def _shadow_compare_ok(**kwargs):  # type: ignore[no-untyped-def]
    return {"status": "COMPLETED", "matched_mode_count": 593, "blocking": False}


class ShadowBatchCompactionOrderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        _seed_five_official_runs(self.repo)
        build_official_rom_surrogate_from_runs(
            repo_root=self.repo,
            shape_name="classic",
            allowed_run_ids=list(OFFICIAL_INITIAL_RUN_IDS),
            min_mode_count=1,
        )
        self.sample_id = "sample_006"
        self.run_id = "sample_006_rom_shadow_v1"
        self.lhs_row_index = 6
        self.run_root = _make_shadow_batch_run(
            self.repo,
            sample_id=self.sample_id,
            run_id=self.run_id,
            lhs_row_index=self.lhs_row_index,
        )
        self.pool = json.loads(
            _write_lhs_pool(
                self.repo,
                sample_id=self.sample_id,
                run_id=self.run_id,
                lhs_row_index=self.lhs_row_index,
            ).read_text(encoding="utf-8")
        )
        self.context = {
            "sample_id": self.sample_id,
            "run_id": self.run_id,
            "shape_name": "classic",
            "lhs_row_index": self.lhs_row_index,
            "parameters": json.loads((self.run_root / "sample/sample_input.json").read_text())[
                "parameters"
            ],
        }
        run_shadow_rom_prepredict_nonblocking(
            repo_root=self.repo, run_root=self.run_root, context=self.context
        )
        mark_fom_pipeline_started(self.run_root)
        run_shadow_rom_compare_nonblocking(
            repo_root=self.repo, run_root=self.run_root, context=self.context
        )
        _strip_post_cleanup_gates(self.run_root)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_compaction_before_cleanup_order(self) -> None:
        calls: list[str] = []
        row = _production_row(
            run_root=self.run_root,
            sample_id=self.sample_id,
            run_id=self.run_id,
            lhs_row_index=self.lhs_row_index,
        )

        def _track_compaction(**kwargs):  # type: ignore[no-untyped-def]
            calls.append("compaction")
            from compact_completed_m4_runs import compact_one_completed_run

            return compact_one_completed_run(**kwargs)

        def _track_cleanup(**kwargs):  # type: ignore[no-untyped-def]
            calls.append("cleanup")
            return run_sample_cleanup_barrier(**kwargs)

        with unittest.mock.patch(
            "v2_b3_m4_lhs_production_batch.compact_one_completed_run",
            side_effect=_track_compaction,
        ), unittest.mock.patch(
            "v2_b3_m4_lhs_production_batch.run_sample_cleanup_barrier",
            side_effect=_track_cleanup,
        ):
            ok = _run_sample_post_export_finalization(
                row=row,
                repo_root=self.repo,
                pool=self.pool,
                compact_after_sample=True,
                compact_keep_full_samples=set(),
                compact_nonblocking=False,
                run_rom_compare=False,
                use_shadow_rom=True,
                strict_production=True,
            )
        self.assertTrue(ok)
        self.assertEqual(calls, ["compaction", "cleanup"])

    def test_cleanup_barrier_sees_compaction_manifest(self) -> None:
        row = _production_row(
            run_root=self.run_root,
            sample_id=self.sample_id,
            run_id=self.run_id,
            lhs_row_index=self.lhs_row_index,
        )
        ok = _run_sample_post_export_finalization(
            row=row,
            repo_root=self.repo,
            pool=self.pool,
            compact_after_sample=True,
            compact_keep_full_samples=set(),
            compact_nonblocking=False,
            run_rom_compare=False,
            use_shadow_rom=True,
            strict_production=True,
        )
        self.assertTrue(ok)
        self.assertTrue((self.run_root / "compaction" / "compaction_manifest.json").is_file())
        self.assertEqual(row["cleanup_barrier"]["status"], "completed")

    def test_cleanup_blocked_when_compaction_skipped(self) -> None:
        row = _production_row(
            run_root=self.run_root,
            sample_id=self.sample_id,
            run_id=self.run_id,
            lhs_row_index=self.lhs_row_index,
        )
        skipped = unittest.mock.Mock()
        skipped.to_dict.return_value = {
            "status": "skipped",
            "skip_reason": "production_gate:rom_compare_not_recorded",
            "deleted_bytes": 0,
        }
        skipped.status = "skipped"
        skipped.deleted_bytes = 0
        skipped.runtime_s = 0.01

        with unittest.mock.patch(
            "v2_b3_m4_lhs_production_batch.compact_one_completed_run",
            return_value=skipped,
        ):
            ok = _run_sample_post_export_finalization(
                row=row,
                repo_root=self.repo,
                pool=self.pool,
                compact_after_sample=True,
                compact_keep_full_samples=set(),
                compact_nonblocking=False,
                run_rom_compare=False,
                use_shadow_rom=True,
                strict_production=True,
            )
        self.assertFalse(ok)
        self.assertNotIn("cleanup_barrier", row)

    def test_registration_only_after_cleanup_in_batch(self) -> None:
        reg_calls: list[str] = []

        def _track_register(**kwargs):  # type: ignore[no-untyped-def]
            reg_calls.append("register")
            from v2_b3_m4_rom_shadow_pipeline_lib import attempt_register_and_retrain_after_cleanup

            return attempt_register_and_retrain_after_cleanup(**kwargs)

        spec_path = self.repo / "batch_spec.json"
        batch_id = "lhs_rom_shadow_v1_test"
        spec = _batch_spec(
            batch_id=batch_id,
            sample_id=self.sample_id,
            run_id=self.run_id,
            lhs_row_index=self.lhs_row_index,
            run_root=self.run_root,
        )
        _write_json(spec_path, spec)
        shared = self.repo / "shared"
        shared.mkdir(parents=True, exist_ok=True)

        with unittest.mock.patch(
            "v2_b3_m4_lhs_production_batch.run_pipeline",
            return_value=0,
        ) as mock_pipeline, unittest.mock.patch(
            "v2_b3_m4_lhs_production_batch.attempt_register_and_retrain_after_cleanup",
            side_effect=_track_register,
        ), unittest.mock.patch(
            "v2_b3_m4_lhs_production_batch._ensure_run_tree",
        ), unittest.mock.patch(
            "v2_b3_m4_production_contracts.evaluate_production_acceptance",
            side_effect=_acceptance_pass,
        ), unittest.mock.patch(
            "v2_b3_m4_lhs_production_batch.run_shadow_rom_compare_nonblocking",
            side_effect=_shadow_compare_ok,
        ):
            summary = run_production_batch(
                repo_root=self.repo,
                spec_path=spec_path,
                batch_id=batch_id,
                samples=spec["samples"],
                spec=spec,
                workers=3,
                execute=True,
                continue_on_fail=False,
                force=True,
                stop_after=None,
                resume=False,
                production_mode=True,
                shared_root=shared,
                run_rom_shadow=True,
                rom_nonblocking=True,
                pool=self.pool,
                compact_after_sample=True,
                compact_nonblocking=False,
                strict_production=True,
                mesh_profile=MESH_PROFILE_ROM,
                dataset_version=DATASET_VERSION_ROM,
            )
        mock_pipeline.assert_called_once()
        self.assertEqual(reg_calls, ["register"])
        self.assertEqual(summary["compaction_sample_count"], 1)
        self.assertEqual(summary["compaction_status"], "completed")
        self.assertEqual(summary["completed_count"], 1)

    def test_batch_summary_honest_when_compaction_count_zero(self) -> None:
        row = _production_row(
            run_root=self.run_root,
            sample_id=self.sample_id,
            run_id=self.run_id,
            lhs_row_index=self.lhs_row_index,
        )
        skipped = unittest.mock.Mock()
        skipped.to_dict.return_value = {"status": "skipped", "skip_reason": "test", "deleted_bytes": 0}
        skipped.status = "skipped"
        skipped.deleted_bytes = 0
        skipped.runtime_s = 0.0

        spec_path = self.repo / "batch_spec_fail.json"
        batch_id = "lhs_rom_shadow_v1_fail"
        spec = _batch_spec(
            batch_id=batch_id,
            sample_id=self.sample_id,
            run_id=self.run_id,
            lhs_row_index=self.lhs_row_index,
            run_root=self.run_root,
        )
        _write_json(spec_path, spec)

        with unittest.mock.patch(
            "v2_b3_m4_lhs_production_batch.run_pipeline",
            return_value=0,
        ), unittest.mock.patch(
            "v2_b3_m4_lhs_production_batch.compact_one_completed_run",
            return_value=skipped,
        ), unittest.mock.patch(
            "v2_b3_m4_lhs_production_batch._ensure_run_tree",
        ), unittest.mock.patch(
            "v2_b3_m4_production_contracts.evaluate_production_acceptance",
            side_effect=_acceptance_pass,
        ), unittest.mock.patch(
            "v2_b3_m4_lhs_production_batch.run_shadow_rom_compare_nonblocking",
            side_effect=_shadow_compare_ok,
        ):
            summary = run_production_batch(
                repo_root=self.repo,
                spec_path=spec_path,
                batch_id=batch_id,
                samples=spec["samples"],
                spec=spec,
                workers=3,
                execute=True,
                continue_on_fail=False,
                force=True,
                stop_after=None,
                resume=False,
                production_mode=True,
                pool=self.pool,
                compact_after_sample=True,
                compact_nonblocking=False,
                strict_production=True,
                run_rom_shadow=True,
            )
        self.assertEqual(summary["compaction_sample_count"], 0)
        self.assertNotEqual(summary["compaction_status"], "completed")
        self.assertEqual(summary["failed_count"], 1)

    def test_sample_006_simulation_passes_without_recovery(self) -> None:
        spec_path = self.repo / "batch_spec_006.json"
        batch_id = "lhs_rom_shadow_v1_20260610"
        spec = _batch_spec(
            batch_id=batch_id,
            sample_id=self.sample_id,
            run_id=self.run_id,
            lhs_row_index=self.lhs_row_index,
            run_root=self.run_root,
        )
        _write_json(spec_path, spec)
        shared = self.repo / "shared"
        shared.mkdir(parents=True, exist_ok=True)

        with unittest.mock.patch(
            "v2_b3_m4_lhs_production_batch.run_pipeline",
            return_value=0,
        ) as mock_pipeline, unittest.mock.patch(
            "v2_b3_m4_lhs_production_batch._ensure_run_tree",
        ), unittest.mock.patch(
            "v2_b3_m4_production_contracts.evaluate_production_acceptance",
            side_effect=_acceptance_pass,
        ), unittest.mock.patch(
            "v2_b3_m4_lhs_production_batch.run_shadow_rom_compare_nonblocking",
            side_effect=_shadow_compare_ok,
        ):
            summary = run_production_batch(
                repo_root=self.repo,
                spec_path=spec_path,
                batch_id=batch_id,
                samples=spec["samples"],
                spec=spec,
                workers=3,
                execute=True,
                continue_on_fail=False,
                force=True,
                stop_after=None,
                resume=False,
                production_mode=True,
                shared_root=shared,
                run_rom_shadow=True,
                rom_nonblocking=True,
                pool=self.pool,
                compact_after_sample=True,
                compact_nonblocking=False,
                strict_production=True,
                mesh_profile=MESH_PROFILE_ROM,
                dataset_version=DATASET_VERSION_ROM,
            )
        mock_pipeline.assert_called_once()
        self.assertEqual(summary["completed_count"], 1)
        self.assertEqual(summary["failed_count"], 0)
        self.assertEqual(summary["compaction_sample_count"], 1)
        self.assertEqual(summary["compaction_status"], "completed")
        self.assertTrue((self.run_root / "compaction" / "compaction_manifest.json").is_file())
        self.assertTrue((self.run_root / "cleanup" / "sample_cleanup_barrier.json").is_file())
        completed = summary["completed"][0]
        self.assertTrue(completed.get("rom_dataset_registration", {}).get("registered"))

    def test_pre_cleanup_gate_requires_manifest_and_no_heavy_artifacts(self) -> None:
        ready, errors = _assert_compaction_ready_before_cleanup(
            run_root=self.run_root,
            compact_after_sample=True,
            compact_blocking=True,
        )
        self.assertFalse(ready)
        self.assertTrue(any("compaction_manifest" in e for e in errors))

    def test_no_fem_subprocess_in_batch_tests(self) -> None:
        """Guard: batch tests must mock run_pipeline (no solver launch)."""
        with unittest.mock.patch(
            "v2_b3_m4_lhs_production_batch.run_pipeline",
            return_value=0,
        ) as mock_pipeline, unittest.mock.patch(
            "v2_b3_m4_lhs_production_batch._run_pipeline_isolated_subprocess",
            side_effect=AssertionError("FEM subprocess must not run in tests"),
        ), unittest.mock.patch(
            "v2_b3_m4_lhs_production_batch._ensure_run_tree",
        ), unittest.mock.patch(
            "v2_b3_m4_production_contracts.evaluate_production_acceptance",
            side_effect=_acceptance_pass,
        ), unittest.mock.patch(
            "v2_b3_m4_lhs_production_batch.run_shadow_rom_compare_nonblocking",
            side_effect=_shadow_compare_ok,
        ):
            spec = _batch_spec(
                batch_id="guard_batch",
                sample_id=self.sample_id,
                run_id=self.run_id,
                lhs_row_index=self.lhs_row_index,
                run_root=self.run_root,
            )
            run_production_batch(
                repo_root=self.repo,
                spec_path=self.repo / "guard_spec.json",
                batch_id="guard_batch",
                samples=spec["samples"],
                spec=spec,
                workers=3,
                execute=True,
                continue_on_fail=False,
                force=True,
                stop_after=None,
                resume=False,
                production_mode=True,
                pool=self.pool,
                compact_after_sample=True,
                compact_nonblocking=False,
                strict_production=True,
                run_rom_shadow=True,
            )
        mock_pipeline.assert_called_once()


if __name__ == "__main__":
    unittest.main()
