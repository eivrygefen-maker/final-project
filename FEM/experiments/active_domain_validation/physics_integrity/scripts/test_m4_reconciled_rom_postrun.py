#!/usr/bin/env python3
"""Tests for reconciled-run compare preconditions and minimal ROM compaction."""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from test_m4_mesh_profile import (  # noqa: E402
    _TEST_GEOMETRY,
    _make_compare_ready_run,
    _write_barrier_pass,
    _write_modes_catalog,
)
from v2_b3_m4_lhs_pool_bridge import specs_generated_dir  # noqa: E402
from v2_b3_petsc_util import write_json_atomic  # noqa: E402
from v2_b3_m4_mesh_profile_compare_lib import (  # noqa: E402
    EXIT_PASS,
    EXIT_PRECONDITION_FAIL,
    RECOMMEND_ACCEPT_ROM_BALANCED,
    compare_exit_code,
    compare_runs,
    verify_reconciled_historical_compare_precondition,
    verify_run_compare_barrier_precondition,
)
from v2_b3_m4_minimal_rom_compaction import (  # noqa: E402
    MINIMAL_ROM_DELETE_REL_FILES,
    collect_minimal_rom_deletable_paths,
    compact_minimal_rom_durable_run,
    minimal_rom_retain_rel_paths,
    verify_minimal_rom_retention_sufficient,
)
from v2_b3_m4_mesh_profile_lib import (  # noqa: E402
    MESH_PROFILE_REFERENCE,
    MESH_PROFILE_ROM,
    materialize_validation_input_package,
)
from v2_b3_m4_sample_cleanup_barrier import BARRIER_MANIFEST_REL  # noqa: E402


def _synthetic_modes(offset_hz: float = 0.0, scale: float = 1.0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for hz in (80.0, 120.0, 180.0, 220.0, 280.0, 320.0, 380.0, 420.0, 480.0, 520.0):
        rows.append(
            {
                "frequency_hz": hz + offset_hz,
                "mode_id": f"mode_{int(hz)}",
                "coupling_class": "structural" if hz < 300 else "mixed",
                "dominant_region": "top" if hz < 250 else "back",
                "top_share": 0.5,
                "back_share": 0.3,
                "air_share": 0.2,
                "bridge_excitation_abs": 0.01 * scale,
                "bridge_excitation_coupling": 0.008 * scale,
                "mic_output_proxy": 0.02 * scale,
                "radiation_proxy": 0.015 * scale,
            }
        )
    return rows


def _write_reconcile_report(
    repo_root: Path,
    *,
    sample_id: str,
    run_id: str,
    outcome: str = "pass",
) -> Path:
    path = (
        specs_generated_dir(repo_root)
        / f"bookkeeping_reconcile_{sample_id}_{run_id}_20260602.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        path,
        {
            "schema": "m4_lhs_run_bookkeeping_reconcile_v1",
            "sample_id": sample_id,
            "run_id": run_id,
            "outcome": outcome,
            "generated_utc": "2026-06-02T00:00:00Z",
        },
    )
    return path


def _write_reconciled_false_success_barrier(
    repo_root: Path,
    run_root: Path,
    *,
    sample_id: str,
    run_id: str,
) -> None:
    from v2_b3_m4_sample_cleanup_barrier import verify_cleanup_barrier  # noqa: WPS433

    verify = verify_cleanup_barrier(
        repo_root=repo_root,
        run_root=run_root,
        sample_id=sample_id,
        run_id=run_id,
        sample_success=False,
    )
    write_json_atomic(
        run_root / BARRIER_MANIFEST_REL,
        {
            "schema": "m4_sample_cleanup_barrier_v1",
            "status": "completed",
            "sample_success": False,
            "verification": verify,
        },
    )
    write_json_atomic(
        run_root / "cleanup" / "sample_failure_retention.json",
        {"outcome": "fail", "error_message": "historical_misclassification"},
    )
    (run_root / "logs").mkdir(parents=True, exist_ok=True)
    (run_root / "logs" / "sample_failure_diagnostic.log").write_text("historical\n", encoding="utf-8")


def _stamp_completed_terminal(run_root: Path) -> None:
    manifest_path = run_root / "pipeline_run_manifest.json"
    doc = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    doc["terminal_status"] = "COMPLETED"
    write_json_atomic(manifest_path, doc)


def _add_compaction_debris(run_root: Path) -> None:
    for rel in (
        "lprod/worker_chunk_plan.preview.json",
        "lprod/lprod_execution_plan.preview.json",
        "aggregation/aggregation_plan.preview.json",
        "run_one_sample_plan.json",
        "pipeline_run_manifest.m4_4_partial_aggregation_preview.json",
        "README.md",
    ):
        path = run_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    (run_root / "lprod" / "mesh").mkdir(parents=True, exist_ok=True)
    (run_root / "lprod" / "mesh" / "placeholder.msh").write_text("mesh", encoding="utf-8")
    (run_root / "worker_results").mkdir(parents=True, exist_ok=True)
    (run_root / "worker_results" / "chunk_01.json").write_text("{}", encoding="utf-8")


def _make_minimal_rom_fixture(
    run_root: Path,
    *,
    repo_root: Path,
    sample_id: str,
    run_id: str,
    reconciled: bool,
    with_compaction_debris: bool = False,
) -> None:
    modes = _synthetic_modes()
    _make_compare_ready_run(
        run_root,
        repo_root=repo_root,
        sample_id=sample_id,
        run_id=run_id,
        mesh_profile=MESH_PROFILE_ROM,
        modes=modes,
        worker_wall_s=600.0,
        peak_rss_bytes=int(4 * 1024**3),
        geometry=dict(_TEST_GEOMETRY),
    )
    _stamp_completed_terminal(run_root)
    if reconciled:
        _write_reconciled_false_success_barrier(
            repo_root, run_root, sample_id=sample_id, run_id=run_id,
        )
        _write_reconcile_report(repo_root, sample_id=sample_id, run_id=run_id)
    else:
        _write_barrier_pass(repo_root, run_root, sample_id=sample_id, run_id=run_id)
    if with_compaction_debris:
        _add_compaction_debris(run_root)


def _materialize_shared_validation(
    repo_root: Path,
    *,
    ref_root: Path,
    cand_root: Path,
    sample_id: str,
) -> None:
    shared = repo_root / "shared_target_plan.json"
    write_json_atomic(
        shared,
        {
            "sample_id": sample_id,
            "targets_hz": [100.0, 200.0, 300.0, 400.0, 500.0],
            "frequency_range_hz": [60.0, 550.0],
            "coverage_check": {"pass": True},
        },
    )
    for run_root, run_id in (
        (ref_root, ref_root.name),
        (cand_root, cand_root.name),
    ):
        materialize_validation_input_package(
            run_root=run_root,
            source_path=shared,
            input_name="target_plan",
            sample_id=sample_id,
            run_id=run_id,
        )


class ReconciledComparePreconditionTests(unittest.TestCase):
    def test_reconciled_historical_run_accepted_when_evidence_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            run_root = (
                repo_root
                / "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars"
                / "sample_002/runs/sample_002_rom_prod_004"
            )
            _make_minimal_rom_fixture(
                run_root,
                repo_root=repo_root,
                sample_id="sample_002",
                run_id="sample_002_rom_prod_004",
                reconciled=True,
            )
            ok, meta, errors = verify_run_compare_barrier_precondition(
                repo_root=repo_root,
                run_root=run_root,
                label="candidate",
            )
            self.assertTrue(ok, errors)
            self.assertEqual(meta.get("precondition_contract"), "reconciled_historical")

    def test_cleanup_false_without_reconciliation_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            run_root = (
                repo_root
                / "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars"
                / "sample_002/runs/sample_002_rom_prod_004"
            )
            _make_minimal_rom_fixture(
                run_root,
                repo_root=repo_root,
                sample_id="sample_002",
                run_id="sample_002_rom_prod_004",
                reconciled=False,
            )
            _write_reconciled_false_success_barrier(
                repo_root, run_root, sample_id="sample_002", run_id="sample_002_rom_prod_004",
            )
            ok, _, errors = verify_run_compare_barrier_precondition(
                repo_root=repo_root,
                run_root=run_root,
                label="candidate",
            )
            self.assertFalse(ok)
            self.assertTrue(any("bookkeeping" in e for e in errors))

    def test_forbidden_heavy_artifact_fails_precondition(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            run_root = (
                repo_root
                / "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars"
                / "sample_002/runs/sample_002_rom_prod_004"
            )
            _make_minimal_rom_fixture(
                run_root,
                repo_root=repo_root,
                sample_id="sample_002",
                run_id="sample_002_rom_prod_004",
                reconciled=True,
            )
            (run_root / "lprod" / "checkpoint").mkdir(parents=True, exist_ok=True)
            (run_root / "lprod" / "checkpoint" / "forbidden.bin").write_bytes(b"x" * 10)
            ok, _, errors = verify_reconciled_historical_compare_precondition(
                repo_root=repo_root,
                run_root=run_root,
                label="candidate",
            )
            self.assertFalse(ok)
            self.assertTrue(any("forbidden_heavy" in e for e in errors))


class MinimalRomCompactionTests(unittest.TestCase):
    def test_minimal_compaction_retain_allowlist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            run_root = (
                repo_root
                / "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars"
                / "sample_002/runs/sample_002_rom_prod_004"
            )
            _make_minimal_rom_fixture(
                run_root,
                repo_root=repo_root,
                sample_id="sample_002",
                run_id="sample_002_rom_prod_004",
                reconciled=True,
            )
            retain = minimal_rom_retain_rel_paths(run_root)
            self.assertIn("aggregation/modes_catalog_deduped.jsonl", retain)
            self.assertIn("freeze/physics_identity_manifest.json", retain)
            self.assertIn("cleanup/sample_cleanup_barrier.json", retain)

    def test_minimal_compaction_removes_previews_and_failure_leftovers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            run_root = (
                repo_root
                / "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars"
                / "sample_002/runs/sample_002_rom_prod_004"
            )
            _make_minimal_rom_fixture(
                run_root,
                repo_root=repo_root,
                sample_id="sample_002",
                run_id="sample_002_rom_prod_004",
                reconciled=True,
                with_compaction_debris=True,
            )
            deletable = collect_minimal_rom_deletable_paths(run_root)
            deleted_rels = {
                str(p.relative_to(run_root)).replace("\\", "/")
                for p in deletable
                if p.exists() and p.is_relative_to(run_root)
            }
            for rel in MINIMAL_ROM_DELETE_REL_FILES:
                if (run_root / rel).exists():
                    self.assertTrue(
                        any(d.endswith(rel) or rel in d for d in deleted_rels),
                        f"expected deletable: {rel}",
                    )

            outcome = compact_minimal_rom_durable_run(
                repo_root=repo_root,
                run_root=run_root,
                sample_id="sample_002",
                run_id="sample_002_rom_prod_004",
            )
            self.assertEqual(outcome.status, "completed")
            self.assertFalse((run_root / "lprod/worker_chunk_plan.preview.json").exists())
            self.assertFalse((run_root / "cleanup/sample_failure_retention.json").exists())
            self.assertFalse((run_root / "logs/sample_failure_diagnostic.log").exists())
            self.assertTrue((run_root / "aggregation/modes_catalog_deduped.jsonl").is_file())
            ok, errs = verify_minimal_rom_retention_sufficient(run_root)
            self.assertTrue(ok, errs)


class MathematicalCompareTests(unittest.TestCase):
    def _pair_runs(
        self,
        tmp: str,
        *,
        cand_offset: float = 0.0,
        cand_scale: float = 1.0,
        reconciled: bool = True,
    ) -> tuple[Path, Path, Path]:
        repo_root = Path(tmp)
        ref_root = (
            repo_root
            / "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars"
            / "sample_002/runs/sample_002_m4prod2_strict_clean5"
        )
        cand_root = (
            repo_root
            / "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars"
            / "sample_002/runs/sample_002_rom_prod_004"
        )
        ref_modes = _synthetic_modes()
        cand_modes = _synthetic_modes(offset_hz=cand_offset, scale=cand_scale)
        _make_compare_ready_run(
            ref_root,
            repo_root=repo_root,
            sample_id="sample_002",
            run_id="sample_002_m4prod2_strict_clean5",
            mesh_profile=MESH_PROFILE_REFERENCE,
            modes=ref_modes,
            worker_wall_s=1200.0,
            peak_rss_bytes=int(6 * 1024**3),
            geometry=dict(_TEST_GEOMETRY),
        )
        _make_compare_ready_run(
            cand_root,
            repo_root=repo_root,
            sample_id="sample_002",
            run_id="sample_002_rom_prod_004",
            mesh_profile=MESH_PROFILE_ROM,
            modes=cand_modes,
            worker_wall_s=600.0,
            peak_rss_bytes=int(4 * 1024**3),
            geometry=dict(_TEST_GEOMETRY),
        )
        _stamp_completed_terminal(ref_root)
        _stamp_completed_terminal(cand_root)
        _materialize_shared_validation(
            repo_root,
            ref_root=ref_root,
            cand_root=cand_root,
            sample_id="sample_002",
        )
        _write_barrier_pass(
            repo_root, ref_root, sample_id="sample_002", run_id="sample_002_m4prod2_strict_clean5",
        )
        if reconciled:
            _write_reconciled_false_success_barrier(
                repo_root, cand_root, sample_id="sample_002", run_id="sample_002_rom_prod_004",
            )
            _write_reconcile_report(repo_root, sample_id="sample_002", run_id="sample_002_rom_prod_004")
        else:
            _write_barrier_pass(
                repo_root, cand_root, sample_id="sample_002", run_id="sample_002_rom_prod_004",
            )
        return repo_root, ref_root, cand_root

    def test_comparison_works_after_minimal_compaction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root, ref_root, cand_root = self._pair_runs(tmp)
            compact_minimal_rom_durable_run(
                repo_root=repo_root,
                run_root=cand_root,
                sample_id="sample_002",
                run_id="sample_002_rom_prod_004",
            )
            report = compare_runs(
                reference_run=ref_root,
                candidate_run=cand_root,
                repo_root=repo_root,
            )
            self.assertTrue(report.get("comparison_executed"))
            self.assertEqual(compare_exit_code(report), EXIT_PASS)

    def test_synthetic_exact_match_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root, ref_root, cand_root = self._pair_runs(tmp)
            report = compare_runs(
                reference_run=ref_root,
                candidate_run=cand_root,
                repo_root=repo_root,
            )
            self.assertTrue(report.get("acceptance_pass"))
            self.assertEqual(
                (report.get("recommendation") or {}).get("recommendation"),
                RECOMMEND_ACCEPT_ROM_BALANCED,
            )

    def test_synthetic_frequency_drift_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root, ref_root, cand_root = self._pair_runs(tmp, cand_offset=8.0)
            report = compare_runs(
                reference_run=ref_root,
                candidate_run=cand_root,
                repo_root=repo_root,
            )
            self.assertFalse(report.get("acceptance_pass"))

    def test_synthetic_proxy_scale_change_rank_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root, ref_root, cand_root = self._pair_runs(tmp, cand_scale=2.0)
            report = compare_runs(
                reference_run=ref_root,
                candidate_run=cand_root,
                repo_root=repo_root,
            )
            proxy = report.get("proxy_comparison") or {}
            self.assertTrue(proxy.get("normalization_scale_warning"))
            self.assertTrue(report.get("comparison_executed"))

    def test_synthetic_missing_family_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root, ref_root, cand_root = self._pair_runs(tmp)
            sparse = [
                {
                    "frequency_hz": 90.0,
                    "mode_id": "only_low",
                    "coupling_class": "air_dominant",
                    "dominant_region": "air",
                    "top_share": 0.1,
                    "back_share": 0.1,
                    "air_share": 0.8,
                    "bridge_excitation_abs": 0.001,
                    "mic_output_proxy": 0.001,
                    "radiation_proxy": 0.001,
                }
            ]
            _write_modes_catalog(cand_root, sparse)
            report = compare_runs(
                reference_run=ref_root,
                candidate_run=cand_root,
                repo_root=repo_root,
            )
            self.assertTrue(report.get("comparison_executed"))
            families = report.get("mode_family_survival") or {}
            self.assertFalse(families.get("family_survival_pass", True))


if __name__ == "__main__":
    unittest.main()
