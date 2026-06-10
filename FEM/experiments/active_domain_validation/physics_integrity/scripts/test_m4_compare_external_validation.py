#!/usr/bin/env python3
"""Tests for external validation-input package comparison preconditions."""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from test_m4_mesh_profile import (  # noqa: E402
    _TEST_GEOMETRY,
    _make_compare_ready_run,
    _write_barrier_pass,
)
from test_m4_reconciled_rom_postrun import (  # noqa: E402
    _materialize_shared_validation,
    _stamp_completed_terminal,
    _synthetic_modes,
    _write_reconcile_report,
    _write_reconciled_false_success_barrier,
)
from v2_b3_m4_mesh_profile_compare_lib import compare_exit_code, compare_runs  # noqa: E402
from v2_b3_m4_mesh_profile_lib import (  # noqa: E402
    MESH_PROFILE_REFERENCE,
    MESH_PROFILE_ROM,
    VALIDATION_INPUT_PACKAGE_REL,
    load_target_plan_file,
)
from v2_b3_m4_minimal_rom_compaction import compact_minimal_rom_durable_run  # noqa: E402
from v2_b3_petsc_util import write_json_atomic  # noqa: E402

TARGETS_HZ = [100.0, 200.0, 300.0, 400.0, 500.0]


def _write_external_validation_package(
    repo_root: Path,
    *,
    sample_id: str,
    targets_hz: List[float],
    manifest_sha_override: Optional[str] = None,
    plan_sample_id: Optional[str] = None,
) -> Tuple[Path, str]:
    pkg = (
        repo_root
        / "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/validation_inputs"
        / f"sample_{sample_id}_external_test"
    )
    pkg.mkdir(parents=True, exist_ok=True)
    plan_body: Dict[str, Any] = {
        "sample_id": plan_sample_id or sample_id,
        "targets_hz": targets_hz,
        "frequency_range_hz": [60.0, 550.0],
        "coverage_check": {"pass": True},
    }
    write_json_atomic(pkg / "target_plan.json", plan_body)
    _, plan_sha = load_target_plan_file(pkg / "target_plan.json")
    write_json_atomic(
        pkg / "validation_input_manifest.json",
        {
            "schema": "m4_mesh_validation_input_package_v1",
            "sample_id": sample_id,
            "inputs": [
                {
                    "name": "target_plan",
                    "sha256": manifest_sha_override or plan_sha,
                    "sample_id": sample_id,
                    "targets_hz": targets_hz,
                    "geometry_fingerprint": None,
                }
            ],
        },
    )
    return pkg, plan_sha


def _build_historical_pair(
    repo_root: Path,
    *,
    reconciled_candidate: bool = True,
    ref_lprod_plan: bool = False,
    cand_validation_package: bool = False,
    external_targets: Optional[List[float]] = None,
) -> Tuple[Path, Path, Path]:
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
    modes = _synthetic_modes()
    _make_compare_ready_run(
        ref_root,
        repo_root=repo_root,
        sample_id="sample_002",
        run_id="sample_002_m4prod2_strict_clean5",
        mesh_profile=MESH_PROFILE_REFERENCE,
        modes=modes,
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
        modes=modes,
        worker_wall_s=600.0,
        peak_rss_bytes=int(4 * 1024**3),
        geometry=dict(_TEST_GEOMETRY),
    )
    _stamp_completed_terminal(ref_root)
    _stamp_completed_terminal(cand_root)
    _write_barrier_pass(
        repo_root, ref_root, sample_id="sample_002", run_id="sample_002_m4prod2_strict_clean5",
    )
    if reconciled_candidate:
        _write_reconciled_false_success_barrier(
            repo_root, cand_root, sample_id="sample_002", run_id="sample_002_rom_prod_004",
        )
        _write_reconcile_report(repo_root, sample_id="sample_002", run_id="sample_002_rom_prod_004")
    else:
        _write_barrier_pass(
            repo_root, cand_root, sample_id="sample_002", run_id="sample_002_rom_prod_004",
        )

    targets = external_targets if external_targets is not None else TARGETS_HZ
    ext_pkg, plan_sha = _write_external_validation_package(
        repo_root, sample_id="sample_002", targets_hz=targets,
    )

    if ref_lprod_plan:
        write_json_atomic(
            ref_root / "lprod" / "lprod_target_plan.json",
            {
                "sample_id": "sample_002",
                "targets_hz": targets,
                "frequency_range_hz": [60.0, 550.0],
                "coverage_check": {"pass": True},
            },
        )
    else:
        (ref_root / "lprod" / "lprod_target_plan.json").unlink(missing_ok=True)

    if cand_validation_package:
        _materialize_shared_validation(
            repo_root,
            ref_root=ref_root,
            cand_root=cand_root,
            sample_id="sample_002",
        )
    else:
        shutil.rmtree(cand_root / VALIDATION_INPUT_PACKAGE_REL, ignore_errors=True)
        shutil.rmtree(ref_root / VALIDATION_INPUT_PACKAGE_REL, ignore_errors=True)

    return ref_root, cand_root, ext_pkg


class ExternalValidationCompareTests(unittest.TestCase):
    def test_external_package_accepted_for_reconciled_historical_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            ref_root, cand_root, ext_pkg = _build_historical_pair(
                repo_root, ref_lprod_plan=True, cand_validation_package=False,
            )
            report = compare_runs(
                reference_run=ref_root,
                candidate_run=cand_root,
                repo_root=repo_root,
                validation_input_package=ext_pkg,
            )
            self.assertTrue(report.get("comparison_executed"), report.get("precondition_errors"))
            val_meta = (report.get("cleanup_barrier") or {}).get("validation_input") or {}
            self.assertEqual(val_meta.get("validation_input_contract"), "external_authoritative")

    def test_external_manifest_sha_mismatch_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            ref_root, cand_root, ext_pkg = _build_historical_pair(repo_root, ref_lprod_plan=True)
            ext_pkg, _ = _write_external_validation_package(
                repo_root,
                sample_id="sample_002",
                targets_hz=TARGETS_HZ,
                manifest_sha_override="0" * 64,
            )
            report = compare_runs(
                reference_run=ref_root,
                candidate_run=cand_root,
                repo_root=repo_root,
                validation_input_package=ext_pkg,
            )
            self.assertEqual(report.get("status"), "PRECONDITION_FAILED")
            self.assertTrue(
                any("sha256" in str(e).lower() for e in (report.get("precondition_errors") or []))
            )

    def test_external_sample_identity_mismatch_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            ref_root, cand_root, ext_pkg = _build_historical_pair(repo_root, ref_lprod_plan=True)
            ext_pkg, _ = _write_external_validation_package(
                repo_root,
                sample_id="sample_002",
                targets_hz=TARGETS_HZ,
                plan_sample_id="sample_999",
            )
            report = compare_runs(
                reference_run=ref_root,
                candidate_run=cand_root,
                repo_root=repo_root,
                validation_input_package=ext_pkg,
            )
            self.assertEqual(report.get("status"), "PRECONDITION_FAILED")
            self.assertTrue(
                any("sample_id" in str(e) for e in (report.get("precondition_errors") or []))
            )

    def test_legacy_reference_lprod_identical_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            ref_root, cand_root, ext_pkg = _build_historical_pair(
                repo_root, ref_lprod_plan=True, cand_validation_package=False,
            )
            report = compare_runs(
                reference_run=ref_root,
                candidate_run=cand_root,
                repo_root=repo_root,
                validation_input_package=ext_pkg,
            )
            legacy = (report.get("cleanup_barrier") or {}).get("reference_legacy_lprod_target_plan") or {}
            self.assertTrue(legacy.get("legacy_lprod_target_plan_present"))
            self.assertEqual(legacy.get("classification"), "legacy_durable_provenance")
            self.assertNotIn("PRECONDITION_FAILED", [report.get("status")])

    def test_legacy_reference_lprod_differs_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            ref_root, cand_root, ext_pkg = _build_historical_pair(
                repo_root,
                ref_lprod_plan=True,
                external_targets=TARGETS_HZ,
            )
            write_json_atomic(
                ref_root / "lprod" / "lprod_target_plan.json",
                {
                    "sample_id": "sample_002",
                    "targets_hz": [111.0, 222.0],
                    "frequency_range_hz": [60.0, 550.0],
                },
            )
            report = compare_runs(
                reference_run=ref_root,
                candidate_run=cand_root,
                repo_root=repo_root,
                validation_input_package=ext_pkg,
            )
            self.assertEqual(report.get("status"), "PRECONDITION_FAILED")
            self.assertTrue(
                any("legacy_lprod_target_plan" in str(e) for e in (report.get("precondition_errors") or []))
            )

    def test_missing_internal_and_external_package_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            ref_root, cand_root, _ext_pkg = _build_historical_pair(
                repo_root, ref_lprod_plan=False, cand_validation_package=False,
            )
            report = compare_runs(
                reference_run=ref_root,
                candidate_run=cand_root,
                repo_root=repo_root,
            )
            self.assertEqual(report.get("status"), "PRECONDITION_FAILED")
            self.assertEqual(compare_exit_code(report), 2)

    def test_comparison_works_after_minimal_rom_compaction_with_external_package(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            ref_root, cand_root, ext_pkg = _build_historical_pair(
                repo_root, ref_lprod_plan=True, cand_validation_package=False,
            )
            shutil.rmtree(cand_root / "compaction", ignore_errors=True)
            compact_minimal_rom_durable_run(
                repo_root=repo_root,
                run_root=cand_root,
                sample_id="sample_002",
                run_id="sample_002_rom_prod_004",
            )
            self.assertFalse((cand_root / VALIDATION_INPUT_PACKAGE_REL).exists())
            report = compare_runs(
                reference_run=ref_root,
                candidate_run=cand_root,
                repo_root=repo_root,
                validation_input_package=ext_pkg,
            )
            self.assertTrue(report.get("comparison_executed"), report.get("precondition_errors"))


if __name__ == "__main__":
    unittest.main()
