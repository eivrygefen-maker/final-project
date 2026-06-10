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
from v2_b3_m4_mesh_profile_compare_lib import (  # noqa: E402
    compare_exit_code,
    compare_runs,
    verify_legacy_reference_lprod_target_plan,
)
from v2_b3_m4_mesh_profile_lib import (  # noqa: E402
    EXTERNAL_VALIDATION_INPUT_PACKAGE_SCHEMA_V1,
    MESH_PROFILE_REFERENCE,
    MESH_PROFILE_ROM,
    VALIDATION_INPUT_PACKAGE_REL,
    load_external_validation_package,
    load_target_plan_file,
)
from v2_b3_m4_mesh_profile_provenance_lib import (  # noqa: E402
    compare_physical_identity_projections,
    compare_target_plan_semantic,
    material_fingerprint,
    physics_identity_hash,
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


def _write_flat_v1_external_validation_package(
    repo_root: Path,
    *,
    sample_id: str,
    targets_hz: List[float],
    manifest_sha_override: Optional[str] = None,
    manifest_targets_override: Optional[List[float]] = None,
    write_plan: bool = True,
    ref_root: Optional[Path] = None,
) -> Tuple[Path, str]:
    pkg = (
        repo_root
        / "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/validation_inputs"
        / f"sample_{sample_id}_flat_v1_test"
    )
    pkg.mkdir(parents=True, exist_ok=True)
    plan_body: Dict[str, Any] = {
        "sample_id": sample_id,
        "targets_hz": targets_hz,
        "frequency_range_hz": [64.5, 547.0],
        "coverage_check": {"pass": True},
    }
    plan_sha = "0" * 64
    if write_plan:
        write_json_atomic(pkg / "target_plan.json", plan_body)
        _, plan_sha = load_target_plan_file(pkg / "target_plan.json")
    manifest_targets = manifest_targets_override if manifest_targets_override is not None else targets_hz
    manifest: Dict[str, Any] = {
        "schema": EXTERNAL_VALIDATION_INPUT_PACKAGE_SCHEMA_V1,
        "sample_id": sample_id,
        "target_plan_sha256": manifest_sha_override or plan_sha,
        "target_count": len(manifest_targets),
        "targets_hz": manifest_targets,
        "frequency_range_hz": [64.5, 547.0],
        "chunk_count": 13,
    }
    if ref_root is not None:
        sample_in_path = ref_root / "sample" / "sample_input.json"
        if sample_in_path.is_file():
            sample_in = json.loads(sample_in_path.read_text(encoding="utf-8"))
            from v2_b3_m4_lprod_interfaces import extract_geometry_dict, geometry_fingerprint  # noqa: WPS433

            geom = extract_geometry_dict(sample_in)
            if geom:
                manifest["geometry_fingerprint"] = geometry_fingerprint(geom)
            manifest["material_fingerprint"] = material_fingerprint(sample_in)
    write_json_atomic(pkg / "validation_input_manifest.json", manifest)
    return pkg, plan_sha


class ExternalValidationPackageLoaderTests(unittest.TestCase):
    def test_valid_flat_v1_package_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            pkg, plan_sha = _write_flat_v1_external_validation_package(
                repo_root, sample_id="sample_002", targets_hz=TARGETS_HZ,
            )
            loaded, errors = load_external_validation_package(pkg)
            self.assertFalse(errors, errors)
            assert loaded is not None
            self.assertEqual(loaded.target_plan_sha256, plan_sha)
            self.assertEqual(len(loaded.target_plan.get("targets_hz") or []), len(TARGETS_HZ))

    def test_valid_nested_schema_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            pkg, plan_sha = _write_external_validation_package(
                repo_root, sample_id="sample_002", targets_hz=TARGETS_HZ,
            )
            loaded, errors = load_external_validation_package(pkg)
            self.assertFalse(errors, errors)
            assert loaded is not None
            self.assertEqual(loaded.target_plan_sha256, plan_sha)

    def test_flat_v1_sha_mismatch_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            pkg, _ = _write_flat_v1_external_validation_package(
                repo_root,
                sample_id="sample_002",
                targets_hz=TARGETS_HZ,
                manifest_sha_override="f" * 64,
            )
            loaded, errors = load_external_validation_package(pkg)
            self.assertIsNone(loaded)
            self.assertIn("external_validation_input_sha256_mismatch", errors)

    def test_flat_v1_target_list_mismatch_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            pkg, _ = _write_flat_v1_external_validation_package(
                repo_root,
                sample_id="sample_002",
                targets_hz=TARGETS_HZ,
                manifest_targets_override=[111.0, 222.0],
            )
            loaded, errors = load_external_validation_package(pkg)
            self.assertIsNone(loaded)
            self.assertIn("external_validation_input_targets_hz_mismatch", errors)

    def test_missing_target_plan_json_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            pkg, _ = _write_flat_v1_external_validation_package(
                repo_root, sample_id="sample_002", targets_hz=TARGETS_HZ, write_plan=False,
            )
            loaded, errors = load_external_validation_package(pkg)
            self.assertIsNone(loaded)
            self.assertIn("missing_external_target_plan", errors)


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
    ext_pkg, plan_sha = _write_flat_v1_external_validation_package(
        repo_root,
        sample_id="sample_002",
        targets_hz=targets,
        ref_root=ref_root,
    )

    if ref_lprod_plan:
        (ref_root / "lprod").mkdir(parents=True, exist_ok=True)
        shutil.copy2(ext_pkg / "target_plan.json", ref_root / "lprod" / "lprod_target_plan.json")
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

    def test_flat_v1_package_advances_past_external_validation_preconditions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            ref_root, cand_root, ext_pkg = _build_historical_pair(
                repo_root, ref_lprod_plan=True, cand_validation_package=False,
            )
            loaded, load_errors = load_external_validation_package(ext_pkg)
            self.assertFalse(load_errors, load_errors)
            self.assertIsNotNone(loaded)
            self.assertEqual(
                str((loaded.manifest or {}).get("schema")),
                EXTERNAL_VALIDATION_INPUT_PACKAGE_SCHEMA_V1,
            )
            report = compare_runs(
                reference_run=ref_root,
                candidate_run=cand_root,
                repo_root=repo_root,
                validation_input_package=ext_pkg,
            )
            self.assertTrue(report.get("comparison_executed"), report.get("precondition_errors"))
            self.assertNotEqual(report.get("status"), "PRECONDITION_FAILED")

    def test_external_manifest_sha_mismatch_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            ref_root, cand_root, ext_pkg = _build_historical_pair(repo_root, ref_lprod_plan=True)
            ext_pkg, _ = _write_flat_v1_external_validation_package(
                repo_root,
                sample_id="sample_002",
                targets_hz=TARGETS_HZ,
                manifest_sha_override="0" * 64,
                ref_root=ref_root,
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
            ext_pkg, _ = _write_flat_v1_external_validation_package(
                repo_root,
                sample_id="sample_999",
                targets_hz=TARGETS_HZ,
                ref_root=ref_root,
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


class SemanticValidationIdentityTests(unittest.TestCase):
    def test_legacy_lprod_different_serialization_same_semantics_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            ext_pkg, _ = _write_flat_v1_external_validation_package(
                repo_root, sample_id="sample_002", targets_hz=TARGETS_HZ,
            )
            loaded, load_errors = load_external_validation_package(ext_pkg)
            self.assertFalse(load_errors)
            assert loaded is not None
            ref_root = (
                repo_root
                / "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars"
                / "sample_002/runs/sample_002_ref"
            )
            ref_root.mkdir(parents=True)
            lprod_plan = {
                "coverage_check": {"pass": True},
                "frequency_range_hz": [64.5, 547.0],
                "sample_id": "sample_002",
                "targets_hz": TARGETS_HZ,
            }
            lprod_path = ref_root / "lprod" / "lprod_target_plan.json"
            lprod_path.parent.mkdir(parents=True, exist_ok=True)
            lprod_path.write_text(json.dumps(lprod_plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            _, lprod_sha = load_target_plan_file(lprod_path)
            self.assertNotEqual(lprod_sha, loaded.target_plan_sha256)
            plan_cmp = compare_target_plan_semantic(
                lprod_plan,
                loaded.target_plan,
                left_sha256=lprod_sha,
                right_sha256=loaded.target_plan_sha256,
                left_manifest_chunk_count=13,
                right_manifest_chunk_count=13,
            )
            self.assertTrue(plan_cmp.get("semantic_match"))
            self.assertEqual(plan_cmp.get("target_plan_match_mode"), "semantic_exact")
            self.assertFalse(plan_cmp.get("raw_sha_match"))
            legacy_errors, legacy_meta = verify_legacy_reference_lprod_target_plan(
                ref_root,
                external=loaded,
                barrier_meta={"barrier_status": "completed", "forbidden_heavy_artifact_count": 0, "shared_sample_artifact_count": 0},
            )
            self.assertEqual(legacy_errors, [])
            self.assertEqual(legacy_meta.get("target_plan_match_mode"), "semantic_exact")
            self.assertFalse(legacy_meta.get("raw_sha_match"))

    def test_target_plan_order_or_content_mismatch_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            left = {"targets_hz": TARGETS_HZ, "frequency_range_hz": [64.5, 547.0]}
            right = {"targets_hz": list(reversed(TARGETS_HZ)), "frequency_range_hz": [64.5, 547.0]}
            plan_cmp = compare_target_plan_semantic(left, right)
            self.assertFalse(plan_cmp.get("semantic_match"))
            self.assertIn("targets_hz", plan_cmp.get("differences") or [])

    def test_rom_mesh_only_physics_hash_difference_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            ref_root, cand_root, ext_pkg = _build_historical_pair(
                repo_root, ref_lprod_plan=True, cand_validation_package=False,
            )
            self.assertNotEqual(physics_identity_hash(ref_root), physics_identity_hash(cand_root))
            report = compare_runs(
                reference_run=ref_root,
                candidate_run=cand_root,
                repo_root=repo_root,
                validation_input_package=ext_pkg,
            )
            self.assertTrue(report.get("comparison_executed"), report.get("precondition_errors"))
            phys = ((report.get("cleanup_barrier") or {}).get("validation_input") or {}).get("physical_identity") or {}
            self.assertTrue(phys.get("physical_identity_invariants_match"))
            self.assertTrue(phys.get("allowed_mesh_identity_differences"))
            self.assertEqual(phys.get("unexpected_identity_differences"), [])

    def test_geometry_material_solver_mismatch_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            ref_root, cand_root, _ext_pkg = _build_historical_pair(
                repo_root, ref_lprod_plan=False, cand_validation_package=False,
            )
            cand_in_path = cand_root / "sample" / "sample_input.json"
            cand_in = json.loads(cand_in_path.read_text(encoding="utf-8"))
            cand_in["top_wood_id"] = "cedar"
            write_json_atomic(cand_in_path, cand_in)
            write_json_atomic(
                cand_root / "lprod" / "resolved_core_config.json",
                {
                    "solver": {
                        "eps_eigenvalue_semantics": "slepc_backtransformed",
                        "rtol": 1e-4,
                    }
                },
            )
            identity_cmp = compare_physical_identity_projections(ref_root, cand_root)
            self.assertFalse(identity_cmp.get("physical_identity_invariants_match"))
            unexpected = identity_cmp.get("unexpected_identity_differences") or []
            self.assertTrue(
                any("material" in path or "solver_config" in path for path in unexpected),
                unexpected,
            )


if __name__ == "__main__":
    unittest.main()
