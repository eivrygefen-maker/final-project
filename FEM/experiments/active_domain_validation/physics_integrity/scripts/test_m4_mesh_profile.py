#!/usr/bin/env python3
"""Lightweight mesh profile contract tests (no FEM solve)."""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_m4_lhs_pool_bridge import build_lhs_batch_spec, build_sample_input, write_per_sample_spec  # noqa: E402
from test_m4_compaction_run_selection import _make_strict_completed_run  # noqa: E402
from v2_b3_m4_mesh_profile_compare import compare_runs, scan_candidate_references_other_run  # noqa: E402
from v2_b3_m4_mesh_profile_compare_lib import (  # noqa: E402
    EXIT_ACCEPTANCE_FAIL,
    EXIT_INCOMPLETE,
    EXIT_PASS,
    EXIT_PRECONDITION_FAIL,
    compare_exit_code,
    evaluate_acceptance,
)
from v2_b3_m4_mesh_profile_provenance_lib import (  # noqa: E402
    compare_intrinsic_band_third_coverage,
    compare_mode_family_survival,
    derive_band_third_counts_from_catalog,
    major_mode_families,
)
from v2_b3_m4_validation_readiness_audit import audit_legacy_reference, audit_readiness  # noqa: E402
from v2_b3_m4_production_contracts import evaluate_post_cleanup_region_dof_evidence  # noqa: E402
from v2_b3_m4_mesh_profile_lib import (  # noqa: E402
    DATASET_VERSION_LEGACY,
    DATASET_VERSION_REFERENCE,
    DATASET_VERSION_ROM,
    LEVEL_PROD_REFERENCE,
    LEVEL_ROM_PROD,
    MESH_PROFILE_REFERENCE,
    MESH_PROFILE_ROM,
    MeshProfileError,
    REFERENCE_CONTROLS_M,
    VALIDATION_INPUT_PACKAGE_REL,
    apply_mesh_profile_to_sample_input,
    assert_l_prod_alias_controls_match,
    evaluate_legacy_reference_compatibility,
    install_explicit_target_plan,
    load_durable_target_plan,
    materialize_validation_input_package,
    preserve_target_plan_before_cleanup,
    production_mesh_levels_for_cleanup,
    resolve_mesh_profile,
    validate_mesh_profile_reuse,
    validate_profile_dataset_pairing,
)
from unittest.mock import patch  # noqa: E402

import v2_b3_m4_pipeline_dry_run as pipeline_dry_run  # noqa: E402
from v2_b3_m4_pipeline_dry_run import build_dry_run_plan, _write_tree  # noqa: E402
from v2_b3_m4_sample_cleanup_barrier import (  # noqa: E402
    BARRIER_MANIFEST_REL,
    MESH_LEVELS,
    require_cleanup_barrier_passed_for_validation,
    verify_cleanup_barrier,
)
from v2_b3_petsc_util import write_json_atomic  # noqa: E402
from v2_mesh_convergence_common import load_manifest, mesh_path  # noqa: E402


def _write_modes_catalog(run_root: Path, rows: List[Dict[str, Any]]) -> None:
    path = run_root / "aggregation" / "modes_catalog_deduped.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _write_barrier_pass(repo_root: Path, run_root: Path, *, sample_id: str, run_id: str) -> None:
    verify = verify_cleanup_barrier(
        repo_root=repo_root,
        run_root=run_root,
        sample_id=sample_id,
        run_id=run_id,
        sample_success=True,
    )
    write_json_atomic(
        run_root / BARRIER_MANIFEST_REL,
        {
            "schema": "m4_sample_cleanup_barrier_v1",
            "status": "completed",
            "sample_success": True,
            "verification": verify,
        },
    )


def _write_validation_package(
    run_root: Path,
    *,
    sample_id: str,
    run_id: str,
    targets_hz: List[float],
    source_path: Optional[Path] = None,
) -> None:
    src = source_path or (run_root / "external_plan.json")
    if not src.is_file():
        write_json_atomic(
            src,
            {
                "sample_id": sample_id,
                "run_id": run_id,
                "targets_hz": targets_hz,
                "frequency_range_hz": [60.0, 550.0],
                "coverage_check": {"pass": True},
            },
        )
    materialize_validation_input_package(
        run_root=run_root,
        source_path=src,
        input_name="target_plan",
        sample_id=sample_id,
        run_id=run_id,
    )


_TEST_GEOMETRY = {
    "length": 0.45,
    "width": 0.35,
    "depth": 0.12,
    "hole_radius": 0.04,
    "top_thickness": 0.003,
    "back_thickness": 0.003,
}


def _make_compare_ready_run(
    run_root: Path,
    *,
    repo_root: Path,
    sample_id: str,
    run_id: str,
    mesh_profile: str,
    modes: List[Dict[str, Any]],
    worker_wall_s: float = 1000.0,
    peak_rss_bytes: int = 2 * 1024**3,
    geometry: Optional[Dict[str, Any]] = None,
) -> None:
    _make_strict_completed_run(run_root, sample_id=sample_id, run_id=run_id)
    for rel in (
        "lprod/checkpoint",
        "lprod/mesh",
        "scout/checkpoint",
        "scout/mesh",
        "scout/discovery",
        "worker_results",
    ):
        shutil.rmtree(run_root / rel, ignore_errors=True)
    (run_root / "lprod" / "lprod_target_plan.json").unlink(missing_ok=True)

    resolved = resolve_mesh_profile(
        mesh_profile=mesh_profile,
        dataset_version=DATASET_VERSION_REFERENCE if mesh_profile == MESH_PROFILE_REFERENCE else DATASET_VERSION_ROM,
    )
    geom = geometry or dict(_TEST_GEOMETRY)
    sample_in = apply_mesh_profile_to_sample_input(
        {
            "sample_id": sample_id,
            "geometry": geom,
            "top_wood_id": "spruce",
            "back_wood_id": "mahogany",
        },
        resolved,
    )
    write_json_atomic(run_root / "sample" / "sample_input.json", sample_in)

    from v2_b3_m4_lprod_interfaces import geometry_fingerprint  # noqa: WPS433

    identity = {
        "schema": "m4_physics_identity_v1",
        "sample_id": sample_id,
        "run_id": run_id,
        "production_acceptance_pass": True,
        "mesh_profile": resolved.mesh_profile,
        "mesh_level_id": resolved.mesh_level_id,
        "dataset_version": resolved.dataset_version,
        "effective_controls_m": dict(resolved.effective_controls_m),
        "generated_mesh_sha256": f"gen_{mesh_profile}",
        "operator_mesh_sha256": f"op_{mesh_profile}",
        "operator_mesh_matches_generated": True,
        "geometry_fingerprint": geometry_fingerprint(geom),
        "active_dimension": 1000 if mesh_profile == MESH_PROFILE_REFERENCE else 400,
        "masks": {"p_idx_aperture_count": 4},
        "fallback_flags": {"cross_sample_reuse": False},
        "path_contamination": {"contamination_detected": False},
    }
    write_json_atomic(run_root / "freeze" / "physics_identity_manifest.json", identity)
    write_json_atomic(
        run_root / "freeze" / "freeze_manifest.json",
        {"production_acceptance_pass": True, "status": "ok"},
    )
    write_json_atomic(
        run_root / "pipeline_run_manifest.json",
        {
            "sample_id": sample_id,
            "run_id": run_id,
            "mesh_profile": resolved.mesh_profile,
            "mesh_level_id": resolved.mesh_level_id,
            "dataset_version": resolved.dataset_version,
            "frequency_policy": {"band_hz": [60.0, 550.0]},
        },
    )
    write_json_atomic(
        run_root / "compaction" / "compaction_manifest.json",
        {"schema": "m4_compaction_v1", "status": "completed", "sample_id": sample_id, "run_id": run_id},
    )
    write_json_atomic(
        run_root / "aggregation" / "runtime_summary.json",
        {"schema": "m4_runtime_summary_v1", "sample_id": sample_id, "run_id": run_id},
    )
    _write_modes_catalog(run_root, modes)
    _write_validation_package(
        run_root,
        sample_id=sample_id,
        run_id=run_id,
        targets_hz=[100.0, 200.0, 300.0, 400.0, 500.0],
    )
    write_json_atomic(
        run_root / "m4_sample_runtime_provenance.json",
        {
            "stage_wall_times_s": {"stage5_workers": worker_wall_s},
            "peak_rss_bytes_max_worker": peak_rss_bytes,
            "worker_resource_records": [
                {
                    "chunk_id": "chunk_01",
                    "worker_pid": 1234,
                    "peak_rss_bytes": peak_rss_bytes,
                    "rss_measurement_method": "linux_proc_status_vmhwm",
                    "wall_seconds": worker_wall_s,
                    "exit_status": 0,
                }
            ],
            "rss_aggregate_note": "peak is per-worker VmHWM",
        },
    )
    _write_barrier_pass(repo_root, run_root, sample_id=sample_id, run_id=run_id)


class MeshProfileLibTest(unittest.TestCase):
    def test_default_profile_is_reference_canonical(self) -> None:
        r = resolve_mesh_profile()
        self.assertEqual(r.mesh_profile, MESH_PROFILE_REFERENCE)
        self.assertEqual(r.mesh_level_id, LEVEL_PROD_REFERENCE)
        self.assertEqual(r.dataset_version, DATASET_VERSION_REFERENCE)

    def test_rom_profile_auto_dataset(self) -> None:
        r = resolve_mesh_profile(mesh_profile="rom")
        self.assertEqual(r.mesh_level_id, LEVEL_ROM_PROD)
        self.assertEqual(r.dataset_version, DATASET_VERSION_ROM)

    def test_legacy_dataset_rejected_for_new_reference(self) -> None:
        with self.assertRaises(MeshProfileError):
            resolve_mesh_profile(mesh_profile="reference", dataset_version=DATASET_VERSION_LEGACY)

    def test_rom_profile_resolution(self) -> None:
        r = resolve_mesh_profile(mesh_profile="rom", dataset_version=DATASET_VERSION_ROM)
        self.assertEqual(r.mesh_level_id, LEVEL_ROM_PROD)
        self.assertAlmostEqual(r.effective_controls_m["wood_thickness_size_m"], 0.00125)

    def test_profile_dataset_mismatch_fails(self) -> None:
        with self.assertRaises(MeshProfileError):
            validate_profile_dataset_pairing(MESH_PROFILE_ROM, DATASET_VERSION_REFERENCE)
        with self.assertRaises(MeshProfileError):
            validate_profile_dataset_pairing(MESH_PROFILE_REFERENCE, DATASET_VERSION_ROM)

    def test_manifest_levels_exist(self) -> None:
        manifest = load_manifest()
        levels = manifest.get("mesh_levels") or {}
        self.assertIn(LEVEL_PROD_REFERENCE, levels)
        self.assertIn(LEVEL_ROM_PROD, levels)
        self.assertIn("L_prod", levels)

    def test_l_prod_alias_controls_match_reference(self) -> None:
        assert_l_prod_alias_controls_match()

    def test_checkpoint_reuse_profile_mismatch_fails(self) -> None:
        expected = resolve_mesh_profile(mesh_profile="rom", dataset_version=DATASET_VERSION_ROM)
        errors = validate_mesh_profile_reuse(
            expected=expected,
            existing={"mesh_profile": "reference", "mesh_level_id": LEVEL_PROD_REFERENCE},
            context="test",
        )
        self.assertTrue(errors)

    def test_cleanup_levels_include_both_profiles(self) -> None:
        levels = production_mesh_levels_for_cleanup()
        self.assertIn(LEVEL_PROD_REFERENCE, levels)
        self.assertIn(LEVEL_ROM_PROD, levels)
        self.assertEqual(MESH_LEVELS, levels)

    def test_no_shared_mesh_path_collision_between_profiles(self) -> None:
        sid = "sample_042"
        ref_path = mesh_path(LEVEL_PROD_REFERENCE, sid)
        rom_path = mesh_path(LEVEL_ROM_PROD, sid)
        self.assertNotEqual(ref_path, rom_path)


class MeshProfilePropagationTest(unittest.TestCase):
    def test_batch_spec_contains_profile(self) -> None:
        pool = {"shape_name": "classic", "entries": [{"id": "sample_001", "parameters": {}}]}
        entry = build_sample_input(
            pool=pool,
            entry=pool["entries"][0],
            lhs_row_index=0,
            batch_id="b1",
            lhs_source_path="pool.json",
            mesh_profile="rom",
            dataset_version=DATASET_VERSION_ROM,
        )
        self.assertEqual(entry["mesh_profile"], "rom")
        self.assertEqual(entry["mesh_level_id"], LEVEL_ROM_PROD)

        batch = build_lhs_batch_spec(
            pool=pool,
            samples=[{"sample_id": "sample_001", "run_id": "sample_001_t", "sample_input": entry}],
            batch_id="b1",
            lhs_source_path="pool.json",
            run_id_suffix="t",
            mesh_profile="rom",
            dataset_version=DATASET_VERSION_ROM,
        )
        self.assertEqual(batch["mesh_profile"], "rom")
        self.assertEqual(batch["dataset_version"], DATASET_VERSION_ROM)

    def test_generated_per_sample_spec_propagates_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            pool = {"shape_name": "classic", "entries": [{"id": "sample_003", "parameters": {}}]}
            resolved = resolve_mesh_profile(mesh_profile="reference", dataset_version=DATASET_VERSION_REFERENCE)
            sample_input = apply_mesh_profile_to_sample_input({"sample_id": "sample_003"}, resolved)
            batch = build_lhs_batch_spec(
                pool=pool,
                samples=[
                    {
                        "sample_id": "sample_003",
                        "run_id": "sample_003_run",
                        "sample_input": sample_input,
                    }
                ],
                batch_id="batch_x",
                lhs_source_path="pool.json",
                run_id_suffix="run",
                mesh_profile="reference",
                dataset_version=DATASET_VERSION_REFERENCE,
            )
            out = write_per_sample_spec(
                repo_root=repo,
                batch_spec=batch,
                sample_entry=batch["samples"][0],
                lhs_source_path="pool.json",
            )
            doc = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(doc["mesh_profile"], "reference")
            self.assertEqual(doc["mesh_level_id"], LEVEL_PROD_REFERENCE)


class DryRunProfileTest(unittest.TestCase):
    def test_reference_dry_run_uses_l_prod_reference_not_l_prod(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / ".git").mkdir()
            guitars = repo / "guitars"
            sample = {"sample_id": "sample_099", "schema": "m4_sample_input_v1"}
            with patch.object(pipeline_dry_run, "GUITARS_ROOT", guitars):
                plan = build_dry_run_plan(
                    repo_root=repo,
                    sample=sample,
                    run_id="dry_ref",
                    freq_min=60.0,
                    freq_max=550.0,
                    scout_spacing_hz=7.5,
                    scout_half_width_hz=3.75,
                    zone_spacing_dense=6.0,
                    zone_spacing_medium=9.0,
                    zone_spacing_sparse=12.5,
                    workers=3,
                    prod_python="python",
                    solver_python="python",
                    mesh_profile="reference",
                )
                self.assertEqual(plan["lprod_mesh_level"], LEVEL_PROD_REFERENCE)
                _write_tree(plan, force=True)
            self.assertTrue((plan["run_root"] / "lprod" / "mesh" / LEVEL_PROD_REFERENCE).is_dir())
            self.assertFalse((plan["run_root"] / "lprod" / "mesh" / "L_prod").exists())

    def test_rom_dry_run_uses_l_rom_prod(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / ".git").mkdir()
            guitars = repo / "guitars"
            sample = {"sample_id": "sample_100", "schema": "m4_sample_input_v1"}
            with patch.object(pipeline_dry_run, "GUITARS_ROOT", guitars):
                plan = build_dry_run_plan(
                    repo_root=repo,
                    sample=sample,
                    run_id="dry_rom",
                    freq_min=60.0,
                    freq_max=550.0,
                    scout_spacing_hz=7.5,
                    scout_half_width_hz=3.75,
                    zone_spacing_dense=6.0,
                    zone_spacing_medium=9.0,
                    zone_spacing_sparse=12.5,
                    workers=3,
                    prod_python="python",
                    solver_python="python",
                    mesh_profile="rom",
                )
                self.assertEqual(plan["lprod_mesh_level"], LEVEL_ROM_PROD)
                _write_tree(plan, force=True)
            self.assertTrue((plan["run_root"] / "lprod" / "mesh" / LEVEL_ROM_PROD).is_dir())
            self.assertFalse((plan["run_root"] / "lprod" / "mesh" / "L_prod").exists())


class TargetPlanDurabilityTest(unittest.TestCase):
    def test_durable_target_plan_survives_cleanup_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            run_root = repo / "guitars" / "sample_002" / "runs" / "sample_002_ref"
            run_root.mkdir(parents=True)
            write_json_atomic(
                run_root / "lprod" / "lprod_target_plan.json",
                {"sample_id": "sample_002", "targets_hz": [80.0, 160.0], "coverage_check": {"pass": True}},
            )
            result = preserve_target_plan_before_cleanup(
                run_root=run_root, sample_id="sample_002", run_id="sample_002_ref",
            )
            self.assertIsNotNone(result)
            durable = run_root / VALIDATION_INPUT_PACKAGE_REL / "target_plan.json"
            self.assertTrue(durable.is_file())
            (run_root / "lprod" / "lprod_target_plan.json").unlink()
            body, sha, errs = load_durable_target_plan(run_root)
            self.assertIsNotNone(body)
            self.assertIsNotNone(sha)
            self.assertNotIn("TARGET_PLAN_UNAVAILABLE", errs)

    def test_comparison_does_not_require_lprod_target_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            run_root = repo / "guitars" / "sample_002" / "runs" / "run_x"
            _write_validation_package(
                run_root, sample_id="sample_002", run_id="run_x", targets_hz=[100.0, 200.0],
            )
            body, sha, errs = load_durable_target_plan(run_root)
            self.assertIsNotNone(body)
            self.assertFalse((run_root / "lprod" / "lprod_target_plan.json").exists())


class LegacyReferenceCompatibilityTest(unittest.TestCase):
    def _legacy_run(self, repo: Path, complete: bool = True) -> Path:
        from v2_b3_m4_lprod_interfaces import geometry_fingerprint  # noqa: WPS433

        run_root = repo / "guitars" / "sample_002" / "runs" / "sample_002_legacy"
        _make_strict_completed_run(run_root, sample_id="sample_002", run_id="sample_002_legacy")
        for rel in ("lprod", "worker_results", "scout/mesh", "scout/checkpoint", "scout/discovery"):
            shutil.rmtree(run_root / rel, ignore_errors=True)
        geom = dict(_TEST_GEOMETRY)
        write_json_atomic(
            run_root / "sample" / "sample_input.json",
            {
                "sample_id": "sample_002",
                "geometry": geom,
                "top_wood_id": "spruce",
                "back_wood_id": "mahogany",
                "effective_controls_m": dict(REFERENCE_CONTROLS_M),
            },
        )
        identity = {
            "schema": "m4_physics_identity_v1",
            "sample_id": "sample_002",
            "run_id": "sample_002_legacy",
            "production_acceptance_pass": True,
            "generated_mesh_sha256": "legacy_gen",
            "operator_mesh_sha256": "legacy_op",
            "operator_mesh_matches_generated": True,
            "effective_controls_m": dict(REFERENCE_CONTROLS_M),
            "geometry_fingerprint": geometry_fingerprint(geom),
            "active_dimension": 1000,
            "masks": {"p_idx_aperture_count": 4},
            "fallback_flags": {"cross_sample_reuse": False},
            "path_contamination": {"contamination_detected": False},
            "dataset_version": DATASET_VERSION_LEGACY,
        }
        if not complete:
            identity.pop("operator_mesh_sha256")
        write_json_atomic(run_root / "freeze" / "physics_identity_manifest.json", identity)
        write_json_atomic(
            run_root / "freeze" / "freeze_manifest.json",
            {
                "production_acceptance_pass": True,
                "production_acceptance_failures": [],
                "effective_controls_m": dict(REFERENCE_CONTROLS_M),
                "p_idx_aperture_count": 4,
            },
        )
        write_json_atomic(
            run_root / "compaction" / "compaction_manifest.json",
            {"schema": "m4_compaction_v1", "status": "completed"},
        )
        write_json_atomic(
            run_root / "aggregation" / "runtime_summary.json",
            {"schema": "m4_runtime_summary_v1"},
        )
        _write_barrier_pass(repo, run_root, sample_id="sample_002", run_id="sample_002_legacy")
        return run_root

    def test_legacy_reference_accepted_with_full_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            run_root = self._legacy_run(repo, complete=True)
            ok, meta, errors = evaluate_legacy_reference_compatibility(run_root=run_root, repo_root=repo)
            self.assertTrue(ok, errors)
            self.assertTrue(meta.get("legacy_reference_compatibility"))
            self.assertEqual(meta.get("resolved_reference_profile"), MESH_PROFILE_REFERENCE)

    def test_missing_legacy_evidence_precondition_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            run_root = self._legacy_run(repo, complete=False)
            ok, meta, errors = evaluate_legacy_reference_compatibility(run_root=run_root, repo_root=repo)
            self.assertFalse(ok)
            self.assertFalse(meta.get("legacy_reference_compatibility"))
            self.assertTrue(errors)


class MeshProfileValidationPreconditionTest(unittest.TestCase):
    def test_require_cleanup_barrier_fails_without_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            run_root = repo / "guitars" / "sample_001" / "runs" / "sample_001_run"
            run_root.mkdir(parents=True)
            ok, _, errors = require_cleanup_barrier_passed_for_validation(
                repo_root=repo,
                run_root=run_root,
                label="reference",
            )
            self.assertFalse(ok)
            self.assertTrue(any("missing_cleanup_barrier_manifest" in e for e in errors))

    def test_compare_aborts_when_cleanup_barrier_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            ref = repo / "guitars" / "sample_001" / "runs" / "ref_run"
            cand = repo / "guitars" / "sample_001" / "runs" / "cand_run"
            ref.mkdir(parents=True)
            cand.mkdir(parents=True)
            report = compare_runs(reference_run=ref, candidate_run=cand, repo_root=repo)
            self.assertFalse(report.get("comparison_executed"))
            self.assertEqual(report.get("status"), "PRECONDITION_FAILED")
            self.assertEqual(compare_exit_code(report), EXIT_PRECONDITION_FAIL)

    def test_scan_detects_candidate_reference_to_other_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            ref = repo / "guitars" / "sample_002" / "runs" / "ref_run"
            cand = repo / "guitars" / "sample_002" / "runs" / "cand_run"
            ref.mkdir(parents=True)
            cand.mkdir(parents=True)
            write_json_atomic(
                cand / "sample" / "sample_input.json",
                {
                    "sample_id": "sample_002",
                    "mesh_profile": "rom",
                    "mesh_file": str(ref / "lprod" / "mesh" / "sample_002.msh"),
                },
            )
            hits = scan_candidate_references_other_run(cand, forbidden_root=ref)
            self.assertTrue(hits)

    def test_validation_input_package_materialized_with_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            run_root = repo / "guitars" / "sample_002" / "runs" / "sample_002_rom"
            run_root.mkdir(parents=True)
            source = repo / "external_target_plan.json"
            write_json_atomic(
                source,
                {"sample_id": "sample_002", "targets_hz": [100.0, 110.0], "coverage_check": {"pass": True}},
            )
            plan = install_explicit_target_plan(
                run_root=run_root,
                target_plan_path=source,
                sample_id="sample_002",
                run_id="sample_002_rom",
            )
            self.assertTrue(plan.get("explicit_target_plan"))
            self.assertIn(VALIDATION_INPUT_PACKAGE_REL, str(plan.get("validation_input_package")))
            pkg = run_root / str(plan["validation_input_package"])
            self.assertTrue(pkg.is_file())
            self.assertIsNotNone(plan.get("validation_input_sha256"))


def _synthetic_modes_pass() -> List[Dict[str, Any]]:
    modes = []
    for i, f in enumerate([80.0, 120.0, 180.0, 220.0, 280.0, 380.0, 420.0, 480.0, 520.0]):
        modes.append(
            {
                "mode_id": f"m{i}",
                "frequency_hz": f,
                "bridge_excitation_abs": 1.0 - i * 0.05,
                "mic_output_proxy": 0.9 - i * 0.04,
                "coupling_class": "structural" if f < 300 else "mixed",
                "dominant_region": "bridge",
            }
        )
    return modes


class SyntheticComparisonTest(unittest.TestCase):
    def _build_pair(
        self,
        repo: Path,
        *,
        ref_modes: List[Dict[str, Any]],
        cand_modes: List[Dict[str, Any]],
        ref_wall: float = 1000.0,
        cand_wall: float = 600.0,
        cand_peak_rss: int = 3 * 1024**3,
    ) -> tuple[Path, Path]:
        ref = repo / "guitars" / "sample_005" / "runs" / "ref_run"
        cand = repo / "guitars" / "sample_005" / "runs" / "cand_run"
        shared_plan = repo / "shared_target_plan.json"
        write_json_atomic(
            shared_plan,
            {
                "sample_id": "sample_005",
                "targets_hz": [100.0, 200.0, 300.0, 400.0, 500.0],
                "frequency_range_hz": [60.0, 550.0],
                "coverage_check": {"pass": True},
            },
        )
        _make_compare_ready_run(
            ref, repo_root=repo, sample_id="sample_005", run_id="ref_run",
            mesh_profile=MESH_PROFILE_REFERENCE, modes=ref_modes, worker_wall_s=ref_wall,
        )
        _make_compare_ready_run(
            cand, repo_root=repo, sample_id="sample_005", run_id="cand_run",
            mesh_profile=MESH_PROFILE_ROM, modes=cand_modes, worker_wall_s=cand_wall,
            peak_rss_bytes=cand_peak_rss,
        )
        _write_validation_package(
            ref, sample_id="sample_005", run_id="ref_run",
            targets_hz=[100.0, 200.0, 300.0, 400.0, 500.0], source_path=shared_plan,
        )
        _write_validation_package(
            cand, sample_id="sample_005", run_id="cand_run",
            targets_hz=[100.0, 200.0, 300.0, 400.0, 500.0], source_path=shared_plan,
        )
        _write_barrier_pass(repo, ref, sample_id="sample_005", run_id="ref_run")
        _write_barrier_pass(repo, cand, sample_id="sample_005", run_id="cand_run")
        return ref, cand

    def test_synthetic_comparison_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            modes = _synthetic_modes_pass()
            ref, cand = self._build_pair(repo, ref_modes=modes, cand_modes=modes)
            report = compare_runs(reference_run=ref, candidate_run=cand, repo_root=repo)
            self.assertTrue(report.get("comparison_executed"))
            self.assertTrue(report.get("acceptance_pass"), report.get("acceptance_evaluation"))
            self.assertEqual(compare_exit_code(report), EXIT_PASS)

    def test_synthetic_acceptance_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            ref_modes = _synthetic_modes_pass()
            cand_modes = [
                {**m, "frequency_hz": m["frequency_hz"] + 4.0, "mode_id": f"c{m['mode_id']}"}
                for m in ref_modes
            ]
            ref, cand = self._build_pair(repo, ref_modes=ref_modes, cand_modes=cand_modes)
            report = compare_runs(reference_run=ref, candidate_run=cand, repo_root=repo)
            self.assertTrue(report.get("comparison_executed"))
            self.assertFalse(report.get("acceptance_pass"))
            self.assertEqual(compare_exit_code(report), EXIT_ACCEPTANCE_FAIL)

    def test_synthetic_precondition_missing_validation_package(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            modes = _synthetic_modes_pass()
            ref, cand = self._build_pair(repo, ref_modes=modes, cand_modes=modes)
            shutil.rmtree(cand / VALIDATION_INPUT_PACKAGE_REL)
            report = compare_runs(reference_run=ref, candidate_run=cand, repo_root=repo)
            self.assertEqual(report.get("status"), "PRECONDITION_FAILED")
            self.assertEqual(compare_exit_code(report), EXIT_PRECONDITION_FAIL)

    def test_synthetic_incomplete_missing_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            modes = _synthetic_modes_pass()
            ref, cand = self._build_pair(repo, ref_modes=modes, cand_modes=modes)
            (cand / "m4_sample_runtime_provenance.json").unlink()
            report = compare_runs(reference_run=ref, candidate_run=cand, repo_root=repo)
            self.assertIn(report.get("status"), ("INCOMPLETE", "ACCEPTANCE_FAILED"))
            if report.get("status") == "INCOMPLETE":
                self.assertEqual(compare_exit_code(report), EXIT_INCOMPLETE)

    def test_band_recall_and_overlap_metrics_populated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            modes = _synthetic_modes_pass()
            ref, cand = self._build_pair(repo, ref_modes=modes, cand_modes=modes)
            report = compare_runs(reference_run=ref, candidate_run=cand, repo_root=repo)
            self.assertGreaterEqual(report["coupling_output"]["bridge_top10"]["overlap_count"], 8)
            self.assertGreaterEqual(report["coupling_output"]["mic_top10"]["overlap_count"], 7)
            self.assertGreaterEqual(
                report["modal_retention"]["recall_below_350"]["recall"], 0.95,
            )


class ScoutBandThirdAndFamilyTest(unittest.TestCase):
    def _rows(self) -> List[Dict[str, Any]]:
        freqs = [80, 120, 180, 220, 280, 380, 420, 480, 520]
        return [
            {
                "mode_id": f"m{f}",
                "frequency_hz": float(f),
                "coupling_class": "structural" if f < 300 else "mixed",
                "dominant_region": "bridge",
                "bridge_excitation_abs": 1.0 - i * 0.05,
            }
            for i, f in enumerate(freqs)
        ]

    def test_band_third_counts_derived_from_catalog(self) -> None:
        counts = derive_band_third_counts_from_catalog(self._rows())
        self.assertGreaterEqual(counts["low_third"], 2)
        self.assertGreaterEqual(counts["mid_third"], 2)
        self.assertGreaterEqual(counts["high_third"], 2)

    def test_intrinsic_band_third_pass_identical_catalogs(self) -> None:
        rows = self._rows()
        result = compare_intrinsic_band_third_coverage(rows, rows)
        self.assertTrue(result["intrinsic_band_third_no_loss_pass"])

    def test_intrinsic_band_third_fails_when_third_lost(self) -> None:
        ref = self._rows()
        cand = [r for r in ref if float(r["frequency_hz"]) < 350]
        result = compare_intrinsic_band_third_coverage(ref, cand)
        self.assertFalse(result["intrinsic_band_third_no_loss_pass"])
        self.assertTrue(result["missing_covered_band_thirds"] or result["material_population_changes"])

    def test_mode_family_survival_pass(self) -> None:
        rows = self._rows()
        result = compare_mode_family_survival(rows, rows)
        self.assertTrue(result["family_survival_pass"])
        self.assertEqual(result["unexplained_family_loss_count"], 0)

    def test_mode_family_survival_fails_when_major_family_missing(self) -> None:
        ref = self._rows()
        families = major_mode_families(ref)
        self.assertTrue(families)
        cand = [r for r in ref if float(r["frequency_hz"]) < 200]
        result = compare_mode_family_survival(ref, cand)
        self.assertFalse(result["family_survival_pass"])
        self.assertGreater(result["unexplained_family_loss_count"], 0)

    def test_missing_mandatory_intrinsic_produces_incomplete(self) -> None:
        report = {
            "frequencies": {
                "global_median_rel_error": 0.0,
                "global_p95_rel_error": 0.0,
                "bands": {
                    "60_150": {"max_rel_error": 0.0, "matched_count": 1},
                    "150_350": {"median_rel_error": 0.0, "max_rel_error": 0.0},
                    "350_550": {"median_rel_error": 0.0, "max_rel_error": 0.0},
                },
            },
            "modal_retention": {
                "recall_below_350": {"recall": 1.0},
                "recall_350_550": {"recall": 1.0},
            },
            "coupling_output": {
                "coupling_class_agreement": 1.0,
                "bridge_top10": {"overlap_count": 10},
                "mic_top10": {"overlap_count": 8},
            },
            "performance": {
                "runtime_reduction_fraction": 0.5,
                "candidate_peak_rss_bytes_max_worker": 1024**3,
            },
            "mac": {"MAC_STATUS": "UNAVAILABLE"},
            "intrinsic_coverage": {},
            "mode_family_survival": {"family_survival_pass": True},
        }
        _, acceptance_pass, incomplete = evaluate_acceptance(report)
        self.assertFalse(acceptance_pass)
        self.assertTrue(incomplete)


class ValidationReadinessAuditTest(unittest.TestCase):
    def test_audit_blocked_when_no_completed_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            report = audit_readiness(repo_root=repo, sample_id="sample_002")
            self.assertEqual(report["FINAL_STATUS"], "BLOCKED")
            self.assertFalse(report["LEGACY_REFERENCE_READY"])
            self.assertFalse(report["TARGET_PLAN_READY"])


class ValidationReadinessPostCleanupTest(unittest.TestCase):
    def test_post_cleanup_region_dof_evidence_passes_without_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            run_root = LegacyReferenceCompatibilityTest()._legacy_run(repo, complete=True)
            ok, errors, meta = evaluate_post_cleanup_region_dof_evidence(run_root)
            self.assertTrue(ok, errors)
            self.assertEqual(meta.get("evidence_mode"), "durable_post_cleanup")
            self.assertFalse((run_root / "lprod" / "checkpoint").exists())

    def test_legacy_reference_audit_passes_after_cleanup_without_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            run_root = LegacyReferenceCompatibilityTest()._legacy_run(repo, complete=True)
            report = audit_legacy_reference(repo_root=repo, run_root=run_root)
            self.assertTrue(report["LEGACY_REFERENCE_READY"], report.get("errors"))
            self.assertEqual(
                (report.get("region_dof_gate") or {}).get("evidence_mode"),
                "durable_post_cleanup",
            )
            self.assertFalse((report.get("errors") or []))


if __name__ == "__main__":
    unittest.main()
