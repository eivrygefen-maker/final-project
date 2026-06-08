#!/usr/bin/env python3
"""Unit tests for strict production contracts and mode provenance fingerprints."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_m4_mode_provenance import (  # noqa: E402
    eigenvector_fingerprint_sha256,
    eigenvector_sketch_sha256,
)
from v2_b3_m4_physics_identity_lib import (  # noqa: E402
    foreign_sample_ids_in_text,
    scan_cross_sample_path_contamination,
    validate_physics_identity_manifest,
)
from v2_b3_m4_production_contracts import (  # noqa: E402
    DATASET_VERSION,
    _evaluate_scout_strict_failures,
    is_strict_production_mode,
    require_aperture_mask_production,
)
from v2_b3_run_coarse_scout_lhs_batch import _verify_density_result  # noqa: E402


class StrictProductionTest(unittest.TestCase):
    def test_strict_mode_always_on_for_corrected_dataset(self) -> None:
        self.assertTrue(is_strict_production_mode(dataset_version=DATASET_VERSION))

    def test_strict_mode_ignores_env_disable(self) -> None:
        import os

        os.environ["B3_ALLOW_CAVITY_MAX_MIC_FALLBACK"] = "1"
        os.environ["B3_DIAGNOSTIC_MIC_FALLBACK_ONLY"] = "1"
        try:
            self.assertTrue(require_aperture_mask_production(dataset_version=DATASET_VERSION))
        finally:
            os.environ.pop("B3_ALLOW_CAVITY_MAX_MIC_FALLBACK", None)
            os.environ.pop("B3_DIAGNOSTIC_MIC_FALLBACK_ONLY", None)

    def test_eigenvector_fingerprint_stable(self) -> None:
        v = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        a = eigenvector_fingerprint_sha256(v)
        b = eigenvector_fingerprint_sha256(v * 1.0000001)
        self.assertEqual(a, b)
        c = eigenvector_fingerprint_sha256(np.array([1.0, 2.0, 3.0, 4.1]))
        self.assertNotEqual(a, c)

    def test_eigenvector_sketch_differs_for_different_vectors(self) -> None:
        v1 = np.linspace(0, 1, 128)
        v2 = np.linspace(0, 1, 128) + 1e-6
        self.assertNotEqual(
            eigenvector_sketch_sha256(v1),
            eigenvector_sketch_sha256(v2),
        )

    def test_path_contamination_detects_other_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sample").mkdir()
            bad = root / "lprod" / "resolved_core_config.json"
            bad.parent.mkdir(parents=True)
            bad.write_text(
                json.dumps({"solver": {"mesh_file": "guitars/sample_001/runs/x/lprod/mesh/L_prod/sample_001.msh"}}),
                encoding="utf-8",
            )
            rep = scan_cross_sample_path_contamination(root, sample_id="sample_002")
            self.assertTrue(rep["contamination_detected"])
            self.assertEqual(rep["contamination_hits"][0]["foreign_sample_ids"], ["sample_001"])

    def test_current_sample_000_path_reference_passes(self) -> None:
        text = "guitars/sample_000/runs/sample_000_m4prod2_strict_val/lprod/mesh/L_prod/sample_000.msh"
        self.assertEqual(foreign_sample_ids_in_text(text, current_sample_id="sample_000"), set())

    def test_current_run_id_reference_passes(self) -> None:
        text = "run_id=sample_000_m4prod2_strict_val"
        self.assertEqual(foreign_sample_ids_in_text(text, current_sample_id="sample_000"), set())

    def test_current_chunk_id_reference_passes(self) -> None:
        text = "worker_results/sample_000_chunk_12/solver_result.json"
        self.assertEqual(foreign_sample_ids_in_text(text, current_sample_id="sample_000"), set())

    def test_foreign_sample_001_in_sample_000_run_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = root / "lprod" / "resolved_core_config.json"
            cfg.parent.mkdir(parents=True)
            cfg.write_text(
                json.dumps(
                    {"solver": {"mesh_file": "guitars/sample_001/runs/x/lprod/mesh/L_prod/sample_001.msh"}}
                ),
                encoding="utf-8",
            )
            rep = scan_cross_sample_path_contamination(root, sample_id="sample_000")
            self.assertTrue(rep["contamination_detected"])
            self.assertEqual(rep["contamination_hits"][0]["foreign_sample_ids"], ["sample_001"])

    def test_only_sample_000_references_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "pipeline_run_manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "sample_id": "sample_000",
                        "run_id": "sample_000_m4prod2_strict_val",
                        "chunk_id": "sample_000_chunk_04",
                    }
                ),
                encoding="utf-8",
            )
            rep = scan_cross_sample_path_contamination(root, sample_id="sample_000")
            self.assertFalse(rep["contamination_detected"])

    def test_sample_000_plus_sample_002_fails_with_sample_002(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            solver = root / "worker_results" / "sample_000_chunk_01" / "solver_result.json"
            solver.parent.mkdir(parents=True)
            solver.write_text(
                json.dumps(
                    {
                        "targets": [
                            {
                                "accepted_modes": [
                                    {
                                        "frequency_hz": 100.0,
                                        "source_solver_result": (
                                            "guitars/sample_002/runs/x/worker_results/"
                                            "sample_002_chunk_01/solver_result.json"
                                        ),
                                    }
                                ]
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            rep = scan_cross_sample_path_contamination(root, sample_id="sample_000")
            self.assertTrue(rep["contamination_detected"])
            self.assertEqual(rep["contamination_hits"][0]["foreign_sample_ids"], ["sample_002"])

    def test_historical_e2e_note_mentioning_sample_001_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            e2e = root / "freeze" / "sample_e2e_run_manifest.json"
            e2e.parent.mkdir(parents=True)
            e2e.write_text(
                json.dumps({"notes": ["Only sample_001 validated end-to-end on VM"]}),
                encoding="utf-8",
            )
            rep = scan_cross_sample_path_contamination(root, sample_id="sample_000")
            self.assertFalse(rep["contamination_detected"])

    def test_generated_physics_identity_manifest_audit_results_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ident = root / "freeze" / "physics_identity_manifest.json"
            ident.parent.mkdir(parents=True)
            ident.write_text(
                json.dumps(
                    {
                        "path_contamination": {
                            "contamination_detected": True,
                            "contamination_hits": [{"foreign_sample_ids": ["sample_001"]}],
                        }
                    }
                ),
                encoding="utf-8",
            )
            rep = scan_cross_sample_path_contamination(root, sample_id="sample_000")
            self.assertFalse(rep["contamination_detected"])

    def test_foreign_sample_in_operator_mesh_file_used_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            built = root / "lprod" / "checkpoint" / "built_metadata.json"
            built.parent.mkdir(parents=True)
            built.write_text(
                json.dumps(
                    {
                        "operator_mesh_file_used": (
                            "guitars/sample_001/runs/x/lprod/mesh/L_prod/sample_001.msh"
                        )
                    }
                ),
                encoding="utf-8",
            )
            rep = scan_cross_sample_path_contamination(root, sample_id="sample_000")
            self.assertTrue(rep["contamination_detected"])
            self.assertEqual(rep["contamination_hits"][0]["file"], "lprod/checkpoint/built_metadata.json")

    def test_foreign_sample_in_aggregation_source_path_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog = root / "aggregation" / "modes_catalog.jsonl"
            catalog.parent.mkdir(parents=True)
            catalog.write_text(
                json.dumps(
                    {
                        "frequency_hz": 529.5,
                        "source_worker_result": (
                            "guitars/sample_003/runs/x/worker_results/"
                            "sample_003_chunk_02/worker_result.json"
                        ),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            rep = scan_cross_sample_path_contamination(root, sample_id="sample_000")
            self.assertTrue(rep["contamination_detected"])
            self.assertIn("sample_003", rep["contamination_hits"][0]["foreign_sample_ids"])

    def test_strict_acceptance_cannot_pass_when_cross_sample_reuse_true(self) -> None:
        man = {
            "schema": "m4_physics_identity_v1",
            "sample_id": "sample_000",
            "run_id": "sample_000_m4prod2_strict_val",
            "generated_mesh_sha256": "abc",
            "operator_mesh_matches_generated": True,
            "active_dimension": 100,
            "production_acceptance_pass": True,
            "fallback_flags": {"cross_sample_reuse": True},
            "masks": {"p_idx_aperture_count": 10},
            "path_contamination": {"contamination_detected": False},
        }
        ok, errs = validate_physics_identity_manifest(man)
        self.assertFalse(ok)
        self.assertIn("fallback_flag_true:cross_sample_reuse", errs)
        self.assertIn("production_acceptance_pass_inconsistent_with_cross_sample_reuse", errs)
        self.assertIn("production_acceptance_pass!=true", errs)

    def test_strict_density_verify_rejects_partial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "density_result.json"
            path.write_text(
                json.dumps(
                    {
                        "status": "PARTIAL",
                        "coverage_policy": "intrinsic_discovered_modes_v1",
                        "intrinsic_coverage_pass": False,
                        "intrinsic_coverage_failures": ["stub"],
                        "spacings": [
                            {
                                "spacing_hz": 7.5,
                                "coverage_pass": False,
                                "unique_accepted_count": 3,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            ok, detail, _ = _verify_density_result(path, strict=True)
            self.assertFalse(ok)
            self.assertIn("status=PARTIAL", detail)

    def test_strict_density_verify_accepts_intrinsic_pass(self) -> None:
        from test_m4_scout_intrinsic_coverage import _intrinsic_density_payload, _rich_freqs  # noqa: WPS433

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "density_result.json"
            path.write_text(json.dumps(_intrinsic_density_payload(freqs=_rich_freqs())), encoding="utf-8")
            ok, detail, _ = _verify_density_result(path, strict=True)
            self.assertTrue(ok, detail)

    def test_scout_strict_failures_reject_partial_density_and_best_effort(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            discovery = root / "scout" / "discovery"
            discovery.mkdir(parents=True)
            (discovery / "density_result.json").write_text(
                json.dumps(
                    {
                        "status": "PARTIAL",
                        "coverage_policy": "legacy_external_reference",
                        "intrinsic_coverage_pass": False,
                        "spacings": [{"spacing_hz": 7.5, "coverage_pass": False, "unique_accepted_count": 1}],
                    }
                ),
                encoding="utf-8",
            )
            scout = root / "scout"
            scout.mkdir(parents=True, exist_ok=True)
            (scout / "scout_result.json").write_text(
                json.dumps({"discovery": {"experiment_status": "PARTIAL"}}),
                encoding="utf-8",
            )
            lprod_ckpt = root / "lprod" / "checkpoint"
            lprod_ckpt.mkdir(parents=True)
            (lprod_ckpt / "synthesis_metadata.json").write_text(
                json.dumps({"region_dof_indices_status": "BEST_EFFORT_PASS"}),
                encoding="utf-8",
            )
            failures = _evaluate_scout_strict_failures(root)
            self.assertTrue(any("scout_density_intrinsic" in f for f in failures))
            self.assertIn("scout_discovery_experiment_status=PARTIAL", failures)
            self.assertTrue(
                any("region_dof_indices_status=BEST_EFFORT_PASS" in f for f in failures)
            )

    def test_physics_identity_manifest_validation_requires_aperture(self) -> None:
        man = {
            "schema": "m4_physics_identity_v1",
            "sample_id": "sample_001",
            "run_id": "sample_001_m4prod2",
            "generated_mesh_sha256": "abc",
            "operator_mesh_matches_generated": True,
            "active_dimension": 100,
            "production_acceptance_pass": True,
            "fallback_flags": {},
            "masks": {"p_idx_aperture_count": 0},
            "path_contamination": {"contamination_detected": False},
        }
        ok, errs = validate_physics_identity_manifest(man)
        self.assertFalse(ok)
        self.assertIn("p_idx_aperture_count<=0", errs)


if __name__ == "__main__":
    unittest.main()
