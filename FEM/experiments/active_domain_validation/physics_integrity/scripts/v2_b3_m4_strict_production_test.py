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
    scan_cross_sample_path_contamination,
    validate_physics_identity_manifest,
)
from v2_b3_m4_production_contracts import (  # noqa: E402
    DATASET_VERSION,
    is_strict_production_mode,
    require_aperture_mask_production,
)


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
