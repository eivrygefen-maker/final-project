#!/usr/bin/env python3
"""Regression: region-DOF PASS semantics and strict production freeze gates."""
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

from v2_b3_m4_freeze_first_e2e_run import (  # noqa: E402
    AGG_STATUS_PASS,
    TERMINAL_E2E,
    mark_freeze_stage_failed,
    promote_pipeline_terminal_status,
)
from v2_b3_m4_mesh_profile_lib import DATASET_VERSION_REFERENCE  # noqa: E402
from v2_b3_m4_production_contracts import (  # noqa: E402
    DATASET_VERSION,
    evaluate_production_region_dof_gate,
    validate_post_export_region_dof_contract,
)
from v2_b3_m4_production_freeze import replay_production_freeze  # noqa: E402
from v2_b3_m4_production_freeze_test import _write_minimal_production_aggregated_run  # noqa: E402
from v2_b3_m4_run_one_sample import _stage_pass_checkpoint  # noqa: E402
from v2_b3_synthesis_export import (  # noqa: E402
    REGION_DOF_STATUS_PASS,
    export_region_dof_indices_from_operator_build,
    region_dof_status_is_pass,
)
from v2_b3_petsc_util import write_json_atomic  # noqa: E402


def _write_core_config(path: Path) -> None:
    write_json_atomic(
        path,
        {
            "dataset_version": DATASET_VERSION,
            "m4_run_metadata": {"dataset_version": DATASET_VERSION},
            "geometry_numeric_parameters": {"length": 0.48, "width": 0.325, "depth": 0.1},
            "solver": {"mesh_file": "mesh.msh"},
        },
    )


def _write_valid_region_dof_checkpoint(
    ckpt: Path,
    *,
    status: str = "PASS",
    mode: str = "best_effort",
    include_aperture: bool = True,
) -> None:
    ckpt.mkdir(parents=True, exist_ok=True)
    if include_aperture:
        np.savez_compressed(
            ckpt / "region_dof_indices.npz",
            p_idx_aperture=np.array([10, 11, 12, 13], dtype=np.int32),
            u_idx_top=np.array([0, 1], dtype=np.int32),
            u_idx_back=np.array([2], dtype=np.int32),
            u_idx_ribs=np.array([], dtype=np.int32),
            u_idx_soundhole=np.array([], dtype=np.int32),
            p_idx_air=np.array([10, 11, 12, 13], dtype=np.int32),
            p_idx_all=np.array([10, 11, 12, 13], dtype=np.int32),
            u_idx_all=np.arange(5, dtype=np.int32),
        )
        write_json_atomic(
            ckpt / "region_dof_metadata.json",
            {
                "aperture_selection_method": "facet_adjacent_air_cell_dofs_v1",
                "p_idx_aperture_count": 4,
            },
        )
    else:
        np.savez_compressed(
            ckpt / "region_dof_indices.npz",
            p_idx_aperture=np.array([], dtype=np.int32),
            u_idx_top=np.array([0, 1], dtype=np.int32),
            u_idx_back=np.array([2], dtype=np.int32),
            u_idx_ribs=np.array([], dtype=np.int32),
            u_idx_soundhole=np.array([], dtype=np.int32),
            p_idx_air=np.array([10, 11, 12, 13], dtype=np.int32),
            p_idx_all=np.array([10, 11, 12, 13], dtype=np.int32),
            u_idx_all=np.arange(5, dtype=np.int32),
        )
    write_json_atomic(
        ckpt / "built_metadata.json",
        {
            "n_u_b3": 5,
            "p_idx": [10, 11, 12, 13],
            "active_dimension": 100,
            "p_idx_aperture_count": 4 if include_aperture else 0,
            "aperture_selection_method": "facet_adjacent_air_cell_dofs_v1",
        },
    )
    write_json_atomic(
        ckpt / "synthesis_metadata.json",
        {
            "region_dof_indices_mode": mode,
            "region_dof_indices_status": status,
        },
    )
    write_json_atomic(
        ckpt / "checkpoint_export_manifest.json",
        {"status": "PASS", "export_pass": True},
    )


class RegionDofStrictContractTests(unittest.TestCase):
    def test_best_effort_build_mode_with_complete_bundle_is_pass(self) -> None:
        region_dof_build = {
            "u_idx_top": np.array([0, 1], dtype=np.int32),
            "u_idx_back": np.array([2], dtype=np.int32),
            "u_idx_ribs": np.array([], dtype=np.int32),
            "u_idx_soundhole": np.array([], dtype=np.int32),
            "p_idx_all": np.array([10, 11, 12, 13], dtype=np.int32),
            "p_idx_aperture": np.array([10, 11], dtype=np.int32),
            "u_idx_all": np.arange(5, dtype=np.int32),
            "region_dof_mesh_file": "/tmp/sample.msh",
            "aperture_selection_method": "facet_adjacent_air_cell_dofs_v1",
        }
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp)
            status, err = export_region_dof_indices_from_operator_build(
                ckpt,
                region_dof_build=region_dof_build,
            )
            self.assertIsNone(err)
            self.assertEqual(status, REGION_DOF_STATUS_PASS)
            self.assertTrue(region_dof_status_is_pass(status))
            self.assertTrue((ckpt / "region_dof_indices.npz").is_file())
            self.assertEqual(status, "PASS")

    def test_missing_aperture_contract_fails_post_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "ckpt"
            core = Path(tmp) / "core.json"
            _write_core_config(core)
            _write_valid_region_dof_checkpoint(ckpt, include_aperture=False)
            errors = validate_post_export_region_dof_contract(ckpt, core_config_path=core)
            self.assertTrue(errors)

    def test_strict_run_blocked_with_best_effort_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            core = run_root / "lprod" / "resolved_core_config.json"
            _write_core_config(core)
            _write_valid_region_dof_checkpoint(
                run_root / "lprod" / "checkpoint",
                status="BEST_EFFORT_PASS",
            )
            _write_valid_region_dof_checkpoint(
                run_root / "scout" / "checkpoint",
                status="BEST_EFFORT_PASS",
            )
            write_json_atomic(
                run_root / "sample" / "resolved_core_config.json",
                json.loads(core.read_text(encoding="utf-8")),
            )
            ok, errors = evaluate_production_region_dof_gate(run_root, repo_root=Path(tmp))
            self.assertFalse(ok)
            self.assertTrue(any("BEST_EFFORT_PASS" in e for e in errors))
            self.assertFalse(_stage_pass_checkpoint(run_root, production_mode=True))

    def test_freeze_failure_stays_fail_not_pass_with_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            write_json_atomic(
                run_root / "pipeline_run_manifest.json",
                {
                    "terminal_status": TERMINAL_E2E,
                    "stages": {
                        "stage6_aggregate": {"status": "PASS", "aggregation_status": AGG_STATUS_PASS},
                        "stage6_freeze": {"status": "PLANNED"},
                    },
                },
            )
            mark_freeze_stage_failed(run_root, reason="production_acceptance_failed")
            pipeline = json.loads((run_root / "pipeline_run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(pipeline["stages"]["stage6_freeze"]["status"], "FAIL")
            self.assertNotEqual(pipeline["stages"]["stage6_freeze"]["status"], "PASS_WITH_WARNING")

    def test_aggregation_pass_alone_does_not_promote_freeze_warning_in_strict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            write_json_atomic(
                run_root / "pipeline_run_manifest.json",
                {"terminal_status": "RUNNING", "stages": {}},
            )
            promote_pipeline_terminal_status(
                run_root,
                aggregation_status=AGG_STATUS_PASS,
                allow_freeze_warning=False,
            )
            pipeline = json.loads((run_root / "pipeline_run_manifest.json").read_text(encoding="utf-8"))
            freeze_st = pipeline["stages"]["stage6_freeze"]
            self.assertNotEqual(freeze_st.get("status"), "PASS_WITH_WARNING")
            self.assertNotEqual(pipeline["terminal_status"], "COMPLETED")

    def test_valid_region_dof_pass_reaches_production_freeze(self) -> None:
        from test_m4_scout_intrinsic_coverage import _intrinsic_density_payload, _rich_freqs  # noqa: WPS433

        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "sample_001_smoke"
            _write_minimal_production_aggregated_run(run_root)
            core = run_root / "lprod" / "resolved_core_config.json"
            _write_core_config(core)
            _write_valid_region_dof_checkpoint(
                run_root / "lprod" / "checkpoint",
                status="PASS",
                mode="best_effort",
            )
            built = run_root / "lprod" / "checkpoint" / "built_metadata.json"
            built_doc = json.loads(built.read_text(encoding="utf-8"))
            built_doc.update(
                {
                    "dataset_version": DATASET_VERSION_REFERENCE,
                    "operator_mesh_matches_generated": True,
                    "generated_mesh_sha256": "abc123",
                    "operator_mesh_file_used": "guitars/sample_001/runs/sample_001_smoke/lprod/mesh/L_prod/sample_001.msh",
                }
            )
            write_json_atomic(built, built_doc)
            discovery = run_root / "scout" / "discovery"
            discovery.mkdir(parents=True, exist_ok=True)
            write_json_atomic(
                discovery / "density_result.json",
                _intrinsic_density_payload(freqs=_rich_freqs()),
            )
            catalog_line = {
                "mic_output_method": "aperture_pressure_rms_proxy_v1",
                "frequency_hz": 100.0,
                "lambda_real": 39578.4,
                "convergence_status": "converged",
            }
            catalog = run_root / "aggregation" / "modes_catalog.jsonl"
            catalog.write_text(json.dumps(catalog_line) + "\n", encoding="utf-8")
            (run_root / "aggregation" / "modes_catalog_deduped.jsonl").write_text(
                catalog.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            (run_root / "aggregation" / "mode_provenance.jsonl").write_text(
                '{"mode_index": 0, "frequency_hz": 100.0}\n',
                encoding="utf-8",
            )
            rc, msg = replay_production_freeze(repo_root=Path(tmp), run_root=run_root, force=False)
            self.assertEqual(rc, 0, msg)
            pipeline = json.loads((run_root / "pipeline_run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(pipeline["terminal_status"], "COMPLETED")
            self.assertEqual(pipeline["stages"]["stage6_freeze"]["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
