#!/usr/bin/env python3
"""Regression tests for production freeze finalization."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_m4_freeze_first_e2e_run import AGG_STATUS_PASS, TERMINAL_E2E  # noqa: E402
from v2_b3_m4_production_contracts import DATASET_VERSION, PRODUCTION_MIC_METHOD  # noqa: E402
from v2_b3_m4_production_freeze import (  # noqa: E402
    PRODUCTION_FREEZE_MANIFEST,
    TERMINAL_PRODUCTION_COMPLETED,
    assess_production_completion,
    production_freeze_complete,
    replay_production_freeze,
)
from v2_b3_petsc_util import write_json_atomic  # noqa: E402


def _write_minimal_production_aggregated_run(run_root: Path) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    sample_id = "sample_001"
    chunk_id = f"{sample_id}_chunk_01"
    write_json_atomic(
        run_root / "pipeline_run_manifest.json",
        {
            "sample_id": sample_id,
            "run_id": run_root.name,
            "terminal_status": TERMINAL_E2E,
            "stages": {
                "stage5_workers": {"status": "PASS"},
                "stage6_aggregate": {"status": "PASS", "aggregation_status": AGG_STATUS_PASS},
                "stage6_freeze": {"status": "PASS_WITH_WARNING"},
            },
        },
    )
    write_json_atomic(
        run_root / "sample" / "sample_input.json",
        {
            "sample_id": sample_id,
            "geometry": {
                "length": 0.48,
                "width": 0.325,
                "depth": 0.1,
                "hole_radius": 0.047,
                "top_thickness": 0.003,
                "back_thickness": 0.003,
            },
        },
    )
    write_json_atomic(run_root / "lprod" / "lprod_target_plan.json", {"targets_hz": [100.0]})
    write_json_atomic(
        run_root / "lprod" / "worker_chunk_plan.preview.json",
        {"chunks": [{"chunk_id": chunk_id}]},
    )
    write_json_atomic(
        run_root / "lprod" / "resolved_core_config.json",
        {
            "geometry_numeric_parameters": {
                "length": 0.48,
                "width": 0.325,
                "depth": 0.1,
                "hole_radius": 0.047,
                "top_thickness": 0.003,
                "back_thickness": 0.003,
            },
            "m4_run_metadata": {"shape_name": "classic", "dataset_version": DATASET_VERSION},
        },
    )
    ckpt = run_root / "lprod" / "checkpoint"
    ckpt.mkdir(parents=True)
    write_json_atomic(
        ckpt / "checkpoint_export_manifest.json",
        {"status": "PASS", "export_pass": True},
    )
    write_json_atomic(
        ckpt / "built_metadata.json",
        {
            "mesh_level": "L_prod",
            "active_dimension": 100,
            "operator_mesh_matches_generated": True,
            "generated_mesh_sha256": "abc123",
            "dataset_version": DATASET_VERSION,
            "p_idx_aperture_count": 4,
        },
    )
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
    chunk_dir = run_root / "worker_results" / chunk_id
    chunk_dir.mkdir(parents=True)
    write_json_atomic(chunk_dir / "worker_result.json", {"status": "PASS", "unique_modes": [{"f_hz": 100.0}]})
    write_json_atomic(
        run_root / "aggregation" / "aggregation_result.json",
        {
            "status": AGG_STATUS_PASS,
            "final_aggregation_ready": True,
            "sample_id": sample_id,
            "run_id": run_root.name,
            "planned_chunk_count": 1,
            "completed_chunk_count": 1,
            "missing_chunk_count": 0,
            "failed_chunk_count": 0,
            "deduped_mode_count": 1,
        },
    )
    catalog = run_root / "aggregation" / "modes_catalog.jsonl"
    catalog.parent.mkdir(parents=True, exist_ok=True)
    catalog.write_text(
        json.dumps({"mic_output_method": PRODUCTION_MIC_METHOD, "frequency_hz": 100.0}) + "\n",
        encoding="utf-8",
    )


def test_acceptance_accepts_aggregation_pass() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        run_root = Path(tmp)
        _write_minimal_production_aggregated_run(run_root)
        from v2_b3_m4_production_contracts import evaluate_production_acceptance  # noqa: WPS433

        acceptance = evaluate_production_acceptance(
            run_root=run_root,
            sample_input=json.loads((run_root / "sample" / "sample_input.json").read_text(encoding="utf-8")),
        )
        assert acceptance["acceptance_pass"] is True
        assert acceptance["failures"] == []


def test_replay_writes_freeze_manifest_and_completed_terminal() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        run_root = Path(tmp) / "sample_001_smoke"
        _write_minimal_production_aggregated_run(run_root)
        repo_root = Path(tmp)
        before = assess_production_completion(run_root)
        assert before["complete"] is False
        assert any("freeze/freeze_manifest.json" in f for f in before["failures"])

        rc, msg = replay_production_freeze(repo_root=repo_root, run_root=run_root, force=False)
        assert rc == 0, msg
        assert production_freeze_complete(run_root)

        freeze_doc = json.loads(
            (run_root / "freeze" / PRODUCTION_FREEZE_MANIFEST).read_text(encoding="utf-8")
        )
        assert freeze_doc["production_acceptance_pass"] is True
        assert freeze_doc["production_acceptance_failures"] == []
        assert freeze_doc["mic_output_method"] == PRODUCTION_MIC_METHOD
        assert freeze_doc["terminal_status"] == TERMINAL_PRODUCTION_COMPLETED

        pipeline = json.loads((run_root / "pipeline_run_manifest.json").read_text(encoding="utf-8"))
        assert pipeline["terminal_status"] == TERMINAL_PRODUCTION_COMPLETED
        assert pipeline["production_acceptance_pass"] is True
        assert pipeline["stages"]["stage6_freeze"]["status"] == "PASS"


def test_replay_idempotent() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        run_root = Path(tmp) / "sample_001_smoke"
        _write_minimal_production_aggregated_run(run_root)
        repo_root = Path(tmp)
        rc1, _ = replay_production_freeze(repo_root=repo_root, run_root=run_root)
        rc2, msg2 = replay_production_freeze(repo_root=repo_root, run_root=run_root)
        assert rc1 == 0
        assert rc2 == 0
        assert "already complete" in msg2


def main() -> int:
    tests = [
        test_acceptance_accepts_aggregation_pass,
        test_replay_writes_freeze_manifest_and_completed_terminal,
        test_replay_idempotent,
    ]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"ALL {len(tests)} TESTS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
