#!/usr/bin/env python3
"""Regression tests for strict M4 stage reuse integrity."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_m4_freeze_first_e2e_run import (  # noqa: E402
    AGG_STATUS_PASS,
    CHECKPOINT_TERMINAL_READY,
    SCOUT_TERMINAL_READY,
    TERMINAL_E2E,
)
from v2_b3_m4_production_freeze import TERMINAL_PRODUCTION_COMPLETED  # noqa: E402
from v2_b3_m4_reuse_integrity_lib import (  # noqa: E402
    REUSE_INTEGRITY_FAIL,
    assess_stages_with_integrity,
    repair_inconsistent_reuse_state,
    worker_plan_artifact_contract_pass,
)
from v2_b3_petsc_util import write_json_atomic  # noqa: E402


def _write_chunk_targets(run_root: Path, chunk_id: str = "sample_000_chunk_01") -> None:
    chunk_dir = run_root / "worker_results" / chunk_id
    chunk_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        chunk_dir / "chunk_targets.json",
        {
            "schema": "m4_worker_chunk_targets_v1",
            "chunk_id": chunk_id,
            "targets": [{"target_hz": 200.0, "window_hz": [196.0, 204.0]}],
        },
    )


def _write_worker_plan_contract(run_root: Path) -> None:
    write_json_atomic(
        run_root / "lprod" / "worker_chunk_plan.preview.json",
        {"chunks": [{"chunk_id": "sample_000_chunk_01"}]},
    )
    write_json_atomic(run_root / "lprod" / "worker_commands.json", {"commands": []})
    write_json_atomic(run_root / "lprod" / "lprod_execution_plan.json", {"schema": "m4_lprod_execution_plan_v1"})
    write_json_atomic(
        run_root / "lprod" / "aggregation_plan.json",
        {"schema": "m4_aggregation_plan_v1", "chunk_count": 1},
    )
    write_json_atomic(
        run_root / "lprod" / "lprod_mesh_checkpoint_readiness.json",
        {"ready": True},
    )
    _write_chunk_targets(run_root)


def _write_minimal_checkpoint(run_root: Path) -> None:
    ckpt = run_root / "lprod" / "checkpoint"
    ckpt.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        ckpt / "checkpoint_export_manifest.json",
        {
            "status": "PASS",
            "export_pass": True,
            "matrix_verify_pass": True,
            "mesh_level": "L_prod",
            "active_dimension": 100,
        },
    )
    write_json_atomic(
        ckpt / "built_metadata.json",
        {
            "mesh_level": "L_prod",
            "active_dimension": 100,
            "p_idx_aperture_count": 4,
        },
    )
    np.savez_compressed(ckpt / "A_active_csr.npz", data=np.array([1], dtype=np.float64))
    np.savez_compressed(ckpt / "M_active_csr.npz", data=np.array([1], dtype=np.float64))
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
        {"aperture_selection_method": "facet_adjacent_air_cell_dofs_v1", "p_idx_aperture_count": 4},
    )


def _write_manifest(run_root: Path, *, terminal_status: str, stage5: str = "PLANNED_READY") -> None:
    write_json_atomic(
        run_root / "pipeline_run_manifest.json",
        {
            "sample_id": "box_sample_000",
            "run_id": "box_sample_000_box_fom_v1",
            "terminal_status": terminal_status,
            "stages": {
                "stage3_zones_plan": {"status": "PASS"},
                "stage4_lprod_mesh": {"status": "PASS"},
                "stage4_lprod_export": {"status": "PASS"},
                "stage5_workers": {"status": stage5},
                "stage6_aggregate": {"status": "PLANNED_READY"},
            },
        },
    )


def _write_worker_pass(run_root: Path, chunk_id: str = "sample_000_chunk_01") -> None:
    chunk_dir = run_root / "worker_results" / chunk_id
    chunk_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        chunk_dir / "worker_result.json",
        {
            "status": "PASS",
            "mode": "m4_worker_real",
            "minibatch_executed": True,
        },
    )


def _write_aggregate_pass(run_root: Path) -> None:
    agg = run_root / "aggregation"
    agg.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        agg / "aggregation_result.json",
        {"status": AGG_STATUS_PASS, "final_aggregation_ready": True},
    )
    (agg / "modes_catalog.jsonl").write_text('{"mode_index":0,"frequency_hz":200.0}\n', encoding="utf-8")


def test_checkpoint_ready_with_worker_results_not_workers_pass() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        run_root = Path(tmp)
        _write_worker_plan_contract(run_root)
        _write_manifest(run_root, terminal_status=CHECKPOINT_TERMINAL_READY)
        _write_worker_pass(run_root)

        stages = assess_stages_with_integrity(run_root, production_mode=True)
        assert stages["workers"]["pass"] is False
        assert stages["workers"]["reuse_status"] == REUSE_INTEGRITY_FAIL
        assert "terminal_status_incompatible" in stages["workers"]["integrity_error"]
        assert CHECKPOINT_TERMINAL_READY in stages["workers"]["integrity_error"]


def test_checkpoint_ready_with_aggregation_not_aggregate_pass() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        run_root = Path(tmp)
        _write_worker_plan_contract(run_root)
        _write_manifest(run_root, terminal_status=CHECKPOINT_TERMINAL_READY)
        _write_aggregate_pass(run_root)

        stages = assess_stages_with_integrity(run_root, production_mode=True)
        assert stages["aggregate"]["pass"] is False
        assert stages["aggregate"]["reuse_status"] == REUSE_INTEGRITY_FAIL


def test_stale_execution_plan_alone_not_worker_plan_pass() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        run_root = Path(tmp)
        write_json_atomic(run_root / "lprod" / "lprod_execution_plan.json", {"schema": "x"})
        assert worker_plan_artifact_contract_pass(run_root) is False
        stages = assess_stages_with_integrity(run_root, production_mode=True)
        assert stages["worker_plan"]["pass"] is False


def test_completed_classic_state_reuses() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        run_root = Path(tmp)
        write_json_atomic(run_root / "scout" / "density_zones.json", {"zones": []})
        write_json_atomic(run_root / "lprod" / "lprod_target_plan.json", {"targets_hz": [200.0]})
        _write_worker_plan_contract(run_root)
        _write_minimal_checkpoint(run_root)
        _write_worker_pass(run_root)
        _write_aggregate_pass(run_root)
        write_json_atomic(
            run_root / "freeze" / "freeze_manifest.json",
            {"production_acceptance_pass": True},
        )
        _write_manifest(
            run_root,
            terminal_status=TERMINAL_PRODUCTION_COMPLETED,
            stage5="PASS",
        )
        manifest_path = run_root / "pipeline_run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["stages"]["stage6_aggregate"] = {"status": "PASS"}
        write_json_atomic(manifest_path, manifest)

        stages = assess_stages_with_integrity(run_root, production_mode=True)
        assert stages["scout"]["pass"] is True
        assert stages["worker_plan"]["pass"] is True
        assert stages["checkpoint"]["pass"] is True
        assert stages["workers"]["pass"] is True
        assert stages["aggregate"]["pass"] is True
        assert stages["freeze"]["pass"] is True


def test_partial_failed_state_does_not_advance_to_freeze() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        run_root = Path(tmp)
        _write_worker_plan_contract(run_root)
        _write_manifest(run_root, terminal_status=CHECKPOINT_TERMINAL_READY)
        _write_worker_pass(run_root)
        _write_aggregate_pass(run_root)

        stages = assess_stages_with_integrity(run_root, production_mode=True)
        assert stages["aggregate"]["pass"] is False
        assert stages["freeze"]["pass"] is False


def test_repair_quarantines_stale_downstream() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        run_root = Path(tmp)
        _write_worker_plan_contract(run_root)
        _write_manifest(run_root, terminal_status=CHECKPOINT_TERMINAL_READY)
        _write_worker_pass(run_root)
        _write_aggregate_pass(run_root)

        repair = repair_inconsistent_reuse_state(run_root, production_mode=True)
        assert repair.get("repaired") is True

        stages = assess_stages_with_integrity(run_root, production_mode=True)
        assert stages["workers"]["pass"] is False
        assert stages["aggregate"]["pass"] is False
        assert not (run_root / "aggregation" / "aggregation_result.json").is_file()


def test_diagnostics_retained_not_reused_as_pass() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        run_root = Path(tmp)
        _write_worker_plan_contract(run_root)
        _write_manifest(run_root, terminal_status=CHECKPOINT_TERMINAL_READY)
        diag = run_root / "logs" / "sample_failure_diagnostic.log"
        diag.parent.mkdir(parents=True, exist_ok=True)
        diag.write_text("freeze failed rc=2\n", encoding="utf-8")
        _write_aggregate_pass(run_root)

        repair = repair_inconsistent_reuse_state(run_root, production_mode=True)
        assert repair.get("repaired") is True
        assert diag.is_file()
        stages = assess_stages_with_integrity(run_root, production_mode=True)
        assert stages["aggregate"]["pass"] is False


def test_e2e_terminal_allows_workers_aggregate_reuse() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        run_root = Path(tmp)
        _write_worker_plan_contract(run_root)
        _write_worker_pass(run_root)
        _write_aggregate_pass(run_root)
        _write_manifest(run_root, terminal_status=TERMINAL_E2E, stage5="PASS")
        manifest_path = run_root / "pipeline_run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["stages"]["stage6_aggregate"] = {"status": "PASS"}
        write_json_atomic(manifest_path, manifest)

        stages = assess_stages_with_integrity(run_root, production_mode=False)
        assert stages["workers"]["pass"] is True
        assert stages["aggregate"]["pass"] is True


def main() -> int:
    tests = [
        test_checkpoint_ready_with_worker_results_not_workers_pass,
        test_checkpoint_ready_with_aggregation_not_aggregate_pass,
        test_stale_execution_plan_alone_not_worker_plan_pass,
        test_completed_classic_state_reuses,
        test_partial_failed_state_does_not_advance_to_freeze,
        test_repair_quarantines_stale_downstream,
        test_diagnostics_retained_not_reused_as_pass,
        test_e2e_terminal_allows_workers_aggregate_reuse,
    ]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}", flush=True)
    print(f"[B3_reuse_integrity] PASS all {len(tests)} tests", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
