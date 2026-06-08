#!/usr/bin/env python3
"""Regression tests for stale RUNNING terminal repair."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_m4_freeze_first_e2e_run import CHECKPOINT_TERMINAL_READY  # noqa: E402
from v2_b3_m4_run_status_repair import (  # noqa: E402
    STALE_RUNNING_REPAIR_REASON,
    assess_stale_running_repair,
    promote_checkpoint_ready_terminal,
)
from v2_b3_petsc_util import write_json_atomic  # noqa: E402


def _write_minimal_checkpoint_ready_run(run_root: Path) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        run_root / "pipeline_run_manifest.json",
        {
            "terminal_status": "RUNNING",
            "stages": {"stage3_zones_plan": {"status": "PASS"}},
        },
    )
    write_json_atomic(run_root / "scout" / "density_zones.json", {"zones": []})
    write_json_atomic(run_root / "lprod" / "lprod_target_plan.json", {"targets_hz": [100.0]})
    write_json_atomic(run_root / "lprod" / "worker_chunk_plan.preview.json", {"chunks": [{"chunk_id": "sample_001_chunk_01"}]})
    write_json_atomic(run_root / "lprod" / "worker_commands.json", {"commands": []})
    chunk_dir = run_root / "worker_results" / "sample_001_chunk_01"
    chunk_dir.mkdir(parents=True)
    write_json_atomic(chunk_dir / "chunk_targets.json", {"schema": "m4_worker_chunk_targets_v1", "chunk_id": "sample_001_chunk_01", "targets": [{"target_hz": 100.0, "window_hz": [96.0, 104.0]}]})

    ckpt = run_root / "lprod" / "checkpoint"
    ckpt.mkdir(parents=True)
    write_json_atomic(
        ckpt / "checkpoint_export_manifest.json",
        {
            "status": "PASS",
            "export_pass": True,
            "matrix_verify_pass": True,
            "core_config_mode": "override",
            "mesh_level": "L_prod",
            "active_dimension": 100,
            "core_config_path": "lprod/resolved_core_config.json",
        },
    )
    write_json_atomic(
        ckpt / "built_metadata.json",
        {
            "mesh_level": "L_prod",
            "active_dimension": 100,
            "n_w": 200,
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
        {
            "aperture_selection_method": "facet_adjacent_air_cell_dofs_v1",
            "p_idx_aperture_count": 4,
        },
    )


def test_assess_and_repair_stale_running() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        run_root = Path(tmp)
        _write_minimal_checkpoint_ready_run(run_root)
        assessment = assess_stale_running_repair(run_root)
        assert assessment["eligible"] is True
        result = promote_checkpoint_ready_terminal(
            run_root,
            repair_reason=STALE_RUNNING_REPAIR_REASON,
        )
        assert result["status"] == "PASS"
        manifest = json.loads((run_root / "pipeline_run_manifest.json").read_text(encoding="utf-8"))
        assert manifest["terminal_status"] == CHECKPOINT_TERMINAL_READY
        assert manifest["stale_running_repair"]["repair_reason"] == STALE_RUNNING_REPAIR_REASON
        repair = json.loads((run_root / "stale_running_repair.json").read_text(encoding="utf-8"))
        assert repair["previous_status"] == "RUNNING"
        assert repair["repaired_status"] == CHECKPOINT_TERMINAL_READY


def test_skip_when_not_running() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        run_root = Path(tmp)
        _write_minimal_checkpoint_ready_run(run_root)
        manifest_path = run_root / "pipeline_run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["terminal_status"] = CHECKPOINT_TERMINAL_READY
        write_json_atomic(manifest_path, manifest)
        assessment = assess_stale_running_repair(run_root)
        assert assessment["eligible"] is False


def main() -> int:
    tests = [test_assess_and_repair_stale_running, test_skip_when_not_running]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}", flush=True)
    print(f"[B3_run_status_repair] PASS all {len(tests)} tests", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
