#!/usr/bin/env python3
"""Regression tests for geometry-corrected dataset reset planner."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_m4_geometry_corrected_dataset_reset import (  # noqa: E402
    DEFAULT_SMOKE_RUN_ID,
    apply_lhs_pool_reset,
    build_reset_plan,
    classify_run_dir,
    discover_rom_quarantine_files,
    execute_reset_plan,
    verify_reset_state,
)
from v2_b3_m4_lhs_pool_bridge import LHS_PENDING, load_lhs_pool  # noqa: E402
from v2_b3_m4_production_contracts import DATASET_VERSION  # noqa: E402
from v2_b3_m4_production_freeze import TERMINAL_PRODUCTION_COMPLETED  # noqa: E402
from v2_b3_petsc_util import write_json_atomic  # noqa: E402


def _write_run(
    repo: Path,
    *,
    sample_id: str,
    run_id: str,
    terminal_status: str,
    dataset_version: str,
) -> Path:
    run_root = (
        repo
        / "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars"
        / sample_id
        / "runs"
        / run_id
    )
    run_root.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        run_root / "pipeline_run_manifest.json",
        {"terminal_status": terminal_status, "dataset_version": dataset_version},
    )
    write_json_atomic(
        run_root / "freeze" / "freeze_manifest.json",
        {"terminal_status": terminal_status, "dataset_version": dataset_version, "production_acceptance_pass": True},
    )
    write_json_atomic(
        run_root / "lprod" / "checkpoint" / "built_metadata.json",
        {"dataset_version": dataset_version},
    )
    write_json_atomic(
        run_root / "aggregation" / "aggregation_result.json",
        {"status": "AGGREGATION_PASS", "final_aggregation_ready": True},
    )
    (run_root / "lprod" / "scout_marker").mkdir(parents=True, exist_ok=True)
    return run_root


def _write_lhs_pool(repo: Path) -> None:
    entries = []
    for i in range(37):
        entries.append(
            {
                "id": f"sample_{i:03d}",
                "parameters": {"geometry.length": 0.48},
                "status": "COMPLETED",
                "last_run_id": f"sample_{i:03d}_m4prod1",
                "last_run_dir": "stale",
                "error": "old",
            }
        )
    write_json_atomic(
        repo / "ROM/classic/lhs_pool.json",
        {"shape_name": "classic", "entries": entries},
    )


def test_plan_preserves_smoke_and_deletes_m4prod1() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        _write_lhs_pool(repo)
        smoke = _write_run(
            repo,
            sample_id="sample_001",
            run_id=DEFAULT_SMOKE_RUN_ID,
            terminal_status=TERMINAL_PRODUCTION_COMPLETED,
            dataset_version=DATASET_VERSION,
        )
        invalid = _write_run(
            repo,
            sample_id="sample_001",
            run_id="sample_001_m4prod1",
            terminal_status="LPROD_WORKERS_AND_AGGREGATION_PASS",
            dataset_version="legacy_v0",
        )
        (invalid / "heavy.bin").write_bytes(b"x" * 1024)

        classic = repo / "ROM/classic"
        (classic / "m4_modal_surrogate.json").write_text("{}", encoding="utf-8")

        plan = build_reset_plan(repo_root=repo)
        delete_ids = {row["run_id"] for row in plan["runs_to_delete"]}
        preserve_ids = {row["run_id"] for row in plan["runs_preserved"]}
        assert "sample_001_m4prod1" in delete_ids
        assert DEFAULT_SMOKE_RUN_ID in preserve_ids
        assert plan["estimated_run_delete_bytes"] >= 1024
        assert any("m4_modal_surrogate.json" in row["source"] for row in plan["rom_files_to_quarantine"])
        assert classify_run_dir(smoke, smoke_run_id=DEFAULT_SMOKE_RUN_ID) == "preserve_smoke"
        assert classify_run_dir(invalid, smoke_run_id=DEFAULT_SMOKE_RUN_ID) == "delete"


def test_execute_and_verify() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        _write_lhs_pool(repo)
        _write_run(
            repo,
            sample_id="sample_001",
            run_id=DEFAULT_SMOKE_RUN_ID,
            terminal_status=TERMINAL_PRODUCTION_COMPLETED,
            dataset_version=DATASET_VERSION,
        )
        bad = _write_run(
            repo,
            sample_id="sample_005",
            run_id="sample_005_m4prod1",
            terminal_status="LPROD_WORKERS_AND_AGGREGATION_PASS",
            dataset_version="legacy_v0",
        )
        classic = repo / "ROM/classic"
        (classic / "m4_modal_surrogate.json").write_text("{}", encoding="utf-8")
        (classic / "m4_modal_surrogate.npz").write_bytes(b"npz")

        index_dir = repo / "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/index"
        index_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(
            index_dir / "lhs_pool_status.json",
            {
                "samples": {
                    "sample_001": {"run_id": DEFAULT_SMOKE_RUN_ID, "status": "pass"},
                    "sample_005": {"run_id": "sample_005_m4prod1", "status": "pass"},
                }
            },
        )
        (index_dir / "lhs_production_runs_index.jsonl").write_text(
            json.dumps({"sample_id": "sample_001", "run_id": DEFAULT_SMOKE_RUN_ID}) + "\n"
            + json.dumps({"sample_id": "sample_005", "run_id": "sample_005_m4prod1"}) + "\n",
            encoding="utf-8",
        )

        plan = build_reset_plan(repo_root=repo)
        report = execute_reset_plan(repo_root=repo, plan=plan)
        assert report["bytes_deleted"] > 0
        assert not bad.exists()

        pool = load_lhs_pool(repo / "ROM/classic/lhs_pool.json")
        assert pool["dataset_version"] == DATASET_VERSION
        for i in range(37):
            entry = pool["entries"][i]
            assert str(entry["id"]) == f"sample_{i:03d}"
            assert str(entry["status"]).upper() == LHS_PENDING
            assert "last_run_id" not in entry

        verify = verify_reset_state(repo_root=repo)
        assert verify["verify_pass"] is True


def test_apply_lhs_pool_reset_preserves_parameters() -> None:
    pool = {
        "entries": [
            {
                "id": "sample_001",
                "parameters": {"geometry.length": 0.48, "top_wood_id": "spruce"},
                "status": "COMPLETED",
                "last_run_id": "sample_001_m4prod1",
            }
        ]
    }
    out = apply_lhs_pool_reset(pool)
    entry = out["entries"][0]
    assert entry["parameters"]["geometry.length"] == 0.48
    assert entry["status"] == LHS_PENDING
    assert "last_run_id" not in entry


def main() -> int:
    tests = [
        test_plan_preserves_smoke_and_deletes_m4prod1,
        test_execute_and_verify,
        test_apply_lhs_pool_reset_preserves_parameters,
    ]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"ALL {len(tests)} TESTS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
