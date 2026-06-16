#!/usr/bin/env python3
"""Tests for scout terminal artifact contract and worker_plan preview ownership."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[5]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_m4_reuse_integrity_lib import (  # noqa: E402
    remove_stale_worker_plan_outputs,
    scout_artifact_contract_pass,
    worker_plan_artifact_contract_pass,
)
from v2_b3_m4_scout_planner_lib import render_chunk_preview_md  # noqa: E402
from v2_b3_m4_stage_artifact_contract import (  # noqa: E402
    SCOUT_CHUNK_PREVIEW_JSON_REL,
    SCOUT_TERMINAL_ARTIFACTS,
    assert_scout_terminal_contract_or_raise,
    format_scout_stage_contract_fail_line,
    validate_scout_terminal_artifacts,
    write_worker_chunk_preview_artifacts,
)
from v2_b3_petsc_util import write_json_atomic  # noqa: E402


def _write_scout_terminal_bundle(run_root: Path, *, with_chunk_json: bool = True) -> None:
    (run_root / "scout" / "discovery").mkdir(parents=True, exist_ok=True)
    (run_root / "scout").mkdir(parents=True, exist_ok=True)
    (run_root / "lprod").mkdir(parents=True, exist_ok=True)
    write_json_atomic(run_root / "scout" / "discovery" / "density_result.json", {"status": "PASS"})
    write_json_atomic(run_root / "scout" / "density_zones.json", {"bins": []})
    write_json_atomic(
        run_root / "lprod" / "lprod_target_plan.json",
        {"targets_hz": [100.0, 110.0], "target_windows_hz": [[97, 103], [107, 113]]},
    )
    if with_chunk_json:
        preview = {
            "schema": "m4_worker_chunk_plan_v1",
            "sample_id": "box_sample_000",
            "run_id": "box_sample_000_box_fom_v1",
            "chunks": [
                {
                    "chunk_id": "box_sample_000_chunk_01",
                    "freq_range_hz": [100.0, 110.0],
                    "target_count": 2,
                    "targets_hz": [100.0, 110.0],
                }
            ],
        }
        write_worker_chunk_preview_artifacts(
            lprod_dir=run_root / "lprod",
            chunk_preview=preview,
            md_renderer=render_chunk_preview_md,
        )
    else:
        (run_root / "lprod" / "worker_chunk_plan.preview.md").write_text("# preview\n", encoding="utf-8")
    write_json_atomic(run_root / "scout" / "scout_result.json", {"status": "PASS"})


def test_scout_contract_fails_md_without_json() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        run_root = Path(tmp)
        _write_scout_terminal_bundle(run_root, with_chunk_json=False)
        ok, errors = validate_scout_terminal_artifacts(run_root)
        assert ok is False
        assert any(SCOUT_CHUNK_PREVIEW_JSON_REL in e for e in errors)
        assert scout_artifact_contract_pass(run_root) is False


def test_scout_contract_pass_with_json_and_md() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        run_root = Path(tmp)
        _write_scout_terminal_bundle(run_root, with_chunk_json=True)
        ok, errors = validate_scout_terminal_artifacts(run_root)
        assert ok is True
        assert errors == []
        target_count, chunk_count = assert_scout_terminal_contract_or_raise(run_root)
        assert target_count == 2
        assert chunk_count == 1


def test_remove_stale_worker_plan_preserves_scout_preview_json() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        run_root = Path(tmp)
        _write_scout_terminal_bundle(run_root, with_chunk_json=True)
        preview_json = run_root / SCOUT_CHUNK_PREVIEW_JSON_REL
        preview_md = run_root / "lprod" / "worker_chunk_plan.preview.md"
        assert preview_json.is_file()
        removed = remove_stale_worker_plan_outputs(run_root)
        assert preview_json.is_file()
        assert preview_md.is_file()
        assert SCOUT_CHUNK_PREVIEW_JSON_REL not in removed
        assert worker_plan_artifact_contract_pass(run_root) is False


def test_write_worker_chunk_preview_generates_expected_chunks() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        lprod_dir = Path(tmp) / "lprod"
        lprod_dir.mkdir(parents=True)
        preview = {
            "schema": "m4_worker_chunk_plan_v1",
            "sample_id": "sample_001",
            "run_id": "sample_001_m4prod1",
            "target_count": 2,
            "chunk_count": 1,
            "planned_workers": 3,
            "shape_context": {"shape_name": "classic", "geometry_shape_type": "Classical"},
            "chunks": [{"chunk_id": "sample_001_chunk_01", "freq_range_hz": [100.0, 110.0], "target_count": 2, "targets_hz": [100.0, 110.0]}],
        }
        json_path = write_worker_chunk_preview_artifacts(
            lprod_dir=lprod_dir,
            chunk_preview=preview,
            md_renderer=render_chunk_preview_md,
        )
        body = json.loads(json_path.read_text(encoding="utf-8"))
        assert body["chunks"][0]["chunk_id"] == "sample_001_chunk_01"
        assert (lprod_dir / "worker_chunk_plan.preview.md").is_file()


def test_contract_same_for_all_shapes() -> None:
    for shape in ("classic", "box", "acoustic"):
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp)
            _write_scout_terminal_bundle(run_root, with_chunk_json=True)
            preview = json.loads((run_root / SCOUT_CHUNK_PREVIEW_JSON_REL).read_text(encoding="utf-8"))
            preview["shape_context"] = {
                "shape_name": shape,
                "geometry_shape_type": shape.capitalize() if shape != "classic" else "Classical",
            }
            write_worker_chunk_preview_artifacts(
                lprod_dir=run_root / "lprod",
                chunk_preview=preview,
                md_renderer=render_chunk_preview_md,
            )
            ok, _ = validate_scout_terminal_artifacts(run_root)
            assert ok is True
        assert len(SCOUT_TERMINAL_ARTIFACTS) >= 5


def test_assert_scout_contract_fail_line() -> None:
    line = format_scout_stage_contract_fail_line(["lprod/worker_chunk_plan.preview.json"])
    assert "SCOUT_STAGE_ARTIFACT_CONTRACT_FAIL" in line
    assert "worker_chunk_plan.preview.json" in line


def main() -> int:
    tests = [
        test_scout_contract_fails_md_without_json,
        test_scout_contract_pass_with_json_and_md,
        test_remove_stale_worker_plan_preserves_scout_preview_json,
        test_write_worker_chunk_preview_generates_expected_chunks,
        test_contract_same_for_all_shapes,
        test_assert_scout_contract_fail_line,
    ]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}", flush=True)
    print(f"[B3_stage_artifact_contract] PASS all {len(tests)} tests", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
