#!/usr/bin/env python3
"""Tests for LHS pool truncation guard and in-place status updates."""
from __future__ import annotations

import json
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[4]
FEM_SCRIPTS = SCRIPT_DIR.parents[3] / "scripts"
TOOLS = REPO_ROOT / "tools"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(FEM_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(FEM_SCRIPTS))
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from m4_shape_registry import resolve_shape_config  # noqa: E402
from v2_b3_m4_lhs_pool_bridge import (  # noqa: E402
    LhsPoolTruncationError,
    lhs_pool_entry_patch_from_run,
    load_lhs_pool,
    select_lhs_samples,
    sync_lhs_pool_entry,
    write_lhs_pool,
)
from generate_shape_lhs_pool import build_pool_document, resolve_target_pool_size, write_pool  # noqa: E402


def _fake_pool(n: int, *, shape_key: str = "box") -> dict:
    cfg = resolve_shape_config(shape_key)
    entries = []
    for i in range(n):
        sid = cfg.sample_id(i)
        entries.append(
            {
                "id": sid,
                "status": "PENDING",
                "parameters": {"top_wood_id": 1, "back_wood_id": 1},
            }
        )
    return {
        "shape_name": shape_key,
        "total_samples": n,
        "entries": entries,
    }


def test_update_one_status_preserves_500_entries():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "lhs_pool.json"
        pool = _fake_pool(500)
        write_lhs_pool(path, pool)
        patch = lhs_pool_entry_patch_from_run(
            run_id="box_sample_002_box_fom_v1",
            run_dir="/tmp/run",
            batch_id="batch",
            outcome="pass",
            summary={"aggregation_status": "AGGREGATION_PASS", "deduped_modes": 9},
            elapsed_s=1.0,
            started_at="2026-01-01T00:00:00Z",
        )
        sync_lhs_pool_entry(pool, sample_id="box_sample_002", patch=patch)
        write_lhs_pool(path, pool)
        loaded = load_lhs_pool(path)
        assert len(loaded["entries"]) == 500
        entry = next(e for e in loaded["entries"] if e["id"] == "box_sample_002")
        assert entry["status"] == "COMPLETED"
        assert entry["last_run_id"] == "box_sample_002_box_fom_v1"


def test_update_sample_002_does_not_remove_others():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "lhs_pool.json"
        pool = _fake_pool(500)
        write_lhs_pool(path, pool)
        sync_lhs_pool_entry(
            pool,
            sample_id="box_sample_002",
            patch={"status": "RUNNING", "last_run_id": "box_sample_002_box_fom_v1"},
        )
        write_lhs_pool(path, pool)
        loaded = load_lhs_pool(path)
        ids = {e["id"] for e in loaded["entries"]}
        assert "box_sample_000" in ids
        assert "box_sample_001" in ids
        assert "box_sample_002" in ids
        assert len(ids) == 500


def test_box_update_does_not_touch_classic_pool():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        classic_path = root / "ROM" / "classic" / "lhs_pool.json"
        box_path = root / "ROM" / "box" / "lhs_pool.json"
        classic_pool = _fake_pool(67, shape_key="classic")
        box_pool = _fake_pool(500, shape_key="box")
        classic_path.parent.mkdir(parents=True, exist_ok=True)
        box_path.parent.mkdir(parents=True, exist_ok=True)
        write_lhs_pool(classic_path, classic_pool)
        write_lhs_pool(box_path, box_pool)
        before = classic_path.read_text(encoding="utf-8")
        sync_lhs_pool_entry(
            box_pool,
            sample_id="box_sample_002",
            patch={"status": "COMPLETED", "last_run_id": "box_sample_002_box_fom_v1"},
        )
        write_lhs_pool(box_path, box_pool)
        after = classic_path.read_text(encoding="utf-8")
        assert before == after
        assert len(load_lhs_pool(box_path)["entries"]) == 500


def test_truncation_raises_guard():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "lhs_pool.json"
        pool = _fake_pool(500)
        write_lhs_pool(path, pool)
        truncated = deepcopy(pool)
        truncated["entries"] = truncated["entries"][:1]
        try:
            write_lhs_pool(path, truncated)
            raise AssertionError("expected LhsPoolTruncationError")
        except LhsPoolTruncationError as exc:
            assert "LHS_POOL_TRUNCATION_GUARD_FAIL original_entries=500 new_entries=1" in str(exc)


def test_select_lhs_samples_start_2_count_1():
    pool = _fake_pool(500)
    status_doc = {"samples": {}}
    selected, _ = select_lhs_samples(
        pool,
        status_doc,
        max_samples=1,
        start_index=2,
        skip_completed=False,
        run_id_suffix="box_fom_v1",
    )
    assert len(selected) == 1
    assert selected[0]["sample_id"] == "box_sample_002"
    assert selected[0]["lhs_row_index"] == 2


def test_build_pool_document_never_shrinks_existing():
    existing = _fake_pool(500)
    doc = build_pool_document(
        shape_key="box",
        count=1,
        seed=1,
        existing=existing,
        force=False,
    )
    assert len(doc["entries"]) == 500


def test_resolve_target_pool_size_with_batch_count_one():
    existing = _fake_pool(500)
    size = resolve_target_pool_size(shape_key="box", count=1, existing=existing, force=False)
    assert size == 500


def test_write_creates_backup():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "lhs_pool.json"
        pool = _fake_pool(5)
        write_lhs_pool(path, pool)
        sync_lhs_pool_entry(pool, sample_id="box_sample_000", patch={"status": "RUNNING"})
        write_lhs_pool(path, pool)
        backups = list(path.parent.glob("lhs_pool.json.bak.*"))
        assert backups, "expected timestamped .bak file"


def main() -> int:
    tests = [
        test_update_one_status_preserves_500_entries,
        test_update_sample_002_does_not_remove_others,
        test_box_update_does_not_touch_classic_pool,
        test_truncation_raises_guard,
        test_select_lhs_samples_start_2_count_1,
        test_build_pool_document_never_shrinks_existing,
        test_resolve_target_pool_size_with_batch_count_one,
        test_write_creates_backup,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
