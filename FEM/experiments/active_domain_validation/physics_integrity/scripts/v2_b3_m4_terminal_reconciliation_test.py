#!/usr/bin/env python3
"""Tests for failure_retention reconciliation and compaction deferral."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_m4_freeze_first_e2e_run import (  # noqa: E402
    AGG_STATUS_PASS,
    CHECKPOINT_TERMINAL_READY,
    SCOUT_TERMINAL_READY,
    TERMINAL_E2E,
)
from v2_b3_m4_terminal_status_lib import (  # noqa: E402
    FAILURE_RETENTION_ARCHIVE_REL,
    FAILURE_RETENTION_REL,
    TERMINAL_STATUS_INCONSISTENCY,
    check_terminal_status_consistency,
    promote_after_aggregation_pass,
)
from v2_b3_petsc_util import write_json_atomic  # noqa: E402


def _write_aggregate_pass(run_root: Path) -> None:
    agg = run_root / "aggregation"
    agg.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        agg / "aggregation_result.json",
        {
            "status": AGG_STATUS_PASS,
            "final_aggregation_ready": True,
            "planned_chunk_count": 12,
            "completed_chunk_count": 12,
        },
    )
    (agg / "modes_catalog.jsonl").write_text('{"mode_index":0,"frequency_hz":200.0}\n', encoding="utf-8")


def test_stale_scout_retention_cleared_after_aggregation_promote() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        run_root = Path(tmp)
        write_json_atomic(
            run_root / "pipeline_run_manifest.json",
            {"terminal_status": CHECKPOINT_TERMINAL_READY, "stages": {}},
        )
        (run_root / "cleanup").mkdir(parents=True)
        write_json_atomic(
            run_root / FAILURE_RETENTION_REL,
            {
                "schema": "m4_sample_failure_retention_v1",
                "terminal_status": SCOUT_TERMINAL_READY,
                "outcome": "fail",
            },
        )
        _write_aggregate_pass(run_root)
        promote = promote_after_aggregation_pass(run_root)
        assert promote.get("failure_retention", {}).get("action") == "cleared"
        assert not (run_root / FAILURE_RETENTION_REL).is_file()
        assert (run_root / FAILURE_RETENTION_ARCHIVE_REL).is_file()
        ok, errors = check_terminal_status_consistency(run_root)
        assert ok is True
        assert errors == []
        manifest = json.loads((run_root / "pipeline_run_manifest.json").read_text(encoding="utf-8"))
        assert manifest["terminal_status"] == TERMINAL_E2E


def test_true_contradiction_still_blocks_consistency() -> None:
    """Retention ahead of manifest is a real contradiction, not stale scout diagnostics."""
    with tempfile.TemporaryDirectory() as tmp:
        run_root = Path(tmp)
        write_json_atomic(
            run_root / "pipeline_run_manifest.json",
            {"terminal_status": SCOUT_TERMINAL_READY},
        )
        (run_root / "cleanup").mkdir()
        write_json_atomic(
            run_root / FAILURE_RETENTION_REL,
            {"terminal_status": TERMINAL_E2E, "outcome": "fail"},
        )
        ok, errors = check_terminal_status_consistency(run_root)
        assert ok is False
        assert any(TERMINAL_STATUS_INCONSISTENCY in e for e in errors)
        assert (run_root / FAILURE_RETENTION_REL).is_file()


def test_compaction_deferred_when_freeze_pending() -> None:
    from v2_b3_m4_lhs_production_batch import _run_sample_compaction_for_batch  # noqa: WPS433

    with tempfile.TemporaryDirectory() as tmp:
        repo_root = Path(tmp)
        run_root = repo_root / "run"
        run_root.mkdir()
        row = {
            "sample_id": "box_sample_000",
            "run_id": "box_sample_000_box_fom_v1",
            "run_root_abs": str(run_root),
            "outcome": "pass_freeze_warning",
            "terminal_status": TERMINAL_E2E,
            "aggregation_status": "AGGREGATION_PASS",
            "final_aggregation_ready": True,
        }
        ok = _run_sample_compaction_for_batch(
            row=row,
            repo_root=repo_root,
            pool={"entries": [{"id": "box_sample_000"}]},
            compact_after_sample=True,
            compact_keep_full_samples=set(),
            compact_blocking=True,
        )
        assert ok is True
        assert row["compaction"]["status"] == "deferred"
        assert "freeze_pending" in row["compaction"]["skip_reason"]


def main() -> int:
    tests = [
        test_stale_scout_retention_cleared_after_aggregation_promote,
        test_true_contradiction_still_blocks_consistency,
        test_compaction_deferred_when_freeze_pending,
    ]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}", flush=True)
    print(f"[B3_terminal_reconciliation] PASS all {len(tests)} tests", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
