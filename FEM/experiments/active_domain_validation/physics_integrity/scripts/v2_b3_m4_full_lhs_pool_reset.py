#!/usr/bin/env python3
"""Full LHS pool bookkeeping reset — all entries to PENDING (dry-run default)."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_m4_lhs_pool_bridge import (  # noqa: E402
    LHS_PENDING,
    STATUS_SCHEMA,
    lhs_pool_status_path,
    lhs_runs_index_path,
    load_lhs_pool,
    load_lhs_pool_status,
    normalize_lhs_entry_status,
    write_lhs_pool_status,
    write_lhs_pool_with_backup,
)
from v2_b3_m4_worker_run_lib import detect_repo_root, rel, utc_now  # noqa: E402
from v2_b3_petsc_util import write_json_atomic  # noqa: E402

DEFAULT_LHS_REL = "ROM/classic/lhs_pool.json"
RESET_SCHEMA = "m4_full_lhs_pool_reset_plan_v1"

LHS_ENTRY_COMPLETION_KEYS = (
    "last_run_id",
    "last_run_dir",
    "last_batch_id",
    "last_started_at",
    "last_finished_at",
    "last_elapsed_s",
    "last_aggregation_status",
    "last_deduped_mode_count",
    "last_participation_computed_count",
    "last_audio_coupling_computed_count",
    "last_error",
    "last_terminal_status",
    "last_outcome",
    "last_deduped_modes",
    "last_raw_modes",
    "last_rom_status",
    "last_rom_comparison_status",
    "last_rom_comparison_path",
    "last_rom_median_relative_error",
    "last_rom_mean_relative_error",
    "last_rom_p90_relative_error",
    "last_rom_mean_abs_error_hz",
    "last_rom_median_abs_error_hz",
    "last_rom_max_abs_error_hz",
    "last_rom_matched_mode_count",
    "last_rom_meets_accuracy_target",
    "last_rom_accuracy_meaningful",
    "last_rom_validation_mode",
    "last_rom_training_sample_count",
    "last_rom_prediction_runtime_s",
    "last_rom_comparison_runtime_s",
    "last_rom_total_runtime_s",
    "last_rom_top_share_mae",
    "last_rom_coupling_class_accuracy",
    "last_rom_error",
    "snapshot_file",
)


def summarize_pool_statuses(pool: Mapping[str, Any]) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for entry in pool.get("entries") or []:
        status = normalize_lhs_entry_status(entry.get("status"))
        counts[status] += 1
    return dict(counts)


def reset_pool_entries(pool: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of pool with every entry reset to PENDING and completion fields cleared."""
    out = copy.deepcopy(pool)
    entries: List[Dict[str, Any]] = []
    for entry in out.get("entries") or []:
        row = dict(entry)
        for key in LHS_ENTRY_COMPLETION_KEYS:
            row.pop(key, None)
        row["status"] = LHS_PENDING
        row["error"] = None
        entries.append(row)
    out["entries"] = entries
    out["full_reset_utc"] = utc_now()
    return out


def plan_full_lhs_pool_reset(
    *,
    repo_root: Path,
    lhs_path: Path,
    run_id_suffix: Optional[str] = None,
) -> Dict[str, Any]:
    repo_root = repo_root.resolve()
    pool = load_lhs_pool(lhs_path)
    entries = list(pool.get("entries") or [])
    status_counts = summarize_pool_statuses(pool)
    status_path = lhs_pool_status_path(repo_root)
    index_path = lhs_runs_index_path(repo_root)
    status_doc = load_lhs_pool_status(
        status_path,
        lhs_path=lhs_path,
        run_id_suffix=run_id_suffix or "pending",
        repo_root=repo_root,
    )

    lhs_rows: List[Dict[str, Any]] = []
    for i, entry in enumerate(entries):
        cleared = [k for k in LHS_ENTRY_COMPLETION_KEYS if k in entry]
        lhs_rows.append(
            {
                "lhs_index": i,
                "sample_id": str(entry.get("id") or ""),
                "previous_status": entry.get("status"),
                "previous_last_run_id": entry.get("last_run_id"),
                "action": "reset_to_pending",
                "fields_to_clear": cleared,
            }
        )

    status_entries = list((status_doc.get("samples") or {}).keys())
    index_row_count = 0
    if index_path.is_file():
        index_row_count = sum(1 for line in index_path.read_text(encoding="utf-8").splitlines() if line.strip())

    return {
        "schema": RESET_SCHEMA,
        "generated_utc": utc_now(),
        "lhs_json": rel(lhs_path, repo_root=repo_root),
        "total_lhs_entries": len(entries),
        "status_counts_before_reset": status_counts,
        "entries_currently_pending": int(status_counts.get(LHS_PENDING, 0)),
        "entries_currently_completed": int(status_counts.get("COMPLETED", 0)),
        "entries_currently_failed": int(status_counts.get("FAILED", 0))
        + int(status_counts.get("FAILED_RETRYABLE", 0)),
        "entries_currently_running": int(status_counts.get("RUNNING", 0)),
        "fields_cleared_per_entry": list(LHS_ENTRY_COMPLETION_KEYS),
        "lhs_entries_to_reset": lhs_rows,
        "index_cleanup": {
            "lhs_pool_status_path": rel(status_path, repo_root=repo_root) if status_path.is_file() else None,
            "lhs_runs_index_path": rel(index_path, repo_root=repo_root) if index_path.is_file() else None,
            "status_sample_ids_to_clear": status_entries,
            "status_sample_id_count": len(status_entries),
            "index_rows_to_clear": index_row_count,
            "batch_sidecar_action": "clear_lhs_pool_status_and_truncate_runs_index",
        },
        "run_trees_are_not_deleted": True,
        "run_directory_deletions": 0,
    }


def apply_full_lhs_pool_reset(
    *,
    repo_root: Path,
    lhs_path: Path,
    run_id_suffix: Optional[str] = None,
) -> Dict[str, Any]:
    plan = plan_full_lhs_pool_reset(
        repo_root=repo_root,
        lhs_path=lhs_path,
        run_id_suffix=run_id_suffix,
    )
    pool = load_lhs_pool(lhs_path)
    pool = reset_pool_entries(pool)
    write_lhs_pool_with_backup(lhs_path, pool, explicit_lhs_regeneration=True)

    status_path = lhs_pool_status_path(repo_root)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    write_lhs_pool_status(
        status_path,
        {
            "schema": STATUS_SCHEMA,
            "lhs_json": rel(lhs_path, repo_root=repo_root),
            "run_id_suffix_default": run_id_suffix,
            "pipeline_version": "M4 production v1",
            "updated_utc": utc_now(),
            "samples": {},
            "full_reset_utc": utc_now(),
        },
    )

    index_path = lhs_runs_index_path(repo_root)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text("", encoding="utf-8")

    after_counts = summarize_pool_statuses(pool)
    return {
        "executed_utc": utc_now(),
        "entries_reset": len(plan.get("lhs_entries_to_reset") or []),
        "status_counts_after_reset": after_counts,
        "all_entries_pending": after_counts == {LHS_PENDING: len(pool.get("entries") or [])},
        "plan": plan,
    }


def verify_all_entries_pending(pool: Mapping[str, Any]) -> bool:
    entries = pool.get("entries") or []
    if not entries:
        return False
    return all(normalize_lhs_entry_status(e.get("status")) == LHS_PENDING for e in entries)


def _print_plan(plan: Mapping[str, Any]) -> None:
    print(f"total_lhs_entries={plan.get('total_lhs_entries')}")
    print(f"entries_currently_pending={plan.get('entries_currently_pending')}")
    print(f"entries_currently_completed={plan.get('entries_currently_completed')}")
    print(f"entries_currently_failed={plan.get('entries_currently_failed')}")
    print(f"entries_currently_running={plan.get('entries_currently_running')}")
    print("fields_cleared_per_entry:")
    for key in plan.get("fields_cleared_per_entry") or []:
        print(f"  - {key}")
    idx = plan.get("index_cleanup") or {}
    print(f"status_sample_ids_to_clear={idx.get('status_sample_id_count')}")
    print(f"index_rows_to_clear={idx.get('index_rows_to_clear')}")
    print(f"run_trees_are_not_deleted={str(plan.get('run_trees_are_not_deleted')).lower()}")
    print(f"run_directory_deletions={plan.get('run_directory_deletions')}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Full LHS pool bookkeeping reset (dry-run default).")
    parser.add_argument("--lhs-json", type=Path, default=Path(DEFAULT_LHS_REL))
    parser.add_argument("--run-id-suffix", help="Recorded as default suffix in fresh lhs_pool_status.json.")
    parser.add_argument("--execute", action="store_true", help="Apply reset (default: dry-run report only).")
    parser.add_argument("--report-path", type=Path, help="Write JSON plan/report to this path.")
    args = parser.parse_args(argv)

    repo_root = detect_repo_root(SCRIPT_DIR)
    lhs_path = args.lhs_json if args.lhs_json.is_absolute() else repo_root / args.lhs_json
    if not lhs_path.is_file():
        print(f"error: missing --lhs-json: {lhs_path}", file=sys.stderr)
        return 2

    plan = plan_full_lhs_pool_reset(
        repo_root=repo_root,
        lhs_path=lhs_path,
        run_id_suffix=args.run_id_suffix,
    )
    report_path = args.report_path
    if report_path is None:
        report_path = (
            repo_root
            / "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/specs/generated"
            / f"full_lhs_pool_reset_{utc_now()[:10].replace('-', '')}.json"
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(report_path, plan)
    _print_plan(plan)
    print(f"report={rel(report_path, repo_root=repo_root)}")
    print(f"will_execute={str(bool(args.execute)).lower()}")

    if not args.execute:
        print("no lhs bookkeeping modified")
        return 0

    result = apply_full_lhs_pool_reset(
        repo_root=repo_root,
        lhs_path=lhs_path,
        run_id_suffix=args.run_id_suffix,
    )
    write_json_atomic(report_path.with_name(report_path.stem + "_executed.json"), result)
    print(f"entries_reset={result.get('entries_reset')}")
    print(f"all_entries_pending={result.get('all_entries_pending')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
