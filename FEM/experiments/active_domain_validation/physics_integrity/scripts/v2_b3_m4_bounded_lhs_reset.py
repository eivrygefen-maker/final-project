#!/usr/bin/env python3
"""Bounded LHS bookkeeping reset for a selected index range (dry-run default)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_m4_lhs_pool_bridge import (  # noqa: E402
    LHS_PENDING,
    lhs_pool_status_path,
    lhs_runs_index_path,
    load_lhs_pool,
    load_lhs_pool_status,
    write_lhs_pool_status,
    write_lhs_pool_with_backup,
)
from v2_b3_m4_worker_run_lib import detect_repo_root, load_json, rel, utc_now  # noqa: E402
from v2_b3_petsc_util import write_json_atomic  # noqa: E402

DEFAULT_LHS_REL = "ROM/classic/lhs_pool.json"
RESET_SCHEMA = "m4_bounded_lhs_reset_plan_v1"

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
)


def sample_id_for_index(index: int) -> str:
    return f"sample_{index:03d}"


def indexes_in_range(*, start_index: int, end_index: int) -> List[int]:
    if start_index < 0:
        raise ValueError("start_index must be >= 0")
    if end_index < start_index:
        raise ValueError("end_index must be >= start_index")
    return list(range(start_index, end_index + 1))


def plan_bounded_lhs_reset(
    *,
    repo_root: Path,
    lhs_path: Path,
    start_index: int,
    end_index: int,
    preserved_run_ids: Optional[Sequence[str]] = None,
    run_id_suffix: Optional[str] = None,
) -> Dict[str, Any]:
    repo_root = repo_root.resolve()
    pool = load_lhs_pool(lhs_path)
    entries = list(pool.get("entries") or [])
    target_indexes = set(indexes_in_range(start_index=start_index, end_index=end_index))
    preserved = set(preserved_run_ids or ())
    status_path = lhs_pool_status_path(repo_root)
    index_path = lhs_runs_index_path(repo_root)
    status_doc = load_lhs_pool_status(
        status_path,
        lhs_path=lhs_path,
        run_id_suffix=run_id_suffix or "pending",
        repo_root=repo_root,
    )

    lhs_rows: List[Dict[str, Any]] = []
    untouched_rows: List[Dict[str, Any]] = []
    for i, entry in enumerate(entries):
        sid = str(entry.get("id") or "")
        if i in target_indexes:
            cleared = [k for k in LHS_ENTRY_COMPLETION_KEYS if k in entry]
            lhs_rows.append(
                {
                    "lhs_index": i,
                    "sample_id": sid,
                    "previous_status": entry.get("status"),
                    "previous_last_run_id": entry.get("last_run_id"),
                    "action": "reset_to_pending",
                    "fields_to_clear": cleared,
                    "run_tree_preserved": True,
                }
            )
        elif i > end_index:
            untouched_rows.append(
                {
                    "lhs_index": i,
                    "sample_id": sid,
                    "action": "untouched_out_of_range",
                }
            )

    status_remove: List[str] = []
    status_preserve: List[str] = []
    for sid, row in (status_doc.get("samples") or {}).items():
        if not isinstance(row, dict):
            continue
        run_id = str(row.get("run_id") or "")
        idx = None
        for i, entry in enumerate(entries):
            if str(entry.get("id")) == sid:
                idx = i
                break
        key = f"{sid}:{run_id or 'no_run_id'}"
        if idx is not None and idx in target_indexes:
            if run_id and run_id in preserved:
                status_preserve.append(key)
            else:
                status_remove.append(key)

    index_remove: List[str] = []
    index_preserve: List[str] = []
    if index_path.is_file():
        for line in index_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                index_remove.append("unparseable_line")
                continue
            run_id = str(row.get("run_id") or "")
            sample_id = str(row.get("sample_id") or "")
            key = f"{sample_id}:{run_id}"
            idx = None
            for i, entry in enumerate(entries):
                if str(entry.get("id")) == sample_id:
                    idx = i
                    break
            if idx is not None and idx in target_indexes:
                if run_id in preserved:
                    index_preserve.append(key)
                else:
                    index_remove.append(key)

    return {
        "schema": RESET_SCHEMA,
        "generated_utc": utc_now(),
        "lhs_json": rel(lhs_path, repo_root=repo_root),
        "start_index": start_index,
        "end_index": end_index,
        "target_indexes": sorted(target_indexes),
        "preserved_run_ids": sorted(preserved),
        "lhs_entries_to_reset": lhs_rows,
        "lhs_entries_untouched_after_end": untouched_rows[:5],
        "index_cleanup": {
            "lhs_pool_status_path": rel(status_path, repo_root=repo_root) if status_path.is_file() else None,
            "lhs_runs_index_path": rel(index_path, repo_root=repo_root) if index_path.is_file() else None,
            "status_entries_to_remove": status_remove,
            "status_entries_to_preserve": status_preserve,
            "index_rows_to_remove": index_remove,
            "index_rows_to_preserve": index_preserve,
        },
        "run_trees_are_not_deleted": True,
    }


def apply_bounded_lhs_reset(
    *,
    repo_root: Path,
    lhs_path: Path,
    start_index: int,
    end_index: int,
    preserved_run_ids: Optional[Sequence[str]] = None,
    run_id_suffix: Optional[str] = None,
) -> Dict[str, Any]:
    plan = plan_bounded_lhs_reset(
        repo_root=repo_root,
        lhs_path=lhs_path,
        start_index=start_index,
        end_index=end_index,
        preserved_run_ids=preserved_run_ids,
        run_id_suffix=run_id_suffix,
    )
    pool = load_lhs_pool(lhs_path)
    target_indexes = set(plan["target_indexes"])
    preserved = set(preserved_run_ids or ())
    entries: List[Dict[str, Any]] = []
    for i, entry in enumerate(pool.get("entries") or []):
        row = dict(entry)
        if i in target_indexes:
            for key in LHS_ENTRY_COMPLETION_KEYS:
                row.pop(key, None)
            row["status"] = LHS_PENDING
            row["error"] = None
        entries.append(row)
    pool["entries"] = entries
    pool["bounded_reset_utc"] = utc_now()
    write_lhs_pool_with_backup(lhs_path, pool)

    status_path = lhs_pool_status_path(repo_root)
    status_doc = load_lhs_pool_status(
        status_path,
        lhs_path=lhs_path,
        run_id_suffix=run_id_suffix or "pending",
        repo_root=repo_root,
    )
    samples = dict(status_doc.get("samples") or {})
    for sid in list(samples.keys()):
        idx = None
        for i, entry in enumerate(entries):
            if str(entry.get("id")) == sid:
                idx = i
                break
        if idx is None or idx not in target_indexes:
            continue
        row = dict(samples.get(sid) or {})
        run_id = str(row.get("run_id") or "")
        if run_id and run_id in preserved:
            continue
        samples.pop(sid, None)
    status_doc["samples"] = samples
    status_doc["updated_utc"] = utc_now()
    write_lhs_pool_status(status_path, status_doc)

    index_path = lhs_runs_index_path(repo_root)
    if index_path.is_file():
        kept_lines: List[str] = []
        for line in index_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                kept_lines.append(line)
                continue
            run_id = str(row.get("run_id") or "")
            sample_id = str(row.get("sample_id") or "")
            idx = None
            for i, entry in enumerate(entries):
                if str(entry.get("id")) == sample_id:
                    idx = i
                    break
            if idx is not None and idx in target_indexes and run_id not in preserved:
                continue
            kept_lines.append(line)
        index_path.write_text("\n".join(kept_lines) + ("\n" if kept_lines else ""), encoding="utf-8")

    return {
        "executed_utc": utc_now(),
        "entries_reset": len(plan.get("lhs_entries_to_reset") or []),
        "plan": plan,
    }


def _print_plan(plan: Mapping[str, Any]) -> None:
    print(f"start_index={plan.get('start_index')} end_index={plan.get('end_index')}")
    print(f"target_indexes={plan.get('target_indexes')}")
    print(f"preserved_run_ids={plan.get('preserved_run_ids')}")
    print("LHS entries to reset:")
    for row in plan.get("lhs_entries_to_reset") or []:
        print(
            f"  lhs_index={row.get('lhs_index')} sample_id={row.get('sample_id')} "
            f"prev_status={row.get('previous_status')} prev_run_id={row.get('previous_last_run_id')}"
        )
    idx = plan.get("index_cleanup") or {}
    print("status entries to remove:")
    for item in idx.get("status_entries_to_remove") or []:
        print(f"  - {item}")
    print("index rows to remove:")
    for item in idx.get("index_rows_to_remove") or []:
        print(f"  - {item}")
    print("run_trees_are_not_deleted=true")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Bounded LHS bookkeeping reset (dry-run default).")
    parser.add_argument("--lhs-json", type=Path, default=Path(DEFAULT_LHS_REL))
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, required=True)
    parser.add_argument(
        "--preserve-run-id",
        action="append",
        default=[],
        help="Run IDs whose run trees and index rows must be preserved (repeatable).",
    )
    parser.add_argument("--run-id-suffix", help="Sidecar default suffix when loading lhs_pool_status.")
    parser.add_argument("--execute", action="store_true", help="Apply reset (default: dry-run report only).")
    parser.add_argument("--report-path", type=Path, help="Write JSON plan/report to this path.")
    args = parser.parse_args(argv)

    repo_root = detect_repo_root(SCRIPT_DIR)
    lhs_path = args.lhs_json if args.lhs_json.is_absolute() else repo_root / args.lhs_json
    if not lhs_path.is_file():
        print(f"error: missing --lhs-json: {lhs_path}", file=sys.stderr)
        return 2

    try:
        plan = plan_bounded_lhs_reset(
            repo_root=repo_root,
            lhs_path=lhs_path,
            start_index=int(args.start_index),
            end_index=int(args.end_index),
            preserved_run_ids=args.preserve_run_id,
            run_id_suffix=args.run_id_suffix,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    report_path = args.report_path
    if report_path is None:
        report_path = (
            repo_root
            / "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/specs/generated"
            / f"bounded_lhs_reset_{args.start_index}_{args.end_index}_{utc_now()[:10].replace('-', '')}.json"
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(report_path, plan)
    _print_plan(plan)
    print(f"report={rel(report_path, repo_root=repo_root)}")
    print(f"will_execute={str(bool(args.execute)).lower()}")

    if not args.execute:
        print("no lhs bookkeeping modified")
        return 0

    result = apply_bounded_lhs_reset(
        repo_root=repo_root,
        lhs_path=lhs_path,
        start_index=int(args.start_index),
        end_index=int(args.end_index),
        preserved_run_ids=args.preserve_run_id,
        run_id_suffix=args.run_id_suffix,
    )
    write_json_atomic(report_path.with_name(report_path.stem + "_executed.json"), result)
    print(f"entries_reset={result.get('entries_reset')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
