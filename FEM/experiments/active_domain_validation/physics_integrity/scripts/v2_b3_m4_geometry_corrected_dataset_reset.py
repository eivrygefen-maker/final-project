#!/usr/bin/env python3
"""Guarded migration reset for m4_geometry_corrected_v1 (dry-run default, --execute required)."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_m4_lhs_pool_bridge import (  # noqa: E402
    LHS_PENDING,
    lhs_pool_status_path,
    lhs_runs_index_path,
    load_lhs_pool,
    write_lhs_pool_with_backup,
)
from v2_b3_m4_production_contracts import DATASET_VERSION  # noqa: E402
from v2_b3_m4_production_freeze import TERMINAL_PRODUCTION_COMPLETED  # noqa: E402
from v2_b3_m4_worker_run_lib import detect_repo_root, load_json, rel, utc_now  # noqa: E402
from v2_b3_petsc_util import write_json_atomic  # noqa: E402

DEFAULT_LHS_REL = "ROM/classic/lhs_pool.json"
DEFAULT_SMOKE_RUN_ID = "sample_001_m4prod2_geometryfix_smoke"
RESET_SAMPLE_IDS = tuple(f"sample_{i:03d}" for i in range(37))  # sample_000 .. sample_036

ROM_QUARANTINE_NAMES = (
    "m4_modal_surrogate.json",
    "m4_modal_surrogate.npz",
    "rom_model_manifest.json",
    "m4_rom_build_report.json",
)
ROM_QUARANTINE_GLOBS = (
    "experimental_v22*",
    "experimental_v22b*",
)

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
)

GUITARS_REL = "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars"
ROM_CLASSIC_REL = "ROM/classic"
ROM_INVALID_REL = "ROM/classic_INVALID_pre_geometryfix_v1"
ROM_CORRECTED_REL = "ROM/m4_geometry_corrected_v1"


def guitars_root(repo_root: Path) -> Path:
    return repo_root / GUITARS_REL


def rom_classic_dir(repo_root: Path) -> Path:
    return repo_root / ROM_CLASSIC_REL


def rom_invalid_dir(repo_root: Path) -> Path:
    return repo_root / ROM_INVALID_REL


def rom_corrected_dir(repo_root: Path) -> Path:
    return repo_root / ROM_CORRECTED_REL


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                continue
    return total


def _load_json_optional(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        doc = load_json(path)
        return doc if isinstance(doc, dict) else {}
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def is_corrected_completed_run(run_root: Path) -> bool:
    pipeline = _load_json_optional(run_root / "pipeline_run_manifest.json")
    freeze = _load_json_optional(run_root / "freeze" / "freeze_manifest.json")
    built = _load_json_optional(run_root / "lprod" / "checkpoint" / "built_metadata.json")
    terminal = str(pipeline.get("terminal_status") or freeze.get("terminal_status") or "")
    dataset = str(
        pipeline.get("dataset_version")
        or freeze.get("dataset_version")
        or built.get("dataset_version")
        or ""
    )
    return dataset == DATASET_VERSION and terminal == TERMINAL_PRODUCTION_COMPLETED


def classify_run_dir(run_root: Path, *, smoke_run_id: str) -> str:
    if not run_root.is_dir():
        return "missing"
    sample_id = run_root.parent.parent.name
    if sample_id not in RESET_SAMPLE_IDS:
        return "out_of_scope"
    if run_root.name == smoke_run_id:
        return "preserve_smoke"
    if is_corrected_completed_run(run_root):
        return "preserve_corrected_completed"
    return "delete"


def iter_reset_run_dirs(repo_root: Path) -> Iterable[Path]:
    root = guitars_root(repo_root)
    if not root.is_dir():
        return
    for sample_id in RESET_SAMPLE_IDS:
        runs_dir = root / sample_id / "runs"
        if not runs_dir.is_dir():
            continue
        for run_dir in sorted(runs_dir.iterdir()):
            if run_dir.is_dir():
                yield run_dir


def discover_rom_quarantine_files(repo_root: Path) -> List[Path]:
    classic = rom_classic_dir(repo_root)
    if not classic.is_dir():
        return []
    found: List[Path] = []
    for name in ROM_QUARANTINE_NAMES:
        path = classic / name
        if path.is_file():
            found.append(path)
    for pattern in ROM_QUARANTINE_GLOBS:
        for path in sorted(classic.glob(pattern)):
            if path.is_file() and path.name != "lhs_pool.json":
                found.append(path)
    # comparisons/ and other dirs are intentionally left in ROM/classic
    return sorted(set(found), key=lambda p: str(p))


def _lhs_entry_index(pool: Mapping[str, Any], sample_id: str) -> Optional[int]:
    for i, row in enumerate(pool.get("entries") or []):
        if str(row.get("id")) == sample_id:
            return i
    return None


def plan_lhs_resets(pool: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for sample_id in RESET_SAMPLE_IDS:
        idx = _lhs_entry_index(pool, sample_id)
        if idx is None:
            rows.append({"sample_id": sample_id, "action": "skip_missing_entry"})
            continue
        entry = pool["entries"][idx]
        cleared = [k for k in LHS_ENTRY_COMPLETION_KEYS if k in entry]
        rows.append(
            {
                "sample_id": sample_id,
                "lhs_row_index": idx,
                "previous_status": entry.get("status"),
                "action": "reset_to_pending",
                "fields_to_clear": cleared,
            }
        )
    return rows


def apply_lhs_pool_reset(pool: Dict[str, Any]) -> Dict[str, Any]:
    pool = dict(pool)
    pool["dataset_version"] = DATASET_VERSION
    entries = []
    for entry in pool.get("entries") or []:
        row = dict(entry)
        sid = str(row.get("id") or "")
        if sid in RESET_SAMPLE_IDS:
            for key in LHS_ENTRY_COMPLETION_KEYS:
                row.pop(key, None)
            row["status"] = LHS_PENDING
            row["error"] = None
        entries.append(row)
    pool["entries"] = entries
    pool["reset_utc"] = utc_now()
    return pool


def plan_index_cleanup(
    *,
    repo_root: Path,
    preserved_run_ids: Sequence[str],
    smoke_run_id: str,
) -> Dict[str, Any]:
    preserved = set(preserved_run_ids) | {smoke_run_id}
    status_path = lhs_pool_status_path(repo_root)
    index_path = lhs_runs_index_path(repo_root)

    status_remove: List[str] = []
    status_preserve: List[str] = []
    if status_path.is_file():
        doc = _load_json_optional(status_path)
        for sid, row in (doc.get("samples") or {}).items():
            run_id = str((row or {}).get("run_id") or "")
            if run_id and run_id in preserved:
                status_preserve.append(f"{sid}:{run_id}")
            elif sid in RESET_SAMPLE_IDS:
                status_remove.append(f"{sid}:{run_id or 'no_run_id'}")

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
            if run_id in preserved:
                index_preserve.append(key)
            elif sample_id in RESET_SAMPLE_IDS:
                index_remove.append(key)

    return {
        "lhs_pool_status_path": rel(status_path, repo_root=repo_root) if status_path.is_file() else None,
        "lhs_runs_index_path": rel(index_path, repo_root=repo_root) if index_path.is_file() else None,
        "status_entries_to_remove": status_remove,
        "status_entries_to_preserve": status_preserve,
        "index_rows_to_remove": index_remove,
        "index_rows_to_preserve": index_preserve,
    }


def build_reset_plan(
    *,
    repo_root: Path,
    smoke_run_id: str = DEFAULT_SMOKE_RUN_ID,
) -> Dict[str, Any]:
    repo_root = repo_root.resolve()
    lhs_path = repo_root / DEFAULT_LHS_REL
    pool = load_lhs_pool(lhs_path) if lhs_path.is_file() else {"entries": []}

    runs_delete: List[Dict[str, Any]] = []
    runs_preserve: List[Dict[str, Any]] = []
    bytes_delete = 0
    preserved_run_ids: List[str] = []

    for run_dir in iter_reset_run_dirs(repo_root):
        action = classify_run_dir(run_dir, smoke_run_id=smoke_run_id)
        size = _dir_size(run_dir)
        row = {
            "sample_id": run_dir.parent.parent.name,
            "run_id": run_dir.name,
            "run_dir": rel(run_dir, repo_root=repo_root),
            "action": action,
            "bytes": size,
        }
        if action in ("preserve_smoke", "preserve_corrected_completed"):
            runs_preserve.append(row)
            preserved_run_ids.append(run_dir.name)
        elif action == "delete":
            runs_delete.append(row)
            bytes_delete += size

    rom_files = discover_rom_quarantine_files(repo_root)
    rom_rows = [
        {
            "source": rel(path, repo_root=repo_root),
            "dest": rel(rom_invalid_dir(repo_root) / path.name, repo_root=repo_root),
            "bytes": _dir_size(path),
        }
        for path in rom_files
    ]
    rom_bytes = sum(int(r["bytes"]) for r in rom_rows)

    index_plan = plan_index_cleanup(
        repo_root=repo_root,
        preserved_run_ids=preserved_run_ids,
        smoke_run_id=smoke_run_id,
    )

    smoke_path = guitars_root(repo_root) / "sample_001" / "runs" / smoke_run_id

    return {
        "schema": "m4_geometry_corrected_reset_plan_v1",
        "generated_utc": utc_now(),
        "repo_root": str(repo_root),
        "dataset_version": DATASET_VERSION,
        "smoke_run_id": smoke_run_id,
        "smoke_run_path": rel(smoke_path, repo_root=repo_root) if smoke_path.is_dir() else None,
        "lhs_json": DEFAULT_LHS_REL,
        "runs_to_delete": runs_delete,
        "runs_preserved": runs_preserve,
        "estimated_run_delete_bytes": bytes_delete,
        "rom_files_to_quarantine": rom_rows,
        "estimated_rom_quarantine_bytes": rom_bytes,
        "estimated_total_freed_bytes": bytes_delete + rom_bytes,
        "lhs_entries_to_reset": plan_lhs_resets(pool),
        "index_cleanup": index_plan,
        "directories_to_create": [ROM_CORRECTED_REL],
    }


def _print_plan(plan: Mapping[str, Any]) -> None:
    print(f"dataset_version={plan.get('dataset_version')}")
    print(f"smoke_run_path={plan.get('smoke_run_path')}")
    print(f"estimated_total_freed_bytes={plan.get('estimated_total_freed_bytes')}")
    print("")
    print("run directories to delete:")
    for row in plan.get("runs_to_delete") or []:
        print(f"  - {row.get('run_dir')} ({row.get('bytes')} bytes)")
    print("")
    print("run directories preserved:")
    for row in plan.get("runs_preserved") or []:
        print(f"  - {row.get('run_dir')} [{row.get('action')}]")
    print("")
    print("ROM files to quarantine:")
    for row in plan.get("rom_files_to_quarantine") or []:
        print(f"  - {row.get('source')} -> {row.get('dest')}")
    print("")
    print("LHS entries to reset:")
    for row in plan.get("lhs_entries_to_reset") or []:
        print(
            f"  - {row.get('sample_id')}: {row.get('action')} "
            f"(clear {len(row.get('fields_to_clear') or [])} fields)"
        )
    idx = plan.get("index_cleanup") or {}
    print("")
    print("index entries to remove:")
    for item in idx.get("status_entries_to_remove") or []:
        print(f"  - lhs_pool_status: {item}")
    for item in idx.get("index_rows_to_remove") or []:
        print(f"  - lhs_production_runs_index: {item}")
    print("")
    print("index entries to preserve:")
    for item in idx.get("status_entries_to_preserve") or []:
        print(f"  - lhs_pool_status: {item}")
    for item in idx.get("index_rows_to_preserve") or []:
        print(f"  - lhs_production_runs_index: {item}")


def apply_index_cleanup(
    *,
    repo_root: Path,
    plan: Mapping[str, Any],
    report: Dict[str, Any],
) -> None:
    preserved_run_ids = {row["run_id"] for row in plan.get("runs_preserved") or []}
    preserved_run_ids.add(str(plan.get("smoke_run_id")))

    status_path = lhs_pool_status_path(repo_root)
    if status_path.is_file():
        doc = dict(_load_json_optional(status_path))
        samples = dict(doc.get("samples") or {})
        removed: List[str] = []
        kept: List[str] = []
        for sid in list(samples.keys()):
            if sid not in RESET_SAMPLE_IDS:
                continue
            row = dict(samples.get(sid) or {})
            run_id = str(row.get("run_id") or "")
            if run_id and run_id in preserved_run_ids:
                kept.append(f"{sid}:{run_id}")
                continue
            removed.append(f"{sid}:{run_id or 'no_run_id'}")
            samples.pop(sid, None)
        doc["samples"] = samples
        doc["updated_utc"] = utc_now()
        write_json_atomic(status_path, doc)
        report["index_actions"].append(
            {"path": rel(status_path, repo_root=repo_root), "removed": removed, "preserved": kept}
        )

    index_path = lhs_runs_index_path(repo_root)
    if index_path.is_file():
        kept_lines: List[str] = []
        removed: List[str] = []
        preserved: List[str] = []
        for line in index_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                removed.append("unparseable_line")
                continue
            run_id = str(row.get("run_id") or "")
            sample_id = str(row.get("sample_id") or "")
            key = f"{sample_id}:{run_id}"
            if run_id in preserved_run_ids:
                kept_lines.append(line)
                preserved.append(key)
            elif sample_id in RESET_SAMPLE_IDS:
                removed.append(key)
            else:
                kept_lines.append(line)
        index_path.write_text("\n".join(kept_lines) + ("\n" if kept_lines else ""), encoding="utf-8")
        report["index_actions"].append(
            {
                "path": rel(index_path, repo_root=repo_root),
                "removed": removed,
                "preserved": preserved,
            }
        )


def execute_reset_plan(
    *,
    repo_root: Path,
    plan: Mapping[str, Any],
) -> Dict[str, Any]:
    repo_root = repo_root.resolve()
    report: Dict[str, Any] = {
        "schema": "m4_geometry_corrected_reset_report_v1",
        "executed_utc": utc_now(),
        "repo_root": str(repo_root),
        "dataset_version": DATASET_VERSION,
        "smoke_run_id": plan.get("smoke_run_id"),
        "actions": [],
        "index_actions": [],
        "bytes_deleted": 0,
        "bytes_quarantined": 0,
    }

    for row in plan.get("runs_to_delete") or []:
        run_dir = repo_root / str(row["run_dir"])
        size = _dir_size(run_dir)
        if run_dir.is_dir():
            shutil.rmtree(run_dir)
        report["actions"].append(
            {"action": "delete_run_dir", "path": row["run_dir"], "bytes": size, "status": "done"}
        )
        report["bytes_deleted"] += size

    invalid_dir = rom_invalid_dir(repo_root)
    invalid_dir.mkdir(parents=True, exist_ok=True)
    corrected_dir = rom_corrected_dir(repo_root)
    corrected_dir.mkdir(parents=True, exist_ok=True)
    readme = corrected_dir / "README.txt"
    if not readme.is_file():
        readme.write_text(
            "M4 geometry-corrected ROM artifacts (m4_geometry_corrected_v1).\n"
            "Train new surrogates here after >=3 corrected production samples complete.\n"
            "Do not load models from ROM/classic_INVALID_pre_geometryfix_v1.\n",
            encoding="utf-8",
        )
    report["actions"].append(
        {"action": "create_dir", "path": ROM_CORRECTED_REL, "status": "done"}
    )

    for row in plan.get("rom_files_to_quarantine") or []:
        src = repo_root / str(row["source"])
        dest = repo_root / str(row["dest"])
        size = _dir_size(src)
        if src.is_file():
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.is_file():
                dest.unlink()
            shutil.move(str(src), str(dest))
        report["actions"].append(
            {
                "action": "quarantine_rom_file",
                "source": row["source"],
                "dest": row["dest"],
                "bytes": size,
                "status": "done" if not src.exists() else "failed",
            }
        )
        report["bytes_quarantined"] += size

    lhs_path = repo_root / DEFAULT_LHS_REL
    pool = load_lhs_pool(lhs_path)
    write_lhs_pool_with_backup(lhs_path, apply_lhs_pool_reset(pool))
    report["actions"].append(
        {
            "action": "reset_lhs_pool",
            "path": DEFAULT_LHS_REL,
            "entries_reset": len(plan.get("lhs_entries_to_reset") or []),
            "status": "done",
        }
    )

    apply_index_cleanup(repo_root=repo_root, plan=plan, report=report)
    report["estimated_total_freed_bytes"] = int(plan.get("estimated_total_freed_bytes") or 0)
    report["runs_preserved"] = plan.get("runs_preserved")
    report["smoke_run_path"] = plan.get("smoke_run_path")
    return report


def verify_reset_state(
    *,
    repo_root: Path,
    smoke_run_id: str = DEFAULT_SMOKE_RUN_ID,
) -> Dict[str, Any]:
    repo_root = repo_root.resolve()
    checks: List[Dict[str, Any]] = []

    lhs_path = repo_root / DEFAULT_LHS_REL
    ok_lhs_exists = lhs_path.is_file()
    checks.append({"check": "lhs_pool_exists", "pass": ok_lhs_exists, "path": DEFAULT_LHS_REL})

    pool = load_lhs_pool(lhs_path) if ok_lhs_exists else {"entries": []}
    ds_ok = str(pool.get("dataset_version") or "") == DATASET_VERSION
    checks.append(
        {
            "check": "lhs_pool_dataset_version",
            "pass": ds_ok,
            "expected": DATASET_VERSION,
            "actual": pool.get("dataset_version"),
        }
    )

    pending_ok = True
    bad_entries: List[str] = []
    for sample_id in RESET_SAMPLE_IDS:
        idx = _lhs_entry_index(pool, sample_id)
        if idx is None:
            pending_ok = False
            bad_entries.append(f"{sample_id}:missing")
            continue
        entry = pool["entries"][idx]
        status = str(entry.get("status") or "").upper()
        if status != LHS_PENDING:
            pending_ok = False
            bad_entries.append(f"{sample_id}:{status}")
        if entry.get("last_run_id"):
            pending_ok = False
            bad_entries.append(f"{sample_id}:has_last_run_id")
    checks.append(
        {
            "check": "lhs_entries_000_036_pending",
            "pass": pending_ok,
            "failures": bad_entries,
        }
    )

    smoke_path = guitars_root(repo_root) / "sample_001" / "runs" / smoke_run_id
    smoke_exists = smoke_path.is_dir()
    smoke_completed = smoke_exists and is_corrected_completed_run(smoke_path)
    checks.append(
        {
            "check": "smoke_run_exists_and_completed",
            "pass": smoke_exists and smoke_completed,
            "path": rel(smoke_path, repo_root=repo_root) if smoke_exists else None,
            "terminal_status": _load_json_optional(smoke_path / "pipeline_run_manifest.json").get(
                "terminal_status"
            )
            if smoke_exists
            else None,
        }
    )

    stale_runs: List[str] = []
    for run_dir in iter_reset_run_dirs(repo_root):
        action = classify_run_dir(run_dir, smoke_run_id=smoke_run_id)
        if action == "delete":
            stale_runs.append(rel(run_dir, repo_root=repo_root))
    checks.append(
        {
            "check": "invalid_m4prod1_runs_gone",
            "pass": len(stale_runs) == 0,
            "remaining_invalid_runs": stale_runs,
        }
    )

    from v2_b3_m4_modal_surrogate_lib import surrogate_is_available  # noqa: WPS433

    rom_loadable = surrogate_is_available(repo_root, "classic")
    checks.append(
        {
            "check": "old_rom_cannot_load_from_classic",
            "pass": not rom_loadable,
            "rom_classic_surrogate_available": rom_loadable,
        }
    )

    quarantined = discover_rom_quarantine_files(repo_root)
    checks.append(
        {
            "check": "rom_quarantine_files_removed_from_classic",
            "pass": len(quarantined) == 0,
            "remaining_in_classic": [rel(p, repo_root=repo_root) for p in quarantined],
        }
    )

    all_pass = all(bool(c.get("pass")) for c in checks)
    return {"verify_pass": all_pass, "checks": checks}


def _default_report_path(repo_root: Path) -> Path:
    stamp = utc_now().replace(":", "").replace("-", "")
    return (
        repo_root
        / "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/index"
        / f"geometry_corrected_reset_report_{stamp}.json"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Guarded reset for m4_geometry_corrected_v1 dataset migration."
    )
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--smoke-run-id", default=DEFAULT_SMOKE_RUN_ID)
    parser.add_argument("--execute", action="store_true", help="Apply planned actions (default: dry-run).")
    parser.add_argument("--verify", action="store_true", help="Verify post-reset state and exit.")
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Write reset report JSON (execute mode).",
    )
    args = parser.parse_args(argv)

    repo_root = (args.repo_root or detect_repo_root(SCRIPT_DIR)).resolve()

    if args.verify:
        result = verify_reset_state(repo_root=repo_root, smoke_run_id=str(args.smoke_run_id))
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("verify_pass") else 2

    plan = build_reset_plan(repo_root=repo_root, smoke_run_id=str(args.smoke_run_id))
    if not args.execute:
        print("DRY RUN — no changes written (pass --execute to apply)")
        print("")
        _print_plan(plan)
        print("")
        print(f"report_preview={json.dumps({'estimated_total_freed_bytes': plan.get('estimated_total_freed_bytes')})}")
        return 0

    report = execute_reset_plan(repo_root=repo_root, plan=plan)
    report_path = (
        args.report_json
        if args.report_json is not None
        else _default_report_path(repo_root)
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"report_json={rel(report_path, repo_root=repo_root)}")
    verify = verify_reset_state(repo_root=repo_root, smoke_run_id=str(args.smoke_run_id))
    print(json.dumps(verify, indent=2, sort_keys=True))
    return 0 if verify.get("verify_pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
