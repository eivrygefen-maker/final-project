#!/usr/bin/env python3
"""Repair stale pipeline_run_manifest terminal_status after checkpoint-ready partial runs."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_m4_freeze_first_e2e_run import (  # noqa: E402
    CHECKPOINT_TERMINAL_READY,
    SCOUT_TERMINAL_READY,
    TERMINAL_E2E,
)
from v2_b3_m4_lhs_pool_bridge import is_run_usably_complete, read_run_production_summary  # noqa: E402
from v2_b3_m4_production_freeze import TERMINAL_PRODUCTION_COMPLETED  # noqa: E402
from v2_b3_m4_reuse_integrity_lib import (  # noqa: E402
    quarantine_stale_downstream_artifacts,
    terminal_status_rank,
)
from v2_b3_m4_worker_run_lib import (  # noqa: E402
    TERMINAL_CHECKPOINT_READY,
    chunk_ids_from_worker_plan,
    chunk_worker_pass_status,
    load_json,
    verify_lprod_checkpoint,
)
from v2_b3_petsc_util import write_json_atomic  # noqa: E402

STALE_RUNNING_REPAIR_JSON = "stale_running_repair.json"
M4_WORKER_ACTIVE_LOCK = "m4_worker_active.lock.json"
STALE_RUNNING_REPAIR_REASON = "stale_running_after_checkpoint_failure"
REPAIRABLE_TERMINAL_STATUSES = frozenset({"RUNNING"})


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def detect_active_worker_lock(
    run_root: Path,
    *,
    incomplete_chunk_grace_s: float = 900.0,
) -> Tuple[bool, str]:
    """Return (active, detail). Conservative: refuse repair when workers may still be running."""
    run_root = run_root.expanduser().resolve()
    lock_path = run_root / M4_WORKER_ACTIVE_LOCK
    if lock_path.is_file():
        try:
            lock = load_json(lock_path)
        except (OSError, ValueError, json.JSONDecodeError):
            lock = {}
        pid = int(lock.get("pid") or 0)
        if pid and _pid_alive(pid):
            return True, f"worker_lock_active pid={pid}"
        started = str(lock.get("started_utc") or "")
        return True, f"stale_worker_lock_present pid={pid} started={started}"

    worker_root = run_root / "worker_results"
    if worker_root.is_dir():
        now = time.time()
        for chunk_dir in sorted(worker_root.iterdir()):
            if not chunk_dir.is_dir():
                continue
            solver_path = chunk_dir / "solver_result.json"
            worker_path = chunk_dir / "worker_result.json"
            if solver_path.is_file() and not worker_path.is_file():
                age_s = now - solver_path.stat().st_mtime
                if age_s < incomplete_chunk_grace_s:
                    return True, f"incomplete_chunk_recent chunk={chunk_dir.name} age_s={age_s:.0f}"
    return False, "no_active_worker_lock"


def write_worker_active_lock(run_root: Path, *, pid: Optional[int] = None) -> Path:
    run_root = run_root.expanduser().resolve()
    body = {
        "schema": "m4_worker_active_lock_v1",
        "pid": int(pid or os.getpid()),
        "started_utc": utc_now(),
    }
    path = run_root / M4_WORKER_ACTIVE_LOCK
    write_json_atomic(path, body)
    return path


def clear_worker_active_lock(run_root: Path) -> None:
    path = run_root.expanduser().resolve() / M4_WORKER_ACTIVE_LOCK
    if path.is_file():
        path.unlink()


def _scout_pass(run_root: Path) -> bool:
    manifest_path = run_root / "pipeline_run_manifest.json"
    manifest = load_json(manifest_path) if manifest_path.is_file() else {}
    if str(manifest.get("terminal_status") or "") == SCOUT_TERMINAL_READY:
        return True
    st3 = (manifest.get("stages") or {}).get("stage3_zones_plan") or {}
    if str(st3.get("status") or "") != "PASS":
        return False
    return (run_root / "lprod" / "lprod_target_plan.json").is_file() and (
        run_root / "scout" / "density_zones.json"
    ).is_file()


def _worker_plan_pass(run_root: Path) -> bool:
    cmds = run_root / "lprod" / "worker_commands.json"
    chunk_plan = run_root / "lprod" / "worker_chunk_plan.preview.json"
    if not cmds.is_file() or not chunk_plan.is_file():
        return False
    worker_root = run_root / "worker_results"
    if not worker_root.is_dir():
        return False
    for cid in chunk_ids_from_worker_plan(run_root):
        if not (worker_root / cid / "chunk_targets.json").is_file():
            return False
    return True


def _workers_complete(run_root: Path) -> bool:
    planned = chunk_ids_from_worker_plan(run_root)
    if not planned:
        return False
    return all(chunk_worker_pass_status(run_root, cid) for cid in planned)


def _checkpoint_artifacts_ready(run_root: Path) -> Tuple[bool, Dict[str, Any]]:
    ckpt_dir = run_root / "lprod" / "checkpoint"
    ckpt_ok, ckpt_detail = verify_lprod_checkpoint(ckpt_dir)
    checks: Dict[str, Any] = {
        "lprod_checkpoint_pass": ckpt_ok,
        "lprod_checkpoint_detail": ckpt_detail,
    }
    if not ckpt_ok:
        return False, checks

    built_path = ckpt_dir / "built_metadata.json"
    built_meta: Dict[str, Any] = {}
    if built_path.is_file():
        try:
            built_meta = load_json(built_path)
        except (OSError, ValueError, json.JSONDecodeError):
            built_meta = {}

    p_count = int(built_meta.get("p_idx_aperture_count") or 0)
    if p_count <= 0:
        try:
            from v2_b3_rich_modal_lib import load_region_dof_bundle  # noqa: WPS433

            region_ctx = load_region_dof_bundle(ckpt_dir, built_meta, validate_aperture=True)
            p_count = int(region_ctx.get("p_idx_aperture_count") or 0)
        except Exception as exc:
            checks["region_dof_error"] = f"{type(exc).__name__}:{exc}"
            return False, checks

    checks["p_idx_aperture_count"] = p_count
    checks["active_dimension"] = int(built_meta.get("active_dimension") or 0)
    if p_count <= 0:
        return False, checks
    if int(built_meta.get("active_dimension") or 0) <= 0:
        return False, checks
    return True, checks


def assess_stale_running_repair(run_root: Path) -> Dict[str, Any]:
    """Evaluate whether a RUNNING manifest can be repaired to LPROD_CHECKPOINT_READY."""
    run_root = run_root.expanduser().resolve()
    manifest_path = run_root / "pipeline_run_manifest.json"
    manifest = load_json(manifest_path) if manifest_path.is_file() else {}
    summary = read_run_production_summary(run_root)
    previous_status = str(manifest.get("terminal_status") or "")
    out: Dict[str, Any] = {
        "run_root": str(run_root),
        "previous_status": previous_status,
        "eligible": False,
        "checks": {},
        "failures": [],
    }

    if previous_status not in REPAIRABLE_TERMINAL_STATUSES:
        out["failures"].append(f"terminal_status_not_repairable:{previous_status!r}")
        return out

    if is_run_usably_complete(summary):
        out["failures"].append("run_already_usably_complete")
        return out

    if _workers_complete(run_root):
        out["failures"].append("workers_already_complete_use_reconcile")
        return out

    active, active_detail = detect_active_worker_lock(run_root)
    out["checks"]["worker_lock_active"] = active
    out["checks"]["worker_lock_detail"] = active_detail
    if active:
        out["failures"].append(f"worker_activity_detected:{active_detail}")
        return out

    out["checks"]["scout_pass"] = _scout_pass(run_root)
    if not out["checks"]["scout_pass"]:
        out["failures"].append("scout_not_pass")

    out["checks"]["worker_plan_pass"] = _worker_plan_pass(run_root)
    if not out["checks"]["worker_plan_pass"]:
        out["failures"].append("worker_plan_not_pass")

    ckpt_ok, ckpt_checks = _checkpoint_artifacts_ready(run_root)
    out["checks"].update(ckpt_checks)
    if not ckpt_ok:
        out["failures"].append("checkpoint_artifacts_not_ready")

    if out["failures"]:
        return out

    out["eligible"] = True
    out["repaired_status"] = TERMINAL_CHECKPOINT_READY
    out["repair_reason"] = STALE_RUNNING_REPAIR_REASON
    return out


def promote_checkpoint_ready_terminal(
    run_root: Path,
    *,
    repair_reason: str,
    previous_status: Optional[str] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Set pipeline_run_manifest.terminal_status to LPROD_CHECKPOINT_READY."""
    run_root = run_root.expanduser().resolve()
    manifest_path = run_root / "pipeline_run_manifest.json"
    manifest = load_json(manifest_path) if manifest_path.is_file() else {}
    prev = str(previous_status if previous_status is not None else manifest.get("terminal_status") or "")
    prev_rank = terminal_status_rank(prev)
    checkpoint_rank = terminal_status_rank(TERMINAL_CHECKPOINT_READY)
    e2e_rank = terminal_status_rank(TERMINAL_E2E)
    completed_rank = terminal_status_rank(TERMINAL_PRODUCTION_COMPLETED)

    if prev_rank >= completed_rank:
        return {
            "status": "SKIP",
            "reason": "run_production_completed",
            "previous_status": prev,
        }

    if prev_rank >= e2e_rank:
        return {
            "status": "SKIP",
            "reason": "terminal_already_at_or_beyond_workers",
            "previous_status": prev,
        }

    if prev == TERMINAL_CHECKPOINT_READY:
        return {
            "status": "SKIP",
            "reason": "already_checkpoint_ready",
            "previous_status": prev,
            "repaired_status": TERMINAL_CHECKPOINT_READY,
        }

    assessment = assess_stale_running_repair(run_root)
    if prev in REPAIRABLE_TERMINAL_STATUSES:
        if not assessment.get("eligible"):
            return {
                "status": "FAIL",
                "previous_status": prev,
                "failures": list(assessment.get("failures") or []),
                "checks": assessment.get("checks") or {},
            }
    else:
        if is_run_usably_complete(read_run_production_summary(run_root)):
            return {"status": "SKIP", "reason": "run_complete", "previous_status": prev}
        if _workers_complete(run_root):
            quarantine_stale_downstream_artifacts(
                run_root,
                reason="promote_checkpoint_stale_downstream",
            )
        ckpt_ok, checks = _checkpoint_artifacts_ready(run_root)
        if not ckpt_ok or not _scout_pass(run_root) or not _worker_plan_pass(run_root):
            return {
                "status": "SKIP",
                "reason": "preconditions_not_met",
                "previous_status": prev,
                "checks": checks,
            }
        active, detail = detect_active_worker_lock(run_root)
        if active:
            return {
                "status": "SKIP",
                "reason": f"worker_activity:{detail}",
                "previous_status": prev,
            }
        assessment = {
            "checks": checks,
            "eligible": True,
        }

    repair_body = {
        "schema": "m4_stale_running_repair_v1",
        "previous_status": prev,
        "repaired_status": CHECKPOINT_TERMINAL_READY,
        "repair_reason": repair_reason,
        "repaired_utc": utc_now(),
        "checks": assessment.get("checks") or {},
    }
    if dry_run:
        repair_body["status"] = "DRY_RUN"
        return repair_body

    manifest["terminal_status"] = CHECKPOINT_TERMINAL_READY
    manifest["updated_utc"] = utc_now()
    stages = manifest.setdefault("stages", {})
    for key, status in (
        ("stage4_lprod_mesh", "PASS"),
        ("stage4_lprod_export", "PASS"),
        ("stage5_workers", "PLANNED_READY"),
        ("stage6_aggregate", "PLANNED_READY"),
    ):
        st = stages.setdefault(key, {})
        if str(st.get("status") or "") not in ("PASS",):
            st["status"] = status
        st["updated_utc"] = utc_now()
    manifest.pop("failure_reason", None)
    manifest["stale_running_repair"] = {
        "previous_status": prev,
        "repaired_status": CHECKPOINT_TERMINAL_READY,
        "repair_reason": repair_reason,
        "repaired_utc": repair_body["repaired_utc"],
    }
    write_json_atomic(manifest_path, manifest)
    write_json_atomic(run_root / STALE_RUNNING_REPAIR_JSON, repair_body)
    repair_body["status"] = "PASS"
    return repair_body


def maybe_promote_checkpoint_ready_terminal(
    run_root: Path,
    *,
    repair_reason: str = "checkpoint_stage_complete",
) -> bool:
    """Idempotent helper for pipeline resume paths after checkpoint is ready."""
    result = promote_checkpoint_ready_terminal(run_root, repair_reason=repair_reason)
    return result.get("status") in ("PASS", "SKIP")


def repair_stale_running_runs(
    *,
    run_roots: List[Path],
    dry_run: bool = False,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    repaired = 0
    failed = 0
    for run_root in run_roots:
        run_root = run_root.expanduser().resolve()
        assessment = assess_stale_running_repair(run_root)
        row = {
            "run_root": str(run_root),
            "previous_status": assessment.get("previous_status"),
            "eligible": bool(assessment.get("eligible")),
            "failures": list(assessment.get("failures") or []),
        }
        if not assessment.get("eligible"):
            row["action"] = (
                "skip_not_stale_running"
                if assessment.get("previous_status") not in REPAIRABLE_TERMINAL_STATUSES
                else "skip_not_eligible"
            )
            if row["action"] == "skip_not_eligible":
                failed += 1
            rows.append(row)
            continue
        result = promote_checkpoint_ready_terminal(
            run_root,
            repair_reason=STALE_RUNNING_REPAIR_REASON,
            previous_status=str(assessment.get("previous_status") or "RUNNING"),
            dry_run=dry_run,
        )
        row["repair"] = result
        row["action"] = "would_repair_stale_running" if dry_run else "repaired_stale_running"
        repaired += 1
        rows.append(row)
    return {
        "schema": "m4_stale_running_repair_report_v1",
        "generated_utc": utc_now(),
        "dry_run": dry_run,
        "repaired_count": repaired,
        "failed_count": failed,
        "runs": rows,
    }
