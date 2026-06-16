#!/usr/bin/env python3
"""Strict stage reuse contracts — terminal status must agree with durable artifacts."""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from v2_b3_m4_freeze_first_e2e_run import (  # noqa: E402
    AGG_STATUS_PASS,
    CHECKPOINT_TERMINAL_READY,
    SCOUT_TERMINAL_READY,
    TERMINAL_E2E,
    freeze_outputs_present,
)
from v2_b3_m4_production_freeze import (  # noqa: E402
    TERMINAL_PRODUCTION_COMPLETED,
    production_freeze_complete,
)
from v2_b3_m4_stage_artifact_contract import (  # noqa: E402
    SCOUT_CHUNK_PREVIEW_JSON_REL,
    SCOUT_TERMINAL_ARTIFACTS,
    validate_scout_terminal_artifacts,
    WORKER_PLAN_OUTPUT_ARTIFACTS,
)
from v2_b3_m4_worker_run_lib import (  # noqa: E402
    chunk_ids_from_worker_plan,
    chunk_worker_pass_status,
    load_json,
    utc_now,
)
from v2_b3_petsc_util import write_json_atomic  # noqa: E402

REUSE_INTEGRITY_FAIL = "REUSE_INTEGRITY_FAIL"
QUARANTINE_SCHEMA = "m4_stale_reuse_quarantine_v1"

# Scout Stage 3 owns the chunk preview JSON; worker_plan consumes it and must not delete it.
WORKER_PLAN_REL_PATHS: Tuple[str, ...] = WORKER_PLAN_OUTPUT_ARTIFACTS + (
    "lprod/aggregation_plan.json",
    "lprod/lprod_mesh_checkpoint_readiness.json",
)

WORKER_PLAN_OPTIONAL_REL: Tuple[str, ...] = (
    "lprod/lprod_execution_plan.md",
    "lprod/worker_commands.md",
    "lprod/aggregation_plan.md",
)

DOWNSTREAM_WORKER_RESULT_FILES: Tuple[str, ...] = (
    "worker_result.json",
    "solver_result.json",
    "worker_manifest.json",
)

TERMINAL_RANK: Dict[str, int] = {
    "": 0,
    "FAIL": 0,
    "PLANNED": 1,
    "RUNNING": 1,
    SCOUT_TERMINAL_READY: 2,
    CHECKPOINT_TERMINAL_READY: 3,
    TERMINAL_E2E: 4,
    TERMINAL_PRODUCTION_COMPLETED: 5,
}

STAGE_MIN_TERMINAL_RANK: Dict[str, int] = {
    "scout": 2,
    "worker_plan": 2,
    "checkpoint": 3,
    "workers": 4,
    "aggregate": 4,
    "freeze": 4,
}


def terminal_status_rank(terminal_status: str) -> int:
    return TERMINAL_RANK.get(str(terminal_status or ""), 0)


def manifest_path(run_root: Path) -> Path:
    return run_root / "pipeline_run_manifest.json"


def read_manifest(run_root: Path) -> Dict[str, Any]:
    path = manifest_path(run_root)
    if not path.is_file():
        return {}
    try:
        doc = load_json(path)
        return doc if isinstance(doc, dict) else {}
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def format_integrity_fail(
    *,
    stage: str,
    reason: str,
    terminal_status: str = "",
    detail: str = "",
) -> str:
    parts = [REUSE_INTEGRITY_FAIL, f"stage={stage}", f"reason={reason}"]
    if terminal_status:
        parts.append(f"terminal_status={terminal_status}")
    if detail:
        parts.append(detail)
    return " ".join(parts)


def scout_artifact_contract_pass(run_root: Path) -> bool:
    manifest = read_manifest(run_root)
    if str(manifest.get("terminal_status")) == SCOUT_TERMINAL_READY:
        ok, _ = validate_scout_terminal_artifacts(run_root)
        return ok
    st3 = (manifest.get("stages") or {}).get("stage3_zones_plan") or {}
    if str(st3.get("status")) != "PASS":
        return False
    ok, _ = validate_scout_terminal_artifacts(run_root)
    return ok


def worker_plan_artifact_contract_pass(run_root: Path) -> bool:
    if not (run_root / SCOUT_CHUNK_PREVIEW_JSON_REL).is_file():
        return False
    for rel in WORKER_PLAN_REL_PATHS:
        if not (run_root / rel).is_file():
            return False
    worker_root = run_root / "worker_results"
    if not worker_root.is_dir():
        return False
    try:
        planned = chunk_ids_from_worker_plan(run_root)
    except (OSError, ValueError, json.JSONDecodeError, KeyError, TypeError):
        return False
    if not planned:
        return False
    for cid in planned:
        if not (worker_root / cid / "chunk_targets.json").is_file():
            return False
    return True


def checkpoint_artifact_contract_pass(run_root: Path, *, production_mode: bool = False) -> bool:
    manifest = read_manifest(run_root)
    terminal = str(manifest.get("terminal_status") or "")
    if production_mode and terminal_status_rank(terminal) >= terminal_status_rank(
        TERMINAL_PRODUCTION_COMPLETED
    ):
        ck = run_root / "lprod" / "checkpoint" / "checkpoint_export_manifest.json"
        if not ck.is_file():
            return False
        try:
            data = load_json(ck)
            return bool(data.get("export_pass")) or str(data.get("status")) == "PASS"
        except (OSError, ValueError, json.JSONDecodeError):
            return False

    ck = run_root / "lprod" / "checkpoint" / "checkpoint_export_manifest.json"
    if not ck.is_file():
        return False
    try:
        data = load_json(ck)
        export_ok = bool(data.get("export_pass")) or str(data.get("status")) == "PASS"
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if not export_ok:
        return False
    if not (run_root / "lprod" / "checkpoint").is_dir():
        return False
    if production_mode:
        from v2_b3_m4_production_contracts import validate_post_export_region_dof_contract  # noqa: WPS433

        core = run_root / "lprod" / "resolved_core_config.json"
        contract_errors = validate_post_export_region_dof_contract(
            run_root / "lprod" / "checkpoint",
            core_config_path=core if core.is_file() else None,
        )
        if contract_errors:
            return False
    return True


def workers_artifact_contract_pass(run_root: Path) -> bool:
    try:
        planned = chunk_ids_from_worker_plan(run_root)
    except (OSError, ValueError, json.JSONDecodeError, KeyError, TypeError):
        return False
    if not planned:
        return False
    return all(chunk_worker_pass_status(run_root, cid) for cid in planned)


def aggregate_artifact_contract_pass(run_root: Path) -> bool:
    agg_path = run_root / "aggregation" / "aggregation_result.json"
    if not agg_path.is_file():
        return False
    try:
        agg = load_json(agg_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if str(agg.get("status")) != AGG_STATUS_PASS:
        return False
    if not bool(agg.get("final_aggregation_ready")):
        return False
    catalog = run_root / "aggregation" / "modes_catalog.jsonl"
    if not catalog.is_file():
        return False
    try:
        text = catalog.read_text(encoding="utf-8").strip()
    except OSError:
        return False
    return bool(text)


def freeze_artifact_contract_pass(
    run_root: Path,
    *,
    production_mode: bool = False,
) -> bool:
    if production_mode:
        return production_freeze_complete(run_root)
    return freeze_outputs_present(run_root)


def _manifest_stage_status(manifest: Dict[str, Any], stage_key: str) -> str:
    return str((manifest.get("stages") or {}).get(stage_key, {}).get("status") or "")


def terminal_integrity_for_stage(
    stage: str,
    *,
    manifest: Dict[str, Any],
    production_mode: bool,
    artifact_pass: bool,
) -> Tuple[bool, str]:
    if not artifact_pass:
        return True, ""

    terminal = str(manifest.get("terminal_status") or "")
    rank = terminal_status_rank(terminal)
    min_rank = STAGE_MIN_TERMINAL_RANK.get(stage, 0)

    if stage == "freeze":
        if production_mode:
            if terminal != TERMINAL_PRODUCTION_COMPLETED:
                return False, "terminal_status_incompatible"
        return True, ""

    if stage in ("workers", "aggregate"):
        if rank < terminal_status_rank(TERMINAL_E2E):
            return False, "terminal_status_incompatible"
        return True, ""

    if stage == "checkpoint":
        if rank < terminal_status_rank(CHECKPOINT_TERMINAL_READY):
            st4_mesh = _manifest_stage_status(manifest, "stage4_lprod_mesh")
            st4_export = _manifest_stage_status(manifest, "stage4_lprod_export")
            if st4_mesh == "PASS" and st4_export == "PASS":
                return True, ""
            return False, "terminal_status_incompatible"
        return True, ""

    if stage in ("scout", "worker_plan"):
        if rank >= terminal_status_rank(SCOUT_TERMINAL_READY):
            return True, ""
        if rank < terminal_status_rank(SCOUT_TERMINAL_READY) and terminal not in (
            "",
            "RUNNING",
            "PLANNED",
            "FAIL",
        ):
            return False, "terminal_status_incompatible"
        return True, ""

    if rank < min_rank:
        return False, "terminal_status_incompatible"
    return True, ""


def assess_stage_with_integrity(
    stage: str,
    run_root: Path,
    *,
    production_mode: bool = False,
) -> Dict[str, Any]:
    manifest = read_manifest(run_root)
    terminal = str(manifest.get("terminal_status") or "")

    artifact_fns = {
        "scout": lambda: scout_artifact_contract_pass(run_root),
        "worker_plan": lambda: worker_plan_artifact_contract_pass(run_root),
        "checkpoint": lambda: checkpoint_artifact_contract_pass(
            run_root, production_mode=production_mode
        ),
        "workers": lambda: workers_artifact_contract_pass(run_root),
        "aggregate": lambda: aggregate_artifact_contract_pass(run_root),
        "freeze": lambda: freeze_artifact_contract_pass(
            run_root, production_mode=production_mode
        ),
    }
    artifact_pass = artifact_fns[stage]()
    integrity_ok, reason = terminal_integrity_for_stage(
        stage,
        manifest=manifest,
        production_mode=production_mode,
        artifact_pass=artifact_pass,
    )

    out: Dict[str, Any] = {
        "artifact_pass": artifact_pass,
        "terminal_status": terminal,
    }

    if artifact_pass and integrity_ok:
        out["pass"] = True
        out["reuse_status"] = "PASS_reuse"
        return out

    if artifact_pass and not integrity_ok:
        out["pass"] = False
        out["reuse_status"] = REUSE_INTEGRITY_FAIL
        out["integrity_error"] = format_integrity_fail(
            stage=stage,
            reason=reason or "terminal_status_incompatible",
            terminal_status=terminal,
        )
        return out

    out["pass"] = False
    if run_root.is_dir():
        out["reuse_status"] = "resume_possible"
    else:
        out["reuse_status"] = "planned_new"
    return out


def assess_stages_with_integrity(
    run_root: Path,
    *,
    production_mode: bool = False,
) -> Dict[str, Dict[str, Any]]:
    stages = ("scout", "worker_plan", "checkpoint", "workers", "aggregate", "freeze")
    return {
        name: assess_stage_with_integrity(name, run_root, production_mode=production_mode)
        for name in stages
    }


def collect_integrity_failures(stages: Dict[str, Dict[str, Any]]) -> List[str]:
    failures: List[str] = []
    for name, st in stages.items():
        err = st.get("integrity_error")
        if err:
            failures.append(str(err))
        elif st.get("reuse_status") == REUSE_INTEGRITY_FAIL:
            failures.append(
                format_integrity_fail(
                    stage=name,
                    reason="terminal_status_incompatible",
                    terminal_status=str(st.get("terminal_status") or ""),
                )
            )
    return failures


def has_downstream_stale_evidence(run_root: Path) -> bool:
    terminal = str(read_manifest(run_root).get("terminal_status") or "")
    if terminal_status_rank(terminal) >= terminal_status_rank(TERMINAL_E2E):
        return False
    if workers_artifact_contract_pass(run_root):
        return True
    if aggregate_artifact_contract_pass(run_root):
        return True
    agg_dir = run_root / "aggregation"
    if agg_dir.is_dir() and any(agg_dir.iterdir()):
        return True
    worker_root = run_root / "worker_results"
    if not worker_root.is_dir():
        return False
    for chunk_dir in worker_root.iterdir():
        if not chunk_dir.is_dir():
            continue
        for fname in DOWNSTREAM_WORKER_RESULT_FILES:
            if (chunk_dir / fname).is_file():
                return True
    return False


def quarantine_stale_downstream_artifacts(
    run_root: Path,
    *,
    reason: str,
    keep_chunk_targets: bool = True,
) -> Dict[str, Any]:
    """Move downstream worker/aggregation PASS evidence out of the reuse path."""
    run_root = run_root.expanduser().resolve()
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    quarantine_root = run_root / "cleanup" / "stale_reuse_quarantine" / stamp
    quarantine_root.mkdir(parents=True, exist_ok=True)
    moved: List[str] = []

    agg_dir = run_root / "aggregation"
    if agg_dir.is_dir():
        try:
            has_content = any(agg_dir.iterdir())
        except OSError:
            has_content = False
        if has_content:
            dest = quarantine_root / "aggregation"
            shutil.move(str(agg_dir), str(dest))
            moved.append("aggregation")

    worker_root = run_root / "worker_results"
    if worker_root.is_dir():
        for chunk_dir in sorted(worker_root.iterdir()):
            if not chunk_dir.is_dir():
                continue
            for fname in DOWNSTREAM_WORKER_RESULT_FILES:
                src = chunk_dir / fname
                if not src.is_file():
                    continue
                dest_dir = quarantine_root / "worker_results" / chunk_dir.name
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dest_dir / fname))
                moved.append(f"worker_results/{chunk_dir.name}/{fname}")

    manifest = read_manifest(run_root)
    if manifest:
        stages = manifest.setdefault("stages", {})
        for key in ("stage5_workers", "stage6_aggregate"):
            st = stages.setdefault(key, {})
            if str(st.get("status") or "") == "PASS":
                st["status"] = "PLANNED_READY"
            st["updated_utc"] = utc_now()
            st.pop("failure_reason", None)
        manifest["updated_utc"] = utc_now()
        write_json_atomic(manifest_path(run_root), manifest)

    report = {
        "schema": QUARANTINE_SCHEMA,
        "reason": reason,
        "quarantined_utc": utc_now(),
        "moved_paths": moved,
        "keep_chunk_targets": keep_chunk_targets,
    }
    write_json_atomic(quarantine_root / "quarantine_manifest.json", report)
    return report


def remove_stale_worker_plan_outputs(run_root: Path) -> List[str]:
    """Remove partial worker_plan outputs only — never scout-owned chunk preview JSON."""
    if worker_plan_artifact_contract_pass(run_root):
        return []
    removed: List[str] = []
    for rel in WORKER_PLAN_REL_PATHS + WORKER_PLAN_OPTIONAL_REL:
        path = run_root / rel
        if path.is_file():
            path.unlink()
            removed.append(rel)
    return removed


def repair_inconsistent_reuse_state(
    run_root: Path,
    *,
    production_mode: bool = False,
) -> Dict[str, Any]:
    """Auto-quarantine downstream stale evidence when terminal rank is below workers-complete."""
    run_root = run_root.expanduser().resolve()
    manifest = read_manifest(run_root)
    terminal = str(manifest.get("terminal_status") or "")
    rank = terminal_status_rank(terminal)
    out: Dict[str, Any] = {
        "repaired": False,
        "terminal_status": terminal,
        "terminal_rank": rank,
        "actions": [],
    }

    if rank >= terminal_status_rank(TERMINAL_E2E):
        return out

    if has_downstream_stale_evidence(run_root):
        report = quarantine_stale_downstream_artifacts(
            run_root,
            reason="repair_inconsistent_reuse_state",
        )
        out["repaired"] = True
        out["actions"].append("quarantine_downstream")
        out["quarantine"] = report

    removed = remove_stale_worker_plan_outputs(run_root)
    if removed:
        out["repaired"] = True
        out["actions"].append("removed_stale_worker_plan_outputs")
        out["removed_worker_plan_outputs"] = removed

    stages = assess_stages_with_integrity(run_root, production_mode=production_mode)
    failures = collect_integrity_failures(stages)
    out["remaining_integrity_failures"] = failures
    return out


def earliest_rerun_stage(stages: Dict[str, Dict[str, Any]]) -> Optional[str]:
    order = ("scout", "worker_plan", "checkpoint", "workers", "aggregate", "freeze")
    for name in order:
        st = stages.get(name) or {}
        if not st.get("pass"):
            return name
    return None
