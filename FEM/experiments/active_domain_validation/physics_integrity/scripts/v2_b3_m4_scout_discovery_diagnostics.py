#!/usr/bin/env python3
"""Durable scout discovery failure diagnostics (per-sample run tree only)."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from v2_b3_petsc_util import write_json_atomic

SCOUT_DISCOVERY_REL = "scout/discovery"
DENSITY_RESULT_NAME = "density_result.json"
DENSITY_MD_NAME = "density_result.md"
RETENTION_SCHEMA = "m4_scout_discovery_failure_retention_v1"

SCOUT_DISCOVERY_DIAGNOSTIC_FILES = frozenset(
    {
        DENSITY_RESULT_NAME,
        DENSITY_MD_NAME,
    }
)


def discovery_dir_for_run(run_root: Path) -> Path:
    return Path(run_root).resolve() / "scout" / "discovery"


def density_result_path(run_root: Path) -> Path:
    return discovery_dir_for_run(run_root) / DENSITY_RESULT_NAME


def slim_density_result_body(body: Mapping[str, Any]) -> Dict[str, Any]:
    """Drop heavy per-target solver rows; keep spacing summaries and intrinsic fields."""
    slim: Dict[str, Any] = dict(body)
    slim_spacings: List[Dict[str, Any]] = []
    for row in body.get("spacings") or []:
        if not isinstance(row, dict):
            continue
        slim_row = {k: v for k, v in row.items() if k != "per_target"}
        slim_spacings.append(slim_row)
    slim["spacings"] = slim_spacings
    retention = dict(slim.get("failure_retention") or {})
    retention.update(
        {
            "schema": RETENTION_SCHEMA,
            "retained_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "stripped_fields": ["spacings[].per_target"],
            "not_reused_by_next_sample": True,
        }
    )
    slim["failure_retention"] = retention
    return slim


def scout_discovery_is_diagnostic_only(run_root: Path) -> bool:
    """True when scout/discovery contains only lightweight failure JSON (not heavy solver dumps)."""
    disc = discovery_dir_for_run(run_root)
    if not disc.is_dir():
        return False
    for path in disc.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(disc)
        if len(rel.parts) != 1:
            return False
        if path.name not in SCOUT_DISCOVERY_DIAGNOSTIC_FILES:
            return False
    density = disc / DENSITY_RESULT_NAME
    if density.is_file():
        try:
            body = json.loads(density.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        for row in body.get("spacings") or []:
            if isinstance(row, dict) and row.get("per_target"):
                return False
    return True


def preserve_scout_discovery_failure_diagnostics(
    run_root: Path,
    *,
    reason: Optional[str] = None,
) -> List[str]:
    """Slim and retain scout/discovery JSON before failed-sample heavy cleanup."""
    run_root = Path(run_root).resolve()
    disc = discovery_dir_for_run(run_root)
    preserved: List[str] = []
    density = disc / DENSITY_RESULT_NAME
    if density.is_file():
        try:
            body = json.loads(density.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            body = {"status": "FAIL", "failure_reason": "density_result_unreadable"}
        slim = slim_density_result_body(body)
        if reason:
            slim.setdefault("failure_retention", {})["preserve_reason"] = reason
        write_json_atomic(density, slim)
        preserved.append(str(density))
    for path in list(disc.iterdir()) if disc.is_dir() else []:
        if path.is_file() and path.name not in SCOUT_DISCOVERY_DIAGNOSTIC_FILES:
            try:
                path.unlink()
            except OSError:
                pass
        elif path.is_dir():
            try:
                import shutil

                shutil.rmtree(path)
            except OSError:
                pass
    return preserved


def ensure_scout_discovery_failure_artifacts(
    run_root: Path,
    *,
    reason: str = "scout_discovery_stage_failed",
) -> bool:
    """Ensure slim density_result.json exists in run tree after scout discovery failure."""
    run_root = Path(run_root).resolve()
    density = density_result_path(run_root)
    if not density.is_file():
        fallback = {
            "status": "FAIL",
            "failure_reason": f"missing_density_result_at_{reason}",
            "failure_retention": {
                "schema": RETENTION_SCHEMA,
                "preserve_reason": reason,
                "not_reused_by_next_sample": True,
            },
        }
        disc = discovery_dir_for_run(run_root)
        disc.mkdir(parents=True, exist_ok=True)
        write_json_atomic(density, fallback)
        return False
    preserve_scout_discovery_failure_diagnostics(run_root, reason=reason)
    return True
