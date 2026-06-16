#!/usr/bin/env python3
"""Authoritative terminal_status promotion and cross-source consistency checks."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from v2_b3_m4_freeze_first_e2e_run import (  # noqa: E402
    AGG_STATUS_PASS,
    TERMINAL_E2E,
    promote_pipeline_terminal_status,
)
from v2_b3_m4_lhs_pool_bridge import read_run_production_summary  # noqa: E402
from v2_b3_m4_reuse_integrity_lib import (  # noqa: E402
    aggregate_artifact_contract_pass,
    read_manifest,
    terminal_status_rank,
)
from v2_b3_m4_worker_run_lib import load_json  # noqa: E402

TERMINAL_STATUS_INCONSISTENCY = "TERMINAL_STATUS_INCONSISTENCY"
AGG_PREVIEW_REL = "pipeline_run_manifest.m4_4_full_aggregation_preview.json"
AGG_RESULT_REL = "aggregation/aggregation_result.json"
FAILURE_RETENTION_REL = "cleanup/sample_failure_retention.json"


def _safe_terminal(path: Path, key: str = "terminal_status") -> str:
    if not path.is_file():
        return ""
    try:
        doc = load_json(path)
        return str(doc.get(key) or "")
    except (OSError, ValueError, json.JSONDecodeError):
        return ""


def collect_terminal_status_sources(run_root: Path) -> Dict[str, str]:
    run_root = run_root.expanduser().resolve()
    manifest = read_manifest(run_root)
    summary = read_run_production_summary(run_root)
    return {
        "manifest": str(manifest.get("terminal_status") or ""),
        "aggregation_preview": _safe_terminal(run_root / AGG_PREVIEW_REL),
        "aggregation_result": _safe_terminal(run_root / AGG_RESULT_REL),
        "production_summary": str(summary.get("terminal_status") or ""),
        "failure_retention": _safe_terminal(run_root / FAILURE_RETENTION_REL),
    }


def expected_terminal_after_aggregation(run_root: Path) -> Optional[str]:
    if not aggregate_artifact_contract_pass(run_root):
        return None
    agg_path = run_root / AGG_RESULT_REL
    if not agg_path.is_file():
        return None
    try:
        agg = load_json(agg_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if str(agg.get("status")) != AGG_STATUS_PASS:
        return None
    if not bool(agg.get("final_aggregation_ready")):
        return None
    return TERMINAL_E2E


def promote_after_aggregation_pass(run_root: Path) -> Dict[str, Any]:
    """Persist workers+aggregation terminal on the authoritative pipeline manifest."""
    run_root = run_root.expanduser().resolve()
    expected = expected_terminal_after_aggregation(run_root)
    out: Dict[str, Any] = {
        "promoted": False,
        "expected_terminal": expected,
        "previous_manifest_terminal": str(read_manifest(run_root).get("terminal_status") or ""),
    }
    if not expected:
        out["reason"] = "aggregation_contract_not_ready"
        return out

    prev = out["previous_manifest_terminal"]
    if terminal_status_rank(prev) >= terminal_status_rank(expected):
        out["reason"] = "manifest_already_at_or_beyond_expected"
        out["manifest_terminal"] = prev
        return out

    promote_pipeline_terminal_status(run_root, terminal_status=expected, aggregation_status=AGG_STATUS_PASS)
    out["promoted"] = True
    out["manifest_terminal"] = expected
    out["reason"] = "promoted_after_aggregation_pass"
    return out


def check_terminal_status_consistency(
    run_root: Path,
    *,
    context: str = "",
) -> Tuple[bool, List[str]]:
    """Return (ok, errors). Fail if authoritative sources disagree after aggregation PASS."""
    run_root = run_root.expanduser().resolve()
    sources = collect_terminal_status_sources(run_root)
    errors: List[str] = []
    manifest = sources["manifest"]
    expected = expected_terminal_after_aggregation(run_root)

    if expected and manifest and terminal_status_rank(manifest) < terminal_status_rank(expected):
        errors.append(
            f"{TERMINAL_STATUS_INCONSISTENCY} "
            f"aggregation={expected} manifest={manifest}"
            + (f" context={context}" if context else "")
        )

    preview = sources["aggregation_preview"]
    if expected and preview and preview != expected:
        errors.append(
            f"{TERMINAL_STATUS_INCONSISTENCY} "
            f"aggregation={expected} preview={preview}"
            + (f" context={context}" if context else "")
        )

    if manifest and sources["production_summary"] and manifest != sources["production_summary"]:
        errors.append(
            f"{TERMINAL_STATUS_INCONSISTENCY} "
            f"manifest={manifest} production_summary={sources['production_summary']}"
        )

    retention = sources["failure_retention"]
    if expected and retention and terminal_status_rank(retention) < terminal_status_rank(manifest):
        if manifest and retention != manifest:
            errors.append(
                f"{TERMINAL_STATUS_INCONSISTENCY} "
                f"manifest={manifest} failure_retention={retention}"
            )

    return len(errors) == 0, errors


def reconcile_terminal_status_before_freeze(run_root: Path) -> Dict[str, Any]:
    """Promote manifest if aggregation PASS but terminal still checkpoint-ready."""
    promote = promote_after_aggregation_pass(run_root)
    ok, errors = check_terminal_status_consistency(run_root, context="pre_freeze")
    return {
        "promote": promote,
        "consistent": ok,
        "errors": errors,
    }
