#!/usr/bin/env python3
"""Canonical M4 stage artifact paths — shape-agnostic pipeline contract."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from v2_b3_petsc_util import write_json_atomic  # noqa: E402

SCOUT_CHUNK_PREVIEW_JSON_REL = "lprod/worker_chunk_plan.preview.json"
SCOUT_CHUNK_PREVIEW_MD_REL = "lprod/worker_chunk_plan.preview.md"

SCOUT_TERMINAL_ARTIFACTS: Tuple[str, ...] = (
    "scout/discovery/density_result.json",
    "scout/density_zones.json",
    "lprod/lprod_target_plan.json",
    SCOUT_CHUNK_PREVIEW_JSON_REL,
    "scout/scout_result.json",
)

LPROD_CHECKPOINT_ARTIFACTS: Tuple[str, ...] = (
    "lprod/checkpoint/checkpoint_export_manifest.json",
    "lprod/lprod_execution_plan.json",
)

WORKER_PLAN_OUTPUT_ARTIFACTS: Tuple[str, ...] = (
    "lprod/worker_commands.json",
    "lprod/lprod_execution_plan.json",
)

WORKER_PLAN_ARTIFACTS: Tuple[str, ...] = (
    SCOUT_CHUNK_PREVIEW_JSON_REL,
    *WORKER_PLAN_OUTPUT_ARTIFACTS,
)

MESH_MANIFEST_SUFFIX = ".mesh_manifest.json"


def run_artifact_path(run_root: Path, rel: str) -> Path:
    return run_root / rel


def validate_artifacts(
    run_root: Path,
    rel_paths: Sequence[str],
    *,
    label: str,
) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    for rel in rel_paths:
        path = run_root / rel
        if not path.is_file():
            errors.append(f"missing {label} artifact: {rel}")
    return len(errors) == 0, errors


def validate_scout_terminal_artifacts(run_root: Path) -> Tuple[bool, List[str]]:
    return validate_artifacts(run_root, SCOUT_TERMINAL_ARTIFACTS, label="scout_terminal")


def validate_worker_plan_artifacts(run_root: Path) -> Tuple[bool, List[str]]:
    return validate_artifacts(run_root, WORKER_PLAN_ARTIFACTS, label="worker_plan")


def format_stage_contract_error(errors: List[str]) -> str:
    return "STAGE_ARTIFACT_CONTRACT_FAIL " + "; ".join(errors)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def write_worker_chunk_preview_artifacts(
    *,
    lprod_dir: Path,
    chunk_preview: Mapping[str, Any],
    md_renderer: Any,
) -> Path:
    """Write scout-owned chunk preview JSON + MD; verify JSON exists before returning."""
    json_path = lprod_dir / Path(SCOUT_CHUNK_PREVIEW_JSON_REL).name
    md_path = lprod_dir / Path(SCOUT_CHUNK_PREVIEW_MD_REL).name
    payload = _json_safe(dict(chunk_preview))
    write_json_atomic(json_path, payload)
    if not json_path.is_file() or json_path.stat().st_size < 2:
        raise RuntimeError(f"worker_chunk_plan.preview.json not written: {json_path}")
    md_path.write_text(md_renderer(payload), encoding="utf-8")
    return json_path


def format_scout_stage_contract_pass_line(
    *,
    target_count: int,
    chunk_count: int,
) -> str:
    return (
        f"SCOUT_STAGE_ARTIFACT_CONTRACT_PASS target_count={target_count} "
        f"chunk_count={chunk_count} worker_chunk_plan_json=present"
    )


def format_scout_stage_contract_fail_line(missing: Sequence[str]) -> str:
    return "SCOUT_STAGE_ARTIFACT_CONTRACT_FAIL missing=" + ",".join(missing)


def assert_scout_terminal_contract_or_raise(run_root: Path) -> Tuple[int, int]:
    """Validate scout terminal artifacts; return (target_count, chunk_count) from on-disk plans."""
    ok, errors = validate_scout_terminal_artifacts(run_root)
    if not ok:
        missing = [e.split(": ", 1)[-1] for e in errors if "artifact:" in e]
        raise RuntimeError(format_scout_stage_contract_fail_line(missing))

    target_plan = json.loads((run_root / "lprod" / "lprod_target_plan.json").read_text(encoding="utf-8"))
    chunk_plan = json.loads((run_root / SCOUT_CHUNK_PREVIEW_JSON_REL).read_text(encoding="utf-8"))
    target_count = len(target_plan.get("targets_hz") or [])
    chunk_count = len(chunk_plan.get("chunks") or [])
    if chunk_count <= 0:
        raise RuntimeError(format_scout_stage_contract_fail_line([SCOUT_CHUNK_PREVIEW_JSON_REL + ":empty_chunks"]))
    return target_count, chunk_count
