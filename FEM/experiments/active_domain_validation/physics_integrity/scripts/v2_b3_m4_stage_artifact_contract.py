#!/usr/bin/env python3
"""Canonical M4 stage artifact paths — shape-agnostic pipeline contract."""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

SCOUT_TERMINAL_ARTIFACTS: Tuple[str, ...] = (
    "scout/density_zones.json",
    "lprod/lprod_target_plan.json",
    "lprod/worker_chunk_plan.preview.json",
    "scout/scout_result.json",
)

LPROD_CHECKPOINT_ARTIFACTS: Tuple[str, ...] = (
    "lprod/checkpoint/checkpoint_export_manifest.json",
    "lprod/lprod_execution_plan.json",
)

WORKER_PLAN_ARTIFACTS: Tuple[str, ...] = (
    "lprod/worker_chunk_plan.preview.json",
    "lprod/worker_commands.json",
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
