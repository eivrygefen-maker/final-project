#!/usr/bin/env python3
"""Production batch control signals (graceful stop between samples)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from v2_b3_m4_worker_run_lib import detect_repo_root, utc_now  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
STOP_FILENAME = "STOP_AFTER_CURRENT_SAMPLE"


def production_control_dir(repo_root: Path) -> Path:
    return (
        repo_root
        / "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/control"
    )


def stop_after_current_path(repo_root: Path) -> Path:
    return production_control_dir(repo_root) / STOP_FILENAME


def is_stop_after_current_requested(repo_root: Path) -> bool:
    return stop_after_current_path(repo_root).is_file()


def request_stop_after_current(repo_root: Path) -> Path:
    path = stop_after_current_path(repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"requested_utc={utc_now()}\n"
        "action=stop_after_current_sample_completes\n",
        encoding="utf-8",
    )
    return path


def clear_stop_after_current(repo_root: Path) -> bool:
    path = stop_after_current_path(repo_root)
    if path.is_file():
        path.unlink()
        return True
    return False
