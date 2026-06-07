#!/usr/bin/env python3
"""Read-only legacy/large-artifact audit for M4 project storage cleanup (no deletions)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_m4_worker_run_lib import detect_repo_root, rel, utc_now  # noqa: E402
from v2_b3_petsc_util import write_json_atomic  # noqa: E402

TOOL_VERSION = "m4_legacy_storage_audit_v1"

# Scripts referenced by current M4 production path (non-exhaustive guard set)
M4_ACTIVE_SCRIPTS = {
    "v2_b3_m4_run_one_sample.py",
    "v2_b3_m4_lhs_production_batch.py",
    "v2_b3_m4_aggregate_worker_results.py",
    "v2_b3_m4_freeze_first_e2e_run.py",
    "v2_b3_m4_lhs_pool_bridge.py",
    "v2_b3_m4_modal_surrogate_lib.py",
    "build_m4_rom_from_completed_fom.py",
    "run_m4_rom_compare.py",
    "run_m4_production_pipeline.py",
    "compact_completed_m4_runs.py",
}

LEGACY_SCRIPT_HINTS = (
    "v2_b3_m3_orchestrator",
    "v2_b3_lhs_orchestrator",
    "v2_b3_run_coarse_scout",
    "v2_b3_coarse_mesh_scout",
    "v2_b3_frequency_coarse_planner",
    "rich_modal",
    "checkpoint_.*smoke",
    "operator_.*audit",
    "run_v2_B3_trace",
    "run_v2_cleaned_mass",
)

LARGE_FILE_THRESHOLD_BYTES = 50 * 1024 * 1024

CLASSIFICATIONS = (
    "SAFE_TO_REMOVE_FROM_WORKTREE",
    "KEEP_FOR_HISTORY",
    "MOVE_TO_LEGACY_ARCHIVE",
    "UNKNOWN_REQUIRES_REVIEW",
)


def _size(path: Path) -> int:
    try:
        if path.is_file():
            return path.stat().st_size
        total = 0
        for p in path.rglob("*"):
            if p.is_file() and not p.is_symlink():
                total += p.stat().st_size
        return total
    except OSError:
        return 0


def _scan_large_files(root: Path, *, threshold: int) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []
    skip_parts = {".git", "__pycache__", "node_modules"}
    for p in root.rglob("*"):
        if any(part in skip_parts for part in p.parts):
            continue
        if not p.is_file() or p.is_symlink():
            continue
        try:
            sz = p.stat().st_size
        except OSError:
            continue
        if sz >= threshold:
            hits.append({"path": rel(p, repo_root=root), "bytes": sz})
    hits.sort(key=lambda x: -int(x["bytes"]))
    return hits


def _classify_path(path_str: str) -> str:
    p = path_str.replace("\\", "/")
    if p.startswith("FEM/SORTING/"):
        return "SAFE_TO_REMOVE_FROM_WORKTREE"
    if "/snapshots/" in p and p.startswith("ROM/"):
        return "MOVE_TO_LEGACY_ARCHIVE"
    if p.endswith("reduced_basis.npz"):
        return "KEEP_FOR_HISTORY"
    if "pipeline_runs/guitars/" in p:
        return "MOVE_TO_LEGACY_ARCHIVE"
    if "solver_benchmarks" in p or "/diagnostics/" in p:
        return "MOVE_TO_LEGACY_ARCHIVE"
    if "config_overlays" in p:
        return "SAFE_TO_REMOVE_FROM_WORKTREE"
    return "UNKNOWN_REQUIRES_REVIEW"


def _audit_scripts(scripts_dir: Path, *, repo_root: Path) -> List[Dict[str, Any]]:
    import re

    rows: List[Dict[str, Any]] = []
    for path in sorted(scripts_dir.glob("*.py")):
        name = path.name
        if name in M4_ACTIVE_SCRIPTS:
            decision = "KEEP_FOR_HISTORY"
            reason = "M4 active entry or library"
        elif any(re.search(pat, name) for pat in LEGACY_SCRIPT_HINTS):
            decision = "MOVE_TO_LEGACY_ARCHIVE"
            reason = "Legacy M2/M3/B3 diagnostic; superseded by M4"
        elif name.startswith("v2_b3_m4_"):
            decision = "KEEP_FOR_HISTORY"
            reason = "M4 module"
        elif name.startswith("audit_") or name.startswith("compare_"):
            decision = "KEEP_FOR_HISTORY"
            reason = "ROM/M4 audit tooling"
        else:
            decision = "UNKNOWN_REQUIRES_REVIEW"
            reason = "Not in active set; manual review"
        rows.append(
            {
                "path": rel(path, repo_root=repo_root),
                "classification": decision,
                "reason": reason,
            }
        )
    return rows


def _audit_tree_sizes(repo_root: Path) -> List[Dict[str, Any]]:
    candidates: Tuple[Tuple[str, str], ...] = (
        ("pipeline_runs/guitars", "ACTIVE_REQUIRED"),
        ("ROM/classic", "ACTIVE_REQUIRED"),
        ("FEM/SORTING", "LEGACY_SAFE_TO_REMOVE"),
        ("FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/batches", "ARCHIVE_RECOMMENDED"),
        ("FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/config_overlays", "SAFE_TO_DELETE_AFTER_COMPLETION"),
        ("FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/specs/generated", "ACTIVE_OPTIONAL"),
        ("FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/index", "ACTIVE_OPTIONAL"),
        ("FEM/experiments/active_domain_validation/solver_benchmarks", "LEGACY_SAFE_TO_REMOVE"),
    )
    rows: List[Dict[str, Any]] = []
    for rel_dir, default_class in candidates:
        path = repo_root / rel_dir
        if not path.exists():
            rows.append(
                {
                    "path": rel_dir,
                    "exists": False,
                    "bytes": 0,
                    "classification": default_class,
                }
            )
            continue
        rows.append(
            {
                "path": rel_dir,
                "exists": True,
                "bytes": _size(path),
                "classification": default_class,
            }
        )
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--large-file-mb", type=float, default=50.0)
    args = parser.parse_args(argv)

    repo_root = detect_repo_root(SCRIPT_DIR)
    scripts_dir = SCRIPT_DIR
    threshold = int(float(args.large_file_mb) * 1024 * 1024)

    report = {
        "schema": "m4_legacy_storage_audit_v1",
        "tool_version": TOOL_VERSION,
        "generated_utc": utc_now(),
        "repo_root": str(repo_root),
        "tree_sizes": _audit_tree_sizes(repo_root),
        "large_files": _scan_large_files(repo_root, threshold=threshold)[:200],
        "script_classifications": _audit_scripts(scripts_dir, repo_root=repo_root),
        "notes": [
            "No files deleted by this audit.",
            "Worktree classifications require operator approval before removal.",
            "pipeline_runs/guitars may be absent in dev checkout; sizes are VM-specific.",
        ],
    }

    out = args.out_json or (
        repo_root
        / "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/index/legacy_storage_audit.json"
    )
    write_json_atomic(out, report)
    print(f"wrote {rel(out, repo_root=repo_root)}", flush=True)
    print(f"large_files>={args.large_file_mb}MB: {len(report['large_files'])}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
