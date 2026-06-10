#!/usr/bin/env python3
"""Minimal durable-data compaction for completed ROM production runs (no FEM rerun)."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compact_completed_m4_runs import (  # noqa: E402
    HEAVY_ARCHIVE_REL_DIRS,
    HEAVY_ARCHIVE_REL_FILES,
    guitars_root,
)
from v2_b3_m4_lhs_pool_bridge import (  # noqa: E402
    AGG_PASS,
    is_run_usably_complete,
    read_run_production_summary,
)
from v2_b3_m4_mesh_profile_lib import (  # noqa: E402
    DURABLE_VALIDATION_INPUT_REL,
    MESH_PROFILE_ROM,
    VALIDATION_INPUT_PACKAGE_REL,
)
from v2_b3_m4_physics_identity_lib import (  # noqa: E402
    PHYSICS_IDENTITY_MANIFEST,
    count_forbidden_heavy_artifacts,
    validate_physics_identity_manifest,
)
from v2_b3_m4_sample_cleanup_barrier import (  # noqa: E402
    BARRIER_MANIFEST_REL,
    collect_shared_sample_artifact_paths,
    verify_success_durable_outputs,
)
from v2_b3_m4_worker_run_lib import detect_repo_root, load_json, rel, utc_now  # noqa: E402
from v2_b3_petsc_util import write_json_atomic  # noqa: E402

TOOL_VERSION = "m4_minimal_rom_durable_compaction_v1"
MANIFEST_SCHEMA = "m4_minimal_rom_durable_compaction_manifest_v1"

MINIMAL_ROM_RETAIN_REQUIRED_FILES: Tuple[str, ...] = (
    "aggregation/aggregation_result.json",
    "aggregation/modes_catalog.jsonl",
    "aggregation/modes_catalog_deduped.jsonl",
    "aggregation/mode_provenance.jsonl",
    "aggregation/modes_summary.json",
    "aggregation/runtime_summary.json",
    "freeze/freeze_manifest.json",
    "freeze/physics_identity_manifest.json",
    "sample/sample_input.json",
    "cleanup/sample_cleanup_barrier.json",
    "pipeline_run_manifest.json",
    "m4_sample_runtime_provenance.json",
)

MINIMAL_ROM_RETAIN_OPTIONAL_FILES: Tuple[str, ...] = (
    "aggregation/warnings_and_failures.json",
    "freeze/artifact_index.json",
    "freeze/sample_e2e_run_manifest.json",
    "sample/sample_resolved_config_manifest.json",
    "compaction/compaction_manifest.json",
    f"compaction/{MANIFEST_SCHEMA}.json",
)

MINIMAL_ROM_RETAIN_FILES: Tuple[str, ...] = (
    MINIMAL_ROM_RETAIN_REQUIRED_FILES + MINIMAL_ROM_RETAIN_OPTIONAL_FILES
)

MINIMAL_ROM_RETAIN_VALIDATION_FILES: Tuple[str, ...] = (
    f"{VALIDATION_INPUT_PACKAGE_REL}/validation_input_manifest.json",
    f"{VALIDATION_INPUT_PACKAGE_REL}/target_plan.json",
)

MINIMAL_ROM_RETAIN_DIR_PREFIXES: Tuple[str, ...] = (
    "validation/",
)

MINIMAL_ROM_DELETE_NAME_PATTERNS: Tuple[str, ...] = (
    ".preview.json",
    "_preview.json",
    "dry_run",
    "placeholder",
    "sample_failure_retention.json",
    "sample_failure_diagnostic.log",
    "sample_cleanup_failure_report.json",
    "README.md",
)

MINIMAL_ROM_DELETE_REL_FILES: Tuple[str, ...] = (
    "pipeline_run_manifest.m4_4_partial_aggregation_preview.json",
    "pipeline_run_manifest.m4_4_full_aggregation_preview.json",
    "lprod/worker_chunk_plan.preview.json",
    "lprod/worker_commands.json",
    "lprod/lprod_target_plan.json",
    "lprod/lprod_execution_plan.preview.json",
    "aggregation/aggregation_plan.preview.json",
    "run_one_sample_plan.json",
    "batch_dry_run_plan.json",
)

MINIMAL_ROM_DELETE_REL_DIRS: Tuple[str, ...] = (
    "rom",
    "logs",
    "specs",
    "config_overlays",
)


@dataclass
class MinimalRomCompactionOutcome:
    sample_id: str
    run_id: str
    status: str
    deleted_paths: List[str] = field(default_factory=list)
    retained_paths: List[str] = field(default_factory=list)
    deleted_bytes: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    runtime_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _aggregation_plot_paths(run_root: Path) -> List[Path]:
    agg = run_root / "aggregation"
    if not agg.is_dir():
        return []
    plots: List[Path] = []
    for pattern in ("mode_frequency*.png", "mode_frequency_plot.png"):
        plots.extend(p for p in agg.glob(pattern) if p.is_file())
    return sorted(set(plots))


def minimal_rom_retain_rel_paths(run_root: Path) -> Set[str]:
    retained = {p.replace("\\", "/") for p in MINIMAL_ROM_RETAIN_FILES}
    for rel_path in MINIMAL_ROM_RETAIN_VALIDATION_FILES:
        if (run_root / rel_path).is_file():
            retained.add(rel_path.replace("\\", "/"))
    for rel_path in DURABLE_VALIDATION_INPUT_REL:
        if (run_root / rel_path).is_file():
            retained.add(rel_path.replace("\\", "/"))
    for plot in _aggregation_plot_paths(run_root):
        retained.add(plot.relative_to(run_root).as_posix())
    return retained


def verify_minimal_rom_retention_sufficient(run_root: Path) -> Tuple[bool, List[str]]:
    """Pre-delete check: durable package can support ROM ingest, compare, audit."""
    errors: List[str] = []
    for rel_path in MINIMAL_ROM_RETAIN_REQUIRED_FILES:
        if not (run_root / rel_path).is_file():
            errors.append(f"missing_required:{rel_path}")

    durable_ok, durable_errors = verify_success_durable_outputs(run_root)
    if not durable_ok:
        errors.extend(durable_errors)

    phys_path = run_root / PHYSICS_IDENTITY_MANIFEST
    if phys_path.is_file():
        try:
            phys = load_json(phys_path)
            ok, man_errs = validate_physics_identity_manifest(phys)
            if not ok:
                errors.extend([f"physics_identity:{e}" for e in man_errs])
        except (OSError, ValueError, json.JSONDecodeError):
            errors.append("physics_identity_unreadable")
    else:
        errors.append("missing_physics_identity_manifest")

    if not _aggregation_plot_paths(run_root):
        errors.append("missing_aggregation_plots")

    return len(errors) == 0, errors


def _path_matches_delete_pattern(rel_posix: str) -> bool:
    lower = rel_posix.lower()
    if any(pat in lower for pat in MINIMAL_ROM_DELETE_NAME_PATTERNS):
        return True
    if rel_posix.endswith(".md") and not rel_posix.startswith("validation/"):
        return True
    return False


def collect_minimal_rom_deletable_paths(run_root: Path) -> List[Path]:
    run_root = run_root.resolve()
    retain = minimal_rom_retain_rel_paths(run_root)
    deletable: List[Path] = []

    for rel_dir in HEAVY_ARCHIVE_REL_DIRS + MINIMAL_ROM_DELETE_REL_DIRS:
        path = run_root / rel_dir
        if path.exists() and not path.is_symlink():
            deletable.append(path)

    for rel_file in HEAVY_ARCHIVE_REL_FILES + MINIMAL_ROM_DELETE_REL_FILES:
        path = run_root / rel_file
        if path.is_file() and not path.is_symlink():
            deletable.append(path)

    for path in sorted(run_root.rglob("*")):
        if not path.exists() or path.is_symlink():
            continue
        try:
            rel_posix = path.relative_to(run_root).as_posix()
        except ValueError:
            continue
        if rel_posix in retain:
            continue
        if any(rel_posix.startswith(prefix) for prefix in MINIMAL_ROM_RETAIN_DIR_PREFIXES):
            if path.is_file() and rel_posix in retain:
                continue
            if path.is_dir():
                continue
        if path.is_dir():
            if rel_posix in {d.rstrip("/") for d in MINIMAL_ROM_DELETE_REL_DIRS}:
                continue
            if any(rel_posix.startswith(d.rstrip("/") + "/") for d in MINIMAL_ROM_DELETE_REL_DIRS):
                continue
            if any(rel_posix == d.rstrip("/") for d in HEAVY_ARCHIVE_REL_DIRS):
                continue
            if any(rel_posix.startswith(d.rstrip("/") + "/") for d in HEAVY_ARCHIVE_REL_DIRS):
                continue
            continue
        if _path_matches_delete_pattern(rel_posix):
            deletable.append(path)
            continue
        top = rel_posix.split("/", 1)[0]
        if top in {"lprod", "scout", "worker_results", "rom", "logs", "specs", "config_overlays"}:
            deletable.append(path)

    unique: Dict[str, Path] = {}
    for path in deletable:
        unique[str(path.resolve())] = path
    return sorted(unique.values(), key=lambda p: len(str(p)), reverse=True)


def _file_size(path: Path) -> int:
    try:
        if path.is_file():
            return path.stat().st_size
        if path.is_dir():
            return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    except OSError:
        return 0
    return 0


def _delete_paths(paths: Sequence[Path]) -> Tuple[List[str], int]:
    deleted: List[str] = []
    nbytes = 0
    for path in paths:
        if not path.exists():
            continue
        nbytes += _file_size(path)
        rel_posix = path.name
        try:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
                rel_posix = str(path)
            elif path.is_file():
                path.unlink()
                rel_posix = str(path)
            else:
                continue
            deleted.append(rel_posix)
        except OSError:
            continue
    return deleted, nbytes


def verify_minimal_rom_post_compaction(
    *,
    repo_root: Path,
    run_root: Path,
    sample_id: str,
    run_id: str,
) -> Tuple[bool, Dict[str, Any], List[str]]:
    errors: List[str] = []
    retain_ok, retain_errors = verify_minimal_rom_retention_sufficient(run_root)
    if not retain_ok:
        errors.extend(retain_errors)

    forbidden_count, forbidden_paths = count_forbidden_heavy_artifacts(run_root)
    shared_paths = [
        p for p in collect_shared_sample_artifact_paths(
            repo_root=repo_root,
            sample_id=sample_id,
            run_id=run_id,
        )
        if p.exists()
    ]
    meta = {
        "forbidden_heavy_artifact_count": forbidden_count,
        "shared_sample_artifact_count": len(shared_paths),
        "retained_rel_paths": sorted(minimal_rom_retain_rel_paths(run_root)),
    }
    if forbidden_count != 0:
        errors.append(f"forbidden_heavy_artifact_count={forbidden_count}")
    if shared_paths:
        errors.append(f"shared_sample_artifact_count={len(shared_paths)}")

    return len(errors) == 0, meta, errors


def _eligibility_errors(run_root: Path) -> List[str]:
    errors: List[str] = []
    sample_in_path = run_root / "sample" / "sample_input.json"
    sample_in = load_json(sample_in_path) if sample_in_path.is_file() else {}
    profile = str(sample_in.get("mesh_profile") or "")
    if profile and profile != MESH_PROFILE_ROM:
        errors.append(f"mesh_profile={profile or 'missing'}")
    elif not profile:
        phys = run_root / PHYSICS_IDENTITY_MANIFEST
        if phys.is_file():
            doc = load_json(phys)
            if str(doc.get("mesh_profile") or "") != MESH_PROFILE_ROM:
                errors.append(f"physics_mesh_profile={doc.get('mesh_profile')!r}")
        else:
            errors.append("mesh_profile_unknown")

    summary = read_run_production_summary(run_root)
    if str(summary.get("terminal_status") or "") != "COMPLETED":
        errors.append(f"terminal_status={summary.get('terminal_status') or 'missing'}")
    if not is_run_usably_complete(summary):
        errors.append("aggregation_not_usably_complete")

    barrier_path = run_root / BARRIER_MANIFEST_REL
    if not barrier_path.is_file():
        errors.append("missing_cleanup_barrier_manifest")
    else:
        barrier = load_json(barrier_path)
        if str(barrier.get("status") or "") != "completed":
            errors.append(f"cleanup_barrier_status={barrier.get('status')!r}")

    pre_ok, pre_errors = verify_minimal_rom_retention_sufficient(run_root)
    if not pre_ok:
        errors.extend(pre_errors)
    return errors


def compact_minimal_rom_durable_run(
    *,
    repo_root: Path,
    run_root: Path,
    sample_id: str,
    run_id: str,
    dry_run: bool = False,
) -> MinimalRomCompactionOutcome:
    t0 = time.perf_counter()
    outcome = MinimalRomCompactionOutcome(sample_id=sample_id, run_id=run_id, status="planned")
    gate_errors = _eligibility_errors(run_root)
    if gate_errors:
        outcome.status = "skipped"
        outcome.errors = gate_errors
        outcome.runtime_s = round(time.perf_counter() - t0, 4)
        return outcome

    deletable = collect_minimal_rom_deletable_paths(run_root)
    outcome.retained_paths = sorted(minimal_rom_retain_rel_paths(run_root))
    if dry_run:
        outcome.status = "dry_run"
        outcome.deleted_paths = [
            str(p.relative_to(run_root)).replace("\\", "/")
            if p.exists() and str(p).startswith(str(run_root))
            else str(p)
            for p in deletable
        ]
        outcome.runtime_s = round(time.perf_counter() - t0, 4)
        return outcome

    deleted, nbytes = _delete_paths(deletable)
    outcome.deleted_paths = deleted
    outcome.deleted_bytes = nbytes

    post_ok, post_meta, post_errors = verify_minimal_rom_post_compaction(
        repo_root=repo_root,
        run_root=run_root,
        sample_id=sample_id,
        run_id=run_id,
    )
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "tool_version": TOOL_VERSION,
        "generated_utc": utc_now(),
        "sample_id": sample_id,
        "run_id": run_id,
        "mode": "minimal_rom_durable",
        "deleted_paths": outcome.deleted_paths,
        "retained_paths": outcome.retained_paths,
        "deleted_bytes": outcome.deleted_bytes,
        "post_compaction": post_meta,
        "status": "completed" if post_ok else "failed",
        "errors": post_errors,
    }
    write_json_atomic(run_root / "compaction" / f"{MANIFEST_SCHEMA}.json", manifest)
    write_json_atomic(
        run_root / "compaction" / "compaction_manifest.json",
        {
            "schema": "m4_run_compaction_manifest_v1",
            "tool_version": TOOL_VERSION,
            "mode": "minimal_rom_durable",
            "status": manifest["status"],
            "timestamp": utc_now(),
            "deleted_paths": outcome.deleted_paths,
            "retained_paths": outcome.retained_paths,
            "deleted_bytes": outcome.deleted_bytes,
        },
    )

    if post_ok:
        outcome.status = "completed"
    else:
        outcome.status = "failed"
        outcome.errors = post_errors
    outcome.runtime_s = round(time.perf_counter() - t0, 4)
    return outcome


def resolve_run_root(repo_root: Path, *, sample_id: str, run_id: str) -> Path:
    return guitars_root(repo_root) / sample_id / "runs" / run_id


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Minimal durable-data compaction for completed ROM production runs.",
    )
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    repo_root = (args.repo_root or detect_repo_root(SCRIPT_DIR)).resolve()
    sample_id = str(args.sample_id)
    run_id = str(args.run_id)
    run_root = resolve_run_root(repo_root, sample_id=sample_id, run_id=run_id)
    if not run_root.is_dir():
        print(f"error: run_root missing: {run_root}", file=sys.stderr)
        return 2

    outcome = compact_minimal_rom_durable_run(
        repo_root=repo_root,
        run_root=run_root,
        sample_id=sample_id,
        run_id=run_id,
        dry_run=bool(args.dry_run),
    )
    print(f"status={outcome.status}")
    print(f"deleted_count={len(outcome.deleted_paths)}")
    print(f"deleted_bytes={outcome.deleted_bytes}")
    print(f"retained_count={len(outcome.retained_paths)}")
    print(f"run_root={rel(run_root, repo_root=repo_root)}")
    if outcome.errors:
        print(f"errors={';'.join(outcome.errors)}", file=sys.stderr)
    return 0 if outcome.status in ("completed", "dry_run") else 1


if __name__ == "__main__":
    raise SystemExit(main())
