#!/usr/bin/env python3
"""Safe compaction/archive for completed M4 FOM runs (ROM-retention aware)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_m4_lhs_pool_bridge import (  # noqa: E402
    AGG_PASS,
    DEFAULT_RUN_ID_SUFFIX,
    LHS_COMPLETED,
    LHS_FAILED,
    LHS_FAILED_RETRYABLE,
    LHS_RUNNING,
    is_lhs_entry_completed,
    is_run_usably_complete,
    load_lhs_pool,
    normalize_lhs_entry_status,
    read_run_production_summary,
)
from v2_b3_m4_worker_run_lib import detect_repo_root, rel, utc_now  # noqa: E402
from v2_b3_petsc_util import write_json_atomic  # noqa: E402

TOOL_VERSION = "m4_run_compaction_v2"
MANIFEST_SCHEMA = "m4_run_compaction_manifest_v1"
MODE_DELETE_WITHOUT_ARCHIVE = "delete_without_archive"
MODE_ARCHIVE_HEAVY = "archive_heavy"

# ROM rebuild / compare / STK minimum (code-audited)
ROM_REQUIRED_REL_PATHS: Tuple[str, ...] = (
    "aggregation/modes_catalog.jsonl",
    "aggregation/aggregation_result.json",
)

ROM_OPTIONAL_REL_PATHS: Tuple[str, ...] = (
    "aggregation/modes_summary.json",
    "aggregation/runtime_summary.json",
    "aggregation/warnings_and_failures.json",
    "rom/rom_fom_comparison.json",
    "rom/rom_prediction_pre_fom.json",
    "sample/sample_input.json",
)

RETAIN_ALWAYS_REL: Tuple[str, ...] = (
    "aggregation",
    "rom",
    "freeze",
    "logs",
    "sample",
    "compaction",
    "pipeline_run_manifest.json",
    "m4_sample_runtime_provenance.json",
)

RETAIN_METADATA_REL_FILES: Tuple[str, ...] = (
    "scout/scout_plan.json",
    "scout/scout_result.json",
    "scout/density_zones.json",
    "lprod/lprod_target_plan.json",
    "lprod/worker_chunk_plan.preview.json",
    "lprod/worker_commands.json",
    "worker_results/remaining_workers_m4_4_1b_4_manifest.json",
)

HEAVY_ARCHIVE_REL_DIRS: Tuple[str, ...] = (
    "lprod/mesh",
    "lprod/checkpoint",
    "scout/mesh",
    "scout/checkpoint",
    "scout/discovery",
    "worker_results",
)

# Large verification dumps inside lprod (archived with lprod/checkpoint)
HEAVY_ARCHIVE_REL_FILES: Tuple[str, ...] = (
    "lprod/lprod_checkpoint_verify.json",
)

REFERENCE_SAMPLES = ("sample_001",)
SYMBOLIC_FULL_SAMPLES = ("sample_000", "sample_001", "sample_034")
DEFAULT_RECOMMENDED_FULL = SYMBOLIC_FULL_SAMPLES


@dataclass
class RunRecord:
    sample_id: str
    run_id: str
    run_root: Path
    eligible: bool = False
    skip_reason: str = ""
    keep_full: bool = False
    keep_full_reason: str = ""
    already_compacted: bool = False
    rom_comparison_present: bool = False
    rom_comparison_path: str = ""
    original_bytes: int = 0
    retained_bytes: int = 0
    archivable_bytes: int = 0
    archived_bytes: int = 0
    freed_bytes: int = 0
    archive_path: str = ""
    sha256: str = ""
    archived_paths: List[str] = field(default_factory=list)
    deleted_paths: List[str] = field(default_factory=list)
    retained_paths: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    status: str = "planned"
    compaction_mode: str = ""
    rom_rebuild_safe: bool = False
    rom_dir_present: bool = False
    aggregation_plots_present: bool = False
    deleted_bytes: int = 0


PRODUCTION_PASS_OUTCOMES = frozenset({"pass", "reused_complete", "pass_freeze_warning"})


@dataclass
class CompactionOutcome:
    sample_id: str
    run_id: str
    status: str
    deleted_bytes: int = 0
    archivable_bytes: int = 0
    retained_bytes: int = 0
    runtime_s: float = 0.0
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    skip_reason: str = ""
    keep_full: bool = False
    rom_rebuild_safe: bool = False
    compaction_mode: str = MODE_DELETE_WITHOUT_ARCHIVE

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _parse_sample_range(text: str) -> List[str]:
    out: List[str] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part and part.replace("-", "").isdigit():
            lo, hi = part.split("-", 1)
            for i in range(int(lo), int(hi) + 1):
                out.append(f"sample_{i:03d}")
        elif part.isdigit():
            out.append(f"sample_{int(part):03d}")
        else:
            out.append(part)
    return sorted(set(out))


def _parse_sample_list(text: str) -> List[str]:
    return [s.strip() for s in str(text).split(",") if s.strip()]


def _dir_size(path: Path, *, follow_symlinks: bool = False) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    for p in path.rglob("*"):
        if p.is_symlink() and not follow_symlinks:
            continue
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            continue
    return total


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size if path.is_file() else 0
    except OSError:
        return 0


def guitars_root(repo_root: Path) -> Path:
    return repo_root / "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars"


def _safe_resolve_under(base: Path, child: Path) -> Optional[Path]:
    base = base.resolve()
    try:
        target = child.resolve(strict=False)
    except OSError:
        return None
    if target.is_symlink():
        return None
    try:
        target.relative_to(base)
    except ValueError:
        return None
    return target


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _catalog_readable(path: Path) -> Tuple[bool, str]:
    if not path.is_file():
        return False, "missing"
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    return True, "ok"
        return False, "empty"
    except OSError as exc:
        return False, f"read_error:{exc}"


def _rom_comparison_status(run_root: Path, repo_root: Path, entry: Mapping[str, Any]) -> Tuple[bool, str, str]:
    run_cmp = run_root / "rom" / "rom_fom_comparison.json"
    if run_cmp.is_file():
        return True, str(run_cmp), "run_tree"
    last_path = str(entry.get("last_rom_comparison_path") or "")
    if last_path:
        p = Path(last_path)
        if not p.is_absolute():
            p = repo_root / last_path
        if p.is_file():
            return True, str(p), "lhs_pool_pointer"
    shape = str(entry.get("shape_name") or "classic")
    sid = str(entry.get("id") or "")
    run_id = str(entry.get("last_run_id") or "")
    classic_cmp = repo_root / "ROM" / shape / "comparisons" / f"{sid}__{run_id}_rom_fom_comparison.json"
    if classic_cmp.is_file():
        return True, str(classic_cmp), "rom_classic_comparisons"
    return False, "", "missing"


def _is_resume_needed(entry: Mapping[str, Any], summary: Mapping[str, Any]) -> bool:
    status = normalize_lhs_entry_status(entry.get("status"))
    if status in (LHS_RUNNING, LHS_FAILED, LHS_FAILED_RETRYABLE):
        return True
    agg = str(summary.get("aggregation_status") or entry.get("last_aggregation_status") or "")
    if agg and agg != AGG_PASS:
        return True
    if int(summary.get("missing_chunks") or 0) > 0:
        return True
    if int(summary.get("failed_chunks") or 0) > 0:
        return True
    terminal = str(summary.get("terminal_status") or entry.get("last_terminal_status") or "")
    if terminal and "RUNNING" in terminal.upper():
        return True
    return False


def _existing_manifest(run_root: Path) -> Optional[Dict[str, Any]]:
    path = run_root / "compaction" / "compaction_manifest.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _collect_archivable_paths(run_root: Path) -> List[Path]:
    found: List[Path] = []
    for rel_dir in HEAVY_ARCHIVE_REL_DIRS:
        p = run_root / rel_dir
        if p.exists() and not p.is_symlink():
            if _safe_resolve_under(run_root, p) is not None:
                found.append(p)
    for rel_file in HEAVY_ARCHIVE_REL_FILES:
        p = run_root / rel_file
        if p.is_file() and not p.is_symlink():
            if _safe_resolve_under(run_root, p) is not None:
                found.append(p)
    return sorted(found, key=lambda x: str(x))


def _collect_retained_paths(run_root: Path) -> List[str]:
    retained: List[str] = []
    for name in RETAIN_ALWAYS_REL:
        p = run_root / name
        if p.exists():
            retained.append(name.replace("\\", "/"))
    for rel_file in RETAIN_METADATA_REL_FILES:
        p = run_root / rel_file
        if p.is_file():
            retained.append(rel_file.replace("\\", "/"))
    return sorted(set(retained))


def _estimate_bytes(run_root: Path, archivable: Sequence[Path]) -> Tuple[int, int, int]:
    original = _dir_size(run_root)
    archivable_bytes = sum(_dir_size(p) if p.is_dir() else _file_size(p) for p in archivable)
    retained = max(0, original - archivable_bytes)
    return original, retained, archivable_bytes


def _eligible_run(
    *,
    repo_root: Path,
    entry: Mapping[str, Any],
    run_id: str,
) -> RunRecord:
    sample_id = str(entry.get("id") or "")
    run_root = guitars_root(repo_root) / sample_id / "runs" / run_id
    rec = RunRecord(sample_id=sample_id, run_id=run_id, run_root=run_root)

    if normalize_lhs_entry_status(entry.get("status")) != LHS_COMPLETED:
        rec.skip_reason = f"lhs_status={entry.get('status')}"
        return rec

    if not is_lhs_entry_completed(entry, run_id=run_id):
        rec.skip_reason = "lhs_completed_run_id_mismatch"
        return rec

    if not run_root.is_dir():
        rec.skip_reason = "run_root_missing"
        return rec

    summary = read_run_production_summary(run_root)
    if _is_resume_needed(entry, summary):
        rec.skip_reason = "resume_needed_or_incomplete"
        return rec

    agg_status = str(summary.get("aggregation_status") or entry.get("last_aggregation_status") or "")
    if agg_status != AGG_PASS:
        rec.skip_reason = f"aggregation_status={agg_status or 'missing'}"
        return rec

    if not is_run_usably_complete(summary):
        rec.skip_reason = "aggregation_not_usably_complete"
        return rec

    cat_ok, cat_detail = _catalog_readable(run_root / "aggregation" / "modes_catalog.jsonl")
    if not cat_ok:
        rec.skip_reason = f"modes_catalog:{cat_detail}"
        return rec

    rom_present, rom_path, _rom_src = _rom_comparison_status(run_root, repo_root, entry)
    rec.rom_comparison_present = rom_present
    rec.rom_comparison_path = rom_path
    if not rom_present:
        rec.warnings.append("rom_fom_comparison_missing (reported; compaction still allowed)")

    rec.rom_dir_present = (run_root / "rom").is_dir()
    rec.aggregation_plots_present = _aggregation_plots_present(run_root)
    rom_safe, retention_warnings, blocking = _verify_rom_retention(run_root)
    rec.rom_rebuild_safe = rom_safe
    rec.warnings.extend(retention_warnings)
    if blocking:
        rec.skip_reason = f"rom_retention_incomplete:{','.join(blocking)}"
        return rec

    manifest = _existing_manifest(run_root)
    archivable = _collect_archivable_paths(run_root)
    if manifest and str(manifest.get("status")) in ("completed", "archived_no_delete") and not archivable:
        rec.already_compacted = True
        rec.status = "already_compacted"
        archive_path = Path(str(manifest.get("archive_path") or ""))
        if archive_path.is_file():
            rec.archive_path = str(archive_path)
            rec.sha256 = str(manifest.get("sha256") or "")
    if not archivable and rec.already_compacted:
        rec.eligible = False
        rec.skip_reason = "already_compacted_no_heavy_local"
        return rec
    if not archivable:
        rec.skip_reason = "no_heavy_artifacts_present"
        return rec

    original, retained, archivable_bytes = _estimate_bytes(run_root, archivable)
    rec.original_bytes = original
    rec.retained_bytes = retained
    rec.archivable_bytes = archivable_bytes
    rec.archived_paths = [str(p.relative_to(run_root)).replace("\\", "/") for p in archivable]
    rec.retained_paths = _collect_retained_paths(run_root)
    rec.eligible = True
    return rec


def _completion_sort_key(entry: Mapping[str, Any]) -> str:
    for key in ("last_completed_at", "last_run_finished_at", "last_run_started_at", "updated_at"):
        val = str(entry.get(key) or "")
        if val:
            return val
    return str(entry.get("id") or "")


def recommend_representative_full_samples(
    pool: Mapping[str, Any],
    *,
    completed_ids: Sequence[str],
) -> List[Dict[str, str]]:
    """Conservative recommendation — operator must approve before --keep-full-samples."""
    recs: List[Dict[str, str]] = []
    for sid in DEFAULT_RECOMMENDED_FULL:
        if sid not in completed_ids:
            continue
        entry = next((e for e in pool.get("entries") or [] if str(e.get("id")) == sid), None)
        params = dict((entry or {}).get("parameters") or {})
        reason_parts = []
        if sid in REFERENCE_SAMPLES:
            reason_parts.append("M4 reference E2E run")
        if sid == "sample_000":
            reason_parts.append("first LHS anchor")
        if sid == "sample_034":
            reason_parts.append("symbolic latest/high-index reference in 0-34 range")
        recs.append(
            {
                "sample_id": sid,
                "reason": "; ".join(reason_parts) or "representative",
                "top_wood": str(params.get("top_wood_id") or ""),
                "back_wood": str(params.get("back_wood_id") or ""),
            }
        )
    return recs


def _verify_shared_root(path: Optional[Path]) -> Path:
    if path is None:
        raise SystemExit("error: --shared-root required for --archive-heavy / --delete-heavy-after-verify")
    root = path.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"error: shared root unavailable: {root}")
    test_dir = root / ".m4_compaction_write_test"
    try:
        test_dir.mkdir(parents=True, exist_ok=True)
        probe = test_dir / "probe.tmp"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        test_dir.rmdir()
    except OSError as exc:
        raise SystemExit(f"error: shared root not writable: {root} ({exc})") from exc
    return root


def _archive_name(sample_id: str, run_id: str) -> str:
    return f"{sample_id}__{run_id}__heavy.tar.zst"


def _create_archive(
    *,
    run_root: Path,
    archivable: Sequence[Path],
    archive_final: Path,
) -> Tuple[int, List[str]]:
    archive_final.parent.mkdir(parents=True, exist_ok=True)
    tmp = archive_final.with_suffix(archive_final.suffix + ".partial")
    if tmp.exists():
        tmp.unlink()

    rel_members = [str(p.relative_to(run_root)).replace("\\", "/") for p in archivable]
    cmd = [
        "tar",
        "-C",
        str(run_root),
        "--use-compress-program=zstd",
        "-cf",
        str(tmp),
        *rel_members,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise RuntimeError(f"tar failed: {proc.stderr.strip() or proc.stdout.strip()}")

    verify = subprocess.run(["tar", "-tf", str(tmp)], capture_output=True, text=True)
    if verify.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"archive unreadable: {verify.stderr.strip()}")

    listed = [ln.strip() for ln in verify.stdout.splitlines() if ln.strip()]
    if not listed:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("archive empty")

    size = tmp.stat().st_size
    tmp.replace(archive_final)
    return size, listed


def _verify_rom_required(run_root: Path) -> List[str]:
    missing = []
    for rel_path in ROM_REQUIRED_REL_PATHS:
        if not (run_root / rel_path).is_file():
            missing.append(rel_path)
    return missing


def _aggregation_plots_present(run_root: Path) -> bool:
    agg = run_root / "aggregation"
    if not agg.is_dir():
        return False
    return any(agg.glob("mode_frequency*.png")) or (agg / "mode_frequency_plot.png").is_file()


def _verify_rom_retention(run_root: Path) -> Tuple[bool, List[str], List[str]]:
    """Return (rom_rebuild_safe, warnings, blocking_missing)."""
    warnings: List[str] = []
    missing = _verify_rom_required(run_root)
    ms = run_root / "aggregation" / "modes_summary.json"
    ar = run_root / "aggregation" / "aggregation_result.json"
    if not ms.is_file() and not ar.is_file():
        missing.append("aggregation/modes_summary.json_or_aggregation_result.json")
    if not (run_root / "rom").is_dir():
        warnings.append("rom/ directory missing (reported; ROM compare cache optional)")
    if not _aggregation_plots_present(run_root):
        warnings.append("no aggregation plots found (shared export optional)")
    return len(missing) == 0, warnings, missing


def _delete_archived_paths(run_root: Path, archivable: Sequence[Path]) -> List[str]:
    deleted: List[str] = []
    for p in sorted(archivable, key=lambda x: len(str(x)), reverse=True):
        rel_s = str(p.relative_to(run_root)).replace("\\", "/")
        if p.is_symlink():
            continue
        if not _safe_resolve_under(run_root, p):
            continue
        if p.is_dir():
            shutil.rmtree(p)
        elif p.is_file():
            p.unlink()
        deleted.append(rel_s)
    return deleted


def _write_manifest(run_root: Path, payload: Dict[str, Any]) -> None:
    write_json_atomic(run_root / "compaction" / "compaction_manifest.json", payload)


def _build_delete_manifest(rec: RunRecord, *, production_trigger: bool = False) -> Dict[str, Any]:
    return {
        "schema": MANIFEST_SCHEMA,
        "tool_version": TOOL_VERSION,
        "sample_id": rec.sample_id,
        "run_id": rec.run_id,
        "timestamp": utc_now(),
        "mode": MODE_DELETE_WITHOUT_ARCHIVE,
        "production_trigger": production_trigger,
        "original_bytes": rec.original_bytes,
        "retained_local_bytes": rec.retained_bytes,
        "deleted_bytes": rec.deleted_bytes,
        "freed_bytes": rec.deleted_bytes,
        "deleted_paths": rec.deleted_paths,
        "retained_paths": rec.retained_paths,
        "rom_comparison_present": rec.rom_comparison_present,
        "rom_dir_present": rec.rom_dir_present,
        "aggregation_plots_present": rec.aggregation_plots_present,
        "rom_rebuild_safe": rec.rom_rebuild_safe,
        "warnings": rec.warnings,
        "status": rec.status,
    }


def production_compaction_preconditions(
    *,
    row: Mapping[str, Any],
    pool_entry: Mapping[str, Any],
    run_rom_compare: bool,
) -> Tuple[bool, str, List[str]]:
    """Gates for auto-compaction after a production batch sample finishes."""
    warnings: List[str] = []
    outcome = str(row.get("outcome") or "")
    if outcome not in PRODUCTION_PASS_OUTCOMES:
        return False, f"outcome={outcome or 'missing'}", warnings

    if str(row.get("aggregation_status") or "") != AGG_PASS:
        return False, f"aggregation_status={row.get('aggregation_status') or 'missing'}", warnings

    if not bool(row.get("final_aggregation_ready")):
        return False, "final_aggregation_ready=false", warnings

    lhs_completed = normalize_lhs_entry_status(pool_entry.get("status")) == LHS_COMPLETED
    row_passed = outcome in PRODUCTION_PASS_OUTCOMES
    if not lhs_completed and not row_passed:
        return False, f"lhs_status={pool_entry.get('status') or 'missing'}", warnings

    if row.get("shared_export") or row.get("shared_export_warning"):
        pass
    else:
        warnings.append("shared_export_not_reported")

    if run_rom_compare:
        rom_cmp = row.get("rom_compare")
        if not isinstance(rom_cmp, dict):
            return False, "rom_compare_not_recorded", warnings
        if rom_cmp.get("error"):
            warnings.append(f"rom_compare_error:{rom_cmp.get('error')}")

    return True, "ok", warnings


def apply_keep_full_policy(
    records: Sequence[RunRecord],
    pool: Mapping[str, Any],
    *,
    keep_full_samples: Sequence[str],
    keep_full_latest: int = 0,
) -> None:
    keep_set = set(keep_full_samples)
    eligible = [r for r in records if r.eligible and not r.already_compacted]
    if keep_full_latest > 0 and eligible:
        entry_by_id = {str(e.get("id")): e for e in pool.get("entries") or []}

        def _recency_key(rec: RunRecord) -> str:
            return _completion_sort_key(entry_by_id.get(rec.sample_id) or {})

        latest_ids = [
            r.sample_id for r in sorted(eligible, key=_recency_key, reverse=True)[: int(keep_full_latest)]
        ]
        keep_set.update(latest_ids)
    for rec in records:
        if rec.sample_id in keep_set and rec.eligible:
            rec.keep_full = True
            rec.keep_full_reason = (
                "keep_full_samples" if rec.sample_id in keep_full_samples else "keep_full_latest"
            )


def _record_to_outcome(rec: RunRecord, *, runtime_s: float = 0.0, error: Optional[str] = None) -> CompactionOutcome:
    return CompactionOutcome(
        sample_id=rec.sample_id,
        run_id=rec.run_id,
        status=rec.status,
        deleted_bytes=int(rec.deleted_bytes or rec.freed_bytes or 0),
        archivable_bytes=int(rec.archivable_bytes),
        retained_bytes=int(rec.retained_bytes),
        runtime_s=round(runtime_s, 4),
        error=error,
        warnings=list(rec.warnings),
        skip_reason=rec.skip_reason,
        keep_full=bool(rec.keep_full),
        rom_rebuild_safe=bool(rec.rom_rebuild_safe),
        compaction_mode=rec.compaction_mode or MODE_DELETE_WITHOUT_ARCHIVE,
    )


def compact_one_completed_run(
    *,
    repo_root: Path,
    pool: Mapping[str, Any],
    sample_id: str,
    run_id: Optional[str] = None,
    keep_full: bool = False,
    dry_run: bool = False,
    production_row: Optional[Mapping[str, Any]] = None,
    run_rom_compare: bool = False,
    production_trigger: bool = False,
) -> CompactionOutcome:
    """Delete heavy artifacts for one completed run (reuses standalone compaction logic)."""
    t0 = time.perf_counter()
    entry = next((e for e in pool.get("entries") or [] if str(e.get("id")) == sample_id), None)
    if entry is None:
        return CompactionOutcome(
            sample_id=sample_id,
            run_id=run_id or "",
            status="skipped",
            skip_reason="not_in_lhs_pool",
            runtime_s=round(time.perf_counter() - t0, 4),
        )

    rid = str(run_id or entry.get("last_run_id") or f"{sample_id}_{DEFAULT_RUN_ID_SUFFIX}")
    rec = _eligible_run(repo_root=repo_root, entry=entry, run_id=rid)

    if production_row is not None:
        ok, reason, gate_warnings = production_compaction_preconditions(
            row=production_row,
            pool_entry=entry,
            run_rom_compare=run_rom_compare,
        )
        rec.warnings.extend(gate_warnings)
        if not ok:
            rec.eligible = False
            rec.skip_reason = f"production_gate:{reason}"

    if keep_full:
        rec.keep_full = True
        rec.keep_full_reason = "keep_full_samples"

    if not rec.eligible:
        return _record_to_outcome(rec, runtime_s=time.perf_counter() - t0)

    if rec.keep_full:
        rec.status = "keep_full"
        return _record_to_outcome(rec, runtime_s=time.perf_counter() - t0)

    if rec.already_compacted:
        return _record_to_outcome(rec, runtime_s=time.perf_counter() - t0)

    try:
        _process_delete_without_archive(rec, dry_run=dry_run)
        if not dry_run and rec.status == "completed":
            manifest = _build_delete_manifest(rec, production_trigger=production_trigger)
            _write_manifest(rec.run_root, manifest)
    except Exception as exc:
        rec.status = "failed"
        rec.warnings.append(str(exc))
        return _record_to_outcome(rec, runtime_s=time.perf_counter() - t0, error=str(exc))

    return _record_to_outcome(rec, runtime_s=time.perf_counter() - t0)


def compact_runs_for_samples(
    *,
    repo_root: Path,
    pool: Mapping[str, Any],
    sample_specs: Sequence[Tuple[str, str]],
    keep_full_samples: Sequence[str] = (),
    keep_full_latest: int = 0,
    dry_run: bool = False,
    production_rows_by_sid: Optional[Mapping[str, Mapping[str, Any]]] = None,
    run_rom_compare: bool = False,
    production_trigger: bool = False,
) -> Dict[str, Any]:
    """Compact multiple completed runs; applies keep-full-latest at batch scope."""
    t0 = time.perf_counter()
    records: List[RunRecord] = []
    entry_by_id = {str(e.get("id")): e for e in pool.get("entries") or []}

    for sample_id, run_id in sample_specs:
        entry = entry_by_id.get(sample_id)
        if entry is None:
            records.append(
                RunRecord(
                    sample_id=sample_id,
                    run_id=run_id,
                    run_root=guitars_root(repo_root) / sample_id / "runs" / run_id,
                    skip_reason="not_in_lhs_pool",
                )
            )
            continue
        rec = _eligible_run(repo_root=repo_root, entry=entry, run_id=run_id)
        prod_row = (production_rows_by_sid or {}).get(sample_id)
        if prod_row is not None:
            ok, reason, gate_warnings = production_compaction_preconditions(
                row=prod_row,
                pool_entry=entry,
                run_rom_compare=run_rom_compare,
            )
            rec.warnings.extend(gate_warnings)
            if not ok:
                rec.eligible = False
                rec.skip_reason = f"production_gate:{reason}"
        records.append(rec)

    apply_keep_full_policy(
        records,
        pool,
        keep_full_samples=keep_full_samples,
        keep_full_latest=keep_full_latest,
    )

    outcomes: List[CompactionOutcome] = []
    failed = 0
    freed = 0
    for rec in records:
        if not rec.eligible or rec.keep_full or rec.already_compacted:
            outcomes.append(_record_to_outcome(rec))
            continue
        try:
            _process_delete_without_archive(rec, dry_run=dry_run)
            if not dry_run and rec.status == "completed":
                _write_manifest(rec.run_root, _build_delete_manifest(rec, production_trigger=production_trigger))
            out = _record_to_outcome(rec)
            outcomes.append(out)
            if out.status == "failed":
                failed += 1
            else:
                freed += int(out.deleted_bytes)
        except Exception as exc:
            rec.status = "failed"
            rec.warnings.append(str(exc))
            failed += 1
            outcomes.append(_record_to_outcome(rec, error=str(exc)))

    compacted_count = sum(1 for o in outcomes if o.status == "completed")
    return {
        "schema": "m4_production_compaction_summary_v1",
        "tool_version": TOOL_VERSION,
        "compaction_mode": MODE_DELETE_WITHOUT_ARCHIVE,
        "compaction_status": "completed" if failed == 0 else "partial_failed",
        "compaction_runtime_s": round(time.perf_counter() - t0, 4),
        "compaction_sample_count": compacted_count,
        "compaction_failed_count": failed,
        "compaction_bytes_freed": freed,
        "dry_run": dry_run,
        "keep_full_samples": list(keep_full_samples),
        "keep_full_latest": int(keep_full_latest),
        "outcomes": [o.to_dict() for o in outcomes],
    }


def _process_delete_without_archive(rec: RunRecord, *, dry_run: bool) -> None:
    if not rec.eligible or rec.keep_full or rec.already_compacted:
        if rec.keep_full:
            rec.status = "keep_full"
        elif rec.already_compacted:
            rec.status = "already_compacted"
        return

    rec.compaction_mode = MODE_DELETE_WITHOUT_ARCHIVE
    archivable = _collect_archivable_paths(rec.run_root)
    if not archivable:
        rec.status = "skipped_no_heavy"
        return

    rom_safe, warnings, blocking = _verify_rom_retention(rec.run_root)
    rec.rom_rebuild_safe = rom_safe
    for w in warnings:
        if w not in rec.warnings:
            rec.warnings.append(w)
    if blocking:
        raise RuntimeError(f"refuse delete: ROM retention incomplete: {blocking}")
    if not rom_safe:
        raise RuntimeError("refuse delete: rom_rebuild_safe=false")

    if dry_run:
        rec.status = "dry_run_planned_delete"
        return

    rec.deleted_paths = _delete_archived_paths(rec.run_root, archivable)
    rec.deleted_bytes = rec.archivable_bytes
    rec.freed_bytes = rec.deleted_bytes
    rec.status = "completed"


def _process_record(
    rec: RunRecord,
    *,
    repo_root: Path,
    shared_root: Path,
    dry_run: bool,
    archive_heavy: bool,
    delete_after_verify: bool,
) -> None:
    if not rec.eligible or rec.keep_full or rec.already_compacted:
        return

    archivable = _collect_archivable_paths(rec.run_root)
    if not archivable:
        rec.status = "skipped_no_heavy"
        return

    archive_dir = shared_root / "classic" / "archives"
    archive_path = archive_dir / _archive_name(rec.sample_id, rec.run_id)

    if dry_run and not archive_heavy:
        rec.status = "dry_run_planned"
        return

    if not archive_heavy:
        rec.status = "dry_run_planned"
        return

    if archive_path.is_file():
        existing_digest = _sha256_file(archive_path)
        if rec.sha256 and existing_digest != rec.sha256:
            raise RuntimeError(f"archive checksum mismatch: {archive_path}")
        rec.sha256 = existing_digest
        rec.archive_path = str(archive_path)
        rec.archived_bytes = archive_path.stat().st_size
        if delete_after_verify:
            missing = _verify_rom_required(rec.run_root)
            if missing:
                raise RuntimeError(f"refuse delete: ROM required missing: {missing}")
            rec.deleted_paths = _delete_archived_paths(rec.run_root, archivable)
            rec.freed_bytes = rec.archivable_bytes
            rec.status = "completed"
            manifest = {
                "schema": MANIFEST_SCHEMA,
                "tool_version": TOOL_VERSION,
                "sample_id": rec.sample_id,
                "run_id": rec.run_id,
                "timestamp": utc_now(),
                "original_bytes": rec.original_bytes,
                "retained_local_bytes": rec.retained_bytes,
                "archived_bytes": rec.archived_bytes,
                "freed_bytes": rec.freed_bytes,
                "archive_path": rec.archive_path,
                "sha256": rec.sha256,
                "archived_paths": rec.archived_paths,
                "deleted_paths": rec.deleted_paths,
                "retained_paths": rec.retained_paths,
                "rom_comparison_present": rec.rom_comparison_present,
                "status": rec.status,
            }
            _write_manifest(rec.run_root, manifest)
        else:
            rec.status = "archive_exists_verified"
        return

    listed: List[str] = []
    try:
        archived_size, listed = _create_archive(
            run_root=rec.run_root,
            archivable=archivable,
            archive_final=archive_path,
        )
    except OSError as exc:
        rec.status = "archive_failed"
        rec.warnings.append(str(exc))
        raise

    digest = _sha256_file(archive_path)
    checksum_path = archive_path.with_suffix(archive_path.suffix + ".sha256")
    checksum_path.write_text(f"{digest}  {archive_path.name}\n", encoding="utf-8")
    contents_path = archive_path.with_suffix(archive_path.suffix + ".contents.txt")
    contents_path.write_text("\n".join(listed) + "\n", encoding="utf-8")

    rec.archived_bytes = archived_size
    rec.archive_path = str(archive_path)
    rec.sha256 = digest

    missing = _verify_rom_required(rec.run_root)
    if missing:
        rec.status = "archive_ok_retention_incomplete"
        rec.warnings.append(f"rom_required_missing:{missing}")
        raise RuntimeError(f"refuse delete: ROM required files missing: {missing}")

    if delete_after_verify:
        rec.deleted_paths = _delete_archived_paths(rec.run_root, archivable)
        rec.freed_bytes = rec.archivable_bytes
        rec.status = "completed"
    else:
        rec.status = "archived_no_delete"

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "tool_version": TOOL_VERSION,
        "sample_id": rec.sample_id,
        "run_id": rec.run_id,
        "timestamp": utc_now(),
        "original_bytes": rec.original_bytes,
        "retained_local_bytes": rec.retained_bytes,
        "archived_bytes": rec.archived_bytes,
        "freed_bytes": rec.freed_bytes,
        "archive_path": rec.archive_path,
        "sha256": rec.sha256,
        "archive_contents_count": len(listed),
        "archived_paths": rec.archived_paths,
        "deleted_paths": rec.deleted_paths,
        "retained_paths": rec.retained_paths,
        "rom_comparison_present": rec.rom_comparison_present,
        "status": rec.status,
    }
    if not dry_run:
        _write_manifest(rec.run_root, manifest)


def _print_summary(
    *,
    title: str,
    records: Sequence[RunRecord],
    recommendations: Sequence[Mapping[str, str]],
    dry_run: bool,
    compaction_mode: str,
) -> None:
    eligible = [r for r in records if r.eligible]
    skipped = [r for r in records if not r.eligible]
    keep_full = [r for r in eligible if r.keep_full]
    planned = [r for r in eligible if not r.keep_full and not r.already_compacted]

    print(f"\n=== {title} ===", flush=True)
    print(f"compaction_mode={compaction_mode or 'dry_run_plan'}", flush=True)
    print(
        f"eligible_completed={len(eligible)} skipped={len(skipped)} "
        f"keep_full={len(keep_full)} planned_action={len(planned)}",
        flush=True,
    )
    print(f"dry_run={dry_run}", flush=True)

    if recommendations:
        print("\nrepresentative_full_retention_recommendation (approve before --keep-full-samples):", flush=True)
        for row in recommendations:
            print(
                f"  {row['sample_id']}: {row['reason']} "
                f"(top={row.get('top_wood')}, back={row.get('back_wood')})",
                flush=True,
            )

    if skipped:
        print("\nskipped_samples:", flush=True)
        for r in skipped[:40]:
            print(f"  {r.sample_id}/{r.run_id}: {r.skip_reason or r.status}", flush=True)
        if len(skipped) > 40:
            print(f"  ... +{len(skipped) - 40} more", flush=True)

    if keep_full:
        print("\nkept_full_samples (heavy artifacts retained locally):", flush=True)
        for r in keep_full:
            print(f"  {r.sample_id}: {r.keep_full_reason}", flush=True)

    if planned:
        label = (
            "planned_direct_delete_samples"
            if compaction_mode == MODE_DELETE_WITHOUT_ARCHIVE
            else "planned_compaction_samples"
        )
        print(f"\n{label}:", flush=True)
        for r in planned[:40]:
            print(
                f"  {r.sample_id}: archivable={r.archivable_bytes} retained={r.retained_bytes} "
                f"rom_rebuild_safe={r.rom_rebuild_safe}",
                flush=True,
            )
        if len(planned) > 40:
            print(f"  ... +{len(planned) - 40} more", flush=True)

    est_freed = (
        sum(r.archivable_bytes for r in planned)
        if dry_run
        else sum(r.deleted_bytes or r.freed_bytes for r in planned)
    )
    est_retained = sum(r.retained_bytes for r in planned)
    if compaction_mode == MODE_ARCHIVE_HEAVY:
        est_archive = sum(r.archivable_bytes for r in planned)
        print(
            f"\nestimated_archive_bytes={est_archive} (~{est_archive / (1024**3):.2f} GiB)",
            flush=True,
        )
    print(
        f"estimated_local_bytes_freed={est_freed} (~{est_freed / (1024**3):.2f} GiB)",
        flush=True,
    )
    if planned:
        avg_ret = est_retained / len(planned)
        print(
            f"estimated_local_bytes_retained={est_retained} (~{avg_ret / (1024**2):.1f} MiB per planned run)",
            flush=True,
        )

    print("\nrequired_rom_files_retained_per_planned_run:", flush=True)
    for rel_path in ROM_REQUIRED_REL_PATHS + ("aggregation/modes_summary.json",):
        print(f"  {rel_path}", flush=True)

    if dry_run:
        print("\nno files deleted (dry-run)", flush=True)


def _write_reports(
    *,
    out_dir: Path,
    records: Sequence[RunRecord],
    recommendations: Sequence[Mapping[str, str]],
    args: argparse.Namespace,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "m4_run_compaction_report_v1",
        "tool_version": TOOL_VERSION,
        "generated_utc": utc_now(),
        "dry_run": bool(args.dry_run),
        "compaction_mode": (
            MODE_DELETE_WITHOUT_ARCHIVE
            if args.delete_heavy_without_archive
            else (MODE_ARCHIVE_HEAVY if args.archive_heavy else "dry_run_plan")
        ),
        "archive_heavy": bool(args.archive_heavy),
        "delete_heavy_without_archive": bool(args.delete_heavy_without_archive),
        "delete_heavy_after_verify": bool(args.delete_heavy_after_verify),
        "shape_name": args.shape_name,
        "sample_range": args.sample_range,
        "keep_full_latest": args.keep_full_latest,
        "keep_full_samples": list(args.keep_full_samples or []),
        "representative_recommendation": list(recommendations),
        "records": [
            {
                "sample_id": r.sample_id,
                "run_id": r.run_id,
                "eligible": r.eligible,
                "skip_reason": r.skip_reason,
                "keep_full": r.keep_full,
                "status": r.status,
                "original_bytes": r.original_bytes,
                "retained_bytes": r.retained_bytes,
                "archivable_bytes": r.archivable_bytes,
                "compaction_mode": r.compaction_mode,
                "archived_bytes": r.archived_bytes,
                "deleted_bytes": r.deleted_bytes,
                "freed_bytes": r.freed_bytes,
                "archive_path": r.archive_path,
                "sha256": r.sha256,
                "rom_comparison_present": r.rom_comparison_present,
                "rom_rebuild_safe": r.rom_rebuild_safe,
                "warnings": r.warnings,
                "archived_paths": r.archived_paths,
                "deleted_paths": r.deleted_paths,
            }
            for r in records
        ],
    }
    write_json_atomic(out_dir / "compaction_report.json", payload)

    csv_path = out_dir / "compaction_report.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "sample_id",
                "run_id",
                "eligible",
                "skip_reason",
                "keep_full",
                "status",
                "original_bytes",
                "archivable_bytes",
                "deleted_bytes",
                "freed_bytes",
                "rom_comparison_present",
                "rom_rebuild_safe",
                "compaction_mode",
            ],
        )
        writer.writeheader()
        for r in records:
            writer.writerow(
                {
                    "sample_id": r.sample_id,
                    "run_id": r.run_id,
                    "eligible": r.eligible,
                    "skip_reason": r.skip_reason,
                    "keep_full": r.keep_full,
                    "status": r.status,
                    "original_bytes": r.original_bytes,
                    "archivable_bytes": r.archivable_bytes,
                    "deleted_bytes": r.deleted_bytes,
                    "freed_bytes": r.freed_bytes,
                    "rom_comparison_present": r.rom_comparison_present,
                    "rom_rebuild_safe": r.rom_rebuild_safe,
                    "compaction_mode": r.compaction_mode,
                }
            )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Dry-run plan (default):\n"
            "  python .../compact_completed_m4_runs.py --lhs-json ROM/classic/lhs_pool.json "
            "--shape-name classic --sample-range 0-34 --keep-full-latest 1 "
            "--keep-full-samples sample_000,sample_001 --delete-heavy-without-archive --dry-run\n\n"
            "Direct delete (no archive) for non-keep-full completed samples:\n"
            "  same command without --dry-run\n\n"
            "Legacy archive mode (representative full samples only):\n"
            "  add --archive-heavy --delete-heavy-after-verify --shared-root /media/sf_gmar\n"
        ),
    )
    parser.add_argument("--lhs-json", type=Path, default=Path("ROM/classic/lhs_pool.json"))
    parser.add_argument("--shape-name", default="classic")
    parser.add_argument("--sample-range", default="0-34")
    parser.add_argument("--shared-root", type=Path, default=Path("/media/sf_gmar"))
    parser.add_argument("--keep-full-latest", type=int, default=2)
    parser.add_argument("--keep-full-samples", default="")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Plan only (default). No archives or deletes.",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--archive-heavy",
        action="store_true",
        help="Create tar.zst archives on --shared-root (legacy; for keep-full/reference runs).",
    )
    mode_group.add_argument(
        "--delete-heavy-without-archive",
        action="store_true",
        help="Delete heavy artifacts directly after ROM-retention verification (default policy).",
    )
    parser.add_argument(
        "--delete-heavy-after-verify",
        action="store_true",
        help="With --archive-heavy: delete local heavy files after archive verify.",
    )
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Force destructive action (disables dry-run).",
    )
    args = parser.parse_args(argv)

    argv_list = list(argv) if argv is not None else sys.argv[1:]
    explicit_dry_run = "--dry-run" in argv_list
    destructive_mode = bool(args.archive_heavy or args.delete_heavy_without_archive)
    if args.execute:
        args.dry_run = False
    elif destructive_mode and not explicit_dry_run:
        args.dry_run = False
    if args.delete_heavy_after_verify and not args.archive_heavy:
        print("error: --delete-heavy-after-verify requires --archive-heavy", file=sys.stderr)
        return 2
    if destructive_mode and args.dry_run:
        print(
            "note: destructive mode with --dry-run will plan only; omit --dry-run or pass --execute to apply",
            flush=True,
        )

    repo_root = detect_repo_root(SCRIPT_DIR)
    lhs_path = args.lhs_json if args.lhs_json.is_absolute() else repo_root / args.lhs_json
    pool = load_lhs_pool(lhs_path)
    if str(pool.get("shape_name") or "classic") != str(args.shape_name):
        print(
            f"warning: pool shape={pool.get('shape_name')} != --shape-name {args.shape_name}",
            flush=True,
        )

    sample_ids = _parse_sample_range(args.sample_range)
    keep_full_samples = _parse_sample_list(args.keep_full_samples)

    shared_root: Optional[Path] = None
    if args.archive_heavy or args.delete_heavy_after_verify:
        shared_root = _verify_shared_root(args.shared_root)

    compaction_mode = ""
    if args.delete_heavy_without_archive:
        compaction_mode = MODE_DELETE_WITHOUT_ARCHIVE
    elif args.archive_heavy:
        compaction_mode = MODE_ARCHIVE_HEAVY

    records: List[RunRecord] = []
    completed_ids: List[str] = []
    entry_by_id = {str(e.get("id")): e for e in pool.get("entries") or []}

    for sid in sample_ids:
        entry = entry_by_id.get(sid)
        if not entry:
            records.append(
                RunRecord(
                    sample_id=sid,
                    run_id="",
                    run_root=guitars_root(repo_root) / sid / "runs" / "missing",
                    skip_reason="not_in_lhs_pool",
                )
            )
            continue
        run_id = str(entry.get("last_run_id") or f"{sid}_{DEFAULT_RUN_ID_SUFFIX}")
        rec = _eligible_run(repo_root=repo_root, entry=entry, run_id=run_id)
        records.append(rec)
        if rec.eligible:
            completed_ids.append(sid)

    apply_keep_full_policy(
        records,
        pool,
        keep_full_samples=keep_full_samples,
        keep_full_latest=int(args.keep_full_latest),
    )

    recommendations = recommend_representative_full_samples(pool, completed_ids=completed_ids)

    report_dir = args.report_dir or (
        repo_root
        / "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/index/compaction_reports"
    )

    _print_summary(
        title="M4 compaction plan (pre)",
        records=records,
        recommendations=recommendations,
        dry_run=bool(args.dry_run),
        compaction_mode=compaction_mode,
    )

    if args.delete_heavy_without_archive and not args.dry_run:
        for rec in records:
            if not rec.eligible or rec.keep_full:
                continue
            try:
                _process_delete_without_archive(rec, dry_run=False)
            except Exception as exc:
                rec.status = "failed"
                rec.warnings.append(str(exc))
                print(f"error: {rec.sample_id}: {exc}", file=sys.stderr)
                print("error: aborting further deletes after failure", file=sys.stderr)
                break

    if args.archive_heavy and not args.dry_run:
        for rec in records:
            if not rec.eligible or rec.keep_full:
                continue
            try:
                _process_record(
                    rec,
                    repo_root=repo_root,
                    shared_root=shared_root,  # type: ignore[arg-type]
                    dry_run=False,
                    archive_heavy=True,
                    delete_after_verify=bool(args.delete_heavy_after_verify),
                )
            except Exception as exc:
                rec.status = "failed"
                rec.warnings.append(str(exc))
                print(f"error: {rec.sample_id}: {exc}", file=sys.stderr)
                if args.delete_heavy_after_verify:
                    print("error: aborting further deletes after failure", file=sys.stderr)
                    break

    _write_reports(out_dir=report_dir, records=records, recommendations=recommendations, args=args)

    _print_summary(
        title="M4 compaction plan (post)",
        records=records,
        recommendations=recommendations,
        dry_run=bool(args.dry_run),
        compaction_mode=compaction_mode,
    )
    print(f"\nwrote report: {rel(report_dir / 'compaction_report.json', repo_root=repo_root)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
