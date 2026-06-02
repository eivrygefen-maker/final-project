#!/usr/bin/env python3
"""Dry-run manifest for physics_integrity diagnostics cleanup (never deletes or moves files)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_checkpoint_pipeline_lib import (  # noqa: E402
    CONV_DIAG,
    PIPELINE_EXPORT_MANIFEST,
    SOLVER_BENCHMARKS_ROOT,
    write_json,
)

PHYSICS_ROOT = SCRIPT_DIR.parent
CONV_MESH_L_PROD = PHYSICS_ROOT / "v2_mesh_convergence" / "mesh" / "L_prod"
MESH_CONVERGENCE_MANIFEST = PHYSICS_ROOT / "configs" / "v2_mesh_convergence_manifest.json"
SCRIPTS_ROOT = PHYSICS_ROOT / "scripts"
CONFIGS_ROOT = PHYSICS_ROOT / "configs"

# Official A+B+C rich pipeline PASS (2026-06-01).
P0_EXACT_DIR_NAMES = frozenset(
    {
        "st_worker_scaling_L_prod_rich_safe_20260601T164739Z",
        "checkpoint_solve_mkl_pardiso_full9_20260601T203438Z",
    }
)
P1_TIMING_CHECKPOINT_NAME = "st_worker_scaling_L_prod_20260531T083626Z"

P0_SUBPATH_MARKERS = (
    "rich_modal",
    "rich_modal_post",
    "A_B_C_RICH_PIPELINE_PASS.md",
)

KEEP_KEYWORDS_P1 = (
    "rich_safe_20260601T164739Z",
    "checkpoint_solve_mkl_pardiso_full9_20260601T203438Z",
    P1_TIMING_CHECKPOINT_NAME,
    "checkpoint_solve_mkl_pardiso",
    "checkpoint_multi_mkl_pardiso_full9",
    "env_audit_",
    "cleanup_dry_run_",
    "cleanup_inventory_pre_lhs_",
)

ARCHIVE_KEYWORDS = (
    "target_density_experiment_",
    "target_alignment_experiment_",
    "checkpoint_multi_",
)

ALLOWED_SCAN_ROOTS = (
    SOLVER_BENCHMARKS_ROOT.resolve(),
    CONV_DIAG.resolve(),
)


def _dir_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
        return body if isinstance(body, dict) else None
    except Exception:
        return None


def _is_under_allowed_scan_roots(path: Path) -> bool:
    resolved = path.expanduser().resolve()
    for root in ALLOWED_SCAN_ROOTS:
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _solve_dir_has_official_abc_artifacts(solve_dir: Path) -> bool:
    if not solve_dir.is_dir():
        return False
    return (
        (solve_dir / "rich_modal").is_dir()
        and (solve_dir / "rich_modal_post").is_dir()
        and (solve_dir / "A_B_C_RICH_PIPELINE_PASS.md").is_file()
    )


def _is_p0_exact_name(name: str) -> bool:
    return name in P0_EXACT_DIR_NAMES


def _probe_checkpoint_dir(path: Path) -> Dict[str, Any]:
    manifest_path = path / PIPELINE_EXPORT_MANIFEST
    body = _read_json(manifest_path)
    status = body.get("status") if body else None
    return {
        "has_export_manifest": body is not None,
        "export_manifest_status": status,
        "export_manifest_path": str(manifest_path.resolve()) if body else None,
    }


def _probe_solve_dir(path: Path) -> Dict[str, Any]:
    result_path = path / "result.json"
    body = _read_json(result_path)
    rich = body.get("rich_modal_export") if body else None
    requested: Optional[bool] = None
    if isinstance(rich, dict) and "requested" in rich:
        requested = bool(rich.get("requested"))
    mode_count: Optional[int] = None
    if body is not None:
        rm = rich if isinstance(rich, dict) else {}
        if rm.get("mode_count") is not None:
            try:
                mode_count = int(rm["mode_count"])
            except (TypeError, ValueError):
                pass
        if mode_count is None and (path / "rich_modal" / "rich_modal_manifest.json").is_file():
            rm_man = _read_json(path / "rich_modal" / "rich_modal_manifest.json")
            if rm_man and rm_man.get("mode_count") is not None:
                try:
                    mode_count = int(rm_man["mode_count"])
                except (TypeError, ValueError):
                    pass
    summary_status = None
    if body and isinstance(body.get("summary"), dict):
        summary_status = body["summary"].get("status")
    return {
        "has_result_json": body is not None,
        "result_status": body.get("status") if body else None,
        "summary_status": summary_status,
        "rich_modal_export_requested": requested,
        "mode_count": mode_count,
        "has_official_abc_artifacts": _solve_dir_has_official_abc_artifacts(path),
    }


def _name_matches_keyword(name: str, keywords: Tuple[str, ...]) -> bool:
    lower = name.lower()
    return any(k.lower() in lower for k in keywords)


def _classify_entry(
    path: Path,
    *,
    label: str,
) -> Dict[str, Any]:
    name = path.name
    is_dir = path.is_dir()
    probe: Dict[str, Any] = {}
    if label == "solver_benchmark_run" and is_dir:
        probe = _probe_solve_dir(path)
    elif label == "checkpoint_or_diagnostic" and is_dir:
        probe = _probe_checkpoint_dir(path)

    official_abc = _is_p0_exact_name(name) or bool(probe.get("has_official_abc_artifacts"))
    protected = official_abc or _is_p0_exact_name(name) or name == P1_TIMING_CHECKPOINT_NAME

    export_status = probe.get("export_manifest_status")
    rich_requested = probe.get("rich_modal_export_requested")
    has_export_manifest = bool(probe.get("has_export_manifest"))
    result_status = probe.get("result_status")

    category = "review_needed"
    reason = "default review"

    if _is_p0_exact_name(name):
        category = "keep"
        protected = True
        official_abc = True
        reason = "official A+B+C PASS path (exact name)"
    elif probe.get("has_official_abc_artifacts"):
        category = "keep"
        protected = True
        official_abc = True
        reason = "official A+B+C solve artifacts (rich_modal, rich_modal_post, PASS marker)"
    elif name == P1_TIMING_CHECKPOINT_NAME:
        category = "keep"
        protected = True
        reason = "P1 MKL timing baseline checkpoint (documented reference)"
    elif _name_matches_keyword(name, KEEP_KEYWORDS_P1) and (
        export_status == "PASS" or rich_requested is True or result_status == "PASS"
    ):
        category = "keep"
        protected = True
        reason = "P1 keep keyword with PASS manifest or rich modal / solve PASS"
    elif _name_matches_keyword(name, ("cleanup_dry_run_", "cleanup_inventory_pre_lhs_")):
        category = "keep"
        protected = True
        reason = "cleanup audit trail manifest"
    elif _name_matches_keyword(name, ("env_audit_",)):
        category = "keep"
        protected = True
        reason = "environment audit artifact"
    elif label == "checkpoint_or_diagnostic" and is_dir:
        if not has_export_manifest:
            category = "delete"
            reason = "checkpoint dir without checkpoint_export_manifest.json (incomplete or crashed export)"
        elif export_status != "PASS":
            category = "delete"
            reason = f"export manifest status={export_status!r} (not PASS)"
        elif _name_matches_keyword(name, ARCHIVE_KEYWORDS):
            category = "archive"
            reason = "experiment checkpoint; archive before delete"
        else:
            category = "archive"
            reason = "non-official checkpoint with PASS manifest; archive after confirming superseded"
    elif label == "solver_benchmark_run" and is_dir:
        if rich_requested is False:
            category = "delete"
            reason = "solve dir with rich_modal_export.requested=false (pre-fix or timing-only; not synthesis)"
        elif rich_requested is True and result_status == "PASS":
            category = "archive"
            reason = "rich modal solve PASS but not official PASS dir name; archive as reference"
        elif result_status == "PASS":
            category = "archive"
            reason = "timing/parity solve PASS; archive if not referenced in docs"
        elif _name_matches_keyword(name, ARCHIVE_KEYWORDS):
            category = "archive"
            reason = "legacy multi-benchmark run; archive"
        elif not probe.get("has_result_json"):
            category = "delete"
            reason = "solver benchmark dir without result.json"
        else:
            category = "review_needed"
            reason = "solver benchmark run needs manual review"
    elif not is_dir and name.endswith((".json", ".md")):
        if _name_matches_keyword(name, ("v2_mesh_convergence_summary", "cleanup_dry_run", "cleanup_inventory")):
            category = "keep"
            protected = True
            reason = "summary or cleanup inventory artifact"
        else:
            category = "review_needed"
            reason = "top-level diagnostic file; review before archive"

    entry: Dict[str, Any] = {
        "label": label,
        "path": str(path.resolve()),
        "name": name,
        "is_dir": is_dir,
        "size_bytes": _dir_size_bytes(path),
        "modified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(path.stat().st_mtime))
        if path.exists()
        else None,
        "category": category,
        "protected": bool(protected),
        "official_abc_pass": bool(official_abc),
        "reason": reason,
        "export_manifest_status": export_status,
        "rich_modal_export_requested": rich_requested,
        "mode_count": probe.get("mode_count"),
    }
    entry.update({k: v for k, v in probe.items() if k not in entry})
    return entry


def _infrastructure_awareness(*, inventory_extended: bool) -> List[Dict[str, Any]]:
    """P0/P1 paths outside solver_benchmarks top-level scan (never cleanup targets)."""
    entries: List[Dict[str, Any]] = []
    if inventory_extended:
        if CONV_MESH_L_PROD.is_dir():
            entries.append(
                {
                    "label": "infrastructure_p0",
                    "path": str(CONV_MESH_L_PROD.resolve()),
                    "name": "L_prod",
                    "is_dir": True,
                    "size_bytes": _dir_size_bytes(CONV_MESH_L_PROD),
                    "category": "keep",
                    "protected": True,
                    "official_abc_pass": False,
                    "reason": "Stage A rebuild input mesh (v2_mesh_convergence/mesh/L_prod); not a diagnostics dump",
                    "export_manifest_status": None,
                    "rich_modal_export_requested": None,
                    "mode_count": None,
                }
            )
        if MESH_CONVERGENCE_MANIFEST.is_file():
            entries.append(
                {
                    "label": "infrastructure_p0",
                    "path": str(MESH_CONVERGENCE_MANIFEST.resolve()),
                    "name": MESH_CONVERGENCE_MANIFEST.name,
                    "is_dir": False,
                    "size_bytes": _dir_size_bytes(MESH_CONVERGENCE_MANIFEST),
                    "category": "keep",
                    "protected": True,
                    "official_abc_pass": False,
                    "reason": "Tracked L_prod mesh/case manifest required to reproduce Stage A",
                    "export_manifest_status": None,
                    "rich_modal_export_requested": None,
                    "mode_count": None,
                }
            )
    entries.append(
        {
            "label": "hard_safety",
            "path": str(SCRIPTS_ROOT.resolve()),
            "name": "scripts",
            "is_dir": True,
            "size_bytes": None,
            "category": "keep",
            "protected": True,
            "official_abc_pass": False,
            "reason": "Active pipeline source (v2_b3_*); never treat as generated output",
            "export_manifest_status": None,
            "rich_modal_export_requested": None,
            "mode_count": None,
        }
    )
    entries.append(
        {
            "label": "hard_safety",
            "path": str(CONFIGS_ROOT.resolve()),
            "name": "configs",
            "is_dir": True,
            "size_bytes": None,
            "category": "keep",
            "protected": True,
            "official_abc_pass": False,
            "reason": "Tracked experiment configs; not cleanup targets (use git for config changes)",
            "export_manifest_status": None,
            "rich_modal_export_requested": None,
            "mode_count": None,
        }
    )
    return entries


def _scan_root(root: Path, *, label: str) -> List[Dict[str, Any]]:
    if not _is_under_allowed_scan_roots(root):
        raise ValueError(f"refusing to scan outside allowed diagnostics scope: {root}")
    entries: List[Dict[str, Any]] = []
    if not root.is_dir():
        return entries
    for child in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True):
        entries.append(_classify_entry(child, label=label))
    return entries


def _bucket_entries(entries: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    buckets: Dict[str, List[Dict[str, Any]]] = {
        "recommended_keep": [],
        "recommended_archive": [],
        "recommended_delete": [],
        "review_needed": [],
    }
    for entry in entries:
        cat = str(entry.get("category") or "review_needed")
        if cat == "keep":
            buckets["recommended_keep"].append(entry)
        elif cat == "archive":
            buckets["recommended_archive"].append(entry)
        elif cat == "delete":
            buckets["recommended_delete"].append(entry)
        else:
            buckets["review_needed"].append(entry)
    return buckets


def _size_totals_by_category(entries: List[Dict[str, Any]]) -> Dict[str, int]:
    totals: Dict[str, int] = {"keep": 0, "archive": 0, "delete": 0, "review_needed": 0}
    for entry in entries:
        cat = str(entry.get("category") or "review_needed")
        if cat not in totals:
            totals[cat] = 0
        totals[cat] += int(entry.get("size_bytes") or 0)
    return totals


def _p0_do_not_delete_list(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for e in entries:
        if not e.get("protected"):
            continue
        name = str(e.get("name") or "")
        label = str(e.get("label") or "")
        if e.get("official_abc_pass") or _is_p0_exact_name(name):
            out.append(e)
        elif label in ("infrastructure_p0", "hard_safety"):
            out.append(e)
        elif name == P1_TIMING_CHECKPOINT_NAME:
            out.append(e)
    return out


def _write_markdown(
    path: Path,
    *,
    manifest: Dict[str, Any],
    buckets: Dict[str, List[Dict[str, Any]]],
    p0_list: List[Dict[str, Any]],
) -> None:
    totals = manifest.get("size_bytes_by_category") or {}
    lines = [
        "# Diagnostics cleanup dry-run",
        "",
        f"- manifest: `{manifest.get('output_json')}`",
        f"- generated_utc: `{manifest.get('generated_utc')}`",
        f"- mode: `{manifest.get('mode')}`",
        f"- deletions performed: **no**",
        f"- files moved: **no**",
        "",
        "## Size by category",
        "",
        f"| category | count | size (MiB) |",
        f"|----------|------:|-----------:|",
    ]
    counts = manifest.get("counts") or {}
    for cat in ("keep", "archive", "delete", "review_needed"):
        n = counts.get(f"category_{cat}", 0)
        mib = int(totals.get(cat, 0)) / (1024 * 1024)
        lines.append(f"| {cat} | {n} | {mib:.1f} |")
    lines.extend(
        [
            "",
            "## DO NOT DELETE (P0 / protected)",
            "",
            "These paths must not be deleted or moved during cleanup without explicit sign-off.",
            "",
        ]
    )
    if not p0_list:
        lines.append("- (none detected — verify official PASS dirs exist on disk)")
    else:
        for e in p0_list:
            lines.append(
                f"- `{e.get('path')}` — {e.get('reason')} "
                f"(protected={e.get('protected')}, official_abc_pass={e.get('official_abc_pass')})"
            )
    lines.extend(
        [
            "",
            "## Counts",
            "",
            f"- entries total: {counts.get('entries_total', 0)}",
            f"- recommended_keep: {counts.get('recommended_keep', 0)}",
            f"- recommended_archive: {counts.get('recommended_archive', 0)}",
            f"- recommended_delete: {counts.get('recommended_delete', 0)}",
            f"- review_needed: {counts.get('review_needed', 0)}",
            "",
            "See JSON manifest for full per-entry metadata (`export_manifest_status`, "
            "`rich_modal_export_requested`, `mode_count`).",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_cleanup_dry_run(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate cleanup dry-run manifest (no deletions, no moves).",
    )
    parser.add_argument(
        "--output-json",
        help="Output manifest path (default: solver_benchmarks/cleanup_dry_run_<utc>.json)",
    )
    parser.add_argument(
        "--inventory-extended",
        action="store_true",
        help="Include P0 infrastructure rows (mesh/L_prod, v2_mesh_convergence_manifest.json).",
    )
    args = parser.parse_args(argv)

    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = (
        Path(args.output_json).expanduser().resolve()
        if args.output_json
        else SOLVER_BENCHMARKS_ROOT / f"cleanup_dry_run_{ts}.json"
    )
    if not _is_under_allowed_scan_roots(out.parent):
        raise SystemExit(
            f"[B3_cleanup_dry_run] refuse to write manifest outside diagnostics scope: {out.parent}"
        )
    out.parent.mkdir(parents=True, exist_ok=True)

    benchmark_entries = _scan_root(SOLVER_BENCHMARKS_ROOT, label="solver_benchmark_run")
    checkpoint_entries: List[Dict[str, Any]] = []
    for child_name in sorted(CONV_DIAG.iterdir(), key=lambda p: p.name if p.exists() else ""):
        if not child_name.exists():
            continue
        if child_name.name == "solver_benchmarks":
            continue
        if child_name.is_dir() and not (
            child_name.name.startswith("st_worker_scaling_")
            or child_name.name.startswith("env_audit_")
            or child_name.name.startswith("target_")
            or child_name.name.startswith("cleanup_")
            or child_name.name.startswith("v2_mesh_convergence")
        ):
            checkpoint_entries.append(
                _classify_entry(
                    child_name,
                    label="checkpoint_or_diagnostic",
                )
            )
            continue
        if child_name.is_dir() and (
            child_name.name.startswith("st_worker_scaling_")
            or child_name.name.startswith("env_audit_")
            or child_name.name.startswith("target_")
        ):
            checkpoint_entries.append(_classify_entry(child_name, label="checkpoint_or_diagnostic"))
        elif child_name.is_file() and (
            child_name.name.startswith("cleanup_")
            or child_name.name.startswith("v2_mesh_convergence_summary")
        ):
            checkpoint_entries.append(_classify_entry(child_name, label="checkpoint_or_diagnostic"))

    infra = _infrastructure_awareness(inventory_extended=bool(args.inventory_extended))
    all_entries = benchmark_entries + checkpoint_entries + infra
    buckets = _bucket_entries(all_entries)
    size_by_cat = _size_totals_by_category(all_entries)

    p0_paths = [
        e for e in all_entries
        if e.get("official_abc_pass") or _is_p0_exact_name(str(e.get("name") or ""))
    ]
    p0_do_not_delete = _p0_do_not_delete_list(all_entries)

    manifest: Dict[str, Any] = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "cleanup_dry_run_only",
        "deletions_performed": False,
        "files_moved": False,
        "output_json": str(out),
        "notes": [
            "Advisory only. No files were deleted or moved.",
            "Official A+B+C PASS dirs are P0 protected.",
            "scripts/ and configs/ are never cleanup targets.",
            "Use recommended_archive before recommended_delete.",
        ],
        "roots_scanned": [str(r) for r in ALLOWED_SCAN_ROOTS],
        "p0_exact_dir_names": sorted(P0_EXACT_DIR_NAMES),
        "p0_subpath_markers": list(P0_SUBPATH_MARKERS),
        "p0_paths": [e.get("path") for e in p0_paths],
        "counts": {
            "entries_total": len(all_entries),
            "recommended_keep": len(buckets["recommended_keep"]),
            "recommended_archive": len(buckets["recommended_archive"]),
            "recommended_delete": len(buckets["recommended_delete"]),
            "review_needed": len(buckets["review_needed"]),
            "category_keep": sum(1 for e in all_entries if e.get("category") == "keep"),
            "category_archive": sum(1 for e in all_entries if e.get("category") == "archive"),
            "category_delete": sum(1 for e in all_entries if e.get("category") == "delete"),
            "category_review_needed": sum(1 for e in all_entries if e.get("category") == "review_needed"),
        },
        "size_bytes_by_category": size_by_cat,
        "recommended_keep": buckets["recommended_keep"],
        "recommended_archive": buckets["recommended_archive"],
        "recommended_delete": buckets["recommended_delete"],
        "review_needed": buckets["review_needed"],
        "do_not_delete_p0": p0_do_not_delete,
    }
    write_json(out, manifest)
    md_path = out.with_suffix(".md")
    _write_markdown(md_path, manifest=manifest, buckets=buckets, p0_list=p0_do_not_delete)

    print(f"[B3_cleanup_dry_run] wrote {out}", flush=True)
    print(f"[B3_cleanup_dry_run] wrote {md_path}", flush=True)
    print(
        f"[B3_cleanup_dry_run] keep={len(buckets['recommended_keep'])} "
        f"archive={len(buckets['recommended_archive'])} "
        f"delete={len(buckets['recommended_delete'])} "
        f"review={len(buckets['review_needed'])} "
        f"P0={len(p0_paths)}",
        flush=True,
    )
    return 0


def main() -> int:
    return run_cleanup_dry_run()


if __name__ == "__main__":
    raise SystemExit(main())
