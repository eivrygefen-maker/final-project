#!/usr/bin/env python3
"""Dry-run manifest for old solver diagnostics cleanup (does not delete anything)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_checkpoint_pipeline_lib import CONV_DIAG, SOLVER_BENCHMARKS_ROOT, write_json

KEEP_KEYWORDS = (
    "checkpoint_multi_mkl_pardiso_full9",
    "checkpoint_solve_mkl_pardiso",
    "st_worker_scaling_L_prod_20260531T083626Z",
    "env_audit_",
)


def _dir_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def _should_keep(name: str) -> bool:
    lower = name.lower()
    return any(k.lower() in lower for k in KEEP_KEYWORDS)


def _scan_root(root: Path, *, label: str) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    if not root.is_dir():
        return entries
    for child in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True):
        name = child.name
        entry: Dict[str, Any] = {
            "label": label,
            "path": str(child.resolve()),
            "name": name,
            "is_dir": child.is_dir(),
            "size_bytes": _dir_size_bytes(child),
            "modified_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(child.stat().st_mtime)
            )
            if child.exists()
            else None,
            "recommended_action": "keep" if _should_keep(name) else "review_for_archive_or_delete",
            "has_result_json": (child / "result.json").is_file() if child.is_dir() else False,
            "has_export_manifest": (child / "checkpoint_export_manifest.json").is_file()
            if child.is_dir()
            else False,
            "has_summary": False,
        }
        if entry["has_result_json"]:
            try:
                body = json.loads((child / "result.json").read_text(encoding="utf-8"))
                entry["has_summary"] = isinstance(body.get("summary"), dict)
                entry["summary_status"] = (body.get("summary") or {}).get("status")
            except Exception:
                pass
        entries.append(entry)
    return entries


def run_cleanup_dry_run(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate cleanup dry-run manifest (no deletions).")
    parser.add_argument(
        "--output-json",
        help="Output manifest path (default: solver_benchmarks/cleanup_dry_run_<utc>.json)",
    )
    args = parser.parse_args(argv)

    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = (
        Path(args.output_json).expanduser().resolve()
        if args.output_json
        else SOLVER_BENCHMARKS_ROOT / f"cleanup_dry_run_{ts}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    benchmark_entries = _scan_root(SOLVER_BENCHMARKS_ROOT, label="solver_benchmark_run")
    checkpoint_entries = [
        e
        for e in _scan_root(CONV_DIAG, label="checkpoint_or_diagnostic")
        if e["name"].startswith("st_worker_scaling_") or e["name"].startswith("env_audit_")
    ]

    all_entries = benchmark_entries + checkpoint_entries
    keep = [e for e in all_entries if e["recommended_action"] == "keep"]
    review = [e for e in all_entries if e["recommended_action"] != "keep"]
    total_review_bytes = sum(int(e.get("size_bytes") or 0) for e in review)

    manifest: Dict[str, Any] = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "cleanup_dry_run_only",
        "deletions_performed": False,
        "notes": [
            "This manifest is advisory only. No files were deleted.",
            "Keep entries include validated MKL full9 runs and the canonical L_prod checkpoint.",
            "Review entries may be archived after official pipeline promotion.",
        ],
        "roots_scanned": [str(SOLVER_BENCHMARKS_ROOT.resolve()), str(CONV_DIAG.resolve())],
        "counts": {
            "entries_total": len(all_entries),
            "recommended_keep": len(keep),
            "recommended_review": len(review),
        },
        "review_size_bytes": total_review_bytes,
        "recommended_keep": keep,
        "recommended_review": review,
    }
    write_json(out, manifest)
    md = out.with_suffix(".md")
    lines = [
        "# Diagnostics cleanup dry-run",
        "",
        f"- manifest: `{out}`",
        f"- deletions performed: **no**",
        f"- entries total: {len(all_entries)}",
        f"- recommended keep: {len(keep)}",
        f"- recommended review: {len(review)}",
        f"- review size: {total_review_bytes / (1024 * 1024):.1f} MiB",
        "",
        "Review candidates are listed in the JSON manifest under `recommended_review`.",
        "",
    ]
    md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[B3_cleanup_dry_run] wrote {out}", flush=True)
    print(f"[B3_cleanup_dry_run] review candidates={len(review)} size={total_review_bytes / (1024 * 1024):.1f} MiB", flush=True)
    return 0


def main() -> int:
    return run_cleanup_dry_run()


if __name__ == "__main__":
    raise SystemExit(main())
