#!/usr/bin/env python3
"""Collect per-shape physical validation baselines from run dirs or shared summaries."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[4]
FEM_SCRIPTS = SCRIPT_DIR.parents[3] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(FEM_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(FEM_SCRIPTS))

from m4_shape_registry import normalize_shape_key  # noqa: E402
from evaluate_shape_physical_acceptance import (  # noqa: E402
    ACCEPTANCE_JSON_REL,
    RECOMMENDED_MIN_SAMPLES,
    evaluate_shape_physical_acceptance,
    render_markdown_report,
    write_shape_physical_acceptance,
)
from v2_b3_petsc_util import write_json_atomic  # noqa: E402

BASELINE_SCHEMA = "m4_shape_validation_baseline_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_acceptance(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        return doc if isinstance(doc, dict) else None
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _discover_acceptance_reports(
    *,
    shape: str,
    runs_root: Optional[Path],
    shared_root: Optional[Path],
) -> List[Dict[str, Any]]:
    reports: List[Dict[str, Any]] = []
    seen: set[str] = set()

    def _add(path: Path) -> None:
        doc = _load_acceptance(path)
        if not doc:
            return
        sid = str(doc.get("sample_id") or path.parent.parent.parent.name)
        key = f"{doc.get('run_id')}:{sid}"
        if key in seen:
            return
        seen.add(key)
        reports.append({**doc, "_source_path": str(path)})

    if shared_root is not None:
        summary_dir = shared_root / shape / "summaries"
        if summary_dir.is_dir():
            for path in sorted(summary_dir.glob("*_shape_physical_acceptance.json")):
                _add(path)

    if runs_root is not None:
        guitars = runs_root / "guitars"
        if guitars.is_dir():
            for sample_dir in sorted(guitars.iterdir()):
                if not sample_dir.is_dir():
                    continue
                for run_dir in sorted((sample_dir / "runs").glob("*")) if (sample_dir / "runs").is_dir() else []:
                    _add(run_dir / ACCEPTANCE_JSON_REL)

    return reports


def collect_shape_validation_baseline(
    *,
    shape: str,
    runs_root: Optional[Path] = None,
    shared_root: Optional[Path] = None,
    evaluate_missing: bool = False,
) -> Dict[str, Any]:
    shape_name = normalize_shape_key(shape)
    reports = _discover_acceptance_reports(
        shape=shape_name,
        runs_root=runs_root,
        shared_root=shared_root,
    )

    if evaluate_missing and runs_root is not None:
        guitars = runs_root / "guitars"
        if guitars.is_dir():
            for sample_dir in guitars.iterdir():
                if not sample_dir.is_dir():
                    continue
                runs = sample_dir / "runs"
                if not runs.is_dir():
                    continue
                for run_dir in runs.iterdir():
                    if not run_dir.is_dir():
                        continue
                    acc_path = run_dir / ACCEPTANCE_JSON_REL
                    if acc_path.is_file():
                        continue
                    agg = run_dir / "aggregation" / "aggregation_result.json"
                    if not agg.is_file():
                        continue
                    sample_input_path = run_dir / "sample" / "sample_input.json"
                    sample_doc: Dict[str, Any] = {}
                    if sample_input_path.is_file():
                        try:
                            sample_doc = json.loads(sample_input_path.read_text(encoding="utf-8"))
                        except (OSError, ValueError, json.JSONDecodeError):
                            sample_doc = {}
                    run_shape = str(sample_doc.get("shape_name") or "")
                    if run_shape and normalize_shape_key(run_shape) != shape_name:
                        continue
                    report = evaluate_shape_physical_acceptance(run_root=run_dir, shape_key=shape_name)
                    write_shape_physical_acceptance(run_dir, report)
                    reports.append({**report, "_source_path": str(run_dir / ACCEPTANCE_JSON_REL)})

    sample_count = len(reports)
    baseline_status = (
        "SUFFICIENT_SAMPLE_COUNT"
        if sample_count >= RECOMMENDED_MIN_SAMPLES
        else "INSUFFICIENT_SAMPLE_COUNT"
    )

    status_counts: Dict[str, int] = {}
    deduped_counts: List[int] = []
    for rep in reports:
        st = str(rep.get("status") or "UNKNOWN")
        status_counts[st] = status_counts.get(st, 0) + 1
        try:
            deduped_counts.append(int((rep.get("metrics") or {}).get("deduped_mode_count") or 0))
        except (TypeError, ValueError):
            pass

    return {
        "schema": BASELINE_SCHEMA,
        "generated_utc": utc_now(),
        "shape_name": shape_name,
        "baseline_status": baseline_status,
        "recommended_min_samples": RECOMMENDED_MIN_SAMPLES,
        "sample_count": sample_count,
        "status_counts": status_counts,
        "deduped_mode_count_values": deduped_counts,
        "samples": [
            {
                "sample_id": rep.get("sample_id"),
                "run_id": rep.get("run_id"),
                "status": rep.get("status"),
                "profile_id": rep.get("profile_id"),
                "deduped_mode_count": (rep.get("metrics") or {}).get("deduped_mode_count"),
                "source_path": rep.get("_source_path"),
            }
            for rep in reports
        ],
        "notes": (
            "Baseline is advisory. Musical usefulness and profile tuning require "
            f">= {RECOMMENDED_MIN_SAMPLES} completed samples per shape."
        ),
    }


def write_baseline_outputs(
    *,
    out_dir: Path,
    shape: str,
    baseline: Dict[str, Any],
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    shape_name = normalize_shape_key(shape)
    json_path = out_dir / f"{shape_name}_baseline_summary.json"
    md_path = out_dir / f"{shape_name}_baseline_summary.md"
    write_json_atomic(json_path, baseline)
    lines = [
        f"# Shape validation baseline — {shape_name}",
        "",
        f"- baseline_status: **{baseline.get('baseline_status')}**",
        f"- sample_count: {baseline.get('sample_count')}",
        f"- recommended_min_samples: {baseline.get('recommended_min_samples')}",
        f"- status_counts: {baseline.get('status_counts')}",
        "",
        "## Samples",
    ]
    for row in baseline.get("samples") or []:
        lines.append(
            f"- `{row.get('sample_id')}` / `{row.get('run_id')}`: "
            f"status={row.get('status')} deduped={row.get('deduped_mode_count')}"
        )
    lines.extend(["", str(baseline.get("notes") or "")])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Collect shape validation baseline summaries.")
    parser.add_argument("--shape", required=True)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=REPO_ROOT
        / "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs",
    )
    parser.add_argument("--shared-root", type=Path, default=Path("/media/sf_gmar"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT
        / "FEM/experiments/active_domain_validation/physics_integrity/validation_baseline",
    )
    parser.add_argument("--evaluate-missing", action="store_true")
    parser.add_argument("--no-shared-scan", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    baseline = collect_shape_validation_baseline(
        shape=args.shape,
        runs_root=args.runs_root,
        shared_root=None if args.no_shared_scan else args.shared_root,
        evaluate_missing=bool(args.evaluate_missing),
    )
    json_path, md_path = write_baseline_outputs(out_dir=args.out_dir, shape=args.shape, baseline=baseline)
    print(f"baseline_status={baseline.get('baseline_status')}")
    print(f"recommended_min_samples={baseline.get('recommended_min_samples')}")
    print(f"sample_count={baseline.get('sample_count')}")
    print(f"written={json_path}")
    print(f"written={md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
