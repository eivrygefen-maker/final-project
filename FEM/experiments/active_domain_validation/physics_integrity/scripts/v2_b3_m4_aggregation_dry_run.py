#!/usr/bin/env python3
"""M4.4.1a — aggregation dry-run validator (planned worker outputs; no aggregation)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_m4_lprod_interfaces import CHUNK_TARGETS_SCHEMA, validate_chunk_targets_doc  # noqa: E402
from v2_b3_petsc_util import write_json_atomic  # noqa: E402
from v2_b3_resolve_pilot_core_config import _repo_relative  # noqa: E402


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _detect_repo_root(start: Path) -> Path:
    cur = start.resolve()
    while cur.parent != cur:
        if (cur / ".git").exists():
            return cur
        cur = cur.parent
    raise RuntimeError("Could not detect repository root (missing .git ancestor)")


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected JSON object")
    return data


def validate_aggregation_dry_run(
    *,
    repo_root: Path,
    run_root: Path,
) -> Dict[str, Any]:
    lprod_dir = run_root / "lprod"
    target_plan_path = lprod_dir / "lprod_target_plan.json"
    chunk_plan_path = lprod_dir / "worker_chunk_plan.preview.json"
    agg_plan_path = lprod_dir / "aggregation_plan.json"
    worker_cmds_path = lprod_dir / "worker_commands.json"

    errors: List[str] = []
    warnings: List[str] = []

    for p, label in (
        (target_plan_path, "lprod_target_plan.json"),
        (chunk_plan_path, "worker_chunk_plan.preview.json"),
    ):
        if not p.is_file():
            errors.append(f"missing {label}")

    if errors:
        return {
            "schema": "m4_aggregation_dry_run_v1",
            "will_execute": False,
            "status": "FAIL",
            "errors": errors,
            "warnings": warnings,
        }

    target_plan = _load_json(target_plan_path)
    chunk_plan = _load_json(chunk_plan_path)
    plan_targets = [float(t) for t in (target_plan.get("targets_hz") or [])]
    chunks = chunk_plan.get("chunks") or []

    if not chunks:
        errors.append("worker_chunk_plan has no chunks")

    assigned: List[float] = []
    chunk_rows: List[Dict[str, Any]] = []
    seen_targets: List[float] = []

    for chunk in chunks:
        chunk_id = str(chunk.get("chunk_id"))
        chunk_dir = run_root / "worker_results" / chunk_id
        targets_path = chunk_dir / "chunk_targets.json"
        row: Dict[str, Any] = {
            "chunk_id": chunk_id,
            "chunk_dir": _repo_relative(chunk_dir, repo_root=repo_root),
            "chunk_targets_json": _repo_relative(targets_path, repo_root=repo_root),
            "chunk_targets_exists": targets_path.is_file(),
            "worker_result_json": _repo_relative(chunk_dir / "worker_result.json", repo_root=repo_root),
            "solver_result_json": _repo_relative(chunk_dir / "solver_result.json", repo_root=repo_root),
            "command_preview_sh": _repo_relative(chunk_dir / "worker_command.sh", repo_root=repo_root),
            "target_count": len(chunk.get("targets_hz") or []),
        }
        if not targets_path.is_file():
            errors.append(f"missing chunk_targets.json: {targets_path}")
        else:
            try:
                doc = _load_json(targets_path)
                if doc.get("schema") != CHUNK_TARGETS_SCHEMA:
                    errors.append(f"{chunk_id}: schema={doc.get('schema')!r}")
                val = validate_chunk_targets_doc(doc)
                for e in val:
                    errors.append(f"{chunk_id}: {e}")
                for t in doc.get("targets") or []:
                    seen_targets.append(float(t["target_hz"]))
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                errors.append(f"{chunk_id}: invalid chunk_targets.json ({exc})")

        assigned.extend(float(t) for t in (chunk.get("targets_hz") or []))
        chunk_rows.append(row)

    if len(assigned) != len(plan_targets):
        errors.append(
            f"chunk assignment count {len(assigned)} != plan targets {len(plan_targets)}"
        )
    else:
        a_sorted = sorted(assigned)
        t_sorted = sorted(plan_targets)
        for a, t in zip(a_sorted, t_sorted):
            if abs(a - t) > 1e-4:
                errors.append("chunk targets_hz do not match lprod_target_plan.targets_hz")
                break

    if len(seen_targets) != len(plan_targets):
        errors.append(
            f"chunk_targets.json assignment {len(seen_targets)} != plan {len(plan_targets)}"
        )
    else:
        s_sorted = sorted(seen_targets)
        t_sorted = sorted(plan_targets)
        for s, t in zip(s_sorted, t_sorted):
            if abs(s - t) > 1e-4:
                errors.append("chunk_targets frequencies do not match lprod_target_plan")
                break

    dup_check: Set[float] = set()
    for hz in seen_targets:
        key = round(hz, 4)
        if key in dup_check:
            errors.append(f"duplicate target assignment at {hz} Hz")
        dup_check.add(key)

    agg_out = run_root / "aggregation"
    output_paths = {
        "aggregation_result_json": agg_out / "aggregation_result.json",
        "modes_catalog_jsonl": agg_out / "modes_catalog.jsonl",
        "modes_summary_json": agg_out / "modes_summary.json",
        "runtime_summary_json": agg_out / "runtime_summary.json",
    }
    outputs_ready = {k: p.parent.exists() for k, p in output_paths.items()}
    if not agg_out.is_dir():
        warnings.append("aggregation/ directory missing — will be created at execution")

    uses_target_list = False
    if worker_cmds_path.is_file():
        wdoc = _load_json(worker_cmds_path)
        for c in wdoc.get("chunks") or []:
            cmd = (c.get("commands") or {}).get("m4_4_target_list_solve") or ""
            if "v2_b3_checkpoint_solve_target_list.py" in cmd:
                uses_target_list = True
                break
    else:
        warnings.append("worker_commands.json missing — cannot verify command interface")

    if not uses_target_list:
        errors.append("worker_commands do not reference v2_b3_checkpoint_solve_target_list.py")

    status = "PASS" if not errors else "FAIL"
    return {
        "schema": "m4_aggregation_dry_run_v1",
        "will_execute": False,
        "mode": "m4_4_1a_dry_run",
        "generated_utc": _utc_now(),
        "status": status,
        "sample_id": target_plan.get("sample_id"),
        "run_id": target_plan.get("run_id"),
        "target_assignment": {
            "plan_target_count": len(plan_targets),
            "chunk_assigned_count": len(assigned),
            "chunk_targets_json_count": len(seen_targets),
            "complete": not errors and len(seen_targets) == len(plan_targets),
        },
        "chunk_count": len(chunks),
        "chunks": chunk_rows,
        "aggregation_plan_path": _repo_relative(agg_plan_path, repo_root=repo_root)
        if agg_plan_path.is_file()
        else None,
        "aggregation_output_paths": {
            k: _repo_relative(v, repo_root=repo_root) for k, v in output_paths.items()
        },
        "aggregation_dirs_ready": outputs_ready,
        "errors": errors,
        "warnings": warnings,
    }


def run_dry_run(*, repo_root: Path, run_root: Path, force: bool) -> int:
    out_path = run_root / "aggregation" / "aggregation_dry_run.json"
    if out_path.is_file() and not force:
        raise FileExistsError(f"aggregation dry-run exists (use --force): {out_path}")

    report = validate_aggregation_dry_run(repo_root=repo_root, run_root=run_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out_path, report)
    md_lines = [
        f"# Aggregation dry-run — {report.get('sample_id')}",
        "",
        f"- will_execute: **false**",
        f"- status: `{report.get('status')}`",
        f"- target assignment: **{report.get('target_assignment', {}).get('chunk_targets_json_count')}**"
        f"/{report.get('target_assignment', {}).get('plan_target_count')}**",
        "",
    ]
    if report.get("errors"):
        md_lines.append("## Errors")
        for e in report["errors"]:
            md_lines.append(f"- {e}")
    (run_root / "aggregation" / "aggregation_dry_run.md").write_text(
        "\n".join(md_lines) + "\n",
        encoding="utf-8",
    )

    print("will_execute=false")
    ta = report.get("target_assignment") or {}
    print(
        f"target assignment complete: {ta.get('chunk_targets_json_count')}/{ta.get('plan_target_count')}"
    )
    print(f"chunk_count={report.get('chunk_count')}")
    print(f"status={report.get('status')}")
    print(f"wrote {out_path.name}")
    return 0 if report.get("status") == "PASS" else 2


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="M4.4.1a aggregation dry-run validator.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    repo_root = _detect_repo_root(SCRIPT_DIR)
    run_root = args.run_dir if args.run_dir.is_absolute() else repo_root / args.run_dir
    run_root = run_root.resolve()

    try:
        return run_dry_run(repo_root=repo_root, run_root=run_root, force=bool(args.force))
    except FileExistsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
