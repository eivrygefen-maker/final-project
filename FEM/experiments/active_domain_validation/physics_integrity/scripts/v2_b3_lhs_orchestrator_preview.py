#!/usr/bin/env python3
"""Generate dry-run command preview for M2 pilot samples (no execution)."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent

# Local utility used across physics_integrity scripts.
import sys

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_petsc_util import write_json_atomic  # noqa: E402


def _detect_repo_root(start: Path) -> Path:
    cur = start.resolve()
    while cur.parent != cur:
        if (cur / ".git").exists():
            return cur
        cur = cur.parent
    raise RuntimeError("Could not detect repository root (missing .git ancestor)")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                row = json.loads(s)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: invalid JSONL on line {i}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}: line {i} is not a JSON object")
            rows.append(row)
    return rows


def _mode_from_flags(row: Dict[str, Any]) -> str:
    if bool(row.get("synthesis_requested")):
        return "synthesis"
    if bool(row.get("rich_requested")):
        return "rich"
    return "timing"


def _has_placeholder_payload(row: Dict[str, Any]) -> bool:
    payload = row.get("parameter_payload") or {}
    geom = payload.get("geometry_delta")
    mat = payload.get("material_delta")
    geom_empty = isinstance(geom, dict) and len(geom) == 0
    mat_empty = isinstance(mat, dict) and len(mat) == 0
    return geom_empty and mat_empty


def _stage_initial_statuses(mode: str) -> Tuple[str, str, str]:
    if mode == "synthesis":
        return "PENDING", "PENDING", "PENDING"
    return "PENDING", "PENDING", "SKIPPED"


def _format_path(path: Path, *, repo_root: Path, absolute_paths: bool) -> str:
    if absolute_paths:
        return str(path.resolve())
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _repo_root_field(repo_root: Path, *, absolute_paths: bool) -> str:
    if absolute_paths:
        return str(repo_root.resolve())
    return "."


def _build_paths(
    repo_root: Path,
    mesh_level: str,
    run_id: str,
    *,
    absolute_paths: bool,
) -> Dict[str, str]:
    base = repo_root / "FEM" / "experiments" / "active_domain_validation" / "physics_integrity"
    checkpoint_dir = (
        base
        / "v2_mesh_convergence"
        / "diagnostics"
        / f"st_worker_scaling_{mesh_level}_{run_id}"
    )
    solve_dir = (
        base
        / "v2_mesh_convergence"
        / "diagnostics"
        / "solver_benchmarks"
        / f"checkpoint_solve_mkl_pardiso_full9_{run_id}"
    )
    rich_dir = solve_dir / "rich_modal"
    synthesis_dir = solve_dir / "rich_modal_post"
    return {
        "checkpoint_dir": _format_path(checkpoint_dir, repo_root=repo_root, absolute_paths=absolute_paths),
        "solve_dir": _format_path(solve_dir, repo_root=repo_root, absolute_paths=absolute_paths),
        "rich_modal_dir": _format_path(rich_dir, repo_root=repo_root, absolute_paths=absolute_paths),
        "synthesis_dir": _format_path(synthesis_dir, repo_root=repo_root, absolute_paths=absolute_paths),
    }


def _cmd_stage_a(paths: Dict[str, str], mesh_level: str) -> str:
    return (
        "python FEM/experiments/active_domain_validation/physics_integrity/scripts/"
        "v2_b3_checkpoint_export.py "
        f"--mesh-level {mesh_level} "
        "--B3-block-compose-backend csr_bulk "
        "--B3-synthesis-region-dofs off "
        f"--output-dir \"{paths['checkpoint_dir']}\""
    )


def _cmd_stage_b(paths: Dict[str, str], *, rich: bool) -> str:
    cmd = (
        "python FEM/experiments/active_domain_validation/physics_integrity/scripts/"
        "v2_b3_checkpoint_solve.py "
        f"--checkpoint-dir \"{paths['checkpoint_dir']}\" "
        "--factor-solver mkl_pardiso "
        "--target-set full9 "
        f"--output-dir \"{paths['solve_dir']}\""
    )
    if rich:
        cmd += " --B3-export-rich-modal-data"
    return cmd


def _cmd_stage_c(paths: Dict[str, str]) -> str:
    return (
        "python FEM/experiments/active_domain_validation/physics_integrity/scripts/"
        "v2_b3_rich_modal_post.py "
        f"--checkpoint-dir \"{paths['checkpoint_dir']}\" "
        f"--rich-modal-dir \"{paths['rich_modal_dir']}\" "
        f"--output-dir \"{paths['synthesis_dir']}\""
    )


def _preview_for_row(repo_root: Path, row: Dict[str, Any], *, absolute_paths: bool) -> Dict[str, Any]:
    sample_id = str(row.get("sample_id") or "").strip()
    if not sample_id:
        raise ValueError("sample row missing required sample_id")
    run_id = sample_id
    mesh_level = str(row.get("mesh_level") or "L_prod")
    target_set = str(row.get("target_set") or "full9")
    selection_reason = str(row.get("selection_reason") or "unspecified")
    mode = _mode_from_flags(row)
    a_status, b_status, c_status = _stage_initial_statuses(mode)
    placeholder_payload = _has_placeholder_payload(row)
    paths = _build_paths(repo_root, mesh_level, run_id, absolute_paths=absolute_paths)

    policy_rich_requested = bool(row.get("rich_requested"))
    policy_synthesis_requested = bool(row.get("synthesis_requested"))
    rich = mode in ("rich", "synthesis")
    c_requested = mode == "synthesis"

    warnings: List[str] = []
    if placeholder_payload:
        warnings.append("placeholder_parameter_payload")
        warnings.append("physical_lhs_ready=false")
    if policy_synthesis_requested and not policy_rich_requested:
        warnings.append("synthesis_implies_rich_export")

    return {
        "sample_id": sample_id,
        "run_id": run_id,
        "mode": mode,
        "selection_reason": selection_reason,
        "mesh_level": mesh_level,
        "target_set": target_set,
        "source_refs": row.get("source_refs"),
        "parameter_payload": row.get("parameter_payload"),
        "placeholder_parameter_payload": placeholder_payload,
        "physical_lhs_ready": not placeholder_payload,
        "warnings": warnings,
        "policy_flags": {
            "timing_only": bool(row.get("timing_only")),
            "rich_requested": policy_rich_requested,
            "synthesis_requested": policy_synthesis_requested,
            "effective_rich_requested": bool(rich),
        },
        "initial_stage_status": {"A": a_status, "B": b_status, "C": c_status},
        "predicted_commands": {
            "stage_a": _cmd_stage_a(paths, mesh_level=mesh_level),
            "stage_b": _cmd_stage_b(paths, rich=rich),
            "stage_c": _cmd_stage_c(paths) if c_requested else None,
        },
        "expected_environment": {
            "stage_a": "production .venv",
            "stage_b": "solver-mkl",
            "stage_c": "production .venv",
        },
        "predicted_output_paths": {
            **paths,
            "note": "preview_expected_only_not_executed",
        },
        "will_execute": False,
    }


def _write_markdown(path: Path, body: Dict[str, Any]) -> None:
    lines = [
        "# M2.2 dry-run orchestrator preview",
        "",
        f"- generated_utc: `{body.get('generated_utc')}`",
        f"- samples_jsonl: `{body.get('samples_jsonl')}`",
        f"- will_execute: `{body.get('will_execute')}`",
        "",
        "## Summary",
        "",
        f"- sample_count: `{body['summary']['sample_count']}`",
        f"- timing_count: `{body['summary']['timing_count']}`",
        f"- rich_count: `{body['summary']['rich_count']}`",
        f"- synthesis_count: `{body['summary']['synthesis_count']}`",
        f"- placeholder_warning_count: `{body['summary']['placeholder_warning_count']}`",
        "",
        "## Per sample",
        "",
        "| sample_id | run_id | mode | A | B | C | placeholder_warning |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in body.get("samples", []):
        st = row.get("initial_stage_status") or {}
        lines.append(
            f"| {row.get('sample_id')} | {row.get('run_id')} | {row.get('mode')} | "
            f"{st.get('A')} | {st.get('B')} | {st.get('C')} | "
            f"{row.get('placeholder_parameter_payload')} |"
        )
    lines.extend(
        [
            "",
            "## Warnings",
            "",
            "- `placeholder_parameter_payload` indicates this is orchestration smoke preview only.",
            "- `physical_lhs_ready=false` means no physical LHS interpretation should be made.",
            "- No commands were executed; this file is preview metadata only.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_preview(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate non-executing A/B/C command preview from pilot JSONL."
    )
    parser.add_argument("--samples-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", help="Optional markdown output path (default: sibling .md)")
    parser.add_argument("--repo-root", help="Optional explicit repo root (default: auto-detect)")
    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        help="Emit absolute paths in preview output (default: repo-relative).",
    )
    parser.add_argument("--force", action="store_true", help="Allow overwrite of preview outputs")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).expanduser().resolve() if args.repo_root else _detect_repo_root(SCRIPT_DIR)
    samples_path = Path(args.samples_jsonl).expanduser().resolve()
    out_json = Path(args.output_json).expanduser().resolve()
    out_md = Path(args.output_md).expanduser().resolve() if args.output_md else out_json.with_suffix(".md")

    if out_json.exists() and not args.force:
        raise SystemExit(f"output exists: {out_json} (use --force)")
    if out_md.exists() and not args.force:
        raise SystemExit(f"output exists: {out_md} (use --force)")

    rows = _read_jsonl(samples_path)
    previews = [_preview_for_row(repo_root, row, absolute_paths=bool(args.absolute_paths)) for row in rows]

    summary = {
        "sample_count": len(previews),
        "timing_count": sum(1 for r in previews if r["mode"] == "timing"),
        "rich_count": sum(1 for r in previews if r["mode"] in ("rich", "synthesis")),
        "synthesis_count": sum(1 for r in previews if r["mode"] == "synthesis"),
        "placeholder_warning_count": sum(1 for r in previews if r["placeholder_parameter_payload"]),
    }

    body = {
        "schema": "b3_lhs_orchestrator_preview_v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "samples_jsonl": _format_path(samples_path, repo_root=repo_root, absolute_paths=bool(args.absolute_paths)),
        "repo_root": _repo_root_field(repo_root, absolute_paths=bool(args.absolute_paths)),
        "will_execute": False,
        "notes": [
            "Dry-run preview only: no stage commands executed.",
            "No runtime manifests or index updates were performed.",
            "No subprocess invocation, no env activation, no file moves/deletes.",
        ],
        "summary": summary,
        "samples": previews,
    }

    write_json_atomic(out_json, body)
    _write_markdown(out_md, body)

    print(f"[B3_lhs_preview] wrote {out_json}", flush=True)
    print(f"[B3_lhs_preview] wrote {out_md}", flush=True)
    print(
        f"[B3_lhs_preview] sample_count={summary['sample_count']} "
        f"timing={summary['timing_count']} rich={summary['rich_count']} "
        f"synthesis={summary['synthesis_count']} placeholders={summary['placeholder_warning_count']}",
        flush=True,
    )
    print("[B3_lhs_preview] no stage execution performed", flush=True)
    return 0


def main() -> int:
    return run_preview()


if __name__ == "__main__":
    raise SystemExit(main())
