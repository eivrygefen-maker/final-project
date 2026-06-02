#!/usr/bin/env python3
"""M3.2 dry-run orchestrator preview (no Stage A/B/C execution, no runtime manifests)."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PHYSICS_ROOT = SCRIPT_DIR.parent
SCRIPTS_REL = Path("FEM/experiments/active_domain_validation/physics_integrity/scripts")

DEFAULT_PROD_PYTHON = "/home/vboxuser/final-project/.venv/bin/python"
DEFAULT_SOLVER_PYTHON = "/home/vboxuser/solver-mkl/venv/bin/python"

PROD_ENV_VARS = {
    "PETSC_DIR": "/usr/lib/petscdir/petsc3.15/x86_64-linux-gnu-real",
    "SLEPC_DIR": "/usr/lib/slepcdir/slepc3.15/x86_64-linux-gnu-real",
    "PYTHONPATH": (
        "$PETSC_DIR/lib/python3/dist-packages:"
        "$SLEPC_DIR/lib/python3/dist-packages:"
        "/usr/lib/python3/dist-packages:$PYTHONPATH"
    ),
}

STAGE_A_SCRIPT = "v2_b3_checkpoint_export.py"
STAGE_B_SCRIPT = "v2_b3_checkpoint_solve.py"
STAGE_C_SCRIPT = "v2_b3_rich_modal_post.py"

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


def _format_path(path: Path, *, repo_root: Path, absolute_paths: bool) -> str:
    if absolute_paths:
        return str(path.resolve())
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve_mode(row: Dict[str, Any]) -> str:
    mode = str(row.get("mode") or "").strip().lower()
    if mode in ("timing", "rich", "synthesis"):
        return mode
    if bool(row.get("synthesis_requested")):
        return "synthesis"
    if bool(row.get("rich_requested")):
        return "rich"
    return "timing"


def _stage_plan(mode: str) -> Dict[str, str]:
    if mode == "synthesis":
        return {"A": "PENDING", "B": "PENDING", "C": "PENDING"}
    if mode == "rich":
        return {"A": "PENDING", "B": "PENDING", "C": "SKIPPED"}
    return {"A": "PENDING", "B": "PENDING", "C": "SKIPPED"}


def _build_paths(
    repo_root: Path,
    *,
    mesh_level: str,
    target_set: str,
    run_id: str,
    absolute_paths: bool,
) -> Dict[str, str]:
    base = repo_root / "FEM" / "experiments" / "active_domain_validation" / "physics_integrity"
    checkpoint = base / "v2_mesh_convergence" / "diagnostics" / f"st_worker_scaling_{mesh_level}_{run_id}"
    solve = (
        base
        / "v2_mesh_convergence"
        / "diagnostics"
        / "solver_benchmarks"
        / f"checkpoint_solve_mkl_pardiso_{target_set}_{run_id}"
    )
    rich_modal = solve / "rich_modal"
    synthesis = solve / "rich_modal_post"
    return {
        "checkpoint_dir": _format_path(checkpoint, repo_root=repo_root, absolute_paths=absolute_paths),
        "solve_dir": _format_path(solve, repo_root=repo_root, absolute_paths=absolute_paths),
        "rich_modal_dir": _format_path(rich_modal, repo_root=repo_root, absolute_paths=absolute_paths),
        "synthesis_dir": _format_path(synthesis, repo_root=repo_root, absolute_paths=absolute_paths),
    }


def _resolved_core_config(repo_root: Path, sample_id: str, *, absolute_paths: bool) -> Tuple[Optional[str], Path]:
    rel = (
        Path("FEM")
        / "experiments"
        / "active_domain_validation"
        / "physics_integrity"
        / "pipeline_runs"
        / "config_overlays"
        / sample_id
        / "resolved_core_config.json"
    )
    path = repo_root / rel
    if not path.is_file():
        return None, path
    return _format_path(path, repo_root=repo_root, absolute_paths=absolute_paths), path


def _intended_manifest_path(
    repo_root: Path, run_id: str, *, absolute_paths: bool
) -> str:
    p = (
        repo_root
        / "FEM"
        / "experiments"
        / "active_domain_validation"
        / "physics_integrity"
        / "pipeline_runs"
        / "manifests"
        / f"run_{run_id}.json"
    )
    return _format_path(p, repo_root=repo_root, absolute_paths=absolute_paths)


def _cmd_stage_a(
    *,
    python: str,
    mesh_level: str,
    core_config: str,
    checkpoint_dir: str,
) -> str:
    return (
        f"{python} {SCRIPTS_REL.as_posix()}/{STAGE_A_SCRIPT} "
        f"--mesh-level {mesh_level} "
        "--B3-block-compose-backend csr_bulk "
        "--B3-synthesis-region-dofs off "
        f"--core-config \"{core_config}\" "
        f"--output-dir \"{checkpoint_dir}\""
    )


def _cmd_stage_b(
    *,
    python: str,
    checkpoint_dir: str,
    solve_dir: str,
    target_set: str,
    rich: bool,
) -> str:
    cmd = (
        f"{python} {SCRIPTS_REL.as_posix()}/{STAGE_B_SCRIPT} "
        f"--checkpoint-dir \"{checkpoint_dir}\" "
        "--factor-solver mkl_pardiso "
        f"--target-set {target_set} "
        f"--output-dir \"{solve_dir}\""
    )
    if rich:
        cmd += " --B3-export-rich-modal-data"
    return cmd


def _cmd_stage_c(
    *,
    python: str,
    checkpoint_dir: str,
    rich_modal_dir: str,
    synthesis_dir: str,
) -> str:
    return (
        f"{python} {SCRIPTS_REL.as_posix()}/{STAGE_C_SCRIPT} "
        f"--checkpoint-dir \"{checkpoint_dir}\" "
        f"--rich-modal-dir \"{rich_modal_dir}\" "
        f"--output-dir \"{synthesis_dir}\""
    )


def _stage_env_preview(*, prod_python: str, solver_python: str) -> Dict[str, Any]:
    return {
        "stage_a": {
            "profile": "production_venv",
            "python": prod_python,
            "env_vars": dict(PROD_ENV_VARS),
        },
        "stage_b": {
            "profile": "solver_mkl",
            "python": solver_python,
            "env_vars": {},
        },
        "stage_c": {
            "profile": "production_venv",
            "python": prod_python,
            "env_vars": dict(PROD_ENV_VARS),
        },
    }


def _preflight(
    *,
    repo_root: Path,
    row: Dict[str, Any],
    mode: str,
    paths_abs: Dict[str, Path],
    resolved_path: Optional[Path],
    prod_python: Path,
    solver_python: Path,
) -> Dict[str, Any]:
    checks: Dict[str, Any] = {}
    warnings: List[str] = []
    blockers: List[str] = []

    sample_id = str(row.get("sample_id") or "")
    run_id = str(row.get("run_id") or "")

    for name, script in (
        ("stage_a_script", STAGE_A_SCRIPT),
        ("stage_b_script", STAGE_B_SCRIPT),
        ("stage_c_script", STAGE_C_SCRIPT),
    ):
        p = repo_root / SCRIPTS_REL / script
        checks[f"{name}_exists"] = p.is_file()
        if not p.is_file():
            blockers.append(f"missing_script:{script}")

    checks["prod_python_exists_on_host"] = prod_python.is_file()
    checks["solver_python_exists_on_host"] = solver_python.is_file()
    if not prod_python.is_file():
        warnings.append("prod_python_not_found_on_host:vm_path_expected")
    if not solver_python.is_file():
        warnings.append("solver_python_not_found_on_host:vm_path_expected")

    if resolved_path is None or not resolved_path.is_file():
        checks["resolved_core_config_exists"] = False
        blockers.append(f"missing_resolved_core_config:config_overlays/{sample_id}/resolved_core_config.json")
    else:
        checks["resolved_core_config_exists"] = True
        try:
            cfg = json.loads(resolved_path.read_text(encoding="utf-8"))
            clamp = (cfg.get("solver") or {}).get("clamp_ribs")
            checks["solver_clamp_ribs"] = clamp
            if clamp is not False:
                blockers.append("resolved_config_solver_clamp_ribs_not_false")
        except Exception as exc:
            blockers.append(f"resolved_core_config_invalid_json:{type(exc).__name__}")

    for key in ("checkpoint_dir", "solve_dir"):
        exists = paths_abs[key].exists()
        checks[f"{key}_exists"] = exists
        if exists:
            blockers.append(f"output_dir_exists:{key}:{paths_abs[key]}")

    if mode == "synthesis":
        syn_exists = paths_abs["synthesis_dir"].exists()
        checks["synthesis_dir_exists"] = syn_exists
        if syn_exists:
            blockers.append(f"output_dir_exists:synthesis_dir:{paths_abs['synthesis_dir']}")

    rich_req = bool(row.get("rich_requested"))
    syn_req = bool(row.get("synthesis_requested"))
    c_req = bool(row.get("stage_c_requested"))

    checks["policy_rich_requested"] = rich_req
    checks["policy_synthesis_requested"] = syn_req
    checks["policy_stage_c_requested"] = c_req

    if syn_req and not rich_req:
        warnings.append("synthesis_implies_rich_export")
    if mode == "synthesis" and not (rich_req and syn_req and c_req):
        blockers.append("mode_synthesis_policy_inconsistent")
    if mode == "timing" and (syn_req or c_req):
        blockers.append("mode_timing_stage_c_must_be_skipped")
    if mode == "timing" and rich_req:
        warnings.append("timing_mode_with_rich_requested_unusual")

    checks["stage_c_skipped_for_timing"] = mode == "timing"

    return {
        "checks": checks,
        "warnings": warnings,
        "blockers": blockers,
        "ready": len(blockers) == 0,
    }


def _preview_row(
    repo_root: Path,
    row: Dict[str, Any],
    *,
    absolute_paths: bool,
    prod_python: str,
    solver_python: str,
) -> Dict[str, Any]:
    sample_id = str(row.get("sample_id") or "").strip()
    run_id = str(row.get("run_id") or sample_id).strip()
    if not sample_id:
        raise ValueError("sample row missing sample_id")
    if not run_id:
        raise ValueError("sample row missing run_id")

    mesh_level = str(row.get("mesh_level") or "L_prod")
    target_set = str(row.get("target_set") or "full9")
    selection_reason = str(row.get("selection_reason") or "unspecified")
    mode = _resolve_mode(row)

    paths_str = _build_paths(
        repo_root,
        mesh_level=mesh_level,
        target_set=target_set,
        run_id=run_id,
        absolute_paths=absolute_paths,
    )
    paths_abs = {
        "checkpoint_dir": (repo_root / paths_str["checkpoint_dir"]).resolve()
        if not Path(paths_str["checkpoint_dir"]).is_absolute()
        else Path(paths_str["checkpoint_dir"]).resolve(),
        "solve_dir": (repo_root / paths_str["solve_dir"]).resolve()
        if not Path(paths_str["solve_dir"]).is_absolute()
        else Path(paths_str["solve_dir"]).resolve(),
        "rich_modal_dir": (repo_root / paths_str["rich_modal_dir"]).resolve()
        if not Path(paths_str["rich_modal_dir"]).is_absolute()
        else Path(paths_str["rich_modal_dir"]).resolve(),
        "synthesis_dir": (repo_root / paths_str["synthesis_dir"]).resolve()
        if not Path(paths_str["synthesis_dir"]).is_absolute()
        else Path(paths_str["synthesis_dir"]).resolve(),
    }

    core_config_str, core_config_path = _resolved_core_config(
        repo_root, sample_id, absolute_paths=absolute_paths
    )

    rich = mode in ("rich", "synthesis")
    c_requested = mode == "synthesis"

    commands: Dict[str, Optional[str]] = {
        "stage_a": None,
        "stage_b": None,
        "stage_c": None,
    }
    if core_config_str:
        commands["stage_a"] = _cmd_stage_a(
            python=prod_python,
            mesh_level=mesh_level,
            core_config=core_config_str,
            checkpoint_dir=paths_str["checkpoint_dir"],
        )
        commands["stage_b"] = _cmd_stage_b(
            python=solver_python,
            checkpoint_dir=paths_str["checkpoint_dir"],
            solve_dir=paths_str["solve_dir"],
            target_set=target_set,
            rich=rich,
        )
        if c_requested:
            commands["stage_c"] = _cmd_stage_c(
                python=prod_python,
                checkpoint_dir=paths_str["checkpoint_dir"],
                rich_modal_dir=paths_str["rich_modal_dir"],
                synthesis_dir=paths_str["synthesis_dir"],
            )

    preflight = _preflight(
        repo_root=repo_root,
        row=row,
        mode=mode,
        paths_abs=paths_abs,
        resolved_path=core_config_path if core_config_str else None,
        prod_python=Path(prod_python),
        solver_python=Path(solver_python),
    )

    warnings = list(preflight.get("warnings") or [])
    blockers = list(preflight.get("blockers") or [])

    return {
        "sample_id": sample_id,
        "run_id": run_id,
        "mode": mode,
        "selection_reason": selection_reason,
        "mesh_level": mesh_level,
        "target_set": target_set,
        "resolved_core_config": core_config_str,
        "intended_runtime_manifest": _intended_manifest_path(
            repo_root, run_id, absolute_paths=absolute_paths
        ),
        "stage_plan": _stage_plan(mode),
        "commands": commands,
        "stage_env": _stage_env_preview(prod_python=prod_python, solver_python=solver_python),
        "predicted_output_paths": {
            **paths_str,
            "resolved_core_config": core_config_str,
            "note": "dry_run_preview_only_not_created",
        },
        "preflight": preflight,
        "warnings": warnings,
        "blockers": blockers,
        "will_execute": False,
    }


def _write_markdown(path: Path, body: Dict[str, Any]) -> None:
    lines = [
        "# M3.2 orchestrator dry-run preview",
        "",
        f"- generated_utc: `{body.get('generated_utc')}`",
        f"- samples_jsonl: `{body.get('samples_jsonl')}`",
        f"- will_execute: `{body.get('will_execute')}`",
        "",
        "## Summary",
        "",
        f"- sample_count: `{body['summary']['sample_count']}`",
        f"- timing_count: `{body['summary']['timing_count']}`",
        f"- synthesis_count: `{body['summary']['synthesis_count']}`",
        f"- ready_count: `{body['summary']['ready_count']}`",
        f"- blocker_count: `{body['summary']['blocker_count']}`",
        "",
        "## Per sample",
        "",
        "| sample_id | run_id | mode | A | B | C | blockers | ready |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in body.get("samples", []):
        sp = row.get("stage_plan") or {}
        blockers = row.get("blockers") or []
        ready = (row.get("preflight") or {}).get("ready", False)
        lines.append(
            f"| {row.get('sample_id')} | {row.get('run_id')} | {row.get('mode')} | "
            f"{sp.get('A')} | {sp.get('B')} | {sp.get('C')} | {len(blockers)} | {ready} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Dry-run only: no Stage A/B/C execution, no runtime manifests, no index append.",
            "- Python paths target VM production/solver-mkl environments.",
            "- `run_id` keys output directories; `sample_id` keys config overlays.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_dry_run(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="M3.2 orchestrator dry-run preview (no execution).")
    parser.add_argument("--samples-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", help="Optional markdown output (default: sibling .md)")
    parser.add_argument("--repo-root", help="Optional repo root (default: auto-detect)")
    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        help="Emit absolute paths (default: repo-relative).",
    )
    parser.add_argument("--prod-python", default=DEFAULT_PROD_PYTHON)
    parser.add_argument("--solver-python", default=DEFAULT_SOLVER_PYTHON)
    parser.add_argument("--force", action="store_true", help="Overwrite preview outputs")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).expanduser().resolve() if args.repo_root else _detect_repo_root(SCRIPT_DIR)
    samples_path = Path(args.samples_jsonl).expanduser()
    if not samples_path.is_absolute():
        samples_path = (repo_root / samples_path).resolve()
    out_json = Path(args.output_json).expanduser()
    if not out_json.is_absolute():
        out_json = (repo_root / out_json).resolve()
    out_md = Path(args.output_md).expanduser().resolve() if args.output_md else out_json.with_suffix(".md")

    if out_json.exists() and not args.force:
        raise SystemExit(f"output exists: {out_json} (use --force)")
    if out_md.exists() and not args.force:
        raise SystemExit(f"output exists: {out_md} (use --force)")

    rows = _read_jsonl(samples_path)
    samples = [
        _preview_row(
            repo_root,
            row,
            absolute_paths=bool(args.absolute_paths),
            prod_python=str(args.prod_python),
            solver_python=str(args.solver_python),
        )
        for row in rows
    ]

    summary = {
        "sample_count": len(samples),
        "timing_count": sum(1 for s in samples if s["mode"] == "timing"),
        "synthesis_count": sum(1 for s in samples if s["mode"] == "synthesis"),
        "ready_count": sum(1 for s in samples if (s.get("preflight") or {}).get("ready")),
        "blocker_count": sum(1 for s in samples if s.get("blockers")),
    }

    body = {
        "schema": "b3_m3_orchestrator_dry_run_v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "samples_jsonl": _format_path(samples_path, repo_root=repo_root, absolute_paths=bool(args.absolute_paths)),
        "repo_root": "." if not args.absolute_paths else str(repo_root),
        "will_execute": False,
        "notes": [
            "M3.2 dry-run only: no Stage A/B/C execution.",
            "No runtime manifests under pipeline_runs/manifests/.",
            "No runs_index.jsonl append.",
            "No subprocess stage calls.",
        ],
        "summary": summary,
        "samples": samples,
    }

    write_json_atomic(out_json, body)
    _write_markdown(out_md, body)

    print(f"[B3_m3_dry_run] wrote {out_json}", flush=True)
    print(f"[B3_m3_dry_run] wrote {out_md}", flush=True)
    print(
        f"[B3_m3_dry_run] sample_count={summary['sample_count']} "
        f"timing={summary['timing_count']} synthesis={summary['synthesis_count']} "
        f"ready={summary['ready_count']} blockers={summary['blocker_count']}",
        flush=True,
    )
    print("[B3_m3_dry_run] no stage execution performed", flush=True)
    return 0 if summary["blocker_count"] == 0 else 2


def main() -> int:
    return run_dry_run()


if __name__ == "__main__":
    raise SystemExit(main())
