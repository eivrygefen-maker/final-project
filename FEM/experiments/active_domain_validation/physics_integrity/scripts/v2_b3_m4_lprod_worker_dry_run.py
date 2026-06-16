#!/usr/bin/env python3
"""M4.4.1a — L_prod worker execution interfaces + dry-run planner (no real solve)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PHYSICS_ROOT = SCRIPT_DIR.parent
SCRIPTS_REL = Path("FEM/experiments/active_domain_validation/physics_integrity/scripts")

DEFAULT_PROD_PYTHON = "/home/vboxuser/final-project/.venv/bin/python"
DEFAULT_PROD_VENV = "/home/vboxuser/final-project/.venv"
DEFAULT_SOLVER_PYTHON = "/home/vboxuser/solver-mkl/venv/bin/python"
DEFAULT_SOLVER_VENV = "/home/vboxuser/solver-mkl/venv"

LPROD_MESH_SCRIPT = f"{SCRIPTS_REL.as_posix()}/run_v2_mesh_convergence.py"
STAGE_A_SCRIPT = f"{SCRIPTS_REL.as_posix()}/v2_b3_checkpoint_export.py"
STAGE_B_EXISTING = f"{SCRIPTS_REL.as_posix()}/v2_b3_checkpoint_solve.py"
STAGE_B_PLANNED = f"{SCRIPTS_REL.as_posix()}/v2_b3_checkpoint_solve_target_list.py"

LPROD_SEC_PER_TARGET = 95.0
DEDUPE_TOLERANCE_HZ = 0.5
MESH_LEVEL = "L_prod"

REQUIRED_TERMINAL = "SCOUT_PASS_TARGET_PLAN_READY"
PLAN_OUTPUTS = (
    "lprod_execution_plan.json",
    "lprod_execution_plan.md",
    "lprod_mesh_checkpoint_readiness.json",
    "worker_commands.json",
    "worker_commands.md",
    "aggregation_plan.json",
    "aggregation_plan.md",
)

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_m4_lprod_interfaces import (  # noqa: E402
    build_chunk_targets_payload,
    build_solver_result_placeholder,
    build_worker_command_line,
    build_worker_result_placeholder,
    evaluate_lprod_mesh_checkpoint_readiness,
)
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


def _rel(path: Path, *, repo_root: Path) -> str:
    return _repo_relative(path, repo_root=repo_root)


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected JSON object")
    return data


def _comma_targets(targets: Sequence[float]) -> str:
    return ",".join(f"{float(t):g}" for t in targets)


def _validate_inputs(
    *,
    run_root: Path,
    manifest: Dict[str, Any],
    target_plan: Dict[str, Any],
    chunk_plan: Dict[str, Any],
    targets_hz: Sequence[float],
) -> List[str]:
    errors: List[str] = []
    term = str(manifest.get("terminal_status") or "")
    if term != REQUIRED_TERMINAL:
        errors.append(f"terminal_status={term!r} expected {REQUIRED_TERMINAL!r}")

    st3 = (manifest.get("stages") or {}).get("stage3_zones_plan") or {}
    if str(st3.get("status")) != "PASS":
        errors.append(f"stage3_zones_plan.status={st3.get('status')!r} expected PASS")

    cov = target_plan.get("coverage_check") or {}
    if not cov.get("pass"):
        errors.append(f"lprod_target_plan.coverage_check.pass={cov.get('pass')!r}")

    chunks = chunk_plan.get("chunks") or []
    if not chunks:
        errors.append("worker_chunk_plan.preview has no chunks")

    assigned: List[float] = []
    for c in chunks:
        assigned.extend(float(t) for t in (c.get("targets_hz") or []))

    if len(assigned) != len(targets_hz):
        errors.append(
            f"chunk target assignment count {len(assigned)} != plan targets {len(targets_hz)}"
        )
    else:
        a_sorted = sorted(assigned)
        t_sorted = sorted(float(t) for t in targets_hz)
        for a, t in zip(a_sorted, t_sorted):
            if abs(a - t) > 1e-4:
                errors.append("chunk targets do not match lprod_target_plan.targets_hz")
                break

    for c in chunks:
        if not c.get("targets_hz"):
            errors.append(f"{c.get('chunk_id')}: empty targets_hz")

    return errors


def _fcfs_schedule(
    chunks: Sequence[Dict[str, Any]],
    *,
    workers: int,
    sec_per_target: float,
) -> Dict[str, Any]:
    """Greedy FCFS wall-time estimate (seconds)."""
    n_workers = max(1, int(workers))
    costs = [
        float((c.get("estimated_cost") or {}).get("estimated_seconds")
              or len(c.get("targets_hz") or []) * sec_per_target)
        for c in chunks
    ]
    worker_free = [0.0] * n_workers
    assignments: List[Dict[str, Any]] = []
    for i, chunk in enumerate(chunks):
        w = min(range(n_workers), key=lambda j: worker_free[j])
        start = worker_free[w]
        cost = costs[i]
        worker_free[w] = start + cost
        assignments.append(
            {
                "chunk_id": chunk.get("chunk_id"),
                "worker_slot": f"W{w}",
                "queue_position": i,
                "estimated_start_s": round(start, 1),
                "estimated_finish_s": round(start + cost, 1),
                "estimated_wall_s": round(cost, 1),
                "target_count": len(chunk.get("targets_hz") or []),
            }
        )
    makespan = max(worker_free) if worker_free else 0.0
    serial = sum(costs)
    return {
        "policy": "fcfs_v1",
        "workers": n_workers,
        "chunk_count": len(chunks),
        "serial_wall_s": round(serial, 1),
        "estimated_makespan_s": round(makespan, 1),
        "critical_path_note": "Makespan = max worker load under FCFS; serial = sum(chunk costs).",
        "assignments": assignments,
    }


def _env_profiles() -> Dict[str, Any]:
    return {
        "stage4_mesh_checkpoint": {
            "profile": "production_venv_strict",
            "python": DEFAULT_PROD_PYTHON,
            "virtual_env": DEFAULT_PROD_VENV,
            "used_for": ["lprod_mesh_build", "lprod_stage_a_export"],
            "env_vars": {
                "PETSC_DIR": "/usr/lib/petscdir/petsc3.15/x86_64-linux-gnu-real",
                "SLEPC_DIR": "/usr/lib/slepcdir/slepc3.15/x86_64-linux-gnu-real",
                "PYTHONPATH": (
                    "$PETSC_DIR/lib/python3/dist-packages:"
                    "$SLEPC_DIR/lib/python3/dist-packages:/usr/lib/python3/dist-packages"
                ),
            },
            "note": "Explicit subprocess env; do not inherit parent VIRTUAL_ENV.",
        },
        "stage5_workers": {
            "profile": "solver_mkl_strict",
            "python": DEFAULT_SOLVER_PYTHON,
            "virtual_env": DEFAULT_SOLVER_VENV,
            "used_for": ["lprod_worker_chunk_solve"],
            "unset_at_execution": ["PYTHONPATH", "PETSC_DIR", "SLEPC_DIR", "PYTHONHOME"],
            "note": "Isolated solver-mkl; dolfinx must not import.",
        },
    }


def build_lprod_execution_plan(
    *,
    repo_root: Path,
    run_root: Path,
    sample_id: str,
    run_id: str,
    manifest: Dict[str, Any],
    sample_input: Dict[str, Any],
    target_plan: Dict[str, Any],
    chunk_plan: Dict[str, Any],
    workers: int,
) -> Dict[str, Any]:
    rel_run = _rel(run_root, repo_root=repo_root)
    lprod_dir = run_root / "lprod"
    mesh_path = lprod_dir / "mesh" / MESH_LEVEL / f"{sample_id}.msh"
    checkpoint_dir = lprod_dir / "checkpoint"
    resolved_lprod = lprod_dir / "resolved_core_config.json"
    resolved_sample = run_root / "sample" / "resolved_core_config.json"

    mesh_readiness = evaluate_lprod_mesh_checkpoint_readiness(
        repo_root=repo_root,
        run_root=run_root,
        sample_id=sample_id,
        sample_input=sample_input,
        rel_path_fn=lambda p, **kw: _rel(p, repo_root=repo_root),
    )

    targets_hz = [float(t) for t in (target_plan.get("targets_hz") or [])]
    chunks = chunk_plan.get("chunks") or []

    cmd_mesh = mesh_readiness.get("commands", {}).get("mesh_build_planned", "")
    cmd_stage_a = mesh_readiness.get("commands", {}).get("stage_a_export_planned", "")

    schedules: Dict[str, Any] = {}
    for w in (1, 2, 3, workers):
        if w <= max(3, workers):
            schedules[str(w)] = _fcfs_schedule(chunks, workers=w, sec_per_target=LPROD_SEC_PER_TARGET)

    return {
        "schema": "m4_lprod_execution_plan_v1",
        "will_execute": False,
        "mode": "m4_4_1a_dry_run",
        "sample_id": sample_id,
        "run_id": run_id,
        "generated_utc": _utc_now(),
        "input_status": manifest.get("terminal_status"),
        "target_count": len(targets_hz),
        "chunk_count": len(chunks),
        "planned_workers": workers,
        "stage4_lprod_mesh_checkpoint": {
            "status": "PLANNED_READY",
            "lprod_mesh_status": mesh_readiness.get("lprod_mesh_status"),
            "lprod_checkpoint_status": mesh_readiness.get("lprod_checkpoint_status"),
            "mesh_path": _rel(mesh_path, repo_root=repo_root),
            "mesh_source_recommended": (mesh_readiness.get("paths") or {}).get("mesh_source_recommended"),
            "resolved_core_config_path": _rel(resolved_lprod, repo_root=repo_root),
            "resolved_core_config_source": _rel(resolved_sample, repo_root=repo_root),
            "checkpoint_dir": _rel(checkpoint_dir, repo_root=repo_root),
            "geometry_compatibility": mesh_readiness.get("geometry_compatibility"),
            "readiness_path": _rel(lprod_dir / "lprod_mesh_checkpoint_readiness.json", repo_root=repo_root),
            "commands": {"mesh_build": cmd_mesh, "stage_a_export": cmd_stage_a},
            "reuse_policy": {
                "skip_mesh_if_pass": True,
                "skip_checkpoint_if_pass": True,
                "require_force_to_overwrite_pass": True,
                "reuse_baseline_mesh_if_geometry_hash_matches": True,
            },
        },
        "stage5_workers": {
            "status": "PLANNED_READY",
            "worker_count": workers,
            "scheduling": schedules,
            "solver_interface": {
                "primary_m4_4_1a": {
                    "script": STAGE_B_PLANNED,
                    "cli": (
                        f"{DEFAULT_SOLVER_PYTHON} {STAGE_B_PLANNED} "
                        '--checkpoint-dir "<lprod/checkpoint>" '
                        '--targets-json "<worker_results/chunk_id>/chunk_targets.json" '
                        "--factor-solver mkl_pardiso "
                        '--output-dir "<worker_results/chunk_id>" [--dry-run]'
                    ),
                    "notes": "Per-target window_hz from adaptive lprod_target_plan (m4_worker_chunk_targets_v1).",
                },
                "legacy_interim": {
                    "script": STAGE_B_EXISTING,
                    "supports": "--targets-hz comma-separated",
                    "gap": "Single global half-width only; not used for M4.4.1a worker commands.",
                },
            },
        },
        "stage6_aggregation": {
            "status": "PLANNED_READY",
            "aggregation_plan_path": f"{rel_run}/lprod/aggregation_plan.json",
        },
        "environment_profiles": _env_profiles(),
        "safety": {
            "will_execute": False,
            "no_subprocess_execution": True,
            "no_lprod_mesh_build": True,
            "no_lprod_checkpoint_export": True,
            "no_worker_solves": True,
            "no_stage_c": True,
            "no_rich_modal_export": True,
            "do_not_modify_stage0_3_pass_artifacts": True,
        },
        "shape_name": sample_input.get("shape_name"),
        "target_plan_policy": target_plan.get("target_generation_policy"),
    }


def _write_chunk_worker_artifacts(
    *,
    repo_root: Path,
    run_root: Path,
    sample_id: str,
    run_id: str,
    chunk: Dict[str, Any],
    target_plan: Dict[str, Any],
    checkpoint_dir: Path,
    force: bool,
) -> Dict[str, Any]:
    chunk_id = str(chunk.get("chunk_id"))
    targets = [float(t) for t in (chunk.get("targets_hz") or [])]
    out_dir = run_root / "worker_results" / chunk_id
    out_dir.mkdir(parents=True, exist_ok=True)
    chunk_targets_path = out_dir / "chunk_targets.json"
    if chunk_targets_path.is_file() and not force:
        raise FileExistsError(f"chunk artifact exists (use --force): {chunk_targets_path}")

    targets_payload = build_chunk_targets_payload(
        sample_id=sample_id,
        run_id=run_id,
        chunk=chunk,
        target_plan=target_plan,
    )
    write_json_atomic(chunk_targets_path, targets_payload)

    cmd_line = (
        build_worker_command_line(
            repo_root=repo_root,
            checkpoint_dir=checkpoint_dir,
            chunk_targets_path=chunk_targets_path,
            output_dir=out_dir,
            solver_python=DEFAULT_SOLVER_PYTHON,
        )
        + " --dry-run"
    )
    sh_path = out_dir / "worker_command.sh"
    sh_path.write_text(
        "#!/bin/bash\n# M4.4.1a dry-run command preview (solver-mkl strict env at execution)\n"
        f"set -euo pipefail\n{cmd_line}\n",
        encoding="utf-8",
    )
    preview_path = out_dir / "worker_command.preview.txt"
    preview_path.write_text(cmd_line + "\n", encoding="utf-8")

    solver_placeholder = build_solver_result_placeholder(
        chunk_targets=targets_payload,
        checkpoint_dir=checkpoint_dir,
        factor_solver="mkl_pardiso",
    )
    worker_placeholder = build_worker_result_placeholder(
        chunk_id=chunk_id,
        worker_id=None,
        chunk_targets=targets_payload,
        output_dir=out_dir,
    )
    write_json_atomic(out_dir / "solver_result.json", solver_placeholder)
    write_json_atomic(out_dir / "worker_result.json", worker_placeholder)
    (out_dir / "log.txt").write_text(
        "M4.4.1a dry-run placeholder — worker not executed.\n", encoding="utf-8"
    )

    checkpoint_rel = _rel(checkpoint_dir, repo_root=repo_root)
    targets_csv = _comma_targets(targets)
    cmd_legacy = (
        f"{DEFAULT_SOLVER_PYTHON} {STAGE_B_EXISTING} "
        f'--checkpoint-dir "{checkpoint_rel}" '
        f"--factor-solver mkl_pardiso "
        f'--targets-hz "{targets_csv}" '
        f'--output-dir "{_rel(out_dir, repo_root=repo_root)}"'
    )
    return {
        "chunk_id": chunk_id,
        "freq_range_hz": chunk.get("freq_range_hz"),
        "zone_ids": chunk.get("zone_ids"),
        "target_count": len(targets),
        "targets_hz": targets,
        "output_dir": _rel(out_dir, repo_root=repo_root),
        "artifacts": {
            "chunk_targets_json": _rel(chunk_targets_path, repo_root=repo_root),
            "worker_command_sh": _rel(sh_path, repo_root=repo_root),
            "worker_command_preview": _rel(preview_path, repo_root=repo_root),
            "worker_result_json": _rel(out_dir / "worker_result.json", repo_root=repo_root),
            "solver_result_json": _rel(out_dir / "solver_result.json", repo_root=repo_root),
            "log_txt": _rel(out_dir / "log.txt", repo_root=repo_root),
        },
        "commands": {
            "m4_4_target_list_solve": cmd_line,
            "legacy_targets_hz_solve": cmd_legacy,
        },
        "status": "DRY_RUN_PLANNED",
        "assigned_worker_id": None,
        "estimated_cost_seconds": len(targets) * LPROD_SEC_PER_TARGET,
    }


def build_worker_commands(
    *,
    repo_root: Path,
    run_root: Path,
    sample_id: str,
    run_id: str,
    chunk_plan: Dict[str, Any],
    target_plan: Dict[str, Any],
    workers: int,
    force: bool,
) -> Dict[str, Any]:
    lprod_dir = run_root / "lprod"
    checkpoint_dir = lprod_dir / "checkpoint"
    chunks_out: List[Dict[str, Any]] = []

    for chunk in chunk_plan.get("chunks") or []:
        chunks_out.append(
            _write_chunk_worker_artifacts(
                repo_root=repo_root,
                run_root=run_root,
                sample_id=sample_id,
                run_id=run_id,
                chunk=chunk,
                target_plan=target_plan,
                checkpoint_dir=checkpoint_dir,
                force=force,
            )
        )

    return {
        "schema": "m4_worker_commands_v1",
        "will_execute": False,
        "mode": "m4_4_1a_dry_run",
        "sample_id": sample_id,
        "run_id": run_id,
        "generated_utc": _utc_now(),
        "workers": workers,
        "solver_script": STAGE_B_PLANNED,
        "fcfs_policy": "assign first N chunks; on finish assign next queued",
        "chunks": chunks_out,
        "static_command_count": len(chunks_out),
    }


def build_aggregation_plan(
    *,
    repo_root: Path,
    run_root: Path,
    sample_id: str,
    run_id: str,
    chunk_plan: Dict[str, Any],
    target_plan: Dict[str, Any],
) -> Dict[str, Any]:
    agg_dir = run_root / "aggregation"
    chunks = chunk_plan.get("chunks") or []
    worker_globs = [
        _rel(run_root / "worker_results" / str(c.get("chunk_id")) / "worker_result.json", repo_root=repo_root)
        for c in chunks
    ]

    return {
        "schema": "m4_aggregation_plan_v1",
        "will_execute": False,
        "sample_id": sample_id,
        "run_id": run_id,
        "generated_utc": _utc_now(),
        "inputs": {
            "worker_result_globs": worker_globs,
            "solver_result_globs": [
                p.replace("worker_result.json", "solver_result.json") for p in worker_globs
            ],
            "lprod_target_plan": _rel(run_root / "lprod" / "lprod_target_plan.json", repo_root=repo_root),
            "chunk_plan": _rel(run_root / "lprod" / "worker_chunk_plan.preview.json", repo_root=repo_root),
        },
        "outputs": {
            "aggregation_result_json": _rel(agg_dir / "aggregation_result.json", repo_root=repo_root),
            "modes_catalog_jsonl": _rel(agg_dir / "modes_catalog.jsonl", repo_root=repo_root),
            "modes_summary_json": _rel(agg_dir / "modes_summary.json", repo_root=repo_root),
            "modal_data_npz": _rel(agg_dir / "modal_data.npz", repo_root=repo_root),
            "mode_frequency_plot_png": _rel(agg_dir / "mode_frequency_plot.png", repo_root=repo_root),
            "runtime_summary_json": _rel(agg_dir / "runtime_summary.json", repo_root=repo_root),
            "warnings_and_failures_json": _rel(agg_dir / "warnings_and_failures.json", repo_root=repo_root),
        },
        "rules": {
            "collect": "all accepted modes from worker_result.json per chunk",
            "dedupe_tolerance_hz": DEDUPE_TOLERANCE_HZ,
            "sort": "by frequency_hz ascending",
            "provenance_fields": ["chunk_id", "worker_id", "target_hz", "source_result_path"],
            "validate": "every chunk terminal PASS/PARTIAL; all targets attempted; report missing chunks",
        },
        "expected_mode_count_hint": len(target_plan.get("targets_hz") or []),
        "chunk_count": len(chunks),
    }


def build_manifest_preview(
    manifest: Dict[str, Any],
    *,
    run_id: str,
    sample_id: str,
    workers: int,
    target_count: int,
    chunk_count: int,
) -> Dict[str, Any]:
    preview = json.loads(json.dumps(manifest))
    preview["updated_utc"] = _utc_now()
    preview["will_execute"] = False
    preview["mode"] = "m4_4_1a_dry_run"
    preview["terminal_status"] = "LPROD_WORKER_PLAN_READY"
    preview["lprod_worker_plan"] = {
        "planned_workers": workers,
        "target_count": target_count,
        "chunk_count": chunk_count,
    }
    stages = preview.setdefault("stages", {})
    for key in ("stage4_lprod_mesh", "stage4_lprod_export", "stage5_workers", "stage6_aggregate"):
        st = stages.setdefault(key, {})
        if st.get("status") in ("PASS", "FAIL"):
            continue
        st["status"] = "PLANNED_READY"
    return preview


def _render_execution_plan_md(plan: Dict[str, Any]) -> str:
    s4 = plan.get("stage4_lprod_mesh_checkpoint") or {}
    s5 = plan.get("stage5_workers") or {}
    sched = (s5.get("scheduling") or {}).get(str(plan.get("planned_workers"))) or {}
    lines = [
        f"# L_prod execution plan (dry-run) — {plan.get('sample_id')}",
        "",
        f"- run_id: `{plan.get('run_id')}`",
        f"- will_execute: **false**",
        f"- targets: **{plan.get('target_count')}**",
        f"- chunks: **{plan.get('chunk_count')}**",
        f"- workers: **{plan.get('planned_workers')}**",
        f"- input status: `{plan.get('input_status')}`",
        "",
        "## Stage 4 — L_prod mesh + checkpoint",
        "",
        f"- mesh: `{s4.get('mesh_path')}`",
        f"- checkpoint: `{s4.get('checkpoint_dir')}`",
        f"- lprod_mesh_status: `{s4.get('lprod_mesh_status')}`",
        f"- lprod_checkpoint_status: `{s4.get('lprod_checkpoint_status')}`",
        "",
        "```bash",
        str((s4.get("commands") or {}).get("mesh_build", "")),
        str((s4.get("commands") or {}).get("stage_a_export", "")),
        "```",
        "",
        "## Stage 5 — workers (FCFS)",
        "",
        f"- makespan estimate ({plan.get('planned_workers')} workers): **{sched.get('estimated_makespan_s')} s**",
        f"- serial estimate: **{sched.get('serial_wall_s')} s**",
        "",
        "### Solver interface (M4.4.1a)",
        "",
        f"- Primary: `{STAGE_B_PLANNED}` with `--targets-json` per chunk",
        f"- Legacy (not used): `{STAGE_B_EXISTING}` `--targets-hz`",
        "",
        "## Safety",
        "",
        "- No L_prod execution in M4.4.1a.",
    ]
    return "\n".join(lines) + "\n"


def _render_worker_commands_md(doc: Dict[str, Any]) -> str:
    lines = [
        f"# Worker commands (dry-run) — {doc.get('sample_id')}",
        "",
        f"- chunks: **{doc.get('static_command_count')}**",
        f"- FCFS: {doc.get('fcfs_policy')}",
        "",
        "| chunk | targets | est. s |",
        "|-------|---------|--------|",
    ]
    for c in doc.get("chunks") or []:
        lines.append(
            f"| {c.get('chunk_id')} | {c.get('target_count')} | {c.get('estimated_cost_seconds')} |"
        )
    lines.append("")
    lines.append("## Example planned command (M4.4)")
    lines.append("")
    if doc.get("chunks"):
        lines.append("```bash")
        lines.append(str((doc["chunks"][0].get("commands") or {}).get("m4_4_target_list_solve", "")))
        lines.append("```")
    return "\n".join(lines) + "\n"


def _render_aggregation_plan_md(plan: Dict[str, Any]) -> str:
    lines = [
        f"# Aggregation plan (dry-run) — {plan.get('sample_id')}",
        "",
        f"- chunks: **{plan.get('chunk_count')}**",
        f"- dedupe tolerance: **{plan.get('rules', {}).get('dedupe_tolerance_hz')} Hz**",
        "",
        "## Outputs",
        "",
    ]
    for k, v in (plan.get("outputs") or {}).items():
        lines.append(f"- `{k}`: `{v}`")
    lines.append("")
    lines.append("## Rules")
    for k, v in (plan.get("rules") or {}).items():
        lines.append(f"- **{k}:** {v}")
    return "\n".join(lines) + "\n"


def run_dry_run(
    *,
    repo_root: Path,
    run_root: Path,
    workers: int,
    force: bool,
) -> int:
    lprod_dir = run_root / "lprod"
    if not force:
        from v2_b3_m4_reuse_integrity_lib import (  # noqa: WPS433
            remove_stale_worker_plan_outputs,
            worker_plan_artifact_contract_pass,
        )

        if worker_plan_artifact_contract_pass(run_root):
            print("skip: worker_plan PASS contract already satisfied (reuse)", flush=True)
            return 0
        remove_stale_worker_plan_outputs(run_root)

    manifest_path = run_root / "pipeline_run_manifest.json"
    target_plan_path = lprod_dir / "lprod_target_plan.json"
    chunk_plan_path = lprod_dir / "worker_chunk_plan.preview.json"

    for p, label in (
        (manifest_path, "pipeline_run_manifest.json"),
        (run_root / "sample" / "sample_input.json", "sample_input.json"),
        (target_plan_path, "lprod_target_plan.json"),
        (chunk_plan_path, "worker_chunk_plan.preview.json"),
    ):
        if not p.is_file():
            print(f"error: missing {label}: {p}", file=sys.stderr)
            return 2

    manifest = _load_json(manifest_path)
    sample_input = _load_json(run_root / "sample" / "sample_input.json")
    target_plan = _load_json(target_plan_path)
    chunk_plan = _load_json(chunk_plan_path)
    sample_id = str(manifest.get("sample_id") or sample_input.get("sample_id") or "")
    run_id = str(manifest.get("run_id") or run_root.name)
    targets_hz = [float(t) for t in (target_plan.get("targets_hz") or [])]

    val_errors = _validate_inputs(
        run_root=run_root,
        manifest=manifest,
        target_plan=target_plan,
        chunk_plan=chunk_plan,
        targets_hz=targets_hz,
    )
    if val_errors:
        print("error: input validation failed:", file=sys.stderr)
        for e in val_errors:
            print(f"  - {e}", file=sys.stderr)
        return 2

    exec_plan = build_lprod_execution_plan(
        repo_root=repo_root,
        run_root=run_root,
        sample_id=sample_id,
        run_id=run_id,
        manifest=manifest,
        sample_input=sample_input,
        target_plan=target_plan,
        chunk_plan=chunk_plan,
        workers=workers,
    )
    mesh_readiness = evaluate_lprod_mesh_checkpoint_readiness(
        repo_root=repo_root,
        run_root=run_root,
        sample_id=sample_id,
        sample_input=sample_input,
        rel_path_fn=lambda p, **kw: _rel(p, repo_root=repo_root),
    )
    write_json_atomic(lprod_dir / "lprod_mesh_checkpoint_readiness.json", mesh_readiness)

    worker_cmds = build_worker_commands(
        repo_root=repo_root,
        run_root=run_root,
        sample_id=sample_id,
        run_id=run_id,
        chunk_plan=chunk_plan,
        target_plan=target_plan,
        workers=workers,
        force=force,
    )
    agg_plan = build_aggregation_plan(
        repo_root=repo_root,
        run_root=run_root,
        sample_id=sample_id,
        run_id=run_id,
        chunk_plan=chunk_plan,
        target_plan=target_plan,
    )

    write_json_atomic(lprod_dir / "lprod_execution_plan.json", exec_plan)
    (lprod_dir / "lprod_execution_plan.md").write_text(
        _render_execution_plan_md(exec_plan), encoding="utf-8"
    )
    write_json_atomic(lprod_dir / "worker_commands.json", worker_cmds)
    (lprod_dir / "worker_commands.md").write_text(
        _render_worker_commands_md(worker_cmds), encoding="utf-8"
    )
    write_json_atomic(lprod_dir / "aggregation_plan.json", agg_plan)
    (lprod_dir / "aggregation_plan.md").write_text(
        _render_aggregation_plan_md(agg_plan), encoding="utf-8"
    )

    manifest_preview = build_manifest_preview(
        manifest,
        run_id=run_id,
        sample_id=sample_id,
        workers=workers,
        target_count=len(targets_hz),
        chunk_count=len(chunk_plan.get("chunks") or []),
    )
    write_json_atomic(run_root / "pipeline_run_manifest.m4_4_dry_run_preview.json", manifest_preview)

    print("will_execute=false")
    print(f"input status={manifest.get('terminal_status')}")
    print(f"target_count={len(targets_hz)}")
    print(f"chunk_count={len(chunk_plan.get('chunks') or [])}")
    print(f"planned workers={workers}")
    print("wrote lprod_execution_plan.json/md")
    print("wrote lprod_mesh_checkpoint_readiness.json")
    print(f"chunk_targets written for {len(chunk_plan.get('chunks') or [])} chunks")
    print("wrote worker_commands.json/md + per-chunk worker_command.sh previews")
    print("wrote aggregation_plan.json/md")
    print("wrote pipeline_run_manifest.m4_4_dry_run_preview.json")
    s4 = exec_plan.get("stage4_lprod_mesh_checkpoint") or {}
    print(f"lprod_mesh_status={s4.get('lprod_mesh_status')}")
    print(f"lprod_checkpoint_status={s4.get('lprod_checkpoint_status')}")
    print("no L_prod executed")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="M4.4.1a L_prod worker execution dry-run planner.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true", default=True, help="Planning only (default).")
    parser.add_argument(
        "--no-dry-run",
        action="store_false",
        dest="dry_run",
        help="Refused: this script is dry-run only.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing M4.4-pre plan outputs.")
    args = parser.parse_args(argv)

    if not args.dry_run:
        print("error: M4.4-pre script is dry-run only (--no-dry-run refused)", file=sys.stderr)
        return 2
    if args.workers < 1:
        print("error: --workers must be >= 1", file=sys.stderr)
        return 2

    repo_root = _detect_repo_root(SCRIPT_DIR)
    run_root = args.run_dir if args.run_dir.is_absolute() else repo_root / args.run_dir
    run_root = run_root.resolve()

    return run_dry_run(
        repo_root=repo_root,
        run_root=run_root,
        workers=int(args.workers),
        force=bool(args.force),
    )


if __name__ == "__main__":
    raise SystemExit(main())
