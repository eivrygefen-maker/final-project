#!/usr/bin/env python3
"""M4.4.1b-4 — run remaining L_prod worker chunks (skip/reuse PASS; solver-mkl)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_m4_runtime_provenance import (  # noqa: E402
    collect_m4_runtime_provenance,
    production_worker_thread_settings,
)
from v2_b3_m4_worker_run_lib import (  # noqa: E402
    PASS_LIKE,
    TERMINAL_CHECKPOINT_READY,
    build_chunk_plan,
    detect_repo_root,
    execute_worker_chunk,
    load_json,
    plan_remaining_worker_chunks,
    production_worker_subprocess_env,
    run_chunks_fcfs_parallel,
    run_solver_env_probe,
    utc_now,
    validate_chunk_preconditions,
    validate_global_preconditions,
)
from v2_b3_petsc_util import write_json_atomic  # noqa: E402
from v2_b3_run_coarse_scout_lhs_batch import (  # noqa: E402
    DEFAULT_SOLVER_PYTHON,
)

DEFAULT_SOLVER_VENV = "/home/vboxuser/solver-mkl/venv"

REMAINING_ID = "m4_4_1b_4"
REMAINING_MANIFEST = "remaining_workers_m4_4_1b_4_manifest.json"
REMAINING_SUMMARY_MD = "remaining_workers_m4_4_1b_4_summary.md"
REMAINING_PLAN_JSON = "remaining_workers_m4_4_1b_4_plan.json"
TERMINAL_PASS = "WORKER_REMAINING_PASS"
TERMINAL_WARN = "WORKER_REMAINING_PASS_WITH_WARNING"
TERMINAL_FAIL = "WORKER_REMAINING_FAIL"


def _aggregate_remaining_status(chunk_results: Sequence[Dict[str, Any]]) -> str:
    if any(r.get("status") == "FAIL" for r in chunk_results):
        return TERMINAL_FAIL
    if any(r.get("status") == "PASS_WITH_WARNING" for r in chunk_results):
        return TERMINAL_WARN
    if chunk_results and all(r.get("status") in PASS_LIKE for r in chunk_results):
        return TERMINAL_PASS
    return TERMINAL_FAIL


def _render_summary_md(report: Dict[str, Any]) -> str:
    lines = [
        f"# Remaining L_prod workers ({REMAINING_ID}) — {report.get('sample_id')}",
        "",
        f"- run_id: `{report.get('run_id')}`",
        f"- status: **{report.get('status')}**",
        f"- planned_chunk_count: **{report.get('planned_chunk_count')}**",
        f"- preexisting_pass_chunks: `{report.get('preexisting_pass_chunks')}`",
        f"- executed_chunks: `{report.get('executed_chunks')}`",
        f"- skipped (reuse): `{report.get('skipped_chunks')}`",
        f"- pass: `{report.get('pass_chunks')}`",
        f"- warning: `{report.get('warning_chunks')}`",
        f"- failed: `{report.get('failed_chunks')}`",
        f"- wall_time_s: **{report.get('wall_time_s')}**",
        "",
        "## Totals",
        "",
        f"- targets_attempted: **{report.get('total_targets_attempted')}**",
        f"- targets_passed: **{report.get('total_targets_passed')}**",
        f"- unique_modes (sum per chunk): **{report.get('total_unique_modes')}**",
        "",
        "## Per chunk",
        "",
        "| chunk | action | status | targets | modes | wall_s |",
        "|-------|--------|--------|---------|-------|--------|",
    ]
    for row in report.get("chunk_results") or []:
        lines.append(
            f"| {row.get('chunk_id')} | {row.get('action')} | {row.get('status')} | "
            f"{row.get('targets_passed')}/{row.get('targets_attempted')} | "
            f"{row.get('unique_mode_count')} | {row.get('wall_seconds')} |"
        )
    lines.append("")
    return "\n".join(lines)


def build_remaining_plan(
    *,
    repo_root: Path,
    run_root: Path,
    manifest: Dict[str, Any],
    solver_python: str,
    force: bool,
) -> Dict[str, Any]:
    classification = plan_remaining_worker_chunks(run_root, force=force)
    chunk_plans: List[Dict[str, Any]] = []

    for chunk_id in classification["planned_chunk_ids"]:
        errs = validate_chunk_preconditions(run_root=run_root, chunk_id=chunk_id, force=force)
        hard = [e for e in errs if "use --force" not in e]
        plan = build_chunk_plan(
            repo_root=repo_root,
            run_root=run_root,
            chunk_id=chunk_id,
            solver_python=solver_python,
            force=force,
        )
        if hard:
            plan["precheck_errors"] = hard
            plan["runnable"] = False
        else:
            plan["precheck_errors"] = []
            plan["runnable"] = True
        chunk_plans.append(plan)

    return {
        "schema": "m4_worker_remaining_plan_v1",
        "will_execute": False,
        "remaining_batch_id": REMAINING_ID,
        "sample_id": manifest.get("sample_id"),
        "run_id": manifest.get("run_id"),
        **classification,
        "chunk_plans": chunk_plans,
    }


def run_dry_run(*, repo_root: Path, run_root: Path, solver_python: str, force: bool) -> int:
    manifest_path = run_root / "pipeline_run_manifest.json"
    manifest = load_json(manifest_path)

    global_errors = validate_global_preconditions(run_root=run_root, manifest=manifest)
    if global_errors:
        print("error: global preconditions failed:", file=sys.stderr)
        for e in global_errors:
            print(f"  - {e}", file=sys.stderr)
        return 2

    plan = build_remaining_plan(
        repo_root=repo_root,
        run_root=run_root,
        manifest=manifest,
        solver_python=solver_python,
        force=force,
    )
    worker_root = run_root / "worker_results"
    write_json_atomic(worker_root / REMAINING_PLAN_JSON, plan)

    hard_any = False
    for cp in plan["chunk_plans"]:
        if cp.get("chunk_id") in plan["chunks_to_execute"] and cp.get("precheck_errors"):
            hard_any = True
            for e in cp["precheck_errors"]:
                print(f"  - {e}", file=sys.stderr)

    if hard_any:
        print("error: one or more chunks to execute failed precheck", file=sys.stderr)
        return 2

    print("will_execute=false")
    print(f"planned_chunk_count={plan.get('planned_chunk_count')}")
    print(f"preexisting_pass_chunks={plan.get('preexisting_pass_chunks')}")
    print(f"to_execute={plan.get('chunks_to_execute')}")
    print(f"to_skip_reuse={plan.get('chunks_to_skip_reuse')}")
    print("no other chunks will be executed")
    for cp in plan["chunk_plans"]:
        action = "skip_reuse" if cp.get("skip_solve") else "execute"
        print(f"  {cp['chunk_id']}: action={action} targets={cp.get('target_count')}")
        if action == "execute":
            print(f"    cmd={cp.get('command_preview')}")
    return 0


def run_execute(
    *,
    repo_root: Path,
    run_root: Path,
    solver_python: str,
    solver_venv: str,
    force: bool,
    stop_on_fail: bool,
    n_workers: int,
) -> int:
    manifest_path = run_root / "pipeline_run_manifest.json"
    manifest = load_json(manifest_path)
    worker_root = run_root / "worker_results"

    global_errors = validate_global_preconditions(run_root=run_root, manifest=manifest)
    if global_errors:
        print("error: global preconditions failed:", file=sys.stderr)
        for e in global_errors:
            print(f"  - {e}", file=sys.stderr)
        return 2

    plan = build_remaining_plan(
        repo_root=repo_root,
        run_root=run_root,
        manifest=manifest,
        solver_python=solver_python,
        force=force,
    )
    planned_ids = list(plan["planned_chunk_ids"])

    for chunk_id in plan["chunks_to_execute"]:
        errs = validate_chunk_preconditions(run_root=run_root, chunk_id=chunk_id, force=force)
        hard = [e for e in errs if "use --force" not in e]
        if hard:
            print("error: chunk preconditions failed:", file=sys.stderr)
            for e in hard:
                print(f"  - {e}", file=sys.stderr)
            return 2

    if not plan["chunks_to_execute"] and plan["preexisting_pass_chunks"]:
        print("note: all planned chunks already PASS; nothing to execute", flush=True)

    n_workers = max(1, int(n_workers))
    env_b = production_worker_subprocess_env(
        solver_python=solver_python, solver_venv=solver_venv
    )
    thread_settings = production_worker_thread_settings(env_b)
    t_wall0 = time.perf_counter()
    env_pass, env_body = run_solver_env_probe(
        repo_root=repo_root,
        solver_python=solver_python,
        solver_venv=solver_venv,
        env_b=env_b,
        chunk_dir=worker_root,
    )
    if not env_pass:
        print("env_probe FAIL (remaining workers)", flush=True)
        return 1
    print("env_probe PASS", flush=True)
    print(f"checkpoint PASS ({TERMINAL_CHECKPOINT_READY})", flush=True)
    print(f"preexisting_pass={plan.get('preexisting_pass_chunks')}", flush=True)
    print(f"to_execute={plan.get('chunks_to_execute')}", flush=True)
    print(
        f"workers_requested={n_workers} thread_settings={thread_settings}",
        flush=True,
    )

    chunk_results: List[Dict[str, Any]] = []
    result_by_chunk: Dict[str, Dict[str, Any]] = {}
    workers_actual = 1
    skip_reuse = set(plan.get("chunks_to_skip_reuse") or [])

    for i, chunk_id in enumerate(planned_ids):
        if chunk_id not in skip_reuse:
            continue
        print(f"[remaining] chunk {i + 1}/{len(planned_ids)}: {chunk_id} (reuse)", flush=True)
        result = execute_worker_chunk(
            repo_root=repo_root,
            run_root=run_root,
            chunk_id=chunk_id,
            solver_python=solver_python,
            solver_venv=solver_venv,
            force=force,
            env_b=env_b,
            env_probe_body=env_body,
            run_env_probe=False,
            worker_id=f"remaining_reuse_{i}",
            mode="m4_4_1b_4_worker_remaining",
            minibatch_id=REMAINING_ID,
            label_prefix="worker_remaining",
        )
        result_by_chunk[chunk_id] = result
        print(
            f"  {chunk_id}: action={result.get('action')} status={result.get('status')}",
            flush=True,
        )

    to_execute = [cid for cid in planned_ids if cid in plan.get("chunks_to_execute", [])]
    if to_execute:
        print(
            f"[remaining] FCFS parallel execute n_workers={n_workers} chunks={len(to_execute)}",
            flush=True,
        )
        parallel_results, workers_actual = run_chunks_fcfs_parallel(
            repo_root=repo_root,
            run_root=run_root,
            chunk_ids=to_execute,
            solver_python=solver_python,
            solver_venv=solver_venv,
            force=force,
            env_b=env_b,
            env_probe_body=env_body,
            n_workers=n_workers,
            minibatch_id=REMAINING_ID,
            label_prefix="worker_remaining",
            stop_on_fail=stop_on_fail,
        )
        for j, result in enumerate(parallel_results):
            chunk_id = result.get("chunk_id")
            print(
                f"[remaining] chunk done {j + 1}/{len(to_execute)}: {chunk_id} "
                f"action={result.get('action')} status={result.get('status')} "
                f"exit={result.get('solve_exit_code')} "
                f"targets={result.get('targets_passed')}/{result.get('targets_attempted')} "
                f"wall_s={result.get('wall_seconds')}",
                flush=True,
            )
            result_by_chunk[chunk_id] = result
            if (
                stop_on_fail
                and result.get("status") == "FAIL"
                and result.get("action") in ("executed", "failed_precheck")
            ):
                print(f"error: stopping after FAIL on {chunk_id}", file=sys.stderr)
                break

    chunk_results = [result_by_chunk[cid] for cid in planned_ids if cid in result_by_chunk]
    print(f"workers_actual_parallel={workers_actual}", flush=True)

    wall_s = time.perf_counter() - t_wall0
    executed = [r["chunk_id"] for r in chunk_results if r.get("action") == "executed"]
    skipped = [r["chunk_id"] for r in chunk_results if r.get("action") == "skipped_reuse"]
    pass_chunks = [r["chunk_id"] for r in chunk_results if r.get("status") == "PASS"]
    warn_chunks = [r["chunk_id"] for r in chunk_results if r.get("status") == "PASS_WITH_WARNING"]
    fail_chunks = [r["chunk_id"] for r in chunk_results if r.get("status") == "FAIL"]

    total_attempted = sum(int(r.get("targets_attempted") or 0) for r in chunk_results)
    total_passed = sum(int(r.get("targets_passed") or 0) for r in chunk_results)
    total_unique = sum(int(r.get("unique_mode_count") or 0) for r in chunk_results)

    status = _aggregate_remaining_status(chunk_results)

    report = {
        "schema": "m4_worker_remaining_manifest_v1",
        "will_execute": True,
        "remaining_batch_id": REMAINING_ID,
        "generated_utc": utc_now(),
        "sample_id": manifest.get("sample_id"),
        "run_id": manifest.get("run_id"),
        "status": status,
        "planned_chunk_count": plan.get("planned_chunk_count"),
        "preexisting_pass_chunks": plan.get("preexisting_pass_chunks"),
        "chunks_to_execute_planned": plan.get("chunks_to_execute"),
        "skipped_chunks": skipped,
        "executed_chunks": executed,
        "pass_chunks": pass_chunks,
        "warning_chunks": warn_chunks,
        "failed_chunks": fail_chunks,
        "total_targets_attempted": total_attempted,
        "total_targets_passed": total_passed,
        "total_unique_modes": total_unique,
        "wall_time_s": round(wall_s, 2),
        "chunk_results": chunk_results,
        "env_probe_ok": True,
        "stop_on_fail": stop_on_fail,
        "only_missing_chunks_executed": True,
        "workers_requested": n_workers,
        "workers_actual_parallel": workers_actual,
        "worker_thread_settings": thread_settings,
        "execution_mode": "fcfs_process_pool" if workers_actual > 1 else "sequential",
    }
    write_json_atomic(worker_root / REMAINING_MANIFEST, report)
    write_json_atomic(
        run_root / "m4_sample_runtime_provenance.json",
        collect_m4_runtime_provenance(
            run_root=run_root,
            workers_requested=n_workers,
            worker_remaining_manifest=report,
        ),
    )
    (worker_root / REMAINING_SUMMARY_MD).write_text(_render_summary_md(report), encoding="utf-8")

    preview = json.loads(json.dumps(manifest))
    preview["updated_utc"] = utc_now()
    preview["will_execute"] = False
    preview["mode"] = "m4_4_1b_4_worker_remaining"
    preview["worker_remaining"] = report
    preview["worker_remaining_terminal"] = status
    preview["pipeline_terminal_unchanged"] = manifest.get("terminal_status")
    st5 = preview.setdefault("stages", {}).setdefault("stage5_workers", {})
    st5["remaining_batch_status"] = status
    st5["remaining_executed_chunks"] = executed
    st5["remaining_preexisting_pass_chunks"] = plan.get("preexisting_pass_chunks")
    st5["updated_utc"] = utc_now()
    if status in (TERMINAL_PASS, TERMINAL_WARN) and not fail_chunks:
        st5["status"] = "PASS"
    elif fail_chunks:
        st5["status"] = "FAIL"
    write_json_atomic(run_root / "pipeline_run_manifest.m4_4_workers_complete_preview.json", preview)

    print(f"status={status}")
    print(f"executed_chunks={executed}")
    print(f"skipped_chunks={skipped}")
    print(f"pass_chunks={pass_chunks}")
    print(f"failed_chunks={fail_chunks}")
    print(f"wall_time_s={report['wall_time_s']}")

    if status == TERMINAL_FAIL:
        return 1
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="M4.4.1b-4: run remaining L_prod worker chunks (skip/reuse PASS)."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-run PASS chunks too.")
    parser.add_argument(
        "--no-stop-on-fail",
        action="store_true",
        help="Continue after a chunk FAIL (logs preserved).",
    )
    parser.add_argument("--solver-python", default=DEFAULT_SOLVER_PYTHON)
    parser.add_argument("--solver-venv", default=DEFAULT_SOLVER_VENV)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Max concurrent chunk solver processes (FCFS). Default 1 = sequential.",
    )
    args = parser.parse_args(argv)

    if args.dry_run and args.execute:
        print("error: use --dry-run or --execute, not both", file=sys.stderr)
        return 2
    if not args.dry_run and not args.execute:
        print("error: specify --dry-run or --execute", file=sys.stderr)
        return 2
    if int(args.workers) < 1:
        print("error: --workers must be >= 1", file=sys.stderr)
        return 2

    repo_root = detect_repo_root(SCRIPT_DIR)
    run_root = args.run_dir if args.run_dir.is_absolute() else repo_root / args.run_dir
    run_root = run_root.resolve()

    if args.dry_run:
        return run_dry_run(
            repo_root=repo_root,
            run_root=run_root,
            solver_python=str(args.solver_python),
            force=bool(args.force),
        )
    return run_execute(
        repo_root=repo_root,
        run_root=run_root,
        solver_python=str(args.solver_python),
        solver_venv=str(args.solver_venv),
        force=bool(args.force),
        stop_on_fail=not bool(args.no_stop_on_fail),
        n_workers=int(args.workers),
    )


if __name__ == "__main__":
    raise SystemExit(main())
