#!/usr/bin/env python3
"""M4.4.1b-2 — limited multi-chunk L_prod worker mini-batch (solver-mkl)."""
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

from v2_b3_m4_worker_run_lib import (  # noqa: E402
    DEFAULT_MINIBATCH_CHUNKS,
    PASS_LIKE,
    RECOMMENDED_SMOKE_CHUNK,
    TERMINAL_CHECKPOINT_READY,
    auto_pick_minibatch_chunks,
    build_chunk_plan,
    detect_repo_root,
    execute_worker_chunk,
    load_json,
    rel,
    run_solver_env_probe,
    utc_now,
    validate_chunk_preconditions,
    validate_global_preconditions,
)
from v2_b3_petsc_util import write_json_atomic  # noqa: E402
from v2_b3_run_coarse_scout_lhs_batch import (  # noqa: E402
    DEFAULT_SOLVER_PYTHON,
    _solver_mkl_subprocess_env_strict,
)

DEFAULT_SOLVER_VENV = "/home/vboxuser/solver-mkl/venv"

MINIBATCH_ID = "m4_4_1b_2"
MINIBATCH_MANIFEST = "minibatch_m4_4_1b_2_manifest.json"
MINIBATCH_SUMMARY_MD = "minibatch_m4_4_1b_2_summary.md"
MINIBATCH_TERMINAL_PASS = "WORKER_MINIBATCH_PASS"
MINIBATCH_TERMINAL_WARN = "WORKER_MINIBATCH_PASS_WITH_WARNING"
MINIBATCH_TERMINAL_FAIL = "WORKER_MINIBATCH_FAIL"


def _parse_chunk_ids(raw: Optional[str]) -> Optional[List[str]]:
    if not raw:
        return None
    return [c.strip() for c in str(raw).split(",") if c.strip()]


def _render_summary_md(report: Dict[str, Any]) -> str:
    lines = [
        f"# Worker mini-batch ({MINIBATCH_ID}) — {report.get('sample_id')}",
        "",
        f"- run_id: `{report.get('run_id')}`",
        f"- status: **{report.get('status')}**",
        f"- will_execute: **{report.get('will_execute', False)}**",
        f"- wall_time_s: **{report.get('wall_time_s')}**",
        "",
        "## Chunks",
        "",
        f"- selected: `{report.get('selected_chunk_ids')}`",
        f"- executed: `{report.get('executed_chunks')}`",
        f"- skipped (reuse): `{report.get('skipped_chunks')}`",
        f"- pass: `{report.get('pass_chunks')}`",
        f"- warning: `{report.get('warning_chunks')}`",
        f"- failed: `{report.get('failed_chunks')}`",
        "",
        "## Totals",
        "",
        f"- targets_attempted: **{report.get('total_targets_attempted')}**",
        f"- targets_passed: **{report.get('total_targets_passed')}**",
        f"- accepted_modes: **{report.get('total_accepted_modes')}**",
        f"- unique_modes: **{report.get('total_unique_modes')}**",
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


def _aggregate_minibatch_status(chunk_results: Sequence[Dict[str, Any]]) -> str:
    if any(r.get("status") == "FAIL" for r in chunk_results):
        return MINIBATCH_TERMINAL_FAIL
    if any(r.get("status") == "PASS_WITH_WARNING" for r in chunk_results):
        return MINIBATCH_TERMINAL_WARN
    if all(r.get("status") in PASS_LIKE for r in chunk_results):
        return MINIBATCH_TERMINAL_PASS
    return MINIBATCH_TERMINAL_FAIL


def build_minibatch_plan(
    *,
    repo_root: Path,
    run_root: Path,
    chunk_ids: List[str],
    manifest: Dict[str, Any],
    solver_python: str,
    force: bool,
) -> Dict[str, Any]:
    chunk_plans: List[Dict[str, Any]] = []
    skipped: List[str] = []
    to_run: List[str] = []

    for cid in chunk_ids:
        errs = validate_chunk_preconditions(run_root=run_root, chunk_id=cid, force=force)
        hard = [e for e in errs if "use --force" not in e]
        plan = build_chunk_plan(
            repo_root=repo_root,
            run_root=run_root,
            chunk_id=cid,
            solver_python=solver_python,
            force=force,
        )
        if hard:
            plan["precheck_errors"] = hard
            plan["runnable"] = False
        else:
            plan["runnable"] = True
            if plan["skip_solve"]:
                skipped.append(cid)
            else:
                to_run.append(cid)
        chunk_plans.append(plan)

    return {
        "schema": "m4_worker_minibatch_plan_v1",
        "will_execute": False,
        "minibatch_id": MINIBATCH_ID,
        "sample_id": manifest.get("sample_id"),
        "run_id": manifest.get("run_id"),
        "selected_chunk_ids": chunk_ids,
        "chunks_to_execute": to_run,
        "chunks_to_skip_reuse": skipped,
        "chunk_plans": chunk_plans,
        "exclude_smoke_chunk_by_default": RECOMMENDED_SMOKE_CHUNK,
        "only_selected_chunks": True,
    }


def run_dry_run(
    *,
    repo_root: Path,
    run_root: Path,
    chunk_ids: Optional[List[str]],
    solver_python: str,
    force: bool,
) -> int:
    manifest_path = run_root / "pipeline_run_manifest.json"
    manifest = load_json(manifest_path)

    global_errors = validate_global_preconditions(run_root=run_root, manifest=manifest)
    if global_errors:
        print("error: global preconditions failed:", file=sys.stderr)
        for e in global_errors:
            print(f"  - {e}", file=sys.stderr)
        return 2

    if not chunk_ids:
        chunk_ids = auto_pick_minibatch_chunks(run_root, max_chunks=3)
        selection_note = f"auto-picked {len(chunk_ids)} chunks (excludes {RECOMMENDED_SMOKE_CHUNK} if PASS)"
    else:
        selection_note = "user --chunk-ids"

    plan = build_minibatch_plan(
        repo_root=repo_root,
        run_root=run_root,
        chunk_ids=chunk_ids,
        manifest=manifest,
        solver_python=solver_python,
        force=force,
    )
    plan["selection_note"] = selection_note
    worker_root = run_root / "worker_results"
    write_json_atomic(worker_root / MINIBATCH_MANIFEST.replace(".json", "_plan.json"), plan)

    hard_any = False
    for cp in plan["chunk_plans"]:
        if cp.get("precheck_errors"):
            hard_any = True
            for e in cp["precheck_errors"]:
                print(f"  - {e}", file=sys.stderr)

    if hard_any:
        print("error: one or more chunks failed precheck", file=sys.stderr)
        return 2

    print("will_execute=false")
    print(f"selected_chunks={len(chunk_ids)}")
    print(f"chunk_ids={chunk_ids}")
    print(f"selection={selection_note}")
    print(f"to_execute={plan.get('chunks_to_execute')}")
    print(f"to_skip_reuse={plan.get('chunks_to_skip_reuse')}")
    print("no other chunks will be executed")
    for cp in plan["chunk_plans"]:
        print(f"  {cp['chunk_id']}: targets={cp.get('target_count')} skip={cp.get('skip_solve')}")
        if not cp.get("skip_solve"):
            print(f"    cmd={cp.get('command_preview')}")
    return 0


def run_execute(
    *,
    repo_root: Path,
    run_root: Path,
    chunk_ids: Optional[List[str]],
    solver_python: str,
    solver_venv: str,
    force: bool,
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

    if not chunk_ids:
        chunk_ids = auto_pick_minibatch_chunks(run_root, max_chunks=3)

    if not chunk_ids:
        print("error: no chunks selected for mini-batch", file=sys.stderr)
        return 2

    for cid in chunk_ids:
        errs = validate_chunk_preconditions(run_root=run_root, chunk_id=cid, force=force)
        hard = [e for e in errs if "use --force" not in e]
        if hard:
            print("error: chunk preconditions failed:", file=sys.stderr)
            for e in hard:
                print(f"  - {e}", file=sys.stderr)
            return 2

    env_b = _solver_mkl_subprocess_env_strict(
        solver_python=solver_python, solver_venv=solver_venv
    )
    t_wall0 = time.perf_counter()
    env_pass, env_body = run_solver_env_probe(
        repo_root=repo_root,
        solver_python=solver_python,
        solver_venv=solver_venv,
        env_b=env_b,
        chunk_dir=worker_root,
    )
    if not env_pass:
        print("env_probe FAIL (minibatch)", flush=True)
        return 1
    print("env_probe PASS", flush=True)
    print(f"checkpoint PASS ({TERMINAL_CHECKPOINT_READY})", flush=True)
    print(f"selected chunks = {len(chunk_ids)}: {chunk_ids}", flush=True)

    chunk_results: List[Dict[str, Any]] = []
    for i, cid in enumerate(chunk_ids):
        print(f"[minibatch] chunk {i + 1}/{len(chunk_ids)}: {cid}", flush=True)
        result = execute_worker_chunk(
            repo_root=repo_root,
            run_root=run_root,
            chunk_id=cid,
            solver_python=solver_python,
            solver_venv=solver_venv,
            force=force,
            env_b=env_b,
            env_probe_body=env_body,
            run_env_probe=False,
            worker_id=f"minibatch_W{i}",
            mode="m4_4_1b_2_worker_minibatch",
            minibatch_id=MINIBATCH_ID,
            label_prefix="worker_minibatch",
        )
        chunk_results.append(result)
        print(
            f"  {cid}: action={result.get('action')} status={result.get('status')} "
            f"exit={result.get('solve_exit_code')} "
            f"targets={result.get('targets_passed')}/{result.get('targets_attempted')}",
            flush=True,
        )

    wall_s = time.perf_counter() - t_wall0
    executed = [r["chunk_id"] for r in chunk_results if r.get("action") == "executed"]
    skipped = [r["chunk_id"] for r in chunk_results if r.get("action") == "skipped_reuse"]
    pass_chunks = [r["chunk_id"] for r in chunk_results if r.get("status") == "PASS"]
    warn_chunks = [r["chunk_id"] for r in chunk_results if r.get("status") == "PASS_WITH_WARNING"]
    fail_chunks = [r["chunk_id"] for r in chunk_results if r.get("status") == "FAIL"]

    total_attempted = sum(int(r.get("targets_attempted") or 0) for r in chunk_results)
    total_passed = sum(int(r.get("targets_passed") or 0) for r in chunk_results)
    total_accepted = sum(int(r.get("accepted_mode_count") or 0) for r in chunk_results)
    total_unique = sum(int(r.get("unique_mode_count") or 0) for r in chunk_results)

    status = _aggregate_minibatch_status(chunk_results)

    report = {
        "schema": "m4_worker_minibatch_manifest_v1",
        "will_execute": True,
        "minibatch_id": MINIBATCH_ID,
        "generated_utc": utc_now(),
        "sample_id": manifest.get("sample_id"),
        "run_id": manifest.get("run_id"),
        "status": status,
        "selected_chunk_ids": chunk_ids,
        "skipped_chunks": skipped,
        "executed_chunks": executed,
        "pass_chunks": pass_chunks,
        "warning_chunks": warn_chunks,
        "failed_chunks": fail_chunks,
        "total_targets_attempted": total_attempted,
        "total_targets_passed": total_passed,
        "total_accepted_modes": total_accepted,
        "total_unique_modes": total_unique,
        "wall_time_s": round(wall_s, 2),
        "chunk_results": chunk_results,
        "env_probe_ok": True,
        "only_selected_chunks_executed": True,
    }
    write_json_atomic(worker_root / MINIBATCH_MANIFEST, report)
    (worker_root / MINIBATCH_SUMMARY_MD).write_text(_render_summary_md(report), encoding="utf-8")

    preview = json.loads(json.dumps(manifest))
    preview["updated_utc"] = utc_now()
    preview["will_execute"] = False
    preview["mode"] = "m4_4_1b_2_worker_minibatch"
    preview["worker_minibatch"] = report
    preview["worker_minibatch_terminal"] = status
    preview["pipeline_terminal_unchanged"] = manifest.get("terminal_status")
    st5 = preview.setdefault("stages", {}).setdefault("stage5_workers", {})
    st5["minibatch_status"] = status
    st5["minibatch_chunk_ids"] = chunk_ids
    st5["updated_utc"] = utc_now()
    if st5.get("status") not in ("PASS",):
        st5["status"] = "MINIBATCH_PARTIAL"
    write_json_atomic(run_root / "pipeline_run_manifest.m4_4_worker_minibatch_preview.json", preview)

    print(f"minibatch_status={status}")
    print(f"pass_chunks={pass_chunks}")
    print(f"warning_chunks={warn_chunks}")
    print(f"failed_chunks={fail_chunks}")
    print(f"wall_time_s={report['wall_time_s']}")
    print("no other chunks executed")

    if status == MINIBATCH_TERMINAL_FAIL:
        return 1
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="M4.4.1b-2: limited L_prod worker mini-batch (2–3 chunks, solver-mkl)."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--chunk-ids",
        help=f"Comma-separated chunk IDs (default auto: {','.join(DEFAULT_MINIBATCH_CHUNKS)}).",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--solver-python", default=DEFAULT_SOLVER_PYTHON)
    parser.add_argument("--solver-venv", default=DEFAULT_SOLVER_VENV)
    args = parser.parse_args(argv)

    if args.dry_run and args.execute:
        print("error: use --dry-run or --execute, not both", file=sys.stderr)
        return 2
    if not args.dry_run and not args.execute:
        print("error: specify --dry-run or --execute", file=sys.stderr)
        return 2

    repo_root = detect_repo_root(SCRIPT_DIR)
    run_root = args.run_dir if args.run_dir.is_absolute() else repo_root / args.run_dir
    run_root = run_root.resolve()
    chunk_ids = _parse_chunk_ids(args.chunk_ids)

    if args.dry_run:
        return run_dry_run(
            repo_root=repo_root,
            run_root=run_root,
            chunk_ids=chunk_ids,
            solver_python=str(args.solver_python),
            force=bool(args.force),
        )
    return run_execute(
        repo_root=repo_root,
        run_root=run_root,
        chunk_ids=chunk_ids,
        solver_python=str(args.solver_python),
        solver_venv=str(args.solver_venv),
        force=bool(args.force),
    )


if __name__ == "__main__":
    raise SystemExit(main())
