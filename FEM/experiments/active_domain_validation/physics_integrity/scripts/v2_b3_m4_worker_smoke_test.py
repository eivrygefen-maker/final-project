#!/usr/bin/env python3
"""M4.4.1b-1 — single-chunk L_prod worker smoke test (solver-mkl, one chunk only)."""
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
SOLVE_TARGET_LIST_REL = SCRIPTS_REL / "v2_b3_checkpoint_solve_target_list.py"

DEFAULT_SOLVER_PYTHON = "/home/vboxuser/solver-mkl/venv/bin/python"
DEFAULT_SOLVER_VENV = "/home/vboxuser/solver-mkl/venv"

TERMINAL_CHECKPOINT_READY = "LPROD_CHECKPOINT_READY"
SMOKE_TERMINAL_PASS = "WORKER_SMOKE_TEST_PASS"
RECOMMENDED_SMOKE_CHUNK = "sample_001_chunk_04"
TARGET_COUNT_LO = 3
TARGET_COUNT_HI = 5

MKL_PROBE_SCRIPT = """
from v2_b3_operator_checkpoint_portable import probe_pc_lu_factor_solver
p = probe_pc_lu_factor_solver("mkl_pardiso")
print("mkl_available", bool(p.get("available")))
print("mkl_error", p.get("error"))
""".strip()

REAL_WORKER_STATUSES = frozenset({"PASS", "PASS_WITH_WARNING", "PARTIAL"})

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_m3_orchestrator_run_one import (  # noqa: E402
    _run_subprocess,
    _verify_stage_a_export,
)
from v2_b3_m4_lprod_interfaces import (  # noqa: E402
    CHUNK_TARGETS_SCHEMA,
    build_worker_command_line,
    validate_chunk_targets_doc,
)
from v2_b3_petsc_util import write_json_atomic  # noqa: E402
from v2_b3_resolve_pilot_core_config import _repo_relative  # noqa: E402
from v2_b3_run_coarse_scout_lhs_batch import (  # noqa: E402
    STAGE_B_ENV_PROBE,
    _path_for_subprocess,
    _run_env_probe,
    _solver_mkl_subprocess_env_strict,
    _verify_stage_b_env_probe,
)


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


def _append_log(log_path: Path, text: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(text)
        if not text.endswith("\n"):
            fh.write("\n")


def _chunk_target_count(chunk_dir: Path) -> Optional[int]:
    path = chunk_dir / "chunk_targets.json"
    if not path.is_file():
        return None
    try:
        doc = _load_json(path)
        return len(doc.get("targets") or [])
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def recommend_smoke_chunk(run_root: Path) -> Tuple[str, str]:
    """Pick a 3–5 target chunk; prefer RECOMMENDED_SMOKE_CHUNK when valid."""
    worker_root = run_root / "worker_results"
    preferred = worker_root / RECOMMENDED_SMOKE_CHUNK
    n_pref = _chunk_target_count(preferred)
    if n_pref is not None and TARGET_COUNT_LO <= n_pref <= TARGET_COUNT_HI:
        return RECOMMENDED_SMOKE_CHUNK, (
            f"recommended default ({n_pref} targets, ZONE_2 medium band ~184–220 Hz)"
        )

    candidates: List[Tuple[str, int]] = []
    for chunk_dir in sorted(worker_root.iterdir()) if worker_root.is_dir() else []:
        if not chunk_dir.is_dir():
            continue
        n = _chunk_target_count(chunk_dir)
        if n is None:
            continue
        if TARGET_COUNT_LO <= n <= TARGET_COUNT_HI:
            candidates.append((chunk_dir.name, n))

    if not candidates:
        raise RuntimeError(
            "no worker chunk with 3–5 targets found; pass --chunk-id explicitly"
        )
    chunk_id, n = min(candidates, key=lambda row: (abs(row[1] - 4), row[0]))
    return chunk_id, f"auto-selected smallest medium chunk ({n} targets)"


def _verify_lprod_checkpoint(checkpoint_dir: Path) -> Tuple[bool, str]:
    manifest = checkpoint_dir / "checkpoint_export_manifest.json"
    if not manifest.is_file():
        return False, "missing checkpoint_export_manifest.json"
    ok, detail = _verify_stage_a_export(manifest)
    if not ok:
        return False, detail
    try:
        data = _load_json(manifest)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return False, str(exc)
    if not data.get("export_pass"):
        return False, "export_pass=false"
    if not data.get("matrix_verify_pass"):
        return False, "matrix_verify_pass=false"
    for name in ("A_active_csr.npz", "M_active_csr.npz", "built_metadata.json"):
        if not (checkpoint_dir / name).is_file():
            return False, f"missing {name}"
    built = checkpoint_dir / "built_metadata.json"
    built_meta = _load_json(built)
    if str(built_meta.get("mesh_level") or data.get("mesh_level") or "") != "L_prod":
        return False, "mesh_level is not L_prod"
    return True, "ok"


def _existing_real_worker_result(worker_result_path: Path) -> bool:
    if not worker_result_path.is_file():
        return False
    try:
        data = _load_json(worker_result_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if data.get("mode") in ("m4_4_1a_dry_run", "m4_4_1b_1_smoke_dry_run"):
        return False
    if data.get("status") in ("DRY_RUN_PLANNED",):
        return False
    if data.get("smoke_test_executed") and data.get("status") in (
        "PASS",
        "PASS_WITH_WARNING",
        "PARTIAL",
    ):
        return True
    if data.get("status") in REAL_WORKER_STATUSES and data.get("mode") != "m4_4_1a_dry_run":
        return True
    return False


def _run_solver_env_probe(
    *,
    repo_root: Path,
    solver_python: str,
    solver_venv: str,
    env_b: Dict[str, str],
    chunk_dir: Path,
) -> Tuple[bool, Dict[str, Any]]:
    rc_b, out_b = _run_env_probe(
        python=solver_python,
        script=STAGE_B_ENV_PROBE,
        env=env_b,
        cwd=repo_root,
    )
    ok_b, detail_b = _verify_stage_b_env_probe(
        out_b, solver_python=solver_python, solver_venv=solver_venv
    )

    rc_mkl, out_mkl = _run_env_probe(
        python=solver_python,
        script=MKL_PROBE_SCRIPT,
        env=env_b,
        cwd=repo_root,
    )
    mkl_available = "mkl_available True" in out_mkl
    mkl_ok = rc_mkl == 0 and mkl_available

    body = {
        "schema": "m4_worker_smoke_env_probe_v1",
        "generated_utc": _utc_now(),
        "stage_b_probe": {
            "ok": ok_b and rc_b == 0,
            "exit_code": rc_b,
            "detail": detail_b,
            "stdout": out_b,
        },
        "mkl_pardiso_probe": {
            "ok": mkl_ok,
            "exit_code": rc_mkl,
            "stdout": out_mkl,
            "note": "optional; failure is warning only unless stage_b fails",
        },
    }
    write_json_atomic(chunk_dir / "env_probe.json", body)
    env_pass = ok_b and rc_b == 0
    return env_pass, body


def _derive_smoke_worker_status(
    *,
    solve_rc: int,
    solver_result: Optional[Dict[str, Any]],
) -> Tuple[str, List[str], List[str]]:
    warnings: List[str] = []
    errors: List[str] = []
    if solver_result is None:
        errors.append("solver_result.json missing after solve")
        return "FAIL", warnings, errors

    st = str(solver_result.get("status") or "FAIL")
    if solve_rc != 0:
        warnings.append(f"solve process exit_code={solve_rc} (solver may still have written result)")
    agg = solver_result.get("aggregate") or {}
    attempted = int(agg.get("targets_attempted") or 0)
    succeeded = int(agg.get("targets_succeeded") or 0)
    unique = list(agg.get("unique_accepted_frequencies_hz") or [])

    if attempted <= 0:
        errors.append("targets_attempted=0")
        return "FAIL", warnings, errors

    if st == "PASS" and unique:
        return "PASS", warnings, errors
    if st == "PASS" and not unique:
        warnings.append("solver_status=PASS but no unique_accepted_frequencies_hz")
        return "PASS_WITH_WARNING", warnings, errors
    if st == "PARTIAL" or succeeded > 0:
        warnings.append(f"solver_status={st}; targets_succeeded={succeeded}/{attempted}")
        return "PASS_WITH_WARNING", warnings, errors
    if st == "FAIL" and solver_result.get("failure_reason"):
        warnings.append(f"solver failure_reason={solver_result.get('failure_reason')}")
    warnings.append(
        "solver ran (exit 0) but no modes accepted in per-target window_hz — check acceptance windows"
    )
    return "PASS_WITH_WARNING", warnings, errors


def _augment_worker_result(
    *,
    repo_root: Path,
    chunk_dir: Path,
    chunk_id: str,
    smoke_status: str,
    warnings: Sequence[str],
    errors: Sequence[str],
    solve_rc: int,
    env_probe: Dict[str, Any],
) -> Dict[str, Any]:
    worker_path = chunk_dir / "worker_result.json"
    worker = _load_json(worker_path) if worker_path.is_file() else {}
    solver_path = chunk_dir / "solver_result.json"
    solver = _load_json(solver_path) if solver_path.is_file() else {}

    agg = solver.get("aggregate") or {}
    unique = list(agg.get("unique_accepted_frequencies_hz") or [])
    worker.update(
        {
            "schema": "m4_worker_result_v1",
            "mode": "m4_4_1b_1_worker_smoke",
            "smoke_test_executed": True,
            "chunk_id": chunk_id,
            "worker_id": "smoke_W0",
            "status": smoke_status,
            "targets_attempted": int(
                agg.get("targets_attempted") or worker.get("targets_attempted") or 0
            ),
            "targets_passed": int(
                agg.get("targets_succeeded") or worker.get("targets_passed") or 0
            ),
            "accepted_modes": unique,
            "unique_modes": unique,
            "timing": {
                "wall_seconds": agg.get("total_wall_seconds"),
                "setup_seconds": agg.get("total_setup_seconds"),
                "solve_seconds": agg.get("total_solve_seconds"),
            },
            "warnings": list(warnings) + list(worker.get("warnings") or []),
            "errors": list(errors),
            "solver_result_json": _rel(solver_path, repo_root=repo_root),
            "solve_exit_code": solve_rc,
            "env_probe_ok": bool((env_probe.get("stage_b_probe") or {}).get("ok")),
            "per_target_windows_from_plan": bool(
                (solver.get("per_target_windows_from_plan"))
                or (solver.get("accepted_frequency_policy") == "discovery_band_and_target_window")
            ),
            "updated_utc": _utc_now(),
        }
    )
    write_json_atomic(worker_path, worker)
    return worker


def build_smoke_plan(
    *,
    repo_root: Path,
    run_root: Path,
    chunk_id: str,
    selection_note: str,
    solver_python: str,
    force: bool,
) -> Dict[str, Any]:
    checkpoint_dir = run_root / "lprod" / "checkpoint"
    chunk_dir = run_root / "worker_results" / chunk_id
    targets_path = chunk_dir / "chunk_targets.json"
    chunk_targets = _load_json(targets_path) if targets_path.is_file() else {}
    targets_hz = [float(t["target_hz"]) for t in (chunk_targets.get("targets") or [])]

    cmd_line = build_worker_command_line(
        repo_root=repo_root,
        checkpoint_dir=checkpoint_dir,
        chunk_targets_path=targets_path,
        output_dir=chunk_dir,
        solver_python=solver_python,
    )
    argv = [
        solver_python,
        _path_for_subprocess(repo_root / SOLVE_TARGET_LIST_REL, repo_root=repo_root),
        "--checkpoint-dir",
        _rel(checkpoint_dir, repo_root=repo_root),
        "--targets-json",
        _rel(targets_path, repo_root=repo_root),
        "--factor-solver",
        "mkl_pardiso",
        "--output-dir",
        _rel(chunk_dir, repo_root=repo_root),
    ]

    return {
        "schema": "m4_worker_smoke_plan_v1",
        "will_execute": False,
        "chunk_id": chunk_id,
        "chunk_selection_note": selection_note,
        "target_count": len(targets_hz),
        "targets_hz": targets_hz,
        "freq_range_hz": chunk_targets.get("freq_range_hz"),
        "only_chunk_executed": chunk_id,
        "argv_solve": argv,
        "command_preview": cmd_line,
        "paths": {
            "checkpoint_dir": _rel(checkpoint_dir, repo_root=repo_root),
            "chunk_dir": _rel(chunk_dir, repo_root=repo_root),
            "chunk_targets_json": _rel(targets_path, repo_root=repo_root),
        },
        "skip_solve": _existing_real_worker_result(chunk_dir / "worker_result.json") and not force,
    }


def _validate_preconditions(
    *,
    run_root: Path,
    chunk_id: str,
    manifest: Dict[str, Any],
    force: bool,
) -> List[str]:
    errors: List[str] = []
    term = str(manifest.get("terminal_status") or "")
    if term != TERMINAL_CHECKPOINT_READY:
        errors.append(f"terminal_status={term!r} expected {TERMINAL_CHECKPOINT_READY!r}")

    checkpoint_dir = run_root / "lprod" / "checkpoint"
    ok, detail = _verify_lprod_checkpoint(checkpoint_dir)
    if not ok:
        errors.append(f"lprod checkpoint: {detail}")

    chunk_dir = run_root / "worker_results" / chunk_id
    if not chunk_dir.is_dir():
        errors.append(f"missing chunk dir: {chunk_dir}")
        return errors

    targets_path = chunk_dir / "chunk_targets.json"
    if not targets_path.is_file():
        errors.append(f"missing chunk_targets.json for {chunk_id}")
    else:
        try:
            doc = _load_json(targets_path)
            if doc.get("schema") != CHUNK_TARGETS_SCHEMA:
                errors.append(f"{chunk_id}: schema mismatch")
            errors.extend(f"{chunk_id}: {e}" for e in validate_chunk_targets_doc(doc))
            if str(doc.get("chunk_id")) != chunk_id:
                errors.append(f"chunk_id mismatch in JSON: {doc.get('chunk_id')!r}")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"invalid chunk_targets.json: {exc}")

    if _existing_real_worker_result(chunk_dir / "worker_result.json") and not force:
        errors.append(
            f"{chunk_id}: real worker_result exists (use --force to re-run smoke solve)"
        )

    return errors


def run_dry_run(
    *,
    repo_root: Path,
    run_root: Path,
    chunk_id: Optional[str],
    solver_python: str,
    force: bool,
) -> int:
    manifest_path = run_root / "pipeline_run_manifest.json"
    manifest = _load_json(manifest_path)

    if not chunk_id:
        chunk_id, note = recommend_smoke_chunk(run_root)
    else:
        note = "user-specified --chunk-id"

    errors = _validate_preconditions(
        run_root=run_root, chunk_id=chunk_id, manifest=manifest, force=force
    )
    hard_errors = [e for e in errors if "use --force" not in e]
    if hard_errors:
        print("error: preconditions failed:", file=sys.stderr)
        for e in hard_errors:
            print(f"  - {e}", file=sys.stderr)
        return 2
    if errors:
        print("note: real worker_result exists; use --force on --execute to re-run", flush=True)

    plan = build_smoke_plan(
        repo_root=repo_root,
        run_root=run_root,
        chunk_id=chunk_id,
        selection_note=note,
        solver_python=solver_python,
        force=force,
    )
    write_json_atomic(run_root / "worker_results" / chunk_id / "worker_smoke_plan.json", plan)

    print("will_execute=false")
    print(f"selected_chunk={chunk_id}")
    print(f"chunk_selection={note}")
    print(f"target_count={plan.get('target_count')}")
    print(f"targets_hz={plan.get('targets_hz')}")
    print(f"checkpoint={plan['paths']['checkpoint_dir']}")
    print(f"command_preview={plan.get('command_preview')}")
    print("no other chunks will be executed")
    return 0


def run_execute(
    *,
    repo_root: Path,
    run_root: Path,
    chunk_id: Optional[str],
    solver_python: str,
    solver_venv: str,
    force: bool,
) -> int:
    manifest_path = run_root / "pipeline_run_manifest.json"
    manifest = _load_json(manifest_path)

    if not chunk_id:
        chunk_id, note = recommend_smoke_chunk(run_root)
    else:
        note = "user-specified --chunk-id"

    errors = _validate_preconditions(
        run_root=run_root, chunk_id=chunk_id, manifest=manifest, force=force
    )
    if errors:
        print("error: preconditions failed:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 2

    checkpoint_dir = run_root / "lprod" / "checkpoint"
    chunk_dir = run_root / "worker_results" / chunk_id
    log_path = chunk_dir / "log.txt"
    plan = build_smoke_plan(
        repo_root=repo_root,
        run_root=run_root,
        chunk_id=chunk_id,
        selection_note=note,
        solver_python=solver_python,
        force=force,
    )

    if plan.get("skip_solve"):
        print(f"[worker_smoke] reuse existing real worker_result for {chunk_id}", flush=True)
        worker = _load_json(chunk_dir / "worker_result.json")
        smoke_status = str(worker.get("status") or "PASS")
    else:
        _append_log(log_path, f"[{_utc_now()}] M4.4.1b-1 worker smoke test chunk={chunk_id}\n")

        env_b = _solver_mkl_subprocess_env_strict(
            solver_python=solver_python, solver_venv=solver_venv
        )
        env_pass, env_body = _run_solver_env_probe(
            repo_root=repo_root,
            solver_python=solver_python,
            solver_venv=solver_venv,
            env_b=env_b,
            chunk_dir=chunk_dir,
        )
        if not env_pass:
            _append_log(log_path, f"env_probe FAIL: {env_body}\n")
            _write_smoke_failure(
                run_root=run_root,
                manifest=manifest,
                chunk_id=chunk_id,
                chunk_dir=chunk_dir,
                reason="env_probe_failed",
            )
            print("env_probe FAIL", flush=True)
            return 1
        print("env_probe PASS", flush=True)
        _append_log(log_path, "env_probe PASS\n")

        if not (env_body.get("mkl_pardiso_probe") or {}).get("ok"):
            _append_log(log_path, "WARN: mkl_pardiso probe not confirmed available\n")

        print(f"checkpoint PASS", flush=True)
        print(f"selected chunk = {chunk_id} ({plan.get('target_count')} targets)", flush=True)
        print("worker solve starts", flush=True)

        rc = _run_subprocess(
            plan["argv_solve"],
            env=env_b,
            cwd=repo_root,
            log_path=log_path,
            label=f"worker_smoke_{chunk_id}",
        )
        print(f"worker solve exit_code={rc}", flush=True)
        _append_log(log_path, f"worker solve exit_code={rc}\n")

        solver_result = None
        solver_path = chunk_dir / "solver_result.json"
        if solver_path.is_file():
            solver_result = _load_json(solver_path)

        smoke_status, warnings, errors = _derive_smoke_worker_status(
            solve_rc=rc, solver_result=solver_result
        )
        worker = _augment_worker_result(
            repo_root=repo_root,
            chunk_dir=chunk_dir,
            chunk_id=chunk_id,
            smoke_status=smoke_status,
            warnings=warnings,
            errors=errors,
            solve_rc=rc,
            env_probe=env_body,
        )
        print(f"worker_result.status={worker.get('status')}", flush=True)
        if warnings:
            for w in warnings:
                print(f"  warning: {w}", flush=True)

    smoke_manifest = {
        "schema": "m4_worker_smoke_manifest_v1",
        "generated_utc": _utc_now(),
        "chunk_id": chunk_id,
        "chunk_selection_note": note,
        "only_chunk_executed": chunk_id,
        "other_chunks_touched": False,
        "terminal_status": SMOKE_TERMINAL_PASS
        if smoke_status in ("PASS", "PASS_WITH_WARNING")
        else "WORKER_SMOKE_TEST_FAIL",
        "worker_result_status": smoke_status,
        "target_count": plan.get("target_count"),
        "targets_hz": plan.get("targets_hz"),
        "paths": plan.get("paths"),
    }
    write_json_atomic(chunk_dir / "worker_smoke_manifest.json", smoke_manifest)

    preview = json.loads(json.dumps(manifest))
    preview["updated_utc"] = _utc_now()
    preview["will_execute"] = False
    preview["mode"] = "m4_4_1b_1_worker_smoke"
    preview["worker_smoke_test"] = smoke_manifest
    preview["worker_smoke_terminal"] = smoke_manifest["terminal_status"]
    preview["pipeline_terminal_unchanged"] = manifest.get("terminal_status")
    stages = preview.setdefault("stages", {})
    st5 = stages.setdefault("stage5_workers", {})
    if st5.get("status") not in ("PASS",):
        st5["status"] = "SMOKE_PASS" if smoke_status in ("PASS", "PASS_WITH_WARNING") else "SMOKE_FAIL"
        st5["smoke_chunk_id"] = chunk_id
        st5["updated_utc"] = _utc_now()
    write_json_atomic(run_root / "pipeline_run_manifest.m4_4_worker_smoke_preview.json", preview)

    if smoke_status not in ("PASS", "PASS_WITH_WARNING"):
        print("WORKER_SMOKE_TEST_FAIL", flush=True)
        return 1

    print(SMOKE_TERMINAL_PASS, flush=True)
    print("no other chunks executed", flush=True)
    return 0


def _write_smoke_failure(
    *,
    run_root: Path,
    manifest: Dict[str, Any],
    chunk_id: str,
    chunk_dir: Path,
    reason: str,
) -> None:
    body = {
        "schema": "m4_worker_smoke_manifest_v1",
        "generated_utc": _utc_now(),
        "chunk_id": chunk_id,
        "terminal_status": "WORKER_SMOKE_TEST_FAIL",
        "failure_reason": reason,
    }
    write_json_atomic(chunk_dir / "worker_smoke_manifest.json", body)
    preview = json.loads(json.dumps(manifest))
    preview["worker_smoke_test"] = body
    preview["worker_smoke_terminal"] = "WORKER_SMOKE_TEST_FAIL"
    write_json_atomic(run_root / "pipeline_run_manifest.m4_4_worker_smoke_preview.json", preview)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="M4.4.1b-1: single-chunk L_prod worker smoke test (solver-mkl)."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--chunk-id",
        help=f"Worker chunk to solve (recommended: {RECOMMENDED_SMOKE_CHUNK}, 5 targets).",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-run even if real worker_result exists.")
    parser.add_argument("--solver-python", default=DEFAULT_SOLVER_PYTHON)
    parser.add_argument("--solver-venv", default=DEFAULT_SOLVER_VENV)
    args = parser.parse_args(argv)

    if args.dry_run and args.execute:
        print("error: use --dry-run or --execute, not both", file=sys.stderr)
        return 2
    if not args.dry_run and not args.execute:
        print("error: specify --dry-run or --execute", file=sys.stderr)
        return 2

    repo_root = _detect_repo_root(SCRIPT_DIR)
    run_root = args.run_dir if args.run_dir.is_absolute() else repo_root / args.run_dir
    run_root = run_root.resolve()

    if args.dry_run:
        return run_dry_run(
            repo_root=repo_root,
            run_root=run_root,
            chunk_id=str(args.chunk_id) if args.chunk_id else None,
            solver_python=str(args.solver_python),
            force=bool(args.force),
        )
    return run_execute(
        repo_root=repo_root,
        run_root=run_root,
        chunk_id=str(args.chunk_id) if args.chunk_id else None,
        solver_python=str(args.solver_python),
        solver_venv=str(args.solver_venv),
        force=bool(args.force),
    )


if __name__ == "__main__":
    raise SystemExit(main())
