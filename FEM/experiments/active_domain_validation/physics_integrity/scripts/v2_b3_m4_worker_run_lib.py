"""M4.4.1b — shared L_prod worker chunk execution (solver-mkl, target-list solve)."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_REL = Path("FEM/experiments/active_domain_validation/physics_integrity/scripts")
SOLVE_TARGET_LIST_REL = SCRIPTS_REL / "v2_b3_checkpoint_solve_target_list.py"

TERMINAL_CHECKPOINT_READY = "LPROD_CHECKPOINT_READY"
RECOMMENDED_SMOKE_CHUNK = "sample_001_chunk_04"
DEFAULT_MINIBATCH_CHUNKS = (
    "sample_001_chunk_08",
    "sample_001_chunk_10",
    "sample_001_chunk_11",
)
TARGET_COUNT_LO = 3
TARGET_COUNT_HI = 5

MKL_PROBE_SCRIPT = """
from v2_b3_operator_checkpoint_portable import probe_pc_lu_factor_solver
p = probe_pc_lu_factor_solver("mkl_pardiso")
print("mkl_available", bool(p.get("available")))
print("mkl_error", p.get("error"))
""".strip()

REAL_WORKER_STATUSES = frozenset({"PASS", "PASS_WITH_WARNING", "PARTIAL"})
PASS_LIKE = frozenset({"PASS", "PASS_WITH_WARNING"})

import sys

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_m3_orchestrator_run_one import _run_subprocess, _verify_stage_a_export  # noqa: E402
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


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def detect_repo_root(start: Path) -> Path:
    cur = start.resolve()
    while cur.parent != cur:
        if (cur / ".git").exists():
            return cur
        cur = cur.parent
    raise RuntimeError("Could not detect repository root (missing .git ancestor)")


def rel(path: Path, *, repo_root: Path) -> str:
    return _repo_relative(path, repo_root=repo_root)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected JSON object")
    return data


def append_log(log_path: Path, text: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(text)
        if not text.endswith("\n"):
            fh.write("\n")


def chunk_target_count(chunk_dir: Path) -> Optional[int]:
    path = chunk_dir / "chunk_targets.json"
    if not path.is_file():
        return None
    try:
        doc = load_json(path)
        return len(doc.get("targets") or [])
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def chunk_freq_range(chunk_dir: Path) -> Optional[List[float]]:
    path = chunk_dir / "chunk_targets.json"
    if not path.is_file():
        return None
    try:
        doc = load_json(path)
        fr = doc.get("freq_range_hz")
        if isinstance(fr, list) and len(fr) == 2:
            return [float(fr[0]), float(fr[1])]
    except (OSError, ValueError, json.JSONDecodeError, TypeError):
        return None
    return None


def verify_lprod_checkpoint(checkpoint_dir: Path) -> Tuple[bool, str]:
    manifest = checkpoint_dir / "checkpoint_export_manifest.json"
    if not manifest.is_file():
        return False, "missing checkpoint_export_manifest.json"
    ok, detail = _verify_stage_a_export(manifest)
    if not ok:
        return False, detail
    try:
        data = load_json(manifest)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return False, str(exc)
    if not data.get("export_pass"):
        return False, "export_pass=false"
    if not data.get("matrix_verify_pass"):
        return False, "matrix_verify_pass=false"
    for name in ("A_active_csr.npz", "M_active_csr.npz", "built_metadata.json"):
        if not (checkpoint_dir / name).is_file():
            return False, f"missing {name}"
    built_meta = load_json(checkpoint_dir / "built_metadata.json")
    if str(built_meta.get("mesh_level") or data.get("mesh_level") or "") != "L_prod":
        return False, "mesh_level is not L_prod"
    return True, "ok"


def existing_real_worker_result(worker_result_path: Path) -> bool:
    if not worker_result_path.is_file():
        return False
    try:
        data = load_json(worker_result_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if data.get("mode") in ("m4_4_1a_dry_run", "m4_4_1b_1_smoke_dry_run"):
        return False
    if data.get("status") in ("DRY_RUN_PLANNED",):
        return False
    if data.get("smoke_test_executed") or data.get("minibatch_executed"):
        if data.get("status") in PASS_LIKE or data.get("status") in REAL_WORKER_STATUSES:
            return True
    if data.get("status") in REAL_WORKER_STATUSES and data.get("mode") != "m4_4_1a_dry_run":
        return True
    return False


def validate_global_preconditions(
    *,
    run_root: Path,
    manifest: Dict[str, Any],
) -> List[str]:
    errors: List[str] = []
    term = str(manifest.get("terminal_status") or "")
    if term != TERMINAL_CHECKPOINT_READY:
        errors.append(f"terminal_status={term!r} expected {TERMINAL_CHECKPOINT_READY!r}")
    ok, detail = verify_lprod_checkpoint(run_root / "lprod" / "checkpoint")
    if not ok:
        errors.append(f"lprod checkpoint: {detail}")
    return errors


def validate_chunk_preconditions(
    *,
    run_root: Path,
    chunk_id: str,
    force: bool,
) -> List[str]:
    errors: List[str] = []
    chunk_dir = run_root / "worker_results" / chunk_id
    if not chunk_dir.is_dir():
        errors.append(f"missing chunk dir: {chunk_dir}")
        return errors

    targets_path = chunk_dir / "chunk_targets.json"
    if not targets_path.is_file():
        errors.append(f"missing chunk_targets.json for {chunk_id}")
    else:
        try:
            doc = load_json(targets_path)
            if doc.get("schema") != CHUNK_TARGETS_SCHEMA:
                errors.append(f"{chunk_id}: schema mismatch")
            errors.extend(f"{chunk_id}: {e}" for e in validate_chunk_targets_doc(doc))
            if str(doc.get("chunk_id")) != chunk_id:
                errors.append(f"chunk_id mismatch in JSON: {doc.get('chunk_id')!r}")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"invalid chunk_targets.json: {exc}")

    if existing_real_worker_result(chunk_dir / "worker_result.json") and not force:
        errors.append(f"{chunk_id}: real worker_result exists (use --force to re-run)")
    return errors


def run_solver_env_probe(
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
    mkl_ok = rc_mkl == 0 and "mkl_available True" in out_mkl
    body = {
        "schema": "m4_worker_smoke_env_probe_v1",
        "generated_utc": utc_now(),
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
    return ok_b and rc_b == 0, body


def derive_worker_status(
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
        "solver ran but no modes accepted in per-target window_hz — check acceptance windows"
    )
    return "PASS_WITH_WARNING", warnings, errors


def augment_worker_result(
    *,
    repo_root: Path,
    chunk_dir: Path,
    chunk_id: str,
    worker_status: str,
    warnings: Sequence[str],
    errors: Sequence[str],
    solve_rc: int,
    env_probe: Dict[str, Any],
    worker_id: str,
    mode: str,
    minibatch_id: Optional[str] = None,
) -> Dict[str, Any]:
    worker_path = chunk_dir / "worker_result.json"
    worker = load_json(worker_path) if worker_path.is_file() else {}
    solver_path = chunk_dir / "solver_result.json"
    solver = load_json(solver_path) if solver_path.is_file() else {}

    agg = solver.get("aggregate") or {}
    unique = list(agg.get("unique_accepted_frequencies_hz") or [])
    accepted = unique
    if isinstance(worker.get("accepted_modes"), list) and worker.get("accepted_modes"):
        accepted = worker["accepted_modes"]

    worker.update(
        {
            "schema": "m4_worker_result_v1",
            "mode": mode,
            "smoke_test_executed": mode.startswith("m4_4_1b_1"),
            "minibatch_executed": mode.startswith("m4_4_1b_2"),
            "minibatch_id": minibatch_id,
            "chunk_id": chunk_id,
            "worker_id": worker_id,
            "status": worker_status,
            "targets_attempted": int(
                agg.get("targets_attempted") or worker.get("targets_attempted") or 0
            ),
            "targets_passed": int(
                agg.get("targets_succeeded") or worker.get("targets_passed") or 0
            ),
            "accepted_modes": accepted,
            "unique_modes": unique,
            "timing": {
                "wall_seconds": agg.get("total_wall_seconds"),
                "setup_seconds": agg.get("total_setup_seconds"),
                "solve_seconds": agg.get("total_solve_seconds"),
            },
            "warnings": list(warnings) + list(worker.get("warnings") or []),
            "errors": list(errors),
            "solver_result_json": rel(solver_path, repo_root=repo_root),
            "solve_exit_code": solve_rc,
            "env_probe_ok": bool((env_probe.get("stage_b_probe") or {}).get("ok")),
            "updated_utc": utc_now(),
        }
    )
    write_json_atomic(worker_path, worker)
    return worker


def build_chunk_plan(
    *,
    repo_root: Path,
    run_root: Path,
    chunk_id: str,
    solver_python: str,
    force: bool,
) -> Dict[str, Any]:
    checkpoint_dir = run_root / "lprod" / "checkpoint"
    chunk_dir = run_root / "worker_results" / chunk_id
    targets_path = chunk_dir / "chunk_targets.json"
    chunk_targets = load_json(targets_path) if targets_path.is_file() else {}
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
        rel(checkpoint_dir, repo_root=repo_root),
        "--targets-json",
        rel(targets_path, repo_root=repo_root),
        "--factor-solver",
        "mkl_pardiso",
        "--output-dir",
        rel(chunk_dir, repo_root=repo_root),
    ]
    skip = existing_real_worker_result(chunk_dir / "worker_result.json") and not force
    return {
        "chunk_id": chunk_id,
        "target_count": len(targets_hz),
        "targets_hz": targets_hz,
        "freq_range_hz": chunk_targets.get("freq_range_hz"),
        "argv_solve": argv,
        "command_preview": cmd_line,
        "skip_solve": skip,
        "paths": {
            "chunk_dir": rel(chunk_dir, repo_root=repo_root),
            "chunk_targets_json": rel(targets_path, repo_root=repo_root),
        },
    }


def worker_result_summary(worker: Dict[str, Any]) -> Dict[str, Any]:
    accepted = worker.get("accepted_modes") or []
    unique = worker.get("unique_modes") or []
    return {
        "status": worker.get("status"),
        "targets_attempted": int(worker.get("targets_attempted") or 0),
        "targets_passed": int(worker.get("targets_passed") or 0),
        "accepted_mode_count": len(accepted) if isinstance(accepted, list) else 0,
        "unique_mode_count": len(unique) if isinstance(unique, list) else 0,
        "warnings": list(worker.get("warnings") or []),
        "errors": list(worker.get("errors") or []),
        "solve_exit_code": worker.get("solve_exit_code"),
    }


def auto_pick_minibatch_chunks(
    run_root: Path,
    *,
    max_chunks: int = 3,
    exclude: Optional[Sequence[str]] = None,
) -> List[str]:
    """Pick chunks in different bands, skip already-PASS, never auto-include smoke chunk."""
    exclude_set = set(exclude or ())
    exclude_set.add(RECOMMENDED_SMOKE_CHUNK)

    picked: List[str] = []
    worker_root = run_root / "worker_results"
    for cid in DEFAULT_MINIBATCH_CHUNKS:
        if cid in exclude_set:
            continue
        chunk_dir = worker_root / cid
        if chunk_target_count(chunk_dir) is None:
            continue
        if existing_real_worker_result(chunk_dir / "worker_result.json"):
            continue
        picked.append(cid)
        if len(picked) >= max_chunks:
            return picked

    candidates: List[Tuple[str, float, int]] = []
    for chunk_dir in sorted(worker_root.iterdir()) if worker_root.is_dir() else []:
        if not chunk_dir.is_dir():
            continue
        cid = chunk_dir.name
        if cid in exclude_set or cid in picked:
            continue
        n = chunk_target_count(chunk_dir)
        if n is None or not (TARGET_COUNT_LO <= n <= TARGET_COUNT_HI):
            continue
        if existing_real_worker_result(chunk_dir / "worker_result.json"):
            continue
        fr = chunk_freq_range(chunk_dir)
        lo = float(fr[0]) if fr else 0.0
        candidates.append((cid, lo, n))

    candidates.sort(key=lambda row: row[1])
    for cid, _, _ in candidates:
        if cid not in picked:
            picked.append(cid)
        if len(picked) >= max_chunks:
            break
    return picked[:max_chunks]


def chunk_ids_from_worker_plan(run_root: Path) -> List[str]:
    plan_path = run_root / "lprod" / "worker_chunk_plan.preview.json"
    if not plan_path.is_file():
        return []
    try:
        plan = load_json(plan_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return []
    return [
        str(c.get("chunk_id"))
        for c in (plan.get("chunks") or [])
        if c.get("chunk_id")
    ]


def chunk_worker_pass_status(run_root: Path, chunk_id: str) -> Optional[str]:
    """Return PASS / PASS_WITH_WARNING when a real worker result exists, else None."""
    worker_path = run_root / "worker_results" / chunk_id / "worker_result.json"
    if not existing_real_worker_result(worker_path):
        return None
    try:
        status = str(load_json(worker_path).get("status") or "")
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if status in PASS_LIKE:
        return status
    return None


def plan_remaining_worker_chunks(
    run_root: Path,
    *,
    force: bool,
) -> Dict[str, Any]:
    """Classify planned chunks into reuse (PASS) vs still to execute."""
    planned = chunk_ids_from_worker_plan(run_root)
    preexisting: List[str] = []
    to_execute: List[str] = []
    for chunk_id in planned:
        if chunk_worker_pass_status(run_root, chunk_id) and not force:
            preexisting.append(chunk_id)
        else:
            to_execute.append(chunk_id)
    return {
        "planned_chunk_ids": planned,
        "planned_chunk_count": len(planned),
        "preexisting_pass_chunks": preexisting,
        "chunks_to_execute": to_execute,
        "chunks_to_skip_reuse": preexisting,
    }


def execute_worker_chunk(
    *,
    repo_root: Path,
    run_root: Path,
    chunk_id: str,
    solver_python: str,
    solver_venv: str,
    force: bool,
    env_b: Dict[str, str],
    env_probe_body: Optional[Dict[str, Any]] = None,
    run_env_probe: bool = True,
    worker_id: str = "W0",
    mode: str = "m4_4_1b_2_worker_minibatch",
    minibatch_id: Optional[str] = None,
    label_prefix: str = "worker",
) -> Dict[str, Any]:
    """Execute or reuse one chunk. Returns per-chunk result record."""
    t0 = time.perf_counter()
    chunk_dir = run_root / "worker_results" / chunk_id
    log_path = chunk_dir / "log.txt"

    chunk_errors = validate_chunk_preconditions(
        run_root=run_root, chunk_id=chunk_id, force=force
    )
    hard = [e for e in chunk_errors if "use --force" not in e]
    if hard:
        return {
            "chunk_id": chunk_id,
            "action": "failed_precheck",
            "status": "FAIL",
            "errors": hard,
            "wall_seconds": time.perf_counter() - t0,
        }

    plan = build_chunk_plan(
        repo_root=repo_root,
        run_root=run_root,
        chunk_id=chunk_id,
        solver_python=solver_python,
        force=force,
    )

    if plan["skip_solve"]:
        worker = load_json(chunk_dir / "worker_result.json")
        summary = worker_result_summary(worker)
        return {
            "chunk_id": chunk_id,
            "action": "skipped_reuse",
            "status": str(summary["status"] or "PASS"),
            "wall_seconds": time.perf_counter() - t0,
            **summary,
        }

    append_log(log_path, f"[{utc_now()}] worker chunk execute {chunk_id} mode={mode}\n")

    if run_env_probe:
        env_pass, env_body = run_solver_env_probe(
            repo_root=repo_root,
            solver_python=solver_python,
            solver_venv=solver_venv,
            env_b=env_b,
            chunk_dir=chunk_dir,
        )
    else:
        env_body = env_probe_body or {}
        env_pass = bool((env_body.get("stage_b_probe") or {}).get("ok"))
        write_json_atomic(chunk_dir / "env_probe.json", env_body)

    if not env_pass:
        append_log(log_path, "env_probe FAIL\n")
        return {
            "chunk_id": chunk_id,
            "action": "executed",
            "status": "FAIL",
            "errors": ["env_probe_failed"],
            "wall_seconds": time.perf_counter() - t0,
        }

    append_log(log_path, "env_probe PASS\n")
    rc = _run_subprocess(
        plan["argv_solve"],
        env=env_b,
        cwd=repo_root,
        log_path=log_path,
        label=f"{label_prefix}_{chunk_id}",
    )
    append_log(log_path, f"solve exit_code={rc}\n")

    solver_result = None
    solver_path = chunk_dir / "solver_result.json"
    if solver_path.is_file():
        solver_result = load_json(solver_path)

    worker_status, warnings, errors = derive_worker_status(
        solve_rc=rc, solver_result=solver_result
    )
    worker = augment_worker_result(
        repo_root=repo_root,
        chunk_dir=chunk_dir,
        chunk_id=chunk_id,
        worker_status=worker_status,
        warnings=warnings,
        errors=errors,
        solve_rc=rc,
        env_probe=env_body,
        worker_id=worker_id,
        mode=mode,
        minibatch_id=minibatch_id,
    )
    summary = worker_result_summary(worker)
    return {
        "chunk_id": chunk_id,
        "action": "executed",
        "solve_exit_code": rc,
        "wall_seconds": time.perf_counter() - t0,
        **summary,
    }
