#!/usr/bin/env python3
"""M3.3 single-sample timing-only execution orchestrator (Stage A + B; Stage C skipped)."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PHYSICS_ROOT = SCRIPT_DIR.parent
PIPELINE_RUNS = PHYSICS_ROOT / "pipeline_runs"
MANIFEST_SCHEMA = "b3_pipeline_run_manifest_v1"
PLAN_SCHEMA = "b3_m3_orchestrator_run_one_plan_v1"

DEFAULT_PROD_PYTHON = "/home/vboxuser/final-project/.venv/bin/python"
DEFAULT_SOLVER_PYTHON = "/home/vboxuser/solver-mkl/venv/bin/python"
DEFAULT_SOLVER_VENV = "/home/vboxuser/solver-mkl/venv"
SOLVER_MKL_VENV = DEFAULT_SOLVER_VENV

PETSC_DIR_PROD = "/usr/lib/petscdir/petsc3.15/x86_64-linux-gnu-real"
SLEPC_DIR_PROD = "/usr/lib/slepcdir/slepc3.15/x86_64-linux-gnu-real"

STAGE_A_ENV_UNSET: Tuple[str, ...] = ()
STAGE_B_ENV_UNSET: Tuple[str, ...] = ("PYTHONPATH", "PETSC_DIR", "SLEPC_DIR", "PYTHONHOME")

STAGE_A_ENV_PROBE = """
import sys
import petsc4py
print(sys.executable)
print(petsc4py.__file__)
import dolfinx  # noqa: F401
import mpi4py  # noqa: F401
print("ok")
""".strip()

STAGE_B_ENV_PROBE = """
import petsc4py, slepc4py, sys
print(sys.executable)
print(petsc4py.__file__)
print(slepc4py.__file__)
try:
    import dolfinx
    raise SystemExit("unexpected dolfinx importable")
except ModuleNotFoundError:
    pass
""".strip()

SCRIPTS_REL = Path("FEM/experiments/active_domain_validation/physics_integrity/scripts")
STAGE_A_SCRIPT = "v2_b3_checkpoint_export.py"
STAGE_B_SCRIPT = "v2_b3_checkpoint_solve.py"

MESH_CONVERGENCE_MANIFEST = (
    "FEM/experiments/active_domain_validation/physics_integrity/configs/v2_mesh_convergence_manifest.json"
)

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_m3_orchestrator_dry_run import (  # noqa: E402
    _build_paths,
    _cmd_stage_a,
    _cmd_stage_b,
    _detect_repo_root,
    _format_path,
    _intended_manifest_path,
    _resolved_core_config,
    _stage_plan,
)
from v2_b3_petsc_util import write_json_atomic  # noqa: E402


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read_run_spec(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"run spec must be a JSON object: {path}")
    return data


def _solver_venv_root(solver_python: str, *, solver_venv: Optional[str] = None) -> Path:
    """Return venv root without resolving python (venv/bin/python is often a symlink)."""
    if solver_venv:
        return Path(solver_venv).expanduser()
    env_override = os.environ.get("SOLVER_MKL_VENV", "").strip()
    if env_override:
        return Path(env_override).expanduser()
    p = Path(solver_python).expanduser()
    if p.name in ("python", "python3") and p.parent.name == "bin":
        return p.parent.parent
    return Path(SOLVER_MKL_VENV)


def _solver_venv_bin(solver_python: str, *, solver_venv: Optional[str] = None) -> Path:
    return _solver_venv_root(solver_python, solver_venv=solver_venv) / "bin"


def _prod_subprocess_env(*, base: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    env = dict(os.environ if base is None else base)
    py_parts = [
        f"{PETSC_DIR_PROD}/lib/python3/dist-packages",
        f"{SLEPC_DIR_PROD}/lib/python3/dist-packages",
        "/usr/lib/python3/dist-packages",
    ]
    existing = env.get("PYTHONPATH", "")
    if existing:
        py_parts.append(existing)
    env["PETSC_DIR"] = PETSC_DIR_PROD
    env["SLEPC_DIR"] = SLEPC_DIR_PROD
    env["PYTHONPATH"] = ":".join(py_parts)
    return env


def _solver_mkl_subprocess_env(
    *,
    solver_python: str,
    solver_venv: Optional[str] = None,
    base: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Isolated solver-mkl env: strip production PETSc PYTHONPATH contamination."""
    env = dict(os.environ if base is None else base)
    for key in STAGE_B_ENV_UNSET:
        env.pop(key, None)
    venv_root = _solver_venv_root(solver_python, solver_venv=solver_venv)
    venv_bin = _solver_venv_bin(solver_python, solver_venv=solver_venv)
    env["VIRTUAL_ENV"] = str(venv_root)
    path_existing = env.get("PATH", "")
    env["PATH"] = f"{venv_bin}:{path_existing}" if path_existing else str(venv_bin)
    return env


def _stage_env_preview_m33(
    *, prod_python: str, solver_python: str, solver_venv: Optional[str] = None
) -> Dict[str, Any]:
    venv_root = str(_solver_venv_root(solver_python, solver_venv=solver_venv))
    venv_bin = str(_solver_venv_bin(solver_python, solver_venv=solver_venv))
    py_path = ":".join(
        [
            f"{PETSC_DIR_PROD}/lib/python3/dist-packages",
            f"{SLEPC_DIR_PROD}/lib/python3/dist-packages",
            "/usr/lib/python3/dist-packages",
            "<existing PYTHONPATH if any>",
        ]
    )
    return {
        "stage_a": {
            "profile": "production_venv",
            "python": prod_python,
            "env_vars": {
                "PETSC_DIR": PETSC_DIR_PROD,
                "SLEPC_DIR": SLEPC_DIR_PROD,
                "PYTHONPATH": py_path,
            },
            "unset_env_vars": list(STAGE_A_ENV_UNSET),
        },
        "stage_b": {
            "profile": "solver_mkl_isolated",
            "python": solver_python,
            "env_vars": {
                "VIRTUAL_ENV": venv_root,
                "PATH": f"{venv_bin}:<existing PATH>",
            },
            "unset_env_vars": list(STAGE_B_ENV_UNSET),
            "isolated_from_stage_a": True,
            "note": "PYTHONPATH/PETSC_DIR/SLEPC_DIR/PYTHONHOME removed before Stage B",
        },
    }


def _run_env_probe(
    *,
    python: str,
    script: str,
    env: Dict[str, str],
    cwd: Path,
) -> Tuple[int, str]:
    proc = subprocess.run(
        [python, "-c", script],
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return int(proc.returncode), proc.stdout or ""


def _verify_stage_a_env_probe(output: str, prod_python: str) -> Tuple[bool, str]:
    lines = [ln.strip() for ln in output.strip().splitlines() if ln.strip()]
    if len(lines) < 3:
        return False, f"probe_output_incomplete:{output!r}"
    exe, petsc_file, last = lines[0], lines[1], lines[-1]
    try:
        if Path(exe).resolve() != Path(prod_python).resolve():
            return False, f"executable_mismatch:{exe}"
    except OSError as exc:
        return False, f"executable_resolve_error:{exc}"
    petsc_norm = petsc_file.replace("\\", "/")
    if PETSC_DIR_PROD not in petsc_norm and "/usr/lib/petscdir/" not in petsc_norm:
        return False, f"petsc4py_not_system_petsc:{petsc_file}"
    if last != "ok":
        return False, f"probe_missing_ok_marker:{last!r}"
    return True, "ok"


def _verify_stage_b_env_probe(
    output: str, solver_python: str, *, solver_venv: Optional[str] = None
) -> Tuple[bool, str]:
    lines = [ln.strip() for ln in output.strip().splitlines() if ln.strip()]
    if len(lines) < 3:
        return False, f"probe_output_incomplete:{output!r}"
    exe, petsc_file, slepc_file = lines[0], lines[1], lines[2]
    venv_root = _solver_venv_root(solver_python, solver_venv=solver_venv)
    venv_marker = str(venv_root).replace("\\", "/")
    exe_path = Path(exe).expanduser()
    expected_python = Path(solver_python).expanduser()
    if exe_path != expected_python:
        try:
            if exe_path.resolve() != expected_python.resolve():
                exe_posix = exe_path.as_posix()
                if not exe_posix.startswith((venv_root / "bin").as_posix()):
                    return False, f"executable_mismatch:{exe}"
        except OSError as exc:
            return False, f"executable_resolve_error:{exc}"
    petsc_norm = petsc_file.replace("\\", "/")
    slepc_norm = slepc_file.replace("\\", "/")
    if venv_marker not in petsc_norm:
        return False, f"petsc4py_not_in_solver_venv:{petsc_file}"
    if venv_marker not in slepc_norm:
        return False, f"slepc4py_not_in_solver_venv:{slepc_file}"
    if "unexpected dolfinx importable" in output:
        return False, "dolfinx_importable_in_solver_env"
    return True, "ok"


def _resolve_paths_abs(
    repo_root: Path,
    *,
    mesh_level: str,
    target_set: str,
    run_id: str,
) -> Dict[str, Path]:
    rel = _build_paths(
        repo_root,
        mesh_level=mesh_level,
        target_set=target_set,
        run_id=run_id,
        absolute_paths=False,
    )
    out: Dict[str, Path] = {}
    for key in ("checkpoint_dir", "solve_dir", "rich_modal_dir", "synthesis_dir"):
        p = Path(rel[key])
        out[key] = (repo_root / p).resolve() if not p.is_absolute() else p.resolve()
    return out


def _argv_stage_a(
    *,
    prod_python: Path,
    script: Path,
    mesh_level: str,
    core_config: Path,
    checkpoint_dir: Path,
) -> List[str]:
    return [
        str(prod_python),
        str(script),
        "--mesh-level",
        mesh_level,
        "--B3-block-compose-backend",
        "csr_bulk",
        "--B3-synthesis-region-dofs",
        "off",
        "--core-config",
        str(core_config),
        "--output-dir",
        str(checkpoint_dir),
    ]


def _argv_stage_b(
    *,
    solver_python: Path,
    script: Path,
    checkpoint_dir: Path,
    solve_dir: Path,
    target_set: str,
) -> List[str]:
    return [
        str(solver_python),
        str(script),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--factor-solver",
        "mkl_pardiso",
        "--target-set",
        target_set,
        "--output-dir",
        str(solve_dir),
    ]


def _preflight_m33(
    *,
    repo_root: Path,
    spec: Dict[str, Any],
    paths_abs: Dict[str, Path],
    resolved_path: Optional[Path],
    prod_python: Path,
    solver_python: Path,
    manifest_path: Path,
    for_execution: bool,
) -> Dict[str, Any]:
    checks: Dict[str, Any] = {}
    warnings: List[str] = []
    blockers: List[str] = []

    sample_id = str(spec.get("sample_id") or "").strip()
    run_id = str(spec.get("run_id") or "").strip()
    mode = str(spec.get("mode") or "").strip().lower()

    if not sample_id:
        blockers.append("missing_sample_id")
    if not run_id:
        blockers.append("missing_run_id")

    if mode != "timing":
        blockers.append(f"mode_not_timing:{mode or 'missing'}")

    rich_req = bool(spec.get("rich_requested"))
    syn_req = bool(spec.get("synthesis_requested"))
    c_req = bool(spec.get("stage_c_requested"))
    checks["policy_rich_requested"] = rich_req
    checks["policy_synthesis_requested"] = syn_req
    checks["policy_stage_c_requested"] = c_req

    if rich_req:
        blockers.append("rich_requested_must_be_false_for_m3_3_timing")
    if syn_req:
        blockers.append("synthesis_requested_must_be_false_for_m3_3_timing")
    if c_req:
        blockers.append("stage_c_requested_must_be_false_for_m3_3_timing")

    checks["stage_c_skipped"] = True

    for name, script in (
        ("stage_a_script", STAGE_A_SCRIPT),
        ("stage_b_script", STAGE_B_SCRIPT),
    ):
        p = repo_root / SCRIPTS_REL / script
        checks[f"{name}_exists"] = p.is_file()
        if not p.is_file():
            blockers.append(f"missing_script:{script}")

    checks["prod_python_exists"] = prod_python.is_file()
    checks["solver_python_exists"] = solver_python.is_file()
    if not prod_python.is_file():
        if for_execution:
            blockers.append(f"prod_python_missing:{prod_python}")
        else:
            warnings.append("prod_python_not_found_on_host:vm_path_expected")
    if not solver_python.is_file():
        if for_execution:
            blockers.append(f"solver_python_missing:{solver_python}")
        else:
            warnings.append("solver_python_not_found_on_host:vm_path_expected")

    if resolved_path is None or not resolved_path.is_file():
        checks["resolved_core_config_exists"] = False
        blockers.append(
            f"missing_resolved_core_config:pipeline_runs/config_overlays/{sample_id}/resolved_core_config.json"
        )
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

    if for_execution:
        checks["runtime_manifest_exists"] = manifest_path.is_file()
        if manifest_path.is_file():
            blockers.append(f"runtime_manifest_exists:{manifest_path}")

    return {
        "checks": checks,
        "warnings": warnings,
        "blockers": blockers,
        "ready": len(blockers) == 0,
    }


def _build_plan(
    repo_root: Path,
    spec: Dict[str, Any],
    *,
    absolute_paths: bool,
    prod_python: str,
    solver_python: str,
    solver_venv: str,
    for_execution: bool,
) -> Dict[str, Any]:
    sample_id = str(spec["sample_id"]).strip()
    run_id = str(spec["run_id"]).strip()
    mesh_level = str(spec.get("mesh_level") or "L_prod")
    target_set = str(spec.get("target_set") or "full9")
    selection_reason = str(spec.get("selection_reason") or "unspecified")
    mode = "timing"

    paths_abs = _resolve_paths_abs(
        repo_root, mesh_level=mesh_level, target_set=target_set, run_id=run_id
    )
    core_config_str, core_config_path = _resolved_core_config(
        repo_root, sample_id, absolute_paths=absolute_paths
    )
    manifest_path = (
        repo_root
        / "FEM"
        / "experiments"
        / "active_domain_validation"
        / "physics_integrity"
        / "pipeline_runs"
        / "manifests"
        / f"run_{run_id}.json"
    )

    preflight = _preflight_m33(
        repo_root=repo_root,
        spec=spec,
        paths_abs=paths_abs,
        resolved_path=core_config_path if core_config_str else None,
        prod_python=Path(prod_python),
        solver_python=Path(solver_python),
        manifest_path=manifest_path,
        for_execution=for_execution,
    )

    paths_str = {
        k: _format_path(paths_abs[k], repo_root=repo_root, absolute_paths=absolute_paths)
        for k in ("checkpoint_dir", "solve_dir", "rich_modal_dir", "synthesis_dir")
    }

    commands: Dict[str, Optional[str]] = {"stage_a": None, "stage_b": None, "stage_c": None}
    argv: Dict[str, Optional[List[str]]] = {"stage_a": None, "stage_b": None}

    if core_config_str and preflight.get("ready"):
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
            rich=False,
        )
        if core_config_path and core_config_path.is_file():
            argv["stage_a"] = _argv_stage_a(
                prod_python=Path(prod_python),
                script=repo_root / SCRIPTS_REL / STAGE_A_SCRIPT,
                mesh_level=mesh_level,
                core_config=core_config_path,
                checkpoint_dir=paths_abs["checkpoint_dir"],
            )
            argv["stage_b"] = _argv_stage_b(
                solver_python=Path(solver_python),
                script=repo_root / SCRIPTS_REL / STAGE_B_SCRIPT,
                checkpoint_dir=paths_abs["checkpoint_dir"],
                solve_dir=paths_abs["solve_dir"],
                target_set=target_set,
            )

    log_dir = PIPELINE_RUNS / "logs"
    logs = {
        "stage_a_env_probe": _format_path(
            log_dir / f"run_{run_id}_stageA_env_probe.log",
            repo_root=repo_root,
            absolute_paths=absolute_paths,
        ),
        "stage_a": _format_path(
            log_dir / f"run_{run_id}_stageA.log",
            repo_root=repo_root,
            absolute_paths=absolute_paths,
        ),
        "stage_b_env_probe": _format_path(
            log_dir / f"run_{run_id}_stageB_env_probe.log",
            repo_root=repo_root,
            absolute_paths=absolute_paths,
        ),
        "stage_b": _format_path(
            log_dir / f"run_{run_id}_stageB.log",
            repo_root=repo_root,
            absolute_paths=absolute_paths,
        ),
    }

    return {
        "schema": PLAN_SCHEMA,
        "generated_utc": _utc_now(),
        "will_execute": False,
        "dry_run": not for_execution,
        "repo_root": "." if not absolute_paths else str(repo_root),
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
        "stage_c_note": "SKIPPED — timing-only M3.3; Stage C not executed",
        "commands": commands,
        "argv": argv,
        "stage_env": _stage_env_preview_m33(
            prod_python=prod_python,
            solver_python=solver_python,
            solver_venv=solver_venv,
        ),
        "solver_venv": solver_venv,
        "env_probes": {
            "stage_a": "runs before Stage A when executing (production env)",
            "stage_b": "runs before Stage B when executing (isolated solver-mkl env)",
        },
        "predicted_output_paths": paths_str,
        "logs": logs,
        "preflight": preflight,
        "warnings": list(preflight.get("warnings") or []),
        "blockers": list(preflight.get("blockers") or []),
        "paths_abs": {k: str(v) for k, v in paths_abs.items()},
        "manifest_path": str(manifest_path.resolve()),
        "core_config_path": str(core_config_path.resolve()) if core_config_path else None,
    }


def _build_initial_manifest(
    plan: Dict[str, Any],
    *,
    repo_root: Path,
    prod_python: str,
    solver_python: str,
) -> Dict[str, Any]:
    run_id = plan["run_id"]
    sample_id = plan["sample_id"]
    paths = plan["predicted_output_paths"]
    ckpt = paths["checkpoint_dir"]
    solve = paths["solve_dir"]
    core = plan["resolved_core_config"]

    return {
        "schema": MANIFEST_SCHEMA,
        "run_id": run_id,
        "sample_id": sample_id,
        "created_utc": _utc_now(),
        "updated_utc": _utc_now(),
        "terminal_status": "RUNNING",
        "source": {
            "mesh_level": plan["mesh_level"],
            "mesh_convergence_manifest": MESH_CONVERGENCE_MANIFEST,
            "resolved_core_config": core,
        },
        "policy": {
            "mode": "timing",
            "rich_export": False,
            "synthesis_requested": False,
            "stage_c_requested": False,
            "selection_reason": plan["selection_reason"],
        },
        "stages": {
            "A": {
                "status": "PENDING",
                "script": f"{SCRIPTS_REL.as_posix()}/{STAGE_A_SCRIPT}",
                "command": plan["commands"]["stage_a"],
                "checkpoint_dir": ckpt,
                "export_manifest": f"{ckpt.rstrip('/')}/checkpoint_export_manifest.json",
                "started_utc": None,
                "finished_utc": None,
            },
            "B": {
                "status": "PENDING",
                "script": f"{SCRIPTS_REL.as_posix()}/{STAGE_B_SCRIPT}",
                "command": plan["commands"]["stage_b"],
                "solve_dir": solve,
                "result_json": f"{solve.rstrip('/')}/result.json",
                "rich_modal_requested": False,
                "rich_modal_dir": None,
                "started_utc": None,
                "finished_utc": None,
            },
            "C": {
                "status": "SKIPPED",
                "script": f"{SCRIPTS_REL.as_posix()}/v2_b3_rich_modal_post.py",
                "command": None,
                "synthesis_dir": None,
                "modes_synthesis_json": None,
                "note": "timing-only M3.3 — not executed",
            },
        },
        "environment": plan.get("stage_env")
        or _stage_env_preview_m33(
            prod_python=prod_python,
            solver_python=solver_python,
            solver_venv=plan.get("solver_venv"),
        ),
        "orchestrator": {
            "schema": "b3_m3_orchestrator_run_one_v1",
            "milestone": "M3.3",
            "timing_only": True,
        },
        "logs": plan.get("logs"),
    }


def _append_index(repo_root: Path, manifest: Dict[str, Any]) -> None:
    index_path = PIPELINE_RUNS / "index" / "runs_index.jsonl"
    row = {
        "run_id": manifest["run_id"],
        "sample_id": manifest.get("sample_id"),
        "created_utc": manifest["created_utc"],
        "updated_utc": manifest.get("updated_utc"),
        "terminal_status": manifest.get("terminal_status"),
        "mode": manifest["policy"]["mode"],
        "selection_reason": manifest["policy"]["selection_reason"],
        "stage_a_status": manifest["stages"]["A"]["status"],
        "stage_b_status": manifest["stages"]["B"]["status"],
        "stage_c_status": manifest["stages"]["C"]["status"],
        "checkpoint_dir": manifest["stages"]["A"]["checkpoint_dir"],
        "solve_dir": manifest["stages"]["B"]["solve_dir"],
        "synthesis_dir": manifest["stages"]["C"].get("synthesis_dir"),
        "rich_modal_requested": manifest["stages"]["B"]["rich_modal_requested"],
    }
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


def _run_subprocess(
    argv: List[str],
    *,
    env: Dict[str, str],
    cwd: Path,
    log_path: Path,
    label: str,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[B3_m3_run_one] {label} starting", flush=True)
    print(f"[B3_m3_run_one] log={log_path}", flush=True)
    proc = subprocess.run(
        argv,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    output = proc.stdout or ""
    log_path.write_text(output, encoding="utf-8")
    tail = "\n".join(output.strip().splitlines()[-8:])
    if tail:
        print(f"[B3_m3_run_one] {label} log tail:\n{tail}", flush=True)
    print(
        f"[B3_m3_run_one] {label} finished exit_code={proc.returncode}",
        flush=True,
    )
    return int(proc.returncode)


def _verify_stage_a_export(export_manifest_path: Path) -> Tuple[bool, str]:
    if not export_manifest_path.is_file():
        return False, f"missing_export_manifest:{export_manifest_path}"
    try:
        data = json.loads(export_manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return False, f"invalid_export_manifest_json:{exc}"
    if data.get("status") != "PASS":
        return False, f"status={data.get('status')!r}"
    if not data.get("export_pass"):
        return False, "export_pass=false"
    if not data.get("matrix_verify_pass"):
        return False, "matrix_verify_pass=false"
    if data.get("core_config_mode") != "override":
        return False, f"core_config_mode={data.get('core_config_mode')!r}"
    return True, "ok"


def _verify_stage_b_result(result_path: Path) -> Tuple[bool, str]:
    if not result_path.is_file():
        return False, f"missing_result_json:{result_path}"
    try:
        data = json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return False, f"invalid_result_json:{exc}"
    if data.get("status") != "PASS":
        return False, f"status={data.get('status')!r}"
    rich = data.get("rich_modal_export") or {}
    if rich.get("requested"):
        return False, "rich_modal_export.requested=true"
    targets = data.get("targets") or []
    if not targets:
        return False, "no_targets_in_result"
    for row in targets:
        if row.get("status") != "PASS":
            hz = row.get("target_hz")
            return False, f"target_not_pass:hz={hz}:status={row.get('status')!r}"
    return True, "ok"


def _finalize_manifest(
    manifest_path: Path,
    manifest: Dict[str, Any],
    *,
    terminal_status: str,
    append_index: bool,
    repo_root: Path,
) -> None:
    manifest["terminal_status"] = terminal_status
    manifest["updated_utc"] = _utc_now()
    write_json_atomic(manifest_path, manifest)
    if append_index:
        _append_index(repo_root, manifest)


def _execute_run(
    repo_root: Path,
    plan: Dict[str, Any],
    *,
    prod_python: str,
    solver_python: str,
    append_index: bool,
) -> int:
    if not plan["preflight"].get("ready"):
        print("[B3_m3_run_one] preflight not ready; aborting", flush=True)
        return 2

    manifest_path = Path(plan["manifest_path"])
    manifest = _build_initial_manifest(
        plan, repo_root=repo_root, prod_python=prod_python, solver_python=solver_python
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(manifest_path, manifest)
    print(f"[B3_m3_run_one] wrote runtime manifest {manifest_path}", flush=True)

    argv_a = plan["argv"]["stage_a"]
    argv_b = plan["argv"]["stage_b"]
    if not argv_a or not argv_b:
        print("[B3_m3_run_one] missing stage argv; aborting", flush=True)
        return 2

    paths_abs = {k: Path(v) for k, v in plan["paths_abs"].items()}
    run_id = plan["run_id"]
    log_a_probe = PIPELINE_RUNS / "logs" / f"run_{run_id}_stageA_env_probe.log"
    log_a = PIPELINE_RUNS / "logs" / f"run_{run_id}_stageA.log"
    log_b_probe = PIPELINE_RUNS / "logs" / f"run_{run_id}_stageB_env_probe.log"
    log_b = PIPELINE_RUNS / "logs" / f"run_{run_id}_stageB.log"

    env_a = _prod_subprocess_env()
    solver_venv = plan.get("solver_venv") or SOLVER_MKL_VENV
    env_b = _solver_mkl_subprocess_env(solver_python=solver_python, solver_venv=solver_venv)

    # Stage A env probe
    print("[B3_m3_run_one] Stage A env probe", flush=True)
    rc_a_probe, out_a_probe = _run_env_probe(
        python=prod_python,
        script=STAGE_A_ENV_PROBE,
        env=env_a,
        cwd=repo_root,
    )
    log_a_probe.parent.mkdir(parents=True, exist_ok=True)
    log_a_probe.write_text(out_a_probe, encoding="utf-8")
    ok_a_probe, detail_a_probe = _verify_stage_a_env_probe(out_a_probe, prod_python)
    manifest["stages"]["A"]["env_preflight"] = {
        "ok": ok_a_probe and rc_a_probe == 0,
        "exit_code": rc_a_probe,
        "detail": detail_a_probe,
        "log": str(log_a_probe),
    }
    if rc_a_probe != 0 or not ok_a_probe:
        manifest["stages"]["A"]["status"] = "FAIL"
        reason = f"stage_a_env_preflight_failed:rc={rc_a_probe}:verify={detail_a_probe}"
        manifest["failure_reason"] = reason
        _finalize_manifest(
            manifest_path,
            manifest,
            terminal_status="FAIL",
            append_index=append_index,
            repo_root=repo_root,
        )
        print(f"[B3_m3_run_one] FAIL {reason}", flush=True)
        return 1

    # Stage A
    manifest["stages"]["A"]["started_utc"] = _utc_now()
    manifest["updated_utc"] = _utc_now()
    write_json_atomic(manifest_path, manifest)

    rc_a = _run_subprocess(
        argv_a,
        env=env_a,
        cwd=repo_root,
        log_path=log_a,
        label="Stage A",
    )
    export_manifest = paths_abs["checkpoint_dir"] / "checkpoint_export_manifest.json"
    ok_a, detail_a = _verify_stage_a_export(export_manifest)
    manifest["stages"]["A"]["finished_utc"] = _utc_now()
    manifest["stages"]["A"]["export_manifest"] = str(export_manifest)
    manifest["stages"]["A"]["exit_code"] = rc_a
    manifest["stages"]["A"]["verify_detail"] = detail_a

    if rc_a != 0 or not ok_a:
        manifest["stages"]["A"]["status"] = "FAIL"
        reason = f"stage_a_failed:rc={rc_a}:verify={detail_a}"
        manifest["failure_reason"] = reason
        _finalize_manifest(
            manifest_path,
            manifest,
            terminal_status="FAIL",
            append_index=append_index,
            repo_root=repo_root,
        )
        print(f"[B3_m3_run_one] FAIL {reason}", flush=True)
        return 1

    manifest["stages"]["A"]["status"] = "PASS"
    manifest["updated_utc"] = _utc_now()
    write_json_atomic(manifest_path, manifest)
    print("[B3_m3_run_one] Stage A PASS", flush=True)

    # Stage B env probe (isolated solver-mkl env; must not inherit Stage A PYTHONPATH)
    print("[B3_m3_run_one] Stage B env probe", flush=True)
    rc_b_probe, out_b_probe = _run_env_probe(
        python=solver_python,
        script=STAGE_B_ENV_PROBE,
        env=env_b,
        cwd=repo_root,
    )
    log_b_probe.parent.mkdir(parents=True, exist_ok=True)
    log_b_probe.write_text(out_b_probe, encoding="utf-8")
    ok_b_probe, detail_b_probe = _verify_stage_b_env_probe(
        out_b_probe, solver_python, solver_venv=solver_venv
    )
    manifest["stages"]["B"]["env_preflight"] = {
        "ok": ok_b_probe and rc_b_probe == 0,
        "exit_code": rc_b_probe,
        "detail": detail_b_probe,
        "log": str(log_b_probe),
    }
    manifest["updated_utc"] = _utc_now()
    write_json_atomic(manifest_path, manifest)

    if rc_b_probe != 0 or not ok_b_probe:
        manifest["stages"]["B"]["status"] = "FAIL"
        reason = f"stage_b_env_preflight_failed:rc={rc_b_probe}:verify={detail_b_probe}"
        manifest["failure_reason"] = reason
        _finalize_manifest(
            manifest_path,
            manifest,
            terminal_status="FAIL",
            append_index=append_index,
            repo_root=repo_root,
        )
        print(f"[B3_m3_run_one] FAIL {reason}", flush=True)
        return 1

    # Stage B
    manifest["stages"]["B"]["started_utc"] = _utc_now()
    manifest["updated_utc"] = _utc_now()
    write_json_atomic(manifest_path, manifest)

    rc_b = _run_subprocess(
        argv_b,
        env=env_b,
        cwd=repo_root,
        log_path=log_b,
        label="Stage B",
    )
    result_json = paths_abs["solve_dir"] / "result.json"
    ok_b, detail_b = _verify_stage_b_result(result_json)
    manifest["stages"]["B"]["finished_utc"] = _utc_now()
    manifest["stages"]["B"]["result_json"] = str(result_json)
    manifest["stages"]["B"]["exit_code"] = rc_b
    manifest["stages"]["B"]["verify_detail"] = detail_b

    if rc_b != 0 or not ok_b:
        manifest["stages"]["B"]["status"] = "FAIL"
        reason = f"stage_b_failed:rc={rc_b}:verify={detail_b}"
        manifest["failure_reason"] = reason
        _finalize_manifest(
            manifest_path,
            manifest,
            terminal_status="FAIL",
            append_index=append_index,
            repo_root=repo_root,
        )
        print(f"[B3_m3_run_one] FAIL {reason}", flush=True)
        return 1

    manifest["stages"]["B"]["status"] = "PASS"
    manifest["stages"]["C"]["status"] = "SKIPPED"
    _finalize_manifest(
        manifest_path,
        manifest,
        terminal_status="PASS",
        append_index=append_index,
        repo_root=repo_root,
    )
    print("[B3_m3_run_one] terminal PASS (Stage C SKIPPED)", flush=True)
    return 0


def run_one(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="M3.3 single-sample timing orchestrator (Stage A+B only)."
    )
    parser.add_argument("--run-spec", required=True, help="JSON run spec (single object)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print execution plan only; no subprocess, no runtime manifest.",
    )
    parser.add_argument(
        "--no-index",
        action="store_true",
        help="Do not append pipeline_runs/index/runs_index.jsonl after terminal state.",
    )
    parser.add_argument("--repo-root", help="Optional repo root (default: auto-detect)")
    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        help="Emit absolute paths in plan output (default: repo-relative).",
    )
    parser.add_argument("--prod-python", default=DEFAULT_PROD_PYTHON)
    parser.add_argument("--solver-python", default=DEFAULT_SOLVER_PYTHON)
    parser.add_argument(
        "--solver-venv",
        default=SOLVER_MKL_VENV,
        help="solver-mkl virtualenv root (default: /home/vboxuser/solver-mkl/venv)",
    )
    args = parser.parse_args(argv)

    repo_root = (
        Path(args.repo_root).expanduser().resolve()
        if args.repo_root
        else _detect_repo_root(SCRIPT_DIR)
    )
    spec_path = Path(args.run_spec).expanduser()
    if not spec_path.is_absolute():
        spec_path = (repo_root / spec_path).resolve()
    if not spec_path.is_file():
        raise SystemExit(f"run spec not found: {spec_path}")

    spec = _read_run_spec(spec_path)
    plan = _build_plan(
        repo_root,
        spec,
        absolute_paths=bool(args.absolute_paths),
        prod_python=str(args.prod_python),
        solver_python=str(args.solver_python),
        solver_venv=str(args.solver_venv),
        for_execution=not args.dry_run,
    )
    plan["will_execute"] = not args.dry_run and plan["preflight"].get("ready", False)
    plan["run_spec"] = _format_path(
        spec_path, repo_root=repo_root, absolute_paths=bool(args.absolute_paths)
    )

    if args.dry_run:
        plan.pop("paths_abs", None)
        plan.pop("manifest_path", None)
        plan.pop("core_config_path", None)
        print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
        ready = bool(plan["preflight"].get("ready"))
        print(
            f"[B3_m3_run_one] dry_run will_execute=false ready={ready} "
            f"blockers={len(plan.get('blockers') or [])}",
            flush=True,
        )
        return 0 if ready else 2

    return _execute_run(
        repo_root,
        plan,
        prod_python=str(args.prod_python),
        solver_python=str(args.solver_python),
        append_index=not args.no_index,
    )


def main() -> int:
    print(
        "LEGACY (M3.3): for production LHS use "
        "v2_b3_m4_lhs_production_batch.py or v2_b3_m4_run_one_sample.py --production-mode.",
        file=sys.stderr,
    )
    return run_one()


if __name__ == "__main__":
    raise SystemExit(main())
