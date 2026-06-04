#!/usr/bin/env python3
"""Overnight coarse-mesh (L_scout_coarse) LHS material scout batch: Stage A + discovery density + zone reports."""
from __future__ import annotations

import argparse
import copy
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PHYSICS_ROOT = SCRIPT_DIR.parent
PIPELINE_RUNS = PHYSICS_ROOT / "pipeline_runs"
REPORTS_ROOT = PIPELINE_RUNS / "scout_density_reports"
SUMMARY_DIR = REPORTS_ROOT / "summary"
CONFIG_OVERLAYS = PIPELINE_RUNS / "config_overlays"
CONV_DIAG = PHYSICS_ROOT / "v2_mesh_convergence" / "diagnostics"
CONV_MESH = PHYSICS_ROOT / "v2_mesh_convergence" / "mesh"
BASELINE_CORE = PHYSICS_ROOT / "configs" / "coupled_physical_core_v2.json"
DEFAULT_SAMPLES_JSONL = PIPELINE_RUNS / "specs" / "m3_4_coarse_scout_lhs_batch.jsonl"
DEFAULT_REFERENCE_STUB = PIPELINE_RUNS / "specs" / "scout_discovery_reference_stub.json"
MESH_CASE_ID = "baseline_coupled_v2"
RUN_SUFFIX = "m34"
BIN_WIDTH_HZ = 25.0

SCRIPTS_REL = Path("FEM/experiments/active_domain_validation/physics_integrity/scripts")
STAGE_A_SCRIPT = str(SCRIPTS_REL / "v2_b3_checkpoint_export.py")
STAGE_B_SCRIPT = str(SCRIPTS_REL / "v2_b3_checkpoint_target_density_experiment.py")

DEFAULT_PROD_PYTHON = "/home/vboxuser/final-project/.venv/bin/python"
DEFAULT_PROD_VENV = "/home/vboxuser/final-project/.venv"
DEFAULT_SOLVER_PYTHON = "/home/vboxuser/solver-mkl/venv/bin/python"
SOLVER_MKL_VENV = "/home/vboxuser/solver-mkl/venv"

PETSC_DIR_PROD = "/usr/lib/petscdir/petsc3.15/x86_64-linux-gnu-real"
SLEPC_DIR_PROD = "/usr/lib/slepcdir/slepc3.15/x86_64-linux-gnu-real"
STAGE_B_ENV_UNSET: Tuple[str, ...] = ("PYTHONPATH", "PETSC_DIR", "SLEPC_DIR", "PYTHONHOME")

STAGE_A_ENV_PROBE = """
import os
import sys
import petsc4py
print(sys.executable)
print(petsc4py.__file__)
print(os.environ.get("VIRTUAL_ENV", ""))
import dolfinx  # noqa: F401
import mpi4py  # noqa: F401
print("ok")
""".strip()

STAGE_B_ENV_PROBE = """
import os
import petsc4py
import slepc4py
import sys
print(sys.executable)
print(petsc4py.__file__)
print(slepc4py.__file__)
print(os.environ.get("VIRTUAL_ENV", ""))
try:
    import dolfinx
    raise SystemExit("unexpected dolfinx importable")
except ModuleNotFoundError:
    pass
print("ok")
""".strip()

# Minimal inherited keys for subprocess (avoid parent VIRTUAL_ENV / PYTHONPATH).
_SUBPROCESS_INHERIT_KEYS = ("HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "TMPDIR", "TERM")

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_m3_orchestrator_run_one import (  # noqa: E402
    DEFAULT_PROD_PYTHON as M3_PROD_PYTHON,
    DEFAULT_SOLVER_PYTHON as M3_SOLVER_PYTHON,
    _run_subprocess,
    _verify_stage_a_export,
)
from v2_b3_petsc_util import write_json_atomic  # noqa: E402


def _dedupe_frequencies_hz(freqs: List[float], *, tol_hz: float = 0.05) -> List[float]:
    out: List[float] = []
    for f in sorted(float(x) for x in freqs):
        if not out or abs(f - out[-1]) > tol_hz:
            out.append(f)
    return out

# Reuse pilot resolver merge helpers
from v2_b3_resolve_pilot_core_config import (  # noqa: E402
    _apply_material_delta,
    _build_changed_material_values,
    _repo_relative,
    _sha256_file,
    _sha256_json,
)


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _venv_root_from_python(python_exe: str, *, fallback: str) -> Path:
    p = Path(python_exe).expanduser()
    if p.name in ("python", "python3") and p.parent.name == "bin":
        return p.parent.parent.resolve()
    return Path(fallback).expanduser().resolve()


def _path_without_venv_bins(path_str: str, *, exclude_venv_roots: List[Path]) -> str:
    """Drop PATH entries under excluded venv roots (e.g. solver-mkl, production .venv)."""
    if not path_str:
        return ""
    exclude = [str(v.resolve()) for v in exclude_venv_roots]
    kept: List[str] = []
    for part in path_str.split(os.pathsep):
        if not part:
            continue
        part_res = str(Path(part).resolve())
        skip = False
        for root in exclude:
            if part_res == root or part_res.startswith(root + os.sep):
                skip = True
                break
        if not skip:
            kept.append(part)
    return os.pathsep.join(kept)


def _minimal_subprocess_base() -> Dict[str, str]:
    return {k: os.environ[k] for k in _SUBPROCESS_INHERIT_KEYS if k in os.environ}


def _prod_subprocess_env_strict(
    *,
    prod_python: str,
    prod_venv: str,
) -> Dict[str, str]:
    """Isolated production env; never inherit parent VIRTUAL_ENV (solver-mkl guard)."""
    venv_root = Path(prod_venv).expanduser().resolve()
    venv_bin = venv_root / "bin"
    prod_root = _venv_root_from_python(prod_python, fallback=prod_venv)
    solver_root = Path(SOLVER_MKL_VENV).expanduser().resolve()
    env = _minimal_subprocess_base()
    env["VIRTUAL_ENV"] = str(venv_root)
    base_path = _path_without_venv_bins(os.environ.get("PATH", ""), exclude_venv_roots=[solver_root, prod_root])
    env["PATH"] = f"{venv_bin}{os.pathsep}{base_path}" if base_path else str(venv_bin)
    env["PETSC_DIR"] = PETSC_DIR_PROD
    env["SLEPC_DIR"] = SLEPC_DIR_PROD
    env["PYTHONPATH"] = os.pathsep.join(
        [
            f"{PETSC_DIR_PROD}/lib/python3/dist-packages",
            f"{SLEPC_DIR_PROD}/lib/python3/dist-packages",
            "/usr/lib/python3/dist-packages",
        ]
    )
    env.pop("PYTHONHOME", None)
    return env


def _solver_mkl_subprocess_env_strict(
    *,
    solver_python: str,
    solver_venv: str,
) -> Dict[str, str]:
    """Isolated solver-mkl env; strip production PETSc PYTHONPATH contamination."""
    venv_root = Path(solver_venv).expanduser().resolve()
    venv_bin = venv_root / "bin"
    prod_root = Path(DEFAULT_PROD_VENV).expanduser().resolve()
    env = _minimal_subprocess_base()
    for key in STAGE_B_ENV_UNSET:
        env.pop(key, None)
    env["VIRTUAL_ENV"] = str(venv_root)
    base_path = _path_without_venv_bins(os.environ.get("PATH", ""), exclude_venv_roots=[venv_root, prod_root])
    env["PATH"] = f"{venv_bin}{os.pathsep}{base_path}" if base_path else str(venv_bin)
    env.pop("PYTHONHOME", None)
    return env


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


def _verify_stage_a_env_probe(
    output: str,
    *,
    prod_python: str,
    prod_venv: str,
) -> Tuple[bool, str]:
    lines = [ln.strip() for ln in output.strip().splitlines() if ln.strip()]
    if len(lines) < 4:
        return False, f"probe_output_incomplete:{output!r}"
    exe, petsc_file, virtual_env, last = lines[0], lines[1], lines[2], lines[-1]
    try:
        if Path(exe).resolve() != Path(prod_python).expanduser().resolve():
            return False, f"executable_mismatch:{exe}"
    except OSError as exc:
        return False, f"executable_resolve_error:{exc}"
    petsc_norm = petsc_file.replace("\\", "/")
    if PETSC_DIR_PROD not in petsc_norm and "/usr/lib/petscdir/" not in petsc_norm:
        return False, f"petsc4py_not_system_petsc:{petsc_file}"
    venv_norm = virtual_env.replace("\\", "/")
    prod_norm = str(Path(prod_venv).expanduser().resolve()).replace("\\", "/")
    if "solver-mkl" in venv_norm:
        return False, f"virtual_env_solver_mkl:{virtual_env}"
    if prod_norm not in venv_norm and venv_norm != prod_norm:
        return False, f"virtual_env_mismatch:{virtual_env}"
    if last != "ok":
        return False, f"probe_missing_ok_marker:{last!r}"
    return True, "ok"


def _verify_stage_b_env_probe(
    output: str,
    *,
    solver_python: str,
    solver_venv: str,
) -> Tuple[bool, str]:
    lines = [ln.strip() for ln in output.strip().splitlines() if ln.strip()]
    if len(lines) < 5:
        return False, f"probe_output_incomplete:{output!r}"
    exe, petsc_file, slepc_file, virtual_env, last = lines[0], lines[1], lines[2], lines[3], lines[-1]
    venv_marker = str(Path(solver_venv).expanduser().resolve()).replace("\\", "/")
    try:
        if Path(exe).resolve() != Path(solver_python).expanduser().resolve():
            return False, f"executable_mismatch:{exe}"
    except OSError as exc:
        return False, f"executable_resolve_error:{exc}"
    petsc_norm = petsc_file.replace("\\", "/")
    slepc_norm = slepc_file.replace("\\", "/")
    if venv_marker not in petsc_norm:
        return False, f"petsc4py_not_in_solver_venv:{petsc_file}"
    if venv_marker not in slepc_norm:
        return False, f"slepc4py_not_in_solver_venv:{slepc_file}"
    if "solver-mkl" not in virtual_env.replace("\\", "/"):
        return False, f"virtual_env_not_solver_mkl:{virtual_env}"
    if "unexpected dolfinx importable" in output:
        return False, "dolfinx_importable_in_solver_env"
    if last != "ok":
        return False, f"probe_missing_ok_marker:{last!r}"
    return True, "ok"


def _run_stage_env_probes(
    *,
    repo_root: Path,
    prod_python: str,
    prod_venv: str,
    solver_python: str,
    solver_venv: str,
    log_dir: Path,
) -> Tuple[bool, Dict[str, Any]]:
    """Probe Stage A and Stage B env before any sample work."""
    log_dir.mkdir(parents=True, exist_ok=True)
    env_a = _prod_subprocess_env_strict(prod_python=prod_python, prod_venv=prod_venv)
    env_b = _solver_mkl_subprocess_env_strict(solver_python=solver_python, solver_venv=solver_venv)

    rc_a, out_a = _run_env_probe(
        python=prod_python, script=STAGE_A_ENV_PROBE, env=env_a, cwd=repo_root
    )
    log_a = log_dir / "stage_a_env_probe.log"
    log_a.write_text(out_a, encoding="utf-8")
    ok_a, detail_a = _verify_stage_a_env_probe(out_a, prod_python=prod_python, prod_venv=prod_venv)

    rc_b, out_b = _run_env_probe(
        python=solver_python, script=STAGE_B_ENV_PROBE, env=env_b, cwd=repo_root
    )
    log_b = log_dir / "stage_b_env_probe.log"
    log_b.write_text(out_b, encoding="utf-8")
    ok_b, detail_b = _verify_stage_b_env_probe(
        out_b, solver_python=solver_python, solver_venv=solver_venv
    )

    payload = {
        "stage_a": {
            "ok": bool(rc_a == 0 and ok_a),
            "exit_code": rc_a,
            "detail": detail_a,
            "log": str(log_a),
            "VIRTUAL_ENV": env_a.get("VIRTUAL_ENV"),
            "PATH_head": (env_a.get("PATH") or "").split(os.pathsep)[:3],
        },
        "stage_b": {
            "ok": bool(rc_b == 0 and ok_b),
            "exit_code": rc_b,
            "detail": detail_b,
            "log": str(log_b),
            "VIRTUAL_ENV": env_b.get("VIRTUAL_ENV"),
            "PATH_head": (env_b.get("PATH") or "").split(os.pathsep)[:3],
        },
    }
    return bool(payload["stage_a"]["ok"] and payload["stage_b"]["ok"]), payload


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
                raise ValueError(f"{path}: line {i}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}: line {i} is not a JSON object")
            rows.append(row)
    return rows


def _scout_run_id(sample_id: str) -> str:
    return f"scout_{sample_id}_{RUN_SUFFIX}"


def _scout_mesh_file_rel(mesh_level: str, *, case_id: str = MESH_CASE_ID) -> str:
    return (
        Path("FEM")
        / "experiments"
        / "active_domain_validation"
        / "physics_integrity"
        / "v2_mesh_convergence"
        / "mesh"
        / mesh_level
        / f"{case_id}.msh"
    ).as_posix()


def _paths_for_sample(sample_id: str, mesh_level: str) -> Dict[str, Path]:
    run_id = _scout_run_id(sample_id)
    return {
        "overlay_dir": CONFIG_OVERLAYS / run_id,
        "resolved_core_config": CONFIG_OVERLAYS / run_id / "resolved_core_config.json",
        "checkpoint_dir": CONV_DIAG / f"st_worker_scaling_{mesh_level}_{run_id}",
        "stage_b_dir": CONV_DIAG
        / "solver_benchmarks"
        / f"target_density_discovery_60_550_step7p5_{run_id}",
        "report_dir": REPORTS_ROOT / run_id,
        "log_dir": PIPELINE_RUNS / "logs" / run_id,
    }


def _resolve_scout_sample_overlay(
    row: Dict[str, Any],
    *,
    mesh_level: str,
    repo_root: Path,
    overlay_dir: Path,
    force: bool,
) -> Dict[str, Any]:
    sample_id = str(row.get("sample_id") or "").strip()
    if not sample_id:
        raise ValueError("sample row missing sample_id")

    resolved_path = overlay_dir / "resolved_core_config.json"
    if resolved_path.is_file() and not force:
        resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
        mats = resolved.get("materials") or {}
        return {
            "sample_id": sample_id,
            "skipped_resolve": True,
            "resolved_config_path": str(resolved_path),
            "readiness_status": "REUSED",
            "top_density": float((mats.get("top") or {}).get("density", float("nan"))),
            "back_density": float((mats.get("back") or {}).get("density", float("nan"))),
            "mesh_file": (resolved.get("solver") or {}).get("mesh_file"),
            "errors": [],
        }

    overlay_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = BASELINE_CORE.resolve()
    baseline_cfg = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_sha = _sha256_file(baseline_path)

    payload = row.get("parameter_payload") or {}
    material_delta = payload.get("material_delta") or {}
    geometry_delta = payload.get("geometry_delta") or {}
    requires_mesh = bool(payload.get("requires_mesh_regeneration", False))

    resolved = copy.deepcopy(baseline_cfg)
    changed_fields = _apply_material_delta(resolved, material_delta if isinstance(material_delta, dict) else {})
    mesh_file = _scout_mesh_file_rel(mesh_level)
    solver = resolved.setdefault("solver", {})
    solver["mesh_file"] = mesh_file
    solver["clamp_ribs"] = False

    generated_utc = _utc_now()
    overlay_payload = {
        "schema": "b3_scout_lhs_overlay_applied_v1",
        "generated_utc": generated_utc,
        "run_id": _scout_run_id(sample_id),
        "sample_id": sample_id,
        "purpose": "scout_modal_density_only",
        "mesh_level": mesh_level,
        "mesh_case_id": MESH_CASE_ID,
        "mesh_file": mesh_file,
        "base_config_path": _repo_relative(baseline_path, repo_root=repo_root),
        "base_config_sha256": baseline_sha,
        "geometry_delta": geometry_delta,
        "material_delta": material_delta,
        "requires_mesh_regeneration": requires_mesh,
        "fields_changed": changed_fields,
    }
    overlay_payload["overlay_payload_sha256"] = _sha256_json(
        {k: v for k, v in overlay_payload.items() if k != "overlay_payload_sha256"}
    )

    errors: List[str] = []
    warnings: List[str] = []
    if geometry_delta:
        errors.append("geometry_delta must be empty for scout material-only batch")
    if requires_mesh:
        errors.append("requires_mesh_regeneration must be false (shared L_scout_coarse mesh)")
    if mesh_level not in mesh_file.replace("\\", "/"):
        errors.append(f"solver.mesh_file must reference {mesh_level}")
    if "L_prod" in mesh_file.replace("\\", "/"):
        errors.append("solver.mesh_file must not reference L_prod")
    if solver.get("clamp_ribs") is not False:
        errors.append("solver.clamp_ribs must be false")

    mesh_abs = (repo_root / mesh_file).resolve()
    mesh_exists = mesh_abs.is_file()
    if not mesh_exists:
        warnings.append(f"mesh file not found: {mesh_file}")

    mats = resolved.get("materials") or {}
    top_d = float((mats.get("top") or {}).get("density", float("nan")))
    back_d = float((mats.get("back") or {}).get("density", float("nan")))

    overlay_path = overlay_dir / "overlay_applied.json"
    readiness_path = overlay_dir / "readiness_check.json"

    write_json_atomic(resolved_path, resolved)
    write_json_atomic(overlay_path, overlay_payload)
    readiness = {
        "schema": "b3_scout_lhs_readiness_check_v1",
        "generated_utc": generated_utc,
        "run_id": _scout_run_id(sample_id),
        "sample_id": sample_id,
        "status": "PASS" if (not errors and mesh_exists) else ("PENDING_MESH" if not errors else "FAIL"),
        "mesh_level": mesh_level,
        "mesh_file": mesh_file,
        "mesh_file_exists": mesh_exists,
        "solver_clamp_ribs": False,
        "effective_materials": {"top.density": top_d, "back.density": back_d},
        "changed_material_values": _build_changed_material_values(
            baseline_cfg, resolved, material_delta if isinstance(material_delta, dict) else {}
        ),
        "lhs_perturbation_applied": bool(material_delta),
        "warnings": warnings,
        "errors": errors,
    }
    write_json_atomic(readiness_path, readiness)

    return {
        "sample_id": sample_id,
        "resolved_config_path": str(resolved_path),
        "readiness_status": readiness["status"],
        "top_density": top_d,
        "back_density": back_d,
        "mesh_file": mesh_file,
        "errors": errors,
    }


def _verify_density_result(path: Path) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    if not path.is_file():
        return False, f"missing:{path}", None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return False, f"invalid_json:{exc}", None
    status = str(data.get("status") or "")
    if status not in ("PASS", "PARTIAL"):
        return False, f"status={status!r}", data
    spacings = data.get("spacings") or []
    if not spacings:
        return False, "no_spacings", data
    total_modes = 0
    for row in spacings:
        total_modes += int(row.get("unique_accepted_count") or 0)
    if total_modes <= 0:
        return False, "no_unique_accepted_modes", data
    return True, "ok", data


def _extract_unique_frequencies(density_body: Dict[str, Any], *, dedupe_tol: float = 0.05) -> List[float]:
    freqs: List[float] = []
    for row in density_body.get("spacings") or []:
        freqs.extend(float(x) for x in (row.get("unique_accepted_frequencies_hz") or []))
    return _dedupe_frequencies_hz(freqs, tol_hz=dedupe_tol)


def _postprocess_sample_density(
    *,
    sample_id: str,
    run_id: str,
    density_body: Dict[str, Any],
    report_dir: Path,
    freq_min_hz: float,
    freq_max_hz: float,
    bin_width_hz: float,
) -> Dict[str, Any]:
    report_dir.mkdir(parents=True, exist_ok=True)
    unique_hz = _extract_unique_frequencies(density_body)
    bin_edges: List[float] = []
    lo = float(freq_min_hz)
    hi = float(freq_max_hz)
    w = float(bin_width_hz)
    cur = lo
    while cur < hi - 1e-9:
        bin_edges.append(cur)
        cur += w
    bin_edges.append(hi)

    bins: List[Dict[str, Any]] = []
    counts: List[int] = []
    for i in range(len(bin_edges) - 1):
        b_lo, b_hi = bin_edges[i], bin_edges[i + 1]
        in_bin = [f for f in unique_hz if b_lo <= f < b_hi - 1e-9 or (i == len(bin_edges) - 2 and abs(f - b_hi) < 1e-9)]
        count = len(in_bin)
        counts.append(count)
        width = b_hi - b_lo
        bins.append(
            {
                "bin_lo_hz": b_lo,
                "bin_hi_hz": b_hi,
                "bin_label": f"{b_lo:g}-{b_hi:g}",
                "mode_count": count,
                "mode_frequencies_hz": sorted(in_bin),
                "density_per_hz": (count / width) if width > 0 else 0.0,
            }
        )

    median_count = statistics.median(counts) if counts else 0.0
    for b in bins:
        c = b["mode_count"]
        if median_count > 0 and c >= 1.5 * median_count:
            zone = "ZONE1_dense"
        elif median_count > 0 and c <= 0.5 * median_count:
            zone = "ZONE3_sparse"
        else:
            zone = "ZONE2_moderate"
        b["zone_candidate"] = zone

    body = {
        "schema": "b3_scout_density_report_v1",
        "generated_utc": _utc_now(),
        "sample_id": sample_id,
        "run_id": run_id,
        "unique_mode_count": len(unique_hz),
        "unique_mode_frequencies_hz": unique_hz,
        "bin_width_hz": bin_width_hz,
        "freq_range_hz": [freq_min_hz, freq_max_hz],
        "bins": bins,
        "median_count_per_bin": median_count,
        "density_result_status": density_body.get("status"),
        "spacing_hz": (density_body.get("spacings") or [{}])[0].get("spacing_hz"),
        "experiment_wall_s": density_body.get("experiment_wall_s"),
    }
    write_json_atomic(report_dir / "density_zone_report.json", body)
    md_lines = [
        f"# Scout density report — {sample_id}",
        "",
        f"- unique modes: **{len(unique_hz)}**",
        f"- bins: **{bin_width_hz} Hz** over **{freq_min_hz}–{freq_max_hz} Hz**",
        "",
        "| bin (Hz) | count | density/Hz | zone |",
        "|----------|-------|------------|------|",
    ]
    for b in bins:
        md_lines.append(
            f"| {b['bin_label']} | {b['mode_count']} | {b['density_per_hz']:.4f} | {b['zone_candidate']} |"
        )
    (report_dir / "density_zone_report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return body


def _build_cross_sample_summary(
    *,
    sample_reports: List[Dict[str, Any]],
    batch_meta: Dict[str, Any],
) -> Dict[str, Any]:
    if not sample_reports:
        return {"status": "FAIL", "reason": "no_sample_reports"}

    bin_labels = [b["bin_label"] for b in sample_reports[0].get("bins") or []]
    per_bin: List[Dict[str, Any]] = []
    all_means: List[float] = []

    for label in bin_labels:
        counts_by_sample: Dict[str, int] = {}
        counts: List[int] = []
        zones: List[str] = []
        for rep in sample_reports:
            sid = str(rep.get("sample_id"))
            for b in rep.get("bins") or []:
                if b.get("bin_label") == label:
                    c = int(b.get("mode_count") or 0)
                    counts_by_sample[sid] = c
                    counts.append(c)
                    zones.append(str(b.get("zone_candidate")))
                    break
        mean_c = statistics.mean(counts) if counts else 0.0
        stdev_c = statistics.pstdev(counts) if len(counts) > 1 else 0.0
        all_means.append(mean_c)
        per_bin.append(
            {
                "bin_label": label,
                "counts_by_sample": counts_by_sample,
                "mean_count": mean_c,
                "stdev_count": stdev_c,
                "zone_votes": zones,
            }
        )

    global_median = statistics.median(all_means) if all_means else 0.0
    for row in per_bin:
        m = row["mean_count"]
        if global_median > 0 and m >= 1.5 * global_median:
            row["consensus_zone"] = "ZONE1_dense"
            row["stability"] = "consensus_dense"
        elif global_median > 0 and m <= 0.5 * global_median:
            row["consensus_zone"] = "ZONE3_sparse"
            row["stability"] = "consensus_sparse"
        else:
            row["consensus_zone"] = "ZONE2_moderate"
            votes = row.get("zone_votes") or []
            if len(set(votes)) == 1:
                row["stability"] = "unanimous_moderate"
            else:
                row["stability"] = "unstable_mixed_votes"

    runtime_notes = {
        "L_prod_uniform_5p5hz_full_60_550": {
            "description": "Reference: ~89 targets on L_prod active_dim~316k",
            "relative_cost": "highest",
        },
        "zone_policy_spacing_hz": {"dense": 6, "moderate": 9, "sparse": 15},
        "scout_overhead_per_sample": "Stage A on L_scout_coarse + ~66 discovery targets at 7.5 Hz spacing",
        "scout_mesh_note": "Shared L_scout_coarse mesh; material-only LHS deltas per sample",
    }

    summary = {
        "schema": "b3_coarse_scout_lhs_zone_consensus_v1",
        "generated_utc": _utc_now(),
        "batch": batch_meta,
        "sample_count": len(sample_reports),
        "sample_ids": [r.get("sample_id") for r in sample_reports],
        "bin_width_hz": sample_reports[0].get("bin_width_hz"),
        "per_bin": per_bin,
        "global_median_mean_count": global_median,
        "runtime_comparison_notes": runtime_notes,
        "status": "OK",
    }
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    write_json_atomic(SUMMARY_DIR / "coarse_scout_lhs_zone_consensus.json", summary)
    md = [
        "# Coarse scout LHS zone consensus",
        "",
        f"- samples: **{len(sample_reports)}**",
        f"- bin width: **{sample_reports[0].get('bin_width_hz')} Hz**",
        "",
        "| bin | mean count | stdev | consensus | stability |",
        "|-----|------------|-------|-----------|-----------|",
    ]
    for row in per_bin:
        md.append(
            f"| {row['bin_label']} | {row['mean_count']:.2f} | {row['stdev_count']:.2f} | "
            f"{row['consensus_zone']} | {row['stability']} |"
        )
    md.append("")
    md.append("## Runtime notes (rough)")
    md.append("")
    for k, v in runtime_notes.items():
        md.append(f"- **{k}:** {v}")
    (SUMMARY_DIR / "coarse_scout_lhs_zone_consensus.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return summary


def _path_for_subprocess(path: Path, *, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _cmd_stage_a(
    *,
    repo_root: Path,
    prod_python: str,
    mesh_level: str,
    core_config_rel: str,
    checkpoint_dir: Path,
) -> List[str]:
    return [
        prod_python,
        _path_for_subprocess(repo_root / STAGE_A_SCRIPT, repo_root=repo_root),
        "--mesh-level",
        mesh_level,
        "--B3-block-compose-backend",
        "csr_bulk",
        "--B3-synthesis-region-dofs",
        "off",
        "--core-config",
        core_config_rel,
        "--output-dir",
        _path_for_subprocess(checkpoint_dir, repo_root=repo_root),
    ]


def _cmd_stage_b(
    *,
    repo_root: Path,
    solver_python: str,
    checkpoint_dir: Path,
    reference_json: Path,
    stage_b_dir: Path,
    freq_min_hz: float,
    freq_max_hz: float,
    spacing_hz: float,
    half_width_hz: float,
) -> List[str]:
    return [
        solver_python,
        _path_for_subprocess(repo_root / STAGE_B_SCRIPT, repo_root=repo_root),
        "--checkpoint-dir",
        _path_for_subprocess(checkpoint_dir, repo_root=repo_root),
        "--reference-json",
        _path_for_subprocess(reference_json, repo_root=repo_root),
        "--start-hz",
        str(freq_min_hz),
        "--stop-hz",
        str(freq_max_hz),
        "--spacings-hz",
        str(spacing_hz),
        "--B3-discovery-mode",
        "--discovery-band-hz",
        str(freq_min_hz),
        str(freq_max_hz),
        "--target-window-half-width-hz",
        str(half_width_hz),
        "--output-dir",
        _path_for_subprocess(stage_b_dir, repo_root=repo_root),
    ]


def _print_execution_plan(plan: Dict[str, Any]) -> None:
    print("[scout_batch] ========== EXECUTION PLAN ==========", flush=True)
    print(f"[scout_batch] dry_run={plan.get('dry_run')} will_execute={plan.get('will_execute')}", flush=True)
    print(f"[scout_batch] mesh_level={plan.get('mesh_level')} mesh_exists={plan.get('mesh_exists')}", flush=True)
    print(f"[scout_batch] samples_selected={len(plan.get('samples') or [])}", flush=True)
    for s in plan.get("samples") or []:
        print(f"[scout_batch] --- {s.get('sample_id')} run_id={s.get('run_id')} ---", flush=True)
        print(f"[scout_batch]   top_density={s.get('top_density')} back_density={s.get('back_density')}", flush=True)
        print(f"[scout_batch]   mesh_file={s.get('core_config_mesh_file')}", flush=True)
        print(f"[scout_batch]   checkpoint_exists={s.get('checkpoint_exists')} density_exists={s.get('density_exists')}", flush=True)
        print(f"[scout_batch]   skip_stage_a={s.get('skip_stage_a')} skip_stage_b={s.get('skip_stage_b')}", flush=True)
        if plan.get("dry_run"):
            print(f"[scout_batch]   stage_a: {s.get('stage_a_cmd')}", flush=True)
            print(f"[scout_batch]   stage_b: {s.get('stage_b_cmd')}", flush=True)
    print("[scout_batch] ====================================", flush=True)


def build_batch_plan(
    *,
    repo_root: Path,
    samples_path: Path,
    mesh_level: str,
    freq_min_hz: float,
    freq_max_hz: float,
    spacing_hz: float,
    half_width_hz: float,
    max_samples: int,
    prod_python: str,
    solver_python: str,
    reference_json: Path,
    dry_run: bool,
    force: bool,
) -> Dict[str, Any]:
    rows = _read_jsonl(samples_path)[: max(1, int(max_samples))]
    mesh_rel = _scout_mesh_file_rel(mesh_level)
    mesh_abs = (repo_root / mesh_rel).resolve()
    mesh_exists = mesh_abs.is_file()
    ref_rel = _repo_relative(reference_json, repo_root=repo_root)

    sample_plans: List[Dict[str, Any]] = []
    for row in rows:
        sid = str(row.get("sample_id") or "").strip()
        if not sid:
            continue
        run_id = _scout_run_id(sid)
        paths = _paths_for_sample(sid, mesh_level)
        overlay_dir = paths["overlay_dir"]
        resolve_info = _resolve_scout_sample_overlay(
            row, mesh_level=mesh_level, repo_root=repo_root, overlay_dir=overlay_dir, force=force
        )
        core_rel = _repo_relative(paths["resolved_core_config"], repo_root=repo_root)
        resolved = json.loads(paths["resolved_core_config"].read_text(encoding="utf-8"))
        ckpt = paths["checkpoint_dir"]
        stage_b = paths["stage_b_dir"]
        density_json = stage_b / "density_result.json"
        export_manifest = ckpt / "checkpoint_export_manifest.json"

        ckpt_ok = False
        if export_manifest.is_file():
            ckpt_ok, _ = _verify_stage_a_export(export_manifest)
        dens_ok = False
        if density_json.is_file():
            dens_ok, _, _ = _verify_density_result(density_json)

        skip_a = bool(ckpt_ok and not force)
        skip_b = bool(dens_ok and not force)

        argv_a = _cmd_stage_a(
            repo_root=repo_root,
            prod_python=prod_python,
            mesh_level=mesh_level,
            core_config_rel=core_rel,
            checkpoint_dir=ckpt,
        )
        argv_b = _cmd_stage_b(
            repo_root=repo_root,
            solver_python=solver_python,
            checkpoint_dir=ckpt,
            reference_json=reference_json,
            stage_b_dir=stage_b,
            freq_min_hz=freq_min_hz,
            freq_max_hz=freq_max_hz,
            spacing_hz=spacing_hz,
            half_width_hz=half_width_hz,
        )

        sample_plans.append(
            {
                "sample_id": sid,
                "run_id": run_id,
                "selection_reason": row.get("selection_reason"),
                "note": row.get("note"),
                "top_density": resolve_info.get("top_density")
                or (resolved.get("materials") or {}).get("top", {}).get("density"),
                "back_density": resolve_info.get("back_density")
                or (resolved.get("materials") or {}).get("back", {}).get("density"),
                "core_config_path": core_rel,
                "core_config_mesh_file": (resolved.get("solver") or {}).get("mesh_file"),
                "clamp_ribs": (resolved.get("solver") or {}).get("clamp_ribs"),
                "overlay_dir": _repo_relative(overlay_dir, repo_root=repo_root),
                "checkpoint_dir": _repo_relative(ckpt, repo_root=repo_root),
                "stage_b_dir": _repo_relative(stage_b, repo_root=repo_root),
                "report_dir": _repo_relative(paths["report_dir"], repo_root=repo_root),
                "checkpoint_exists": ckpt.is_dir(),
                "checkpoint_pass": ckpt_ok,
                "density_exists": density_json.is_file(),
                "density_pass": dens_ok,
                "skip_stage_a": skip_a,
                "skip_stage_b": skip_b,
                "stage_a_cmd": " ".join(argv_a),
                "stage_b_cmd": " ".join(argv_b),
                "readiness_status": resolve_info.get("readiness_status"),
            }
        )

    return {
        "schema": "b3_coarse_scout_lhs_batch_plan_v1",
        "generated_utc": _utc_now(),
        "dry_run": dry_run,
        "will_execute": not dry_run,
        "mesh_level": mesh_level,
        "mesh_file": mesh_rel,
        "mesh_exists": mesh_exists,
        "samples_jsonl": _repo_relative(samples_path, repo_root=repo_root),
        "reference_json": ref_rel,
        "discovery": {
            "freq_min_hz": freq_min_hz,
            "freq_max_hz": freq_max_hz,
            "spacing_hz": spacing_hz,
            "half_width_hz": half_width_hz,
        },
        "env": {
            "stage_a": {
                "python": prod_python,
                "prod_venv": DEFAULT_PROD_VENV,
                "profile": "production_venv_strict",
                "VIRTUAL_ENV_set": DEFAULT_PROD_VENV,
                "PETSC_DIR": PETSC_DIR_PROD,
                "note": "Does not inherit parent VIRTUAL_ENV; PATH strips solver-mkl/bin",
            },
            "stage_b": {
                "python": solver_python,
                "solver_venv": SOLVER_MKL_VENV,
                "profile": "solver_mkl_isolated_strict",
                "unset": list(STAGE_B_ENV_UNSET),
            },
        },
        "samples": sample_plans,
        "sample_count_note": (
            "5 samples: 3 pilot + 2 synthetic extensions (lhs_scout_ext_004/005). "
            "Replace JSONL when additional LHS material specs are available for 6-8."
        ),
    }


def run_batch(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Coarse L_scout_coarse LHS scout overnight batch.")
    parser.add_argument("--samples-jsonl", default=str(DEFAULT_SAMPLES_JSONL))
    parser.add_argument("--mesh-level", default="L_scout_coarse")
    parser.add_argument("--freq-min-hz", type=float, default=60.0)
    parser.add_argument("--freq-max-hz", type=float, default=550.0)
    parser.add_argument("--spacing-hz", type=float, default=7.5)
    parser.add_argument("--target-window-half-width-hz", type=float, default=3.75)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--continue-on-fail", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--force", action="store_true", default=False)
    parser.add_argument("--prod-python", default=os.environ.get("B3_PROD_PYTHON", M3_PROD_PYTHON))
    parser.add_argument("--prod-venv", default=os.environ.get("B3_PROD_VENV", DEFAULT_PROD_VENV))
    parser.add_argument("--solver-python", default=os.environ.get("B3_SOLVER_PYTHON", M3_SOLVER_PYTHON))
    parser.add_argument("--solver-venv", default=os.environ.get("B3_SOLVER_MKL_VENV", SOLVER_MKL_VENV))
    parser.add_argument("--reference-json", default=str(DEFAULT_REFERENCE_STUB))
    parser.add_argument(
        "--bin-width-hz",
        type=float,
        default=BIN_WIDTH_HZ,
        help="Post-process bin width for zone reports (default 25 Hz).",
    )
    args = parser.parse_args(argv)

    repo_root = _detect_repo_root(SCRIPT_DIR)
    samples_path = Path(args.samples_jsonl).expanduser()
    if not samples_path.is_absolute():
        samples_path = (repo_root / samples_path).resolve()
    reference_json = Path(args.reference_json).expanduser()
    if not reference_json.is_absolute():
        reference_json = (repo_root / reference_json).resolve()

    plan = build_batch_plan(
        repo_root=repo_root,
        samples_path=samples_path,
        mesh_level=str(args.mesh_level),
        freq_min_hz=float(args.freq_min_hz),
        freq_max_hz=float(args.freq_max_hz),
        spacing_hz=float(args.spacing_hz),
        half_width_hz=float(args.target_window_half_width_hz),
        max_samples=int(args.max_samples),
        prod_python=str(args.prod_python),
        solver_python=str(args.solver_python),
        reference_json=reference_json,
        dry_run=bool(args.dry_run),
        force=bool(args.force),
    )

    plan_path = SUMMARY_DIR / ("batch_plan_dry_run.json" if args.dry_run else "batch_plan.json")
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    write_json_atomic(plan_path, plan)
    _print_execution_plan(plan)

    if args.dry_run:
        print(f"[scout_batch] wrote plan {plan_path}", flush=True)
        return 0

    if not plan.get("mesh_exists"):
        print(f"[scout_batch] FAIL shared mesh missing: {plan.get('mesh_file')}", flush=True)
        return 2

    prod_venv = str(args.prod_venv)
    solver_venv = str(args.solver_venv)
    env_probe_log_dir = PIPELINE_RUNS / "logs" / "scout_lhs_batch_env_probe"
    print("[scout_batch] Stage A/B environment probes (strict isolation)", flush=True)
    ok_probe, probe_payload = _run_stage_env_probes(
        repo_root=repo_root,
        prod_python=str(args.prod_python),
        prod_venv=prod_venv,
        solver_python=str(args.solver_python),
        solver_venv=solver_venv,
        log_dir=env_probe_log_dir,
    )
    write_json_atomic(SUMMARY_DIR / "env_probe.json", probe_payload)
    print(f"[scout_batch] env_probe stage_a_ok={probe_payload['stage_a']['ok']} stage_b_ok={probe_payload['stage_b']['ok']}", flush=True)
    if not ok_probe:
        print("[scout_batch] FAIL environment probe; see pipeline_runs/logs/scout_lhs_batch_env_probe/", flush=True)
        return 2

    env_a = _prod_subprocess_env_strict(prod_python=str(args.prod_python), prod_venv=prod_venv)
    env_b = _solver_mkl_subprocess_env_strict(
        solver_python=str(args.solver_python), solver_venv=solver_venv
    )

    sample_reports: List[Dict[str, Any]] = []
    failures = 0
    t_batch0 = time.perf_counter()

    for s in plan["samples"]:
        sid = str(s["sample_id"])
        paths = _paths_for_sample(sid, str(args.mesh_level))
        paths["log_dir"].mkdir(parents=True, exist_ok=True)
        sample_status = "PASS"
        fail_reason: Optional[str] = None

        print(f"[scout_batch] ===== sample {sid} =====", flush=True)

        if not s.get("skip_stage_a"):
            rc_a = _run_subprocess(
                _cmd_stage_a(
                    repo_root=repo_root,
                    prod_python=str(args.prod_python),
                    mesh_level=str(args.mesh_level),
                    core_config_rel=str(s["core_config_path"]),
                    checkpoint_dir=paths["checkpoint_dir"],
                ),
                env=env_a,
                cwd=repo_root,
                log_path=paths["log_dir"] / "stage_a.log",
                label=f"Stage A {sid}",
            )
            ok_a, detail_a = _verify_stage_a_export(paths["checkpoint_dir"] / "checkpoint_export_manifest.json")
            if rc_a != 0 or not ok_a:
                sample_status = "FAIL"
                fail_reason = f"stage_a:rc={rc_a}:{detail_a}"
        else:
            print(f"[scout_batch] skip Stage A (existing PASS) {sid}", flush=True)

        if sample_status == "PASS" and not s.get("skip_stage_b"):
            rc_b = _run_subprocess(
                _cmd_stage_b(
                    repo_root=repo_root,
                    solver_python=str(args.solver_python),
                    checkpoint_dir=paths["checkpoint_dir"],
                    reference_json=reference_json,
                    stage_b_dir=paths["stage_b_dir"],
                    freq_min_hz=float(args.freq_min_hz),
                    freq_max_hz=float(args.freq_max_hz),
                    spacing_hz=float(args.spacing_hz),
                    half_width_hz=float(args.target_window_half_width_hz),
                ),
                env=env_b,
                cwd=repo_root,
                log_path=paths["log_dir"] / "stage_b.log",
                label=f"Stage B {sid}",
            )
            ok_b, detail_b, _ = _verify_density_result(paths["stage_b_dir"] / "density_result.json")
            if rc_b != 0 or not ok_b:
                sample_status = "FAIL"
                fail_reason = f"stage_b:rc={rc_b}:{detail_b}"
        elif sample_status == "PASS":
            print(f"[scout_batch] skip Stage B (existing PASS) {sid}", flush=True)

        if sample_status == "PASS":
            _, _, dens = _verify_density_result(paths["stage_b_dir"] / "density_result.json")
            if dens:
                rep = _postprocess_sample_density(
                    sample_id=sid,
                    run_id=str(s["run_id"]),
                    density_body=dens,
                    report_dir=paths["report_dir"],
                    freq_min_hz=float(args.freq_min_hz),
                    freq_max_hz=float(args.freq_max_hz),
                    bin_width_hz=float(args.bin_width_hz),
                )
                sample_reports.append(rep)
        else:
            failures += 1
            print(f"[scout_batch] FAIL {sid}: {fail_reason}", flush=True)
            if not args.continue_on_fail:
                break

    if sample_reports:
        _build_cross_sample_summary(
            sample_reports=sample_reports,
            batch_meta={
                "samples_jsonl": plan.get("samples_jsonl"),
                "mesh_level": plan.get("mesh_level"),
                "elapsed_s": time.perf_counter() - t_batch0,
                "failures": failures,
            },
        )

    manifest = {
        "generated_utc": _utc_now(),
        "terminal_status": "PASS" if failures == 0 else "PARTIAL",
        "failures": failures,
        "samples_completed": len(sample_reports),
        "elapsed_s": time.perf_counter() - t_batch0,
        "plan_path": str(plan_path),
    }
    write_json_atomic(SUMMARY_DIR / "batch_run_manifest.json", manifest)
    print(f"[scout_batch] done terminal={manifest['terminal_status']} reports={len(sample_reports)}", flush=True)
    return 0 if failures == 0 else 1


def main() -> int:
    print(
        "LEGACY (M3.4): coarse scout-only batch. Full LHS production: "
        "v2_b3_m4_lhs_production_batch.py",
        file=sys.stderr,
    )
    return run_batch()


if __name__ == "__main__":
    raise SystemExit(main())
