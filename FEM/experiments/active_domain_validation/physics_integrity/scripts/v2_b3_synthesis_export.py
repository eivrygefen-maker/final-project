#!/usr/bin/env python3
"""Stage A synthesis metadata and region DOF index export (production .venv)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_mesh_convergence_common import mesh_path  # noqa: E402
from v2_b3_rich_modal_lib import (  # noqa: E402
    REGION_DOF_INDICES_NPZ,
    REGION_DOF_LAYOUT,
    SYNTHESIS_METADATA_JSON,
    SYNTHESIS_METADATA_SCHEMA,
    TAG_PROTOCOL_V1,
)
from v2_b3_petsc_util import write_json_atomic  # noqa: E402

CASE_ID = "baseline_coupled_v2"
PHYSICS_CONFIG = SCRIPT_DIR.parent / "configs" / "coupled_physical_core_v2.json"
REGION_DOF_WORKER = SCRIPT_DIR / "v2_b3_synthesis_region_dof_worker.py"
REGION_DOF_SUBPROCESS_TIMEOUT_S = 600
REGION_DOF_STATUS_PASS = "BEST_EFFORT_PASS"
REGION_DOF_SOURCE_OPERATOR_BUILD = "operator_build_context"

TAG_TOP = 1
TAG_SOUNDHOLE = 2
TAG_BACK = 3
TAG_RIBS = 4

SynthesisRegionDofsMode = Literal["off", "best_effort"]


def _detect_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(16):
        if (cur / ".git").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return start.resolve().parents[4]


def bootstrap_fem_import_paths(*, start: Optional[Path] = None) -> Path:
    """
    Match B3 checkpoint/audit layout: physics_integrity/scripts + FEM/scripts on sys.path.
    Required for ``import fem_main_3d`` in the isolated region-DOF worker subprocess.
    """
    anchor = (start or SCRIPT_DIR).resolve()
    script_dir = anchor.parent if anchor.suffix == ".py" else anchor
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    repo_root = _detect_repo_root(script_dir)
    fem_scripts = repo_root / "FEM" / "scripts"
    if fem_scripts.is_dir() and str(fem_scripts) not in sys.path:
        sys.path.insert(0, str(fem_scripts))
    return repo_root


def fem_import_diagnostics(*, start: Optional[Path] = None, module_name: str = "fem_main_3d") -> Dict[str, Any]:
    """Fail-fast context for ``import fem_main_3d`` (validation/subprocess parity with Stage A)."""
    anchor = (start or SCRIPT_DIR).resolve()
    script_dir = anchor.parent if anchor.suffix == ".py" else anchor
    repo_root = bootstrap_fem_import_paths(start=script_dir)
    fem_scripts = repo_root / "FEM" / "scripts"
    module_file = fem_scripts / f"{module_name}.py"
    return {
        "repo_root": str(repo_root.resolve()),
        "cwd": str(Path.cwd().resolve()),
        "sys_executable": sys.executable,
        "sys_path": [str(Path(p).resolve()) if p else p for p in sys.path],
        "module_searched": module_name,
        "fem_scripts_dir": str(fem_scripts.resolve()),
        "fem_scripts_on_sys_path": str(fem_scripts) in sys.path,
        "resolved_module_path": str(module_file.resolve()) if module_file.is_file() else None,
        "module_file_exists": module_file.is_file(),
    }


def import_fem_main_3d(*, start: Optional[Path] = None, module_name: str = "fem_main_3d") -> Tuple[Any, Dict[str, Any]]:
    """
    Import ``fem_main_3d`` using the same sys.path bootstrap as B3 Stage A / region-DOF workers.

    On failure, prints diagnostics to stderr and re-raises (never silently returns empty masks).
    """
    diag = fem_import_diagnostics(start=start, module_name=module_name)
    if not diag.get("module_file_exists"):
        print(json.dumps(diag, indent=2, sort_keys=True), file=sys.stderr)
        raise ModuleNotFoundError(
            f"No module named '{module_name}'; expected file missing: {diag.get('resolved_module_path')}"
        )
    try:
        import importlib

        mod = importlib.import_module(module_name)
        diag["imported_module_file"] = str(Path(mod.__file__).resolve())
        return mod, diag
    except ModuleNotFoundError as exc:
        print(json.dumps(diag, indent=2, sort_keys=True), file=sys.stderr)
        raise ModuleNotFoundError(f"{exc}; see fem_import_diagnostics on stderr") from exc


def region_dof_subprocess_env(
    *,
    repo_root: Path,
    base_env: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    """Subprocess env for region-DOF worker (production python + FEM import paths)."""
    env = dict(base_env or os.environ)
    prepend = [
        str((repo_root / "FEM" / "scripts").resolve()),
        str(
            (
                repo_root
                / "FEM"
                / "experiments"
                / "active_domain_validation"
                / "physics_integrity"
                / "scripts"
            ).resolve()
        ),
    ]
    existing = str(env.get("PYTHONPATH") or "")
    env["PYTHONPATH"] = os.pathsep.join(prepend + ([existing] if existing else []))
    return env


def resolve_region_dof_mesh_file(
    checkpoint: Path,
    *,
    mesh_level: str,
    built_meta: Dict[str, Any],
    core_config_path: Optional[Path] = None,
) -> Tuple[Optional[Path], str]:
    """Resolve M4 per-sample mesh or baseline fallback for region DOF locate."""
    repo_root = _detect_repo_root(checkpoint)
    candidates: List[Tuple[str, Path]] = []

    for key in ("region_dof_mesh_file", "mesh_file"):
        raw = built_meta.get(key)
        if raw:
            p = Path(str(raw))
            candidates.append((key, p if p.is_absolute() else repo_root / p))

    for label, cfg_path in (
        ("core_config_arg", core_config_path),
        ("lprod_resolved", checkpoint.parent / "resolved_core_config.json"),
        ("run_lprod_resolved", checkpoint.parent.parent / "resolved_core_config.json"),
    ):
        if not cfg_path:
            continue
        path = Path(cfg_path).expanduser()
        if not path.is_file():
            continue
        try:
            cfg = json.loads(path.read_text(encoding="utf-8"))
            rel = (cfg.get("solver") or {}).get("mesh_file")
            if rel:
                p = Path(str(rel))
                candidates.append((label, p if p.is_absolute() else repo_root / p))
        except (OSError, ValueError, json.JSONDecodeError):
            continue

    baseline = mesh_path(str(mesh_level), CASE_ID)
    candidates.append(("baseline_fallback", baseline))

    tried: List[str] = []
    for label, path in candidates:
        tried.append(f"{label}={path}")
        if path.is_file() and path.stat().st_size > 1000:
            return path.resolve(), label

    return None, f"mesh_missing; tried: {'; '.join(tried)}"


def _trace_rows_to_b3_u_w(trace_rows: np.ndarray, *, n_u_b3: int) -> np.ndarray:
    """Map shell-trace displacement DOF rows to B3 monolithic W u-block rows (0..n_u_b3-1)."""
    tr = np.asarray(trace_rows, dtype=np.int32).ravel()
    if tr.size == 0 or n_u_b3 <= 0:
        return np.asarray([], dtype=np.int32)
    valid = tr[(tr >= 0) & (tr < int(n_u_b3))]
    if valid.size == 0:
        return np.asarray([], dtype=np.int32)
    return np.unique(valid.astype(np.int32, copy=False))


def _default_solver_physics() -> Dict[str, float]:
    out = {"pressure_dof_scale": 30.0, "fsi_coupling_gain": 1.0}
    if PHYSICS_CONFIG.is_file():
        try:
            cfg = json.loads(PHYSICS_CONFIG.read_text(encoding="utf-8"))
            sc = cfg.get("solver") or {}
            out["pressure_dof_scale"] = float(sc.get("pressure_dof_scale", out["pressure_dof_scale"]))
            out["fsi_coupling_gain"] = float(sc.get("fsi_coupling_gain", out["fsi_coupling_gain"]))
        except Exception:
            pass
    return out


def _gnhep_scales_from_built(built: Dict[str, Any]) -> Dict[str, float]:
    op_meta = built.get("op_meta") or {}
    for key in ("gnhep_scales", "B3_gnhep_scales"):
        block = op_meta.get(key)
        if isinstance(block, dict) and block.get("s_uu") is not None:
            return {
                "s_uu": float(block["s_uu"]),
                "s_pp": float(block["s_pp"]),
                "s_couple": float(block.get("s_couple", 1.0)),
            }
    return {"s_uu": None, "s_pp": None, "s_couple": None, "source": "not_captured_in_op_meta"}


def region_dof_status_is_pass(status: Optional[str]) -> bool:
    return str(status or "") in ("present", REGION_DOF_STATUS_PASS)


def _region_arr(region_dof_build: Mapping[str, Any], key: str) -> np.ndarray:
    val = region_dof_build.get(key)
    if val is None:
        return np.asarray([], dtype=np.int32)
    return np.asarray(val, dtype=np.int32).ravel()


def export_region_dof_indices_from_operator_build(
    checkpoint: Path,
    *,
    region_dof_build: Mapping[str, Any],
) -> Tuple[str, Optional[str]]:
    """Write region_dof_indices.npz from Stage A operator build masks (in-process, no subprocess)."""
    checkpoint = checkpoint.expanduser().resolve()
    u_idx_top = _region_arr(region_dof_build, "u_idx_top")
    u_idx_back = _region_arr(region_dof_build, "u_idx_back")
    u_idx_ribs = _region_arr(region_dof_build, "u_idx_ribs")
    u_idx_soundhole = _region_arr(region_dof_build, "u_idx_soundhole")
    p_idx_all = _region_arr(region_dof_build, "p_idx_all")
    if p_idx_all.size == 0:
        p_idx_all = _region_arr(region_dof_build, "p_idx_air")
    u_idx_all = _region_arr(region_dof_build, "u_idx_all")
    mesh_file = str(region_dof_build.get("region_dof_mesh_file") or "")
    source = str(region_dof_build.get("region_dof_source") or REGION_DOF_SOURCE_OPERATOR_BUILD)

    if u_idx_top.size == 0 and u_idx_back.size == 0 and u_idx_ribs.size == 0:
        counts = region_dof_build.get("counts") or {}
        return (
            "deferred_to_stage_c",
            f"operator_build_context_empty_structural counts={counts}",
        )

    np.savez_compressed(
        checkpoint / REGION_DOF_INDICES_NPZ,
        u_idx_top=u_idx_top,
        u_idx_back=u_idx_back,
        u_idx_ribs=u_idx_ribs,
        u_idx_soundhole=u_idx_soundhole,
        p_idx_air=p_idx_all.copy(),
        p_idx_all=p_idx_all.copy(),
        u_idx_all=u_idx_all,
        region_dof_mesh_file=np.asarray([mesh_file]),
        region_dof_source=np.asarray([source]),
        layout=np.asarray([str(region_dof_build.get("layout") or REGION_DOF_LAYOUT)]),
        back_includes_ribs=np.asarray([bool(region_dof_build.get("back_includes_ribs", True))]),
    )
    return REGION_DOF_STATUS_PASS, None


def export_region_dof_indices_npz(
    checkpoint: Path,
    *,
    mesh_level: str,
    built_meta: Dict[str, Any],
    mesh_file: Optional[Path] = None,
    core_config_path: Optional[Path] = None,
) -> Tuple[str, Optional[str]]:
    """Locate region DOFs via DOLFINx; indices are global W rows (u_idx / p_idx)."""
    bootstrap_fem_import_paths(start=checkpoint)

    import dolfinx.mesh as dmesh
    from dolfinx import fem

    import fem_main_3d as fem3d

    checkpoint = checkpoint.expanduser().resolve()
    if mesh_file is None or not mesh_file.is_file():
        mesh_file, resolve_detail = resolve_region_dof_mesh_file(
            checkpoint,
            mesh_level=mesh_level,
            built_meta=built_meta,
            core_config_path=core_config_path,
        )
        if mesh_file is None:
            return "deferred_to_stage_c", resolve_detail

    msh, _cell_tags, facet_tags = fem3d._load_mesh_and_tags(mesh_file)
    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim - 1, tdim)

    f_top = np.asarray(facet_tags.find(TAG_TOP), dtype=np.int32)
    f_back = np.asarray(facet_tags.find(TAG_BACK), dtype=np.int32)
    f_ribs = np.asarray(facet_tags.find(TAG_RIBS), dtype=np.int32)
    f_soundhole = np.asarray(facet_tags.find(TAG_SOUNDHOLE), dtype=np.int32)
    shell_facets = np.unique(np.concatenate([f_top, f_back, f_ribs]).astype(np.int32, copy=False))

    shell_mesh, shell_to_parent, _, _ = dmesh.create_submesh(msh, tdim - 1, shell_facets)
    V_u_trace = fem.functionspace(shell_mesh, fem3d._displacement_element(shell_mesh, 1))

    def _trace_u_rows(facets: np.ndarray) -> np.ndarray:
        if facets.size == 0:
            return np.asarray([], dtype=np.int32)
        dofs = fem3d._locate_facet_displacement_dofs(V_u_trace, shell_mesh, facets)
        return np.unique(np.asarray(dofs, dtype=np.int32).ravel())

    n_u_b3 = int(built_meta.get("n_u_b3") or built_meta.get("n_u") or 0)
    p_idx_all = np.asarray(built_meta.get("p_idx") or [], dtype=np.int32).ravel()
    if n_u_b3 <= 0:
        return "deferred_to_stage_c", "built_metadata_missing_n_u_b3"

    trace_top = _trace_u_rows(f_top)
    trace_back = _trace_u_rows(f_back)
    trace_ribs = _trace_u_rows(f_ribs)
    trace_soundhole = _trace_u_rows(f_soundhole)

    # B3 checkpoint u_idx = arange(n_u_b3): trace shell DOF rows are W u-block row indices.
    u_idx_top = _trace_rows_to_b3_u_w(trace_top, n_u_b3=n_u_b3)
    u_idx_back = _trace_rows_to_b3_u_w(trace_back, n_u_b3=n_u_b3)
    u_idx_ribs = _trace_rows_to_b3_u_w(trace_ribs, n_u_b3=n_u_b3)
    u_idx_soundhole = _trace_rows_to_b3_u_w(trace_soundhole, n_u_b3=n_u_b3)

    if u_idx_top.size == 0 and u_idx_back.size == 0:
        return (
            "deferred_to_stage_c",
            f"no_b3_u_region_dofs: trace_top={trace_top.size} trace_back={trace_back.size} "
            f"trace_ribs={trace_ribs.size} n_u_b3={n_u_b3}",
        )

    p_idx_air = p_idx_all.copy()

    np.savez_compressed(
        checkpoint / REGION_DOF_INDICES_NPZ,
        u_idx_top=u_idx_top,
        u_idx_back=u_idx_back,
        u_idx_ribs=u_idx_ribs,
        u_idx_soundhole=u_idx_soundhole,
        p_idx_air=p_idx_air,
        p_idx_all=p_idx_all,
        u_idx_all=np.arange(n_u_b3, dtype=np.int32),
        region_dof_mesh_file=np.asarray([str(mesh_file.resolve())]),
        layout=np.asarray([REGION_DOF_LAYOUT]),
        back_includes_ribs=np.asarray([True]),
    )
    return REGION_DOF_STATUS_PASS, None


def _build_synthesis_metadata_body(
    checkpoint: Path,
    *,
    built: Dict[str, Any],
    built_meta: Dict[str, Any],
    mesh_level: str,
    compose_backend: str,
    region_status: str,
    region_error: Optional[str],
    region_dofs_mode: SynthesisRegionDofsMode,
    core_config_provenance: Optional[Dict[str, Any]] = None,
    mesh_file: Optional[Path] = None,
) -> Dict[str, Any]:
    if mesh_file is None:
        mesh_file, _ = resolve_region_dof_mesh_file(
            checkpoint,
            mesh_level=mesh_level,
            built_meta=built_meta,
        )
    physics = _default_solver_physics()
    gnhep = _gnhep_scales_from_built(built)
    n_u = int(built_meta.get("n_u_b3") or built.get("n_u_b3") or 0)
    p_idx = np.asarray(built_meta.get("p_idx") or built.get("p_idx") or [], dtype=np.int32)
    n_p = int(p_idx.size) if p_idx.size else 0

    body: Dict[str, Any] = {
        "schema": SYNTHESIS_METADATA_SCHEMA,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mesh_file": str(mesh_file.resolve()) if mesh_file and mesh_file.is_file() else str(mesh_file or ""),
        "mesh_level": str(mesh_level),
        "case_id": CASE_ID,
        "tag_protocol": dict(TAG_PROTOCOL_V1),
        "gnhep_scales": gnhep,
        "pressure_dof_scale": float(physics["pressure_dof_scale"]),
        "fsi_coupling_gain": float(physics["fsi_coupling_gain"]),
        "compose_backend": str(compose_backend),
        "n_w": int(built_meta.get("n_w") or built.get("n_w") or 0),
        "n_u_b3": n_u,
        "n_p_active": n_p,
        "active_dimension": int(built_meta.get("active_dimension") or 0),
        "region_dof_indices_mode": region_dofs_mode,
        "region_dof_indices_status": region_status,
        "region_dof_indices_file": (
            str((checkpoint / REGION_DOF_INDICES_NPZ).resolve())
            if region_dof_status_is_pass(region_status)
            else None
        ),
        "region_dof_indices_error": region_error,
        "layout": REGION_DOF_LAYOUT,
        "back_includes_ribs": True,
        "region_dof_source": (
            REGION_DOF_SOURCE_OPERATOR_BUILD
            if region_dof_status_is_pass(region_status)
            else None
        ),
    }
    prov = core_config_provenance or built_meta.get("core_config_provenance")
    if isinstance(prov, dict) and prov:
        body["core_config_provenance"] = dict(prov)
    return body


def export_region_dof_indices_isolated(
    checkpoint: Path,
    *,
    mesh_level: str,
    built_meta: Dict[str, Any],
    mesh_file: Optional[Path] = None,
    core_config_path: Optional[Path] = None,
    python_executable: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    """Run DOLFINx region locate in a child process so segfaults cannot kill Stage A."""
    checkpoint = checkpoint.expanduser().resolve()
    meta_tmp = checkpoint / ".built_meta_for_region_worker.json"
    result_path = checkpoint / ".region_dof_export_result.json"

    meta_out = dict(built_meta)
    if mesh_file and mesh_file.is_file():
        meta_out["region_dof_mesh_file"] = str(mesh_file.resolve())
    if core_config_path and Path(core_config_path).is_file():
        meta_out["region_dof_core_config_path"] = str(Path(core_config_path).resolve())
    write_json_atomic(meta_tmp, meta_out)
    if result_path.is_file():
        result_path.unlink()

    py = str(python_executable or sys.executable)
    cmd = [
        py,
        str(REGION_DOF_WORKER),
        "--checkpoint",
        str(checkpoint),
        "--mesh-level",
        str(mesh_level),
        "--built-meta-json",
        str(meta_tmp),
        "--result-json",
        str(result_path),
    ]
    if mesh_file and mesh_file.is_file():
        cmd.extend(["--mesh-file", str(mesh_file.resolve())])
    if core_config_path and Path(core_config_path).is_file():
        cmd.extend(["--core-config", str(Path(core_config_path).resolve())])

    repo_root = bootstrap_fem_import_paths(start=checkpoint)
    sub_env = region_dof_subprocess_env(repo_root=repo_root)

    proc_error: Optional[str] = None
    proc_tail = ""
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(repo_root),
            env=sub_env,
            capture_output=True,
            text=True,
            timeout=REGION_DOF_SUBPROCESS_TIMEOUT_S,
            check=False,
        )
        proc_tail = (proc.stderr or proc.stdout or "").strip()[-800:]
        if proc.returncode != 0:
            proc_error = f"subprocess_exit_{proc.returncode}" + (f":{proc_tail}" if proc_tail else "")
    except subprocess.TimeoutExpired:
        proc_error = "region_dof_subprocess_timeout"
    except Exception as exc:
        proc_error = f"subprocess_spawn_failed:{type(exc).__name__}:{exc}"
    finally:
        if meta_tmp.is_file():
            meta_tmp.unlink()

    if result_path.is_file():
        try:
            body = json.loads(result_path.read_text(encoding="utf-8"))
        finally:
            result_path.unlink(missing_ok=True)
        status = str(body.get("status") or "deferred_to_stage_c")
        return status, body.get("error")

    detail = proc_error or "region_dof_subprocess_no_result"
    if proc_tail and proc_tail not in detail:
        detail = f"{detail}; {proc_tail}"
    return "deferred_to_stage_c", detail


def write_stage_a_synthesis_artifacts(
    checkpoint: Path,
    *,
    built: Dict[str, Any],
    built_meta: Dict[str, Any],
    mesh_level: str,
    compose_backend: str,
    region_dofs_mode: SynthesisRegionDofsMode = "off",
    core_config_provenance: Optional[Dict[str, Any]] = None,
    core_config_path: Optional[Path] = None,
    python_executable: Optional[str] = None,
) -> Dict[str, Any]:
    checkpoint = checkpoint.expanduser().resolve()
    mode: SynthesisRegionDofsMode = region_dofs_mode if region_dofs_mode in ("off", "best_effort") else "off"

    mesh_file, mesh_resolve = resolve_region_dof_mesh_file(
        checkpoint,
        mesh_level=mesh_level,
        built_meta=built_meta,
        core_config_path=core_config_path,
    )

    region_status = "deferred_to_stage_c"
    region_error: Optional[str] = None
    region_source: Optional[str] = None
    if mode == "off":
        region_error = "disabled_default_no_dolfinx_locate"
    else:
        region_dof_build = built.get("region_dof_build") if isinstance(built, dict) else None
        if isinstance(region_dof_build, dict) and region_dof_build:
            region_status, region_error = export_region_dof_indices_from_operator_build(
                checkpoint,
                region_dof_build=region_dof_build,
            )
            region_source = str(
                region_dof_build.get("region_dof_source") or REGION_DOF_SOURCE_OPERATOR_BUILD
            )
        else:
            region_error = "operator_build_context_missing_region_dof_build"

    body = _build_synthesis_metadata_body(
        checkpoint,
        built=built,
        built_meta=built_meta,
        mesh_level=mesh_level,
        compose_backend=compose_backend,
        region_status=region_status,
        region_error=region_error,
        region_dofs_mode=mode,
        core_config_provenance=core_config_provenance,
        mesh_file=mesh_file,
    )
    write_json_atomic(checkpoint / SYNTHESIS_METADATA_JSON, body)

    warning: Optional[str] = None
    if not region_dof_status_is_pass(region_status):
        warning = (
            f"region_dof_indices_status={region_status}; "
            f"region_dof_indices_mode={mode}; "
            f"detail={region_error}"
        )

    out: Dict[str, Any] = {
        "synthesis_metadata_json": True,
        "region_dof_indices_npz": region_dof_status_is_pass(region_status),
        "region_dof_indices_status": region_status,
        "region_dof_indices_mode": mode,
        "region_dof_indices_error": region_error,
        "region_dof_indices_file": body["region_dof_indices_file"],
        "region_dof_mesh_resolve": mesh_resolve,
        "region_dof_source": region_source or body.get("region_dof_source"),
    }
    if warning:
        out["warning"] = warning
        print(f"[B3_synthesis_export] {warning}", flush=True)
    elif region_dof_status_is_pass(region_status):
        print(
            f"[B3_synthesis_export] region_dof_indices_status={region_status} "
            f"source={region_source or REGION_DOF_SOURCE_OPERATOR_BUILD} "
            f"npz={checkpoint / REGION_DOF_INDICES_NPZ}",
            flush=True,
        )
    return out
