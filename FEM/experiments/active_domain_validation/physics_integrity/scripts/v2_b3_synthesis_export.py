#!/usr/bin/env python3
"""Stage A synthesis metadata and region DOF index export (production .venv)."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

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

SynthesisRegionDofsMode = Literal["off", "best_effort"]


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


def export_region_dof_indices_npz(
    checkpoint: Path,
    *,
    mesh_level: str,
    built_meta: Dict[str, Any],
) -> Tuple[str, Optional[str]]:
    """Locate region DOFs via DOLFINx. Only call from isolated subprocess (segfault risk)."""
    import dolfinx.mesh as dmesh
    from dolfinx import fem

    import fem_main_3d as fem3d

    mesh_file = mesh_path(str(mesh_level), CASE_ID)
    if not mesh_file.is_file():
        return "deferred_to_stage_c", f"mesh_file missing: {mesh_file}"

    msh, _cell_tags, facet_tags = fem3d._load_mesh_and_tags(mesh_file)
    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim - 1, tdim)

    f_top = np.asarray(facet_tags.find(1), dtype=np.int32)
    f_back = np.asarray(facet_tags.find(3), dtype=np.int32)
    f_ribs = np.asarray(facet_tags.find(4), dtype=np.int32)
    f_soundhole = np.asarray(facet_tags.find(2), dtype=np.int32)
    shell_facets = np.unique(np.concatenate([f_top, f_back, f_ribs]).astype(np.int32, copy=False))

    shell_mesh, shell_to_parent, _, _ = dmesh.create_submesh(msh, tdim - 1, shell_facets)
    V_u_trace = fem.functionspace(shell_mesh, fem3d._displacement_element(shell_mesh, 1))

    def _trace_u_rows(facets: np.ndarray) -> np.ndarray:
        if facets.size == 0:
            return np.asarray([], dtype=np.int32)
        dofs = fem3d._locate_facet_displacement_dofs(V_u_trace, shell_mesh, facets)
        return np.unique(np.asarray(dofs, dtype=np.int32).ravel())

    u_idx_top = _trace_u_rows(f_top)
    u_idx_back = _trace_u_rows(f_back)
    u_idx_ribs = _trace_u_rows(f_ribs)
    u_idx_soundhole = _trace_u_rows(f_soundhole)

    u_idx_all = np.asarray(built_meta.get("u_idx") or [], dtype=np.int32).ravel()
    p_idx_all = np.asarray(built_meta.get("p_idx") or [], dtype=np.int32).ravel()
    p_idx_air = p_idx_all.copy()

    np.savez_compressed(
        checkpoint / REGION_DOF_INDICES_NPZ,
        u_idx_top=u_idx_top,
        u_idx_back=u_idx_back,
        u_idx_ribs=u_idx_ribs,
        u_idx_soundhole=u_idx_soundhole,
        p_idx_air=p_idx_air,
        p_idx_all=p_idx_all,
        u_idx_all=u_idx_all,
        layout=np.asarray([REGION_DOF_LAYOUT]),
    )
    return "present", None


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
) -> Dict[str, Any]:
    mesh_file = mesh_path(str(mesh_level), CASE_ID)
    physics = _default_solver_physics()
    gnhep = _gnhep_scales_from_built(built)
    n_u = int(built_meta.get("n_u_b3") or built.get("n_u_b3") or 0)
    p_idx = np.asarray(built_meta.get("p_idx") or built.get("p_idx") or [], dtype=np.int32)
    n_p = int(p_idx.size) if p_idx.size else 0

    return {
        "schema": SYNTHESIS_METADATA_SCHEMA,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mesh_file": str(mesh_file.resolve()) if mesh_file.is_file() else str(mesh_file),
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
            if region_status == "present"
            else None
        ),
        "region_dof_indices_error": region_error,
        "layout": REGION_DOF_LAYOUT,
    }


def export_region_dof_indices_isolated(
    checkpoint: Path,
    *,
    mesh_level: str,
    built_meta: Dict[str, Any],
) -> Tuple[str, Optional[str]]:
    """Run DOLFINx region locate in a child process so segfaults cannot kill Stage A."""
    checkpoint = checkpoint.expanduser().resolve()
    meta_tmp = checkpoint / ".built_meta_for_region_worker.json"
    result_path = checkpoint / ".region_dof_export_result.json"
    write_json_atomic(meta_tmp, built_meta)
    if result_path.is_file():
        result_path.unlink()

    cmd = [
        sys.executable,
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
    proc_error: Optional[str] = None
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=REGION_DOF_SUBPROCESS_TIMEOUT_S,
            check=False,
        )
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip()[-500:]
            proc_error = f"subprocess_exit_{proc.returncode}" + (f":{tail}" if tail else "")
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
        return str(body.get("status") or "deferred_to_stage_c"), body.get("error")

    return "deferred_to_stage_c", proc_error or "region_dof_subprocess_no_result"


def write_stage_a_synthesis_artifacts(
    checkpoint: Path,
    *,
    built: Dict[str, Any],
    built_meta: Dict[str, Any],
    mesh_level: str,
    compose_backend: str,
    region_dofs_mode: SynthesisRegionDofsMode = "off",
) -> Dict[str, Any]:
    checkpoint = checkpoint.expanduser().resolve()
    mode: SynthesisRegionDofsMode = region_dofs_mode if region_dofs_mode in ("off", "best_effort") else "off"

    region_status = "deferred_to_stage_c"
    region_error: Optional[str] = None
    if mode == "off":
        region_error = "disabled_default_no_dolfinx_locate"
    else:
        region_status, region_error = export_region_dof_indices_isolated(
            checkpoint,
            mesh_level=mesh_level,
            built_meta=built_meta,
        )

    body = _build_synthesis_metadata_body(
        checkpoint,
        built=built,
        built_meta=built_meta,
        mesh_level=mesh_level,
        compose_backend=compose_backend,
        region_status=region_status,
        region_error=region_error,
        region_dofs_mode=mode,
    )
    write_json_atomic(checkpoint / SYNTHESIS_METADATA_JSON, body)

    warning: Optional[str] = None
    if region_status != "present":
        warning = (
            f"region_dof_indices_status={region_status}; "
            f"region_dof_indices_mode={mode}; "
            f"detail={region_error}"
        )

    out: Dict[str, Any] = {
        "synthesis_metadata_json": True,
        "region_dof_indices_npz": region_status == "present",
        "region_dof_indices_status": region_status,
        "region_dof_indices_mode": mode,
        "region_dof_indices_error": region_error,
        "region_dof_indices_file": body["region_dof_indices_file"],
    }
    if warning:
        out["warning"] = warning
    return out
