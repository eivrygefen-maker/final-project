#!/usr/bin/env python3
"""Shared helpers for the two-stage checkpoint solver pipeline."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
CONV_DIAG = SCRIPT_DIR.parent / "v2_mesh_convergence" / "diagnostics"
SOLVER_BENCHMARKS_ROOT = CONV_DIAG / "solver_benchmarks"
SOLVER_MKL_ROOT = Path.home() / "solver-mkl"

PIPELINE_EXPORT_MANIFEST = "checkpoint_export_manifest.json"
PIPELINE_SOLVE_MANIFEST = "checkpoint_solve_manifest.json"

B3_EXPORT_RICH_MODAL_DATA_ARG = "--B3-export-rich-modal-data"

# Required artifacts before expensive LHS / wide sweeps used for audio, STK, or microphone synthesis.
RICH_MODAL_EXPORT_CHECKLIST: Tuple[str, ...] = (
    "eigenvectors_or_mode_shapes",
    "mode_normalization_convention",
    "excitation_coupling_bridge_string_inputs",
    "output_coupling_microphone_listener_body_observation",
    "dof_mapping_metadata",
    "per_mode_material_plate_participation_for_damping_Q",
)

RICH_MODAL_EXPORT_STATUS = "not_implemented_opt_in_only"


def _venv_path() -> str:
    return str(os.environ.get("VIRTUAL_ENV") or "").replace("\\", "/")


def is_solver_mkl_venv_active() -> bool:
    venv = _venv_path()
    return bool(venv and "solver-mkl" in venv)


def is_production_venv_active() -> bool:
    venv = _venv_path()
    if not venv:
        return False
    if "solver-mkl" in venv:
        return False
    return True


def verify_production_stage_environment() -> Tuple[bool, List[str]]:
    """Production stage must have DOLFINx and must not use solver-mkl venv."""
    errors: List[str] = []
    warnings: List[str] = []

    if is_solver_mkl_venv_active():
        errors.append(
            "solver-mkl venv is active; production export requires the project production .venv "
            "(e.g. source ~/final-project/.venv/bin/activate)"
        )
    elif not is_production_venv_active():
        warnings.append("VIRTUAL_ENV unset or unrecognized; continuing if DOLFINx is importable")

    try:
        import dolfinx  # noqa: F401
    except ImportError as exc:
        errors.append(f"dolfinx not importable in production stage: {type(exc).__name__}:{exc}")

    try:
        import petsc4py  # noqa: F401
    except ImportError as exc:
        errors.append(f"petsc4py not importable: {type(exc).__name__}:{exc}")

    petsc_path = ""
    try:
        import petsc4py

        petsc_path = str(Path(petsc4py.__file__).resolve())
        if "solver-mkl" in petsc_path.replace("\\", "/"):
            errors.append(f"petsc4py resolves to solver-mkl env: {petsc_path}")
    except Exception:
        pass

    return len(errors) == 0, errors + [f"WARN:{w}" for w in warnings]


def verify_solver_mkl_stage_environment(*, require_mkl_pardiso: bool = True) -> Tuple[bool, List[str]]:
    """Solver stage must use solver-mkl venv and optional MKL PARDISO probe."""
    errors: List[str] = []
    warnings: List[str] = []

    if not is_solver_mkl_venv_active():
        errors.append(
            "solver-mkl venv is not active; run: source ~/solver-mkl/activate_solver_mkl.sh"
        )

    try:
        import dolfinx  # noqa: F401

        warnings.append("dolfinx is importable in solver-mkl env (unexpected but not fatal)")
    except ImportError:
        pass

    try:
        import petsc4py

        petsc_path = str(Path(petsc4py.__file__).resolve())
        if "solver-mkl" not in petsc_path.replace("\\", "/"):
            errors.append(f"petsc4py is not from solver-mkl venv: {petsc_path}")
    except ImportError as exc:
        errors.append(f"petsc4py not importable: {type(exc).__name__}:{exc}")

    if require_mkl_pardiso:
        try:
            from v2_b3_operator_checkpoint_portable import probe_pc_lu_factor_solver

            probe = probe_pc_lu_factor_solver("mkl_pardiso")
            if not probe.get("available"):
                errors.append(f"mkl_pardiso probe failed: {probe.get('error')}")
        except Exception as exc:
            errors.append(f"mkl_pardiso probe exception: {type(exc).__name__}:{exc}")

    return len(errors) == 0, errors + [f"WARN:{w}" for w in warnings]


def verify_mumps_available() -> Tuple[bool, Optional[str]]:
    try:
        from v2_b3_operator_checkpoint_portable import probe_pc_lu_factor_solver

        probe = probe_pc_lu_factor_solver("mumps")
        if probe.get("available"):
            return True, None
        return False, str(probe.get("error"))
    except Exception as exc:
        return False, f"{type(exc).__name__}:{exc}"


def verify_checkpoint_complete(
    checkpoint: Path,
    *,
    require_csr: bool = False,
) -> Tuple[bool, List[str], Dict[str, Any]]:
    from v2_b3_operator_checkpoint_portable import verify_portable_checkpoint_export

    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_dir():
        return False, [f"checkpoint directory not found: {checkpoint}"], {"checkpoint_dir": str(checkpoint)}

    export_pass, missing, detail = verify_portable_checkpoint_export(
        checkpoint,
        require_csr=require_csr,
    )
    errors: List[str] = []
    if not export_pass:
        if require_csr:
            errors.append(f"checkpoint incomplete; missing: {missing}")
        else:
            errors.append(f"checkpoint missing required files: {missing}")
    elif not require_csr and detail.get("warnings"):
        for warn in detail["warnings"]:
            print(f"[B3_checkpoint] WARN: {warn}", flush=True)
    if not (checkpoint / "built_metadata.json").is_file():
        errors.append("missing built_metadata.json")
    detail["checkpoint_dir"] = str(checkpoint)
    detail["export_pass"] = bool(export_pass)
    detail["csr_required"] = bool(require_csr)
    return len(errors) == 0, errors, detail


def verify_checkpoint_matrices(checkpoint: Path) -> Tuple[bool, List[str], Dict[str, Any]]:
    """Load A/M once and verify shape/nnz without running EPS."""
    from v2_b3_operator_checkpoint_portable import load_operators_with_portable_fallback
    from v2_b3_petsc_util import mat_shape
    from v2_b3_st_sinvert_solver_lib import mat_global_nnz_used

    checkpoint = checkpoint.expanduser().resolve()
    mats: List[Any] = []
    try:
        A_active, M_active, load_diag = load_operators_with_portable_fallback(checkpoint)
        mats.extend([A_active, M_active])
        a_shape = mat_shape(A_active)
        m_shape = mat_shape(M_active)
        a_nnz = mat_global_nnz_used(A_active)
        m_nnz = mat_global_nnz_used(M_active)
        errors: List[str] = []
        if not a_shape or not m_shape:
            errors.append("matrix shape unavailable after checkpoint load")
        elif a_shape != m_shape:
            errors.append(f"A/M shape mismatch: A={a_shape} M={m_shape}")
        if a_nnz is None or m_nnz is None or int(a_nnz) <= 0 or int(m_nnz) <= 0:
            errors.append(f"invalid nnz after load: A={a_nnz} M={m_nnz}")
        detail = {
            "checkpoint_dir": str(checkpoint),
            "A_shape": a_shape,
            "M_shape": m_shape,
            "A_nnz_used": a_nnz,
            "M_nnz_used": m_nnz,
            "load_path": load_diag.get("load_path_summary"),
            "load_path_summary": load_diag.get("load_path_summary"),
            "load_path_by_matrix": load_diag.get("load_path_by_matrix"),
            "csr_present": bool(load_diag.get("csr_metadata_present")),
            "csr_required": False,
            "csr_verification_pass": load_diag.get("csr_verification_pass"),
            "binary_load_errors": load_diag.get("binary_load_errors"),
            "csr_load_error": load_diag.get("csr_load_error"),
        }
        return len(errors) == 0, errors, detail
    except Exception as exc:
        return False, [f"{type(exc).__name__}:{exc}"], {"checkpoint_dir": str(checkpoint)}
    finally:
        for mat in mats:
            try:
                mat.destroy()
            except Exception:
                pass


def default_checkpoint_dir(mesh_level: str, *, run_id: Optional[str] = None) -> Path:
    rid = run_id or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return CONV_DIAG / f"st_worker_scaling_{mesh_level}_{rid}"


def default_solve_output_dir(*, factor_solver: str, target_set: str) -> Path:
    rid = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return SOLVER_BENCHMARKS_ROOT / f"checkpoint_solve_{factor_solver}_{target_set}_{rid}"


def default_target_density_output_dir() -> Path:
    rid = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return SOLVER_BENCHMARKS_ROOT / f"target_density_experiment_{rid}"


def write_json(path: Path, body: Dict[str, Any]) -> None:
    from v2_b3_petsc_util import write_json_atomic

    write_json_atomic(path, body)


def fail_with_messages(prefix: str, messages: List[str], *, exit_code: int = 2) -> None:
    print(f"[{prefix}] environment/check failed:", flush=True)
    for msg in messages:
        print(f"  - {msg}", flush=True)
    raise SystemExit(exit_code)


def parse_rich_modal_export_flag(argv: Optional[Sequence[str]]) -> bool:
    if not argv:
        return False
    return B3_EXPORT_RICH_MODAL_DATA_ARG in argv


def rich_modal_export_manifest_block(*, requested: bool) -> Dict[str, Any]:
    """Metadata block for manifests; export body is not implemented yet."""
    return {
        "requested": bool(requested),
        "enabled_by_default": False,
        "status": RICH_MODAL_EXPORT_STATUS if requested else "disabled",
        "cli_flag": B3_EXPORT_RICH_MODAL_DATA_ARG,
        "required_checklist": list(RICH_MODAL_EXPORT_CHECKLIST),
        "solver_benchmark_default": "disabled",
        "note": (
            "Rich modal export is opt-in only and not implemented in checkpoint pipeline yet. "
            "Before LHS or wide sweeps for audio/STK/microphone synthesis, verify checklist items "
            "in production FOM outputs or future rich-export implementation."
        ),
    }


def ensure_rich_modal_export_allowed(*, requested: bool, context: str) -> None:
    """Fail clearly when opt-in rich export is requested but not implemented."""
    if not requested:
        return
    fail_with_messages(
        context,
        [
            f"{B3_EXPORT_RICH_MODAL_DATA_ARG} was requested but rich modal export is not implemented yet",
            "See physics_integrity/docs/B3_RICH_MODAL_EXPORT_TODO.md for required artifacts",
            "Disable the flag for solver timing benchmarks (default)",
        ],
    )
