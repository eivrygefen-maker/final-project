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
B3_DISCOVERY_MODE_ARG = "--B3-discovery-mode"
B3_DISCOVERY_BAND_HZ_ARG = "--discovery-band-hz"
B3_TARGET_WINDOW_HALF_WIDTH_HZ_ARG = "--target-window-half-width-hz"
B3_SYNTHESIS_REGION_DOFS_ARG = "--B3-synthesis-region-dofs"
B3_SYNTHESIS_REGION_DOFS_ENV = "B3_SYNTHESIS_REGION_DOFS"
SYNTHESIS_REGION_DOFS_OFF = frozenset({"off", "safe_off", "0", "false", "no", ""})
SYNTHESIS_REGION_DOFS_BEST_EFFORT = frozenset(
    {"best_effort", "best-effort", "on", "1", "true", "yes"}
)

# Required artifacts before expensive LHS / wide sweeps used for audio, STK, or microphone synthesis.
RICH_MODAL_EXPORT_CHECKLIST: Tuple[str, ...] = (
    "eigenvectors_or_mode_shapes",
    "mode_normalization_convention",
    "excitation_coupling_bridge_string_inputs",
    "output_coupling_microphone_listener_body_observation",
    "dof_mapping_metadata",
    "per_mode_material_plate_participation_for_damping_Q",
)

RICH_MODAL_EXPORT_STATUS = "v1_active_basis_opt_in"


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
    elif _venv_path() and not is_production_venv_active():
        errors.append(
            f"VIRTUAL_ENV is not the project production .venv (got: {_venv_path()}); "
            "deactivate solver-mkl and activate production .venv before Stage A export"
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


def verify_rich_modal_post_environment(*, require_dolfinx: bool = False) -> Tuple[bool, List[str]]:
    """Stage C post: production .venv, numpy; DOLFINx/PETSc only for region_dof best_effort."""
    errors: List[str] = []
    warnings: List[str] = []

    if is_solver_mkl_venv_active():
        errors.append(
            "solver-mkl venv is active; Stage C rich modal post requires the project production .venv "
            "(e.g. source ~/final-project/.venv/bin/activate)"
        )
    elif _venv_path() and not is_production_venv_active():
        warnings.append(
            f"VIRTUAL_ENV is not the project production .venv (got: {_venv_path()}); "
            "continuing if numpy is importable"
        )

    try:
        import numpy as _np  # noqa: F401
    except ImportError as exc:
        errors.append(f"numpy not importable: {type(exc).__name__}:{exc}")

    if require_dolfinx:
        try:
            import dolfinx  # noqa: F401
        except ImportError as exc:
            errors.append(
                f"dolfinx required for --B3-synthesis-region-dofs best_effort: {type(exc).__name__}:{exc}"
            )

    return len(errors) == 0, errors + [f"WARN:{w}" for w in warnings]


def verify_solver_mkl_stage_environment(*, require_mkl_pardiso: bool = True) -> Tuple[bool, List[str]]:
    """Solver stage must use solver-mkl venv and optional MKL PARDISO probe."""
    errors: List[str] = []
    warnings: List[str] = []

    if not is_solver_mkl_venv_active():
        if is_production_venv_active():
            errors.append(
                "production .venv is active; Stage B requires solver-mkl "
                "(e.g. source ~/solver-mkl/activate_solver_mkl.sh)"
            )
        else:
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


def resolve_synthesis_region_dofs_mode(
    cli_value: Optional[str] = None,
    *,
    env_value: Optional[str] = None,
) -> str:
    """Stage A region DOF locate mode: off (default) or best_effort (subprocess isolated)."""
    raw = cli_value if cli_value is not None else (env_value or os.environ.get(B3_SYNTHESIS_REGION_DOFS_ENV, "off"))
    token = str(raw).strip().lower()
    if token in SYNTHESIS_REGION_DOFS_OFF:
        return "off"
    if token in SYNTHESIS_REGION_DOFS_BEST_EFFORT:
        return "best_effort"
    raise ValueError(
        f"invalid {B3_SYNTHESIS_REGION_DOFS_ARG}={raw!r}; use off|best_effort "
        f"(env {B3_SYNTHESIS_REGION_DOFS_ENV})"
    )


def verify_official_export_manifest(checkpoint: Path) -> Tuple[bool, List[str], Dict[str, Any]]:
    """Stage B requires a completed Stage A export manifest (status PASS)."""
    checkpoint = checkpoint.expanduser().resolve()
    manifest_path = checkpoint / PIPELINE_EXPORT_MANIFEST
    detail: Dict[str, Any] = {"manifest_path": str(manifest_path)}
    if not manifest_path.is_file():
        return False, [
            f"missing {PIPELINE_EXPORT_MANIFEST} — Stage A export incomplete or crashed before manifest write"
        ], detail
    try:
        body = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, [f"unreadable {PIPELINE_EXPORT_MANIFEST}: {type(exc).__name__}:{exc}"], detail
    detail["manifest_status"] = body.get("status")
    detail["manifest_stage"] = body.get("stage")
    if body.get("status") != "PASS":
        return False, [
            f"{PIPELINE_EXPORT_MANIFEST} status={body.get('status')!r} (expected PASS)"
        ], detail
    return True, [], detail


def verify_checkpoint_complete(
    checkpoint: Path,
    *,
    require_csr: bool = False,
    require_export_manifest: bool = False,
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
    if require_export_manifest:
        man_ok, man_errors, man_detail = verify_official_export_manifest(checkpoint)
        detail["export_manifest"] = man_detail
        if not man_ok:
            errors.extend(man_errors)
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


def default_target_density_output_dir(*, nev: Optional[int] = None, ncv: Optional[int] = None) -> Path:
    rid = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    if nev is not None and ncv is not None:
        return SOLVER_BENCHMARKS_ROOT / f"target_density_experiment_nev{int(nev)}_ncv{int(ncv)}_{rid}"
    return SOLVER_BENCHMARKS_ROOT / f"target_density_experiment_{rid}"


def default_target_alignment_output_dir() -> Path:
    rid = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return SOLVER_BENCHMARKS_ROOT / f"target_alignment_experiment_{rid}"


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


def add_b3_discovery_cli_arguments(parser: Any) -> None:
    """Opt-in wide-band discovery acceptance (Gate A / Option C)."""
    parser.add_argument(
        B3_DISCOVERY_MODE_ARG,
        dest="discovery_mode",
        action="store_true",
        help="Opt-in discovery acceptance: global band + per-target window (default: legacy 220–265).",
    )
    parser.add_argument(
        B3_DISCOVERY_BAND_HZ_ARG,
        dest="discovery_band_hz",
        nargs=2,
        type=float,
        metavar=("LO", "HI"),
        help="Discovery global band in Hz (required with --B3-discovery-mode).",
    )
    parser.add_argument(
        B3_TARGET_WINDOW_HALF_WIDTH_HZ_ARG,
        dest="target_window_half_width_hz",
        type=float,
        help="Half-width Hz around each shift target (required with --B3-discovery-mode).",
    )


def extend_argv_with_discovery_options(
    argv: List[str],
    *,
    discovery_mode: bool = False,
    discovery_band_hz: Optional[Sequence[float]] = None,
    target_window_half_width_hz: Optional[float] = None,
) -> List[str]:
    if not discovery_mode:
        return argv
    out = list(argv)
    out.append(B3_DISCOVERY_MODE_ARG)
    if discovery_band_hz is not None:
        out.extend(
            [
                B3_DISCOVERY_BAND_HZ_ARG,
                str(float(discovery_band_hz[0])),
                str(float(discovery_band_hz[1])),
            ]
        )
    if target_window_half_width_hz is not None:
        out.extend([B3_TARGET_WINDOW_HALF_WIDTH_HZ_ARG, str(float(target_window_half_width_hz))])
    return out


def build_checkpoint_multi_benchmark_argv(
    *,
    checkpoint_dir: str,
    factor_solver: str,
    target_set: str,
    nev: int,
    ncv: int,
    output_dir: str,
    targets_hz: Optional[str] = None,
    baseline_json: Optional[str] = None,
    export_rich_modal_data: bool = False,
    discovery_mode: bool = False,
    discovery_band_hz: Optional[Sequence[float]] = None,
    target_window_half_width_hz: Optional[float] = None,
) -> List[str]:
    """Argv for v2_b3_checkpoint_solver_multi_benchmark (Stage B inner runner)."""
    bench_argv: List[str] = [
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--factor-solver",
        str(factor_solver),
        "--target-set",
        str(target_set),
        "--nev",
        str(int(nev)),
        "--ncv",
        str(int(ncv)),
        "--output-dir",
        str(output_dir),
    ]
    if targets_hz:
        bench_argv.extend(["--targets-hz", str(targets_hz)])
    if baseline_json:
        bench_argv.extend(["--baseline-json", str(baseline_json)])
    if export_rich_modal_data:
        bench_argv.append(B3_EXPORT_RICH_MODAL_DATA_ARG)
    return extend_argv_with_discovery_options(
        bench_argv,
        discovery_mode=discovery_mode,
        discovery_band_hz=discovery_band_hz,
        target_window_half_width_hz=target_window_half_width_hz,
    )


def rich_modal_export_manifest_block(*, requested: bool) -> Dict[str, Any]:
    """Metadata block for manifests."""
    return {
        "requested": bool(requested),
        "enabled_by_default": False,
        "status": "v1_enabled" if requested else "disabled",
        "cli_flag": B3_EXPORT_RICH_MODAL_DATA_ARG,
        "required_checklist": list(RICH_MODAL_EXPORT_CHECKLIST),
        "solver_benchmark_default": "disabled",
        "note": (
            "Rich modal v1: Stage B writes active eigenvectors under rich_modal/ when flag set. "
            "Stage A always writes synthesis_metadata.json on successful export. "
            "Stage C: v2_b3_rich_modal_post.py for region participation and audio output proxies."
        ),
    }


def ensure_rich_modal_export_allowed(*, requested: bool, context: str) -> None:
    """Stage B rich export v1; no-op when flag omitted."""
    if not requested:
        return
